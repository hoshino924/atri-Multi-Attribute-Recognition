# -*- coding: utf-8 -*-
"""Train the ATRI outfit, pose, and expression classifier."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import random
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

from calibration import build_calibration, collect_logits
from dataset import (
    AtriDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SCALE_CODES,
    build_dual_view_transform,
    scan_records,
    stratified_split,
)
from labels import TASK_CODES
from model import AtriNet


TASKS = tuple(TASK_CODES.keys())
CHECKPOINT_VERSION = 3
PROTECTED_OUTPUT_NAMES = {
    "atri_net_best.pth",
    "atri_net_last.pth",
    "metrics.json",
}


def seed_everything(seed):
    """Seed Python and PyTorch for a repeatable dataset split and training run."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_output_directory(args):
    """Resolve one run directory and guard existing training artifacts."""
    if args.run_name:
        if (
            args.run_name in {".", ".."}
            or os.path.basename(args.run_name) != args.run_name
        ):
            raise ValueError("--run_name must be a single directory name")
        out_dir = os.path.join(args.out_dir, args.run_name)
    else:
        out_dir = args.out_dir

    if not args.resume and os.path.isdir(out_dir) and not args.overwrite:
        existing = PROTECTED_OUTPUT_NAMES.intersection(os.listdir(out_dir))
        if existing:
            names = ", ".join(sorted(existing))
            raise FileExistsError(
                f"output directory already contains {names}: {out_dir}. "
                "Use --run_name for a new run or --overwrite to replace them."
            )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def dataset_signature(records, root):
    """Fingerprint selected filenames and sizes to protect resumed splits."""
    root = os.path.abspath(root)
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.path.name):
        path = os.path.abspath(record.path)
        relative_path = os.path.relpath(path, root).replace("\\", "/")
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(os.path.getsize(path)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def save_json(path, value):
    """Write one UTF-8 JSON artifact."""
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def save_run_metadata(out_dir, args, device):
    """Record arguments and the software/hardware environment."""
    save_json(os.path.join(out_dir, "run_config.json"), vars(args))
    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
    }
    save_json(os.path.join(out_dir, "environment.json"), environment)


def print_record_summary(name, records):
    """Print label counts for one dataset split."""
    print(f"{name}: {len(records)} images")
    for task in TASKS:
        counts = Counter(getattr(record, task) for record in records)
        detail = ", ".join(
            f"{code}={counts.get(code, 0)}"
            for code in TASK_CODES[task]
        )
        print(f"  {task}: {detail}")


def save_split_manifest(
    path,
    root,
    train_records,
    val_records,
    args,
    data_signature,
):
    """Record the exact split used for this run."""
    root = os.path.abspath(root)

    def relative_names(records):
        return [
            os.path.relpath(os.path.abspath(record.path), root)
            for record in records
        ]

    manifest = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "scale": args.scale,
        "dataset_signature": data_signature,
        "train": relative_names(train_records),
        "validation": relative_names(val_records),
    }
    save_json(path, manifest)


def move_targets(targets, device):
    """Move a collated target dictionary to the selected device."""
    return {
        task: targets[task].to(device, non_blocking=True)
        for task in TASKS
    }


def move_views(views, device):
    """Move both image views to the selected device."""
    return {
        name: tensor.to(device, non_blocking=True)
        for name, tensor in views.items()
    }


def run_epoch(
    model,
    loader,
    loss_fn,
    device,
    amp_enabled,
    optimizer=None,
    scaler=None,
    gradient_clip=0.0,
    backbone_trainable=True,
    freeze_bn_statistics=False,
):
    """Run one training or validation epoch and return aggregate metrics."""
    training = optimizer is not None
    model.train(training)
    if training and not backbone_trainable:
        model.backbone.eval()
    elif training and freeze_bn_statistics:
        model.freeze_backbone_batchnorm()

    sample_count = 0
    total_loss = 0.0
    task_loss_totals = {task: 0.0 for task in TASKS}
    task_correct = {task: 0 for task in TASKS}
    joint_correct = 0
    expression_classes = len(TASK_CODES["expression"])
    expression_confusion = torch.zeros(
        (expression_classes, expression_classes),
        dtype=torch.long,
    )

    for views, targets in loader:
        views = move_views(views, device)
        targets = move_targets(targets, device)
        batch_size = views["full"].size(0)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                outputs = model(views["full"], views["expression"])
                losses = {
                    task: loss_fn(outputs[task], targets[task])
                    for task in TASKS
                }
                loss = sum(losses.values())

            if training:
                scaler.scale(loss).backward()
                if gradient_clip > 0.0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()

        sample_count += batch_size
        total_loss += loss.item() * batch_size
        batch_joint = torch.ones(batch_size, dtype=torch.bool, device=device)
        for task in TASKS:
            task_loss_totals[task] += losses[task].item() * batch_size
            predictions = outputs[task].argmax(dim=1)
            correct = predictions.eq(targets[task])
            task_correct[task] += correct.sum().item()
            batch_joint &= correct
            if task == "expression":
                indices = (
                    targets[task].detach().cpu() * expression_classes
                    + predictions.detach().cpu()
                )
                expression_confusion += torch.bincount(
                    indices,
                    minlength=expression_classes * expression_classes,
                ).reshape(expression_classes, expression_classes)
        joint_correct += batch_joint.sum().item()

    if sample_count == 0:
        raise RuntimeError("the data loader produced no samples")

    expression_support = expression_confusion.sum(dim=1)
    expression_class_accuracy = {}
    for index, code in enumerate(TASK_CODES["expression"]):
        support = expression_support[index].item()
        expression_class_accuracy[code] = (
            expression_confusion[index, index].item() / support
            if support
            else None
        )

    return {
        "loss": total_loss / sample_count,
        "task_loss": {
            task: task_loss_totals[task] / sample_count
            for task in TASKS
        },
        "accuracy": {
            task: task_correct[task] / sample_count
            for task in TASKS
        },
        "joint_accuracy": joint_correct / sample_count,
        "expression_class_accuracy": expression_class_accuracy,
        "expression_confusion": expression_confusion.tolist(),
    }


def format_metrics(name, metrics):
    """Format one compact console metrics line."""
    accuracy = " ".join(
        f"{task}={metrics['accuracy'][task] * 100:.1f}%"
        for task in TASKS
    )
    return (
        f"{name} loss={metrics['loss']:.4f} "
        f"{accuracy} joint={metrics['joint_accuracy'] * 100:.1f}%"
    )


def make_checkpoint(
    model,
    args,
    epoch,
    best_expression_loss,
    history,
    calibration=None,
    data_signature=None,
):
    """Create the inference-compatible part of a checkpoint."""
    return {
        "format_version": CHECKPOINT_VERSION,
        "model_state": model.state_dict(),
        "epoch": epoch,
        "best_expression_loss": best_expression_loss,
        "dataset_signature": data_signature,
        "model_config": {
            "backbone": "resnet18",
            "architecture": "dual_view_shared_backbone",
            "dropout": args.dropout,
            "task_sizes": {
                task: len(codes)
                for task, codes in TASK_CODES.items()
            },
        },
        "preprocess": {
            "scale": args.scale,
            "full": {
                "height": args.height,
                "width": args.width,
                "margin": args.margin,
            },
            "expression": {
                "size": args.expression_size,
                "width_fraction": args.expression_width_fraction,
                "height_fraction": args.expression_height_fraction,
                "margin": args.margin,
            },
            "background": [0, 0, 0],
            "normalization": {
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
            },
        },
        "label_codes": {
            task: list(codes)
            for task, codes in TASK_CODES.items()
        },
        "train_args": vars(args),
        "history": history,
        "calibration": calibration or {},
    }


def load_torch_checkpoint(path, device):
    """Load a trusted local training checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def validate_resume_config(checkpoint, args, data_signature):
    """Reject a resume checkpoint built with incompatible data settings."""
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("resume checkpoint format is not compatible")
    if checkpoint.get("label_codes") != TASK_CODES:
        raise ValueError("resume checkpoint label definitions do not match")
    checkpoint_signature = checkpoint.get("dataset_signature")
    if checkpoint_signature and checkpoint_signature != data_signature:
        raise ValueError(
            "the selected dataset changed since this checkpoint was created"
        )

    model_config = checkpoint.get("model_config", {})
    if model_config.get("dropout") != args.dropout:
        raise ValueError("resume checkpoint dropout does not match --dropout")

    preprocess = checkpoint.get("preprocess", {})
    full_expected = {
        "height": args.height,
        "width": args.width,
        "margin": args.margin,
    }
    expression_expected = {
        "size": args.expression_size,
        "width_fraction": args.expression_width_fraction,
        "height_fraction": args.expression_height_fraction,
        "margin": args.margin,
    }
    mismatched = []
    if preprocess.get("scale") != args.scale:
        mismatched.append("scale")
    mismatched.extend(
        f"full.{key}"
        for key, value in full_expected.items()
        if preprocess.get("full", {}).get(key) != value
    )
    mismatched.extend(
        f"expression.{key}"
        for key, value in expression_expected.items()
        if preprocess.get("expression", {}).get(key) != value
    )
    if mismatched:
        names = ", ".join(mismatched)
        raise ValueError(f"resume checkpoint settings differ: {names}")


def restore_training_state(
    checkpoint,
    model,
    optimizer,
    scheduler,
    scaler,
):
    """Restore model and optional optimizer state for resume training."""
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(
            "the resume file uses the old checkpoint format; retrain with the "
            "three-task model"
        )

    model.load_state_dict(checkpoint["model_state"])
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])

    return (
        checkpoint.get("epoch", 0),
        checkpoint.get("best_expression_loss", float("inf")),
        checkpoint.get("epochs_without_improvement", 0),
        checkpoint.get("history", []),
        checkpoint.get("calibration", {}),
    )


def save_plots(history, out_dir):
    """Save loss and validation accuracy curves."""
    if not history:
        return

    epochs = [item["epoch"] for item in history]

    plt.figure()
    plt.plot(epochs, [item["train"]["loss"] for item in history], label="train")
    plt.plot(epochs, [item["validation"]["loss"] for item in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure()
    for task in TASKS:
        plt.plot(
            epochs,
            [item["validation"]["accuracy"][task] for item in history],
            label=task,
        )
    plt.plot(
        epochs,
        [item["validation"]["joint_accuracy"] for item in history],
        label="joint",
        linestyle="--",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=200)
    plt.close()


def save_metrics(history, out_dir):
    """Save complete JSON history and a compact comparison-friendly CSV."""
    save_json(os.path.join(out_dir, "metrics.json"), history)
    fieldnames = [
        "epoch",
        "split",
        "loss",
        "outfit_loss",
        "pose_loss",
        "expression_loss",
        "outfit_accuracy",
        "pose_accuracy",
        "expression_accuracy",
        "joint_accuracy",
        "backbone_lr",
        "heads_lr",
    ]
    with open(
        os.path.join(out_dir, "metrics.csv"),
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            learning_rates = item.get("learning_rates", {})
            for split_name in ("train", "validation"):
                metrics = item[split_name]
                writer.writerow({
                    "epoch": item["epoch"],
                    "split": split_name,
                    "loss": metrics["loss"],
                    "outfit_loss": metrics["task_loss"]["outfit"],
                    "pose_loss": metrics["task_loss"]["pose"],
                    "expression_loss": metrics["task_loss"]["expression"],
                    "outfit_accuracy": metrics["accuracy"]["outfit"],
                    "pose_accuracy": metrics["accuracy"]["pose"],
                    "expression_accuracy": metrics["accuracy"]["expression"],
                    "joint_accuracy": metrics["joint_accuracy"],
                    "backbone_lr": learning_rates.get("backbone"),
                    "heads_lr": learning_rates.get("heads"),
                })


def save_expression_report(metrics, out_dir):
    """Save per-class expression accuracy and a confusion matrix."""
    codes = TASK_CODES["expression"]
    confusion = metrics["expression_confusion"]
    support = [sum(row) for row in confusion]
    report = {
        "overall_accuracy": metrics["accuracy"]["expression"],
        "per_class_accuracy": metrics["expression_class_accuracy"],
        "support": {
            code: support[index]
            for index, code in enumerate(codes)
        },
        "confusion_matrix": confusion,
    }
    report_path = os.path.join(out_dir, "expression_report.json")
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)

    plt.figure(figsize=(10, 9))
    plt.imshow(confusion, interpolation="nearest", cmap="Blues")
    plt.title("Expression Confusion Matrix")
    plt.colorbar()
    positions = range(len(codes))
    plt.xticks(positions, codes, rotation=90)
    plt.yticks(positions, codes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "expression_confusion_matrix.png"),
        dpi=200,
    )
    plt.close()


def validate_args(args):
    """Validate command-line values before loading data or allocating a model."""
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive")
    if args.expression_size <= 0:
        raise ValueError("--expression_size must be positive")
    if not 0.0 < args.expression_width_fraction <= 1.0:
        raise ValueError("--expression_width_fraction must be in (0, 1]")
    if not 0.0 < args.expression_height_fraction <= 1.0:
        raise ValueError("--expression_height_fraction must be in (0, 1]")
    if not 0.0 <= args.margin < 0.5:
        raise ValueError("--margin must be in [0, 0.5)")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("--warmup_epochs must be in [0, epochs)")
    if args.patience <= 0:
        raise ValueError("--patience must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label_smoothing must be in [0, 1)")
    if args.gradient_clip < 0.0:
        raise ValueError("--gradient_clip cannot be negative")
    if not 0.0 <= args.threshold_quantile <= 0.5:
        raise ValueError("--threshold_quantile must be in [0, 0.5]")


def train(args):
    """Run the complete training and validation workflow."""
    validate_args(args)
    seed_everything(args.seed)
    args.out_dir = prepare_output_directory(args)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    print("Device:", device)
    print("Automatic mixed precision:", "enabled" if amp_enabled else "disabled")
    print("BatchNorm statistics:", "train" if args.train_bn else "frozen")
    print(
        "Input views:",
        f"full={args.height}x{args.width}",
        f"expression={args.expression_size}x{args.expression_size}",
    )
    print("Output directory:", args.out_dir)
    save_run_metadata(args.out_dir, args, device)

    records = scan_records(args.train_dir, scale=args.scale)
    data_signature = dataset_signature(records, args.train_dir)
    train_records, val_records = stratified_split(
        records,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print_record_summary("Train", train_records)
    print_record_summary("Validation", val_records)
    save_split_manifest(
        os.path.join(args.out_dir, "split.json"),
        args.train_dir,
        train_records,
        val_records,
        args,
        data_signature,
    )

    train_dataset = AtriDataset(
        train_records,
        paired_transform=build_dual_view_transform(
            height=args.height,
            width=args.width,
            expression_size=args.expression_size,
            train=True,
            expression_width_fraction=args.expression_width_fraction,
            expression_height_fraction=args.expression_height_fraction,
            margin=args.margin,
        ),
    )
    val_dataset = AtriDataset(
        val_records,
        paired_transform=build_dual_view_transform(
            height=args.height,
            width=args.width,
            expression_size=args.expression_size,
            train=False,
            expression_width_fraction=args.expression_width_fraction,
            expression_height_fraction=args.expression_height_fraction,
            margin=args.margin,
        ),
    )

    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    if args.workers > 0:
        loader_options["persistent_workers"] = True
        loader_options["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_options,
    )

    model = AtriNet(
        pretrained=not args.no_pretrained and not args.resume,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.heads.parameters(), "lr": args.lr_heads},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=args.min_lr,
    )
    train_loss_fn = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing,
    )
    validation_loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    start_epoch = 0
    best_expression_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    best_calibration = {}
    if args.resume:
        checkpoint = load_torch_checkpoint(args.resume, device)
        validate_resume_config(checkpoint, args, data_signature)
        (
            start_epoch,
            best_expression_loss,
            epochs_without_improvement,
            history,
            best_calibration,
        ) = restore_training_state(
            checkpoint,
            model,
            optimizer,
            scheduler,
            scaler,
        )
        print(f"Resumed from epoch {start_epoch}: {args.resume}")

    best_path = os.path.join(args.out_dir, "atri_net_best.pth")
    last_path = os.path.join(args.out_dir, "atri_net_last.pth")

    print("Training started.")
    for epoch_index in range(start_epoch, args.epochs):
        epoch = epoch_index + 1
        backbone_trainable = epoch_index >= args.warmup_epochs
        model.set_backbone_trainable(backbone_trainable)

        train_metrics = run_epoch(
            model,
            train_loader,
            train_loss_fn,
            device,
            amp_enabled,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip=args.gradient_clip,
            backbone_trainable=backbone_trainable,
            freeze_bn_statistics=not args.train_bn,
        )
        with torch.inference_mode():
            val_metrics = run_epoch(
                model,
                val_loader,
                validation_loss_fn,
                device,
                amp_enabled,
            )

        if backbone_trainable:
            scheduler.step()

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "learning_rates": {
                "backbone": optimizer.param_groups[0]["lr"],
                "heads": optimizer.param_groups[1]["lr"],
            },
        })
        print(f"Epoch [{epoch}/{args.epochs}]")
        print(" ", format_metrics("train", train_metrics))
        print(" ", format_metrics("validation", val_metrics))

        monitored_loss = val_metrics["task_loss"]["expression"]
        improved = monitored_loss < best_expression_loss
        if improved:
            best_expression_loss = monitored_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if improved:
            collected = collect_logits(
                model,
                val_loader,
                device,
                TASKS,
                amp_enabled=amp_enabled,
            )
            best_calibration = build_calibration(
                collected,
                threshold_quantile=args.threshold_quantile,
            )

        checkpoint = make_checkpoint(
            model,
            args,
            epoch,
            best_expression_loss,
            history,
            calibration=best_calibration,
            data_signature=data_signature,
        )
        checkpoint.update({
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epochs_without_improvement": epochs_without_improvement,
        })
        torch.save(checkpoint, last_path)

        if improved:
            best_checkpoint = make_checkpoint(
                model,
                args,
                epoch,
                best_expression_loss,
                history,
                calibration=best_calibration,
                data_signature=data_signature,
            )
            torch.save(best_checkpoint, best_path)
            save_expression_report(val_metrics, args.out_dir)
            save_json(
                os.path.join(args.out_dir, "calibration.json"),
                best_calibration,
            )
            print("  Saved new best checkpoint.")

        save_metrics(history, args.out_dir)

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after {args.patience} epochs "
                "without validation improvement."
            )
            break

    save_plots(history, args.out_dir)
    print("Saved:", best_path)
    print("Saved:", last_path)
    print("Saved:", os.path.join(args.out_dir, "loss_curve.png"))
    print("Saved:", os.path.join(args.out_dir, "accuracy_curve.png"))
    print("Saved:", os.path.join(args.out_dir, "metrics.json"))
    print("Saved:", os.path.join(args.out_dir, "metrics.csv"))
    print("Saved:", os.path.join(args.out_dir, "calibration.json"))
    print("Saved:", os.path.join(args.out_dir, "expression_report.json"))
    print(
        "Saved:",
        os.path.join(args.out_dir, "expression_confusion_matrix.png"),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the ATRI three-task dual-view classifier."
    )
    parser.add_argument("--train_dir", default="atridataset/train")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--run_name", help="optional subdirectory for this run")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--expression_size", type=int, default=512)
    parser.add_argument("--expression_width_fraction", type=float, default=0.65)
    parser.add_argument("--expression_height_fraction", type=float, default=0.50)
    parser.add_argument("--margin", type=float, default=0.04)
    parser.add_argument("--scale", choices=SCALE_CODES, default="w")
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_heads", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gradient_clip", type=float, default=5.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--threshold_quantile", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume")
    parser.add_argument(
        "--train_bn",
        action="store_true",
        help="update backbone BatchNorm statistics instead of freezing them",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
