# -*- coding: utf-8 -*-
"""Evaluate a checkpoint on a labeled, independent image set."""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from calibration import (
    apply_temperature,
    expected_calibration_error,
    suggested_threshold,
)
from dataset import (
    AtriDataset,
    IMAGE_EXTENSIONS,
    LabeledImageRecord,
    load_label_manifest,
    parse_filename,
)
from infer import load_model
from labels import TASK_CODES


TASKS = tuple(TASK_CODES.keys())


def load_evaluation_records(image_dir, manifest_path=None):
    """Load labels from CSV or from conventional six-field filenames."""
    image_dir = Path(image_dir)
    if manifest_path:
        return load_label_manifest(image_dir, manifest_path)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {image_dir}")

    records = []
    errors = []
    for path in sorted(image_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            parsed = parse_filename(path)
        except ValueError as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        records.append(
            LabeledImageRecord(
                path=path,
                outfit=parsed.outfit,
                pose=parsed.pose,
                expression=parsed.expression,
            )
        )
    if errors:
        details = "\n".join(f"  - {error}" for error in errors[:20])
        raise ValueError(
            "evaluation images need conventional filenames or --labels CSV:\n"
            f"{details}"
        )
    if not records:
        raise ValueError(f"no labeled images found in: {image_dir}")
    return records


def confusion_matrix(targets, predictions, class_count):
    indices = targets * class_count + predictions
    return torch.bincount(
        indices,
        minlength=class_count * class_count,
    ).reshape(class_count, class_count)


def task_metrics(logits, targets, threshold):
    """Calculate full-set and accepted-subset metrics for one task."""
    probabilities = torch.softmax(logits, dim=1)
    confidence, predictions = probabilities.max(dim=1)
    accepted = (
        torch.ones_like(confidence, dtype=torch.bool)
        if threshold is None
        else confidence.ge(threshold)
    )
    correct = predictions.eq(targets)
    matrix = confusion_matrix(targets, predictions, logits.shape[1])
    support = matrix.sum(dim=1)
    true_positive = matrix.diag()
    predicted = matrix.sum(dim=0)
    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    included = support.gt(0)

    return {
        "accuracy": correct.float().mean().item(),
        "macro_f1": f1[included].mean().item() if included.any() else None,
        "nll": functional.cross_entropy(logits, targets).item(),
        "ece": expected_calibration_error(logits, targets),
        "coverage": accepted.float().mean().item(),
        "selective_accuracy": (
            correct[accepted].float().mean().item()
            if accepted.any()
            else None
        ),
        "confusion_matrix": matrix.tolist(),
        "support": support.tolist(),
        "per_class_accuracy": [
            true_positive[index].item() / support[index].item()
            if support[index].item()
            else None
            for index in range(len(support))
        ],
        "predictions": predictions.tolist(),
        "confidence": confidence.tolist(),
        "accepted": accepted.tolist(),
        "correct": correct.tolist(),
    }


def save_confusion_plot(matrix, codes, task, output_dir):
    plt.figure(figsize=(10, 9) if len(codes) > 10 else (7, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(f"{task.title()} Confusion Matrix")
    plt.colorbar()
    positions = range(len(codes))
    plt.xticks(positions, codes, rotation=90)
    plt.yticks(positions, codes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{task}_confusion_matrix.png"),
        dpi=200,
    )
    plt.close()


def evaluate(args):
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.min_confidence is not None and not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min_confidence must be in [0, 1]")
    if os.path.exists(os.path.join(args.output_dir, "evaluation.json")) and not args.overwrite:
        raise FileExistsError(
            f"evaluation output already exists: {args.output_dir}; use --overwrite"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )
    model, image_transforms, label_codes = load_model(args.weight, device)
    records = load_evaluation_records(args.test_dir, args.labels)
    dataset = AtriDataset(
        records,
        full_transform=image_transforms["full"],
        expression_transform=image_transforms["expression"],
        task_codes=label_codes,
    )
    loader_options = {
        "batch_size": args.batch,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    if args.workers > 0:
        loader_options["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_options)

    all_logits = {task: [] for task in TASKS}
    all_targets = {task: [] for task in TASKS}
    amp_enabled = device.type == "cuda" and not args.no_amp
    with torch.inference_mode():
        for views, targets in loader:
            full_image = views["full"].to(device, non_blocking=True)
            expression_image = views["expression"].to(
                device,
                non_blocking=True,
            )
            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                outputs = model(full_image, expression_image)
            for task in TASKS:
                logits = apply_temperature(
                    outputs[task],
                    task,
                    model.calibration,
                )
                all_logits[task].append(logits.detach().float().cpu())
                all_targets[task].append(targets[task].long().cpu())

    metrics = {}
    for task in TASKS:
        logits = torch.cat(all_logits[task])
        targets = torch.cat(all_targets[task])
        threshold = None
        if not args.accept_all:
            threshold = (
                args.min_confidence
                if args.min_confidence is not None
                else suggested_threshold(task, model.calibration)
            )
        task_result = task_metrics(logits, targets, threshold)
        task_result["threshold"] = threshold
        task_result["support_by_code"] = {
            code: task_result["support"][index]
            for index, code in enumerate(label_codes[task])
        }
        task_result["per_class_accuracy_by_code"] = {
            code: task_result["per_class_accuracy"][index]
            for index, code in enumerate(label_codes[task])
        }
        metrics[task] = task_result
        save_confusion_plot(
            task_result["confusion_matrix"],
            label_codes[task],
            task,
            args.output_dir,
        )

    sample_count = len(records)
    joint_correct = []
    joint_accepted = []
    for index in range(sample_count):
        joint_correct.append(all(metrics[task]["correct"][index] for task in TASKS))
        joint_accepted.append(all(metrics[task]["accepted"][index] for task in TASKS))
    accepted_and_correct = [
        accepted and correct
        for accepted, correct in zip(joint_accepted, joint_correct)
    ]
    report = {
        "checkpoint": str(args.weight),
        "test_dir": str(args.test_dir),
        "labels": str(args.labels) if args.labels else None,
        "samples": sample_count,
        "joint_accuracy": sum(joint_correct) / sample_count,
        "joint_coverage": sum(joint_accepted) / sample_count,
        "joint_selective_accuracy": (
            sum(accepted_and_correct) / sum(joint_accepted)
            if any(joint_accepted)
            else None
        ),
        "tasks": {
            task: {
                key: value
                for key, value in metrics[task].items()
                if key not in {"predictions", "confidence", "accepted", "correct"}
            }
            for task in TASKS
        },
    }
    with open(
        os.path.join(args.output_dir, "evaluation.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)

    with open(
        os.path.join(args.output_dir, "predictions.csv"),
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        fieldnames = ["filename"]
        for task in TASKS:
            fieldnames.extend([
                f"{task}_actual",
                f"{task}_predicted",
                f"{task}_confidence",
                f"{task}_accepted",
            ])
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(records):
            row = {"filename": record.path.name}
            for task in TASKS:
                actual = getattr(record, task)
                predicted_index = metrics[task]["predictions"][index]
                row.update({
                    f"{task}_actual": actual,
                    f"{task}_predicted": label_codes[task][predicted_index],
                    f"{task}_confidence": metrics[task]["confidence"][index],
                    f"{task}_accepted": metrics[task]["accepted"][index],
                })
            writer.writerow(row)

    print(f"Evaluated {sample_count} images on {device}.")
    for task in TASKS:
        print(
            f"{task}: accuracy={metrics[task]['accuracy']:.1%} "
            f"macro_f1={metrics[task]['macro_f1']:.3f} "
            f"coverage={metrics[task]['coverage']:.1%}"
        )
    print(f"joint_accuracy={report['joint_accuracy']:.1%}")
    print("Saved:", os.path.join(args.output_dir, "evaluation.json"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on an independent labeled set."
    )
    parser.add_argument("--weight", default="outputs/atri_net_best.pth")
    parser.add_argument("--test_dir", default="atridataset/test")
    parser.add_argument("--labels", help="CSV with filename,outfit,pose,expression")
    parser.add_argument("--output_dir", default="evaluation")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    confidence_group = parser.add_mutually_exclusive_group()
    confidence_group.add_argument("--min_confidence", type=float)
    confidence_group.add_argument("--accept_all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
