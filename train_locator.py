# -*- coding: utf-8 -*-
"""Train a single-face locator from the project's manual square annotations."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from dataset import scan_records, split_content_keys, stratified_split
from face_box_cache import file_sha256
from face_locator import FaceLocatorNet, LOCATOR_VERSION
from face_regions import annotation_signature, load_face_annotations
from locator_data import LocatorDataset
from train import dataset_signature, seed_everything
from image_rotation import DEFAULT_ANGLES, ROTATION_CONTRACT, angle_text
from rotation_data import ContentAngleSampler, angle_policy, load_confirmed_rotation_annotations, rotation_records


def box_iou(predicted, target):
    p0, p1 = predicted[:, :2] - predicted[:, 2:] / 2, predicted[:, :2] + predicted[:, 2:] / 2
    t0, t1 = target[:, :2] - target[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2
    overlap = (torch.minimum(p1, t1) - torch.maximum(p0, t0)).clamp_min(0).prod(dim=1)
    union = predicted[:, 2:].prod(dim=1) + target[:, 2:].prod(dim=1) - overlap
    return overlap / union.clamp_min(1e-8)


def new_totals():
    return dict(positive_iou=0., localized_iou=0., true_positive=0,
                false_positive=0, positives=0, negatives=0, count=0)


def accumulate(totals, positive, detected, iou):
    totals["positive_iou"] += iou[positive].sum().item()
    totals["localized_iou"] += (iou[positive] * detected[positive]).sum().item()
    totals["true_positive"] += (detected & positive).sum().item()
    totals["false_positive"] += (detected & ~positive).sum().item()
    totals["positives"] += positive.sum().item()
    totals["negatives"] += (~positive).sum().item()
    totals["count"] += positive.numel()


def summarize(totals):
    return {
        "positive_iou": totals["positive_iou"] / max(1, totals["positives"]),
        "localized_iou": totals["localized_iou"] / max(1, totals["positives"]),
        "face_recall": totals["true_positive"] / max(1, totals["positives"]),
        "false_positive_rate": totals["false_positive"] / max(1, totals["negatives"]),
        "positives": totals["positives"], "negatives": totals["negatives"], "samples": totals["count"],
    }


def run_epoch(model, loader, device, threshold, optimizer=None, scaler=None):
    model.train(optimizer is not None)
    totals, groups = new_totals(), {}
    loss_sum, steps, skipped = 0., 0, 0
    for batch in loader:
        images, presence, target = batch[:3]
        metadata = batch[3] if len(batch) == 4 else None
        images, presence, target = images.to(device), presence.to(device), target.to(device)
        positive = presence.bool()
        with torch.set_grad_enabled(optimizer is not None):
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits, boxes = model(images)
                loss = F.binary_cross_entropy_with_logits(logits, presence)
                if positive.any():
                    loss = loss + 10 * F.smooth_l1_loss(boxes[positive], target[positive], beta=0.05)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    skipped += 1
                else:
                    steps += 1
        detected = logits.detach().sigmoid() >= threshold
        iou = box_iou(boxes.detach().float(), target)
        loss_sum += loss.item() * len(images)
        accumulate(totals, positive, detected, iou)
        if metadata is not None:
            angles = metadata["angle"].to(device)
            for angle in angles.unique().tolist():
                mask = angles == angle
                group = groups.setdefault(angle_text(angle), new_totals())
                accumulate(group, positive[mask], detected[mask], iou[mask])
    if not totals["count"]:
        raise ValueError("empty locator loader")
    return {**summarize(totals), "loss": loss_sum / totals["count"],
            "optimizer_steps": steps, "skipped_optimizer_steps": skipped,
            "by_angle": {key: summarize(value) for key, value in groups.items()}}


def selection_score(metrics, policy=None):
    def score(values):
        return values["localized_iou"] - 0.1 * values["false_positive_rate"]
    if policy is None:
        return score(metrics)
    return sum(weight * score(metrics["by_angle"][angle_text(angle)])
               for angle, weight in zip(policy["angles"], policy["weights"]))


def save_locator_previews(dataset, output_dir, per_angle=3):
    """Write clean/negative/synthetic inputs with auditable target overlays."""
    directory = Path(output_dir) / "locator_previews"
    directory.mkdir(exist_ok=False)
    counts, rows = {}, []
    for index, record in enumerate(dataset.records):
        angle = float(getattr(record, "angle", 0))
        if counts.get(angle, 0) >= per_angle:
            continue
        counts[angle] = counts.get(angle, 0) + 1
        for sample in range(4):
            tensor, presence, target = dataset[index * dataset.samples_per_image + sample][:3]
            image = TF.to_pil_image(tensor)
            number = len(rows) + 1
            image.save(directory / f"{number:04d}_input.png")
            if presence.item():
                cx, cy, width, height = (float(value) * dataset.size for value in target)
                ImageDraw.Draw(image).rectangle((cx - width / 2, cy - height / 2,
                                                cx + width / 2, cy + height / 2), outline="lime", width=2)
            image.save(directory / f"{number:04d}_target.png")
            rows.append({"index": number, "filename": record.path.name, "angle": angle,
                         "sample": sample, "kind": "clean" if sample == 0 else ("negative" if sample == 1 else "synthetic"),
                         "present": bool(presence.item()), "target_cx_cy_w_h": target.tolist()})
    (Path(output_dir) / "locator_preview.json").write_text(
        json.dumps({"samples": rows, "note": "Actual locator inputs; green is the training target. Sources unchanged."},
                   ensure_ascii=False, indent=2), encoding="utf-8")


def train(args):
    if min(args.epochs, args.batch, args.samples_per_image) <= 0 or args.input_size < 32:
        raise ValueError("epochs, batch and samples_per_image must be positive; input_size >= 32")
    if args.samples_per_image < 4 or args.workers < 0 or not 0 < args.threshold <= 1 or args.lr <= 0:
        raise ValueError("need >= 4 samples/image, workers >= 0, lr > 0 and threshold in (0, 1]")
    seed_everything(args.seed)
    records = scan_records(args.train_dir, args.scale)
    rotated = getattr(args, "rotated_annotations", False)
    policy = None
    if rotated:
        policy = angle_policy(args.angles or DEFAULT_ANGLES, args.angle_weights)
        _, boxes, manifest = load_confirmed_rotation_annotations(records, args.annotations, policy["angles"])
        args.rotation_policy = {**policy, "rotation": ROTATION_CONTRACT,
                                "pillow_version": manifest["pillow_version"], "synthesis_version": 2}
    else:
        if getattr(args, "angles", None) or getattr(args, "angle_weights", None):
            raise ValueError("--angles/--angle_weights require --rotated_annotations")
        boxes = load_face_annotations(records, args.annotations)
        args.rotation_policy = None
    for record in records:
        with Image.open(record.path) as source:
            alpha = source.convert("RGBA").getchannel("A")
            if alpha.getextrema()[0] == 255:
                raise ValueError(f"locator synthesis requires transparent sprites: {record.path.name}")
    signature = dataset_signature(records, args.train_dir) + ":" + annotation_signature(args.annotations)
    if rotated:
        signature += ":" + file_sha256(Path(args.annotations).with_suffix(".meta.json"))
    train_records, val_records = stratified_split(records, args.val_ratio, args.seed)
    if getattr(args, "reference_split", None):
        reference = split_content_keys(json.loads(Path(args.reference_split).read_text(encoding="utf-8-sig")))
        for name, part in (("train", train_records), ("validation", val_records)):
            if {record.content_key for record in part} != reference[name]:
                raise ValueError(f"{name} contents differ from --reference_split")
        args.reference_split_sha256 = file_sha256(args.reference_split)
    source_train, source_val = train_records, val_records
    if rotated:
        train_records = rotation_records(source_train, policy["angles"])
        val_records = rotation_records(source_val, policy["angles"])
    datasets = [LocatorDataset(part, boxes, args.input_size, args.samples_per_image, seed,
                               include_metadata=rotated)
                for part, seed in ((train_records, args.seed), (val_records, args.seed + 100000000))]
    sampler = ContentAngleSampler(train_records, policy, args.samples_per_image, args.seed) if rotated else None
    args.sampling_metadata = sampler.metadata(args.batch) if sampler else None
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    history, start_epoch, best = [], 0, float("-inf")
    config = vars(args).copy()
    checkpoint = None
    if args.resume:
        from infer import load_torch_checkpoint
        checkpoint = load_torch_checkpoint(args.resume, device)
        if checkpoint.get("locator_version") != LOCATOR_VERSION or checkpoint.get("dataset_signature") != signature:
            raise ValueError("locator resume data or checkpoint format differs")
        for key in ("input_size", "samples_per_image", "seed", "val_ratio", "scale", "threshold", "lr"):
            if checkpoint["train_args"][key] != config[key]:
                raise ValueError(f"locator resume setting differs: {key}")
        if checkpoint["train_args"].get("rotation_policy") != args.rotation_policy:
            raise ValueError("locator resume rotation/synthesis policy differs")
        if rotated:
            for key in ("batch", "sampling_metadata"):
                if checkpoint["train_args"].get(key) != config[key]:
                    raise ValueError(f"locator resume setting differs: {key}")
        if not {"optimizer_state", "scaler_state", "history"}.issubset(checkpoint):
            raise ValueError("resume requires face_locator_last.pth with complete training state")
        start_epoch, best, history = checkpoint["epoch"], checkpoint["best_score"], checkpoint["history"]
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.resume or Path(args.resume).resolve().parent != out_dir.resolve():
            raise FileExistsError("choose a new locator output directory, or resume its last checkpoint")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    split = {"train": [r.path.name for r in source_train], "validation": [r.path.name for r in source_val],
             "dataset_signature": signature, "seed": args.seed, "val_ratio": args.val_ratio,
             "rotation_policy": args.rotation_policy}
    (out_dir / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    if sampler:
        preview = {"sampling": args.sampling_metadata,
                   "note": "Planned first complete angle cycle, not executed training.",
                   "validation": {"views": len(val_records), "samples": len(datasets[1]),
                                  "angles": policy["angles"], "distribution": "all angles; score weighted by training policy"},
                   "epochs": [sampler.epoch_summary(epoch, include_views=True) for epoch in range(policy["cycle_epochs"])]}
        (out_dir / "sampling_preview.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
    if getattr(args, "preview_only", False):
        save_locator_previews(datasets[0], out_dir)
        print(f"Saved locator previews, validated split and sampling plan: {out_dir}; no model/optimizer created")
        return
    model = FaceLocatorNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        del checkpoint
    loaders = [DataLoader(data, batch_size=args.batch, shuffle=index == 0 and sampler is None,
                          sampler=sampler if index == 0 else None, num_workers=args.workers,
                          pin_memory=device.type == "cuda") for index, data in enumerate(datasets)]
    print(f"Device: {device}; source split: {len(source_train)} train / {len(source_val)} validation")
    if sampler:
        print("Rotation sampling:", args.sampling_metadata)
    print("Validation covers synthetic scenes from held-out records; test real images separately.")
    for epoch in range(start_epoch, args.epochs):
        torch.manual_seed(args.seed + epoch)
        datasets[0].epoch = epoch
        if sampler:
            sampler.set_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics = run_epoch(model, loaders[0], device, args.threshold, optimizer, scaler)
        val_metrics = run_epoch(model, loaders[1], device, args.threshold)
        if sampler:
            planned = sampler.epoch_summary(epoch)
            actual = {key: value["samples"] for key, value in train_metrics["by_angle"].items()}
            if any(actual.get(key, 0) != count for key, count in planned["angle_sample_counts"].items()):
                raise RuntimeError("executed locator angle counts differ from the sampler")
            train_metrics["sampling"] = planned
        score = selection_score(val_metrics, policy)
        improved = score > best
        best = max(best, score)
        memory = ({"allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                   "reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20} if device.type == "cuda" else None)
        history.append({"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics,
                        "selection_score": score, "cuda_peak_memory": memory})
        checkpoint = {"locator_version": LOCATOR_VERSION, "model_state": model.state_dict(),
                      "input_size": args.input_size, "threshold": args.threshold, "epoch": epoch + 1,
                      "dataset_signature": signature, "train_args": config, "best_score": best}
        checkpoint["rotation_policy"] = args.rotation_policy
        if improved:
            torch.save(checkpoint, out_dir / "face_locator_best.pth")
        checkpoint.update(optimizer_state=optimizer.state_dict(), scaler_state=scaler.state_dict(), history=history)
        temporary = out_dir / "face_locator_last.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(out_dir / "face_locator_last.pth")
        (out_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={train_metrics['loss']:.4f}, val={val_metrics}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_dir", default="atridataset/train")
    parser.add_argument("--annotations", help="annotation CSV; defaults to the upright or rotated table")
    parser.add_argument("--scale", default="l", choices=("s", "w", "m", "l", "ll"))
    parser.add_argument("--out_dir")
    parser.add_argument("--rotated_annotations", action="store_true", help="read confirmed rotation CSV + .meta.json")
    parser.add_argument("--angles", type=float, nargs="+", help="default six angles with rotated annotations")
    parser.add_argument("--angle_weights", type=float, nargs="+", help="weights paired with --angles; default 45/20/5/5/5/20 percent")
    parser.add_argument("--reference_split", help="require exactly the same content partitions as this previous split.json")
    parser.add_argument("--preview_only", action="store_true", help="validate sources and export training inputs, without allocating a model")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--samples_per_image", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.annotations is None:
        args.annotations = ("annotations/face_boxes_l_rotated.csv" if args.rotated_annotations
                            else "annotations/face_boxes_l.csv")
    if args.preview_only and args.resume:
        parser.error("preview_only requires a fresh directory and cannot resume")
    if args.out_dir is None:
        args.out_dir = "outputs/face_locator_rotated_l" if args.rotated_annotations else "outputs/face_locator_l"
    return args


if __name__ == "__main__":
    train(parse_args())
