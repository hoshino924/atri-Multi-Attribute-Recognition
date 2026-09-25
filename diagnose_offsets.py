# -*- coding: utf-8 -*-
"""Evaluate a fixed face-box offset grid with normal deterministic preprocessing."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform

from PIL import Image, ImageDraw
import torch
from torchvision.transforms import functional as TF

from dataset import SCALE_CODES, scan_records
from diagnose_scales import (
    BASE_FIELDS, TASK_FIELDS, TASKS, content_id, face_view, match_records,
    prediction_fields, read_split, summarize, write_csv,
)
from face_locator import FaceLocator, FaceNotFoundError
from face_regions import annotation_signature, offset_face_box
from infer import load_model


OFFSET_FIELDS = (
    "requested_dx", "requested_dy", "actual_dx", "actual_dy", "source_dx", "source_dy",
    "output_scale", "base_x_left", "base_y_bottom", "base_side",
)


def offset_grid(values):
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("offsets must be finite and nonempty")
    if len(set(values)) != len(values) or 0 not in values:
        raise ValueError("offsets must be unique and include zero for the paired baseline")
    return [(dx, dy) for dy in values for dx in values]


def offset_name(dx, dy):
    return f"dx={dx:.17g},dy={dy:.17g}"


def summarize_offsets(rows):
    """Keep infeasible geometry visible and compare feasible predictions to zero."""
    summaries = summarize(rows)
    grouped = defaultdict(list)
    baseline = {}
    for row in rows:
        grouped[(row["scale"], row["mode"])].append(row)
        if row["requested_dx"] == row["requested_dy"] == 0:
            baseline[row["filename"]] = row
    for item in summaries:
        selected = grouped[(item["scale"], item["mode"])]
        successful = [row for row in selected if row["status"] == "ok"]
        item.update(requested_dx=selected[0]["requested_dx"], requested_dy=selected[0]["requested_dy"],
                    status_counts=dict(Counter(row["status"] for row in selected)),
                    out_of_bounds=sum(row["status"] == "out_of_bounds" for row in selected),
                    joint_accuracy=sum(all(row[f"{task}_correct"] for task in TASKS) for row in successful) / len(selected))
        paired = [row for row in successful if baseline[row["filename"]]["status"] == "ok"]
        item["valid_pairs_to_zero"] = len(paired)
        for task in TASKS:
            metrics = item["tasks"][task]
            metrics["accuracy_on_feasible_predictions"] = (
                sum(row[f"{task}_correct"] for row in successful) / len(successful) if successful else None)
            metrics["correct_to_wrong_from_zero"] = sum(
                baseline[row["filename"]][f"{task}_correct"] and not row[f"{task}_correct"] for row in paired)
            metrics["wrong_to_correct_from_zero"] = sum(
                not baseline[row["filename"]][f"{task}_correct"] and row[f"{task}_correct"] for row in paired)
            metrics["prediction_changed_from_zero"] = sum(
                baseline[row["filename"]][f"{task}_predicted"] != row[f"{task}_predicted"] for row in paired)
    return summaries


def image_stability(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["filename"]].append(row)
    results = []
    for filename, group in grouped.items():
        successful = [row for row in group if row["status"] == "ok"]
        base = next(row for row in group if row["requested_dx"] == row["requested_dy"] == 0)
        item = {"filename": filename, "content_id": group[0]["content_id"], "scale": group[0]["scale"],
                "conditions": len(group), "predicted": len(successful), "zero_status": base["status"],
                "out_of_bounds": sum(row["status"] == "out_of_bounds" for row in group)}
        for task in TASKS:
            correct = sum(row[f"{task}_correct"] for row in successful)
            item.update({
                f"{task}_zero_correct": base.get(f"{task}_correct"),
                f"{task}_correct": correct,
                f"{task}_all_requests_correct": len(successful) == len(group) and correct == len(group),
                f"{task}_all_feasible_correct": bool(successful) and correct == len(successful),
                f"{task}_accepted_wrong": sum(row[f"{task}_accepted"] and not row[f"{task}_correct"] for row in successful),
                f"{task}_correct_rejected": sum(row[f"{task}_correct"] and not row[f"{task}_accepted"] for row in successful),
                f"{task}_distinct_predictions": len({row[f"{task}_predicted"] for row in successful}),
            })
        results.append(item)
    return results


def save_report(out, summaries, metadata, stability):
    percent = lambda value: "n/a" if value is None else f"{value:.2%}"
    lines = ["# Fixed face-offset diagnostics", "",
             f"- Classifier: `{metadata['classifier']}`; locator: `{metadata['locator']}`.",
             f"- {metadata['content_count']} validation contents / {metadata['image_count']} resolution variants / {len(metadata['offset_grid'])} offsets per image.",
             "- These are development-validation contents used for model selection, not an independent test set.",
             "- Only the face crop moves. Full image, box side, preprocessing, calibration and thresholds stay fixed.",
             "- Offsets are face-input pixels: positive x moves the box right, positive y up; visible content moves oppositely.",
             "- Source offsets use round-to-even. Requested and realized values are in predictions.csv.",
             "- Out-of-bounds requests are reported without clamping, shrinking or falling back. All-request accuracy counts them as failures; feasible accuracy excludes them and must be read with its coverage.",
             "- Zero offset is the ordinary inference baseline. No random augmentation or recalibration is applied.", "",
             "| Scale | dx | dy | N | Predicted | Out of bounds | Expression all-request | Expression feasible | Correct to wrong | Wrong accepted | Correct rejected | Outfit | Pose |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in summaries:
        metrics = item["tasks"]["expression"]
        lines.append(f"| {item['scale']} | {item['requested_dx']:g} | {item['requested_dy']:g} | {item['samples']} | "
                     f"{item['predicted']} | {item['out_of_bounds']} | {percent(metrics['top1_accuracy'])} | "
                     f"{percent(metrics['accuracy_on_feasible_predictions'])} | {metrics['correct_to_wrong_from_zero']} | "
                     f"{metrics['accepted_but_wrong']} | {metrics['correct_but_rejected']} | "
                     f"{percent(item['tasks']['outfit']['top1_accuracy'])} | {percent(item['tasks']['pose']['top1_accuracy'])} |")
    lines.extend(["", "## Expression stability by image", "",
                  "| Scale | Images | All requests correct | All feasible requests correct | Images with infeasible offsets |",
                  "|---|---:|---:|---:|---:|"])
    for scale in metadata["scales"]:
        selected = [row for row in stability if row["scale"] == scale]
        lines.append(f"| {scale} | {len(selected)} | {sum(row['expression_all_requests_correct'] for row in selected)} | "
                     f"{sum(row['expression_all_feasible_correct'] for row in selected)} | {sum(row['out_of_bounds'] > 0 for row in selected)} |")
    lines.extend(["", "Detailed predictions: predictions.csv. Per-image robustness: by_image.csv. "
                  "All three tasks, paired changes and joint accuracy: summary.json. Conditions and scales reuse the same content.", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def diagnose(args):
    grid = offset_grid(args.offsets)
    if len(set(args.scales)) != len(args.scales):
        raise ValueError("scales must be unique")
    if args.batch < 1 or args.preview_images < 0:
        raise ValueError("batch must be positive and preview_images nonnegative")
    if args.locator_threshold is not None and not 0 < args.locator_threshold <= 1:
        raise ValueError("locator_threshold must be in (0, 1]")
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f"choose a new diagnostic output directory: {out}")
    classifier_split = Path(args.split or Path(args.weight).parent / "split.json")
    locator_split = Path(args.locator_split or Path(args.locator_weight).parent / "split.json")
    selected = read_split(classifier_split)["validation"]
    records = match_records({scale: scan_records(args.image_dir, scale) for scale in args.scales},
                            args.scales, selected, read_split(locator_split)["train"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, transforms, label_codes = load_model(args.weight, device)
    if not hasattr(model, "face_preview"):
        raise ValueError("fixed offsets require an annotated-face classifier")
    locator = FaceLocator.from_checkpoint(args.locator_weight, device, args.locator_threshold)
    config = model.preprocess_config
    size, background = config["expression"]["size"], config["background"]
    mean, std = config["normalization"]["mean"], config["normalization"]["std"]
    out.mkdir(parents=True, exist_ok=False)
    if args.preview_images:
        (out / "previews").mkdir()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "classifier": str(args.weight),
        "locator": str(args.locator_weight), "classifier_sha256": annotation_signature(args.weight),
        "locator_sha256": annotation_signature(args.locator_weight),
        "classifier_split": str(classifier_split), "locator_split": str(locator_split),
        "classifier_split_sha256": annotation_signature(classifier_split),
        "locator_split_sha256": annotation_signature(locator_split),
        "classifier_metadata": model.checkpoint_metadata, "preprocess": config, "calibration": model.calibration,
        "locator_threshold": locator.threshold, "subset": "validation", "scales": args.scales,
        "offset_grid": [{"dx": dx, "dy": dy} for dx, dy in grid],
        "content_count": len(selected), "image_count": len(records),
        "precision": "float32 (matching normal inference)", "device": str(device),
        "python": platform.python_version(), "torch": str(torch.__version__), "arguments": vars(args),
    }
    (out / "run_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    rows, preview_counts = [], Counter()
    with torch.inference_mode():
        for index, record in enumerate(records, 1):
            base = {"filename": record.path.name, "content_id": content_id(record), "scale": record.scale,
                    "box_source": "locator", **{f"{task}_actual": getattr(record, task) for task in TASKS}}
            image_rows = [{**base, "requested_dx": dx, "requested_dy": dy, "mode": offset_name(dx, dy)} for dx, dy in grid]
            rows.extend(image_rows)
            try:
                with Image.open(record.path) as source:
                    image = source.convert("RGBA")
            except OSError as exc:
                for row in image_rows:
                    row.update(status="read_error", error=str(exc))
                continue
            try:
                box, confidence = locator.locate(image)
            except FaceNotFoundError as exc:
                for row in image_rows:
                    row.update(status="face_not_found", error=str(exc))
                continue
            sheet = None
            if preview_counts[record.scale] < args.preview_images:
                sheet = Image.new("RGB", (len(args.offsets) * size, len(args.offsets) * (size + 36)), (35, 39, 46))
                preview_counts[record.scale] += 1
            pending, tensors = [], []
            full = transforms["full"](image).unsqueeze(0).to(device)
            for position, row in enumerate(image_rows):
                changed, geometry = offset_face_box(box, *image.size, row["requested_dx"], row["requested_dy"], min(1.0, size / box.side))
                row.update(geometry, image_width=image.width, image_height=image.height,
                           base_x_left=box.x_left, base_y_bottom=box.y_bottom, base_side=box.side,
                           locator_confidence=confidence)
                if changed is None:
                    row.update(status="out_of_bounds", error="requested crop leaves source image", actual_dx=None, actual_dy=None)
                    if sheet:
                        ImageDraw.Draw(sheet).text(((position % len(args.offsets)) * size + 4,
                                                   (position // len(args.offsets)) * (size + 36) + 4),
                                                  f"{row['mode']} OUT OF BOUNDS", fill="orange")
                    continue
                preview, fitted = face_view(image, changed, size, background, "pad")
                row.update(status="ok", error="", x_left=changed.x_left, y_bottom=changed.y_bottom, side=changed.side, **fitted)
                if sheet:
                    x, y = (position % len(args.offsets)) * size, (position // len(args.offsets)) * (size + 36)
                    sheet.paste(preview, (x, y + 36))
                    ImageDraw.Draw(sheet).text((x + 4, y + 4), row["mode"], fill="white")
                pending.append(row)
                tensors.append(TF.normalize(TF.to_tensor(preview), mean, std))
            for start in range(0, len(tensors), args.batch):
                batch = torch.stack(tensors[start:start + args.batch]).to(device)
                outputs = model(full.expand(len(batch), -1, -1, -1), batch)
                for batch_index, row in enumerate(pending[start:start + args.batch]):
                    row.update(prediction_fields(outputs, batch_index, record, model, label_codes))
            if sheet:
                sheet.save(out / "previews" / f"{record.path.stem}.png")
            if index % 10 == 0 or index == len(records):
                print(f"Processed {index}/{len(records)} images ({len(grid)} offsets each)", flush=True)
    summaries, stability = summarize_offsets(rows), image_stability(rows)
    write_csv(out / "predictions.csv", rows, [*BASE_FIELDS, *OFFSET_FIELDS,
              *(f"{task}_{field}" for task in TASKS for field in TASK_FIELDS)])
    write_csv(out / "by_image.csv", stability, list(stability[0]))
    (out / "summary.json").write_text(json.dumps({"metadata": metadata, "conditions": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    save_report(out, summaries, metadata, stability)
    print(f"Saved offset diagnostics: {out / 'report.md'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", required=True)
    from app_assets import DEFAULT_LOCATOR_WEIGHT
    parser.add_argument("--locator_weight", default=DEFAULT_LOCATOR_WEIGHT)
    parser.add_argument("--image_dir", default="atridataset/train")
    parser.add_argument("--split", help="classifier split.json; defaults to checkpoint directory")
    parser.add_argument("--locator_split", help="locator split.json; defaults to checkpoint directory")
    parser.add_argument("--locator_threshold", type=float)
    parser.add_argument("--scales", choices=SCALE_CODES, nargs="+", default=list(SCALE_CODES))
    parser.add_argument("--offsets", type=float, nargs="+", default=[-25, -10, 0, 10, 25])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--preview_images", type=int, default=0, help="save offset contact sheets for this many images per scale")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    diagnose(parse_args())
