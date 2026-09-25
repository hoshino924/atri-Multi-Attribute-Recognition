# -*- coding: utf-8 -*-
"""Compare matched resolution variants and face padding/upscaling, without training."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics

from PIL import Image
import torch
from torchvision.transforms import functional as TF

from calibration import apply_temperature, suggested_threshold
from dataset import SCALE_CODES, scan_records, split_content_keys
from face_locator import FaceLocator, FaceNotFoundError
from face_regions import FaceCanvas, SquareBox, crop_face, load_face_annotations
from infer import load_model
from labels import TASK_CODES


MODES = ("pad", "upscale")
SOURCES = ("locator", "reference")
TASKS = tuple(TASK_CODES)
BASE_FIELDS = (
    "content_id", "filename", "scale", "box_source", "mode", "status", "error",
    "image_width", "image_height", "x_left", "y_bottom", "side",
    "locator_confidence", "locator_reference_iou", "canvas_size",
    "rendered_side", "resize_factor", "added_padding_fraction",
)
TASK_FIELDS = (
    "actual", "predicted", "raw_confidence", "confidence", "target_confidence",
    "top2_code", "top2_confidence", "margin", "threshold", "accepted", "correct",
)


def content_id(record):
    return "_".join(record.content_key)


def read_split(path):
    """Resolve content identity, ignoring resolution and platform path separators."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"split manifest not found: {path}; specify its path or explicitly use --subset all")
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    return split_content_keys(manifest)


def match_records(records_by_scale, scales, selected_keys=None, locator_train_keys=()):
    """Require identical content coverage so every condition has the same denominator."""
    indexed = {}
    for scale in scales:
        records = records_by_scale[scale]
        indexed[scale] = {record.content_key: record for record in records}
        if len(indexed[scale]) != len(records):
            raise ValueError(f"duplicate content for scale {scale}")
    keys = set(selected_keys) if selected_keys is not None else set().union(*(set(items) for items in indexed.values()))
    if not keys:
        raise ValueError("no content selected")
    leaked = keys & set(locator_train_keys)
    if leaked:
        raise ValueError(f"{len(leaked)} selected validation contents were used to train the locator; use aligned splits")
    for scale in scales:
        missing = keys - indexed[scale].keys()
        if missing:
            example = "_".join(sorted(missing)[0])
            raise ValueError(f"scale {scale} is missing {len(missing)} matched contents, for example {example}")
    return [indexed[scale][key] for key in sorted(keys) for scale in scales]


def map_reference_box(box, source_size, target_size):
    """Proportionally map bottom-origin annotations, preserving a square crop."""
    if not box.fits(*source_size):
        raise ValueError("reference box does not fit its image")
    sx, sy = target_size[0] / source_size[0], target_size[1] / source_size[1]
    if abs(sx / sy - 1.0) > 0.02:
        raise ValueError("image aspect ratios differ by more than 2%; cannot use a proportional reference")
    side = max(1, min(*target_size, round(box.side * max(sx, sy))))
    left = round((box.x_left + box.side / 2) * sx - side / 2)
    bottom = round((box.y_bottom + box.side / 2) * sy - side / 2)
    return SquareBox(left, bottom, side).moved(0, 0, *target_size)


def square_iou(first, second):
    width = max(0, min(first.x_left + first.side, second.x_left + second.side) - max(first.x_left, second.x_left))
    height = max(0, min(first.y_bottom + first.side, second.y_bottom + second.side) - max(first.y_bottom, second.y_bottom))
    intersection = width * height
    return intersection / (first.side**2 + second.side**2 - intersection)


def face_view(image, box, size, background, mode):
    """Only change face occupancy; keep the same crop and alpha compositing."""
    if mode not in MODES:
        raise ValueError(f"unsupported diagnostic mode: {mode}")
    face = crop_face(image, box)
    if mode == "upscale" and box.side < size:
        face = face.resize((size, size), Image.Resampling.LANCZOS)
    preview = FaceCanvas(size, background)(face)
    rendered_side = size if mode == "upscale" else min(box.side, size)
    return preview, {
        "canvas_size": size, "rendered_side": rendered_side,
        "resize_factor": rendered_side / box.side,
        "added_padding_fraction": 1.0 - (rendered_side / size)**2,
    }


def prediction_fields(outputs, row_index, record, model, label_codes):
    fields = {}
    for task in TASKS:
        logits = outputs[task][row_index:row_index + 1].float()
        if not torch.isfinite(logits).all():
            raise ValueError(f"non-finite {task} output for {record.path.name}")
        raw = logits.softmax(dim=1)[0]
        probabilities = apply_temperature(logits, task, model.calibration).softmax(dim=1)[0]
        values, indices = probabilities.topk(2)
        predicted = label_codes[task][indices[0].item()]
        actual = getattr(record, task)
        threshold = suggested_threshold(task, model.calibration)
        confidence = values[0].item()
        result = {
            "actual": actual, "predicted": predicted,
            "raw_confidence": raw[indices[0]].item(), "confidence": confidence,
            "target_confidence": probabilities[label_codes[task].index(actual)].item(),
            "top2_code": label_codes[task][indices[1].item()],
            "top2_confidence": values[1].item(), "margin": (values[0] - values[1]).item(),
            "threshold": threshold, "accepted": threshold is None or confidence >= threshold,
            "correct": predicted == actual,
        }
        fields.update({f"{task}_{key}": value for key, value in result.items()})
    return fields


def summarize(rows):
    """Count localization/read failures as errors and rejections, not exclusions."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scale"], row["box_source"], row["mode"])].append(row)
    summaries = []
    for (scale, source, mode), group in grouped.items():
        successful = [row for row in group if row["status"] == "ok"]
        result = dict(scale=scale, box_source=source, mode=mode, samples=len(group),
                      predicted=len(successful), failures=len(group) - len(successful), tasks={})
        result["mean_added_padding_fraction"] = (
            statistics.mean(row["added_padding_fraction"] for row in successful) if successful else None
        )
        for task in TASKS:
            correct = [row for row in successful if row[f"{task}_correct"]]
            accepted = [row for row in successful if row[f"{task}_accepted"]]
            accepted_correct = [row for row in accepted if row[f"{task}_correct"]]
            confidence = [row[f"{task}_confidence"] for row in successful]
            result["tasks"][task] = {
                "top1_accuracy": len(correct) / len(group),
                "acceptance_rate": len(accepted) / len(group),
                "selective_accuracy": len(accepted_correct) / len(accepted) if accepted else None,
                "correct_but_rejected": len(correct) - len(accepted_correct),
                "accepted_but_wrong": len(accepted) - len(accepted_correct),
                "mean_confidence": statistics.mean(confidence) if confidence else None,
                "median_confidence": statistics.median(confidence) if confidence else None,
                "min_confidence": min(confidence) if confidence else None,
                "nll_on_localized_images": (
                    statistics.mean(-math.log(max(row[f"{task}_target_confidence"], 1e-12)) for row in successful)
                    if successful else None
                ),
            }
        summaries.append(result)
    return summaries


def paired_comparisons(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["content_id"], row["scale"], row["box_source"])][row["mode"]] = row
    result = []
    for (key, scale, source), pair in grouped.items():
        pad, upscale = pair["pad"], pair["upscale"]
        item = {"content_id": key, "filename": pad["filename"], "scale": scale, "box_source": source}
        for mode, row in pair.items():
            item[f"{mode}_status"] = row["status"]
            for field in ("predicted", "confidence", "accepted", "correct"):
                item[f"{mode}_{field}"] = row.get(f"expression_{field}")
        valid = pad["status"] == upscale["status"] == "ok"
        item["confidence_change"] = upscale["expression_confidence"] - pad["expression_confidence"] if valid else None
        item["correctness_change"] = int(upscale["expression_correct"]) - int(pad["expression_correct"]) if valid else None
        result.append(item)
    return result


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(out_dir, rows, summaries, pairs, metadata):
    percent = lambda value: "n/a" if value is None else f"{value:.2%}"
    expression_calibration = (metadata["calibration"] or {}).get("tasks", {}).get("expression", {})
    selection_note = (
        "Validation contents are excluded from both recorded training splits. This is a diagnostic validation set, not an independent test set."
        if metadata["subset"] == "validation" else
        "ALL includes training contents; these results diagnose preprocessing and must not be reported as held-out accuracy."
    )
    lines = ["# Matched-resolution face diagnostics", "",
             f"- Classifier: `{metadata['classifier']}`",
             f"- Locator: `{metadata['locator']}`",
             f"- Selection: {metadata['subset']}; {metadata['content_count']} distinct contents; {metadata['image_count']} resolution variants.",
             f"- {selection_note}",
             f"- Expression temperature: {expression_calibration.get('temperature', 1.0)}; acceptance threshold: {percent(suggested_threshold('expression', metadata['calibration']))}.",
             "- Each condition uses the same matched contents. Resolution variants are not independent artworks.",
             "- `pad` uses checkpoint preprocessing; `upscale` is an experiment with the same weights and thresholds, not newly calibrated probabilities.",
             "- `reference` uses proportional annotation mapping, approximate away from the annotated resolution. It is not independent localization.",
             "- Padding measures added canvas area only, not black/transparent pixels already inside the crop.",
             "- Top-1 accuracy and acceptance use all selected images as their denominator, including failures. Confidence/padding means use successful predictions only.", "",
             "## Expression results", "",
             "| Scale | Box | Mode | N | Failures | Top-1 | Accepted | Correct, rejected | Accepted, wrong | Mean confidence | Added padding |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in summaries:
        metrics = item["tasks"]["expression"]
        lines.append(f"| {item['scale']} | {item['box_source']} | {item['mode']} | {item['samples']} | {item['failures']} | "
                     f"{percent(metrics['top1_accuracy'])} | {percent(metrics['acceptance_rate'])} | {metrics['correct_but_rejected']} | "
                     f"{metrics['accepted_but_wrong']} | {percent(metrics['mean_confidence'])} | {percent(item['mean_added_padding_fraction'])} |")
    lines.extend(["", "## Paired upscale changes", "",
                  "| Scale | Box | Valid pairs | Wrong to correct | Correct to wrong | Newly accepted | Newly accepted, wrong |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for scale in metadata["scales"]:
        for source in SOURCES:
            selected = [item for item in pairs if item["scale"] == scale and item["box_source"] == source and item["confidence_change"] is not None]
            gained = sum(item["correctness_change"] == 1 for item in selected)
            lost = sum(item["correctness_change"] == -1 for item in selected)
            new = [item for item in selected if item["upscale_accepted"] and not item["pad_accepted"]]
            lines.append(f"| {scale} | {source} | {len(selected)} | {gained} | {lost} | {len(new)} | {sum(not item['upscale_correct'] for item in new)} |")
    difficult = [row for row in rows if row["status"] != "ok" or not row["expression_accepted"] or not row["expression_correct"]]
    difficult.sort(key=lambda row: (row["scale"] != "s", row.get("expression_confidence", -1)))
    lines.extend(["", "## First 20 difficult cases (s prioritized)", "",
                  "| File | Box / mode | Status | Actual | Predicted | Confidence | Threshold | Side |",
                  "|---|---|---|---|---|---|---:|---:|"])
    for row in difficult[:20]:
        filename = row["filename"].replace("|", "\\|")
        lines.append(f"| {filename} | {row['box_source']} / {row['mode']} | {row['status']} | {row['expression_actual']} | "
                     f"{row.get('expression_predicted', '')} | {percent(row.get('expression_confidence'))} | "
                     f"{percent(row.get('expression_threshold'))} | {row.get('side', '')} |")
    lines.extend(["", "All task metrics/configuration are in summary.json. Detailed errors and predictions are in predictions.csv; "
                  "paired mode changes in comparisons.csv; resolution variants side by side in by_content.csv.", ""])
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def diagnose(args):
    if len(set(args.scales)) != len(args.scales):
        raise ValueError("--scales must not contain duplicates")
    if args.locator_threshold is not None and not 0 < args.locator_threshold <= 1:
        raise ValueError("--locator_threshold must be in (0, 1]")
    if Path(args.output_dir).exists():
        raise FileExistsError(f"choose a new diagnostic output directory: {args.output_dir}")
    records_by_scale = {scale: scan_records(args.image_dir, scale) for scale in dict.fromkeys([*args.scales, args.reference_scale])}
    classifier_split_path = args.split or str(Path(args.weight).parent / "split.json")
    locator_split_path = args.locator_split or str(Path(args.locator_weight).parent / "split.json")
    selected_keys, locator_train_keys = None, ()
    if args.subset == "validation":
        selected_keys = read_split(classifier_split_path)["validation"]
        locator_train_keys = read_split(locator_split_path)["train"]
    records = match_records(records_by_scale, args.scales, selected_keys, locator_train_keys)
    reference_records = records_by_scale[args.reference_scale]
    annotations = load_face_annotations(reference_records, args.annotations)
    references = {}
    for record in reference_records:
        with Image.open(record.path) as source:
            references[record.content_key] = annotations[record.path.name], source.size
    missing = {record.content_key for record in records} - references.keys()
    if missing:
        raise ValueError(f"{len(missing)} selected contents have no reference annotation")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, transforms, label_codes = load_model(args.weight, device)
    if not hasattr(model, "face_preview"):
        raise ValueError("this diagnostic requires an annotated-face classifier (format 4)")
    locator = FaceLocator.from_checkpoint(args.locator_weight, device, args.locator_threshold)
    config = model.preprocess_config
    size, background = config["expression"]["size"], config["background"]
    mean, std = config["normalization"]["mean"], config["normalization"]["std"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    if args.save_previews:
        (out_dir / "previews").mkdir()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "classifier": str(args.weight),
        "locator": str(args.locator_weight), "annotations": str(args.annotations),
        "subset": args.subset, "scales": args.scales, "reference_scale": args.reference_scale,
        "classifier_split": classifier_split_path if args.subset == "validation" else None,
        "locator_split": locator_split_path if args.subset == "validation" else None,
        "content_count": len({record.content_key for record in records}), "image_count": len(records),
        "classifier_metadata": model.checkpoint_metadata,
        "preprocess": config, "calibration": model.calibration, "locator_threshold": locator.threshold,
        "python": platform.python_version(), "torch": str(torch.__version__), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "float32 (matching normal inference)", "arguments": vars(args),
    }
    (out_dir / "run_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    print(f"{metadata['content_count']} matched contents / {len(records)} images; device={device}; subset={args.subset}")
    with torch.inference_mode():
        for image_index, record in enumerate(records, start=1):
            base = {"content_id": content_id(record), "filename": record.path.name, "scale": record.scale}
            base.update({f"{task}_actual": getattr(record, task) for task in TASKS})
            try:
                with Image.open(record.path) as source:
                    image = source.convert("RGBA")
            except OSError as exc:
                rows.extend({**base, "box_source": source, "mode": mode, "status": "read_error", "error": str(exc)} for source in SOURCES for mode in MODES)
                continue
            base.update(image_width=image.width, image_height=image.height)
            boxes, errors = {}, {}
            try:
                boxes["locator"], base["locator_confidence"] = locator.locate(image)
            except FaceNotFoundError as exc:
                errors["locator"] = ("face_not_found", str(exc))
            try:
                ref_box, ref_size = references[record.content_key]
                boxes["reference"] = map_reference_box(ref_box, ref_size, image.size)
            except ValueError as exc:
                errors["reference"] = ("reference_error", str(exc))
            if len(boxes) == 2:
                base["locator_reference_iou"] = square_iou(boxes["locator"], boxes["reference"])
            pending, tensors = [], []
            for source in SOURCES:
                for mode in MODES:
                    row = {**base, "box_source": source, "mode": mode}
                    rows.append(row)
                    if source in errors:
                        row["status"], row["error"] = errors[source]
                        continue
                    box = boxes[source]
                    preview, geometry = face_view(image, box, size, background, mode)
                    row.update(status="ok", error="", x_left=box.x_left, y_bottom=box.y_bottom, side=box.side, **geometry)
                    if args.save_previews:
                        preview.save(out_dir / "previews" / f"{record.path.stem}_{source}_{mode}.png")
                    pending.append(row)
                    tensors.append(TF.normalize(TF.to_tensor(preview), mean, std))
            if tensors:
                # At most four views. The same full image and box are reused in each paired comparison.
                full = transforms["full"](image).unsqueeze(0).to(device)
                outputs = model(full.expand(len(tensors), -1, -1, -1), torch.stack(tensors).to(device))
                for index, row in enumerate(pending):
                    row.update(prediction_fields(outputs, index, record, model, label_codes))
            if image_index % 10 == 0 or image_index == len(records):
                print(f"Processed {image_index}/{len(records)} images", flush=True)
    summaries, pairs = summarize(rows), paired_comparisons(rows)
    fields = [*BASE_FIELDS, *(f"{task}_{field}" for task in TASKS for field in TASK_FIELDS)]
    write_csv(out_dir / "predictions.csv", rows, fields)
    write_csv(out_dir / "comparisons.csv", pairs, list(pairs[0]))
    wide = {}
    for row in rows:
        key = (row["content_id"], row["box_source"], row["mode"])
        item = wide.setdefault(key, {"content_id": key[0], "box_source": key[1], "mode": key[2], "expression_actual": row["expression_actual"]})
        for field in ("status", "side", "expression_predicted", "expression_confidence", "expression_accepted", "expression_correct"):
            item[f"{row['scale']}_{field}"] = row.get(field)
    write_csv(out_dir / "by_content.csv", list(wide.values()), list(next(iter(wide.values()))))
    (out_dir / "summary.json").write_text(json.dumps({"metadata": metadata, "conditions": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir, rows, summaries, pairs, metadata)
    print(f"Saved diagnostic report: {out_dir / 'report.md'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    from app_assets import DEFAULT_CLASSIFIER_WEIGHT, DEFAULT_LOCATOR_WEIGHT
    parser.add_argument("--weight", default=DEFAULT_CLASSIFIER_WEIGHT)
    parser.add_argument("--locator_weight", default=DEFAULT_LOCATOR_WEIGHT)
    parser.add_argument("--image_dir", default="atridataset/train")
    parser.add_argument("--annotations", default="annotations/face_boxes_l.csv")
    parser.add_argument("--reference_scale", choices=SCALE_CODES, default="l")
    parser.add_argument("--scales", choices=SCALE_CODES, nargs="+", default=list(SCALE_CODES))
    parser.add_argument("--subset", choices=("validation", "all"), default="validation",
                        help="validation reuses both recorded splits; all also includes training contents")
    parser.add_argument("--split", help="classifier split.json; defaults to checkpoint directory")
    parser.add_argument("--locator_split", help="locator split.json; defaults to checkpoint directory")
    parser.add_argument("--locator_threshold", type=float)
    parser.add_argument("--output_dir", default="evaluation/scale_diagnostics_smallface")
    parser.add_argument("--save_previews", action="store_true", help="write diagnostic crops to the new output directory")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    diagnose(parse_args())
