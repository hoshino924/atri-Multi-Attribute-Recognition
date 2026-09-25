# -*- coding: utf-8 -*-
"""Evaluate a fixed locator on matched original sprites rotated before inference."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

from PIL import Image, ImageDraw, ImageOps
import torch

from dataset import SCALE_CODES, scan_records, split_content_keys
from diagnose_scales import map_reference_box, match_records
from face_box_cache import file_sha256
from face_locator import FaceLocator, FaceNotFoundError
from face_regions import FaceCanvas, SquareBox, crop_face, locator_canvas
from image_rotation import DEFAULT_ANGLES, ROTATION_CONTRACT, angle_text, parse_angles, rotate_image, rotation_geometry
from rotation_data import load_confirmed_rotation_annotations


FIELDS = ("filename", "content_id", "scale", "angle", "status", "error", "image_width", "image_height",
          "confidence", "x_left", "y_bottom", "side", "reference_kind", "reference_x_left",
          "reference_y_bottom", "reference_side", "iou", "center_error_relative", "side_error_relative", "preview")


def box_errors(box, reference):
    overlap_w = max(0, min(box.x_left + box.side, reference.x_left + reference.side)
                    - max(box.x_left, reference.x_left))
    overlap_h = max(0, min(box.y_bottom + box.side, reference.y_bottom + reference.side)
                    - max(box.y_bottom, reference.y_bottom))
    intersection = overlap_w * overlap_h
    dx = box.x_left + box.side / 2 - reference.x_left - reference.side / 2
    dy = box.y_bottom + box.side / 2 - reference.y_bottom - reference.side / 2
    return {"iou": intersection / (box.side ** 2 + reference.side ** 2 - intersection),
            "center_error_relative": math.hypot(dx, dy) / reference.side,
            "side_error_relative": (box.side - reference.side) / reference.side}


def summarize_rows(rows, iou_threshold):
    localized = [row for row in rows if row["status"] == "ok"]
    references = [row for row in rows if row["reference_kind"] != "none"]
    measured = [row for row in references if row["status"] == "ok"]
    good = sum(row["iou"] >= iou_threshold for row in measured)
    return {
        "samples": len(rows), "localized": len(localized), "failures": len(rows) - len(localized),
        "detection_rate": len(localized) / len(rows) if rows else 0,
        "reference_kind": sorted({row["reference_kind"] for row in rows}),
        "reference_samples": len(references), "measured_boxes": len(measured),
        "iou_threshold": iou_threshold, "localized_at_iou_threshold": good,
        "localization_rate_at_iou_threshold": good / len(references) if references else None,
        "mean_iou_localized": statistics.mean(row["iou"] for row in measured) if measured else None,
        "mean_iou_all_references": sum(row["iou"] for row in measured) / len(references) if references else None,
        "mean_center_error_relative": statistics.mean(row["center_error_relative"] for row in measured) if measured else None,
        "mean_absolute_side_error_relative": statistics.mean(abs(row["side_error_relative"]) for row in measured) if measured else None,
    }


def reference_for_view(record, angle, target_size, references, derived):
    item = references.get((record.content_key, angle))
    if item is None:
        return None, "none"
    box, size, filename = item
    if filename == record.path.name:
        if target_size != size:
            raise ValueError("manual reference dimensions changed")
        return box, "manual"
    if derived:
        return map_reference_box(box, size, target_size), "scaled_reference"
    return None, "none"


def save_preview(row, image, path, locator_size):
    """Show source, exact locator input and exact unnormalized 300px face input."""
    canvas = Image.new("RGBA", image.size, (0, 0, 0, 255))
    canvas.alpha_composite(image)
    display = ImageOps.contain(canvas.convert("RGB"), (400, 460))
    drawing = ImageDraw.Draw(display)
    boxes = []
    if row["reference_kind"] != "none" and "reference_side" in row:
        boxes.append((SquareBox(row["reference_x_left"], row["reference_y_bottom"], row["reference_side"]),
                      "lime" if row["reference_kind"] == "manual" else "yellow"))
    if row["status"] == "ok":
        boxes.append((SquareBox(row["x_left"], row["y_bottom"], row["side"]), "red"))
    for box, color in boxes:
        left, top, right, bottom = box.pil_bounds(image.height)
        drawing.rectangle((left * display.width / image.width, top * display.height / image.height,
                           right * display.width / image.width, bottom * display.height / image.height),
                          outline=color, width=2)
    locator_input, _ = locator_canvas(image, locator_size)
    # Preserve exact input PNG separately; the comparison may fit large inputs.
    locator_input.save(path.with_name(path.stem + "_locator_input.png"))
    card = Image.new("RGB", (1000, 520), (35, 39, 46))
    card.paste(display, ((410 - display.width) // 2, 42))
    card.paste(ImageOps.contain(locator_input, (280, 460)), (410, 42))
    if row["status"] == "ok":
        face = FaceCanvas(300)(crop_face(image, boxes[-1][0]))
        face.save(path.with_name(path.stem + "_face_input.png"))
        card.paste(face, (700, 42))
    draw = ImageDraw.Draw(card)
    draw.text((8, 8), f"{row['scale']} angle={angle_text(row['angle'])} | red=predicted; green=manual; yellow=derived", fill="white")
    draw.text((410, 25), "Locator input", fill="white")
    draw.text((700, 25), "Face input (300 px)", fill="white")
    draw.text((8, 503), f"status={row['status']} reference={row['reference_kind']} IoU={row.get('iou', '')}", fill="white")
    card.save(path)


def diagnose(args):
    if len(set(args.scales)) != len(args.scales) or not args.scales:
        raise ValueError("choose distinct resolution scales")
    angles = parse_angles(args.angles)
    if not 0 < args.iou_threshold <= 1 or args.preview_per_group < 0:
        raise ValueError("iou_threshold must be in (0, 1]; preview_per_group must be nonnegative")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"choose a new diagnostics directory: {output}")
    weight = Path(args.locator_weight)
    split_path = Path(args.locator_split) if args.locator_split else weight.parent / "split.json"
    manifest = json.loads(split_path.read_text(encoding="utf-8-sig"))
    partitions = split_content_keys(manifest)
    checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
    if not checkpoint.get("dataset_signature") or checkpoint["dataset_signature"] != manifest.get("dataset_signature"):
        raise ValueError("locator checkpoint and split signature differ")
    for key in ("seed", "val_ratio"):
        if checkpoint.get("train_args", {}).get(key) != manifest.get(key):
            raise ValueError(f"locator checkpoint and split disagree on {key}")
    policy = checkpoint.get("rotation_policy")
    if policy and policy.get("rotation") != ROTATION_CONTRACT:
        raise ValueError("locator uses a different rotation contract")
    epoch = checkpoint.get("epoch")
    del checkpoint
    selected = match_records({scale: scan_records(args.train_dir, scale) for scale in args.scales},
                             args.scales, partitions["validation"], partitions["train"])
    sources = scan_records(args.train_dir, args.reference_scale)
    annotation_path = Path(args.annotations)
    annotation_meta = json.loads(annotation_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    annotation_views, boxes, annotation_contract = load_confirmed_rotation_annotations(
        sources, annotation_path, annotation_meta["angles"])
    source_sizes = {name: (value["width"], value["height"]) for name, value in annotation_contract["sources"].items()}
    references = {(view.content_key, view.angle):
                  (boxes[view.view_key], rotation_geometry(source_sizes[view.path.name], view.angle)[0], view.path.name)
                  for view in annotation_views}
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    locator = FaceLocator.from_checkpoint(weight, device)
    output.mkdir(parents=True, exist_ok=False)
    rows, hashes, preview_errors = [], {}, []
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "building",
        "locator": str(weight), "locator_sha256": file_sha256(weight), "epoch": epoch,
        "locator_split_sha256": file_sha256(split_path), "rotation": ROTATION_CONTRACT,
        "trained_rotation_policy": policy, "annotations_sha256": file_sha256(annotation_path),
        "annotation_manifest_sha256": file_sha256(annotation_path.with_suffix(".meta.json")),
        "scales": args.scales, "angles": list(angles), "device": str(device), "input_size": locator.size,
        "presence_threshold": locator.threshold, "iou_threshold": args.iou_threshold,
        "subset": "development_validation", "contents": len(partitions["validation"]),
        "derived_references": args.derived_references,
        "note": "Detection alone does not prove crop correctness. Only matching annotated files/angles are manual ground truth; scaled references are approximate. No generated box becomes a label or cache.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    for index, record in enumerate(selected, 1):
        source, source_error = None, ""
        try:
            data = record.path.read_bytes()
            hashes[record.path.name] = hashlib.sha256(data).hexdigest()
            with Image.open(io.BytesIO(data)) as opened:
                source = opened.convert("RGBA")
        except (OSError, ValueError) as exc:
            source_error = str(exc)
        for angle in angles:
            available = references.get((record.content_key, angle))
            reference_kind = ("manual" if available and available[2] == record.path.name else
                              "scaled_reference" if available and args.derived_references else "none")
            row = {"filename": record.path.name, "content_id": "_".join(record.content_key),
                   "scale": record.scale, "angle": angle, "status": "image_error" if source_error else "ok",
                   "error": source_error, "reference_kind": reference_kind, "preview": ""}
            try:
                if source is None:
                    raise OSError(source_error)
                image = rotate_image(source, angle)
                row.update(image_width=image.width, image_height=image.height)
                reference, kind = reference_for_view(record, angle, image.size, references, args.derived_references)
                row["reference_kind"] = kind
                if reference is not None:
                    row.update(reference_x_left=reference.x_left, reference_y_bottom=reference.y_bottom,
                               reference_side=reference.side)
                box, confidence = locator.locate(image)
                if not box.fits(*image.size) or not math.isfinite(confidence) or confidence < locator.threshold:
                    raise FaceNotFoundError("invalid geometry/confidence from locator")
                row.update(confidence=confidence, x_left=box.x_left, y_bottom=box.y_bottom, side=box.side)
                if reference is not None:
                    row.update(box_errors(box, reference))
            except FaceNotFoundError as exc:
                row.update(status="locator_failure", error=str(exc))
            except (OSError, ValueError) as exc:
                row.update(status="image_error", error=str(exc))
            rows.append(row)
        if index % 25 == 0 or index == len(selected):
            print(f"Processed {index}/{len(selected)} sources, {len(angles)} angles each")
    groups = defaultdict(list)
    for row in rows:
        groups[row["scale"], row["angle"]].append(row)
    previews = output / "previews"
    if args.preview_per_group:
        previews.mkdir()
        paths = {record.path.name: record.path for record in selected}
        number = 0
        for key, group in sorted(groups.items()):
            worst = sorted(group, key=lambda r: (r["status"] == "ok", r.get("iou", r.get("confidence", -1))))
            for row in worst[:args.preview_per_group]:
                number += 1
                path = previews / f"{number:04d}_{key[0]}_{angle_text(key[1])}.png"
                try:
                    with Image.open(paths[row["filename"]]) as opened:
                        image = rotate_image(opened, row["angle"])
                    save_preview(row, image, path, locator.size)
                    row["preview"] = path.relative_to(output).as_posix()
                except (OSError, ValueError) as exc:
                    preview_errors.append({"filename": row["filename"], "angle": row["angle"], "error": str(exc)})
    with (output / "predictions.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    conditions = [{"scale": scale, "angle": angle, **summarize_rows(group, args.iou_threshold)}
                  for (scale, angle), group in sorted(groups.items())]
    metadata.update(status="complete", source_sha256=hashes, preview_errors=preview_errors)
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({"metadata": metadata, "conditions": conditions},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# Rotated locator diagnostics", "", metadata["note"], "",
              "Same held-out contents used during locator model selection; this is not an independent test set.", "",
              "| Scale | Angle | N | Detected | Failures | Reference | Mean IoU (failures=0) | Recall at IoU threshold |",
              "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for item in conditions:
        iou = "n/a" if item["mean_iou_all_references"] is None else f"{item['mean_iou_all_references']:.4f}"
        recall = "n/a" if item["localization_rate_at_iou_threshold"] is None else f"{item['localization_rate_at_iou_threshold']:.2%}"
        report.append(f"| {item['scale']} | {angle_text(item['angle'])} | {item['samples']} | {item['localized']} | "
                      f"{item['failures']} | {', '.join(item['reference_kind'])} | {iou} | {recall} |")
    report.extend(["", f"IoU threshold: {args.iou_threshold}. Worst per-group previews: previews/; preview errors: {len(preview_errors)}.",
                   "Raw failures remain in predictions.csv. Other resolutions' derived references are not manually confirmed labels."])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Saved:", output / "report.md")
    return {"metadata": metadata, "conditions": conditions}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_dir", default="atridataset/train")
    parser.add_argument("--locator_weight", required=True)
    parser.add_argument("--locator_split")
    parser.add_argument("--annotations", default="annotations/face_boxes_l_rotated.csv")
    parser.add_argument("--reference_scale", choices=SCALE_CODES, default="l")
    parser.add_argument("--scales", choices=SCALE_CODES, nargs="+", default=list(SCALE_CODES))
    parser.add_argument("--angles", type=float, nargs="+", default=list(DEFAULT_ANGLES))
    parser.add_argument("--derived_references", action="store_true", help="explicitly allow approximate same-angle references at other resolutions")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--preview_per_group", type=int, default=3)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    diagnose(parse_args())
