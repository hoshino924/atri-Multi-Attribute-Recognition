# -*- coding: utf-8 -*-
"""Matched angle/scale diagnostics for all three tasks using live localization."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time

from PIL import Image, __version__ as PILLOW_VERSION
import torch
from torchvision.transforms import functional as TF

from calibration import apply_temperature
from dataset import SCALE_CODES, build_full_preview, scan_records, split_content_keys
from diagnose_scales import TASKS, TASK_FIELDS, face_view, match_records, prediction_fields, summarize, write_csv
from face_box_cache import file_sha256, write_json
from face_locator import FaceNotFoundError
from image_rotation import DEFAULT_ANGLES, ROTATION_CONTRACT, angle_text, parse_angles, rotate_image
from infer import load_model
from labels import TASK_CODES


def confidence_error(rows, task, field):
    """ECE on successfully predicted views; failures remain in accuracy denominators."""
    if not rows:
        return None
    result = 0.0
    for index in range(10):
        group = [r for r in rows if index / 10 < r[f"{task}_{field}"] <= (index + 1) / 10]
        if group:
            result += abs(sum(r[f"{task}_{field}"] - int(r[f"{task}_correct"]) for r in group)) / len(rows)
    return result


def summarize_rotations(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["angle"], row["scale"])].append(row)
    result = []
    for (angle, scale), group in sorted(grouped.items()):
        item = summarize(group)[0]
        item["angle"] = angle
        successful = [r for r in group if r["status"] == "ok"]
        item["joint_accuracy"] = sum(all(r[f"{t}_correct"] for t in TASKS) for r in successful) / len(group)
        item["joint_acceptance_rate"] = sum(all(r[f"{t}_accepted"] for t in TASKS) for r in successful) / len(group)
        item["joint_accepted_but_wrong"] = sum(all(r[f"{t}_accepted"] for t in TASKS)
            and not all(r[f"{t}_correct"] for t in TASKS) for r in successful)
        for task in TASKS:
            codes = TASK_CODES[task]
            matrix = [[0 for _ in codes] for _ in codes]
            support = {code: sum(r[f"{task}_actual"] == code for r in group) for code in codes}
            for row in successful:
                matrix[codes.index(row[f"{task}_actual"])][codes.index(row[f"{task}_predicted"])] += 1
            item["tasks"][task].update(
                label_codes=codes, confusion_matrix=matrix, support=support,
                per_class_recall={code: matrix[i][i] / support[code] if support[code] else None for i, code in enumerate(codes)},
                ece_raw_on_predictions=confidence_error(successful, task, "raw_confidence"),
                ece_calibrated_on_predictions=confidence_error(successful, task, "confidence"),
                mean_top2_margin=sum(r[f"{task}_margin"] for r in successful) / len(successful) if successful else None)
        wrong = [r for r in successful if not r["expression_correct"]]
        ambiguous = [r for r in wrong if scale == "s" and {r["expression_actual"], r["expression_predicted"]} == {"f1", "fe"}]
        item["expression_ambiguity"] = {
            "policy": "s-scale f1/fe cross-confusions only; strict labels and decisions are unchanged",
            "known_pair_errors": len(ambiguous),
            "known_pair_accepted_errors": sum(r["expression_accepted"] for r in ambiguous),
            "other_expression_errors": len(wrong) - len(ambiguous),
            "failures": len(group) - len(successful),
        }
        result.append(item)
    return result


def consistency(rows):
    """Do not treat consistent wrong predictions or missing views as success."""
    result = {}
    for axis in ("scales_at_angle", "angles_at_scale"):
        groups = defaultdict(list)
        for row in rows:
            fixed = row["angle"] if axis == "scales_at_angle" else row["scale"]
            groups[(row["content_id"], fixed)].append(row)
        result[axis] = {task: {"groups": len(groups),
            "complete_groups": sum(all(r["status"] == "ok" for r in g) for g in groups.values()),
            "consistent_groups": sum(all(r["status"] == "ok" for r in g)
                and len({r.get(f"{task}_predicted") for r in g}) == 1 for g in groups.values()),
            "all_correct_groups": sum(all(r["status"] == "ok" and r[f"{task}_correct"] for r in g) for g in groups.values())}
            for task in TASKS}
    return result


def diagnose(args):
    angles = parse_angles(args.angles)
    if not args.scales or len(set(args.scales)) != len(args.scales):
        raise ValueError("choose distinct scales")
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f"choose a new diagnostic directory: {out}")
    split_path = Path(args.split) if args.split else Path(args.weight).parent / "split.json"
    locator_split_path = Path(args.locator_split) if args.locator_split else Path(args.locator_weight).parent / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8-sig"))
    locator_split = json.loads(locator_split_path.read_text(encoding="utf-8-sig"))
    partitions, locator_partitions = split_content_keys(split), split_content_keys(locator_split)
    if partitions != locator_partitions:
        raise ValueError("classifier and locator content partitions differ")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, transforms, codes = load_model(args.weight, device, args.locator_weight)
    if not hasattr(model, "face_preview"):
        raise ValueError("rotation diagnostics require a face-crop classifier")
    if model.checkpoint_metadata.get("dataset_signature") != split.get("dataset_signature"):
        raise ValueError("classifier checkpoint and split signatures differ")
    training_rotation = model.checkpoint_metadata.get("rotation_training")
    if training_rotation and training_rotation.get("pillow_version") != PILLOW_VERSION:
        raise ValueError("Pillow version differs from classifier rotation training")
    cache_locator = (model.checkpoint_metadata.get("face_cache_metadata") or {}).get("locator", {})
    if cache_locator.get("split_sha256") and cache_locator["split_sha256"] != file_sha256(locator_split_path):
        raise ValueError("locator split differs from the classifier cache provenance")
    locator = model.face_preview.locator
    locator_checkpoint = torch.load(args.locator_weight, map_location="cpu", weights_only=True)
    if locator_checkpoint.get("dataset_signature") != locator_split.get("dataset_signature"):
        raise ValueError("locator checkpoint and split signatures differ")
    for key in ("seed", "val_ratio"):
        if locator_checkpoint.get("train_args", {}).get(key) != locator_split.get(key):
            raise ValueError(f"locator checkpoint and split disagree on {key}")
    if locator_checkpoint.get("rotation_policy") != locator_split.get("rotation_policy"):
        raise ValueError("locator rotation policy and split differ")
    locator_epoch = locator_checkpoint.get("epoch")
    locator_rotation = locator_checkpoint.get("rotation_policy")
    del locator_checkpoint
    records = match_records({s: scan_records(args.image_dir, s) for s in args.scales}, args.scales,
                            partitions["validation"], locator_partitions["train"])
    config = model.preprocess_config
    full_preview = build_full_preview(**config["full"], background=config["background"])
    mean, std = config["normalization"]["mean"], config["normalization"]["std"]
    out.mkdir(parents=True, exist_ok=False)
    if args.save_previews:
        (out / "previews").mkdir()
    trained_angles = (model.checkpoint_metadata.get("rotation_training") or {}).get("angles", [0])
    metadata = {"status": "building", "created_utc": datetime.now(timezone.utc).isoformat(),
                "classifier": str(args.weight), "classifier_sha256": file_sha256(args.weight),
                "locator": str(args.locator_weight), "locator_sha256": file_sha256(args.locator_weight),
                "locator_epoch": locator_epoch, "locator_rotation_policy": locator_rotation,
                "classifier_metadata": model.checkpoint_metadata,
                "classifier_split_sha256": file_sha256(split_path), "locator_split_sha256": file_sha256(locator_split_path),
                "angles": list(angles), "classifier_trained_angles": trained_angles,
                "unseen_angles": [a for a in angles if a not in trained_angles], "scales": args.scales,
                "contents": len(partitions["validation"]), "views": len(records) * len(angles),
                "rotation": ROTATION_CONTRACT, "pillow_version": PILLOW_VERSION, "device": str(device),
                "precision": "float32", "preprocess": config, "calibration": model.calibration,
                "source_sha256": {}, "preview_errors": [], "subset": "development_validation",
                "note": "Shared development contents, correlated views, not independent test data. Unseen angles reuse training calibration as stress tests. No geometric proposal is used as a label."}
    write_json(out / "metadata.json", metadata)
    rows = []
    with torch.inference_mode():
        for index, record in enumerate(records, 1):
            source, source_error = None, None
            try:
                data = record.path.read_bytes()
                metadata["source_sha256"][record.path.name] = hashlib.sha256(data).hexdigest()
                with Image.open(io.BytesIO(data)) as opened:
                    source = opened.convert("RGBA")
            except (OSError, ValueError) as exc:
                source_error = str(exc)
            for angle in angles:
                row = {"content_id": "_".join(record.content_key), "filename": record.path.name,
                       "scale": record.scale, "angle": angle, "box_source": "locator", "mode": "pad",
                       "status": "pending", "error": "", **{f"{t}_actual": getattr(record, t) for t in TASKS}}
                rows.append(row)
                started = time.perf_counter()
                try:
                    if source is None:
                        raise OSError(source_error)
                    image = rotate_image(source, angle)
                    box, confidence = locator.locate(image)
                    face, geometry = face_view(image, box, config["expression"]["size"], config["background"], "pad")
                    full_tensor = transforms["full"](image).unsqueeze(0).to(device)
                    face_tensor = TF.normalize(TF.to_tensor(face), mean, std).unsqueeze(0).to(device)
                    outputs = model(full_tensor, face_tensor)
                    row.update(prediction_fields(outputs, 0, record, model, codes))
                    probabilities = apply_temperature(outputs["expression"].float(), "expression", model.calibration).softmax(1)[0]
                    row["expression_f1_fe_probability"] = sum(probabilities[codes["expression"].index(c)].item() for c in ("f1", "fe"))
                    row.update(status="ok", image_width=image.width, image_height=image.height,
                               x_left=box.x_left, y_bottom=box.y_bottom, side=box.side,
                               locator_confidence=confidence, **geometry)
                    if args.save_previews:
                        name = f"{index:04d}_{record.scale}_{angle_text(angle)}"
                        try:
                            face.save(out / "previews" / f"{name}_face.png")
                            full_preview(image).save(out / "previews" / f"{name}_full.png")
                        except OSError as exc:
                            metadata["preview_errors"].append({"filename": record.path.name, "angle": angle, "error": str(exc)})
                except (OSError, ValueError, FaceNotFoundError) as exc:
                    row.update(status="face_not_found" if isinstance(exc, FaceNotFoundError) else "error", error=str(exc))
                row["elapsed_seconds"] = time.perf_counter() - started
            if index % 25 == 0 or index == len(records):
                print(f"Processed {index}/{len(records)} sources, {len(angles)} angles each", flush=True)
    metadata.update(status="complete", failures=sum(r["status"] != "ok" for r in rows))
    conditions = summarize_rotations(rows)
    fields = ["content_id", "filename", "scale", "angle", "box_source", "mode", "status", "error",
              "image_width", "image_height", "x_left", "y_bottom", "side", "locator_confidence",
              "canvas_size", "rendered_side", "resize_factor", "added_padding_fraction", "elapsed_seconds",
              "expression_f1_fe_probability", *(f"{t}_{f}" for t in TASKS for f in TASK_FIELDS)]
    write_csv(out / "predictions.csv", rows, fields)
    write_json(out / "metadata.json", metadata)
    write_json(out / "summary.json", {"metadata": metadata, "conditions": conditions, "consistency": consistency(rows)})
    expression_head = metadata["classifier_metadata"].get("model_config", {}).get("expression_head", "fusion")
    lines = ["# Three-task rotation diagnostics", "", f"Expression head: `{expression_head}` (shared backbone).", "",
             "Development validation; all failures remain in denominators.",
             "Strict labels/decisions are unchanged. s-scale f1/fe cross-confusions are also counted separately.",
             "Unseen angles reuse the trained thresholds; their calibration is not validated.", "",
             "| Angle | Scale | N | Fail | Outfit | Pose | Expression | Joint | Expr wrong accepted | f1/fe errors | Other expr errors |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in conditions:
        t, a = item["tasks"], item["expression_ambiguity"]
        lines.append(f"| {angle_text(item['angle'])} | {item['scale']} | {item['samples']} | {item['failures']} | "
                     f"{t['outfit']['top1_accuracy']:.2%} | {t['pose']['top1_accuracy']:.2%} | {t['expression']['top1_accuracy']:.2%} | "
                     f"{item['joint_accuracy']:.2%} | {t['expression']['accepted_but_wrong']} | {a['known_pair_errors']} | {a['other_expression_errors']} |")
    lines.extend(["", "See summary.json for task confusion matrices, recalls, ECE, rejection counts and consistency; predictions.csv for all views.", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved:", out / "report.md")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--locator_weight", required=True)
    parser.add_argument("--image_dir", default="atridataset/train")
    parser.add_argument("--angles", type=float, nargs="+", default=list(DEFAULT_ANGLES))
    parser.add_argument("--scales", choices=SCALE_CODES, nargs="+", default=list(SCALE_CODES))
    parser.add_argument("--split")
    parser.add_argument("--locator_split")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_previews", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    diagnose(parse_args())
