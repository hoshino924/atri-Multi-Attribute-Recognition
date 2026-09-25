# -*- coding: utf-8 -*-
"""Cache fixed-locator face boxes, optionally on rotated source canvases."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path

from PIL import Image, __version__ as PILLOW_VERSION
import torch

from dataset import SCALE_CODES, scan_records, split_content_keys, stratified_split
from face_locator import FaceLocator, FaceNotFoundError
from face_regions import COORDINATE_SYSTEM, CSV_FIELDS, SquareBox, load_face_annotations
from image_rotation import ROTATION_CONTRACT, normalize_angle, parse_angles, rotate_image, rotation_geometry


CACHE_VERSION = 1
ROTATED_CACHE_VERSION = 2
ROTATED_CSV_FIELDS = (*CSV_FIELDS, "angle", "source_width", "source_height", "confidence")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_contents(manifest):
    """Read source identities, ignoring image resolution and path separators."""
    return split_content_keys(manifest)


def check_locator_overlap(val_records, locator_split, train_records=None):
    locator_keys = split_contents(locator_split)
    overlap = {record.content_key for record in val_records} & locator_keys["train"]
    if overlap:
        raise ValueError(
            f"{len(overlap)} classifier validation contents were used to train the locator; "
            "use the same source split (seed / val_ratio) as the locator"
        )
    if train_records is not None:
        for name, records in (("train", train_records), ("validation", val_records)):
            if {record.content_key for record in records} != locator_keys[name]:
                raise ValueError(f"classifier and locator {name} content partitions differ; use aligned splits")


def classifier_split(train_records, val_records, scale, seed, val_ratio):
    return {
        "scale": scale, "seed": seed, "val_ratio": val_ratio,
        "train": sorted(record.path.name for record in train_records),
        "validation": sorted(record.path.name for record in val_records),
    }


def load_face_cache(records, directory, train_records, val_records, scale, seed, val_ratio, angles=None):
    """Validate the complete cache before any training outputs are written."""
    directory = Path(directory)
    manifest_path, csv_path = directory / "manifest.json", directory / "face_boxes.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") not in (CACHE_VERSION, ROTATED_CACHE_VERSION) or manifest.get("status") != "complete":
        raise ValueError("face cache is incomplete or unsupported; regenerate it in a new directory")
    rotated = manifest["version"] == ROTATED_CACHE_VERSION
    coordinate_system = ROTATION_CONTRACT["coordinate_system"] if rotated else COORDINATE_SYSTEM
    if manifest.get("box_source") != "fixed_locator" or manifest.get("coordinate_system") != coordinate_system:
        raise ValueError("unsupported face cache coordinate/source contract")
    expected_split = classifier_split(train_records, val_records, scale, seed, val_ratio)
    if manifest.get("classifier_split") != expected_split:
        raise ValueError("face cache classifier split differs (scale / seed / val_ratio / image list)")
    locator = manifest.get("locator", {})
    check_locator_overlap(val_records, locator.get("split", {}), train_records)
    if manifest.get("csv_sha256") != file_sha256(csv_path):
        raise ValueError("face cache CSV changed; regenerate the cache instead of editing automatic boxes")
    images = manifest.get("images", {})
    if set(images) != {record.path.name for record in records}:
        raise ValueError("face cache image coverage differs from the selected dataset")
    for record in records:
        if images[record.path.name].get("sha256") != file_sha256(record.path):
            raise ValueError(f"face cache source image changed: {record.path.name}")
    if rotated:
        if angles is None:
            raise ValueError("rotated cache requires explicit training --angles (use --angles 0 for upright)")
        if manifest.get("rotation") != ROTATION_CONTRACT or manifest.get("pillow_version") != PILLOW_VERSION:
            raise ValueError("face cache rotation contract or Pillow version differs; regenerate in a new directory")
        available = parse_angles(manifest.get("angles", []))
        requested = parse_angles(angles)
        if not set(requested).issubset(available):
            raise ValueError("face cache is missing requested angles")
        boxes = load_rotated_boxes(records, csv_path, manifest, available)
        boxes = {key: box for key, box in boxes.items() if key[1] in requested}
    else:
        if angles is not None:
            raise ValueError("--angles requires a rotated cache, even for the 0-degree control")
        boxes = load_face_annotations(records, csv_path)
    signature = file_sha256(manifest_path) + ":" + file_sha256(csv_path)
    provenance = {
        "version": manifest["version"], "directory": str(directory.resolve()), "signature": signature,
        "image_count": len(records),
        "locator": {key: value for key, value in locator.items() if key != "split"},
        "classifier_split": {key: expected_split[key] for key in ("scale", "seed", "val_ratio")},
    }
    if rotated:
        provenance.update(angles=list(requested), rotation=manifest["rotation"], pillow_version=PILLOW_VERSION,
                          cached_angles=list(available), view_count=len(boxes))
    return boxes, signature, provenance


def load_rotated_boxes(records, path, manifest, angles):
    """Check every cached view, including angles not selected for this run."""
    indexed = {r.path.name: r for r in records}
    sizes = {}
    for record in records:
        with Image.open(record.path) as source:
            sizes[record.path.name] = source.size
        saved = manifest["images"][record.path.name]
        if (saved.get("width"), saved.get("height")) != sizes[record.path.name]:
            raise ValueError("face cache source dimensions differ")
    boxes = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(ROTATED_CSV_FIELDS):
            raise ValueError("rotated cache CSV header differs")
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("invalid CSV row")
                name, angle = row["filename"], normalize_angle(row["angle"])
                key = name, angle
                if name not in indexed or angle not in angles or key in boxes:
                    raise ValueError("unknown or duplicate filename/angle")
                source_size = sizes[name]
                size = rotation_geometry(source_size, angle)[0]
                if (int(row["source_width"]), int(row["source_height"])) != source_size:
                    raise ValueError("source dimensions differ")
                if (int(row["image_width"]), int(row["image_height"])) != size or row["coordinate_system"] != ROTATION_CONTRACT["coordinate_system"]:
                    raise ValueError("rotated dimensions or coordinate system differ")
                for field in ("character", "shoe_variant", "scale", "outfit", "pose", "expression"):
                    if row[field] != getattr(indexed[name], field):
                        raise ValueError(f"label differs: {field}")
                box = SquareBox(*(int(row[field]) for field in ("x_left", "y_bottom", "side")))
                confidence = float(row["confidence"])
                if not box.fits(*size) or not math.isfinite(confidence) or not manifest["locator"]["threshold"] <= confidence <= 1:
                    raise ValueError("invalid cached box or confidence")
                boxes[key] = box
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"invalid rotated cache row {line}: {exc}") from exc
    if set(boxes) != {(name, angle) for name in indexed for angle in angles}:
        raise ValueError("rotated cache has missing filename/angle views")
    return boxes


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_cache(args):
    """Locate each requested view; never skip a failure or reuse another angle's box."""
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"face cache output already exists: {output_dir}; choose a new directory")
    records = scan_records(args.train_dir, scale=args.scale)
    train_records, val_records = stratified_split(records, args.val_ratio, args.seed)
    weight_path = Path(args.locator_weight)
    split_path = Path(args.locator_split) if args.locator_split else weight_path.parent / "split.json"
    locator_split = json.loads(split_path.read_text(encoding="utf-8-sig"))
    check_locator_overlap(val_records, locator_split, train_records)
    # Bind the supplied split to the locator checkpoint, not just its filename.
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not checkpoint.get("dataset_signature") or checkpoint["dataset_signature"] != locator_split.get("dataset_signature"):
        raise ValueError("locator checkpoint and split manifest have different dataset signatures")
    for key in ("seed", "val_ratio"):
        if key not in checkpoint.get("train_args", {}) or checkpoint["train_args"][key] != locator_split.get(key):
            raise ValueError(f"locator checkpoint and split manifest disagree on {key}")
    angles = parse_angles(args.angles) if getattr(args, "angles", None) is not None else None
    rotation_policy = checkpoint.get("rotation_policy")
    if angles is not None:
        if (not rotation_policy or rotation_policy.get("rotation") != ROTATION_CONTRACT
                or rotation_policy.get("pillow_version") != PILLOW_VERSION
                or locator_split.get("rotation_policy") != rotation_policy):
            raise ValueError("rotated cache requires a matching rotated locator, split and Pillow contract")
        if not set(angles).issubset(rotation_policy["angles"]):
            raise ValueError("training cache angles must be covered by the locator training policy")
    locator_epoch = checkpoint.get("epoch")
    del checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    locator = FaceLocator.from_checkpoint(weight_path, device)
    manifest = {
        "version": ROTATED_CACHE_VERSION if angles is not None else CACHE_VERSION,
        "status": "building", "box_source": "fixed_locator",
        "coordinate_system": ROTATION_CONTRACT["coordinate_system"] if angles is not None else COORDINATE_SYSTEM,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(Path(args.train_dir).resolve()), "device": str(device),
        "locator": {
            "weight": str(weight_path.resolve()), "sha256": file_sha256(weight_path),
            "epoch": locator_epoch, "input_size": locator.size, "threshold": locator.threshold,
            "split_path": str(split_path.resolve()), "split_sha256": file_sha256(split_path),
            "split": locator_split,
        },
        "classifier_split": classifier_split(train_records, val_records, args.scale, args.seed, args.val_ratio),
        "images": {},
    }
    if angles is not None:
        manifest.update(angles=list(angles), rotation=ROTATION_CONTRACT, pillow_version=PILLOW_VERSION)
        manifest["locator"]["rotation_policy"] = rotation_policy
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "manifest.json", manifest)
    rows, failures = [], []
    print(f"Device: {device}; {len(records)} {args.scale}-scale images; fixed locator epoch {locator_epoch}")
    print(f"Classifier split: {len(train_records)} train / {len(val_records)} validation; no locator-training overlap in validation")
    print(f"Content groups: {len({r.content_key for r in train_records})} train / {len({r.content_key for r in val_records})} validation; angles={angles or (0,)}")
    for index, record in enumerate(records, start=1):
        try:
            # Hash the same bytes used for decoding; source images remain untouched.
            source_bytes = record.path.read_bytes()
            with Image.open(io.BytesIO(source_bytes)) as source:
                image = source.convert("RGBA")
            for angle in angles or (0,):
                try:
                    rotated = rotate_image(image, angle) if angles is not None else image
                    box, confidence = locator.locate(rotated)
                    if not box.fits(*rotated.size) or not math.isfinite(confidence) or not locator.threshold <= confidence <= 1:
                        raise FaceNotFoundError("invalid geometry or confidence from the locator")
                    row = {key: getattr(record, key) for key in ("character", "shoe_variant", "scale", "outfit", "pose", "expression")}
                    row.update(filename=record.path.name, image_width=rotated.width, image_height=rotated.height,
                               coordinate_system=manifest["coordinate_system"], x_left=box.x_left, y_bottom=box.y_bottom, side=box.side)
                    if angles is not None:
                        row.update(angle=angle, source_width=image.width, source_height=image.height, confidence=confidence)
                    rows.append(row)
                except (OSError, ValueError, FaceNotFoundError) as exc:
                    failures.append({"filename": record.path.name, "angle": angle, "error": str(exc), "type": type(exc).__name__})
            if rows and rows[-1]["filename"] == record.path.name:
                manifest["images"][record.path.name] = {
                    "sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "width": image.width, "height": image.height,
                }
                if angles is None:
                    manifest["images"][record.path.name]["confidence"] = confidence
        except (OSError, ValueError, FaceNotFoundError) as exc:
            failures.append({"filename": record.path.name, "error": str(exc), "type": type(exc).__name__})
        if index % 25 == 0 or index == len(records):
            print(f"Located {index}/{len(records)}; failures={len(failures)}")
    if failures:
        manifest.update(status="failed", failure_count=len(failures))
        write_json(output_dir / "failures.json", failures)
        write_json(output_dir / "manifest.json", manifest)
        unit = "views/sources" if angles is not None else "images"
        raise ValueError(f"{len(failures)} {unit} failed; no usable cache was produced. See {output_dir / 'failures.json'}")
    csv_path = output_dir / "face_boxes.csv"
    with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROTATED_CSV_FIELDS if angles is not None else CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest.update(status="complete", csv_sha256=file_sha256(csv_path))
    write_json(output_dir / "manifest.json", manifest)
    print("Saved:", csv_path)
    print("Saved:", output_dir / "manifest.json")
    print("Use train.py --face_cache with this directory; keep the manual annotations for comparison.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_dir", default="atridataset/train")
    from app_assets import DEFAULT_LOCATOR_WEIGHT
    parser.add_argument("--locator_weight", default=DEFAULT_LOCATOR_WEIGHT)
    parser.add_argument("--locator_split", help="defaults to split.json beside the locator checkpoint")
    parser.add_argument("--scale", choices=(*SCALE_CODES, "all"), default="w",
                        help="all requires every content at s/w/m/l/ll")
    parser.add_argument("--angles", type=float, nargs="+", help="rotate original canvases before localization; omitted: upright v1 cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--output_dir", help="default: annotations/locator_<scale>_seed<seed>")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        suffix = "_rotated" if args.angles is not None else ""
        args.output_dir = f"annotations/locator_{args.scale}{suffix}_seed{args.seed}"
    return args


if __name__ == "__main__":
    generate_cache(parse_args())
