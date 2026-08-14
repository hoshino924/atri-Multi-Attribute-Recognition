# -*- coding: utf-8 -*-
"""Audit dataset filenames, image integrity, scales, and near duplicates."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image

from dataset import (
    ForegroundRegionCrop,
    IMAGE_EXTENSIONS,
    SCALE_CODES,
    load_label_manifest,
    parse_filename,
)


def image_hash(path, expression_crop=False):
    """Calculate a compact difference hash after optional face-region crop."""
    with Image.open(path) as source:
        image = source.convert("RGBA")
    if expression_crop:
        image = ForegroundRegionCrop()(image)
    background = Image.new("RGBA", image.size, (0, 0, 0, 255))
    background.alpha_composite(image)
    grayscale = background.convert("L")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    grayscale = grayscale.resize((9, 8), resampling)
    pixels = list(grayscale.getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value <<= 1
            value |= pixels[offset + column] > pixels[offset + column + 1]
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hamming_distance(first, second):
    return (first ^ second).bit_count()


def inspect_image(path):
    """Read enough image data to detect corruption and alpha availability."""
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        bands = image.getbands()
        has_alpha = "A" in bands
        has_transparency = False
        if has_alpha:
            minimum, maximum = image.getchannel("A").getextrema()
            has_transparency = minimum < 255 or maximum < 255
        return {
            "width": width,
            "height": height,
            "mode": image.mode,
            "has_alpha": has_alpha,
            "has_transparency": has_transparency,
        }


def find_near_duplicates(items, key, maximum_distance, maximum_pairs):
    pairs = []
    total_found = 0
    for left_index, left in enumerate(items):
        for right in items[left_index + 1:]:
            distance = hamming_distance(left[key], right[key])
            if distance > maximum_distance:
                continue
            total_found += 1
            if len(pairs) < maximum_pairs:
                pairs.append({
                    "left": left["name"],
                    "right": right["name"],
                    "left_source": left["source"],
                    "right_source": right["source"],
                    "distance": distance,
                })
    return {"total": total_found, "shown": pairs}


def audit(args):
    train_dir = Path(args.train_dir)
    if not train_dir.is_dir():
        raise FileNotFoundError(f"training directory does not exist: {train_dir}")

    errors = []
    warnings = []
    records = []
    metadata = {}
    paths = sorted(
        path
        for path in train_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for path in paths:
        try:
            record = parse_filename(path)
            info = inspect_image(path)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        records.append(record)
        metadata[path] = info
        if not info["has_alpha"] or not info["has_transparency"]:
            warnings.append(
                f"training image has no transparent foreground: {path.name}"
            )

    groups = defaultdict(list)
    for record in records:
        groups[record.content_key].append(record)

    missing_scales = []
    aspect_mismatches = []
    expected_scales = set(SCALE_CODES)
    for content_key, group in groups.items():
        scales = {record.scale for record in group}
        missing = sorted(expected_scales - scales)
        if missing:
            missing_scales.append({
                "content_key": list(content_key),
                "missing": missing,
            })
        ratios = [
            metadata[record.path]["width"] / metadata[record.path]["height"]
            for record in group
        ]
        if ratios and max(ratios) - min(ratios) > args.aspect_tolerance:
            aspect_mismatches.append({
                "content_key": list(content_key),
                "ratios": ratios,
            })

    selected_records = [record for record in records if record.scale == args.scale]
    label_counts = {
        task: dict(sorted(Counter(
            getattr(record, task) for record in selected_records
        ).items()))
        for task in ("outfit", "pose", "expression")
    }

    hash_items = []
    for record in selected_records:
        try:
            hash_items.append({
                "name": record.path.name,
                "source": "train",
                "full_hash": image_hash(record.path),
                "expression_hash": image_hash(
                    record.path,
                    expression_crop=True,
                ),
                "sha256": file_sha256(record.path),
            })
        except Exception as exc:
            errors.append(f"hash failed for {record.path.name}: {exc}")

    test_count = 0
    if args.test_dir:
        test_dir = Path(args.test_dir)
        if not test_dir.is_dir():
            warnings.append(f"test directory does not exist: {test_dir}")
        else:
            test_paths = sorted(
                path
                for path in test_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            test_count = len(test_paths)
            for path in test_paths:
                try:
                    inspect_image(path)
                    hash_items.append({
                        "name": path.name,
                        "source": "test",
                        "full_hash": image_hash(path),
                        "expression_hash": image_hash(
                            path,
                            expression_crop=True,
                        ),
                        "sha256": file_sha256(path),
                    })
                except Exception as exc:
                    errors.append(f"test image {path.name}: {exc}")
            if args.test_labels:
                try:
                    load_label_manifest(test_dir, args.test_labels)
                except Exception as exc:
                    errors.append(f"test label manifest: {exc}")

    exact_groups = defaultdict(list)
    for item in hash_items:
        exact_groups[item["sha256"]].append(item)
    exact_duplicates = [
        [
            {"name": item["name"], "source": item["source"]}
            for item in group
        ]
        for group in exact_groups.values()
        if len(group) > 1
    ]
    near_full = find_near_duplicates(
        hash_items,
        "full_hash",
        args.near_duplicate_distance,
        args.max_pairs,
    )
    near_expression = find_near_duplicates(
        hash_items,
        "expression_hash",
        args.near_duplicate_distance,
        args.max_pairs,
    )

    report = {
        "train_dir": str(train_dir),
        "selected_scale": args.scale,
        "image_files": len(paths),
        "valid_records": len(records),
        "unique_content": len(groups),
        "selected_records": len(selected_records),
        "test_images": test_count,
        "label_counts": label_counts,
        "missing_scale_groups": missing_scales,
        "aspect_ratio_mismatches": aspect_mismatches,
        "exact_duplicate_groups": exact_duplicates,
        "near_duplicates_full": near_full,
        "near_duplicates_expression": near_expression,
        "errors": errors,
        "warnings": warnings,
    }
    output_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)

    print(f"Valid training images: {len(records)} / {len(paths)}")
    print(f"Unique content groups: {len(groups)}")
    print(f"Selected '{args.scale}' images: {len(selected_records)}")
    print(f"Missing scale groups: {len(missing_scales)}")
    print(f"Aspect ratio mismatches: {len(aspect_mismatches)}")
    print(f"Exact duplicate groups: {len(exact_duplicates)}")
    print(f"Near full-image pairs: {near_full['total']}")
    print(f"Near expression pairs: {near_expression['total']}")
    print(f"Errors: {len(errors)}; warnings: {len(warnings)}")
    print("Saved:", output_path)
    if errors:
        raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit the ATRI image dataset.")
    parser.add_argument("--train_dir", default="atridataset/train")
    parser.add_argument("--scale", choices=SCALE_CODES, default="w")
    parser.add_argument("--test_dir", default="atridataset/test")
    parser.add_argument("--test_labels")
    parser.add_argument("--report", default="dataset_report.json")
    parser.add_argument("--near_duplicate_distance", type=int, default=2)
    parser.add_argument("--max_pairs", type=int, default=200)
    parser.add_argument("--aspect_tolerance", type=float, default=0.005)
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
