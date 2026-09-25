# -*- coding: utf-8 -*-
"""Shared face annotation geometry and Pillow-only crop preprocessing."""

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

from PIL import Image, ImageOps


CSV_FIELDS = (
    "filename", "character", "shoe_variant", "scale", "outfit", "pose",
    "expression", "image_width", "image_height", "coordinate_system",
    "x_left", "y_bottom", "side",
)
COORDINATE_SYSTEM = "bottom_left_pixels"
FACE_CROP_MODE = "annotated_face"


@dataclass(frozen=True)
class SquareBox:
    """Original-image pixels, measured from the left and bottom image edges."""

    x_left: int
    y_bottom: int
    side: int

    def fits(self, width, height):
        return (
            self.side > 0 and self.x_left >= 0 and self.y_bottom >= 0
            and self.x_left + self.side <= width
            and self.y_bottom + self.side <= height
        )

    def pil_bounds(self, height):
        return (
            self.x_left, height - self.y_bottom - self.side,
            self.x_left + self.side, height - self.y_bottom,
        )

    def moved(self, dx, dy, width, height):
        return SquareBox(
            max(0, min(width - self.side, self.x_left + dx)),
            max(0, min(height - self.side, self.y_bottom + dy)), self.side,
        )

    def resized(self, delta, width, height):
        side = max(1, min(width, height, self.side + delta))
        shift = (self.side - side) // 2
        return SquareBox(self.x_left + shift, self.y_bottom + shift, side).moved(
            0, 0, width, height,
        )


def load_face_annotations(records, path):
    """Read a complete annotation table; reject stale, missing or extra rows."""
    records = list(records)
    by_name = {record.path.name: record for record in records}
    if len(by_name) != len(records):
        raise ValueError("face annotations require unique image filenames")
    boxes = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError("face annotation CSV header does not match the annotation tool")
        for line, row in enumerate(reader, start=2):
            try:
                filename = row["filename"]
                if filename not in by_name or filename in boxes:
                    raise ValueError("unknown or duplicate image filename")
                record = by_name[filename]
                with Image.open(record.path) as source:
                    width, height = source.size
                box = SquareBox(*(int(row[key]) for key in ("x_left", "y_bottom", "side")))
                if not box.fits(width, height):
                    raise ValueError("face box is outside the image")
                if (
                    row["coordinate_system"] != COORDINATE_SYSTEM
                    or int(row["image_width"]) != width
                    or int(row["image_height"]) != height
                    or None in row or any(value is None for value in row.values())
                ):
                    raise ValueError("image dimensions or coordinate metadata do not match")
                for key in ("character", "shoe_variant", "scale", "outfit", "pose", "expression"):
                    if not row[key] or (hasattr(record, key) and row[key] != getattr(record, key)):
                        raise ValueError(f"annotation metadata does not match: {key}")
                boxes[filename] = box
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"invalid face annotation at line {line}: {exc}") from exc
    missing = sorted(by_name.keys() - boxes.keys())
    if missing:
        raise ValueError(f"missing face annotations for {len(missing)} images: {', '.join(missing[:5])}")
    if not boxes:
        raise ValueError("face annotation table contains no images")
    return boxes


def annotation_signature(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FaceCanvas:
    """Fit a previously cropped face without upscaling or adding a margin."""

    def __init__(self, size=626, background=(0, 0, 0)):
        if size <= 0:
            raise ValueError("face canvas size must be positive")
        self.size = size
        self.background = tuple(background)

    def __call__(self, image):
        image = image.convert("RGBA")
        if max(image.size) > self.size:
            image = ImageOps.contain(image, (self.size, self.size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (self.size, self.size), (*self.background, 255))
        canvas.alpha_composite(image, ((self.size - image.width) // 2, (self.size - image.height) // 2))
        return canvas.convert("RGB")


def crop_face(image, box):
    if not box.fits(*image.size):
        raise ValueError("face box is outside the image")
    return image.crop(box.pil_bounds(image.height))


def jitter_face_box(box, width, height, rng, size_ratio=0.08):
    """Simulate modest detector errors without removing face context entirely."""
    changed = box.resized(round(box.side * rng.uniform(-size_ratio, size_ratio)), width, height)
    return changed.moved(
        round(box.side * rng.uniform(-0.025, 0.025)),
        round(box.side * rng.uniform(-0.025, 0.025)), width, height,
    )


def offset_face_box(box, width, height, dx, dy, output_scale):
    """Move a crop in input-equivalent pixels; +x right, +y up.

    output_scale includes crop resizing and any later affine scale. Python's
    round (ties to even) is used on source pixels. Never clamp or resize an
    invalid request: return None so callers can resample or report it.
    """
    if not box.fits(width, height):
        raise ValueError("base face box is outside the image")
    if not all(math.isfinite(value) for value in (dx, dy, output_scale)) or output_scale <= 0:
        raise ValueError("offsets must be finite and output_scale must be positive")
    source_dx, source_dy = round(dx / output_scale), round(dy / output_scale)
    candidate = SquareBox(box.x_left + source_dx, box.y_bottom + source_dy, box.side)
    metadata = {
        "requested_dx": dx, "requested_dy": dy,
        "source_dx": source_dx, "source_dy": source_dy,
        "actual_dx": source_dx * output_scale, "actual_dy": source_dy * output_scale,
        "output_scale": output_scale,
    }
    return (candidate if candidate.fits(width, height) else None), metadata


def locator_canvas(image, size=256):
    """Letterbox an image; return exact scale/offset for mapping both ways."""
    image = image.convert("RGBA")
    resized = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    offset_x, offset_y = (size - resized.width) // 2, (size - resized.height) // 2
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    canvas.alpha_composite(resized, (offset_x, offset_y))
    mapping = (resized.width / image.width, resized.height / image.height, offset_x, offset_y)
    return canvas.convert("RGB"), mapping


def map_locator_box(box, mapping, image_size, input_size):
    """Map normalized canvas cx,cy,w,h to an in-bounds original square."""
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise ValueError("invalid face locator coordinates")
    cx, cy, width, height = box
    if width <= 0 or height <= 0:
        raise ValueError("face locator predicted an empty region")
    sx, sy, ox, oy = mapping
    image_width, image_height = image_size
    cx, cy = (cx * input_size - ox) / sx, (cy * input_size - oy) / sy
    if not (0 <= cx < image_width and 0 <= cy < image_height):
        raise ValueError("face locator predicted a center outside the image")
    side = max(1, min(image_width, image_height, round(max(width * input_size / sx, height * input_size / sy))))
    left = max(0, min(image_width - side, round(cx - side / 2)))
    top = max(0, min(image_height - side, round(cy - side / 2)))
    return SquareBox(left, image_height - top - side, side)
