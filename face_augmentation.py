# -*- coding: utf-8 -*-
"""Explicit face-crop augmentation policy, independent of model execution."""

from dataclasses import asdict, dataclass
import math

from face_regions import offset_face_box


@dataclass(frozen=True)
class FaceAugmentation:
    position_mode: str = "legacy"
    size_jitter: float = 0.08
    shrink_probability: float = 0.35
    min_scale: float = 0.30
    keep_probability: float = 0.25
    small_probability: float = 0.50
    small_pixels: float = 10.0
    wide_pixels: float = 25.0
    attempts: int = 16
    face_translate: float = 0.02
    full_translate: float = 0.02
    flip_probability: float = 0.50
    affine_degrees: float = 3.0
    affine_scale_min: float = 0.95
    affine_scale_max: float = 1.02
    color_jitter: float = 0.08

    def __post_init__(self):
        if self.position_mode not in ("legacy", "mixed", "none"):
            raise ValueError("unknown face position mode")
        for name, value in asdict(self).items():
            if name != "position_mode" and not math.isfinite(value):
                raise ValueError(f"augmentation {name} must be finite")
        for name in ("shrink_probability", "keep_probability", "small_probability", "flip_probability"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"augmentation {name} must be in [0, 1]")
        if self.keep_probability + self.small_probability > 1:
            raise ValueError("keep_probability + small_probability must not exceed 1")
        if not 0 < self.min_scale <= 1 or not 0 <= self.size_jitter < 1:
            raise ValueError("invalid face min_scale or size_jitter")
        if not 0 <= self.small_pixels <= self.wide_pixels:
            raise ValueError("require 0 <= small_pixels <= wide_pixels")
        if not isinstance(self.attempts, int) or self.attempts < 1:
            raise ValueError("augmentation attempts must be a positive integer")
        if not 0 < self.affine_scale_min <= self.affine_scale_max:
            raise ValueError("invalid affine scale range")
        if not 0 <= self.affine_degrees <= 180 or not 0 <= self.color_jitter <= 1:
            raise ValueError("invalid affine degrees or color jitter")
        if not 0 <= self.face_translate <= 1 or not 0 <= self.full_translate <= 1:
            raise ValueError("affine translation fractions must be in [0, 1]")
        if self.position_mode != "legacy" and self.face_translate != 0:
            raise ValueError("mixed/none face position modes require face_translate=0 to avoid stacked translations")

    def metadata(self):
        return {"version": 1, **asdict(self)}

    @classmethod
    def from_metadata(cls, metadata):
        values = dict(metadata)
        if values.pop("version", None) != 1:
            raise ValueError("unsupported face augmentation policy version")
        return cls(**values)


def sample_face_offset(box, image_size, output_scale, policy, rng):
    """Keep the chosen mixture component when retrying near an image edge."""
    choice = rng.random()
    if policy.position_mode == "none" or choice < policy.keep_probability:
        bucket, limit = "keep", 0.0
    elif choice < policy.keep_probability + policy.small_probability:
        bucket, limit = "small", policy.small_pixels
    else:
        bucket, limit = "wide", policy.wide_pixels
    for attempt in range(1, policy.attempts + 1):
        dx, dy = (rng.uniform(-limit, limit), rng.uniform(-limit, limit)) if limit else (0.0, 0.0)
        changed, info = offset_face_box(box, *image_size, dx, dy, output_scale)
        if changed is not None:
            return changed, {**info, "bucket": bucket, "rejected_attempts": attempt - 1,
                             "fallback": False}
    # The size-adjusted box was already valid. Do not shrink it to fit a shift.
    return box, {**info, "source_dx": 0, "source_dy": 0, "actual_dx": 0.0, "actual_dy": 0.0,
                 "bucket": bucket, "rejected_attempts": policy.attempts, "fallback": True}


AUGMENTATION_BUCKETS = ("keep", "small", "wide", "legacy")


def augmentation_stat_values(info):
    return [AUGMENTATION_BUCKETS.index(info["bucket"]), info["rejected_attempts"],
            int(info["fallback"]), info["actual_dx"], info["actual_dy"], int(info["shrunk"])]


def summarize_augmentation(values):
    rows = list(values)
    if not rows:
        return {}
    return {
        "samples": len(rows),
        "bucket_counts": {name: sum(int(row[0]) == index for row in rows)
                          for index, name in enumerate(AUGMENTATION_BUCKETS)},
        "boundary_limited_samples": sum(row[1] > 0 for row in rows),
        "rejected_attempts": int(sum(row[1] for row in rows)),
        "fallback_samples": int(sum(row[2] for row in rows)),
        "shrunk_samples": int(sum(row[5] for row in rows)),
        "nonzero_offset_samples": sum(row[3] != 0 or row[4] != 0 for row in rows),
        "mean_abs_dx": sum(abs(row[3]) for row in rows) / len(rows),
        "mean_abs_dy": sum(abs(row[4]) for row in rows) / len(rows),
        "max_abs_dx": max(abs(row[3]) for row in rows),
        "max_abs_dy": max(abs(row[4]) for row in rows),
    }
