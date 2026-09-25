# -*- coding: utf-8 -*-
"""Confirmed rotation views and deterministic content/angle sampling."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

from PIL import Image
from torch.utils.data import Sampler

from dataset import ImageRecord
from image_rotation import DEFAULT_ANGLES, DEFAULT_ANGLE_WEIGHTS, angle_text, normalize_angle, parse_angles
from rotation_annotations import LABEL_FIELDS, RotatedAnnotationStore, expand_views


@dataclass(frozen=True)
class RotatedImageRecord(ImageRecord):
    angle: float = 0.0

    @property
    def view_key(self):
        return self.path.name, self.angle


def rotation_records(records, angles):
    """Expand identities only; every rotated view still reads its original file."""
    return [RotatedImageRecord(record.path, *(getattr(record, key) for key in LABEL_FIELDS), angle)
            for angle in parse_angles(angles) for record in records]


def load_confirmed_rotation_annotations(records, path, angles):
    """Read the annotator's exact contract, source hashes and confirmed boxes."""
    path = Path(path)
    if not path.is_file() or not path.with_suffix(".meta.json").is_file():
        raise ValueError("rotated annotations require both CSV and its .meta.json manifest")
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if set(metadata.get("sources", {})) != {record.path.name for record in records}:
        raise ValueError("rotated annotation source coverage differs from the selected scale")
    requested = parse_angles(angles)
    if not set(requested).issubset(metadata.get("angles", [])):
        raise ValueError("requested angles lack manually confirmed annotation coverage")
    sources = []
    for record in records:
        with Image.open(record.path) as image:
            width, height = image.size
        sources.append(SimpleNamespace(path=record.path, width=width, height=height,
                                       **{key: getattr(record, key) for key in LABEL_FIELDS}))
    # Reuse the annotator's strict reader, without saving anything. It checks
    # pixel hashes, dimensions, labels, angle keys, confirmation and provenance.
    store = RotatedAnnotationStore(path, expand_views(sources, metadata["angles"]))
    selected = rotation_records(records, requested)
    missing = [view.view_key for view in selected if view.view_key not in store.boxes]
    if missing:
        raise ValueError(f"{len(missing)} rotation views are not confirmed; first: {missing[0]}")
    return selected, {view.view_key: store.boxes[view.view_key] for view in selected}, store.manifest


def angle_policy(angles, weights=None):
    """Canonicalize paired CLI values and derive an exact finite weight cycle."""
    original = tuple(normalize_angle(value) for value in angles)
    parse_angles(original)
    if weights is None:
        if len(original) == 1:
            weights = (1,)
        elif set(original) == set(DEFAULT_ANGLES):
            defaults = dict(zip(DEFAULT_ANGLES, DEFAULT_ANGLE_WEIGHTS))
            weights = tuple(defaults[angle] for angle in original)
        else:
            raise ValueError("custom angle sets require explicit --angle_weights")
    if len(original) != len(weights) or any(not math.isfinite(float(w)) or float(w) <= 0 for w in weights):
        raise ValueError("each angle needs one finite positive weight")
    pairs = sorted(zip(original, (Fraction(str(w)) for w in weights)))
    total = sum(weight for _, weight in pairs)
    normalized = [weight / total for _, weight in pairs]
    period = math.lcm(*(weight.denominator for weight in normalized))
    if period > 1000:
        raise ValueError("angle weights require a cycle longer than 1000 epochs; use simple ratios")
    return {
        "angles": [angle for angle, _ in pairs],
        "weights": [float(weight) for weight in normalized],
        "cycle_epochs": period,
        "cycle_counts": [int(weight * period) for weight in normalized],
    }


class ContentAngleSampler(Sampler):
    """One angle per source per epoch, all synthetic views of that angle.

    Each source receives exact angle proportions over a complete cycle. Angle
    slots and source order are shuffled deterministically per cycle/epoch.
    This locator sampler requires a single resolution per content.
    """

    def __init__(self, records, policy, samples_per_image=8, seed=42):
        self.records, self.policy = list(records), policy
        self.samples_per_image, self.seed, self.epoch = samples_per_image, seed, 0
        if samples_per_image <= 0:
            raise ValueError("samples_per_image must be positive")
        self.groups = defaultdict(dict)
        for index, record in enumerate(self.records):
            group = self.groups[record.content_key]
            if record.angle in group:
                raise ValueError("angle sampler requires one scale/view per content and angle")
            group[record.angle] = index
        expected = set(policy["angles"])
        if not self.groups or any(set(group) != expected for group in self.groups.values()):
            raise ValueError("angle sampler requires complete content/angle coverage")
        self.keys = sorted(self.groups)
        self.schedule = [angle for angle, count in zip(policy["angles"], policy["cycle_counts"])
                         for _ in range(count)]

    def __len__(self):
        return len(self.keys) * self.samples_per_image

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampling epoch must be a nonnegative integer")
        self.epoch = epoch

    def selected_records(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampling epoch must be a nonnegative integer")
        cycle, phase = divmod(epoch, len(self.schedule))
        keys, schedule = list(self.keys), list(self.schedule)
        rng = random.Random(self.seed + cycle)
        rng.shuffle(keys)
        rng.shuffle(schedule)
        return [self.groups[key][schedule[(slot + phase) % len(schedule)]] for slot, key in enumerate(keys)]

    def indices_for_epoch(self, epoch):
        indices = [index * self.samples_per_image + sample
                   for index in self.selected_records(epoch) for sample in range(self.samples_per_image)]
        random.Random(self.seed + epoch).shuffle(indices)
        return indices

    def __iter__(self):
        return iter(self.indices_for_epoch(self.epoch))

    def metadata(self, batch):
        return {"version": 1, "mode": "content_then_angle_cycle", "seed": self.seed,
                **self.policy, "content_count": len(self.keys), "samples_per_image": self.samples_per_image,
                "samples_per_epoch": len(self), "batches_per_epoch": (len(self) + batch - 1) // batch,
                "drop_last": False}

    def epoch_summary(self, epoch, include_views=False):
        records = [self.records[index] for index in self.selected_records(epoch)]
        counts = Counter(angle_text(record.angle) for record in records)
        result = {"epoch": epoch + 1, "contents": len(records), "samples": len(self),
                  "angle_content_counts": {angle_text(a): counts[angle_text(a)] for a in self.policy["angles"]},
                  "angle_sample_counts": {angle_text(a): counts[angle_text(a)] * self.samples_per_image
                                          for a in self.policy["angles"]}}
        if include_views:
            result["views"] = [{"filename": r.path.name, "angle": r.angle} for r in records]
        return result
