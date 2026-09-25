# -*- coding: utf-8 -*-
"""Versioned, manually confirmed rotated annotations; no training dependency."""

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time

from PIL import Image, __version__ as PILLOW_VERSION

from annotation_i18n import AnnotationError
from face_regions import CSV_FIELDS, COORDINATE_SYSTEM, SquareBox
from image_rotation import (
    ROTATION_CONTRACT, angle_text, parse_angles, rotate_box_proposal,
    rotate_image, rotation_geometry, scale_box_proposal,
)


LABEL_FIELDS = ("character", "shoe_variant", "scale", "outfit", "pose", "expression")
ROTATED_CSV_FIELDS = (
    "filename", "content_id", *LABEL_FIELDS, "image_width", "image_height",
    "source_sha256", "angle", "canvas_width", "canvas_height", "coordinate_system",
    "x_left", "y_bottom", "side", "origin", "reference", "confirmed",
)
ORIGINS = {"manual", "manual_adjusted", "inherited", "rotated_reference", "scaled_reference"}
_WINDOWS_RETRY_DELAYS = (0.05, 0.10, 0.20, 0.40, 0.80)


def digest(data):
    return hashlib.sha256(data).hexdigest() if data is not None else None


@dataclass(frozen=True)
class AnnotationView:
    path: Path
    character: str
    shoe_variant: str
    scale: str
    outfit: str
    pose: str
    expression: str
    source_width: int
    source_height: int
    source_sha256: str
    angle: float
    width: int
    height: int

    @property
    def key(self):
        return self.path.name, self.angle

    @property
    def group(self):
        return self.angle, self.character, self.scale, self.pose

    @property
    def content_id(self):
        return "_".join((self.character, self.shoe_variant, self.outfit, self.pose, self.expression))


def expand_views(records, angles):
    """Preserve source order inside angle-major groups, hashing each source once."""
    sources = []
    for item in records:
        data = item.path.read_bytes()
        with Image.open(io.BytesIO(data)) as source:
            if source.size != (item.width, item.height):
                raise AnnotationError("dimensions_changed")
        sources.append((item, digest(data)))
    views = []
    for angle in parse_angles(angles):
        for item, signature in sources:
            width, height = rotation_geometry((item.width, item.height), angle)[0]
            views.append(AnnotationView(
                item.path, *(getattr(item, field) for field in LABEL_FIELDS),
                item.width, item.height, signature, angle, width, height,
            ))
    return views


def atomic_write(path, data, before_replace):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=f".{path.stem}.", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(len(_WINDOWS_RETRY_DELAYS) + 1):
            try:
                # Recheck on every attempt: another process may edit the file
                # while a Windows access/sharing/lock violation delays us.
                before_replace()
                os.replace(temporary_path, path)
                break
            except OSError as exc:
                if (getattr(exc, "winerror", None) not in (5, 32, 33)
                        or attempt == len(_WINDOWS_RETRY_DELAYS)):
                    raise
                time.sleep(_WINDOWS_RETRY_DELAYS[attempt])
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                # A locked temporary file may remain; never mask the save error.
                pass


class RotatedAnnotationStore:
    """Immutable session manifest plus atomic CSV updates of confirmed rows only."""

    def __init__(self, path, records):
        self.path = Path(path).resolve()
        self.manifest_path = self.path.with_suffix(".meta.json")
        if self.path.suffix.lower() != ".csv":
            raise AnnotationError("csv_only")
        self.records = {item.key: item for item in records}
        if not self.records or len(self.records) != len(records):
            raise AnnotationError("csv_image")
        source_paths = {item.path.resolve() for item in records}
        if self.path in source_paths or self.manifest_path in source_paths:
            raise AnnotationError("output_is_image")
        self.boxes, self.provenance, self.last_boxes, self.last_keys = {}, {}, {}, {}
        self.references = {}
        sources = {}
        for item in records:
            sources[item.path.name] = {
                "sha256": item.source_sha256, "width": item.source_width,
                "height": item.source_height,
                **{field: getattr(item, field) for field in LABEL_FIELDS},
            }
        self.manifest = {
            "version": 2, "rotation": ROTATION_CONTRACT, "pillow_version": PILLOW_VERSION,
            "angles": list(dict.fromkeys(item.angle for item in records)),
            "sources": sources,
        }
        data = self.path.read_bytes() if self.path.exists() else None
        metadata = self.manifest_path.read_bytes() if self.manifest_path.exists() else None
        self.signature, self.meta_signature = digest(data), digest(metadata)
        if metadata is not None:
            if json.loads(metadata.decode("utf-8")) != self.manifest:
                raise AnnotationError("rotation_metadata")
        elif data is not None:
            raise AnnotationError("rotation_metadata")
        if data is not None:
            self._load(data)

    def _row(self, item, box, origin, reference):
        return {
            "filename": item.path.name, "content_id": item.content_id,
            **{field: getattr(item, field) for field in LABEL_FIELDS},
            "image_width": item.source_width, "image_height": item.source_height,
            "source_sha256": item.source_sha256, "angle": angle_text(item.angle),
            "canvas_width": item.width, "canvas_height": item.height,
            "coordinate_system": ROTATION_CONTRACT["coordinate_system"],
            "x_left": box.x_left, "y_bottom": box.y_bottom, "side": box.side,
            "origin": origin, "reference": reference, "confirmed": "true",
        }

    def _load(self, data):
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        if reader.fieldnames != list(ROTATED_CSV_FIELDS):
            raise AnnotationError("csv_header")
        for line, row in enumerate(reader, 2):
            try:
                key = row["filename"], float(row["angle"])
                if key not in self.records or key in self.boxes:
                    raise AnnotationError("csv_image")
                item = self.records[key]
                box = SquareBox(*(int(row[field]) for field in ("x_left", "y_bottom", "side")))
                if not box.fits(item.width, item.height):
                    raise AnnotationError("csv_box")
                origin, reference = row["origin"], row["reference"]
                if origin not in ORIGINS or reference is None:
                    raise AnnotationError("csv_metadata")
                expected = {field: str(value) for field, value in self._row(item, box, origin, reference).items()}
                if row != expected:
                    raise AnnotationError("csv_metadata")
                self.boxes[key] = box
                self.provenance[key] = origin, reference
                self.last_boxes[item.group], self.last_keys[item.group] = box, key
            except (KeyError, ValueError, TypeError) as exc:
                raise AnnotationError("csv_row", line=line, error=exc) from exc

    def _check_external_change(self):
        for path, signature in ((self.path, self.signature), (self.manifest_path, self.meta_signature)):
            data = path.read_bytes() if path.exists() else None
            if digest(data) != signature:
                raise AnnotationError("csv_changed")

    def load_view(self, item):
        data = item.path.read_bytes()
        if digest(data) != item.source_sha256:
            raise AnnotationError("source_changed", filename=item.path.name)
        with Image.open(io.BytesIO(data)) as source:
            image = rotate_image(source, item.angle)
        if image.size != (item.width, item.height):
            raise AnnotationError("dimensions_changed")
        return image

    def save(self, item, box, origin="manual", reference=""):
        if item.key not in self.records or self.records[item.key] != item:
            raise AnnotationError("csv_image")
        if not box.fits(item.width, item.height):
            raise AnnotationError("box_bounds")
        if origin not in ORIGINS:
            raise AnnotationError("csv_metadata")
        if digest(item.path.read_bytes()) != item.source_sha256:
            raise AnnotationError("source_changed", filename=item.path.name)
        self._check_external_change()
        # The manifest is immutable. An interrupted first CSV save can resume
        # from a valid manifest with zero confirmed rows.
        if self.meta_signature is None:
            metadata = (json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            atomic_write(self.manifest_path, metadata, self._check_external_change)
            self.meta_signature = digest(metadata)
        boxes, provenance = dict(self.boxes), dict(self.provenance)
        boxes.pop(item.key, None)
        boxes[item.key] = box
        provenance[item.key] = origin, reference
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=ROTATED_CSV_FIELDS)
        writer.writeheader()
        for key, current in boxes.items():
            writer.writerow(self._row(self.records[key], current, *provenance[key]))
        data = stream.getvalue().encode("utf-8-sig")
        atomic_write(self.path, data, self._check_external_change)
        self.signature = digest(data)
        self.boxes, self.provenance = boxes, provenance
        self.last_boxes[item.group], self.last_keys[item.group] = box, item.key

    def suggestion(self, item):
        if item.key in self.boxes:
            return self.boxes[item.key], *self.provenance[item.key]
        if item.group in self.last_boxes:
            return self.last_boxes[item.group], "inherited", json.dumps(self.last_keys[item.group], ensure_ascii=False)
        reference = self.references.get((item.content_id, item.angle))
        if reference is not None:
            box, size, identity = reference
            return scale_box_proposal(box, size, (item.width, item.height)), "scaled_reference", identity
        upright = self.records.get((item.path.name, 0.0))
        if upright is not None and upright.key in self.boxes:
            box = rotate_box_proposal(self.boxes[upright.key], (upright.width, upright.height), item.angle)
            return box, "rotated_reference", json.dumps(upright.key, ensure_ascii=False)
        reference = self.references.get((item.content_id, 0.0))
        if reference is not None:
            box, size, identity = reference
            box = scale_box_proposal(box, size, (item.source_width, item.source_height))
            return rotate_box_proposal(box, (item.source_width, item.source_height), item.angle), "rotated_reference", identity
        return None, "manual", ""

    def add_reference(self, path, source_records):
        """Load old 0-degree or new rotated annotations as unconfirmed proposals."""
        path = Path(path).resolve()
        if path == self.path or path == self.manifest_path:
            raise AnnotationError("reference_is_output")
        data = path.read_bytes()
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        source_records = {item.path.name: item for item in source_records}
        references = {}
        if reader.fieldnames == list(ROTATED_CSV_FIELDS):
            metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
            selected = [source_records[name] for name in metadata["sources"] if name in source_records]
            if len(selected) != len(metadata["sources"]):
                raise AnnotationError("csv_image")
            other = RotatedAnnotationStore(path, expand_views(selected, metadata["angles"]))
            if other.signature != digest(data):
                raise AnnotationError("csv_changed")
            for key, box in other.boxes.items():
                item = other.records[key]
                if (item.content_id, item.angle) in references:
                    raise AnnotationError("reference_duplicate")
                identity = json.dumps([str(path), digest(data), item.path.name, item.angle], ensure_ascii=False)
                references[item.content_id, item.angle] = box, (item.width, item.height), identity
        elif reader.fieldnames == list(CSV_FIELDS):
            seen = set()
            for line, row in enumerate(reader, 2):
                try:
                    item = source_records[row["filename"]]
                    if item.path.name in seen:
                        raise AnnotationError("csv_image")
                    seen.add(item.path.name)
                    box = SquareBox(*(int(row[field]) for field in ("x_left", "y_bottom", "side")))
                    expected = {
                        "filename": item.path.name,
                        **{field: getattr(item, field) for field in LABEL_FIELDS},
                        "image_width": item.width, "image_height": item.height,
                        "coordinate_system": COORDINATE_SYSTEM,
                        "x_left": box.x_left, "y_bottom": box.y_bottom, "side": box.side,
                    }
                    if row != {field: str(value) for field, value in expected.items()} or not box.fits(item.width, item.height):
                        raise AnnotationError("csv_metadata")
                    content = "_".join((item.character, item.shoe_variant, item.outfit, item.pose, item.expression))
                    if (content, 0.0) in references:
                        raise AnnotationError("reference_duplicate")
                    identity = json.dumps([str(path), digest(data), item.path.name, 0], ensure_ascii=False)
                    references[content, 0.0] = box, (item.width, item.height), identity
                except (KeyError, ValueError, TypeError) as exc:
                    raise AnnotationError("csv_row", line=line, error=exc) from exc
        else:
            raise AnnotationError("csv_header")
        self.references = references
