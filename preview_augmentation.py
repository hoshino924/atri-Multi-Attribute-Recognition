# -*- coding: utf-8 -*-
"""Export actual training inputs and crop metadata without constructing a model."""

from collections import defaultdict
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageOps

from face_augmentation import augmentation_stat_values, summarize_augmentation
from face_regions import FaceCanvas, SquareBox, crop_face
from dataset import open_record_image, record_box_key
from image_rotation import angle_text


def save_augmentation_previews(records, boxes, transform, output_dir, count=21, repeats=4, seed=42):
    """Choose up to count training views per scale/angle; export repeats per view."""
    out = Path(output_dir)
    previews = out / "augmentation_previews"
    previews.mkdir(exist_ok=False)
    groups = defaultdict(list)
    for record in records:
        groups[(record.scale, getattr(record, "angle", 0))].append(record)
    chooser = random.Random(seed)  # Selection does not consume the augmentation RNG.
    rows = []
    for (scale, angle), group in sorted(groups.items()):
        chosen = chooser.sample(sorted(group, key=lambda item: item.path.name), min(count, len(group)))
        for record in chosen:
            image = open_record_image(record)
            box = boxes[record_box_key(record)]
            clean = FaceCanvas(transform.expression_preview.size, transform.background)(crop_face(image, box))
            for repeat in range(repeats):
                full, face, info = transform.render(image, box)
                index = len(rows) + 1
                # Face files contain the exact input before tensor normalization.
                face.save(previews / f"{index:04d}_face.png")
                full.save(previews / f"{index:04d}_full.png")
                rgba = Image.new("RGBA", image.size, (0, 0, 0, 255))
                rgba.alpha_composite(image)
                original = ImageOps.contain(rgba.convert("RGB"), (256, 384))
                draw = ImageDraw.Draw(original)
                shifted = SquareBox(info["x_left"], info["y_bottom"], info["side"])
                for rectangle, color in ((box, "lime"), (shifted, "red")):
                    left, top, right, bottom = rectangle.pil_bounds(image.height)
                    draw.rectangle((left * original.width / image.width, top * original.height / image.height,
                                    right * original.width / image.width, bottom * original.height / image.height),
                                   outline=color, width=2)
                card = Image.new("RGB", (856, 440), (35, 39, 46))
                card.paste(original, ((256 - original.width) // 2, 36))
                card.paste(ImageOps.contain(clean, (300, 300)), (256, 36))
                card.paste(ImageOps.contain(face, (300, 300)), (556, 36))
                draw = ImageDraw.Draw(card)
                draw.text((8, 8), f"#{index} {scale} angle={angle_text(angle)} green=original red=training | clean | augmented", fill="white")
                draw.text((8, 422), f"{info['bucket']} dx={info['actual_dx']:.2f} dy={info['actual_dy']:.2f} "
                          f"retries={info['rejected_attempts']} fallback={info['fallback']}", fill="white")
                card.save(previews / f"{index:04d}_comparison.png")
                rows.append({"index": index, "filename": record.path.name, "scale": scale, "angle": angle,
                             "repeat": repeat + 1, **info})
    result = {
        "description": "Preview draws only, not executed training counts. Source files are unchanged.",
        "policy": transform.augmentation.metadata(),
        "offset_units": "face-input pixels including shrink, canvas fit and affine scale; axes before shared flip/rotation; +x right, +y up for the crop (content moves oppositely)",
        "statistics": summarize_augmentation(augmentation_stat_values(row) for row in rows),
        "by_scale": {scale: summarize_augmentation(augmentation_stat_values(row) for row in rows if row["scale"] == scale)
                     for scale in sorted({r.scale for r in records})},
        "by_angle_scale": {f"{angle_text(angle)}/{scale}": summarize_augmentation(
            augmentation_stat_values(row) for row in rows if row["scale"] == scale and row["angle"] == angle)
            for scale, angle in groups},
        "samples": rows,
    }
    (out / "augmentation_preview.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
