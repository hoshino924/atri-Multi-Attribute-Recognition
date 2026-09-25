# -*- coding: utf-8 -*-
"""Reproducible on-demand synthetic scenes; never save or alter source images."""

from collections import OrderedDict
import math
import random

from PIL import Image, ImageDraw, ImageEnhance
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from face_regions import locator_canvas
from image_rotation import rotate_image


def face_free_body(image, bounds):
    """Choose foreground outside the face rectangle in any orientation."""
    left, top, right, bottom = bounds
    rectangles = ((0, 0, image.width, max(0, math.floor(top))),
                  (0, min(image.height, math.ceil(bottom)), image.width, image.height),
                  (0, 0, max(0, math.floor(left)), image.height),
                  (min(image.width, math.ceil(right)), 0, image.width, image.height))
    choices = []
    for rectangle in rectangles:
        if rectangle[2] > rectangle[0] and rectangle[3] > rectangle[1]:
            crop = image.crop(rectangle)
            foreground = crop.getchannel("A").getbbox()
            if foreground:
                area = (foreground[2] - foreground[0]) * (foreground[3] - foreground[1])
                choices.append((area, crop))
    return max(choices, key=lambda choice: choice[0])[1] if choices else None


class LocatorDataset(Dataset):
    def __init__(self, records, boxes, size=256, samples_per_image=8, seed=42, include_metadata=False):
        self.records = list(records)
        self.boxes = boxes
        self.size = size
        self.samples_per_image = samples_per_image
        self.seed = seed
        self.epoch = 0
        self.include_metadata = include_metadata
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.records) * self.samples_per_image

    def _source(self, record):
        rotated = hasattr(record, "angle")
        name = record.view_key if rotated else record.path.name
        if name not in self.cache:
            with Image.open(record.path) as source:
                image = rotate_image(source, record.angle) if rotated else source.convert("RGBA")
            bounds = self.boxes[name].pil_bounds(image.height)
            if not self.boxes[name].fits(*image.size):
                raise ValueError(f"locator annotation outside rotated canvas: {name}")
            # New rotation runs use the same direct clean resize as live
            # localization. Legacy runs retain their existing double resize.
            clean = self._clean_view(image, bounds) if rotated else None
            original_size = image.size
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)
            sx, sy = image.width / original_size[0], image.height / original_size[1]
            bounds = (bounds[0] * sx, bounds[1] * sy, bounds[2] * sx, bounds[3] * sy)
            self.cache[name] = image, bounds, clean
            if len(self.cache) > 128:
                self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        return self.cache[name]

    def _clean_view(self, image, bounds):
        left, top, right, bottom = bounds
        scene, (sx, sy, ox, oy) = locator_canvas(image, self.size)
        target = ((left + right) / 2 * sx + ox, (top + bottom) / 2 * sy + oy,
                  (right - left) * sx, (bottom - top) * sy)
        return scene, target

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rng = random.Random(self.seed + self.epoch * 1000003 + index)
        record = self.records[index // self.samples_per_image]
        image, bounds, clean = self._source(record)
        left, top, right, bottom = bounds
        n = self.size
        # One deterministic clean view per artwork also covers normal sprite use.
        if index % self.samples_per_image == 0:
            scene, target = clean if clean is not None else self._clean_view(image, bounds)
            present = 1.0
        else:
            color = lambda: tuple(rng.randrange(256) for _ in range(3))
            scene = Image.new("RGBA", (n, n), (*color(), 255))
            draw = ImageDraw.Draw(scene)
            for _ in range(rng.randrange(3, 12)):
                x, y = rng.randrange(n), rng.randrange(n)
                draw.rectangle((x, y, x + rng.randrange(5, n), y + rng.randrange(5, n)), fill=(*color(), 255))
            # Always include a negative per source; remaining synthetic views
            # are positive so even small validation splits exercise both cases.
            present = float(index % self.samples_per_image != 1)
            target = (0, 0, 0, 0)
            if present:
                face_side = rng.uniform(0.10, 0.60) * n
                scale = face_side / max(right - left, bottom - top)
                resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
                sx, sy = resized.width / image.width, resized.height / image.height
                w, h = (right - left) * sx, (bottom - top) * sy
                cx, cy = rng.uniform(w / 2, n - w / 2), rng.uniform(h / 2, n - h / 2)
                ox = round(cx - (left + right) / 2 * sx)
                oy = round(cy - (top + bottom) / 2 * sy)
                scene.alpha_composite(resized, (ox, oy))
                target = ((left + right) / 2 * sx + ox, (top + bottom) / 2 * sy + oy, w, h)
            elif rng.random() < 0.7:
                # Body-only negatives help avoid recognizing any sprite as a face.
                if hasattr(record, "angle"):
                    body = face_free_body(image, bounds)
                else:
                    body = (image.crop((0, min(image.height - 1, int(bottom) + 1), image.width, image.height))
                            if bottom + 1 < image.height else None)
                if body is not None:
                    body.thumbnail((n, n), Image.Resampling.LANCZOS)
                    scene.alpha_composite(body, (rng.randrange(n - body.width + 1), rng.randrange(n - body.height + 1)))
            scene = ImageEnhance.Brightness(scene.convert("RGB")).enhance(rng.uniform(0.7, 1.2))
        result = TF.to_tensor(scene), torch.tensor(present), torch.tensor(target, dtype=torch.float32) / n
        if self.include_metadata:
            return (*result, {"angle": float(getattr(record, "angle", 0)), "scale": record.scale,
                              "clean": index % self.samples_per_image == 0})
        return result
