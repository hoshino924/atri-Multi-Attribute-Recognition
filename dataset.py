# -*- coding: utf-8 -*-
"""Dataset parsing, splitting, and image transforms."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random

from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from labels import EXPR_CODES, OUTFIT_CODES, POSE_CODES


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SCALE_CODES = ("s", "w", "m", "l", "ll")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageRecord:
    """One image and the metadata encoded in its filename."""

    path: Path
    character: str
    shoe_variant: str
    scale: str
    outfit: str
    pose: str
    expression: str

    @property
    def content_key(self):
        """Identify resolution variants of the same source artwork."""
        return (
            self.character,
            self.shoe_variant,
            self.outfit,
            self.pose,
            self.expression,
        )


def parse_filename(path):
    """Parse and validate one dataset filename."""
    path = Path(path)
    parts = path.stem.split("_")
    if len(parts) != 6:
        raise ValueError(
            "expected 6 underscore-separated fields: "
            "character_shoe-variant_scale_outfit_pose_expression"
        )

    character, shoe_variant, scale, outfit, pose, expression = parts
    if not character or not shoe_variant:
        raise ValueError("character and shoe-variant fields cannot be empty")
    if scale not in SCALE_CODES:
        raise ValueError(f"unknown scale code: {scale}")
    if outfit not in OUTFIT_CODES:
        raise ValueError(f"unknown outfit code: {outfit}")
    if pose not in POSE_CODES:
        raise ValueError(f"unknown pose code: {pose}")
    if expression not in EXPR_CODES:
        raise ValueError(f"unknown expression code: {expression}")

    return ImageRecord(
        path=path,
        character=character,
        shoe_variant=shoe_variant,
        scale=scale,
        outfit=outfit,
        pose=pose,
        expression=expression,
    )


def scan_records(root, scale="w"):
    """Scan a dataset directory and select one resolution level."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {root}")
    if scale not in SCALE_CODES:
        raise ValueError(f"unknown scale code: {scale}")

    records = []
    errors = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            record = parse_filename(path)
        except ValueError as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        if record.scale == scale:
            records.append(record)

    if errors:
        details = "\n".join(f"  - {error}" for error in errors[:20])
        if len(errors) > 20:
            details += f"\n  - ... and {len(errors) - 20} more"
        raise ValueError(f"invalid dataset filenames:\n{details}")
    if not records:
        raise ValueError(f"no '{scale}' images found in {root}")

    seen = set()
    duplicates = []
    for record in records:
        if record.content_key in seen:
            duplicates.append(record.path.name)
        seen.add(record.content_key)
    if duplicates:
        names = ", ".join(duplicates[:10])
        raise ValueError(f"duplicate content records for scale '{scale}': {names}")

    return records


def stratified_split(records, val_ratio=0.25, seed=42):
    """Split by pose and expression while keeping each content item intact."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("at least two records are required for a train/val split")

    rng = random.Random(seed)
    strata = defaultdict(list)
    for record in records:
        strata[(record.pose, record.expression)].append(record)

    train_records = []
    val_records = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda item: item.path.name)
        rng.shuffle(group)
        if len(group) == 1:
            train_records.extend(group)
            continue

        val_count = max(1, round(len(group) * val_ratio))
        val_count = min(val_count, len(group) - 1)
        val_records.extend(group[:val_count])
        train_records.extend(group[val_count:])

    if not val_records:
        shuffled = list(train_records)
        rng.shuffle(shuffled)
        val_records = [shuffled.pop()]
        train_records = shuffled

    return train_records, val_records


class FitPad:
    """Fit an image inside a fixed canvas without changing its aspect ratio."""

    def __init__(self, height, width, margin=0.04, background=(0, 0, 0)):
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        if not 0.0 <= margin < 0.5:
            raise ValueError("margin must be in [0, 0.5)")
        self.height = height
        self.width = width
        self.margin = margin
        self.background = background

    def __call__(self, image):
        image = image.convert("RGBA")
        inner_width = max(1, round(self.width * (1.0 - 2.0 * self.margin)))
        inner_height = max(1, round(self.height * (1.0 - 2.0 * self.margin)))
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        resized = ImageOps.contain(
            image,
            (inner_width, inner_height),
            method=resampling,
        )

        canvas = Image.new(
            "RGBA",
            (self.width, self.height),
            (*self.background, 255),
        )
        left = (self.width - resized.width) // 2
        top = (self.height - resized.height) // 2
        canvas.alpha_composite(resized, (left, top))
        return canvas.convert("RGB")


class ForegroundRegionCrop:
    """Crop the upper-center region of the non-transparent foreground."""

    def __init__(self, width_fraction=0.65, height_fraction=0.50):
        if not 0.0 < width_fraction <= 1.0:
            raise ValueError("width_fraction must be in (0, 1]")
        if not 0.0 < height_fraction <= 1.0:
            raise ValueError("height_fraction must be in (0, 1]")
        self.width_fraction = width_fraction
        self.height_fraction = height_fraction

    def __call__(self, image):
        image = image.convert("RGBA")
        foreground = image.getchannel("A").getbbox()
        if foreground is None:
            foreground = (0, 0, image.width, image.height)

        left, top, right, bottom = foreground
        foreground_width = right - left
        foreground_height = bottom - top
        crop_width = max(1, round(foreground_width * self.width_fraction))
        crop_height = max(1, round(foreground_height * self.height_fraction))
        center_x = (left + right) / 2.0

        crop_left = max(left, round(center_x - crop_width / 2.0))
        crop_right = min(right, crop_left + crop_width)
        crop_left = max(left, crop_right - crop_width)
        crop_bottom = min(bottom, top + crop_height)
        return image.crop((crop_left, top, crop_right, crop_bottom))


def _post_fit_operations(train, mean, std):
    """Build augmentations and tensor normalization after geometric fitting."""
    operations = []
    if train:
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=3,
                translate=(0.02, 0.02),
                scale=(0.95, 1.02),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.ColorJitter(
                brightness=0.08,
                contrast=0.08,
                saturation=0.08,
            ),
        ])
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return operations


def build_transform(
    height=512,
    width=320,
    train=False,
    margin=0.04,
    background=(0, 0, 0),
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
):
    """Build matching train or inference preprocessing."""
    operations = [
        FitPad(
            height=height,
            width=width,
            margin=margin,
            background=background,
        )
    ]
    operations.extend(_post_fit_operations(train, mean, std))
    return transforms.Compose(operations)


def build_expression_transform(
    size=512,
    train=False,
    width_fraction=0.65,
    height_fraction=0.50,
    margin=0.04,
    background=(0, 0, 0),
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
):
    """Build preprocessing for the focused upper-body expression view."""
    operations = [
        ForegroundRegionCrop(
            width_fraction=width_fraction,
            height_fraction=height_fraction,
        ),
        FitPad(
            height=size,
            width=size,
            margin=margin,
            background=background,
        ),
    ]
    operations.extend(_post_fit_operations(train, mean, std))
    return transforms.Compose(operations)


class AtriDataset(Dataset):
    """Dataset for outfit, pose, and expression classification."""

    def __init__(self, records, full_transform, expression_transform):
        self.records = list(records)
        self.full_transform = full_transform
        self.expression_transform = expression_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.path) as source:
            image = source.convert("RGBA")
        views = {
            "full": self.full_transform(image),
            "expression": self.expression_transform(image),
        }

        targets = {
            "outfit": OUTFIT_CODES.index(record.outfit),
            "pose": POSE_CODES.index(record.pose),
            "expression": EXPR_CODES.index(record.expression),
        }
        return views, targets
