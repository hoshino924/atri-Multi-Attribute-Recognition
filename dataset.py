# -*- coding: utf-8 -*-
"""Dataset parsing, splitting, and image transforms."""

from collections import defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path
import random

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as transform_functional

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


@dataclass(frozen=True)
class LabeledImageRecord:
    """One externally named image with labels supplied by a CSV manifest."""

    path: Path
    outfit: str
    pose: str
    expression: str


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


def load_label_manifest(root, manifest_path):
    """Load filename and three task labels from a UTF-8 CSV manifest."""
    root = Path(root).resolve()
    manifest_path = Path(manifest_path)
    required_columns = {"filename", "outfit", "pose", "expression"}
    records = []
    seen_paths = set()

    if not root.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"label manifest does not exist: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        missing = required_columns - fieldnames
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"label manifest is missing columns: {names}")

        for line_number, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            if not filename:
                raise ValueError(f"empty filename at manifest line {line_number}")

            path = (root / filename).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"filename escapes the image directory at line {line_number}: "
                    f"{filename}"
                ) from exc
            if not path.is_file():
                raise FileNotFoundError(
                    f"manifest image does not exist at line {line_number}: {path}"
                )
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(
                    f"unsupported image extension at line {line_number}: {filename}"
                )
            if path in seen_paths:
                raise ValueError(
                    f"duplicate manifest filename at line {line_number}: {filename}"
                )

            labels = {
                task: (row.get(task) or "").strip()
                for task in ("outfit", "pose", "expression")
            }
            for task, code in labels.items():
                valid_codes = {
                    "outfit": OUTFIT_CODES,
                    "pose": POSE_CODES,
                    "expression": EXPR_CODES,
                }[task]
                if code not in valid_codes:
                    raise ValueError(
                        f"unknown {task} code at line {line_number}: {code}"
                    )

            records.append(
                LabeledImageRecord(
                    path=path,
                    outfit=labels["outfit"],
                    pose=labels["pose"],
                    expression=labels["expression"],
                )
            )
            seen_paths.add(path)

    if not records:
        raise ValueError(f"label manifest contains no records: {manifest_path}")
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


def build_full_preview(
    height=512,
    width=320,
    margin=0.04,
    background=(0, 0, 0),
):
    """Build the deterministic full-image view as a displayable PIL image."""
    return FitPad(
        height=height,
        width=width,
        margin=margin,
        background=background,
    )


def build_expression_preview(
    size=512,
    width_fraction=0.65,
    height_fraction=0.50,
    margin=0.04,
    background=(0, 0, 0),
):
    """Build the deterministic expression crop as a displayable PIL image."""
    return transforms.Compose([
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
    ])


class PairedViewTransform:
    """Create both views while sharing all randomly sampled augmentations."""

    def __init__(
        self,
        full_preview,
        expression_preview,
        train=False,
        background=(0, 0, 0),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ):
        self.full_preview = full_preview
        self.expression_preview = expression_preview
        self.train = train
        self.background = tuple(background)
        self.mean = tuple(mean)
        self.std = tuple(std)

    @staticmethod
    def _uniform(low, high):
        return torch.empty(1).uniform_(low, high).item()

    def _augment_pair(self, full_image, expression_image):
        images = [full_image, expression_image]
        if torch.rand(1).item() < 0.5:
            images = [transform_functional.hflip(image) for image in images]

        angle = self._uniform(-3.0, 3.0)
        translate_x = self._uniform(-0.02, 0.02)
        translate_y = self._uniform(-0.02, 0.02)
        scale = self._uniform(0.95, 1.02)
        images = [
            transform_functional.affine(
                image,
                angle=angle,
                translate=[
                    round(translate_x * image.width),
                    round(translate_y * image.height),
                ],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=self.background,
            )
            for image in images
        ]

        brightness = self._uniform(0.92, 1.08)
        contrast = self._uniform(0.92, 1.08)
        saturation = self._uniform(0.92, 1.08)
        adjusted = []
        for image in images:
            image = transform_functional.adjust_brightness(image, brightness)
            image = transform_functional.adjust_contrast(image, contrast)
            image = transform_functional.adjust_saturation(image, saturation)
            adjusted.append(image)
        return adjusted

    def _to_tensor(self, image):
        tensor = transform_functional.to_tensor(image)
        return transform_functional.normalize(tensor, self.mean, self.std)

    def __call__(self, image):
        full_image = self.full_preview(image)
        expression_image = self.expression_preview(image)
        if self.train:
            full_image, expression_image = self._augment_pair(
                full_image,
                expression_image,
            )
        return {
            "full": self._to_tensor(full_image),
            "expression": self._to_tensor(expression_image),
        }


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
    operations = [build_full_preview(height, width, margin, background)]
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
        build_expression_preview(
            size=size,
            width_fraction=width_fraction,
            height_fraction=height_fraction,
            margin=margin,
            background=background,
        )
    ]
    operations.extend(_post_fit_operations(train, mean, std))
    return transforms.Compose(operations)


def build_dual_view_transform(
    height=512,
    width=320,
    expression_size=512,
    train=False,
    expression_width_fraction=0.65,
    expression_height_fraction=0.50,
    margin=0.04,
    background=(0, 0, 0),
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
):
    """Build paired preprocessing for training or deterministic evaluation."""
    return PairedViewTransform(
        full_preview=build_full_preview(
            height=height,
            width=width,
            margin=margin,
            background=background,
        ),
        expression_preview=build_expression_preview(
            size=expression_size,
            width_fraction=expression_width_fraction,
            height_fraction=expression_height_fraction,
            margin=margin,
            background=background,
        ),
        train=train,
        background=background,
        mean=mean,
        std=std,
    )


class AtriDataset(Dataset):
    """Dataset for outfit, pose, and expression classification."""

    def __init__(
        self,
        records,
        full_transform=None,
        expression_transform=None,
        paired_transform=None,
        task_codes=None,
    ):
        self.records = list(records)
        self.full_transform = full_transform
        self.expression_transform = expression_transform
        self.paired_transform = paired_transform
        task_codes = task_codes or {
            "outfit": OUTFIT_CODES,
            "pose": POSE_CODES,
            "expression": EXPR_CODES,
        }
        self.code_to_index = {
            task: {code: index for index, code in enumerate(codes)}
            for task, codes in task_codes.items()
        }
        if paired_transform is None and (
            full_transform is None or expression_transform is None
        ):
            raise ValueError(
                "provide paired_transform or both full and expression transforms"
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.path) as source:
            image = source.convert("RGBA")
        if self.paired_transform is not None:
            views = self.paired_transform(image)
        else:
            views = {
                "full": self.full_transform(image),
                "expression": self.expression_transform(image),
            }

        targets = {
            task: self.code_to_index[task][getattr(record, task)]
            for task in ("outfit", "pose", "expression")
        }
        return views, targets
