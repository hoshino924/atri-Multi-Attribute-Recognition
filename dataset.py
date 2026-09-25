# -*- coding: utf-8 -*-
"""Dataset parsing, splitting, and image transforms."""

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path
import random

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
from torchvision.transforms import functional as transform_functional

from labels import EXPR_CODES, OUTFIT_CODES, POSE_CODES
from face_regions import FaceCanvas, crop_face, jitter_face_box
from face_augmentation import FaceAugmentation, augmentation_stat_values, sample_face_offset
from image_rotation import rotate_image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SCALE_CODES = ("s", "w", "m", "l", "ll")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_FACE_MIN_SCALE = 0.30
LEGACY_FACE_MIN_SCALE = 0.55


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
    """Select one resolution, or require complete matched coverage for all five."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {root}")
    if scale not in (*SCALE_CODES, "all"):
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
        if scale == "all" or record.scale == scale:
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
        key = (record.content_key, record.scale)
        if key in seen:
            duplicates.append(record.path.name)
        seen.add(key)
    if duplicates:
        names = ", ".join(duplicates[:10])
        raise ValueError(f"duplicate content records for scale '{scale}': {names}")

    if scale == "all":
        content_scale_indices(records, SCALE_CODES)
    return records


def content_scale_indices(records, scales):
    """Index each content/scale once, rejecting incomplete resolution groups."""
    groups = defaultdict(dict)
    for index, record in enumerate(records):
        if record.scale not in scales:
            raise ValueError(f"unexpected scale: {record.path.name}")
        group = groups[record.content_key]
        if record.scale in group:
            raise ValueError(f"duplicate content/scale: {record.path.name}")
        group[record.scale] = index
    if not groups:
        raise ValueError("no content records")
    for key, group in groups.items():
        missing = set(scales) - group.keys()
        if missing:
            raise ValueError(f"content {'_'.join(key)} is missing scales: {', '.join(sorted(missing))}")
    return dict(groups)


def split_content_keys(manifest):
    """Read grouped split identities; resolution variants may share one side."""
    if not isinstance(manifest, dict):
        raise ValueError("invalid split manifest")
    result = {}
    for name in ("train", "validation"):
        entries = manifest.get(name)
        if not isinstance(entries, list) or not entries or not all(isinstance(item, str) for item in entries):
            raise ValueError(f"invalid {name} split")
        records = [parse_filename(Path(item.replace("\\", "/")).name) for item in entries]
        views = [(record.content_key, record.scale) for record in records]
        if len(set(views)) != len(views):
            raise ValueError(f"duplicate content/scale in {name} split")
        result[name] = {record.content_key for record in records}
    if result["train"] & result["validation"]:
        raise ValueError("training and validation contain the same artwork")
    return result


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
    contents = defaultdict(list)
    for record in records:
        contents[record.content_key].append(record)
    if len(contents) < 2:
        raise ValueError("at least two contents are required for a train/val split")

    rng = random.Random(seed)
    strata = defaultdict(list)
    for key, variants in contents.items():
        record = variants[0]
        strata[(record.pose, record.expression)].append(key)

    train_records = []
    val_records = []
    for key in sorted(strata):
        # For a single scale this has the same order as sorting filenames.
        # Shuffle contents first so adding resolutions cannot change the split.
        group = sorted(strata[key])
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

    def expand(keys):
        return [record for key in keys for record in sorted(contents[key], key=lambda item: item.path.name)]

    return expand(train_records), expand(val_records)


class ContentBalancedSampler(Sampler):
    """One view per content per epoch; all scales once in every scale cycle.

    Each epoch's scale counts differ by at most one. The remainder rotates, so
    a complete five-epoch cycle has exactly 20% of each scale. Sampling is a
    function of seed and absolute epoch, independent of loader/worker RNG.
    """

    VERSION = 1

    def __init__(self, records, scales=SCALE_CODES, seed=42):
        self.records = list(records)
        self.scales = tuple(scales)
        if not self.scales or len(set(self.scales)) != len(self.scales):
            raise ValueError("sampling scales must be nonempty and unique")
        if any(scale not in SCALE_CODES for scale in self.scales):
            raise ValueError("unknown sampling scale")
        self.groups = content_scale_indices(self.records, self.scales)
        self.keys = sorted(self.groups)
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.keys)

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampling epoch must be a nonnegative integer")
        self.epoch = epoch

    def indices_for_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampling epoch must be a nonnegative integer")
        cycle, phase = divmod(epoch, len(self.scales))
        slots = list(self.keys)
        random.Random(self.seed + cycle).shuffle(slots)
        selected = {
            key: self.groups[key][self.scales[(index + phase) % len(self.scales)]]
            for index, key in enumerate(slots)
        }
        order = list(self.keys)
        random.Random(self.seed + epoch).shuffle(order)
        return [selected[key] for key in order]

    def __iter__(self):
        return iter(self.indices_for_epoch(self.epoch))

    def metadata(self, batch_size):
        return {
            "version": self.VERSION,
            "mode": "content_then_scale_cycle",
            "seed": self.seed,
            "scales": list(self.scales),
            "scale_weights": {scale: 1.0 / len(self.scales) for scale in self.scales},
            "cycle_epochs": len(self.scales),
            "content_count": len(self),
            "samples_per_epoch": len(self),
            "batches_per_epoch": (len(self) + batch_size - 1) // batch_size,
            "drop_last": False,
            "angle": 0,
        }

    def epoch_summary(self, epoch, include_filenames=False):
        selected = [self.records[index] for index in self.indices_for_epoch(epoch)]
        counts = Counter(record.scale for record in selected)
        summary = {
            "epoch": epoch + 1,
            "samples": len(selected),
            "scale_counts": {scale: counts[scale] for scale in self.scales},
        }
        if include_filenames:
            summary["filenames"] = [record.path.name for record in selected]
        return summary


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

    def bounds(self, image):
        """Return the original-image PIL rectangle used by the legacy crop."""
        if image.mode != "RGBA":
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
        return crop_left, top, crop_right, crop_bottom

    def __call__(self, image):
        image = image.convert("RGBA")
        return image.crop(self.bounds(image))


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
    """Share appearance/orientation parameters; control face-crop shifts separately."""

    def __init__(
        self,
        full_preview,
        expression_preview,
        train=False,
        background=(0, 0, 0),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        face_min_scale=DEFAULT_FACE_MIN_SCALE,
        augmentation=None,
        collect_augmentation=False,
    ):
        if not 0.0 < face_min_scale <= 1.0:
            raise ValueError("face_min_scale must be in (0, 1]")
        self.full_preview = full_preview
        self.expression_preview = expression_preview
        self.train = train
        self.background = tuple(background)
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.face_min_scale = face_min_scale
        self.augmentation = augmentation or FaceAugmentation(min_scale=face_min_scale)
        self.collect_augmentation = collect_augmentation

    @staticmethod
    def _uniform(low, high):
        return torch.empty(1).uniform_(low, high).item()

    def _sample_pair_parameters(self):
        policy = self.augmentation
        return {
            "flip": torch.rand(1).item() < policy.flip_probability,
            "angle": self._uniform(-policy.affine_degrees, policy.affine_degrees),
            "translate_x": self._uniform(-1.0, 1.0),
            "translate_y": self._uniform(-1.0, 1.0),
            "scale": self._uniform(policy.affine_scale_min, policy.affine_scale_max),
            "brightness": self._uniform(1 - policy.color_jitter, 1 + policy.color_jitter),
            "contrast": self._uniform(1 - policy.color_jitter, 1 + policy.color_jitter),
            "saturation": self._uniform(1 - policy.color_jitter, 1 + policy.color_jitter),
        }

    def _augment_pair(self, full_image, expression_image, parameters=None):
        parameters = parameters or self._sample_pair_parameters()
        images = [full_image, expression_image]
        if parameters["flip"]:
            images = [transform_functional.hflip(image) for image in images]

        policy = self.augmentation
        images = [
            transform_functional.affine(
                image,
                angle=parameters["angle"],
                translate=[
                    round(parameters["translate_x"] * fraction * image.width),
                    round(parameters["translate_y"] * fraction * image.height),
                ],
                scale=parameters["scale"],
                shear=[0.0, 0.0],
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=self.background,
            )
            for image, fraction in zip(images, (policy.full_translate, policy.face_translate))
        ]

        adjusted = []
        for image in images:
            image = transform_functional.adjust_brightness(image, parameters["brightness"])
            image = transform_functional.adjust_contrast(image, parameters["contrast"])
            image = transform_functional.adjust_saturation(image, parameters["saturation"])
            adjusted.append(image)
        return adjusted

    def _to_tensor(self, image):
        tensor = transform_functional.to_tensor(image)
        return transform_functional.normalize(tensor, self.mean, self.std)

    def render(self, image, face_box=None):
        """Return the actual pre-normalization inputs and auditable crop metadata."""
        info = {}
        parameters = self._sample_pair_parameters() if self.train else None
        full_image = self.full_preview(image)
        if isinstance(self.expression_preview, FaceCanvas):
            if face_box is None:
                raise ValueError("annotated face preprocessing requires a face box")
            if not face_box.fits(*image.size):
                raise ValueError("face box is outside the image")
            original_box = face_box
            policy = self.augmentation
            if self.train:
                if policy.position_mode == "legacy":
                    face_box = jitter_face_box(face_box, *image.size, random, policy.size_jitter)
                    anchor = original_box.resized(face_box.side - original_box.side, *image.size)
                else:
                    face_box = face_box.resized(round(face_box.side * random.uniform(
                        -policy.size_jitter, policy.size_jitter)), *image.size)
                    anchor = face_box
            else:
                anchor = face_box
            side = face_box.side
            if self.train and random.random() < policy.shrink_probability:
                side = max(1, round(side * random.uniform(policy.min_scale, 1.0)))
            rendered_side = min(side, self.expression_preview.size)
            # Account for actual rounded shrink size and the later affine scale.
            # Offset axes are defined before the shared flip/rotation, so that
            # these orientation transforms do not change the sampled strength.
            output_scale = rendered_side / face_box.side * (parameters["scale"] if self.train else 1.0)
            if self.train and policy.position_mode != "legacy":
                face_box, info = sample_face_offset(face_box, image.size, output_scale, policy, random)
            else:
                dx, dy = face_box.x_left - anchor.x_left, face_box.y_bottom - anchor.y_bottom
                info = {"bucket": "legacy" if self.train else "keep", "rejected_attempts": 0,
                        "fallback": False, "source_dx": dx, "source_dy": dy,
                        "requested_dx": dx * output_scale, "requested_dy": dy * output_scale,
                        "actual_dx": dx * output_scale, "actual_dy": dy * output_scale,
                        "output_scale": output_scale}
            info.update(original_x=original_box.x_left, original_y=original_box.y_bottom,
                        original_side=original_box.side, anchor_x=anchor.x_left, anchor_y=anchor.y_bottom,
                        x_left=face_box.x_left, y_bottom=face_box.y_bottom, side=face_box.side,
                        resized_side=side, rendered_side=rendered_side, shrunk=side < face_box.side,
                        affine=parameters)
            face_image = crop_face(image, face_box)
            if side != face_box.side:
                face_image = face_image.resize((side, side), Image.Resampling.LANCZOS)
            expression_image = self.expression_preview(face_image)
        else:
            expression_image = self.expression_preview(image)
        if self.train:
            full_image, expression_image = self._augment_pair(
                full_image,
                expression_image,
                parameters,
            )
        return full_image, expression_image, info

    def __call__(self, image, face_box=None):
        full_image, expression_image, info = self.render(image, face_box)
        views = {
            "full": self._to_tensor(full_image),
            "expression": self._to_tensor(expression_image),
        }
        if self.train and self.collect_augmentation and info:
            views["_face_augmentation"] = torch.tensor(augmentation_stat_values(info), dtype=torch.float64)
        return views


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
    face_crop=False,
    face_min_scale=DEFAULT_FACE_MIN_SCALE,
    augmentation=None,
    collect_augmentation=False,
):
    """Build paired preprocessing for training or deterministic evaluation."""
    return PairedViewTransform(
        full_preview=build_full_preview(
            height=height,
            width=width,
            margin=margin,
            background=background,
        ),
        expression_preview=FaceCanvas(expression_size, background) if face_crop else build_expression_preview(
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
        face_min_scale=face_min_scale,
        augmentation=augmentation,
        collect_augmentation=collect_augmentation,
    )


def open_record_image(record):
    """Decode once and rotate the source canvas before either classifier view."""
    with Image.open(record.path) as source:
        image = source.convert("RGBA")
    return rotate_image(image, record.angle) if hasattr(record, "angle") else image


def record_box_key(record):
    return (record.path.name, record.angle) if hasattr(record, "angle") else record.path.name


class AtriDataset(Dataset):
    """Dataset for outfit, pose, and expression classification."""

    def __init__(
        self,
        records,
        full_transform=None,
        expression_transform=None,
        paired_transform=None,
        task_codes=None,
        face_boxes=None,
        view_metadata=False,
        angle_weights=None,
    ):
        self.records = list(records)
        self.full_transform = full_transform
        self.expression_transform = expression_transform
        self.paired_transform = paired_transform
        self.face_boxes = face_boxes
        self.view_metadata = view_metadata
        self.angle_weights = angle_weights
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
        image = open_record_image(record)
        if self.paired_transform is not None:
            views = (
                self.paired_transform(image, self.face_boxes[record_box_key(record)])
                if self.face_boxes is not None else self.paired_transform(image)
            )
        else:
            views = {
                "full": self.full_transform(image),
                "expression": (
                    self.expression_transform(image, self.face_boxes[record_box_key(record)])
                    if self.face_boxes is not None else self.expression_transform(image)
                ),
            }

        if self.view_metadata:
            views["_view"] = torch.tensor([SCALE_CODES.index(record.scale), record.angle], dtype=torch.float64)
        if self.angle_weights is not None:
            views["_sample_weight"] = torch.tensor(self.angle_weights[record.angle], dtype=torch.float64)
        targets = {
            task: self.code_to_index[task][getattr(record, task)]
            for task in ("outfit", "pose", "expression")
        }
        return views, targets
