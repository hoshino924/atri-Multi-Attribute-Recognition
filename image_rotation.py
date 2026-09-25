# -*- coding: utf-8 -*-
"""Pillow-only, centered counterclockwise rotation shared by all image views."""

import math

from PIL import Image

from face_regions import SquareBox


DEFAULT_ANGLES = (0, 45, 90, 180, 270, 315)
DEFAULT_ANGLE_WEIGHTS = (0.45, 0.20, 0.05, 0.05, 0.05, 0.20)
ROTATION_CONTRACT = {
    "version": 1,
    "direction": "counterclockwise",
    "center": "source_canvas_center",
    "canvas": "ceil_rotated_extent_centered",
    "interpolation": "bicubic",
    "quarter_turns": "exact_transpose",
    "fill": [0, 0, 0, 0],
    "coordinate_system": "rotated_bottom_left_pixels",
}


def normalize_angle(angle):
    angle = float(angle)
    if not math.isfinite(angle):
        raise ValueError("Rotation angles must be finite numbers")
    return angle % 360.0


def parse_angles(values):
    angles = tuple(normalize_angle(value) for value in values)
    if not angles or len(set(angles)) != len(angles):
        raise ValueError("Choose distinct angles; 0 and 360 are the same direction")
    return tuple(sorted(angles))


def angle_text(angle):
    angle = normalize_angle(angle)
    return str(int(angle)) if angle.is_integer() else repr(angle)


def rotation_geometry(size, angle):
    """Return expanded size and forward rotation coefficients in pixel edges."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    angle = normalize_angle(angle)
    radians = math.radians(angle)
    cosine = round(math.cos(radians), 15)
    sine = round(math.sin(radians), 15)
    output_size = (
        math.ceil(abs(width * cosine) + abs(height * sine)),
        math.ceil(abs(width * sine) + abs(height * cosine)),
    )
    return output_size, cosine, sine


def rotate_image(image, angle):
    """Rotate directly from the source; preserve alpha and the complete canvas."""
    angle = normalize_angle(angle)
    image = image.convert("RGBA")
    if angle == 0:
        return image.copy()
    exact = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180,
             270: Image.Transpose.ROTATE_270}
    if angle in exact:
        return image.transpose(exact[angle])
    size, cosine, sine = rotation_geometry(image.size, angle)
    cx, cy = image.width / 2, image.height / 2
    ox, oy = size[0] / 2, size[1] / 2
    # Pillow requests destination-to-source coordinates; image y points down.
    inverse = (cosine, -sine, cx - cosine * ox + sine * oy,
               sine, cosine, cy - sine * ox - cosine * oy)
    return image.transform(size, Image.Transform.AFFINE, inverse,
                           resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0))


def rotate_box_proposal(box, source_size, angle):
    """Enclose a transformed box with a square; this is never a human label."""
    if not box.fits(*source_size):
        raise ValueError("Reference box is outside its canvas")
    size, cosine, sine = rotation_geometry(source_size, angle)
    cx, cy = source_size[0] / 2, source_size[1] / 2
    ox, oy = size[0] / 2, size[1] / 2
    left, top, right, bottom = box.pil_bounds(source_size[1])
    points = [(ox + cosine * (x - cx) + sine * (y - cy),
               oy - sine * (x - cx) + cosine * (y - cy))
              for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))]
    x0, x1 = min(p[0] for p in points), max(p[0] for p in points)
    y0, y1 = min(p[1] for p in points), max(p[1] for p in points)
    side = min(*size, max(1, math.ceil(max(x1 - x0, y1 - y0) - 1e-9)))
    left = max(0, min(size[0] - side, round((x0 + x1 - side) / 2)))
    top = max(0, min(size[1] - side, round((y0 + y1 - side) / 2)))
    return SquareBox(left, size[1] - top - side, side)


def scale_box_proposal(box, source_size, target_size):
    """Map same-angle reference pixels between resolution variants."""
    if not box.fits(*source_size):
        raise ValueError("Reference box is outside its canvas")
    sx, sy = target_size[0] / source_size[0], target_size[1] / source_size[1]
    side = min(*target_size, max(1, round(box.side * max(sx, sy))))
    result = SquareBox(round((box.x_left + box.side / 2) * sx - side / 2),
                       round((box.y_bottom + box.side / 2) * sy - side / 2), side)
    return result.moved(0, 0, *target_size)
