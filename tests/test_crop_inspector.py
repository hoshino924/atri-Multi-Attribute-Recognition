from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from torchvision.transforms import functional as TF

from dataset import ForegroundRegionCrop, build_expression_transform, build_transform
from face_locator import FaceNotFoundError, LocatedFacePreview, LocatedFaceTransform
from face_regions import SquareBox
from inspect_faces import display_geometry, filter_paths, inspect_image, screen_bounds
from labels import TASK_CODES


class RecordingClassifier(torch.nn.Module):
    def forward(self, full, expression):
        self.expression_seen = expression.clone()
        return {task: torch.zeros((len(full), len(codes))) for task, codes in TASK_CODES.items()}


class CountingLocator:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def locate(self, image):
        self.calls += 1
        if self.fail:
            raise FaceNotFoundError("test: no face")
        return SquareBox(4, 26, 8), .99


class CropInspectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "ATRI_tatr01_w_d1_p1_f1.png"
        self.image = Image.new("RGBA", (20, 40))
        self.image.paste((190, 70, 30, 255), (4, 6, 16, 38))
        self.image.save(self.path)

    def model_and_transforms(self, face=True):
        model = RecordingClassifier().eval()
        model.calibration = {}
        model.preprocess_config = {
            "full": {"height": 20, "width": 10, "margin": 0},
            "expression": {"size": 12, "width_fraction": .5, "height_fraction": .25, "margin": .1},
            "background": (0, 0, 0), "normalization": {"mean": (0, 0, 0), "std": (1, 1, 1)},
        }
        transforms = {"full": build_transform(height=20, width=10, margin=0, mean=(0, 0, 0), std=(1, 1, 1))}
        if face:
            model.face_preview = LocatedFacePreview(12, (0, 0, 0), CountingLocator())
            transforms["expression"] = LocatedFaceTransform(model.face_preview, (0, 0, 0), (1, 1, 1))
        else:
            transforms["expression"] = build_expression_transform(
                size=12, width_fraction=.5, height_fraction=.25, margin=.1,
                mean=(0, 0, 0), std=(1, 1, 1),
            )
        return model, transforms

    def test_legacy_shared_bounds_preserve_crop_pixels_and_rgb_fallback(self):
        cropper = ForegroundRegionCrop(.5, .25)
        self.assertEqual(cropper.bounds(self.image), (7, 6, 13, 14))
        self.assertEqual(cropper(self.image).tobytes(), self.image.crop((7, 6, 13, 14)).tobytes())
        rgb = Image.new("RGB", (20, 40), "white")
        self.assertEqual(cropper.bounds(rgb), (5, 0, 15, 10))
        self.assertEqual(cropper(rgb).mode, "RGBA")
        transparent = Image.new("RGBA", (20, 40))
        self.assertEqual(cropper.bounds(transparent), (5, 0, 15, 10))

    def test_face_views_match_actual_inference_and_reuse_the_detected_box(self):
        model, transforms = self.model_and_transforms()
        before = self.path.read_bytes()
        result = inspect_image(model, transforms, TASK_CODES, torch.device("cpu"), self.path)
        self.assertEqual(result.status, "ok")
        self.assertEqual(model.face_preview.locator.calls, 1)
        self.assertEqual(result.bounds, (4, 6, 12, 14))
        self.assertEqual(result.crop.size, (8, 8))
        self.assertEqual(result.model_input.size, (12, 12))
        torch.testing.assert_close(TF.to_tensor(result.model_input), model.expression_seen[0])
        self.assertEqual(result.crop.tobytes(), self.image.crop(result.bounds).tobytes())
        self.assertEqual(self.path.read_bytes(), before)
        self.assertAlmostEqual(result.locator_confidence, .99)

    def test_legacy_preview_matches_the_tensor_including_margin(self):
        model, transforms = self.model_and_transforms(face=False)
        result = inspect_image(model, transforms, TASK_CODES, torch.device("cpu"), self.path)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.bounds, (7, 6, 13, 14))
        self.assertEqual(result.crop.size, (6, 8))
        self.assertIsNone(result.locator_confidence)
        torch.testing.assert_close(TF.to_tensor(result.model_input), model.expression_seen[0])

    def test_failed_localization_keeps_original_but_clears_prior_crop(self):
        model, transforms = self.model_and_transforms()
        first = inspect_image(model, transforms, TASK_CODES, torch.device("cpu"), self.path)
        self.assertEqual(first.status, "ok")
        model.face_preview.locator.fail = True
        failed = inspect_image(model, transforms, TASK_CODES, torch.device("cpu"), self.path)
        self.assertEqual(failed.status, "no_face")
        self.assertEqual(failed.image.tobytes(), self.image.tobytes())
        for field in ("bounds", "crop", "model_input", "predictions", "locator_confidence"):
            self.assertIsNone(getattr(failed, field))
        model.face_preview.locator.fail = False
        recovered = inspect_image(model, transforms, TASK_CODES, torch.device("cpu"), self.path)
        self.assertEqual(recovered.status, "ok")
        self.assertEqual(model.face_preview.locator.calls, 3)

    def test_unreadable_file_never_shows_previous_image(self):
        model, transforms = self.model_and_transforms()
        inspect_image(model, transforms, TASK_CODES, torch.device("cpu"), self.path)
        failed = inspect_image(model, transforms, TASK_CODES, torch.device("cpu"), self.path.with_name("missing.png"))
        self.assertEqual(failed.status, "read_error")
        self.assertIsNone(failed.image)
        self.assertIsNone(failed.bounds)
        self.assertIsNone(failed.predictions)

    def test_filter_pairs_resolution_variants_and_accepts_arbitrary_filenames(self):
        root = self.path.parent
        w = root / "ATRI_tatr01_w_d1_p1_f1.png"
        s = root / "ATRI_tatr01_s_d1_p1_f1.png"
        other = root / "ATRI_tatr01_s_d1_p1_f2.png"
        plain = root / "custom.png"
        self.assertEqual(filter_paths([plain, other, w, s]), [s, w, other, plain])
        self.assertEqual(filter_paths([plain, other, w, s], "s", "F1"), [s])
        self.assertEqual(filter_paths([plain, w], "w"), [w])
        self.assertEqual(filter_paths([plain], text="missing"), [])

    def test_display_overlay_uses_exact_rounded_scale_and_native_coordinates(self):
        rendered, offset = display_geometry((101, 203), (350, 420))
        box = (10, 20, 91, 193)
        projected = screen_bounds(box, (101, 203), rendered, offset)
        self.assertAlmostEqual((projected[0] - offset[0]) / (rendered[0] / 101), 10)
        self.assertAlmostEqual((projected[3] - offset[1]) / (rendered[1] / 203), 193)
        rendered, offset = display_geometry((101, 203), (80, 90), native=True)
        self.assertEqual(rendered, (101, 203))
        self.assertEqual(screen_bounds(box, (101, 203), rendered, offset), (18, 28, 99, 201))


if __name__ == "__main__":
    unittest.main()
