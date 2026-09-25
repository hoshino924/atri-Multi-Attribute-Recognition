import copy
from dataclasses import replace
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from dataset import build_dual_view_transform, parse_filename
from face_augmentation import FaceAugmentation, sample_face_offset, summarize_augmentation
from face_regions import FaceCanvas, SquareBox, crop_face, offset_face_box
from labels import TASK_CODES
from preview_augmentation import save_augmentation_previews
from train import make_checkpoint, parse_args, resolve_augmentation, resolve_face_min_scale, run_epoch, validate_args, validate_resume_config


class FixedRandom:
    def __init__(self, choice, offsets):
        self.choice, self.offsets = choice, iter(offsets)

    def random(self):
        return self.choice

    def uniform(self, lower, upper):
        value = next(self.offsets)
        assert lower <= value <= upper
        return value


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.biases = torch.nn.ParameterDict({task: torch.nn.Parameter(torch.zeros(len(codes)))
                                             for task, codes in TASK_CODES.items()})

    def forward(self, full, face):
        return {task: bias.unsqueeze(0).expand(len(full), -1) for task, bias in self.biases.items()}


class FaceAugmentationTests(unittest.TestCase):
    def test_offsets_have_consistent_input_units_at_different_resolutions(self):
        for side, scale, expected in ((200, 1, 25), (600, .5, 50), (1000, .3, 83)):
            base = SquareBox(200, 200, side)
            changed, info = offset_face_box(base, 2000, 2000, 25, -25, scale)
            self.assertEqual(changed, SquareBox(200 + expected, 200 - expected, side))
            self.assertLessEqual(abs(info["actual_dx"] - 25), scale / 2 + 1e-10)
        self.assertEqual(offset_face_box(SquareBox(10, 10, 20), 100, 100, 2.5, -3.5, 1)[0],
                         SquareBox(12, 6, 20))
        for values in ((float("nan"), 0, 1), (0, 0, 0), (0, 0, float("inf"))):
            with self.assertRaises(ValueError):
                offset_face_box(SquareBox(10, 10, 20), 100, 100, *values)

    def test_landmark_motion_matches_bottom_origin_and_crop_direction(self):
        image = Image.new("RGB", (400, 400))
        image.putpixel((120, 170), (255, 0, 0))
        box = SquareBox(100, 100, 200)
        changed, _ = offset_face_box(box, *image.size, 10, 15, 1)
        original = FaceCanvas(300)(crop_face(image, box))
        shifted = FaceCanvas(300)(crop_face(image, changed))
        self.assertEqual(original.getpixel((70, 120)), (255, 0, 0))
        self.assertEqual(shifted.getpixel((60, 135)), (255, 0, 0))
        self.assertEqual(image.getpixel((120, 170)), (255, 0, 0))

    def test_boundary_resampling_and_fallback_do_not_clamp_or_shrink(self):
        box = SquareBox(0, 0, 100)
        policy = FaceAugmentation(position_mode="mixed", face_translate=0, attempts=2)
        self.assertIsNone(offset_face_box(box, 200, 200, -10, 0, 1)[0])
        changed, info = sample_face_offset(box, (200, 200), 1, policy,
                                          FixedRandom(.9, [-25, -25, 10, 20]))
        self.assertEqual(changed, SquareBox(10, 20, 100))
        self.assertEqual((info["bucket"], info["rejected_attempts"], info["fallback"]), ("wide", 1, False))
        changed, info = sample_face_offset(box, (100, 100), 1, policy,
                                          FixedRandom(.9, [-25, -25, 25, 25]))
        self.assertEqual(changed, box)
        self.assertTrue(info["fallback"])
        self.assertEqual((info["actual_dx"], info["actual_dy"], info["rejected_attempts"]), (0, 0, 2))
        self.assertEqual(info["requested_dx"], 25)

    def test_mixture_components_and_no_position_mode(self):
        box = SquareBox(100, 100, 100)
        policy = FaceAugmentation(position_mode="mixed", face_translate=0)
        for choice, offsets, bucket in ((.1, [], "keep"), (.4, [-10, 10], "small"), (.9, [25, -25], "wide")):
            _, info = sample_face_offset(box, (400, 400), 1, policy, FixedRandom(choice, offsets))
            self.assertEqual(info["bucket"], bucket)
        changed, info = sample_face_offset(box, (400, 400), 1, replace(policy, position_mode="none"), FixedRandom(.99, []))
        self.assertEqual(changed, box)
        self.assertEqual(info["bucket"], "keep")
        for values in (dict(keep_probability=.6, small_probability=.5), dict(small_pixels=26),
                       dict(attempts=0), dict(min_scale=float("nan")), dict(face_translate=.02)):
            with self.assertRaises(ValueError):
                replace(policy, **values)

    def test_shrink_and_affine_scale_are_included_in_offset_conversion(self):
        policy = FaceAugmentation(position_mode="mixed", face_translate=0, size_jitter=0,
                                  shrink_probability=1, min_scale=.5, keep_probability=0, small_probability=0,
                                  affine_degrees=0, affine_scale_min=2, affine_scale_max=2,
                                  color_jitter=0, flip_probability=0, full_translate=0)
        transform = build_dual_view_transform(height=100, width=100, expression_size=100,
                                             face_crop=True, train=True, augmentation=policy)
        with patch("dataset.random.random", return_value=.99), patch("dataset.random.uniform", side_effect=lambda low, high: low):
            _, face, info = transform.render(Image.new("RGBA", (600, 600)), SquareBox(200, 200, 200))
        self.assertEqual(face.size, (100, 100))
        self.assertEqual((info["resized_side"], info["rendered_side"], info["output_scale"]), (100, 100, 1))
        self.assertEqual((info["source_dx"], info["actual_dx"]), (-25, -25))

    def test_pair_translation_is_disabled_only_for_face_and_evaluation_is_clean(self):
        policy = FaceAugmentation(position_mode="mixed", face_translate=0)
        transform = build_dual_view_transform(height=100, width=100, expression_size=100,
                                             face_crop=True, train=True, augmentation=policy)
        parameters = dict(flip=False, angle=0, translate_x=1, translate_y=-1, scale=1,
                          brightness=1, contrast=1, saturation=1)
        image = Image.new("RGB", (100, 100))
        with patch("dataset.transform_functional.affine", side_effect=lambda image, **kwargs: image) as affine:
            transform._augment_pair(image, image, parameters)
        self.assertEqual(affine.call_args_list[0].kwargs["translate"], [2, -2])
        self.assertEqual(affine.call_args_list[1].kwargs["translate"], [0, 0])
        clean = build_dual_view_transform(height=100, width=100, expression_size=100,
                                         face_crop=True, augmentation=policy, collect_augmentation=True)
        source = Image.new("RGBA", (200, 200), (70, 80, 90, 255))
        box = SquareBox(50, 50, 100)
        with patch("dataset.sample_face_offset", side_effect=AssertionError("validation must be deterministic")):
            _, face, info = clean.render(source, box)
            views = clean(source, box)
        self.assertEqual(face.tobytes(), crop_face(source, box).convert("RGB").tobytes())
        self.assertEqual(info["actual_dx"], 0)
        self.assertNotIn("_face_augmentation", views)

    def test_preview_and_training_use_the_same_rendered_pixels_and_preserve_sources(self):
        image = Image.new("RGBA", (100, 160), (200, 70, 30, 255))
        box = SquareBox(20, 60, 60)
        policy = FaceAugmentation(position_mode="mixed", face_translate=0)
        transform = build_dual_view_transform(height=80, width=64, expression_size=64, face_crop=True,
                                             train=True, augmentation=policy, collect_augmentation=True)
        random.seed(12)
        torch.manual_seed(12)
        full, face, info = transform.render(image, box)
        random.seed(12)
        torch.manual_seed(12)
        views = transform(image, box)
        torch.testing.assert_close(views["full"], transform._to_tensor(full))
        torch.testing.assert_close(views["expression"], transform._to_tensor(face))
        self.assertEqual(summarize_augmentation([views["_face_augmentation"].tolist()])["samples"], 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ATRI_tatr01_s_d1_p1_f1.png"
            image.save(path)
            before = path.read_bytes()
            result = save_augmentation_previews([parse_filename(path)], {path.name: box}, transform, root, 1, 2)
            self.assertEqual(result["statistics"]["samples"], 2)
            self.assertEqual(path.read_bytes(), before)
            self.assertTrue((root / "augmentation_previews/0001_face.png").is_file())
            with self.assertRaises(FileExistsError):
                save_augmentation_previews([parse_filename(path)], {path.name: box}, transform, root)

    def test_cli_resolution_and_resume_policy_guard(self):
        with patch("sys.argv", ["train.py", "--scale", "all", "--face_cache", "cache"]):
            args = parse_args()
        validate_args(args)
        resolve_face_min_scale(args)
        policy = resolve_augmentation(args)
        self.assertEqual((policy.position_mode, policy.face_translate), ("mixed", 0))
        checkpoint = make_checkpoint(SimpleNamespace(state_dict=lambda: {}), args, 1, 1, [], data_signature="test")
        validate_resume_config(checkpoint, args, "test")
        changed = copy.deepcopy(args)
        changed.face_wide_pixels = 20
        resolve_augmentation(changed)
        with self.assertRaisesRegex(ValueError, "augmentation policy"):
            validate_resume_config(checkpoint, changed, "test")
        with patch("sys.argv", ["train.py", "--scale", "all", "--face_cache", "cache"]):
            old_args = parse_args()
        validate_args(old_args)
        old_checkpoint = copy.deepcopy(checkpoint)
        old_checkpoint["train_args"].pop("augmentation_config")
        resolve_face_min_scale(old_args, old_checkpoint)
        old_policy = resolve_augmentation(old_args, old_checkpoint)
        self.assertEqual((old_policy.position_mode, old_policy.face_translate), ("legacy", .02))
        validate_resume_config(old_checkpoint, old_args, "test")

    def test_worker_metadata_is_counted_without_becoming_model_input(self):
        model = TinyClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        batches = [({"full": torch.zeros(2, 3, 4, 4), "expression": torch.zeros(2, 3, 4, 4),
                     "_face_augmentation": torch.tensor([[0, 0, 0, 0, 0, 0], [2, 2, 1, 0, 0, 1]], dtype=torch.float64)},
                    {task: torch.zeros(2, dtype=torch.long) for task in TASK_CODES})]
        metrics = run_epoch(model, batches, torch.nn.CrossEntropyLoss(), torch.device("cpu"), False,
                            optimizer=optimizer, scaler=torch.amp.GradScaler("cpu", enabled=False))
        stats = metrics["face_augmentation"]
        self.assertEqual((metrics["samples"], metrics["optimizer_steps"], stats["samples"]), (2, 1, 2))
        self.assertEqual((stats["bucket_counts"]["keep"], stats["bucket_counts"]["wide"]), (1, 1))
        self.assertEqual((stats["fallback_samples"], stats["rejected_attempts"], stats["shrunk_samples"]), (1, 2, 1))


if __name__ == "__main__":
    unittest.main()
