from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from dataset import AtriDataset, build_dual_view_transform, parse_filename
from evaluate import task_metrics
from face_locator import FaceLocator, FaceLocatorNet, FaceNotFoundError
from face_regions import SquareBox, locator_canvas
from infer import build_preview_transforms, load_model, prepare_tensors
from locator_data import LocatorDataset
from model import AtriNet
from train import make_checkpoint, resolve_face_min_scale, validate_resume_config
from train_locator import run_epoch


class ConstantLocator(torch.nn.Module):
    def __init__(self, confidence_logit, box):
        super().__init__()
        self.confidence_logit, self.box = confidence_logit, box

    def forward(self, image):
        return torch.tensor([self.confidence_logit]), torch.tensor([self.box])


class FacePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".face_test_", dir=Path.cwd())
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path.cwd().resolve()))
        self.addCleanup(self.temporary.cleanup)
        path = self.root / "ATRI_tatr01_l_d1_p1_f1.png"
        self.image = Image.new("RGBA", (80, 160))
        self.image.paste((180, 70, 30, 255), (20, 10, 60, 150))
        self.image.save(path)
        self.record = parse_filename(path)
        self.box = SquareBox(20, 110, 40)
        self.boxes = {path.name: self.box}

    def args(self, face=True):
        return SimpleNamespace(face_annotations="annotations.csv" if face else None, expression_size=626 if face else 512,
                               height=512, width=320, margin=.04, expression_width_fraction=.65,
                               expression_height_fraction=.5, scale="l" if face else "w", dropout=.2,
                               seed=42, val_ratio=.25)

    def test_locator_data_and_train_step(self):
        data = LocatorDataset([self.record], self.boxes, size=64, samples_per_image=32)
        sample = data[1]
        torch.testing.assert_close(sample[0], data[1][0])
        positives, negatives = 0, 0
        for image, present, box in data:
            self.assertEqual(tuple(image.shape), (3, 64, 64))
            if present:
                positives += 1
                self.assertTrue((box[:2] - box[2:] / 2 >= -1 / 64).all())
                self.assertTrue((box[:2] + box[2:] / 2 <= 1 + 1 / 64).all())
            else:
                negatives += 1
        self.assertGreater(positives, 0)
        self.assertGreater(negatives, 0)
        model = FaceLocatorNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        metrics = run_epoch(model, torch.utils.data.DataLoader(data, batch_size=8), torch.device("cpu"), .5,
                            optimizer, torch.amp.GradScaler("cpu", enabled=False))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))

    def test_locator_mapping_and_presence_rejection(self):
        _, (sx, sy, ox, oy) = locator_canvas(self.image, 64)
        left, top, right, bottom = self.box.pil_bounds(self.image.height)
        box = [((left + right) / 2 * sx + ox) / 64, ((top + bottom) / 2 * sy + oy) / 64,
               self.box.side * sx / 64, self.box.side * sy / 64]
        locator = FaceLocator(ConstantLocator(10., box), torch.device("cpu"), 64)
        self.assertEqual(locator.locate(self.image)[0], self.box)
        locator.model.confidence_logit = -10.
        with self.assertRaises(FaceNotFoundError):
            locator.locate(self.image)

    def test_small_face_training_keeps_reduced_pixels_on_canvas(self):
        image = Image.new("RGBA", (626, 626), (255, 0, 0, 255))
        box = SquareBox(0, 0, 626)
        transform = build_dual_view_transform(
            height=64, width=64, expression_size=626, face_crop=True, train=True,
        )
        with (
            patch("dataset.jitter_face_box", return_value=box),
            patch("dataset.random.random", return_value=0.0),
            patch("dataset.random.uniform", side_effect=lambda lower, upper: lower),
            patch.object(transform, "_augment_pair", side_effect=lambda full, face, parameters=None: (full, face)),
        ):
            views = transform(image, box)
        # 30% of 626 rounds to 188: this is small enough to cover s-size faces.
        self.assertEqual(views["expression"][0].gt(0).sum().item(), 188 * 188)
        evaluation = build_dual_view_transform(
            height=64, width=64, expression_size=626, face_crop=True, train=False,
        )
        self.assertEqual(evaluation(image, box)["expression"][0].gt(0).sum().item(), 626 * 626)

    def test_face_scale_defaults_resume_and_validation(self):
        args = self.args()
        args.face_min_scale = None
        resolve_face_min_scale(args)
        self.assertEqual(args.face_min_scale, 0.30)
        args.face_min_scale = None
        resolve_face_min_scale(args, {"train_args": {}})
        self.assertEqual(args.face_min_scale, 0.55)
        args.face_min_scale = None
        resolve_face_min_scale(args, {"train_args": {"face_min_scale": 0.4}})
        self.assertEqual(args.face_min_scale, 0.4)
        for invalid in (0, -0.1, 1.1, float("nan")):
            args.face_min_scale = invalid
            with self.assertRaises(ValueError):
                resolve_face_min_scale(args)
            with self.assertRaises(ValueError):
                build_dual_view_transform(face_crop=True, face_min_scale=invalid)

    def test_checkpoint_modes_and_shared_preprocessing(self):
        model = AtriNet(pretrained=False)
        for face in (False, True):
            args = self.args(face)
            checkpoint = make_checkpoint(model, args, 1, 1., [], data_signature="data")
            path = self.root / "classifier.pth"
            torch.save(checkpoint, path)
            restored, transforms, _ = load_model(path, torch.device("cpu"))
            paired = build_dual_view_transform(face_crop=face, expression_size=args.expression_size)
            if face:
                with self.assertRaisesRegex(ValueError, "locator_weight"):
                    prepare_tensors(self.image, torch.device("cpu"), transforms)
                actual = transforms["expression"](self.image, self.box)
                expected = paired(self.image, self.box)["expression"]
                preview = build_preview_transforms(restored)["expression"](self.image, self.box)
                self.assertEqual(preview.size, (626, 626))
                views = AtriDataset([self.record], paired_transform=paired, face_boxes=self.boxes)[0][0]
                with torch.inference_mode():
                    output = restored(views["full"].unsqueeze(0), views["expression"].unsqueeze(0))
                self.assertEqual(tuple(output["expression"].shape), (1, 21))
            else:
                actual = transforms["expression"](self.image)
                expected = paired(self.image)["expression"]
            torch.testing.assert_close(actual, expected)
            validate_resume_config(checkpoint, args, "data")
            if face:
                # Historical format-4 runs omit the augmentation parameter.
                resume_args = self.args()
                resume_args.face_min_scale = None
                resolve_face_min_scale(resume_args, checkpoint)
                validate_resume_config(checkpoint, resume_args, "data")
                resume_args.face_min_scale = 0.30
                with self.assertRaisesRegex(ValueError, "face_min_scale"):
                    validate_resume_config(checkpoint, resume_args, "data")
            with self.assertRaises(ValueError):
                validate_resume_config(checkpoint, self.args(not face), "data")
            with self.assertRaises(ValueError):
                validate_resume_config(checkpoint, args, "changed")

    def test_failed_localization_is_counted_in_evaluation(self):
        logits = torch.tensor([[10., -10.], [10., -10.]])
        metrics = task_metrics(logits, torch.tensor([0, 0]), None, torch.tensor([True, False]))
        self.assertEqual(metrics["accuracy"], .5)
        self.assertEqual(metrics["coverage"], .5)
        self.assertEqual(metrics["per_class_accuracy"][0], .5)
        self.assertEqual(metrics["predictions"], [0, -1])


if __name__ == "__main__":
    unittest.main()
