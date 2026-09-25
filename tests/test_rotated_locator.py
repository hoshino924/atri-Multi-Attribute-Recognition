from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torchvision.transforms import functional as TF

from dataset import parse_filename, scan_records, split_content_keys
from diagnose_locator_rotations import diagnose, reference_for_view, summarize_rows
from face_locator import FaceNotFoundError
from face_regions import SquareBox, locator_canvas
from image_rotation import DEFAULT_ANGLES, rotate_box_proposal, rotate_image
from locator_data import LocatorDataset, face_free_body
from rotation_annotations import RotatedAnnotationStore, expand_views
from rotation_data import ContentAngleSampler, angle_policy, load_confirmed_rotation_annotations, rotation_records
from train_locator import parse_args, run_epoch, selection_score, train


class RotationSamplingTests(unittest.TestCase):
    def test_weight_pairs_remain_aligned_and_bad_policies_fail(self):
        policy = angle_policy([315, 0, 45, 270, 180, 90], [20, 45, 20, 5, 5, 5])
        self.assertEqual(policy["angles"], list(DEFAULT_ANGLES))
        self.assertEqual(policy["cycle_counts"], [9, 4, 1, 1, 1, 4])
        self.assertEqual(policy["cycle_epochs"], 20)
        for angles, weights in (([0, 360], [1, 1]), ([0, 45], [1]), ([0, 45], [0, 1]),
                                ([0, 45], [float("nan"), 1]), ([0, 30], None),
                                ([0, 45], [1, 10000])):
            with self.subTest(angles=angles, weights=weights), self.assertRaises(ValueError):
                angle_policy(angles, weights)

    def test_each_content_gets_exact_ratios_without_expanding_epoch_budget(self):
        sources = [parse_filename(f"ATRI_tatr{index:02d}_l_d1_p1_f1.png") for index in range(7)]
        records = rotation_records(sources, DEFAULT_ANGLES)
        sampler = ContentAngleSampler(records, angle_policy(DEFAULT_ANGLES), samples_per_image=4, seed=42)
        counts = defaultdict(Counter)
        for epoch in range(40):
            indices = sampler.indices_for_epoch(epoch)
            self.assertEqual(len(indices), 28)
            self.assertEqual(len(set(indices)), 28)
            selected = [records[index] for index in sampler.selected_records(epoch)]
            self.assertEqual(len({record.content_key for record in selected}), 7)
            for record in selected:
                counts[record.content_key][record.angle] += 1
        for values in counts.values():
            self.assertEqual([values[angle] for angle in DEFAULT_ANGLES], [18, 8, 2, 2, 2, 8])
        sampler.set_epoch(27)
        restored = ContentAngleSampler(records, angle_policy(DEFAULT_ANGLES), 4, 42)
        restored.set_epoch(27)
        self.assertEqual(list(sampler), list(restored))
        with self.assertRaises(ValueError):
            ContentAngleSampler(records[:-1], angle_policy(DEFAULT_ANGLES), 4, 42)


class RotatedLocatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".rotated_locator_test_", dir=Path.cwd())
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path.cwd().resolve()))
        self.addCleanup(self.temporary.cleanup)
        self.source = Image.new("RGBA", (96, 192))
        self.source.paste((255, 0, 0, 255), (28, 12, 68, 52))
        self.source.paste((0, 255, 0, 255), (20, 62, 76, 180))
        self.box = SquareBox(28, 140, 40)
        for outfit in ("d1", "d2"):
            self.source.save(self.root / f"ATRI_tatr01_l_{outfit}_p1_f1.png")
        self.records = scan_records(self.root, "l")
        annotation_records = [SimpleNamespace(**vars(record), width=96, height=192) for record in self.records]
        views = expand_views(annotation_records, DEFAULT_ANGLES)
        self.annotation = self.root / "rotation.csv"
        store = RotatedAnnotationStore(self.annotation, views)
        # Synthetic square geometry is exact; fixture confirmations do not
        # replace the real project's manually drawn rotated annotations.
        for view in views:
            store.save(view, rotate_box_proposal(self.box, self.source.size, view.angle))
        self.views, self.boxes, _ = load_confirmed_rotation_annotations(self.records, self.annotation, DEFAULT_ANGLES)
        self.hashes = {record.path: hashlib.sha256(record.path.read_bytes()).hexdigest() for record in self.records}

    def test_annotation_reader_rejects_missing_confirmations_and_changed_pixels(self):
        original = self.annotation.read_bytes()
        lines = self.annotation.read_text(encoding="utf-8-sig").splitlines()
        self.annotation.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8-sig")
        with self.assertRaisesRegex(ValueError, "not confirmed"):
            load_confirmed_rotation_annotations(self.records, self.annotation, DEFAULT_ANGLES)
        self.annotation.write_bytes(original)
        altered = self.source.copy()
        altered.putpixel((0, 0), (5, 5, 5, 255))
        altered.save(self.records[0].path)
        with self.assertRaises(ValueError):
            load_confirmed_rotation_annotations(self.records, self.annotation, DEFAULT_ANGLES)

    def test_clean_views_match_live_rotation_and_targets_for_every_angle(self):
        dataset = LocatorDataset(self.views, self.boxes, size=64, samples_per_image=4, include_metadata=True)
        for index, view in enumerate(self.views):
            tensor, present, target, metadata = dataset[index * 4]
            image = rotate_image(self.source, view.angle)
            clean, (sx, sy, ox, oy) = locator_canvas(image, 64)
            torch.testing.assert_close(tensor, TF.to_tensor(clean))
            left, top, right, bottom = self.boxes[view.view_key].pil_bounds(image.height)
            expected = torch.tensor([((left + right) / 2 * sx + ox) / 64,
                                     ((top + bottom) / 2 * sy + oy) / 64,
                                     (right - left) * sx / 64, (bottom - top) * sy / 64])
            torch.testing.assert_close(target, expected)
            self.assertEqual(present.item(), 1)
            self.assertEqual(metadata["angle"], view.angle)
            self.assertEqual(dataset[index * 4 + 1][1].item(), 0)
        self.assertEqual(self.hashes, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.hashes})

    def test_large_clean_source_does_not_double_resize(self):
        path = self.root / "ATRI_large_l_d1_p1_f1.png"
        image = Image.new("RGBA", (1000, 1800))
        image.paste((130, 60, 250, 255), (200, 73, 789, 1651))
        image.save(path)
        view = rotation_records([parse_filename(path)], [45])[0]
        box = rotate_box_proposal(SquareBox(200, 1127, 600), image.size, 45)
        dataset = LocatorDataset([view], {view.view_key: box}, size=64, samples_per_image=4)
        expected, _ = locator_canvas(rotate_image(image, 45), 64)
        torch.testing.assert_close(dataset[0][0], TF.to_tensor(expected))

    def test_body_negatives_exclude_face_in_all_quarter_turns(self):
        for angle in (0, 90, 180, 270):
            image = rotate_image(self.source, angle)
            box = rotate_box_proposal(self.box, self.source.size, angle)
            body = face_free_body(image, box.pil_bounds(image.height))
            self.assertIsNotNone(body)
            self.assertEqual(body.getchannel("R").getextrema()[1], 0)
            self.assertEqual(body.getchannel("G").getextrema()[1], 255)

    def test_preview_validates_split_without_constructing_a_model(self):
        output = self.root / "preview"
        argv = ["train_locator.py", "--train_dir", str(self.root), "--annotations", str(self.annotation),
                "--rotated_annotations", "--angles", "0", "90", "--angle_weights", "3", "1",
                "--out_dir", str(output), "--input_size", "64", "--samples_per_image", "4", "--preview_only", "--cpu"]
        with patch("sys.argv", argv), patch("train_locator.FaceLocatorNet") as model:
            train(parse_args())
        model.assert_not_called()
        split = json.loads((output / "split.json").read_text(encoding="utf-8"))
        partitions = split_content_keys(split)
        self.assertFalse(partitions["train"] & partitions["validation"])
        preview = json.loads((output / "sampling_preview.json").read_text(encoding="utf-8"))
        self.assertEqual(preview["sampling"]["samples_per_epoch"], 4)
        self.assertEqual(preview["validation"]["samples"], 8)
        self.assertFalse(list(output.glob("*.pth")))
        with patch("sys.argv", argv), self.assertRaises(FileExistsError):
            train(parse_args())

    def test_reference_mapping_is_explicitly_approximate_and_unknown_angles_have_no_label(self):
        view = self.views[0]
        references = {(view.content_key, 0): (self.box, self.source.size, view.path.name)}
        small = parse_filename("ATRI_tatr01_s_d1_p1_f1.png")
        self.assertEqual(reference_for_view(small, 0, (48, 96), references, False), (None, "none"))
        box, kind = reference_for_view(small, 0, (48, 96), references, True)
        self.assertEqual(kind, "scaled_reference")
        self.assertEqual(box, SquareBox(14, 70, 20))
        self.assertEqual(reference_for_view(small, 15, (60, 100), references, True), (None, "none"))

    def test_diagnostics_keep_failures_in_denominator_and_bind_split(self):
        train_record, val_record = self.records
        weight = self.root / "face_locator_best.pth"
        weight.write_bytes(b"mock locator for metadata hashing")
        split = {"train": [train_record.path.name], "validation": [val_record.path.name],
                 "dataset_signature": "fixture", "seed": 42, "val_ratio": .25}
        (self.root / "split.json").write_text(json.dumps(split), encoding="utf-8")
        checkpoint = {"dataset_signature": "fixture", "train_args": {"seed": 42, "val_ratio": .25}, "epoch": 1}
        outer = self
        class Locator:
            size, threshold = 64, .5
            def locate(self, image):
                if image.width > image.height:
                    raise FaceNotFoundError("fixture failure at 90 degrees")
                return outer.box, .99
        args = SimpleNamespace(train_dir=str(self.root), locator_weight=str(weight), locator_split=None,
                               annotations=str(self.annotation), reference_scale="l", scales=["l"],
                               angles=[0, 90], derived_references=False, iou_threshold=.5,
                               preview_per_group=1, output_dir=str(self.root / "diagnostics"), cpu=True)
        with patch("diagnose_locator_rotations.torch.load", return_value=checkpoint), \
             patch("diagnose_locator_rotations.FaceLocator.from_checkpoint", return_value=Locator()):
            result = diagnose(args)
        good, failed = result["conditions"]
        self.assertEqual(good["mean_iou_all_references"], 1)
        self.assertEqual(failed["mean_iou_all_references"], 0)
        self.assertEqual(failed["reference_samples"], 1)
        self.assertEqual(failed["localization_rate_at_iou_threshold"], 0)
        self.assertEqual(result["metadata"]["preview_errors"], [])
        with (Path(args.output_dir) / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2)
        args.output_dir = str(self.root / "bad_split")
        checkpoint["dataset_signature"] = "wrong"
        with patch("diagnose_locator_rotations.torch.load", return_value=checkpoint), self.assertRaises(ValueError):
            diagnose(args)
        self.assertFalse(Path(args.output_dir).exists())


class LocatorMetricsTests(unittest.TestCase):
    def test_missed_positive_is_zero_for_localized_iou_and_weighted_selection(self):
        class Model(torch.nn.Module):
            def forward(self, images):
                return images[:, 0, 0, 0], torch.tensor([[.5, .5, .25, .25]]).repeat(len(images), 1)
        images = torch.zeros(2, 3, 32, 32)
        images[0, 0, 0, 0], images[1, 0, 0, 0] = 10, -10
        batch = images, torch.ones(2), torch.tensor([[.5, .5, .25, .25]]).repeat(2, 1), {"angle": torch.tensor([0., 90.])}
        metrics = run_epoch(Model(), [batch], torch.device("cpu"), .5)
        self.assertEqual(metrics["by_angle"]["90"]["localized_iou"], 0)
        self.assertEqual(metrics["face_recall"], .5)
        self.assertEqual(selection_score(metrics, angle_policy([0, 90], [3, 1])), .75)

    def test_image_error_with_reference_stays_in_quality_denominator(self):
        values = summarize_rows([{"status": "image_error", "reference_kind": "manual"}], .5)
        self.assertEqual(values["reference_samples"], 1)
        self.assertEqual(values["mean_iou_all_references"], 0)
        self.assertIsNone(values["mean_iou_localized"])


if __name__ == "__main__":
    unittest.main()
