"""D2 regression checks; execute with unittest on the user's Python environment."""

from collections import Counter, defaultdict
import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image, __version__ as PILLOW_VERSION
import torch
from torchvision.transforms import functional as TF

from calibration import build_calibration, collect_logits, sample_weights
from classifier_rotation import ContentScaleAngleSampler, validation_selection
from dataset import AtriDataset, SCALE_CODES, build_dual_view_transform, parse_filename, scan_records, stratified_split
from diagnose_classifier_rotations import consistency, diagnose, summarize_rotations
from face_box_cache import ROTATED_CSV_FIELDS, file_sha256, generate_cache, load_face_cache
from face_regions import SquareBox
from image_rotation import DEFAULT_ANGLES, ROTATION_CONTRACT, rotate_image
from infer import load_model
from labels import TASK_CODES
from rotation_data import angle_policy, rotation_records
from train import make_checkpoint, parse_args, run_epoch, train, validate_args, validate_resume_config


class JointSamplingTests(unittest.TestCase):
    def records(self, count=17):
        sources = [parse_filename(f"ART{i:03d}_tatr01_{s}_d1_p1_f1.png") for i in range(count) for s in SCALE_CODES]
        return rotation_records(sources, DEFAULT_ANGLES)

    def test_every_content_covers_every_scale_at_each_angle_in_two_full_cycles(self):
        records = self.records()
        policy = angle_policy(DEFAULT_ANGLES)
        sampler = ContentScaleAngleSampler(records, SCALE_CODES, policy)
        self.assertEqual(sampler.cycle_epochs, 100)
        for start in (0, 100):
            seen = defaultdict(Counter)
            for epoch in range(start, start + 100):
                selected = [records[i] for i in sampler.indices_for_epoch(epoch)]
                self.assertEqual(len({r.content_key for r in selected}), 17)
                self.assertEqual(len(selected), 17)
                counts = Counter(r.scale for r in selected)
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
                for record in selected:
                    seen[record.content_key][record.scale, record.angle] += 1
            expected = Counter({(s, a): c for s in SCALE_CODES for a, c in zip(policy["angles"], policy["cycle_counts"])})
            for values in seen.values():
                self.assertEqual(values, expected)

    def test_resume_reordering_and_upright_control_keep_content_order_and_budget(self):
        records = self.records()
        upright = [r for r in records if r.angle == 0]
        a = ContentScaleAngleSampler(records, SCALE_CODES, angle_policy(DEFAULT_ANGLES))
        reversed_records = list(reversed(records))
        b = ContentScaleAngleSampler(reversed_records, SCALE_CODES, angle_policy(DEFAULT_ANGLES))
        control = ContentScaleAngleSampler(upright, SCALE_CODES, angle_policy([0]))
        for epoch in (0, 4, 99, 100, 239, 499):
            a.set_epoch(epoch)
            b.set_epoch(epoch)
            self.assertEqual([records[i].view_key for i in a], [reversed_records[i].view_key for i in b])
            self.assertEqual([records[i].content_key for i in a],
                             [upright[i].content_key for i in control.indices_for_epoch(epoch)])
        self.assertEqual(control.metadata(16)["batches_per_epoch"], a.metadata(16)["batches_per_epoch"])

    def test_incomplete_duplicate_and_invalid_sampling_inputs_fail(self):
        records = self.records(1)
        for bad in (records[:-1], records + records[:1]):
            with self.assertRaises(ValueError):
                ContentScaleAngleSampler(bad, SCALE_CODES, angle_policy(DEFAULT_ANGLES))
        sampler = ContentScaleAngleSampler(records, SCALE_CODES, angle_policy(DEFAULT_ANGLES))
        with self.assertRaises(ValueError):
            sampler.indices_for_epoch(-1)

    def test_weighted_selection_cannot_ignore_outfit_and_pose(self):
        groups = {"0/s": {"task_loss": {"outfit": 3, "pose": 2, "expression": 1},
                           "accuracy": dict.fromkeys(TASK_CODES, 1), "joint_accuracy": 1},
                  "90/s": {"task_loss": dict.fromkeys(TASK_CODES, 10),
                            "accuracy": dict.fromkeys(TASK_CODES, 0), "joint_accuracy": 0}}
        result = validation_selection(groups, angle_policy([0, 90], [3, 1]), ["s"], TASK_CODES)
        self.assertAlmostEqual(result["loss"], 4)
        self.assertAlmostEqual(result["joint_accuracy"], .75)


class SyntheticLocator:
    size, threshold = 64, .5

    def __init__(self):
        self.inputs = []

    def locate(self, image):
        self.inputs.append((image.size, image.tobytes()))
        return SquareBox(3, 4, 12), .99


class RotatedClassifierPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".classifier_rotation_test_", dir=Path.cwd())
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        source = Image.new("RGBA", (40, 80))
        source.paste((255, 0, 0, 255), (10, 4, 26, 22))
        source.paste((0, 255, 0, 255), (6, 25, 30, 74))
        for outfit in ("d1", "d2"):
            for scale in SCALE_CODES:
                source.save(self.images / f"ATRI_tatr01_{scale}_{outfit}_p1_f1.png")
        self.records = scan_records(self.images, "all")
        self.training, self.validation = stratified_split(self.records)
        self.policy = {**angle_policy(DEFAULT_ANGLES, [1] * 6), "rotation": ROTATION_CONTRACT,
                       "pillow_version": PILLOW_VERSION, "synthesis_version": 2}
        self.split = {"train": [r.path.name for r in self.training if r.scale == "l"],
                      "validation": [r.path.name for r in self.validation if r.scale == "l"],
                      "dataset_signature": "fixture", "seed": 42, "val_ratio": .25,
                      "rotation_policy": self.policy}
        self.weight = self.root / "locator.pth"
        self.weight.write_bytes(b"fixture: torch loading is mocked")
        (self.root / "split.json").write_text(json.dumps(self.split), encoding="utf-8")
        self.checkpoint = {"dataset_signature": "fixture", "epoch": 231,
                           "rotation_policy": self.policy, "train_args": {"seed": 42, "val_ratio": .25}}
        self.args = SimpleNamespace(train_dir=str(self.images), scale="all", seed=42, val_ratio=.25,
                                    output_dir=str(self.root / "cache"), locator_weight=str(self.weight),
                                    locator_split=None, angles=list(DEFAULT_ANGLES), cpu=True)

    def generate(self):
        locator = SyntheticLocator()
        with patch("face_box_cache.torch.load", return_value=self.checkpoint), \
                patch("face_box_cache.FaceLocator.from_checkpoint", return_value=locator):
            generate_cache(self.args)
        return locator

    def load(self, angles=DEFAULT_ANGLES):
        return load_face_cache(self.records, self.args.output_dir, self.training, self.validation,
                               "all", 42, .25, angles=angles)

    def training_args(self, angles=DEFAULT_ANGLES):
        with patch("sys.argv", ["train.py", "--train_dir", str(self.images), "--face_cache", self.args.output_dir,
                   "--scale", "all", "--angles", *map(str, angles), "--epochs", "100", "--cpu",
                   "--out_dir", str(self.root / "run"), "--face_shrink_probability", "0"]):
            return parse_args()

    def test_cache_rotates_from_source_and_train_views_match_live_preprocessing(self):
        before = {r.path: r.path.read_bytes() for r in self.records}
        locator = self.generate()
        expected = []
        for record in self.records:
            with Image.open(record.path) as opened:
                source = opened.convert("RGBA")
            for angle in DEFAULT_ANGLES:
                rotated = rotate_image(source, angle)
                expected.append((rotated.size, rotated.tobytes()))
        self.assertEqual(locator.inputs, expected)
        boxes, _, metadata = self.load()
        self.assertEqual(metadata["view_count"], 60)
        views = rotation_records(self.records, DEFAULT_ANGLES)
        transform = build_dual_view_transform(height=80, width=40, expression_size=30, face_crop=True)
        dataset = AtriDataset(views, face_boxes=boxes, paired_transform=transform, view_metadata=True)
        for index in (0, 10, 20, 30, 40, 50):
            record = views[index]
            actual, targets = dataset[index]
            with Image.open(record.path) as opened:
                image = rotate_image(opened, record.angle)
            reference = transform(image, boxes[record.view_key])
            for name in ("full", "expression"):
                torch.testing.assert_close(actual[name], reference[name])
            self.assertEqual(actual["_view"][1].item(), record.angle)
            self.assertEqual(targets["expression"], TASK_CODES["expression"].index("f1"))
        self.assertTrue(all(r.path.read_bytes() == before[r.path] for r in self.records))

    def test_upright_subset_is_explicit_and_other_angles_still_validated(self):
        self.generate()
        boxes, _, _ = self.load([0])
        self.assertEqual(len(boxes), 10)
        with self.assertRaisesRegex(ValueError, "explicit"):
            self.load(None)
        with self.assertRaisesRegex(ValueError, "missing requested"):
            self.load([15])
        path = Path(self.args.output_dir) / "face_boxes.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[-1]["x_left"] = "99999"  # Corrupt a nonzero angle and update the digest: geometry must still reject it.
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=ROTATED_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        manifest_path = Path(self.args.output_dir) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["csv_sha256"] = file_sha256(path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cached box"):
            self.load([0])

    def test_rotated_cache_rejects_changed_source_and_rotation_contract(self):
        self.generate()
        path = Path(self.args.output_dir) / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["pillow_version"] = "different"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Pillow"):
            self.load()
        manifest["pillow_version"] = PILLOW_VERSION
        path.write_text(json.dumps(manifest), encoding="utf-8")
        self.records[0].path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source image changed"):
            self.load()

    def test_sampling_preview_has_unique_source_split_and_complete_product_plan(self):
        self.generate()
        args = self.training_args()
        args.sampling_preview_only = True
        with patch("train.AtriNet") as model, patch("train.DataLoader") as loader:
            train(args)
            model.assert_not_called()
            loader.assert_not_called()
        out = Path(args.out_dir)
        preview = json.loads((out / "sampling_preview.json").read_text(encoding="utf-8"))
        self.assertEqual(len(preview["epochs"]), 100)
        self.assertEqual(preview["validation"]["samples"], 30)
        split = json.loads((out / "split.json").read_text(encoding="utf-8"))
        self.assertEqual(len(split["train"]), 5)
        self.assertEqual(len(set(split["train"])), 5)
        checkpoint = make_checkpoint(SimpleNamespace(state_dict=lambda: {}), args, 1, .5, [], data_signature="test")
        validate_resume_config(checkpoint, args, "test")
        changed = copy.deepcopy(args)
        changed.rotation_policy["weights"][0] = .5
        with self.assertRaisesRegex(ValueError, "rotation policy"):
            validate_resume_config(checkpoint, changed, "test")

    def test_augmentation_preview_covers_angles_and_both_inputs_without_model(self):
        self.generate()
        args = self.training_args()
        args.augmentation_preview_only = True
        args.augmentation_preview_count, args.augmentation_preview_repeats = 1, 1
        with patch("train.AtriNet") as model:
            train(args)
            model.assert_not_called()
        preview = json.loads((Path(args.out_dir) / "augmentation_preview.json").read_text(encoding="utf-8"))
        self.assertEqual(len(preview["by_angle_scale"]), 30)
        self.assertEqual(len(preview["samples"]), 30)
        self.assertEqual(len(list((Path(args.out_dir) / "augmentation_previews").glob("*_full.png"))), 30)

    def test_locator_rotation_split_mismatch_rejected_before_cache_output(self):
        self.checkpoint["rotation_policy"] = None
        with self.assertRaisesRegex(ValueError, "matching rotated locator"):
            self.generate()
        self.assertFalse(Path(self.args.output_dir).exists())

    def test_inference_rejects_mismatched_locator_before_model_construction(self):
        checkpoint = {"format_version": 4, "model_state": {}, "rotation_training": self.policy,
                      "face_cache_metadata": {"locator": {"sha256": "wrong", "threshold": .5}}}
        with patch("infer.load_torch_checkpoint", return_value=checkpoint), patch("infer.AtriNet") as model:
            with self.assertRaisesRegex(ValueError, "locator weights differ"):
                load_model("unused", torch.device("cpu"), self.weight)
            model.assert_not_called()

    def test_live_diagnostics_write_all_angle_scale_groups_and_true_input_previews(self):
        class Model(torch.nn.Module):
            def forward(self, full, expression):
                return {task: torch.zeros((full.shape[0], len(codes))) for task, codes in TASK_CODES.items()}
        model = Model()
        model.calibration = {}
        model.checkpoint_metadata = {"dataset_signature": "fixture", "rotation_training": self.policy}
        model.face_preview = SimpleNamespace(locator=SyntheticLocator())
        transform = build_dual_view_transform(height=80, width=40, expression_size=30, face_crop=True)
        model.preprocess_config = {"full": {"height": 80, "width": 40, "margin": .04},
                                  "expression": {"size": 30}, "background": [0, 0, 0],
                                  "normalization": {"mean": list(transform.mean), "std": list(transform.std)}}
        transforms = {"full": lambda image: TF.normalize(TF.to_tensor(transform.full_preview(image)), transform.mean, transform.std)}
        args = SimpleNamespace(angles=[0, 90], scales=list(SCALE_CODES), output_dir=str(self.root / "diagnostic"),
            split=str(self.root / "split.json"), locator_split=None, weight=str(self.weight), locator_weight=str(self.weight),
            image_dir=str(self.images), cpu=True, save_previews=True)
        with patch("diagnose_classifier_rotations.load_model", return_value=(model, transforms, TASK_CODES)), \
                patch("diagnose_classifier_rotations.torch.load", return_value=self.checkpoint):
            diagnose(args)
        summary = json.loads((Path(args.output_dir) / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(summary["conditions"]), 10)
        self.assertEqual(summary["metadata"]["failures"], 0)
        self.assertEqual(summary["metadata"]["views"], 10)
        self.assertEqual(len(list((Path(args.output_dir) / "previews").glob("*_full.png"))), 10)


class RotatedMetricsTests(unittest.TestCase):
    def row(self, scale="s", angle=0, predicted="fe", status="ok"):
        result = {"content_id": "fixture", "scale": scale, "angle": angle, "status": status,
                  "box_source": "locator", "mode": "pad", "added_padding_fraction": 0.0}
        for task, code in (("outfit", "d1"), ("pose", "p1"), ("expression", "f1")):
            pred = predicted if task == "expression" else code
            result.update({f"{task}_actual": code, f"{task}_predicted": pred,
                           f"{task}_correct": pred == code, f"{task}_accepted": True,
                           f"{task}_confidence": .99, f"{task}_raw_confidence": .98,
                           f"{task}_target_confidence": .01 if pred != code else .99, f"{task}_margin": .98})
        return result

    def test_known_ambiguity_is_separate_without_changing_strict_accuracy(self):
        rows = [self.row(), self.row(predicted="f2"), self.row(status="error"), self.row(scale="l")]
        groups = {g["scale"]: g for g in summarize_rotations(rows)}
        self.assertEqual(groups["s"]["tasks"]["expression"]["top1_accuracy"], 0)
        self.assertEqual(groups["s"]["expression_ambiguity"]["known_pair_errors"], 1)
        self.assertEqual(groups["s"]["expression_ambiguity"]["other_expression_errors"], 1)
        self.assertEqual(groups["s"]["failures"], 1)
        self.assertEqual(groups["l"]["expression_ambiguity"]["known_pair_errors"], 0)
        self.assertEqual(groups["s"]["tasks"]["expression"]["per_class_recall"]["f1"], 0)

    def test_consistency_excludes_failed_groups_and_exposes_consistent_errors(self):
        rows = [self.row(), self.row(scale="l"), self.row(angle=90, status="error"), self.row(scale="l", angle=90)]
        metrics = consistency(rows)["scales_at_angle"]["expression"]
        self.assertEqual(metrics["groups"], 2)
        self.assertEqual(metrics["consistent_groups"], 1)
        self.assertEqual(metrics["all_correct_groups"], 0)

    def test_weighted_calibration_preserves_raw_support_and_validates_weights(self):
        values = {"expression": {"logits": torch.tensor([[3., 0.], [3., 0.]]),
                                 "targets": torch.tensor([0, 1]), "weights": torch.tensor([9., 1.])}}
        with patch("calibration.fit_temperature", return_value=1.):
            calibrated = build_calibration(values)["tasks"]["expression"]
        self.assertAlmostEqual(calibrated["validation_accuracy"], .9, places=6)
        self.assertEqual(calibrated["support"], 2)
        self.assertTrue(calibrated["weighted"])
        for weights in (torch.tensor([1., 0.]), torch.tensor([1.]), torch.tensor([float("nan"), 1.])):
            with self.assertRaises(ValueError):
                sample_weights(values["expression"]["targets"], weights)

    def test_validation_grouping_and_logit_collection_ignore_view_metadata_as_inputs(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, full, expression):
                return {task: torch.zeros((full.shape[0], len(codes))) for task, codes in TASK_CODES.items()}
        views = {"full": torch.zeros(2, 3, 8, 8), "expression": torch.zeros(2, 3, 8, 8),
                 "_view": torch.tensor([[0, 0], [3, 90]]), "_sample_weight": torch.tensor([.45, .05])}
        targets = {task: torch.zeros(2, dtype=torch.long) for task in TASK_CODES}
        loader = [(views, targets)]
        collected = collect_logits(ConstantModel(), loader, torch.device("cpu"), TASK_CODES)
        torch.testing.assert_close(collected["expression"]["weights"], torch.tensor([.45, .05]))
        metrics = run_epoch(ConstantModel(), loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"), False)
        self.assertEqual(set(metrics["by_angle_scale"]), {"0/s", "90/l"})
        self.assertEqual(metrics["samples"], 2)


if __name__ == "__main__":
    unittest.main()
