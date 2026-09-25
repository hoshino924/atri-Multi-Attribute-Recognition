"""Public bundles preserve tensors and model pairing without training records."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from export_release_weights import export, sha256


class ReleaseWeightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.classifier_dir = self.root / "private_classifier"
        self.locator_dir = self.root / "private_locator"
        self.classifier_dir.mkdir()
        self.locator_dir.mkdir()
        self.locator_path = self.locator_dir / "best.pth"
        self.classifier_path = self.classifier_dir / "best.pth"
        policy = {"angles": [0, 45], "weights": [0.5, 0.5]}
        self.locator = {
            "locator_version": 1, "input_size": 256, "threshold": 0.5, "epoch": 4,
            "model_state": {"weight": torch.tensor([1.25, -2.0])},
            "dataset_signature": "locator-data", "rotation_policy": policy,
            "train_args": {"seed": 42, "val_ratio": 0.25, "out_dir": "C:/PRIVATE/locator"},
            "history": ["PRIVATE training log"], "optimizer_state": {"PRIVATE": 1},
        }
        locator_split = {"dataset_signature": "locator-data", "seed": 42,
                         "val_ratio": 0.25, "rotation_policy": policy}
        (self.locator_dir / "split.json").write_text(json.dumps(locator_split), encoding="utf-8")
        torch.save(self.locator, self.locator_path)
        self.classifier = {
            "format_version": 4, "epoch": 7, "dataset_signature": "classifier-data",
            "model_state": {"weight": torch.tensor([[3.0, -0.5]])},
            "model_config": {"expression_head": "fusion", "dropout": 0.2},
            "label_codes": {"expression": ["f1", "fe"]}, "rotation_training": policy,
            "preprocess": {"full": {"width": 512, "height": 768, "margin": 0.04},
                "expression": {"mode": "annotated_face", "size": 300, "margin": 0.0,
                               "allow_upscale": False, "coordinate_system": "bottom_left_pixels"},
                "background": [0, 0, 0], "normalization": {"mean": [0.5] * 3, "std": [0.2] * 3}},
            "face_cache_metadata": {"version": 2, "directory": "C:/PRIVATE/cache",
                "classifier_split": {"scale": "all", "seed": 42, "val_ratio": 0.25},
                "locator": {"sha256": sha256(self.locator_path), "weight": "C:/PRIVATE/locator",
                    "threshold": 0.5, "split_sha256": sha256(self.locator_dir / "split.json")}},
            "calibration": {"method": "temperature_scaling", "tasks": {
                "expression": {"temperature": 0.25, "suggested_threshold": 0.95,
                               "validation_accuracy": 1.0}},
                "distribution": {"angles": [0, 45], "locator_sha256": sha256(self.locator_path)}},
            "train_args": {"train_dir": "C:/PRIVATE/images"}, "history": ["PRIVATE log"],
        }
        torch.save(self.classifier, self.classifier_path)
        (self.classifier_dir / "split.json").write_text(
            json.dumps({"dataset_signature": "classifier-data"}), encoding="utf-8")
        self.args = SimpleNamespace(classifier=self.classifier_path, locator=self.locator_path,
                                    output_dir=self.root / "public_models")

    @staticmethod
    def fake_loader(classifier_path, device, locator_path):
        checkpoint = torch.load(classifier_path, map_location="cpu", weights_only=True)
        if checkpoint["face_cache_metadata"]["locator"]["sha256"] != sha256(locator_path):
            raise ValueError("exported locator identity was not rebound")
        return SimpleNamespace(calibration=checkpoint["calibration"]), {}, {}

    def test_bundle_preserves_tensors_and_pair_without_private_records(self):
        source_hashes = (sha256(self.classifier_path), sha256(self.locator_path))
        with patch("infer.load_model", side_effect=self.fake_loader):
            result = export(self.args)
        classifier = torch.load(self.args.output_dir / "classifier/atri_net_best.pth", weights_only=True)
        locator_path = self.args.output_dir / "locator/face_locator_best.pth"
        locator = torch.load(locator_path, weights_only=True)
        self.assertTrue(torch.equal(classifier["model_state"]["weight"], self.classifier["model_state"]["weight"]))
        self.assertTrue(torch.equal(locator["model_state"]["weight"], self.locator["model_state"]["weight"]))
        for checkpoint in (classifier, locator):
            self.assertNotIn("history", checkpoint)
            self.assertNotIn("optimizer_state", checkpoint)
            metadata = {key: value for key, value in checkpoint.items() if key != "model_state"}
            self.assertNotIn("PRIVATE", json.dumps(metadata))
        self.assertNotIn("train_args", classifier)
        self.assertEqual(locator["train_args"], {"seed": 42, "val_ratio": 0.25})
        self.assertEqual(classifier["calibration"]["tasks"]["expression"],
                         {"temperature": 0.25, "suggested_threshold": 0.95})
        self.assertEqual(classifier["calibration"]["distribution"]["locator_sha256"], sha256(locator_path))
        self.assertEqual(sha256(self.args.output_dir / "locator/split.json"), sha256(self.locator_dir / "split.json"))
        self.assertEqual(source_hashes, (sha256(self.classifier_path), sha256(self.locator_path)))
        for relative, digest in result["files"].items():
            self.assertEqual(sha256(self.args.output_dir / relative), digest)

    def test_wrong_source_locator_is_rejected(self):
        self.classifier["face_cache_metadata"]["locator"]["sha256"] = "wrong"
        torch.save(self.classifier, self.classifier_path)
        with self.assertRaisesRegex(ValueError, "source locator does not match"):
            export(self.args)
        self.assertFalse(self.args.output_dir.exists())

    def test_existing_directory_is_preserved(self):
        self.args.output_dir.mkdir()
        marker = self.args.output_dir / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            export(self.args)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_failed_loading_does_not_commit_partial_bundle(self):
        with patch("infer.load_model", side_effect=ValueError("loading failed")):
            with self.assertRaisesRegex(ValueError, "loading failed"):
                export(self.args)
        self.assertFalse(self.args.output_dir.exists())
        self.assertTrue(self.classifier_path.is_file())
        self.assertTrue(self.locator_path.is_file())


if __name__ == "__main__":
    unittest.main()
