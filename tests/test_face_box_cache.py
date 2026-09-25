import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from dataset import AtriDataset, SCALE_CODES, build_dual_view_transform, scan_records, stratified_split
from face_box_cache import file_sha256, generate_cache, load_face_cache, split_contents
from face_locator import FaceNotFoundError, LocatedFacePreview, LocatedFaceTransform
from face_regions import SquareBox
from infer import load_model
from model import AtriNet
from train import make_checkpoint, parse_args, resolve_face_min_scale, train, validate_args, validate_resume_config


class RecordingLocator:
    size = 64
    threshold = .5

    def __init__(self, fail_first=False):
        self.calls = 0
        self.fail_first = fail_first

    def locate(self, image):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise FaceNotFoundError("fixture: no face")
        return SquareBox(8, 40, 24), .99


class FaceBoxCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        for index in range(1, 5):
            image = Image.new("RGBA", (40, 80))
            image.paste((index * 40, 80, 120, 255), (8, 12, 32, 72))
            image.save(self.images / f"ATRI_tatr01_w_d{index}_p1_f1.png")
        self.records = scan_records(self.images, "w")
        self.train_records, self.val_records = stratified_split(self.records, .25, 42)
        self.output = self.root / "cache"
        self.weight = self.root / "locator.pth"
        self.weight.write_bytes(b"locator fixture (loading is mocked)")
        self.split_path = self.root / "split.json"
        self.split = {
            "train": [record.path.name.replace("_w_", "_l_") for record in self.train_records],
            "validation": [record.path.name.replace("_w_", "_l_") for record in self.val_records],
            "dataset_signature": "locator-data", "seed": 42, "val_ratio": .25,
        }
        self.split_path.write_text(json.dumps(self.split), encoding="utf-8")
        self.checkpoint = {"dataset_signature": "locator-data", "epoch": 4,
                           "train_args": {"seed": 42, "val_ratio": .25}}
        self.args = SimpleNamespace(train_dir=str(self.images), output_dir=str(self.output),
                                    locator_weight=str(self.weight), locator_split=None,
                                    scale="w", seed=42, val_ratio=.25, cpu=True)

    def generate(self, locator=None):
        locator = locator or RecordingLocator()
        with (
            patch("face_box_cache.torch.load", return_value=self.checkpoint),
            patch("face_box_cache.FaceLocator.from_checkpoint", return_value=locator),
            patch("face_box_cache.torch.cuda.is_available", return_value=False),
        ):
            generate_cache(self.args)
        return locator

    def load(self, **changes):
        settings = dict(scale="w", seed=42, val_ratio=.25)
        settings.update(changes)
        return load_face_cache(self.records, self.output, self.train_records, self.val_records, **settings)

    def train_args(self, manual=False):
        source = ["--face_annotations", "manual.csv"] if manual else ["--face_cache", str(self.output)]
        with patch("sys.argv", ["train.py", *source, "--scale", "w", "--expression_size", "300", "--cpu",
                                "--train_dir", str(self.images), "--out_dir", str(self.root / "run")]):
            args = parse_args()
        validate_args(args)
        resolve_face_min_scale(args)
        return args

    def test_cache_round_trip_matches_live_locator_without_modifying_sources(self):
        before = {record.path.name: record.path.read_bytes() for record in self.records}
        manual = self.root / "manual.csv"
        manual.write_text("manual annotations stay unchanged", encoding="utf-8")
        locator = self.generate()
        self.assertEqual(locator.calls, 4)
        boxes, signature, metadata = self.load()
        self.assertEqual(len(boxes), 4)
        self.assertEqual(metadata["locator"]["sha256"], file_sha256(self.weight))
        self.assertEqual(metadata["signature"], signature)
        self.assertEqual(manual.read_text(encoding="utf-8"), "manual annotations stay unchanged")
        for record in self.records:
            self.assertEqual(record.path.read_bytes(), before[record.path.name])
        paired = build_dual_view_transform(height=64, width=40, expression_size=300, face_crop=True)
        cached = AtriDataset(self.records, paired_transform=paired, face_boxes=boxes)[0][0]["expression"]
        preview = LocatedFacePreview(300, (0, 0, 0), locator)
        live = LocatedFaceTransform(preview, paired.mean, paired.std)
        with Image.open(self.records[0].path) as source:
            image = source.convert("RGBA")
        torch.testing.assert_close(cached, live(image))
        self.assertEqual(tuple(cached.shape), (3, 300, 300))

    def test_cache_rejects_source_changes_even_with_unchanged_file_size(self):
        self.generate()
        path = self.records[0].path
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "source image changed"):
            self.load()

    def test_cache_rejects_csv_edits_and_different_training_split(self):
        self.generate()
        for setting in ({"scale": "l"}, {"seed": 7}, {"val_ratio": .5}):
            with self.subTest(setting=setting), self.assertRaisesRegex(ValueError, "classifier split differs"):
                self.load(**setting)
        path = self.output / "face_boxes.csv"
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "CSV changed"):
            self.load()

    def test_failed_localization_produces_report_but_no_trainable_cache(self):
        with self.assertRaisesRegex(ValueError, "1 images failed"):
            self.generate(RecordingLocator(fail_first=True))
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        failures = json.loads((self.output / "failures.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(len(manifest["images"]), 3)
        self.assertEqual(failures[0]["filename"], self.records[0].path.name)
        self.assertFalse((self.output / "face_boxes.csv").exists())
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.load()

    def test_unreadable_image_is_reported_instead_of_silently_excluded(self):
        self.records[0].path.write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "1 images failed"):
            self.generate()
        failures = json.loads((self.output / "failures.json").read_text(encoding="utf-8"))
        self.assertEqual(failures[0]["filename"], self.records[0].path.name)
        self.assertFalse((self.output / "face_boxes.csv").exists())

    def test_validation_overlap_is_detected_across_resolutions_before_loading_locator(self):
        self.split["train"], self.split["validation"] = self.split["validation"], self.split["train"]
        self.split_path.write_text(json.dumps(self.split), encoding="utf-8")
        with patch("face_box_cache.FaceLocator.from_checkpoint") as load_locator:
            with self.assertRaisesRegex(ValueError, "used to train the locator"):
                generate_cache(self.args)
            load_locator.assert_not_called()
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, "same artwork"):
            split_contents({"train": [self.records[0].path.name],
                            "validation": [self.records[0].path.name.replace("_w_", "_l_")]})

    def test_checkpoint_and_locator_split_must_match(self):
        for updates in ({"dataset_signature": "different"}, {"train_args": {"seed": 7, "val_ratio": .25}}):
            with self.subTest(updates=updates):
                checkpoint = {**self.checkpoint, **updates}
                with patch("face_box_cache.torch.load", return_value=checkpoint):
                    with self.assertRaisesRegex(ValueError, "locator checkpoint and split manifest"):
                        generate_cache(self.args)
                self.assertFalse(self.output.exists())

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            generate_cache(self.args)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_all_scale_cache_and_training_preview_use_grouped_partitions(self):
        for record in self.records:
            for scale in SCALE_CODES:
                if scale != "w":
                    (self.images / record.path.name.replace("_w_", f"_{scale}_")).write_bytes(record.path.read_bytes())
        self.args.scale = "all"
        self.assertEqual(self.generate().calls, 20)
        records = scan_records(self.images, "all")
        training, validation = stratified_split(records)
        boxes, _, _ = load_face_cache(records, self.output, training, validation, "all", 42, .25)
        self.assertEqual((len(boxes), len(training), len(validation)), (20, 15, 5))
        args = self.train_args()
        args.scale = "all"
        args.sampling_preview_only = True
        with patch("train.AtriNet") as model, patch("train.DataLoader") as loader:
            train(args)
            model.assert_not_called()
            loader.assert_not_called()
        preview = json.loads((Path(args.out_dir) / "sampling_preview.json").read_text(encoding="utf-8"))
        self.assertEqual(preview["sampling"]["samples_per_epoch"], 3)
        self.assertEqual(preview["validation"]["samples"], 5)
        self.assertEqual(sum(epoch["samples"] for epoch in preview["epochs"]), 15)
        manifest = json.loads((Path(args.out_dir) / "split.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["content_counts"], {"train": 3, "validation": 1})
        self.assertEqual(split_contents(manifest), split_contents(self.split))

    def test_locator_partition_must_cover_classifier_contents_exactly(self):
        self.split["train"] = self.split["train"][:-1]
        self.split_path.write_text(json.dumps(self.split), encoding="utf-8")
        with patch("face_box_cache.torch.load") as load:
            with self.assertRaisesRegex(ValueError, "content partitions differ"):
                generate_cache(self.args)
            load.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_augmentation_preview_uses_cache_without_allocating_model_or_loader(self):
        self.generate()
        args = self.train_args()
        args.augmentation_preview_only = True
        args.augmentation_preview_count = 1
        args.augmentation_preview_repeats = 2
        with patch("train.AtriNet") as model, patch("train.DataLoader") as loader:
            train(args)
            model.assert_not_called()
            loader.assert_not_called()
        preview = json.loads((Path(args.out_dir) / "augmentation_preview.json").read_text(encoding="utf-8"))
        self.assertEqual(preview["statistics"]["samples"], 2)
        self.assertEqual(preview["policy"]["position_mode"], "mixed")
        training, validation = stratified_split(self.records)
        self.assertTrue({row["filename"] for row in preview["samples"]} <= {row.path.name for row in training})
        self.assertFalse({row["filename"] for row in preview["samples"]} & {row.path.name for row in validation})

    def test_invalid_cache_stops_training_before_output_metadata_or_model_allocation(self):
        self.generate()
        with (self.output / "face_boxes.csv").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with patch("train.prepare_output_directory") as prepare, patch("train.AtriNet") as model:
            with self.assertRaisesRegex(ValueError, "CSV changed"):
                train(self.train_args())
            prepare.assert_not_called()
            model.assert_not_called()

    def test_inference_only_checkpoint_cannot_silently_reset_resumed_optimizer(self):
        self.generate()
        args = self.train_args()
        args.resume = "best.pth"
        with (patch("train.load_torch_checkpoint", return_value={}),
              patch("train.validate_resume_config"), patch("train.prepare_output_directory") as prepare):
            with self.assertRaisesRegex(ValueError, "complete training state"):
                train(args)
            prepare.assert_not_called()

    def test_defaults_and_conflicting_box_inputs(self):
        for arguments, expected in (([], ("w", 512)), (["--face_annotations", "manual.csv"], ("l", 626)),
                                    (["--face_cache", "cache"], ("w", 300))):
            with patch("sys.argv", ["train.py", *arguments]):
                args = parse_args()
            validate_args(args)
            self.assertEqual((args.scale, args.expression_size), expected)
        args.face_annotations = "manual.csv"
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_args(args)

    def test_checkpoint_inference_contract_provenance_and_resume_source_protection(self):
        self.generate()
        _, signature, metadata = self.load()
        args = self.train_args()
        args.face_cache_metadata = metadata
        model = AtriNet(pretrained=False)
        checkpoint = make_checkpoint(model, args, 1, 1., [], data_signature=signature)
        self.assertEqual(checkpoint["format_version"], 4)
        self.assertEqual(checkpoint["face_box_source"], "locator_cache")
        self.assertEqual(checkpoint["face_cache_metadata"]["locator"]["sha256"], file_sha256(self.weight))
        self.assertEqual(checkpoint["preprocess"]["expression"]["size"], 300)
        self.assertFalse(checkpoint["preprocess"]["expression"]["allow_upscale"])
        path = self.root / "classifier.pth"
        torch.save(checkpoint, path)
        restored, _, _ = load_model(path, torch.device("cpu"))
        self.assertEqual(restored.face_preview.canvas.size, 300)
        validate_resume_config(checkpoint, args, signature)
        with self.assertRaisesRegex(ValueError, "dataset changed"):
            validate_resume_config(checkpoint, args, "different-cache")
        changed = copy.deepcopy(args)
        changed.seed = 1
        with self.assertRaisesRegex(ValueError, "split setting"):
            validate_resume_config(checkpoint, changed, signature)
        manual_args = self.train_args(manual=True)
        with self.assertRaisesRegex(ValueError, "face box source"):
            validate_resume_config(checkpoint, manual_args, signature)
        # Historical format-4 checkpoints did not record a separate box source.
        manual_checkpoint = make_checkpoint(model, manual_args, 1, 1., [], data_signature=signature)
        manual_checkpoint.pop("face_box_source")
        validate_resume_config(manual_checkpoint, manual_args, signature)
        with self.assertRaisesRegex(ValueError, "face box source"):
            validate_resume_config(manual_checkpoint, args, signature)


if __name__ == "__main__":
    unittest.main()
