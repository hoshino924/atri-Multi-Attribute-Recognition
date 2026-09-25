"""Content isolation and training-budget checks; run locally with unittest."""

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dataset import ContentBalancedSampler, SCALE_CODES, parse_filename, scan_records, stratified_split
from diagnose_scales import read_split
from face_box_cache import split_contents
from train import make_checkpoint, parse_args, validate_args, validate_resume_config


class BalancedSamplingTests(unittest.TestCase):
    @staticmethod
    def records(count=189, scales=SCALE_CODES):
        return [parse_filename(f"ART{index:03d}_tatr01_{scale}_d1_p1_f1.png")
                for index in range(count) for scale in scales]

    def test_grouped_split_preserves_historical_single_scale_partition(self):
        records = [parse_filename(f"ATRI_tatr01_{scale}_d{outfit}_p{pose}_f{expression}.png")
                   for outfit in range(1, 5) for pose in range(1, 4)
                   for expression in range(1, 3) for scale in SCALE_CODES]
        upright = [record for record in records if record.scale == "w"]
        # Independent reference: the original splitter shuffled filenames in
        # each pose/expression stratum before selecting the validation quarter.
        rng = random.Random(42)
        strata = defaultdict(list)
        for record in upright:
            strata[(record.pose, record.expression)].append(record)
        expected_validation = set()
        for key in sorted(strata):
            group = sorted(strata[key], key=lambda record: record.path.name)
            rng.shuffle(group)
            expected_validation.add(group[0].content_key)
        for selected in (records, upright, list(reversed(records))):
            training, validation = stratified_split(selected, seed=42)
            self.assertEqual({r.content_key for r in validation}, expected_validation)
            self.assertFalse({r.content_key for r in training} & expected_validation)
            self.assertEqual(len(training) + len(validation), len(selected))
        # Multiple resolutions of one artwork do not provide a valid split.
        with self.assertRaisesRegex(ValueError, "two contents"):
            stratified_split(self.records(count=1))

    def test_scanning_all_requires_one_of_every_scale_for_every_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for record in self.records(count=2):
                (root / record.path).touch()  # Parsing must not decode an image.
            self.assertEqual(len(scan_records(root, "all")), 10)
            missing = root / self.records(count=1)[0].path
            missing.unlink()
            self.assertEqual(len(scan_records(root, "w")), 2)
            with self.assertRaisesRegex(ValueError, "missing scales"):
                scan_records(root, "all")
            missing.touch()
            missing.with_suffix(".jpg").touch()
            with self.assertRaisesRegex(ValueError, "duplicate"):
                scan_records(root, "all")

    def test_every_content_is_seen_once_per_epoch_and_every_scale_once_per_cycle(self):
        records = self.records()
        sampler = ContentBalancedSampler(records)
        self.assertEqual(len(sampler), 189)
        self.assertEqual(sampler.metadata(16)["batches_per_epoch"], 12)
        for start in (0, 5, 85):
            coverage = defaultdict(Counter)
            total = Counter()
            for epoch in range(start, start + 5):
                sampler.set_epoch(epoch)
                selected = [records[index] for index in sampler]
                self.assertEqual(len({r.content_key for r in selected}), 189)
                counts = Counter(record.scale for record in selected)
                self.assertEqual(sorted(counts.values()), [37, 38, 38, 38, 38])
                self.assertEqual(dict(counts), sampler.epoch_summary(epoch)["scale_counts"])
                for record in selected:
                    coverage[record.content_key][record.scale] += 1
                total.update(counts)
            self.assertEqual(total, Counter({scale: 189 for scale in SCALE_CODES}))
            self.assertTrue(all(counts == Counter(SCALE_CODES) for counts in coverage.values()))

    def test_resume_and_record_order_do_not_change_epoch_selection(self):
        records = self.records(count=17)
        sampler = ContentBalancedSampler(records, seed=7)
        reversed_records = list(reversed(records))
        resumed = ContentBalancedSampler(reversed_records, seed=7)
        for epoch in (0, 1, 4, 5, 42):
            sampler.set_epoch(epoch)
            expected = [records[index].path.name for index in sampler]
            random.random()  # Global augmentation RNG must not affect sampling.
            resumed.set_epoch(epoch)
            self.assertEqual(expected, [reversed_records[index].path.name for index in resumed])
            self.assertEqual(expected, sampler.epoch_summary(epoch, True)["filenames"])
        with self.assertRaises(ValueError):
            sampler.set_epoch(-1)

    def test_single_scale_baseline_has_same_content_order_and_training_budget(self):
        records = self.records(count=17)
        single_records = [record for record in records if record.scale == "w"]
        baseline = ContentBalancedSampler(single_records, ("w",), seed=42)
        multi = ContentBalancedSampler(records, seed=42)
        self.assertEqual(len(baseline), len(multi))
        for epoch in range(10):
            first = [single_records[index].content_key for index in baseline.indices_for_epoch(epoch)]
            second = [records[index].content_key for index in multi.indices_for_epoch(epoch)]
            self.assertEqual(first, second)
        for selected in (records[:-1], records + records[:1]):
            with self.assertRaises(ValueError):
                ContentBalancedSampler(selected)

    def test_small_content_count_is_balanced_over_a_complete_cycle(self):
        sampler = ContentBalancedSampler(self.records(count=2))
        total = Counter()
        for epoch in range(5):
            total.update(sampler.epoch_summary(epoch)["scale_counts"])
        self.assertEqual(total, Counter({scale: 2 for scale in SCALE_CODES}))

    def test_split_readers_allow_scale_variants_but_reject_duplicates_and_leakage(self):
        records = self.records(count=2)
        manifest = {"train": [r.path.name for r in records[:5]],
                    "validation": [r.path.name for r in records[5:]]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(read_split(path), split_contents(manifest))
            self.assertEqual(len(read_split(path)["validation"]), 1)
            bad = copy.deepcopy(manifest)
            bad["train"].append(bad["train"][0].replace(".png", ".jpg"))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                split_contents(bad)
            bad = copy.deepcopy(manifest)
            bad["validation"].append(bad["train"][0])
            with self.assertRaisesRegex(ValueError, "same artwork"):
                split_contents(bad)

    def test_new_defaults_and_resume_reject_changed_sampling_or_update_budget(self):
        with patch("sys.argv", ["train.py", "--scale", "all", "--face_cache", "cache"]):
            args = parse_args()
        validate_args(args)
        self.assertEqual((args.width, args.height, args.expression_size, args.batch), (512, 768, 300, 16))
        args.sampling_metadata = ContentBalancedSampler(self.records()).metadata(args.batch)
        checkpoint = make_checkpoint(SimpleNamespace(state_dict=lambda: {}), args, 1, 1.0, [], data_signature="fixture")
        validate_resume_config(checkpoint, args, "fixture")
        for key, value in (("batch", 8), ("epochs", 100), ("warmup_epochs", 2)):
            changed = copy.deepcopy(args)
            setattr(changed, key, value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "training setting"):
                validate_resume_config(checkpoint, changed, "fixture")
        changed = copy.deepcopy(args)
        changed.sampling_metadata["samples_per_epoch"] *= 5
        with self.assertRaisesRegex(ValueError, "sampling policy"):
            validate_resume_config(checkpoint, changed, "fixture")
        old = copy.deepcopy(checkpoint)
        old["train_args"].pop("sampling_metadata")
        with self.assertRaisesRegex(ValueError, "sampling policy"):
            validate_resume_config(old, args, "fixture")


if __name__ == "__main__":
    unittest.main()
