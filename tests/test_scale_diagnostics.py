import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from dataset import parse_filename
from diagnose_scales import (
    diagnose, face_view, map_reference_box, match_records, paired_comparisons,
    read_split, square_iou, summarize,
)
from face_regions import SquareBox
from labels import TASK_CODES


class ScaleDiagnosticTests(unittest.TestCase):
    def record(self, scale, outfit="d1"):
        return parse_filename(f"ATRI_tatr01_{scale}_{outfit}_p1_f1.png")

    def test_matching_uses_same_validation_content_in_every_scale(self):
        records = {scale: [self.record(scale, outfit) for outfit in ("d1", "d2")] for scale in ("s", "l")}
        key = self.record("l").content_key
        matched = match_records(records, ("s", "l"), {key})
        self.assertEqual([record.scale for record in matched], ["s", "l"])
        self.assertEqual({record.content_key for record in matched}, {key})
        with self.assertRaisesRegex(ValueError, "locator"):
            match_records(records, ("s", "l"), {key}, {key})
        with self.assertRaisesRegex(ValueError, "missing"):
            match_records({"s": records["s"][1:], "l": records["l"]}, ("s", "l"), {key})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            match_records({"s": [records["s"][0]] * 2}, ("s",))

    def test_split_rejects_same_artwork_across_resolutions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps({"train": ["nested\\ATRI_tatr01_l_d1_p1_f1.png"],
                                        "validation": ["ATRI_tatr01_s_d1_p1_f1.png"]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same artwork"):
                read_split(path)
            path.write_text(json.dumps({"train": ["nested\\ATRI_tatr01_l_d1_p1_f1.png"],
                                        "validation": ["ATRI_tatr01_l_d2_p1_f1.png"]}), encoding="utf-8")
            split = read_split(path)
            self.assertEqual(split["validation"], {self.record("l", "d2").content_key})

    def test_proportional_bottom_origin_reference_mapping(self):
        box = SquareBox(10, 100, 50)
        self.assertEqual(map_reference_box(box, (100, 200), (50, 100)), SquareBox(5, 50, 25))
        self.assertEqual(map_reference_box(box, (100, 200), (100, 200)), box)
        self.assertEqual(square_iou(box, box), 1.0)
        self.assertEqual(square_iou(box, SquareBox(60, 100, 40)), 0.0)
        with self.assertRaisesRegex(ValueError, "aspect ratios"):
            map_reference_box(box, (100, 200), (60, 100))

    def test_upscaling_changes_occupancy_without_changing_source_or_box(self):
        image = Image.new("RGBA", (10, 20), (255, 0, 0, 255))
        before = image.tobytes()
        box = SquareBox(2, 8, 4)
        pad, pad_geometry = face_view(image, box, 12, (0, 0, 0), "pad")
        upscale, upscale_geometry = face_view(image, box, 12, (0, 0, 0), "upscale")
        self.assertEqual(sum(pixel == (255, 0, 0) for pixel in pad.getdata()), 16)
        self.assertEqual(sum(pixel == (255, 0, 0) for pixel in upscale.getdata()), 144)
        self.assertAlmostEqual(pad_geometry["added_padding_fraction"], 8 / 9)
        self.assertEqual(upscale_geometry["added_padding_fraction"], 0)
        self.assertEqual(upscale_geometry["resize_factor"], 3)
        self.assertEqual(image.tobytes(), before)

    def test_large_crop_is_identical_in_both_modes(self):
        image = Image.new("RGBA", (20, 20), (255, 0, 0, 120))
        box = SquareBox(0, 0, 20)
        pad, _ = face_view(image, box, 12, (0, 0, 0), "pad")
        upscale, _ = face_view(image, box, 12, (0, 0, 0), "upscale")
        self.assertEqual(pad.tobytes(), upscale.tobytes())

    def test_failures_and_confident_errors_remain_visible_in_summaries(self):
        rows = []
        for mode, correct, accepted, confidence in (("pad", True, False, .9), ("upscale", False, True, .98)):
            row = dict(content_id="A", filename="A.png", scale="s", box_source="locator", mode=mode,
                       status="ok", added_padding_fraction=0.8)
            for task in TASK_CODES:
                row.update({f"{task}_correct": correct, f"{task}_accepted": accepted,
                            f"{task}_confidence": confidence, f"{task}_target_confidence": .9 if correct else .01,
                            f"{task}_predicted": "f1" if correct else "f2"})
            rows.append(row)
            rows.append(dict(content_id="B", filename="B.png", scale="s", box_source="locator", mode=mode, status="face_not_found"))
        conditions = {item["mode"]: item for item in summarize(rows)}
        pad = conditions["pad"]["tasks"]["expression"]
        self.assertEqual(conditions["pad"]["samples"], 2)
        self.assertEqual(pad["top1_accuracy"], .5)
        self.assertEqual(pad["correct_but_rejected"], 1)
        self.assertEqual(conditions["upscale"]["tasks"]["expression"]["accepted_but_wrong"], 1)
        pairs = {item["content_id"]: item for item in paired_comparisons(rows)}
        self.assertEqual(pairs["A"]["correctness_change"], -1)
        self.assertGreater(pairs["A"]["confidence_change"], 0)
        self.assertIsNone(pairs["B"]["confidence_change"])

    def test_output_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(scales=["s", "l"], locator_threshold=None, output_dir=directory)
            with self.assertRaises(FileExistsError):
                diagnose(args)


if __name__ == "__main__":
    unittest.main()
