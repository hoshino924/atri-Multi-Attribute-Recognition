from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from dataset import parse_filename
from diagnose_offsets import diagnose, image_stability, offset_grid, offset_name, summarize_offsets
from face_regions import SquareBox
from labels import TASK_CODES


def prediction_row(filename, dx, status="ok", correct=True, accepted=True):
    row = dict(filename=filename, content_id=filename, scale="s", box_source="locator",
               mode=offset_name(dx, 0), requested_dx=dx, requested_dy=0, status=status,
               added_padding_fraction=.5)
    if status == "ok":
        for task in TASK_CODES:
            row.update({f"{task}_correct": correct, f"{task}_accepted": accepted,
                        f"{task}_confidence": .99, f"{task}_target_confidence": .99 if correct else .01,
                        f"{task}_predicted": "A" if correct else "B"})
    return row


class FakeModel:
    face_preview = True
    checkpoint_metadata = {}
    calibration = {}
    preprocess_config = {"expression": {"size": 32}, "background": [0, 0, 0],
                         "normalization": {"mean": [0, 0, 0], "std": [1, 1, 1]}}

    def __call__(self, full, face):
        results = {}
        for task, codes in TASK_CODES.items():
            logits = torch.zeros(len(face), len(codes))
            logits[:, 0] = 10
            results[task] = logits
        return results


class OffsetDiagnosticTests(unittest.TestCase):
    def test_grid_requires_a_unique_zero_baseline(self):
        self.assertEqual(len(offset_grid([-25, -10, 0, 10, 25])), 25)
        for values in ([], [10], [0, 0], [0, float("nan")]):
            with self.assertRaises(ValueError):
                offset_grid(values)

    def test_geometry_failures_and_prediction_failures_are_distinguished(self):
        rows = [prediction_row("A", 0), prediction_row("A", 25, correct=False),
                prediction_row("B", 0), prediction_row("B", 25, status="out_of_bounds"),
                prediction_row("C", 0, status="face_not_found"),
                prediction_row("C", 25, status="face_not_found")]
        groups = {row["requested_dx"]: row for row in summarize_offsets(rows)}
        changed = groups[25]
        self.assertEqual((changed["samples"], changed["predicted"], changed["out_of_bounds"]), (3, 1, 1))
        self.assertEqual(changed["valid_pairs_to_zero"], 1)
        self.assertEqual(changed["tasks"]["expression"]["accepted_but_wrong"], 1)
        self.assertEqual(changed["tasks"]["expression"]["correct_to_wrong_from_zero"], 1)
        self.assertEqual(groups[0]["tasks"]["expression"]["top1_accuracy"], 2 / 3)
        images = {row["filename"]: row for row in image_stability(rows)}
        self.assertFalse(images["B"]["expression_all_requests_correct"])
        self.assertTrue(images["B"]["expression_all_feasible_correct"])
        self.assertFalse(images["C"]["expression_all_feasible_correct"])

    def test_full_diagnostic_writes_all_conditions_and_never_changes_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ATRI_tatr01_s_d1_p1_f1.png"
            image = Image.new("RGBA", (80, 100), (100, 60, 20, 255))
            image.save(path)
            original = path.read_bytes()
            record = parse_filename(path)
            args = SimpleNamespace(offsets=[-25, 0, 25], scales=["s"], batch=2, preview_images=1,
                                   locator_threshold=None, output_dir=str(root / "output"),
                                   split=None, locator_split=None, weight="fake.pth", locator_weight="locator.pth",
                                   image_dir=str(root), cpu=True)
            box = SquareBox(0, 0, 32)
            transforms = {"full": lambda source: torch.zeros(3, 32, 32)}
            with (patch("diagnose_offsets.read_split", return_value={"train": set(), "validation": {record.content_key}}),
                  patch("diagnose_offsets.load_model", return_value=(FakeModel(), transforms, TASK_CODES)),
                  patch("diagnose_offsets.FaceLocator.from_checkpoint", return_value=SimpleNamespace(threshold=.5, locate=lambda image: (box, 1))),
                  patch("diagnose_offsets.annotation_signature", return_value="fixture")):
                diagnose(args)
            import csv
            import json
            with (root / "output/predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 9)
            self.assertEqual(sum(row["status"] == "out_of_bounds" for row in rows), 5)
            zero = next(row for row in rows if float(row["requested_dx"]) == float(row["requested_dy"]) == 0)
            self.assertEqual((zero["x_left"], zero["y_bottom"], zero["side"]), ("0", "0", "32"))
            self.assertEqual(zero["expression_correct"], "True")
            summary = json.loads((root / "output/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary["conditions"]), 9)
            self.assertTrue((root / "output/report.md").is_file())
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                diagnose(args)


if __name__ == "__main__":
    unittest.main()
