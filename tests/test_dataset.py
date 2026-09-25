import csv
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from dataset import (
    ImageRecord,
    build_dual_view_transform,
    load_label_manifest,
    parse_filename,
    stratified_split,
)


class DatasetTests(unittest.TestCase):
    def test_parse_filename(self):
        record = parse_filename("ATRI_tatr01_w_d2_p1_fa.png")
        self.assertEqual(record.scale, "w")
        self.assertEqual(record.outfit, "d2")
        self.assertEqual(record.expression, "fa")

    def test_stratified_split_keeps_each_group_in_training(self):
        records = [
            ImageRecord(
                path=Path(f"ATRI_tatr01_w_d{index}_p1_f1.png"),
                character="ATRI",
                shoe_variant="tatr01",
                scale="w",
                outfit=f"d{index}",
                pose="p1",
                expression="f1",
            )
            for index in range(1, 5)
        ]
        train_records, validation_records = stratified_split(records, seed=42)
        self.assertEqual(len(train_records), 3)
        self.assertEqual(len(validation_records), 1)

    def test_paired_transform_uses_matching_random_parameters(self):
        image = Image.new("RGBA", (16, 16), (0, 0, 0, 255))
        for x in range(8):
            for y in range(16):
                image.putpixel((x, y), (255, 0, 0, 255))
        transform = build_dual_view_transform(
            height=16,
            width=16,
            expression_size=16,
            train=True,
            expression_width_fraction=1.0,
            expression_height_fraction=1.0,
            margin=0.0,
        )
        torch.manual_seed(7)
        views = transform(image)
        torch.testing.assert_close(views["full"], views["expression"])

    def test_load_label_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(root / "test.png")
            manifest = root / "labels.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["filename", "outfit", "pose", "expression"],
                )
                writer.writeheader()
                writer.writerow({
                    "filename": "test.png",
                    "outfit": "d1",
                    "pose": "p1",
                    "expression": "f1",
                })
            records = load_label_manifest(root, manifest)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].expression, "f1")


if __name__ == "__main__":
    unittest.main()
