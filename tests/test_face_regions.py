import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from face_regions import (
    COORDINATE_SYSTEM, CSV_FIELDS, FaceCanvas, SquareBox, crop_face,
    load_face_annotations, locator_canvas, map_locator_box,
)


class FaceRegionTests(unittest.TestCase):
    def test_bottom_origin_and_top_padding(self):
        image = Image.new("RGB", (20, 40))
        image.paste((10, 20, 30), (4, 12, 12, 20))
        box = SquareBox(4, 20, 8)
        self.assertEqual(crop_face(image, box).getpixel((0, 0)), (10, 20, 30))
        taller = Image.new("RGB", (20, 47))
        taller.paste(image, (0, 7))
        self.assertEqual(crop_face(image, box).tobytes(), crop_face(taller, box).tobytes())

    def test_exact_size_preserves_pixels(self):
        image = Image.new("RGB", (626, 626), (25, 50, 75))
        image.putpixel((19, 27), (99, 8, 230))
        self.assertEqual(FaceCanvas()(image).tobytes(), image.tobytes())

    def test_small_crop_is_padded_not_upscaled(self):
        output = FaceCanvas(10)(Image.new("RGBA", (4, 4), (255, 0, 0, 255)))
        self.assertEqual(sum(pixel == (255, 0, 0) for pixel in output.getdata()), 16)
        self.assertEqual(output.getpixel((3, 3)), (255, 0, 0))

    def test_larger_crop_and_alpha(self):
        self.assertEqual(FaceCanvas(8)(Image.new("RGB", (16, 16))).size, (8, 8))
        self.assertEqual(FaceCanvas(8)(Image.new("RGBA", (8, 8))).getbbox(), None)

    def test_letterbox_roundtrip_for_odd_dimensions(self):
        for size in ((1156, 2740), (2129, 2578), (1000, 333)):
            image = Image.new("RGB", size)
            original = SquareBox(100, 100, 200)
            _, mapping = locator_canvas(image)
            sx, sy, ox, oy = mapping
            left, top, right, bottom = original.pil_bounds(image.height)
            normalized = ((left + right) / 2 * sx + ox, (top + bottom) / 2 * sy + oy, 200 * sx, 200 * sy)
            recovered = map_locator_box([value / 256 for value in normalized], mapping, size, 256)
            self.assertEqual(recovered, original)
        with self.assertRaises(ValueError):
            map_locator_box((float("nan"), .5, .2, .2), (1, 1, 0, 0), (256, 256), 256)

    def test_annotation_validation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "ATRI_tatr01_l_d1_p1_f1.png"
            Image.new("RGBA", (10, 20)).save(image)
            record = SimpleNamespace(path=image, character="ATRI", shoe_variant="tatr01", scale="l", outfit="d1", pose="p1", expression="f1")
            row = dict(filename=image.name, character="ATRI", shoe_variant="tatr01", scale="l", outfit="d1", pose="p1", expression="f1", image_width=10, image_height=20, coordinate_system=COORDINATE_SYSTEM, x_left=1, y_bottom=2, side=5)
            path = root / "boxes.csv"

            def write(rows):
                with path.open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)

            write([row])
            self.assertEqual(load_face_annotations([record], path)[image.name], SquareBox(1, 2, 5))
            for rows in ([], [row, row], [{**row, "image_height": 21}], [{**row, "scale": "w"}], [{**row, "side": 11}]):
                write(rows)
                with self.assertRaises(ValueError):
                    load_face_annotations([record], path)


if __name__ == "__main__":
    unittest.main()
