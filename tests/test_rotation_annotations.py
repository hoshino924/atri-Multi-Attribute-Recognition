import csv
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from annotate_faces import AnnotationStore, FaceAnnotator, scan_images
from face_regions import SquareBox
from image_rotation import (
    DEFAULT_ANGLES, angle_text, parse_angles, rotate_box_proposal, rotate_image,
    rotation_geometry, scale_box_proposal,
)
from rotation_annotations import ROTATED_CSV_FIELDS, RotatedAnnotationStore, atomic_write, expand_views


def windows_file_error(code):
    error = PermissionError("Simulated Windows file access error")
    error.winerror = code
    return error


class RotationGeometryTests(unittest.TestCase):
    def test_zero_and_quarter_turns_preserve_exact_pixels_and_box(self):
        source = Image.new("RGBA", (320, 600), (0, 0, 0, 0))
        ImageDraw.Draw(source).rectangle((80, 100, 219, 239), fill=(200, 40, 70, 255))
        box = SquareBox(80, 360, 140)
        expected = {
            0: box, 90: SquareBox(100, 80, 140),
            180: SquareBox(100, 100, 140), 270: SquareBox(360, 100, 140),
        }
        for angle, target_box in expected.items():
            with self.subTest(angle=angle):
                rotated = rotate_image(source, angle)
                self.assertEqual(rotated.size, rotation_geometry(source.size, angle)[0])
                self.assertEqual(rotated.tobytes(), source.rotate(angle, expand=True).tobytes())
                mapped = rotate_box_proposal(box, source.size, angle)
                self.assertEqual(mapped, target_box)
                self.assertEqual(rotated.crop(mapped.pil_bounds(rotated.height)).getchannel("A").getextrema(), (255, 255))

    def test_oblique_rotation_preserves_alpha_and_is_counterclockwise(self):
        source = Image.new("RGBA", (128, 192), (0, 0, 0, 0))
        ImageDraw.Draw(source).rectangle((98, 48, 102, 52), fill=(200, 100, 50, 255))
        before = source.tobytes()
        rotated = rotate_image(source, 45)
        self.assertEqual(rotated.size, (227, 227))
        left, top, right, bottom = rotated.getchannel("A").getbbox()
        # The marker above/right of center moves above/slightly left of center.
        self.assertLess((left + right) / 2, rotated.width / 2)
        self.assertLess((top + bottom) / 2, rotated.height / 2 - 40)
        self.assertEqual(rotated.getpixel((0, 0))[3], 0)
        self.assertEqual(source.tobytes(), before)
        proposal = rotate_box_proposal(SquareBox(90, 130, 25), source.size, 45)
        self.assertTrue(proposal.fits(*rotated.size))
        x0, y0, x1, y1 = proposal.pil_bounds(rotated.height)
        self.assertTrue(x0 <= (left + right) / 2 <= x1)
        self.assertTrue(y0 <= (top + bottom) / 2 <= y1)

    def test_custom_angles_are_normalized_unique_and_round_trip(self):
        self.assertEqual(parse_angles([315, 0, 45, -90]), (0, 45, 270, 315))
        for values in ([0, 360], [], [float("nan")], [float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                parse_angles(values)
        angle = 22.12345678912345
        self.assertEqual(float(angle_text(angle)), angle)
        self.assertEqual(scale_box_proposal(SquareBox(80, 360, 140), (320, 600), (160, 300)),
                         SquareBox(40, 180, 70))


class RotationAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        source = Image.new("RGBA", (320, 600), (0, 0, 0, 0))
        ImageDraw.Draw(source).rectangle((80, 100, 219, 239), fill=(120, 60, 180, 255))
        for pose, expression in (("p1", "f1"), ("p1", "fh"), ("p2", "f1")):
            image = source
            if expression == "fh":
                image = Image.new("RGBA", (320, 720), (0, 0, 0, 0))
                image.paste(source, (0, 120))
            image.save(self.directory / f"アトリ_tatr01_w_d1_{pose}_{expression}.png")
        source.resize((160, 300)).save(self.directory / "アトリ_tatr01_s_d1_p1_f1.png")
        self.sources = scan_images(self.directory, "w")
        self.views = expand_views(self.sources, DEFAULT_ANGLES)
        self.output = self.directory / "rotated.csv"
        self.store = RotatedAnnotationStore(self.output, self.views)
        self.box = SquareBox(80, 360, 140)

    def test_angle_major_order_and_separate_row_identity(self):
        self.assertEqual([item.angle for item in self.views], [angle for angle in DEFAULT_ANGLES for _ in self.sources])
        self.assertEqual(len({item.key for item in self.views}), 18)
        self.assertNotEqual(self.views[0].group, self.views[3].group)
        self.assertNotEqual(self.views[0].group, self.views[2].group)
        self.assertEqual(self.views[0].content_id, self.views[3].content_id)

    def test_confirmed_rows_round_trip_and_source_files_are_unchanged(self):
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in self.directory.glob("*.png")}
        self.store.save(self.views[0], self.box)
        angle45 = self.views[3]
        proposal, origin, reference = self.store.suggestion(angle45)
        self.assertEqual(origin, "rotated_reference")
        self.assertNotIn(angle45.key, self.store.boxes)
        self.store.save(angle45, proposal, origin, reference)
        restored = RotatedAnnotationStore(self.output, expand_views(self.sources, DEFAULT_ANGLES))
        self.assertEqual(restored.boxes, self.store.boxes)
        self.assertEqual(restored.provenance, self.store.provenance)
        with self.output.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, list(ROTATED_CSV_FIELDS))
            rows = list(reader)
        self.assertEqual([row["angle"] for row in rows], ["0", "45"])
        self.assertTrue(all(row["confirmed"] == "true" for row in rows))
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before})

    def test_inheritance_is_within_angle_pose_and_resolution_and_stays_pending(self):
        self.store.save(self.views[0], self.box)
        proposal, origin, _ = self.store.suggestion(self.views[1])
        self.assertEqual((proposal, origin), (self.box, "inherited"))
        self.assertNotIn(self.views[1].key, self.store.boxes)
        self.assertIsNone(self.store.suggestion(self.views[2])[0])
        self.assertEqual(self.store.suggestion(self.views[3])[1], "rotated_reference")
        newer = SquareBox(75, 350, 150)
        self.store.save(self.views[1], newer)
        restored = RotatedAnnotationStore(self.output, self.views)
        self.assertEqual(restored.suggestion(self.views[1])[0], newer)
        self.assertEqual(restored.last_boxes[self.views[0].group], newer)

    def test_legacy_import_is_a_proposal_and_never_overwritten(self):
        old_path = self.directory / "old.csv"
        old = AnnotationStore(old_path, self.sources)
        old.save(self.sources[0], self.box)
        before = old_path.read_bytes()
        self.store.add_reference(old_path, scan_images(self.directory, "all"))
        self.assertEqual(self.store.suggestion(self.views[0])[0], self.box)
        self.assertEqual(self.store.boxes, {})
        self.assertFalse(self.output.exists())
        self.assertEqual(self.store.suggestion(self.views[3])[1], "rotated_reference")
        with self.assertRaises(ValueError):
            RotatedAnnotationStore(old_path, self.views)
        self.assertEqual(old_path.read_bytes(), before)

    def test_same_angle_cross_resolution_reference_requires_confirmation(self):
        self.store.save(self.views[0], self.box)
        angle45 = self.views[3]
        box45, origin, reference = self.store.suggestion(angle45)
        self.store.save(angle45, box45, origin, reference)
        small_views = expand_views(scan_images(self.directory, "s"), DEFAULT_ANGLES)
        target = RotatedAnnotationStore(self.directory / "small.csv", small_views)
        target.add_reference(self.output, scan_images(self.directory, "all"))
        proposed, origin, _ = target.suggestion(small_views[1])
        self.assertEqual(origin, "scaled_reference")
        self.assertTrue(proposed.fits(small_views[1].width, small_views[1].height))
        self.assertEqual(target.boxes, {})

    def test_source_pixel_change_is_rejected_even_when_dimensions_match(self):
        self.store.save(self.views[0], self.box)
        before = self.output.read_bytes()
        Image.new("RGBA", (320, 600), "red").save(self.views[0].path)
        with self.assertRaises(ValueError):
            self.store.load_view(self.views[0])
        with self.assertRaises(ValueError):
            self.store.save(self.views[0], self.box)
        with self.assertRaises(ValueError):
            RotatedAnnotationStore(self.output, expand_views(scan_images(self.directory, "w"), DEFAULT_ANGLES))
        self.assertEqual(self.output.read_bytes(), before)

    def test_changed_manifest_or_csv_is_not_overwritten(self):
        self.store.save(self.views[0], self.box)
        for path in (self.output, self.store.manifest_path):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaises(ValueError):
                    self.store.save(self.views[1], self.box)
                self.assertEqual(path.read_bytes(), original + b"\n")
                path.write_bytes(original)

    def test_failed_csv_replace_preserves_confirmations_and_can_retry(self):
        self.store.save(self.views[0], self.box)
        before = self.output.read_bytes()
        for error in (PermissionError("CSV is open"), windows_file_error(112)):
            with self.subTest(error=error), patch("rotation_annotations.time.sleep") as sleep:
                with patch("rotation_annotations.os.replace", side_effect=error) as replace:
                    with self.assertRaises(PermissionError):
                        self.store.save(self.views[1], self.box)
                self.assertEqual(replace.call_count, 1)
                sleep.assert_not_called()
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(self.store.boxes, {self.views[0].key: self.box})
        self.assertFalse(list(self.directory.glob("*.tmp")))
        self.store.save(self.views[1], self.box)
        self.assertEqual(len(self.store.boxes), 2)

    def test_transient_windows_replace_errors_retry_and_save_complete_csv(self):
        real_replace = os.replace
        for code in (5, 32, 33):
            with self.subTest(winerror=code):
                output = self.directory / f"retry_{code}.csv"
                store = RotatedAnnotationStore(output, self.views)
                store.save(self.views[0], self.box)
                attempts = []

                def replace_after_busy(source, destination):
                    attempts.append((source, destination))
                    if len(attempts) <= 2:
                        raise windows_file_error(code)
                    return real_replace(source, destination)

                with patch("rotation_annotations.time.sleep") as sleep:
                    with patch("rotation_annotations.os.replace", side_effect=replace_after_busy):
                        store.save(self.views[1], self.box)
                self.assertGreaterEqual(sleep.call_count, 2)
                restored = RotatedAnnotationStore(output, self.views)
                self.assertEqual(restored.boxes, {view.key: self.box for view in self.views[:2]})
                self.assertEqual(restored.boxes, store.boxes)
                self.assertFalse(list(self.directory.glob("*.tmp")))

    def test_persistent_windows_denial_is_bounded_and_preserves_saved_state(self):
        self.store.save(self.views[0], self.box)
        before = self.output.read_bytes()
        manifest = self.store.manifest_path.read_bytes()
        state = (self.store.signature, dict(self.store.boxes), dict(self.store.provenance),
                 dict(self.store.last_boxes), dict(self.store.last_keys))
        error = windows_file_error(5)
        with patch("rotation_annotations.time.sleep") as sleep:
            with patch("rotation_annotations.os.replace", side_effect=error) as replace:
                with self.assertRaises(PermissionError) as raised:
                    self.store.save(self.views[1], self.box)
        self.assertIs(raised.exception, error)
        self.assertEqual(replace.call_count, 6)
        self.assertEqual(sleep.call_count, 5)
        self.assertAlmostEqual(sum(call.args[0] for call in sleep.call_args_list), 1.55)
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(self.store.manifest_path.read_bytes(), manifest)
        self.assertEqual((self.store.signature, self.store.boxes, self.store.provenance,
                          self.store.last_boxes, self.store.last_keys), state)
        self.assertFalse(list(self.directory.glob("*.tmp")))

    def test_external_change_during_retry_is_not_overwritten(self):
        self.store.save(self.views[0], self.box)
        for path in (self.output, self.store.manifest_path):
            with self.subTest(path=path.name):
                original = path.read_bytes()

                def external_edit(_delay):
                    path.write_bytes(original + b"\n")

                with patch("rotation_annotations.time.sleep", side_effect=external_edit):
                    with patch("rotation_annotations.os.replace", side_effect=windows_file_error(32)) as replace:
                        with self.assertRaises(ValueError) as raised:
                            self.store.save(self.views[1], self.box)
                self.assertEqual(raised.exception.key, "csv_changed")
                self.assertEqual(replace.call_count, 1)
                self.assertEqual(path.read_bytes(), original + b"\n")
                self.assertEqual(self.store.boxes, {self.views[0].key: self.box})
                self.assertFalse(list(self.directory.glob("*.tmp")))
                path.write_bytes(original)

    def test_transient_access_error_in_guard_retries_before_replace(self):
        check = Mock(side_effect=[windows_file_error(32), None])
        with patch("rotation_annotations.time.sleep") as sleep:
            with patch("rotation_annotations.os.replace", wraps=os.replace) as replace:
                atomic_write(self.output, b"complete data", check)
        self.assertEqual(check.call_count, 2)
        self.assertEqual(replace.call_count, 1)
        sleep.assert_called_once()
        self.assertEqual(self.output.read_bytes(), b"complete data")

    def test_locked_temporary_cleanup_does_not_hide_replace_error(self):
        self.store.save(self.views[0], self.box)
        before = self.output.read_bytes()
        error = windows_file_error(5)
        with patch("rotation_annotations.time.sleep"):
            with patch("rotation_annotations.os.replace", side_effect=error):
                with patch("rotation_annotations.Path.unlink", side_effect=windows_file_error(32)):
                    with self.assertRaises(PermissionError) as raised:
                        self.store.save(self.views[1], self.box)
        self.assertIs(raised.exception, error)
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(self.store.boxes, {self.views[0].key: self.box})

    def test_manifest_only_resume_after_interrupted_initial_csv_save(self):
        from rotation_annotations import atomic_write
        def fail_csv(path, data, check):
            if path == self.output.resolve():
                raise PermissionError("CSV blocked")
            return atomic_write(path, data, check)
        with patch("rotation_annotations.atomic_write", side_effect=fail_csv):
            with self.assertRaises(PermissionError):
                self.store.save(self.views[0], self.box)
        restored = RotatedAnnotationStore(self.output, self.views)
        self.assertEqual(restored.boxes, {})
        restored.save(self.views[0], self.box)
        self.assertTrue(self.output.exists())

    def test_resume_rejects_different_angles_and_unconfirmed_csv_rows(self):
        self.store.save(self.views[0], self.box)
        with self.assertRaises(ValueError):
            RotatedAnnotationStore(self.output, expand_views(self.sources, [0, 90]))
        original = self.output.read_text(encoding="utf-8-sig")
        self.output.write_text(original.replace(",true", ",false"), encoding="utf-8-sig")
        with self.assertRaises(ValueError):
            RotatedAnnotationStore(self.output, self.views)

    def test_fractional_angle_and_odd_canvas_survive_save_and_reload(self):
        views = expand_views(self.sources[:1], [22.12345678912345])
        output = self.directory / "fractional.csv"
        store = RotatedAnnotationStore(output, views)
        proposed = rotate_box_proposal(self.box, (320, 600), views[0].angle)
        store.save(views[0], proposed)
        restored = RotatedAnnotationStore(output, views)
        self.assertEqual(restored.boxes[views[0].key], proposed)
        self.assertEqual(restored.load_view(views[0]).size, (views[0].width, views[0].height))

    def test_gui_angle_jump_language_and_pending_filter(self):
        try:
            import tkinter as tk
            root = tk.Tk()
        except ImportError as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        except tk.TclError as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        root.withdraw()
        self.addCleanup(root.destroy)
        app = FaceAnnotator(root, self.views, self.store)
        app.set_box(self.box)
        self.assertTrue(app.save_current())
        app.angle_variable.set("45")
        app.jump_angle()
        self.assertEqual(app.item.angle, 45)
        self.assertEqual(app.image.size, (app.item.width, app.item.height))
        self.assertNotIn(app.item.key, self.store.boxes)
        self.assertIn("45°", app.heading.get())
        app.set_language("zh")
        self.assertIn("旋转画布", app.heading.get())
        app.set_language("en")
        self.assertTrue(app.save_current())
        app.unconfirmed_only.set(True)
        app.filter_unconfirmed()
        self.assertNotIn(app.item.key, self.store.boxes)
        self.assertIn("0°: 1/3", app.angle_progress.get())
        self.assertIn("45°: 1/3", app.angle_progress.get())


if __name__ == "__main__":
    unittest.main()
