import csv
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from annotation_i18n import AnnotationError
from annotate_faces import (
    AnnotationStore,
    CSV_FIELDS,
    FaceAnnotator,
    SquareBox,
    scan_images,
    square_from_drag,
)


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        original = Image.new("RGBA", (320, 600), (0, 0, 0, 0))
        ImageDraw.Draw(original).rectangle((80, 100, 219, 239), fill=(120, 60, 180, 255))
        original.save(self.directory / "アトリ_tatr01_w_d1_p1_f1.png")
        taller = Image.new("RGBA", (320, 720), (0, 0, 0, 0))
        taller.paste(original, (0, 120))
        taller.save(self.directory / "アトリ_tatr01_w_d1_p1_ff.png")
        original.save(self.directory / "アトリ_tatr01_w_d2_p1_f1.png")
        original.save(self.directory / "アトリ_tatr01_w_d1_p2_f1.png")
        original.resize((160, 300)).save(self.directory / "アトリ_tatr01_s_d1_p1_f1.png")
        self.records = scan_images(self.directory, "w")
        self.output = self.directory / "labels.csv"
        self.box = SquareBox(80, 360, 140)

    def test_scan_filters_scale_and_groups_pose_before_outfit(self):
        self.assertEqual(len(self.records), 4)
        self.assertEqual([(item.pose, item.outfit, item.expression) for item in self.records], [
            ("p1", "d1", "f1"), ("p1", "d1", "ff"),
            ("p1", "d2", "f1"), ("p2", "d1", "f1"),
        ])

    def test_bottom_origin_crops_identical_content_after_top_padding(self):
        self.assertEqual(self.box.pil_bounds(600), (80, 100, 220, 240))
        self.assertEqual(self.box.pil_bounds(720), (80, 220, 220, 360))
        with Image.open(self.records[0].path) as first, Image.open(self.records[1].path) as taller:
            first_crop = first.crop(self.box.pil_bounds(first.height))
            taller_crop = taller.crop(self.box.pil_bounds(taller.height))
            self.assertEqual(first_crop.tobytes(), taller_crop.tobytes())

    def test_square_drawing_handles_all_directions_and_edges(self):
        for end in ((95, 80), (5, 80), (95, 20), (5, 20), (300, 400), (-50, -60)):
            with self.subTest(end=end):
                box = square_from_drag((50, 50), end, 100, 120)
                self.assertTrue(box.fits(100, 120))
                left, top, right, bottom = box.pil_bounds(120)
                self.assertEqual(right - left, bottom - top)
                self.assertTrue(left <= 50 <= right and top <= 50 <= bottom)
        self.assertIsNone(square_from_drag((50, 50), (50, 50), 100, 120))
        moved = self.box.moved(1000, -1000, 320, 600)
        self.assertEqual(moved, SquareBox(180, 0, 140))
        self.assertTrue(self.box.resized(1000, 320, 600).fits(320, 600))

    def test_csv_round_trip_remembers_latest_pose_box_without_touching_images(self):
        before = {path: hashlib.sha256(path.read_bytes()).digest()
                  for path in self.directory.glob("*.png")}
        store = AnnotationStore(self.output, self.records)
        store.save(self.records[0], self.box)
        store.save(self.records[1], self.box)
        store.save(self.records[3], SquareBox(40, 100, 180))
        revised = SquareBox(75, 355, 150)
        store.save(self.records[0], revised)
        restored = AnnotationStore(self.output, self.records)
        self.assertEqual(len(restored.boxes), 3)
        self.assertEqual(restored.last_boxes[self.records[2].group], revised)
        self.assertEqual(restored.last_boxes[self.records[3].group], SquareBox(40, 100, 180))
        self.assertEqual(restored.boxes[self.records[1].path.name], self.box)
        self.assertTrue(self.output.read_bytes().startswith(b"\xef\xbb\xbf"))
        with self.output.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, list(CSV_FIELDS))
            rows = list(reader)
        self.assertEqual(rows[-1]["filename"], self.records[0].path.name)
        self.assertEqual(rows[-1]["coordinate_system"], "bottom_left_pixels")
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).digest() for path in before})

    def test_bad_box_or_non_csv_cannot_overwrite_sources(self):
        store = AnnotationStore(self.output, self.records)
        with self.assertRaises(ValueError):
            store.save(self.records[0], SquareBox(300, 0, 140))
        self.assertFalse(self.output.exists())
        with self.assertRaises(ValueError):
            AnnotationStore(self.records[0].path, self.records)

    def test_resume_rejects_changed_dimensions_and_wrong_resolution(self):
        store = AnnotationStore(self.output, self.records)
        store.save(self.records[0], self.box)
        previous = self.output.read_bytes()
        with self.assertRaises(ValueError):
            AnnotationStore(self.output, scan_images(self.directory, "s"))
        Image.new("RGBA", (321, 600)).save(self.records[0].path)
        with self.assertRaises(ValueError):
            AnnotationStore(self.output, scan_images(self.directory, "w"))
        self.assertEqual(self.output.read_bytes(), previous)

    def test_failed_atomic_replace_keeps_previous_annotation(self):
        store = AnnotationStore(self.output, self.records)
        store.save(self.records[0], self.box)
        previous = self.output.read_bytes()
        with patch("annotate_faces.os.replace", side_effect=PermissionError("CSV is open")):
            with self.assertRaises(PermissionError):
                store.save(self.records[1], self.box)
        self.assertEqual(self.output.read_bytes(), previous)
        self.assertEqual(store.boxes, {self.records[0].path.name: self.box})
        self.assertFalse(list(self.directory.glob("*.tmp")))
        store.save(self.records[1], self.box)
        self.assertEqual(len(store.boxes), 2)

    def test_external_csv_change_is_not_overwritten(self):
        store = AnnotationStore(self.output, self.records)
        store.save(self.records[0], self.box)
        changed = self.output.read_bytes() + b"\r\n"
        self.output.write_bytes(changed)
        with self.assertRaises(ValueError):
            store.save(self.records[1], self.box)
        self.assertEqual(self.output.read_bytes(), changed)

    def test_unrelated_csv_is_rejected_without_overwriting(self):
        path = self.directory / "unrelated.csv"
        original = "filename,label\nexample.png,smile\n"
        path.write_text(original, encoding="utf-8")
        with self.assertRaises(ValueError):
            AnnotationStore(path, self.records)
        self.assertEqual(path.read_text(encoding="utf-8"), original)

    def make_gui(self):
        try:
            import tkinter as tk
            root = tk.Tk()
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        except tk.TclError as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        root.withdraw()
        self.addCleanup(root.destroy)
        app = FaceAnnotator(root, self.records, AnnotationStore(self.output, self.records))
        root.update_idletasks()
        return app

    def test_gui_confirmation_inherits_bottom_coordinates_and_resume(self):
        app = self.make_gui()
        app.set_box(self.box)
        app.confirm_next()
        self.assertEqual(app.index, 1)
        self.assertEqual(app.box, self.box)
        self.assertFalse(app.dirty)
        self.assertEqual([var.get() for var in app.coordinate_vars], ["80", "360", "140"])
        self.assertNotIn(app.item.path.name, app.store.boxes)
        app.confirm_next()
        self.assertEqual(app.index, 2)
        self.assertEqual(app.box, self.box)
        app.navigate(1)
        self.assertIsNone(app.box)

    def test_gui_typed_coordinates_are_saved_and_survive_redraw(self):
        app = self.make_gui()
        for variable, value in zip(app.coordinate_vars, (80, 360, 140)):
            variable.set(str(value))
        app.render_image()
        self.assertTrue(app.pending_coordinates())
        app.confirm_next()
        self.assertEqual(app.store.boxes[self.records[0].path.name], self.box)
        self.assertEqual(app.box, self.box)
        with patch.object(app.messagebox, "askyesnocancel", return_value=None):
            app.set_box(SquareBox(70, 350, 150))
            app.navigate(1)
            self.assertEqual(app.index, 1)

    def test_gui_mouse_drawing_and_dragging_use_original_pixels(self):
        app = self.make_gui()
        app.scale_x = app.scale_y = 0.5
        app.offset_x, app.offset_y = 10, 20
        app.start_drag(SimpleNamespace(x=50, y=70, state=0))
        app.end_drag(SimpleNamespace(x=120, y=130))
        self.assertEqual(app.box, self.box)
        app.start_drag(SimpleNamespace(x=70, y=90, state=0))
        app.end_drag(SimpleNamespace(x=75, y=80))
        self.assertEqual(app.box, SquareBox(90, 380, 140))
        _, _, right, bottom = app.screen_bounds()
        app.start_drag(SimpleNamespace(x=right, y=bottom, state=0))
        app.end_drag(SimpleNamespace(x=right + 10, y=bottom + 10))
        self.assertEqual(app.box, SquareBox(90, 360, 160))

    def test_gui_language_switch_preserves_pending_coordinates_and_csv(self):
        app = self.make_gui()
        self.assertEqual(app.root.title(), "ATRI Face Box Annotation")
        app.set_box(self.box)
        app.save_current()
        saved_csv = self.output.read_bytes()
        revised = SquareBox(70, 350, 150)
        app.set_box(revised)
        app.coordinate_vars[0].set("071")

        app.language_selector.set("简体中文")
        app.language_selector.event_generate("<<ComboboxSelected>>")
        app.root.update_idletasks()
        self.assertEqual(app.language, "zh")
        self.assertEqual(app.root.title(), "ATRI 面部方框标注")
        widgets = {key: widget for widget, key in app.text_widgets}
        self.assertEqual(widgets["confirm_next"].cget("text"), "确认并下一张")
        self.assertIn("已保存", app.status.get())
        self.assertIn("尚未保存", app.box_status.get())

        app.set_language("en")
        app.root.update_idletasks()
        self.assertEqual(widgets["confirm_next"].cget("text"), "Confirm & next")
        self.assertEqual(app.box, revised)
        self.assertTrue(app.dirty)
        self.assertEqual(app.index, 0)
        self.assertEqual([var.get() for var in app.coordinate_vars], ["071", "350", "150"])
        self.assertEqual(self.output.read_bytes(), saved_csv)
        app.confirm_next()
        self.assertEqual(app.store.boxes[self.records[0].path.name], SquareBox(71, 350, 150))
        self.assertEqual(app.index, 1)

    def test_gui_dialogs_follow_selected_language(self):
        app = self.make_gui()
        app.set_box(self.box)
        for language, title, body, help_title, error_title, error_detail in (
            ("en", "Unsaved box", "Confirm and save", "Controls and shortcuts",
             "Save failed", "Another program changed the CSV"),
            ("zh", "当前方框尚未保存", "是否先确认", "操作快捷键",
             "保存失败", "CSV 已被其他程序修改"),
        ):
            with self.subTest(language=language):
                app.set_language(language)
                with patch.object(app.messagebox, "askyesnocancel", return_value=None) as question:
                    self.assertFalse(app.allow_leave())
                    self.assertEqual(question.call_args.args[0], title)
                    self.assertIn(body, question.call_args.args[1])
                with patch.object(app.messagebox, "showinfo") as help_dialog:
                    app.show_shortcuts()
                    self.assertEqual(help_dialog.call_args.args[0], help_title)
                with patch.object(app.store, "save", side_effect=AnnotationError("csv_changed")):
                    with patch.object(app.messagebox, "showerror") as error_dialog:
                        self.assertFalse(app.save_current())
                        self.assertEqual(error_dialog.call_args.args[0], error_title)
                        self.assertIn(error_detail, error_dialog.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
