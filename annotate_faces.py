# -*- coding: utf-8 -*-
"""Annotate square face regions with Pillow/Tkinter; never write source images."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import tempfile

from PIL import Image

from annotation_i18n import AnnotationError, LANGUAGES, error_text, translate
from labels import EXPR_CODES, OUTFIT_CODES, POSE_CODES
from face_regions import COORDINATE_SYSTEM, CSV_FIELDS, SquareBox
from image_rotation import DEFAULT_ANGLES, angle_text
from rotation_annotations import RotatedAnnotationStore, expand_views
from app_assets import configure_taskbar, set_app_icon


SCALE_CODES = ("s", "w", "m", "l", "ll")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
@dataclass(frozen=True)
class AnnotationImage:
    path: Path
    character: str
    shoe_variant: str
    scale: str
    outfit: str
    pose: str
    expression: str
    width: int
    height: int

    @property
    def group(self):
        return self.character, self.scale, self.pose

    @property
    def key(self):
        return self.path.name

    @property
    def angle(self):
        return 0.0


def square_from_drag(start, end, width, height):
    """Fit a square in any drag direction, using top-left image coordinates."""
    x0, y0 = start
    dx, dy = end[0] - x0, end[1] - y0
    limit_x = width - x0 if dx >= 0 else x0
    limit_y = height - y0 if dy >= 0 else y0
    side = min(max(abs(dx), abs(dy)), limit_x, limit_y)
    if side < 1:
        return None
    left = x0 if dx >= 0 else x0 - side
    top = y0 if dy >= 0 else y0 - side
    return SquareBox(left, height - top - side, side)


def scan_images(root, scale):
    """Read one resolution level without importing the training/Torch stack."""
    if scale not in (*SCALE_CODES, "all"):
        raise AnnotationError("bad_scale", scale=scale)
    root = Path(root).resolve()
    if not root.is_dir():
        raise AnnotationError("missing_directory", root=root)
    records = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = path.stem.split("_")
        if len(parts) != 6:
            raise AnnotationError("filename_fields", filename=path.name)
        character, variant, image_scale, outfit, pose, expression = parts
        if scale != "all" and image_scale != scale:
            continue
        if (
            image_scale not in SCALE_CODES or not character or not variant or outfit not in OUTFIT_CODES
            or pose not in POSE_CODES or expression not in EXPR_CODES
        ):
            raise AnnotationError("filename_labels", filename=path.name)
        with Image.open(path) as source:
            width, height = source.size
        records.append(AnnotationImage(
            path, character, variant, image_scale, outfit, pose, expression,
            width, height,
        ))
    if not records:
        raise AnnotationError("no_images", scale=scale, root=root)
    return sorted(records, key=lambda item: (
        item.character, POSE_CODES.index(item.pose),
        OUTFIT_CODES.index(item.outfit), item.shoe_variant,
        EXPR_CODES.index(item.expression), SCALE_CODES.index(item.scale), item.path.name,
    ))


class AnnotationStore:
    """Validated, resumable CSV with atomic saves and external-change checks."""

    def __init__(self, path, records):
        self.path = Path(path).resolve()
        if self.path.suffix.lower() != ".csv":
            raise AnnotationError("csv_only")
        self.records = {item.path.name: item for item in records}
        if self.path in {item.path.resolve() for item in records}:
            raise AnnotationError("output_is_image")
        self.boxes = {}
        self.last_boxes = {}
        data = self.path.read_bytes() if self.path.exists() else None
        self.signature = self._signature(data)
        if data is not None:
            self._load(data)

    @staticmethod
    def _signature(data):
        return hashlib.sha256(data).digest() if data is not None else None

    def _row(self, filename, box):
        item = self.records[filename]
        return {
            "filename": filename,
            "character": item.character,
            "shoe_variant": item.shoe_variant,
            "scale": item.scale,
            "outfit": item.outfit,
            "pose": item.pose,
            "expression": item.expression,
            "image_width": item.width,
            "image_height": item.height,
            "coordinate_system": COORDINATE_SYSTEM,
            "x_left": box.x_left,
            "y_bottom": box.y_bottom,
            "side": box.side,
        }

    def _load(self, data):
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        if reader.fieldnames != list(CSV_FIELDS):
            raise AnnotationError("csv_header")
        for line, row in enumerate(reader, start=2):
            try:
                filename = row["filename"]
                if filename not in self.records or filename in self.boxes:
                    raise AnnotationError("csv_image")
                box = SquareBox(*(int(row[key]) for key in (
                    "x_left", "y_bottom", "side",
                )))
                item = self.records[filename]
                if not box.fits(item.width, item.height):
                    raise AnnotationError("csv_box")
                expected = {key: str(value) for key, value in self._row(filename, box).items()}
                if row != expected:
                    raise AnnotationError("csv_metadata")
                self.boxes[filename] = box
                self.last_boxes[item.group] = box
            except (ValueError, TypeError, KeyError) as exc:
                raise AnnotationError("csv_row", line=line, error=exc) from exc

    def _check_external_change(self):
        data = self.path.read_bytes() if self.path.exists() else None
        if self._signature(data) != self.signature:
            raise AnnotationError("csv_changed")

    def load_view(self, item):
        with Image.open(item.path) as source:
            if source.size != (item.width, item.height):
                raise AnnotationError("dimensions_changed")
            return source.convert("RGBA")

    def suggestion(self, item):
        if item.key in self.boxes:
            return self.boxes[item.key], "manual", ""
        return self.last_boxes.get(item.group), "inherited", ""

    def save(self, item, box, origin="manual", reference=""):
        if not box.fits(item.width, item.height):
            raise AnnotationError("box_bounds")
        self._check_external_change()
        updated = dict(self.boxes)
        # Keep confirmation order so each pose resumes with its latest box.
        updated.pop(item.path.name, None)
        updated[item.path.name] = box
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8-sig", newline="", delete=False,
                dir=self.path.parent, prefix=f".{self.path.stem}.", suffix=".tmp",
            ) as stream:
                temporary_path = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for filename, current_box in updated.items():
                    writer.writerow(self._row(filename, current_box))
                stream.flush()
                os.fsync(stream.fileno())
            signature = self._signature(temporary_path.read_bytes())
            self._check_external_change()
            os.replace(temporary_path, self.path)
            self.signature = signature
            self.boxes = updated
            self.last_boxes[item.group] = box
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


class FaceAnnotator:
    def __init__(self, root, records, store, language="en"):
        import tkinter as tk
        from tkinter import messagebox, ttk
        from PIL import ImageTk

        self.tk = tk
        self.messagebox = messagebox
        self.ImageTk = ImageTk
        self.root = root
        self.records = records
        self.store = store
        self.language = language
        self.text_widgets = []
        self.index = 0
        self.box = None
        self.box_origin = "manual"
        self.box_reference = ""
        self.dirty = False
        self.image = None
        self.drag = None
        self.resize_job = None
        self.scale_x = self.scale_y = 1.0
        self.offset_x = self.offset_y = 0

        root.title(self.tr("window_title"))
        root.geometry(f"1160x{max(680, min(850, root.winfo_screenheight() - 100))}")
        root.minsize(900, 680)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        self.heading = tk.StringVar()
        header = ttk.Frame(root)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=6)
        header.columnconfigure(0, weight=1)
        ttk.Label(header, textvariable=self.heading, font=("Microsoft YaHei UI", 11)).grid(
            row=0, column=0, sticky="w",
        )
        language_panel = ttk.Frame(header)
        language_panel.grid(row=0, column=1, sticky="ne", padx=(12, 0))
        ttk.Label(language_panel, text="Language / 语言").pack(anchor="w")
        language_label = next(label for label, code in LANGUAGES.items() if code == language)
        self.language_variable = tk.StringVar(value=language_label)
        self.language_selector = ttk.Combobox(
            language_panel, textvariable=self.language_variable,
            values=tuple(LANGUAGES), state="readonly", width=13,
        )
        self.language_selector.pack()
        self.language_selector.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.set_language(LANGUAGES[self.language_variable.get()]),
        )
        navigation = ttk.Frame(header)
        navigation.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.localized(ttk.Label(navigation), "angle_label").pack(side="left")
        self.angles = tuple(dict.fromkeys(item.angle for item in records))
        self.angle_variable = tk.StringVar(value=angle_text(self.angles[0]))
        self.angle_selector = ttk.Combobox(
            navigation, textvariable=self.angle_variable, state="readonly", width=8,
            values=tuple(angle_text(angle) for angle in self.angles),
        )
        self.angle_selector.pack(side="left", padx=8)
        self.angle_selector.bind("<<ComboboxSelected>>", self.jump_angle)
        self.unconfirmed_only = tk.BooleanVar(value=False)
        self.localized(ttk.Checkbutton(
            navigation, variable=self.unconfirmed_only, command=self.filter_unconfirmed,
        ), "unconfirmed_only").pack(side="left", padx=8)
        self.angle_progress = tk.StringVar()
        ttk.Label(header, textvariable=self.angle_progress).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(5, 0),
        )
        self.canvas = tk.Canvas(root, background="#242830", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=(12, 6))
        panel = ttk.Frame(root, padding=(10, 0, 12, 0))
        panel.grid(row=1, column=1, sticky="ns")
        self.localized(ttk.Label(panel), "preview_title").pack(anchor="w")
        self.preview = ttk.Label(panel, anchor="center")
        self.preview.pack(pady=6)
        self.localized(ttk.Label(panel), "coordinates_title").pack(anchor="w")
        coordinates = ttk.Frame(panel)
        coordinates.pack(fill="x", pady=6)
        self.coordinate_vars = [tk.StringVar() for _ in range(3)]
        for row, (key, variable) in enumerate(zip(
            ("x_left", "y_bottom", "side"), self.coordinate_vars,
        )):
            self.localized(ttk.Label(coordinates), key).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(coordinates, textvariable=variable, width=14).grid(
                row=row, column=1, padx=(15, 0), pady=3,
            )
        self.localized(ttk.Button(panel, command=self.apply_coordinates), "apply").pack(fill="x")
        self.box_status = tk.StringVar()
        ttk.Label(panel, textvariable=self.box_status, wraplength=260).pack(
            fill="x", pady=6,
        )
        self.localized(ttk.Label(panel, wraplength=260), "legend").pack(anchor="w", pady=6)
        controls = ttk.Frame(root)
        controls.grid(row=2, column=0, columnspan=2, pady=10)
        for column, (key, callback) in enumerate((
            ("previous", lambda: self.navigate(-1)),
            ("confirm_next", self.confirm_next),
            ("next", lambda: self.navigate(1)),
            ("unannotated", self.next_unannotated),
            ("shortcuts", self.show_shortcuts),
        )):
            self.localized(ttk.Button(controls, command=callback), key).grid(row=0, column=column, padx=4)
        self.status = tk.StringVar()
        self.set_status("status_output", path=store.path)
        ttk.Label(root, textvariable=self.status, wraplength=1050).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 10),
        )
        self.canvas.bind("<Configure>", self.schedule_render)
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.end_drag)
        self.canvas.bind("<MouseWheel>", self.wheel)
        self.canvas.bind("<Button-4>", lambda event: self.change_size(1))
        self.canvas.bind("<Button-5>", lambda event: self.change_size(-1))
        root.bind("<KeyPress>", self.keypress)
        first = next((i for i, item in enumerate(records) if item.key not in store.boxes), 0)
        self.load_image(first)
        if self.image is None:
            root.destroy()

    @property
    def item(self):
        return self.records[self.index]

    def tr(self, key, **values):
        return translate(self.language, key, **values)

    def localized(self, widget, key):
        self.text_widgets.append((widget, key))
        widget.configure(text=self.tr(key))
        return widget

    def set_status(self, key, **values):
        self.status_key = key
        self.status_values = values
        self.status.set(self.tr(key, **values))

    def set_language(self, language):
        self.language = language
        self.language_variable.set(next(label for label, code in LANGUAGES.items() if code == language))
        self.root.title(self.tr("window_title"))
        for widget, key in self.text_widgets:
            widget.configure(text=self.tr(key))
        self.set_status(self.status_key, **self.status_values)
        self.refresh_box()

    def show_shortcuts(self):
        self.messagebox.showinfo(self.tr("help_title"), self.tr("help_text"), parent=self.root)

    def load_image(self, index):
        item = self.records[index]
        try:
            image = self.store.load_view(item)
            box, origin, reference = self.store.suggestion(item)
        except (OSError, ValueError) as exc:
            self.messagebox.showerror(self.tr("open_error"), error_text(exc, self.language), parent=self.root)
            return
        self.index = index
        self.image = image
        self.box, self.box_origin, self.box_reference = box, origin, reference
        self.angle_variable.set(angle_text(item.angle))
        values = ("", "", "") if self.box is None else (
            self.box.x_left, self.box.y_bottom, self.box.side,
        )
        for variable, value in zip(self.coordinate_vars, values):
            variable.set(str(value))
        self.dirty = False
        self.drag = None
        self.render_image()
        self.canvas.focus_set()

    def schedule_render(self, _event=None):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(60, self.render_image)

    def render_image(self):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
            self.resize_job = None
        if self.image is None:
            return
        width = max(1, self.canvas.winfo_width() - 32)
        height = max(1, self.canvas.winfo_height() - 32)
        scale = min(width / self.item.width, height / self.item.height)
        size = (max(1, round(self.item.width * scale)), max(1, round(self.item.height * scale)))
        self.scale_x = size[0] / self.item.width
        self.scale_y = size[1] / self.item.height
        self.offset_x = (self.canvas.winfo_width() - size[0]) / 2
        self.offset_y = (self.canvas.winfo_height() - size[1]) / 2
        display = self.image.resize(size, Image.Resampling.LANCZOS)
        # A checkerboard makes transparent padding visible without changing it.
        from PIL import ImageDraw
        background = Image.new("RGBA", size, (76, 81, 91, 255))
        draw = ImageDraw.Draw(background)
        for y in range(0, size[1], 16):
            for x in range(0, size[0], 16):
                if (x // 16 + y // 16) % 2:
                    draw.rectangle((x, y, x + 15, y + 15), fill=(91, 97, 107, 255))
        background.alpha_composite(display)
        self.photo = self.ImageTk.PhotoImage(background.convert("RGB"))
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo, anchor="nw")
        self.refresh_box()

    def screen_bounds(self):
        left, top, right, bottom = self.box.pil_bounds(self.item.height)
        return (
            self.offset_x + left * self.scale_x,
            self.offset_y + top * self.scale_y,
            self.offset_x + right * self.scale_x,
            self.offset_y + bottom * self.scale_y,
        )

    def image_point(self, event):
        return (
            round((event.x - self.offset_x) / self.scale_x),
            round((event.y - self.offset_y) / self.scale_y),
        )

    def refresh_box(self):
        pending_coordinates = self.pending_coordinates()
        self.heading.set(self.tr(
            "heading", index=self.index + 1, total=len(self.records), saved=len(self.store.boxes),
            filename=self.item.path.name, scale=self.item.scale, pose=self.item.pose,
            outfit=self.item.outfit, expression=self.item.expression,
            width=self.item.width, height=self.item.height,
            angle=angle_text(self.item.angle),
            source_width=getattr(self.item, "source_width", self.item.width),
            source_height=getattr(self.item, "source_height", self.item.height),
        ))
        counts = {angle: [0, 0] for angle in self.angles}
        for item in self.records:
            counts[item.angle][1] += 1
            counts[item.angle][0] += item.key in self.store.boxes
        self.angle_progress.set(self.tr("angle_progress", progress="   |   ".join(
            f"{angle_text(angle)}°: {saved}/{total}"
            for angle, (saved, total) in counts.items()
        )))
        self.canvas.delete("box")
        valid = self.box is not None and self.box.fits(self.item.width, self.item.height)
        saved = self.box is not None and self.box == self.store.boxes.get(self.item.key)
        if self.box is None:
            values = ("", "", "")
            self.box_status.set(self.tr("box_empty"))
        else:
            values = (self.box.x_left, self.box.y_bottom, self.box.side)
            bounds = self.screen_bounds()
            color = "#69df9d" if saved else "#ffd166" if valid else "#ff6b6b"
            self.canvas.create_rectangle(*bounds, outline=color, width=2, tags="box")
            right, bottom = bounds[2:]
            self.canvas.create_rectangle(right - 5, bottom - 5, right + 5, bottom + 5,
                                         fill=color, outline=color, tags="box")
            self.box_status.set(self.tr(
                "box_saved" if saved else "box_outside" if not valid else
                "box_dirty" if self.dirty else "box_proposal",
            ))
        if not pending_coordinates:
            for variable, value in zip(self.coordinate_vars, values):
                variable.set(str(value))
        preview = Image.new("RGBA", (240, 240), (76, 81, 91, 255))
        if valid:
            crop = self.image.crop(self.box.pil_bounds(self.item.height))
            preview.alpha_composite(crop.resize((240, 240), Image.Resampling.LANCZOS))
        self.preview_photo = self.ImageTk.PhotoImage(preview.convert("RGB"))
        self.preview.configure(image=self.preview_photo)

    def set_box(self, box):
        if box is not None:
            self.box = box
            self.dirty = box != self.store.boxes.get(self.item.key)
            self.box_origin = "manual_adjusted" if self.box_reference else "manual"
            for variable, value in zip(self.coordinate_vars, (box.x_left, box.y_bottom, box.side)):
                variable.set(str(value))
            self.refresh_box()

    def pending_coordinates(self):
        values = ("", "", "") if self.box is None else tuple(
            str(value) for value in (self.box.x_left, self.box.y_bottom, self.box.side)
        )
        return tuple(variable.get() for variable in self.coordinate_vars) != values

    def start_drag(self, event):
        self.canvas.focus_set()
        point = self.image_point(event)
        if not (0 <= point[0] <= self.item.width and 0 <= point[1] <= self.item.height):
            return
        mode = "draw"
        if self.box is not None and not event.state & 0x0001:
            left, top, right, bottom = self.screen_bounds()
            if abs(event.x - right) <= 9 and abs(event.y - bottom) <= 9:
                mode = "resize"
            elif left <= event.x <= right and top <= event.y <= bottom:
                mode = "move"
        self.drag = (mode, point, self.box)

    def drag_motion(self, event):
        if self.drag is None:
            return
        mode, start, initial = self.drag
        point = self.image_point(event)
        if mode == "move":
            box = initial.moved(point[0] - start[0], start[1] - point[1],
                                self.item.width, self.item.height)
        elif mode == "resize":
            left, top, _, _ = initial.pil_bounds(self.item.height)
            side = max(1, min(max(point[0] - left, point[1] - top),
                              self.item.width - left, self.item.height - top))
            box = SquareBox(left, self.item.height - top - side, side)
        else:
            box = square_from_drag(start, point, self.item.width, self.item.height)
        self.set_box(box)

    def end_drag(self, event):
        self.drag_motion(event)
        self.drag = None

    def apply_coordinates(self):
        try:
            try:
                box = SquareBox(*(int(variable.get()) for variable in self.coordinate_vars))
            except ValueError as exc:
                raise AnnotationError("integer_required") from exc
            if not box.fits(self.item.width, self.item.height):
                raise AnnotationError("box_bounds")
        except ValueError as exc:
            self.messagebox.showerror(
                self.tr("coordinates_error_title"), error_text(exc, self.language), parent=self.root,
            )
            return False
        self.set_box(box)
        self.canvas.focus_set()
        return True

    def change_size(self, delta):
        if self.box is not None:
            self.set_box(self.box.resized(delta, self.item.width, self.item.height))
        return "break"

    def wheel(self, event):
        if event.delta:
            step = 10 if event.state & 0x0001 else 1
            self.change_size(step if event.delta > 0 else -step)
        return "break"

    def keypress(self, event):
        if event.state & 0x0004 and event.keysym.lower() == "s":
            self.save_current()
            return "break"
        if event.widget.winfo_class() in {"TEntry", "Entry", "TSpinbox", "TCombobox"}:
            return
        step = 10 if event.state & 0x0001 else 1
        if event.keysym in {"Return", "space"}:
            self.confirm_next()
        elif event.keysym in {"plus", "equal", "KP_Add"}:
            self.change_size(step)
        elif event.keysym in {"minus", "underscore", "KP_Subtract"}:
            self.change_size(-step)
        elif event.keysym in {"Left", "Right", "Up", "Down"} and self.box is not None:
            dx, dy = {"Left": (-step, 0), "Right": (step, 0),
                      "Up": (0, step), "Down": (0, -step)}[event.keysym]
            self.set_box(self.box.moved(dx, dy, self.item.width, self.item.height))
        elif event.keysym in {"Prior", "Next"}:
            self.navigate(-1 if event.keysym == "Prior" else 1)
        else:
            return
        return "break"

    def save_current(self):
        if self.pending_coordinates() and not self.apply_coordinates():
            return False
        if self.box is None:
            self.messagebox.showinfo(self.tr("draw_first_title"), self.tr("draw_first"), parent=self.root)
            return False
        try:
            self.store.save(self.item, self.box, self.box_origin, self.box_reference)
        except (OSError, ValueError) as exc:
            self.messagebox.showerror(
                self.tr("save_error"), self.tr("save_error_detail", error=error_text(exc, self.language)),
                parent=self.root,
            )
            return False
        self.dirty = False
        self.set_status("status_saved", filename=self.item.path.name, path=self.store.path,
                        angle=angle_text(self.item.angle))
        self.refresh_box()
        return True

    def allow_leave(self):
        if not self.dirty and not self.pending_coordinates():
            return True
        answer = self.messagebox.askyesnocancel(
            self.tr("unsaved_title"), self.tr("unsaved_prompt"),
            parent=self.root,
        )
        return answer is False or (answer is True and self.save_current())

    def navigate(self, delta):
        indices = range(self.index + delta, len(self.records) if delta > 0 else -1, delta)
        target = next((i for i in indices if not self.unconfirmed_only.get()
                       or self.records[i].key not in self.store.boxes), None)
        if target is not None and self.allow_leave():
            self.load_image(target)

    def jump_angle(self, _event=None):
        selected = float(self.angle_variable.get())
        target = next((i for i, item in enumerate(self.records)
                       if item.angle == selected and (not self.unconfirmed_only.get()
                       or item.key not in self.store.boxes)), None)
        if target is not None and self.allow_leave():
            self.load_image(target)
        elif target is None:
            self.set_status("angle_complete", angle=angle_text(selected))
        self.angle_variable.set(angle_text(self.item.angle))

    def filter_unconfirmed(self):
        if self.unconfirmed_only.get() and self.item.key in self.store.boxes:
            if not self.allow_leave():
                self.unconfirmed_only.set(False)
                return
            self.next_unannotated()

    def confirm_next(self):
        if self.save_current():
            target = next((i for i in range(self.index + 1, len(self.records))
                           if not self.unconfirmed_only.get()
                           or self.records[i].key not in self.store.boxes), None)
            if target is not None:
                self.load_image(target)
            else:
                remaining = len(self.records) - len(self.store.boxes)
                self.set_status(
                    "status_remaining" if remaining else "status_complete",
                    remaining=remaining, total=len(self.records), path=self.store.path,
                )

    def next_unannotated(self):
        if not self.allow_leave():
            return
        for offset in range(1, len(self.records) + 1):
            index = (self.index + offset) % len(self.records)
            if self.records[index].key not in self.store.boxes:
                self.load_image(index)
                return
        self.set_status("status_review")

    def close(self):
        if self.allow_leave():
            if self.resize_job is not None:
                self.root.after_cancel(self.resize_job)
                self.resize_job = None
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Annotate square face boxes in a CSV without modifying images.")
    parser.add_argument("--train_dir", default="atridataset/train", help="image directory with six-field filenames")
    parser.add_argument("--scale", choices=SCALE_CODES, default="l", help="reference resolution to annotate (default: l)")
    parser.add_argument("--angles", type=float, nargs="+", default=DEFAULT_ANGLES,
                        help="distinct angles in degrees; annotation order is ascending")
    parser.add_argument("--output", help="CSV path (default: annotations/face_boxes_<scale>_rotated.csv); resume if it exists")
    parser.add_argument("--reference_annotations", help="old 0-degree or rotated CSV to use as unconfirmed proposals")
    parser.add_argument("--language", choices=tuple(LANGUAGES.values()), default="en",
                        help="initial GUI language; can also be changed in the window (default: en)")
    args = parser.parse_args()
    try:
        records = expand_views(scan_images(args.train_dir, args.scale), args.angles)
        output = args.output or f"annotations/face_boxes_{args.scale}_rotated.csv"
        store = RotatedAnnotationStore(output, records)
        if args.reference_annotations:
            store.add_reference(args.reference_annotations, scan_images(args.train_dir, "all"))
    except (OSError, ValueError, UnicodeError, csv.Error, KeyError, TypeError) as exc:
        parser.exit(1, translate(args.language, "startup_error", error=error_text(exc, args.language)) + "\n")
    import tkinter as tk
    configure_taskbar("FaceAnnotator")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        parser.exit(1, translate(args.language, "tk_error", error=exc) + "\n")
    set_app_icon(root)
    FaceAnnotator(root, records, store, language=args.language)
    root.mainloop()


if __name__ == "__main__":
    main()
