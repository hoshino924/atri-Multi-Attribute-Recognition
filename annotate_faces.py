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

from labels import EXPR_CODES, OUTFIT_CODES, POSE_CODES


SCALE_CODES = ("s", "w", "m", "l", "ll")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CSV_FIELDS = (
    "filename", "character", "shoe_variant", "scale", "outfit", "pose",
    "expression", "image_width", "image_height", "coordinate_system",
    "x_left", "y_bottom", "side",
)
COORDINATE_SYSTEM = "bottom_left_pixels"


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


@dataclass(frozen=True)
class SquareBox:
    """Original-image pixels, measured from the left and bottom image edges."""

    x_left: int
    y_bottom: int
    side: int

    def fits(self, width, height):
        return (
            self.side > 0
            and self.x_left >= 0
            and self.y_bottom >= 0
            and self.x_left + self.side <= width
            and self.y_bottom + self.side <= height
        )

    def pil_bounds(self, height):
        return (
            self.x_left,
            height - self.y_bottom - self.side,
            self.x_left + self.side,
            height - self.y_bottom,
        )

    def moved(self, dx, dy, width, height):
        return SquareBox(
            max(0, min(width - self.side, self.x_left + dx)),
            max(0, min(height - self.side, self.y_bottom + dy)),
            self.side,
        )

    def resized(self, delta, width, height):
        side = max(1, min(width, height, self.side + delta))
        shift = (self.side - side) // 2
        return SquareBox(
            self.x_left + shift, self.y_bottom + shift, side,
        ).moved(0, 0, width, height)


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
    if scale not in SCALE_CODES:
        raise ValueError(f"不支持的分辨率档位：{scale}")
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"图片目录不存在：{root}")
    records = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = path.stem.split("_")
        if len(parts) != 6:
            raise ValueError(f"文件名必须包含六个字段：{path.name}")
        character, variant, image_scale, outfit, pose, expression = parts
        if image_scale != scale:
            continue
        if (
            not character or not variant or outfit not in OUTFIT_CODES
            or pose not in POSE_CODES or expression not in EXPR_CODES
        ):
            raise ValueError(f"文件名标签无效：{path.name}")
        with Image.open(path) as source:
            width, height = source.size
        records.append(AnnotationImage(
            path, character, variant, image_scale, outfit, pose, expression,
            width, height,
        ))
    if not records:
        raise ValueError(f"目录中没有 {scale} 档图片：{root}")
    return sorted(records, key=lambda item: (
        item.character, POSE_CODES.index(item.pose),
        OUTFIT_CODES.index(item.outfit), item.shoe_variant,
        EXPR_CODES.index(item.expression), item.path.name,
    ))


class AnnotationStore:
    """Validated, resumable CSV with atomic saves and external-change checks."""

    def __init__(self, path, records):
        self.path = Path(path).resolve()
        if self.path.suffix.lower() != ".csv":
            raise ValueError("标注输出必须是 .csv 文件")
        self.records = {item.path.name: item for item in records}
        if self.path in {item.path.resolve() for item in records}:
            raise ValueError("标注输出不能是源图片")
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
            raise ValueError("已有 CSV 的表头不属于此标注工具，请指定新的输出文件")
        for line, row in enumerate(reader, start=2):
            try:
                filename = row["filename"]
                if filename not in self.records or filename in self.boxes:
                    raise ValueError("图片不在所选目录/分辨率中，或文件名重复")
                box = SquareBox(*(int(row[key]) for key in (
                    "x_left", "y_bottom", "side",
                )))
                item = self.records[filename]
                if not box.fits(item.width, item.height):
                    raise ValueError("方框超出原图边界或边长无效")
                expected = {key: str(value) for key, value in self._row(filename, box).items()}
                if row != expected:
                    raise ValueError("图片尺寸、标签或坐标格式与已有标注不一致")
                self.boxes[filename] = box
                self.last_boxes[item.group] = box
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"CSV 第 {line} 行无效：{exc}") from exc

    def _check_external_change(self):
        data = self.path.read_bytes() if self.path.exists() else None
        if self._signature(data) != self.signature:
            raise ValueError("CSV 已被其他程序修改。请保留当前文件并重新打开标注工具")

    def save(self, item, box):
        if not box.fits(item.width, item.height):
            raise ValueError("请将完整正方形框放在图片范围内")
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
    def __init__(self, root, records, store):
        import tkinter as tk
        from tkinter import messagebox, ttk
        from PIL import ImageTk

        self.tk = tk
        self.messagebox = messagebox
        self.ImageTk = ImageTk
        self.root = root
        self.records = records
        self.store = store
        self.index = 0
        self.box = None
        self.dirty = False
        self.image = None
        self.drag = None
        self.resize_job = None
        self.scale_x = self.scale_y = 1.0
        self.offset_x = self.offset_y = 0

        root.title("ATRI 面部方框标注")
        root.geometry(f"1160x{max(680, min(850, root.winfo_screenheight() - 100))}")
        root.minsize(900, 680)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        self.heading = tk.StringVar()
        ttk.Label(root, textvariable=self.heading, font=("Microsoft YaHei UI", 11)).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=12, pady=6,
        )
        self.canvas = tk.Canvas(root, background="#242830", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=(12, 6))
        panel = ttk.Frame(root, padding=(10, 0, 12, 0))
        panel.grid(row=1, column=1, sticky="ns")
        ttk.Label(panel, text="面部区域预览（可包含头发）").pack(anchor="w")
        self.preview = ttk.Label(panel, anchor="center")
        self.preview.pack(pady=6)
        ttk.Label(panel, text="原图像素 · 左下角为原点").pack(anchor="w")
        coordinates = ttk.Frame(panel)
        coordinates.pack(fill="x", pady=6)
        self.coordinate_vars = [tk.StringVar() for _ in range(3)]
        for row, (caption, variable) in enumerate(zip(
            ("左边距 x", "下边距 y", "正方形边长"), self.coordinate_vars,
        )):
            ttk.Label(coordinates, text=caption).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(coordinates, textvariable=variable, width=14).grid(
                row=row, column=1, padx=(15, 0), pady=3,
            )
        ttk.Button(panel, text="应用输入坐标", command=self.apply_coordinates).pack(fill="x")
        self.box_status = tk.StringVar()
        ttk.Label(panel, textvariable=self.box_status, wraplength=260).pack(
            fill="x", pady=6,
        )
        self.help_text = (
            "左键拖动：绘制正方形\n"
            "框内拖动：移动\n"
            "右下角手柄：调整边长\n"
            "Shift + 拖动：重新画框\n"
            "方向键：移动 1 原图像素\n"
            "+ / - 或滚轮：调整边长\n"
            "按住 Shift：键盘步长为 10\n"
            "Enter / 空格：确认并下一张\n"
            "Ctrl+S：确认并保存当前图片\n"
            "PageUp / PageDown：前后浏览"
        )
        ttk.Label(panel, text=(
            "黄色：待确认；绿色：已保存。\n"
            "每张图片需单独确认，继承框不会自动标注。\n"
            "点击“快捷键”查看绘制与微调操作。"
        ), wraplength=260).pack(anchor="w", pady=6)
        controls = ttk.Frame(root)
        controls.grid(row=2, column=0, columnspan=2, pady=10)
        for column, (caption, callback) in enumerate((
            ("上一张", lambda: self.navigate(-1)),
            ("确认并下一张", self.confirm_next),
            ("下一张 / 暂不确认", lambda: self.navigate(1)),
            ("下一张未标注", self.next_unannotated),
            ("快捷键", lambda: self.messagebox.showinfo("操作快捷键", self.help_text, parent=root)),
        )):
            ttk.Button(controls, text=caption, command=callback).grid(row=0, column=column, padx=4)
        self.status = tk.StringVar(value=f"标注表格：{store.path}")
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
        first = next((i for i, item in enumerate(records) if item.path.name not in store.boxes), 0)
        self.load_image(first)
        if self.image is None:
            root.destroy()

    @property
    def item(self):
        return self.records[self.index]

    def load_image(self, index):
        item = self.records[index]
        try:
            with Image.open(item.path) as source:
                if source.size != (item.width, item.height):
                    raise ValueError("图片尺寸在启动后发生了变化，请重新打开工具")
                image = source.convert("RGBA")
        except (OSError, ValueError) as exc:
            self.messagebox.showerror("无法打开图片", str(exc), parent=self.root)
            return
        self.index = index
        self.image = image
        self.box = self.store.boxes.get(item.path.name, self.store.last_boxes.get(item.group))
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
        self.heading.set(
            f"{self.index + 1} / {len(self.records)}    已标注 {len(self.store.boxes)} 张    "
            f"{self.item.path.name}\n"
            f"分辨率：{self.item.scale}    姿势：{self.item.pose}    "
            f"服装：{self.item.outfit}    表情：{self.item.expression}    "
            f"原图：{self.item.width} × {self.item.height}"
        )
        self.canvas.delete("box")
        valid = self.box is not None and self.box.fits(self.item.width, self.item.height)
        saved = self.box is not None and self.box == self.store.boxes.get(self.item.path.name)
        if self.box is None:
            values = ("", "", "")
            self.box_status.set("尚无方框，请拖动鼠标绘制。")
        else:
            values = (self.box.x_left, self.box.y_bottom, self.box.side)
            bounds = self.screen_bounds()
            color = "#69df9d" if saved else "#ffd166" if valid else "#ff6b6b"
            self.canvas.create_rectangle(*bounds, outline=color, width=2, tags="box")
            right, bottom = bounds[2:]
            self.canvas.create_rectangle(right - 5, bottom - 5, right + 5, bottom + 5,
                                         fill=color, outline=color, tags="box")
            self.box_status.set(
                "已确认并保存。" if saved else
                "方框超出图片边界，请调整后确认。" if not valid else
                "已调整，尚未保存。" if self.dirty else
                "沿用同姿势上次确认的方框，请检查并确认。"
            )
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
            self.dirty = box != self.store.boxes.get(self.item.path.name)
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
            box = SquareBox(*(int(variable.get()) for variable in self.coordinate_vars))
            if not box.fits(self.item.width, self.item.height):
                raise ValueError("框必须为图片范围内的正方形，边长至少为 1 像素")
        except ValueError as exc:
            self.messagebox.showerror("坐标无效", str(exc), parent=self.root)
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
            self.messagebox.showinfo("请先画框", "请先绘制面部正方形框。", parent=self.root)
            return False
        try:
            self.store.save(self.item, self.box)
        except (OSError, ValueError) as exc:
            self.messagebox.showerror(
                "保存失败", f"{exc}\n\n若 CSV 正在 Excel 中打开，请关闭表格后重试。",
                parent=self.root,
            )
            return False
        self.dirty = False
        self.status.set(f"已保存 {self.item.path.name} → {self.store.path}")
        self.refresh_box()
        return True

    def allow_leave(self):
        if not self.dirty and not self.pending_coordinates():
            return True
        answer = self.messagebox.askyesnocancel(
            "当前方框尚未保存", "是否先确认并保存当前方框？\n选择“否”放弃本次调整。",
            parent=self.root,
        )
        return answer is False or (answer is True and self.save_current())

    def navigate(self, delta):
        target = self.index + delta
        if 0 <= target < len(self.records) and self.allow_leave():
            self.load_image(target)

    def confirm_next(self):
        if self.save_current():
            if self.index + 1 < len(self.records):
                self.load_image(self.index + 1)
            else:
                remaining = len(self.records) - len(self.store.boxes)
                self.status.set(
                    f"已到最后一张；还有 {remaining} 张未标注。可点击“下一张未标注”。"
                    if remaining else f"全部 {len(self.records)} 张已确认，CSV：{self.store.path}"
                )

    def next_unannotated(self):
        if not self.allow_leave():
            return
        for offset in range(1, len(self.records) + 1):
            index = (self.index + offset) % len(self.records)
            if self.records[index].path.name not in self.store.boxes:
                self.load_image(index)
                return
        self.status.set("所有图片均已标注，可以继续浏览和修改已有方框。")

    def close(self):
        if self.allow_leave():
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="逐图标注面部正方形框，仅保存 CSV，不修改图片。")
    parser.add_argument("--train_dir", default="atridataset/train", help="标准六字段文件名的图片目录")
    parser.add_argument("--scale", choices=SCALE_CODES, default="w", help="筛选分辨率档位，默认 w")
    parser.add_argument("--output", help="CSV 路径，默认 annotations/face_boxes_<scale>.csv；存在时续标")
    args = parser.parse_args()
    try:
        records = scan_images(args.train_dir, args.scale)
        output = args.output or f"annotations/face_boxes_{args.scale}.csv"
        store = AnnotationStore(output, records)
    except (OSError, ValueError, UnicodeError, csv.Error) as exc:
        parser.exit(1, f"无法载入标注任务：{exc}\n")
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        parser.exit(1, f"无法启动 Tk 窗口，请使用带完整 Tcl/Tk 的 Python 和图形桌面环境：{exc}\n")
    FaceAnnotator(root, records, store)
    root.mainloop()


if __name__ == "__main__":
    main()
