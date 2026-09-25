# -*- coding: utf-8 -*-
"""Read-only GUI for inspecting the exact crop used by a classifier checkpoint."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import queue
import threading

from PIL import Image
import torch

from dataset import ForegroundRegionCrop, SCALE_CODES, parse_filename
from face_locator import FaceNotFoundError
from app_assets import DEFAULT_CLASSIFIER_WEIGHT, DEFAULT_LOCATOR_WEIGHT, configure_taskbar, set_app_icon
from infer import build_preview_transforms, collect_input_paths, load_model, predict


LANGUAGES = {"English": "en", "简体中文": "zh"}
TEXTS = {
    "en": {
        "title": "ATRI Crop Inspector",
        "classifier": "Classifier checkpoint",
        "locator": "Locator checkpoint (face-crop models)",
        "input": "Image folder / image",
        "browse": "Browse...",
        "folder": "Folder...",
        "file": "Image...",
        "load": "Load / Start",
        "cpu": "CPU only",
        "recursive": "Include subfolders",
        "scale": "Resolution",
        "all": "All",
        "filter": "Filename contains",
        "apply": "Apply filter",
        "previous": "Previous",
        "next": "Next",
        "go": "Go to",
        "native": "Display at 100% (use scrollbars)",
        "original": "Original image + crop rectangle",
        "crop": "Raw crop (before resizing / padding)",
        "input_view": "Expression input (before normalization)",
        "empty": "No image",
        "hint": "Select checkpoints and images, then click Load / Start.",
        "loading": "Loading checkpoints and image list...",
        "working": "Inspecting {name}...",
        "no_matches": "No images match the current filters.",
        "failed": "Operation failed: {error}",
        "error_title": "Crop inspection",
        "required": "Select a classifier checkpoint and an image folder or image.",
        "jump_error": "Enter an image number from 1 to {count}.",
        "counter": "{index} / {count}",
        "model_info": "{name} | epoch {epoch} | {mode} | {head} | expression {size} x {size} | {device}",
        "head_fusion": "Expression: full + face",
        "head_face_only": "Expression: face only",
        "face_mode": "Face locator",
        "legacy_mode": "Legacy foreground crop (locator unused)",
        "file_info": "{name} | original {width} x {height}",
        "box_info": "Bottom-left origin: x={x}, y={y}, width={width}, height={height}",
        "locator_score": "Locator confidence: {score:.2%}",
        "prediction": "Expression: {code} ({label}), {confidence:.2%} | {decision} | threshold {threshold}",
        "accepted": "Accepted",
        "rejected": "Low confidence",
        "none": "none",
        "candidates": "Top candidates: {values}",
        "expected": "Expected expression from filename: {code} | {correct}",
        "correct": "Correct",
        "incorrect": "Incorrect",
        "no_face": "No valid face found. No fallback crop is displayed.",
        "read_error": "Cannot read this image.",
        "inference_error": "Inference or crop preparation failed.",
        "view_info": "{width} x {height} px | display {zoom:.0%}",
        "legend": "Green rectangle: actual crop. Display zoom does not change model input. Left/Right or Page Up/Down: previous/next. Images and annotations are never saved or modified.",
    },
    "zh": {
        "title": "ATRI 裁剪检查工具",
        "classifier": "分类模型权重",
        "locator": "定位器权重（面部裁剪模型使用）",
        "input": "图片目录 / 图片",
        "browse": "浏览…",
        "folder": "目录…",
        "file": "图片…",
        "load": "加载 / 开始",
        "cpu": "仅使用 CPU",
        "recursive": "包含子目录",
        "scale": "分辨率档位",
        "all": "全部",
        "filter": "文件名包含",
        "apply": "应用筛选",
        "previous": "上一张",
        "next": "下一张",
        "go": "跳转",
        "native": "按 100% 显示（可滚动查看）",
        "original": "原图与实际裁剪框",
        "crop": "原始裁剪区域（缩放、补边前）",
        "input_view": "表情模型输入（归一化前）",
        "empty": "暂无图片",
        "hint": "选择权重和图片，点击“加载 / 开始”。",
        "loading": "正在加载权重和图片列表…",
        "working": "正在检查 {name}…",
        "no_matches": "没有符合当前筛选条件的图片。",
        "failed": "操作失败：{error}",
        "error_title": "裁剪检查",
        "required": "请选择分类模型权重，以及图片目录或图片。",
        "jump_error": "请输入 1 到 {count} 之间的图片编号。",
        "counter": "{index} / {count}",
        "model_info": "{name} | 第 {epoch} 轮 | {mode} | {head} | 表情输入 {size} x {size} | {device}",
        "head_fusion": "表情：整图 + 面部",
        "head_face_only": "表情：仅面部",
        "face_mode": "面部定位器",
        "legacy_mode": "旧版前景裁剪（不使用定位器）",
        "file_info": "{name} | 原图 {width} x {height}",
        "box_info": "左下角原点：x={x}，y={y}，宽={width}，高={height}",
        "locator_score": "定位置信度：{score:.2%}",
        "prediction": "表情：{code}（{label}），{confidence:.2%} | {decision} | 阈值 {threshold}",
        "accepted": "已接受",
        "rejected": "低置信度",
        "none": "无",
        "candidates": "候选：{values}",
        "expected": "文件名中的表情标签：{code} | {correct}",
        "correct": "预测正确",
        "incorrect": "预测错误",
        "no_face": "未定位到有效面部，不显示替代裁剪区域。",
        "read_error": "无法读取这张图片。",
        "inference_error": "推理或裁剪准备失败。",
        "view_info": "{width} x {height} 像素 | 显示比例 {zoom:.0%}",
        "legend": "绿色框表示实际裁剪范围。显示缩放不会改变模型输入。左右方向键或 Page Up/Down 切换图片。工具不会保存或修改图片和标注。",
    },
}


@dataclass
class Inspection:
    path: Path
    image: Image.Image | None = None
    bounds: tuple | None = None
    crop: Image.Image | None = None
    model_input: Image.Image | None = None
    predictions: dict | None = None
    locator_confidence: float | None = None
    status: str = "ok"
    error: str = ""


def inspect_image(model, transforms, label_codes, device, path):
    """Use normal inference, then reuse its exact localized box and preview."""
    path = Path(path)
    result = Inspection(path)
    try:
        image, predictions = predict(model, path, device, transforms, label_codes, top_k=3)
        previews = build_preview_transforms(model)
        expression_input = previews["expression"](image)
        if hasattr(model, "face_preview"):
            bounds = model.face_preview.last_box.pil_bounds(image.height)
            confidence = model.face_preview.last_confidence
        else:
            config = model.preprocess_config["expression"]
            bounds = ForegroundRegionCrop(config["width_fraction"], config["height_fraction"]).bounds(image)
            confidence = None
        result.image, result.bounds = image, bounds
        result.crop, result.model_input = image.crop(bounds), expression_input
        result.predictions, result.locator_confidence = predictions, confidence
    except Exception as exc:
        # Keep the failed original inspectable, never reuse the preceding box.
        result = Inspection(path, status="no_face" if isinstance(exc, FaceNotFoundError) else "inference_error", error=str(exc))
        try:
            with Image.open(path) as source:
                result.image = source.convert("RGBA")
        except OSError as read_error:
            result.status, result.error = "read_error", str(read_error)
    return result


def filter_paths(paths, scale="all", text=""):
    """Keep resolution variants adjacent, while allowing arbitrary image names."""
    selected = []
    for path in paths:
        if text.casefold() not in path.name.casefold():
            continue
        try:
            record = parse_filename(path)
        except ValueError:
            if scale == "all":
                selected.append(((1, path.name.casefold(), str(path)), path))
            continue
        if scale == "all" or record.scale == scale:
            selected.append(((0, *record.content_key, SCALE_CODES.index(record.scale), str(path)), path))
    return [path for _, path in sorted(selected, key=lambda item: item[0])]


def display_geometry(image_size, viewport, native=False):
    """Return exact rendered dimensions/offsets, accounting for integer rounding."""
    width, height = image_size
    view_width, view_height = viewport
    factor = 1.0 if native else min(max(1, view_width - 16) / width, max(1, view_height - 16) / height)
    rendered = max(1, round(width * factor)), max(1, round(height * factor))
    offset = max(8, (view_width - rendered[0]) // 2), max(8, (view_height - rendered[1]) // 2)
    return rendered, offset


def screen_bounds(bounds, image_size, rendered, offset):
    sx, sy = rendered[0] / image_size[0], rendered[1] / image_size[1]
    left, top, right, bottom = bounds
    return offset[0] + left * sx, offset[1] + top * sy, offset[0] + right * sx, offset[1] + bottom * sy


class ImagePanel:
    def __init__(self, app, parent, column, title):
        tk, ttk = app.tk, app.ttk
        self.app, self.image, self.bounds, self.photo = app, None, None, None
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=column, sticky="nsew", padx=4)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        app.localize(ttk.Label(frame), title).grid(row=0, column=0, columnspan=2, sticky="w", pady=4)
        self.canvas = tk.Canvas(frame, background="#242830", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=self.canvas.xview)
        vertical.grid(row=1, column=1, sticky="ns")
        horizontal.grid(row=2, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.details = tk.StringVar()
        ttk.Label(frame, textvariable=self.details).grid(row=3, column=0, columnspan=2, sticky="w", pady=4)
        self.canvas.bind("<Configure>", lambda _event: app.schedule_render())
        self.canvas.bind("<Button-1>", lambda _event: self.canvas.focus_set())

    def set_image(self, image, bounds=None):
        self.image, self.bounds = image, bounds
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.render()

    def render(self):
        canvas = self.canvas
        canvas.delete("all")
        self.photo = None
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        if self.image is None:
            canvas.configure(scrollregion=(0, 0, width, height))
            canvas.create_text(width / 2, height / 2, text=self.app.tr("empty"), fill="#cbd1db")
            self.details.set("")
            return
        rendered, offset = display_geometry(self.image.size, (width, height), self.app.native.get())
        preview = self.image if rendered == self.image.size else self.image.resize(rendered, Image.Resampling.LANCZOS)
        self.photo = self.app.ImageTk.PhotoImage(preview)
        canvas.create_image(*offset, image=self.photo, anchor="nw")
        if self.bounds is not None:
            canvas.create_rectangle(*screen_bounds(self.bounds, self.image.size, rendered, offset),
                                    outline="#4cff89", width=2)
        canvas.configure(scrollregion=(0, 0, max(width, rendered[0] + offset[0] + 8),
                                       max(height, rendered[1] + offset[1] + 8)))
        self.details.set(self.app.tr("view_info", width=self.image.width, height=self.image.height,
                                     zoom=rendered[0] / self.image.width))


class CropInspector:
    def __init__(self, root, args):
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from PIL import ImageTk

        self.root, self.tk, self.ttk = root, tk, ttk
        self.filedialog, self.messagebox, self.ImageTk = filedialog, messagebox, ImageTk
        self.language = args.language
        self.text_widgets, self.editable, self.action_widgets = [], [], []
        self.session, self.inspection = None, None
        self.all_paths, self.paths, self.index = [], [], 0
        self.busy, self.closed = False, False
        self.render_job, self.poll_job = None, None
        self.status_key, self.status_values = "hint", {}
        root.geometry(f"{min(1480, root.winfo_screenwidth() - 60)}x{min(900, root.winfo_screenheight() - 100)}")
        root.minsize(1000, 660)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.variables = {name: tk.StringVar(value=value or "") for name, value in
                          (("classifier", args.weight), ("locator", args.locator_weight), ("input", args.input))}
        self.cpu, self.recursive, self.native = tk.BooleanVar(value=args.cpu), tk.BooleanVar(value=args.recursive), tk.BooleanVar()
        config = ttk.Frame(root, padding=(12, 8))
        config.grid(row=0, column=0, sticky="ew")
        config.columnconfigure(1, weight=1)
        for row, name in enumerate(self.variables):
            self.localize(ttk.Label(config), name).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            entry = ttk.Entry(config, textvariable=self.variables[name])
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            self.editable.append(entry)
            button = self.localize(ttk.Button(config, command=lambda name=name: self.browse(name)),
                                   "folder" if name == "input" else "browse")
            button.grid(row=row, column=2, padx=4)
            self.action_widgets.append(button)
        image_button = self.localize(ttk.Button(config, command=lambda: self.browse("input", image=True)), "file")
        image_button.grid(row=2, column=3)
        self.action_widgets.append(image_button)
        options = ttk.Frame(config)
        options.grid(row=0, column=3, rowspan=2, padx=6, sticky="n")
        for variable, key in ((self.cpu, "cpu"), (self.recursive, "recursive")):
            check = self.localize(ttk.Checkbutton(options, variable=variable, command=self.invalidate), key)
            check.pack(anchor="w")
            self.action_widgets.append(check)
        self.load_button = self.localize(ttk.Button(config, command=self.load), "load")
        self.load_button.grid(row=0, column=4, rowspan=2, padx=8)
        language = ttk.Frame(config)
        language.grid(row=2, column=4, padx=8)
        ttk.Label(language, text="Language / 语言").pack(side="left")
        self.language_var = tk.StringVar(value=next(label for label, code in LANGUAGES.items() if code == args.language))
        language_choice = ttk.Combobox(language, textvariable=self.language_var, values=tuple(LANGUAGES), state="readonly", width=10)
        language_choice.pack(side="left", padx=4)
        language_choice.bind("<<ComboboxSelected>>", lambda _event: self.set_language())

        filters = ttk.Frame(root, padding=(12, 2))
        filters.grid(row=1, column=0, sticky="ew")
        filters.columnconfigure(3, weight=1)
        self.localize(ttk.Label(filters), "scale").grid(row=0, column=0, padx=(0, 6))
        self.scale_var = tk.StringVar(value=args.scale if args.scale != "all" else self.tr("all"))
        self.scale_choice = ttk.Combobox(filters, textvariable=self.scale_var, state="readonly", width=7)
        self.scale_choice.grid(row=0, column=1)
        self.localize(ttk.Label(filters), "filter").grid(row=0, column=2, padx=8)
        self.filter_var = tk.StringVar(value=args.filter)
        search = ttk.Entry(filters, textvariable=self.filter_var)
        search.grid(row=0, column=3, sticky="ew")
        search.bind("<Return>", lambda _event: self.apply_filter())
        self.editable.append(search)
        self.apply_button = self.localize(ttk.Button(filters, command=self.apply_filter), "apply")
        self.apply_button.grid(row=0, column=4, padx=8)
        self.localize(ttk.Checkbutton(filters, variable=self.native, command=self.schedule_render), "native").grid(row=0, column=5)
        self.model_info = tk.StringVar()
        ttk.Label(root, textvariable=self.model_info, padding=(12, 5)).grid(row=2, column=0, sticky="ew")
        views = ttk.Frame(root, padding=(8, 0))
        views.grid(row=3, column=0, sticky="nsew")
        views.rowconfigure(0, weight=1)
        for column in range(3):
            views.columnconfigure(column, weight=2 if column == 0 else 1, uniform="views")
        self.panels = [ImagePanel(self, views, i, key) for i, key in enumerate(("original", "crop", "input_view"))]

        footer = ttk.Frame(root, padding=(12, 6))
        footer.grid(row=4, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self.details = tk.Text(footer, height=5, wrap="word", relief="flat", state="disabled")
        self.details.grid(row=0, column=0, sticky="ew")
        self.status = tk.StringVar()
        ttk.Label(footer, textvariable=self.status).grid(row=1, column=0, sticky="w", pady=4)
        navigation = ttk.Frame(footer)
        navigation.grid(row=2, column=0, sticky="w")
        self.previous_button = self.localize(ttk.Button(navigation, command=lambda: self.navigate(-1)), "previous")
        self.previous_button.pack(side="left")
        self.counter = tk.StringVar()
        ttk.Label(navigation, textvariable=self.counter, width=16, anchor="center").pack(side="left")
        self.next_button = self.localize(ttk.Button(navigation, command=lambda: self.navigate(1)), "next")
        self.next_button.pack(side="left")
        self.jump_var = tk.StringVar()
        self.jump_entry = ttk.Entry(navigation, textvariable=self.jump_var, width=7)
        self.jump_entry.pack(side="left", padx=(16, 4))
        self.jump_entry.bind("<Return>", lambda _event: self.jump())
        self.jump_button = self.localize(ttk.Button(navigation, command=self.jump), "go")
        self.jump_button.pack(side="left")
        self.localize(ttk.Label(footer, wraplength=950), "legend").grid(row=3, column=0, sticky="w", pady=(6, 0))
        root.bind("<KeyPress>", self.keypress)
        for variable in self.variables.values():
            variable.trace_add("write", lambda *_args: self.invalidate())
        self.set_language()
        self.controls()
        if args.weight:
            root.after(100, self.load)

    def tr(self, key, **values):
        return TEXTS[self.language][key].format(**values)

    def localize(self, widget, key):
        self.text_widgets.append((widget, key))
        widget.configure(text=self.tr(key))
        return widget

    def set_language(self):
        scale = self.selected_scale()
        self.language = LANGUAGES[self.language_var.get()]
        self.root.title(self.tr("title"))
        for widget, key in self.text_widgets:
            widget.configure(text=self.tr(key))
        self.scale_choice.configure(values=(self.tr("all"), *SCALE_CODES))
        self.scale_var.set(self.tr("all") if scale == "all" else scale)
        self.refresh_text()
        self.schedule_render()

    def selected_scale(self):
        return self.scale_var.get() if self.scale_var.get() in SCALE_CODES else "all"

    def set_status(self, key, **values):
        self.status_key, self.status_values = key, values
        self.status.set(self.tr(key, **values) if key else "")

    def controls(self):
        ready = self.session is not None and not self.busy
        for widget in [*self.editable, *self.action_widgets, self.load_button]:
            widget.configure(state="disabled" if self.busy else "normal")
        self.scale_choice.configure(state="disabled" if self.busy else "readonly")
        self.apply_button.configure(state="normal" if ready else "disabled")
        self.previous_button.configure(state="normal" if ready and self.paths and self.index > 0 else "disabled")
        self.next_button.configure(state="normal" if ready and self.paths and self.index < len(self.paths) - 1 else "disabled")
        for widget in (self.jump_entry, self.jump_button):
            widget.configure(state="normal" if ready and self.paths else "disabled")
        self.counter.set(self.tr("counter", index=self.index + 1 if self.paths else 0, count=len(self.paths)))

    def clear_views(self):
        self.inspection = None
        for panel in self.panels:
            panel.set_image(None)
        self.refresh_text()

    def invalidate(self):
        if self.busy:
            return
        self.session, self.all_paths, self.paths, self.index = None, [], [], 0
        self.set_status("hint")
        self.clear_views()
        self.controls()

    def browse(self, name, image=False):
        if name == "input" and not image:
            path = self.filedialog.askdirectory(parent=self.root)
        else:
            filetypes = [("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")] if image else [
                ("PyTorch checkpoints", "*.pth *.pt"), ("All files", "*.*")]
            path = self.filedialog.askopenfilename(parent=self.root, filetypes=filetypes)
        if path:
            self.variables[name].set(path)

    def run_job(self, work, completed):
        """Serialize jobs; all Tk operations stay on the main thread."""
        if self.busy or self.closed:
            return
        self.busy = True
        self.controls()
        mailbox = queue.Queue()

        def worker():
            try:
                mailbox.put((True, work()))
            except Exception as exc:
                mailbox.put((False, str(exc)))

        def poll():
            self.poll_job = None
            if self.closed:
                return
            try:
                successful, value = mailbox.get_nowait()
            except queue.Empty:
                self.poll_job = self.root.after(60, poll)
                return
            self.busy = False
            if successful:
                completed(value)
            else:
                self.set_status("failed", error=value)
                self.messagebox.showerror(self.tr("error_title"), value, parent=self.root)
            self.controls()

        threading.Thread(target=worker, daemon=True).start()
        self.poll_job = self.root.after(60, poll)

    def load(self):
        if self.busy:
            return
        values = {key: variable.get().strip() for key, variable in self.variables.items()}
        if not values["classifier"] or not values["input"]:
            self.messagebox.showerror(self.tr("error_title"), self.tr("required"), parent=self.root)
            return
        cpu, recursive = self.cpu.get(), self.recursive.get()
        self.invalidate()
        self.set_status("loading")

        def work():
            paths = collect_input_paths(values["input"], recursive=recursive)
            device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
            model, transforms, labels = load_model(values["classifier"], device, values["locator"] or None)
            if hasattr(model, "face_preview") and model.face_preview.locator is None:
                raise ValueError("Select a locator checkpoint for this face-crop classifier.")
            return (model, transforms, labels, device), paths

        def completed(value):
            self.session, self.all_paths = value
            self.apply_filter()

        self.run_job(work, completed)

    def apply_filter(self):
        if self.busy or self.session is None:
            return
        previous = self.paths[self.index] if self.paths else None
        self.paths = filter_paths(self.all_paths, self.selected_scale(), self.filter_var.get().strip())
        self.index = self.paths.index(previous) if previous in self.paths else 0
        self.show_current()

    def show_current(self):
        self.clear_views()
        self.controls()
        if not self.paths:
            self.set_status("no_matches")
            return
        path, session = self.paths[self.index], self.session
        self.jump_var.set(str(self.index + 1))
        self.set_status("working", name=path.name)

        def completed(result):
            self.inspection = result
            self.panels[0].set_image(result.image, result.bounds)
            self.panels[1].set_image(result.crop)
            self.panels[2].set_image(result.model_input)
            self.set_status("" if result.status == "ok" else result.status)
            self.refresh_text()

        self.run_job(lambda: inspect_image(*session, path), completed)

    def refresh_text(self):
        lines = []
        if self.session is None:
            self.model_info.set("")
        else:
            model, _, _, device = self.session
            self.model_info.set(self.tr("model_info", name=Path(self.variables["classifier"].get()).name,
                                       epoch=model.checkpoint_metadata.get("epoch", "?"),
                                       mode=self.tr("face_mode" if hasattr(model, "face_preview") else "legacy_mode"),
                                       head=self.tr("head_" + model.expression_head),
                                       size=model.preprocess_config["expression"]["size"], device=str(device)))
        result = self.inspection
        if result is not None:
            if result.image is not None:
                lines.append(self.tr("file_info", name=result.path.name, width=result.image.width, height=result.image.height))
            else:
                lines.append(str(result.path))
            if result.bounds is not None:
                left, top, right, bottom = result.bounds
                line = self.tr("box_info", x=left, y=result.image.height - bottom, width=right - left, height=bottom - top)
                if result.locator_confidence is not None:
                    line += " | " + self.tr("locator_score", score=result.locator_confidence)
                lines.append(line)
            if result.predictions is not None:
                prediction = result.predictions["expression"]
                threshold = prediction["threshold"]
                lines.append(self.tr("prediction", code=prediction["code"], label=prediction["label"],
                                     confidence=prediction["confidence"], decision=self.tr("accepted" if prediction["accepted"] else "rejected"),
                                     threshold=self.tr("none") if threshold is None else f"{threshold:.2%}"))
                lines.append(self.tr("candidates", values=", ".join(f"{item['code']} {item['confidence']:.2%}" for item in prediction["candidates"])))
                try:
                    actual = parse_filename(result.path).expression
                    lines.append(self.tr("expected", code=actual, correct=self.tr("correct" if actual == prediction["code"] else "incorrect")))
                except ValueError:
                    pass
            if result.error:
                lines.append(result.error)
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", "\n".join(lines))
        self.details.configure(state="disabled")
        self.status.set(self.tr(self.status_key, **self.status_values) if self.status_key else "")
        self.controls()

    def navigate(self, delta):
        if self.busy or not self.paths or self.session is None:
            return
        index = max(0, min(len(self.paths) - 1, self.index + delta))
        if index != self.index:
            self.index = index
            self.show_current()

    def jump(self):
        if self.busy or not self.paths or self.session is None:
            return
        try:
            index = int(self.jump_var.get()) - 1
            if not 0 <= index < len(self.paths):
                raise ValueError()
        except ValueError:
            self.messagebox.showerror(self.tr("error_title"), self.tr("jump_error", count=len(self.paths)), parent=self.root)
            return
        self.index = index
        self.show_current()

    def keypress(self, event):
        if event.widget.winfo_class() in {"Entry", "TEntry", "TCombobox", "Text", "Spinbox", "TSpinbox"}:
            return
        if event.keysym in {"Left", "Prior", "Right", "Next"}:
            self.navigate(-1 if event.keysym in {"Left", "Prior"} else 1)
            return "break"

    def schedule_render(self):
        if self.closed:
            return
        if self.render_job is not None:
            self.root.after_cancel(self.render_job)
        self.render_job = self.root.after(80, self.render)

    def render(self):
        self.render_job = None
        for panel in self.panels:
            panel.render()

    def close(self):
        self.closed = True
        for job in (self.render_job, self.poll_job):
            if job is not None:
                self.root.after_cancel(job)
        self.root.destroy()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", default=DEFAULT_CLASSIFIER_WEIGHT,
                        help="classifier checkpoint; can also be selected in the GUI")
    parser.add_argument("--locator_weight", default=DEFAULT_LOCATOR_WEIGHT,
                        help="required for annotated-face classifiers; unused for legacy checkpoints")
    parser.add_argument("--input", default="atridataset/train", help="image folder or single image")
    parser.add_argument("--scale", choices=("all", *SCALE_CODES), default="all")
    parser.add_argument("--filter", default="", help="filename substring, for example d1_p3_fb")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    return parser.parse_args()


def main():
    import tkinter as tk

    args = parse_args()
    configure_taskbar("CropInspector")
    root = tk.Tk()
    set_app_icon(root)
    CropInspector(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
