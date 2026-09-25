# -*- coding: utf-8 -*-
"""English and Simplified Chinese text for the face annotation tool."""


LANGUAGES = {"English": "en", "简体中文": "zh"}

TEXTS = {
    "en": {
        "window_title": "ATRI Face Box Annotation",
        "preview_title": "Face preview (hair optional)",
        "coordinates_title": "Rotated canvas pixels · bottom-left origin",
        "angle_label": "Angle (counterclockwise)",
        "unconfirmed_only": "Browse unconfirmed only",
        "angle_progress": "Confirmed per angle: {progress}",
        "angle_complete": "No unconfirmed views at {angle}°. Disable the filter to review them.",
        "box_proposal": "Suggested box; check this rotated view and confirm. Not saved yet.",
        "rotation_metadata": "The rotation session metadata is missing or does not match the images, angles, or Pillow version. Use the original settings or choose a new output CSV.",
        "source_changed": "Source image changed: {filename}. Reopen the annotation tool; existing labels were not overwritten.",
        "reference_is_output": "The reference CSV and the output must be different files.",
        "reference_duplicate": "The reference CSV contains multiple resolution boxes for the same content and angle. Use one reference resolution at a time.",
        "x_left": "Left offset x",
        "y_bottom": "Bottom offset y",
        "side": "Square side",
        "apply": "Apply coordinates",
        "legend": "Yellow: pending; green: saved.\nConfirm every inherited box.\nSee Shortcuts for controls.",
        "previous": "Previous",
        "confirm_next": "Confirm & next",
        "next": "Next / skip",
        "unannotated": "Next unannotated",
        "shortcuts": "Shortcuts",
        "help_title": "Controls and shortcuts",
        "help_text": (
            "Left drag: draw a square\n"
            "Drag inside the box: move\n"
            "Drag the bottom-right handle: resize\n"
            "Shift + drag: draw a new box\n"
            "Arrow keys: move by 1 rotated-canvas pixel\n"
            "+ / - or mouse wheel: resize\n"
            "Hold Shift: use a step of 10 pixels\n"
            "Enter / Space on the canvas: confirm & next\n"
            "Ctrl+S: confirm and save the current image\n"
            "PageUp / PageDown: previous / next image"
        ),
        "heading": "{index} / {total}    Saved: {saved}    {filename}\nResolution: {scale}    Pose: {pose}    Outfit: {outfit}    Expression: {expression}    Angle: {angle}°\nSource: {source_width} × {source_height}    Rotated canvas: {width} × {height}",
        "status_output": "Annotation CSV: {path}",
        "status_saved": "Saved {filename} @ {angle}° → {path}",
        "status_remaining": "Last view reached; {remaining} views remain unconfirmed. Select 'Next unannotated' to continue.",
        "status_complete": "All {total} views confirmed. CSV: {path}",
        "status_review": "All views are confirmed. You can still review and edit their boxes.",
        "box_empty": "Drag on the image to draw a square.",
        "box_saved": "Confirmed and saved.",
        "box_outside": "Box outside image. Adjust before confirming.",
        "box_dirty": "Box adjusted; not saved yet.",
        "box_inherited": "Inherited from this pose. Check and confirm.",
        "coordinates_error_title": "Invalid coordinates",
        "integer_required": "Enter whole numbers for the left offset, bottom offset and side length.",
        "box_bounds": "Keep the entire square inside the image, with a side length of at least 1 pixel.",
        "draw_first_title": "Draw a box first",
        "draw_first": "Draw a square around the face before saving.",
        "unsaved_title": "Unsaved box",
        "unsaved_prompt": "Confirm and save the current box before leaving?\nChoose No to discard this adjustment.",
        "open_error": "Cannot open image",
        "save_error": "Save failed",
        "save_error_detail": "{error}\n\nIf the CSV is open in Excel, close it and try again.",
        "dimensions_changed": "The image dimensions changed after startup. Reopen the annotation tool.",
        "bad_scale": "Unsupported resolution level: {scale}",
        "missing_directory": "Image directory does not exist: {root}",
        "filename_fields": "The filename must contain six fields: {filename}",
        "filename_labels": "Invalid filename labels: {filename}",
        "no_images": "No images at resolution '{scale}' in: {root}",
        "csv_only": "The annotation output must be a .csv file.",
        "output_is_image": "The annotation output cannot be a source image.",
        "csv_header": "The existing CSV has an incompatible header. Choose a new output file.",
        "csv_image": "The image is outside the selected directory/resolution, or its filename is duplicated.",
        "csv_box": "The square is outside the original image, or its side length is invalid.",
        "csv_metadata": "Image dimensions, labels or coordinate format do not match the existing annotations.",
        "csv_row": "Invalid CSV row {line}: {error}",
        "csv_changed": "Another program changed the CSV. Keep that file and reopen the annotation tool.",
        "startup_error": "Cannot load annotation session: {error}",
        "tk_error": "Cannot start Tk. Use Python with a complete Tcl/Tk installation and a graphical desktop: {error}",
    },
    "zh": {
        "window_title": "ATRI 面部方框标注",
        "preview_title": "面部区域预览（可包含头发）",
        "coordinates_title": "旋转画布像素 · 左下角为原点",
        "angle_label": "角度（逆时针）",
        "unconfirmed_only": "仅浏览未确认项",
        "angle_progress": "各角度已确认：{progress}",
        "angle_complete": "{angle}° 已无待确认项，取消筛选后可复查。",
        "box_proposal": "候选框：请检查当前旋转图并确认，尚未保存。",
        "rotation_metadata": "旋转任务元数据缺失，或与源图、角度、Pillow 版本不一致。请恢复原设置或另选新 CSV。",
        "source_changed": "源图已变化：{filename}。请重新打开标注器；已有标注未被覆盖。",
        "reference_is_output": "参考标注和输出必须使用不同文件。",
        "reference_duplicate": "参考 CSV 对同一内容和角度含有多个分辨率的框，请一次使用一个参考分辨率。",
        "x_left": "左边距 x",
        "y_bottom": "下边距 y",
        "side": "正方形边长",
        "apply": "应用输入坐标",
        "legend": "黄色：待确认；绿色：已保存。\n每张图片需单独确认，继承框不会自动标注。\n点击“快捷键”查看绘制与微调操作。",
        "previous": "上一张",
        "confirm_next": "确认并下一张",
        "next": "下一张 / 暂不确认",
        "unannotated": "下一张未标注",
        "shortcuts": "快捷键",
        "help_title": "操作快捷键",
        "help_text": (
            "左键拖动：绘制正方形\n"
            "框内拖动：移动\n"
            "右下角手柄：调整边长\n"
            "Shift + 拖动：重新画框\n"
            "方向键：移动 1 旋转画布像素\n"
            "+ / - 或滚轮：调整边长\n"
            "按住 Shift：步长为 10 像素\n"
            "画布上的 Enter / 空格：确认并下一张\n"
            "Ctrl+S：确认并保存当前图片\n"
            "PageUp / PageDown：前后浏览"
        ),
        "heading": "{index} / {total}    已确认 {saved} 项    {filename}\n分辨率：{scale}    姿势：{pose}    服装：{outfit}    表情：{expression}    角度：{angle}°\n原图：{source_width} × {source_height}    旋转画布：{width} × {height}",
        "status_output": "标注表格：{path}",
        "status_saved": "已保存 {filename} @ {angle}° → {path}",
        "status_remaining": "已到最后一个视图；还有 {remaining} 项未确认。可点击“下一张未标注”。",
        "status_complete": "全部 {total} 个视图已确认，CSV：{path}",
        "status_review": "所有视图均已确认，可以继续浏览和修改已有方框。",
        "box_empty": "尚无方框，请拖动鼠标绘制。",
        "box_saved": "已确认并保存。",
        "box_outside": "方框超出图片边界，请调整后确认。",
        "box_dirty": "已调整，尚未保存。",
        "box_inherited": "沿用同姿势上次确认的方框，请检查并确认。",
        "coordinates_error_title": "坐标无效",
        "integer_required": "请为左边距、下边距和边长输入整数。",
        "box_bounds": "请将完整正方形框放在图片范围内，边长至少为 1 像素。",
        "draw_first_title": "请先画框",
        "draw_first": "请先绘制面部正方形框。",
        "unsaved_title": "当前方框尚未保存",
        "unsaved_prompt": "是否先确认并保存当前方框？\n选择“否”放弃本次调整。",
        "open_error": "无法打开图片",
        "save_error": "保存失败",
        "save_error_detail": "{error}\n\n若 CSV 正在 Excel 中打开，请关闭表格后重试。",
        "dimensions_changed": "图片尺寸在启动后发生了变化，请重新打开工具。",
        "bad_scale": "不支持的分辨率档位：{scale}",
        "missing_directory": "图片目录不存在：{root}",
        "filename_fields": "文件名必须包含六个字段：{filename}",
        "filename_labels": "文件名标签无效：{filename}",
        "no_images": "目录中没有 {scale} 档图片：{root}",
        "csv_only": "标注输出必须是 .csv 文件。",
        "output_is_image": "标注输出不能是源图片。",
        "csv_header": "已有 CSV 的表头不属于此标注工具，请指定新的输出文件。",
        "csv_image": "图片不在所选目录/分辨率中，或文件名重复。",
        "csv_box": "方框超出原图边界或边长无效。",
        "csv_metadata": "图片尺寸、标签或坐标格式与已有标注不一致。",
        "csv_row": "CSV 第 {line} 行无效：{error}",
        "csv_changed": "CSV 已被其他程序修改。请保留当前文件并重新打开标注工具。",
        "startup_error": "无法载入标注任务：{error}",
        "tk_error": "无法启动 Tk 窗口，请使用带完整 Tcl/Tk 的 Python 和图形桌面环境：{error}",
    },
}


def translate(language, key, **values):
    return TEXTS[language][key].format(**values)


class AnnotationError(ValueError):
    """Keep error details translatable when the GUI language changes."""

    def __init__(self, key, **values):
        self.key = key
        self.values = values
        super().__init__(self.localized("en"))

    def localized(self, language):
        values = {key: value.localized(language) if isinstance(value, AnnotationError) else value
                  for key, value in self.values.items()}
        return translate(language, self.key, **values)


def error_text(error, language):
    return error.localized(language) if isinstance(error, AnnotationError) else str(error)
