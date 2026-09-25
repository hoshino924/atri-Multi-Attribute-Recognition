# -*- coding: utf-8 -*-
"""Tkinter inference application for the three-task ATRI model."""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image
import torch

from calibration import apply_temperature, suggested_threshold
from dataset import (
    IMAGE_EXTENSIONS,
    build_expression_preview,
    build_expression_transform,
    build_full_preview,
    build_transform,
)
from labels import TASK_CODES, TASK_LABELS
from model import AtriNet, checkpoint_expression_head
from visualization import generate_gradcam, overlay_gradcam
from face_regions import FACE_CROP_MODE
from face_locator import FaceLocator, FaceNotFoundError, LocatedFacePreview, LocatedFaceTransform
from image_rotation import ROTATION_CONTRACT
from app_assets import DEFAULT_CLASSIFIER_WEIGHT, DEFAULT_LOCATOR_WEIGHT, configure_taskbar, set_app_icon


TASKS = tuple(TASK_CODES.keys())


def load_torch_checkpoint(path, device):
    """Load a checkpoint without executing arbitrary serialized objects."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(weight_path, device, locator_weight=None, locator_threshold=None):
    """Load the model and preprocessing configuration from a checkpoint."""
    checkpoint = load_torch_checkpoint(weight_path, device)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(
            "this weight file uses the old four-task format; retrain the new "
            "three-task model first"
        )
    if checkpoint.get("format_version") not in (3, 4):
        raise ValueError(
            "this checkpoint predates the focused expression view; retrain "
            "the dual-view model first"
        )
    rotation_training = checkpoint.get("rotation_training")
    if rotation_training:
        if rotation_training.get("rotation") != ROTATION_CONTRACT:
            raise ValueError("unsupported classifier rotation contract")
        expected_locator = (checkpoint.get("face_cache_metadata") or {}).get("locator", {})
        if not expected_locator.get("sha256"):
            raise ValueError("rotated classifier lacks its fixed locator identity")
        if locator_weight:
            from face_box_cache import file_sha256
            if file_sha256(locator_weight) != expected_locator["sha256"]:
                raise ValueError("locator weights differ from classifier training/calibration")
            if locator_threshold is not None and locator_threshold != expected_locator.get("threshold"):
                raise ValueError("locator threshold differs from classifier training/calibration")

    label_codes = checkpoint.get("label_codes", TASK_CODES)
    if set(label_codes.keys()) != set(TASKS):
        raise ValueError(
            f"checkpoint tasks {tuple(label_codes.keys())} do not match {TASKS}"
        )
    label_codes = {
        task: list(label_codes[task])
        for task in TASKS
    }
    for task in TASKS:
        if set(label_codes[task]) != set(TASK_CODES[task]):
            raise ValueError(
                f"checkpoint {task} labels do not match this program"
            )

    model_config = checkpoint.get("model_config", {})
    expression_head = checkpoint_expression_head(checkpoint)
    task_sizes = {
        task: len(label_codes[task])
        for task in TASKS
    }
    model = AtriNet(
        pretrained=False,
        dropout=model_config.get("dropout", 0.2),
        task_sizes=task_sizes,
        expression_head=expression_head,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    preprocess = checkpoint.get("preprocess", {})
    full_config = preprocess.get("full", {})
    expression_config = preprocess.get("expression", {})
    face_mode = checkpoint["format_version"] == 4
    if face_mode:
        if (expression_config.get("mode") != FACE_CROP_MODE
                or expression_config.get("allow_upscale") is not False
                or expression_config.get("margin") != 0.0
                or expression_config.get("coordinate_system") != "bottom_left_pixels"):
            raise ValueError("unsupported face preprocessing contract in checkpoint")
    elif expression_config.get("mode", "foreground") != "foreground":
        raise ValueError("unsupported legacy expression crop mode")
    background = tuple(preprocess.get("background", (0, 0, 0)))
    normalization = preprocess.get("normalization", {})
    mean = tuple(normalization.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(normalization.get("std", (0.229, 0.224, 0.225)))
    resolved_preprocess = {
        "full": {
            "height": full_config.get("height", 512),
            "width": full_config.get("width", 320),
            "margin": full_config.get("margin", 0.04),
        },
        "expression": {
            "size": expression_config.get("size", 512),
            "width_fraction": expression_config.get("width_fraction", 0.65),
            "height_fraction": expression_config.get("height_fraction", 0.50),
            "margin": expression_config.get("margin", 0.04),
        },
        "background": background,
        "normalization": {"mean": mean, "std": std},
    }
    image_transforms = {
        "full": build_transform(
            height=resolved_preprocess["full"]["height"],
            width=resolved_preprocess["full"]["width"],
            train=False,
            margin=resolved_preprocess["full"]["margin"],
            background=background,
            mean=mean,
            std=std,
        ),
        "expression": build_expression_transform(
            size=resolved_preprocess["expression"]["size"],
            train=False,
            width_fraction=resolved_preprocess["expression"]["width_fraction"],
            height_fraction=resolved_preprocess["expression"]["height_fraction"],
            margin=resolved_preprocess["expression"]["margin"],
            background=background,
            mean=mean,
            std=std,
        ),
    }
    if face_mode:
        resolved_preprocess["expression"] = dict(expression_config)
        locator = FaceLocator.from_checkpoint(locator_weight, device, locator_threshold) if locator_weight else None
        preview = LocatedFacePreview(expression_config["size"], background, locator)
        image_transforms["expression"] = LocatedFaceTransform(preview, mean, std)
        model.face_preview = preview
    model.calibration = checkpoint.get("calibration", {})
    model.preprocess_config = resolved_preprocess
    model.checkpoint_metadata = {
        "epoch": checkpoint.get("epoch"),
        "format_version": checkpoint.get("format_version"),
        "model_config": {**model_config, "expression_head": expression_head},
        "rotation_training": rotation_training,
        "dataset_signature": checkpoint.get("dataset_signature"),
        "face_cache_metadata": checkpoint.get("face_cache_metadata"),
        "selection_metric": checkpoint.get("selection_metric", "expression_cross_entropy"),
    }
    return model, image_transforms, label_codes


def prepare_tensors(image, device, image_transforms):
    """Create batched tensors for one already opened image."""
    full_tensor = image_transforms["full"](image).unsqueeze(0).to(device)
    expression_tensor = (
        image_transforms["expression"](image).unsqueeze(0).to(device)
    )
    return full_tensor, expression_tensor


def predict(
    model,
    image_path,
    device,
    image_transforms,
    label_codes,
    top_k=1,
    min_confidence=None,
    accept_all=False,
):
    """Run calibrated prediction for one image."""
    with Image.open(image_path) as source:
        image = source.convert("RGBA")
    full_tensor, expression_tensor = prepare_tensors(
        image,
        device,
        image_transforms,
    )

    with torch.inference_mode():
        outputs = model(full_tensor, expression_tensor)

    result = {}
    for task in TASKS:
        calibrated_logits = apply_temperature(
            outputs[task],
            task,
            getattr(model, "calibration", {}),
        )
        probabilities = torch.softmax(calibrated_logits, dim=1)
        count = min(max(1, top_k), probabilities.shape[1])
        confidences, indices = probabilities.topk(count, dim=1)
        candidates = []
        for confidence, index in zip(confidences[0], indices[0]):
            code = label_codes[task][index.item()]
            candidates.append({
                "code": code,
                "label": TASK_LABELS[task].get(code, code),
                "confidence": confidence.item(),
            })

        best = candidates[0]
        threshold = None
        if not accept_all:
            threshold = (
                min_confidence
                if min_confidence is not None
                else suggested_threshold(
                    task,
                    getattr(model, "calibration", {}),
                )
            )
        accepted = threshold is None or best["confidence"] >= threshold
        result[task] = {
            **best,
            "accepted": accepted,
            "threshold": threshold,
            "candidates": candidates,
        }

    if hasattr(model, "face_preview"):
        box = model.face_preview.last_box
        result["face"] = {
            "coordinate_system": "bottom_left_pixels",
            "x_left": box.x_left, "y_bottom": box.y_bottom, "side": box.side,
            "confidence": model.face_preview.last_confidence,
        }
    return image, result


def collect_input_paths(input_path, recursive=False):
    """Resolve one image or a sorted directory of supported images."""
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported image extension: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"input path does not exist: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    paths = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"no supported images found in: {input_path}")
    return paths


def batch_predict(
    model,
    paths,
    device,
    image_transforms,
    label_codes,
    top_k=1,
    min_confidence=None,
    accept_all=False,
):
    """Predict a sequence of files and return JSON-serializable records."""
    records = []
    for path in paths:
        try:
            _, result = predict(
                model, path, device, image_transforms, label_codes,
                top_k=top_k, min_confidence=min_confidence, accept_all=accept_all,
            )
            records.append({"file": str(path), "status": "ok", "predictions": result})
        except FaceNotFoundError as exc:
            records.append({"file": str(path), "status": "face_not_found", "error": str(exc), "predictions": None})
    return records


def save_batch_results(records, output_path):
    """Write nested JSON or flattened CSV based on the output extension."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        fieldnames = ["file", "status", "error"]
        for task in TASKS:
            fieldnames.extend([
                f"{task}_code",
                f"{task}_label",
                f"{task}_confidence",
                f"{task}_accepted",
                f"{task}_threshold",
                f"{task}_top_k",
            ])
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = {"file": record["file"], "status": record.get("status", "ok"), "error": record.get("error", "")}
                if record["predictions"] is None:
                    writer.writerow(row)
                    continue
                for task in TASKS:
                    prediction = record["predictions"][task]
                    row.update({
                        f"{task}_code": prediction["code"],
                        f"{task}_label": prediction["label"],
                        f"{task}_confidence": prediction["confidence"],
                        f"{task}_accepted": prediction["accepted"],
                        f"{task}_threshold": prediction["threshold"],
                        f"{task}_top_k": json.dumps(
                            prediction["candidates"],
                            ensure_ascii=False,
                        ),
                    })
                writer.writerow(row)
    else:
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(records, stream, ensure_ascii=False, indent=2)


def build_preview_transforms(model):
    """Build exact display views from the checkpoint preprocessing config."""
    config = model.preprocess_config
    full = config["full"]
    expression = config["expression"]
    background = config["background"]
    return {
        "full": build_full_preview(
            height=full["height"],
            width=full["width"],
            margin=full["margin"],
            background=background,
        ),
        "expression": model.face_preview if hasattr(model, "face_preview") else build_expression_preview(
            size=expression["size"],
            width_fraction=expression["width_fraction"],
            height_fraction=expression["height_fraction"],
            margin=expression["margin"],
            background=background,
        ),
    }


def start_gui(model, device, image_transforms, label_codes):
    """Start the preview, calibrated prediction, and Grad-CAM interface."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk

    configure_taskbar("Recognition")
    root = tk.Tk()
    set_app_icon(root)
    root.title("ATRI Multi-Attribute Recognition")
    root.geometry("1120x850")
    root.minsize(900, 720)
    preview_transforms = build_preview_transforms(model)
    state = {"image": None, "result": None, "photos": {}}

    title = ttk.Label(
        root,
        text="ATRI Multi-Attribute Recognition",
        font=("Arial", 18, "bold"),
    )
    title.pack(pady=(12, 8))

    previews = ttk.Frame(root)
    previews.pack(fill="both", expand=True, padx=12)
    preview_labels = {}
    for column, (name, caption) in enumerate((
        ("full", "Full model view"),
        ("expression", "Expression crop"),
        ("gradcam", "Grad-CAM"),
    )):
        panel = ttk.Frame(previews)
        panel.grid(row=0, column=column, sticky="nsew", padx=6)
        ttk.Label(panel, text=caption, font=("Arial", 11, "bold")).pack(
            pady=(0, 6)
        )
        preview_labels[name] = ttk.Label(panel, anchor="center")
        preview_labels[name].pack(fill="both", expand=True)
        previews.columnconfigure(column, weight=1, uniform="preview")
    previews.rowconfigure(0, weight=1)

    result_label = ttk.Label(
        root,
        text="Please select an image.",
        font=("Consolas", 12),
        justify="left",
    )
    result_label.pack(pady=10)

    controls = ttk.Frame(root)
    controls.pack(pady=(2, 10))
    task_variable = tk.StringVar(value="expression")
    task_selector = ttk.Combobox(
        controls,
        textvariable=task_variable,
        values=TASKS,
        state="readonly",
        width=12,
    )
    task_selector.grid(row=0, column=1, padx=6)
    status_label = ttk.Label(root, text="")
    status_label.pack(pady=(0, 8))

    def show_image(name, image, maximum):
        display = image.convert("RGB").copy()
        display.thumbnail(maximum)
        photo = ImageTk.PhotoImage(display)
        preview_labels[name].configure(image=photo)
        state["photos"][name] = photo

    def result_text(result):
        lines = []
        captions = {
            "outfit": "Outfit",
            "pose": "Pose",
            "expression": "Expression",
        }
        for task in TASKS:
            prediction = result[task]
            if prediction["accepted"]:
                value = prediction["label"]
            else:
                value = f"uncertain (best: {prediction['label']})"
            threshold = prediction["threshold"]
            threshold_text = (
                f", threshold {threshold:.1%}"
                if threshold is not None
                else ""
            )
            lines.append(
                f"{captions[task]:10}: {value} "
                f"({prediction['confidence']:.1%}{threshold_text})"
            )
        return "\n".join(lines)

    def open_image():
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            image, result = predict(
                model,
                path,
                device,
                image_transforms,
                label_codes,
                top_k=3,
            )
            full_preview = preview_transforms["full"](image)
            expression_preview = preview_transforms["expression"](image)
            show_image("full", full_preview, (320, 520))
            show_image("expression", expression_preview, (320, 420))
            preview_labels["gradcam"].configure(image="")
            state["photos"].pop("gradcam", None)
            state["image"] = image
            state["result"] = result
            result_label.configure(text=result_text(result))
            status_label.configure(text=Path(path).name)
        except Exception as exc:
            state["image"] = state["result"] = None
            for name in preview_labels:
                preview_labels[name].configure(image="")
            state["photos"].clear()
            result_label.configure(text="No prediction available.")
            messagebox.showerror("Inference failed", str(exc), parent=root)

    def show_gradcam():
        if state["image"] is None:
            messagebox.showinfo(
                "Grad-CAM",
                "Select an image first.",
                parent=root,
            )
            return
        task = task_variable.get()
        try:
            full_tensor, expression_tensor = prepare_tensors(
                state["image"],
                device,
                image_transforms,
            )
            code = state["result"][task]["code"]
            class_index = label_codes[task].index(code)
            cam = generate_gradcam(
                model,
                full_tensor,
                expression_tensor,
                task,
                class_index=class_index,
            )
            view_name = "expression" if task == "expression" else "full"
            base = preview_transforms[view_name](state["image"])
            overlay = overlay_gradcam(base, cam)
            show_image("gradcam", overlay, (320, 520))
            status_label.configure(text=f"Grad-CAM: {task} / {code}")
        except Exception as exc:
            messagebox.showerror("Grad-CAM failed", str(exc), parent=root)

    ttk.Button(controls, text="Select Image", command=open_image).grid(
        row=0,
        column=0,
        padx=6,
    )
    ttk.Button(controls, text="Show Grad-CAM", command=show_gradcam).grid(
        row=0,
        column=2,
        padx=6,
    )
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="Run GUI, single-image, or directory inference."
    )
    parser.add_argument("--weight", default=DEFAULT_CLASSIFIER_WEIGHT)
    parser.add_argument("--locator_weight", default=DEFAULT_LOCATOR_WEIGHT,
                        help="independent face locator checkpoint for face-crop models")
    parser.add_argument("--locator_threshold", type=float, help="override locator confidence threshold")
    parser.add_argument("--input", help="image file or directory; omit for GUI")
    parser.add_argument("--output", help="batch result path ending in .json or .csv")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--top_k", type=int, default=3)
    confidence_group = parser.add_mutually_exclusive_group()
    confidence_group.add_argument("--min_confidence", type=float)
    confidence_group.add_argument(
        "--accept_all",
        action="store_true",
        help="disable checkpoint confidence thresholds",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top_k must be positive")
    if args.min_confidence is not None and not 0.0 <= args.min_confidence <= 1.0:
        parser.error("--min_confidence must be in [0, 1]")
    if args.output and not args.input:
        parser.error("--output requires --input")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )
    print("Using device:", device)

    model, image_transforms, label_codes = load_model(args.weight, device, args.locator_weight, args.locator_threshold)
    if hasattr(model, "face_preview") and model.face_preview.locator is None:
        parser.error("this face-crop classifier requires --locator_weight")
    if args.input:
        paths = collect_input_paths(args.input, recursive=args.recursive)
        records = batch_predict(
            model,
            paths,
            device,
            image_transforms,
            label_codes,
            top_k=args.top_k,
            min_confidence=args.min_confidence,
            accept_all=args.accept_all,
        )
        if args.output:
            save_batch_results(records, args.output)
            print(f"Saved {len(records)} predictions: {args.output}")
        else:
            print(json.dumps(records, ensure_ascii=False, indent=2))
    else:
        start_gui(model, device, image_transforms, label_codes)


if __name__ == "__main__":
    main()
