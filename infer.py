# -*- coding: utf-8 -*-
"""Tkinter inference application for the three-task ATRI model."""

import argparse
import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk
import torch

from dataset import build_expression_transform, build_transform
from labels import TASK_CODES, TASK_LABELS
from model import AtriNet


TASKS = tuple(TASK_CODES.keys())


def load_torch_checkpoint(path, device):
    """Load a checkpoint without executing arbitrary serialized objects."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(weight_path, device):
    """Load the model and preprocessing configuration from a checkpoint."""
    checkpoint = load_torch_checkpoint(weight_path, device)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(
            "this weight file uses the old four-task format; retrain the new "
            "three-task model first"
        )
    if checkpoint.get("format_version") != 3:
        raise ValueError(
            "this checkpoint predates the focused expression view; retrain "
            "the dual-view model first"
        )

    label_codes = checkpoint.get("label_codes", TASK_CODES)
    if set(label_codes.keys()) != set(TASKS):
        raise ValueError(
            f"checkpoint tasks {tuple(label_codes.keys())} do not match {TASKS}"
        )
    label_codes = {
        task: list(label_codes[task])
        for task in TASKS
    }

    model_config = checkpoint.get("model_config", {})
    task_sizes = {
        task: len(label_codes[task])
        for task in TASKS
    }
    model = AtriNet(
        pretrained=False,
        dropout=model_config.get("dropout", 0.2),
        task_sizes=task_sizes,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    preprocess = checkpoint.get("preprocess", {})
    full_config = preprocess.get("full", {})
    expression_config = preprocess.get("expression", {})
    background = tuple(preprocess.get("background", (0, 0, 0)))
    normalization = preprocess.get("normalization", {})
    mean = tuple(normalization.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(normalization.get("std", (0.229, 0.224, 0.225)))
    image_transforms = {
        "full": build_transform(
            height=full_config.get("height", 512),
            width=full_config.get("width", 320),
            train=False,
            margin=full_config.get("margin", 0.04),
            background=background,
            mean=mean,
            std=std,
        ),
        "expression": build_expression_transform(
            size=expression_config.get("size", 512),
            train=False,
            width_fraction=expression_config.get("width_fraction", 0.65),
            height_fraction=expression_config.get("height_fraction", 0.50),
            margin=expression_config.get("margin", 0.04),
            background=background,
            mean=mean,
            std=std,
        ),
    }
    return model, image_transforms, label_codes


def predict(model, image_path, device, image_transforms, label_codes):
    """Run prediction for one image."""
    with Image.open(image_path) as source:
        image = source.convert("RGBA")
    full_tensor = image_transforms["full"](image).unsqueeze(0).to(device)
    expression_tensor = (
        image_transforms["expression"](image).unsqueeze(0).to(device)
    )

    with torch.inference_mode():
        outputs = model(full_tensor, expression_tensor)

    result = {}
    for task in TASKS:
        probabilities = torch.softmax(outputs[task], dim=1)
        confidence, index = probabilities.max(dim=1)
        code = label_codes[task][index.item()]
        result[task] = {
            "code": code,
            "label": TASK_LABELS[task].get(code, code),
            "confidence": confidence.item(),
        }

    return image, result


def start_gui(model, device, image_transforms, label_codes):
    """Start the image selection and prediction interface."""
    root = tk.Tk()
    root.title("ATRI Multi-Attribute Recognition")
    root.geometry("720x900")

    title = tk.Label(
        root,
        text="ATRI Multi-Attribute Recognition",
        font=("Arial", 18, "bold"),
    )
    title.pack(pady=10)

    image_label = tk.Label(root)
    image_label.pack(pady=10)

    result_label = tk.Label(
        root,
        text="Please select an image.",
        font=("Arial", 14),
        justify="left",
    )
    result_label.pack(pady=10)

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

        image, result = predict(
            model,
            path,
            device,
            image_transforms,
            label_codes,
        )
        display_image = image.copy()
        display_image.thumbnail((500, 650))
        tk_image = ImageTk.PhotoImage(display_image)
        image_label.configure(image=tk_image)
        image_label.image = tk_image

        text = (
            f"Outfit     : {result['outfit']['label']} "
            f"({result['outfit']['confidence']:.1%})\n"
            f"Pose       : {result['pose']['label']} "
            f"({result['pose']['confidence']:.1%})\n"
            f"Expression : {result['expression']['label']} "
            f"({result['expression']['confidence']:.1%})"
        )
        result_label.configure(text=text)

    button = tk.Button(
        root,
        text="Select Image",
        command=open_image,
        font=("Arial", 14),
        width=20,
    )
    button.pack(pady=15)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default="outputs/atri_net_best.pth")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )
    print("Using device:", device)

    model, image_transforms, label_codes = load_model(args.weight, device)
    start_gui(model, device, image_transforms, label_codes)


if __name__ == "__main__":
    main()
