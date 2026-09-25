# -*- coding: utf-8 -*-
"""Export a trained ATRI checkpoint to ONNX with a JSON sidecar."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from infer import load_model
from labels import TASK_CODES


TASKS = tuple(TASK_CODES.keys())


class OnnxWrapper(nn.Module):
    """Convert the model's task dictionary into stable named ONNX outputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, full_image, expression_image):
        outputs = self.model(full_image, expression_image)
        return tuple(outputs[task] for task in TASKS)


def export(args):
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if args.opset <= 0:
        raise ValueError("--opset must be positive")
    output_path = Path(args.output)
    metadata_path = output_path.with_suffix(".json")
    if (output_path.exists() or metadata_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"ONNX output already exists: {output_path}; use --overwrite"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )
    model, _image_transforms, label_codes = load_model(args.weight, device)
    wrapper = OnnxWrapper(model).to(device).eval()
    preprocess = model.preprocess_config
    full = preprocess["full"]
    expression = preprocess["expression"]
    full_input = torch.zeros(
        args.batch,
        3,
        full["height"],
        full["width"],
        device=device,
    )
    expression_input = torch.zeros(
        args.batch,
        3,
        expression["size"],
        expression["size"],
        device=device,
    )

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "full_image": {0: "batch"},
            "expression_image": {0: "batch"},
            **{task: {0: "batch"} for task in TASKS},
        }
    torch.onnx.export(
        wrapper,
        (full_input, expression_input),
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["full_image", "expression_image"],
        output_names=list(TASKS),
        dynamic_axes=dynamic_axes,
    )

    metadata = {
        "source_checkpoint": str(args.weight),
        "opset": args.opset,
        "dynamic_batch": args.dynamic_batch,
        "inputs": {
            "full_image": ["batch", 3, full["height"], full["width"]],
            "expression_image": [
                "batch",
                3,
                expression["size"],
                expression["size"],
            ],
        },
        "outputs": list(TASKS),
        "label_codes": label_codes,
        "model_config": model.checkpoint_metadata["model_config"],
        "preprocess": preprocess,
        "external_face_localization_required": hasattr(model, "face_preview"),
        "face_localization_note": (
            "Run the independent locator, map its square to original-image pixels, "
            "crop, downscale only if larger than expression size, then center-pad. "
            "Localization, cropping and face-presence rejection are not inside this ONNX graph."
            if hasattr(model, "face_preview") else None
        ),
        "calibration": model.calibration,
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)

    print("Saved:", output_path)
    print("Saved:", metadata_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Export an ATRI model to ONNX.")
    from app_assets import DEFAULT_CLASSIFIER_WEIGHT
    parser.add_argument("--weight", default=DEFAULT_CLASSIFIER_WEIGHT)
    parser.add_argument("--output", default="outputs/atri_net.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--fixed_batch",
        action="store_false",
        dest="dynamic_batch",
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
