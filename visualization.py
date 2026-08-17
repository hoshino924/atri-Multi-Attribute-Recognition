# -*- coding: utf-8 -*-
"""Grad-CAM generation and image overlay helpers."""

import matplotlib
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional


def generate_gradcam(
    model,
    full_tensor,
    expression_tensor,
    task,
    class_index=None,
):
    """Generate a Grad-CAM map for the view used most directly by one task."""
    activations = []
    gradients = []

    def capture_activation(_module, _inputs, output):
        index = len(activations)
        activations.append(output)
        gradients.append(None)

        def capture_gradient(gradient, output_index=index):
            gradients[output_index] = gradient

        output.register_hook(capture_gradient)

    target_layer = model.backbone[-2]
    handle = target_layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            outputs = model(full_tensor, expression_tensor)
            logits = outputs[task]
            if class_index is None:
                class_index = logits.argmax(dim=1).item()
            logits[0, class_index].backward()

        view_index = 1 if task == "expression" else 0
        activation = activations[view_index]
        gradient = gradients[view_index]
        if gradient is None:
            raise RuntimeError(f"no gradient was produced for task '{task}'")

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True).relu()
        target_size = (
            expression_tensor.shape[-2:]
            if task == "expression"
            else full_tensor.shape[-2:]
        )
        cam = functional.interpolate(
            cam,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam -= cam.min()
        maximum = cam.max()
        if maximum > 0:
            cam /= maximum
        return cam.detach().cpu()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def overlay_gradcam(image, cam, alpha=0.45):
    """Blend a normalized Grad-CAM heatmap over a PIL image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    base = image.convert("RGB")
    values = cam.clamp(0.0, 1.0).numpy()
    colormap = matplotlib.colormaps["turbo"]
    heatmap = (colormap(values)[..., :3] * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap, mode="RGB")
    if heatmap_image.size != base.size:
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        heatmap_image = heatmap_image.resize(base.size, resampling)
    return Image.blend(base, heatmap_image, alpha)
