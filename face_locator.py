# -*- coding: utf-8 -*-
"""Small independently trained single-ATRI-face locator (no external weights)."""

import math

import torch
from torch import nn
from torchvision.transforms import functional as TF

from face_regions import FaceCanvas, crop_face, locator_canvas, map_locator_box


LOCATOR_VERSION = 1


class FaceNotFoundError(ValueError):
    """No sufficiently confident, geometrically valid face was localized."""


class FaceLocatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = 3
        for output_channels in (16, 32, 64, 96):
            layers.extend([
                nn.Conv2d(channels, output_channels, 3, stride=2, padding=1),
                nn.GroupNorm(4, output_channels), nn.SiLU(),
            ])
            channels = output_channels
        # Retain spatial information for coordinate regression.
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((4, 4)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(96 * 4 * 4, 128), nn.SiLU())
        self.presence = nn.Linear(128, 1)
        self.box = nn.Linear(128, 4)

    def forward(self, images):
        features = self.head(self.features(images))
        return self.presence(features).squeeze(1), self.box(features).sigmoid()


class FaceLocator:
    def __init__(self, model, device, size=256, threshold=0.5):
        if size < 32 or not 0 < threshold <= 1:
            raise ValueError("invalid locator input size or confidence threshold")
        self.model = model.eval()
        self.device = device
        self.size = size
        self.threshold = threshold

    @classmethod
    def from_checkpoint(cls, path, device, threshold=None):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("locator_version") != LOCATOR_VERSION:
            raise ValueError("unsupported face locator checkpoint")
        if checkpoint.get("epoch", 0) < 1:
            raise ValueError("face locator checkpoint has not completed a training epoch")
        model = FaceLocatorNet().to(device)
        model.load_state_dict(checkpoint["model_state"])
        return cls(model, device, checkpoint["input_size"],
                   checkpoint["threshold"] if threshold is None else threshold)

    def locate(self, image):
        preview, mapping = locator_canvas(image, self.size)
        tensor = TF.to_tensor(preview).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            presence, boxes = self.model(tensor)
        confidence = presence.sigmoid()[0].item()
        if not math.isfinite(confidence) or confidence < self.threshold:
            raise FaceNotFoundError(f"No confident ATRI face found (score={confidence:.3f}, threshold={self.threshold:.3f}).")
        try:
            box = map_locator_box(boxes[0].tolist(), mapping, image.size, self.size)
        except ValueError as exc:
            raise FaceNotFoundError(str(exc)) from exc
        return box, confidence


class LocatedFacePreview:
    """Use a locator or an explicit per-image annotation for the same canvas."""

    def __init__(self, size=626, background=(0, 0, 0), locator=None):
        self.canvas = FaceCanvas(size, background)
        self.locator = locator
        self.last_image = None
        self.last_box = None
        self.last_confidence = None

    def __call__(self, image, box=None):
        if box is None:
            if self.locator is None:
                raise ValueError("This face-crop checkpoint requires --locator_weight or explicit face annotations.")
            if image is self.last_image:
                box = self.last_box
            else:
                box, confidence = self.locator.locate(image)
                self.last_image, self.last_box, self.last_confidence = image, box, confidence
        return self.canvas(crop_face(image, box))


class LocatedFaceTransform:
    def __init__(self, preview, mean, std):
        self.preview = preview
        self.mean, self.std = mean, std

    def __call__(self, image, box=None):
        return TF.normalize(TF.to_tensor(self.preview(image, box)), self.mean, self.std)
