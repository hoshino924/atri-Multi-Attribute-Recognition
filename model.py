# -*- coding: utf-8 -*-
"""Model definition for the ATRI multi-attribute classifier."""

import torch
import torch.nn as nn
from torchvision import models
from labels import TASK_CODES


EXPRESSION_HEADS = ("fusion", "face_only")


def checkpoint_expression_head(checkpoint):
    """Read the head contract; checkpoints before stage E use fusion."""
    head = checkpoint.get("model_config", {}).get("expression_head", "fusion")
    if head not in EXPRESSION_HEADS:
        raise ValueError(f"unsupported expression head: {head}")
    recorded = checkpoint.get("train_args", {}).get("expression_head")
    if recorded is not None and recorded != head:
        raise ValueError("checkpoint expression head disagrees with train_args")
    return head


class AtriNet(nn.Module):
    """ResNet18 backbone with configurable attribute heads."""

    def __init__(self, pretrained=True, dropout=0.2, task_sizes=None, expression_head="fusion"):
        super().__init__()
        if expression_head not in EXPRESSION_HEADS:
            raise ValueError(f"unsupported expression head: {expression_head}")
        self.expression_head = expression_head

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)

        task_sizes = task_sizes or {
            task: len(codes)
            for task, codes in TASK_CODES.items()
        }
        self.task_sizes = dict(task_sizes)
        feature_size = net.fc.in_features
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(
                    feature_size * 2 if task == "expression" and expression_head == "fusion" else feature_size,
                    size,
                ),
            )
            for task, size in self.task_sizes.items()
        })

    def extract_features(self, image):
        """Extract one pooled feature vector with the shared backbone."""
        return self.backbone(image).flatten(1)

    def forward(self, full_image, expression_image):
        full_features = self.extract_features(full_image)
        expression_features = self.extract_features(expression_image)

        outputs = {}
        for task, head in self.heads.items():
            features = full_features
            if task == "expression":
                features = expression_features
                if self.expression_head == "fusion":
                    features = torch.cat((full_features, expression_features), dim=1)
            outputs[task] = head(features)
        return outputs

    def set_backbone_trainable(self, trainable):
        """Freeze or unfreeze the shared feature extractor."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def freeze_backbone_batchnorm(self):
        """Keep pretrained BatchNorm running statistics fixed while training."""
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
