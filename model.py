# -*- coding: utf-8 -*-
"""Model definition for the ATRI multi-attribute classifier."""

import torch.nn as nn
from torchvision import models
from labels import SHOE_CODES, OUTFIT_CODES, POSE_CODES, EXPR_CODES


class AtriNet(nn.Module):
    """ResNet18 backbone with four classification heads."""

    def __init__(self, pretrained=True):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)

        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.shoe_head = nn.Linear(512, len(SHOE_CODES))
        self.outfit_head = nn.Linear(512, len(OUTFIT_CODES))
        self.pose_head = nn.Linear(512, len(POSE_CODES))
        self.expr_head = nn.Linear(512, len(EXPR_CODES))

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        shoe = self.shoe_head(feat)
        outfit = self.outfit_head(feat)
        pose = self.pose_head(feat)
        expr = self.expr_head(feat)
        return shoe, outfit, pose, expr
