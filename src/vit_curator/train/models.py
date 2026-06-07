"""Model architectures for image classification."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def create_vit_model(
    num_classes: int,
    img_size: int = 224,
    pretrained: bool = True,
) -> nn.Module:
    """Create Vision Transformer model."""
    try:
        import timm  # noqa: PLC0415

        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
        )
    except ImportError:
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    return model


def create_resnet_model(
    num_classes: int,
    arch: str = "resnet50",
    pretrained: bool = True,
) -> nn.Module:
    """Create ResNet model."""
    arch_map = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
        "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2),
        "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V2),
    }

    if arch not in arch_map:
        raise ValueError(f"Unknown architecture: {arch}. Choose from {list(arch_map.keys())}")

    model_fn, weights = arch_map[arch]
    if not pretrained:
        weights = None

    model = model_fn(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def create_multilabel_head(
    backbone: nn.Module,
    num_classes: int,
    dropout: float = 0.5,
    hidden_dim: int | None = None,
) -> nn.Module:
    """Create multi-label classification head with optional hidden layers and dropout."""
    if hasattr(backbone, "fc"):
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif hasattr(backbone, "head"):
        in_features = backbone.head.in_features
        backbone.head = nn.Identity()
    elif hasattr(backbone, "classifier"):
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
    else:
        raise ValueError("Could not determine input features from backbone")

    if hidden_dim is None:
        head = nn.Linear(in_features, num_classes)
    else:
        head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    class MultiLabelModel(nn.Module):
        def __init__(self, backbone: nn.Module, head: nn.Module):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.backbone(x)
            if isinstance(x, tuple):
                x = x[0]
            return self.head(x)

    return MultiLabelModel(backbone, head)
