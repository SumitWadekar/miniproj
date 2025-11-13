from dataclasses import dataclass
import torch
import torch.nn as nn
from torchvision import models


@dataclass
class ModelConfig:
    pretrained: bool = True
    backbone: str = "efficientnet_b0"  # or "resnet18"
    out_dim: int = 5


class VisionStockRegressor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_f = self.backbone.classifier[1].in_features
            # Replace classifier head
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_f, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, cfg.out_dim),
            )
        elif cfg.backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_f = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_f, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, cfg.out_dim),
            )
        else:
            raise ValueError("Unsupported backbone")

    def forward(self, x):
        return self.backbone(x)
