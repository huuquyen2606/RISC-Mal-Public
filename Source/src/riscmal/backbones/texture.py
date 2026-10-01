"""2D Texture multi-scale feature extractor backbone using EfficientNet ensemble.

Processes 224x224 RGB byte-texture maps through frozen pre-trained EfficientNet
backbones (B0: 1280-d, B1: 1280-d, B2: 1408-d), concatenates their Global Average
Pooling representations (3968 dimensions), and projects down to 64 dimensions.
Gracefully falls back to offline initialization or architectural surrogates if
torchvision is missing or remote pretrained weights are unreachable.
"""

from typing import Optional
import torch
import torch.nn as nn
from riscmal.backbones.common import DNNBlock


class _SurrogateEfficientNetGAP(nn.Module):
    """Architectural surrogate for EfficientNet GAP when torchvision is not installed.

    Provides identical output tensor dimensionality [B, out_dim] from [B, 3, H, W]
    inputs with frozen weights and persistent eval mode.
    """

    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.out_features = out_features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Conv2d(3, out_features, kernel_size=1)
        # Freeze parameters
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects image tensor [B, 3, H, W] to pooled representation [B, out_features]."""
        with torch.no_grad():
            x = self.pool(x)
            x = self.proj(x)
            return torch.flatten(x, start_dim=1)


def _build_backbone_column(model_name: str, out_dim: int) -> nn.Module:
    """Attempts to construct a frozen EfficientNet backbone from torchvision with offline fallback."""
    try:
        from torchvision import models as torch_models

        factory = getattr(torch_models, model_name, None)
        if factory is None:
            return _SurrogateEfficientNetGAP(out_dim)

        try:
            # Try default pretrained weights
            net = factory(weights="DEFAULT")
        except Exception:
            # Graceful offline fallback
            try:
                net = factory(weights=None)
            except Exception:
                net = factory(pretrained=False)

        # Replace classification head with Identity to extract GAP feature vector
        net.classifier = nn.Identity()
        for p in net.parameters():
            p.requires_grad = False
        net.eval()
        return net
    except (ImportError, ModuleNotFoundError):
        return _SurrogateEfficientNetGAP(out_dim)


class TextureBackbone(nn.Module):
    """Backbone for 2D PE texture multi-scale representation.

    Architecture:
        Input [B, 3, 224, 224] ->
        Concatenate(GAP(B0), GAP(B1), GAP(B2)) [B, 1280 + 1280 + 1408 = 3968] ->
        DNNBlock(3968, 512) ->
        DNNBlock(512, 256) ->
        DNNBlock(256, 64) ->
        Output [B, 64].
    """

    def __init__(
        self,
        out_features: int = 64,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.out_features = out_features
        self.dropout_rate = dropout_rate

        # Multi-scale EfficientNet ensemble
        self.efn_b0 = _build_backbone_column("efficientnet_b0", 1280)
        self.efn_b1 = _build_backbone_column("efficientnet_b1", 1280)
        self.efn_b2 = _build_backbone_column("efficientnet_b2", 1408)

        # Ensure frozen eval mode
        for net in [self.efn_b0, self.efn_b1, self.efn_b2]:
            for p in net.parameters():
                p.requires_grad = False
            net.eval()

        # Dense projection: 3968 -> 512 -> 256 -> 64
        self.img2d_dnn = nn.Sequential(
            DNNBlock(3968, 512, dropout_rate=dropout_rate),
            DNNBlock(512, 256, dropout_rate=dropout_rate),
            DNNBlock(256, out_features, dropout_rate=dropout_rate),
        )

    def train(self, mode: bool = True) -> "TextureBackbone":
        """Overrides train mode to ensure frozen EfficientNet backbones remain in eval mode."""
        super().train(mode)
        self.efn_b0.eval()
        self.efn_b1.eval()
        self.efn_b2.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 64-dimensional features from 2D RGB byte texture images.

        Args:
            x: Tensor of shape [B, 3, 224, 224].

        Returns:
            Tensor of shape [B, 64].
        """
        with torch.no_grad():
            feat_b0 = self.efn_b0(x)
            feat_b1 = self.efn_b1(x)
            feat_b2 = self.efn_b2(x)

        # Concatenate 1280 + 1280 + 1408 = 3968
        img2_fusion = torch.cat([feat_b0, feat_b1, feat_b2], dim=1)
        return self.img2d_dnn(img2_fusion)


# Backward compatibility alias
Texture2DBackbone = TextureBackbone

__all__ = ["TextureBackbone", "Texture2DBackbone"]
