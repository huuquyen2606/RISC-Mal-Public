"""PE-Header multi-view feature extractor backbone.

Maps 4 raw PE header fields (SizeOfData, VirtualSize, Entropy, Characteristics)
to a 64-dimensional latent representation through a 5-layer MLP hierarchy.
"""

import torch
import torch.nn as nn
from riscmal.backbones.common import DNNBlock


class HeaderBackbone(nn.Module):
    """Backbone for PE-Header feature extraction.

    Architecture:
        Input [B, 4] ->
        DNNBlock(4, 1000) ->
        DNNBlock(1000, 750) ->
        DNNBlock(750, 500) ->
        DNNBlock(500, 250) ->
        DNNBlock(250, 64) ->
        Output [B, 64].
    """

    def __init__(
        self,
        in_features: int = 4,
        out_features: int = 64,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate

        self.net = nn.Sequential(
            DNNBlock(in_features, 1000, dropout_rate=dropout_rate),
            DNNBlock(1000, 750, dropout_rate=dropout_rate),
            DNNBlock(750, 500, dropout_rate=dropout_rate),
            DNNBlock(500, 250, dropout_rate=dropout_rate),
            DNNBlock(250, out_features, dropout_rate=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 64-dimensional features from PE-Header vector.

        Args:
            x: Tensor of shape [B, 4] containing normalized header fields.

        Returns:
            Tensor of shape [B, 64].
        """
        return self.net(x)


# Backward compatibility alias
PEHeaderBackbone = HeaderBackbone

__all__ = ["HeaderBackbone", "PEHeaderBackbone"]
