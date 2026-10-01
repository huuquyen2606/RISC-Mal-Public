"""1D Convolutional multi-view feature extractor backbone for byte streams.

Processes 1024-byte raw byte sequences using a 3-layer 1D CNN with MaxPooling,
followed by a 2-stage MLP projection down to 64 dimensions.
"""

import torch
import torch.nn as nn
from riscmal.backbones.common import DNNBlock


class Conv1DBackbone(nn.Module):
    """Backbone for 1D PE binary byte stream representation.

    Architecture:
        Input [B, 1024] (or [B, 1, 1024]) ->
        Conv1d(1, 256, k=3, p=1) -> BatchNorm1d(256) -> ReLU -> MaxPool1d(2) -> [B, 256, 512]
        Conv1d(256, 128, k=3, p=1) -> BatchNorm1d(128) -> ReLU -> MaxPool1d(2) -> [B, 128, 256]
        Conv1d(128, 64, k=3, p=1) -> BatchNorm1d(64) -> ReLU -> MaxPool1d(2) -> [B, 64, 128]
        Flatten -> [B, 8192]
        DNNBlock(8192, 128) ->
        DNNBlock(128, 64) ->
        Output [B, 64].
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 64,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.dropout_rate = dropout_rate

        self.conv1d_1 = nn.Conv1d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv1d_2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv1d_3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()

        self.img1d_dnn = nn.Sequential(
            DNNBlock(8192, 128, dropout_rate=dropout_rate),
            DNNBlock(128, out_features, dropout_rate=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 64-dimensional features from 1D byte stream.

        Args:
            x: Tensor of shape [B, 1024] or [B, 1, 1024].

        Returns:
            Tensor of shape [B, 64].
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1024] -> [B, 1, 1024]

        # Stage 1
        x = self.conv1d_1(x)
        if self.training and x.size(0) == 1:
            self.bn1.eval()
            x = self.relu(self.bn1(x))
            self.bn1.train()
        else:
            x = self.relu(self.bn1(x))
        x = self.pool(x)

        # Stage 2
        x = self.conv1d_2(x)
        if self.training and x.size(0) == 1:
            self.bn2.eval()
            x = self.relu(self.bn2(x))
            self.bn2.train()
        else:
            x = self.relu(self.bn2(x))
        x = self.pool(x)

        # Stage 3
        x = self.conv1d_3(x)
        if self.training and x.size(0) == 1:
            self.bn3.eval()
            x = self.relu(self.bn3(x))
            self.bn3.train()
        else:
            x = self.relu(self.bn3(x))
        x = self.pool(x)

        # Flatten and dense projection
        x = torch.flatten(x, start_dim=1)
        return self.img1d_dnn(x)
