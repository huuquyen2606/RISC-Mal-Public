"""Common neural network building blocks for RISC-Mal multi-view backbones.

Implements standard dense blocks (Linear -> ReLU -> BatchNorm1d -> Dropout)
with robust handling for single-sample evaluation and boundary shapes.
"""

from typing import Optional
import torch
import torch.nn as nn


class DNNBlock(nn.Module):
    """Dense block consisting of Linear -> ReLU -> BatchNorm1d -> Dropout.

    Architecture:
        1. Linear(in_features, out_features)
        2. ReLU()
        3. BatchNorm1d(out_features)
        4. Dropout(p=dropout_rate)

    Args:
        in_features: Dimensionality of the input tensor.
        out_features: Dimensionality of the output tensor.
        dropout_rate: Dropout probability (default: 0.2).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.BatchNorm1d(out_features),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Linear, ReLU, BatchNorm1d, and Dropout.

        Gracefully handles single-sample batches (B=1) during training by
        temporarily evaluating running stats in BatchNorm1d to prevent PyTorch crashes.
        """
        if self.training and x.size(0) == 1:
            bn = self.block[2]
            was_training = bn.training
            bn.eval()
            out = self.block(x)
            if was_training:
                bn.train()
            return out
        return self.block(x)
