"""Dynamic API Call Sequence multi-view feature extractor backbone.

Processes variable sequence length API invocations (default max length 100)
using an Embedding layer, 3-stage unidirectional LSTM hierarchy,
Self-Attention, Global Average Pooling across the temporal axis, and a DNNBlock.
"""

import torch
import torch.nn as nn
from riscmal.backbones.common import DNNBlock


class APISequenceBackbone(nn.Module):
    """Backbone for dynamic API invocation sequence representation.

    Architecture:
        Input [B, 100] ->
        Embedding(101, 128) -> [B, 100, 128] ->
        LSTM(128 -> 512, batch_first=True) -> [B, 100, 512] ->
        LSTM(512 -> 256, batch_first=True) -> [B, 100, 256] ->
        LSTM(256 -> 128, batch_first=True) -> [B, 100, 128] ->
        MultiheadAttention(128, num_heads=1, batch_first=True) -> [B, 100, 128] ->
        Global Average Pooling (temporal mean across dim=1) -> [B, 128] ->
        DNNBlock(128, 64) ->
        Output [B, 64].
    """

    def __init__(
        self,
        num_embeddings: int = 101,
        embedding_dim: int = 128,
        out_features: int = 64,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.out_features = out_features
        self.dropout_rate = dropout_rate

        self.api_embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

        self.lstm1 = nn.LSTM(128, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, batch_first=True)
        self.lstm3 = nn.LSTM(256, 128, batch_first=True)

        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=1,
            batch_first=True,
        )

        self.api_dnn = DNNBlock(128, out_features, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 64-dimensional features from dynamic API call tokens.

        Args:
            x: LongTensor of shape [B, 100] (or similar sequence length).

        Returns:
            Tensor of shape [B, 64].
        """
        # Ensure integers within vocabulary bounds [0, num_embeddings - 1]
        x_api = self.api_embedding(x.long().clamp(0, self.num_embeddings - 1))

        # Hierarchical LSTM sequence encoding
        x_api, _ = self.lstm1(x_api)
        x_api, _ = self.lstm2(x_api)
        x_api, _ = self.lstm3(x_api)

        # Multi-head Self-Attention
        att_out, _ = self.attention(x_api, x_api, x_api)

        # Global Average Pooling along sequence length dimension
        api_gap = torch.mean(att_out, dim=1)

        # Final projection to 64 dimensions
        return self.api_dnn(api_gap)
