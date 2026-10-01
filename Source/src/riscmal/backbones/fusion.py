"""Multi-view feature extraction and fusion module.

Orchestrates all 5 specialized domain backbones (Header, Imports, Conv1D byte stream,
2D Texture ensemble, and Dynamic API Sequence), concatenates their 64-dimensional
representations into a 320-dimensional unified vector, and applies multi-layer
fusion projection down to the final 64-dimensional latent embedding.
"""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from riscmal.backbones.common import DNNBlock
from riscmal.backbones.conv1d import Conv1DBackbone
from riscmal.backbones.header import HeaderBackbone
from riscmal.backbones.imports import ImportsBackbone
from riscmal.backbones.sequence import APISequenceBackbone
from riscmal.backbones.texture import TextureBackbone


def _unpack_multiview_inputs(
    *args: Any,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper to flexibly extract the 5 modality tensors from dict, tuple, or positional args."""
    if len(args) == 1:
        first = args[0]
        if isinstance(first, Mapping):
            return (
                first["header"],
                first["imports"],
                first["img1d"],
                first["img2d"],
                first["apis"],
            )
        elif isinstance(first, (tuple, list)) and len(first) >= 5:
            return first[0], first[1], first[2], first[3], first[4]
    elif len(args) >= 5:
        return args[0], args[1], args[2], args[3], args[4]

    # Check kwargs
    if "header" in kwargs and "imports" in kwargs and "img1d" in kwargs and "img2d" in kwargs and "apis" in kwargs:
        return kwargs["header"], kwargs["imports"], kwargs["img1d"], kwargs["img2d"], kwargs["apis"]

    raise ValueError(
        "Invalid inputs for MultiViewFeatureExtractor. Expected dict/MultiViewBatch, "
        "5 positional tensors (header, imports, img1d, img2d, apis), or keyword arguments."
    )



class MultiViewFusion(nn.Module):
    """Multi-layer projection network fusing 320-d concatenated representations down to 64-d."""

    def __init__(
        self,
        in_features: int = 320,
        out_features: int = 64,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.net = nn.Sequential(
            DNNBlock(in_features, 320, dropout_rate=dropout_rate),
            DNNBlock(320, 256, dropout_rate=dropout_rate),
            DNNBlock(256, 128, dropout_rate=dropout_rate),
            nn.Linear(128, out_features),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects 320-d concatenated representations down to 64-d fused latent features."""
        return self.net(x)


class MultiViewFeatureExtractor(nn.Module):
    """Unified multi-view feature extractor and fusion network.

    Combines 5 modalities into a cohesive 64-dimensional feature embedding:
        1. PE-Header: [B, 4] -> [B, 64]
        2. PE-Imports: [B, 1000] -> [B, 64]
        3. Conv1D 1D-Image: [B, 1024] -> [B, 64]
        4. Texture 2D: [B, 3, 224, 224] -> [B, 64]
        5. API Calls Sequence: [B, 100] -> [B, 64]

    Fusion Architecture:
        Concatenate [B, 5 x 64 = 320] ->
        DNNBlock(320, 320) ->
        DNNBlock(320, 256) ->
        DNNBlock(256, 128) ->
        Linear(128, 64) ->
        ReLU() ->
        Output [B, 64].
    """

    def __init__(
        self,
        dropout_rate: float = 0.2,
        header_backbone: Optional[nn.Module] = None,
        imports_backbone: Optional[nn.Module] = None,
        conv1d_backbone: Optional[nn.Module] = None,
        texture_backbone: Optional[nn.Module] = None,
        sequence_backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        # Branch 1: PE Header
        self.header_net = header_backbone or HeaderBackbone(dropout_rate=dropout_rate)
        # Branch 2: PE Imports
        self.imports_net = imports_backbone or ImportsBackbone(dropout_rate=dropout_rate)
        # Branch 3: PE 1D Byte Stream
        self.conv1d_net = conv1d_backbone or Conv1DBackbone(dropout_rate=dropout_rate)
        # Branch 4: Multi-Scale Texture 2D
        self.texture_net = texture_backbone or TextureBackbone(dropout_rate=dropout_rate)
        # Branch 5: Dynamic API Sequence
        self.sequence_net = sequence_backbone or APISequenceBackbone(dropout_rate=dropout_rate)

        # Aliases for notebook backward-compatibility
        self.img1d_net = self.conv1d_net
        self.img2d_net = self.texture_net
        self.api_net = self.sequence_net

        # Final fusion layer (320 -> 320 -> 256 -> 128 -> 64)
        self.final_fusion = MultiViewFusion(in_features=320, out_features=64, dropout_rate=dropout_rate)

    # Convenience properties for contract and test compatibility
    @property
    def fc_header(self) -> HeaderBackbone:
        """Reference to PE-Header backbone branch."""
        return self.header_net

    @property
    def fc_imports(self) -> ImportsBackbone:
        """Reference to PE-Imports backbone branch."""
        return self.imports_net

    @property
    def conv1d(self) -> Conv1DBackbone:
        """Reference to PE 1D Conv byte stream backbone branch."""
        return self.conv1d_net

    @property
    def texture(self) -> TextureBackbone:
        """Reference to 2D Texture EfficientNet ensemble backbone branch."""
        return self.texture_net

    @property
    def api_embed(self) -> nn.Embedding:
        """Reference to API sequence token embedding layer."""
        return self.sequence_net.api_embedding

    @property
    def api_lstm(self) -> nn.LSTM:
        """Reference to API sequence primary LSTM layer."""
        return self.sequence_net.lstm1

    @property
    def fusion(self) -> MultiViewFusion:
        """Reference to multi-layer fusion projection block."""
        return self.final_fusion

    def train(self, mode: bool = True) -> "MultiViewFeatureExtractor":
        """Ensures frozen backbones (e.g. EfficientNet) remain in eval mode."""
        super().train(mode)
        self.texture_net.eval()
        return self

    def extract_individual_features(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Extracts 64-dimensional feature representations for each individual modality.

        Args:
            *args, **kwargs: Flexible inputs (dict, MultiViewBatch, or 5 tensors).

        Returns:
            Dictionary containing:
                "header": Tensor[B, 64]
                "imports": Tensor[B, 64]
                "img1d": Tensor[B, 64]
                "img2d": Tensor[B, 64]
                "apis": Tensor[B, 64]
        """
        h, imp, i1d, i2d, api = _unpack_multiview_inputs(*args, **kwargs)

        h_feat = self.header_net(h)
        imp_feat = self.imports_net(imp)
        i1d_feat = self.conv1d_net(i1d)
        i2d_feat = self.texture_net(i2d)
        api_feat = self.sequence_net(api)

        return {
            "header": h_feat,
            "imports": imp_feat,
            "img1d": i1d_feat,
            "img2d": i2d_feat,
            "apis": api_feat,
        }

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through all 5 branches followed by concatenation and fusion.

        Args:
            *args, **kwargs: Flexible inputs (dict, MultiViewBatch, or 5 tensors).

        Returns:
            Fused feature representation of shape [B, 64].
        """
        indiv = self.extract_individual_features(*args, **kwargs)

        combined = torch.cat(
            [
                indiv["header"],
                indiv["imports"],
                indiv["img1d"],
                indiv["img2d"],
                indiv["apis"],
            ],
            dim=1,
        )

        return self.final_fusion(combined)


__all__ = ["MultiViewFusion", "MultiViewFeatureExtractor", "_unpack_multiview_inputs"]
