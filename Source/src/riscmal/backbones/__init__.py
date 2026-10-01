"""Multi-view backbone architectures for RISC-Mal.

Provides specialized neural network extractors for:
- PE-Header (MLP hierarchy)
- PE-Imports (binary indicator MLP hierarchy)
- Conv1D (1D byte stream convolution and pooling)
- Texture (multi-scale 2D byte texture ensemble)
- Dynamic API Sequence (Embedding + 3x LSTM + MultiheadAttention)
- MultiViewFeatureExtractor (unified 5-view fusion module)
"""

from riscmal.backbones.common import DNNBlock
from riscmal.backbones.conv1d import Conv1DBackbone
from riscmal.backbones.fusion import MultiViewFeatureExtractor, MultiViewFusion
from riscmal.backbones.header import HeaderBackbone, PEHeaderBackbone
from riscmal.backbones.imports import ImportsBackbone
from riscmal.backbones.sequence import APISequenceBackbone
from riscmal.backbones.texture import TextureBackbone, Texture2DBackbone

__all__ = [
    "DNNBlock",
    "HeaderBackbone",
    "PEHeaderBackbone",
    "ImportsBackbone",
    "Conv1DBackbone",
    "TextureBackbone",
    "Texture2DBackbone",
    "APISequenceBackbone",
    "MultiViewFeatureExtractor",
    "MultiViewFusion",
]
