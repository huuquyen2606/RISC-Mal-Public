"""Unified multi-view feature extractor module."""

from riscmal.backbones.fusion import (
    MultiViewFeatureExtractor,
    MultiViewFusion,
    _unpack_multiview_inputs,
)

__all__ = [
    "MultiViewFeatureExtractor",
    "MultiViewFusion",
    "_unpack_multiview_inputs",
]
