"""Model architectures for RISC-Mal and continual learning baselines.

Provides:
- MalwareMultiViewClassifier: Central multi-view classifier with norm-matched expansion.
- DERBackbone & DERClassifier: Dynamically Expandable Representation models.
- FOSTERDynamicBackbone & FOSTERDynamicHead: Dynamic boosting/compression models.
- LatentCVAE: Conditional VAE for Latent Generative Replay (LGR).
- ConditionalVAE: Expandable GMM-prior VAE for Brain-Inspired Replay (BI-R).
- ClassVAE & GCHead: Class-specific generative classifier for GC.
"""

from riscmal.models.classifier import MalwareMultiViewClassifier
from riscmal.models.dynamic import (
    DERBackbone,
    DERClassifier,
    FOSTERClassifier,
    FOSTERDynamicBackbone,
    FOSTERDynamicHead,
    bkd_loss,
    build_compression_student,
    kd_loss,
)
from riscmal.models.vae import (
    ClassVAE,
    ConditionalVAE,
    GCHead,
    LatentCVAE,
)

__all__ = [
    "MalwareMultiViewClassifier",
    "DERBackbone",
    "DERClassifier",
    "FOSTERClassifier",
    "FOSTERDynamicBackbone",
    "FOSTERDynamicHead",
    "build_compression_student",
    "kd_loss",
    "bkd_loss",
    "LatentCVAE",
    "ConditionalVAE",
    "ClassVAE",
    "GCHead",
]

