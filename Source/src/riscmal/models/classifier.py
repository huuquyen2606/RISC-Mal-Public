"""Unified multi-view classifier model with norm-matched output-only expansion.

Implements the central model architecture for RISC-Mal and static continual learning
baselines, featuring decoupled feature extraction and dynamic class-incremental
head expansion using exact Euclidean norm matching.
"""

from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn

from riscmal.backbones.fusion import MultiViewFeatureExtractor


class MalwareMultiViewClassifier(nn.Module):
    """Unified multi-view classifier for incremental malware detection.

    Consists of:
        1. A multi-view feature extractor (backbone) producing 64-dimensional embeddings.
        2. A dynamic linear classification head mapping 64 features to class logits.

    Formula for novel class weight vector initialization (norm-matched expansion):
        w_new = r_bar * (u_k / ||u_k||_2)
        b_new = 0.0
    where r_bar = (1 / C_old) * sum_{c=1}^{C_old} ||w_c||_2.

    Args:
        feature_extractor: Optional backbone module. If None, instantiates a default
            MultiViewFeatureExtractor.
        initial_classes: Number of initial classes for Task 1 (default: 2).
        feat_dim: Dimensionality of latent feature representations (default: 64).
        **kwargs: Backwards compatibility options (e.g. backbone_block).
    """

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        initial_classes: int = 2,
        feat_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Support backward-compatible kwargs
        if feature_extractor is None:
            feature_extractor = kwargs.get("backbone_block", None)

        if feature_extractor is None:
            feature_extractor = MultiViewFeatureExtractor()

        self.feature_extractor = feature_extractor
        self.backbone = feature_extractor  # Standard alias
        self.feat_dim = feat_dim
        self.num_classes = initial_classes

        # Dynamic classification head
        self.classifier_head = nn.Linear(feat_dim, initial_classes)

    def extract_features(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Extracts 64-dimensional feature representations without computing classification logits."""
        return self.feature_extractor(*args, **kwargs)

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through feature extractor and dynamic classifier head.

        Args:
            *args, **kwargs: Input data as dictionary/MultiViewBatch, tuple of 5 tensors,
                or individual keyword/positional arguments.

        Returns:
            Tuple of (logits, features):
                logits: Tensor[B, num_classes]
                features: Tensor[B, 64]
        """
        features = self.feature_extractor(*args, **kwargs)
        logits = self.classifier_head(features)
        return logits, features

    def expand_classes(
        self,
        num_new_classes: int,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Dynamically expands the classification head using exact norm-matched initialization.

        Preserves existing class weights and biases perfectly in the [:C_old] slice.
        Initializes novel class weight vectors with unit Gaussian samples projected onto
        the unit sphere and scaled to the average Euclidean norm of existing classes (r_bar).
        Novel class biases are initialized to zero.

        Args:
            num_new_classes: Number of novel classes to append to the classifier.
            device: Target torch device for tensor allocation.
        """
        if num_new_classes <= 0:
            return

        old_head = self.classifier_head
        old_classes = old_head.out_features
        new_total_classes = old_classes + num_new_classes
        feat_dim = old_head.in_features

        old_weights = old_head.weight.data
        old_bias = old_head.bias.data

        # 1. Compute average Euclidean norm of existing class weight vectors
        r_bar = float(torch.norm(old_weights, dim=1).mean().item())

        # 2. Sample random vectors on the unit sphere and scale by r_bar
        u_k = torch.randn(num_new_classes, feat_dim, device=device)
        u_k_norm = u_k / torch.norm(u_k, dim=1, keepdim=True).clamp(min=1e-8)
        w_new = r_bar * u_k_norm

        # 3. Novel class biases are initialized to 0.0
        new_biases = torch.zeros(num_new_classes, device=device)

        # 4. Allocate expanded linear layer and copy weights
        expanded_head = nn.Linear(feat_dim, new_total_classes).to(device)
        with torch.no_grad():
            expanded_head.weight.data[:old_classes] = old_weights.to(device)
            expanded_head.weight.data[old_classes:] = w_new.to(device)
            expanded_head.bias.data[:old_classes] = old_bias.to(device)
            expanded_head.bias.data[old_classes:] = new_biases.to(device)

        self.classifier_head = expanded_head
        self.num_classes = new_total_classes
