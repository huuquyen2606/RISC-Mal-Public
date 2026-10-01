"""Dynamic multi-column and expanding architectures for DER and FOSTER.

Implements:
1. DER (Dynamically Expandable Representation):
   - DERBackbone: Column-growing feature extractor (64 * T dimensions).
   - DERClassifier: Multi-column classifier with auxiliary head for feature isolation.
2. FOSTER:
   - FOSTERDynamicBackbone: Dynamic multi-column backbone with old/new feature decoupling.
   - FOSTERDynamicHead: Dynamic classifier head supporting Feature Boosting snapshotting.
   - build_compression_student: Compact student generation for Feature Compression.
   - kd_loss and bkd_loss: Ordinary and Balanced Knowledge Distillation formulations.
"""

import copy
from typing import Any, Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_DIM = 64
DEFAULT_CLASSES_PER_TASK = 2


# =============================================================================
# 1. DER ARCHITECTURE COMPONENTS
# =============================================================================

class DERBackbone(nn.Module):
    """Dynamically Expandable Representation (DER) multi-column backbone.

    Each incremental task allocates an independent feature extractor column (64-d).
    Output representation is the concatenation of all column outputs [B, 64 * T].
    """

    def __init__(self, base_feature_extractor: nn.Module) -> None:
        super().__init__()
        self.backbones = nn.ModuleList([base_feature_extractor])

    @property
    def feature_dim(self) -> int:
        """Total feature dimensionality after concatenating all columns."""
        return FEATURE_DIM * len(self.backbones)

    def add_backbone_column(self, new_column: nn.Module) -> None:
        """Appends a new backbone column for novel task learning."""
        self.backbones.append(new_column)

    def freeze_old_columns(self) -> None:
        """Freezes historical backbone columns to preserve existing representations."""
        for col in self.backbones[:-1]:
            col.eval()
            for param in col.parameters():
                param.requires_grad_(False)

    def freeze_all_columns(self) -> None:
        """Freezes all currently registered backbone columns."""
        for col in self.backbones:
            col.eval()
            for param in col.parameters():
                param.requires_grad_(False)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Extracts features across all columns and concatenates them."""
        features = [bb(*args, **kwargs) for bb in self.backbones]
        return torch.cat(features, dim=1)  # [B, 64 * T]


class DERClassifier(nn.Module):
    """Full DER classifier holding a multi-column backbone, main head, and aux head.

    Args:
        base_feature_extractor: Backbone column for the initial task.
        initial_classes: Number of initial classes (default: 2).
        aux_classes: Number of classes for the auxiliary classifier (default: 3).
        feat_dim: Feature dimension per column (default: 64).
    """

    def __init__(
        self,
        base_feature_extractor: nn.Module,
        initial_classes: int = 2,
        aux_classes: int = DEFAULT_CLASSES_PER_TASK + 1,
        feat_dim: int = FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.backbone = DERBackbone(base_feature_extractor)
        self.classifier_head = nn.Linear(feat_dim, initial_classes)
        self.aux_classifier = nn.Linear(feat_dim, aux_classes)
        self.feat_dim = feat_dim
        self.num_classes = initial_classes

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through multi-column backbone and both classification heads.

        Returns:
            Tuple of (main_logits, aux_logits):
                main_logits: [B, current_classes]
                aux_logits: [B, aux_classes] from the novel column
        """
        features_all = self.backbone(*args, **kwargs)  # [B, 64 * T]
        logits = self.classifier_head(features_all)

        # Extract features solely from the novel (last) column for the auxiliary head
        feat_new = self.backbone.backbones[-1](*args, **kwargs)  # [B, 64]
        aux_logits = self.aux_classifier(feat_new)

        return logits, aux_logits

    def add_column(
        self,
        new_column: nn.Module,
        num_new_classes: int = DEFAULT_CLASSES_PER_TASK,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Expands DER architecture before a new incremental task.

        1. Freezes all existing backbone columns.
        2. Appends new_column to the multi-column backbone.
        3. Expands classifier_head across both column input dimensions and output class dimensions,
           zeroing cross-connections to prevent novel column noise from leaking into old classes.
        4. Re-initializes auxiliary classifier for novel classes.
        """
        # 1. Physically freeze all existing historical columns
        self.backbone.freeze_all_columns()
        new_column.train()
        for p in new_column.parameters():
            p.requires_grad_(True)
        self.backbone.add_backbone_column(new_column)

        num_cols = len(self.backbone.backbones)
        new_feat_dim = FEATURE_DIM * num_cols

        old_head = self.classifier_head
        old_out = old_head.out_features
        old_in = old_head.in_features
        new_out = old_out + num_new_classes

        new_head = nn.Linear(new_feat_dim, new_out).to(device)
        with torch.no_grad():
            # Copy old weights into top-left corner
            new_head.weight[:old_out, :old_in] = old_head.weight.clone()
            new_head.bias[:old_out] = old_head.bias.clone()
            # Zero out influence of novel column on historical classes
            new_head.weight[:old_out, old_in:].fill_(0.0)

        self.classifier_head = new_head
        self.num_classes = new_out
        self.aux_classifier = nn.Linear(FEATURE_DIM, num_new_classes + 1).to(device)


# =============================================================================
# 2. FOSTER ARCHITECTURE COMPONENTS
# =============================================================================

class FOSTERDynamicBackbone(nn.Module):
    """Dynamic multi-column backbone used during FOSTER Feature Boosting.

    Concatenates feature representations from previous (frozen) backbones and the
    current (trainable) backbone.
    """

    def __init__(self, first_backbone: nn.Module) -> None:
        super().__init__()
        self.backbones = nn.ModuleList([first_backbone])

    @property
    def out_dim(self) -> int:
        """Total concatenated feature dimensions."""
        return FEATURE_DIM * len(self.backbones)

    def add_backbone(self, new_backbone: nn.Module) -> None:
        """Adds a new backbone column for Feature Boosting."""
        self.backbones.append(new_backbone)

    def add_backbone_column(self, new_backbone: nn.Module) -> None:
        """Alias for add_backbone."""
        self.add_backbone(new_backbone)

    def freeze_old_backbones(self) -> None:
        """Freezes historical columns, leaving only the newly added column trainable."""
        for bb in self.backbones[:-1]:
            bb.eval()
            for p in bb.parameters():
                p.requires_grad_(False)

    def freeze_old_columns(self) -> None:
        """Alias for freeze_old_backbones."""
        self.freeze_old_backbones()

    def get_old_features(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Extracts concatenated features from historical columns only (for teacher distillation)."""
        assert len(self.backbones) > 1, "No historical backbone column exists at Task 1."
        with torch.no_grad():
            old_feats = [bb(*args, **kwargs) for bb in self.backbones[:-1]]
        return torch.cat(old_feats, dim=1)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass concatenating features across all columns."""
        feats = [bb(*args, **kwargs) for bb in self.backbones]
        return torch.cat(feats, dim=1)


class FOSTERDynamicHead(nn.Module):
    """Dynamic classification head for FOSTER with teacher snapshotting and auxiliary head."""

    def __init__(self, num_classes: int, feat_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.fc = nn.Linear(feat_dim, num_classes)
        self.old_fc: Optional[nn.Linear] = None
        self.fe_fc: Optional[nn.Linear] = None

    @property
    def weight(self) -> torch.Tensor:
        """Weight tensor of the primary linear classification head."""
        return self.fc.weight

    @property
    def bias(self) -> torch.Tensor:
        """Bias tensor of the primary linear classification head."""
        return self.fc.bias

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass projecting features to class logits."""
        return self.fc(features)

    def initialize_from_linear(self, source_fc: nn.Linear) -> None:
        """Copies weights and biases from an existing compact linear head."""
        if source_fc.in_features != self.feat_dim:
            raise ValueError(f"Expected in_features={self.feat_dim}, got {source_fc.in_features}")
        if source_fc.out_features != self.num_classes:
            raise ValueError(f"Class mismatch: source={source_fc.out_features}, target={self.num_classes}")

        with torch.no_grad():
            self.fc.weight.copy_(source_fc.weight)
            self.fc.bias.copy_(source_fc.bias)

    def expand_for_boosting(self, new_in_dim: int, new_num_classes: int) -> None:
        """Expands head for Feature Boosting and preserves a frozen teacher snapshot (old_fc)."""
        old_out = self.fc.out_features
        old_in = self.fc.in_features

        if old_out >= new_num_classes:
            raise ValueError(f"Invalid expansion: old_out={old_out}, new_num_classes={new_num_classes}")
        if new_in_dim < old_in:
            raise ValueError(f"Invalid feature expansion: old_in={old_in}, new_in_dim={new_in_dim}")

        # Snapshot current classifier as teacher BEFORE expanding to new classes
        self.old_fc = copy.deepcopy(self.fc)
        self.old_fc.eval()
        for p in self.old_fc.parameters():
            p.requires_grad_(False)

        # Build expanded linear head
        new_fc = nn.Linear(new_in_dim, new_num_classes).to(self.fc.weight.device)
        with torch.no_grad():
            new_fc.weight[:old_out, :old_in].copy_(self.fc.weight)
            new_fc.bias[:old_out].copy_(self.fc.bias)
            if new_in_dim > old_in:
                new_fc.weight[:old_out, old_in:].zero_()

        self.fc = new_fc
        self.num_classes = new_num_classes

        # Auxiliary classifier for Feature Boosting
        self.fe_fc = nn.Linear(FEATURE_DIM, new_num_classes).to(self.fc.weight.device)

    def get_old_logits(self, old_features: torch.Tensor) -> torch.Tensor:
        """Evaluates frozen teacher classification head on historical features."""
        if self.old_fc is None:
            raise RuntimeError("old_fc has not been initialized.")
        return self.old_fc(old_features)

    def get_fe_logits(self, new_features_64: torch.Tensor) -> torch.Tensor:
        """Evaluates auxiliary classifier on new backbone column features."""
        if self.fe_fc is None:
            raise RuntimeError("fe_fc has not been initialized.")
        return self.fe_fc(new_features_64)


# =============================================================================
# 3. FOSTER UTILITIES & DISTILLATION LOSSES
# =============================================================================

def kd_loss(
    pred_logits: torch.Tensor,
    soft_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Ordinary Knowledge Distillation (OKD) cross-entropy loss."""
    pred = F.log_softmax(pred_logits / temperature, dim=1)
    soft = F.softmax(soft_logits / temperature, dim=1)
    return -torch.mul(soft, pred).sum() / pred.shape[0]


def bkd_loss(
    pred_logits: torch.Tensor,
    soft_logits: torch.Tensor,
    temperature: float = 2.0,
    per_cls_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Balanced Knowledge Distillation (BKD) loss for FOSTER Feature Compression."""
    pred = F.log_softmax(pred_logits / temperature, dim=1)
    soft = F.softmax(soft_logits / temperature, dim=1)

    if per_cls_weights is not None:
        soft = soft * per_cls_weights.to(pred.device)
        soft = soft / soft.sum(dim=1, keepdim=True).clamp(min=1e-8)

    return -torch.mul(soft, pred).sum() / pred.shape[0]


def build_compression_student(
    teacher_model: nn.Module,
    backbone_factory: Callable[[], nn.Module],
    total_classes: int,
    known_classes: int,
    device: Union[torch.device, str] = "cpu",
) -> Tuple[nn.Module, nn.Linear]:
    """Constructs a compact student model initialized from the teacher's compact components.

    Used during FOSTER Feature Compression phase to reduce multi-column capacity
    back to a single compact backbone.
    """
    snet_backbone = backbone_factory().to(device)
    snet_head = nn.Linear(FEATURE_DIM, total_classes).to(device)

    # Initialize backbone from column 0 of teacher
    old_compact_backbone = teacher_model.backbone.backbones[0]
    snet_backbone.load_state_dict(old_compact_backbone.state_dict())

    old_fc = getattr(teacher_model.classifier_head, "old_fc", None)
    if old_fc is not None:
        with torch.no_grad():
            snet_head.weight[:known_classes].copy_(old_fc.weight)
            snet_head.bias[:known_classes].copy_(old_fc.bias)

    return snet_backbone, snet_head



class FOSTERClassifier(nn.Module):
    """Unified model for FOSTER continual learning."""

    def __init__(self, base_feature_extractor: nn.Module, initial_classes: int = 2):
        super().__init__()
        self.backbone = FOSTERDynamicBackbone(base_feature_extractor)
        self.classifier_head = nn.Linear(FEATURE_DIM, initial_classes)

    def forward(self, *args, **kwargs):
        """Forward pass through dynamic backbone and classifier head returning (logits, feats)."""
        if len(args) == 1:
            x = args[0]
            if isinstance(x, (dict, MultiViewBatch)):
                h, imp, i1d, i2d, api = x["header"], x["imports"], x["img1d"], x["img2d"], x["apis"]
            elif isinstance(x, (tuple, list)):
                h, imp, i1d, i2d, api = x[:5]
            else:
                return self.classifier_head(self.backbone(x))
        elif len(args) >= 5:
            h, imp, i1d, i2d, api = args[:5]
        else:
            raise ValueError(f"Invalid input to FOSTERClassifier: {len(args)} arguments")

        feats = self.backbone(h, imp, i1d, i2d, api)
        logits = self.classifier_head(feats)
        return logits, feats

