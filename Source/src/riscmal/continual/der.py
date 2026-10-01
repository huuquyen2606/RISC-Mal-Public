"""Dynamically Expandable Representation (DER) Continual Learning Strategy."""

import copy
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from riscmal.memory.buffer import RehearsalMemoryManager
from .base import BaseContinualStrategy

FEATURE_DIM = 64
CLASSES_PER_TASK = 2


class DERStrategy(BaseContinualStrategy):
    """Dynamically Expandable Representation (DER) Strategy.
    
    Dynamically grows a dedicated feature extractor column for each incremental task.
    Previous columns are frozen to completely eliminate catastrophic forgetting, while
    a shared classifier head routes across the concatenated multi-column representation:
        h = [f_1(x); f_2(x); ...; f_T(x)] in R^{64 * T}.
    An auxiliary classifier head guides new representation learning.
    """

    def __init__(
        self,
        num_known_classes: int = 0,
        replay_ratio: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="DER", num_known_classes=num_known_classes, replay_ratio=replay_ratio, **kwargs)
        self.memory_manager = RehearsalMemoryManager(replay_ratio=replay_ratio)

    def adapt_architecture_before_task(self, device: Union[torch.device, str]) -> None:
        """Dynamically expands DER architecture before task training.

        Allocates a new trainable backbone column, freezes historical columns,
        and expands the unified classifier head and auxiliary head.

        Args:
            device: Target torch device for tensor allocation.
        """
        dev = torch.device(device) if isinstance(device, str) else device
        if self.model is None or not hasattr(self.model, "backbone"):
            return

        current_num_cols = len(self.model.backbone.backbones)
        new_num_cols = current_num_cols + 1
        new_feature_dim = FEATURE_DIM * new_num_cols
        num_new_classes = CLASSES_PER_TASK

        # 1. Freeze previous backbone columns
        for old_col in self.model.backbone.backbones:
            old_col.eval()
            for param in old_col.parameters():
                param.requires_grad_(False)

        # 2. Append new trainable backbone column
        new_backbone_col = copy.deepcopy(self.model.backbone.backbones[0])
        new_backbone_col.train()
        for param in new_backbone_col.parameters():
            param.requires_grad_(True)
        self.model.backbone.add_backbone_column(new_backbone_col)

        # 3. Expand unified classifier head in both input and output dimensions
        old_head = self.model.classifier_head
        old_out = old_head.out_features
        new_out = old_out + num_new_classes

        new_head = nn.Linear(new_feature_dim, new_out).to(dev)
        with torch.no_grad():
            old_in = old_head.in_features
            new_head.weight[:old_out, :old_in] = old_head.weight.clone()
            new_head.bias[:old_out].copy_(old_head.bias.clone())
            new_head.weight[:old_out, old_in:].fill_(0.0)

        self.model.classifier_head = new_head

        # 4. Expand auxiliary classifier head
        if hasattr(self.model, "aux_classifier"):
            self.model.aux_classifier = nn.Linear(FEATURE_DIM, num_new_classes + 1).to(dev)

        print(
            f"\n[DER] Architecture dynamically expanded: "
            f"{new_num_cols} columns ({new_feature_dim} dims) -> {new_out} classes."
        )

    def compute_additional_loss(
        self,
        model: nn.Module,
        features_64: torch.Tensor,
        outputs: torch.Tensor,
        batch: Any,
        current_task_id: int = 1,
    ) -> torch.Tensor:
        """Computes DER auxiliary classification loss on the newly allocated backbone column.

        Args:
            model: Current DER model.
            features_64: Multi-column concatenated features.
            outputs: Logits from main classifier head.
            batch: Current training mini-batch.
            current_task_id: Current incremental task identifier.

        Returns:
            Auxiliary cross-entropy loss scaled by 0.5.
        """
        if current_task_id == 1 or not hasattr(model, "aux_classifier"):
            return torch.tensor(0.0, device=features_64.device)

        if isinstance(batch, (tuple, list)):
            y = batch[-1].to(features_64.device)
            h, imp, i1d, i2d, api = [item.to(features_64.device) for item in batch[:5]]
        elif isinstance(batch, dict):
            y = batch["label"].to(features_64.device)
            h, imp, i1d, i2d, api = (
                batch["header"].to(features_64.device),
                batch["imports"].to(features_64.device),
                batch["img1d"].to(features_64.device),
                batch["img2d"].to(features_64.device),
                batch["apis"].to(features_64.device),
            )
        else:
            return torch.tensor(0.0, device=features_64.device)

        feat_new = model.backbone.backbones[-1](h, imp, i1d, i2d, api)
        aux_logits = model.aux_classifier(feat_new)

        # Auxiliary loss: map new class labels to 0..CLASSES_PER_TASK - 1, others to CLASSES_PER_TASK
        min_class = self._known_classes
        max_class = self._known_classes + CLASSES_PER_TASK - 1
        aux_targets = torch.where(
            (y >= min_class) & (y <= max_class),
            y - min_class,
            torch.full_like(y, CLASSES_PER_TASK),
        )

        loss_aux = F.cross_entropy(aux_logits, aux_targets.long())
        return 0.5 * loss_aux

    def update_memory_after_task(
        self,
        train_loader: Any,
        new_classes: Union[List[int], int],
    ) -> None:
        """Post-task lifecycle hook: updates class-proportional rehearsal memory buffer.

        Args:
            train_loader: DataLoader for the completed task.
            new_classes: Class IDs introduced in the completed task.
        """
        num_new = new_classes if isinstance(new_classes, int) else len(new_classes)
        self._known_classes += num_new
        if self.memory_manager is not None:
            classes_list = list(range(self._known_classes - num_new, self._known_classes))
            self.memory_manager.update_memory_after_task(train_loader, classes_list)
        print(f"[DER] Completed Task. Total known classes: {self._known_classes}")
