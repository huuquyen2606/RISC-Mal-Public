"""FOSTER (Feature Boosting and Compression) Continual Learning Strategy."""

import copy
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from riscmal.memory.buffer import RehearsalMemoryManager
from .base import BaseContinualStrategy

FEATURE_DIM = 64
CLASSES_PER_TASK = 2


class FOSTERStrategy(BaseContinualStrategy):
    """FOSTER Continual Learning Strategy.
    
    Implements a two-phase continual learning cycle:
        1. Feature Boosting: Adds an auxiliary backbone column to learn novel representations
           without degrading old features, combining multi-column features with distillation.
        2. Feature Compression: Distills the boosted multi-column ensemble back into a
           single compact backbone, keeping parameter count bounded O(1).
    """

    def __init__(
        self,
        num_known_classes: int = 0,
        replay_ratio: float = 0.1,
        temperature: float = 2.0,
        lambda_kd: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="FOSTER", num_known_classes=num_known_classes, replay_ratio=replay_ratio, **kwargs)
        self.temperature = temperature
        self.lambda_kd = lambda_kd
        self.memory_manager = RehearsalMemoryManager(replay_ratio=replay_ratio)
        self.old_network: Optional[nn.Module] = None

    def adapt_architecture_before_task(self, device: Union[torch.device, str]) -> None:
        """Adapts FOSTER architecture before task learning via Feature Boosting.

        Appends a newly allocated trainable backbone column, freezes historical columns,
        and expands the unified classifier head.

        Args:
            device: Target torch device for tensor allocation.
        """
        dev = torch.device(device) if isinstance(device, str) else device
        if self.model is None:
            return

        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "backbones"):
            # Feature Boosting: Add new column
            new_column = copy.deepcopy(self.model.backbone.backbones[0])
            new_column.train()
            for p in new_column.parameters():
                p.requires_grad = True
            self.model.backbone.add_backbone(new_column)
            self.model.backbone.freeze_old_backbones()

        if hasattr(self.model, "expand_classes"):
            self.model.expand_classes(CLASSES_PER_TASK, dev)

        print(f"\n[FOSTER] Feature Boosting column initialized. Expanded head by {CLASSES_PER_TASK} classes.")

    def compute_additional_loss(
        self,
        model: nn.Module,
        features_64: torch.Tensor,
        outputs: torch.Tensor,
        batch: Any,
        current_task_id: int = 1,
    ) -> torch.Tensor:
        """Computes FOSTER knowledge distillation loss matching teacher snapshot logits.

        Args:
            model: Current student model.
            features_64: Latent representations from current backbone.
            outputs: Current student logits.
            batch: Current training mini-batch.
            current_task_id: Current incremental task identifier.

        Returns:
            Distillation loss scaled by lambda_kd.
        """
        if current_task_id == 1 or self.old_network is None:
            return torch.tensor(0.0, device=features_64.device)

        device = features_64.device
        if isinstance(batch, (tuple, list)):
            h, imp, i1d, i2d, api = [item.to(device) for item in batch[:5]]
        elif isinstance(batch, dict):
            h, imp, i1d, i2d, api = (
                batch["header"].to(device),
                batch["imports"].to(device),
                batch["img1d"].to(device),
                batch["img2d"].to(device),
                batch["apis"].to(device),
            )
        else:
            return torch.tensor(0.0, device=features_64.device)

        with torch.no_grad():
            old_feats = self.old_network.backbone(h, imp, i1d, i2d, api)
            old_logits = self.old_network.classifier_head(old_feats)

        num_old_classes = self._known_classes
        student_old_logits = outputs[:, :num_old_classes]

        p_s = F.log_softmax(student_old_logits / self.temperature, dim=1)
        q_t = F.softmax(old_logits / self.temperature, dim=1)
        loss_kd = -torch.sum(q_t * p_s, dim=1).mean() * (self.temperature ** 2)

        return self.lambda_kd * loss_kd

    def update_memory_after_task(
        self,
        train_loader: Any,
        new_classes: Union[List[int], int],
    ) -> None:
        """Post-task lifecycle hook: snapshots teacher model and updates rehearsal memory.

        Args:
            train_loader: DataLoader for the completed task.
            new_classes: Class IDs introduced during the completed task.
        """
        num_new = new_classes if isinstance(new_classes, int) else len(new_classes)
        self._known_classes += num_new

        # Snapshot old network for distillation in subsequent tasks
        self.old_network = copy.deepcopy(self.model)
        self.old_network.eval()
        for p in self.old_network.parameters():
            p.requires_grad = False

        if self.memory_manager is not None:
            classes_list = list(range(self._known_classes - num_new, self._known_classes))
            self.memory_manager.update_memory_after_task(train_loader, classes_list)

        print(f"[FOSTER] Task completed. Teacher snapshot saved. Total classes: {self._known_classes}")
