"""Learning without Forgetting (LwF) Continual Learning Strategy."""

import copy
from typing import Any, Dict, List, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseContinualStrategy


class LwFStrategy(BaseContinualStrategy):
    """Learning without Forgetting (LwF) Strategy.

    Protects historical task representations without exemplar storage by snapshotting
    a frozen teacher model before task transitions and enforcing knowledge distillation
    over previous class logits:
        L_{LwF} = L_{CE} + lambda * T^2 * D_{KL}(sigma(z_{old}^{student} / T) || sigma(z_{old}^{teacher} / T))

    Hyperparameters:
        T: Softmax distillation temperature (default: 2.0).
        lambda_distill: Loss balancing scalar for distillation penalty (default: 3.0).
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: Union[torch.device, str] = "cpu",
        config: Optional[Dict[str, Any]] = None,
        temperature: float = 2.0,
        lambda_distill: float = 3.0,
        num_known_classes: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            device=device,
            config=config,
            name="LwF",
            num_known_classes=num_known_classes,
            **kwargs,
        )
        self.T: float = float(self.config.get("temperature", self.config.get("T", temperature)))
        self.lambda_distill: float = float(
            self.config.get("lambda_distill", self.config.get("lamda", lambda_distill))
        )
        self.old_model: Optional[nn.Module] = None

    def adapt_architecture_before_task(self, device: Union[torch.device, str]) -> None:
        """Expands the student classifier head for incoming task classes."""
        if self.model is None:
            return

        num_new = len(self.current_task_classes) if self.current_task_classes else 2
        if hasattr(self.model, "expand_classes"):
            self.model.expand_classes(num_new, device=device)

    def compute_additional_loss(
        self,
        model: nn.Module,
        features_64: torch.Tensor,
        outputs: torch.Tensor,
        batch: Any,
        current_task_id: int = 1,
    ) -> torch.Tensor:
        """Computes knowledge distillation loss against the frozen snapshot teacher."""
        if self.old_model is None:
            return torch.tensor(0.0, device=features_64.device, requires_grad=True)

        h, imp, i1d, i2d, api, _ = self._unpack_batch(batch)

        with torch.no_grad():
            if hasattr(self.old_model, "backbone") and hasattr(self.old_model, "classifier_head"):
                old_features = self.old_model.backbone(h, imp, i1d, i2d, api)
                old_logits = self.old_model.classifier_head(old_features)
            else:
                out = self.old_model(h, imp, i1d, i2d, api)
                old_logits = out[0] if isinstance(out, tuple) else out

        num_old_classes = old_logits.shape[1]
        if num_old_classes == 0:
            return torch.tensor(0.0, device=features_64.device, requires_grad=True)

        student_old_logits = outputs[:, :num_old_classes]
        loss_kd = self._kd_loss(student_old_logits, old_logits, self.T)
        return self.lambda_distill * loss_kd

    @staticmethod
    def _kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
        """Calculates temperature-scaled Kullback-Leibler divergence distillation loss."""
        log_pred = F.log_softmax(student_logits / T, dim=1)
        soft_tgt = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(log_pred, soft_tgt, reduction="batchmean") * (T * T)

    def update_memory_after_task(
        self,
        train_loader: Any,
        current_task_id: int,
        device: Union[torch.device, str],
    ) -> None:
        """Snapshots current model as frozen teacher and updates class tracking."""
        dev = torch.device(device) if isinstance(device, str) else device
        num_new = len(self.current_task_classes) if self.current_task_classes else 2

        self.old_model = copy.deepcopy(self.model)
        self.old_model.to(dev)
        self.old_model.eval()
        for param in self.old_model.parameters():
            param.requires_grad_(False)

        self._known_classes += num_new

    def get_extra_state(self) -> Dict[str, Any]:
        """Serializes teacher model weights for checkpoint restoration."""
        old_state = self.old_model.state_dict() if self.old_model is not None else None
        return {
            "T": self.T,
            "lambda_distill": self.lambda_distill,
            "old_model_state_dict": old_state,
        }

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        """Restores teacher model weights from checkpoint state."""
        self.T = float(state.get("T", self.T))
        self.lambda_distill = float(state.get("lambda_distill", self.lambda_distill))
        old_state = state.get("old_model_state_dict", None)
        if old_state is not None and self.model is not None:
            self.old_model = copy.deepcopy(self.model)
            self.old_model.load_state_dict(old_state)
            self.old_model.eval()
            for p in self.old_model.parameters():
                p.requires_grad_(False)
