"""Naive Baseline Sequential Fine-Tuning Strategy (Lower Bound)."""

from typing import Any, Dict, List, Optional, Sequence, Union
import torch
import torch.nn as nn
from .base import BaseContinualStrategy


class BaselineStrategy(BaseContinualStrategy):
    """Naive Sequential Fine-Tuning Strategy (Lower Bound).

    Trained strictly with Cross-Entropy on current task samples with no memory replay
    or regularization mechanism to counter catastrophic forgetting.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: Union[torch.device, str] = "cpu",
        config: Optional[Dict[str, Any]] = None,
        num_known_classes: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            device=device,
            config=config,
            name="Baseline",
            num_known_classes=num_known_classes,
            **kwargs,
        )

    def adapt_architecture_before_task(self, device: Union[torch.device, str]) -> None:
        """Dynamically expands the classification head for novel classes."""
        if self.model is None:
            return

        num_new_classes = len(self.current_task_classes) if self.current_task_classes else 2
        if hasattr(self.model, "expand_classes"):
            self.model.expand_classes(num_new_classes, device=device)

    def compute_additional_loss(
        self,
        model: nn.Module,
        features_64: torch.Tensor,
        outputs: torch.Tensor,
        batch: Any,
        current_task_id: int = 1,
    ) -> torch.Tensor:
        """Baseline applies zero auxiliary penalty (pure cross-entropy)."""
        return torch.tensor(0.0, device=features_64.device, requires_grad=True)

    def update_memory_after_task(
        self,
        train_loader: Any,
        current_task_id: int,
        device: Union[torch.device, str],
    ) -> None:
        """Updates known class count without retaining exemplars or parameter snapshots."""
        num_new_classes = len(self.current_task_classes) if self.current_task_classes else 2
        self._known_classes += num_new_classes
