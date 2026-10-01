"""Online Elastic Weight Consolidation (EWC) Continual Learning Strategy."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from .base import BaseContinualStrategy


class EWCStrategy(BaseContinualStrategy):
    """Online Elastic Weight Consolidation (EWC) Strategy.

    Penalizes deviations from consolidated historical parameter weights using an online
    accumulated diagonal Fisher Information Matrix:
        L_{EWC} = L_{CE} + (lambda / 2) * sum_i F_i * (theta_i - theta_i^*)^2

    Features:
        - Online Fisher accumulation across multiple sequential tasks.
        - Fisher ceiling clamping (fishermax=1e-4) to prevent penalty explosion.
        - Dimension-aware weight slicing for dynamically expanded classifier heads.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: Union[torch.device, str] = "cpu",
        config: Optional[Dict[str, Any]] = None,
        lamda: float = 1000.0,
        fishermax: float = 0.0001,
        lr_fisher: float = 1e-4,
        num_known_classes: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            device=device,
            config=config,
            name="EWC",
            num_known_classes=num_known_classes,
            **kwargs,
        )
        self.lamda: float = float(self.config.get("lamda", self.config.get("lambda_ewc", lamda)))
        self.fishermax: float = float(self.config.get("fishermax", fishermax))
        self.lr_fisher: float = float(self.config.get("lr_fisher", lr_fisher))

        self.fisher: Optional[Dict[str, torch.Tensor]] = None
        self.mean: Optional[Dict[str, torch.Tensor]] = None

    def adapt_architecture_before_task(self, device: Union[torch.device, str]) -> None:
        """Expands the classifier head before learning novel classes."""
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
        """Computes quadratic EWC parameter regularization penalty."""
        if self.fisher is None or self.mean is None:
            return torch.tensor(0.0, device=features_64.device, requires_grad=True)

        loss_ewc = self._compute_ewc_penalty(model, device=features_64.device)
        return self.lamda * loss_ewc

    def _compute_ewc_penalty(self, model: nn.Module, device: torch.device) -> torch.Tensor:
        """Calculates quadratic drift penalty between current weights and target mean."""
        loss = torch.tensor(0.0, device=device)
        for n, p in model.named_parameters():
            if n in self.fisher and n in self.mean:
                mean_n = self.mean[n].to(device)
                fish_n = self.fisher[n].to(device)

                # Accommodate expanding output dimensions safely
                min_len = min(p.shape[0], mean_n.shape[0], fish_n.shape[0])
                p_slice = p[:min_len]
                m_slice = mean_n[:min_len]
                f_slice = fish_n[:min_len]

                loss = loss + torch.sum(f_slice * (p_slice - m_slice).pow(2)) / 2.0

        return loss

    def _get_fisher_diagonal(self, train_loader: Any, device: Union[torch.device, str]) -> Dict[str, torch.Tensor]:
        """Approximates the diagonal elements of the empirical Fisher Information Matrix."""
        dev = torch.device(device) if isinstance(device, str) else device
        fisher: Dict[str, torch.Tensor] = {
            n: torch.zeros(p.shape, device=dev)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.train()
        num_batches = 0

        for raw_batch in train_loader:
            self.model.zero_grad()
            h, imp, i1d, i2d, api, y = self._unpack_batch(raw_batch)

            if hasattr(self.model, "backbone") and hasattr(self.model, "classifier_head"):
                features_64 = self.model.backbone(h, imp, i1d, i2d, api)
                logits = self.model.classifier_head(features_64)
            else:
                out = self.model(h, imp, i1d, i2d, api)
                logits = out[0] if isinstance(out, tuple) else out

            loss = F.cross_entropy(logits, y.long())
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] = fisher[n] + p.grad.detach().pow(2)

            num_batches += 1

        if num_batches > 0:
            for n in fisher:
                fisher[n] = fisher[n] / float(num_batches)
                fisher[n] = torch.clamp(fisher[n], max=self.fishermax)

        return fisher

    def update_memory_after_task(
        self,
        train_loader: Any,
        current_task_id: int,
        device: Union[torch.device, str],
    ) -> None:
        """Calculates task Fisher diagonal, accumulates online, and saves parameter anchor mean."""
        dev = torch.device(device) if isinstance(device, str) else device
        num_new = len(self.current_task_classes) if self.current_task_classes else 2
        total_classes = self._known_classes + num_new

        # 1. Compute empirical Fisher diagonal for completed task
        new_fisher = self._get_fisher_diagonal(train_loader, dev)

        # 2. Online Fisher accumulation
        if self.fisher is None:
            self.fisher = {n: f.clone().detach() for n, f in new_fisher.items()}
        else:
            alpha = float(self._known_classes) / float(max(total_classes, 1))
            for n in new_fisher:
                if n in self.fisher:
                    old_len = min(self.fisher[n].shape[0], new_fisher[n].shape[0])
                    new_fisher[n][:old_len] = (
                        alpha * self.fisher[n][:old_len]
                        + (1.0 - alpha) * new_fisher[n][:old_len]
                    )
            self.fisher = {n: f.clone().detach() for n, f in new_fisher.items()}

        # 3. Snapshot current optimal parameters as anchor mean
        self.mean = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        # 4. Update known classes
        self._known_classes = total_classes

    def get_extra_state(self) -> Dict[str, Any]:
        """Serializes Fisher Information Matrix and mean parameter anchors."""
        fisher_cpu = {n: f.cpu() for n, f in self.fisher.items()} if self.fisher else None
        mean_cpu = {n: m.cpu() for n, m in self.mean.items()} if self.mean else None
        return {
            "lamda": self.lamda,
            "fishermax": self.fishermax,
            "fisher": fisher_cpu,
            "mean": mean_cpu,
        }

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        """Restores Fisher Information Matrix and mean parameter anchors."""
        self.lamda = float(state.get("lamda", self.lamda))
        self.fishermax = float(state.get("fishermax", self.fishermax))
        self.fisher = state.get("fisher", None)
        self.mean = state.get("mean", None)
