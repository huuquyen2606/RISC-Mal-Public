"""Generative Classifier (GC) Continual Learning Strategy."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from riscmal.models.vae import ClassVAE, GCHead
from .base import BaseContinualStrategy


class GCStrategy(BaseContinualStrategy):
    """Generative Classifier (GC) Strategy.

    Eliminates catastrophic forgetting by training a separate, independent ClassVAE
    for each newly introduced class. Historical class VAEs are frozen and retained.
    At inference time, classification logits are determined by the Evidence Lower Bound
    (negative reconstruction error minus KL divergence) per class VAE:
        score_c(x) = ELBO_c(x) = E_{q_c}[log p_c(x|z)] - D_{KL}(q_c(z|x) || p(z))
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: Union[torch.device, str] = "cpu",
        config: Optional[Dict[str, Any]] = None,
        vae_hidden_dim: int = 128,
        vae_z_dim: int = 32,
        vae_epochs: int = 20,
        vae_lr: float = 1e-3,
        vae_batch_size: int = 64,
        kl_weight: float = 1.0,
        num_known_classes: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            device=device,
            config=config,
            name="GC",
            num_known_classes=num_known_classes,
            **kwargs,
        )
        self.vae_hidden_dim: int = int(self.config.get("vae_hidden_dim", vae_hidden_dim))
        self.vae_z_dim: int = int(self.config.get("vae_z_dim", vae_z_dim))
        self.vae_epochs: int = int(self.config.get("vae_epochs", vae_epochs))
        self.vae_lr: float = float(self.config.get("vae_lr", vae_lr))
        self.vae_batch_size: int = int(self.config.get("vae_batch_size", vae_batch_size))
        self.kl_weight: float = float(self.config.get("kl_weight", kl_weight))

        self._gc_head: Optional[GCHead] = None

    def adapt_architecture_before_task(self, device: Union[torch.device, str]) -> None:
        """Expands the GC head or underlying model classification head for novel classes."""
        dev = torch.device(device) if isinstance(device, str) else device
        num_new = len(self.current_task_classes) if self.current_task_classes else 2

        if self._gc_head is not None:
            self._gc_head.expand_classes(num_new, device=dev)
        elif self.model is not None and hasattr(self.model, "expand_classes"):
            self.model.expand_classes(num_new, device=dev)

    def compute_additional_loss(
        self,
        model: nn.Module,
        features_64: torch.Tensor,
        outputs: torch.Tensor,
        batch: Any,
        current_task_id: int = 1,
    ) -> torch.Tensor:
        """Standard task training uses linear backup head (no auxiliary penalty during stream)."""
        return torch.tensor(0.0, device=features_64.device, requires_grad=True)

    def _extract_latent_features(
        self,
        train_loader: Any,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts latent features and labels using the current feature extractor."""
        feat_list: List[torch.Tensor] = []
        label_list: List[torch.Tensor] = []

        self.model.eval()
        with torch.no_grad():
            for raw_batch in train_loader:
                h, imp, i1d, i2d, api, y = self._unpack_batch(raw_batch)

                if hasattr(self.model, "backbone"):
                    features = self.model.backbone(h, imp, i1d, i2d, api)
                elif hasattr(self.model, "feature_extractor"):
                    features = self.model.feature_extractor(h, imp, i1d, i2d, api)
                else:
                    out = self.model(h, imp, i1d, i2d, api)
                    features = out[1] if isinstance(out, tuple) else out

                feat_list.append(features.detach().cpu())
                label_list.append(y.detach().cpu())

        return torch.cat(feat_list, dim=0), torch.cat(label_list, dim=0)

    def _train_class_vae(self, class_features: torch.Tensor, device: torch.device) -> ClassVAE:
        """Trains a dedicated ClassVAE on representations belonging to a single class."""
        feat_dim = class_features.shape[1]
        vae = ClassVAE(
            input_dim=feat_dim,
            hidden_dim=self.vae_hidden_dim,
            z_dim=self.vae_z_dim,
        ).to(device)

        optimizer = torch.optim.Adam(vae.parameters(), lr=self.vae_lr)
        dataset = TensorDataset(class_features)
        bs = min(self.vae_batch_size, max(len(class_features), 1))
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)

        vae.train()
        for epoch in range(1, self.vae_epochs + 1):
            for (x_batch,) in loader:
                x_b = x_batch.to(device)
                x_recon, mu, log_var = vae(x_b)

                recon_loss = F.mse_loss(x_recon, x_b, reduction="none").sum(dim=1).mean()
                kl_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
                loss = recon_loss + self.kl_weight * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        return vae

    def update_memory_after_task(
        self,
        train_loader: Any,
        current_task_id: int,
        device: Union[torch.device, str],
    ) -> None:
        """Extracts latent features, trains ClassVAEs for new classes, and integrates into GCHead."""
        dev = torch.device(device) if isinstance(device, str) else device
        num_new = len(self.current_task_classes) if self.current_task_classes else 2
        total_classes = self._known_classes + num_new

        all_features, all_labels = self._extract_latent_features(train_loader, dev)
        feat_dim = all_features.shape[1]

        # Initialize GCHead on first encounter
        if self._gc_head is None:
            self._gc_head = GCHead(initial_classes=total_classes, feature_dim=feat_dim).to(dev)
            if hasattr(self.model, "classifier_head") and hasattr(self.model.classifier_head, "weight"):
                with torch.no_grad():
                    min_out = min(self._gc_head.linear_backup.out_features, self.model.classifier_head.out_features)
                    self._gc_head.linear_backup.weight.data[:min_out] = self.model.classifier_head.weight.data[:min_out]
                    self._gc_head.linear_backup.bias.data[:min_out] = self.model.classifier_head.bias.data[:min_out]
            self.model.classifier_head = self._gc_head

        unique_classes = torch.unique(all_labels).tolist()
        for cls_id in unique_classes:
            cls_int = int(cls_id)
            cls_mask = all_labels == cls_int
            cls_feats = all_features[cls_mask]
            if len(cls_feats) > 0:
                vae = self._train_class_vae(cls_feats, dev)
                self._gc_head.register_vae(cls_int, vae)

        self._gc_head = self._gc_head.to(dev)
        self.model.classifier_head = self._gc_head
        self._known_classes = total_classes
        self.model.train()

    def get_extra_state(self) -> Dict[str, Any]:
        """Serializes GCHead and ClassVAE weights."""
        head_state = self._gc_head.state_dict() if self._gc_head is not None else None
        return {
            "vae_hidden_dim": self.vae_hidden_dim,
            "vae_z_dim": self.vae_z_dim,
            "gc_head_state": head_state,
        }

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        """Restores GCHead and ClassVAE weights."""
        self.vae_hidden_dim = int(state.get("vae_hidden_dim", self.vae_hidden_dim))
        self.vae_z_dim = int(state.get("vae_z_dim", self.vae_z_dim))
        head_state = state.get("gc_head_state", None)
        if head_state is not None and self.model is not None:
            if self._gc_head is None:
                feat_dim = getattr(self.model, "feat_dim", 64)
                self._gc_head = GCHead(initial_classes=self._known_classes, feature_dim=feat_dim).to(self.device)
            self._gc_head.load_state_dict(head_state)
            self.model.classifier_head = self._gc_head
