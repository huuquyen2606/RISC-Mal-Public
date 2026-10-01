"""Latent Generative Replay (LGR) Continual Learning Strategy."""

import copy
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from riscmal.models.vae import LatentCVAE
from .base import BaseContinualStrategy


class LGRStrategy(BaseContinualStrategy):
    """Latent Generative Replay (LGR) Strategy (Stoychev et al.).

    Partitions the network into Generator (G), Root (R - Backbone), and Top (T - Classifier Head).
    The Root backbone is frozen after Task 1 to anchor the latent feature space R(x).
    In incremental tasks, a conditional VAE generator synthesizes pseudo-features for historical
    classes which are labeled by the previous Top classifier head (T_old) and replayed to train
    the updated classifier head (T_new):
        L_{LGR} = L_{CE}(T_{new}(R(x_{new})), y_{new}) + L_{CE}(T_{new}(R'_{x_{old}}), T_{old}(R'_{x_{old}}))
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: Union[torch.device, str] = "cpu",
        config: Optional[Dict[str, Any]] = None,
        feature_dim: int = 64,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        num_classes: int = 6,
        vae_epochs: int = 30,
        vae_lr: float = 1e-3,
        replay_ratio: float = 0.1,
        num_known_classes: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            device=device,
            config=config,
            name="LGR",
            num_known_classes=num_known_classes,
            replay_ratio=replay_ratio,
            **kwargs,
        )
        self.feature_dim: int = int(self.config.get("feature_dim", feature_dim))
        self.latent_dim: int = int(self.config.get("latent_dim", latent_dim))
        self.hidden_dim: int = int(self.config.get("hidden_dim", hidden_dim))
        self.num_classes: int = int(self.config.get("num_classes", num_classes))
        self.vae_epochs: int = int(self.config.get("vae_epochs", vae_epochs))
        self.vae_lr: float = float(self.config.get("vae_lr", vae_lr))

        self.vae: Optional[LatentCVAE] = None
        self.old_vae: Optional[LatentCVAE] = None
        self.old_top: Optional[nn.Module] = None
        self.seen_classes: List[int] = []
        self.root_frozen: bool = False

    def _freeze_root(self, model: nn.Module) -> None:
        """Locks the Root feature extractor to maintain invariant latent representations."""
        if not self.root_frozen and model is not None:
            if hasattr(model, "backbone"):
                model.backbone.eval()
                for param in model.backbone.parameters():
                    param.requires_grad_(False)
            elif hasattr(model, "feature_extractor"):
                model.feature_extractor.eval()
                for param in model.feature_extractor.parameters():
                    param.requires_grad_(False)
            self.root_frozen = True

    def adapt_architecture_before_task(self, device: Union[torch.device, str]) -> None:
        """Freezes Root backbone after Task 1 and dynamically expands Top classifier head."""
        dev = torch.device(device) if isinstance(device, str) else device
        num_new = len(self.current_task_classes) if self.current_task_classes else 2

        if self.current_task_id > 1:
            self._freeze_root(self.model)

        if hasattr(self.model, "expand_classes"):
            self.model.expand_classes(num_new, device=dev)

    def compute_additional_loss(
        self,
        model: nn.Module,
        features_64: torch.Tensor,
        outputs: torch.Tensor,
        batch: Any,
        current_task_id: int = 1,
    ) -> torch.Tensor:
        """Samples pseudo-features from old VAE, labels via old head, and computes replay CE loss."""
        if current_task_id == 1 or self.old_vae is None or self.old_top is None:
            return torch.tensor(0.0, device=features_64.device, requires_grad=True)

        old_classes = [c for c in self.seen_classes if c not in self.current_task_classes]
        if not old_classes:
            return torch.tensor(0.0, device=features_64.device, requires_grad=True)

        device = features_64.device
        batch_size = features_64.shape[0]

        # Sample pseudo-representations R'_x conditioned on old classes
        sampled_class_ids = random.choices(old_classes, k=batch_size)
        z_fake_list = [self.old_vae.sample(1, cls_idx, device=device) for cls_idx in sampled_class_ids]
        r_x_prime = torch.cat(z_fake_list, dim=0)

        # Label pseudo-representations using old Top classifier (T_old)
        with torch.no_grad():
            old_logits = self.old_top(r_x_prime)
            y_prime = old_logits.argmax(dim=1)

        # Train updated Top classifier (T_new) on replayed pseudo-features
        logits_fake = model.classifier_head(r_x_prime)
        loss_replay = F.cross_entropy(logits_fake, y_prime)
        return loss_replay

    def update_memory_after_task(
        self,
        train_loader: Any,
        current_task_id: int,
        device: Union[torch.device, str],
    ) -> None:
        """Trains new LatentCVAE on real + replayed latent features and snapshots Top classifier."""
        dev = torch.device(device) if isinstance(device, str) else device
        num_new = len(self.current_task_classes) if self.current_task_classes else 2
        new_classes = self.current_task_classes or list(range(self._known_classes, self._known_classes + num_new))

        for c in new_classes:
            if c not in self.seen_classes:
                self.seen_classes.append(c)
        self._known_classes += len(new_classes)

        # Freeze root backbone immediately after task 1
        self._freeze_root(self.model)

        # Extract features R(x) using frozen root backbone
        feat_list: List[torch.Tensor] = []
        label_list: List[torch.Tensor] = []

        self.model.eval()
        with torch.no_grad():
            for raw_batch in train_loader:
                h, imp, i1d, i2d, api, y = self._unpack_batch(raw_batch)
                if hasattr(self.model, "backbone"):
                    r_x = self.model.backbone(h, imp, i1d, i2d, api)
                elif hasattr(self.model, "feature_extractor"):
                    r_x = self.model.feature_extractor(h, imp, i1d, i2d, api)
                else:
                    out = self.model(h, imp, i1d, i2d, api)
                    r_x = out[1] if isinstance(out, tuple) else out

                feat_list.append(r_x.detach().cpu())
                label_list.append(y.detach().cpu())

        train_feats = torch.cat(feat_list, dim=0)
        train_labels = torch.cat(label_list, dim=0)

        # Replay pseudo-features from G_old to train G_new without forgetting
        if current_task_id > 1 and self.old_vae is not None:
            old_classes = [c for c in self.seen_classes if c not in new_classes]
            if old_classes:
                n_per_cls = max(1, int(train_feats.shape[0] * self.replay_ratio))
                old_feats_list: List[torch.Tensor] = []
                old_lbls_list: List[torch.Tensor] = []

                self.old_vae.eval()
                with torch.no_grad():
                    for cls_id in old_classes:
                        fake_z = self.old_vae.sample(n_per_cls, cls_id, dev).cpu()
                        fake_y = torch.full((n_per_cls,), cls_id, dtype=torch.long)
                        old_feats_list.append(fake_z)
                        old_lbls_list.append(fake_y)

                old_feats = torch.cat(old_feats_list, dim=0)
                old_labels = torch.cat(old_lbls_list, dim=0)
                train_feats = torch.cat([train_feats, old_feats], dim=0)
                train_labels = torch.cat([train_labels, old_labels], dim=0)

        # Initialize and optimize G_new
        max_seen_class = max(self.seen_classes) if self.seen_classes else self.num_classes - 1
        cvae_num_classes = max(self.num_classes, max_seen_class + 1)

        self.vae = LatentCVAE(
            feature_dim=train_feats.shape[1],
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            num_classes=cvae_num_classes,
        ).to(dev)

        vae_optimizer = optim.Adam(self.vae.parameters(), lr=self.vae_lr)
        self.vae.train()

        n_samples = train_feats.shape[0]
        batch_size = min(64, max(n_samples, 1))

        for epoch in range(1, self.vae_epochs + 1):
            perm = torch.randperm(n_samples)
            z_shuffled = train_feats[perm]
            y_shuffled = train_labels[perm]

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                z_b = z_shuffled[start:end].to(dev)
                y_b = y_shuffled[start:end].to(dev)

                c_b = F.one_hot(y_b.long(), num_classes=cvae_num_classes).float()
                z_recon, mu, log_var = self.vae(z_b, c_b)

                loss = LatentCVAE.vae_loss(z_recon, z_b, mu, log_var, beta=1.0)
                vae_optimizer.zero_grad()
                loss.backward()
                vae_optimizer.step()

        # Update G_old <- G_new
        self.old_vae = copy.deepcopy(self.vae)
        self.old_vae.eval()
        for param in self.old_vae.parameters():
            param.requires_grad_(False)

        # Update T_old <- T_new (Classifier Head only)
        if hasattr(self.model, "classifier_head"):
            self.old_top = copy.deepcopy(self.model.classifier_head)
            self.old_top.eval()
            for param in self.old_top.parameters():
                param.requires_grad_(False)

        self.model.train()

    def get_extra_state(self) -> Dict[str, Any]:
        """Serializes LatentCVAE generator and old Top classifier head."""
        vae_state = self.old_vae.state_dict() if self.old_vae is not None else None
        top_state = self.old_top.state_dict() if self.old_top is not None else None
        return {
            "seen_classes": list(self.seen_classes),
            "root_frozen": self.root_frozen,
            "old_vae_state": vae_state,
            "old_top_state": top_state,
        }

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        """Restores LatentCVAE generator and old Top classifier head."""
        self.seen_classes = list(state.get("seen_classes", self.seen_classes))
        self.root_frozen = bool(state.get("root_frozen", self.root_frozen))

        vae_state = state.get("old_vae_state", None)
        if vae_state is not None:
            max_class = max(self.seen_classes) if self.seen_classes else self.num_classes - 1
            num_classes = max(self.num_classes, max_class + 1)
            self.old_vae = LatentCVAE(
                feature_dim=self.feature_dim,
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                num_classes=num_classes,
            ).to(self.device)
            self.old_vae.load_state_dict(vae_state)
            self.old_vae.eval()
            for p in self.old_vae.parameters():
                p.requires_grad_(False)

        top_state = state.get("old_top_state", None)
        if top_state is not None and self.model is not None and hasattr(self.model, "classifier_head"):
            self.old_top = copy.deepcopy(self.model.classifier_head)
            self.old_top.load_state_dict(top_state)
            self.old_top.eval()
            for p in self.old_top.parameters():
                p.requires_grad_(False)
