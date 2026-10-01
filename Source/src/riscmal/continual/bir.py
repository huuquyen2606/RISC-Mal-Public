"""Brain-Inspired Replay (BI-R) Continual Learning Strategy."""

import copy
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from riscmal.models.vae import ConditionalVAE
from .base import BaseContinualStrategy


class BIRStrategy(BaseContinualStrategy):
    """Brain-Inspired Replay (BI-R) Continual Learning Strategy.

    Combines internal latent generative replay with expandable GMM conditional VAE priors
    and knowledge distillation to prevent representational drift:
        L_{BIR} = L_{CE} + L_{replay} + lambda_distill * L_{KD}

    Components:
        - Expandable ConditionalVAE with class-wise GMM priors.
        - Internal latent replay sampling pseudo-features to maintain old decision boundaries.
        - Knowledge distillation against snapshot teacher model with temperature T=2.0.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: Union[torch.device, str] = "cpu",
        config: Optional[Dict[str, Any]] = None,
        replay_ratio: float = 0.1,
        T_distill: float = 2.0,
        lambda_distill: float = 1.0,
        cvae_hidden_dim: int = 256,
        cvae_z_dim: int = 64,
        cvae_epochs: int = 20,
        cvae_lr: float = 1e-3,
        gate_dropout: float = 0.1,
        num_known_classes: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            device=device,
            config=config,
            name="BI-R",
            num_known_classes=num_known_classes,
            replay_ratio=replay_ratio,
            **kwargs,
        )
        self.T_distill: float = float(self.config.get("T_distill", self.config.get("temperature", T_distill)))
        self.lambda_distill: float = float(self.config.get("lambda_distill", lambda_distill))
        self.cvae_hidden_dim: int = int(self.config.get("cvae_hidden_dim", cvae_hidden_dim))
        self.cvae_z_dim: int = int(self.config.get("cvae_z_dim", cvae_z_dim))
        self.cvae_epochs: int = int(self.config.get("cvae_epochs", cvae_epochs))
        self.cvae_lr: float = float(self.config.get("cvae_lr", cvae_lr))
        self.gate_dropout: float = float(self.config.get("gate_dropout", gate_dropout))

        self._cvae: Optional[ConditionalVAE] = None
        self._previous_model: Optional[nn.Module] = None
        self._seen_classes: List[int] = []

    def adapt_architecture_before_task(self, device: Union[torch.device, str]) -> None:
        """Freezes backbone after task 1 to stabilize latent space and expands classifier/cVAE."""
        dev = torch.device(device) if isinstance(device, str) else device
        num_new = len(self.current_task_classes) if self.current_task_classes else 2

        if self.current_task_id > 1:
            if hasattr(self.model, "backbone"):
                self.model.backbone.eval()
                for param in self.model.backbone.parameters():
                    param.requires_grad_(False)

        if hasattr(self.model, "expand_classes"):
            self.model.expand_classes(num_new, device=dev)

        if self._cvae is not None:
            self._cvae.expand_classes(num_new, device=dev)

    def compute_additional_loss(
        self,
        model: nn.Module,
        features_64: torch.Tensor,
        outputs: torch.Tensor,
        batch: Any,
        current_task_id: int = 1,
    ) -> torch.Tensor:
        """Calculates combined latent generative replay and knowledge distillation loss."""
        if current_task_id == 1 or self._cvae is None or self._previous_model is None:
            return torch.tensor(0.0, device=features_64.device, requires_grad=True)

        device = features_64.device
        batch_size = features_64.shape[0]

        # Isolate historical classes
        old_classes = [c for c in self._seen_classes if c not in self.current_task_classes]
        if not old_classes:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 1. Latent Generative Replay
        sampled_class_ids = random.choices(old_classes, k=batch_size)
        replay_feats_list: List[torch.Tensor] = []
        replay_labels_list: List[torch.Tensor] = []

        self._cvae.eval()
        for cls_id in sampled_class_ids:
            fake_feat = self._cvae.sample(1, cls_id, device=device)
            fake_label = torch.full((1,), cls_id, dtype=torch.long, device=device)
            replay_feats_list.append(fake_feat)
            replay_labels_list.append(fake_label)

        replay_feats = torch.cat(replay_feats_list, dim=0)
        replay_labels = torch.cat(replay_labels_list, dim=0)

        replay_logits = model.classifier_head(replay_feats)
        loss_replay = F.cross_entropy(replay_logits, replay_labels.long())

        # 2. Knowledge Distillation on incoming representations
        h, imp, i1d, i2d, api, _ = self._unpack_batch(batch)
        with torch.no_grad():
            if hasattr(self._previous_model, "backbone") and hasattr(self._previous_model, "classifier_head"):
                prev_features = self._previous_model.backbone(h, imp, i1d, i2d, api)
                prev_outputs = self._previous_model.classifier_head(prev_features)
            else:
                out = self._previous_model(h, imp, i1d, i2d, api)
                prev_outputs = out[0] if isinstance(out, tuple) else out

        student_old_logits = outputs[:, : prev_outputs.shape[1]]
        log_pred = F.log_softmax(student_old_logits / self.T_distill, dim=1)
        soft_tgt = F.softmax(prev_outputs / self.T_distill, dim=1)
        loss_kd = F.kl_div(log_pred, soft_tgt, reduction="batchmean") * (self.T_distill ** 2)

        return loss_replay + (self.lambda_distill * loss_kd)

    def _extract_features(
        self,
        model: nn.Module,
        train_loader: Any,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts latent features and ground truth labels from the provided loader."""
        feat_list: List[torch.Tensor] = []
        label_list: List[torch.Tensor] = []

        model.eval()
        with torch.no_grad():
            for raw_batch in train_loader:
                h, imp, i1d, i2d, api, y = self._unpack_batch(raw_batch)
                if hasattr(model, "backbone"):
                    features = model.backbone(h, imp, i1d, i2d, api)
                elif hasattr(model, "feature_extractor"):
                    features = model.feature_extractor(h, imp, i1d, i2d, api)
                else:
                    out = model(h, imp, i1d, i2d, api)
                    features = out[1] if isinstance(out, tuple) else out

                feat_list.append(features.detach().cpu())
                label_list.append(y.detach().cpu())

        return torch.cat(feat_list, dim=0), torch.cat(label_list, dim=0)

    def _train_cvae(
        self,
        train_feats: torch.Tensor,
        train_labels: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Trains the conditional VAE on real and replayed latent representations."""
        optimizer = torch.optim.Adam(self._cvae.parameters(), lr=self.cvae_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.cvae_epochs, 1))
        dataset = TensorDataset(train_feats, train_labels)
        bs = min(128, max(len(train_feats), 1))
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)

        self._cvae.train()
        for epoch in range(1, self.cvae_epochs + 1):
            for x_batch, y_batch in loader:
                loss = self._cvae.elbo_loss(x_batch.to(device), y_batch.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        self._cvae.eval()

    def update_memory_after_task(
        self,
        train_loader: Any,
        current_task_id: int,
        device: Union[torch.device, str],
    ) -> None:
        """Trains/updates ConditionalVAE generator, records snapshots, and updates memory."""
        dev = torch.device(device) if isinstance(device, str) else device
        num_new = len(self.current_task_classes) if self.current_task_classes else 2
        new_classes = self.current_task_classes or list(range(self._known_classes, self._known_classes + num_new))

        for c in new_classes:
            if c not in self._seen_classes:
                self._seen_classes.append(c)
        self._known_classes += len(new_classes)

        new_feats, new_labels = self._extract_features(self.model, train_loader, dev)
        feat_dim = new_feats.shape[1]

        if self._cvae is None:
            self._cvae = ConditionalVAE(
                input_dim=feat_dim,
                initial_classes=len(self._seen_classes),
                hidden_dim=self.cvae_hidden_dim,
                z_dim=self.cvae_z_dim,
                gate_dropout=self.gate_dropout,
            ).to(dev)
        else:
            self._cvae = self._cvae.to(dev)

        if current_task_id > 1 and len(self._seen_classes) > len(new_classes):
            old_classes = [c for c in self._seen_classes if c not in new_classes]
            n_per_cls = max(1, int(new_feats.shape[0] * self.replay_ratio))

            old_feats_list: List[torch.Tensor] = []
            old_lbls_list: List[torch.Tensor] = []
            self._cvae.eval()
            with torch.no_grad():
                for cls_id in old_classes:
                    fake = self._cvae.sample(n_per_cls, cls_id, dev).cpu()
                    lbl = torch.full((n_per_cls,), cls_id, dtype=torch.long)
                    old_feats_list.append(fake)
                    old_lbls_list.append(lbl)

            old_feats = torch.cat(old_feats_list, dim=0)
            old_labels = torch.cat(old_lbls_list, dim=0)

            train_feats = torch.cat([new_feats, old_feats], dim=0)
            train_labels = torch.cat([new_labels, old_labels], dim=0)
        else:
            train_feats = new_feats
            train_labels = new_labels

        self._train_cvae(train_feats, train_labels, dev)

        self._previous_model = copy.deepcopy(self.model)
        self._previous_model.eval()
        for p in self._previous_model.parameters():
            p.requires_grad = False

        self.model.train()

    def get_extra_state(self) -> Dict[str, Any]:
        """Serializes ConditionalVAE and teacher model state."""
        cvae_state = self._cvae.state_dict() if self._cvae is not None else None
        prev_state = self._previous_model.state_dict() if self._previous_model is not None else None
        return {
            "seen_classes": list(self._seen_classes),
            "cvae_state": cvae_state,
            "prev_model_state": prev_state,
        }

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        """Restores ConditionalVAE and teacher model state."""
        self._seen_classes = list(state.get("seen_classes", self._seen_classes))
        cvae_state = state.get("cvae_state", None)
        if cvae_state is not None:
            if self._cvae is None:
                feat_dim = getattr(self.model, "feat_dim", 64)
                self._cvae = ConditionalVAE(
                    input_dim=feat_dim,
                    initial_classes=len(self._seen_classes),
                    hidden_dim=self.cvae_hidden_dim,
                    z_dim=self.cvae_z_dim,
                    gate_dropout=self.gate_dropout,
                ).to(self.device)
            self._cvae.load_state_dict(cvae_state)

        prev_state = state.get("prev_model_state", None)
        if prev_state is not None and self.model is not None:
            self._previous_model = copy.deepcopy(self.model)
            self._previous_model.load_state_dict(prev_state)
            self._previous_model.eval()
            for p in self._previous_model.parameters():
                p.requires_grad = False
