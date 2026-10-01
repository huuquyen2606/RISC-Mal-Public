"""Risk-Informed Selective Class-Incremental Learning (RISC-Mal) Strategy."""

import copy
import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from riscmal.memory.buffer import RehearsalMemoryManager
from .base import BaseContinualStrategy

CLASSES_PER_TASK = 2


class RISCMalStrategy(BaseContinualStrategy):
    r"""Risk-Informed Selective Class-Incremental Learning (RISC-Mal) Strategy.
    
    Addresses catastrophic forgetting under class imbalance through:
        1. Class-Aware Retention Risk (R_c^t, \widehat{R}_c^t):
           Integrates empirical recall degradation F_c^t = max(0, max_{\tau < t} r_c^\tau - r_c^{t-1})
           and replay-exposure imbalance I_c = 1 - E_c / max_k E_k, where rehearsal exposure
           E_c = (1 - \beta_R^{|\mathcal{M}_c|}) / (1 - \beta_R) uses replay-exposure factor
           \beta_R = 0.999 (Eq. 3). Composite risk R_c^t = \alpha F_c^t + (1 - \alpha) I_c is
           normalized via range-safe min-max to \widehat{R}_c^t \in [0, 1].
        2. Analytical Margin Sensitivity Analysis (S_{c,j}^t):
           Analytically evaluates classification margin gradients S_{c,j}^t = |\partial \gamma_c(x) / \partial h_j|
           with respect to latent representation dimensions without backbone backpropagation (Eq. 4).
        3. Continuous Soft Protection Mask (m^t = [m_1^t, ..., m_d^t] \in (0, 1)^d):
           Aggregates importance P_j^t = \sum_{c \in \mathcal{C}_{<t}} \widehat{R}_c^t S_{c,j}^t,
           normalizes to \widehat{P}_j^t via range-safe min-max, and computes sigmoidal gating
           m_j^t = \sigma((\widehat{P}_j^t - \tau_p) / T_p) (Eq. 5), preserving vital historical
           decision boundaries while maintaining plasticity elsewhere.
        4. Selective Representation Preservation Loss (\mathcal{L}_{SP}):
           Selectively constrains latent feature drift on rehearsal exemplars (Eq. 6):
           \mathcal{L}_{SP} = (1 / |\mathcal{B}_{replay}|) \sum_{x \in \mathcal{B}_{replay}} || m^t \odot [f_{\theta_t}(x) - f_{\theta_{t-1}}(x)] ||_2^2.
        5. Incremental Classification Loss (\mathcal{L}_{IC}):
           Re-weights classification objective over mixed batches using mean-normalized effective
           number weights with sample imbalance calibration factor \beta = 0.10 (Eq. 7).
        6. Temperature-Scaled Knowledge Distillation Loss (\mathcal{L}_{KD}):
           Distills historical class logits at temperature T_{KD} = 2.0 scaled by T_{KD}^2 = 4.0 (Eq. 8).
        7. Norm-Matched Output Head Expansion (\bar{r}_t):
           Scales novel class weight vectors to average historical Euclidean norm
           \bar{r}_t = |\mathcal{C}_{<t}|^{-1} \sum_{c \in \mathcal{C}_{<t}} ||W_{t-1}[c, :]||_2.
        8. Total Incremental Objective (\mathcal{L}_{RISC}):
           Joint optimization objective \mathcal{L}_{RISC} = \mathcal{L}_{IC} + \lambda_{SP}\mathcal{L}_{SP} + \lambda_{KD}\mathcal{L}_{KD} (Eq. 9).
    
    Note on Dual Beta Factors:
        - Sample Imbalance Calibration Factor \beta = 0.10: Used in _update_imbalance_weights for
          effective number sample weights \omega_{c,t} = (1 - \beta) / (1 - \beta^{n_{c,t}^{avail}}).
        - Replay-Exposure Factor \beta_R = 0.999: Used in adapt_architecture_before_task for
          effective rehearsal memory volume E_c = (1 - \beta_R^{|\mathcal{M}_c|}) / (1 - \beta_R).
    """

    def __init__(
        self,
        num_known_classes: int = 0,
        alpha: float = 0.5,
        tau_p: float = 0.8,
        T_p: float = 0.2,
        lambda_sp: float = 0.1,
        lambda_kd: float = 0.1,
        T_kd: float = 2.0,
        replay_ratio: float = 0.1,
        use_sp: bool = True,
        use_ic: bool = True,
        use_kd: bool = True,
        protection_strategy: str = "risc_soft",
        topk_ratio: float = 0.5,
        log_dir: str = "results",
        **kwargs: Any,
    ) -> None:
        super().__init__(name="RISC-Mal", num_known_classes=num_known_classes, replay_ratio=replay_ratio, **kwargs)
        self.alpha = alpha
        self.tau_p = tau_p
        self.T_p = T_p
        self.lambda_sp = lambda_sp
        self.lambda_kd = lambda_kd
        self.T_kd = T_kd

        self.use_sp = use_sp
        self.use_ic = use_ic
        self.use_kd = use_kd

        self.protection_strategy = protection_strategy
        self.topk_ratio = topk_ratio
        self.log_dir = log_dir

        self.memory_manager = RehearsalMemoryManager(replay_ratio=replay_ratio)
        self.teacher_model: Optional[nn.Module] = None
        self.protection_mask: Optional[torch.Tensor] = None
        self.best_historical_f1: Dict[int, float] = {}

    def compute_additional_loss(
        self,
        model: nn.Module,
        features_64: torch.Tensor,
        outputs: torch.Tensor,
        batch: Any,
        current_task_id: int = 1,
    ) -> torch.Tensor:
        r"""Computes auxiliary selective preservation and distillation losses.

        Combines:
            - Selective Representation Preservation Loss (\mathcal{L}_{SP}, Eq. 5):
              Penalizes latent drift on rehearsal exemplars, gated by continuous protection mask m^t.
            - Temperature-Scaled Knowledge Distillation Loss (\mathcal{L}_{KD}, Eq. 8):
              Matches historical logit distributions scaled by T_{KD}^2 = 4.0.

        Args:
            model: Current student model being trained.
            features_64: Latent representations extracted by current backbone [B, 64].
            outputs: Current classifier head logits [B, current_classes].
            batch: Current mixed mini-batch.
            current_task_id: Current incremental task identifier (1-indexed).

        Returns:
            Scalar tensor representing \lambda_{SP}\mathcal{L}_{SP} + \lambda_{KD}\mathcal{L}_{KD}.
        """
        if self.teacher_model is None or self._known_classes == 0 or current_task_id == 1:
            return torch.tensor(0.0, device=features_64.device)

        device = features_64.device
        if isinstance(batch, (tuple, list)):
            h, imp, i1d, i2d, api = [item.to(device) for item in batch[:5]]
            y = batch[-1].to(device)
        elif isinstance(batch, dict):
            h, imp, i1d, i2d, api = (
                batch["header"].to(device),
                batch["imports"].to(device),
                batch["img1d"].to(device),
                batch["img2d"].to(device),
                batch["apis"].to(device),
            )
            y = batch["label"].to(device)
        else:
            return torch.tensor(0.0, device=device)

        is_old = (y < self._known_classes)
        if not is_old.any():
            return torch.tensor(0.0, device=device)

        loss_total = torch.tensor(0.0, device=device)

        with torch.no_grad():
            teacher_features = self.teacher_model.backbone(
                h[is_old], imp[is_old], i1d[is_old], i2d[is_old], api[is_old]
            )
            teacher_logits = self.teacher_model.classifier_head(teacher_features)

        student_features = features_64[is_old]
        student_logits_for_kd = outputs[is_old, :teacher_logits.shape[1]]

        # 1. Selective Representation Preservation Loss (\mathcal{L}_{SP}, Eq. 6)
        # Selectively penalizes latent feature drift on replay exemplars gated element-wise by mask m^t
        if self.use_sp and self.protection_mask is not None:
            mask = self.protection_mask.to(device).detach()
            diff = (student_features - teacher_features) * mask
            loss_sp = (diff ** 2).sum(dim=1).mean()
            loss_total = loss_total + (self.lambda_sp * loss_sp)

        # 2. Temperature-scaled Knowledge Distillation Loss (\mathcal{L}_{KD}, Eq. 8)
        # Distills historical class logits at temperature T_kd scaled by T_kd^2 = 4.0
        if self.use_kd:
            T = self.T_kd
            p = F.log_softmax(student_logits_for_kd / T, dim=1)
            q = F.softmax(teacher_logits / T, dim=1)  # Teacher probability distribution p_{t-1}^{(T_KD)}
            loss_kd = -torch.sum(q * p, dim=1).mean()
            loss_total = loss_total + (self.lambda_kd * (T ** 2) * loss_kd)

        # loss_total contributes \lambda_{SP}\mathcal{L}_{SP} + \lambda_{KD}\mathcal{L}_{KD} to \mathcal{L}_{RISC} (Eq. 9)
        return loss_total

    def adapt_architecture_before_task(self, device: Union[torch.device, str]) -> None:
        r"""Pre-task lifecycle hook: evaluates historical retention risk and generates protection mask.

        Steps:
            1. Measures empirical recall degradation F_c^t across historical classes on rehearsal buffer.
            2. Estimates replay-exposure imbalance I_c using factor \beta_R = 0.999 (Eq. 3).
            3. Derives class-aware retention risk R_c^t and normalized risk \widehat{R}_c^t.
            4. Performs analytical margin sensitivity analysis S_{c,j}^t (Eq. 4).
            5. Synthesizes continuous soft protection mask m^t (Eq. 5).
            6. Snapshots frozen teacher model and expands output classifier head (Eq. 6).

        Args:
            device: Target torch device for tensor computations.
        """
        dev = torch.device(device) if isinstance(device, str) else device
        if self._known_classes > 0:
            print(f"\n[{self.name}] Measuring empirical recall and computing retention risk for {self._known_classes} old classes...")
            old_classes = list(range(self._known_classes))
            self.model.eval()
            current_recalls: Dict[int, float] = {}

            # A. Measure Recall on Rehearsal Memory
            with torch.no_grad():
                for c in old_classes:
                    if c not in self.memory_manager.memory or not self.memory_manager.memory[c]:
                        current_recalls[c] = 0.0
                        continue

                    c_mem = self.memory_manager.memory[c]
                    correct = 0
                    batch_size = 32
                    for i in range(0, len(c_mem), batch_size):
                        batch_samples = c_mem[i : i + batch_size]
                        tensors = [torch.stack([s[idx] for s in batch_samples]).to(dev) for idx in range(6)]
                        h, imp, i1d, i2d, api, y = tensors
                        out = self.model(h, imp, i1d, i2d, api)
                        logits = out[0] if isinstance(out, tuple) else out
                        preds = logits.argmax(dim=1)
                        correct += (preds == y).sum().item()

                    current_recalls[c] = correct / max(1, len(c_mem))
                    if c not in self.best_historical_f1:
                        self.best_historical_f1[c] = current_recalls[c]
                    else:
                        self.best_historical_f1[c] = max(self.best_historical_f1[c], current_recalls[c])

            # B. Compute Class-Aware Retention Risk (R_c^t = \alpha F_c^t + (1 - \alpha) I_c, Eq. 3)
            # Replay-exposure factor \beta_R = 0.999 governs effective rehearsal volume E_c
            beta_R = 0.999  # Replay-exposure factor \beta_R = 0.999 (Eq. 3, distinct from calibration \beta = 0.10)
            beta = beta_R
            effective_nums: Dict[int, float] = {}
            for c in old_classes:
                count = len(self.memory_manager.memory.get(c, []))
                effective_nums[c] = (1.0 - beta_R ** max(1, count)) / (1.0 - beta_R)

            max_E = max(effective_nums.values()) if effective_nums else 1.0
            retention_risks: Dict[int, float] = {}
            exposure_imbalances: Dict[int, float] = {}

            for c in old_classes:
                f_c = max(0.0, self.best_historical_f1.get(c, 0.0) - current_recalls.get(c, 0.0))
                i_c = 1.0 - (effective_nums[c] / max_E)
                exposure_imbalances[c] = i_c
                retention_risks[c] = self.alpha * f_c + (1.0 - self.alpha) * i_c

            min_r = min(retention_risks.values()) if retention_risks else 0.0
            max_r = max(retention_risks.values()) if retention_risks else 0.0
            diff_r = max_r - min_r
            for c in retention_risks:
                retention_risks[c] = (retention_risks[c] - min_r) / diff_r if diff_r > 1e-8 else 1.0

            # C. Analytical Margin Sensitivity Analysis S_{c,j}^t (Eq. 4)
            P_j = torch.zeros(64, device=dev)
            for c in old_classes:
                if c not in self.memory_manager.memory or not self.memory_manager.memory[c]:
                    continue

                c_mem = self.memory_manager.memory[c]
                S_c_sum = torch.zeros(64, device=dev)
                batch_size = 32

                for i in range(0, len(c_mem), batch_size):
                    batch_samples = c_mem[i : i + batch_size]
                    tensors = [torch.stack([s[idx] for s in batch_samples]).to(dev) for idx in range(6)]
                    h, imp, i1d, i2d, api, _ = tensors

                    with torch.no_grad():
                        emb = self.model.backbone(h, imp, i1d, i2d, api)

                    batch_emb = emb.clone().detach().requires_grad_(True)
                    logits = self.model.classifier_head(batch_emb)
                    correct_logits = logits[:, c]

                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    mask[:, c] = True
                    competitor_logits = logits.masked_fill(mask, -1e9).max(dim=1).values
                    margin = correct_logits - competitor_logits

                    grad = torch.autograd.grad(margin.sum(), batch_emb)[0]
                    S_c_sum += grad.detach().abs().sum(dim=0)

                S_c = S_c_sum / max(1, len(c_mem))
                P_j += retention_risks.get(c, 0.0) * S_c

            # D. Protection Mask Generation (m^t = [m_1^t, ..., m_d^t] \in (0, 1)^d, Eq. 5)
            min_P = P_j.min()
            max_P = P_j.max()
            diff_P = max_P - min_P
            P_hat = (P_j - min_P) / (diff_P + 1e-8) if diff_P > 1e-8 else torch.full_like(P_j, 0.5)

            if self.protection_strategy == "fine_tuning":
                self.protection_mask = torch.zeros_like(P_hat)
            elif self.protection_strategy == "freeze":
                self.protection_mask = torch.ones_like(P_hat)
            elif self.protection_strategy == "uniform":
                self.protection_mask = torch.full_like(P_hat, 0.5)
            elif self.protection_strategy == "topk_hard":
                k = max(1, int(round(self.topk_ratio * P_hat.shape[-1])))
                topk_idx = torch.topk(P_hat, k).indices
                hard_mask = torch.zeros_like(P_hat)
                hard_mask[topk_idx] = 1.0
                self.protection_mask = hard_mask
            else:
                # Default: Smooth Sigmoidal Soft Protection Mask m_j^t = \sigma((\widehat{P}_j^t - \tau_p) / T_p)
                self.protection_mask = torch.sigmoid((P_hat - self.tau_p) / self.T_p)

            # Freeze Teacher Snapshot
            self.teacher_model = copy.deepcopy(self.model).to(dev)
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad_(False)

        # Output-Only Classifier Expansion (Norm-Matched Initialization with \bar{r}_t)
        print(f"[{self.name}] Expanding Classifier Head by {CLASSES_PER_TASK} classes...")
        self.model.expand_classes(CLASSES_PER_TASK, dev)
        current_num_classes = self._known_classes + CLASSES_PER_TASK
        self._update_imbalance_weights(current_num_classes, dev)

    def update_memory_after_task(
        self,
        train_loader: Any,
        new_classes: Union[List[int], int],
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Post-task lifecycle hook: updates class-proportional rehearsal buffer and effective weights.

        Args:
            train_loader: DataLoader for the completed incremental task.
            new_classes: Class IDs introduced during the completed task.
            device: Optional execution device for recomputing class weights.
        """
        dev = self.device if device is None else (torch.device(device) if isinstance(device, str) else device)
        if isinstance(new_classes, int):
            task_classes = list(range(self._known_classes, self._known_classes + new_classes))
        else:
            task_classes = list(new_classes)

        self.memory_manager.update_memory_after_task(train_loader, task_classes)
        self._known_classes += len(task_classes)
        self._update_imbalance_weights(self._known_classes, dev)
        print(f"[RISC-Mal] Memory updated for task classes {task_classes}. Total classes: {self._known_classes}")

    def _update_imbalance_weights(self, current_num_classes: int, device: torch.device, beta: float = 0.1) -> None:
        r"""Computes mean-normalized class weights \bar{\omega}_{c,t} for \mathcal{L}_{IC} (Eq. 7).
        
        Args:
            current_num_classes: Total cumulative classes seen so far |\mathcal{C}_{\leq t}|.
            device: Target torch device.
            beta: Sample imbalance calibration factor \beta = 0.10 (Eq. 1 & Eq. 7),
                  strictly distinct from replay-exposure factor \beta_R = 0.999 (Eq. 3).
        """
        if not self.use_ic:
            self.class_weights = None
            return

        counts = []
        for c in range(current_num_classes):
            if c < self._known_classes:
                counts.append(max(1, len(self.memory_manager.memory.get(c, []))))
            else:
                counts.append(1000)

        effective_num = [(1.0 - beta ** cnt) / (1.0 - beta) for cnt in counts]
        weights = torch.tensor([1.0 / e for e in effective_num], device=device)
        self.class_weights = (weights / weights.sum()) * current_num_classes
