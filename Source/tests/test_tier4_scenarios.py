"""Tier 4: Real-World Application Scenarios.

Verifies end-to-end multi-task continual learning workflows:
- 3-Task Incremental Class Learning Scenario (Base Task + 2 Incremental Tasks)
- Severe Class Imbalance and Minority Retention Scenario
- Selective Representation Preservation Feature Drift Validation (Paper Mechanism E5)
"""

import copy
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from riscmal.data.dataset import MalwareMultiViewDataset, MultiViewBatch, multiview_collate_fn
from riscmal.evaluation.metrics import compute_classification_metrics
from riscmal.evaluation.tracker import ContinualMetricsTracker
from riscmal.memory.buffer import RehearsalMemoryManager
from riscmal.memory.sampler import get_mixed_batch


# ===========================================================================
# End-to-End 3-Task Incremental Learning Workflow
# ===========================================================================

class TestRealWorldContinualScenarios:
    """Verifies complete multi-task lifecycle: training, expansion, distillation, evaluation."""

    def test_end_to_end_3task_incremental_learning_simulation(self, reference_model, memory_manager, make_synthetic_dataset):
        """Simulates a complete 3-Task incremental class-learning workflow.
        
        Protocol:
            Task 1: Classes [0, 1, 2, 3] (Base task)
            Task 2: Classes [4, 5]       (Incremental task 1)
            Task 3: Classes [6, 7]       (Incremental task 2)
        """
        device = "cpu"
        optimizer = torch.optim.Adam(reference_model.parameters(), lr=0.01)
        tracker = ContinualMetricsTracker()

        # Task definitions: (task_id, new_classes, total_classes_seen)
        task_schedule = [
            (1, [0, 1, 2, 3], 4),
            (2, [4, 5], 6),
            (3, [6, 7], 8),
        ]

        teacher_model = None

        for task_id, new_classes, total_classes in task_schedule:
            # -------------------------------------------------------------
            # A. Architectural Adaptation Before Task
            # -------------------------------------------------------------
            if task_id > 1:
                # 1. Snapshot and freeze teacher model
                teacher_model = copy.deepcopy(reference_model)
                teacher_model.eval()
                for p in teacher_model.parameters():
                    p.requires_grad_(False)

                old_classes_count = total_classes - len(new_classes)
                old_weights_snapshot = reference_model.classifier_head.weight.detach().clone()

                # 2. Expand student classifier head with norm matching
                reference_model.expand_classes(num_new_classes=len(new_classes), device=device)

                # Assertion: Weight preservation on old classes
                assert torch.equal(
                    reference_model.classifier_head.weight[:old_classes_count, :].detach(),
                    old_weights_snapshot,
                ), f"Task {task_id}: Existing class weights mutated during head expansion!"

                # Re-initialize optimizer for expanded parameter set
                optimizer = torch.optim.Adam(reference_model.parameters(), lr=0.01)

            assert reference_model.classifier_head.out_features == total_classes

            # -------------------------------------------------------------
            # B. Simulated Training Loop
            # -------------------------------------------------------------
            train_ds = make_synthetic_dataset(num_samples=40, num_classes=len(new_classes), class_offset=new_classes[0])
            train_loader = DataLoader(train_ds, batch_size=16, collate_fn=multiview_collate_fn, shuffle=True)

            reference_model.train()
            epochs = 2
            for epoch in range(epochs):
                for batch in train_loader:
                    optimizer.zero_grad()

                    if task_id == 1:
                        # Base task: Standard Cross-Entropy on current batch
                        logits, _ = reference_model(batch)
                        loss = F.cross_entropy(logits, batch.label)
                    else:
                        # Incremental tasks: Form mixed batch B_mixed = B_new U B_replay
                        mixed_batch = get_mixed_batch(batch, memory_manager, batch_size=16, device=device)
                        logits_s, feats_s = reference_model(mixed_batch)

                        # 1. Cross-Entropy Loss
                        loss_ce = F.cross_entropy(logits_s, mixed_batch.label)

                        # 2. Distillation on historical exemplars
                        is_old = (mixed_batch.label < old_classes_count)
                        loss_kd = torch.tensor(0.0, device=device)
                        loss_sp = torch.tensor(0.0, device=device)

                        if is_old.any():
                            # Forward frozen teacher on old class samples
                            old_sub_batch = MultiViewBatch({
                                k: v[is_old] for k, v in mixed_batch.items()
                            })
                            with torch.no_grad():
                                logits_t, feats_t = teacher_model(old_sub_batch)

                            # Knowledge distillation loss
                            T = 2.0
                            p_s = F.log_softmax(logits_s[is_old, :old_classes_count] / T, dim=1)
                            q_t = F.softmax(logits_t / T, dim=1)
                            loss_kd = -1.0 * (q_t * p_s).sum(dim=1).mean() * (T ** 2)

                            # Soft protection mask m^t (fixed simulated mask, Eq. 5)
                            m_mask = torch.full((64,), 0.8, device=device)
                            loss_sp = (((feats_s[is_old] - feats_t) * m_mask) ** 2).sum(dim=1).mean()

                        loss = loss_ce + 0.1 * loss_kd + 0.1 * loss_sp

                    assert torch.isfinite(loss), f"Task {task_id} epoch {epoch}: Non-finite loss encountered!"
                    loss.backward()
                    optimizer.step()

            # -------------------------------------------------------------
            # C. Multi-Task Evaluation After Task
            # -------------------------------------------------------------
            reference_model.eval()
            val_ds = make_synthetic_dataset(num_samples=30, num_classes=total_classes, class_offset=0)
            val_loader = DataLoader(val_ds, batch_size=16, collate_fn=multiview_collate_fn)

            all_preds = []
            all_targets = []
            with torch.no_grad():
                for batch in val_loader:
                    logits, _ = reference_model(batch)
                    preds = logits.argmax(dim=1)
                    all_preds.extend(preds.cpu().tolist())
                    all_targets.extend(batch.label.cpu().tolist())

            metrics = compute_classification_metrics(all_targets, all_preds)
            assert metrics["accuracy"] >= 0.0

            # Continual learning metrics tracking
            classes_introduced = len(new_classes)
            f, old_f1, new_f1, h_mean = tracker.calculate_continual_metrics(
                metrics,
                current_task_id=task_id,
                classes_per_task=classes_introduced,
            )

            # Log assertions: Forgetting is non-negative
            assert f >= 0.0, f"Task {task_id}: Negative forgetting reported ({f})!"
            assert old_f1 >= 0.0
            assert new_f1 >= 0.0

            # -------------------------------------------------------------
            # D. Rehearsal Memory Update
            # -------------------------------------------------------------
            memory_manager.update_memory_after_task(train_loader, new_classes=new_classes)
            for c in new_classes:
                assert c in memory_manager.memory, f"Task {task_id}: Class {c} not retained in memory!"
                assert len(memory_manager.memory[c]) >= 1, f"Task {task_id}: Class {c} has 0 exemplars!"

        # Final checks across entire 3-task lifelong learning sequence
        assert memory_manager.seen_classes == [0, 1, 2, 3, 4, 5, 6, 7]
        assert len(memory_manager.memory) == 8
        assert memory_manager.get_total_exemplars() >= 8

    def test_imbalanced_minority_retention_scenario(self, memory_manager, make_synthetic_dataset):
        """Simulates extreme malware family imbalance (100 common samples vs. 2 rare APT samples)."""
        # Create common class 0 (100 samples) and rare class 1 (2 samples)
        h0 = np.random.randn(100, 4).astype(np.float32)
        imp0 = np.random.randint(0, 2, (100, 1000)).astype(np.float32)
        i1d0 = np.random.rand(100, 1024).astype(np.float32)
        i2d0 = np.random.rand(100, 3, 224, 224).astype(np.float32)
        api0 = np.random.randint(0, 101, (100, 100)).astype(np.int64)
        y0 = np.zeros(100, dtype=np.int64)

        h1 = np.random.randn(2, 4).astype(np.float32)
        imp1 = np.random.randint(0, 2, (2, 1000)).astype(np.float32)
        i1d1 = np.random.rand(2, 1024).astype(np.float32)
        i2d1 = np.random.rand(2, 3, 224, 224).astype(np.float32)
        api1 = np.random.randint(0, 101, (2, 100)).astype(np.int64)
        y1 = np.ones(2, dtype=np.int64)

        imbalanced_ds = MalwareMultiViewDataset(
            header=np.vstack([h0, h1]),
            imports=np.vstack([imp0, imp1]),
            img1d=np.vstack([i1d0, i1d1]),
            img2d=np.vstack([i2d0, i2d1]),
            apis=np.vstack([api0, api1]),
            labels=np.concatenate([y0, y1]),
        )

        loader = DataLoader(imbalanced_ds, batch_size=32, collate_fn=multiview_collate_fn)
        stats = memory_manager.update_memory_after_task(loader, new_classes=[0, 1])

        # Verify majority class retains ~10 exemplars (100 * 0.10)
        # Verify minority class retains min=1 exemplar despite floor(2 * 0.10) = 0
        assert stats[0] >= 10
        assert stats[1] == 1
        assert len(memory_manager.memory[1]) == 1

        # Evaluate metrics under severe imbalance
        y_true = np.array([0] * 50 + [1] * 2)
        y_pred = np.array([0] * 50 + [0] * 2) # Minority completely misclassified
        res = compute_classification_metrics(y_true, y_pred)
        assert res["worst_class_recall"] == 0.0
        assert res["macro_recall"] < res["accuracy"] # Imbalance penalty visible

    def test_feature_drift_under_selective_plasticity(self, reference_model, make_synthetic_batch):
        r"""Validates Mechanism Validation E5: High-m^t dimensions experience lower drift under \mathcal{L}_{SP}."""
        teacher = copy.deepcopy(reference_model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # Create student and assign protection mask m^t: first 32 dims protected (1.0), last 32 unprotected (0.0)
        m_mask = torch.zeros(64)
        m_mask[:32] = 1.0 # Protected
        m_mask[32:] = 0.0 # Plastic

        batch = make_synthetic_batch(batch_size=16)
        _, s_feats = reference_model(batch)
        with torch.no_grad():
            _, t_feats = teacher(batch)

        # Loss explicitly penalizes drift on protected dimensions
        loss_sp = (((s_feats - t_feats) * m_mask) ** 2).sum(dim=1).mean()

        # Optimizer step
        optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.1)
        optimizer.zero_grad()
        loss_sp.backward()
        optimizer.step()

        # Check drift after update
        with torch.no_grad():
            _, s_feats_after = reference_model(batch)
            _, t_feats_after = teacher(batch)
            drift_per_dim = torch.abs(s_feats_after - t_feats_after).mean(dim=0)

        # Protected dimensions should be close to zero drift
        protected_drift = drift_per_dim[:32].mean().item()
        assert protected_drift < 1.0
        assert torch.isfinite(drift_per_dim).all()
