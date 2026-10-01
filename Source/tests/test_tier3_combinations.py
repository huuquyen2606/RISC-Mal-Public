"""Tier 3: Cross-Feature Combination Tests.

Verifies end-to-end multi-module pipelines:
- Data loading -> Memory update -> Mixed sampling
- Mixed batch -> Forward pass -> Loss computation & backward pass
- Model snapshotting -> Classifier expansion -> Selective & KD distillation losses
- Data leakage purge -> Memory exemplar integrity
- Model evaluation -> Classification metrics -> Continual metric tracker
"""

import copy
import hashlib
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from riscmal.data.dataset import MalwareMultiViewDataset, MultiViewBatch
from riscmal.data.leakage import purge_data_leakage
from riscmal.evaluation.metrics import compute_classification_metrics
from riscmal.evaluation.tracker import ContinualMetricsTracker
from riscmal.memory.buffer import RehearsalMemoryManager
from riscmal.memory.sampler import get_mixed_batch


# ===========================================================================
# 1. Data -> Memory -> Sampler Pipeline
# ===========================================================================

class TestDataToMemoryToSamplerPipeline:
    """Verifies the integration from dataset loading through memory update and mixed batching."""

    def test_dataset_to_loader_to_memory_update_pipeline(self, memory_manager, make_synthetic_loader):
        """Pipeline: Synthetic Dataset -> DataLoader -> Memory Exemplar Storage."""
        loader = make_synthetic_loader(num_samples=80, num_classes=2, class_offset=0)
        stats = memory_manager.update_memory_after_task(loader, new_classes=[0, 1])

        assert 0 in stats and 1 in stats
        assert stats[0] >= 1 and stats[1] >= 1
        assert memory_manager.seen_classes == [0, 1]

        # Verify exemplars format in memory
        for cls_id in [0, 1]:
            exemplars = memory_manager.memory[cls_id]
            assert len(exemplars) == stats[cls_id]
            for h, imp, i1d, i2d, api, y in exemplars:
                assert h.shape == (4,)
                assert imp.shape == (1000,)
                assert i1d.shape == (1024,)
                assert i2d.shape == (3, 224, 224)
                assert api.shape == (100,)
                assert int(y.item()) == cls_id

    def test_memory_to_mixed_batch_to_device(self, memory_manager, make_synthetic_loader, make_synthetic_batch):
        """Pipeline: Rehearsal Buffer + Incoming Batch -> Mixed Batch B_mixed=2N -> Device."""
        loader = make_synthetic_loader(num_samples=60, num_classes=2, class_offset=0) # Old: 0, 1
        memory_manager.update_memory_after_task(loader, new_classes=[0, 1])

        current_batch = make_synthetic_batch(batch_size=32, num_classes=2, class_offset=2) # New: 2, 3
        mixed_batch = get_mixed_batch(current_batch, memory_manager, batch_size=32, device="cpu")

        assert isinstance(mixed_batch, MultiViewBatch)
        assert mixed_batch.label.shape[0] == 64
        assert mixed_batch.header.shape == (64, 4)

        # Confirm exact composition: first 32 new, next 32 old
        new_labels = mixed_batch.label[:32].tolist()
        replay_labels = mixed_batch.label[32:].tolist()

        assert all(y in [2, 3] for y in new_labels)
        assert all(y in [0, 1] for y in replay_labels)


# ===========================================================================
# 2. Mixed Batch -> Model -> Loss & Optimization Pipeline
# ===========================================================================

class TestMixedBatchToModelToLossPipeline:
    """Verifies that mixed batches pass through the model and produce valid gradients."""

    def test_mixed_batch_forward_and_ce_loss(self, reference_model, memory_manager, make_synthetic_loader, make_synthetic_batch):
        """Pipeline: Mixed Batch -> Forward Pass -> Cross-Entropy Loss."""
        loader = make_synthetic_loader(num_samples=60, num_classes=2, class_offset=0)
        memory_manager.update_memory_after_task(loader, new_classes=[0, 1])

        # Model has 4 output classes (0, 1, 2, 3)
        current_batch = make_synthetic_batch(batch_size=32, num_classes=2, class_offset=2)
        mixed_batch = get_mixed_batch(current_batch, memory_manager, batch_size=32, device="cpu")

        logits, features = reference_model(mixed_batch)
        loss = F.cross_entropy(logits, mixed_batch.label)

        assert logits.shape == (64, 4)
        assert features.shape == (64, 64)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0

    def test_backward_gradient_flow_on_mixed_batch(self, reference_model, memory_manager, make_synthetic_loader, make_synthetic_batch):
        """Pipeline: Forward -> Loss -> Backward verifies gradients populate all subnets."""
        loader = make_synthetic_loader(num_samples=40, num_classes=2, class_offset=0)
        memory_manager.update_memory_after_task(loader, new_classes=[0, 1])

        current_batch = make_synthetic_batch(batch_size=16, num_classes=2, class_offset=2)
        mixed_batch = get_mixed_batch(current_batch, memory_manager, batch_size=16, device="cpu")

        reference_model.train()
        logits, _ = reference_model(mixed_batch)
        loss = F.cross_entropy(logits, mixed_batch.label)
        loss.backward()

        # Check gradients in header MLP, imports MLP, conv1d, LSTM, fusion, and head
        assert reference_model.backbone.fc_header[0].weight.grad is not None
        assert reference_model.backbone.fc_imports[0].weight.grad is not None
        assert reference_model.backbone.conv1d[0].weight.grad is not None
        assert reference_model.backbone.api_lstm.weight_ih_l0.grad is not None
        assert reference_model.backbone.fusion[0].weight.grad is not None
        assert reference_model.classifier_head.weight.grad is not None


# ===========================================================================
# 3. Model Transition -> Classifier Expansion -> Distillation Pipeline
# ===========================================================================

class TestContinualTransitionAndDistillationPipeline:
    """Verifies teacher snapshotting, classifier expansion, and distillation losses."""

    def test_classifier_expansion_with_frozen_teacher(self, reference_model, make_synthetic_batch):
        """Pipeline: Task 1 (C=4) -> Freeze Teacher -> Expand Student (C=6) -> Check Isolation."""
        # 1. Freeze teacher snapshot
        teacher_model = copy.deepcopy(reference_model)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

        # 2. Expand student head from 4 to 6 classes
        reference_model.expand_classes(num_new_classes=2)

        assert reference_model.classifier_head.out_features == 6
        assert teacher_model.classifier_head.out_features == 4

        # Verify old weights match exactly
        assert torch.equal(
            reference_model.classifier_head.weight.data[:4, :],
            teacher_model.classifier_head.weight.data,
        )

        # Verify teacher remains strictly frozen
        assert all(not p.requires_grad for p in teacher_model.parameters())

    def test_selective_plasticity_loss_formulation(self, reference_model, make_synthetic_batch):
        r"""Pipeline: Computes L_SP = lambda_sp * ||(h_S - h_T) \odot m^t||_2^2 (\mathcal{L}_{SP}, Eq. 6)."""
        teacher_model = copy.deepcopy(reference_model)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

        batch = make_synthetic_batch(batch_size=16)

        # Student and teacher forward passes
        _, student_features = reference_model(batch)
        with torch.no_grad():
            _, teacher_features = teacher_model(batch)

        # Soft protection mask m^t in (0, 1)^64 (Eq. 5)
        protection_mask = torch.sigmoid(torch.randn(64))

        # Compute Selective Representation Preservation loss (\mathcal{L}_{SP})
        lambda_sp = 0.1
        loss_sp = lambda_sp * (((student_features - teacher_features) * protection_mask) ** 2).sum(dim=1).mean()

        assert torch.isfinite(loss_sp)
        assert loss_sp.item() >= 0.0

        # Backpropagation updates only student parameters
        loss_sp.backward()
        assert reference_model.backbone.fusion[0].weight.grad is not None
        assert teacher_model.backbone.fusion[0].weight.grad is None

    def test_kd_distillation_loss_formulation(self, reference_model, make_synthetic_batch):
        r"""Pipeline: Evaluates output distillation loss \mathcal{L}_{KD} on historical classes (T_{KD}^2 = 4.0, Eq. 8)."""
        teacher_model = copy.deepcopy(reference_model)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

        # Student expands to 6 classes
        reference_model.expand_classes(num_new_classes=2)

        batch = make_synthetic_batch(batch_size=16, num_classes=2, class_offset=0) # Old classes 0, 1

        student_logits, _ = reference_model(batch)
        with torch.no_grad():
            teacher_logits, _ = teacher_model(batch)

        # Distillation on old 4 classes
        T = 2.0
        p_student = F.log_softmax(student_logits[:, :4] / T, dim=1)
        q_teacher = F.softmax(teacher_logits / T, dim=1)
        loss_kd = -1.0 * (q_teacher * p_student).sum(dim=1).mean() * (T ** 2)

        assert torch.isfinite(loss_kd)
        assert loss_kd.item() > 0.0

    def test_imbalance_calibrated_weights_calculation(self):
        r"""Pipeline: Evaluates rehearsal exposure weighting with \beta_R = 0.999 (Eq. 3)."""
        beta = 0.999
        class_counts = [5, 10, 50, 200]
        effective_nums = [(1.0 - beta**cnt) / (1.0 - beta) for cnt in class_counts]

        # Weights are inversely proportional to effective sample counts
        raw_weights = [1.0 / e for e in effective_nums]
        norm_weights = [w / sum(raw_weights) * len(class_counts) for w in raw_weights]

        # Minority class (count 5) must receive higher penalty weight than majority class (count 200)
        assert norm_weights[0] > norm_weights[-1]
        assert abs(sum(norm_weights) - len(class_counts)) < 1e-5


# ===========================================================================
# 4. Leakage Purge -> Memory Storage Pipeline
# ===========================================================================

class TestDataPurgeToMemoryIntegrityPipeline:
    """Verifies that purged training instances never contaminate rehearsal memory."""

    def test_purge_leakage_then_update_memory(self, memory_manager, make_synthetic_dataset):
        """Pipeline: Duplicate Injection -> SHA-256 Purge -> Memory Update -> Clean Repertoire."""
        train_ds = make_synthetic_dataset(num_samples=50, num_classes=2, class_offset=0)
        test_ds = make_synthetic_dataset(num_samples=20, num_classes=2, class_offset=0)

        # Inject test byte-streams into train
        test_hashes = {hashlib.sha256(test_ds.img1d[i].tobytes()).hexdigest() for i in range(len(test_ds))}
        train_ds.img1d[0] = test_ds.img1d[0].copy()
        train_ds.img1d[1] = test_ds.img1d[1].copy()

        # Execute purge
        purged = purge_data_leakage(train_ds, test_ds, verbose=False)
        assert purged >= 2

        # Ingest purged dataset via DataLoader into memory
        from torch.utils.data import DataLoader
        from riscmal.data.dataset import multiview_collate_fn
        loader = DataLoader(train_ds, batch_size=16, collate_fn=multiview_collate_fn)
        memory_manager.update_memory_after_task(loader, new_classes=[0, 1])

        # Assert no exemplars in memory share test hashes
        for cls_id in [0, 1]:
            for sample in memory_manager.memory[cls_id]:
                sample_img1d_bytes = sample[2].numpy().tobytes()
                sample_hash = hashlib.sha256(sample_img1d_bytes).hexdigest()
                assert sample_hash not in test_hashes, "Purged test exemplar leaked into rehearsal buffer!"


# ===========================================================================
# 5. Inference -> Metrics -> Continual Tracker Pipeline
# ===========================================================================

class TestInferenceToMetricsToTrackerPipeline:
    """Verifies evaluation loop collecting predictions, calculating metrics, and tracking forgetting."""

    def test_end_to_end_predictions_to_metrics_and_tracker(self, reference_model, make_synthetic_loader):
        """Pipeline: Model Inference -> Metric Engine -> Continual Tracker."""
        tracker = ContinualMetricsTracker()
        loader = make_synthetic_loader(num_samples=60, num_classes=4, class_offset=0, batch_size=20)

        reference_model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                logits, _ = reference_model(batch)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(batch.label.cpu().tolist())

        # Step 1: Classification metrics
        metrics = compute_classification_metrics(all_targets, all_preds)
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert metrics["num_classes_seen"] == 4

        # Step 2: Continual Tracker
        f, old_f1, new_f1, h_mean = tracker.calculate_continual_metrics(
            metrics,
            current_task_id=1,
            classes_per_task=4,
        )

        assert f == 0.0 # Base task has no forgetting
        assert tracker.best_historical_recall is not None
        assert len(tracker.best_historical_recall) > 0
