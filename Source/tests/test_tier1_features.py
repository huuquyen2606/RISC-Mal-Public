"""Tier 1: Comprehensive Feature Coverage Tests.

Covers:
- Backbones & Fusion (Features 1-6)
- Unified Classifier Head & Norm Matching (Feature 7)
- Rehearsal Memory Buffer (Feature 9)
- Mixed Batch Sampler (Feature 10)
- Multi-View Dataset & Batching (Feature 11)
- SHA-256 Duplicate Purging & PE Extraction (Features 12, 13)
- Multi-Metric Engine & Continual Tracker (Features 14, 15)
- Deterministic Seed Anchoring (Feature 16)
"""

from pathlib import Path
import copy
import hashlib
import numpy as np
import pytest
import torch
import torch.nn as nn

from riscmal.data.dataset import MalwareMultiViewDataset, MultiViewBatch
from riscmal.data.leakage import purge_data_leakage
from riscmal.data.pe_features import clean_numeric, extract_byte_images
from riscmal.evaluation.metrics import compute_classification_metrics
from riscmal.evaluation.tracker import ContinualMetricsTracker
from riscmal.memory.buffer import RehearsalMemoryManager
from riscmal.memory.sampler import get_mixed_batch
from riscmal.utils.device import get_device
from riscmal.utils.seed import set_deterministic_seed


# ===========================================================================
# 1. Backbones & Fusion Tests (Features 1-6)
# ===========================================================================

class TestBackbonesAndFusion:
    """Verifies dimension mapping, layer activations, and fusion logic."""

    def test_header_backbone_dimension_and_range(self, reference_model, make_synthetic_batch):
        """Feature 1: PE Header MLP maps [B, 4] -> [B, 64]."""
        batch = make_synthetic_batch(batch_size=16)
        h_feat = reference_model.backbone.fc_header(batch["header"])
        assert h_feat.shape == (16, 64), f"Expected (16, 64), got {h_feat.shape}"
        assert not torch.isnan(h_feat).any(), "NaNs detected in header representation"

    def test_imports_backbone_dimension_and_sparsity(self, reference_model, make_synthetic_batch):
        """Feature 2: Imports MLP maps [B, 1000] -> [B, 64]."""
        batch = make_synthetic_batch(batch_size=8)
        imp_feat = reference_model.backbone.fc_imports(batch["imports"])
        assert imp_feat.shape == (8, 64), f"Expected (8, 64), got {imp_feat.shape}"
        assert not torch.isnan(imp_feat).any()

    def test_conv1d_backbone_spatial_downsampling(self, reference_model, make_synthetic_batch):
        """Feature 3: 1D byte stream Conv1D maps [B, 1024] -> [B, 64]."""
        batch = make_synthetic_batch(batch_size=12)
        conv_in = batch["img1d"].unsqueeze(1)
        conv_feat = reference_model.backbone.conv1d(conv_in)
        assert conv_feat.shape == (12, 64), f"Expected (12, 64), got {conv_feat.shape}"
        assert not torch.isinf(conv_feat).any()

    def test_texture_backbone_channels_and_resolution(self, reference_model, make_synthetic_batch):
        """Feature 4: 2D texture representation maps [B, 3, 224, 224] -> [B, 64]."""
        batch = make_synthetic_batch(batch_size=4)
        tex_feat = reference_model.backbone.texture(batch["img2d"])
        assert tex_feat.shape == (4, 64), f"Expected (4, 64), got {tex_feat.shape}"

    def test_api_sequence_embedding_and_recurrent(self, reference_model, make_synthetic_batch):
        """Feature 5: Dynamic API call sequence maps [B, 100] -> [B, 64]."""
        batch = make_synthetic_batch(batch_size=10)
        emb = reference_model.backbone.api_embed(batch["apis"].clamp(0, 100))
        assert emb.shape == (10, 100, 32)
        _, (hn, _) = reference_model.backbone.api_lstm(emb)
        api_feat = hn[-1]
        assert api_feat.shape == (10, 64), f"Expected (10, 64), got {api_feat.shape}"

    def test_multiview_fusion_concatenation_and_projection(self, reference_model, make_synthetic_batch):
        """Feature 6: Multi-View Fusion concatenates 5x64=320 -> fixed representation d=64."""
        batch = make_synthetic_batch(batch_size=14)
        features = reference_model.backbone(
            batch["header"],
            batch["imports"],
            batch["img1d"],
            batch["img2d"],
            batch["apis"],
        )
        assert features.shape == (14, 64), f"Expected (14, 64), got {features.shape}"
        assert not torch.isnan(features).any()


# ===========================================================================
# 2. Unified Classifier Head & Expansion Tests (Feature 7)
# ===========================================================================

class TestUnifiedClassifierHead:
    """Verifies linear head prediction, norm-matched expansion, and weight preservation."""

    def test_classifier_head_forward(self, reference_model, make_synthetic_batch):
        """Tests forward pass returning logits and intermediate features."""
        batch = make_synthetic_batch(batch_size=16, num_classes=4)
        logits, feats = reference_model(batch)
        assert logits.shape == (16, 4), f"Expected (16, 4), got {logits.shape}"
        assert feats.shape == (16, 64), f"Expected (16, 64), got {feats.shape}"

    def test_classifier_expansion_shape(self, reference_model):
        """Tests expanding classifier from C=4 to C=6."""
        assert reference_model.classifier_head.out_features == 4
        reference_model.expand_classes(num_new_classes=2)
        assert reference_model.classifier_head.out_features == 6
        assert reference_model.classifier_head.in_features == 64

    def test_classifier_expansion_old_weight_preservation(self, reference_model):
        """Tests that expanding the head leaves existing class weights 100% untouched."""
        old_weights = reference_model.classifier_head.weight.detach().clone()
        old_bias = reference_model.classifier_head.bias.detach().clone()

        reference_model.expand_classes(num_new_classes=3)

        new_weights = reference_model.classifier_head.weight.detach()
        new_bias = reference_model.classifier_head.bias.detach()

        # Check exact equality for old slices
        assert torch.equal(new_weights[:4, :], old_weights), "Old class weights were mutated during expansion!"
        assert torch.equal(new_bias[:4], old_bias), "Old class biases were mutated during expansion!"

    def test_classifier_expansion_norm_matching(self, reference_model):
        """Verifies mathematical contract: ||w_new||_2 matches average norm r_bar."""
        old_weights = reference_model.classifier_head.weight.detach()
        expected_r_bar = float(torch.norm(old_weights, dim=1).mean().item())

        reference_model.expand_classes(num_new_classes=2)

        new_weights = reference_model.classifier_head.weight.detach()
        novel_norms = torch.norm(new_weights[4:, :], dim=1)

        for idx, norm_val in enumerate(novel_norms):
            assert abs(norm_val.item() - expected_r_bar) < 1e-5, (
                f"Norm of novel class {idx} ({norm_val.item()}) does not match r_bar ({expected_r_bar})"
            )

    def test_classifier_expansion_bias_initialization(self, reference_model):
        """Verifies contract: b_new == 0.0 for novel classes."""
        reference_model.expand_classes(num_new_classes=4)
        new_biases = reference_model.classifier_head.bias.detach()[4:]
        assert torch.all(new_biases == 0.0), f"Expected 0.0 biases for new classes, got {new_biases}"


# ===========================================================================
# 3. Rehearsal Memory Buffer Tests (Feature 9)
# ===========================================================================

class TestRehearsalMemoryBuffer:
    """Verifies class-proportional exemplar storage, minimum floors, and state dicts."""

    def test_memory_buffer_class_proportional_retention(self, memory_manager, make_synthetic_loader):
        """Feature 9: Stores rho=0.10 samples per class with min_per_class=1."""
        loader = make_synthetic_loader(num_samples=100, num_classes=2, class_offset=0)
        # Class 0: ~50 samples -> floor(50 * 0.10) >= 1
        # Class 1: ~50 samples -> floor(50 * 0.10) >= 1
        stats = memory_manager.update_memory_after_task(loader, new_classes=[0, 1])

        assert 0 in memory_manager.memory
        assert 1 in memory_manager.memory
        assert len(memory_manager.memory[0]) >= 1
        assert len(memory_manager.memory[1]) >= 1
        assert memory_manager.get_total_exemplars() == len(memory_manager.memory[0]) + len(memory_manager.memory[1])

    def test_memory_buffer_cpu_tensor_isolation(self, memory_manager, make_synthetic_loader):
        """Verifies that all stored tensors are detached on CPU memory."""
        loader = make_synthetic_loader(num_samples=20, num_classes=2)
        memory_manager.update_memory_after_task(loader, new_classes=[0, 1])

        for cls_id, exemplars in memory_manager.memory.items():
            for sample in exemplars:
                for tensor in sample:
                    assert isinstance(tensor, torch.Tensor)
                    assert tensor.device.type == "cpu", f"Stored exemplar tensor not on CPU: {tensor.device}"
                    assert not tensor.requires_grad, "Stored exemplar tensor retained grad history"

    def test_memory_buffer_multi_class_ingestion(self, memory_manager, make_synthetic_loader):
        """Verifies memory accumulation across consecutive tasks."""
        loader1 = make_synthetic_loader(num_samples=30, num_classes=2, class_offset=0)
        memory_manager.update_memory_after_task(loader1, new_classes=[0, 1])
        assert memory_manager.seen_classes == [0, 1]

        loader2 = make_synthetic_loader(num_samples=30, num_classes=2, class_offset=2)
        memory_manager.update_memory_after_task(loader2, new_classes=[2, 3])
        assert memory_manager.seen_classes == [0, 1, 2, 3]
        assert len(memory_manager.memory) == 4

    def test_memory_buffer_state_dict_serialization(self, memory_manager, make_synthetic_loader):
        """Tests checkpoint serialization and state restoration."""
        loader = make_synthetic_loader(num_samples=40, num_classes=2)
        memory_manager.update_memory_after_task(loader, new_classes=[0, 1])

        state = memory_manager.state_dict()
        assert "memory" in state
        assert "seen_classes" in state

        restored_manager = RehearsalMemoryManager(replay_ratio=0.10)
        restored_manager.load_state_dict(state)

        assert restored_manager.seen_classes == memory_manager.seen_classes
        assert len(restored_manager.memory) == len(memory_manager.memory)

    def test_memory_buffer_clear(self, memory_manager, make_synthetic_loader):
        """Tests wiping the memory buffer."""
        loader = make_synthetic_loader(num_samples=20, num_classes=2)
        memory_manager.update_memory_after_task(loader, new_classes=[0, 1])
        assert memory_manager.get_total_exemplars() > 0

        memory_manager.clear()
        assert memory_manager.get_total_exemplars() == 0
        assert len(memory_manager.seen_classes) == 0


# ===========================================================================
# 4. Mixed Batch Sampler Tests (Feature 10)
# ===========================================================================

class TestMixedBatchSampler:
    """Verifies balanced rehearsal concatenation B_mixed = 2N = 64."""

    def test_mixed_batch_sampler_output_shape(self, memory_manager, make_synthetic_loader, make_synthetic_batch):
        """Feature 10: Concatenates incoming N=32 with replay N=32 to yield B_mixed=64."""
        # Setup memory with old classes 0 and 1
        loader = make_synthetic_loader(num_samples=60, num_classes=2, class_offset=0)
        memory_manager.update_memory_after_task(loader, new_classes=[0, 1])

        # Current incoming batch has classes 2 and 3
        current_batch = make_synthetic_batch(batch_size=32, num_classes=2, class_offset=2)
        mixed_batch = get_mixed_batch(current_batch, memory_manager, batch_size=32, device="cpu")

        assert mixed_batch["label"].shape[0] == 64
        assert mixed_batch["header"].shape == (64, 4)
        assert mixed_batch["imports"].shape == (64, 1000)
        assert mixed_batch["img1d"].shape == (64, 1024)
        assert mixed_batch["img2d"].shape == (64, 3, 224, 224)
        assert mixed_batch["apis"].shape == (64, 100)

    def test_mixed_batch_sampler_preserves_modalities(self, memory_manager, make_synthetic_loader, make_synthetic_batch):
        """Verifies that all 6 modalities are present and correctly concatenated."""
        loader = make_synthetic_loader(num_samples=40, num_classes=2, class_offset=0)
        memory_manager.update_memory_after_task(loader, new_classes=[0, 1])

        current_batch = make_synthetic_batch(batch_size=16, num_classes=2, class_offset=2)
        mixed_batch = get_mixed_batch(current_batch, memory_manager, batch_size=16, device="cpu")

        for key in ["header", "imports", "img1d", "img2d", "apis", "label"]:
            assert key in mixed_batch
            assert len(mixed_batch[key]) == 32

    def test_mixed_batch_sampler_excludes_current_classes(self, memory_manager, make_synthetic_loader, make_synthetic_batch):
        """Verifies that replay exemplars are drawn exclusively from historical (old) classes."""
        loader_old = make_synthetic_loader(num_samples=50, num_classes=2, class_offset=0) # 0, 1
        memory_manager.update_memory_after_task(loader_old, new_classes=[0, 1])

        current_batch = make_synthetic_batch(batch_size=32, num_classes=2, class_offset=2) # 2, 3
        mixed_batch = get_mixed_batch(current_batch, memory_manager, batch_size=32, device="cpu")

        # First 32 labels are from current batch (2, 3)
        # Second 32 labels must be from memory (0, 1)
        replay_labels = mixed_batch["label"][32:].tolist()
        for y in replay_labels:
            assert y in [0, 1], f"Replay exemplar label {y} should be in old classes [0, 1]"

    def test_mixed_batch_sampler_with_tuple(self, memory_manager, make_synthetic_loader, make_synthetic_batch):
        """Verifies compatibility with tuple batches: (h, imp, i1d, i2d, api, y)."""
        loader = make_synthetic_loader(num_samples=40, num_classes=2, class_offset=0)
        memory_manager.update_memory_after_task(loader, new_classes=[0, 1])

        dict_batch = make_synthetic_batch(batch_size=16, num_classes=2, class_offset=2)
        tuple_batch = dict_batch.as_tuple()

        mixed_tuple = get_mixed_batch(tuple_batch, memory_manager, batch_size=16, device="cpu")
        assert isinstance(mixed_tuple, tuple)
        assert len(mixed_tuple) == 6
        assert mixed_tuple[0].shape == (32, 4)
        assert mixed_tuple[-1].shape == (32,)

    def test_mixed_batch_sampler_device_transfer(self, memory_manager, make_synthetic_loader, make_synthetic_batch):
        """Verifies mixed batch is placed onto the target device."""
        loader = make_synthetic_loader(num_samples=30, num_classes=2, class_offset=0)
        memory_manager.update_memory_after_task(loader, new_classes=[0, 1])

        current_batch = make_synthetic_batch(batch_size=10, num_classes=2, class_offset=2)
        mixed = get_mixed_batch(current_batch, memory_manager, batch_size=10, device="cpu")
        assert mixed["header"].device == torch.device("cpu")


# ===========================================================================
# 5. Multi-View Dataset & Batching Tests (Feature 11)
# ===========================================================================

class TestMultiViewDataset:
    """Verifies dataset indexing, modality synchronization, and MultiViewBatch access."""

    def test_dataset_len_and_getitem(self, make_synthetic_dataset):
        """Feature 11: Multi-view dataset delivers 6 modalities per sample."""
        ds = make_synthetic_dataset(num_samples=25, num_classes=3)
        assert len(ds) == 25

        sample = ds[0]
        assert isinstance(sample, dict)
        assert sample["header"].shape == (4,)
        assert sample["imports"].shape == (1000,)
        assert sample["img1d"].shape == (1024,)
        assert sample["img2d"].shape == (3, 224, 224)
        assert sample["apis"].shape == (100,)
        assert isinstance(sample["label"], (int, np.integer, torch.Tensor))

    def test_multiview_batch_dict_and_attribute_access(self, make_synthetic_batch):
        """Tests MultiViewBatch dual dictionary and attribute access semantics."""
        batch = make_synthetic_batch(batch_size=8)
        assert torch.equal(batch["header"], batch.header)
        assert torch.equal(batch["label"], batch.label)
        assert torch.equal(batch["apis"], batch.apis)

    def test_multiview_batch_to_device(self, make_synthetic_batch):
        """Tests transferring all batch tensors to a target device."""
        batch = make_synthetic_batch(batch_size=8)
        transferred = batch.to("cpu")
        assert transferred.header.device.type == "cpu"
        assert transferred.label.device.type == "cpu"


# ===========================================================================
# 6. Data Leakage Purge & PE Feature Extraction Tests (Features 12, 13)
# ===========================================================================

class TestDataLeakageAndExtraction:
    """Verifies SHA-256 byte hashing deduplication and PE feature parsing."""

    def test_sha256_purge_eliminates_duplicates(self, make_synthetic_dataset):
        """Feature 12: In-place elimination of train-test duplicates via byte hashing."""
        train_ds = make_synthetic_dataset(num_samples=30, num_classes=2)
        test_ds = make_synthetic_dataset(num_samples=20, num_classes=2)

        # Deliberately inject duplicate samples from test into train
        train_ds.img1d[0] = test_ds.img1d[5].copy()
        train_ds.img1d[1] = test_ds.img1d[10].copy()

        purged_count = purge_data_leakage(train_ds, test_ds, verbose=False)
        assert purged_count >= 2
        assert len(train_ds) == 30 - purged_count

        # Verify no remaining hashes overlap
        train_hashes = {hashlib.sha256(train_ds.img1d[i].tobytes()).hexdigest() for i in range(len(train_ds))}
        test_hashes = {hashlib.sha256(test_ds.img1d[i].tobytes()).hexdigest() for i in range(len(test_ds))}
        assert len(train_hashes.intersection(test_hashes)) == 0

    def test_clean_numeric_utility(self):
        """Feature 13: Robust float parsing from hex, ints, floats, and malformed strings."""
        assert clean_numeric(42) == 42.0
        assert clean_numeric("3.14") == 3.14
        assert clean_numeric("0x1000") == 4096.0
        assert clean_numeric(None) == 0.0
        assert clean_numeric("invalid_string") == 0.0

    def test_extract_byte_images_with_mock_pe(self, mock_pe_file):
        """Feature 13: Generates 1D (1024) and 2D (224x224x3) byte visualizations from PE."""
        img1d, img2d = extract_byte_images(mock_pe_file)
        assert img1d.shape == (1024,)
        assert img2d.shape == (224, 224, 3)
        assert img1d.dtype == np.float32
        assert img2d.dtype == np.float32
        assert (img1d >= 0.0).all() and (img1d <= 1.0).all()
        assert (img2d >= 0.0).all() and (img2d <= 1.0).all()


# ===========================================================================
# 7. Multi-Metric Engine & Continual Tracker Tests (Features 14, 15)
# ===========================================================================

class TestMetricsAndTracker:
    """Verifies Acc, Prec, Rec, F1, Worst-Class Recall, Forgetting, and BWT."""

    def test_classification_metrics_perfect_predictions(self):
        """Feature 14: Perfect accuracy and F1 score computation."""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        res = compute_classification_metrics(y_true, y_pred)

        assert res["accuracy"] == 1.0
        assert res["macro_f1"] == 1.0
        assert res["worst_class_recall"] == 1.0
        assert res["num_classes_seen"] == 4

    def test_classification_metrics_imbalanced_and_worst_class(self):
        """Feature 14: Evaluates metrics with severe minority class failure."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0]) # Class 1 completely missed
        res = compute_classification_metrics(y_true, y_pred)

        assert res["worst_class_recall"] == 0.0
        assert res["per_class"]["1"]["recall"] == 0.0
        assert res["per_class"]["0"]["recall"] == 1.0

    def test_classification_metrics_empty_array_handling(self):
        """Feature 14: Safe handling of zero-length evaluation inputs."""
        res = compute_classification_metrics([], [])
        assert res["accuracy"] == 0.0
        assert res["macro_f1"] == 0.0
        assert res["worst_class_recall"] == 0.0

    def test_continual_tracker_initial_state_and_task1(self):
        """Feature 15: Task 1 continual metrics report zero forgetting."""
        tracker = ContinualMetricsTracker()
        report = {
            "num_classes_seen": 2,
            "per_class": {
                "0": {"recall": 0.95, "f1-score": 0.94},
                "1": {"recall": 0.90, "f1-score": 0.89},
            }
        }
        f, old_f1, new_f1, h_mean = tracker.calculate_continual_metrics(report, current_task_id=1, classes_per_task=2)
        assert f == 0.0
        assert old_f1 == 0.0
        assert new_f1 > 0.0

    def test_continual_tracker_forgetting_calculation(self):
        """Feature 15: Evaluates catastrophic forgetting on historical classes in Task 2."""
        tracker = ContinualMetricsTracker()
        # Task 1: Classes 0, 1 with 1.0 recall
        report_t1 = {
            "num_classes_seen": 2,
            "per_class": {
                "0": {"recall": 1.0, "f1-score": 0.95},
                "1": {"recall": 1.0, "f1-score": 0.95},
            }
        }
        tracker.calculate_continual_metrics(report_t1, current_task_id=1, classes_per_task=2)

        # Task 2: Classes 0, 1 drop to 0.70 and 0.80; New classes 2, 3 have 0.90
        report_t2 = {
            "num_classes_seen": 4,
            "per_class": {
                "0": {"recall": 0.70, "f1-score": 0.70},
                "1": {"recall": 0.80, "f1-score": 0.80},
                "2": {"recall": 0.90, "f1-score": 0.90},
                "3": {"recall": 0.90, "f1-score": 0.90},
            }
        }
        f, old_f1, new_f1, h_mean = tracker.calculate_continual_metrics(report_t2, current_task_id=2, classes_per_task=2)

        # Forgetting: c0 = 1.0 - 0.7 = 0.3; c1 = 1.0 - 0.8 = 0.2 -> avg = 0.25
        assert abs(f - 0.25) < 1e-4, f"Expected forgetting 0.25, got {f}"
        assert abs(old_f1 - 0.75) < 1e-4
        assert abs(new_f1 - 0.90) < 1e-4


# ===========================================================================
# 8. Deterministic Seed Anchoring Tests (Feature 16)
# ===========================================================================

class TestDeterministicUtilities:
    """Verifies reproducibility across random generators."""

    def test_deterministic_seed_anchoring(self):
        """Feature 16: Seed 42 anchors torch, numpy, and python random."""
        set_deterministic_seed(42)
        r1 = torch.randn(5)
        np1 = np.random.rand(5)

        set_deterministic_seed(42)
        r2 = torch.randn(5)
        np2 = np.random.rand(5)

        assert torch.equal(r1, r2), "PyTorch random outputs differ despite identical seed!"
        assert np.array_equal(np1, np2), "NumPy random outputs differ despite identical seed!"
