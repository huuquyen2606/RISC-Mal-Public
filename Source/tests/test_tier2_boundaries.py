"""Tier 2: Boundary, Edge Case, and Malformed Input Tests.

Covers:
- Empty rehearsal memory buffers (Task 1 cold-start)
- Single-sample rare classes (floor min=1 constraint)
- Extreme shapes and non-standard batch sizes (B=1, B=128)
- Missing modality keys and malformed dictionaries
- Corrupted byte files and zero-byte inputs
- Extreme class imbalance coefficients (beta -> 1.0)
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from riscmal.data.dataset import MalwareMultiViewDataset, MultiViewBatch
from riscmal.data.pe_features import clean_numeric, extract_byte_images
from riscmal.evaluation.metrics import compute_classification_metrics
from riscmal.memory.buffer import RehearsalMemoryManager
from riscmal.memory.sampler import get_mixed_batch


# ===========================================================================
# 1. Empty Buffer & Cold Start Boundaries
# ===========================================================================

class TestEmptyMemoryBoundaries:
    """Verifies behavior when rehearsal buffer is empty or unpopulated."""

    def test_mixed_batch_with_completely_empty_memory(self, memory_manager, make_synthetic_batch):
        """Task 1 Cold-Start: Buffer empty, get_mixed_batch returns current batch unmodified."""
        batch = make_synthetic_batch(batch_size=32)
        assert len(memory_manager.memory) == 0

        mixed = get_mixed_batch(batch, memory_manager, batch_size=32, device="cpu")
        assert len(mixed["label"]) == 32
        assert torch.equal(mixed["label"], batch["label"])
        assert torch.equal(mixed["header"], batch["header"])

    def test_mixed_batch_when_seen_classes_subset_of_current(self, memory_manager, make_synthetic_loader, make_synthetic_batch):
        """When memory only contains classes present in the current batch, no old classes eligible."""
        # Memory populated only with class 0
        loader = make_synthetic_loader(num_samples=20, num_classes=1, class_offset=0)
        memory_manager.update_memory_after_task(loader, new_classes=[0])

        # Current incoming batch also contains only class 0
        current_batch = make_synthetic_batch(batch_size=16, num_classes=1, class_offset=0)
        mixed = get_mixed_batch(current_batch, memory_manager, batch_size=16, device="cpu")

        # Since class 0 is current, no 'old' class is eligible for replay -> returns current batch
        assert len(mixed["label"]) == 16
        assert torch.equal(mixed["label"], current_batch["label"])

    def test_memory_update_with_empty_dataset(self, memory_manager):
        """Updating memory with an empty dataset handles gracefully without crashing."""
        empty_ds = []
        stats = memory_manager.update_memory_after_task(empty_ds, new_classes=[0, 1])
        assert stats[0] == 0
        assert stats[1] == 0
        assert memory_manager.get_total_exemplars() == 0


# ===========================================================================
# 2. Single-Sample & Extreme Class Imbalance Boundaries
# ===========================================================================

class TestSingleSampleAndImbalanceBoundaries:
    """Verifies floor retention min=1 and effective sample math stability."""

    def test_single_sample_class_retention_floor(self, memory_manager):
        """When class has 1 sample, floor(1 * 0.10) = 0, but min_per_class=1 retains 1 exemplar."""
        h = torch.randn(1, 4)
        imp = torch.zeros(1, 1000)
        i1d = torch.zeros(1, 1024)
        i2d = torch.zeros(1, 3, 224, 224)
        api = torch.zeros(1, 100, dtype=torch.long)
        y = torch.tensor([5], dtype=torch.long)

        single_sample_batch = [(h, imp, i1d, i2d, api, y)]
        stats = memory_manager.update_memory_after_task(single_sample_batch, new_classes=[5])

        assert stats[5] == 1, "Floor constraint min=1 failed to retain single exemplar!"
        assert len(memory_manager.memory[5]) == 1

    def test_extreme_class_imbalance_effective_numbers(self):
        """Mathematical stability of (1 - beta^n) / (1 - beta) under extreme beta=0.9999."""
        beta = 0.9999
        counts = [1, 10, 100, 1000, 50000]
        effective_nums = [(1.0 - beta**cnt) / (1.0 - beta) for cnt in counts]

        # Ensure monotonic growth and finite non-NaN numbers
        for idx in range(len(effective_nums) - 1):
            assert effective_nums[idx] < effective_nums[idx + 1]
            assert not np.isnan(effective_nums[idx])
            assert not np.isinf(effective_nums[idx])

    def test_metrics_with_single_class_ground_truth(self):
        """Evaluation when test set contains only 1 class (boundary condition)."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["worst_class_recall"] == 1.0
        assert metrics["num_classes_seen"] == 1


# ===========================================================================
# 3. Extreme Batch Sizes & Shapes
# ===========================================================================

class TestExtremeShapesAndBatchSizes:
    """Verifies pipeline behavior under B=1 and large B=128."""

    def test_single_sample_batch_forward(self, reference_model, make_synthetic_batch):
        """Single-sample batch B=1 passes through model without BatchNorm dimension errors."""
        reference_model.eval()
        batch_1 = make_synthetic_batch(batch_size=1)
        logits, feats = reference_model(batch_1)
        assert logits.shape == (1, 4)
        assert feats.shape == (1, 64)

    def test_large_batch_processing(self, reference_model, make_synthetic_batch):
        """Large batch B=128 executes cleanly."""
        batch_128 = make_synthetic_batch(batch_size=128)
        logits, feats = reference_model(batch_128)
        assert logits.shape == (128, 4)
        assert feats.shape == (128, 64)

    def test_arbitrary_odd_batch_sampler(self, memory_manager, make_synthetic_loader, make_synthetic_batch):
        """Odd batch size B=7 concatenates correctly."""
        loader = make_synthetic_loader(num_samples=20, num_classes=2, class_offset=0)
        memory_manager.update_memory_after_task(loader, new_classes=[0, 1])

        current_batch = make_synthetic_batch(batch_size=7, num_classes=2, class_offset=2)
        mixed = get_mixed_batch(current_batch, memory_manager, batch_size=7, device="cpu")
        assert len(mixed["label"]) == 14


# ===========================================================================
# 4. Malformed Inputs & Corrupt Files
# ===========================================================================

class TestMalformedInputsAndCorruptFiles:
    """Verifies error handling for missing keys, corrupt byte streams, and bad values."""

    def test_missing_modality_key_raises_key_error(self, reference_model, make_synthetic_batch):
        """Passing a batch missing a required modality key raises KeyError."""
        batch = make_synthetic_batch(batch_size=4)
        del batch["header"] # Deliberately remove header
        with pytest.raises(KeyError):
            _ = reference_model(batch)

    def test_corrupt_empty_byte_file_extraction(self, tmp_path: Path):
        """Extracting byte images from an empty 0-byte file generates zeroed arrays safely."""
        empty_file = tmp_path / "empty.bin"
        empty_file.write_bytes(b"")

        img1d, img2d = extract_byte_images(empty_file)
        assert img1d.shape == (1024,)
        assert img2d.shape == (224, 224, 3)
        assert (img1d == 0.0).all()

    def test_clean_numeric_boundary_values(self):
        """Tests whitespace strings, huge negative numbers, and scientific notation."""
        assert clean_numeric("   ") == 0.0
        assert clean_numeric("-123.45") == -123.45
        assert clean_numeric("1e-4") == 0.0001
        assert clean_numeric(float("inf")) == float("inf")
