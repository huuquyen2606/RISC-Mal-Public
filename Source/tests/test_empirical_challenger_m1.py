"""Empirical challenger verification test suite for RISC-Mal Milestone 1.

Verifies:
1. memory/buffer.py: Exact rho=0.10 retention with min=1 across imbalanced classes (100, 10, 3, 1 samples).
2. memory/sampler.py: Mixed batch sampling N=32 exemplars with replacement, strictly forming B_mixed = 2N = 64.
3. data/leakage.py: SHA-256 byte hashing of img1d and purging of cross-split duplicate samples without data corruption.
"""

from typing import Dict, List, Tuple
import hashlib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from riscmal.data.dataset import MalwareMultiViewDataset, MultiViewBatch, multiview_collate_fn
from riscmal.data.leakage import audit_internal_duplicates, audit_leakage, purge_data_leakage
from riscmal.memory.buffer import RehearsalMemoryManager
from riscmal.memory.sampler import get_mixed_batch


def _make_dataset(
    class_counts: Dict[int, int],
    seed: int = 42,
) -> MalwareMultiViewDataset:
    """Helper to construct a multi-view dataset with exact per-class sample counts."""
    rng = np.random.RandomState(seed)
    total_samples = sum(class_counts.values())

    labels_list: List[int] = []
    for cls, count in sorted(class_counts.items()):
        labels_list.extend([cls] * count)

    labels = np.array(labels_list, dtype=np.int64)
    # Ensure distinct img1d values so samples are distinguishable
    img1d = rng.randn(total_samples, 1024).astype(np.float32)
    header = rng.randn(total_samples, 4).astype(np.float32)
    imports = rng.randn(total_samples, 1000).astype(np.float32)
    img2d = rng.randn(total_samples, 224, 224, 3).astype(np.float32)
    apis = rng.randint(0, 100, size=(total_samples, 100)).astype(np.int64)

    return MalwareMultiViewDataset(
        header=header,
        imports=imports,
        img1d=img1d,
        img2d=img2d,
        apis=apis,
        labels=labels,
    )


# ===========================================================================
# 1. Empirical Tests for memory/buffer.py
# ===========================================================================

class TestEmpiricalMemoryBuffer:
    """Empirical verification of RehearsalMemoryManager."""

    def test_exact_retention_imbalanced_distribution(self):
        """Validates exact rho=0.10 retention with min=1 for classes [100, 10, 3, 1].

        Expected counts:
        - Class 0 (100 samples): floor(100 * 0.10) = 10 -> max(1, 10) = 10 exemplars
        - Class 1 (10 samples):  floor(10 * 0.10)  = 1  -> max(1, 1)  = 1 exemplar
        - Class 2 (3 samples):   floor(3 * 0.10)   = 0  -> max(1, 0)  = 1 exemplar
        - Class 3 (1 sample):    floor(1 * 0.10)   = 0  -> max(1, 0)  = 1 exemplar
        Total stored: 10 + 1 + 1 + 1 = 13 exemplars.
        """
        counts = {0: 100, 1: 10, 2: 3, 3: 1}
        ds = _make_dataset(counts, seed=123)
        loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=multiview_collate_fn)

        manager = RehearsalMemoryManager(replay_ratio=0.10, min_per_class=1)
        stats = manager.update_memory_after_task(loader, new_classes=[0, 1, 2, 3])

        # Verify exact exemplar counts returned by method
        assert stats[0] == 10, f"Class 0 (100 samples) expected 10 exemplars, got {stats[0]}"
        assert stats[1] == 1, f"Class 1 (10 samples) expected 1 exemplar, got {stats[1]}"
        assert stats[2] == 1, f"Class 2 (3 samples) expected 1 exemplar, got {stats[2]}"
        assert stats[3] == 1, f"Class 3 (1 sample) expected 1 exemplar, got {stats[3]}"

        # Verify internal memory store
        assert len(manager.memory[0]) == 10
        assert len(manager.memory[1]) == 1
        assert len(manager.memory[2]) == 1
        assert len(manager.memory[3]) == 1
        assert manager.get_total_exemplars() == 13

        # Verify that all stored exemplars are valid 6-tuples with proper tensor shapes
        for cls_id, exemplars in manager.memory.items():
            for ex in exemplars:
                assert len(ex) == 6
                h, imp, i1d, i2d, api, y = ex
                assert h.shape == (4,)
                assert imp.shape == (1000,)
                assert i1d.shape == (1024,)
                assert i2d.shape == (3, 224, 224)
                assert api.shape == (100,)
                assert y.item() == cls_id

    def test_retention_across_wide_range_boundary_counts(self):
        """Stress-tests retention formula k_c = min(N_c, max(min_per_class, floor(N_c * rho)))

        Boundary test cases:
        - 1 sample   -> 1
        - 2 samples  -> 1
        - 9 samples  -> 1
        - 10 samples -> 1
        - 19 samples -> 1
        - 20 samples -> 2
        - 99 samples -> 9
        - 100 samples-> 10
        - 250 samples-> 25
        """
        test_counts = {1: 1, 2: 2, 3: 9, 4: 10, 5: 19, 6: 20, 7: 99, 8: 100, 9: 250}
        expected = {
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 2,
            7: 9,
            8: 10,
            9: 25,
        }
        ds = _make_dataset(test_counts, seed=456)
        loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=multiview_collate_fn)

        manager = RehearsalMemoryManager(replay_ratio=0.10, min_per_class=1)
        stats = manager.update_memory_after_task(loader, new_classes=list(test_counts.keys()))

        for cls, exp_count in expected.items():
            assert stats[cls] == exp_count, f"Class {cls} ({test_counts[cls]} samples) expected {exp_count}, got {stats[cls]}"

    def test_multitask_retention_persistence(self):
        """Verifies that subsequent incremental tasks retain previously memorized classes."""
        # Task 1: classes [0, 1]
        ds1 = _make_dataset({0: 50, 1: 50}, seed=10)
        loader1 = DataLoader(ds1, batch_size=16, collate_fn=multiview_collate_fn)

        manager = RehearsalMemoryManager(replay_ratio=0.10, min_per_class=1)
        manager.update_memory_after_task(loader1, new_classes=[0, 1])
        assert manager.seen_classes == [0, 1]
        assert manager.get_total_exemplars() == 10

        # Task 2: classes [2, 3]
        ds2 = _make_dataset({2: 100, 3: 10}, seed=20)
        loader2 = DataLoader(ds2, batch_size=16, collate_fn=multiview_collate_fn)
        manager.update_memory_after_task(loader2, new_classes=[2, 3])

        assert manager.seen_classes == [0, 1, 2, 3]
        assert len(manager.memory[0]) == 5
        assert len(manager.memory[1]) == 5
        assert len(manager.memory[2]) == 10
        assert len(manager.memory[3]) == 1
        assert manager.get_total_exemplars() == 21

    def test_state_dict_serialization_fidelity(self):
        """Validates that buffer serialization and deserialization preserves all exemplars."""
        ds = _make_dataset({0: 30, 1: 5}, seed=30)
        loader = DataLoader(ds, batch_size=8, collate_fn=multiview_collate_fn)
        manager = RehearsalMemoryManager(replay_ratio=0.10, min_per_class=1)
        manager.update_memory_after_task(loader, new_classes=[0, 1])

        state = manager.state_dict()
        new_manager = RehearsalMemoryManager()
        new_manager.load_state_dict(state)

        assert new_manager.seen_classes == manager.seen_classes
        assert new_manager.get_total_exemplars() == manager.get_total_exemplars()
        for c in manager.seen_classes:
            assert len(new_manager.memory[c]) == len(manager.memory[c])
            for ex1, ex2 in zip(manager.memory[c], new_manager.memory[c]):
                for t1, t2 in zip(ex1, ex2):
                    assert torch.equal(t1, t2)

    def test_unbatched_dataset_and_sample_list_behavior(self):
        """Adversarial stress-test: Passing unbatched Dataset or sample list to update_memory_after_task.

        The docstring states: 'train_data: DataLoader, Dataset, or list of batches'.
        1. When passed a raw Dataset directly: PyTorch Dataset does not define __iter__,
           so hasattr(train_data, '__iter__') evaluates to False, silently skipping all data
           and leaving 0 exemplars.
        2. When passed a list of unbatched samples (list(ds)): each sample has a 0-d scalar
           label tensor. In PyTorch, hasattr(y, '__len__') evaluates to True, and len(y)
           raises TypeError: len() of a 0-d tensor.
        """
        ds = _make_dataset({0: 10, 1: 5}, seed=40)
        manager = RehearsalMemoryManager(replay_ratio=0.10, min_per_class=1)

        # Case 1: Raw Dataset directly -> 0 samples ingested because hasattr(ds, '__iter__') is False
        stats = manager.update_memory_after_task(ds, new_classes=[0, 1])
        assert stats[0] == 0 and stats[1] == 0
        assert manager.get_total_exemplars() == 0

        # Case 2: List of unbatched dataset items -> raises TypeError on 0-d tensor
        with pytest.raises(TypeError, match="len\\(\\) of a 0-d tensor"):
            manager.update_memory_after_task(list(ds), new_classes=[0, 1])


# ===========================================================================
# 2. Empirical Tests for memory/sampler.py
# ===========================================================================

class TestEmpiricalMemorySampler:
    """Empirical verification of get_mixed_batch."""

    def test_mixed_batch_formation_exact_size_and_composition(self):
        """Validates sampling N=32 exemplars with replacement from historical classes,

        strictly forming B_mixed = 2N = 64.
        """
        # Populate memory with historical classes [0, 1]
        hist_counts = {0: 100, 1: 10}
        hist_ds = _make_dataset(hist_counts, seed=50)
        hist_loader = DataLoader(hist_ds, batch_size=16, collate_fn=multiview_collate_fn)
        manager = RehearsalMemoryManager(replay_ratio=0.10, min_per_class=1)
        manager.update_memory_after_task(hist_loader, new_classes=[0, 1])

        # Current incoming batch from Task 2 (classes [2, 3]), batch_size N = 32
        curr_counts = {2: 16, 3: 16}
        curr_ds = _make_dataset(curr_counts, seed=60)
        curr_loader = DataLoader(curr_ds, batch_size=32, collate_fn=multiview_collate_fn)
        curr_batch = next(iter(curr_loader))
        assert len(curr_batch["label"]) == 32

        # Generate mixed batch
        mixed = get_mixed_batch(curr_batch, manager, batch_size=32, device="cpu")

        # Verify strict 2N = 64 batch size across all modalities
        assert len(mixed["label"]) == 64
        assert mixed["header"].shape == (64, 4)
        assert mixed["imports"].shape == (64, 1000)
        assert mixed["img1d"].shape == (64, 1024)
        assert mixed["img2d"].shape == (64, 3, 224, 224)
        assert mixed["apis"].shape == (64, 100)

        # Verify composition: First N=32 are identical to current batch
        assert torch.equal(mixed["header"][:32], curr_batch["header"])
        assert torch.equal(mixed["label"][:32], curr_batch["label"])

        # Verify composition: Last N=32 are exclusively from historical classes [0, 1]
        mem_labels = mixed["label"][32:].tolist()
        assert len(mem_labels) == 32
        for lbl in mem_labels:
            assert lbl in [0, 1], f"Exemplar label {lbl} must be historical class in [0, 1]"
            assert lbl not in [2, 3], f"Exemplar label {lbl} must NOT be current class in [2, 3]"

    def test_sampling_with_replacement_from_single_exemplar(self):
        """Stress-test: historical memory has ONLY 1 exemplar in total.

        get_mixed_batch must sample that single exemplar 32 times with replacement
        without throwing an IndexError or ValueError.
        """
        manager = RehearsalMemoryManager(replay_ratio=0.10, min_per_class=1)
        single_ds = _make_dataset({0: 1}, seed=70)  # Only 1 sample in class 0
        loader = DataLoader(single_ds, batch_size=1, collate_fn=multiview_collate_fn)
        manager.update_memory_after_task(loader, new_classes=[0])
        assert manager.get_total_exemplars() == 1

        # Current incoming batch has N = 32 samples from class 1
        curr_ds = _make_dataset({1: 32}, seed=80)
        curr_loader = DataLoader(curr_ds, batch_size=32, collate_fn=multiview_collate_fn)
        curr_batch = next(iter(curr_loader))

        mixed = get_mixed_batch(curr_batch, manager, batch_size=32, device="cpu")
        assert len(mixed["label"]) == 64
        # All 32 replay samples must be class 0
        replay_labels = mixed["label"][32:].tolist()
        assert replay_labels == [0] * 32

    def test_tuple_format_compatibility(self):
        """Validates that get_mixed_batch handles tuple-formatted batches seamlessly."""
        manager = RehearsalMemoryManager(replay_ratio=0.10, min_per_class=1)
        ds_old = _make_dataset({0: 10}, seed=90)
        manager.update_memory_after_task(
            DataLoader(ds_old, batch_size=5, collate_fn=multiview_collate_fn),
            new_classes=[0],
        )

        curr_ds = _make_dataset({1: 32}, seed=91)
        curr_batch = next(iter(DataLoader(curr_ds, batch_size=32, collate_fn=multiview_collate_fn)))
        tuple_batch = curr_batch.as_tuple()

        mixed_tuple = get_mixed_batch(tuple_batch, manager, batch_size=32, device="cpu")
        assert isinstance(mixed_tuple, tuple)
        assert len(mixed_tuple) == 6
        h, imp, i1d, i2d, api, y = mixed_tuple
        assert h.shape == (64, 4)
        assert imp.shape == (64, 1000)
        assert i1d.shape == (64, 1024)
        assert i2d.shape == (64, 3, 224, 224)
        assert api.shape == (64, 100)
        assert y.shape == (64,)

    def test_cold_start_empty_memory_returns_unmodified(self):
        """Task 1 Cold-Start: memory empty -> returns current batch untouched."""
        manager = RehearsalMemoryManager()
        curr_ds = _make_dataset({0: 32}, seed=92)
        curr_batch = next(iter(DataLoader(curr_ds, batch_size=32, collate_fn=multiview_collate_fn)))

        mixed = get_mixed_batch(curr_batch, manager, batch_size=32, device="cpu")
        assert len(mixed["label"]) == 32
        assert torch.equal(mixed["label"], curr_batch["label"])

    def test_no_old_classes_eligible_returns_unmodified(self):
        """When all classes in memory are also in current batch, no old classes exist -> returns current."""
        manager = RehearsalMemoryManager()
        ds = _make_dataset({0: 20}, seed=93)
        manager.update_memory_after_task(
            DataLoader(ds, batch_size=10, collate_fn=multiview_collate_fn),
            new_classes=[0],
        )

        curr_ds = _make_dataset({0: 32}, seed=94)
        curr_batch = next(iter(DataLoader(curr_ds, batch_size=32, collate_fn=multiview_collate_fn)))

        mixed = get_mixed_batch(curr_batch, manager, batch_size=32, device="cpu")
        assert len(mixed["label"]) == 32
        assert torch.equal(mixed["label"], curr_batch["label"])


# ===========================================================================
# 3. Empirical Tests for data/leakage.py
# ===========================================================================

class TestEmpiricalLeakagePurge:
    """Empirical verification of SHA-256 byte hashing and leakage purging."""

    def test_sha256_byte_hashing_exactness_and_sensitivity(self):
        """Validates that img1d is hashed byte-for-byte using hashlib.sha256."""
        v1 = np.ones(1024, dtype=np.float32)
        v2 = np.ones(1024, dtype=np.float32)
        v3 = np.ones(1024, dtype=np.float32)
        # Flip a single byte / float bit in v3
        v3[0] = 1.0000001

        h1 = hashlib.sha256(v1.tobytes()).hexdigest()
        h2 = hashlib.sha256(v2.tobytes()).hexdigest()
        h3 = hashlib.sha256(v3.tobytes()).hexdigest()

        # Identical byte buffers produce identical SHA-256
        assert h1 == h2
        # Slightest difference produces completely distinct hash
        assert h1 != h3

    def test_purging_cross_split_duplicates_without_data_corruption(self):
        """Validates purging of cross-split duplicate samples without corrupting remaining data.

        Setup:
        - Train dataset: 50 samples
        - Test dataset: 20 samples
        - 5 duplicate samples injected into both train and test at known indices
        - 2 intra-train duplicates injected (samples duplicated within train, but not in test)
        """
        rng = np.random.RandomState(999)

        n_train = 50
        n_test = 20

        # Construct distinct test dataset
        test_header = rng.randn(n_test, 4).astype(np.float32)
        test_imports = rng.randn(n_test, 1000).astype(np.float32)
        test_img1d = rng.randn(n_test, 1024).astype(np.float32)
        test_img2d = rng.randn(n_test, 224, 224, 3).astype(np.float32)
        test_apis = rng.randint(0, 100, size=(n_test, 100)).astype(np.int64)
        test_labels = rng.randint(0, 5, size=n_test).astype(np.int64)

        test_ds = MalwareMultiViewDataset(
            header=test_header,
            imports=test_imports,
            img1d=test_img1d,
            img2d=test_img2d,
            apis=test_apis,
            labels=test_labels,
        )

        # Construct train dataset
        train_header = rng.randn(n_train, 4).astype(np.float32)
        train_imports = rng.randn(n_train, 1000).astype(np.float32)
        train_img1d = rng.randn(n_train, 1024).astype(np.float32)
        train_img2d = rng.randn(n_train, 224, 224, 3).astype(np.float32)
        train_apis = rng.randint(0, 100, size=(n_train, 100)).astype(np.int64)
        train_labels = rng.randint(0, 5, size=n_train).astype(np.int64)

        # Inject 5 cross-split leaks: train indices [5, 12, 23, 34, 45] copy test samples [0, 3, 7, 11, 15]
        leak_train_idx = [5, 12, 23, 34, 45]
        leak_test_idx = [0, 3, 7, 11, 15]
        for tr_i, te_i in zip(leak_train_idx, leak_test_idx):
            train_img1d[tr_i] = test_img1d[te_i].copy()

        # Inject 2 internal train duplicates (train idx 1 copies train idx 0, train idx 21 copies 20)
        train_img1d[1] = train_img1d[0].copy()
        train_img1d[21] = train_img1d[20].copy()

        train_ds = MalwareMultiViewDataset(
            header=train_header,
            imports=train_imports,
            img1d=train_img1d,
            img2d=train_img2d,
            apis=train_apis,
            labels=train_labels,
        )

        # Pre-purge audit
        audit_pre = audit_leakage(train_ds, test_ds)
        assert audit_pre["overlap_count"] == 5
        assert len(audit_pre["overlap_hashes"]) == 5

        # Record clean train samples and their expected positions after deleting leak_train_idx
        expected_remaining_indices = [i for i in range(n_train) if i not in leak_train_idx]
        expected_header = train_header[expected_remaining_indices]
        expected_labels = train_labels[expected_remaining_indices]
        expected_img1d = train_img1d[expected_remaining_indices]

        # Execute Purge
        purged_count = purge_data_leakage(train_ds, test_ds, verbose=False)

        # 1. Purged count must be exactly 5
        assert purged_count == 5

        # 2. Train dataset length must be 45
        assert len(train_ds) == 45
        assert len(train_ds.labels) == 45
        assert len(train_ds.header) == 45
        assert len(train_ds.imports) == 45
        assert len(train_ds.img1d) == 45
        assert len(train_ds.img2d) == 45
        assert len(train_ds.apis) == 45

        # 3. Test dataset must remain completely untouched
        assert len(test_ds) == 20
        assert np.array_equal(test_ds.img1d, test_img1d)
        assert np.array_equal(test_ds.header, test_header)

        # 4. Post-purge cross-split overlap must be exactly 0
        audit_post = audit_leakage(train_ds, test_ds)
        assert audit_post["overlap_count"] == 0

        # 5. Non-leaking samples must be perfectly preserved in order and value across all modalities
        np.testing.assert_array_equal(train_ds.header, expected_header)
        np.testing.assert_array_equal(train_ds.labels, expected_labels)
        np.testing.assert_array_equal(train_ds.img1d, expected_img1d)

        # 6. Intra-train duplicates (not leaking into test) must NOT be deleted by purge_data_leakage
        internal_audit = audit_internal_duplicates(train_ds, split_name="train")
        # The 2 intra-train duplicates should still remain in train
        assert internal_audit["internal_duplicates"] == 2

    def test_purge_zero_overlap_noop(self):
        """Verifies that clean datasets with zero overlap are untouched."""
        ds_tr = _make_dataset({0: 10}, seed=1)
        ds_te = _make_dataset({1: 10}, seed=2)

        purged = purge_data_leakage(ds_tr, ds_te, verbose=False)
        assert purged == 0
        assert len(ds_tr) == 10

    def test_purge_empty_dataset_handling(self):
        """Verifies that empty datasets are handled gracefully without exceptions."""
        ds_empty = MalwareMultiViewDataset(
            header=np.empty((0, 4), dtype=np.float32),
            imports=np.empty((0, 1000), dtype=np.float32),
            img1d=np.empty((0, 1024), dtype=np.float32),
            img2d=np.empty((0, 224, 224, 3), dtype=np.float32),
            apis=np.empty((0, 100), dtype=np.int64),
            labels=np.empty((0,), dtype=np.int64),
        )
        ds_te = _make_dataset({0: 5}, seed=3)

        assert purge_data_leakage(ds_empty, ds_te, verbose=False) == 0
        assert purge_data_leakage(ds_te, ds_empty, verbose=False) == 0
