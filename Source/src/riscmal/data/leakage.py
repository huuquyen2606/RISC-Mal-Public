"""Cryptographic SHA-256 cross-split leakage detection and duplicate purging.

Audits and eliminates train-to-test data leakage by hashing the 1024-byte
img1d layout representations and deleting overlapping training samples in-place.
"""

from typing import Any, Dict, List, Set, Union
import hashlib
import logging
import numpy as np

logger = logging.getLogger(__name__)


def purge_data_leakage(
    train_ds: Any,
    test_ds: Any,
    task_name: str = "Task",
    verbose: bool = True,
) -> int:
    """Detects and deletes training samples leaking into the test split via SHA-256 hashing.

    Operates in-place on the NumPy arrays of train_ds across all 6 modalities:
    header, imports, img1d, img2d, apis, and labels.

    Args:
        train_ds: Training MalwareMultiViewDataset instance.
        test_ds: Test MalwareMultiViewDataset instance.
        task_name: Human-readable task name for reporting.
        verbose: Whether to log warnings and summaries.

    Returns:
        The total number of leaking training samples purged.
    """
    if len(train_ds) == 0 or len(test_ds) == 0:
        return 0

    train_hashes: List[str] = [
        hashlib.sha256(train_ds.img1d[i].tobytes()).hexdigest()
        for i in range(len(train_ds))
    ]
    test_hashes: Set[str] = {
        hashlib.sha256(test_ds.img1d[i].tobytes()).hexdigest()
        for i in range(len(test_ds))
    }

    # Find 0-based indices of all overlapping training samples
    leak_indices: List[int] = [i for i, h in enumerate(train_hashes) if h in test_hashes]

    if leak_indices:
        if verbose:
            logger.warning(
                "⚠️ Leakage Audit: Detected %d overlapping samples in %s. Purging from training set...",
                len(leak_indices),
                task_name,
            )

        # In-place deletion along axis 0 across all 6 modalities
        train_ds.header = np.delete(train_ds.header, leak_indices, axis=0)
        train_ds.imports = np.delete(train_ds.imports, leak_indices, axis=0)
        train_ds.img1d = np.delete(train_ds.img1d, leak_indices, axis=0)
        train_ds.img2d = np.delete(train_ds.img2d, leak_indices, axis=0)
        train_ds.apis = np.delete(train_ds.apis, leak_indices, axis=0)
        train_ds.labels = np.delete(train_ds.labels, leak_indices, axis=0)

        if verbose:
            logger.info(
                "🧹 Purge Complete: %s training set reduced to %d clean samples.",
                task_name,
                len(train_ds.labels),
            )
    else:
        if verbose:
            logger.info("✅ Leakage Audit: %s is clean (0 overlaps detected).", task_name)

    return len(leak_indices)


def audit_leakage(train_ds: Any, test_ds: Any) -> Dict[str, Any]:
    """Audits cross-split duplicates without mutating datasets.

    Args:
        train_ds: Training dataset.
        test_ds: Test dataset.

    Returns:
        Dictionary containing overlap count, unique train IDs, unique test IDs, and leak hashes.
    """
    train_ids = [hashlib.sha256(train_ds.img1d[i].tobytes()).hexdigest() for i in range(len(train_ds))]
    test_ids = set(hashlib.sha256(test_ds.img1d[i].tobytes()).hexdigest() for i in range(len(test_ds)))

    overlap = [h for h in train_ids if h in test_ids]
    return {
        "train_count": len(train_ds),
        "test_count": len(test_ds),
        "train_unique_hashes": len(set(train_ids)),
        "test_unique_hashes": len(test_ids),
        "overlap_count": len(overlap),
        "overlap_hashes": list(set(overlap)),
    }


def audit_internal_duplicates(ds: Any, split_name: str = "train") -> Dict[str, Any]:
    """Inspects duplicate layout instances existing internally within a single partition.

    Args:
        ds: Dataset partition to audit.
        split_name: Name of partition ('train' or 'test').

    Returns:
        Dictionary reporting total samples, unique hash count, and intra-split duplicate count.
    """
    hashes = [hashlib.sha256(ds.img1d[i].tobytes()).hexdigest() for i in range(len(ds))]
    total = len(hashes)
    unique = len(set(hashes))
    duplicates = total - unique

    return {
        "split_name": split_name,
        "total_samples": total,
        "unique_samples": unique,
        "internal_duplicates": duplicates,
        "duplicate_ratio": float(duplicates / total) if total > 0 else 0.0,
    }


def purge_cross_split_duplicates(train_dir: Any, test_dir: Any, **kwargs: Any) -> Any:
    """Convenience wrapper or alias for cross-split duplicate purging."""
    if hasattr(train_dir, "img1d") and hasattr(test_dir, "img1d"):
        count = purge_data_leakage(train_dir, test_dir, **kwargs)
        return train_dir, test_dir, list(range(count))
    return [], [], []

