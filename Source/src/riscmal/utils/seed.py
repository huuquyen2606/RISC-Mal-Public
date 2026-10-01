"""Deterministic seed anchoring for reproducible continual learning experiments.

Anchors random number generators across Python stdlib random, NumPy, PyTorch CPU,
PyTorch CUDA, cuDNN backend flags, and cuBLAS workspace configurations.
"""

import os
import random
import numpy as np
import torch


def set_deterministic_seed(seed: int = 42, warn_only_deterministic: bool = True) -> int:
    """Sets random seeds across all libraries for deterministic execution.

    Args:
        seed: Integer seed value (default: 42).
        warn_only_deterministic: If True, warns rather than raises when PyTorch
            encounters operations without deterministic implementations.

    Returns:
        The integer seed applied.
    """
    # 1. Python environment variables for hashing and cuBLAS workspace
    os.environ["PYTHONHASHSEED"] = str(seed)
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        # Required for deterministic matrix operations in CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 2. Python stdlib and NumPy
    random.seed(seed)
    np.random.seed(seed)

    # 3. PyTorch CPU and GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 4. cuDNN backend determinism
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 5. PyTorch deterministic algorithms check
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only_deterministic)
    except Exception:
        # In environments or older PyTorch versions where this is unsupported
        pass

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init function ensuring unique, deterministic seeds per worker.

    Args:
        worker_id: The ID of the dataloader worker process.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
