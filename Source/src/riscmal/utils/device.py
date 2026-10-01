"""Device placement and tensor dispatch utilities.

Handles automatic detection and graceful fallback between CUDA, Apple Silicon MPS,
and CPU runtimes, along with recursive data structure device transfer.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch

logger = logging.getLogger(__name__)


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Resolves and returns the compute device based on preferences and availability.

    Args:
        preferred: Optional preferred device string ("cuda", "cuda:0", "mps", "cpu").
            If None, prioritizes CUDA -> MPS -> CPU.

    Returns:
        torch.device object representing the selected compute runtime.
    """
    if isinstance(preferred, torch.device):
        return preferred
    if preferred is not None and str(preferred).strip().lower() != "auto":
        preferred_lower = str(preferred).strip().lower()


        if preferred_lower.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(preferred_lower)
            logger.warning("CUDA requested but not available. Checking fallbacks...")
        elif preferred_lower == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning("Apple MPS requested but not available. Falling back to CPU.")
        elif preferred_lower == "cpu":
            return torch.device("cpu")
        else:
            try:
                return torch.device(preferred)
            except Exception as e:
                logger.warning("Unrecognized device %s: %s. Falling back to auto-detect.", preferred, e)

    # Auto-detection hierarchy: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(data: Any, device: Union[torch.device, str], non_blocking: bool = False) -> Any:
    """Recursively moves tensors within collections (dicts, lists, tuples) to target device.

    Args:
        data: Tensor or collection containing tensors.
        device: Target torch.device or device string.
        non_blocking: Whether to use non-blocking asynchronous copy when applicable.

    Returns:
        The structure with all embedded tensors transferred to the target device.
    """
    target_device = torch.device(device) if isinstance(device, str) else device

    if isinstance(data, torch.Tensor):
        return data.to(target_device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {key: to_device(value, target_device, non_blocking=non_blocking) for key, value in data.items()}
    elif isinstance(data, tuple):
        # Handle namedtuples by checking for _fields attribute
        if hasattr(data, "_fields"):
            cls = type(data)
            return cls(*(to_device(item, target_device, non_blocking=non_blocking) for item in data))
        return tuple(to_device(item, target_device, non_blocking=non_blocking) for item in data)
    elif isinstance(data, list):
        return [to_device(item, target_device, non_blocking=non_blocking) for item in data]
    return data


def get_peak_memory_mb(device: Optional[Union[torch.device, str]] = None) -> float:
    """Returns the peak GPU memory allocated in Megabytes (MB).

    Args:
        device: Target device. If None, checks current CUDA device.

    Returns:
        Peak memory in MB as a float, or 0.0 if not running on CUDA.
    """
    if not torch.cuda.is_available():
        return 0.0
    try:
        current_dev = torch.device(device) if device is not None else torch.cuda.current_device()
        peak_bytes = torch.cuda.max_memory_allocated(current_dev)
        return float(peak_bytes) / (1024.0 * 1024.0)
    except Exception:
        return 0.0
