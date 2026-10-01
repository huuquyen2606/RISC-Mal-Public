"""Mixed batch sampler for rehearsal-based continual learning.

Constructs balanced mixed batches B_{mixed} = B_{new} U B_{replay} of size 2N = 64
by drawing exemplars from historical classes with replacement.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import random
import torch

from riscmal.data.dataset import MultiViewBatch
from riscmal.memory.buffer import RehearsalMemoryManager


def get_mixed_batch(
    current_batch: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...], List[torch.Tensor]],
    memory_manager: RehearsalMemoryManager,
    batch_size: Optional[int] = None,
    device: Union[torch.device, str] = "cpu",
) -> Union[MultiViewBatch, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
    """Combines an incoming streaming batch with historical exemplars to form a mixed batch.

    Formulation:
        B_{mixed} = [x_{new} || x_{replay}] in R^{(N + N_mem) x ...}
        With N = 32 and N_mem = 32, yields B_{mixed} = 2N = 64.

    Args:
        current_batch: Incoming batch, either as a dictionary / MultiViewBatch
            or as a 6-tuple (header, imports, img1d, img2d, apis, label).
        memory_manager: RehearsalMemoryManager holding historical class exemplars.
        batch_size: Number of exemplars to sample. If None, matches current batch size N.
        device: Target compute device for output tensors.

    Returns:
        Mixed batch containing both current and rehearsal samples formatted identically
        to current_batch (dict or tuple), transferred to the target device.
    """
    target_device = torch.device(device) if isinstance(device, str) else device
    is_dict = isinstance(current_batch, dict)

    if is_dict:
        current_y = current_batch["label"]
        current_n = current_y.shape[0] if current_y.ndim > 0 else 1
    else:
        current_y = current_batch[5] if len(current_batch) > 5 else current_batch[-1]
        current_n = current_y.shape[0] if current_y.ndim > 0 else 1

    memory_batch_size = batch_size if batch_size is not None else current_n

    # Identify seen classes that are NOT part of the current incoming batch
    current_classes = torch.unique(current_y).cpu().tolist()
    old_classes = [c for c in memory_manager.seen_classes if c not in current_classes]

    # If memory is unpopulated or no old classes are eligible, return current batch
    if not memory_manager.memory or not old_classes:
        if is_dict:
            res_dict = MultiViewBatch()
            for k, v in current_batch.items():
                res_dict[k] = v.to(target_device) if isinstance(v, torch.Tensor) else v
            return res_dict
        else:
            return tuple(
                item.to(target_device) if isinstance(item, torch.Tensor) else item
                for item in current_batch
            )

    # Aggregate candidate exemplars across all eligible old classes
    old_mem_samples: List[Tuple[torch.Tensor, ...]] = []
    for cls in old_classes:
        if cls in memory_manager.memory:
            old_mem_samples.extend(memory_manager.memory[cls])

    if not old_mem_samples:
        if is_dict:
            res_dict = MultiViewBatch()
            for k, v in current_batch.items():
                res_dict[k] = v.to(target_device) if isinstance(v, torch.Tensor) else v
            return res_dict
        else:
            return tuple(
                item.to(target_device) if isinstance(item, torch.Tensor) else item
                for item in current_batch
            )

    # Random sampling with replacement matching memory_batch_size
    sampled_mem = random.choices(old_mem_samples, k=memory_batch_size)

    # Modality order: 0: header, 1: imports, 2: img1d, 3: img2d, 4: apis, 5: label
    if is_dict:
        keys = ["header", "imports", "img1d", "img2d", "apis", "label"]
        mixed_dict = MultiViewBatch()
        for idx, key in enumerate(keys):
            if key in current_batch:
                curr_t = current_batch[key].to(target_device)
                mem_t = torch.stack([s[idx] for s in sampled_mem]).to(target_device)
                mixed_dict[key] = torch.cat([curr_t, mem_t], dim=0)
            else:
                # Fallback if key named differently
                mixed_dict[key] = current_batch.get(key)
        return mixed_dict
    else:
        mixed_list = []
        for idx in range(len(current_batch)):
            curr_t = current_batch[idx].to(target_device)
            mem_t = torch.stack([s[idx] for s in sampled_mem]).to(target_device)
            mixed_list.append(torch.cat([curr_t, mem_t], dim=0))
        return tuple(mixed_list)
