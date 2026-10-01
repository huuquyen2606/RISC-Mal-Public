"""Multi-view Windows PE Malware Dataset and DataLoader structures.

Loads and synchronizes the 6 modalities (header, imports, img1d, img2d, apis, label)
from disk .npy archives or in-memory arrays, with dict-based and tuple-based formats.
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiViewBatch(dict):
    """Specialized dictionary container for multi-view malware sample batches.

    Provides dictionary indexing (batch['header']), attribute access (batch.header),
    batch tuple unpacking (batch.as_tuple()), and recursive device transfers.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'MultiViewBatch' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def as_tuple(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the modalities in canonical order: (header, imports, img1d, img2d, apis, label)."""
        return (
            self["header"],
            self["imports"],
            self["img1d"],
            self["img2d"],
            self["apis"],
            self["label"],
        )

    def to(self, device: Union[torch.device, str], non_blocking: bool = False) -> "MultiViewBatch":
        """Transfers all member tensors to the designated compute device."""
        target_device = torch.device(device) if isinstance(device, str) else device
        transferred = MultiViewBatch()
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                transferred[k] = v.to(target_device, non_blocking=non_blocking)
            else:
                transferred[k] = v
        return transferred


class MalwareMultiViewDataset(Dataset):
    """PyTorch Dataset delivering 6 heterogeneous views per executable sample:

    1. header: float32 Tensor of shape [4]
    2. imports: float32 Tensor of shape [1000]
    3. img1d: float32 Tensor of shape [1024]
    4. img2d: float32 Tensor of shape [3, 224, 224] (transposed from [224, 224, 3])
    5. apis: int64 Tensor of shape [100]
    6. label: int64 scalar Tensor []
    """

    def __init__(
        self,
        task_id: str = "task1",
        set_name: str = "train",
        base_path: Optional[Union[str, Path]] = None,
        header: Optional[Union[np.ndarray, torch.Tensor]] = None,
        imports: Optional[Union[np.ndarray, torch.Tensor]] = None,
        img1d: Optional[Union[np.ndarray, torch.Tensor]] = None,
        img2d: Optional[Union[np.ndarray, torch.Tensor]] = None,
        apis: Optional[Union[np.ndarray, torch.Tensor]] = None,
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        return_dict: bool = True,
    ) -> None:
        """Initializes dataset from either disk archives or in-memory arrays.

        Args:
            task_id: Task name identifier (e.g. 'task1', 'task2', 'task3').
            set_name: Partition name ('train' or 'test').
            base_path: Root directory containing task subdirectories.
            header: In-memory array of shape [N, 4].
            imports: In-memory array of shape [N, 1000].
            img1d: In-memory array of shape [N, 1024].
            img2d: In-memory array of shape [N, 224, 224, 3] or [N, 3, 224, 224].
            apis: In-memory array of shape [N, 100].
            labels: In-memory array of shape [N].
            return_dict: If True, __getitem__ returns MultiViewBatch; if False, returns tuple.
        """
        self.task_id = task_id
        self.set_name = set_name
        self.return_dict = return_dict

        # Direct in-memory initialization path
        if labels is not None:
            self.header = self._to_numpy(header, np.float32)
            self.imports = self._to_numpy(imports, np.float32)
            self.img1d = self._to_numpy(img1d, np.float32)
            self.img2d = self._to_numpy(img2d, np.float32)
            self.apis = self._to_numpy(apis, np.int64)
            self.labels = self._to_numpy(labels, np.int64)
            return

        # Disk archive loading path
        if base_path is None:
            raise ValueError("Either in-memory arrays or base_path must be provided.")

        task_dir = Path(base_path) / task_id if task_id else Path(base_path)
        if not task_dir.is_dir():
            # Check if files reside directly in base_path
            task_dir = Path(base_path)

        prefix = f"{set_name}_{task_id}" if task_id else set_name

        def _try_load(modality_keys: Sequence[str]) -> np.ndarray:
            for key in modality_keys:
                candidate = task_dir / f"{prefix}_{key}.npy"
                if candidate.is_file():
                    return np.load(candidate)
                # Check without prefix
                candidate_noprefix = task_dir / f"{key}.npy"
                if candidate_noprefix.is_file():
                    return np.load(candidate_noprefix)
            raise FileNotFoundError(
                f"Could not locate .npy file for modalities {modality_keys} in {task_dir}"
            )

        self.header = _try_load(["header"]).astype(np.float32)
        self.imports = _try_load(["imports"]).astype(np.float32)
        self.img1d = _try_load(["img_1d", "img1d"]).astype(np.float32)
        self.img2d = _try_load(["img_2d", "img2d"]).astype(np.float32)
        self.apis = _try_load(["apis", "api"]).astype(np.int64)
        self.labels = _try_load(["labels", "label"]).astype(np.int64)

    @staticmethod
    def _to_numpy(arr: Any, dtype: np.dtype) -> np.ndarray:
        if arr is None:
            return np.empty(0, dtype=dtype)
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy().astype(dtype)
        return np.asarray(arr, dtype=dtype)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Union[MultiViewBatch, Tuple[torch.Tensor, ...]]:
        h_t = torch.from_numpy(self.header[idx]).float()
        imp_t = torch.from_numpy(self.imports[idx]).float()
        i1d_t = torch.from_numpy(self.img1d[idx]).float()

        # Handle img2d permutation from (224, 224, 3) to (3, 224, 224)
        raw_img2d = self.img2d[idx]
        if raw_img2d.ndim == 3 and raw_img2d.shape[-1] == 3:
            i2d_t = torch.from_numpy(raw_img2d).permute(2, 0, 1).float()
        else:
            i2d_t = torch.from_numpy(raw_img2d).float()

        api_t = torch.from_numpy(self.apis[idx]).long()
        label_val = self.labels[idx]
        label_t = torch.tensor(label_val, dtype=torch.long)

        if not self.return_dict:
            return h_t, imp_t, i1d_t, i2d_t, api_t, label_t

        return MultiViewBatch(
            header=h_t,
            imports=imp_t,
            img1d=i1d_t,
            img2d=i2d_t,
            apis=api_t,
            label=label_t,
        )


def multiview_collate_fn(batch: Sequence[Any]) -> Union[MultiViewBatch, Tuple[torch.Tensor, ...]]:
    """Custom collator handling both MultiViewBatch (dict) and tuple element lists."""
    if not batch:
        return MultiViewBatch()

    first = batch[0]
    if isinstance(first, dict):
        collated = MultiViewBatch()
        for k in first.keys():
            tensors = [b[k] for b in batch]
            collated[k] = torch.stack(tensors, dim=0)
        return collated
    elif isinstance(first, (list, tuple)):
        num_fields = len(first)
        stacked = []
        for i in range(num_fields):
            tensors = [b[i] for b in batch]
            stacked.append(torch.stack(tensors, dim=0))
        return tuple(stacked)
    else:
        return torch.utils.data.dataloader.default_collate(batch)
