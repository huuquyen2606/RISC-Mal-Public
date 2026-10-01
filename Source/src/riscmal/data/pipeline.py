"""Task-incremental dataset partitioning and global asset generation pipeline.

Orchestrates Phase 1 (global vocabulary mining and StandardScaler fitting)
and Phase 2 (task-incremental feature extraction and sanitized DataLoader generation).
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import logging
import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from riscmal.data.dataset import MalwareMultiViewDataset, multiview_collate_fn
from riscmal.data.leakage import purge_data_leakage
from riscmal.data.pe_features import (
    clean_numeric,
    extract_apis_from_json,
    extract_byte_images,
    extract_header_from_json,
    extract_header_from_pe,
    extract_imports_from_json,
    extract_imports_from_pe,
)

logger = logging.getLogger(__name__)

DEFAULT_LABELS = ["Benign", "Locker", "Mediyes", "Winwebsec", "Zbot", "Zeroaccess"]
DEFAULT_TASKS = {
    "task1": ["Benign", "Locker"],
    "task2": ["Mediyes", "Winwebsec"],
    "task3": ["Zbot", "Zeroaccess"],
}


class GlobalAssetBuilder:
    """Mines global vocabulary assets (Imports and APIs) and fits standard scaler on headers."""

    def __init__(
        self,
        base_json_dir: Union[str, Path],
        output_dir: Union[str, Path],
        labels: Sequence[str] = DEFAULT_LABELS,
        top_k_imports: int = 1000,
        top_k_apis: int = 100,
    ) -> None:
        self.base_json_dir = Path(base_json_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.labels = list(labels)
        self.top_k_imports = top_k_imports
        self.top_k_apis = top_k_apis

        self.imp_map_path = self.output_dir / "global_imp_map.pkl"
        self.api_map_path = self.output_dir / "global_api_map.pkl"
        self.scaler_path = self.output_dir / "global_scaler.pkl"

    def build_assets(self) -> Tuple[Dict[str, int], Dict[str, int], StandardScaler]:
        """Scans all training JSON reports across all classes to construct global assets.

        Returns:
            Tuple of (imp_map, api_map, scaler).
        """
        logger.info("Building Global Assets across all %d classes...", len(self.labels))
        imp_counter: Counter = Counter()
        api_counter: Counter = Counter()
        all_headers: List[List[float]] = []

        for label in self.labels:
            train_json_dir = self.base_json_dir / "train" / label
            if not train_json_dir.is_dir():
                # Try direct label folder
                train_json_dir = self.base_json_dir / label
                if not train_json_dir.is_dir():
                    logger.debug("Directory not found for label %s: %s", label, train_json_dir)
                    continue

            for json_file in train_json_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # 1. Imports frequency
                    for entry in data.get("static", {}).get("pe_imports", []):
                        for imp in entry.get("imports", []):
                            name = imp.get("name")
                            if name:
                                imp_counter[name] += 1

                    # 2. API calls frequency
                    for proc in data.get("behavior", {}).get("processes", []):
                        for call in proc.get("calls", []):
                            api = call.get("api")
                            if api:
                                api_counter[api] += 1

                    # 3. Header features for Scaler
                    header_vec = extract_header_from_json(data)
                    all_headers.append(header_vec)
                except Exception as exc:
                    logger.debug("Skipped corrupt JSON %s: %s", json_file, exc)
                    continue

        # Sort top K elements alphabetically for determinism
        top_imps = sorted([n for n, _ in imp_counter.most_common(self.top_k_imports)])
        top_apis = sorted([n for n, _ in api_counter.most_common(self.top_k_apis)])

        # 0-indexed for imports, 1-indexed for APIs (0 is reserved as PAD_TOKEN)
        imp_map = {n: i for i, n in enumerate(top_imps)}
        api_map = {n: i + 1 for i, n in enumerate(top_apis)}

        with open(self.imp_map_path, "wb") as f:
            pickle.dump(imp_map, f)
        with open(self.api_map_path, "wb") as f:
            pickle.dump(api_map, f)

        # Fit StandardScaler across the aggregated training distribution
        scaler = StandardScaler()
        if all_headers:
            scaler.fit(np.array(all_headers, dtype=np.float32))
        else:
            logger.warning("No header vectors found during scan; fitting dummy identity scaler.")
            scaler.fit(np.zeros((1, 4), dtype=np.float32))

        with open(self.scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        logger.info(
            "Global Assets serialized successfully (Imports: %d, APIs: %d, Scaler fit on %d samples).",
            len(imp_map),
            len(api_map),
            len(all_headers),
        )
        return imp_map, api_map, scaler


class TaskDataExtractor:
    """Extracts and normalizes multi-view features for task-partitioned incremental sets."""

    def __init__(
        self,
        base_json_dir: Union[str, Path],
        base_exe_dir: Union[str, Path],
        output_root: Union[str, Path],
        global_assets_dir: Union[str, Path],
        all_labels: Sequence[str] = DEFAULT_LABELS,
    ) -> None:
        self.base_json_dir = Path(base_json_dir)
        self.base_exe_dir = Path(base_exe_dir)
        self.output_root = Path(output_root)
        self.all_labels = list(all_labels)

        # Load global assets
        g_dir = Path(global_assets_dir)
        with open(g_dir / "global_imp_map.pkl", "rb") as f:
            self.imp_map: Dict[str, int] = pickle.load(f)
        with open(g_dir / "global_api_map.pkl", "rb") as f:
            self.api_map: Dict[str, int] = pickle.load(f)
        with open(g_dir / "global_scaler.pkl", "rb") as f:
            self.scaler: StandardScaler = pickle.load(f)

    def extract_task_split(
        self,
        set_name: str,
        task_name: str,
        target_labels: Sequence[str],
    ) -> Path:
        """Extracts and stores .npy archives for a specific task and split.

        Args:
            set_name: Partition name ('train' or 'test').
            task_name: Task folder name ('task1', 'task2', etc.).
            target_labels: List of class label strings assigned to this task.

        Returns:
            Path to the output task directory containing .npy files.
        """
        task_dir = self.output_root / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        storage: Dict[str, List[Any]] = {
            "header": [],
            "imports": [],
            "img_1d": [],
            "img_2d": [],
            "apis": [],
            "labels": [],
        }

        for label in target_labels:
            if label not in self.all_labels:
                raise ValueError(f"Label '{label}' is not present in global labels {self.all_labels}")
            class_idx = self.all_labels.index(label)

            json_dir = self.base_json_dir / set_name / label
            exe_dir = self.base_exe_dir / set_name / label

            if not json_dir.is_dir():
                continue

            for json_file in json_dir.glob("*.json"):
                # Corresponding binary path
                exe_name = json_file.stem
                exe_path = exe_dir / exe_name
                if not exe_path.is_file():
                    exe_path = exe_dir / f"{exe_name}.exe"
                    if not exe_path.is_file():
                        continue

                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        report = json.load(f)

                    # 1. Header
                    raw_h = extract_header_from_json(report)
                    storage["header"].append(raw_h)

                    # 2. Imports
                    iv = extract_imports_from_json(report, self.imp_map)
                    storage["imports"].append(iv)

                    # 3. APIs
                    aseq = extract_apis_from_json(report, self.api_map)
                    storage["apis"].append(aseq)

                    # 4. Images (1D and 2D)
                    i1, i2 = extract_byte_images(exe_path)
                    storage["img_1d"].append(i1)
                    storage["img_2d"].append(i2)

                    # 5. Label
                    storage["labels"].append(class_idx)
                except Exception as exc:
                    logger.debug("Extraction failed for %s: %s", json_file, exc)
                    continue

        # Standardize header using global scaler
        if storage["header"]:
            header_final = self.scaler.transform(np.array(storage["header"], dtype=np.float32))
        else:
            header_final = np.empty((0, 4), dtype=np.float32)

        prefix = f"{set_name}_{task_name}"
        for key in storage.keys():
            data = header_final if key == "header" else np.array(storage[key])
            np.save(task_dir / f"{prefix}_{key}.npy", data)

        logger.info(
            "Extracted %d samples for %s/%s into %s",
            len(storage["labels"]),
            task_name,
            set_name,
            task_dir,
        )
        return task_dir


def build_incremental_dataloaders(
    data_dir: Union[str, Path],
    tasks: Optional[Dict[str, List[int]]] = None,
    batch_size: int = 32,
    purge_leakage: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> List[Dict[str, DataLoader]]:
    """Builds clean, sanitized training and evaluation DataLoaders for all incremental tasks.

    Args:
        data_dir: Directory containing task1, task2, task3 subdirectories with .npy archives.
        tasks: Optional task configuration dict. Default: task1=[0,1], task2=[2,3], task3=[4,5].
        batch_size: Mini-batch size N (default: 32).
        purge_leakage: Whether to apply cryptographic SHA-256 data-leakage purging.
        num_workers: Number of dataloader worker processes.
        pin_memory: Whether to pin memory for faster GPU transfers.

    Returns:
        List of dictionaries with keys 'train_loader' and 'test_loader' per task.
    """
    task_config = tasks or {
        "task1": [0, 1],
        "task2": [2, 3],
        "task3": [4, 5],
    }

    tasks_data: List[Dict[str, DataLoader]] = []
    base_path = Path(data_dir)
    if not base_path.is_dir():
        for candidate in [
            Path("incremental_data_v2"),
            Path("../incremental_data_v2"),
            Path("../../incremental_data_v2"),
            Path("data"),
            Path("../data"),
            Path("../Data"),
        ]:
            if candidate.is_dir():
                base_path = candidate
                break

    for task_id, class_ids in task_config.items():
        train_ds = MalwareMultiViewDataset(task_id=task_id, set_name="train", base_path=base_path)
        test_ds = MalwareMultiViewDataset(task_id=task_id, set_name="test", base_path=base_path)

        if purge_leakage:
            purge_data_leakage(train_ds, test_ds, task_name=task_id.capitalize())

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True if len(train_ds) > batch_size else False,
            collate_fn=multiview_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=multiview_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        tasks_data.append({"train_loader": train_loader, "test_loader": test_loader})

    return tasks_data
