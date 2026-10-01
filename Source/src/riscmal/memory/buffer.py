"""Class-proportional rehearsal memory buffer management.

Maintains an exemplar reservoir of historical malware samples across incremental tasks
with guaranteed minimum coverage (rho=0.10, min_per_class=1).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import logging
import random
import torch

logger = logging.getLogger(__name__)


class RehearsalMemoryManager:
    """Class-proportional rehearsal buffer storing historical exemplars across task transitions.

    Retention Policy:
        k_c = max(min_per_class, floor(|D_c| * replay_ratio))
        Guarantees at least min_per_class exemplar even for severely minority families.
    """

    def __init__(
        self,
        replay_ratio: float = 0.10,
        min_per_class: int = 1,
    ) -> None:
        """Initializes rehearsal memory manager.

        Args:
            replay_ratio: Proportion of task training samples retained per class (default: 0.10).
            min_per_class: Minimum number of exemplars stored per seen class (default: 1).
        """
        if not (0.0 < replay_ratio <= 1.0):
            raise ValueError(f"replay_ratio must be in (0.0, 1.0], got {replay_ratio}")
        if min_per_class < 1:
            raise ValueError(f"min_per_class must be >= 1, got {min_per_class}")

        self.replay_ratio = float(replay_ratio)
        self.min_per_class = int(min_per_class)
        # Memory storage: class_id -> list of 6-tuples (h, imp, i1d, i2d, api, y) on CPU
        self.memory: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        self.seen_classes: List[int] = []

    def update_memory_after_task(
        self,
        train_data: Any,
        new_classes: Sequence[int],
    ) -> Dict[int, int]:
        """Collects and stores exemplars for newly encountered classes.

        Args:
            train_data: DataLoader, Dataset, or list of batches containing training samples.
            new_classes: Sequence of new class integer IDs introduced in the completed task.

        Returns:
            Dictionary mapping class ID to number of stored exemplars.
        """
        new_classes_list = [int(c) for c in new_classes]
        for c in new_classes_list:
            if c not in self.seen_classes:
                self.seen_classes.append(c)

        new_class_data: Dict[int, List[Tuple[torch.Tensor, ...]]] = {
            cls: [] for cls in new_classes_list
        }

        # Ingestion loop: extract all training samples per new class
        with torch.no_grad():
            if hasattr(train_data, "__iter__"):
                for batch in train_data:
                    if isinstance(batch, dict):
                        h = batch["header"]
                        imp = batch["imports"]
                        i1d = batch["img1d"]
                        i2d = batch["img2d"]
                        api = batch["apis"]
                        y = batch["label"]
                    elif isinstance(batch, (list, tuple)):
                        h, imp, i1d, i2d, api, y = batch
                    else:
                        continue

                    # Process each sample in batch
                    batch_len = len(y) if hasattr(y, "__len__") else 1
                    for i in range(batch_len):
                        label = int(y[i].item() if isinstance(y[i], torch.Tensor) else y[i])
                        if label in new_class_data:
                            sample_tuple = (
                                h[i].detach().cpu(),
                                imp[i].detach().cpu(),
                                i1d[i].detach().cpu(),
                                i2d[i].detach().cpu(),
                                api[i].detach().cpu(),
                                (y[i].detach().cpu() if isinstance(y[i], torch.Tensor) else torch.tensor(y[i], dtype=torch.long)),
                            )
                            new_class_data[label].append(sample_tuple)

        # Class-proportional selection: max(min_per_class, floor(N_c * rho))
        stats_per_class: Dict[int, int] = {}
        for cls in new_classes_list:
            available = new_class_data[cls]
            total_available = len(available)
            if total_available == 0:
                logger.warning("No samples found for new class %d during memory update.", cls)
                self.memory[cls] = []
                stats_per_class[cls] = 0
                continue

            keep_count = max(self.min_per_class, int(total_available * self.replay_ratio))
            # Ensure keep_count does not exceed total available
            keep_count = min(keep_count, total_available)

            # Uniform random sampling without replacement
            self.memory[cls] = random.sample(available, keep_count)
            stats_per_class[cls] = len(self.memory[cls])

        stat_str = ", ".join([f"Class {c}: {len(self.memory.get(c, []))}" for c in self.seen_classes])
        logger.info(
            "📦 [Memory] Replay Ratio: %.1f%% | Exemplars: %s",
            self.replay_ratio * 100.0,
            stat_str,
        )
        return stats_per_class

    def get_total_exemplars(self) -> int:
        """Returns total count of stored exemplars across all seen classes."""
        return sum(len(exemplars) for exemplars in self.memory.values())

    def get_memory_stats(self) -> Dict[str, Any]:
        """Returns diagnostic statistics for the rehearsal buffer."""
        return {
            "seen_classes": list(self.seen_classes),
            "per_class_counts": {c: len(self.memory.get(c, [])) for c in self.seen_classes},
            "total_exemplars": self.get_total_exemplars(),
            "replay_ratio": self.replay_ratio,
            "min_per_class": self.min_per_class,
        }

    def clear(self) -> None:
        """Clears all stored exemplars and seen classes."""
        self.memory.clear()
        self.seen_classes.clear()

    def state_dict(self) -> Dict[str, Any]:
        """Serializes memory state for checkpointing."""
        return {
            "replay_ratio": self.replay_ratio,
            "min_per_class": self.min_per_class,
            "seen_classes": list(self.seen_classes),
            "memory": self.memory,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restores memory state from checkpoint payload."""
        self.replay_ratio = float(state.get("replay_ratio", self.replay_ratio))
        self.min_per_class = int(state.get("min_per_class", self.min_per_class))
        self.seen_classes = list(state.get("seen_classes", []))
        self.memory = state.get("memory", {})
