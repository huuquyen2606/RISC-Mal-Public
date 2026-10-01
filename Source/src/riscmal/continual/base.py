"""Base class and unified interface for Continual Learning strategies."""

from abc import ABC, abstractmethod
import copy
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from riscmal.evaluation.metrics import compute_classification_metrics
from riscmal.memory.buffer import RehearsalMemoryManager
from riscmal.memory.sampler import get_mixed_batch
from riscmal.utils.device import get_device

logger = logging.getLogger(__name__)


class BaseContinualStrategy(ABC):
    """Abstract base class for all Continual Learning (Class-IL) strategies.

    Provides standard lifecycle interfaces:
        - `before_task`: Invoked prior to learning a new task (e.g. head expansion, freezing).
        - `train_task`: Trains on current task dataset over multiple epochs, returning training history.
        - `train_one_epoch`: Single epoch execution across streaming mini-batches.
        - `compute_additional_loss`: Auxiliary strategy-specific regularizers (KD, SP, EWC penalty).
        - `after_task`: Executed once task training concludes (Fisher update, exemplar storage).
        - `evaluate_task`: Evaluates current model on validation or cumulative test dataloader.
        - `save_checkpoint` / `load_checkpoint`: Full state persistence.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: Union[torch.device, str] = "cpu",
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        num_known_classes: int = 0,
        replay_ratio: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initializes base continual learning strategy.

        Args:
            model: PyTorch neural network to be incrementally trained.
            device: Target execution device.
            config: Optional configuration dictionary containing method hyperparameters.
            name: Strategy identifier name.
            num_known_classes: Initial count of previously learned classes.
            replay_ratio: Ratio of exemplar rehearsal memory retention.
            **kwargs: Extra strategy-specific arguments.
        """
        # Backward-compatible handling if model is passed as string identifier
        if isinstance(model, str):
            if name is None:
                name = model
            model = kwargs.pop("model", None)

        self.name = name or kwargs.get("name", getattr(self, "name", "BaseStrategy"))
        self.device = get_device(device)
        self.config: Dict[str, Any] = dict(config or {})

        # Merge remaining kwargs into config
        for k, v in kwargs.items():
            if k not in self.config:
                self.config[k] = v

        self._model = None
        if model is not None:
            self.model = model

        self._known_classes = int(num_known_classes)
        self.replay_ratio = float(replay_ratio)
        self.memory_manager: Optional[RehearsalMemoryManager] = kwargs.get("memory_manager", None)
        self.class_weights: Optional[torch.Tensor] = kwargs.get("class_weights", None)

        self.current_task_id: int = 1
        self.seen_classes: List[int] = []
        self.current_task_classes: List[int] = []
        self.last_train_loader: Optional[DataLoader] = None

    @property
    def model(self) -> Optional[nn.Module]:
        """Active PyTorch neural network model managed by the continual strategy."""
        return self._model

    @model.setter
    def model(self, val: Optional[nn.Module]) -> None:
        """Sets active neural network model and places it on target device."""
        if val is not None:
            self._model = val.to(self.device)
        else:
            self._model = None

    @property
    def known_classes(self) -> int:
        """Total number of classes previously trained and currently retained."""
        return self._known_classes

    @known_classes.setter
    def known_classes(self, val: int) -> None:
        """Updates total number of retained classes."""
        self._known_classes = int(val)

    def adapt_architecture_before_task(self, device: Union[torch.device, str]) -> None:
        """Adapts model architecture before starting training on a new task.

        Default behavior: expands classification head if novel classes exceed capacity.
        Overridden by dynamic architectures (DER, FOSTER) and generative models (GC, BIR, LGR).
        """
        if self.model is None:
            return

        target_classes = len(self.current_task_classes) if self.current_task_classes else 2
        if hasattr(self.model, "expand_classes"):
            current_classes = getattr(self.model, "num_classes", 0)
            if hasattr(self.model, "classifier_head") and hasattr(self.model.classifier_head, "out_features"):
                current_classes = self.model.classifier_head.out_features

            if current_classes < self._known_classes + target_classes:
                num_new = (self._known_classes + target_classes) - current_classes
                self.model.expand_classes(num_new, device=device)

    def before_task(
        self,
        task_id: int,
        task_classes: Sequence[int],
        train_dataset: Optional[Any] = None,
        memory_manager: Optional[RehearsalMemoryManager] = None,
    ) -> None:
        """Lifecycle hook invoked before task training begins.

        Args:
            task_id: Current incremental task identifier (1-indexed).
            task_classes: Class IDs introduced in the current task.
            train_dataset: Optional training dataset for current task.
            memory_manager: Optional rehearsal memory manager.
        """
        self.current_task_id = int(task_id)
        self.current_task_classes = [int(c) for c in task_classes]

        if memory_manager is not None:
            self.memory_manager = memory_manager

        for c in self.current_task_classes:
            if c not in self.seen_classes:
                self.seen_classes.append(c)

        if task_id > 1:
            self.adapt_architecture_before_task(self.device)

    def compute_additional_loss(
        self,
        model: nn.Module,
        features_64: torch.Tensor,
        outputs: torch.Tensor,
        batch: Any,
        current_task_id: int = 1,
    ) -> torch.Tensor:
        """Computes strategy-specific auxiliary loss (e.g. EWC penalty, KD loss, replay loss).

        Returns scalar zero tensor by default (fine-tuning baseline).
        """
        return torch.tensor(0.0, device=features_64.device, requires_grad=True)

    def _unpack_batch(
        self,
        batch: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpacks batch into (header, imports, img1d, img2d, apis, label) on target device."""
        if isinstance(batch, dict):
            h = batch["header"].to(self.device)
            imp = batch["imports"].to(self.device)
            i1d = batch["img1d"].to(self.device)
            i2d = batch["img2d"].to(self.device)
            api = batch["apis"].to(self.device)
            y = batch["label"].to(self.device)
        elif isinstance(batch, (list, tuple)):
            items = [item.to(self.device) for item in batch]
            h, imp, i1d, i2d, api, y = items[0], items[1], items[2], items[3], items[4], items[5]
        else:
            raise TypeError(f"Unsupported batch format: {type(batch).__name__}")
        return h, imp, i1d, i2d, api, y

    def train_one_epoch(
        self,
        train_loader: Any,
        optimizer: torch.optim.Optimizer,
        task_id: int,
    ) -> Tuple[float, float]:
        """Runs a single epoch of training on the current task streaming dataloader.

        Args:
            train_loader: DataLoader yielding streaming task batches.
            optimizer: Optimizer updating active parameters.
            task_id: Current incremental task identifier.

        Returns:
            Tuple of (epoch_loss, epoch_accuracy).
        """
        if self.model is None:
            raise RuntimeError("Model must be assigned before training.")

        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for raw_batch in train_loader:
            optimizer.zero_grad()

            # Optional rehearsal memory augmentation
            batch = raw_batch
            if self.memory_manager is not None and self.memory_manager.get_total_exemplars() > 0:
                batch = get_mixed_batch(raw_batch, self.memory_manager, device=self.device)

            h, imp, i1d, i2d, api, y = self._unpack_batch(batch)

            # Special routing for dynamic multi-column backbones (e.g. DER / FOSTER)
            if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "backbones"):
                if len(self.model.backbone.backbones) > 1:
                    for bb in self.model.backbone.backbones[:-1]:
                        bb.eval()
                        for p in bb.parameters():
                            p.requires_grad = False
                    self.model.backbone.backbones[-1].train(True)
                    for p in self.model.backbone.backbones[-1].parameters():
                        p.requires_grad = True

            # Feature extraction
            if hasattr(self.model, "backbone"):
                features_64 = self.model.backbone(h, imp, i1d, i2d, api)
            elif hasattr(self.model, "feature_extractor"):
                features_64 = self.model.feature_extractor(h, imp, i1d, i2d, api)
            else:
                out = self.model(h, imp, i1d, i2d, api)
                features_64 = out[1] if isinstance(out, tuple) else out

            # Logits calculation
            if getattr(self, "name", "").startswith("DER"):
                outputs, _ = self.model(h, imp, i1d, i2d, api)
            elif hasattr(self.model, "classifier_head"):
                outputs = self.model.classifier_head(features_64)
            else:
                out = self.model(h, imp, i1d, i2d, api)
                outputs = out[0] if isinstance(out, tuple) else out

            # Primary classification loss
            if self.class_weights is not None:
                weights = self.class_weights.to(self.device)
                loss = F.cross_entropy(outputs, y.long(), weight=weights)
            else:
                loss = F.cross_entropy(outputs, y.long())

            # Auxiliary continual regularization loss
            extra_loss = self.compute_additional_loss(self.model, features_64, outputs, batch, task_id)
            loss = loss + extra_loss

            loss.backward()
            optimizer.step()

            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            _, predicted = outputs.max(dim=1)
            total_correct += int(predicted.eq(y.long()).sum().item())
            total_samples += bs

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)
        return avg_loss, avg_acc

    def train_task(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        epochs: int = 10,
        lr: float = 1e-4,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs: Any,
    ) -> Dict[str, List[float]]:
        """Trains the strategy on current task data over multiple epochs.

        Args:
            train_loader: Dataloader yielding training batches.
            val_loader: Optional validation dataloader for evaluation after each epoch.
            epochs: Number of training epochs (default: 10).
            lr: Learning rate for newly initialized optimizer (default: 1e-4).
            optimizer: Optional pre-configured optimizer.
            **kwargs: Extra parameters.

        Returns:
            Dictionary containing history of loss, accuracy, and validation metrics.
        """
        self.last_train_loader = train_loader

        if optimizer is None:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable_params:
                # If all parameters are frozen (e.g. root backbone in LGR before head update), fallback to model parameters
                trainable_params = list(self.model.parameters())
            optimizer = torch.optim.Adam(trainable_params, lr=lr)

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_one_epoch(train_loader, optimizer, self.current_task_id)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if val_loader is not None:
                val_metrics = self.evaluate_task(val_loader)
                history["val_loss"].append(float(val_metrics.get("loss", 0.0)))
                history["val_acc"].append(float(val_metrics.get("accuracy", 0.0)))
                history["val_f1"].append(float(val_metrics.get("macro_f1", 0.0)))

        return history

    def update_memory_after_task(
        self,
        train_loader: Any,
        current_task_id: int,
        device: Union[torch.device, str],
    ) -> None:
        """Hook for post-task memory consolidation and parameter snapshotting.

        Args:
            train_loader: DataLoader from the completed task.
            current_task_id: Task identifier.
            device: Target execution device.
        """
        new_classes_count = len(self.current_task_classes) if self.current_task_classes else 2
        self._known_classes += new_classes_count

    def after_task(
        self,
        task_id: int,
        train_dataset: Optional[Any] = None,
        memory_manager: Optional[RehearsalMemoryManager] = None,
    ) -> None:
        """Lifecycle hook executed after training on a task finishes.

        Args:
            task_id: Identifier of the completed task.
            train_dataset: Dataset or DataLoader from the completed task.
            memory_manager: Optional memory manager to update.
        """
        if memory_manager is not None:
            self.memory_manager = memory_manager

        loader = None
        if train_dataset is not None:
            if isinstance(train_dataset, DataLoader):
                loader = train_dataset
            elif isinstance(train_dataset, Dataset):
                loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
            elif hasattr(train_dataset, "__iter__"):
                loader = train_dataset

        if loader is None and self.last_train_loader is not None:
            loader = self.last_train_loader

        if loader is not None:
            self.update_memory_after_task(loader, task_id, self.device)

            if self.memory_manager is not None:
                classes_to_store = self.current_task_classes or list(range(self._known_classes - 2, self._known_classes))
                self.memory_manager.update_memory_after_task(loader, classes_to_store)
        else:
            new_classes_count = len(self.current_task_classes) if self.current_task_classes else 2
            self._known_classes += new_classes_count

    @torch.no_grad()
    def evaluate_task(self, data_loader: Any) -> Dict[str, Any]:
        """Evaluates model performance on the provided DataLoader.

        Args:
            data_loader: DataLoader containing test or validation samples.

        Returns:
            Dictionary containing comprehensive classification metrics and loss.
        """
        if self.model is None:
            raise RuntimeError("Model must be assigned before evaluation.")

        self.model.eval()
        all_preds: List[int] = []
        all_targets: List[int] = []
        total_loss = 0.0
        total_samples = 0

        for batch in data_loader:
            h, imp, i1d, i2d, api, y = self._unpack_batch(batch)

            if getattr(self, "name", "").startswith("DER"):
                outputs, _ = self.model(h, imp, i1d, i2d, api)
            elif hasattr(self.model, "backbone") and hasattr(self.model, "classifier_head"):
                features_64 = self.model.backbone(h, imp, i1d, i2d, api)
                outputs = self.model.classifier_head(features_64)
            else:
                out = self.model(h, imp, i1d, i2d, api)
                outputs = out[0] if isinstance(out, tuple) else out

            loss = F.cross_entropy(outputs, y.long())
            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

        metrics = compute_classification_metrics(all_targets, all_preds)
        metrics["loss"] = total_loss / max(total_samples, 1)
        return metrics

    def get_extra_state(self) -> Dict[str, Any]:
        """Subclass hook to serialize strategy-specific state."""
        return {}

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        """Subclass hook to restore strategy-specific state."""
        pass

    def save_checkpoint(self, path: Union[str, os.PathLike]) -> None:
        """Serializes complete strategy and model state to file.

        Args:
            path: Destination file path for checkpoint.
        """
        path_str = str(path)
        parent_dir = os.path.dirname(path_str)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        checkpoint: Dict[str, Any] = {
            "name": self.name,
            "known_classes": self._known_classes,
            "seen_classes": list(self.seen_classes),
            "current_task_id": self.current_task_id,
            "current_task_classes": list(self.current_task_classes),
            "config": self.config,
            "model_state_dict": self.model.state_dict() if self.model is not None else None,
            "extra_state": self.get_extra_state(),
        }

        if self.memory_manager is not None and hasattr(self.memory_manager, "state_dict"):
            checkpoint["memory_manager"] = self.memory_manager.state_dict()

        torch.save(checkpoint, path_str)
        logger.info("Saved %s checkpoint to %s", self.name, path_str)

    def load_checkpoint(self, path: Union[str, os.PathLike]) -> None:
        """Restores complete strategy and model state from file.

        Args:
            path: Source checkpoint file path.
        """
        path_str = str(path)
        checkpoint = torch.load(path_str, map_location=self.device)

        self.name = checkpoint.get("name", self.name)
        self._known_classes = checkpoint.get("known_classes", self._known_classes)
        self.seen_classes = checkpoint.get("seen_classes", self.seen_classes)
        self.current_task_id = checkpoint.get("current_task_id", self.current_task_id)
        self.current_task_classes = checkpoint.get("current_task_classes", self.current_task_classes)
        self.config.update(checkpoint.get("config", {}))

        if self.model is not None and checkpoint.get("model_state_dict") is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.memory_manager is not None and "memory_manager" in checkpoint:
            self.memory_manager.load_state_dict(checkpoint["memory_manager"])

        if "extra_state" in checkpoint:
            self.set_extra_state(checkpoint["extra_state"])

        logger.info("Loaded %s checkpoint from %s", self.name, path_str)
