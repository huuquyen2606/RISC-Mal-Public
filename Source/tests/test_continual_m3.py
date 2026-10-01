"""Comprehensive verification test suite for Milestone 3 Continual Learning Strategies.

Verifies:
1. BaselineStrategy (Cell 04): Sequential fine-tuning with expanding head.
2. EWCStrategy (Cell 05): Online EWC with diagonal Fisher consolidation.
3. LwFStrategy (Cell 06): Learning without Forgetting distillation.
4. GCStrategy (Cell 09): Generative Classifier with class-specific VAE ELBO logits.
5. BIRStrategy (Cell 08): Brain-Inspired Replay with ConditionalVAE GMM prior.
6. LGRStrategy (Cell 07): Latent Generative Replay with frozen backbone.
7. Checkpoint persistence (save_checkpoint and load_checkpoint).
"""

import os
import sys
import tempfile
from typing import Dict, List

# Ensure src is on sys.path when running standalone
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from riscmal.continual import (
    BaseContinualStrategy,
    BaselineStrategy,
    BIRStrategy,
    EWCStrategy,
    GCStrategy,
    LGRStrategy,
    LwFStrategy,
    get_strategy,
)
from riscmal.models import MalwareMultiViewClassifier


class SyntheticMultiViewDataset(Dataset):
    """Generates synthetic multi-view malware samples matching project contracts."""

    def __init__(self, n_samples: int = 12, classes: List[int] = None) -> None:
        self.n_samples = n_samples
        self.classes = classes if classes is not None else [0, 1]
        self.labels = [self.classes[i % len(self.classes)] for i in range(n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "header": torch.randn(4),
            "imports": torch.randn(1000),
            "img1d": torch.randn(1024),
            "img2d": torch.randn(3, 224, 224),
            "apis": torch.randint(0, 101, (100,)),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def make_loaders(classes: List[int], batch_size: int = 4):
    """Constructs synthetic multi-view train and validation datasets and loaders."""
    ds_train = SyntheticMultiViewDataset(n_samples=8, classes=classes)
    ds_val = SyntheticMultiViewDataset(n_samples=4, classes=classes)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    return ds_train, loader_train, loader_val


class TestBaselineStrategy:
    """Verifies BaselineStrategy 2-task lifecycle."""

    def test_baseline_two_task_lifecycle(self):
        """Verifies BaselineStrategy 2-task training, head expansion, and evaluation."""
        device = "cpu"
        model = MalwareMultiViewClassifier(initial_classes=2)
        strategy = BaselineStrategy(model=model, device=device)

        # Task 1
        ds1, train_l1, val_l1 = make_loaders(classes=[0, 1])
        strategy.before_task(task_id=1, task_classes=[0, 1], train_dataset=ds1)
        history1 = strategy.train_task(train_l1, val_loader=val_l1, epochs=1, lr=1e-4)
        assert len(history1["train_loss"]) == 1
        assert len(history1["val_f1"]) == 1
        strategy.after_task(task_id=1, train_dataset=train_l1)
        metrics1 = strategy.evaluate_task(val_l1)
        assert "accuracy" in metrics1
        assert "macro_f1" in metrics1

        # Task 2: Output head expansion
        ds2, train_l2, val_l2 = make_loaders(classes=[2, 3])
        strategy.before_task(task_id=2, task_classes=[2, 3], train_dataset=ds2)
        assert model.classifier_head.out_features == 4
        history2 = strategy.train_task(train_l2, val_loader=val_l2, epochs=1, lr=1e-4)
        assert len(history2["train_loss"]) == 1
        strategy.after_task(task_id=2, train_dataset=train_l2)
        metrics2 = strategy.evaluate_task(val_l2)
        assert metrics2["accuracy"] >= 0.0


class TestEWCStrategy:
    """Verifies EWCStrategy diagonal Fisher estimation and online accumulation."""

    def test_ewc_two_task_lifecycle(self):
        """Verifies EWC Fisher information matrix accumulation and quadratic penalty."""
        device = "cpu"
        model = MalwareMultiViewClassifier(initial_classes=2)
        strategy = EWCStrategy(model=model, device=device, lamda=100.0, fishermax=1e-4)

        # Task 1
        ds1, train_l1, val_l1 = make_loaders(classes=[0, 1])
        strategy.before_task(task_id=1, task_classes=[0, 1], train_dataset=ds1)
        strategy.train_task(train_l1, epochs=1)
        assert strategy.fisher is None
        strategy.after_task(task_id=1, train_dataset=train_l1)
        assert strategy.fisher is not None
        assert strategy.mean is not None

        # Task 2
        ds2, train_l2, val_l2 = make_loaders(classes=[2, 3])
        strategy.before_task(task_id=2, task_classes=[2, 3], train_dataset=ds2)
        assert model.classifier_head.out_features == 4
        strategy.train_task(train_l2, epochs=1)
        strategy.after_task(task_id=2, train_dataset=train_l2)
        eval_metrics = strategy.evaluate_task(val_l2)
        assert "macro_f1" in eval_metrics


class TestLwFStrategy:
    """Verifies LwFStrategy snapshot teacher and KD distillation."""

    def test_lwf_two_task_lifecycle(self):
        """Verifies LwF teacher model snapshotting and knowledge distillation."""
        device = "cpu"
        model = MalwareMultiViewClassifier(initial_classes=2)
        strategy = LwFStrategy(model=model, device=device, temperature=2.0, lambda_distill=3.0)

        # Task 1
        ds1, train_l1, val_l1 = make_loaders(classes=[0, 1])
        strategy.before_task(task_id=1, task_classes=[0, 1], train_dataset=ds1)
        strategy.train_task(train_l1, epochs=1)
        assert strategy.old_model is None
        strategy.after_task(task_id=1, train_dataset=train_l1)
        assert strategy.old_model is not None

        # Task 2
        ds2, train_l2, val_l2 = make_loaders(classes=[2, 3])
        strategy.before_task(task_id=2, task_classes=[2, 3], train_dataset=ds2)
        assert model.classifier_head.out_features == 4
        history2 = strategy.train_task(train_l2, epochs=1)
        assert len(history2["train_loss"]) == 1
        strategy.after_task(task_id=2, train_dataset=train_l2)
        eval_metrics = strategy.evaluate_task(val_l2)
        assert "accuracy" in eval_metrics


class TestGCStrategy:
    """Verifies GCStrategy ClassVAE training and ELBO classification."""

    def test_gc_two_task_lifecycle(self):
        """Verifies Generative Classifier ClassVAE training and ELBO-based inference."""
        device = "cpu"
        model = MalwareMultiViewClassifier(initial_classes=2)
        strategy = GCStrategy(model=model, device=device, vae_epochs=2, vae_hidden_dim=32, vae_z_dim=16)

        # Task 1
        ds1, train_l1, val_l1 = make_loaders(classes=[0, 1])
        strategy.before_task(task_id=1, task_classes=[0, 1], train_dataset=ds1)
        strategy.train_task(train_l1, epochs=1)
        strategy.after_task(task_id=1, train_dataset=train_l1)
        assert strategy._gc_head is not None
        assert 0 in strategy._gc_head.vaes and 1 in strategy._gc_head.vaes

        # Task 2
        ds2, train_l2, val_l2 = make_loaders(classes=[2, 3])
        strategy.before_task(task_id=2, task_classes=[2, 3], train_dataset=ds2)
        strategy.train_task(train_l2, epochs=1)
        strategy.after_task(task_id=2, train_dataset=train_l2)
        assert 2 in strategy._gc_head.vaes and 3 in strategy._gc_head.vaes

        # Inference uses ELBO scoring
        eval_metrics = strategy.evaluate_task(val_l2)
        assert "macro_f1" in eval_metrics


class TestBIRStrategy:
    """Verifies BIRStrategy ConditionalVAE GMM prior and latent replay."""

    def test_bir_two_task_lifecycle(self):
        """Verifies Brain-Inspired Replay ConditionalVAE training and feature rehearsal."""
        device = "cpu"
        model = MalwareMultiViewClassifier(initial_classes=2)
        strategy = BIRStrategy(
            model=model,
            device=device,
            cvae_epochs=2,
            cvae_hidden_dim=32,
            cvae_z_dim=16,
            replay_ratio=0.5,
        )

        # Task 1
        ds1, train_l1, val_l1 = make_loaders(classes=[0, 1])
        strategy.before_task(task_id=1, task_classes=[0, 1], train_dataset=ds1)
        strategy.train_task(train_l1, epochs=1)
        strategy.after_task(task_id=1, train_dataset=train_l1)
        assert strategy._cvae is not None

        # Task 2
        ds2, train_l2, val_l2 = make_loaders(classes=[2, 3])
        strategy.before_task(task_id=2, task_classes=[2, 3], train_dataset=ds2)
        # Backbone is frozen after Task 1
        for p in model.backbone.parameters():
            assert not p.requires_grad
        strategy.train_task(train_l2, epochs=1)
        strategy.after_task(task_id=2, train_dataset=train_l2)
        eval_metrics = strategy.evaluate_task(val_l2)
        assert "accuracy" in eval_metrics


class TestLGRStrategy:
    """Verifies LGRStrategy LatentCVAE generator and pseudo-feature replay."""

    def test_lgr_two_task_lifecycle(self):
        """Verifies Latent Generative Replay LatentCVAE training and latent feature replay."""
        device = "cpu"
        model = MalwareMultiViewClassifier(initial_classes=2)
        strategy = LGRStrategy(
            model=model,
            device=device,
            vae_epochs=2,
            hidden_dim=32,
            latent_dim=16,
            replay_ratio=0.5,
        )

        # Task 1
        ds1, train_l1, val_l1 = make_loaders(classes=[0, 1])
        strategy.before_task(task_id=1, task_classes=[0, 1], train_dataset=ds1)
        strategy.train_task(train_l1, epochs=1)
        strategy.after_task(task_id=1, train_dataset=train_l1)
        assert strategy.old_vae is not None
        assert strategy.old_top is not None

        # Task 2
        ds2, train_l2, val_l2 = make_loaders(classes=[2, 3])
        strategy.before_task(task_id=2, task_classes=[2, 3], train_dataset=ds2)
        assert strategy.root_frozen
        for p in model.backbone.parameters():
            assert not p.requires_grad
        strategy.train_task(train_l2, epochs=1)
        strategy.after_task(task_id=2, train_dataset=train_l2)
        eval_metrics = strategy.evaluate_task(val_l2)
        assert "macro_f1" in eval_metrics


class TestCheckpointPersistence:
    """Verifies save_checkpoint and load_checkpoint across strategies."""

    def test_save_load_checkpoint(self):
        """Verifies checkpoint saving and state restoration across strategy instances."""
        model = MalwareMultiViewClassifier(initial_classes=2)
        strategy = EWCStrategy(model=model, device="cpu", lamda=500.0)

        ds1, train_l1, _ = make_loaders(classes=[0, 1])
        strategy.before_task(1, [0, 1], ds1)
        strategy.train_task(train_l1, epochs=1)
        strategy.after_task(1, train_l1)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "ewc_ckpt.pt")
            strategy.save_checkpoint(ckpt_path)
            assert os.path.exists(ckpt_path)

            new_model = MalwareMultiViewClassifier(initial_classes=2)
            restored_strategy = EWCStrategy(model=new_model, device="cpu")
            restored_strategy.load_checkpoint(ckpt_path)

            assert restored_strategy.known_classes == strategy.known_classes
            assert restored_strategy.fisher is not None
            assert restored_strategy.mean is not None


class TestRegistryIntegration:
    """Verifies factory instantiation of Milestone 3 strategies."""

    @pytest.mark.parametrize(
        "name,expected_cls",
        [
            ("baseline", BaselineStrategy),
            ("ewc", EWCStrategy),
            ("lwf", LwFStrategy),
            ("gc", GCStrategy),
            ("bir", BIRStrategy),
            ("lgr", LGRStrategy),
        ],
    )
    def test_get_strategy_factory(self, name, expected_cls):
        """Verifies registry lookup and instantiation for all Continual Learning strategies."""
        model = MalwareMultiViewClassifier(initial_classes=2)
        strat = get_strategy(name, model=model, device="cpu")
        assert isinstance(strat, expected_cls)
        assert strat.model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
