#!/usr/bin/env python
"""CLI tool for profiling parameter counts (total, trainable, frozen, buffer) across Continual Learning methods."""

import argparse
import sys
from pathlib import Path
from typing import Tuple
import torch

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from riscmal.backbones import MultiViewFeatureExtractor
from riscmal.models import DERClassifier, FOSTERClassifier, MalwareMultiViewClassifier
from riscmal.models.vae import ClassVAE, GCHead


def count_params(model: torch.nn.Module) -> Tuple[int, int, int, float]:
    """Computes total, trainable, and frozen parameter counts and trainable ratio."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    ratio = (trainable / total * 100.0) if total > 0 else 0.0
    return total, trainable, frozen, ratio


def print_stats(name: str, task_id: int, total: int, trainable: int, frozen: int, ratio: float) -> None:
    """Prints formatted parameter profiling summary table."""
    print("=" * 60)
    print(f"Model: {name} (Task {task_id})")
    print("=" * 60)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {frozen:,}")
    print(f"Trainable ratio:      {ratio:.2f}%")
    print()


def profile_static(task_id: int, method_label: str = "Static Multi-View Classifier") -> None:
    """Profiles parameter footprint of static backbone with expanding classifier head (RISC-Mal and baselines)."""
    fe = MultiViewFeatureExtractor()
    num_classes = task_id * 2
    model = MalwareMultiViewClassifier(fe, initial_classes=num_classes)
    total, trainable, frozen, ratio = count_params(model)
    print_stats(method_label, task_id, total, trainable, frozen, ratio)


def profile_der(task_id: int) -> None:
    """Profiles parameter footprint of Dynamically Expandable Representation (DER) multi-column network."""
    fe = MultiViewFeatureExtractor()
    model = DERClassifier(fe, initial_classes=2)
    for _ in range(2, task_id + 1):
        model.add_column(MultiViewFeatureExtractor(), num_new_classes=2)
    total, trainable, frozen, ratio = count_params(model)
    print_stats("DER Dynamic Multi-Column Classifier", task_id, total, trainable, frozen, ratio)


def profile_foster(task_id: int) -> None:
    """Profiles parameter footprint of FOSTER dynamic feature boosting network."""
    fe = MultiViewFeatureExtractor()
    num_classes = task_id * 2
    model = FOSTERClassifier(fe, initial_classes=num_classes)
    if task_id > 1:
        model.backbone.add_backbone(MultiViewFeatureExtractor())
        model.backbone.freeze_old_backbones()
    total, trainable, frozen, ratio = count_params(model)
    print_stats("FOSTER Dynamic Classifier", task_id, total, trainable, frozen, ratio)


def profile_gc(task_id: int) -> None:
    """Profiles parameter footprint of Generative Classifier (GC) ensemble of per-class VAEs."""
    fe = MultiViewFeatureExtractor()
    num_classes = task_id * 2
    head = GCHead(initial_classes=num_classes, feature_dim=64)
    for c in range(num_classes):
        head.register_vae(c, ClassVAE(input_dim=64, hidden_dim=128, z_dim=32))

    class CombinedGC(torch.nn.Module):
        """Wrapper combining multi-view backbone and Generative Classifier head for parameter profiling."""

        def __init__(self, bb: torch.nn.Module, h: torch.nn.Module) -> None:
            super().__init__()
            self.backbone = bb
            self.head = h
            if task_id > 1:
                for p in self.backbone.parameters():
                    p.requires_grad_(False)

    model = CombinedGC(fe, head)
    total, trainable, frozen, ratio = count_params(model)
    print_stats("GC Generative Classifier", task_id, total, trainable, frozen, ratio)


def main() -> None:
    """CLI entry point for model parameter profiling."""
    parser = argparse.ArgumentParser(description="RISC-Mal Model Parameter Profiler")
    parser.add_argument("--method", type=str, default="all", help="Method name (riscmal, der, foster, gc, baseline, ewc, lwf, bir, lgr, all)")
    parser.add_argument("--task", type=int, default=1, choices=[1, 2, 3], help="Task ID (1, 2, or 3)")
    args = parser.parse_args()

    method = args.method.strip().lower()
    task = args.task

    if method in ["all", "riscmal", "risc-mal"]:
        profile_static(task, "RISC-Mal (Fixed Multi-View Backbone + Expanded Head)")

    if method in ["all", "baseline", "w/o cil", "ewc", "lwf", "bir", "lgr"]:
        profile_static(task, f"Static Baseline ({method.upper()})")

    if method in ["all", "der"]:
        profile_der(task)

    if method in ["all", "foster"]:
        profile_foster(task)

    if method in ["all", "gc"]:
        profile_gc(task)


if __name__ == "__main__":
    main()
