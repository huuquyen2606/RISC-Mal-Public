#!/usr/bin/env python
"""CLI tool for running incremental continual learning experiments across tasks."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict
import torch
import torch.nn.functional as F
import yaml

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from riscmal.backbones import MultiViewFeatureExtractor
from riscmal.continual import get_strategy
from riscmal.data.dataset import MalwareMultiViewDataset, multiview_collate_fn
from riscmal.data.pipeline import build_incremental_dataloaders
from riscmal.evaluation.metrics import compute_classification_metrics
from riscmal.evaluation.tracker import ContinualMetricsTracker
from riscmal.models import DERClassifier, FOSTERClassifier, MalwareMultiViewClassifier
from riscmal.utils.device import get_device
from riscmal.utils.logger import setup_logger
from riscmal.utils.seed import set_deterministic_seed


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for incremental training."""
    parser = argparse.ArgumentParser(description="RISC-Mal Incremental Training CLI")
    parser.add_argument("--method", type=str, required=True, help="Continual learning method name")
    parser.add_argument("--config", type=str, default="configs/protocol.yaml", help="Path to global protocol config")
    parser.add_argument("--method-config", type=str, default=None, help="Path to method-specific YAML config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per task (overrides config)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save logs and checkpoints")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="incremental_data_v2",
        help="Path to incremental dataset directory containing task1, task2, task3",
    )
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    """Safely loads a YAML configuration file into a Python dictionary."""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def resolve_data_dir(data_dir: str) -> Path:
    """Finds the dataset directory among common candidate paths."""
    candidates = [
        Path(data_dir),
        Path("incremental_data_v2"),
        Path("../incremental_data_v2"),
        Path("../../incremental_data_v2"),
        Path("data"),
        Path("../data"),
    ]
    for c in candidates:
        if c.is_dir() and (c / "task1").is_dir():
            return c
    return Path(data_dir)


def main() -> None:
    """CLI entry point: configures seeds, initializes strategy, and executes incremental training."""
    args = parse_args()
    logger = setup_logger("train_incremental")

    # 1. Deterministic Seeding
    set_deterministic_seed(args.seed)
    device = get_device(args.device)
    logger.info("Running method: %s | Seed: %d | Device: %s", args.method, args.seed, device)

    # 2. Load configurations
    protocol = load_yaml(args.config)
    method_cfg_file = args.method_config or f"configs/methods/{args.method.lower()}.yaml"
    method_cfg = load_yaml(method_cfg_file).get("params", {})

    epochs = args.epochs or protocol.get("training", {}).get("epochs_per_task", 10)
    lr = args.lr or protocol.get("training", {}).get("learning_rate", 1e-4)

    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Initialize Backbone and Classifier
    fe = MultiViewFeatureExtractor()
    method_name = args.method.strip().lower()

    if method_name == "der":
        model = DERClassifier(fe, initial_classes=2).to(device)
    elif method_name == "foster":
        model = FOSTERClassifier(fe, initial_classes=2).to(device)
    else:
        model = MalwareMultiViewClassifier(fe, initial_classes=2).to(device)

    # 4. Instantiate Continual Strategy
    strategy = get_strategy(method_name, device=device, **method_cfg)
    strategy.model = model

    tracker = ContinualMetricsTracker()
    logger.info("Initialized %s strategy successfully.", strategy.name)

    # 5. Check for real dataset
    resolved_data = resolve_data_dir(args.data_dir)
    has_real_data = resolved_data.is_dir() and (resolved_data / "task1").is_dir()

    if has_real_data:
        logger.info("Found real dataset at: %s. Loading incremental tasks...", resolved_data)
        tasks_data = build_incremental_dataloaders(
            data_dir=resolved_data,
            batch_size=protocol.get("data", {}).get("batch_size", 32),
            purge_leakage=protocol.get("data", {}).get("purge_cross_split_leakage", True),
        )

        for task_idx, task_loaders in enumerate(tasks_data):
            task_id = task_idx + 1
            logger.info("Starting Task %d training (%s)...", task_id, strategy.name)

            if task_id > 1 and hasattr(strategy, "adapt_architecture_before_task"):
                strategy.adapt_architecture_before_task(device)

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
            )

            train_loader = task_loaders["train_loader"]
            for epoch in range(1, epochs + 1):
                model.train()
                total_loss = 0.0
                batches = 0
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()

                    if method_name == "der":
                        logits, feats = model(batch.header, batch.imports, batch.img1d, batch.img2d, batch.apis)
                    else:
                        feats = model.backbone(batch.header, batch.imports, batch.img1d, batch.img2d, batch.apis)
                        logits = model.classifier_head(feats)

                    loss = F.cross_entropy(logits, batch.label)
                    add_loss = strategy.compute_additional_loss(model, feats, logits, batch, current_task_id=task_id)
                    loss = loss + add_loss
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    batches += 1

                avg_loss = total_loss / max(1, batches)
                logger.info("Task %d | Epoch %02d/%02d | Loss: %.4f", task_id, epoch, epochs, avg_loss)

            # Update strategy memory
            new_classes = [task_idx * 2, task_idx * 2 + 1]
            strategy.update_memory_after_task(train_loader, new_classes=new_classes)

            # Save checkpoint
            ckpt_path = os.path.join(args.output_dir, f"{args.method.upper()}_S{args.seed}_task{task_id}_checkpoint.pth")
            torch.save({"model_state": model.state_dict(), "method": strategy.name, "seed": args.seed, "task": task_id}, ckpt_path)
            logger.info("Task %d checkpoint saved to: %s", task_id, ckpt_path)
    else:
        logger.warning(
            "Dataset folder not found at '%s'. Executing standalone setup verification. "
            "To train on real data, place or specify --data-dir pointing to incremental_data_v2/.",
            resolved_data,
        )
        ckpt_path = os.path.join(args.output_dir, f"{args.method.upper()}_S{args.seed}_task1_checkpoint.pth")
        torch.save({"model_state": model.state_dict(), "method": strategy.name, "seed": args.seed, "task": 1}, ckpt_path)
        logger.info("Checkpoint saved to: %s", ckpt_path)

    logger.info("Incremental training CLI execution completed successfully.")


if __name__ == "__main__":
    main()
