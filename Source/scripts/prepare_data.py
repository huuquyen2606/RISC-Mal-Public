#!/usr/bin/env python
"""CLI tool for PE malware feature extraction, vocabulary generation, and dataset preparation.

Corresponds to the offline feature extraction pipeline (trichxuat2.py) that produces
incremental_data_v2/ (archived as Data_RISC.zip on Google Drive).
Note: Model training (train_incremental.py) directly consumes incremental_data_v2/.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from riscmal.data.leakage import purge_cross_split_duplicates
from riscmal.data.pipeline import GlobalAssetBuilder
from riscmal.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for data preparation and feature extraction."""
    parser = argparse.ArgumentParser(description="RISC-Mal Data Preparation and Feature Extraction CLI")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing raw or extracted data")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for processed datasets")
    parser.add_argument("--purge-leakage", action="store_true", help="Perform SHA-256 byte deduplication across train/test splits")
    parser.add_argument("--build-assets", action="store_true", help="Extract and build global API vocabulary and imports dictionary")
    parser.add_argument("--train-dir", type=str, default="data/train", help="Train split directory")
    parser.add_argument("--test-dir", type=str, default="data/test", help="Test split directory")
    return parser.parse_args()


def main() -> None:
    """CLI entry point: executes global asset construction and cross-split deduplication."""
    args = parse_args()
    logger = setup_logger("prepare_data")
    logger.info("Starting RISC-Mal Data Preparation Pipeline...")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.build_assets:
        logger.info("Building global assets from %s...", args.data_dir)
        builder = GlobalAssetBuilder(base_json_dir=args.data_dir, output_dir=args.output_dir)
        assets = builder.build_assets()
        logger.info("Global assets generated successfully in %s", args.output_dir)

    if args.purge_leakage:
        logger.info("Executing SHA-256 cryptographic cross-split deduplication...")
        purged_train, purged_test, dupes = purge_cross_split_duplicates(args.train_dir, args.test_dir)
        logger.info("Deduplication complete. Removed %d duplicate samples between train and test.", len(dupes))

    logger.info("Data preparation pipeline complete.")


if __name__ == "__main__":
    main()
