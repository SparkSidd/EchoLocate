"""
EchoLocate — Experiment 4 Standalone DeepGCC Training Script
============================================================
Trains the DeepGCC 1D Residual Convolutional Network on synthetic multi-channel scenes.

Usage:
    python scripts/train_experiment_04.py [--smoke-test] [--config configs/experiment_04.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.training import train_deep_gcc_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EchoLocate.TrainExp04")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepGCC Neural Model for Experiment 4")
    parser.add_argument("--config", type=str, default="configs/experiment_04.yaml", help="Path to config YAML")
    parser.add_argument("--save-path", type=str, default="results/experiment_04/deep_gcc_model.pt", help="Path to save weights")
    parser.add_argument("--history-path", type=str, default="results/experiment_04/training_history.csv", help="Path to save history")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke training")
    parser.add_argument("--no-ipd", action="store_true", help="Train ablation without IPD loss")

    args = parser.parse_args()

    logger.info("Starting DeepGCC training pipeline...")
    train_deep_gcc_model(
        config_path=args.config,
        save_checkpoint_path=args.save_path,
        history_csv_path=args.history_path,
        use_ipd_loss=not args.no_ipd,
        smoke_test=args.smoke_test,
    )
    logger.info("DeepGCC training finished successfully.")


if __name__ == "__main__":
    main()
