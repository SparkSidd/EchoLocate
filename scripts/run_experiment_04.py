"""
EchoLocate — Experiment 4 Master Benchmark Runner Script
========================================================
Orchestrates execution of Experiment 4: Learned Acoustic Feature Enhancement
for Robust Speech Localization Under Reverberation.

Usage:
    python scripts/run_experiment_04.py [--smoke-test] [--model-path path/to/model.pt]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.learned_spatial_likelihood import run_experiment_04_pipeline
from src.visualization.experiment_04_plots import generate_all_experiment_04_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EchoLocate.Exp04")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EchoLocate Experiment 4")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke test of all modules")
    parser.add_argument("--config", type=str, default="configs/experiment_04.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default="results/experiment_04", help="Output directory")
    parser.add_argument("--model-path", type=str, default=None, help="Optional pre-trained model checkpoint")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if args.smoke_test:
        out_dir = out_dir / "smoke_test"

    logger.info("================================================================================")
    logger.info("EchoLocate Experiment 4: Learned Acoustic Feature Enhancement (DeepGCC + SRP)")
    logger.info("Mode: %s | Output: %s", "Smoke Test" if args.smoke_test else "Full Execution", out_dir)
    logger.info("================================================================================")

    # 1. Run simulation and evaluations
    results_dict = run_experiment_04_pipeline(
        config_path=args.config,
        output_dir=out_dir,
        model_checkpoint_path=args.model_path,
        smoke_test=args.smoke_test,
    )

    # 2. Generate all 30 publication figures
    generate_all_experiment_04_figures(
        results_dict=results_dict,
        output_dir=out_dir / "figures",
    )

    logger.info("================================================================================")
    logger.info("Experiment 4 Completed Successfully! Results saved to: %s", out_dir)
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
