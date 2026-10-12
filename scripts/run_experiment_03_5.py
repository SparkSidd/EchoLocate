"""
EchoLocate — Experiment 3.5 Master Runner Script
================================================
Orchestrates execution of Experiment 3.5: Reality Check, Model-Mismatch,
Generalization & Leakage Validation for SRP-PHAT.

Usage:
    python scripts/run_experiment_03_5.py [--smoke-test]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.reality_check import run_experiment_03_5_reality_check
from src.visualization.reality_check_plots import generate_all_experiment_03_5_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EchoLocate.Exp03_5")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EchoLocate Experiment 3.5 (Reality Check)")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast smoke test of all 14 sub-studies",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_03_5.yaml",
        help="Path to experiment configuration YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiment_03_5",
        help="Output directory for CSVs and plots",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if args.smoke_test:
        out_dir = out_dir / "smoke_test"

    logger.info("================================================================================")
    logger.info("EchoLocate Experiment 3.5: Reality Check & Adversarial Model-Mismatch Audit")
    logger.info("Mode: %s | Output: %s", "Smoke Test" if args.smoke_test else "Full Execution", out_dir)
    logger.info("================================================================================")

    # 1. Run all reality check sub-studies
    results_dict = run_experiment_03_5_reality_check(
        config_path=args.config,
        output_dir=out_dir,
        smoke_test=args.smoke_test,
    )

    # 2. Generate Figures
    generate_all_experiment_03_5_figures(
        results_dict=results_dict,
        output_dir=out_dir / "figures",
    )

    logger.info("================================================================================")
    logger.info("Experiment 3.5 Completed Successfully! Results saved to: %s", out_dir)
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
