"""
EchoLocate — Experiment 4.5 Master Runner Script
================================================
Orchestrates execution of Experiment 4.5: Neural Likelihood Calibration,
Peak Contrast Optimization & Localization-Level Training.

Usage:
    python scripts/run_experiment_04_5.py [--smoke-test] [--audit-only]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.neural_likelihood_refinement import run_experiment_04_5_pipeline
from src.visualization.experiment_04_5_plots import generate_all_experiment_04_5_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EchoLocate.Exp04_5")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EchoLocate Experiment 4.5")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke test")
    parser.add_argument("--audit-only", action="store_true", help="Run only the diagnostic audit")
    parser.add_argument("--config", type=str, default="configs/experiment_04_5.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default="results/experiment_04_5", help="Output directory")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if args.smoke_test:
        out_dir = out_dir / "smoke_test"

    logger.info("================================================================================")
    logger.info("EchoLocate Experiment 4.5: Neural Likelihood Calibration & Spatial Training")
    logger.info("Mode: %s | Output: %s", "Smoke Test" if args.smoke_test else "Full Execution", out_dir)
    logger.info("================================================================================")

    # 1. Run Pipeline
    results_dict = run_experiment_04_5_pipeline(
        config_path=args.config,
        output_dir=out_dir,
        smoke_test=args.smoke_test,
    )

    # 2. Generate Figures F1 through F30
    generate_all_experiment_04_5_figures(
        results_dict=results_dict,
        output_dir=out_dir / "figures",
    )

    logger.info("================================================================================")
    logger.info("Experiment 4.5 Completed Successfully! Results saved to: %s", out_dir)
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
