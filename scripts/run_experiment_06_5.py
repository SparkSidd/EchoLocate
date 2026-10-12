"""
EchoLocate — Experiment 6.5 Master Hardware Benchmark & Figure Runner
=====================================================================
Usage:
    python scripts/run_experiment_06_5.py [--output-dir results/experiment_06_5]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.evaluation.experiment_06_5_evaluation import run_experiment_06_5_evaluation
from src.visualization.experiment_06_5_plots import generate_all_experiment_06_5_figures


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "experiment_06_5.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="EchoLocate Experiment 6.5 Master Runner")
    parser.add_argument("--output-dir", type=str, default="results/experiment_06_5", help="Output directory")

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    setup_logging(out_dir)

    logger = logging.getLogger("EchoLocate.Exp06_5")
    logger.info("================================================================================")
    logger.info("EchoLocate — Experiment 6.5: Real Hardware Acquisition & Demonstration")
    logger.info("================================================================================")

    try:
        results = run_experiment_06_5_evaluation(output_dir=out_dir)

        fig_dir = out_dir / "figures"
        logger.info("Rendering 16 Publication Figures (F1 through F16) to: %s", fig_dir)
        generate_all_experiment_06_5_figures(output_dir=fig_dir)

        logger.info("================================================================================")
        logger.info("Experiment 6.5 Completed Successfully! All deliverables saved in: %s", out_dir)
        logger.info("================================================================================")
        return 0

    except Exception as e:
        logger.exception("Experiment 6.5 failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
