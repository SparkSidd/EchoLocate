"""
EchoLocate — Experiment 6 Master Benchmark & Study Runner
=========================================================
Usage:
    python scripts/run_experiment_06.py [--output-dir results/experiment_06]
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

from src.evaluation.experiment_06_evaluation import run_experiment_06_evaluation
from src.visualization.experiment_06_plots import generate_all_experiment_06_figures


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "experiment_06.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="EchoLocate Experiment 6 Runner")
    parser.add_argument("--output-dir", type=str, default="results/experiment_06", help="Output directory")

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    setup_logging(out_dir)

    logger = logging.getLogger("EchoLocate.Exp06")
    logger.info("================================================================================")
    logger.info("EchoLocate — Experiment 6: Real-Time Working Prototype & Deployment Validation")
    logger.info("================================================================================")

    try:
        results = run_experiment_06_evaluation(output_dir=out_dir)

        fig_dir = out_dir / "figures"
        logger.info("Rendering Publication Figures F1 through F16 to: %s", fig_dir)
        generate_all_experiment_06_figures(results, output_dir=fig_dir)

        logger.info("================================================================================")
        logger.info("Experiment 6 Completed Successfully! Outputs saved in: %s", out_dir)
        logger.info("================================================================================")
        return 0

    except Exception as e:
        logger.exception("Experiment 6 execution failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
