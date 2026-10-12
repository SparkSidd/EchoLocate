"""
EchoLocate — Experiment 4.75 Master CLI Runner
===============================================
Usage:
    python scripts/run_experiment_04_75.py [--smoke-test] [--audit-only]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.evaluation.neural_leakage_audit import run_full_neural_leakage_audit
from src.evaluation.statistical_validation import audit_experiment_04_75_dataset_splits
from src.experiments.experiment_04_75 import run_experiment_04_75_pipeline
from src.visualization.experiment_04_75_plots import generate_all_experiment_04_75_figures


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "experiment_04_75.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="EchoLocate Experiment 4.75 Runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test")
    parser.add_argument("--audit-only", action="store_true", help="Run static anti-leakage audit only")
    parser.add_argument("--config", type=str, default="configs/experiment_04_75.yaml", help="Config path")
    parser.add_argument("--output-dir", type=str, default="results/experiment_04_75", help="Output directory")

    args = parser.parse_args()
    out_dir = Path(args.output_dir) if not args.smoke_test else Path("results/experiment_04_75/smoke_test")
    setup_logging(out_dir)

    logger = logging.getLogger("EchoLocate.Exp04_75")
    logger.info("================================================================================")
    logger.info("EchoLocate — Experiment 4.75: Neural Calibration & Adversarial Robustness Audit")
    logger.info("================================================================================")

    if args.audit_only:
        logger.info("Running Static Anti-Leakage Audit...")
        df_leakage = run_full_neural_leakage_audit()
        print("\n--- Leakage Audit Summary ---")
        print(df_leakage.to_string(index=False))

        df_split = audit_experiment_04_75_dataset_splits()
        print("\n--- Dataset Split Isolation Summary ---")
        print(df_split.to_string(index=False))
        return 0

    try:
        results = run_experiment_04_75_pipeline(
            config_path=args.config,
            output_dir=out_dir,
            smoke_test=args.smoke_test,
        )

        fig_dir = out_dir / "figures"
        logger.info("Rendering Publication Figures F1 through F30 to: %s", fig_dir)
        generate_all_experiment_04_75_figures(results, output_dir=fig_dir)

        logger.info("================================================================================")
        logger.info("Experiment 4.75 Completed Successfully! Outputs saved in: %s", out_dir)
        logger.info("================================================================================")
        return 0

    except Exception as e:
        logger.exception("Experiment 4.75 failed with unhandled exception: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
