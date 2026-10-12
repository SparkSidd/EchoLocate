"""
EchoLocate — Experiment 3 Master Runner Script
==============================================
Orchestrates execution of Experiment 3: Steered Response Power (SRP-PHAT)
Spatial Likelihood Localization Baseline against GCC-PHAT + NLS.

Usage:
    python scripts/run_experiment_03.py [--smoke-test]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.spatial_likelihood import run_experiment_03_spatial_likelihood
from src.visualization.experiment_03_plots import generate_all_experiment_03_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EchoLocate.Exp03")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EchoLocate Experiment 3 (SRP-PHAT)")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast smoke test (2 conditions, 3 sources, 2 trials)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_03.yaml",
        help="Path to experiment configuration YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiment_03",
        help="Output directory for CSVs and plots",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if args.smoke_test:
        out_dir = out_dir / "smoke_test"

    logger.info("================================================================================")
    logger.info("EchoLocate Experiment 3: SRP-PHAT Spatial Likelihood Localization Baseline")
    logger.info("Mode: %s | Output: %s", "Smoke Test" if args.smoke_test else "Full Execution", out_dir)
    logger.info("================================================================================")

    # 1. Run paired simulation & evaluation
    df_summary, df_evals, df_per_source, df_resolution, df_rt60_sum, diag_maps = run_experiment_03_spatial_likelihood(
        config_path=args.config,
        output_dir=out_dir,
        smoke_test=args.smoke_test,
    )

    # 2. Generate Figures
    generate_all_experiment_03_figures(
        df_summary=df_summary,
        df_evals=df_evals,
        df_per_source=df_per_source,
        df_resolution=df_resolution,
        diagnostic_maps=diag_maps,
        output_dir=out_dir / "figures",
    )

    logger.info("================================================================================")
    logger.info("Experiment 3 Completed Successfully!")
    logger.info("Summary Results:\n%s", df_summary[["condition_id", "nls_mean_error_m", "srp_refined_mean_error_m", "delta_mean_error_m", "srp_win_rate_pct"]].to_string())
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
