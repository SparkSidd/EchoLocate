"""
EchoLocate — Run Script: Experiment 2.5 Transition-Zone Sweep
==============================================================
Fine-grained reverberation threshold refinement study.
Supports:
  --smoke-test : Fast 3-condition validation
  Default      : Full 7-condition sweep (525 evaluations)
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.transition_zone import run_transition_zone_experiment
from src.visualization.transition_zone_plots import generate_all_transition_zone_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EchoLocate.Exp02_5")


def main():
    parser = argparse.ArgumentParser(description="Run EchoLocate Experiment 2.5")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast 3-condition smoke test")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "configs" / "experiment_02_5.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info("================================================================================")
    logger.info("  EchoLocate — Experiment 2.5: Transition-Zone Verification & Threshold Refinement")
    logger.info("================================================================================")

    if args.smoke_test:
        out_dir = PROJECT_ROOT / "results" / "experiment_02_5" / "smoke_test"
        logger.info("Running PRE-FLIGHT SMOKE TEST into: %s", out_dir)
        df_summary, df_evals, df_rt60_sum, df_per_source = run_transition_zone_experiment(
            cfg=cfg,
            output_dir=out_dir,
            smoke_test=True,
        )
        generate_all_transition_zone_figures(
            df_summary=df_summary,
            df_evals=df_evals,
            df_rt60_summary=df_rt60_sum,
            df_per_source=df_per_source,
            output_dir=out_dir,
        )
        logger.info("Smoke test completed successfully without errors or NaNs!")
        return

    out_dir = PROJECT_ROOT / "results" / "experiment_02_5"
    logger.info("Running FULL EXPERIMENT 2.5 into: %s", out_dir)
    df_summary, df_evals, df_rt60_sum, df_per_source = run_transition_zone_experiment(
        cfg=cfg,
        output_dir=out_dir,
        smoke_test=False,
    )
    generate_all_transition_zone_figures(
        df_summary=df_summary,
        df_evals=df_evals,
        df_rt60_summary=df_rt60_sum,
        df_per_source=df_per_source,
        output_dir=out_dir,
    )
    logger.info("================================================================================")
    logger.info("Experiment 2.5 Complete! Results saved to: %s", out_dir)
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
