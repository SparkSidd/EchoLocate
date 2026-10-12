"""
EchoLocate — Experiment 02 Master Runner & Orchestrator
=========================================================
Robustness and Geometry Analysis of TDOA-Based Indoor Acoustic Source Localization

Executes:
  1. Pre-flight Smoke Test (if requested via --smoke-test)
  2. Part A: Environmental Robustness Study (42 conditions x 5 noise trials)
  3. Part A Visualizations (Figures A1–A9)
  4. Part B: Microphone Geometry Study (4 geometries x 5 conditions x 5 trials)
  5. Part B Visualizations (Figures B1–B8)
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.environment_robustness import run_environmental_robustness_study
from src.experiments.geometry_study import run_microphone_geometry_study
from src.visualization.robustness_plots import generate_all_robustness_figures
from src.visualization.geometry_plots import generate_all_geometry_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EchoLocate.Experiment02")


def main():
    parser = argparse.ArgumentParser(description="Run EchoLocate Experiment 2")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 3-condition smoke test")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "configs" / "experiment_02.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info("============================================================")
    logger.info("  EchoLocate — Experiment 02: Robustness and Geometry Analysis")
    logger.info("============================================================")

    out_part_a = PROJECT_ROOT / "results" / "experiment_02" / "environmental_robustness"
    out_part_b = PROJECT_ROOT / "results" / "experiment_02" / "geometry_study"

    if args.smoke_test:
        logger.info("Running PRE-FLIGHT SMOKE TEST...")
        smoke_out_a = PROJECT_ROOT / "results" / "experiment_02" / "smoke_test" / "environmental_robustness"
        smoke_out_b = PROJECT_ROOT / "results" / "experiment_02" / "smoke_test" / "geometry_study"
        
        # Smoke Test Part A
        df_sum_a, df_all_a, df_cal_a = run_environmental_robustness_study(cfg, smoke_out_a, smoke_test=True)
        generate_all_robustness_figures(df_sum_a, df_all_a, df_cal_a, smoke_out_a)
        
        # Smoke Test Part B
        df_sum_b, df_pre_b, df_rank_b = run_microphone_geometry_study(cfg, smoke_out_b, smoke_test=True)
        generate_all_geometry_figures(cfg, df_sum_b, df_pre_b, df_rank_b, smoke_out_b)
        
        logger.info("Smoke Test Completed Successfully without Errors or NaNs!")
        return

    # --------------------------------------------------------------------------
    # FULL EXPERIMENT 2 EXECUTION
    # --------------------------------------------------------------------------
    
    # PART A
    logger.info(">>> RUNNING PART A: ENVIRONMENTAL ROBUSTNESS STUDY (42 CONDITIONS x 5 TRIALS) <<<")
    df_summary_a, df_all_trials_a, df_rt60_calib = run_environmental_robustness_study(
        cfg=cfg,
        output_dir=out_part_a,
        smoke_test=False,
    )
    generate_all_robustness_figures(
        df_summary=df_summary_a,
        df_all_trials=df_all_trials_a,
        df_rt60_calib=df_rt60_calib,
        output_dir=out_part_a,
    )

    # PART B
    logger.info(">>> RUNNING PART B: MICROPHONE GEOMETRY STUDY (4 GEOMETRIES x 5 CONDITIONS x 5 TRIALS) <<<")
    df_geometry_summary, df_pre_analysis, df_rankings = run_microphone_geometry_study(
        cfg=cfg,
        output_dir=out_part_b,
        smoke_test=False,
    )
    generate_all_geometry_figures(
        cfg=cfg,
        df_geom_summary=df_geometry_summary,
        df_pre_analysis=df_pre_analysis,
        df_rankings=df_rankings,
        output_dir=out_part_b,
    )

    logger.info("============================================================")
    logger.info("EXPERIMENT 2 COMPLETE! All results and figures saved.")
    logger.info("Part A Summary: %s", out_part_a / "summary.csv")
    logger.info("Part B Summary: %s", out_part_b / "summary.csv")
    logger.info("============================================================")


if __name__ == "__main__":
    main()
