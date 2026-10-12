"""
EchoLocate — Run Script: Experiment 2 Part A (Environmental Robustness)
========================================================================
Executes Part A 42-condition grid study and generates Figures A1–A9.
"""

import logging
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.environment_robustness import run_environmental_robustness_study
from src.visualization.robustness_plots import generate_all_robustness_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EchoLocate.Exp02A")


def main():
    config_path = PROJECT_ROOT / "configs" / "experiment_02.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    output_dir = PROJECT_ROOT / "results" / "experiment_02" / "environmental_robustness"
    
    # Run Part A
    df_summary, df_all_trials, df_rt60_calib = run_environmental_robustness_study(
        cfg=cfg,
        output_dir=output_dir,
        smoke_test=False,
    )
    
    # Generate Visualizations
    generate_all_robustness_figures(
        df_summary=df_summary,
        df_all_trials=df_all_trials,
        df_rt60_calib=df_rt60_calib,
        output_dir=output_dir,
    )
    logger.info("Experiment 2 Part A Complete!")


if __name__ == "__main__":
    main()
