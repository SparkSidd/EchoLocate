"""
EchoLocate — Run Script: Experiment 2 Part B (Microphone Geometry Study)
========================================================================
Executes Part B 4-geometry study and generates Figures B1–B8.
"""

import logging
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.geometry_study import run_microphone_geometry_study
from src.visualization.geometry_plots import generate_all_geometry_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EchoLocate.Exp02B")


def main():
    config_path = PROJECT_ROOT / "configs" / "experiment_02.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    output_dir = PROJECT_ROOT / "results" / "experiment_02" / "geometry_study"
    
    # Run Part B
    df_geometry_summary, df_pre_analysis, df_rankings = run_microphone_geometry_study(
        cfg=cfg,
        output_dir=output_dir,
        smoke_test=False,
    )
    
    # Generate Visualizations
    generate_all_geometry_figures(
        cfg=cfg,
        df_geom_summary=df_geometry_summary,
        df_pre_analysis=df_pre_analysis,
        df_rankings=df_rankings,
        output_dir=output_dir,
    )
    logger.info("Experiment 2 Part B Complete!")


if __name__ == "__main__":
    main()
