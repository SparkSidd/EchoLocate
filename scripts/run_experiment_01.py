"""
EchoLocate — Experiment 01 Main Script
========================================
Acoustic Event Localization — Baseline TDOA Pipeline

This script executes the complete Experiment 1 pipeline:

  Stage A (Anechoic):
    1. Generate source signal (chirp)
    2. Simulate room with direct-path only (no reflections)
    3. Verify microphone signals (visual + numerical)
    4. Estimate TDOAs via GCC-PHAT
    5. Compare estimated vs. theoretical TDOAs (sanity check)
    6. Localize single source
    7. Localize 15 distributed source positions
    8. Compute metrics (MLE, MedLE, RMSE, accuracy thresholds)

  Stage B (Reverberant):
    Same pipeline with realistic room reflections (RT60 = 0.4 s)

Outputs:
    results/experiment_01/figures/          — 6+ figures (PNG)
    results/experiment_01/multi_source_results_anechoic.csv
    results/experiment_01/multi_source_results_reverberant.csv

Usage:
    python scripts/run_experiment_01.py

The script is designed to be run from the EchoLocate project root.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# ─── Add project root to sys.path so 'src' is importable ─────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.room_simulator import run_simulation, build_mic_array_3d, SimulationResult
from src.simulation.source_generator import generate_source_signal
from src.localization.gcc_phat import (
    estimate_all_tdoas,
    compute_max_tau_from_geometry,
    TDOAResult,
)
from src.localization.multilateration import localize_with_multiple_starts
from src.evaluation.metrics import (
    compute_tdoa_accuracy,
    compute_localization_summary,
    print_tdoa_table,
    print_localization_summary,
    tdoa_accuracy_to_dataframe,
    results_to_dataframe,
)
from src.visualization.plots import (
    plot_room_layout,
    plot_waveforms,
    plot_waveforms_zoomed,
    plot_gcc_phat,
    plot_localization_single,
    plot_localization_multi,
    plot_error_comparison,
)

# ─── Logging configuration ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EchoLocate.Experiment01")


# =============================================================================
# Utility helpers
# =============================================================================

def load_config(config_path: str | Path) -> dict:
    """Load and return the YAML experiment configuration."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def make_flat_config(cfg: dict) -> dict:
    """
    Create a flat config dict combining room and source sections,
    suitable for passing to run_simulation() and generate_source_signal().
    """
    flat = {
        "sampling_rate": cfg["room"]["sampling_rate"],
        "source": cfg["source"],
    }
    return flat


def ensure_dirs(cfg: dict) -> None:
    """Create output directories if they don't exist."""
    for key in ["results_dir", "figures_dir", "data_dir"]:
        path = PROJECT_ROOT / cfg["output"][key]
        path.mkdir(parents=True, exist_ok=True)


def save_csv(results: list[dict], path: Path) -> None:
    """Save results list as CSV."""
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    logger.info("CSV saved: %s", path)


# =============================================================================
# Core pipeline: single source
# =============================================================================

def run_single_source_pipeline(
    cfg: dict,
    source_position: list[float],
    source_signal: np.ndarray,
    stage: str,
    mic_labels: list[str],
    mic_positions_2d: np.ndarray,
    figures_dir: Path,
    source_id: str = "single",
    save_figures: bool = True,
) -> dict[str, Any]:
    """
    Full pipeline for one source position and one stage.

    Steps:
    1. Simulate room
    2. Estimate TDOAs via GCC-PHAT
    3. Compute TDOA accuracy vs. theoretical
    4. Localize source

    Returns a result dict with all metrics.
    """
    logger.info("=" * 60)
    logger.info("Pipeline: source=%s, stage=%s", source_position, stage)

    # ── Step 1: Simulate ─────────────────────────────────────────────────────
    sim: SimulationResult = run_simulation(
        config=cfg,
        source_position_2d=source_position,
        source_signal=source_signal,
        stage=stage,
    )

    # ── Step 2: Compute max tau from geometry ────────────────────────────────
    max_tau = compute_max_tau_from_geometry(
        mic_positions_2d=mic_positions_2d,
        speed_of_sound=cfg["room"]["speed_of_sound"],
        fs=cfg["room"]["sampling_rate"],
        safety_factor=1.3,
    )

    # ── Step 3: Estimate TDOAs ───────────────────────────────────────────────
    tdoa_results = estimate_all_tdoas(
        mic_signals=sim.mic_signals,
        fs=sim.fs,
        max_tau_samples=max_tau,
    )

    # ── Step 4: TDOA accuracy ────────────────────────────────────────────────
    tdoa_accuracy_rows = compute_tdoa_accuracy(tdoa_results, sim)
    print_tdoa_table(tdoa_accuracy_rows, stage=stage)

    # ── Step 5: Localise ─────────────────────────────────────────────────────
    tdoa_for_solver = {pair: result.tau_seconds for pair, result in tdoa_results.items()}
    loc_result = localize_with_multiple_starts(
        mic_positions_2d=mic_positions_2d,
        tdoa_dict=tdoa_for_solver,
        speed_of_sound=cfg["room"]["speed_of_sound"],
        room_dimensions=cfg["room"]["dimensions"],
        n_restarts=8,
        rng_seed=cfg["experiment"]["random_seed"],
        max_iterations=cfg["localization"]["max_iterations"],
        tolerance=cfg["localization"]["tolerance"],
        true_position=np.array(source_position),
    )

    logger.info(
        "Source %s: true=(%.3f, %.3f), pred=(%.4f, %.4f), error=%.4f m",
        source_id,
        source_position[0], source_position[1],
        loc_result.predicted_position[0], loc_result.predicted_position[1],
        loc_result.localization_error or float("nan"),
    )

    # ── Step 6: Save figures (only for single-source reference case) ─────────
    if save_figures:
        # Figure 1 — Room Layout
        plot_room_layout(
            room_dimensions=cfg["room"]["dimensions"],
            mic_positions_2d=mic_positions_2d,
            mic_labels=mic_labels,
            source_position=np.array(source_position),
            source_label="Source",
            title=f"Room Layout — {stage.capitalize()}",
            save_path=figures_dir / f"fig1_room_layout_{stage}.png",
            dpi=cfg["figures"]["dpi"],
        )

        # Figure 2 — Full waveforms
        plot_waveforms(
            mic_signals=sim.mic_signals,
            fs=sim.fs,
            mic_labels=mic_labels,
            title=f"Received Waveforms — {stage.capitalize()}",
            save_path=figures_dir / f"fig2_waveforms_{stage}.png",
            dpi=cfg["figures"]["dpi"],
        )

        # Figure 3 — Zoomed waveforms
        plot_waveforms_zoomed(
            mic_signals=sim.mic_signals,
            fs=sim.fs,
            mic_labels=mic_labels,
            title=f"Zoomed Waveforms — {stage.capitalize()}",
            save_path=figures_dir / f"fig3_waveforms_zoom_{stage}.png",
            dpi=cfg["figures"]["dpi"],
        )

        # Figure 4 — GCC-PHAT responses
        plot_gcc_phat(
            tdoa_results=tdoa_results,
            speed_of_sound=cfg["room"]["speed_of_sound"],
            room_dimensions=cfg["room"]["dimensions"],
            title=f"GCC-PHAT Responses — {stage.capitalize()}",
            save_path=figures_dir / f"fig4_gcc_phat_{stage}.png",
            dpi=cfg["figures"]["dpi"],
        )

        # Figure 5 — Single-source localization
        plot_localization_single(
            room_dimensions=cfg["room"]["dimensions"],
            mic_positions_2d=mic_positions_2d,
            mic_labels=mic_labels,
            true_position=np.array(source_position),
            predicted_position=loc_result.predicted_position,
            stage=stage,
            title="Single-Source Localization Result",
            save_path=figures_dir / f"fig5_localization_single_{stage}.png",
            dpi=cfg["figures"]["dpi"],
        )

    # Compile result dict
    result = {
        "source_id": source_id,
        "stage": stage,
        "true_x": source_position[0],
        "true_y": source_position[1],
        "pred_x": float(loc_result.predicted_position[0]),
        "pred_y": float(loc_result.predicted_position[1]),
        "localization_error": loc_result.localization_error,
        "converged": loc_result.converged,
    }

    # Add TDOA values
    for (i, j), tdoa_r in tdoa_results.items():
        result[f"tdoa_{i}{j}_estimated_s"] = tdoa_r.tau_seconds
        result[f"tdoa_{i}{j}_theoretical_s"] = sim.theoretical_tdoa(i, j)

    return result


# =============================================================================
# Multi-source experiment
# =============================================================================

def run_multi_source_experiment(
    cfg: dict,
    source_positions: list[list[float]],
    source_signal: np.ndarray,
    stage: str,
    mic_labels: list[str],
    mic_positions_2d: np.ndarray,
    figures_dir: Path,
) -> list[dict]:
    """
    Run the full pipeline over multiple source positions.

    Returns a list of result dicts (one per source position).
    """
    logger.info("=" * 60)
    logger.info("Multi-source experiment: %d positions, stage=%s", len(source_positions), stage)

    all_results = []

    for idx, pos in enumerate(source_positions):
        src_id = f"S{idx+1:02d}"
        result = run_single_source_pipeline(
            cfg=cfg,
            source_position=pos,
            source_signal=source_signal,
            stage=stage,
            mic_labels=mic_labels,
            mic_positions_2d=mic_positions_2d,
            figures_dir=figures_dir,
            source_id=src_id,
            save_figures=False,  # Only save figures for single-source reference
        )
        all_results.append(result)

    return all_results


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║          EchoLocate — Experiment 01                  ║")
    logger.info("║  Acoustic Event Localization — Baseline TDOA Pipeline║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    # ── Load config ─────────────────────────────────────────────────────────
    config_path = PROJECT_ROOT / "configs" / "experiment_01.yaml"
    cfg = load_config(config_path)
    logger.info("Config loaded from: %s", config_path)

    # Set random seed for reproducibility
    np.random.seed(cfg["experiment"]["random_seed"])

    # ── Ensure output directories ─────────────────────────────────────────
    ensure_dirs(cfg)
    results_dir = PROJECT_ROOT / cfg["output"]["results_dir"]
    figures_dir = PROJECT_ROOT / cfg["output"]["figures_dir"]

    # ── Build mic array ──────────────────────────────────────────────────
    mic_3d, mic_labels = build_mic_array_3d(
        positions_2d=cfg["microphones"]["positions"],
        height=cfg["microphones"]["height"],
    )
    mic_positions_2d = mic_3d[:2, :].T  # (n_mics, 2)

    logger.info("Microphone array:")
    for label, pos in zip(mic_labels, mic_positions_2d):
        logger.info("  %s: (%.2f, %.2f)", label, pos[0], pos[1])

    # ── Generate source signal ────────────────────────────────────────────
    flat_cfg = make_flat_config(cfg)
    source_signal, fs = generate_source_signal(flat_cfg)
    logger.info(
        "Source signal: %s, %d samples (%.3f s) at %d Hz",
        cfg["source"]["type"], len(source_signal), len(source_signal) / fs, fs,
    )

    # Print overall geometry summary
    logger.info("\n--- Room Geometry ---")
    logger.info("Room: %.1f m × %.1f m × %.1f m", *cfg["room"]["dimensions"])
    logger.info("Speed of sound: %.1f m/s", cfg["room"]["speed_of_sound"])
    logger.info("Sampling rate: %d Hz", cfg["room"]["sampling_rate"])

    # =========================================================================
    # STAGE A — ANECHOIC
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE A: Anechoic (Direct Path Only)")
    logger.info("=" * 60)

    single_pos = cfg["single_source"]["position"]

    # Single-source run with figures
    single_result_a = run_single_source_pipeline(
        cfg=cfg,
        source_position=single_pos,
        source_signal=source_signal,
        stage="anechoic",
        mic_labels=mic_labels,
        mic_positions_2d=mic_positions_2d,
        figures_dir=figures_dir,
        source_id="single",
        save_figures=True,
    )

    logger.info("\nSingle-source anechoic result:")
    logger.info("  True:      (%.3f, %.3f)", single_result_a["true_x"], single_result_a["true_y"])
    logger.info("  Predicted: (%.4f, %.4f)", single_result_a["pred_x"], single_result_a["pred_y"])
    logger.info("  Error:     %.4f m", single_result_a["localization_error"])

    # Multi-source anechoic
    source_positions = cfg["multi_source"]["positions"]
    multi_results_a = run_multi_source_experiment(
        cfg=cfg,
        source_positions=source_positions,
        source_signal=source_signal,
        stage="anechoic",
        mic_labels=mic_labels,
        mic_positions_2d=mic_positions_2d,
        figures_dir=figures_dir,
    )

    errors_a = np.array([r["localization_error"] for r in multi_results_a])
    summary_a = compute_localization_summary(errors_a)
    print_localization_summary(summary_a, stage="anechoic")

    # Figure 6 — Multi-source anechoic
    true_positions_a = np.array([[r["true_x"], r["true_y"]] for r in multi_results_a])
    pred_positions_a = np.array([[r["pred_x"], r["pred_y"]] for r in multi_results_a])

    plot_localization_multi(
        room_dimensions=cfg["room"]["dimensions"],
        mic_positions_2d=mic_positions_2d,
        mic_labels=mic_labels,
        true_positions=true_positions_a,
        predicted_positions=pred_positions_a,
        errors=errors_a,
        stage="anechoic",
        title="Multi-Source Localization Results",
        save_path=figures_dir / "fig6_localization_multi_anechoic.png",
        dpi=cfg["figures"]["dpi"],
    )

    # Save CSV
    save_csv(multi_results_a, results_dir / "multi_source_results_anechoic.csv")

    # =========================================================================
    # STAGE B — REVERBERANT
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE B: Reverberant (RT60 = %.1f s)", cfg["stage_b"]["rt60"])
    logger.info("=" * 60)

    # Single-source reverberant (figures)
    single_result_b = run_single_source_pipeline(
        cfg=cfg,
        source_position=single_pos,
        source_signal=source_signal,
        stage="reverberant",
        mic_labels=mic_labels,
        mic_positions_2d=mic_positions_2d,
        figures_dir=figures_dir,
        source_id="single",
        save_figures=True,
    )

    logger.info("\nSingle-source reverberant result:")
    logger.info("  True:      (%.3f, %.3f)", single_result_b["true_x"], single_result_b["true_y"])
    logger.info("  Predicted: (%.4f, %.4f)", single_result_b["pred_x"], single_result_b["pred_y"])
    logger.info("  Error:     %.4f m", single_result_b["localization_error"])

    # Multi-source reverberant
    multi_results_b = run_multi_source_experiment(
        cfg=cfg,
        source_positions=source_positions,
        source_signal=source_signal,
        stage="reverberant",
        mic_labels=mic_labels,
        mic_positions_2d=mic_positions_2d,
        figures_dir=figures_dir,
    )

    errors_b = np.array([r["localization_error"] for r in multi_results_b])
    summary_b = compute_localization_summary(errors_b)
    print_localization_summary(summary_b, stage="reverberant")

    # Figure 6 — Multi-source reverberant
    true_positions_b = np.array([[r["true_x"], r["true_y"]] for r in multi_results_b])
    pred_positions_b = np.array([[r["pred_x"], r["pred_y"]] for r in multi_results_b])

    plot_localization_multi(
        room_dimensions=cfg["room"]["dimensions"],
        mic_positions_2d=mic_positions_2d,
        mic_labels=mic_labels,
        true_positions=true_positions_b,
        predicted_positions=pred_positions_b,
        errors=errors_b,
        stage="reverberant",
        title="Multi-Source Localization Results",
        save_path=figures_dir / "fig6_localization_multi_reverberant.png",
        dpi=cfg["figures"]["dpi"],
    )

    # Bonus: comparison figure
    plot_error_comparison(
        errors_anechoic=errors_a,
        errors_reverberant=errors_b,
        save_path=figures_dir / "fig7_error_comparison.png",
        dpi=cfg["figures"]["dpi"],
    )

    # Save CSV
    save_csv(multi_results_b, results_dir / "multi_source_results_reverberant.csv")

    # =========================================================================
    # Final Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 01 — FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info("Stage A (Anechoic):")
    logger.info("  Mean Error:   %.4f m", summary_a.mean_error_m)
    logger.info("  Median Error: %.4f m", summary_a.median_error_m)
    logger.info("  RMSE:         %.4f m", summary_a.rmse_m)
    logger.info("  Within 0.5m:  %.1f%%", summary_a.pct_within_0_5m)
    logger.info("  Within 1.0m:  %.1f%%", summary_a.pct_within_1_0m)
    logger.info("  Within 2.0m:  %.1f%%", summary_a.pct_within_2_0m)
    logger.info("Stage B (Reverberant):")
    logger.info("  Mean Error:   %.4f m", summary_b.mean_error_m)
    logger.info("  Median Error: %.4f m", summary_b.median_error_m)
    logger.info("  RMSE:         %.4f m", summary_b.rmse_m)
    logger.info("  Within 0.5m:  %.1f%%", summary_b.pct_within_0_5m)
    logger.info("  Within 1.0m:  %.1f%%", summary_b.pct_within_1_0m)
    logger.info("  Within 2.0m:  %.1f%%", summary_b.pct_within_2_0m)
    logger.info("\nResults directory: %s", results_dir)
    logger.info("Figures directory: %s", figures_dir)
    logger.info("=" * 60)
    logger.info("Experiment 1 is complete only if the code was actually executed")
    logger.info("and the results were generated.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
