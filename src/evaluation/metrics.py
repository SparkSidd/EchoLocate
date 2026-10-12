"""
EchoLocate — Evaluation Metrics
==================================
Computes quantitative performance measures for:
  1. TDOA estimation accuracy (vs. theoretical geometric delays)
  2. Source localization accuracy (predicted vs. true position)

Metrics for localization:
    - Per-sample localization error (Euclidean distance)
    - Mean Localization Error (MLE)
    - Median Localization Error (MedLE)
    - RMSE (Root Mean Square Error of Euclidean distances)
    - Percentage of predictions within thresholds: 0.5 m, 1.0 m, 2.0 m

Metrics for TDOA estimation:
    - Theoretical TDOA (from geometry)
    - Estimated TDOA (from GCC-PHAT)
    - Absolute delay error (|estimated - theoretical|)
    - Error in metres equivalent: |Δτ| × c
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Localization accuracy thresholds (metres)
ACCURACY_THRESHOLDS = [0.5, 1.0, 2.0]


@dataclass
class TDOAAccuracyRow:
    """One row in the TDOA validation table."""

    pair_label: str      # e.g. "M1–M2"
    mic_i: int
    mic_j: int
    theoretical_tdoa_s: float    # seconds
    estimated_tdoa_s: float      # seconds
    abs_error_s: float           # |estimated - theoretical| in seconds
    abs_error_m: float           # |Δτ| × c in metres equivalent
    abs_error_samples: float     # |Δτ| × fs in samples


@dataclass
class LocalizationSummary:
    """Summary statistics for a set of localization experiments."""

    n_sources: int
    mean_error_m: float
    median_error_m: float
    rmse_m: float
    std_error_m: float
    min_error_m: float
    max_error_m: float
    pct_within_0_5m: float
    pct_within_1_0m: float
    pct_within_2_0m: float
    errors: np.ndarray = field(repr=False)  # All per-source errors


def compute_tdoa_accuracy(
    tdoa_results: dict[tuple[int, int], "TDOAResult"],  # noqa: F821
    simulation_result: "SimulationResult",              # noqa: F821
) -> list[TDOAAccuracyRow]:
    """
    Compare GCC-PHAT estimated TDOAs against theoretical geometric TDOAs.

    Parameters
    ----------
    tdoa_results : dict
        Keys: (i, j) pairs. Values: TDOAResult objects.
    simulation_result : SimulationResult
        Contains mic positions, source position, speed of sound.

    Returns
    -------
    list[TDOAAccuracyRow]
        One row per microphone pair.
    """
    c = simulation_result.speed_of_sound
    fs = simulation_result.fs
    rows = []

    for (i, j), result in sorted(tdoa_results.items()):
        theoretical = simulation_result.theoretical_tdoa(i, j)
        estimated = result.tau_seconds
        abs_err_s = abs(estimated - theoretical)
        abs_err_m = abs_err_s * c
        abs_err_samples = abs_err_s * fs

        row = TDOAAccuracyRow(
            pair_label=result.label,
            mic_i=i,
            mic_j=j,
            theoretical_tdoa_s=theoretical,
            estimated_tdoa_s=estimated,
            abs_error_s=abs_err_s,
            abs_error_m=abs_err_m,
            abs_error_samples=abs_err_samples,
        )
        rows.append(row)
        logger.info(
            "TDOA %s: theoretical=%+.6f s, estimated=%+.6f s, |error|=%.6f s (%.4f m, %.3f samples)",
            result.label, theoretical, estimated, abs_err_s, abs_err_m, abs_err_samples,
        )

    return rows


def tdoa_accuracy_to_dataframe(rows: list[TDOAAccuracyRow]) -> pd.DataFrame:
    """Convert TDOA accuracy rows to a pandas DataFrame for display."""
    data = []
    for r in rows:
        data.append({
            "Pair": r.pair_label,
            "Theoretical TDOA (s)": f"{r.theoretical_tdoa_s:+.6f}",
            "Estimated TDOA (s)": f"{r.estimated_tdoa_s:+.6f}",
            "Absolute Error (s)": f"{r.abs_error_s:.6f}",
            "Error (m equiv.)": f"{r.abs_error_m:.4f}",
            "Error (samples)": f"{r.abs_error_samples:.3f}",
        })
    return pd.DataFrame(data)


def compute_localization_errors(results: list[dict]) -> np.ndarray:
    """
    Extract localization errors from a list of result dicts.

    Parameters
    ----------
    results : list[dict]
        Each dict must contain 'localization_error' key (metres).

    Returns
    -------
    np.ndarray
        Array of errors in metres.
    """
    errors = np.array([r["localization_error"] for r in results if r["localization_error"] is not None])
    return errors


def compute_localization_summary(errors: np.ndarray) -> LocalizationSummary:
    """
    Compute summary statistics for a set of localization errors.

    Parameters
    ----------
    errors : np.ndarray
        1-D array of Euclidean localization errors in metres.

    Returns
    -------
    LocalizationSummary
    """
    n = len(errors)
    if n == 0:
        raise ValueError("Cannot compute summary: no errors provided.")

    mean_e = float(np.mean(errors))
    median_e = float(np.median(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    std_e = float(np.std(errors))
    min_e = float(np.min(errors))
    max_e = float(np.max(errors))

    pct_0_5 = 100.0 * float(np.mean(errors <= 0.5))
    pct_1_0 = 100.0 * float(np.mean(errors <= 1.0))
    pct_2_0 = 100.0 * float(np.mean(errors <= 2.0))

    logger.info(
        "Localization summary: n=%d, mean=%.4f m, median=%.4f m, RMSE=%.4f m",
        n, mean_e, median_e, rmse,
    )
    logger.info(
        "Within 0.5m: %.1f%%, Within 1.0m: %.1f%%, Within 2.0m: %.1f%%",
        pct_0_5, pct_1_0, pct_2_0,
    )

    return LocalizationSummary(
        n_sources=n,
        mean_error_m=mean_e,
        median_error_m=median_e,
        rmse_m=rmse,
        std_error_m=std_e,
        min_error_m=min_e,
        max_error_m=max_e,
        pct_within_0_5m=pct_0_5,
        pct_within_1_0m=pct_1_0,
        pct_within_2_0m=pct_2_0,
        errors=errors,
    )


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Convert a list of per-source result dicts to a summary DataFrame.

    Expected keys in each dict:
        source_id, true_x, true_y, pred_x, pred_y,
        localization_error, stage,
        tdoa_01, tdoa_02, tdoa_12 (optional).

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for r in results:
        row = {
            "Source ID": r.get("source_id", "?"),
            "True X (m)": f"{r.get('true_x', float('nan')):.3f}",
            "True Y (m)": f"{r.get('true_y', float('nan')):.3f}",
            "Predicted X (m)": f"{r.get('pred_x', float('nan')):.4f}",
            "Predicted Y (m)": f"{r.get('pred_y', float('nan')):.4f}",
            "Error (m)": f"{r.get('localization_error', float('nan')):.4f}",
            "Stage": r.get("stage", "?"),
        }
        # Add TDOAs if present
        for key in ["tdoa_01", "tdoa_02", "tdoa_12"]:
            if key in r:
                row[key] = f"{r[key]:+.6f}"
        rows.append(row)
    return pd.DataFrame(rows)


def print_tdoa_table(rows: list[TDOAAccuracyRow], stage: str = "") -> None:
    """Pretty-print the TDOA accuracy table."""
    header = f"\n{'='*70}\nTDOA Accuracy Table"
    if stage:
        header += f" — {stage.upper()}"
    print(header)
    print(f"{'='*70}")
    print(f"{'Pair':<10} {'Theoretical (s)':>16} {'Estimated (s)':>14} {'|Error| (s)':>12} {'|Error| (m)':>12} {'Samples':>8}")
    print(f"{'-'*70}")
    for r in rows:
        print(
            f"{r.pair_label:<10} "
            f"{r.theoretical_tdoa_s:>+16.6f} "
            f"{r.estimated_tdoa_s:>+14.6f} "
            f"{r.abs_error_s:>12.6f} "
            f"{r.abs_error_m:>12.4f} "
            f"{r.abs_error_samples:>8.3f}"
        )
    print(f"{'='*70}\n")


def print_localization_summary(summary: LocalizationSummary, stage: str = "") -> None:
    """Pretty-print the localization summary."""
    print(f"\n{'='*50}")
    print(f"Localization Summary" + (f" — {stage.upper()}" if stage else ""))
    print(f"{'='*50}")
    print(f"  N sources:        {summary.n_sources}")
    print(f"  Mean error:       {summary.mean_error_m:.4f} m")
    print(f"  Median error:     {summary.median_error_m:.4f} m")
    print(f"  RMSE:             {summary.rmse_m:.4f} m")
    print(f"  Std error:        {summary.std_error_m:.4f} m")
    print(f"  Min error:        {summary.min_error_m:.4f} m")
    print(f"  Max error:        {summary.max_error_m:.4f} m")
    print(f"  Within 0.5 m:     {summary.pct_within_0_5m:.1f}%")
    print(f"  Within 1.0 m:     {summary.pct_within_1_0m:.1f}%")
    print(f"  Within 2.0 m:     {summary.pct_within_2_0m:.1f}%")
    print(f"{'='*50}\n")
