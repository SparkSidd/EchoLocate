"""
EchoLocate — Real-World Multi-Room & Multi-Signal Evaluation Module
===================================================================
Executes real-world benchmark evaluations, temporal stability analysis,
and selective prediction abstention sweeps across coverage budgets.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_selective_prediction_curve(
    confidences: list[float] | np.ndarray,
    errors_m: list[float] | np.ndarray,
    coverage_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate mean/median error and catastrophic rate at specified coverage levels."""
    if coverage_levels is None:
        coverage_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    confs = np.asarray(confidences)
    errs = np.asarray(errors_m)
    records = []

    for cov in coverage_levels:
        if cov >= 1.0:
            sub_errs = errs
        else:
            pct_thresh = (1.0 - cov) * 100.0
            thresh = float(np.percentile(confs, pct_thresh))
            mask = confs >= thresh
            sub_errs = errs[mask] if np.any(mask) else errs

        records.append({
            "target_coverage": cov,
            "actual_coverage": float(len(sub_errs) / max(1, len(errs))),
            "retained_mean_error_m": float(np.mean(sub_errs)),
            "retained_median_error_m": float(np.median(sub_errs)),
            "retained_rmse_m": float(np.sqrt(np.mean(sub_errs**2))),
            "catastrophic_rate_pct": float(np.mean(sub_errs > 2.0) * 100.0),
            "success_rate_le_05m_pct": float(np.mean(sub_errs <= 0.50) * 100.0),
        })

    return pd.DataFrame(records)


def compute_temporal_stability_metrics(
    trajectory_pred_xy: list[tuple[float, float]] | np.ndarray,
    trajectory_true_xy: list[tuple[float, float]] | np.ndarray,
    jump_threshold_m: float = 1.0,
) -> dict[str, float]:
    """Compute frame-to-frame position jitter, trajectory RMSE, and sudden jump rate."""
    pred_arr = np.asarray(trajectory_pred_xy)
    true_arr = np.asarray(trajectory_true_xy)

    if len(pred_arr) < 2:
        return {"trajectory_rmse_m": 0.0, "mean_frame_jitter_m": 0.0, "sudden_jump_rate_pct": 0.0}

    # Tracking error
    errs = np.linalg.norm(pred_arr - true_arr, axis=1)
    rmse = float(np.sqrt(np.mean(errs**2)))

    # Frame-to-frame displacement
    diffs = np.linalg.norm(np.diff(pred_arr, axis=0), axis=1)
    mean_jitter = float(np.mean(diffs))
    jumps = int(np.sum(diffs > jump_threshold_m))
    jump_rate = float(jumps / max(1, len(diffs)) * 100.0)

    return {
        "trajectory_rmse_m": rmse,
        "mean_frame_jitter_m": mean_jitter,
        "sudden_jump_rate_pct": jump_rate,
    }
