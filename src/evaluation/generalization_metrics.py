"""
EchoLocate — Generalization Metrics Module
==========================================
Summarizes localization performance across diverse signal categories,
unseen room geometries, unseen source locations, and microphone channel variations.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_signal_generalization_summary(
    df_evals: pd.DataFrame,
    accuracy_thresholds: list[float] = [0.25, 0.50, 1.0, 2.0],
    catastrophic_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Summarize performance across different source signal types.
    """
    records = []
    for (sig_type, cond_id), group in df_evals.groupby(["signal_type", "condition_id"], sort=True):
        errs = group["localization_error_m"].values
        mean_err = float(np.mean(errs))
        med_err = float(np.median(errs))
        rmse = float(np.sqrt(np.mean(errs**2)))
        cat_rate = float(np.mean(errs > catastrophic_threshold) * 100.0)

        succ = {
            f"pct_within_{str(t).replace('.', '_')}m": float(np.mean(errs <= t) * 100.0)
            for t in accuracy_thresholds
        }

        records.append({
            "signal_type": sig_type,
            "condition_id": cond_id,
            "total_evaluations": len(group),
            "mean_error_m": mean_err,
            "median_error_m": med_err,
            "rmse_m": rmse,
            "catastrophic_rate_pct": cat_rate,
            **succ,
        })

    return pd.DataFrame(records)


def compute_room_generalization_summary(
    df_evals: pd.DataFrame,
    accuracy_thresholds: list[float] = [0.25, 0.50, 1.0, 2.0],
    catastrophic_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Summarize performance across different room topologies.
    """
    records = []
    for (room_id, cond_id), group in df_evals.groupby(["room_id", "condition_id"], sort=True):
        errs = group["localization_error_m"].values
        mean_err = float(np.mean(errs))
        med_err = float(np.median(errs))
        rmse = float(np.sqrt(np.mean(errs**2)))
        cat_rate = float(np.mean(errs > catastrophic_threshold) * 100.0)

        succ = {
            f"pct_within_{str(t).replace('.', '_')}m": float(np.mean(errs <= t) * 100.0)
            for t in accuracy_thresholds
        }

        records.append({
            "room_id": room_id,
            "condition_id": cond_id,
            "total_evaluations": len(group),
            "mean_error_m": mean_err,
            "median_error_m": med_err,
            "rmse_m": rmse,
            "catastrophic_rate_pct": cat_rate,
            **succ,
        })

    return pd.DataFrame(records)
