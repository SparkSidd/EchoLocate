"""
EchoLocate — Evaluation: Failure Classification & Mechanism Analysis
======================================================================
Provides fine-grained failure taxonomy, per-source sensitivity rankings,
and TDOA vs. geometric localization error decoupling.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FailureClass(str, Enum):
    VALID_ACCURATE = "valid_accurate"          # error <= 0.5m
    VALID_MODERATE = "valid_moderate"          # 0.5m < error <= 1.0m
    VALID_DEGRADED = "valid_degraded"          # 1.0m < error <= 2.0m
    CATASTROPHIC_ERROR = "catastrophic_error"  # error > 2.0m
    SOLVER_FAILED = "solver_failed"            # NLS failed convergence
    NAN_INF = "nan_inf"                        # NaN or Inf values
    OUT_OF_BOUNDS = "out_of_bounds"            # Predicted outside room


def classify_evaluation_record(
    pred_x: float,
    pred_y: float,
    loc_error: float,
    converged: bool,
    room_dims: list[float] = [10.0, 6.0, 3.0],
    catastrophic_thresh: float = 2.0,
) -> FailureClass:
    """Classify an individual evaluation record into mutually exclusive failure categories."""
    if not np.isfinite(pred_x) or not np.isfinite(pred_y) or not np.isfinite(loc_error):
        return FailureClass.NAN_INF
    
    if pred_x < -0.1 or pred_x > room_dims[0] + 0.1 or pred_y < -0.1 or pred_y > room_dims[1] + 0.1:
        return FailureClass.OUT_OF_BOUNDS
        
    if not converged:
        return FailureClass.SOLVER_FAILED
        
    if loc_error > catastrophic_thresh:
        return FailureClass.CATASTROPHIC_ERROR
    elif loc_error > 1.0:
        return FailureClass.VALID_DEGRADED
    elif loc_error > 0.5:
        return FailureClass.VALID_MODERATE
    else:
        return FailureClass.VALID_ACCURATE


def compute_per_source_sensitivity(
    df_evals: pd.DataFrame,
    df_summary: pd.DataFrame,
    catastrophic_thresh: float = 2.0,
) -> pd.DataFrame:
    """
    Evaluate robustness and degradation profile per source position.

    Divides conditions into:
      - Low RT60 (lowest 2 measured conditions)
      - Transition RT60 (middle 3 measured conditions)
      - High RT60 (highest 2 measured conditions)
    """
    cond_col = "target_rt60_s" if "target_rt60_s" in df_summary.columns else "rt60_target_s"
    conds = df_summary.sort_values("rt60_measured_mean_s")[cond_col].tolist()
    n_conds = len(conds)
    
    if n_conds >= 5:
        low_tgts = conds[:2]
        trans_tgts = conds[2:n_conds-2]
        high_tgts = conds[n_conds-2:]
    else:
        low_tgts = conds[:1]
        trans_tgts = conds[1:n_conds-1]
        high_tgts = conds[n_conds-1:]

    source_records = []
    for src_id, group in df_evals.groupby("source_id"):
        true_x = float(group["true_x"].iloc[0])
        true_y = float(group["true_y"].iloc[0])
        
        eval_tgt_col = "target_rt60_s" if "target_rt60_s" in group.columns else "rt60_target_s"
        low_err = group[group[eval_tgt_col].isin(low_tgts)]["localization_error"].mean()
        trans_err = group[group[eval_tgt_col].isin(trans_tgts)]["localization_error"].mean()
        high_err = group[group[eval_tgt_col].isin(high_tgts)]["localization_error"].mean()
        
        overall_mean = group["localization_error"].mean()
        overall_std = group["localization_error"].std()
        
        cat_count = (group["localization_error"] > catastrophic_thresh).sum()
        cat_rate = float(cat_count / len(group) * 100.0)
        
        tdoa_err = group["mean_tdoa_error_ms"].mean() if "mean_tdoa_error_ms" in group.columns else 0.0
        
        # Relative sensitivity: how much performance explodes during transition vs baseline
        sensitivity = float(trans_err / max(1e-4, low_err))
        
        source_records.append({
            "source_id": src_id,
            "true_x": true_x,
            "true_y": true_y,
            "mean_error_low_rt60_m": float(low_err),
            "mean_error_transition_rt60_m": float(trans_err),
            "mean_error_high_rt60_m": float(high_err),
            "overall_mean_error_m": float(overall_mean),
            "overall_std_error_m": float(overall_std),
            "catastrophic_failure_rate_pct": cat_rate,
            "mean_tdoa_error_ms": float(tdoa_err),
            "relative_sensitivity_index": sensitivity,
        })
        
    df_src = pd.DataFrame(source_records).sort_values("overall_mean_error_m").reset_index(drop=True)
    return df_src


def analyze_failure_mechanisms(
    df_summary: pd.DataFrame,
    speed_of_sound: float = 343.0,
) -> pd.DataFrame:
    """
    Decouple TDOA estimation failure from geometric multilateration amplification.
    """
    df = df_summary.copy().sort_values("rt60_measured_mean_s").reset_index(drop=True)
    
    mech_records = []
    for _, row in df.iterrows():
        tdoa_rmse_s = row["rmse_tdoa_error_ms"] / 1000.0
        loc_rmse_m = row["rmse_localization_error_m"]
        
        # Equivalent range error (metres)
        range_err_m = tdoa_rmse_s * speed_of_sound
        
        # Amplification factor = Localization RMSE / Range RMSE
        amp_factor = loc_rmse_m / max(1e-4, range_err_m)
        
        # Failure classification
        if row["mean_tdoa_error_ms"] > 2.0:
            primary_fail = "Acoustic TDOA Failure (Peak Hopping)"
        elif amp_factor > 3.0:
            primary_fail = "Geometric Amplification"
        else:
            primary_fail = "Combined / Mild Degradation"
            
        mech_records.append({
            "target_rt60_s": row["rt60_target_s"],
            "measured_mean_rt60_s": row["rt60_measured_mean_s"],
            "mean_tdoa_error_ms": row["mean_tdoa_error_ms"],
            "rmse_tdoa_error_ms": row["rmse_tdoa_error_ms"],
            "mean_localization_error_m": row["mean_localization_error_m"],
            "rmse_localization_error_m": loc_rmse_m,
            "range_error_rmse_m": range_err_m,
            "geometric_amplification_factor": amp_factor,
            "catastrophic_failure_rate_pct": row["pct_catastrophic_failure"],
            "primary_failure_mode": primary_fail,
        })
        
    return pd.DataFrame(mech_records)
