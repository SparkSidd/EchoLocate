"""
EchoLocate — Evaluation: Reverberation Threshold & Change-Point Analysis
========================================================================
Performs exploratory change-point analysis, performance slope estimation,
and operational region segmentation based on measured Schroeder RT60.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TransitionThresholdResults:
    """Summary of threshold and transition region analysis."""
    conservative_operating_upper_rt60_s: float
    transition_start_rt60_s: float
    transition_end_rt60_s: float
    failure_region_lower_rt60_s: float
    max_error_slope_measured_rt60_s: float
    max_error_slope_value: float
    max_success_drop_slope_measured_rt60_s: float
    max_success_drop_slope_value: float
    is_sharp_cliff_edge: bool
    interpretation_note: str


def compute_performance_slopes(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate discrete performance slopes between adjacent measured RT60 conditions.

    Parameters
    ----------
    df_summary : pd.DataFrame
        Summary dataframe containing 'rt60_measured_mean_s', 'mean_localization_error_m',
        'pct_within_1_0m', and 'pct_catastrophic_failure'.

    Returns
    -------
    pd.DataFrame
        Dataframe sorted by measured RT60 with slope columns:
        - 'd_mean_error_d_rt60' (m/s)
        - 'd_success_rate_d_rt60' (%/s)
        - 'd_catastrophic_rate_d_rt60' (%/s)
    """
    df = df_summary.copy().sort_values("rt60_measured_mean_s").reset_index(drop=True)
    
    slopes = []
    for i in range(len(df)):
        if i == 0:
            slopes.append({
                "condition_idx": i,
                "target_rt60_s": df.loc[i, "rt60_target_s"],
                "measured_mean_rt60_s": df.loc[i, "rt60_measured_mean_s"],
                "mean_localization_error_m": df.loc[i, "mean_localization_error_m"],
                "pct_within_1_0m": df.loc[i, "pct_within_1_0m"],
                "pct_catastrophic_failure": df.loc[i, "pct_catastrophic_failure"],
                "delta_measured_rt60_s": 0.0,
                "d_mean_error_d_rt60": 0.0,
                "d_success_rate_d_rt60": 0.0,
                "d_catastrophic_rate_d_rt60": 0.0,
            })
        else:
            dt = df.loc[i, "rt60_measured_mean_s"] - df.loc[i - 1, "rt60_measured_mean_s"]
            if dt <= 1e-6:
                dt = 1e-6
            de = df.loc[i, "mean_localization_error_m"] - df.loc[i - 1, "mean_localization_error_m"]
            ds = df.loc[i, "pct_within_1_0m"] - df.loc[i - 1, "pct_within_1_0m"]
            dc = df.loc[i, "pct_catastrophic_failure"] - df.loc[i - 1, "pct_catastrophic_failure"]
            
            slopes.append({
                "condition_idx": i,
                "target_rt60_s": df.loc[i, "rt60_target_s"],
                "measured_mean_rt60_s": df.loc[i, "rt60_measured_mean_s"],
                "mean_localization_error_m": df.loc[i, "mean_localization_error_m"],
                "pct_within_1_0m": df.loc[i, "pct_within_1_0m"],
                "pct_catastrophic_failure": df.loc[i, "pct_catastrophic_failure"],
                "delta_measured_rt60_s": dt,
                "d_mean_error_d_rt60": de / dt,
                "d_success_rate_d_rt60": ds / dt,
                "d_catastrophic_rate_d_rt60": dc / dt,
            })
            
    return pd.DataFrame(slopes)


def analyze_reverberation_thresholds(
    df_summary: pd.DataFrame,
    success_threshold_pct: float = 80.0,
    catastrophic_limit_pct: float = 10.0,
    failure_success_floor_pct: float = 30.0,
    failure_catastrophic_floor_pct: float = 40.0,
) -> tuple[TransitionThresholdResults, pd.DataFrame]:
    """
    Segment measured RT60 space into:
      1. Conservative Operating Region (high success, rare catastrophic failures)
      2. Transition Interval (degrading performance slope)
      3. Failure Region (consistently low success, high catastrophic rate)

    Returns
    -------
    tuple[TransitionThresholdResults, pd.DataFrame]
        Summary result dataclass and detailed per-interval slope dataframe.
    """
    df_slopes = compute_performance_slopes(df_summary)
    
    # 1. Conservative operating upper bound
    stable_mask = (df_slopes["pct_within_1_0m"] >= success_threshold_pct) & (
        df_slopes["pct_catastrophic_failure"] <= catastrophic_limit_pct
    )
    if stable_mask.any():
        conservative_upper = float(df_slopes.loc[stable_mask, "measured_mean_rt60_s"].max())
    else:
        conservative_upper = float(df_slopes["measured_mean_rt60_s"].min())
        
    # 2. Failure region lower bound
    failure_mask = (df_slopes["pct_within_1_0m"] <= failure_success_floor_pct) | (
        df_slopes["pct_catastrophic_failure"] >= failure_catastrophic_floor_pct
    )
    if failure_mask.any():
        failure_lower = float(df_slopes.loc[failure_mask, "measured_mean_rt60_s"].min())
    else:
        failure_lower = float(df_slopes["measured_mean_rt60_s"].max())
        
    # Transition region interval
    trans_start = conservative_upper
    trans_end = max(conservative_upper, failure_lower)
    
    # Peak error slope
    non_zero_slopes = df_slopes[df_slopes["condition_idx"] > 0]
    if len(non_zero_slopes) > 0:
        max_e_row = non_zero_slopes.loc[non_zero_slopes["d_mean_error_d_rt60"].idxmax()]
        max_e_rt60 = float(max_e_row["measured_mean_rt60_s"])
        max_e_val = float(max_e_row["d_mean_error_d_rt60"])
        
        max_s_row = non_zero_slopes.loc[non_zero_slopes["d_success_rate_d_rt60"].idxmin()]
        max_s_rt60 = float(max_s_row["measured_mean_rt60_s"])
        max_s_val = float(max_s_row["d_success_rate_d_rt60"])
    else:
        max_e_rt60, max_e_val = 0.0, 0.0
        max_s_rt60, max_s_val = 0.0, 0.0
        
    # Is it a sharp cliff edge or gradual?
    # Criterion: If max slope is > 5x the average slope across conditions, indicates a steep transition
    mean_slope = non_zero_slopes["d_mean_error_d_rt60"].mean() if len(non_zero_slopes) > 0 else 1.0
    is_sharp = bool(max_e_val > 3.0 * max(0.1, mean_slope))
    
    note = (
        f"Stable region: RT60 <= {conservative_upper:.3f}s. "
        f"Transition zone: [{trans_start:.3f}s, {trans_end:.3f}s]. "
        f"Failure regime: RT60 >= {failure_lower:.3f}s. "
        f"Maximum degradation slope: {max_e_val:.2f} m/s at measured RT60={max_e_rt60:.3f}s."
    )
    
    res = TransitionThresholdResults(
        conservative_operating_upper_rt60_s=conservative_upper,
        transition_start_rt60_s=trans_start,
        transition_end_rt60_s=trans_end,
        failure_region_lower_rt60_s=failure_lower,
        max_error_slope_measured_rt60_s=max_e_rt60,
        max_error_slope_value=max_e_val,
        max_success_drop_slope_measured_rt60_s=max_s_rt60,
        max_success_drop_slope_value=max_s_val,
        is_sharp_cliff_edge=is_sharp,
        interpretation_note=note,
    )
    
    return res, df_slopes
