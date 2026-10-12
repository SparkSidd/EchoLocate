"""
EchoLocate — Evaluation: Robustness & Trial Aggregation Metrics
================================================================
Calculates statistical summaries across independent noise trials and conditions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ConditionSummary:
    """Statistical summary of localization and TDOA performance for one condition."""
    rt60_target_s: float
    rt60_measured_mean_s: float
    rt60_measured_std_s: float
    snr_target_db: float
    snr_measured_mean_db: float
    num_trials: int
    num_sources: int
    total_evaluations: int
    
    # TDOA Metrics (averaged across trials & sources)
    mean_tdoa_error_s: float
    mean_tdoa_error_ms: float
    mean_tdoa_error_samples: float
    median_tdoa_error_ms: float
    rmse_tdoa_error_ms: float
    max_tdoa_error_ms: float
    
    # Localization Metrics (averaged across trials & sources)
    mean_localization_error_m: float
    std_localization_error_m: float
    median_localization_error_m: float
    rmse_localization_error_m: float
    min_localization_error_m: float
    max_localization_error_m: float
    ci95_low_m: float
    ci95_high_m: float
    
    # Threshold Success Rates (%)
    pct_within_0_25m: float
    pct_within_0_5m: float
    pct_within_1_0m: float
    pct_within_2_0m: float
    
    # Spatial Bias (m)
    x_bias_m: float
    y_bias_m: float


def compute_condition_summary(
    df_trials: pd.DataFrame,
    rt60_target: float,
    snr_target: float,
    rt60_measured_stats: dict[str, float] | None = None,
    fs: int = 16000,
) -> ConditionSummary:
    """
    Aggregate per-trial, per-source evaluation records into a unified condition summary.

    Parameters
    ----------
    df_trials : pd.DataFrame
        DataFrame containing records for a single (RT60, SNR) condition across all trials & sources.
    rt60_target : float
        Nominal target RT60 in seconds.
    snr_target : float
        Nominal target SNR in dB.
    rt60_measured_stats : dict or None
        Empirical RT60 distribution stats (mean, std, etc.).
    fs : int
        Sampling frequency in Hz.
    """
    loc_errors = df_trials["localization_error"].values
    n_evals = len(loc_errors)
    num_trials = len(df_trials["trial_idx"].unique())
    num_sources = len(df_trials["source_id"].unique())
    
    # Localization error metrics
    mean_loc = float(np.mean(loc_errors))
    std_loc = float(np.std(loc_errors, ddof=1)) if n_evals > 1 else 0.0
    median_loc = float(np.median(loc_errors))
    rmse_loc = float(np.sqrt(np.mean(loc_errors ** 2)))
    min_loc = float(np.min(loc_errors))
    max_loc = float(np.max(loc_errors))
    
    # 95% Confidence Interval for mean localization error
    if n_evals > 1 and std_loc > 0:
        se = std_loc / np.sqrt(n_evals)
        ci_low, ci_high = stats.t.interval(0.95, df=n_evals - 1, loc=mean_loc, scale=se)
    else:
        ci_low, ci_high = mean_loc, mean_loc
        
    # Thresholds
    pct_025 = float(np.mean(loc_errors <= 0.25) * 100.0)
    pct_050 = float(np.mean(loc_errors <= 0.50) * 100.0)
    pct_100 = float(np.mean(loc_errors <= 1.00) * 100.0)
    pct_200 = float(np.mean(loc_errors <= 2.00) * 100.0)
    
    # Biases
    x_bias = float(np.mean(df_trials["pred_x"] - df_trials["true_x"]))
    y_bias = float(np.mean(df_trials["pred_y"] - df_trials["true_y"]))
    
    # TDOA Errors across all 3 pairs
    tdoa_errs_s = []
    for p in ["01", "02", "12"]:
        if f"tdoa_{p}_estimated_s" in df_trials.columns and f"tdoa_{p}_theoretical_s" in df_trials.columns:
            err_p = np.abs(df_trials[f"tdoa_{p}_estimated_s"] - df_trials[f"tdoa_{p}_theoretical_s"])
            tdoa_errs_s.extend(err_p.tolist())
            
    tdoa_errs_s = np.array(tdoa_errs_s)
    mean_tdoa_s = float(np.mean(tdoa_errs_s)) if len(tdoa_errs_s) > 0 else 0.0
    median_tdoa_s = float(np.median(tdoa_errs_s)) if len(tdoa_errs_s) > 0 else 0.0
    rmse_tdoa_s = float(np.sqrt(np.mean(tdoa_errs_s ** 2))) if len(tdoa_errs_s) > 0 else 0.0
    max_tdoa_s = float(np.max(tdoa_errs_s)) if len(tdoa_errs_s) > 0 else 0.0
    
    # Measured SNR (if present)
    snr_meas = float(df_trials["achieved_snr_db"].mean()) if "achieved_snr_db" in df_trials.columns else snr_target
    
    # Measured RT60
    rt60_m_mean = rt60_measured_stats.get("mean", 0.0) if rt60_measured_stats else 0.0
    rt60_m_std = rt60_measured_stats.get("std", 0.0) if rt60_measured_stats else 0.0
    
    return ConditionSummary(
        rt60_target_s=rt60_target,
        rt60_measured_mean_s=rt60_m_mean,
        rt60_measured_std_s=rt60_m_std,
        snr_target_db=snr_target,
        snr_measured_mean_db=snr_meas,
        num_trials=num_trials,
        num_sources=num_sources,
        total_evaluations=n_evals,
        mean_tdoa_error_s=mean_tdoa_s,
        mean_tdoa_error_ms=mean_tdoa_s * 1000.0,
        mean_tdoa_error_samples=mean_tdoa_s * fs,
        median_tdoa_error_ms=median_tdoa_s * 1000.0,
        rmse_tdoa_error_ms=rmse_tdoa_s * 1000.0,
        max_tdoa_error_ms=max_tdoa_s * 1000.0,
        mean_localization_error_m=mean_loc,
        std_localization_error_m=std_loc,
        median_localization_error_m=median_loc,
        rmse_localization_error_m=rmse_loc,
        min_localization_error_m=min_loc,
        max_localization_error_m=max_loc,
        ci95_low_m=float(ci_low),
        ci95_high_m=float(ci_high),
        pct_within_0_25m=pct_025,
        pct_within_0_5m=pct_050,
        pct_within_1_0m=pct_100,
        pct_within_2_0m=pct_200,
        x_bias_m=x_bias,
        y_bias_m=y_bias,
    )
