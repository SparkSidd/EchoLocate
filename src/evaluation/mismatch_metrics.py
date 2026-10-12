"""
EchoLocate — Model Mismatch Metrics Module
==========================================
Computes parameter mismatch degradation metrics, relative performance drops,
sensitivity slopes, and tolerance thresholds for microphone placement,
speed-of-sound, synchronization, and room geometry uncertainties.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_mismatch_sensitivity_table(
    df_evals: pd.DataFrame,
    mismatch_param_col: str,
    control_val: float = 0.0,
    accuracy_thresholds: list[float] = [0.25, 0.50, 1.0, 2.0],
    catastrophic_threshold: float = 2.0,
    group_cols: list[str] = ["condition_id"],
) -> pd.DataFrame:
    """
    Compute sensitivity summary table for a parameter mismatch sweep.
    """
    records = []

    for group_keys, group in df_evals.groupby(group_cols + [mismatch_param_col], sort=True):
        if isinstance(group_keys, tuple):
            cond_id = group_keys[0]
            mismatch_val = group_keys[1]
        else:
            cond_id = group_keys
            mismatch_val = group[mismatch_param_col].iloc[0]

        n_evals = len(group)
        errs = group["localization_error_m"].values
        mean_err = float(np.mean(errs))
        med_err = float(np.median(errs))
        rmse = float(np.sqrt(np.mean(errs**2)))
        std_err = float(np.std(errs))
        p90 = float(np.percentile(errs, 90))
        p95 = float(np.percentile(errs, 95))

        # Spatial bias
        dx = group["x_error_m"].values
        dy = group["y_error_m"].values
        mean_x_bias = float(np.mean(dx))
        mean_y_bias = float(np.mean(dy))
        bias_magnitude = float(np.sqrt(mean_x_bias**2 + mean_y_bias**2))

        # Success & Catastrophic rates
        succ_rates = {
            f"pct_within_{str(t).replace('.', '_')}m": float(np.mean(errs <= t) * 100.0)
            for t in accuracy_thresholds
        }
        cat_rate = float(np.mean(errs > catastrophic_threshold) * 100.0)

        records.append({
            "condition_id": cond_id,
            mismatch_param_col: mismatch_val,
            "total_evaluations": n_evals,
            "mean_error_m": mean_err,
            "median_error_m": med_err,
            "rmse_m": rmse,
            "std_error_m": std_err,
            "p90_m": p90,
            "p95_m": p95,
            "mean_x_bias_m": mean_x_bias,
            "mean_y_bias_m": mean_y_bias,
            "spatial_bias_magnitude_m": bias_magnitude,
            **succ_rates,
            "catastrophic_rate_pct": cat_rate,
        })

    df_out = pd.DataFrame(records)

    # Compute relative performance drop against control value
    # Relative Performance Drop = (mismatched error - control error) / max(control error, epsilon)
    eps = 0.05  # 5 cm floor to avoid division by zero on near-perfect control
    enhanced_records = []
    for cond_id, cond_df in df_out.groupby("condition_id"):
        ctrl_rows = cond_df[np.isclose(cond_df[mismatch_param_col], control_val, atol=1e-5)]
        ctrl_mean = ctrl_rows["mean_error_m"].iloc[0] if not ctrl_rows.empty else cond_df["mean_error_m"].min()

        for _, row in cond_df.iterrows():
            m_err = row["mean_error_m"]
            abs_drop = m_err - ctrl_mean
            rel_drop = abs_drop / max(ctrl_mean, eps)
            row_dict = row.to_dict()
            row_dict["control_mean_error_m"] = ctrl_mean
            row_dict["absolute_performance_drop_m"] = abs_drop
            row_dict["relative_performance_drop"] = rel_drop
            enhanced_records.append(row_dict)

    return pd.DataFrame(enhanced_records)


def estimate_tolerance_threshold(
    df_mismatch_sum: pd.DataFrame,
    mismatch_param_col: str,
    target_error_limit_m: float = 1.0,
) -> dict[str, float]:
    """
    Estimate maximum parameter error tolerable before mean localization error exceeds target_error_limit_m.
    """
    tolerances = {}
    for cond_id, group in df_mismatch_sum.groupby("condition_id"):
        sorted_g = group.sort_values(mismatch_param_col)
        # Filter where mean_error <= target_error_limit_m
        valid = sorted_g[sorted_g["mean_error_m"] <= target_error_limit_m]
        if not valid.empty:
            tolerances[cond_id] = float(valid[mismatch_param_col].max())
        else:
            tolerances[cond_id] = 0.0
    return tolerances
