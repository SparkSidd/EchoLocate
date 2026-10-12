"""
EchoLocate — Experiment 4 Comprehensive Evaluation & Statistical Comparison
============================================================================
Aggregates localization errors, paired differences, win rates, and Wilcoxon
signed-rank statistical tests across Methods M1, M2, M3, M4, and M5.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)


def compute_method_condition_summary(
    df_evals: pd.DataFrame,
    accuracy_thresholds: list[float] = [0.10, 0.25, 0.50, 1.0, 2.0],
    catastrophic_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Summarize performance by method and environmental condition.
    """
    records = []

    for (method, cond_id), group in df_evals.groupby(["method_id", "condition_id"], sort=True):
        errs = group["localization_error_m"].values
        n_evals = len(errs)
        mean_err = float(np.mean(errs))
        med_err = float(np.median(errs))
        rmse = float(np.sqrt(np.mean(errs**2)))
        std_err = float(np.std(errs))
        p95 = float(np.percentile(errs, 95))
        max_err = float(np.max(errs))
        cat_rate = float(np.mean(errs > catastrophic_threshold) * 100.0)

        # Accuracy rates
        succ = {
            f"pct_within_{str(t).replace('.', '_')}m": float(np.mean(errs <= t) * 100.0)
            for t in accuracy_thresholds
        }

        mean_runtime = float(group["runtime_ms"].mean()) if "runtime_ms" in group.columns else 0.0
        mean_tdoa_err = float(group["mean_tdoa_error_us"].mean()) if "mean_tdoa_error_us" in group.columns else 0.0
        mean_entropy = float(group["mean_correlation_entropy"].mean()) if "mean_correlation_entropy" in group.columns else 0.0
        mean_pcr = float(group["mean_peak_competition_ratio"].mean()) if "mean_peak_competition_ratio" in group.columns else 0.0

        records.append({
            "method_id": method,
            "condition_id": cond_id,
            "total_evaluations": n_evals,
            "mean_error_m": mean_err,
            "median_error_m": med_err,
            "rmse_m": rmse,
            "std_error_m": std_err,
            "p95_m": p95,
            "max_error_m": max_err,
            **succ,
            "catastrophic_rate_pct": cat_rate,
            "mean_runtime_ms": mean_runtime,
            "mean_tdoa_error_us": mean_tdoa_err,
            "mean_correlation_entropy": mean_entropy,
            "mean_peak_competition_ratio": mean_pcr,
        })

    return pd.DataFrame(records)


def compute_paired_statistical_comparison(
    df_evals: pd.DataFrame,
    primary_method: str = "M4_Neural_SRP",
    baseline_method: str = "M2_Raw_SRP",
) -> pd.DataFrame:
    """
    Perform paired statistical comparison (Delta Error, Win Rate, Wilcoxon test).
    """
    records = []

    for cond_id, group in df_evals.groupby("condition_id"):
        df_prim = group[group["method_id"] == primary_method].sort_values("eval_id")
        df_base = group[group["method_id"] == baseline_method].sort_values("eval_id")

        if len(df_prim) == 0 or len(df_base) == 0:
            continue

        err_prim = df_prim["localization_error_m"].values
        err_base = df_base["localization_error_m"].values

        min_len = min(len(err_prim), len(err_base))
        err_prim = err_prim[:min_len]
        err_base = err_base[:min_len]

        delta_err = err_prim - err_base
        mean_delta = float(np.mean(delta_err))
        median_delta = float(np.median(delta_err))

        wins = int(np.sum(err_prim < err_base - 1e-4))
        losses = int(np.sum(err_prim > err_base + 1e-4))
        ties = int(np.sum(np.isclose(err_prim, err_base, atol=1e-4)))
        win_rate = float(wins / min_len * 100.0)

        # Wilcoxon signed-rank test
        try:
            w_stat, p_val = wilcoxon(err_prim, err_base, alternative="two-sided")
            p_val_float = float(p_val)
        except Exception:
            p_val_float = 1.0

        records.append({
            "condition_id": cond_id,
            "primary_method": primary_method,
            "baseline_method": baseline_method,
            "sample_size": min_len,
            "mean_delta_error_m": mean_delta,
            "median_delta_error_m": median_delta,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate_pct": win_rate,
            "wilcoxon_pvalue": p_val_float,
            "statistically_significant": bool(p_val_float < 0.05),
        })

    return pd.DataFrame(records)
