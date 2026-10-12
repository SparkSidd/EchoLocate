"""
EchoLocate — Comparative Evaluation Metrics (Experiment 3)
==========================================================
Computes side-by-side performance metrics, delta improvements, and paired
statistical comparisons between classical GCC-PHAT + NLS and spatial-likelihood SRP-PHAT.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_paired_comparison_summary(
    df_evals: pd.DataFrame,
    accuracy_thresholds: list[float] = [0.25, 0.50, 1.0, 2.0],
    catastrophic_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Compute condition-level summary table comparing GCC-PHAT + NLS vs SRP-PHAT.

    Parameters
    ----------
    df_evals : pd.DataFrame
        DataFrame of paired evaluations containing:
        'condition_id', 'target_rt60_s', 'measured_mean_rt60_s', 'snr_target_db',
        'nls_localization_error', 'srp_raw_error', 'srp_refined_error',
        'srp_peak_ratio', 'srp_peak_separation_m', 'srp_top_k_proximity',
        'nls_runtime_ms', 'srp_runtime_ms'
    """
    summary_rows = []

    for cond_id, group in df_evals.groupby("condition_id", sort=False):
        n_evals = len(group)
        tgt_rt60 = float(group["target_rt60_s"].iloc[0])
        meas_rt60 = float(group["measured_mean_rt60_s"].iloc[0])
        snr_db = float(group["snr_target_db"].iloc[0])

        # 1. Classical GCC-PHAT + NLS metrics
        nls_errs = group["nls_localization_error"].values
        nls_mean_err = float(np.mean(nls_errs))
        nls_med_err = float(np.median(nls_errs))
        nls_rmse = float(np.sqrt(np.mean(nls_errs**2)))
        nls_p90 = float(np.percentile(nls_errs, 90))
        nls_cat_rate = float(np.mean(nls_errs > catastrophic_threshold) * 100.0)
        nls_acc = {
            f"nls_pct_within_{str(t).replace('.', '_')}m": float(np.mean(nls_errs <= t) * 100.0)
            for t in accuracy_thresholds
        }

        # 2. SRP-PHAT Raw Grid metrics
        srp_raw_errs = group["srp_raw_error"].values
        srp_raw_mean = float(np.mean(srp_raw_errs))
        srp_raw_med = float(np.median(srp_raw_errs))
        srp_raw_rmse = float(np.sqrt(np.mean(srp_raw_errs**2)))
        srp_raw_cat_rate = float(np.mean(srp_raw_errs > catastrophic_threshold) * 100.0)
        srp_raw_acc = {
            f"srp_raw_pct_within_{str(t).replace('.', '_')}m": float(np.mean(srp_raw_errs <= t) * 100.0)
            for t in accuracy_thresholds
        }

        # 3. SRP-PHAT Refined metrics (Primary SRP output)
        srp_ref_errs = group["srp_refined_error"].values
        srp_ref_mean = float(np.mean(srp_ref_errs))
        srp_ref_med = float(np.median(srp_ref_errs))
        srp_ref_rmse = float(np.sqrt(np.mean(srp_ref_errs**2)))
        srp_ref_p90 = float(np.percentile(srp_ref_errs, 90))
        srp_ref_cat_rate = float(np.mean(srp_ref_errs > catastrophic_threshold) * 100.0)
        srp_ref_acc = {
            f"srp_refined_pct_within_{str(t).replace('.', '_')}m": float(np.mean(srp_ref_errs <= t) * 100.0)
            for t in accuracy_thresholds
        }

        # 4. Direct Paired Improvement Metrics (Refined vs NLS)
        delta_errs = srp_ref_errs - nls_errs
        mean_delta_err = float(np.mean(delta_errs))
        median_delta_err = float(np.median(delta_errs))
        
        # Improvement in success rate <= 1.0 m (percentage points)
        nls_succ_1m = float(np.mean(nls_errs <= 1.0) * 100.0)
        srp_succ_1m = float(np.mean(srp_ref_errs <= 1.0) * 100.0)
        delta_succ_1m = srp_succ_1m - nls_succ_1m
        
        # Reduction in catastrophic failure rate (percentage points, negative = better)
        delta_cat_rate = srp_ref_cat_rate - nls_cat_rate

        # Algorithm Win Rate: percentage of trials where SRP has strictly lower error
        srp_wins = np.sum(srp_ref_errs < nls_errs)
        nls_wins = np.sum(nls_errs < srp_ref_errs)
        ties = np.sum(np.isclose(srp_ref_errs, nls_errs, atol=1e-3))
        srp_win_rate = float(srp_wins / n_evals * 100.0)

        # 5. Spatial Peak Confidence Metrics
        mean_peak_ratio = float(group["srp_peak_ratio"].mean()) if "srp_peak_ratio" in group.columns else 1.0
        median_peak_ratio = float(group["srp_peak_ratio"].median()) if "srp_peak_ratio" in group.columns else 1.0
        mean_peak_sep = float(group["srp_peak_separation_m"].mean()) if "srp_peak_separation_m" in group.columns else 0.0
        top_k_prox_rate = float(group["srp_top_k_proximity"].mean() * 100.0) if "srp_top_k_proximity" in group.columns else 0.0

        # 6. Runtimes
        nls_mean_rt = float(group["nls_runtime_ms"].mean()) if "nls_runtime_ms" in group.columns else 0.0
        srp_mean_rt = float(group["srp_runtime_ms"].mean()) if "srp_runtime_ms" in group.columns else 0.0

        row = {
            "condition_id": cond_id,
            "target_rt60_s": tgt_rt60,
            "measured_mean_rt60_s": meas_rt60,
            "snr_target_db": snr_db,
            "total_evaluations": n_evals,
            # NLS Baseline
            "nls_mean_error_m": nls_mean_err,
            "nls_median_error_m": nls_med_err,
            "nls_rmse_m": nls_rmse,
            "nls_p90_m": nls_p90,
            "nls_catastrophic_rate_pct": nls_cat_rate,
            **nls_acc,
            # SRP Raw Grid
            "srp_raw_mean_error_m": srp_raw_mean,
            "srp_raw_median_error_m": srp_raw_med,
            "srp_raw_rmse_m": srp_raw_rmse,
            "srp_raw_catastrophic_rate_pct": srp_raw_cat_rate,
            **srp_raw_acc,
            # SRP Refined
            "srp_refined_mean_error_m": srp_ref_mean,
            "srp_refined_median_error_m": srp_ref_med,
            "srp_refined_rmse_m": srp_ref_rmse,
            "srp_refined_p90_m": srp_ref_p90,
            "srp_refined_catastrophic_rate_pct": srp_ref_cat_rate,
            **srp_ref_acc,
            # Direct Deltas & Win Rate
            "delta_mean_error_m": mean_delta_err,
            "delta_median_error_m": median_delta_err,
            "delta_success_rate_1_0m_pct": delta_succ_1m,
            "delta_catastrophic_rate_pct": delta_cat_rate,
            "srp_win_rate_pct": srp_win_rate,
            # Peak Competition
            "mean_peak_ratio": mean_peak_ratio,
            "median_peak_ratio": median_peak_ratio,
            "mean_peak_separation_m": mean_peak_sep,
            "top_k_proximity_rate_pct": top_k_prox_rate,
            # Runtime
            "nls_mean_runtime_ms": nls_mean_rt,
            "srp_mean_runtime_ms": srp_mean_rt,
        }
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def compute_per_source_comparison(
    df_evals: pd.DataFrame,
    catastrophic_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Compute source-wise comparative breakdown across all conditions.
    """
    rows = []
    for src_id, group in df_evals.groupby("source_id", sort=True):
        true_x = float(group["true_x"].iloc[0])
        true_y = float(group["true_y"].iloc[0])
        n_evals = len(group)

        nls_err = group["nls_localization_error"].values
        srp_err = group["srp_refined_error"].values
        raw_err = group["srp_raw_error"].values

        nls_mean = float(np.mean(nls_err))
        srp_mean = float(np.mean(srp_err))
        raw_mean = float(np.mean(raw_err))

        nls_cat = float(np.mean(nls_err > catastrophic_threshold) * 100.0)
        srp_cat = float(np.mean(srp_err > catastrophic_threshold) * 100.0)

        delta_mean = srp_mean - nls_mean
        srp_wins = np.sum(srp_err < nls_err)
        win_rate = float(srp_wins / n_evals * 100.0)

        rows.append({
            "source_id": src_id,
            "true_x": true_x,
            "true_y": true_y,
            "total_evaluations": n_evals,
            "nls_mean_error_m": nls_mean,
            "srp_raw_mean_error_m": raw_mean,
            "srp_refined_mean_error_m": srp_mean,
            "delta_mean_error_m": delta_mean,
            "nls_catastrophic_rate_pct": nls_cat,
            "srp_catastrophic_rate_pct": srp_cat,
            "srp_win_rate_pct": win_rate,
        })

    return pd.DataFrame(rows).sort_values("delta_mean_error_m")
