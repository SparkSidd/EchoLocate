"""
EchoLocate — Confidence Calibration & Uncertainty Analysis Module
=================================================================
Evaluates correlation between spatial peak confidence scores and actual localization errors,
constructs reliability calibration curves, and calculates AUROC for catastrophic failure prediction.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def compute_confidence_calibration_summary(
    df_evals: pd.DataFrame,
    confidence_col: str = "spatial_peak_contrast",
    error_col: str = "localization_error_m",
    catastrophic_thresh_m: float = 2.0,
    n_bins: int = 5,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Compute Spearman correlation, catastrophic AUROC, and binned calibration table.
    """
    conf = df_evals[confidence_col].values
    err = df_evals[error_col].values

    # 1. Spearman Rank Correlation (higher confidence should correlate with lower error -> negative correlation)
    spearman_corr, spearman_p = spearmanr(conf, err)

    # 2. AUROC for Catastrophic Failure Detection
    # True binary label: 1 if failure (> catastrophic_thresh_m), 0 if accurate
    # Score: inverse confidence -conf
    y_true = (err > catastrophic_thresh_m).astype(int)
    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, -conf))
    else:
        auroc = 1.0

    # 3. Reliability Binned Table
    df_evals = df_evals.copy()
    df_evals["conf_bin"] = pd.qcut(df_evals[confidence_col], q=n_bins, duplicates="drop")

    bin_records = []
    for b_label, group in df_evals.groupby("conf_bin", observed=True):
        b_conf_mean = float(group[confidence_col].mean())
        b_err_mean = float(group[error_col].mean())
        b_err_med = float(group[error_col].median())
        b_succ = float(np.mean(group[error_col] <= 1.0) * 100.0)
        b_cat = float(np.mean(group[error_col] > catastrophic_thresh_m) * 100.0)

        bin_records.append({
            "confidence_bin": str(b_label),
            "mean_confidence": b_conf_mean,
            "mean_error_m": b_err_mean,
            "median_error_m": b_err_med,
            "success_rate_pct": b_succ,
            "catastrophic_rate_pct": b_cat,
            "count": len(group),
        })

    df_calib = pd.DataFrame(bin_records)

    summary_metrics = {
        "spearman_correlation": float(spearman_corr),
        "spearman_pvalue": float(spearman_p),
        "catastrophic_failure_auroc": auroc,
    }

    return summary_metrics, df_calib
