"""
EchoLocate — Calibration Generalization & Uncertainty Analysis
==============================================================
Implements temperature transfer, geometric equivariance audits (R*x + t),
and confidence calibration (ECE, AURC, Pearson/Spearman) for Experiment 4.75.
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


def compute_expected_calibration_error(
    confidences: np.ndarray,
    errors_m: np.ndarray,
    error_threshold_m: float = 0.50,
    n_bins: int = 10,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE) treating localization within error_threshold_m
    as the binary target accuracy event.

    Returns
    -------
    ece : float
    bin_confs : np.ndarray, shape (n_bins,)
    bin_accs : np.ndarray, shape (n_bins,)
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    # Clip to [0, 1]
    confidences = np.clip(confidences, 0.0, 1.0)
    accuracies = (np.asarray(errors_m) <= error_threshold_m).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs = np.zeros(n_bins)
    bin_accs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    ece = 0.0
    n_total = len(confidences)

    for b in range(n_bins):
        low, high = bin_edges[b], bin_edges[b + 1]
        mask = (confidences >= low) & (confidences < high if b < n_bins - 1 else confidences <= high)
        count = int(np.sum(mask))
        bin_counts[b] = count

        if count > 0:
            avg_conf = float(np.mean(confidences[mask]))
            avg_acc = float(np.mean(accuracies[mask]))
            bin_confs[b] = avg_conf
            bin_accs[b] = avg_acc
            ece += (count / n_total) * abs(avg_acc - avg_conf)
        else:
            bin_confs[b] = (low + high) / 2.0
            bin_accs[b] = 0.0

    return float(ece), bin_confs, bin_accs


def compute_risk_coverage_curve(
    confidences: np.ndarray,
    errors_m: np.ndarray,
    n_points: int = 50,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Risk-Coverage Curve and Area Under Risk-Coverage (AURC).
    Risk = Mean localization error of retained predictions above confidence threshold.
    Coverage = Fraction of dataset retained.

    Returns
    -------
    aurc : float
    coverages : np.ndarray
    risks : np.ndarray
    """
    confs = np.asarray(confidences)
    errs = np.asarray(errors_m)

    coverages = []
    risks = []

    percentiles = np.linspace(0.0, 95.0, n_points)
    for p in percentiles:
        thresh = float(np.percentile(confs, p))
        mask = confs >= thresh
        if np.any(mask):
            cov = float(np.mean(mask))
            risk = float(np.mean(errs[mask]))
            coverages.append(cov)
            risks.append(risk)

    coverages = np.array(coverages)
    risks = np.array(risks)

    # Sort by coverage ascending
    idx = np.argsort(coverages)
    cov_sorted = coverages[idx]
    risk_sorted = risks[idx]

    aurc = float(np.trapezoid(risk_sorted, cov_sorted)) if len(cov_sorted) > 1 else 0.0
    return aurc, coverages, risks


def compute_confidence_correlation_metrics(
    confidences: np.ndarray,
    errors_m: np.ndarray,
) -> dict[str, float]:
    """
    Compute Pearson and Spearman correlation between predicted confidence and localization error.
    Negative correlation indicates desirable uncertainty awareness (higher conf -> lower error).
    """
    if len(confidences) < 3 or np.all(confidences == confidences[0]) or np.all(errors_m == errors_m[0]):
        return {"pearson_r": 0.0, "spearman_rho": 0.0}

    r_pearson, _ = pearsonr(confidences, errors_m)
    r_spearman, _ = spearmanr(confidences, errors_m)

    return {
        "pearson_r": float(r_pearson),
        "spearman_rho": float(r_spearman),
    }


def audit_geometric_equivariance(
    pred_orig_xy: np.ndarray,
    pred_transformed_xy: np.ndarray,
    rotation_matrix_2d: np.ndarray,
    translation_2d: np.ndarray,
) -> float:
    """
    Measure equivariance deviation: || x'_hat - (R * x_hat + t) ||_2.
    Ideal value = 0.0 m.
    """
    expected_transformed = rotation_matrix_2d @ np.asarray(pred_orig_xy) + np.asarray(translation_2d)
    equivariance_err_m = float(np.linalg.norm(np.asarray(pred_transformed_xy) - expected_transformed))
    return equivariance_err_m
