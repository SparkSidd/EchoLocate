"""
EchoLocate — Mechanistic Acoustic Analysis Module
=================================================
Calculates correlation-domain mechanistic metrics:
1. True-delay peak rank (Rank 1, 2, 3+)
2. Peak competition ratio (PCR = P1 / P2)
3. Peak ambiguity count (number of competing peaks >= 80% of primary)
4. Correlation entropy (H = -sum p log p)
5. TDOA estimation error
"""

from __future__ import annotations

import logging
import numpy as np
import scipy.signal as signal

logger = logging.getLogger(__name__)


def compute_correlation_entropy(curve: np.ndarray) -> float:
    """Calculate Shannon entropy of normalized non-negative correlation curve."""
    pos_curve = np.maximum(0.0, curve)
    sum_c = np.sum(pos_curve)
    if sum_c < 1e-12:
        return 0.0
    p = pos_curve / sum_c
    return -float(np.sum(p * np.log(p + 1e-12)))


def extract_correlation_peaks_and_rank(
    curve: np.ndarray,
    tau_axis: np.ndarray,
    true_tdoa: float,
    tolerance_seconds: float = 0.0002,  # ~3 samples at 16kHz
    ambiguity_threshold_ratio: float = 0.80,
) -> dict[str, float]:
    """
    Extract peak competition, true-delay rank, and ambiguity count from correlation curve.
    """
    pos_curve = np.maximum(0.0, curve)
    # Find local maxima
    peak_indices, _ = signal.find_peaks(pos_curve, distance=3)

    if len(peak_indices) == 0:
        peak_indices = np.array([int(np.argmax(pos_curve))])

    # Sort descending by power
    sorted_peaks = peak_indices[np.argsort(pos_curve[peak_indices])[::-1]]

    p1_idx = sorted_peaks[0]
    p1_power = float(pos_curve[p1_idx])
    p1_tau = float(tau_axis[p1_idx])

    if len(sorted_peaks) > 1:
        p2_power = float(pos_curve[sorted_peaks[1]])
    else:
        p2_power = 1e-6

    pcr = float(p1_power / max(1e-6, p2_power))

    # Count competing peaks >= ambiguity_threshold_ratio * P1
    competing_count = int(np.sum(pos_curve[sorted_peaks] >= (ambiguity_threshold_ratio * p1_power)))

    # Determine rank of the peak closest to true TDOA
    true_peak_rank = 999
    min_dist = float("inf")
    for r_idx, idx in enumerate(sorted_peaks):
        tau_val = tau_axis[idx]
        dist = abs(tau_val - true_tdoa)
        if dist <= tolerance_seconds:
            true_peak_rank = r_idx + 1
            break
        if dist < min_dist:
            min_dist = dist

    # TDOA error of primary peak
    tdoa_error_s = abs(p1_tau - true_tdoa)
    entropy = compute_correlation_entropy(pos_curve)

    return {
        "primary_peak_tau_s": p1_tau,
        "primary_peak_power": p1_power,
        "secondary_peak_power": p2_power,
        "peak_competition_ratio": pcr,
        "competing_peaks_count": float(competing_count),
        "true_tdoa_peak_rank": float(true_peak_rank),
        "tdoa_error_s": float(tdoa_error_s),
        "correlation_entropy": float(entropy),
    }
