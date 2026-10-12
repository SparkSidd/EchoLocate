"""
EchoLocate — Voiced Speech Mechanism & Pitch Ambiguity Analysis
===============================================================
Analyzes speech pitch fundamental frequency F0, pitch period T0, and quantifies
whether competing correlation pseudo-peaks occur at harmonic offsets tau_true +/- n*T0.
"""

from __future__ import annotations

import logging
import numpy as np
import scipy.signal as signal

logger = logging.getLogger(__name__)


def analyze_speech_pitch_correlation_peaks(
    curve: np.ndarray,
    tau_axis: np.ndarray,
    true_tdoa: float,
    f0_hz: float,
    max_harmonic_n: int = 4,
    tolerance_s: float = 0.0003,
) -> dict[str, float]:
    """
    Quantify competing peak energy at pitch lag offsets: tau_k = tau_true +/- n * T_0.
    """
    if f0_hz <= 0.0:
        return {
            "f0_hz": 0.0,
            "pitch_period_ms": 0.0,
            "pitch_harmonic_peak_power_ratio": 0.0,
            "pitch_pseudo_peaks_detected": 0.0,
        }

    t0 = 1.0 / f0_hz
    pos_curve = np.maximum(0.0, curve)

    # Find peaks in correlation
    peak_indices, _ = signal.find_peaks(pos_curve, distance=3)
    if len(peak_indices) == 0:
        peak_indices = np.array([int(np.argmax(pos_curve))])

    p1_power = float(np.max(pos_curve))

    harmonic_targets = []
    for n in range(1, max_harmonic_n + 1):
        harmonic_targets.append(true_tdoa + n * t0)
        harmonic_targets.append(true_tdoa - n * t0)

    harmonic_peak_powers = []
    detected_pseudo_peaks = 0

    for h_tau in harmonic_targets:
        if tau_axis[0] <= h_tau <= tau_axis[-1]:
            # Find closest peak
            closest_idx = None
            min_dist = float("inf")
            for idx in peak_indices:
                tau_val = tau_axis[idx]
                d = abs(tau_val - h_tau)
                if d < min_dist:
                    min_dist = d
                    closest_idx = idx

            if closest_idx is not None and min_dist <= tolerance_s:
                p_pwr = float(pos_curve[closest_idx])
                harmonic_peak_powers.append(p_pwr)
                if p_pwr >= 0.50 * p1_power:
                    detected_pseudo_peaks += 1

    if harmonic_peak_powers:
        mean_harmonic_ratio = float(np.mean(harmonic_peak_powers) / max(1e-6, p1_power))
    else:
        mean_harmonic_ratio = 0.0

    return {
        "f0_hz": float(f0_hz),
        "pitch_period_ms": float(t0 * 1000.0),
        "pitch_harmonic_peak_power_ratio": mean_harmonic_ratio,
        "pitch_pseudo_peaks_detected": float(detected_pseudo_peaks),
    }
