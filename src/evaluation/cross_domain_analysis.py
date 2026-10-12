"""
EchoLocate — Cross-Domain Signal & Speech Mechanism Analysis
============================================================
Implements unseen speech synthesis, pitch stress testing (PPR, HSR, CPD),
and 9-signal class generators for Experiment 4.75.
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.signal import chirp as scipy_chirp

logger = logging.getLogger(__name__)


def generate_adversarial_speech_utterance(
    signal_type: str = "voiced_speech",
    f0_hz: float = 150.0,
    duration: float = 0.20,
    fs: int = 16000,
    sweep_type: str = "constant",  # "constant", "vibrato", "rising", "falling"
    vibrato_depth_hz: float = 10.0,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Generate synthetic speech signals with parameterized fundamental frequency and formants.

    Returns
    -------
    signal : np.ndarray, shape (n_samples,)
    meta : dict containing f0_mean, f0_bin, pitch_stability
    """
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    rng = np.random.default_rng(seed)

    # 1. Generate Instantaneous Frequency f(t)
    if sweep_type == "constant":
        inst_f0 = np.full(n_samples, f0_hz)
    elif sweep_type == "vibrato":
        inst_f0 = f0_hz + vibrato_depth_hz * np.sin(2 * np.pi * 5.0 * t)
    elif sweep_type == "rising":
        inst_f0 = np.linspace(f0_hz * 0.8, f0_hz * 1.2, n_samples)
    elif sweep_type == "falling":
        inst_f0 = np.linspace(f0_hz * 1.2, f0_hz * 0.8, n_samples)
    else:
        inst_f0 = np.full(n_samples, f0_hz)

    # 2. Phase Integration
    phase = 2 * np.pi * np.cumsum(inst_f0) / fs

    # 3. Harmonic Series with Vowel Formants (/a/: F1=700Hz, F2=1200Hz, F3=2500Hz)
    signal = np.zeros(n_samples)
    formants = [(700.0, 100.0, 1.0), (1200.0, 120.0, 0.6), (2500.0, 150.0, 0.3)]

    max_h = int(min(fs / 2, 7000.0) / max(1.0, f0_hz))
    for h in range(1, max_h + 1):
        h_freq = h * f0_hz
        # Spectral envelope based on formants
        gain = 0.0
        for f_center, f_bw, f_amp in formants:
            gain += f_amp * np.exp(-((h_freq - f_center) ** 2) / (2 * (f_bw ** 2)))
        # Glottal source roll-off (-12 dB/octave)
        source_roll = 1.0 / (h ** 1.2)
        signal += gain * source_roll * np.sin(h * phase)

    if signal_type == "unvoiced_speech":
        # Unvoiced fricative /s/: shaped high-pass noise (4-7 kHz)
        noise = rng.normal(0.0, 1.0, size=n_samples)
        from scipy.signal import butter, sosfilt
        sos = butter(4, [4000 / (fs / 2), 7500 / (fs / 2)], btype="bandpass", output="sos")
        signal = sosfilt(sos, noise)

    elif signal_type == "sentence":
        # Multi-syllable modulated sequence
        mod_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
        signal = signal * mod_envelope

    # Normalize amplitude
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))

    # Apply 5% Tukey window to prevent edge transients
    window = np.hanning(n_samples)
    signal = signal * window

    # Assign F0 Bin
    mean_f0 = float(np.mean(inst_f0))
    if mean_f0 < 120.0:
        f0_bin = "80-120"
    elif mean_f0 < 180.0:
        f0_bin = "120-180"
    elif mean_f0 < 250.0:
        f0_bin = "180-250"
    else:
        f0_bin = "250+"

    meta = {
        "mean_f0_hz": mean_f0,
        "f0_bin": f0_bin,
        "sweep_type": sweep_type,
    }
    return signal, meta


def compute_pitch_stress_metrics(
    raw_corr_curve: np.ndarray,
    enhanced_corr_curve: np.ndarray,
    tau_axis: np.ndarray,
    true_tdoa: float,
    f0_hz: float,
) -> dict[str, float]:
    """
    Compute Pseudo-Peak Ratio (PPR), Harmonic Suppression Ratio (HSR),
    and Correct-Peak Dominance (CPD).
    """
    t_pitch = 1.0 / max(10.0, f0_hz)
    true_idx = int(np.argmin(np.abs(tau_axis - true_tdoa)))

    # Identify harmonic pseudo-peak locations (true_tdoa +/- T_pitch)
    harmonic_lag_pos = true_tdoa + t_pitch
    harmonic_lag_neg = true_tdoa - t_pitch

    pos_idx = int(np.argmin(np.abs(tau_axis - harmonic_lag_pos)))
    neg_idx = int(np.argmin(np.abs(tau_axis - harmonic_lag_neg)))

    w_pos_start, w_pos_end = max(0, pos_idx - 2), min(len(tau_axis), pos_idx + 3)
    w_neg_start, w_neg_end = max(0, neg_idx - 2), min(len(tau_axis), neg_idx + 3)

    # Raw curve amplitudes
    raw_true = max(1e-6, float(raw_corr_curve[true_idx]))
    raw_harm = max(1e-6, float(max(np.max(raw_corr_curve[w_pos_start:w_pos_end]), np.max(raw_corr_curve[w_neg_start:w_neg_end]))))

    # Enhanced curve amplitudes
    enh_true = max(1e-6, float(enhanced_corr_curve[true_idx]))
    enh_harm = max(1e-6, float(max(np.max(enhanced_corr_curve[w_pos_start:w_pos_end]), np.max(enhanced_corr_curve[w_neg_start:w_neg_end]))))

    # Exclude true peak region to find second largest peak
    mask = np.ones(len(tau_axis), dtype=bool)
    mask[max(0, true_idx - 5) : min(len(tau_axis), true_idx + 6)] = False
    enh_second = float(np.max(enhanced_corr_curve[mask])) if np.any(mask) else 1e-6

    ppr_raw = float(raw_harm / raw_true)
    ppr_enhanced = float(enh_harm / enh_true)
    hsr = float(raw_harm / max(1e-6, enh_harm))
    cpd = float(enh_true / max(1e-6, enh_second))

    return {
        "raw_ppr": ppr_raw,
        "enhanced_ppr": ppr_enhanced,
        "harmonic_suppression_ratio": hsr,
        "correct_peak_dominance": cpd,
    }


def generate_benchmark_signal(
    signal_type: str,
    fs: int = 16000,
    duration: float = 0.20,
    seed: int = 42,
) -> np.ndarray:
    """Generate signal corresponding to any of the 9 benchmark classes."""
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    rng = np.random.default_rng(seed)

    if signal_type == "broadband_noise":
        noise = rng.normal(0.0, 1.0, size=n_samples)
        return noise / np.max(np.abs(noise))
    elif signal_type == "linear_chirp":
        return scipy_chirp(t, f0=300.0, t1=duration, f1=7000.0, method="linear")
    elif signal_type == "log_chirp":
        return scipy_chirp(t, f0=300.0, t1=duration, f1=7000.0, method="logarithmic")
    elif signal_type == "impulse":
        sig = np.zeros(n_samples)
        sig[n_samples // 4] = 1.0
        return sig
    elif signal_type == "tonal":
        return np.sin(2 * np.pi * 1000.0 * t)
    elif signal_type == "multi_tone":
        return (np.sin(2 * np.pi * 500.0 * t) + np.sin(2 * np.pi * 1500.0 * t) + np.sin(2 * np.pi * 3000.0 * t)) / 3.0
    elif signal_type in ["voiced_speech", "unvoiced_speech", "sentence"]:
        sig, _ = generate_adversarial_speech_utterance(signal_type=signal_type, f0_hz=150.0, duration=duration, fs=fs, seed=seed)
        return sig
    else:
        return scipy_chirp(t, f0=300.0, t1=duration, f1=7000.0, method="linear")
