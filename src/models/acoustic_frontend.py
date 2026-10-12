"""
EchoLocate — Acoustic Frontend & Feature Extraction Module
==========================================================
Provides signal generation, STFT processing, pairwise cross-correlation feature
extraction, and idealized Gaussian target generation for DeepGCC.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy.signal as signal
from scipy.signal import chirp as scipy_chirp
import torch

from src.simulation.source_generator import generate_chirp

logger = logging.getLogger(__name__)


def generate_experiment_04_signal(
    signal_type: str,
    fs: int = 16000,
    duration: float = 0.1,
    seed: int = 42,
    f0_override: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Generate diverse source signals with metadata including pitch fundamental frequency.
    """
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    rng = np.random.default_rng(seed)
    meta: dict[str, Any] = {"signal_type": signal_type, "duration": duration, "fs": fs}

    if signal_type == "broadband_noise":
        noise = rng.normal(0.0, 1.0, size=n_samples)
        window = np.sin(np.pi * np.linspace(0, 1, n_samples))
        sig = noise * window
        meta["f0_hz"] = 0.0

    elif signal_type == "linear_chirp":
        sig = generate_chirp(fs=fs, duration=duration, f_low=300.0, f_high=7000.0)
        meta["f0_hz"] = 0.0

    elif signal_type == "log_chirp":
        sig = scipy_chirp(t, f0=300.0, t1=duration, f1=7000.0, method="logarithmic")
        sig = sig * np.hanning(len(sig))
        meta["f0_hz"] = 0.0

    elif "speech" in signal_type:
        # Determine pitch F0 based on subtype
        if f0_override is not None:
            f0 = float(f0_override)
        elif "female" in signal_type:
            f0 = rng.uniform(200.0, 280.0)
        elif "child" in signal_type:
            f0 = rng.uniform(280.0, 360.0)
        else:  # male default
            f0 = rng.uniform(100.0, 160.0)

        meta["f0_hz"] = float(f0)
        meta["pitch_period_s"] = float(1.0 / f0)
        meta["pitch_period_samples"] = float(fs / f0)

        # Voiced vowel synthesis with 4 formants (F1, F2, F3, F4)
        if "female" in signal_type:
            formants = [800.0, 1600.0, 2900.0, 3900.0]
            bandwidths = [100.0, 120.0, 180.0, 250.0]
        else:
            formants = [700.0, 1200.0, 2500.0, 3500.0]
            bandwidths = [80.0, 100.0, 150.0, 200.0]

        n_harmonics = int(min(7000.0, fs / 2.0 - 100.0) / f0)
        harmonic_sig = np.zeros(n_samples, dtype=np.float64)

        for h in range(1, n_harmonics + 1):
            fh = h * f0
            # Formant shaping filter response
            h_gain = 0.02
            for f_res, bw in zip(formants, bandwidths):
                h_gain += np.exp(-((fh - f_res) ** 2) / (2.0 * (bw ** 2)))
            harmonic_sig += h_gain * np.sin(2.0 * np.pi * fh * t + rng.uniform(0, 2 * np.pi))

        # Dynamic amplitude envelope mimicking phoneme articulation
        envelope = (np.sin(np.pi * np.linspace(0, 1, n_samples)) ** 2)
        sig = harmonic_sig * envelope

    elif signal_type == "impulsive_click":
        decay = np.exp(-t / 0.005)
        carrier = np.sin(2 * np.pi * 2000 * t) + 0.5 * np.sin(2 * np.pi * 4500 * t)
        sig = decay * carrier
        meta["f0_hz"] = 0.0

    else:
        sig = generate_chirp(fs=fs, duration=duration, f_low=300.0, f_high=7000.0)
        meta["f0_hz"] = 0.0

    # Normalize amplitude
    sig = sig / (np.max(np.abs(sig)) + 1e-12)
    return sig, meta


def extract_pairwise_gcc_features(
    mic_signals: np.ndarray,
    fs: int = 16000,
    max_tau_seconds: float = 0.030,
    num_tau_bins: int = 129,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract bounded GCC-PHAT features and Inter-channel Phase Differences for all mic pairs.

    Returns
    -------
    gcc_features : np.ndarray, shape (n_pairs, num_tau_bins)
    ipd_features : np.ndarray, shape (n_pairs, n_freq_bins)
    tau_axis : np.ndarray, shape (num_tau_bins,)
    """
    n_mics, n_samples = mic_signals.shape
    pairs = []
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            pairs.append((i, j))
    n_pairs = len(pairs)

    n_fft = 2 ** int(np.ceil(np.log2(2 * n_samples - 1)))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    X = np.fft.rfft(mic_signals, n=n_fft, axis=1)

    tau_axis = np.linspace(-max_tau_seconds, max_tau_seconds, num_tau_bins, dtype=np.float64)
    gcc_features = np.zeros((n_pairs, num_tau_bins), dtype=np.float32)
    ipd_features = np.zeros((n_pairs, len(freqs)), dtype=np.float32)

    full_lags = np.fft.fftfreq(n_fft, d=fs)
    full_lags = np.fft.fftshift(full_lags)

    for p_idx, (i, j) in enumerate(pairs):
        cps = X[i] * np.conj(X[j])
        phase_diff = np.angle(cps)
        ipd_features[p_idx] = phase_diff.astype(np.float32)

        # PHAT normalization
        denom = np.abs(cps)
        denom[denom < 1e-12] = 1e-12
        phat_spec = cps / denom

        # Centered IFFT
        r_phat = np.fft.fftshift(np.fft.irfft(phat_spec, n=n_fft))
        # Interpolate onto bounded uniform tau_axis
        gcc_interp = np.interp(tau_axis, full_lags, r_phat)
        # Normalize peak to 1.0 for stability
        max_v = np.max(np.abs(gcc_interp))
        if max_v > 1e-12:
            gcc_interp /= max_v
        gcc_features[p_idx] = gcc_interp.astype(np.float32)

    return gcc_features, ipd_features, tau_axis


def generate_gaussian_target_distribution(
    true_tdoa_seconds: float,
    tau_axis: np.ndarray,
    sigma_seconds: float = 0.0001,  # ~1.6 samples at 16kHz
) -> np.ndarray:
    """
    Generate normalized continuous Gaussian target distribution centered at true TDOA.
    """
    dist_sq = (tau_axis - true_tdoa_seconds) ** 2
    target = np.exp(-dist_sq / (2.0 * (sigma_seconds ** 2)))
    sum_t = np.sum(target)
    if sum_t > 1e-12:
        target /= sum_t
    return target.astype(np.float32)
