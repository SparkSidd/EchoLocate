"""
EchoLocate — Live GCC-PHAT Real-Time Correlator (Accuracy Overhaul)
====================================================================
Computes pairwise GCC-PHAT cross-correlation R_12(tau) on streaming
real hardware dual-channel frames.

Signal conditioning pipeline (applied before FFT):
  1. Hann window — eliminates spectral leakage at frame boundaries
  2. Bandpass filter (200–4000 Hz) — removes sub-bass rumble & ultrasonic noise
  3. Per-channel RMS normalization — equalizes level mismatch between devices
  4. PHAT whitening — flattens the cross-power spectrum for sharp time delay peak
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Bandpass design (200 Hz – 4000 Hz) — pre-computed FIR coefficients at 16 kHz
# 63-tap Kaiser window FIR (β=6) for sharp roll-off without scipy dependency
_FIR_TAPS = 63
_FIR_FS = 16000


def _design_bandpass_fir(low_hz: float, high_hz: float, fs: int, ntaps: int) -> np.ndarray:
    """Design a linear-phase FIR bandpass filter using windowed-sinc method."""
    nyq = fs / 2.0
    # High-pass component
    fc_low = low_hz / nyq
    # Low-pass component
    fc_high = high_hz / nyq
    # Kaiser window β=6 for ~53 dB stopband attenuation
    beta = 6.0
    win = np.kaiser(ntaps, beta)
    n = np.arange(ntaps) - (ntaps - 1) / 2.0
    # Low-pass sinc at high cutoff
    lp = np.sinc(2 * fc_high * n) * 2 * fc_high
    # High-pass = delta - low-pass at low cutoff
    hp = np.sinc(2 * fc_low * n) * 2 * fc_low
    # Bandpass = low-pass minus high-pass (spectral subtraction)
    h = (lp - hp) * win
    h /= np.sum(h)  # unity gain at passband centre
    return h.astype(np.float32)


# Cache the FIR kernel once at module load
_BP_FIR: np.ndarray | None = None


def _get_bandpass_fir(fs: int) -> np.ndarray:
    global _BP_FIR
    if _BP_FIR is None or len(_BP_FIR) == 0:
        _BP_FIR = _design_bandpass_fir(200.0, 4000.0, fs, _FIR_TAPS)
    return _BP_FIR


def _condition_signal(sig: np.ndarray, fs: int, window: np.ndarray) -> np.ndarray:
    """
    Apply full signal conditioning pipeline:
    Hann window → bandpass FIR → RMS normalization
    """
    # 1. Hann window to kill spectral leakage
    s = sig.astype(np.float64) * window

    # 2. Bandpass FIR (200–4000 Hz) via linear convolution, trimmed to original length
    fir = _get_bandpass_fir(fs).astype(np.float64)
    filtered = np.convolve(s, fir, mode="same")

    # 3. Per-channel RMS normalization — prevents level mismatch corrupting PHAT
    rms = float(np.sqrt(np.mean(filtered ** 2)))
    if rms > 1e-7:
        filtered /= rms

    return filtered.astype(np.float32)


@dataclass
class LiveGCCResult:
    """Real-time GCC-PHAT output container."""
    tau_axis_ms: list[float]
    correlation_curve: list[float]
    estimated_delay_ms: float
    peak_power: float
    secondary_peak_ms: float
    secondary_peak_power: float
    psr: float
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau_axis_ms": [round(t, 3) for t in self.tau_axis_ms],
            "correlation_curve": [round(c, 5) for c in self.correlation_curve],
            "estimated_delay_ms": round(self.estimated_delay_ms, 3),
            "peak_power": round(self.peak_power, 5),
            "secondary_peak_ms": round(self.secondary_peak_ms, 3),
            "secondary_peak_power": round(self.secondary_peak_power, 5),
            "psr": round(self.psr, 3),
            "latency_ms": round(self.latency_ms, 2),
        }


def compute_live_gcc_phat(
    signal_laptop: np.ndarray,
    signal_phone: np.ndarray,
    fs: int = 16000,
    max_tau_ms: float = 10.0,
) -> LiveGCCResult:
    """
    Compute conditioned GCC-PHAT cross-correlation between laptop and phone microphones.

    Signal conditioning:
      - Hann window (eliminates spectral leakage)
      - Bandpass FIR 200–4000 Hz (removes out-of-band noise)
      - RMS normalization per channel (equalizes level mismatch)
      - PHAT whitening (sharpens inter-arrival peak)
    """
    t0 = time.perf_counter()

    n = len(signal_laptop)
    if n < 64:
        # Not enough data — return flat result
        return _empty_result(max_tau_ms, fs)

    # Pre-compute Hann window once
    hann = np.hanning(n).astype(np.float32)

    # Apply full conditioning pipeline to both channels
    s1 = _condition_signal(signal_laptop, fs, hann)
    s2 = _condition_signal(signal_phone, fs, hann)

    # FFT-domain cross-correlation with PHAT whitening
    nfft = int(2 ** np.ceil(np.log2(2 * n - 1)))  # next power-of-2 for efficiency
    S1 = np.fft.rfft(s1, n=nfft)
    S2 = np.fft.rfft(s2, n=nfft)

    # Cross-power spectrum
    cross = S1 * np.conj(S2)

    # PHAT whitening: divide by magnitude (suppresses frequency-domain amplitude)
    mag = np.abs(cross)
    mag = np.where(mag < 1e-12, 1e-12, mag)
    phat = cross / mag

    # Inverse FFT → correlation function
    gcc = np.fft.irfft(phat, n=nfft).real

    # Centre the correlation at zero-lag
    gcc = np.fft.fftshift(gcc[:n])

    # Restrict to physically meaningful range: |tau| <= max_tau_ms
    max_tau_samples = int(max_tau_ms * 1e-3 * fs)
    centre = n // 2
    lo = max(0, centre - max_tau_samples)
    hi = min(n, centre + max_tau_samples + 1)
    gcc_window = gcc[lo:hi]
    tau_samples = np.arange(lo - centre, hi - centre)
    tau_ms = tau_samples / fs * 1000.0

    # Primary peak
    p1_idx = int(np.argmax(gcc_window))
    p1_tau_ms = float(tau_ms[p1_idx])
    p1_pow = float(gcc_window[p1_idx])

    # Secondary peak — at least 1.5 ms away (avoids sidelobe confusion)
    p2_tau_ms = 0.0
    p2_pow = 1e-9
    min_sep = int(1.5e-3 * fs)
    for idx in np.argsort(gcc_window)[::-1][1:]:
        if abs(int(idx) - p1_idx) >= min_sep:
            p2_tau_ms = float(tau_ms[idx])
            p2_pow = float(gcc_window[idx])
            break

    psr = float(p1_pow / max(1e-9, p2_pow))
    lat_ms = (time.perf_counter() - t0) * 1000.0

    return LiveGCCResult(
        tau_axis_ms=tau_ms.tolist(),
        correlation_curve=gcc_window.tolist(),
        estimated_delay_ms=p1_tau_ms,
        peak_power=p1_pow,
        secondary_peak_ms=p2_tau_ms,
        secondary_peak_power=p2_pow,
        psr=psr,
        latency_ms=lat_ms,
    )


def _empty_result(max_tau_ms: float, fs: int) -> LiveGCCResult:
    """Return an empty zero result when there is insufficient data."""
    n = int(2 * max_tau_ms * 1e-3 * fs) + 1
    tau_ms = np.linspace(-max_tau_ms, max_tau_ms, n).tolist()
    return LiveGCCResult(
        tau_axis_ms=tau_ms,
        correlation_curve=[0.0] * n,
        estimated_delay_ms=0.0,
        peak_power=0.0,
        secondary_peak_ms=0.0,
        secondary_peak_power=0.0,
        psr=0.0,
        latency_ms=0.0,
    )
