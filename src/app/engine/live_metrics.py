"""
EchoLocate — Real-Time Audio Frame Metrics
==========================================
Computes live physical signal characteristics: RMS (dB), Peak Amplitude,
Estimated SNR, Spectral Flatness, Dominant Frequency, and Clipping Detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LiveSignalMetrics:
    """Quantitative characteristics of a live audio frame."""
    rms_energy_db: float
    peak_amplitude: float
    estimated_snr_db: float
    dominant_freq_hz: float
    spectral_flatness: float
    is_clipping: bool
    is_silence: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "rms_energy_db": round(self.rms_energy_db, 1),
            "peak_amplitude": round(self.peak_amplitude, 3),
            "estimated_snr_db": round(self.estimated_snr_db, 1),
            "dominant_freq_hz": round(self.dominant_freq_hz, 1),
            "spectral_flatness": round(self.spectral_flatness, 3),
            "is_clipping": self.is_clipping,
            "is_silence": self.is_silence,
        }


def compute_live_signal_metrics(signal: np.ndarray, fs: int = 16000) -> LiveSignalMetrics:
    """Compute physical signal diagnostics for live microphone validation."""
    if len(signal) == 0:
        return LiveSignalMetrics(-90.0, 0.0, 0.0, 0.0, 0.0, False, True)

    rms = float(np.sqrt(np.mean(signal**2)))
    rms_db = 20.0 * np.log10(max(1e-6, rms))
    peak = float(np.max(np.abs(signal)))

    # FFT for dominant frequency
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    peak_idx = int(np.argmax(fft_vals[1:])) + 1 if len(fft_vals) > 1 else 0
    dom_freq = float(freqs[peak_idx])

    # Spectral flatness
    fft_safe = fft_vals + 1e-12
    geom_mean = np.exp(np.mean(np.log(fft_safe)))
    arith_mean = np.mean(fft_safe)
    flatness = float(geom_mean / max(1e-12, arith_mean))

    snr_est = float(max(0.0, rms_db + 48.0))
    clipping = bool(peak >= 0.98)
    silence = bool(rms_db < -55.0)

    return LiveSignalMetrics(
        rms_energy_db=rms_db,
        peak_amplitude=peak,
        estimated_snr_db=snr_est,
        dominant_freq_hz=dom_freq,
        spectral_flatness=flatness,
        is_clipping=clipping,
        is_silence=silence,
    )
