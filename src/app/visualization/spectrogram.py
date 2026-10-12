"""
EchoLocate — Real-Time Spectrogram & Harmonic Analysis Visualizer
=================================================================
Computes Short-Time Fourier Transform (STFT) linear and log spectrogram matrices,
and detects fundamental pitch frequencies (F0) and harmonic ridges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

from src.app.visualization.provenance import VisualProvenanceMetadata

logger = logging.getLogger(__name__)


@dataclass
class SpectrogramPayload:
    """Spectrogram matrix and spectral analysis payload for web visualization."""
    channel_index: int
    time_bins_ms: list[float]
    frequency_bins_hz: list[float]
    spectrogram_db: list[list[float]]
    dominant_pitch_hz: float | None
    harmonic_peaks_hz: list[float]
    provenance: VisualProvenanceMetadata


def compute_channel_spectrogram(
    channel_audio: np.ndarray,
    fs: int = 16000,
    nperseg: int = 256,
    noverlap: int = 192,
    channel_idx: int = 0,
    frame_id: int = 0,
    provenance: VisualProvenanceMetadata | None = None,
) -> SpectrogramPayload:
    """Compute 2D STFT spectrogram matrix (dB) and detect harmonic structure."""
    prov = provenance or VisualProvenanceMetadata.live(frame_id)

    f, t, sxx = signal.spectrogram(channel_audio, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="spectrum")
    sxx_db = 10.0 * np.log10(np.maximum(1e-9, sxx))

    # Normalize between 0 and 1 for heatmap display
    min_db, max_db = float(np.min(sxx_db)), float(np.max(sxx_db))
    norm_spec = ((sxx_db - min_db) / max(1e-4, max_db - min_db)).tolist()

    # Detect dominant harmonic pitch peak in speech band (80 - 400 Hz)
    speech_mask = (f >= 80) & (f <= 400)
    mean_spec = np.mean(sxx, axis=1)

    dom_pitch = None
    harmonics = []
    if np.any(speech_mask):
        speech_freqs = f[speech_mask]
        speech_powers = mean_spec[speech_mask]
        peak_idx = int(np.argmax(speech_powers))
        if speech_powers[peak_idx] > 1e-6:
            dom_pitch = float(speech_freqs[peak_idx])
            harmonics = [dom_pitch * k for k in [1, 2, 3, 4] if dom_pitch * k <= fs / 2]

    return SpectrogramPayload(
        channel_index=channel_idx,
        time_bins_ms=(t * 1000.0).tolist(),
        frequency_bins_hz=f.tolist(),
        spectrogram_db=norm_spec,
        dominant_pitch_hz=dom_pitch,
        harmonic_peaks_hz=harmonics,
        provenance=prov,
    )
