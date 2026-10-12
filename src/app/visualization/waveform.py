"""
EchoLocate — Multichannel Waveform & Arrival Delay Visualizer
============================================================
Processes multichannel audio frames into synchronized waveforms, calculates
per-channel RMS levels (dB), detects clipping, and models inter-mic arrival delays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.app.visualization.provenance import VisualProvenanceMetadata

logger = logging.getLogger(__name__)


@dataclass
class ChannelWaveformData:
    """Waveform display metrics for a single microphone channel."""
    channel_index: int
    channel_name: str
    samples: list[float]
    rms_energy_db: float
    is_clipped: bool
    peak_amplitude: float


@dataclass
class MultichannelWaveformPayload:
    """Synchronized 4-channel audio display packet."""
    frame_id: int
    sample_rate: int
    duration_ms: float
    time_axis_ms: list[float]
    channels: list[ChannelWaveformData]
    inter_channel_delays: dict[str, float]
    provenance: VisualProvenanceMetadata


def process_multichannel_waveform(
    frame_audio: np.ndarray,
    fs: int = 16000,
    frame_id: int = 0,
    downsample_factor: int = 4,
    provenance: VisualProvenanceMetadata | None = None,
) -> MultichannelWaveformPayload:
    """Process raw multichannel audio array (N_channels, N_samples) for browser display."""
    n_ch, n_s = frame_audio.shape
    t_ms = (np.arange(0, n_s, downsample_factor) / fs * 1000.0).tolist()
    duration_ms = float((n_s / fs) * 1000.0)

    prov = provenance or VisualProvenanceMetadata.live(frame_id)
    ch_data = []

    for c in range(n_ch):
        raw_c = frame_audio[c]
        sub_samples = raw_c[::downsample_factor].tolist()
        rms = float(np.sqrt(np.mean(raw_c**2)))
        rms_db = float(20.0 * np.log10(max(1e-9, rms)))
        peak = float(np.max(np.abs(raw_c)))
        is_clip = bool(peak >= 0.98)

        ch_data.append(
            ChannelWaveformData(
                channel_index=c,
                channel_name=f"MIC {c + 1}",
                samples=sub_samples,
                rms_energy_db=rms_db,
                is_clipped=is_clip,
                peak_amplitude=peak,
            )
        )

    # Compute inter-channel arrival delay peaks (relative to mic 0)
    delays = {}
    if n_ch >= 2:
        ref = frame_audio[0]
        for c in range(1, n_ch):
            corr = np.correlate(frame_audio[c], ref, mode="full")
            lag = int(np.argmax(corr) - (len(ref) - 1))
            delays[f"Pair (0, {c})"] = float(lag / fs * 1000.0)  # ms

    return MultichannelWaveformPayload(
        frame_id=frame_id,
        sample_rate=fs,
        duration_ms=duration_ms,
        time_axis_ms=t_ms,
        channels=ch_data,
        inter_channel_delays=delays,
        provenance=prov,
    )
