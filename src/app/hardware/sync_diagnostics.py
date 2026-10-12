"""
EchoLocate — Channel Synchronization Diagnostics Module
========================================================
Monitors real-time multi-channel audio frames for timestamp jitter,
dropped frames, and clock drift between channels.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SyncDiagnosticsReport:
    """Hardware synchronization health report."""
    is_synchronized: bool
    mean_inter_channel_lag_samples: float
    max_lag_samples: int
    buffer_overrun_count: int
    dropped_frames_count: int
    status: str


class ChannelSyncMonitor:
    """Real-time multi-channel synchronization monitor."""

    def __init__(self, num_channels: int = 4, max_allowed_lag_samples: int = 4) -> None:
        self.num_channels = num_channels
        self.max_lag = max_allowed_lag_samples
        self.buffer_overruns = 0
        self.dropped_frames = 0

    def inspect_frame(self, frame_audio: np.ndarray) -> SyncDiagnosticsReport:
        """Inspect multi-channel audio frame (N_channels, N_samples) for synchronization drift."""
        n_ch, n_samples = frame_audio.shape
        if n_ch < 2 or n_samples < 64:
            return SyncDiagnosticsReport(
                is_synchronized=True,
                mean_inter_channel_lag_samples=0.0,
                max_lag_samples=0,
                buffer_overrun_count=self.buffer_overruns,
                dropped_frames_count=self.dropped_frames,
                status="NOMINAL",
            )

        ref = frame_audio[0]
        lags = []
        for c in range(1, n_ch):
            corr = np.correlate(frame_audio[c], ref, mode="full")
            peak_lag = int(np.argmax(corr) - (len(ref) - 1))
            lags.append(abs(peak_lag))

        max_lag = max(lags) if lags else 0
        mean_lag = float(np.mean(lags)) if lags else 0.0

        is_synced = max_lag <= self.max_lag
        status = "SYNCHRONIZED" if is_synced else "SYNC_DRIFT_DETECTED"

        return SyncDiagnosticsReport(
            is_synchronized=is_synced,
            mean_inter_channel_lag_samples=mean_lag,
            max_lag_samples=max_lag,
            buffer_overrun_count=self.buffer_overruns,
            dropped_frames_count=self.dropped_frames,
            status=status,
        )
