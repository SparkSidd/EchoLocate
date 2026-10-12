"""
EchoLocate — Distributed Clock Synchronization Engine
======================================================
Implements two-way Network Time Protocol (NTP) style clock probing, linear
clock model estimation (offset b, drift a in ppm), RTT jitter filtering,
and synchronization confidence estimation.
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SyncProbeRecord:
    """NTP-style 4-timestamp probe record."""
    t0_client_send_ms: float
    t1_server_recv_ms: float
    t2_server_send_ms: float
    t3_client_recv_ms: float
    rtt_ms: float
    offset_estimate_ms: float


@dataclass
class ClockSyncState:
    """Estimated linear clock synchronization relationship: t_laptop = a * t_phone + b."""
    device_id: str
    clock_offset_ms: float = 0.0          # Intercept b
    clock_drift_ppm: float = 0.0          # Drift: (a - 1.0) * 1e6
    mean_rtt_ms: float = 0.0
    jitter_ms: float = 0.0
    sync_confidence_pct: float = 0.0
    total_probes: int = 0
    is_synchronized: bool = False
    last_updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "clock_offset_ms": round(self.clock_offset_ms, 3),
            "clock_drift_ppm": round(self.clock_drift_ppm, 2),
            "mean_rtt_ms": round(self.mean_rtt_ms, 2),
            "jitter_ms": round(self.jitter_ms, 3),
            "sync_confidence_pct": round(self.sync_confidence_pct, 1),
            "total_probes": self.total_probes,
            "is_synchronized": self.is_synchronized,
            "last_updated_at": self.last_updated_at,
        }


class DistributedClockSynchronizer:
    """Estimates and tracks clock offset, drift, and network jitter for remote clients."""

    def __init__(self, history_len: int = 50, max_rtt_filter_ms: float = 150.0) -> None:
        self.history_len = history_len
        self.max_rtt_filter_ms = max_rtt_filter_ms
        self._probes: dict[str, collections.deque[SyncProbeRecord]] = {}

    def process_probe(
        self,
        device_id: str,
        t0_client_send_ms: float,
        t1_server_recv_ms: float,
        t2_server_send_ms: float,
        t3_client_recv_ms: float,
    ) -> ClockSyncState:
        """Process 4-timestamp probe and update linear clock model."""
        if device_id not in self._probes:
            self._probes[device_id] = collections.deque(maxlen=self.history_len)

        # Standard NTP Calculations
        # RTT = (t3 - t0) - (t2 - t1)
        rtt = (t3_client_recv_ms - t0_client_send_ms) - (t2_server_send_ms - t1_server_recv_ms)
        rtt = max(0.1, rtt)

        # Offset = ((t1 - t0) + (t2 - t3)) / 2
        offset = ((t1_server_recv_ms - t0_client_send_ms) + (t2_server_send_ms - t3_client_recv_ms)) / 2.0

        # Jitter filter: reject probes with excessive RTT
        if rtt <= self.max_rtt_filter_ms or len(self._probes[device_id]) < 5:
            record = SyncProbeRecord(
                t0_client_send_ms=t0_client_send_ms,
                t1_server_recv_ms=t1_server_recv_ms,
                t2_server_send_ms=t2_server_send_ms,
                t3_client_recv_ms=t3_client_recv_ms,
                rtt_ms=rtt,
                offset_estimate_ms=offset,
            )
            self._probes[device_id].append(record)

        return self.compute_sync_state(device_id)

    def compute_sync_state(self, device_id: str) -> ClockSyncState:
        """Fit linear clock regression and compute jitter statistics."""
        records = self._probes.get(device_id, collections.deque())
        if not records:
            return ClockSyncState(device_id=device_id)

        rtts = np.array([r.rtt_ms for r in records], dtype=np.float64)
        offsets = np.array([r.offset_estimate_ms for r in records], dtype=np.float64)
        client_times = np.array([r.t0_client_send_ms for r in records], dtype=np.float64)

        mean_rtt = float(np.mean(rtts))
        jitter = float(np.std(offsets)) if len(offsets) > 1 else 0.0

        # Weighted median / robust offset estimation
        median_offset = float(np.median(offsets))

        # Linear drift estimation: fit t_laptop = a * t_phone + b
        drift_ppm = 0.0
        if len(client_times) >= 6:
            # Shift time to prevent numerical instability
            t_rel = (client_times - client_times[0]) * 1e-3  # seconds
            y_rel = offsets - offsets[0]                      # ms
            if np.ptp(t_rel) > 2.0:
                slope_ms_per_s, _ = np.polyfit(t_rel, y_rel, deg=1)
                # 1 ms/s = 1000 ppm
                drift_ppm = float(slope_ms_per_s * 1000.0)

        # Confidence: high probe count, low jitter, low RTT -> higher confidence
        conf = 100.0 * np.exp(-jitter / 15.0) * np.exp(-mean_rtt / 100.0) * min(1.0, len(records) / 10.0)
        conf = float(np.clip(conf, 5.0, 98.0))

        return ClockSyncState(
            device_id=device_id,
            clock_offset_ms=median_offset,
            clock_drift_ppm=drift_ppm,
            mean_rtt_ms=mean_rtt,
            jitter_ms=jitter,
            sync_confidence_pct=conf,
            total_probes=len(records),
            is_synchronized=(len(records) >= 5 and jitter < 20.0),
            last_updated_at=time.time(),
        )

    def translate_client_time_to_server(self, device_id: str, client_time_ms: float) -> float:
        """Translate remote client timestamp to aligned server local time."""
        state = self.compute_sync_state(device_id)
        # t_server = t_client + offset_ms
        return client_time_ms + state.clock_offset_ms
