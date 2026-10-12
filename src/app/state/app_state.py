"""
EchoLocate — Global Application State & Metrics Manager
=======================================================
Tracks UI mode, active algorithms, room geometry, real-time latency statistics,
and live trajectory points for visualization.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.app.audit.data_provenance import DataProvenance, ProvenanceMetadata
from src.app.engine.pipeline_worker import LocalizationResultPacket

logger = logging.getLogger(__name__)


@dataclass
class ApplicationState:
    """Thread-safe global state for the EchoLocate interactive prototype."""
    active_mode: str = "LIVE"  # "LIVE", "PLAYBACK", "COMPARISON", "SIMULATION"
    active_method: str = "M2_Raw_SRP"
    active_provenance: ProvenanceMetadata = field(
        default_factory=lambda: ProvenanceMetadata.live_hardware("Default 4-Mic Array")
    )
    room_dimensions_m: list[float] = field(default_factory=lambda: [10.0, 6.0, 3.0])
    mic_positions_2d: list[list[float]] = field(
        default_factory=lambda: [[2.0, 1.5], [8.0, 1.5], [5.0, 4.5], [5.0, 1.5]]
    )
    is_running: bool = False
    is_paused: bool = False
    playback_speed: float = 1.0
    tracking_mode: str = "CAUSAL_SMOOTHED"  # "RAW", "CAUSAL_SMOOTHED", "KALMAN"

    # Latest frame results
    latest_result: LocalizationResultPacket | None = None
    trajectory_history: list[dict[str, Any]] = field(default_factory=list)
    max_history_len: int = 50

    # Latency tracking
    latencies_ms: list[float] = field(default_factory=list)
    dropped_frames_total: int = 0
    total_frames_processed: int = 0

    def __post_init__(self) -> None:
        self.lock = threading.Lock()

    def update_result(self, packet: LocalizationResultPacket) -> None:
        """Update active state with new localized output packet."""
        with self.lock:
            self.latest_result = packet
            self.total_frames_processed += 1
            self.latencies_ms.append(packet.processing_latency_ms)
            if len(self.latencies_ms) > 100:
                self.latencies_ms.pop(0)

            point = {
                "frame": packet.frame_index,
                "raw_x": packet.raw_estimate_xy[0],
                "raw_y": packet.raw_estimate_xy[1],
                "x": packet.smoothed_estimate_xy[0],
                "y": packet.smoothed_estimate_xy[1],
                "confidence": packet.confidence,
                "gt_x": packet.ground_truth_xy[0] if packet.ground_truth_xy else None,
                "gt_y": packet.ground_truth_xy[1] if packet.ground_truth_xy else None,
                "error_m": packet.error_m,
                "method": packet.method_id,
            }
            self.trajectory_history.append(point)
            if len(self.trajectory_history) > self.max_history_len:
                self.trajectory_history.pop(0)

    def get_latency_stats(self) -> dict[str, float]:
        """Compute rolling latency metrics."""
        with self.lock:
            if not self.latencies_ms:
                return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "rtf": 0.0}
            l_arr = np.asarray(self.latencies_ms)
            return {
                "mean_ms": float(np.mean(l_arr)),
                "median_ms": float(np.median(l_arr)),
                "p95_ms": float(np.percentile(l_arr, 95)),
                "rtf": float(np.mean(l_arr) / 200.0),
            }

    def clear_history(self) -> None:
        """Clear trajectory tracking points."""
        with self.lock:
            self.trajectory_history.clear()
            self.latencies_ms.clear()
