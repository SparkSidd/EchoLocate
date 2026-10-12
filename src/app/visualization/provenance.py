"""
EchoLocate — Visualization Provenance & Metadata Container
==========================================================
Ensures all visual payloads (waveforms, spectrograms, GCC curves, heatmaps,
trajectories, latency figures) carry immutable provenance attribution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VisualProvenanceType(str, Enum):
    """Source origin categories for all visualization objects."""
    LIVE_COMPUTED = "LIVE_COMPUTED"
    RECORDED_DATA = "RECORDED_DATA"
    SIMULATION = "SIMULATION"
    BENCHMARK_RESULT = "BENCHMARK_RESULT"


@dataclass
class VisualProvenanceMetadata:
    """Provenance container attached to every visualization payload."""
    source_type: VisualProvenanceType
    source_name: str
    frame_id: int
    timestamp_s: float = field(default_factory=time.time)
    computed_at_ms: float = field(default_factory=lambda: time.time() * 1000.0)
    ground_truth_available: bool = False
    badge_color: str = "#22c55e"  # Default green for live
    is_live_stream: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "source_name": self.source_name,
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "computed_at_ms": self.computed_at_ms,
            "ground_truth_available": self.ground_truth_available,
            "badge_color": self.badge_color,
            "is_live_stream": self.is_live_stream,
        }

    @classmethod
    def live(cls, frame_id: int, device_name: str = "Hardware Array") -> VisualProvenanceMetadata:
        return cls(
            source_type=VisualProvenanceType.LIVE_COMPUTED,
            source_name=f"Live Hardware ({device_name})",
            frame_id=frame_id,
            ground_truth_available=False,
            badge_color="#22c55e",  # Green
            is_live_stream=True,
        )

    @classmethod
    def recorded(cls, frame_id: int, dataset_name: str = "LOCATA", sequence_id: str = "Recording_01") -> VisualProvenanceMetadata:
        return cls(
            source_type=VisualProvenanceType.RECORDED_DATA,
            source_name=f"{dataset_name} ({sequence_id})",
            frame_id=frame_id,
            ground_truth_available=True,
            badge_color="#3b82f6",  # Blue
            is_live_stream=False,
        )

    @classmethod
    def simulation(cls, frame_id: int, scenario_name: str = "Acoustic Simulation") -> VisualProvenanceMetadata:
        return cls(
            source_type=VisualProvenanceType.SIMULATION,
            source_name=f"PyRoomAcoustics ({scenario_name})",
            frame_id=frame_id,
            ground_truth_available=True,
            badge_color="#f97316",  # Orange
            is_live_stream=False,
        )

    @classmethod
    def benchmark(cls, frame_id: int, exp_id: str = "Experiment 5 Benchmark") -> VisualProvenanceMetadata:
        return cls(
            source_type=VisualProvenanceType.BENCHMARK_RESULT,
            source_name=f"Research Archive ({exp_id})",
            frame_id=frame_id,
            ground_truth_available=True,
            badge_color="#a855f7",  # Purple
            is_live_stream=False,
        )
