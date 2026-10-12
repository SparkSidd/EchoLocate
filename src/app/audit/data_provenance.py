"""
EchoLocate — Data Provenance Tracking System
============================================
Ensures every audio frame, intermediate representation, and localized coordinate
carries an explicit, verifiable provenance label to guarantee scientific integrity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DataProvenance(str, Enum):
    """Verifiable source origin for all acoustic data processed by EchoLocate."""
    LIVE_HARDWARE = "LIVE_HARDWARE"
    REAL_RECORDED_DATASET = "REAL_RECORDED_DATASET"
    SYNTHETIC_SIMULATION = "SYNTHETIC_SIMULATION"
    PRECOMPUTED_RESEARCH_REPLAY = "PRECOMPUTED_RESEARCH_REPLAY"


@dataclass
class ProvenanceMetadata:
    """Provenance metadata container attached to every localization estimate."""
    provenance: DataProvenance
    source_name: str
    is_live_stream: bool
    is_synthetic: bool
    ground_truth_accessible_during_inference: bool = False
    future_frames_accessible: bool = False
    display_badge_color: str = "green"

    @classmethod
    def live_hardware(cls, device_name: str) -> ProvenanceMetadata:
        return cls(
            provenance=DataProvenance.LIVE_HARDWARE,
            source_name=f"Microphone Device: {device_name}",
            is_live_stream=True,
            is_synthetic=False,
            display_badge_color="#22c55e",  # Green
        )

    @classmethod
    def real_recorded(cls, dataset_name: str, sequence_id: str) -> ProvenanceMetadata:
        return cls(
            provenance=DataProvenance.REAL_RECORDED_DATASET,
            source_name=f"{dataset_name} ({sequence_id})",
            is_live_stream=False,
            is_synthetic=False,
            display_badge_color="#3b82f6",  # Blue
        )

    @classmethod
    def synthetic_simulation(cls, scenario_name: str) -> ProvenanceMetadata:
        return cls(
            provenance=DataProvenance.SYNTHETIC_SIMULATION,
            source_name=f"PyRoomAcoustics Sim ({scenario_name})",
            is_live_stream=False,
            is_synthetic=True,
            display_badge_color="#f97316",  # Orange
        )

    @classmethod
    def archival_replay(cls, experiment_id: str) -> ProvenanceMetadata:
        return cls(
            provenance=DataProvenance.PRECOMPUTED_RESEARCH_REPLAY,
            source_name=f"Archival Log ({experiment_id})",
            is_live_stream=False,
            is_synthetic=False,
            display_badge_color="#a855f7",  # Purple
        )
