"""
EchoLocate — Real Dataset Metadata & Scene Specification Schemas
================================================================
Defines structured schemas for physical recording environments, microphone
array specifications, and acoustic scene ground-truth metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MicrophoneArraySpec:
    """Microphone array physical and geometry specification."""
    name: str
    topology: str  # "planar_4mic", "planar_dicit_15mic", "circular_8mic"
    num_channels: int
    nominal_positions_2d: np.ndarray
    measured_positions_2d: np.ndarray
    estimated_positions_2d: np.ndarray | None = None
    preregistered_subindices: list[int] | None = None
    height_m: float = 1.2
    sampling_rate_hz: int = 16000


@dataclass
class AcousticSceneMetadata:
    """Complete metadata description for an individual real/recorded acoustic scene."""
    scene_id: str
    dataset_name: str
    room_id: str
    room_dimensions_m: list[float]
    estimated_rt60_s: float
    measurement_method: str  # "schroeder_direct", "decay_fit", "interpolated"
    true_source_position_3d: list[float]
    source_signal_class: str
    snr_estimate_db: float
    array_spec: MicrophoneArraySpec
    is_calibration_split: bool = False
    is_validation_split: bool = False
    is_blind_test_split: bool = False
    notes: str = ""


@dataclass
class RealDatasetManifest:
    """Collection manifest summarizing an acoustic localization corpus."""
    corpus_name: str
    total_recordings: int
    num_rooms: int
    num_speakers: int
    signal_classes: list[str]
    sample_rate_hz: int
    creation_date: str
    sub_array_rules: dict[str, Any] = field(default_factory=dict)
