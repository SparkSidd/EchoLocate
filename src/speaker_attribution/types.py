"""
EchoLocate — Speaker Attribution Data Types
===========================================
Core data contracts and event models for speaker recognition and spatial fusion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class AttributionStatus(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    INSUFFICIENT_AUDIO = "insufficient_audio"
    SILENCE = "silence"


@dataclass
class AudioSegment:
    """Represents a discrete audio chunk processed by the speaker pipeline."""
    samples: np.ndarray  # Shape: (N,), 1D float32, mono
    sample_rate: int = 16000
    start_sec: float = 0.0
    end_sec: float = 0.0
    speech_ratio: float = 1.0
    is_speech: bool = True
    channel_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return len(self.samples) / self.sample_rate if self.sample_rate > 0 else 0.0


@dataclass
class CandidateScore:
    """Individual enrolled speaker candidate similarity score."""
    speaker_id: str
    similarity: float
    rank: int = 1
    raw_distance: Optional[float] = None


@dataclass
class SpeakerProfile:
    """Enrolled speaker profile containing normalized centroid and metadata."""
    speaker_id: str
    centroid: np.ndarray  # Shape: (D,), L2-normalized vector
    sample_count: int
    embedding_dim: int
    model_name: str
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        norm = np.linalg.norm(self.centroid)
        if norm > 1e-9:
            self.centroid = (self.centroid / norm).astype(np.float32)


@dataclass
class SpeakerAttributionResult:
    """
    Standard output event for speaker recognition.
    Strictly independent of any UI or visualization.
    """
    speaker_id: str
    status: AttributionStatus
    confidence: float  # Calibrated probability or margin in [0.0, 1.0]
    similarity: float  # Raw cosine similarity to top candidate in [-1.0, 1.0]
    threshold: float   # Decision boundary applied
    embedding_model: str
    segment_start: float = 0.0
    segment_end: float = 0.0
    speech_ratio: float = 1.0
    temporal_stability: float = 1.0
    supporting_frames: int = 1
    alternatives: List[CandidateScore] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "status": self.status.value if isinstance(self.status, AttributionStatus) else str(self.status),
            "confidence": float(self.confidence),
            "similarity": float(self.similarity),
            "threshold": float(self.threshold),
            "embedding_model": self.embedding_model,
            "segment_start": float(self.segment_start),
            "segment_end": float(self.segment_end),
            "speech_ratio": float(self.speech_ratio),
            "temporal_stability": float(self.temporal_stability),
            "supporting_frames": int(self.supporting_frames),
            "alternatives": [
                {"speaker_id": alt.speaker_id, "similarity": float(alt.similarity), "rank": alt.rank}
                for alt in self.alternatives
            ],
            "metadata": self.metadata,
        }


@dataclass
class LocalizationInput:
    """Minimal decoupled representation of spatial localization for fusion."""
    position_xy: Tuple[float, float]
    confidence: float
    timestamp_sec: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialFusionResult:
    """
    Unified WHO + WHERE acoustic event attribution result.
    Maintains an explicit UNKNOWN state when evidence is weak.
    """
    speaker_id: str
    status: AttributionStatus
    position_xy: Optional[Tuple[float, float]]
    voice_confidence: float
    spatial_confidence: float
    joint_confidence: float
    embedding_model: str
    temporal_stability: float = 1.0
    is_spatially_consistent: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "status": self.status.value if isinstance(self.status, AttributionStatus) else str(self.status),
            "position_xy": list(self.position_xy) if self.position_xy is not None else None,
            "voice_confidence": float(self.voice_confidence),
            "spatial_confidence": float(self.spatial_confidence),
            "joint_confidence": float(self.joint_confidence),
            "embedding_model": self.embedding_model,
            "temporal_stability": float(self.temporal_stability),
            "is_spatially_consistent": self.is_spatially_consistent,
            "metadata": self.metadata,
        }
