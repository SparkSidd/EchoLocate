"""
EchoLocate — Speaker Attribution & Spatial Recognition Module
=============================================================
Independent research subsystem for speaker verification, identification,
UNKNOWN rejection, and spatial speaker attribution.
"""

from .adapters.audio_source_adapter import AudioSourceAdapter
from .backends.multichannel.multichannel_research import (
    MultichannelComparisonResult,
    MultichannelSpeakerProcessor,
)
from .diarization.base import (
    DiarizationBackend,
    DiarizationSegment,
    LightweightEnergyDiarizer,
    PyannoteCommunity1Backend,
)
from .encoders.base import SpeakerEncoder
from .encoders.campp_encoder import CAMPPEncoder
from .encoders.ecapa_encoder import ECAPAEncoder
from .encoders.eres2netv2_encoder import ERes2NetV2Encoder
from .encoders.factory import create_encoder
from .enrollment.database import SpeakerEnrollmentDB
from .fusion.spatial_fusion import SpatialSpeakerFusion
from .identifier import SpeakerIdentifier
from .matching.calibration import CalibrationReport, ThresholdCalibrator
from .matching.matcher import CosineSpeakerMatcher
from .preprocessing.audio_processor import AudioProcessor
from .preprocessing.features import compute_fbank
from .preprocessing.vad import BaseVAD, EnergyVAD
from .temporal.tracker import TemporalSpeakerTracker
from .types import (
    AttributionStatus,
    AudioSegment,
    CandidateScore,
    LocalizationInput,
    SpatialFusionResult,
    SpeakerAttributionResult,
    SpeakerProfile,
)

__version__ = "1.0.0"

__all__ = [
    # Core types
    "AttributionStatus",
    "AudioSegment",
    "CandidateScore",
    "SpeakerProfile",
    "SpeakerAttributionResult",
    "LocalizationInput",
    "SpatialFusionResult",
    # Encoders
    "SpeakerEncoder",
    "CAMPPEncoder",
    "ERes2NetV2Encoder",
    "ECAPAEncoder",
    "create_encoder",
    # Preprocessing
    "AudioProcessor",
    "compute_fbank",
    "BaseVAD",
    "EnergyVAD",
    # Enrollment & Matching
    "SpeakerEnrollmentDB",
    "CosineSpeakerMatcher",
    "ThresholdCalibrator",
    "CalibrationReport",
    # Temporal & High-level pipeline
    "TemporalSpeakerTracker",
    "SpeakerIdentifier",
    # Adapters & Diarization
    "AudioSourceAdapter",
    "DiarizationBackend",
    "DiarizationSegment",
    "LightweightEnergyDiarizer",
    "PyannoteCommunity1Backend",
    # Multichannel & Fusion
    "MultichannelSpeakerProcessor",
    "MultichannelComparisonResult",
    "SpatialSpeakerFusion",
]
