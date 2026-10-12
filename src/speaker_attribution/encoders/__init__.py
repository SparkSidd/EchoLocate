"""
EchoLocate — Speaker Encoders Package
"""

from .base import SpeakerEncoder
from .campp_encoder import CAMPPEncoder
from .ecapa_encoder import ECAPAEncoder
from .eres2netv2_encoder import ERes2NetV2Encoder
from .factory import create_encoder
from .mock_encoder import MockAcousticEncoder

__all__ = [
    "SpeakerEncoder",
    "CAMPPEncoder",
    "ERes2NetV2Encoder",
    "ECAPAEncoder",
    "MockAcousticEncoder",
    "create_encoder",
]
