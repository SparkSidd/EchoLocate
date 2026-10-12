"""
EchoLocate — Real Speaker Demonstration Package
================================================
Provides live microphone recording, enrollment quality checks, interactive
identification, UNKNOWN rejection verification, and blind testing.
"""

from .quality_checker import EnrollmentQualityChecker, QualityCheckResult
from .recorder import MicrophoneRecorder
from .session import DemonstrationSession

__all__ = [
    "EnrollmentQualityChecker",
    "QualityCheckResult",
    "MicrophoneRecorder",
    "DemonstrationSession",
]
