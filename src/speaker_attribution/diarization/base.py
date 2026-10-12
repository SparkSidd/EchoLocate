"""
EchoLocate — Diarization Backend Interface & Implementations
============================================================
Optional multi-speaker temporal segmentation and diarization layer.
Provides lightweight energy/continuity diarization and hooks for Pyannote Community-1.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..preprocessing.vad import EnergyVAD

logger = logging.getLogger(__name__)


@dataclass
class DiarizationSegment:
    """Represents a continuous speech turn attributed to a speaker."""
    start_sec: float
    end_sec: float
    speaker_id: str
    confidence: float
    metadata: Dict[str, Any] = None

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


class DiarizationBackend(ABC):
    """Abstract interface for speaker diarization engines."""

    @abstractmethod
    def diarize(self, audio: np.ndarray, sample_rate: int = 16000) -> List[DiarizationSegment]:
        """Process full audio track and output chronological speaker turns."""
        pass


class LightweightEnergyDiarizer(DiarizationBackend):
    """
    Lightweight, zero-dependency diarizer combining VAD segmentation
    with sliding speaker attribution matching.
    """

    def __init__(
        self,
        identifier: Any = None,
        min_speech_duration_sec: float = 0.5,
        min_silence_duration_sec: float = 0.3,
    ):
        self.identifier = identifier
        self.vad = EnergyVAD()
        self.min_speech_duration_sec = min_speech_duration_sec
        self.min_silence_duration_sec = min_silence_duration_sec

    def diarize(self, audio: np.ndarray, sample_rate: int = 16000) -> List[DiarizationSegment]:
        intervals = self.vad.get_speech_segments(
            audio,
            sample_rate=sample_rate,
            min_speech_duration_sec=self.min_speech_duration_sec,
            min_silence_duration_sec=self.min_silence_duration_sec,
        )

        segments: List[DiarizationSegment] = []
        for start_sec, end_sec in intervals:
            start_idx = int(round(start_sec * sample_rate))
            end_idx = int(round(end_sec * sample_rate))
            chunk = audio[start_idx:end_idx]

            if self.identifier is not None:
                res = self.identifier.identify(chunk, sample_rate=sample_rate, start_sec=start_sec)
                speaker_id = res.speaker_id
                conf = res.confidence
            else:
                speaker_id = "SPEAKER_00"
                conf = 1.0

            segments.append(
                DiarizationSegment(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    speaker_id=speaker_id,
                    confidence=conf,
                )
            )

        return segments


class PyannoteCommunity1Backend(DiarizationBackend):
    """
    Optional adapter for pyannote/speaker-diarization-community-1.
    Gracefully degrades if pyannote is not installed or unauthenticated.
    """

    def __init__(self, auth_token: Optional[str] = None):
        self.auth_token = auth_token
        self._pipeline = None
        self._is_available = False

        try:
            import pyannote.audio
            from pyannote.audio import Pipeline
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                use_auth_token=auth_token,
            )
            self._is_available = True
            logger.info("Pyannote Community-1 pipeline loaded successfully.")
        except Exception as e:
            logger.info(
                "Pyannote Community-1 unavailable (%s). Falling back to LightweightEnergyDiarizer.", e
            )

    @property
    def is_available(self) -> bool:
        return self._is_available

    def diarize(self, audio: np.ndarray, sample_rate: int = 16000) -> List[DiarizationSegment]:
        if not self._is_available or self._pipeline is None:
            # Fallback
            fallback = LightweightEnergyDiarizer()
            return fallback.diarize(audio, sample_rate=sample_rate)

        import torch
        tensor_audio = torch.from_numpy(audio).float().unsqueeze(0)
        diarization = self._pipeline({"waveform": tensor_audio, "sample_rate": sample_rate})

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                DiarizationSegment(
                    start_sec=turn.start,
                    end_sec=turn.end,
                    speaker_id=speaker,
                    confidence=0.9,
                )
            )
        return segments
