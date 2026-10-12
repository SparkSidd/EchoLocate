"""
EchoLocate — High-Level Speaker Identifier Pipeline
===================================================
Unified coordinator orchestrating audio preprocessing, VAD,
embedding extraction, cosine similarity matching, UNKNOWN gating,
and causal temporal smoothing into a single clean interface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .encoders.base import SpeakerEncoder
from .enrollment.database import SpeakerEnrollmentDB
from .matching.matcher import CosineSpeakerMatcher
from .preprocessing.audio_processor import AudioProcessor
from .preprocessing.vad import BaseVAD, EnergyVAD
from .temporal.tracker import TemporalSpeakerTracker
from .types import AttributionStatus, SpeakerAttributionResult, SpeakerProfile

logger = logging.getLogger(__name__)


class SpeakerIdentifier:
    """
    Primary user-facing class for isolated speaker recognition in EchoLocate.
    """

    def __init__(
        self,
        encoder: SpeakerEncoder,
        enrollment_db: Optional[SpeakerEnrollmentDB] = None,
        matcher: Optional[CosineSpeakerMatcher] = None,
        preprocessor: Optional[AudioProcessor] = None,
        vad: Optional[BaseVAD] = None,
        enable_temporal_smoothing: bool = True,
        unknown_threshold: float = 0.65,
    ):
        self.encoder = encoder
        self.enrollment_db = enrollment_db if enrollment_db is not None else SpeakerEnrollmentDB()
        self.matcher = matcher if matcher is not None else CosineSpeakerMatcher(default_threshold=unknown_threshold)
        self.preprocessor = preprocessor if preprocessor is not None else AudioProcessor()
        self.vad = vad if vad is not None else EnergyVAD()
        self.enable_temporal_smoothing = enable_temporal_smoothing
        self.temporal_tracker = (
            TemporalSpeakerTracker() if enable_temporal_smoothing else None
        )

    def enroll(
        self,
        speaker_id: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpeakerProfile:
        """
        Enroll a speaker from raw audio.
        """
        # Preprocess
        seg = self.preprocessor.process(audio, sample_rate=sample_rate)
        if not seg.metadata.get("is_duration_valid", True):
            logger.warning(
                "Enrollment audio for '%s' is very short (%.2fs)",
                speaker_id,
                seg.duration_sec,
            )

        emb = self.encoder.embed(seg.samples, sample_rate=seg.sample_rate)
        return self.enrollment_db.enroll_embedding(
            speaker_id=speaker_id,
            embedding=emb,
            model_name=self.encoder.model_name(),
            metadata=metadata,
        )

    def identify(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        start_sec: float = 0.0,
        reset_temporal: bool = False,
    ) -> SpeakerAttributionResult:
        """
        Given an acoustic event / speech segment, identify which enrolled person produced it,
        while strictly preserving an explicit UNKNOWN state.
        """
        if reset_temporal and self.temporal_tracker is not None:
            self.temporal_tracker.reset()

        # 1. Deterministic Preprocessing (Mono, 16kHz, DC removal, Peak Norm)
        seg = self.preprocessor.process(audio, sample_rate=sample_rate, start_sec=start_sec)
        segment_end = start_sec + seg.duration_sec

        # 2. VAD & Duration Validation
        if not seg.metadata.get("is_duration_valid", True):
            res = SpeakerAttributionResult(
                speaker_id="UNKNOWN",
                status=AttributionStatus.INSUFFICIENT_AUDIO,
                confidence=0.0,
                similarity=-1.0,
                threshold=self.matcher.default_threshold,
                embedding_model=self.encoder.model_name(),
                segment_start=start_sec,
                segment_end=segment_end,
                speech_ratio=0.0,
                metadata={"reason": "audio_duration_below_minimum"},
            )
            if self.temporal_tracker is not None:
                return self.temporal_tracker.update(res)
            return res

        vad_mask = self.vad.detect(seg.samples, sample_rate=seg.sample_rate)
        speech_ratio = float(np.mean(vad_mask)) if len(vad_mask) > 0 else 1.0

        if speech_ratio < 0.08:
            res = SpeakerAttributionResult(
                speaker_id="UNKNOWN",
                status=AttributionStatus.SILENCE,
                confidence=0.0,
                similarity=-1.0,
                threshold=self.matcher.default_threshold,
                embedding_model=self.encoder.model_name(),
                segment_start=start_sec,
                segment_end=segment_end,
                speech_ratio=speech_ratio,
                metadata={"reason": "vad_classified_as_silence"},
            )
            if self.temporal_tracker is not None:
                return self.temporal_tracker.update(res)
            return res

        # 3. Model-Agnostic Speaker Embedding Extraction
        embedding = self.encoder.embed(seg.samples, sample_rate=seg.sample_rate)

        # 4. Cosine Similarity Matching & UNKNOWN Gating
        enrolled_ids, centroids = self.enrollment_db.get_all_centroids()
        enrolled_embs = self.enrollment_db.get_all_embeddings()
        instantaneous_result = self.matcher.match(
            query_embedding=embedding,
            enrolled_ids=enrolled_ids,
            enrolled_centroids=centroids,
            enrolled_embeddings=enrolled_embs,
            model_name=self.encoder.model_name(),
            segment_start=start_sec,
            segment_end=segment_end,
            speech_ratio=speech_ratio,
        )

        # 5. Temporal Smoothing & Anti-flicker Hysteresis Filter
        if self.enable_temporal_smoothing and self.temporal_tracker is not None:
            return self.temporal_tracker.update(instantaneous_result)

        return instantaneous_result
