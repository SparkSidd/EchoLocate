"""
EchoLocate — Temporal Speaker Tracking & Hysteresis Filter
==========================================================
Maintains multi-frame score aggregation and hysteresis state to prevent
rapid flickering while accurately exposing instantaneous vs smoothed confidence.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from ..types import AttributionStatus, CandidateScore, SpeakerAttributionResult


class TemporalSpeakerTracker:
    """
    Causal temporal smoother and anti-flicker hysteresis filter for speaker attribution.
    Prevents single-frame dropouts from corrupting confident continuous tracks.
    """

    def __init__(
        self,
        ema_alpha: float = 0.4,
        window_size: int = 5,
        switch_confirmation_frames: int = 3,
        unknown_grace_frames: int = 2,
    ):
        self.ema_alpha = ema_alpha
        self.window_size = window_size
        self.switch_confirmation_frames = switch_confirmation_frames
        self.unknown_grace_frames = unknown_grace_frames

        # State
        self._history: deque[SpeakerAttributionResult] = deque(maxlen=window_size)
        self._active_speaker_id: str = "UNKNOWN"
        self._active_status: AttributionStatus = AttributionStatus.UNKNOWN
        self._smoothed_scores: Dict[str, float] = {}
        self._consecutive_candidate: Optional[str] = None
        self._consecutive_count: int = 0
        self._unknown_dropout_count: int = 0
        self._supporting_frame_count: int = 0

    def reset(self) -> None:
        """Reset all temporal history and trackers."""
        self._history.clear()
        self._active_speaker_id = "UNKNOWN"
        self._active_status = AttributionStatus.UNKNOWN
        self._smoothed_scores.clear()
        self._consecutive_candidate = None
        self._consecutive_count = 0
        self._unknown_dropout_count = 0
        self._supporting_frame_count = 0

    def update(self, instantaneous: SpeakerAttributionResult) -> SpeakerAttributionResult:
        """
        Ingest a new instantaneous speaker attribution result, update smoothed state,
        apply hysteresis, and return a temporally stable attribution event.
        """
        self._history.append(instantaneous)
        inst_speaker = instantaneous.speaker_id
        inst_sim = instantaneous.similarity
        inst_status = instantaneous.status

        # 1. Update Exponential Moving Average of candidate scores
        for alt in instantaneous.alternatives:
            s_id = alt.speaker_id
            sim = alt.similarity
            if s_id in self._smoothed_scores:
                self._smoothed_scores[s_id] = (
                    self.ema_alpha * sim + (1.0 - self.ema_alpha) * self._smoothed_scores[s_id]
                )
            else:
                self._smoothed_scores[s_id] = sim

        # 2. Hysteresis State Machine
        if self._active_speaker_id == "UNKNOWN":
            # Currently UNKNOWN: require N consecutive matching frames to establish identity
            if inst_status == AttributionStatus.KNOWN:
                if inst_speaker == self._consecutive_candidate:
                    self._consecutive_count += 1
                else:
                    self._consecutive_candidate = inst_speaker
                    self._consecutive_count = 1

                if self._consecutive_count >= self.switch_confirmation_frames:
                    self._active_speaker_id = inst_speaker
                    self._active_status = AttributionStatus.KNOWN
                    self._supporting_frame_count = self._consecutive_count
                    self._unknown_dropout_count = 0
            else:
                self._consecutive_candidate = None
                self._consecutive_count = 0
                self._supporting_frame_count = 0

        else:
            # Currently KNOWN as self._active_speaker_id
            if inst_speaker == self._active_speaker_id and inst_status == AttributionStatus.KNOWN:
                # Direct confirmation
                self._supporting_frame_count += 1
                self._consecutive_candidate = None
                self._consecutive_count = 0
                self._unknown_dropout_count = 0
            elif inst_status == AttributionStatus.UNKNOWN:
                # Instantaneous UNKNOWN: check grace frames before tearing down track
                self._unknown_dropout_count += 1
                if self._unknown_dropout_count > self.unknown_grace_frames:
                    self._active_speaker_id = "UNKNOWN"
                    self._active_status = AttributionStatus.UNKNOWN
                    self._supporting_frame_count = 0
                    self._consecutive_candidate = None
                    self._consecutive_count = 0
            else:
                # Proposed switch to a different known speaker
                if inst_speaker == self._consecutive_candidate:
                    self._consecutive_count += 1
                else:
                    self._consecutive_candidate = inst_speaker
                    self._consecutive_count = 1

                if self._consecutive_count >= self.switch_confirmation_frames:
                    # Confirm identity switch
                    self._active_speaker_id = inst_speaker
                    self._active_status = AttributionStatus.KNOWN
                    self._supporting_frame_count = self._consecutive_count
                    self._unknown_dropout_count = 0

        # 3. Compute Temporal Stability Metric in [0.0, 1.0]
        if len(self._history) > 1:
            matching_frames = sum(1 for h in self._history if h.speaker_id == self._active_speaker_id)
            stability = float(matching_frames / len(self._history))
        else:
            stability = 1.0 if self._active_status == AttributionStatus.KNOWN else 0.5

        # 4. Construct Smoothed Alternatives List
        sorted_smoothed = sorted(
            self._smoothed_scores.items(), key=lambda item: item[1], reverse=True
        )
        smoothed_alts = [
            CandidateScore(speaker_id=s_id, similarity=float(score), rank=r)
            for r, (s_id, score) in enumerate(sorted_smoothed, start=1)
        ]

        active_sim = (
            self._smoothed_scores.get(self._active_speaker_id, inst_sim)
            if self._active_speaker_id != "UNKNOWN"
            else inst_sim
        )

        # Smoothed confidence balances instantaneous confidence with stability
        smoothed_confidence = float(
            0.7 * instantaneous.confidence + 0.3 * stability
            if self._active_status == AttributionStatus.KNOWN
            else instantaneous.confidence
        )

        return SpeakerAttributionResult(
            speaker_id=self._active_speaker_id,
            status=self._active_status,
            confidence=max(0.0, min(1.0, smoothed_confidence)),
            similarity=float(active_sim),
            threshold=instantaneous.threshold,
            embedding_model=instantaneous.embedding_model,
            segment_start=instantaneous.segment_start,
            segment_end=instantaneous.segment_end,
            speech_ratio=instantaneous.speech_ratio,
            temporal_stability=stability,
            supporting_frames=self._supporting_frame_count,
            alternatives=smoothed_alts if smoothed_alts else instantaneous.alternatives,
            metadata={
                **instantaneous.metadata,
                "instantaneous_speaker": inst_speaker,
                "instantaneous_similarity": float(inst_sim),
                "unknown_dropout_count": self._unknown_dropout_count,
                "history_length": len(self._history),
            },
        )
