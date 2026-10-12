"""
EchoLocate — Tests: Temporal Tracking & Spatial Fusion Layer
============================================================
Validates anti-flicker hysteresis, causal temporal smoothing,
and thin spatial fusion with explicit UNKNOWN preservation.
"""

import numpy as np
import pytest

from src.speaker_attribution.fusion.spatial_fusion import SpatialSpeakerFusion
from src.speaker_attribution.temporal.tracker import TemporalSpeakerTracker
from src.speaker_attribution.types import (
    AttributionStatus,
    CandidateScore,
    LocalizationInput,
    SpeakerAttributionResult,
)


class TestTemporalTracker:
    def _make_result(self, speaker_id: str, status: AttributionStatus, confidence: float = 0.85):
        return SpeakerAttributionResult(
            speaker_id=speaker_id,
            status=status,
            confidence=confidence,
            similarity=0.85 if status == AttributionStatus.KNOWN else 0.30,
            threshold=0.65,
            embedding_model="mock",
            alternatives=[CandidateScore(speaker_id=speaker_id, similarity=0.85, rank=1)],
        )

    def test_hysteresis_establishment_and_anti_flicker(self):
        tracker = TemporalSpeakerTracker(switch_confirmation_frames=3, unknown_grace_frames=1)

        # Frame 1: Siddharth proposed (not yet confirmed)
        r1 = tracker.update(self._make_result("Siddharth", AttributionStatus.KNOWN))
        assert r1.speaker_id == "UNKNOWN"

        # Frame 2: Siddharth (count = 2)
        r2 = tracker.update(self._make_result("Siddharth", AttributionStatus.KNOWN))
        assert r2.speaker_id == "UNKNOWN"

        # Frame 3: Siddharth (count = 3 -> confirmed!)
        r3 = tracker.update(self._make_result("Siddharth", AttributionStatus.KNOWN))
        assert r3.speaker_id == "Siddharth"
        assert r3.status == AttributionStatus.KNOWN

        # Frame 4: 1 single frame of noise/dropout (within grace frames = 1)
        r4 = tracker.update(self._make_result("UNKNOWN", AttributionStatus.UNKNOWN, confidence=0.2))
        # Must maintain Siddharth track despite 1 drop frame!
        assert r4.speaker_id == "Siddharth"
        assert r4.temporal_stability < 1.0  # Stability dipped slightly

        # Frame 5: Siddharth returns
        r5 = tracker.update(self._make_result("Siddharth", AttributionStatus.KNOWN))
        assert r5.speaker_id == "Siddharth"


class TestSpatialSpeakerFusion:
    def test_fusion_with_spatially_consistent_speaker(self):
        fusion = SpatialSpeakerFusion(weight_voice=0.6, weight_spatial=0.3, weight_temporal=0.1)

        spk = SpeakerAttributionResult(
            speaker_id="Siddharth",
            status=AttributionStatus.KNOWN,
            confidence=0.90,
            similarity=0.88,
            threshold=0.65,
            embedding_model="campp",
            temporal_stability=0.95,
        )
        loc = LocalizationInput(position_xy=(2.0, 3.0), confidence=0.85, timestamp_sec=1.0)
        expected_locations = {"Siddharth": (2.1, 2.9)}

        fused = fusion.fuse(spk, loc, expected_speaker_positions=expected_locations)
        assert fused.speaker_id == "Siddharth"
        assert fused.status == AttributionStatus.KNOWN
        assert fused.is_spatially_consistent is True
        assert fused.joint_confidence > 0.75
        assert fused.position_xy == (2.0, 3.0)

    def test_fusion_preserves_explicit_unknown(self):
        fusion = SpatialSpeakerFusion()

        # Speaker is UNKNOWN, but localization located sound at (4.0, 2.5) with 0.9 confidence
        spk_unknown = SpeakerAttributionResult(
            speaker_id="UNKNOWN",
            status=AttributionStatus.UNKNOWN,
            confidence=0.30,
            similarity=0.45,
            threshold=0.65,
            embedding_model="campp",
        )
        loc = LocalizationInput(position_xy=(4.0, 2.5), confidence=0.92, timestamp_sec=2.0)

        fused = fusion.fuse(spk_unknown, loc)
        # CRITICAL: Must NEVER force an identity
        assert fused.speaker_id == "UNKNOWN"
        assert fused.status == AttributionStatus.UNKNOWN
        # BUT spatial position must still be accurately recorded
        assert fused.position_xy == (4.0, 2.5)
        assert fused.joint_confidence < 0.50

    def test_fusion_spatial_conflict_detection(self):
        fusion = SpatialSpeakerFusion(spatial_tolerance_m=1.0)

        # Voice sounds like Siddharth, but sound arrived from (6.0, 6.0) while Siddharth is at (1.0, 1.0)
        spk = SpeakerAttributionResult(
            speaker_id="Siddharth",
            status=AttributionStatus.KNOWN,
            confidence=0.75,
            similarity=0.72,
            threshold=0.65,
            embedding_model="campp",
        )
        loc = LocalizationInput(position_xy=(6.0, 6.0), confidence=0.90, timestamp_sec=3.0)
        expected_locations = {"Siddharth": (1.0, 1.0)}

        fused = fusion.fuse(spk, loc, expected_speaker_positions=expected_locations)
        assert fused.is_spatially_consistent is False
