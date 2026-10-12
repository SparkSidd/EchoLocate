"""
EchoLocate — Spatial Speaker Attribution Fusion Layer
=====================================================
Combines WHO (Speaker Attribution) + WHERE (EchoLocate Localization) + WHEN (Temporal Stability)
into a unified attribution event, strictly preserving explicit UNKNOWN states.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..types import AttributionStatus, LocalizationInput, SpatialFusionResult, SpeakerAttributionResult


class SpatialSpeakerFusion:
    """
    Transparent, explainable evidence fusion engine:
        P(Person | Voice, Location, Time)
    Weights voice acoustic evidence, 2D spatial localization proximity,
    and temporal stability into a combined attribution score.
    """

    def __init__(
        self,
        weight_voice: float = 0.55,
        weight_spatial: float = 0.30,
        weight_temporal: float = 0.15,
        spatial_tolerance_m: float = 1.5,
        rejection_threshold: float = 0.50,
    ):
        self.weight_voice = weight_voice
        self.weight_spatial = weight_spatial
        self.weight_temporal = weight_temporal
        self.spatial_tolerance_m = spatial_tolerance_m
        self.rejection_threshold = rejection_threshold

    def fuse(
        self,
        speaker_res: SpeakerAttributionResult,
        loc_res: Optional[LocalizationInput] = None,
        expected_speaker_positions: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> SpatialFusionResult:
        """
        Compute joint spatial speaker attribution.
        
        Args:
            speaker_res: Speaker recognition output event
            loc_res: Optional 2D localization output from EchoLocate
            expected_speaker_positions: Optional mapping {speaker_id: (x_m, y_m)}
        """
        voice_conf = float(speaker_res.confidence)
        temp_stability = float(speaker_res.temporal_stability)
        candidate_id = speaker_res.speaker_id
        is_unknown_voice = (
            speaker_res.status != AttributionStatus.KNOWN or candidate_id == "UNKNOWN"
        )

        # 1. Determine Spatial Evidence
        position_xy: Optional[Tuple[float, float]] = None
        spatial_conf = 0.0
        is_spatially_consistent = True
        spatial_distance = None

        if loc_res is not None:
            position_xy = loc_res.position_xy
            loc_conf = float(loc_res.confidence)

            # Check spatial consistency if expected position exists for enrolled candidate
            if expected_speaker_positions and candidate_id in expected_speaker_positions and not is_unknown_voice:
                exp_x, exp_y = expected_speaker_positions[candidate_id]
                est_x, est_y = position_xy
                dist = math.sqrt((est_x - exp_x) ** 2 + (est_y - exp_y) ** 2)
                spatial_distance = dist

                # Gaussian spatial proximity score: exp(-0.5 * (d / sigma)^2)
                sigma = self.spatial_tolerance_m
                prox = math.exp(-0.5 * ((dist / sigma) ** 2))
                spatial_conf = prox * loc_conf
                is_spatially_consistent = dist <= (self.spatial_tolerance_m * 1.5)
            else:
                # No expected prior position: spatial confidence is purely localization confidence
                spatial_conf = loc_conf
        else:
            # No spatial data available: voice-only mode
            spatial_conf = 0.0

        # 2. Re-normalize weights if spatial data is absent
        if loc_res is None:
            w_v = 0.80
            w_s = 0.0
            w_t = 0.20
        else:
            total_w = self.weight_voice + self.weight_spatial + self.weight_temporal
            w_v = self.weight_voice / total_w
            w_s = self.weight_spatial / total_w
            w_t = self.weight_temporal / total_w

        # 3. Compute Joint Evidence Score
        joint_score = (w_v * voice_conf) + (w_s * spatial_conf) + (w_t * temp_stability)

        # 4. Mandatory Explicit UNKNOWN Preservation:
        # If voice evidence was UNKNOWN, NEVER invent an enrolled identity simply because a sound was localized!
        if is_unknown_voice:
            final_speaker = "UNKNOWN"
            final_status = AttributionStatus.UNKNOWN
            final_joint_conf = float(min(0.45, joint_score * 0.7))
        elif not is_spatially_consistent:
            # Severe spatial conflict (e.g. voice looks like Person A, but sound came from 5m away)
            # Downgrade to UNKNOWN or low-confidence ambiguous
            if joint_score < self.rejection_threshold:
                final_speaker = "UNKNOWN"
                final_status = AttributionStatus.UNKNOWN
                final_joint_conf = float(joint_score * 0.5)
            else:
                final_speaker = candidate_id
                final_status = AttributionStatus.KNOWN
                final_joint_conf = float(joint_score * 0.7)
        else:
            final_speaker = candidate_id
            final_status = (
                AttributionStatus.KNOWN
                if joint_score >= self.rejection_threshold
                else AttributionStatus.UNKNOWN
            )
            final_joint_conf = float(joint_score)

        return SpatialFusionResult(
            speaker_id=final_speaker,
            status=final_status,
            position_xy=position_xy,
            voice_confidence=voice_conf,
            spatial_confidence=spatial_conf,
            joint_confidence=final_joint_conf,
            embedding_model=speaker_res.embedding_model,
            temporal_stability=temp_stability,
            is_spatially_consistent=is_spatially_consistent,
            metadata={
                "voice_similarity": float(speaker_res.similarity),
                "voice_threshold": float(speaker_res.threshold),
                "spatial_distance_m": spatial_distance,
                "weights_used": {"voice": w_v, "spatial": w_s, "temporal": w_t},
            },
        )
