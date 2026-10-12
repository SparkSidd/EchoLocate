"""
EchoLocate — Cosine Speaker Matcher & UNKNOWN Rejection
======================================================
Implements cosine similarity matching, calibrated thresholding,
margin gating, and explicit UNKNOWN speaker attribution.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..types import AttributionStatus, CandidateScore, SpeakerAttributionResult

logger = logging.getLogger(__name__)


class CosineSpeakerMatcher:
    """
    Evaluates query embeddings against enrolled centroids and sample embeddings.
    Strictly prevents blind argmax assignment by enforcing threshold gating.
    """

    def __init__(
        self,
        default_threshold: float = 0.42,
        min_margin: float = 0.05,
        confidence_scaling: float = 12.0,
        per_speaker_thresholds: Optional[Dict[str, float]] = None,
        fusion_weights: Optional[Tuple[float, float, float]] = (0.50, 0.30, 0.20),
    ):
        self.default_threshold = default_threshold
        self.min_margin = min_margin
        self.confidence_scaling = confidence_scaling
        self.per_speaker_thresholds = per_speaker_thresholds or {}
        self.fusion_weights = fusion_weights or (0.50, 0.30, 0.20)

    def match(
        self,
        query_embedding: np.ndarray,
        enrolled_ids: List[str],
        enrolled_centroids: np.ndarray,
        enrolled_embeddings: Optional[Dict[str, List[np.ndarray]]] = None,
        model_name: str = "unknown",
        threshold_override: Optional[float] = None,
        segment_start: float = 0.0,
        segment_end: float = 0.0,
        speech_ratio: float = 1.0,
    ) -> SpeakerAttributionResult:
        """
        Match query embedding against enrolled centroids and individual sample embeddings.
        
        Args:
            query_embedding: 1D array of shape (D,), unit normalized
            enrolled_ids: List of K speaker identifiers
            enrolled_centroids: 2D array of shape (K, D), each row unit normalized
            enrolled_embeddings: Optional mapping of speaker ID to list of normalized sample embeddings
            model_name: Name of encoder used
            threshold_override: Optional decision boundary override
        """
        # Base case: empty database
        if len(enrolled_ids) == 0 or enrolled_centroids.shape[0] == 0:
            return SpeakerAttributionResult(
                speaker_id="UNKNOWN",
                status=AttributionStatus.UNKNOWN,
                confidence=0.0,
                similarity=-1.0,
                threshold=threshold_override if threshold_override is not None else self.default_threshold,
                embedding_model=model_name,
                segment_start=segment_start,
                segment_end=segment_end,
                speech_ratio=speech_ratio,
                metadata={"reason": "empty_enrollment_database"},
            )

        # Normalize query vector
        q_norm = np.linalg.norm(query_embedding)
        if q_norm < 1e-9:
            return SpeakerAttributionResult(
                speaker_id="UNKNOWN",
                status=AttributionStatus.INSUFFICIENT_AUDIO,
                confidence=0.0,
                similarity=-1.0,
                threshold=threshold_override if threshold_override is not None else self.default_threshold,
                embedding_model=model_name,
                segment_start=segment_start,
                segment_end=segment_end,
                speech_ratio=speech_ratio,
                metadata={"reason": "zero_norm_query_embedding"},
            )
        q = query_embedding / q_norm

        # Defensive check: verify dimension alignment
        if enrolled_centroids.shape[1] != q.shape[0]:
            logger.error(
                "Dimension mismatch in matcher: enrolled_centroids (%d, %d) vs query (%d,)",
                enrolled_centroids.shape[0],
                enrolled_centroids.shape[1],
                q.shape[0],
            )
            return SpeakerAttributionResult(
                speaker_id="UNKNOWN",
                status=AttributionStatus.UNKNOWN,
                confidence=0.0,
                similarity=-1.0,
                threshold=threshold_override if threshold_override is not None else self.default_threshold,
                embedding_model=model_name,
                segment_start=segment_start,
                segment_end=segment_end,
                speech_ratio=speech_ratio,
                metadata={
                    "reason": "dimension_mismatch",
                    "expected_dim": enrolled_centroids.shape[1],
                    "actual_dim": q.shape[0],
                },
            )

        # Multi-tier score calculation (Phase 8 & Phase 16)
        w_cent, w_top1, w_top3 = self.fusion_weights
        similarities = []
        detailed_scores = {}

        for idx, s_id in enumerate(enrolled_ids):
            centroid = enrolled_centroids[idx]
            centroid_sim = float(np.dot(centroid, q))

            spk_embs = enrolled_embeddings.get(s_id, []) if enrolled_embeddings else []
            if spk_embs:
                sample_sims = [float(np.dot(e / (np.linalg.norm(e) + 1e-9), q)) for e in spk_embs]
                top1_sim = max(sample_sims)
                sorted_sims = sorted(sample_sims, reverse=True)
                top3_mean = float(np.mean(sorted_sims[:min(3, len(sorted_sims))]))
                median_sim = float(np.median(sample_sims))
                fused_sim = float((w_cent * centroid_sim) + (w_top1 * top1_sim) + (w_top3 * top3_mean))
                detailed_scores[s_id] = {
                    "centroid_score": centroid_sim,
                    "top1_score": top1_sim,
                    "top3_mean": top3_mean,
                    "median_score": median_sim,
                    "fused_score": fused_sim,
                }
                similarities.append(fused_sim)
            else:
                detailed_scores[s_id] = {
                    "centroid_score": centroid_sim,
                    "top1_score": centroid_sim,
                    "top3_mean": centroid_sim,
                    "median_score": centroid_sim,
                    "fused_score": centroid_sim,
                }
                similarities.append(centroid_sim)

        similarities = np.array(similarities, dtype=np.float32)

        # Rank all candidates in descending order
        ranked_indices = np.argsort(similarities)[::-1]
        alternatives: List[CandidateScore] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            alternatives.append(
                CandidateScore(
                    speaker_id=enrolled_ids[idx],
                    similarity=float(similarities[idx]),
                    rank=rank,
                )
            )

        top_candidate = alternatives[0]
        top_speaker_id = top_candidate.speaker_id
        top_similarity = top_candidate.similarity

        # Determine applicable threshold
        if threshold_override is not None:
            active_threshold = threshold_override
        elif top_speaker_id in self.per_speaker_thresholds:
            active_threshold = self.per_speaker_thresholds[top_speaker_id]
        else:
            active_threshold = self.default_threshold

        # Calibrate confidence score using logistic sigmoid: 1 / (1 + exp(-k * (s - threshold)))
        diff = top_similarity - active_threshold
        # Clip diff to prevent overflow
        diff = max(-5.0, min(5.0, diff))
        calibrated_confidence = 1.0 / (1.0 + math.exp(-self.confidence_scaling * diff))

        # Check margin condition if multiple speakers exist (Phase 13 audit)
        has_margin_ambiguity = False
        raw_margin = 1.0
        if len(alternatives) > 1:
            second_sim = alternatives[1].similarity
            raw_margin = float(top_similarity - second_sim)
            if self.min_margin > 0 and raw_margin < self.min_margin:
                has_margin_ambiguity = True

        # Decision rule: UNKNOWN if below threshold or ambiguous
        if top_similarity < active_threshold:
            status = AttributionStatus.UNKNOWN
            assigned_speaker = "UNKNOWN"
            # Low confidence when rejected
            confidence = float(max(0.0, min(0.49, calibrated_confidence)))
        elif has_margin_ambiguity:
            # High similarity but indistinguishable between top 2 candidates
            status = AttributionStatus.UNKNOWN
            assigned_speaker = "UNKNOWN"
            confidence = float(max(0.0, min(0.45, calibrated_confidence * 0.5)))
        else:
            status = AttributionStatus.KNOWN
            assigned_speaker = top_speaker_id
            confidence = float(max(0.50, min(0.99, calibrated_confidence)))

        return SpeakerAttributionResult(
            speaker_id=assigned_speaker,
            status=status,
            confidence=confidence,
            similarity=float(top_similarity),
            threshold=float(active_threshold),
            embedding_model=model_name,
            segment_start=segment_start,
            segment_end=segment_end,
            speech_ratio=speech_ratio,
            alternatives=alternatives,
            metadata={
                "top_enrolled_candidate": top_speaker_id,
                "top_similarity": float(top_similarity),
                "second_enrolled_candidate": alternatives[1].speaker_id if len(alternatives) > 1 else None,
                "second_similarity": float(alternatives[1].similarity) if len(alternatives) > 1 else None,
                "raw_margin": raw_margin,
                "min_margin_applied": self.min_margin,
                "has_margin_ambiguity": has_margin_ambiguity,
                "active_threshold": float(active_threshold),
                "scores": detailed_scores.get(top_speaker_id, {}),
            },
        )
