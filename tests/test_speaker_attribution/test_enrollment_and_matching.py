"""
EchoLocate — Tests: Enrollment, Matching & UNKNOWN Rejection
===========================================================
Validates centroid computation, database persistence, cosine similarity,
calibrated thresholding, and mandatory UNKNOWN rejection.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.speaker_attribution.enrollment.database import SpeakerEnrollmentDB
from src.speaker_attribution.matching.calibration import ThresholdCalibrator
from src.speaker_attribution.matching.matcher import CosineSpeakerMatcher
from src.speaker_attribution.types import AttributionStatus


class TestSpeakerEnrollment:
    def test_single_and_incremental_enrollment(self):
        db = SpeakerEnrollmentDB()
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        prof = db.enroll_embedding("Speaker_A", v1, model_name="mock")
        assert prof.sample_count == 1
        assert np.allclose(prof.centroid, v1)

        # Enroll second sample
        prof2 = db.enroll_embedding("Speaker_A", v2, model_name="mock")
        assert prof2.sample_count == 2
        # Expected centroid: normalize([1.0, 1.0, 0.0]) = [1/sqrt(2), 1/sqrt(2), 0.0]
        expected = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0], dtype=np.float32)
        assert np.allclose(prof2.centroid, expected, atol=1e-4)
        assert np.isclose(np.linalg.norm(prof2.centroid), 1.0)

    def test_db_persistence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db = SpeakerEnrollmentDB(db_dir=tmp_dir)
            v = np.array([0.6, 0.8], dtype=np.float32)
            db.enroll_embedding("Siddharth", v, model_name="test_model")
            db.save(tmp_dir)

            # Load into fresh instance
            db_loaded = SpeakerEnrollmentDB(db_dir=tmp_dir)
            prof = db_loaded.get_profile("Siddharth")
            assert prof is not None
            assert prof.speaker_id == "Siddharth"
            assert np.allclose(prof.centroid, v)
            assert prof.model_name == "test_model"


class TestMatchingAndUnknownRejection:
    @pytest.fixture
    def mock_db_data(self):
        ids = ["Siddharth", "Alice"]
        # Orthogonal centroids in 3D
        c1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        c2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        centroids = np.stack([c1, c2], axis=0)
        return ids, centroids

    def test_known_speaker_match(self, mock_db_data):
        ids, centroids = mock_db_data
        matcher = CosineSpeakerMatcher(default_threshold=0.60)

        # Query close to Siddharth (cos ~ 0.96)
        query = np.array([0.96, 0.28, 0.0], dtype=np.float32)
        res = matcher.match(query, ids, centroids, model_name="mock")

        assert res.status == AttributionStatus.KNOWN
        assert res.speaker_id == "Siddharth"
        assert res.confidence > 0.50
        assert res.similarity > 0.90

    def test_mandatory_unknown_rejection(self, mock_db_data):
        ids, centroids = mock_db_data
        matcher = CosineSpeakerMatcher(default_threshold=0.60)

        # Impostor query orthogonal to both enrolled speakers: [0.0, 0.0, 1.0] (cos = 0.0)
        query_impostor = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        res = matcher.match(query_impostor, ids, centroids, model_name="mock")

        assert res.status == AttributionStatus.UNKNOWN
        assert res.speaker_id == "UNKNOWN"
        assert res.confidence < 0.50

    def test_margin_ambiguity_rejection(self, mock_db_data):
        ids, centroids = mock_db_data
        # min_margin = 0.10: requires clear winner
        matcher = CosineSpeakerMatcher(default_threshold=0.60, min_margin=0.10)

        # Ambiguous query equidistant from both candidates: cos ~ 0.707 to both
        ambiguous_query = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0], dtype=np.float32)
        res = matcher.match(ambiguous_query, ids, centroids, model_name="mock")

        # Must reject ambiguous match to UNKNOWN
        assert res.status == AttributionStatus.UNKNOWN
        assert res.speaker_id == "UNKNOWN"
        assert res.metadata["has_margin_ambiguity"] is True


class TestThresholdCalibration:
    def test_calibrator_eer_calculation(self):
        calibrator = ThresholdCalibrator(num_points=100)
        # Genuine scores distributed around 0.85 +- 0.05
        genuine = np.random.normal(0.85, 0.05, 100).tolist()
        # Impostor scores distributed around 0.35 +- 0.05
        impostor = np.random.normal(0.35, 0.05, 100).tolist()

        scores = genuine + impostor
        labels = [True] * len(genuine) + [False] * len(impostor)

        report = calibrator.calibrate(scores, labels)
        assert 0.45 <= report.optimal_threshold <= 0.75
        assert report.eer < 0.05
        assert report.auc_roc > 0.95
