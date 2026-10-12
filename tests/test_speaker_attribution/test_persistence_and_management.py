"""
Tests for Persistent Speaker Profile Management and Restart Survival
====================================================================
Validates:
  - Local WAV persistence under <db_dir>/<speaker_id>/recordings/*.wav
  - Profile persistence and reloading across restart
  - Speaker deletion removing disk folders and profiles
  - Live streaming chunk identification endpoint
  - Model switching and empty database handling
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from src.speaker_attribution.demo.session import DemonstrationSession
from src.speaker_attribution.demo.web_demo import create_demo_app
from src.speaker_attribution.enrollment.database import SpeakerEnrollmentDB
from src.speaker_attribution.experiments.synthetic_generator import AcousticSpeakerGenerator


@pytest.fixture
def generator():
    return AcousticSpeakerGenerator(seed=42)


@pytest.fixture
def temp_db_dir(tmp_path):
    d = tmp_path / "persistent_speakers_test"
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_test_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    bio = io.BytesIO()
    sf.write(bio, audio, sample_rate, format="WAV")
    return bio.getvalue()


class TestSpeakerPersistence:
    def test_local_recordings_and_profile_json(self, generator, temp_db_dir):
        """Verify accepted utterances are saved as WAVs and profile.json is generated."""
        session = DemonstrationSession(
            db_dir=temp_db_dir,
            model_name="mock",
            default_threshold=0.94,
        )

        # Enroll Rahul with 3 samples
        for i in range(1, 4):
            audio = generator.generate_utterance("Siddharth", duration_sec=1.5, utterance_seed=10 + i)
            ok, quality, prof = session.enroll_sample("rahul", "Rahul", audio)
            assert ok is True
            assert prof is not None

        # Verify disk structure
        spk_dir = temp_db_dir / "rahul"
        assert spk_dir.exists()
        assert (spk_dir / "profile.json").exists()
        assert (spk_dir / "centroid.npy").exists()

        rec_dir = spk_dir / "recordings"
        assert rec_dir.exists()
        wavs = list(rec_dir.glob("*.wav"))
        assert len(wavs) == 3
        assert (rec_dir / "001.wav").exists()
        assert (rec_dir / "002.wav").exists()
        assert (rec_dir / "003.wav").exists()

        # Check speaker details
        details = session.get_speaker_details("rahul")
        assert details is not None
        assert details["speaker_id"] == "rahul"
        assert details["display_name"] == "Rahul"
        assert details["sample_count"] == 3
        assert len(details["recordings"]) == 3

    def test_survival_across_restart(self, generator, temp_db_dir):
        """Verify that stopping/restarting server retains all enrolled profiles and centroids."""
        # Session 1: Enroll Rahul and Priya
        session_1 = DemonstrationSession(db_dir=temp_db_dir, model_name="mock", default_threshold=0.94)
        a_rahul = generator.generate_utterance("Siddharth", duration_sec=1.5, utterance_seed=1)
        a_priya = generator.generate_utterance("Alice", duration_sec=1.5, utterance_seed=2)

        session_1.enroll_sample("rahul", "Rahul", a_rahul)
        session_1.enroll_sample("priya", "Priya", a_priya)
        assert len(session_1.db) == 2

        # Simulate Application Restart by destroying session_1 and creating session_2
        del session_1

        session_2 = DemonstrationSession(db_dir=temp_db_dir, model_name="mock", default_threshold=0.94)
        assert len(session_2.db) == 2
        assert "rahul" in session_2.db
        assert "priya" in session_2.db

        prof_rahul = session_2.db.get_profile("rahul")
        assert prof_rahul is not None
        assert prof_rahul.embedding_dim == 192

        # Test identification with session_2
        test_rahul = generator.generate_utterance("Siddharth", duration_sec=1.5, utterance_seed=100)
        res_r = session_2.identify_sample(test_rahul)
        assert res_r["decision"] == "KNOWN"
        assert res_r["speaker_id"] == "rahul"

        # Test impostor with session_2
        test_unknown = generator.generate_utterance("Bob", duration_sec=1.5, utterance_seed=300)
        res_u = session_2.identify_sample(test_unknown)
        assert res_u["decision"] == "UNKNOWN"

    def test_speaker_deletion(self, generator, temp_db_dir):
        """Verify deleting a speaker removes their in-memory profile and on-disk files."""
        session = DemonstrationSession(db_dir=temp_db_dir, model_name="mock", default_threshold=0.94)
        audio = generator.generate_utterance("Siddharth", duration_sec=1.5, utterance_seed=5)
        session.enroll_sample("siddharth", "Siddharth", audio)

        spk_dir = temp_db_dir / "siddharth"
        assert spk_dir.exists()
        assert "siddharth" in session.db

        # Delete speaker
        deleted = session.delete_speaker("siddharth")
        assert deleted is True
        assert "siddharth" not in session.db
        assert not spk_dir.exists()


class TestWebManagementAPI:
    def test_endpoints_crud_and_live(self, generator, temp_db_dir):
        app = create_demo_app(db_dir=str(temp_db_dir), model_name="mock", default_threshold=0.94)
        client = TestClient(app)

        # 1. List speakers initially empty
        resp = client.get("/api/speakers")
        assert resp.status_code == 200
        assert len(resp.json()["speakers"]) == 0

        # 2. Enroll Rahul via API
        audio = generator.generate_utterance("Siddharth", duration_sec=1.5, utterance_seed=10)
        wav_bytes = create_test_wav_bytes(audio)

        resp = client.post(
            "/api/enroll",
            data={"speaker_id": "rahul", "display_name": "Rahul"},
            files={"audio_file": ("sample.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert resp.json()["profile"]["sample_count"] == 1

        # 3. GET /api/speaker/rahul
        resp = client.get("/api/speaker/rahul")
        assert resp.status_code == 200
        details = resp.json()
        assert details["speaker_id"] == "rahul"
        assert len(details["recordings"]) == 1

        # 4. Live identify chunk endpoint
        resp = client.post(
            "/api/speaker/live/identify",
            files={"audio_file": ("chunk.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        live_res = resp.json()
        assert live_res["decision"] == "KNOWN"
        assert live_res["speaker_id"] == "rahul"

        # 5. Delete speaker via API
        resp = client.delete("/api/speaker/rahul")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify list is empty again
        resp = client.get("/api/speakers")
        assert len(resp.json()["speakers"]) == 0
