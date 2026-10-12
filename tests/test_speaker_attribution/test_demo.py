"""
Unit and Integration Tests for Real Speaker Demo Module
========================================================
Tests quality checker gates, session enrollment, identification,
blind testing scorecard, and web API endpoints.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.speaker_attribution.demo.quality_checker import EnrollmentQualityChecker
from src.speaker_attribution.demo.recorder import MicrophoneRecorder
from src.speaker_attribution.demo.session import DemonstrationSession
from src.speaker_attribution.demo.web_demo import create_demo_app
from src.speaker_attribution.experiments.synthetic_generator import AcousticSpeakerGenerator


@pytest.fixture
def generator():
    return AcousticSpeakerGenerator(seed=42)


@pytest.fixture
def temp_db_dir(tmp_path):
    d = tmp_path / "test_demo_speakers"
    d.mkdir(parents=True, exist_ok=True)
    return d


class TestQualityChecker:
    def test_clean_speech_acceptance(self, generator):
        checker = EnrollmentQualityChecker(min_speech_duration_sec=0.5, min_rms=0.005)
        audio = generator.generate_utterance("Siddharth", duration_sec=2.0)
        res = checker.check(audio)
        assert res.passed is True
        assert res.speech_duration_sec >= 0.5
        assert res.rms > 0.005
        assert res.snr_db > 6.0
        assert "ACCEPTED" in res.summary_line()

    def test_silence_rejection(self):
        checker = EnrollmentQualityChecker()
        silence = np.zeros(16000, dtype=np.float32)
        res = checker.check(silence)
        assert res.passed is False
        assert "too_quiet" in res.status_reason

    def test_short_utterance_rejection(self, generator):
        checker = EnrollmentQualityChecker(min_speech_duration_sec=1.5)
        # 0.4s audio is shorter than 1.5s gate
        short_audio = generator.generate_utterance("Siddharth", duration_sec=0.4)
        res = checker.check(short_audio)
        assert res.passed is False
        assert "speech_too_short" in res.status_reason


class TestDemonstrationSession:
    def test_enrollment_and_identification(self, generator, temp_db_dir):
        session = DemonstrationSession(
            db_dir=temp_db_dir,
            model_name="mock",
            default_threshold=0.94,
        )

        # 1. Enroll Rahul with 3 samples
        for i in range(3):
            audio = generator.generate_utterance("Siddharth", duration_sec=1.5, utterance_seed=10 + i)
            ok, quality, prof = session.enroll_sample(
                speaker_id="rahul",
                display_name="Rahul",
                audio=audio,
            )
            assert ok is True
            assert prof is not None
            assert prof.sample_count == i + 1

        assert "rahul" in session.db
        assert len(session.db) == 1

        # 2. Identify Rahul with unseen test utterance
        test_audio = generator.generate_utterance("Siddharth", duration_sec=1.5, utterance_seed=100)
        res_rahul = session.identify_sample(test_audio)
        assert res_rahul["decision"] == "KNOWN"
        assert res_rahul["speaker_id"] == "rahul"
        assert res_rahul["similarity"] >= 0.94

        # 3. Impostor test (un-enrolled person)
        impostor_audio = generator.generate_utterance("Alice", duration_sec=1.5, utterance_seed=200)
        res_imp = session.identify_sample(impostor_audio)
        assert res_imp["decision"] == "UNKNOWN"
        assert res_imp["speaker_id"] == "UNKNOWN"

    def test_blind_evaluation_flow(self, generator, temp_db_dir):
        session = DemonstrationSession(
            db_dir=temp_db_dir,
            model_name="mock",
            default_threshold=0.94,
        )

        # Enroll Speaker 1 (Rahul)
        a1 = generator.generate_utterance("Siddharth", duration_sec=1.5, utterance_seed=1)
        session.enroll_sample("rahul", "Rahul", a1)

        # Run blind test 1: Rahul (Known)
        t1 = generator.generate_utterance("Siddharth", duration_sec=1.5, utterance_seed=55)
        case1 = session.record_blind_test("rahul", t1, condition="normal")
        assert case1.test_id == 1
        assert case1.is_correct is True

        # Run blind test 2: Unseen Impostor
        t2 = generator.generate_utterance("Alice", duration_sec=1.5, utterance_seed=77)
        case2 = session.record_blind_test("unknown_impostor", t2, condition="noise")
        assert case2.test_id == 2
        assert case2.is_correct is True
        assert case2.predicted_speaker == "UNKNOWN"

        # Compute summary
        summary = session.compute_summary()
        assert summary.total_test_utterances == 2
        assert summary.known_tests_count == 1
        assert summary.known_correct_count == 1
        assert summary.known_accuracy_percent == 100.0
        assert summary.unknown_tests_count == 1
        assert summary.unknown_correct_count == 1
        assert summary.unknown_rejection_percent == 100.0
        assert summary.overall_accuracy_percent == 100.0
        assert "REAL SPEAKER ATTRIBUTION EVALUATION REPORT" in summary.format_report()


class TestWebDemoAPI:
    def test_web_endpoints(self, generator, temp_db_dir):
        app = create_demo_app(db_dir=str(temp_db_dir), model_name="mock", default_threshold=0.60)
        client = TestClient(app)

        # 1. GET /
        resp = client.get("/")
        assert resp.status_code == 200
        assert "EchoLocate" in resp.text

        # 2. GET /api/speakers (initially empty)
        resp = client.get("/api/speakers")
        assert resp.status_code == 200
        data = resp.json()
        assert "speakers" in data

        # 3. Create dummy WAV in memory
        import soundfile as sf
        audio_data = generator.generate_utterance("Siddharth", duration_sec=1.5, utterance_seed=1)
        wav_buf = io.BytesIO()
        sf.write(wav_buf, audio_data, 16000, format="WAV")
        wav_bytes = wav_buf.getvalue()

        # 4. POST /api/enroll
        resp = client.post(
            "/api/enroll",
            data={"speaker_id": "priya", "display_name": "Priya"},
            files={"audio_file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        res_json = resp.json()
        assert res_json["success"] is True
        assert res_json["profile"]["speaker_id"] == "priya"

        # 5. POST /api/identify
        resp = client.post(
            "/api/identify",
            files={"audio_file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        id_json = resp.json()
        assert id_json["decision"] == "KNOWN"
        assert id_json["speaker_id"] == "priya"

        # 6. POST /api/blind-test/record
        resp = client.post(
            "/api/blind-test/record",
            data={"true_speaker": "priya", "condition": "normal"},
            files={"audio_file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "recorded"

        # 7. GET /api/blind-test/summary
        resp = client.get("/api/blind-test/summary")
        assert resp.status_code == 200
        assert resp.json()["total_test_utterances"] == 1
