"""
Regression tests for Read-Only Startup/Inference and Model Provenance (Phases 1 & 2)
==================================================================================
Validates:
  1. Startup, identification, and model inspection do not mutate profile files or hashes.
  2. Model switching with persist=False does not overwrite saved disk profiles.
  3. Explicit enrollment persists normally and captures active model provenance.
  4. Dimension mismatch raises ValueError.
  5. Model architecture mismatch raises ValueError.
  6. Checkpoint hash mismatch raises ValueError.
  7. Legacy unverified profiles load cleanly, are marked is_legacy=True / is_verified=False,
     and remain fully functional for matching without crashes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import pytest

from src.speaker_attribution.demo.session import DemonstrationSession
from src.speaker_attribution.enrollment.database import SpeakerEnrollmentDB
from src.speaker_attribution.experiments.synthetic_generator import AcousticSpeakerGenerator
from src.speaker_attribution.types import ModelProvenance, SpeakerProfile


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def snapshot_directory_state(dirpath: Path) -> dict[str, tuple[str, int, float]]:
    """Capture relative path -> (sha256, size, mtime) for all files in directory."""
    state = {}
    for p in dirpath.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(dirpath))
            state[rel] = (compute_file_hash(p), p.stat().st_size, p.stat().st_mtime)
    return state


@pytest.fixture
def generator():
    return AcousticSpeakerGenerator(seed=42)


@pytest.fixture
def enrolled_test_dir(tmp_path, generator):
    """Create a populated db_dir with one enrolled speaker."""
    db_dir = tmp_path / "speakers_db"
    session = DemonstrationSession(db_dir=db_dir, model_name="mock", default_threshold=0.90)

    audio = generator.generate_utterance("Siddharth", duration_sec=1.5, utterance_seed=1)
    ok, quality, prof = session.enroll_sample("sid", "Siddharth", audio)
    assert ok is True
    assert prof is not None
    return db_dir


class TestReadOnlyStartupAndInference:
    def test_startup_and_identification_do_not_mutate_profiles(self, enrolled_test_dir, generator):
        """Proves that initializing DemonstrationSession and identifying samples is strictly read-only."""
        before_state = snapshot_directory_state(enrolled_test_dir)
        assert len(before_state) > 0, "Initial directory must contain files"

        # Initialize session (startup)
        session = DemonstrationSession(db_dir=enrolled_test_dir, model_name="mock", default_threshold=0.90)

        # Run multiple identifications with known and unknown audio
        known_audio = generator.generate_utterance("Siddharth", duration_sec=1.5, utterance_seed=2)
        res_known = session.identify_sample(known_audio)
        assert res_known["decision"] == "KNOWN"

        unknown_audio = generator.generate_utterance("Alice", duration_sec=1.5, utterance_seed=3)
        res_unknown = session.identify_sample(unknown_audio)
        assert res_unknown["decision"] == "UNKNOWN"

        # Inspect speaker details
        details = session.get_speaker_details("sid")
        assert details is not None
        assert details["speaker_id"] == "sid"

        after_state = snapshot_directory_state(enrolled_test_dir)

        # Every file must exist with identical SHA256, size, and mtime
        assert set(before_state.keys()) == set(after_state.keys())
        for rel_path, (b_hash, b_size, b_mtime) in before_state.items():
            a_hash, a_size, a_mtime = after_state[rel_path]
            assert b_hash == a_hash, f"File {rel_path} content hash modified during read-only operation!"
            assert b_size == a_size, f"File {rel_path} size modified during read-only operation!"
            assert b_mtime == a_mtime, f"File {rel_path} timestamp modified during read-only operation!"

    def test_switch_model_without_persist_does_not_mutate_disk(self, enrolled_test_dir):
        """Proves that switch_model(persist=False) updates in-memory centroids but leaves disk files pristine."""
        before_state = snapshot_directory_state(enrolled_test_dir)

        session = DemonstrationSession(db_dir=enrolled_test_dir, model_name="mock", default_threshold=0.90)
        session.switch_model("mock", persist=False)

        after_state = snapshot_directory_state(enrolled_test_dir)
        assert set(before_state.keys()) == set(after_state.keys())
        for rel_path, (b_hash, b_size, b_mtime) in before_state.items():
            a_hash, a_size, a_mtime = after_state[rel_path]
            assert b_hash == a_hash, f"File {rel_path} was modified during switch_model(persist=False)!"


class TestExplicitEnrollmentPersistence:
    def test_explicit_enrollment_persists_with_provenance(self, tmp_path, generator):
        """Proves that explicit user enrollment writes to disk and records model provenance."""
        db_dir = tmp_path / "explicit_enroll_db"
        session = DemonstrationSession(db_dir=db_dir, model_name="mock", default_threshold=0.90)

        audio = generator.generate_utterance("Siddharth", duration_sec=1.5, utterance_seed=11)
        ok, quality, prof = session.enroll_sample("adi", "Aditya", audio)

        assert ok is True
        assert prof is not None

        # Verify disk files exist
        spk_dir = db_dir / "adi"
        prof_file = spk_dir / "profile.json"
        cent_file = spk_dir / "centroid.npy"
        assert prof_file.exists()
        assert cent_file.exists()

        with open(prof_file, "r") as f:
            data = json.load(f)

        assert "provenance" in data
        assert data["provenance"] is not None
        assert data["provenance"]["model_name"] == "mock"
        assert data["provenance"]["embedding_dim"] == 192

        # Re-load DB fresh from disk to verify roundtrip
        fresh_db = SpeakerEnrollmentDB(db_dir=db_dir)
        loaded_prof = fresh_db.get_profile("adi")
        assert loaded_prof is not None
        assert loaded_prof.provenance is not None
        assert loaded_prof.provenance.model_name == "mock"
        assert loaded_prof.provenance.status == "unverified_weights"

    def test_explicit_enrollment_verified_checkpoint_roundtrip(self, tmp_path):
        """Proves that enrolling with verified checkpoint provenance roundtrips is_verified=True."""
        db_dir = tmp_path / "verified_enroll_db"
        db = SpeakerEnrollmentDB(db_dir=db_dir)
        prov = ModelProvenance(
            model_name="campp",
            embedding_dim=192,
            checkpoint_hash="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            is_verified=True,
            status="verified",
        )
        emb = np.random.randn(192).astype(np.float32)
        db.enroll_embedding("verified_user", emb, model_name="campp", provenance=prov)
        db.save()

        fresh_db = SpeakerEnrollmentDB(db_dir=db_dir)
        p = fresh_db.get_profile("verified_user")
        assert p is not None
        assert p.is_verified is True
        assert p.is_legacy is False
        assert p.provenance.checkpoint_hash == "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"


class TestModelProvenanceGuards:
    def test_dimension_mismatch_rejected(self):
        """Proves that attempting to enroll an embedding with mismatched dimension raises ValueError."""
        db = SpeakerEnrollmentDB()
        prov_192 = ModelProvenance(model_name="mock", embedding_dim=192)
        emb_192 = np.ones(192, dtype=np.float32)

        db.enroll_embedding("user1", emb_192, model_name="mock", provenance=prov_192)

        emb_512 = np.ones(512, dtype=np.float32)
        prov_512 = ModelProvenance(model_name="campp", embedding_dim=512)

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            db.enroll_embedding("user1", emb_512, model_name="campp", provenance=prov_512)

    def test_model_architecture_mismatch_rejected(self):
        """Proves that attempting to enroll an embedding with mismatched model architecture raises ValueError."""
        db = SpeakerEnrollmentDB()
        prov_a = ModelProvenance(model_name="campp", embedding_dim=192)
        emb = np.ones(192, dtype=np.float32)
        db.enroll_embedding("user1", emb, model_name="campp", provenance=prov_a)

        prov_b = ModelProvenance(model_name="eres2netv2", embedding_dim=192)
        with pytest.raises(ValueError, match="Model architecture mismatch"):
            db.enroll_embedding("user1", emb, model_name="eres2netv2", provenance=prov_b)

    def test_checkpoint_hash_mismatch_rejected(self):
        """Proves that attempting to mix embeddings from different checkpoints raises ValueError."""
        db = SpeakerEnrollmentDB()
        prov_ckpt1 = ModelProvenance(
            model_name="campp",
            embedding_dim=192,
            checkpoint_hash="aaaa1111222233334444555566667777888899990000aaaa1111222233334444",
        )
        emb = np.ones(192, dtype=np.float32)
        db.enroll_embedding("user1", emb, model_name="campp", provenance=prov_ckpt1)

        prov_ckpt2 = ModelProvenance(
            model_name="campp",
            embedding_dim=192,
            checkpoint_hash="bbbb1111222233334444555566667777888899990000bbbb1111222233334444",
        )
        with pytest.raises(ValueError, match="Checkpoint mismatch"):
            db.enroll_embedding("user1", emb, model_name="campp", provenance=prov_ckpt2)

    def test_legacy_unverified_profile_handling(self, tmp_path):
        """Proves that legacy profiles without provenance are marked is_legacy=True / is_verified=False and work cleanly."""
        db_dir = tmp_path / "legacy_db"
        spk_dir = db_dir / "legacy_user"
        spk_dir.mkdir(parents=True)

        centroid = np.random.randn(192).astype(np.float32)
        centroid /= np.linalg.norm(centroid)
        np.save(spk_dir / "centroid.npy", centroid)

        # Write legacy profile.json without provenance or is_verified fields
        legacy_meta = {
            "speaker_id": "legacy_user",
            "sample_count": 1,
            "embedding_dim": 192,
            "model_name": "mock",
            "created_at": 1000000.0,
            "metadata": {"display_name": "Legacy User"},
        }
        with open(spk_dir / "profile.json", "w") as f:
            json.dump(legacy_meta, f)

        db = SpeakerEnrollmentDB(db_dir=db_dir)
        prof = db.get_profile("legacy_user")

        assert prof is not None
        assert prof.is_legacy is True
        assert prof.is_verified is False
        assert prof.provenance is not None
        assert prof.provenance.status == "legacy_unverified"
        assert prof.provenance.is_verified is False
        assert prof.embedding_dim == 192

        # Verify that legacy profile can still be used in matching without errors
        probe = centroid.copy()
        sim = float(np.dot(prof.centroid, probe))
        assert sim > 0.99

    def test_incompatible_enrollment_does_not_leave_orphaned_files(self, tmp_path, generator):
        """Proves that rejecting an incompatible enrollment does not write or leave orphaned WAVs on disk."""
        db_dir = tmp_path / "atomic_enroll_db"
        session = DemonstrationSession(db_dir=db_dir, model_name="mock", default_threshold=0.90)

        # 1. Enroll initial sample under model "mock"
        audio1 = generator.generate_utterance("Siddharth", duration_sec=1.5, utterance_seed=1)
        ok, q, prof = session.enroll_sample("user1", "User 1", audio1)
        assert ok is True

        rec_dir = db_dir / "user1" / "recordings"
        wav_files_before = list(rec_dir.glob("*.wav"))
        assert len(wav_files_before) == 1

        # 2. Simulate active model mismatch
        session.model_name = "incompatible_model"
        audio2 = generator.generate_utterance("Siddharth", duration_sec=1.5, utterance_seed=2)

        with pytest.raises(ValueError, match="Model architecture mismatch"):
            session.enroll_sample("user1", "User 1", audio2)

        # 3. Verify no new or orphaned WAV recording was written to disk
        wav_files_after = list(rec_dir.glob("*.wav"))
        assert len(wav_files_after) == 1
        assert wav_files_before == wav_files_after
