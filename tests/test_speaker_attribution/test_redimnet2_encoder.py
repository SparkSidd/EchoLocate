"""
EchoLocate — Tests: ReDimNet2-B0 Speaker Encoder Adapter
========================================================
Comprehensive regression tests for the ReDimNet2-B0 adapter:
  - Factory registration and backend initialization with mocked model
  - 16 kHz waveform input, resampling, mono conversion, and output shape (192,)
  - Strict unit L2 normalization and handling of non-finite/zero/silent inputs
  - Model provenance and checkpoint identity verification
  - Clear failures on missing or hash-mismatched checkpoints
  - Safe model switching in DemonstrationSession without disk mutation
  - Preserved operation of all existing encoders (CAM++, ERes2NetV2, ECAPA)
  - Real cached checkpoint smoke test (conditional on local weights)
"""

import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn as nn

from src.speaker_attribution.demo.session import DemonstrationSession
from src.speaker_attribution.encoders.base import SpeakerEncoder
from src.speaker_attribution.encoders.campp_encoder import CAMPPEncoder
from src.speaker_attribution.encoders.ecapa_encoder import ECAPAEncoder
from src.speaker_attribution.encoders.eres2netv2_encoder import ERes2NetV2Encoder
from src.speaker_attribution.encoders.factory import ENCODER_REGISTRY, create_encoder
from src.speaker_attribution.encoders.mock_encoder import MockAcousticEncoder
from src.speaker_attribution.encoders.redimnet2_encoder import (
    EXPECTED_B0_SHA256,
    ReDimNet2B0Encoder,
)
from src.speaker_attribution.types import ModelProvenance


class MockReDimNet2Module(nn.Module):
    """
    Lightweight mock module simulating ReDimNet2's forward behavior.
    Returns unnormalized embeddings with norm != 1.0 to verify that
    the adapter performs strict L2 normalization.
    """

    def __init__(self, dim: int = 192, return_non_finite: bool = False, return_zero: bool = False):
        super().__init__()
        self.dim = dim
        self.return_non_finite = return_non_finite
        self.return_zero = return_zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if self.return_non_finite:
            out = torch.full((batch_size, self.dim), float("nan"), device=x.device, dtype=torch.float32)
            return out
        if self.return_zero:
            return torch.zeros((batch_size, self.dim), device=x.device, dtype=torch.float32)
        # Produce vector with norm != 1 (norm = sqrt(192 * 4) ~ 27.71)
        raw = torch.full((batch_size, self.dim), 2.0, device=x.device, dtype=torch.float32)
        return raw


@pytest.fixture
def synthetic_audio():
    """1.5 seconds of 16 kHz synthetic audio."""
    t = np.linspace(0, 1.5, 24000)
    return (0.5 * np.sin(2 * np.pi * 400 * t) + 0.3 * np.sin(2 * np.pi * 800 * t)).astype(np.float32)


def snapshot_dir(dir_path: Path) -> Dict[str, Tuple[str, int, float]]:
    state = {}
    if not dir_path.exists():
        return state
    for p in sorted(dir_path.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(dir_path)).replace("\\", "/")
            data = p.read_bytes()
            h = hashlib.sha256(data).hexdigest()
            state[rel] = (h, len(data), p.stat().st_mtime)
    return state


class TestReDimNet2B0Adapter:
    def test_factory_registration_and_aliases(self):
        """Verify factory registers 'redimnet2_b0' and aliases, defaulting to dim=192."""
        mock_net = MockReDimNet2Module()
        enc1 = create_encoder("redimnet2_b0", model=mock_net)
        enc2 = create_encoder("redimnet2-b0", model=mock_net)
        enc3 = create_encoder("redimnet2", model=mock_net)

        assert isinstance(enc1, ReDimNet2B0Encoder)
        assert isinstance(enc2, ReDimNet2B0Encoder)
        assert isinstance(enc3, ReDimNet2B0Encoder)
        assert enc1.embedding_dimension() == 192
        assert enc1.model_name() == "redimnet2_b0"

    def test_waveform_input_mono_resampling_and_output_shape(self, synthetic_audio):
        """Test waveform input: mono conversion, 16 kHz resampling, and output shape (192,)."""
        mock_net = MockReDimNet2Module()
        encoder = ReDimNet2B0Encoder(model=mock_net)

        # 1. 16 kHz 1D mono
        emb = encoder.embed(synthetic_audio, sample_rate=16000)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (192,)
        assert emb.dtype == np.float32

        # 2. Multi-channel / Stereo (2, N) -> averaged to mono
        stereo_audio = np.stack([synthetic_audio, synthetic_audio * 0.9], axis=0)
        emb_stereo = encoder.embed(stereo_audio, sample_rate=16000)
        assert emb_stereo.shape == (192,)

        # 3. Non-16kHz (resampled to 16kHz)
        t_48k = np.linspace(0, 1.0, 48000)
        audio_48k = np.sin(2 * np.pi * 440 * t_48k).astype(np.float32)
        emb_48k = encoder.embed(audio_48k, sample_rate=48000)
        assert emb_48k.shape == (192,)

        # 4. Batch embed
        batch = [synthetic_audio, synthetic_audio * 0.8]
        embs = encoder.batch_embed(batch, sample_rate=16000)
        assert embs.shape == (2, 192)

        # 5. Empty batch
        empty_embs = encoder.batch_embed([], sample_rate=16000)
        assert empty_embs.shape == (0, 192)

    def test_strict_l2_normalization_and_edge_cases(self, synthetic_audio):
        """Verify strict L2 unit normalization, and explicit errors on silent/non-finite/zero inputs."""
        mock_net = MockReDimNet2Module()
        encoder = ReDimNet2B0Encoder(model=mock_net)

        # Mock model outputs vector of 2.0s (norm ~ 27.71). Encoder MUST normalize to 1.0.
        emb = encoder.embed(synthetic_audio)
        norm = np.linalg.norm(emb)
        assert np.isclose(norm, 1.0, atol=1e-4), f"Expected unit norm 1.0, got {norm}"

        # Silence input -> raises explicit ValueError
        silence = np.zeros(16000, dtype=np.float32)
        with pytest.raises(ValueError, match="silent audio"):
            encoder.embed(silence)

        # Empty audio -> raises explicit ValueError
        with pytest.raises(ValueError, match="empty or None"):
            encoder.embed(np.array([], dtype=np.float32))

        # Audio containing NaN / Inf -> raises explicit ValueError
        dirty_audio = synthetic_audio.copy()
        dirty_audio[100:150] = np.nan
        dirty_audio[200:250] = np.inf
        with pytest.raises(ValueError, match="non-finite samples"):
            encoder.embed(dirty_audio)

        # Model outputting non-finite values -> raises explicit FloatingPointError
        mock_nan_net = MockReDimNet2Module(return_non_finite=True)
        encoder_nan = ReDimNet2B0Encoder(model=mock_nan_net)
        with pytest.raises(FloatingPointError, match="non-finite embedding vector"):
            encoder_nan.embed(synthetic_audio)

        # Model outputting all zeros -> raises explicit FloatingPointError
        mock_zero_net = MockReDimNet2Module(return_zero=True)
        encoder_zero = ReDimNet2B0Encoder(model=mock_zero_net)
        with pytest.raises(FloatingPointError, match="zero norm"):
            encoder_zero.embed(synthetic_audio)

    def test_provenance_and_checkpoint_identity(self):
        """Verify ModelProvenance fields, verified status, and frontend config."""
        mock_net = MockReDimNet2Module()
        encoder = ReDimNet2B0Encoder(model=mock_net)

        prov = encoder.provenance()
        assert isinstance(prov, ModelProvenance)
        assert prov.model_name == "redimnet2_b0"
        assert prov.embedding_dim == 192
        assert prov.checkpoint_hash == EXPECTED_B0_SHA256
        assert prov.frontend_version == "redimnet2_tfmelbanks_80_v1.0.0"
        assert prov.sample_rate == 16000
        assert prov.is_verified is True

        assert prov.status == "verified"

        fe_cfg = encoder.frontend_config()
        assert fe_cfg["num_mel_bins"] == 80
        assert fe_cfg["sample_rate"] == 16000

    def test_failure_on_missing_checkpoint(self, tmp_path):
        """Verify FileNotFoundError is raised when checkpoint is missing (no silent download)."""
        fake_path = tmp_path / "nonexistent" / "b0-vox2-lm.pt"
        with pytest.raises(FileNotFoundError, match="Automatic network downloads are disabled"):
            ReDimNet2B0Encoder(model_path=str(fake_path))

    def test_failure_on_hash_mismatch(self, tmp_path):
        """Verify ValueError is raised when checkpoint hash diverges from verified SHA-256."""
        corrupt_ckpt = tmp_path / "corrupted_b0.pt"
        corrupt_ckpt.write_bytes(b"corrupted weight data that fails sha256 check")

        with pytest.raises(ValueError, match="checkpoint hash mismatch"):
            ReDimNet2B0Encoder(model_path=str(corrupt_ckpt))

    def test_switch_to_b0_without_overwriting_profiles(self, tmp_path, synthetic_audio):
        """Verify switching to ReDimNet2-B0 does not mutate stored profiles or centroids on disk."""
        db_dir = tmp_path / "speakers_db"
        spk_dir = db_dir / "speaker_test"
        rec_dir = spk_dir / "recordings"
        rec_dir.mkdir(parents=True, exist_ok=True)

        # Write dummy recording
        wav_file = rec_dir / "001.wav"
        sf.write(str(wav_file), synthetic_audio, 16000, subtype="PCM_16")

        # Create session with mock model and enroll
        session = DemonstrationSession(db_dir=db_dir, model_name="mock", default_threshold=0.90)
        ok, qc, prof = session.enroll_sample(
            "speaker_test", "Speaker Test", synthetic_audio, allow_override_quality=True
        )
        assert ok is True

        before_state = snapshot_dir(db_dir)
        assert len(before_state) > 0

        # Inject mock redimnet2 model into factory for offline test execution
        mock_b0 = MockReDimNet2Module()
        orig_cls = ENCODER_REGISTRY["redimnet2_b0"]

        class InjectedB0Encoder(ReDimNet2B0Encoder):
            def __init__(self, *args, **kwargs):
                super().__init__(model=mock_b0)

        ENCODER_REGISTRY["redimnet2_b0"] = InjectedB0Encoder
        try:
            # Switch to B0 with persist=False
            session.switch_model("redimnet2_b0", persist=False)
            assert session.model_name == "redimnet2_b0"
            assert session.encoder.embedding_dimension() == 192

            # In-memory centroids updated
            mem_prof = session.db.get_profile("speaker_test")
            assert mem_prof.embedding_dim == 192
            assert mem_prof.model_name == "redimnet2_b0"

            # Disk state must remain 100% identical
            after_state = snapshot_dir(db_dir)
            assert set(before_state.keys()) == set(after_state.keys())
            for rel_path, (b_hash, b_size, _) in before_state.items():
                a_hash, a_size, _ = after_state[rel_path]
                assert b_hash == a_hash, f"File {rel_path} was modified on disk during switch_model!"
                assert b_size == a_size, f"File {rel_path} size changed during switch_model!"
        finally:
            ENCODER_REGISTRY["redimnet2_b0"] = orig_cls

    def test_existing_encoders_remain_fully_functional(self, synthetic_audio):
        """Ensure CAM++, ERes2NetV2, ECAPA, and default factory instantiate and function unchanged."""
        # 1. Default factory instantiation (must remain CAM++)
        default_enc = create_encoder()
        assert isinstance(default_enc, CAMPPEncoder)
        assert default_enc.model_name() == "campp"

        # 2. CAM++
        campp = create_encoder("campp")
        assert isinstance(campp, CAMPPEncoder)
        assert campp.embedding_dimension() == 192

        # 3. ERes2NetV2
        eres = create_encoder("eres2netv2")
        assert isinstance(eres, ERes2NetV2Encoder)
        assert eres.embedding_dimension() == 512

        # 4. ECAPA
        ecapa = create_encoder("ecapa")
        assert isinstance(ecapa, ECAPAEncoder)
        assert ecapa.embedding_dimension() == 192

        # 5. Mock
        mock_enc = create_encoder("mock")
        assert isinstance(mock_enc, MockAcousticEncoder)

    def test_real_cached_b0_smoke_test(self, synthetic_audio):
        """Smoke test with the real verified B0 checkpoint if cached locally."""
        cached_ckpt = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "b0-vox2-lm.pt"
        if not cached_ckpt.exists():
            pytest.skip(f"Local B0 checkpoint not found at {cached_ckpt}")

        try:
            encoder = ReDimNet2B0Encoder()
        except ImportError as e:
            pytest.skip(f"ReDimNet2 upstream module not importable: {e}")

        assert encoder.model_name() == "redimnet2_b0"
        assert encoder.embedding_dimension() == 192
        assert encoder.checkpoint_hash() == EXPECTED_B0_SHA256
        assert encoder.provenance().is_verified is True

        emb = encoder.embed(synthetic_audio, sample_rate=16000)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (192,)
        assert np.isfinite(emb).all()
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-4)

    def test_invalid_audio_rejection_across_identification_boundary(self, tmp_path, synthetic_audio):
        """Verify that silence, NaN/Inf, and non-finite model outputs safely reject as UNKNOWN."""
        from src.speaker_attribution.enrollment.database import SpeakerEnrollmentDB
        from src.speaker_attribution.identifier import SpeakerIdentifier
        from src.speaker_attribution.matching.matcher import CosineSpeakerMatcher
        from src.speaker_attribution.types import AttributionStatus

        db = SpeakerEnrollmentDB(db_dir=tmp_path / "test_boundary_db")
        mock_net = MockReDimNet2Module()
        encoder = ReDimNet2B0Encoder(model=mock_net)
        matcher = CosineSpeakerMatcher(default_threshold=0.50)
        ident = SpeakerIdentifier(encoder=encoder, enrollment_db=db, matcher=matcher)

        # Enroll Siddharth with valid audio
        prof = ident.enroll("siddharth", synthetic_audio)
        assert prof.sample_count == 1

        # 1. Silent input -> must be UNKNOWN, never assigned identity
        silence = np.zeros(16000, dtype=np.float32)
        res_silence = ident.identify(silence)
        assert res_silence.speaker_id == "UNKNOWN"
        assert res_silence.status in (AttributionStatus.UNKNOWN, AttributionStatus.SILENCE, AttributionStatus.INSUFFICIENT_AUDIO)
        assert res_silence.confidence == 0.0


        # 2. NaN/Inf input -> must be UNKNOWN, never assigned identity
        nan_audio = synthetic_audio.copy()
        nan_audio[100:150] = np.nan
        res_nan = ident.identify(nan_audio)
        assert res_nan.speaker_id == "UNKNOWN"
        assert res_nan.status in (AttributionStatus.UNKNOWN, AttributionStatus.INSUFFICIENT_AUDIO)

        # 3. Model producing non-finite output -> must be UNKNOWN, never assigned identity
        nan_encoder = ReDimNet2B0Encoder(model=MockReDimNet2Module(return_non_finite=True))
        ident_nan = SpeakerIdentifier(encoder=nan_encoder, enrollment_db=db, matcher=matcher)
        res_nan_model = ident_nan.identify(synthetic_audio)
        assert res_nan_model.speaker_id == "UNKNOWN"
        assert res_nan_model.status in (AttributionStatus.UNKNOWN, AttributionStatus.INSUFFICIENT_AUDIO)


    def test_source_revision_reproducibility(self):
        """Verify that source revision constants match immutable v1.0.0 commit 5294667 across all 12 modules."""
        from src.speaker_attribution.encoders.redimnet2_encoder import (
            PINNED_V1_0_0_COMMIT,
            PINNED_V1_0_0_MANIFEST,
            verify_source_tree_integrity,
        )

        assert PINNED_V1_0_0_COMMIT == "5294667"
        assert len(PINNED_V1_0_0_MANIFEST) == 12
        assert PINNED_V1_0_0_MANIFEST["redimnet2/redimnet2.py"] == "33d5d4524f4d53d642f4919a5fb8ffb875316c19"

        # Verify tracked third_party directory
        third_party_dir = Path("third_party/redimnet2")
        assert third_party_dir.exists(), "Tracked third_party/redimnet2 directory must exist for fresh checkouts"
        assert (third_party_dir / "LICENSE").exists(), "Upstream MIT LICENSE must be present in third_party/redimnet2"
        assert verify_source_tree_integrity(third_party_dir) is True

        # Verify offline copy in demo_data/models/redimnet2/v1.0.0_src if present
        offline_src = Path("demo_data/models/redimnet2/v1.0.0_src")
        if offline_src.exists():
            assert verify_source_tree_integrity(offline_src) is True

        # Verify that an unpinned or corrupt directory is rejected
        mock_bad_dir = Path("demo_data/models/redimnet2")
        assert verify_source_tree_integrity(mock_bad_dir) is False

    def test_fresh_checkout_loads_from_tracked_third_party(self):
        """Verify that _resolve_redimnet2_module loads directly from tracked third_party directory."""
        from src.speaker_attribution.encoders.redimnet2_encoder import _resolve_redimnet2_module

        ReDimNet2Wrap = _resolve_redimnet2_module()
        assert ReDimNet2Wrap is not None
        # Check module file path originates from project third_party
        mod_file = Path(sys.modules[ReDimNet2Wrap.__module__].__file__).resolve().as_posix()
        assert "third_party/redimnet2" in mod_file or "v1.0.0" in mod_file

    def test_clean_env_rejects_moving_main_cache(self, tmp_path):
        """Verify that a source tree matching the unpinned moving main branch is strictly rejected."""
        from src.speaker_attribution.encoders.redimnet2_encoder import verify_source_tree_integrity

        # Create a mock source tree mimicking moving main branch (where blocks.py or redimnet2.py differs)
        fake_main = tmp_path / "mock_main"
        fake_main.mkdir(parents=True, exist_ok=True)
        pkg = fake_main / "redimnet2"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "redimnet2.py").write_text("# modified code differing from commit 5294667", encoding="utf-8")

        # Must fail integrity verification
        assert verify_source_tree_integrity(fake_main) is False
