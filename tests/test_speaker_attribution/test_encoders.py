"""
EchoLocate — Tests: Neural Speaker Encoders
===========================================
Validates CAM++, ERes2NetV2, ECAPA-TDNN, and Mock encoders for forward pass,
embedding dimensionality, strict L2 normalization, and factory instantiation.
"""

import numpy as np
import pytest

from src.speaker_attribution.encoders.base import SpeakerEncoder
from src.speaker_attribution.encoders.campp_encoder import CAMPPEncoder
from src.speaker_attribution.encoders.ecapa_encoder import ECAPAEncoder
from src.speaker_attribution.encoders.eres2netv2_encoder import ERes2NetV2Encoder
from src.speaker_attribution.encoders.factory import create_encoder
from src.speaker_attribution.encoders.mock_encoder import MockAcousticEncoder


class TestSpeakerEncoders:
    @pytest.fixture
    def test_audio(self):
        # 1.5 seconds of synthetic audio
        t = np.linspace(0, 1.5, 24000)
        return (0.5 * np.sin(2 * np.pi * 350 * t) + 0.2 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)

    def test_campp_encoder(self, test_audio):
        encoder = CAMPPEncoder(embedding_dim=512)
        emb = encoder.embed(test_audio, sample_rate=16000)

        assert isinstance(emb, np.ndarray)
        assert emb.shape == (512,)
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)
        assert encoder.embedding_dimension() == 512
        assert encoder.model_name() == "campp"

    def test_eres2netv2_encoder(self, test_audio):
        encoder = ERes2NetV2Encoder(embedding_dim=192)
        emb = encoder.embed(test_audio, sample_rate=16000)

        assert isinstance(emb, np.ndarray)
        assert emb.shape == (192,)
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)
        assert encoder.embedding_dimension() == 192
        assert encoder.model_name() == "eres2netv2"

    def test_ecapa_encoder(self, test_audio):
        encoder = ECAPAEncoder(embedding_dim=192)
        emb = encoder.embed(test_audio, sample_rate=16000)

        assert isinstance(emb, np.ndarray)
        assert emb.shape == (192,)
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)
        assert encoder.embedding_dimension() == 192
        assert encoder.model_name() == "ecapa"

    def test_mock_encoder(self, test_audio):
        encoder = MockAcousticEncoder(embedding_dim=192)
        emb1 = encoder.embed(test_audio, sample_rate=16000)
        emb2 = encoder.embed(test_audio, sample_rate=16000)

        assert emb1.shape == (192,)
        assert np.isclose(np.linalg.norm(emb1), 1.0, atol=1e-4)
        # Deterministic
        assert np.allclose(emb1, emb2)

    def test_batch_embed(self, test_audio):
        encoder = MockAcousticEncoder(embedding_dim=192)
        batch = [test_audio, test_audio * 0.5, test_audio * 0.8]
        embs = encoder.batch_embed(batch, sample_rate=16000)

        assert embs.shape == (3, 192)
        for i in range(3):
            assert np.isclose(np.linalg.norm(embs[i]), 1.0, atol=1e-4)

    def test_encoder_factory(self):
        m1 = create_encoder("campp")
        m2 = create_encoder("eres2netv2")
        m3 = create_encoder("ecapa")
        m4 = create_encoder("mock")

        assert isinstance(m1, CAMPPEncoder)
        assert isinstance(m2, ERes2NetV2Encoder)
        assert isinstance(m3, ECAPAEncoder)
        assert isinstance(m4, MockAcousticEncoder)

        with pytest.raises(ValueError):
            create_encoder("nonexistent_model_xyz")
