"""
EchoLocate — Mock / Lightweight Acoustic Feature Encoder
========================================================
High-speed deterministic mock encoder for testing, offline benchmarking,
and lightweight integration tests without heavy deep network overhead.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import SpeakerEncoder


class MockAcousticEncoder(SpeakerEncoder):
    """
    Deterministic acoustic fingerprint encoder based on spectral moments & statistics.
    Outputs normalized 192-dim vectors that vary consistently with pitch, spectral envelope, and harmonics.
    """

    def __init__(self, embedding_dim: int = 192, name: str = "mock"):
        self._dim = embedding_dim
        self._name = name

    def load(self, model_path: Optional[str] = None) -> None:
        pass

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        if len(audio) == 0:
            return np.zeros(self._dim, dtype=np.float32)

        # Compute deterministic acoustic properties
        # 1. Zero crossing rate
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0)
        # 2. RMS energy
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        # 3. FFT energy distribution across bins
        n_fft = 512
        fft = np.abs(np.fft.rfft(audio[: min(len(audio), 4000)], n=n_fft))
        fft_norm = fft / (np.sum(fft) + 1e-9)

        # Construct deterministic embedding
        emb = np.zeros(self._dim, dtype=np.float32)
        n_bins = min(len(fft_norm), self._dim - 10)
        emb[:n_bins] = fft_norm[:n_bins]
        emb[-4] = zcr
        emb[-3] = rms
        emb[-2] = float(np.mean(audio))
        emb[-1] = float(np.std(audio))

        # Modulate by speaker-specific characteristics
        norm = np.linalg.norm(emb)
        if norm > 1e-9:
            emb = emb / norm
        return emb.astype(np.float32)

    def batch_embed(self, audio_batch: List[np.ndarray], sample_rate: int = 16000) -> np.ndarray:
        return np.stack([self.embed(a, sample_rate=sample_rate) for a in audio_batch], axis=0)

    def embedding_dimension(self) -> int:
        return self._dim

    def model_name(self) -> str:
        return self._name
