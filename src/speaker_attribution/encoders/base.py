"""
EchoLocate — Model-Agnostic Speaker Encoder Interface
=====================================================
Common abstract base class for all neural speaker embedding models.
Ensures uniform interface for CAM++, ERes2NetV2, ECAPA-TDNN, and future encoders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class SpeakerEncoder(ABC):
    """
    Abstract Base Class for speaker embedding models.
    All models must output strictly L2-normalized 1D or 2D embeddings.
    """

    @abstractmethod
    def load(self, model_path: Optional[str] = None) -> None:
        """Initialize and load network weights."""
        pass

    @abstractmethod
    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract speaker embedding from single audio waveform.

        Args:
            audio: 1D numpy array of shape (N,), mono float32
            sample_rate: Sampling rate (expected: 16000)

        Returns:
            np.ndarray of shape (D,), strictly L2-normalized (||e||_2 == 1.0)
        """
        pass

    @abstractmethod
    def batch_embed(self, audio_batch: List[np.ndarray], sample_rate: int = 16000) -> np.ndarray:
        """
        Extract speaker embeddings for a batch of audio waveforms.

        Args:
            audio_batch: List of 1D numpy arrays
            sample_rate: Sampling rate

        Returns:
            np.ndarray of shape (B, D), each row strictly L2-normalized
        """
        pass

    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the embedding vector dimensionality (e.g. 192 or 512)."""
        pass

    @abstractmethod
    def model_name(self) -> str:
        """Return unique model architecture identifier."""
        pass

    def checkpoint_hash(self) -> Optional[str]:
        """Return SHA256 hex digest of active model weights, or None if unavailable/unhashed."""
        return None

    def frontend_config(self) -> dict:
        """Return feature extraction and preprocessing configuration dictionary."""
        return {
            "sample_rate": 16000,
            "feature_type": "kaldi_80_cmn",
            "num_mel_bins": 80,
            "cmn": True,
        }

    def provenance(self) -> "ModelProvenance":
        """Construct ModelProvenance for embeddings produced by this encoder."""
        from ..types import ModelProvenance
        c_hash = self.checkpoint_hash()
        return ModelProvenance(
            model_name=self.model_name(),
            embedding_dim=self.embedding_dimension(),
            checkpoint_hash=c_hash,
            frontend_version="kaldi_80_cmn",
            sample_rate=16000,
            is_verified=c_hash is not None,
            status="verified" if c_hash is not None else "unverified_weights",
        )
