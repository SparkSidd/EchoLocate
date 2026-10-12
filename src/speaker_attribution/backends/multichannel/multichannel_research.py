"""
EchoLocate — Multi-Channel Speaker Attribution Research Branch
==============================================================
Inspired by DiariZen & Spatially Aware Multi-Channel Representations.
Compares single-mic, channel-average, best-channel-SNR, and spatial embedding fusion
across multi-microphone arrays to measure empirical spatial gains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from ...encoders.base import SpeakerEncoder
from ...matching.matcher import CosineSpeakerMatcher
from ...types import SpeakerAttributionResult


MultichannelPolicy = Literal["single_channel", "channel_average", "best_channel_snr", "spatial_fusion"]


@dataclass
class MultichannelComparisonResult:
    """Quantitative comparison of multichannel policies on an acoustic event."""
    single_channel_sim: float
    channel_average_sim: float
    best_snr_sim: float
    spatial_fusion_sim: float
    best_policy: str
    gain_over_single_channel: float
    per_channel_snrs: List[float]


class MultichannelSpeakerProcessor:
    """
    Experimental multi-channel speaker representation processor.
    Evaluates empirical gains of spatial information over single-microphone baselines.
    """

    def __init__(
        self,
        encoder: SpeakerEncoder,
        matcher: Optional[CosineSpeakerMatcher] = None,
    ):
        self.encoder = encoder
        self.matcher = matcher or CosineSpeakerMatcher()

    def estimate_channel_snrs(self, multichannel_audio: np.ndarray) -> List[float]:
        """
        Estimate SNR per channel from multi-channel array (C, N).
        """
        if multichannel_audio.ndim == 1:
            return [10.0]

        channels = multichannel_audio.shape[0]
        snrs = []
        for c in range(channels):
            data = multichannel_audio[c]
            rms = np.sqrt(np.mean(data ** 2) + 1e-10)
            noise_floor = np.percentile(np.abs(data), 15) + 1e-6
            snrs.append(float(20.0 * np.log10(rms / noise_floor)))
        return snrs

    def extract_representation(
        self,
        multichannel_audio: np.ndarray,
        policy: MultichannelPolicy = "spatial_fusion",
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Derive speaker embedding from multi-channel audio using specified policy.
        
        Args:
            multichannel_audio: Shape (C, N) where C is number of channels
            policy: "single_channel", "channel_average", "best_channel_snr", or "spatial_fusion"
        """
        if multichannel_audio.ndim == 1:
            return self.encoder.embed(multichannel_audio, sample_rate=sample_rate)

        num_channels = multichannel_audio.shape[0]

        if policy == "single_channel":
            # Primary microphone (Mic 0)
            return self.encoder.embed(multichannel_audio[0], sample_rate=sample_rate)

        elif policy == "channel_average":
            # Direct time-domain average
            avg_audio = np.mean(multichannel_audio, axis=0)
            return self.encoder.embed(avg_audio, sample_rate=sample_rate)

        elif policy == "best_channel_snr":
            # Pick channel with highest estimated SNR
            snrs = self.estimate_channel_snrs(multichannel_audio)
            best_ch = int(np.argmax(snrs))
            return self.encoder.embed(multichannel_audio[best_ch], sample_rate=sample_rate)

        elif policy == "spatial_fusion":
            # Multi-channel embedding fusion:
            # Extract per-channel embeddings and compute SNR-weighted normalized centroid
            snrs = np.array(self.estimate_channel_snrs(multichannel_audio))
            # Softmax weighting over SNR (in linear domain)
            weights = np.exp(snrs / 10.0)
            weights = weights / np.sum(weights)

            ch_embeddings = []
            for c in range(num_channels):
                emb = self.encoder.embed(multichannel_audio[c], sample_rate=sample_rate)
                ch_embeddings.append(emb)

            # Weighted centroid
            stacked = np.stack(ch_embeddings, axis=0)
            weighted_emb = np.sum(stacked * weights[:, np.newaxis], axis=0)
            norm = np.linalg.norm(weighted_emb)
            if norm > 1e-9:
                weighted_emb = weighted_emb / norm
            return weighted_emb.astype(np.float32)

        else:
            return self.encoder.embed(multichannel_audio[0], sample_rate=sample_rate)

    def compare_policies(
        self,
        multichannel_audio: np.ndarray,
        target_centroid: np.ndarray,
        sample_rate: int = 16000,
    ) -> MultichannelComparisonResult:
        """
        Compare all 4 policies against a known enrolled speaker centroid.
        """
        snrs = self.estimate_channel_snrs(multichannel_audio)

        emb_single = self.extract_representation(multichannel_audio, "single_channel", sample_rate)
        emb_avg = self.extract_representation(multichannel_audio, "channel_average", sample_rate)
        emb_best = self.extract_representation(multichannel_audio, "best_channel_snr", sample_rate)
        emb_spatial = self.extract_representation(multichannel_audio, "spatial_fusion", sample_rate)

        sim_single = float(np.dot(emb_single, target_centroid))
        sim_avg = float(np.dot(emb_avg, target_centroid))
        sim_best = float(np.dot(emb_best, target_centroid))
        sim_spatial = float(np.dot(emb_spatial, target_centroid))

        scores = {
            "single_channel": sim_single,
            "channel_average": sim_avg,
            "best_channel_snr": sim_best,
            "spatial_fusion": sim_spatial,
        }
        best_policy = max(scores, key=scores.get)
        gain = scores[best_policy] - sim_single

        return MultichannelComparisonResult(
            single_channel_sim=sim_single,
            channel_average_sim=sim_avg,
            best_snr_sim=sim_best,
            spatial_fusion_sim=sim_spatial,
            best_policy=best_policy,
            gain_over_single_channel=float(gain),
            per_channel_snrs=snrs,
        )
