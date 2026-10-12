"""
EchoLocate — Deterministic Audio Preprocessing Pipeline
======================================================
Strict single-channel standardizer for speaker recognition.
Ensures 16 kHz mono float32 audio with DC removal and normalization.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Union

import numpy as np
from scipy.signal import resample_poly

from ..types import AudioSegment


ChannelPolicy = Literal["first", "average", "mono", "best_snr"]


class AudioProcessor:
    """
    Deterministic audio preprocessor for speaker identification.
    Strictly separates multi-channel spatial processing from single-channel speaker embedding.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        channel_policy: ChannelPolicy = "average",
        remove_dc: bool = True,
        normalize_amplitude: bool = True,
        target_peak: float = 0.95,
        min_duration_sec: float = 0.4,
    ):
        self.target_sample_rate = target_sample_rate
        self.channel_policy = channel_policy
        self.remove_dc = remove_dc
        self.normalize_amplitude = normalize_amplitude
        self.target_peak = target_peak
        self.min_duration_sec = min_duration_sec

    def convert_to_mono(
        self,
        audio: np.ndarray,
        policy: Optional[ChannelPolicy] = None,
    ) -> np.ndarray:
        """
        Convert multi-channel audio to mono based on policy.
        
        Args:
            audio: Shape (N,) or (C, N) or (N, C)
            policy: ChannelPolicy override
        """
        policy = policy or self.channel_policy
        audio = audio.astype(np.float32)

        if audio.ndim == 1:
            return audio

        # Determine channel axis: assuming (C, N) if C < 32 and N > C
        if audio.shape[0] < audio.shape[1]:
            channels, samples = audio.shape[0], audio.shape[1]
            data = audio
        else:
            channels, samples = audio.shape[1], audio.shape[0]
            data = audio.T

        if policy == "first":
            return data[0]
        elif policy == "average" or policy == "mono":
            return np.mean(data, axis=0)
        elif policy == "best_snr":
            # Estimate SNR per channel using variance / noise floor
            snrs = []
            for c in range(channels):
                ch_data = data[c]
                rms = np.sqrt(np.mean(ch_data ** 2) + 1e-10)
                noise_floor = np.percentile(np.abs(ch_data), 10) + 1e-6
                snrs.append(rms / noise_floor)
            best_idx = int(np.argmax(snrs))
            return data[best_idx]
        else:
            return data[0]

    def resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
    ) -> np.ndarray:
        """
        Resample 1D audio to target_sample_rate using polyphase filtering.
        """
        if orig_sr == self.target_sample_rate or len(audio) == 0:
            return audio

        gcd = math.gcd(int(orig_sr), int(self.target_sample_rate))
        up = self.target_sample_rate // gcd
        down = orig_sr // gcd

        resampled = resample_poly(audio, up, down).astype(np.float32)
        return resampled

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        start_sec: float = 0.0,
    ) -> AudioSegment:
        """
        Execute full deterministic preprocessing pipeline:
            Input -> Mono -> Resample(16kHz) -> DC Removal -> Peak Norm -> Validate Duration
        """
        # 1. Mono conversion
        mono = self.convert_to_mono(audio)

        # 2. Resampling
        resampled = self.resample(mono, orig_sr=sample_rate)

        # 3. DC offset removal
        if self.remove_dc and len(resampled) > 0:
            resampled = resampled - np.mean(resampled)

        # 4. Amplitude normalization
        if self.normalize_amplitude and len(resampled) > 0:
            peak = np.max(np.abs(resampled))
            if peak > 1e-6:
                resampled = (resampled / peak) * self.target_peak

        duration_sec = len(resampled) / self.target_sample_rate if self.target_sample_rate > 0 else 0.0
        is_sufficient = duration_sec >= self.min_duration_sec

        return AudioSegment(
            samples=resampled,
            sample_rate=self.target_sample_rate,
            start_sec=start_sec,
            end_sec=start_sec + duration_sec,
            speech_ratio=1.0,
            is_speech=is_sufficient,
            metadata={"is_duration_valid": is_sufficient, "orig_sample_rate": sample_rate},
        )
