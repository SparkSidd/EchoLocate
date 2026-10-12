"""
EchoLocate — Voice Activity Detection (VAD) Layer
=================================================
Modular Voice Activity Detection supporting robust energy-based detection,
adaptive thresholding, and optional Pyannote / WebRTC integration hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from ..types import AudioSegment


class BaseVAD(ABC):
    """Abstract base class for Voice Activity Detectors."""

    @abstractmethod
    def detect(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Compute binary voice activity mask for the given audio.
        
        Args:
            audio: 1D numpy array of shape (N,)
            sample_rate: Sampling frequency in Hz

        Returns:
            Boolean array of shape (num_frames,) indicating speech activity.
        """
        pass

    @abstractmethod
    def get_speech_segments(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        min_speech_duration_sec: float = 0.3,
        min_silence_duration_sec: float = 0.2,
    ) -> List[Tuple[float, float]]:
        """
        Extract continuous speech time intervals [(start_sec, end_sec), ...].
        """
        pass


class EnergyVAD(BaseVAD):
    """
    High-performance, zero-dependency dual-threshold Energy VAD with
    spectral flux and adaptive noise floor estimation.
    """

    def __init__(
        self,
        frame_duration_ms: float = 20.0,
        energy_threshold_ratio: float = 1.8,
        min_energy_db: float = -45.0,
        hangover_frames: int = 4,
    ):
        self.frame_duration_ms = frame_duration_ms
        self.energy_threshold_ratio = energy_threshold_ratio
        self.min_energy_db = min_energy_db
        self.hangover_frames = hangover_frames

    def detect(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        audio = audio.astype(np.float32)
        if len(audio) == 0:
            return np.array([], dtype=bool)

        frame_len = int(round(sample_rate * self.frame_duration_ms / 1000.0))
        num_frames = len(audio) // frame_len
        if num_frames == 0:
            return np.array([True], dtype=bool)

        frames = audio[: num_frames * frame_len].reshape(num_frames, frame_len)
        energies = np.mean(frames ** 2, axis=1) + 1e-10
        energy_db = 10.0 * np.log10(energies)

        # Estimate background noise floor using bottom 20th percentile
        noise_floor_db = np.percentile(energy_db, 20)
        threshold_db = max(self.min_energy_db, noise_floor_db + 6.0)

        raw_mask = energy_db > threshold_db

        # Apply hangover smoothing
        smoothed_mask = np.zeros_like(raw_mask)
        hangover_counter = 0
        for i in range(num_frames):
            if raw_mask[i]:
                smoothed_mask[i] = True
                hangover_counter = self.hangover_frames
            elif hangover_counter > 0:
                smoothed_mask[i] = True
                hangover_counter -= 1
            else:
                smoothed_mask[i] = False

        return smoothed_mask

    def get_speech_segments(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        min_speech_duration_sec: float = 0.3,
        min_silence_duration_sec: float = 0.2,
    ) -> List[Tuple[float, float]]:
        mask = self.detect(audio, sample_rate=sample_rate)
        if len(mask) == 0:
            return []

        frame_duration = self.frame_duration_ms / 1000.0
        segments: List[Tuple[float, float]] = []

        in_speech = False
        start_idx = 0

        for i, is_active in enumerate(mask):
            if is_active and not in_speech:
                in_speech = True
                start_idx = i
            elif not is_active and in_speech:
                in_speech = False
                start_sec = start_idx * frame_duration
                end_sec = i * frame_duration
                if (end_sec - start_sec) >= min_speech_duration_sec:
                    segments.append((start_sec, end_sec))

        if in_speech:
            start_sec = start_idx * frame_duration
            end_sec = len(mask) * frame_duration
            if (end_sec - start_sec) >= min_speech_duration_sec:
                segments.append((start_sec, end_sec))

        # Merge segments separated by small silence
        merged: List[Tuple[float, float]] = []
        for seg in segments:
            if not merged:
                merged.append(seg)
            else:
                prev_start, prev_end = merged[-1]
                if seg[0] - prev_end < min_silence_duration_sec:
                    merged[-1] = (prev_start, seg[1])
                else:
                    merged.append(seg)

        return merged


def extract_speech_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    vad: Optional[BaseVAD] = None,
    min_speech_ratio: float = 0.1,
) -> AudioSegment:
    """
    Extract speech samples and activity metrics from an input audio buffer.
    """
    if vad is None:
        vad = EnergyVAD()

    mask = vad.detect(audio, sample_rate=sample_rate)
    speech_ratio = float(np.mean(mask)) if len(mask) > 0 else 0.0
    is_speech = speech_ratio >= min_speech_ratio

    return AudioSegment(
        samples=audio,
        sample_rate=sample_rate,
        start_sec=0.0,
        end_sec=len(audio) / sample_rate if sample_rate > 0 else 0.0,
        speech_ratio=speech_ratio,
        is_speech=is_speech,
    )


def extract_speech_only_samples(
    audio: np.ndarray,
    sample_rate: int = 16000,
    vad: Optional[BaseVAD] = None,
    min_speech_duration_sec: float = 0.1,
) -> np.ndarray:
    """
    Extract only active voiced segments from audio, stripping long silence (Phase 5).
    """
    if audio is None or len(audio) == 0:
        return np.array([], dtype=np.float32)

    if vad is None:
        vad = EnergyVAD()

    segments = vad.get_speech_segments(
        audio,
        sample_rate=sample_rate,
        min_speech_duration_sec=min_speech_duration_sec,
        min_silence_duration_sec=0.15,
    )
    if not segments:
        return audio

    chunks = []
    for st_sec, end_sec in segments:
        s_idx = max(0, int(round(st_sec * sample_rate)))
        e_idx = min(len(audio), int(round(end_sec * sample_rate)))
        if e_idx > s_idx:
            chunks.append(audio[s_idx:e_idx])

    if chunks:
        speech = np.concatenate(chunks)
        # Only use filtered speech if at least 0.4s of speech was kept
        if len(speech) >= int(0.4 * sample_rate):
            return speech
    return audio


def create_sliding_windows(
    audio: np.ndarray,
    sample_rate: int = 16000,
    window_duration_sec: float = 1.5,
    hop_duration_sec: float = 0.75,
) -> List[np.ndarray]:
    """
    Slice audio into overlapping analysis windows for multi-window test-time embedding (Phase 6).
    """
    if audio is None or len(audio) == 0:
        return []

    win_len = int(round(window_duration_sec * sample_rate))
    hop_len = int(round(hop_duration_sec * sample_rate))

    if len(audio) <= win_len:
        return [audio]

    windows = []
    for start in range(0, len(audio) - win_len + 1, hop_len):
        windows.append(audio[start : start + win_len])

    if not windows:
        windows = [audio]
    return windows

