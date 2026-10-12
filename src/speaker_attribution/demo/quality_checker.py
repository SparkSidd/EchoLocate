"""
EchoLocate — Enrollment Audio Quality Checker
=============================================
Validates speech duration, RMS level, estimated SNR, clipping, and speech activity
before allowing an utterance into a speaker profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..preprocessing.vad import EnergyVAD


@dataclass
class QualityCheckResult:
    """Diagnostic outcome of an enrollment or test utterance quality check."""

    passed: bool
    status_reason: str
    total_duration_sec: float
    speech_duration_sec: float
    speech_ratio: float
    rms: float
    snr_db: float
    is_clipped: bool

    def summary_line(self) -> str:
        verdict = "ACCEPTED" if self.passed else f"REJECTED ({self.status_reason})"
        return (
            f"[{verdict}] Duration: {self.total_duration_sec:.2f}s "
            f"(Speech: {self.speech_duration_sec:.2f}s, {self.speech_ratio * 100:.0f}%) | "
            f"RMS: {self.rms:.4f} | Est SNR: {self.snr_db:.1f} dB"
        )


class EnrollmentQualityChecker:
    """
    Evaluates real recorded utterances against strict quality gates:
      1. Minimum speech duration (default: 0.8s)
      2. Minimum signal RMS (detect dead mic / mute)
      3. Minimum estimated SNR (speech energy vs ambient floor)
      4. Clipping threshold (prevent severely distorted recordings)
      5. Minimum speech activity ratio (reject pure silence/room noise)
    """

    def __init__(
        self,
        min_speech_duration_sec: float = 0.8,
        min_rms: float = 0.005,
        min_snr_db: float = 6.0,
        max_clipping_ratio: float = 0.005,
        min_speech_ratio: float = 0.25,
        sample_rate: int = 16000,
    ):
        self.min_speech_duration_sec = min_speech_duration_sec
        self.min_rms = min_rms
        self.min_snr_db = min_snr_db
        self.max_clipping_ratio = max_clipping_ratio
        self.min_speech_ratio = min_speech_ratio
        self.sample_rate = sample_rate
        self.vad = EnergyVAD()

    def check(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> QualityCheckResult:
        """Analyze 1D float32 audio and return pass/reject verdict with diagnostics."""
        sr = int(sample_rate) if sample_rate is not None else self.sample_rate
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0 if audio.shape[0] < audio.shape[1] else 1)
        audio = audio.astype(np.float32)

        total_samples = len(audio)
        total_duration = total_samples / float(sr)

        if total_samples == 0:
            return QualityCheckResult(
                passed=False,
                status_reason="empty_audio",
                total_duration_sec=0.0,
                speech_duration_sec=0.0,
                speech_ratio=0.0,
                rms=0.0,
                snr_db=-99.0,
                is_clipped=False,
            )

        # 1. RMS level
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < self.min_rms:
            return QualityCheckResult(
                passed=False,
                status_reason="too_quiet_or_silence",
                total_duration_sec=total_duration,
                speech_duration_sec=0.0,
                speech_ratio=0.0,
                rms=rms,
                snr_db=-99.0,
                is_clipped=False,
            )

        # 2. Clipping check
        clip_count = np.sum(np.abs(audio) >= 0.99)
        clipping_ratio = clip_count / float(total_samples)
        is_clipped = clipping_ratio > self.max_clipping_ratio

        # 3. Speech Activity Detection (VAD)
        frame_mask = self.vad.detect(audio, sample_rate=self.sample_rate)
        frame_len = int(round(self.sample_rate * self.vad.frame_duration_ms / 1000.0))
        num_frames = len(frame_mask)
        speech_frames = int(np.sum(frame_mask))
        speech_samples = speech_frames * frame_len
        speech_duration = speech_samples / float(self.sample_rate)
        speech_ratio = speech_samples / float(max(1, total_samples))

        # 4. SNR estimation (speech RMS vs non-speech RMS)
        # Repeat frame mask to sample mask
        sample_mask = np.repeat(frame_mask, frame_len)[:total_samples]
        if len(sample_mask) < total_samples:
            sample_mask = np.pad(sample_mask, (0, total_samples - len(sample_mask)), mode="edge")

        speech_audio = audio[sample_mask]
        noise_audio = audio[~sample_mask]

        speech_rms = float(np.sqrt(np.mean(speech_audio ** 2))) if len(speech_audio) > 0 else rms
        noise_rms = float(np.sqrt(np.mean(noise_audio ** 2))) if len(noise_audio) > 0 else 1e-4

        if noise_rms < 1e-6:
            snr_db = 40.0
        else:
            snr_db = float(20.0 * math.log10(max(1e-5, speech_rms) / max(1e-6, noise_rms)))

        # Validate gates
        if is_clipped:
            reason = "audio_clipped_distorted"
            passed = False
        elif speech_duration < self.min_speech_duration_sec:
            reason = f"speech_too_short ({speech_duration:.2f}s < {self.min_speech_duration_sec:.2f}s)"
            passed = False
        elif speech_ratio < self.min_speech_ratio:
            reason = f"insufficient_speech_ratio ({speech_ratio * 100:.0f}% < {self.min_speech_ratio * 100:.0f}%)"
            passed = False
        elif snr_db < self.min_snr_db:
            reason = f"snr_too_low ({snr_db:.1f}dB < {self.min_snr_db:.1f}dB)"
            passed = False
        else:
            reason = "ok"
            passed = True

        return QualityCheckResult(
            passed=bool(passed),
            status_reason=str(reason),
            total_duration_sec=float(total_duration),
            speech_duration_sec=float(speech_duration),
            speech_ratio=float(speech_ratio),
            rms=float(rms),
            snr_db=float(snr_db),
            is_clipped=bool(is_clipped),
        )
