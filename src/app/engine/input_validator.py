"""
EchoLocate — Input Signal Validator & Edge-Case Guard
=====================================================
Validates incoming audio frames for digital silence, NaN/Inf values,
channel count mismatch, low signal energy, and clipping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameValidationResult:
    """Audio frame integrity report."""
    is_valid: bool
    is_silent: bool
    is_clipped: bool
    has_nan_or_inf: bool
    rms_energy_db: float
    status: str
    rejection_reason: str = ""


def validate_audio_frame(
    frame_audio: np.ndarray,
    expected_channels: int = 4,
    min_rms_energy_db: float = -60.0,
    max_clip_threshold: float = 0.99,
) -> FrameValidationResult:
    """Inspect and validate multichannel frame (N_channels, N_samples)."""
    if not isinstance(frame_audio, np.ndarray):
        return FrameValidationResult(
            is_valid=False, is_silent=False, is_clipped=False, has_nan_or_inf=False,
            rms_energy_db=-100.0, status="INVALID_TYPE", rejection_reason="Audio is not a numpy array.",
        )

    if frame_audio.ndim != 2 or frame_audio.shape[0] != expected_channels:
        return FrameValidationResult(
            is_valid=False, is_silent=False, is_clipped=False, has_nan_or_inf=False,
            rms_energy_db=-100.0, status="CHANNEL_MISMATCH",
            rejection_reason=f"Expected {expected_channels} channels, got shape {frame_audio.shape}.",
        )

    if np.any(np.isnan(frame_audio)) or np.any(np.isinf(frame_audio)):
        return FrameValidationResult(
            is_valid=False, is_silent=False, is_clipped=False, has_nan_or_inf=True,
            rms_energy_db=-100.0, status="NAN_OR_INF", rejection_reason="Audio frame contains NaN or Inf values.",
        )

    rms = float(np.sqrt(np.mean(frame_audio**2)))
    rms_db = float(20.0 * np.log10(max(1e-9, rms)))

    if rms_db < min_rms_energy_db:
        return FrameValidationResult(
            is_valid=False, is_silent=True, is_clipped=False, has_nan_or_inf=False,
            rms_energy_db=rms_db, status="SILENCE_REJECTED", rejection_reason=f"Signal energy ({rms_db:.1f} dB) below minimum threshold.",
        )

    is_clipped = bool(np.max(np.abs(frame_audio)) >= max_clip_threshold)

    return FrameValidationResult(
        is_valid=True,
        is_silent=False,
        is_clipped=is_clipped,
        has_nan_or_inf=False,
        rms_energy_db=rms_db,
        status="VALID_FRAME",
    )
