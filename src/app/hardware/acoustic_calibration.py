"""
EchoLocate — Acoustic Hardware Synchronization & Calibration Engine
===================================================================
Separates physical acoustic propagation delay from hardware clock latency
using known geometric microphone positions and a reference calibration source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.localization.gcc_phat import gcc_phat

logger = logging.getLogger(__name__)


@dataclass
class AcousticCalibrationResult:
    """Outcome of physical acoustic calibration."""
    source_position_2d: tuple[float, float]
    laptop_mic_2d: tuple[float, float]
    phone_mic_2d: tuple[float, float]
    expected_acoustic_delay_ms: float
    observed_total_delay_ms: float
    estimated_hardware_clock_offset_ms: float
    residual_error_ms: float
    correlation_peak_power: float
    calibration_quality_pct: float
    status: str
    is_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_position_2d": list(self.source_position_2d),
            "laptop_mic_2d": list(self.laptop_mic_2d),
            "phone_mic_2d": list(self.phone_mic_2d),
            "expected_acoustic_delay_ms": round(self.expected_acoustic_delay_ms, 3),
            "observed_total_delay_ms": round(self.observed_total_delay_ms, 3),
            "estimated_hardware_clock_offset_ms": round(self.estimated_hardware_clock_offset_ms, 3),
            "residual_error_ms": round(self.residual_error_ms, 3),
            "correlation_peak_power": round(self.correlation_peak_power, 3),
            "calibration_quality_pct": round(self.calibration_quality_pct, 1),
            "status": self.status,
            "is_valid": self.is_valid,
        }


def generate_reference_calibration_chirp(
    fs: int = 16000,
    duration_s: float = 0.15,
    f0: float = 300.0,
    f1: float = 4000.0,
) -> np.ndarray:
    """Generate linear broadband chirp for acoustic synchronization."""
    t = np.linspace(0, duration_s, int(fs * duration_s), endpoint=False)
    chirp = np.sin(2 * np.pi * (f0 + (f1 - f0) / (2 * duration_s) * t) * t)
    # Apply Tukey window to avoid spectral leakage
    window = np.hanning(len(chirp))
    return (chirp * window).astype(np.float32)


def perform_acoustic_calibration(
    laptop_audio: np.ndarray,
    phone_audio: np.ndarray,
    source_xy: tuple[float, float],
    laptop_mic_xy: tuple[float, float],
    phone_mic_xy: tuple[float, float],
    fs: int = 16000,
    speed_of_sound: float = 343.0,
) -> AcousticCalibrationResult:
    """
    Execute acoustic cross-correlation and decouple clock offset from propagation delay.

    tau_observed = tau_acoustic + tau_clock_offset
    => tau_clock_offset = tau_observed - tau_acoustic
    """
    # 1. Theoretical acoustic delay
    d_lap = float(np.hypot(source_xy[0] - laptop_mic_xy[0], source_xy[1] - laptop_mic_xy[1]))
    d_ph = float(np.hypot(source_xy[0] - phone_mic_xy[0], source_xy[1] - phone_mic_xy[1]))

    # tau_acoustic = (d_lap - d_ph) / c (seconds)
    tau_acoustic_s = (d_lap - d_ph) / speed_of_sound
    tau_acoustic_ms = tau_acoustic_s * 1000.0

    # 2. Observed delay via GCC-PHAT
    # Match lengths
    min_len = min(len(laptop_audio), len(phone_audio))
    sig1 = laptop_audio[:min_len]
    sig2 = phone_audio[:min_len]

    max_tau_samples = int(0.100 * fs)  # 100 ms search window
    res = gcc_phat(sig1, sig2, fs=fs, max_tau_samples=max_tau_samples)

    observed_tau_ms = res.tau_seconds * 1000.0
    peak_pow = float(np.max(res.gcc_phat_response))

    # 3. Decouple Clock Offset
    clock_offset_ms = observed_tau_ms - tau_acoustic_ms
    residual_ms = abs(observed_tau_ms - (tau_acoustic_ms + clock_offset_ms))

    # 4. Calibration Quality Score
    quality = 100.0 * min(1.0, max(0.0, peak_pow / 0.8)) * (1.0 - min(1.0, residual_ms / 2.0))
    quality = float(np.clip(quality, 10.0, 99.0))
    is_valid = peak_pow > 0.15

    status = "CALIBRATED_SUCCESS" if is_valid else "POOR_ACOUSTIC_SNR"

    logger.info(
        "Acoustic Calibration: Expected Geometric=%.2f ms, Observed=%.2f ms => Clock Offset=%.2f ms (Quality: %.1f%%)",
        tau_acoustic_ms, observed_tau_ms, clock_offset_ms, quality,
    )

    return AcousticCalibrationResult(
        source_position_2d=source_xy,
        laptop_mic_2d=laptop_mic_xy,
        phone_mic_2d=phone_mic_xy,
        expected_acoustic_delay_ms=tau_acoustic_ms,
        observed_total_delay_ms=observed_tau_ms,
        estimated_hardware_clock_offset_ms=clock_offset_ms,
        residual_error_ms=residual_ms,
        correlation_peak_power=peak_pow,
        calibration_quality_pct=quality,
        status=status,
        is_valid=is_valid,
    )
