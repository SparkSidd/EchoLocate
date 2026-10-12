"""
EchoLocate — Real Recording & Hardware Calibration Protocol
============================================================
Defines the experimental recording guidelines, physical array calibration
protocols, sound level measurement rules, and acoustic test procedures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationProtocolSpec:
    """Hardware and spatial calibration parameters for physical microphone deployment."""
    array_name: str
    target_sampling_rate: int = 16000
    bit_depth: int = 24
    mic_height_m: float = 1.20
    source_height_m: float = 1.20
    min_snr_threshold_db: float = 15.0
    direct_sound_alignment_samples: int = 2
    laser_measurement_accuracy_mm: float = 1.0


def verify_channel_synchronization(multichannel_audio: np.ndarray, threshold_us: float = 10.0, fs: int = 16000) -> bool:
    """
    Verify inter-channel hardware synchronization by checking electrical/acoustic alignment.
    Returns True if maximum cross-correlation peak drift is within threshold.
    """
    n_mics = len(multichannel_audio)
    max_shift_samples = int((threshold_us * 1e-6) * fs)
    ref = multichannel_audio[0]

    for i in range(1, n_mics):
        corr = np.correlate(multichannel_audio[i], ref, mode="full")
        lag = np.argmax(corr) - (len(ref) - 1)
        if abs(lag) > max(1, max_shift_samples):
            logger.warning("Channel %d sync drift exceeds threshold: lag=%d samples", i, lag)
            return False
    return True
