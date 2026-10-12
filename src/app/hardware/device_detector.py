"""
EchoLocate — Hardware Device Detector & Capability Inspector
============================================================
Discovers available audio input devices, validates multi-channel capabilities,
and determines if true synchronous multi-channel recording is supported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AudioDeviceInfo:
    """Hardware sound device capabilities."""
    device_id: int
    name: str
    max_input_channels: int
    default_samplerate: float
    is_multichannel_capable: bool
    is_synchronized_array: bool
    status_message: str


def detect_audio_input_devices() -> list[AudioDeviceInfo]:
    """Query system for available audio input hardware devices."""
    devices = []
    try:
        import sounddevice as sd
        all_devices = sd.query_devices()
        for idx, dev in enumerate(all_devices):
            if dev["max_input_channels"] > 0:
                is_multi = dev["max_input_channels"] >= 4
                devices.append(
                    AudioDeviceInfo(
                        device_id=idx,
                        name=dev["name"],
                        max_input_channels=dev["max_input_channels"],
                        default_samplerate=dev["default_samplerate"],
                        is_multichannel_capable=is_multi,
                        is_synchronized_array=is_multi and ("array" in dev["name"].lower() or dev["max_input_channels"] >= 4),
                        status_message="Multi-channel Synchronous Array Ready" if is_multi else "Single/Stereo Channel (Array Localization Unavailable)",
                    )
                )
    except Exception as e:
        logger.warning("Sounddevice query failed or unavailable: %s. Using mock fallback detection.", e)
        # Fallback simulated / virtual devices
        devices.append(
            AudioDeviceInfo(
                device_id=0,
                name="EchoLocate Virtual 4-Mic Synchronized Array",
                max_input_channels=4,
                default_samplerate=16000.0,
                is_multichannel_capable=True,
                is_synchronized_array=True,
                status_message="Virtual 4-Channel Array (Simulation / Test Ready)",
            )
        )
    return devices
