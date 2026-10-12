"""
EchoLocate — Multi-Device Hardware Registry
============================================
Extensible registry tracking connected microphones (laptop, smartphone, network, USB arrays).
Supports dynamic registration, coordinate assignment, clock synchronization metadata,
and multi-microphone scalability (N >= 2).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    LOCAL_BUILTIN = "LOCAL_BUILTIN"
    LOCAL_USB = "LOCAL_USB"
    SMARTPHONE_NETWORK = "SMARTPHONE_NETWORK"
    SYNCHRONIZED_ARRAY = "SYNCHRONIZED_ARRAY"
    VIRTUAL_EMULATED = "VIRTUAL_EMULATED"


class SyncStatus(str, Enum):
    HARDWARE_LOCKED = "HARDWARE_LOCKED"
    CALIBRATED_ACOUSTIC = "CALIBRATED_ACOUSTIC"
    NETWORK_ESTIMATED = "NETWORK_ESTIMATED"
    UNSYNCHRONIZED = "UNSYNCHRONIZED"


@dataclass
class HardwareDevice:
    """Represents a registered physical or network audio capture device."""
    device_id: str
    name: str
    device_type: DeviceType
    channels: int = 1
    sample_rate: int = 16000
    position_2d: tuple[float, float] = (0.0, 0.0)
    is_active: bool = True
    clock_offset_ms: float = 0.0
    clock_drift_ppm: float = 0.0
    network_rtt_ms: float = 0.0
    sync_status: SyncStatus = SyncStatus.UNSYNCHRONIZED
    last_seen_timestamp: float = field(default_factory=time.time)
    total_frames_received: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "device_type": self.device_type.value,
            "channels": self.channels,
            "sample_rate": self.sample_rate,
            "position_2d": list(self.position_2d),
            "is_active": self.is_active,
            "clock_offset_ms": round(self.clock_offset_ms, 3),
            "clock_drift_ppm": round(self.clock_drift_ppm, 2),
            "network_rtt_ms": round(self.network_rtt_ms, 2),
            "sync_status": self.sync_status.value,
            "last_seen_timestamp": self.last_seen_timestamp,
            "total_frames_received": self.total_frames_received,
        }


class MultiDeviceRegistry:
    """Manages multi-device lifecycle, positions, and synchronization states."""

    def __init__(self) -> None:
        self._devices: dict[str, HardwareDevice] = {}
        self._initialize_default_devices()

    def _initialize_default_devices(self) -> None:
        """Initialize primary laptop microphone and default smartphone placeholder."""
        # Device 1: Laptop Built-in Mic (Default at X=2.0m, Y=2.0m)
        laptop_mic = HardwareDevice(
            device_id="laptop_mic_primary",
            name="Laptop Built-in Microphone",
            device_type=DeviceType.LOCAL_BUILTIN,
            channels=1,
            sample_rate=16000,
            position_2d=(2.0, 2.0),
            sync_status=SyncStatus.HARDWARE_LOCKED,
            is_active=True,
        )
        self.register_device(laptop_mic)

        # Device 2: Smartphone Network Mic (Default at X=4.0m, Y=2.0m -> 2.0m separation)
        phone_mic = HardwareDevice(
            device_id="smartphone_client_primary",
            name="Smartphone Network Microphone",
            device_type=DeviceType.SMARTPHONE_NETWORK,
            channels=1,
            sample_rate=16000,
            position_2d=(4.0, 2.0),
            sync_status=SyncStatus.UNSYNCHRONIZED,
            is_active=False,
        )
        self.register_device(phone_mic)

    def register_device(self, device: HardwareDevice) -> None:
        """Register or update a hardware device."""
        self._devices[device.device_id] = device
        logger.info("Registered hardware device: %s [%s]", device.name, device.device_type.value)

    def get_device(self, device_id: str) -> HardwareDevice | None:
        """Get device by ID."""
        return self._devices.get(device_id)

    def list_devices(self) -> list[HardwareDevice]:
        """List all registered devices."""
        return list(self._devices.values())

    def update_position(self, device_id: str, x: float, y: float) -> bool:
        """Update 2D geometric coordinate for a device."""
        dev = self._devices.get(device_id)
        if dev:
            dev.position_2d = (float(x), float(y))
            return True
        return False

    def update_sync_metrics(
        self,
        device_id: str,
        offset_ms: float,
        drift_ppm: float,
        rtt_ms: float,
        status: SyncStatus,
    ) -> bool:
        """Update clock synchronization and network latency metrics."""
        dev = self._devices.get(device_id)
        if dev:
            dev.clock_offset_ms = offset_ms
            dev.clock_drift_ppm = drift_ppm
            dev.network_rtt_ms = rtt_ms
            dev.sync_status = status
            dev.last_seen_timestamp = time.time()
            return True
        return False

    def get_active_array_geometry(self) -> tuple[list[str], np.ndarray]:
        """Return IDs and coordinates (N, 2) of all active devices."""
        active = [d for d in self._devices.values() if d.is_active]
        ids = [d.device_id for d in active]
        coords = np.array([d.position_2d for d in active], dtype=np.float64)
        return ids, coords
