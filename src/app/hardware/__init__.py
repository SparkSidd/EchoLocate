"""
EchoLocate — Hardware & Sensor Abstraction Layer
"""

from src.app.hardware.device_detector import AudioDeviceInfo, detect_audio_input_devices
from src.app.hardware.geometry_validator import GeometryValidationResult, validate_microphone_geometry
from src.app.hardware.sync_diagnostics import ChannelSyncMonitor, SyncDiagnosticsReport

__all__ = [
    "AudioDeviceInfo",
    "detect_audio_input_devices",
    "GeometryValidationResult",
    "validate_microphone_geometry",
    "ChannelSyncMonitor",
    "SyncDiagnosticsReport",
]
