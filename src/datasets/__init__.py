"""
EchoLocate — Real Datasets & Adaptation Layer
"""

from src.datasets.dataset_metadata import AcousticSceneMetadata, MicrophoneArraySpec, RealDatasetManifest
from src.datasets.real_dataset_adapter import AV163DatasetAdapter, EchoLocateRealRecordingAdapter, LOCATADatasetAdapter
from src.datasets.recording_protocol import CalibrationProtocolSpec, verify_channel_synchronization

__all__ = [
    "AcousticSceneMetadata",
    "MicrophoneArraySpec",
    "RealDatasetManifest",
    "LOCATADatasetAdapter",
    "AV163DatasetAdapter",
    "EchoLocateRealRecordingAdapter",
    "CalibrationProtocolSpec",
    "verify_channel_synchronization",
]
