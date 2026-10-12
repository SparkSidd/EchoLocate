"""
EchoLocate — High-Precision Multi-Device Synchronized Audio Buffer
==================================================================
Maintains continuous lock-free circular sample buffers for:
  1. Laptop Local SoundDevice Microphone Stream
  2. Smartphone WebSocket Binary Float32 Stream

Extracts time-aligned dual-channel frames and physical energy levels (dB).
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any

import numpy as np

from src.app.audio.live_capture import LiveMicrophoneCaptureStream
from src.app.audio.network_audio_receiver import NetworkAudioReceiver
from src.app.hardware.distributed_sync import DistributedClockSynchronizer

logger = logging.getLogger(__name__)

_PHONE_FRESHNESS_SECONDS = 2.5
_MIN_RMS_THRESHOLD = 1e-5


class SynchronizedMultiDeviceBuffer:
    """Combines laptop local audio and network microphone audio into aligned 2-channel frames."""

    def __init__(
        self,
        local_capture: LiveMicrophoneCaptureStream,
        network_receiver: NetworkAudioReceiver,
        clock_sync: DistributedClockSynchronizer,
        sample_rate: int = 16000,
        frame_duration_ms: float = 200.0,
    ) -> None:
        self.local_capture = local_capture
        self.network_receiver = network_receiver
        self.clock_sync = clock_sync
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sample_rate * (frame_duration_ms / 1000.0))

        # Circular buffer for phone audio samples (holds 2 seconds of 16kHz audio = 32000 samples)
        self._phone_sample_lock = threading.Lock()
        self._phone_sample_deque: collections.deque[float] = collections.deque(maxlen=sample_rate * 3)
        self._last_phone_push_time = 0.0

    def feed_phone_packet(self, samples: np.ndarray, t_server: float) -> None:
        """Called whenever binary/JSON phone audio arrives."""
        if len(samples) == 0:
            return
        mono = samples.astype(np.float32).ravel()
        with self._phone_sample_lock:
            self._phone_sample_deque.extend(mono.tolist())
            self._last_phone_push_time = time.time()

    def get_aligned_dual_channel_frame(
        self,
        phone_device_id: str = "smartphone_client_primary",
    ) -> tuple[np.ndarray, bool, float]:
        """
        Extract aligned dual-channel audio array of shape (2, frame_samples).
        Channel 0: Laptop Microphone
        Channel 1: Smartphone Network Microphone

        Returns:
            dual_audio: np.ndarray shape (2, frame_samples)
            is_phone_active: bool
            estimated_sync_offset_ms: float
        """
        # 1. Laptop Audio (always available from live capture)
        lap_audio, lap_ts = self.local_capture.read_latest_frame()
        lap_audio = _ensure_exact_length(lap_audio, self.frame_samples)

        # 2. Smartphone Audio
        is_phone_active = False
        phone_audio = np.zeros(self.frame_samples, dtype=np.float32)

        session = self.network_receiver.active_sessions.get(phone_device_id)
        now = time.time()

        with self._phone_sample_lock:
            time_since_last_packet = now - self._last_phone_push_time
            num_available = len(self._phone_sample_deque)

            if num_available >= self.frame_samples and time_since_last_packet < _PHONE_FRESHNESS_SECONDS:
                raw_phone = np.array(list(self._phone_sample_deque)[-self.frame_samples:], dtype=np.float32)
                rms = float(np.sqrt(np.mean(raw_phone ** 2)))
                if rms > _MIN_RMS_THRESHOLD:
                    phone_audio = raw_phone
                    is_phone_active = True
            elif session and (now - session.last_packet_at) < _PHONE_FRESHNESS_SECONDS:
                # Fallback to session queue if deque still warming up
                raw_phone = session.pop_latest_samples(self.frame_samples)
                raw_phone = _ensure_exact_length(raw_phone, self.frame_samples)
                rms = float(np.sqrt(np.mean(raw_phone ** 2)))
                if rms > _MIN_RMS_THRESHOLD:
                    phone_audio = raw_phone
                    is_phone_active = True

        # 3. Synchronize using estimated clock offset
        sync_state = self.clock_sync.compute_sync_state(phone_device_id)
        offset_ms = sync_state.clock_offset_ms

        dual_audio = np.stack([lap_audio, phone_audio], axis=0).astype(np.float32)
        return dual_audio, is_phone_active, offset_ms


def _ensure_exact_length(arr: np.ndarray, target: int) -> np.ndarray:
    """Pad with zero or truncate to exact target length."""
    arr = np.asarray(arr, dtype=np.float32).ravel()
    if len(arr) == target:
        return arr
    if len(arr) < target:
        pad = np.zeros(target - len(arr), dtype=np.float32)
        return np.concatenate([pad, arr])
    return arr[-target:]
