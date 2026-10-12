"""
EchoLocate — Real-Time Live Microphone Capture
===============================================
Acquires continuous audio from local sound device (e.g. laptop built-in mic / USB mic)
with a non-blocking stream callback, ring buffering, timestamping, and graceful
synthetic fallback if physical audio devices are inaccessible.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    HAVE_SOUNDDEVICE = True
except Exception:
    HAVE_SOUNDDEVICE = False


class LiveMicrophoneCaptureStream:
    """Acquires live physical microphone audio into a timestamped circular ring buffer."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: float = 200.0,
        channels: int = 1,
        device_index: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sample_rate * (frame_duration_ms / 1000.0))
        self.channels = channels
        self.device_index = device_index

        self._buffer: collections.deque[float] = collections.deque(maxlen=self.frame_samples * 10)
        self._is_running = False
        self._lock = threading.Lock()
        self._stream: Any = None
        self._total_samples_captured = 0
        self._last_frame_timestamp = 0.0

    def start(self) -> bool:
        """Start physical capture stream or virtual fallback."""
        if self._is_running:
            return True

        self._is_running = True

        if HAVE_SOUNDDEVICE:
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="float32",
                    device=self.device_index,
                    blocksize=int(self.sample_rate * 0.05),  # 50 ms blocks
                    callback=self._audio_callback,
                )
                self._stream.start()
                logger.info("Physical microphone stream started on device %s at %d Hz", self.device_index, self.sample_rate)
                return True
            except Exception as e:
                logger.warning("Could not open physical sounddevice stream (%s). Starting virtual fallback.", e)

        # Virtual background generator thread for testing / emulated mode
        self._thread = threading.Thread(target=self._virtual_capture_loop, daemon=True)
        self._thread.start()
        logger.info("Live audio capture running in virtual fallback mode.")
        return True

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """SoundDevice real-time input callback."""
        if status:
            logger.debug("SoundDevice status: %s", status)
        mono = indata[:, 0] if indata.ndim > 1 else indata
        with self._lock:
            self._buffer.extend(mono.tolist())
            self._total_samples_captured += len(mono)
            self._last_frame_timestamp = time.time()

    def _virtual_capture_loop(self) -> None:
        """Generates synthetic ambient noise and occasional transient taps."""
        dt = 0.05
        block_size = int(self.sample_rate * dt)
        while self._is_running:
            t = time.time()
            noise = np.random.normal(0, 0.015, block_size).astype(np.float32)
            # Add subtle 200 Hz hum
            t_axis = np.linspace(t, t + dt, block_size, endpoint=False)
            hum = 0.01 * np.sin(2 * np.pi * 200.0 * t_axis).astype(np.float32)
            block = noise + hum

            with self._lock:
                self._buffer.extend(block.tolist())
                self._total_samples_captured += len(block)
                self._last_frame_timestamp = t
            time.sleep(dt)

    def read_latest_frame(self) -> tuple[np.ndarray, float]:
        """Read the most recent continuous frame of audio."""
        with self._lock:
            if len(self._buffer) < self.frame_samples:
                pad = np.zeros(self.frame_samples - len(self._buffer), dtype=np.float32)
                cur = np.array(list(self._buffer), dtype=np.float32)
                return np.concatenate([pad, cur]), self._last_frame_timestamp
            else:
                samples = list(self._buffer)[-self.frame_samples:]
                return np.array(samples, dtype=np.float32), self._last_frame_timestamp

    def stop(self) -> None:
        """Stop audio capture."""
        self._is_running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        logger.info("Microphone capture stream stopped.")
