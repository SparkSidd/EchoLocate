"""
EchoLocate — Audio Stream Manager & Circular Frame Buffer
=========================================================
Manages thread-safe audio input streaming from sounddevice hardware callbacks
into a bounded circular FIFO buffer.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from src.app.audit.data_provenance import ProvenanceMetadata
from src.app.engine.pipeline_worker import PipelineFramePacket, RealTimePipelineWorker

logger = logging.getLogger(__name__)


class CircularAudioBuffer:
    """Thread-safe circular audio buffer for rolling frame extraction."""

    def __init__(self, num_channels: int = 4, buffer_samples: int = 48000) -> None:
        self.num_channels = num_channels
        self.buffer_samples = buffer_samples
        self.buffer = np.zeros((num_channels, buffer_samples), dtype=np.float32)
        self.write_idx = 0
        self.lock = threading.Lock()
        self.total_samples_written = 0

    def write(self, samples: np.ndarray) -> None:
        """Write incoming multichannel chunk (N_channels, N_samples) into circular buffer."""
        n_ch, n_s = samples.shape
        with self.lock:
            if n_s >= self.buffer_samples:
                self.buffer[:] = samples[:, -self.buffer_samples:]
                self.write_idx = 0
            else:
                end_idx = self.write_idx + n_s
                if end_idx <= self.buffer_samples:
                    self.buffer[:, self.write_idx : end_idx] = samples
                else:
                    first_part = self.buffer_samples - self.write_idx
                    self.buffer[:, self.write_idx :] = samples[:, :first_part]
                    self.buffer[:, : end_idx - self.buffer_samples] = samples[:, first_part:]
                self.write_idx = end_idx % self.buffer_samples
            self.total_samples_written += n_s

    def read_latest(self, num_samples: int) -> np.ndarray:
        """Extract the most recent `num_samples` from buffer."""
        with self.lock:
            if self.write_idx >= num_samples:
                return self.buffer[:, self.write_idx - num_samples : self.write_idx].copy()
            else:
                first_part = num_samples - self.write_idx
                tail = self.buffer[:, self.buffer_samples - first_part : self.buffer_samples]
                head = self.buffer[:, : self.write_idx]
                return np.hstack([tail, head])


class LiveAudioStreamManager:
    """Manages continuous hardware audio stream acquisition."""

    def __init__(
        self,
        worker: RealTimePipelineWorker,
        device_id: int = 0,
        sample_rate: int = 16000,
        num_channels: int = 4,
        frame_duration_s: float = 0.20,
        hop_duration_s: float = 0.10,
    ) -> None:
        self.worker = worker
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.frame_samples = int(frame_duration_s * sample_rate)
        self.hop_samples = int(hop_duration_s * sample_rate)

        self.buffer = CircularAudioBuffer(num_channels=num_channels, buffer_samples=sample_rate * 3)
        self.is_streaming = False
        self._timer_thread: threading.Thread | None = None
        self.frame_counter = 0

    def start_streaming(self, device_name: str = "Hardware Array") -> None:
        """Start rolling frame ingestion timer."""
        if self.is_streaming:
            return
        self.is_streaming = True
        self.frame_counter = 0
        self._timer_thread = threading.Thread(target=self._stream_feeder_loop, args=(device_name,), daemon=True)
        self._timer_thread.start()
        logger.info("LiveAudioStreamManager started streaming from device: %s", device_name)

    def stop_streaming(self) -> None:
        """Stop streaming loop."""
        self.is_streaming = False
        if self._timer_thread and self._timer_thread.is_alive():
            self._timer_thread.join(timeout=1.0)
        logger.info("LiveAudioStreamManager stopped.")

    def push_audio_chunk(self, chunk: np.ndarray) -> None:
        """Push external audio chunk into circular buffer."""
        self.buffer.write(chunk)

    def _stream_feeder_loop(self, device_name: str) -> None:
        hop_time_s = self.hop_samples / self.sample_rate
        provenance = ProvenanceMetadata.live_hardware(device_name)

        while self.is_streaming:
            t0 = time.perf_counter()
            frame_data = self.buffer.read_latest(self.frame_samples)

            packet = PipelineFramePacket(
                frame_index=self.frame_counter,
                audio_data=frame_data,
                sample_rate=self.sample_rate,
                timestamp_ingested=time.time(),
                provenance_meta=provenance,
            )
            self.worker.submit_frame(packet)
            self.frame_counter += 1

            elapsed = time.perf_counter() - t0
            sleep_time = max(0.001, hop_time_s - elapsed)
            time.sleep(sleep_time)
