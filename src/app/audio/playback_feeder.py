"""
EchoLocate — Recorded Multichannel Playback Feeder (Mode B1)
============================================================
Feeds sequential multichannel audio frames into the processing worker
at configurable speeds with pause, seek, resume, and ground-truth alignment.
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


class RecordedPlaybackFeeder:
    """Manages sequential playback of multichannel recordings with interactive controls."""

    def __init__(
        self,
        worker: RealTimePipelineWorker,
        multichannel_audio: np.ndarray,
        sample_rate: int = 16000,
        frame_duration_s: float = 0.20,
        hop_duration_s: float = 0.10,
        dataset_name: str = "LOCATA",
        sequence_id: str = "Recording_01",
        ground_truth_trajectory: list[tuple[float, float]] | None = None,
    ) -> None:
        self.worker = worker
        self.audio = multichannel_audio
        self.sample_rate = sample_rate
        self.frame_samples = int(frame_duration_s * sample_rate)
        self.hop_samples = int(hop_duration_s * sample_rate)
        self.provenance = ProvenanceMetadata.real_recorded(dataset_name, sequence_id)
        self.ground_truth = ground_truth_trajectory

        self.current_sample = 0
        self.is_playing = False
        self.playback_speed = 1.0  # 1.0x real-time
        self._thread: threading.Thread | None = None
        self.frame_counter = 0

    def start(self) -> None:
        """Start sequential playback."""
        if self.is_playing:
            return
        self.is_playing = True
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()
        logger.info("RecordedPlaybackFeeder started playback (%s).", self.provenance.source_name)

    def pause(self) -> None:
        """Pause playback."""
        self.is_playing = False
        logger.info("RecordedPlaybackFeeder paused.")

    def resume(self) -> None:
        """Resume playback from current playhead."""
        self.start()

    def seek_frame(self, frame_idx: int) -> None:
        """Seek playhead to specific frame index."""
        self.current_sample = max(0, min(self.audio.shape[1] - self.frame_samples, frame_idx * self.hop_samples))
        self.frame_counter = frame_idx

    def restart(self) -> None:
        """Restart playback from beginning."""
        self.current_sample = 0
        self.frame_counter = 0
        self.start()

    def set_speed(self, speed_multiplier: float) -> None:
        """Set playback rate (e.g. 0.5x, 1.0x, 2.0x)."""
        self.playback_speed = max(0.1, min(5.0, speed_multiplier))

    def _playback_loop(self) -> None:
        total_samples = self.audio.shape[1]
        base_hop_time_s = self.hop_samples / self.sample_rate

        while self.is_playing and self.current_sample + self.frame_samples <= total_samples:
            t0 = time.perf_counter()
            frame_data = self.audio[:, self.current_sample : self.current_sample + self.frame_samples]

            gt_xy = self.ground_truth[self.frame_counter] if self.ground_truth and self.frame_counter < len(self.ground_truth) else None

            packet = PipelineFramePacket(
                frame_index=self.frame_counter,
                audio_data=frame_data,
                sample_rate=self.sample_rate,
                timestamp_ingested=time.time(),
                provenance_meta=self.provenance,
                ground_truth_xy=gt_xy,
            )
            self.worker.submit_frame(packet)

            self.current_sample += self.hop_samples
            self.frame_counter += 1

            elapsed = time.perf_counter() - t0
            target_delay = (base_hop_time_s / self.playback_speed) - elapsed
            if target_delay > 0:
                time.sleep(target_delay)

        self.is_playing = False
        logger.info("RecordedPlaybackFeeder reached end of stream.")
