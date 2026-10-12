"""
EchoLocate — Strictly Causal Benchmark Replay Engine (Mode B2)
=============================================================
Executes sequential frame-by-frame replay with strict causality enforcement:
zero future-frame access, deterministic filter resets, and latency measurement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.app.audit.data_provenance import DataProvenance, ProvenanceMetadata
from src.app.engine.method_registry import EchoLocateMethodRegistry
from src.app.engine.temporal_tracker import CausalTemporalTracker

logger = logging.getLogger(__name__)


@dataclass
class CausalReplayFrameRecord:
    """Detailed log record for an individual causally processed replay frame."""
    frame_index: int
    audio_timestamp_s: float
    ingestion_timestamp_s: float
    processing_latency_ms: float
    estimated_x: float
    estimated_y: float
    smoothed_x: float
    smoothed_y: float
    ground_truth_x: float | None
    ground_truth_y: float | None
    error_m: float | None
    confidence: float
    method_id: str
    provenance: str
    causal_mode: bool = True


def execute_causal_benchmark_replay(
    multichannel_audio: np.ndarray,
    method_registry: EchoLocateMethodRegistry,
    method_id: str = "M2_Raw_SRP",
    fs: int = 16000,
    frame_duration_s: float = 0.20,
    hop_duration_s: float = 0.10,
    ground_truth_trajectory: list[tuple[float, float]] | None = None,
    provenance_meta: ProvenanceMetadata | None = None,
) -> list[CausalReplayFrameRecord]:
    """Execute strictly causal sequential playback across all frames in an audio recording."""
    frame_samples = int(frame_duration_s * fs)
    hop_samples = int(hop_duration_s * fs)
    total_samples = multichannel_audio.shape[1]

    tracker = CausalTemporalTracker(mode="CAUSAL_SMOOTHED")
    records: list[CausalReplayFrameRecord] = []
    prov_str = provenance_meta.provenance.value if provenance_meta else DataProvenance.REAL_RECORDED_DATASET.value

    frame_idx = 0
    start_sample = 0

    while start_sample + frame_samples <= total_samples:
        frame_chunk = multichannel_audio[:, start_sample : start_sample + frame_samples]
        audio_t = float(start_sample / fs)
        t_ingest = time.time()

        t0 = time.perf_counter()
        loc_out = method_registry.execute_method(method_id, frame_chunk)
        tracked = tracker.update(loc_out.pred_x, loc_out.pred_y, confidence=loc_out.confidence)
        proc_lat_ms = (time.perf_counter() - t0) * 1000.0

        gt_xy = ground_truth_trajectory[frame_idx] if ground_truth_trajectory and frame_idx < len(ground_truth_trajectory) else None
        err_m = float(np.hypot(tracked.smoothed_x - gt_xy[0], tracked.smoothed_y - gt_xy[1])) if gt_xy else None

        records.append(
            CausalReplayFrameRecord(
                frame_index=frame_idx,
                audio_timestamp_s=audio_t,
                ingestion_timestamp_s=t_ingest,
                processing_latency_ms=proc_lat_ms,
                estimated_x=tracked.raw_x,
                estimated_y=tracked.raw_y,
                smoothed_x=tracked.smoothed_x,
                smoothed_y=tracked.smoothed_y,
                ground_truth_x=gt_xy[0] if gt_xy else None,
                ground_truth_y=gt_xy[1] if gt_xy else None,
                error_m=err_m,
                confidence=loc_out.confidence,
                method_id=method_id,
                provenance=prov_str,
                causal_mode=True,
            )
        )

        start_sample += hop_samples
        frame_idx += 1

    return records
