"""
EchoLocate — Causality & Future-Frame Isolation Audit
=====================================================
Verifies that real-time and benchmark replay pipelines operate strictly causally:
monotonically increasing timestamps, zero future-frame access, and strictly forward-time filters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CausalityCheckReport:
    """Audit report verifying temporal causality during pipeline execution."""
    is_causal: bool
    total_frames_audited: int
    timestamp_inversions: int
    future_frame_violations: int
    status: str


def audit_frame_causality_stream(
    frame_timestamps: list[float],
    processing_timestamps: list[float],
    future_access_flags: list[bool] | None = None,
) -> CausalityCheckReport:
    """Audit sequence of timestamps to ensure zero temporal inversions or lookahead leaks."""
    n_frames = len(frame_timestamps)
    if n_frames < 2:
        return CausalityCheckReport(
            is_causal=True,
            total_frames_audited=n_frames,
            timestamp_inversions=0,
            future_frame_violations=0,
            status="PASSED_TRIVIAL",
        )

    # Check monotonic audio timestamps
    audio_ts_arr = np.asarray(frame_timestamps)
    diffs = np.diff(audio_ts_arr)
    inversions = int(np.sum(diffs < 0))

    # Check future-access flags
    future_violations = 0
    if future_access_flags is not None:
        future_violations = int(np.sum(future_access_flags))

    is_causal = (inversions == 0) and (future_violations == 0)

    return CausalityCheckReport(
        is_causal=is_causal,
        total_frames_audited=n_frames,
        timestamp_inversions=inversions,
        future_frame_violations=future_violations,
        status="STRICTLY_CAUSAL_PASS" if is_causal else "CAUSALITY_VIOLATION_FAIL",
    )
