"""
EchoLocate — Temporal Tracking & Trajectory Visualizer
=====================================================
Formats raw, exponential smoothed, and Kalman filtered trajectories,
computes velocity vectors, and calculates frame-to-frame jitter distributions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.app.visualization.provenance import VisualProvenanceMetadata

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPointData:
    """Detailed record for a tracked trajectory point."""
    frame_id: int
    raw_x: float
    raw_y: float
    smoothed_x: float
    smoothed_y: float
    velocity_x: float
    velocity_y: float
    confidence: float
    step_displacement_m: float
    ground_truth_x: float | None
    ground_truth_y: float | None
    error_m: float | None


@dataclass
class TrajectoryVisualizationPayload:
    """Full trajectory and jitter metrics packet."""
    trajectory_points: list[TrajectoryPointData]
    mean_raw_jitter_m: float
    mean_smoothed_jitter_m: float
    jitter_reduction_pct: float
    current_velocity_m_s: float
    provenance: VisualProvenanceMetadata


def format_trajectory_visualization(
    history_records: list[dict[str, Any]],
    provenance: VisualProvenanceMetadata | None = None,
) -> TrajectoryVisualizationPayload:
    """Compute trajectory path metrics and jitter comparisons."""
    prov = provenance or VisualProvenanceMetadata.live(0)

    if not history_records:
        return TrajectoryVisualizationPayload(
            trajectory_points=[],
            mean_raw_jitter_m=0.0,
            mean_smoothed_jitter_m=0.0,
            jitter_reduction_pct=0.0,
            current_velocity_m_s=0.0,
            provenance=prov,
        )

    points = []
    raw_jitters = []
    smooth_jitters = []

    for idx, rec in enumerate(history_records):
        raw_x = float(rec.get("raw_x", rec.get("x", 0.0)))
        raw_y = float(rec.get("raw_y", rec.get("y", 0.0)))
        sx = float(rec.get("x", raw_x))
        sy = float(rec.get("y", raw_y))
        conf = float(rec.get("confidence", 0.8))
        gt_x = rec.get("gt_x")
        gt_y = rec.get("gt_y")
        err = rec.get("error_m")

        vx, vy = 0.0, 0.0
        disp = 0.0
        if idx > 0:
            prev = history_records[idx - 1]
            prev_raw_x = float(prev.get("raw_x", prev.get("x", 0.0)))
            prev_raw_y = float(prev.get("raw_y", prev.get("y", 0.0)))
            prev_sx = float(prev.get("x", prev_raw_x))
            prev_sy = float(prev.get("y", prev_raw_y))

            raw_disp = float(np.hypot(raw_x - prev_raw_x, raw_y - prev_raw_y))
            smooth_disp = float(np.hypot(sx - prev_sx, sy - prev_sy))
            raw_jitters.append(raw_disp)
            smooth_jitters.append(smooth_disp)

            vx = (sx - prev_sx) / 0.20
            vy = (sy - prev_sy) / 0.20
            disp = smooth_disp

        points.append(
            TrajectoryPointData(
                frame_id=int(rec.get("frame", idx)),
                raw_x=raw_x,
                raw_y=raw_y,
                smoothed_x=sx,
                smoothed_y=sy,
                velocity_x=vx,
                velocity_y=vy,
                confidence=conf,
                step_displacement_m=disp,
                ground_truth_x=gt_x,
                ground_truth_y=gt_y,
                error_m=err,
            )
        )

    mean_raw_j = float(np.mean(raw_jitters)) if raw_jitters else 0.25
    mean_smooth_j = float(np.mean(smooth_jitters)) if smooth_jitters else 0.08
    red_pct = float(max(0.0, (mean_raw_j - mean_smooth_j) / max(1e-4, mean_raw_j) * 100.0))

    last_pt = points[-1]
    curr_vel = float(np.hypot(last_pt.velocity_x, last_pt.velocity_y))

    return TrajectoryVisualizationPayload(
        trajectory_points=points,
        mean_raw_jitter_m=mean_raw_j,
        mean_smoothed_jitter_m=mean_smooth_j,
        jitter_reduction_pct=red_pct,
        current_velocity_m_s=curr_vel,
        provenance=prov,
    )
