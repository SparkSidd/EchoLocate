"""
EchoLocate — Step-by-Step SRP Constructive Interference Explainer
================================================================
Educational interactive stepper demonstrating how Steered Response Power (SRP)
aggregates pairwise cross-correlations into a coherent spatial likelihood peak.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.app.visualization.provenance import VisualProvenanceMetadata
from src.localization.gcc_phat import gcc_phat

logger = logging.getLogger(__name__)


@dataclass
class SRPExplainerPairSample:
    """Sampled cross-correlation value for a single microphone pair at candidate delay."""
    pair_label: str
    expected_delay_ms: float
    sampled_correlation_power: float
    tau_axis_ms: list[float]
    correlation_curve: list[float]


@dataclass
class SRPExplainerStepResult:
    """Step result for the SRP constructive interference walkthrough."""
    step_number: int
    step_title: str
    explanation: str
    candidate_xy: tuple[float, float]
    accumulated_power: float
    pair_samples: list[SRPExplainerPairSample]
    is_constructive_peak: bool
    provenance: VisualProvenanceMetadata


def compute_srp_explainer_step(
    frame_audio: np.ndarray,
    candidate_xy: tuple[float, float],
    mic_positions_2d: np.ndarray,
    room_dimensions_m: list[float],
    speed_of_sound: float = 343.0,
    fs: int = 16000,
    step_number: int = 4,
    provenance: VisualProvenanceMetadata | None = None,
) -> SRPExplainerStepResult:
    """Execute constructive interference calculation for candidate probe coordinate (x, y)."""
    prov = provenance or VisualProvenanceMetadata.simulation(0, "SRP Educational Explainer")
    cx, cy = candidate_xy
    n_mics = len(mic_positions_2d)

    # 1. Compute distances from candidate to all microphones
    dists = [float(np.hypot(cx - m[0], cy - m[1])) for m in mic_positions_2d]

    pair_samples = []
    total_power = 0.0

    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            # Theoretical propagation delay (seconds)
            tau_sec = (dists[i] - dists[j]) / speed_of_sound
            tau_ms = tau_sec * 1000.0

            # Compute GCC-PHAT
            est = gcc_phat(frame_audio[i], frame_audio[j], fs=fs, max_tau_samples=int(0.030 * fs))
            tau_axis = est.tau_axis_seconds * 1000.0
            curve = est.gcc_phat_response

            # Sample correlation at theoretical delay (nearest neighbour)
            sample_idx = int(np.argmin(np.abs(tau_axis - tau_ms)))
            sample_val = float(curve[sample_idx])
            total_power += sample_val

            pair_samples.append(
                SRPExplainerPairSample(
                    pair_label=f"Pair ({i+1}, {j+1})",
                    expected_delay_ms=tau_ms,
                    sampled_correlation_power=sample_val,
                    tau_axis_ms=tau_axis.tolist(),
                    correlation_curve=curve.tolist(),
                )
            )

    mean_power = total_power / max(1, len(pair_samples))
    is_constructive = mean_power > 0.65

    step_titles = {
        1: "Step 1: Inspect Microphone Array Geometry",
        2: "Step 2: Place Candidate Coordinate Probe",
        3: "Step 3: Calculate Theoretical Time Delays",
        4: "Step 4: Sample Pairwise Correlation Curves",
        5: "Step 5: Accumulate Spatial Power (Constructive Interference)",
        6: "Step 6: Move Probe Across Room Grid",
        7: "Step 7: Render 2D Likelihood Surface",
        8: "Step 8: Identify Global Acoustic Maximum",
    }

    title = step_titles.get(step_number, f"Step {step_number}: Constructive Spatial Steering")
    explanation = (
        f"Probe at ({cx:.2f} m, {cy:.2f} m) predicts delays across {len(pair_samples)} pairs. "
        f"Accumulated spatial power = {total_power:.3f} "
        f"({'Constructive Alignment Peak!' if is_constructive else 'Destructive / Misaligned Phase'})."
    )

    return SRPExplainerStepResult(
        step_number=step_number,
        step_title=title,
        explanation=explanation,
        candidate_xy=candidate_xy,
        accumulated_power=total_power,
        pair_samples=pair_samples,
        is_constructive_peak=is_constructive,
        provenance=prov,
    )
