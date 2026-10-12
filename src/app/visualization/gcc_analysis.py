"""
EchoLocate — Pairwise GCC-PHAT Correlation Curve Visualizer
===========================================================
Extracts GCC-PHAT cross-correlation curves across all microphone pairs,
identifies primary and secondary peaks, sidelobe energy, and PSR metrics.
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
class GCCPairAnalysis:
    """GCC-PHAT cross-correlation analysis for a single microphone pair."""
    pair_id: tuple[int, int]
    pair_label: str
    tau_axis_ms: list[float]
    correlation_curve: list[float]
    primary_peak_tau_ms: float
    primary_peak_power: float
    secondary_peak_tau_ms: float
    secondary_peak_power: float
    peak_to_sidelobe_ratio: float
    sidelobe_energy: float


@dataclass
class GCCAnalysisPayload:
    """Full pairwise GCC-PHAT visual analysis packet."""
    frame_id: int
    pairs: list[GCCPairAnalysis]
    provenance: VisualProvenanceMetadata


def analyze_all_gcc_pairs(
    frame_audio: np.ndarray,
    fs: int = 16000,
    max_tau_ms: float = 30.0,
    frame_id: int = 0,
    provenance: VisualProvenanceMetadata | None = None,
) -> GCCAnalysisPayload:
    """Compute GCC-PHAT cross-correlation for all microphone pairs."""
    n_mics = frame_audio.shape[0]
    max_tau_samples = int(max_tau_ms * 1e-3 * fs)
    prov = provenance or VisualProvenanceMetadata.live(frame_id)

    pairs_data = []
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            est = gcc_phat(frame_audio[i], frame_audio[j], fs=fs, max_tau_samples=max_tau_samples)
            tau_axis = est.tau_axis_seconds * 1000.0
            curve = est.gcc_phat_response

            # Peak extraction
            sorted_indices = np.argsort(curve)[::-1]
            p1_idx = sorted_indices[0]
            p1_tau = float(tau_axis[p1_idx])
            p1_pow = float(curve[p1_idx])

            # Find secondary peak separated by at least 2 samples
            p2_pow = 0.0
            p2_tau = 0.0
            for idx in sorted_indices[1:]:
                if abs(idx - p1_idx) >= 2:
                    p2_pow = float(curve[idx])
                    p2_tau = float(tau_axis[idx])
                    break

            psr = float(p1_pow / max(1e-6, p2_pow))
            sidelobe_energy = float(np.sum(curve**2) - p1_pow**2)

            pairs_data.append(
                GCCPairAnalysis(
                    pair_id=(i, j),
                    pair_label=f"Pair ({i+1}, {j+1})",
                    tau_axis_ms=tau_axis.tolist(),
                    correlation_curve=curve.tolist(),
                    primary_peak_tau_ms=p1_tau,
                    primary_peak_power=p1_pow,
                    secondary_peak_tau_ms=p2_tau,
                    secondary_peak_power=p2_pow,
                    peak_to_sidelobe_ratio=psr,
                    sidelobe_energy=sidelobe_energy,
                )
            )

    return GCCAnalysisPayload(
        frame_id=frame_id,
        pairs=pairs_data,
        provenance=prov,
    )
