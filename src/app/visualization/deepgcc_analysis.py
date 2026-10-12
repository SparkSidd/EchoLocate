"""
EchoLocate — Calibrated DeepGCC & Neural Likelihood Visualizer
=============================================================
Computes neural correlation distributions, highlights harmonic pseudo-peak
suppression, and performs head-to-head raw vs. neural correlation analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.app.visualization.gcc_analysis import GCCPairAnalysis
from src.app.visualization.provenance import VisualProvenanceMetadata
from src.localization.gcc_phat import gcc_phat
from src.models.deepgcc_calibrated import CalibratedDeepGCC

logger = logging.getLogger(__name__)


@dataclass
class DeepGCCComparisonPair:
    """Side-by-side comparison between raw GCC and calibrated DeepGCC."""
    pair_label: str
    tau_axis_ms: list[float]
    raw_gcc_curve: list[float]
    neural_gcc_curve: list[float]
    raw_peak_tau_ms: float
    neural_peak_tau_ms: float
    pseudo_peaks_suppressed: int
    harmonic_suppression_ratio: float
    temperature_sharpening_factor: float


@dataclass
class DeepGCCAnalysisPayload:
    """Full DeepGCC neural comparison packet."""
    frame_id: int
    pairs: list[DeepGCCComparisonPair]
    provenance: VisualProvenanceMetadata


def analyze_deepgcc_comparisons(
    frame_audio: np.ndarray,
    model: CalibratedDeepGCC,
    fs: int = 16000,
    max_tau_ms: float = 30.0,
    frame_id: int = 0,
    provenance: VisualProvenanceMetadata | None = None,
) -> DeepGCCAnalysisPayload:
    """Compute neural enhanced GCC distributions and quantify pseudo-peak suppression."""
    n_mics = frame_audio.shape[0]
    max_tau_samples = int(max_tau_ms * 1e-3 * fs)
    prov = provenance or VisualProvenanceMetadata.live(frame_id)

    pairs_data = []
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            est = gcc_phat(frame_audio[i], frame_audio[j], fs=fs, max_tau_samples=max_tau_samples)
            tau_axis = est.tau_axis_seconds * 1000.0
            raw_curve = est.gcc_phat_response

            # Predict neural likelihood
            inp = torch.from_numpy(raw_curve).float().unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                neural_curve_t = model(inp)[0].squeeze().numpy()

            raw_max_idx = int(np.argmax(raw_curve))
            neural_max_idx = int(np.argmax(neural_curve_t))

            # Quantify suppression of secondary raw sidelobes
            raw_sidelobes = np.sort(raw_curve)[::-1][1:4]
            neural_at_sidelobes = neural_curve_t[np.argsort(raw_curve)[::-1][1:4]]
            suppr_ratio = float(np.mean(raw_sidelobes) / max(1e-6, np.mean(neural_at_sidelobes)))

            pairs_data.append(
                DeepGCCComparisonPair(
                    pair_label=f"Pair ({i+1}, {j+1})",
                    tau_axis_ms=tau_axis.tolist(),
                    raw_gcc_curve=raw_curve.tolist(),
                    neural_gcc_curve=neural_curve_t.tolist(),
                    raw_peak_tau_ms=float(tau_axis[raw_max_idx]),
                    neural_peak_tau_ms=float(tau_axis[neural_max_idx]),
                    pseudo_peaks_suppressed=int(np.sum(neural_at_sidelobes < raw_sidelobes * 0.5)),
                    harmonic_suppression_ratio=suppr_ratio,
                    temperature_sharpening_factor=float(model.temperature) if hasattr(model, "temperature") else 1.0,
                )
            )

    return DeepGCCAnalysisPayload(
        frame_id=frame_id,
        pairs=pairs_data,
        provenance=prov,
    )
