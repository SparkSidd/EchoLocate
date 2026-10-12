"""
EchoLocate — Enhanced GCC Pairwise TDOA & NLS Adapter (Method M3)
=================================================================
Extracts peak delay estimates from DeepGCC enhanced correlation representations
and solves 2D multilateration via Non-Linear Least Squares.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import numpy as np

from src.localization.multilateration import localize_source_2d
from src.models.inference import BlindNeuralGCCPredictor

logger = logging.getLogger(__name__)


@dataclass
class EnhancedNLSResult:
    pred_x: float
    pred_y: float
    estimated_tdoas: dict[tuple[int, int], float]
    tdoa_confidences: dict[tuple[int, int], float]
    solver_converged: bool
    runtime_ms: float


def localize_neural_gcc_nls(
    mic_signals: np.ndarray,
    assumed_mic_positions_2d: np.ndarray,
    predictor: BlindNeuralGCCPredictor,
    fs: int = 16000,
    speed_of_sound: float = 343.0,
    room_dimensions: list[float] = [10.0, 6.0, 3.0],
) -> EnhancedNLSResult:
    """
    Method M3: Audio -> Neural Correlation -> TDOA extraction -> NLS Multilateration.
    """
    import time
    t_start = time.perf_counter()

    enhanced_dict, _, tau_axis = predictor.enhance_correlations(mic_signals, fs=fs)

    est_tdoas = {}
    confidences = {}
    for pair, curve in enhanced_dict.items():
        max_idx = int(np.argmax(curve))
        est_tdoas[pair] = float(tau_axis[max_idx])
        # Peak confidence = primary peak / sum
        confidences[pair] = float(curve[max_idx] / max(1e-12, np.sum(curve)))

    # Solve NLS multilateration
    nls_res = localize_source_2d(
        mic_positions_2d=assumed_mic_positions_2d,
        tdoa_dict=est_tdoas,
        speed_of_sound=speed_of_sound,
        room_dimensions=room_dimensions,
    )

    t_end = time.perf_counter()
    runtime_ms = (t_end - t_start) * 1000.0

    return EnhancedNLSResult(
        pred_x=float(nls_res.predicted_position[0]),
        pred_y=float(nls_res.predicted_position[1]),
        estimated_tdoas=est_tdoas,
        tdoa_confidences=confidences,
        solver_converged=bool(nls_res.converged),
        runtime_ms=runtime_ms,
    )
