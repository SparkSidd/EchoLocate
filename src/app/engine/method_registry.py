"""
EchoLocate — Unified Method Registry & Localization Interface
=============================================================
Provides a standard, clean execution interface across M1 (Classical NLS),
M2 (Raw SRP-PHAT), M6 (Calibrated DeepGCC), M11e (Differentiable SRP), and
M12 (Confidence-Gated Hybrid).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.localization.blind_localizer import BlindSRPLocalizer
from src.localization.confidence_gated_hybrid import ConfidenceGatedHybridLocalizer
from src.localization.gcc_phat import estimate_all_tdoas
from src.localization.multilateration import localize_source_2d
from src.localization.neural_srp import NeuralSRPLocalizer
from src.models.deepgcc_calibrated import CalibratedDeepGCC
from src.models.inference import BlindNeuralGCCPredictor
from src.models.neural_gcc import DeepGCCRes1D

logger = logging.getLogger(__name__)


@dataclass
class MethodLocalizationOutput:
    """Standardized output packet returned by all registered localization methods."""
    method_id: str
    pred_x: float
    pred_y: float
    confidence: float
    latency_ms: float
    spatial_contrast: float
    status: str
    intermediate_info: dict[str, Any]
    heatmap_2d: np.ndarray | None = None


class EchoLocateMethodRegistry:
    """Manages instantiation and unified execution of all EchoLocate localization algorithms."""

    def __init__(
        self,
        room_dimensions_m: list[float] = [10.0, 6.0, 3.0],
        mic_positions_2d: np.ndarray | None = None,
        speed_of_sound: float = 343.0,
        fs: int = 16000,
    ) -> None:
        self.room_dims = room_dimensions_m
        self.mics = mic_positions_2d if mic_positions_2d is not None else np.array([[2.0, 1.5], [8.0, 1.5], [5.0, 4.5], [5.0, 1.5]])
        self.speed_of_sound = speed_of_sound
        self.fs = fs

        self._init_models()
        self._init_localizers()

    def _init_models(self) -> None:
        """Load trained neural models read-only."""
        self.model_calib = CalibratedDeepGCC(num_res_blocks=4, enable_learned_temperature=False)
        p_calib = Path("results/experiment_04_5/calibrated_deep_gcc.pt")
        if p_calib.exists():
            ckpt = torch.load(str(p_calib), map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.model_calib.load_state_dict(state)
        self.model_calib.eval()

        self.model_spatial = CalibratedDeepGCC(num_res_blocks=4, enable_learned_temperature=False)
        p_spatial = Path("results/experiment_04_5/hybrid_spatial_srp.pt")
        if p_spatial.exists():
            ckpt = torch.load(str(p_spatial), map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.model_spatial.load_state_dict(state)
        self.model_spatial.eval()

        self.pred_calib = BlindNeuralGCCPredictor(self.model_calib, max_tau_seconds=0.030)
        self.pred_spatial = BlindNeuralGCCPredictor(self.model_spatial, max_tau_seconds=0.030)

    def _init_localizers(self) -> None:
        """Initialize localizer instances for active room/array geometry."""
        self.localizer_m2 = BlindSRPLocalizer(self.room_dims, self.mics, assumed_speed_of_sound=self.speed_of_sound)
        self.localizer_m6 = NeuralSRPLocalizer(self.room_dims, self.mics, self.pred_calib, assumed_speed_of_sound=self.speed_of_sound)
        self.localizer_m11 = NeuralSRPLocalizer(self.room_dims, self.mics, self.pred_spatial, assumed_speed_of_sound=self.speed_of_sound)
        self.localizer_m12 = ConfidenceGatedHybridLocalizer(self.localizer_m2, self.localizer_m6, gate_policy="threshold")

    def update_geometry(self, room_dimensions_m: list[float], mic_positions_2d: np.ndarray) -> None:
        """Update active room dimensions and microphone positions."""
        self.room_dims = room_dimensions_m
        self.mics = np.asarray(mic_positions_2d, dtype=np.float64)
        self._init_localizers()

    def execute_method(self, method_id: str, frame_audio: np.ndarray) -> MethodLocalizationOutput:
        """Execute specified method on a single multichannel frame."""
        t0 = time.perf_counter()

        if method_id == "M1_Classical_NLS":
            tdoas = estimate_all_tdoas(frame_audio, fs=self.fs, max_tau_samples=int(0.030 * self.fs))
            tdoa_dict = {k: v.tau_seconds for k, v in tdoas.items()}
            nls_res = localize_source_2d(self.mics, tdoa_dict, speed_of_sound=self.speed_of_sound, room_dimensions=self.room_dims)
            lat = (time.perf_counter() - t0) * 1000.0
            return MethodLocalizationOutput(
                method_id=method_id,
                pred_x=float(nls_res.predicted_position[0]),
                pred_y=float(nls_res.predicted_position[1]),
                confidence=0.70 if nls_res.converged else 0.30,
                latency_ms=lat,
                spatial_contrast=1.0,
                status="CONVERGED" if nls_res.converged else "NLS_DIVERGED",
                intermediate_info={"converged": nls_res.converged, "solver_message": nls_res.solver_message},
            )

        elif method_id == "M2_Raw_SRP":
            res = self.localizer_m2.localize(frame_audio, fs=self.fs)
            lat = (time.perf_counter() - t0) * 1000.0
            conf = min(0.98, max(0.10, res.peak_ratio / 3.0))
            return MethodLocalizationOutput(
                method_id=method_id,
                pred_x=float(res.pred_x),
                pred_y=float(res.pred_y),
                confidence=conf,
                latency_ms=lat,
                spatial_contrast=res.peak_ratio,
                status="HIGH_CONFIDENCE" if conf > 0.7 else "MEDIUM_CONFIDENCE",
                intermediate_info={"peak_ratio": res.peak_ratio, "top_k_peaks": len(res.top_k_peaks)},
            )

        elif method_id == "M6_Calibrated_DeepGCC":
            res = self.localizer_m6.localize(frame_audio, fs=self.fs)
            lat = (time.perf_counter() - t0) * 1000.0
            conf = min(0.98, max(0.10, 1.0 / (1.0 + np.exp(-res.spatial_peak_contrast + 1.5))))
            return MethodLocalizationOutput(
                method_id=method_id,
                pred_x=float(res.pred_x),
                pred_y=float(res.pred_y),
                confidence=conf,
                latency_ms=lat,
                spatial_contrast=res.spatial_peak_contrast,
                status="NEURAL_LIKELIHOOD_COMPUTED",
                intermediate_info={"spatial_entropy": res.spatial_entropy},
                heatmap_2d=res.srp_2d,
            )

        elif method_id == "M11e_Hybrid_Spatial_SRP":
            res = self.localizer_m11.localize(frame_audio, fs=self.fs)
            lat = (time.perf_counter() - t0) * 1000.0
            conf = min(0.98, max(0.10, 1.0 / (1.0 + np.exp(-res.spatial_peak_contrast + 1.5))))
            return MethodLocalizationOutput(
                method_id=method_id,
                pred_x=float(res.pred_x),
                pred_y=float(res.pred_y),
                confidence=conf,
                latency_ms=lat,
                spatial_contrast=res.spatial_peak_contrast,
                status="DIFF_SPATIAL_SRP_COMPUTED",
                intermediate_info={"spatial_entropy": res.spatial_entropy},
                heatmap_2d=res.srp_2d,
            )

        elif method_id == "M12_Confidence_Hybrid":
            res_hyb = self.localizer_m12.localize(frame_audio, fs=self.fs)
            lat = (time.perf_counter() - t0) * 1000.0
            return MethodLocalizationOutput(
                method_id=method_id,
                pred_x=float(res_hyb.pred_x),
                pred_y=float(res_hyb.pred_y),
                confidence=res_hyb.gate_confidence,
                latency_ms=lat,
                spatial_contrast=res_hyb.features.srp_spatial_contrast,
                status=f"HYBRID_SELECTED_{res_hyb.selected_method}",
                intermediate_info={
                    "selected_method": res_hyb.selected_method,
                    "srp_entropy": res_hyb.features.srp_spatial_entropy,
                    "disagreement_m": res_hyb.features.cross_method_disagreement_m,
                },
            )

        # Default fallback
        res = self.localizer_m2.localize(frame_audio, fs=self.fs)
        lat = (time.perf_counter() - t0) * 1000.0
        return MethodLocalizationOutput(
            method_id="M2_Raw_SRP",
            pred_x=float(res.pred_x),
            pred_y=float(res.pred_y),
            confidence=0.85,
            latency_ms=lat,
            spatial_contrast=res.peak_ratio,
            status="DEFAULT_SRP",
            intermediate_info={},
        )
