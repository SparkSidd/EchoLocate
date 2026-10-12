"""
EchoLocate — Blind Neural Inference Wrapper
===========================================
Strict black-box interface for DeepGCC neural correlation enhancement.
Has zero access to ground truth coordinates, simulation metadata, or RIR matrices.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.acoustic_frontend import extract_pairwise_gcc_features
from src.models.neural_gcc import DeepGCCRes1D

logger = logging.getLogger(__name__)


class BlindNeuralGCCPredictor:
    """
    Evaluates trained DeepGCC model blindly on raw multi-channel audio arrays.
    """

    def __init__(
        self,
        model_checkpoint_path: str | Path | None = None,
        model: DeepGCCRes1D | None = None,
        max_tau_seconds: float = 0.030,
        num_tau_bins: int = 129,
    ):
        self.max_tau = float(max_tau_seconds)
        self.num_tau_bins = int(num_tau_bins)

        if model is not None:
            self.model = model
        elif isinstance(model_checkpoint_path, nn.Module):
            self.model = model_checkpoint_path
        elif model_checkpoint_path is not None:
            self.model = DeepGCCRes1D()
            ckpt = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            elif isinstance(ckpt, dict):
                self.model.load_state_dict(ckpt)
        else:
            self.model = DeepGCCRes1D()

        self.model.eval()

    def enhance_correlations(
        self,
        mic_signals: np.ndarray,
        fs: int = 16000,
    ) -> tuple[dict[tuple[int, int], np.ndarray], np.ndarray, np.ndarray]:
        """
        Enhance pairwise GCC correlation curves using DeepGCC.

        Returns
        -------
        enhanced_corr_dict : dict[tuple[int, int], np.ndarray]
            Enhanced correlation curve L_ij(tau) for each pair.
        raw_gcc_features : np.ndarray, shape (n_pairs, num_tau_bins)
            Raw GCC-PHAT features before enhancement.
        tau_axis : np.ndarray, shape (num_tau_bins,)
            Uniform delay lag axis in seconds.
        """
        n_mics = mic_signals.shape[0]
        pairs = []
        for i in range(n_mics):
            for j in range(i + 1, n_mics):
                pairs.append((i, j))

        raw_gcc, _, tau_axis = extract_pairwise_gcc_features(
            mic_signals, fs=fs, max_tau_seconds=self.max_tau, num_tau_bins=self.num_tau_bins
        )

        with torch.no_grad():
            x_tensor = torch.from_numpy(raw_gcc).float()
            if x_tensor.dim() == 2:
                x_tensor = x_tensor.unsqueeze(1)
            out = self.model(x_tensor)
            if isinstance(out, tuple):
                pred_likelihood = out[0].cpu().numpy()
            else:
                pred_likelihood = out.cpu().numpy()

        enhanced_dict = {}
        for p_idx, pair in enumerate(pairs):
            # Scale likelihood curve to peak 1.0 for compatibility with standard SRP summation
            curve = pred_likelihood[p_idx]
            max_c = np.max(curve)
            if max_c > 1e-12:
                curve = curve / max_c
            enhanced_dict[pair] = curve

        return enhanced_dict, raw_gcc, tau_axis
