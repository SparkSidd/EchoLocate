"""
EchoLocate — Experiment 4.5 Differentiable SRP Spatial Layer
===========================================================
Implements analytical, fully differentiable spatial response power aggregation
and soft-argmax coordinate regression in PyTorch.

Enables end-to-end backpropagation from spatial localization loss directly
into the neural acoustic frontend.
"""

from __future__ import annotations

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DifferentiableSRPLayer(nn.Module):
    """
    Differentiable 2D Steered Response Power (SRP) Spatial Aggregation Layer.

    Computes:
        S(x) = sum_{(i,j)} w_{ij} * L_{ij}(tau_{ij}(x))
        P(x) = softmax(S(x) / alpha)
        hat{x} = sum_x P(x) * x
    """

    def __init__(
        self,
        room_dimensions: list[float] = [10.0, 6.0, 3.0],
        mic_positions_2d: list[list[float]] | np.ndarray = ((2.0, 1.5), (8.0, 1.5), (5.0, 4.5)),
        speed_of_sound: float = 343.0,
        grid_resolution_m: float = 0.05,
        search_margin_m: float = 0.10,
        max_tau_seconds: float = 0.030,
        num_tau_bins: int = 129,
        temperature_alpha: float = 0.10,
    ) -> None:
        super().__init__()
        self.room_dimensions = room_dimensions
        self.speed_of_sound = float(speed_of_sound)
        self.grid_resolution_m = float(grid_resolution_m)
        self.search_margin_m = float(search_margin_m)
        self.max_tau_seconds = float(max_tau_seconds)
        self.num_tau_bins = int(num_tau_bins)
        self.temperature_alpha = float(temperature_alpha)

        mics = np.array(mic_positions_2d, dtype=np.float32)
        self.n_mics = len(mics)
        self.pairs: list[tuple[int, int]] = []
        for i in range(self.n_mics):
            for j in range(i + 1, self.n_mics):
                self.pairs.append((i, j))
        self.n_pairs = len(self.pairs)

        # 1. Build discrete 2D spatial grid
        x_min, x_max = search_margin_m, room_dimensions[0] - search_margin_m
        y_min, y_max = search_margin_m, room_dimensions[1] - search_margin_m
        xs = np.arange(x_min, x_max + 1e-4, grid_resolution_m, dtype=np.float32)
        ys = np.arange(y_min, y_max + 1e-4, grid_resolution_m, dtype=np.float32)

        X, Y = np.meshgrid(xs, ys, indexing="xy")
        grid_pts = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (N_grid, 2)
        self.n_grid = len(grid_pts)

        # 2. Precompute analytical delays tau_{ij}(x) for all grid points
        delays = np.zeros((self.n_pairs, self.n_grid), dtype=np.float32)
        for p_idx, (i, j) in enumerate(self.pairs):
            d_i = np.linalg.norm(grid_pts - mics[i], axis=-1)
            d_j = np.linalg.norm(grid_pts - mics[j], axis=-1)
            tau_ij = (d_i - d_j) / speed_of_sound
            delays[p_idx, :] = tau_ij

        # 3. Map continuous delays to normalized coordinates in [-1.0, +1.0] for 1D grid_sample
        # tau in [-max_tau, +max_tau] -> normalized in [-1, +1]
        norm_coords = delays / max_tau_seconds
        # Shape for grid_sample: (1, N_pairs, N_grid, 1) or (1, 1, N_grid, 1)
        self.register_buffer("grid_points_2d", torch.from_numpy(grid_pts))  # (N_grid, 2)
        self.register_buffer("theoretical_delays", torch.from_numpy(delays))  # (N_pairs, N_grid)
        self.register_buffer("norm_sampling_coords", torch.from_numpy(norm_coords))  # (N_pairs, N_grid)

        logger.info(
            "Initialized DifferentiableSRPLayer with %d grid points (res=%.2fm, %d pairs).",
            self.n_grid, grid_resolution_m, self.n_pairs
        )

    def sample_delay_likelihoods(
        self,
        pairwise_likelihoods: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable 1D linear delay sampling.
        pairwise_likelihoods: (B, N_pairs, num_tau_bins)
        Returns:
            sampled_values: (B, N_pairs, N_grid)
        """
        B = pairwise_likelihoods.shape[0]
        # Reshape to (B * N_pairs, 1, 1, num_tau_bins) for F.grid_sample
        feat_4d = pairwise_likelihoods.view(B * self.n_pairs, 1, 1, self.num_tau_bins)

        # Expand sampling coords: (B * N_pairs, 1, N_grid, 2) with (x=norm_tau, y=0)
        coords = self.norm_sampling_coords.unsqueeze(0).expand(B, -1, -1)  # (B, N_pairs, N_grid)
        coords_flat = coords.reshape(B * self.n_pairs, 1, self.n_grid)  # (B*N_pairs, 1, N_grid)
        zeros_y = torch.zeros_like(coords_flat)
        grid_2d = torch.stack([coords_flat, zeros_y], dim=-1)  # (B*N_pairs, 1, N_grid, 2)

        # 1D Linear Interpolation via grid_sample (align_corners=True)
        sampled = F.grid_sample(feat_4d, grid_2d, mode="bilinear", padding_mode="border", align_corners=True)
        # sampled shape: (B*N_pairs, 1, 1, N_grid) -> (B, N_pairs, N_grid)
        sampled_3d = sampled.view(B, self.n_pairs, self.n_grid)
        return sampled_3d

    def forward(
        self,
        pairwise_likelihoods: torch.Tensor,
        pair_weights: torch.Tensor | None = None,
        alpha: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward spatial pass.
        Args:
            pairwise_likelihoods: (B, N_pairs, num_tau_bins)
            pair_weights: (B, N_pairs) or None
            alpha: Softmax temperature parameter
        Returns:
            spatial_map: (B, N_grid) - raw unnormalized SRP response
            spatial_prob: (B, N_grid) - normalized spatial probability
            pred_xy: (B, 2) - continuous soft-argmax estimated position
        """
        B = pairwise_likelihoods.shape[0]
        temp_alpha = self.temperature_alpha if alpha is None else alpha

        # 1. Sample continuous likelihood curves at theoretical grid delays
        sampled_vals = self.sample_delay_likelihoods(pairwise_likelihoods)  # (B, N_pairs, N_grid)

        # 2. Weighted spatial accumulation
        if pair_weights is not None:
            w = pair_weights.unsqueeze(-1)  # (B, N_pairs, 1)
            spatial_map = torch.sum(w * sampled_vals, dim=1)  # (B, N_grid)
        else:
            spatial_map = torch.mean(sampled_vals, dim=1)  # (B, N_grid)

        # 3. Differentiable spatial probability distribution
        spatial_prob = F.softmax(spatial_map / max(1e-4, temp_alpha), dim=-1)  # (B, N_grid)

        # 4. Soft-argmax 2D continuous coordinate regression
        # pred_xy = sum_x P(x) * x
        grid_xy = self.grid_points_2d.unsqueeze(0).expand(B, -1, -1)  # (B, N_grid, 2)
        prob_expanded = spatial_prob.unsqueeze(-1)  # (B, N_grid, 1)
        pred_xy = torch.sum(prob_expanded * grid_xy, dim=1)  # (B, 2)

        return spatial_map, spatial_prob, pred_xy
