"""
EchoLocate — DeepGCC 1D Dilated Residual Neural Architecture
============================================================
Implements a 1D Dilated Residual Convolutional Network that maps noisy,
reverberant pairwise cross-correlation curves into idealized unimodal delay likelihoods.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ResidualBlock1D(nn.Module):
    """1D Dilated Residual Block with GroupNorm and GELU activations."""

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + residual)


class DeepGCCRes1D(nn.Module):
    """
    DeepGCC Neural Delay Likelihood Estimator.
    Input: (B, 1, num_tau_bins) - Raw normalized cross-correlation
    Output: (B, num_tau_bins)   - Enhanced normalized delay likelihood L_ij(tau)
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        num_res_blocks: int = 3,
        kernel_size: int = 5,
        dilation_rates: list[int] = [1, 2, 4],
        temperature: float = 0.10,
    ):
        super().__init__()
        self.temperature = float(temperature)

        # 1. Input projection
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(num_groups=4, num_channels=hidden_channels),
            nn.GELU(),
        )

        # 2. Dilated Residual Cascade
        blocks = []
        for i in range(num_res_blocks):
            dil = dilation_rates[i % len(dilation_rates)]
            blocks.append(ResidualBlock1D(hidden_channels, kernel_size=kernel_size, dilation=dil))
        self.res_cascade = nn.Sequential(*blocks)

        # 3. Output Projection Head
        self.out_conv = nn.Sequential(
            nn.Conv1d(hidden_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (B, 1, N_tau) or (B, N_tau)
        Returns: likelihood (B, N_tau) summing to 1.0 along the lag axis
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        feat = self.in_conv(x)
        feat = self.res_cascade(feat)
        logits = self.out_conv(feat).squeeze(1)  # Shape (B, N_tau)

        # Temperature-scaled Softmax to form probability distribution
        likelihood = F.softmax(logits / max(1e-4, self.temperature), dim=-1)
        return likelihood
