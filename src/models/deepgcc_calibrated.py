"""
EchoLocate — Experiment 4.5 Calibrated DeepGCC Architecture
===========================================================
Defines CalibratedDeepGCC with support for:
- 1D Dilated Residual ConvNet backbone
- Temperature-scaled Softmax & post-hoc sharpening
- Dynamic learned temperature & confidence heads (Method M7)
- Differentiable unnormalized logits output for margin contrast losses
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DilatedResidualBlock1D(nn.Module):
    """1D Dilated Residual Block with GroupNorm and GELU activation."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout_p: float = 0.05,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation, bias=False
        )
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation, bias=False
        )
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.act2 = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        out = self.act2(out + res)
        return out


class CalibratedDeepGCC(nn.Module):
    """
    Calibrated 1D Dilated Residual ConvNet for continuous delay likelihood estimation.
    Supports post-hoc sharpening and learned temperature/confidence heads (M7).
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        num_res_blocks: int = 4,
        kernel_size: int = 5,
        dilation_rates: list[int] | None = None,
        num_tau_bins: int = 129,
        temperature: float = 1.0,
        enable_learned_temperature: bool = False,
    ) -> None:
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 1]

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_tau_bins = num_tau_bins
        self.temperature = temperature
        self.enable_learned_temperature = enable_learned_temperature

        # Input projection
        self.input_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.input_norm = nn.GroupNorm(num_groups=min(8, hidden_channels), num_channels=hidden_channels)
        self.input_act = nn.GELU()

        # Residual backbone
        blocks = []
        for i in range(num_res_blocks):
            d = dilation_rates[i % len(dilation_rates)]
            blocks.append(DilatedResidualBlock1D(hidden_channels, kernel_size=kernel_size, dilation=d))
        self.backbone = nn.Sequential(*blocks)

        # Output projection for delay logits z(tau)
        self.out_conv = nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1)

        # Optional Learned Calibration Head (Method M7)
        if enable_learned_temperature:
            self.calib_pool = nn.AdaptiveAvgPool1d(1)
            self.calib_fc = nn.Sequential(
                nn.Linear(hidden_channels, 32),
                nn.GELU(),
                nn.Linear(32, 2),  # [log_temp, confidence_logit]
            )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract raw unnormalized correlation logits z(tau).
        Input: (B, in_channels, num_tau_bins)
        Output: (B, num_tau_bins)
        """
        feat = self.input_act(self.input_norm(self.input_conv(x)))
        feat = self.backbone(feat)
        logits = self.out_conv(feat).squeeze(1)  # (B, num_tau_bins)
        return logits

    def forward(
        self,
        x: torch.Tensor,
        custom_temperature: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass producing normalized probability likelihood distribution L_ij(tau).
        Returns:
            probs: (B, num_tau_bins)
            aux_info: dict with 'temperature', 'confidence', 'logits'
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.input_act(self.input_norm(self.input_conv(x)))
        feat = self.backbone(feat)
        logits = self.out_conv(feat).squeeze(1)  # (B, num_tau_bins)

        temp = self.temperature if custom_temperature is None else custom_temperature
        conf = torch.ones(logits.shape[0], device=logits.device)

        if self.enable_learned_temperature:
            pooled = self.calib_pool(feat).squeeze(-1)  # (B, hidden_channels)
            calib_out = self.calib_fc(pooled)  # (B, 2)
            # Learned temperature T in [0.05, 3.0]
            learned_t = 0.05 + 2.95 * torch.sigmoid(calib_out[:, 0:1])
            learned_c = torch.sigmoid(calib_out[:, 1])
            temp_tensor = learned_t
            conf = learned_c
            probs = F.softmax(logits / temp_tensor, dim=-1)
        else:
            probs = F.softmax(logits / max(1e-4, temp), dim=-1)

        aux_info = {
            "logits": logits,
            "temperature": torch.tensor(temp) if isinstance(temp, float) else temp,
            "confidence": conf,
        }
        return probs, aux_info


def apply_temperature_sharpening(
    probs: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    Post-hoc temperature sharpening function:
    L_T(tau) = L(tau)^(1/T) / sum(L(tau)^(1/T))
    """
    safe_p = torch.clamp(probs, min=1e-12)
    inv_t = 1.0 / max(1e-4, temperature)
    powered = torch.pow(safe_p, inv_t)
    norm = torch.sum(powered, dim=-1, keepdim=True)
    return powered / torch.clamp(norm, min=1e-12)
