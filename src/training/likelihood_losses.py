"""
EchoLocate — Experiment 4.5 Loss Functions
==========================================
Implements multi-level loss objectives:
- L_delay: Pairwise delay distribution KL divergence / cross-entropy
- L_contrast: Differentiable margin-based peak contrast loss (Method M8)
- L_entropy: Output likelihood entropy regularization
- L_spatial: Negative log-likelihood on differentiable spatial grid (Method M10)
- L_position: Soft-argmax Euclidean localization loss (Method M11)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_delay_kl_loss(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    """KL Divergence loss between predicted likelihood L_ij(tau) and target Gaussian."""
    safe_pred = torch.clamp(pred_probs, min=1e-12)
    safe_tgt = torch.clamp(target_probs, min=1e-12)
    kl = F.kl_div(torch.log(safe_pred), safe_tgt, reduction="batchmean")
    return kl


def compute_peak_contrast_loss(
    logits: torch.Tensor,
    true_tdoas: torch.Tensor,
    tau_axis: torch.Tensor,
    margin: float = 0.10,
    neighborhood_radius_s: float = 0.002,
) -> torch.Tensor:
    """
    Margin-based peak contrast loss:
        L_contrast = max(0, margin - [z(tau_true) - max_{tau notin N(tau_true)} z(tau)])
    """
    B = logits.shape[0]
    tau_expanded = tau_axis.unsqueeze(0).expand(B, -1)  # (B, N_tau)
    true_tau_exp = true_tdoas.unsqueeze(-1)  # (B, 1)

    # Neighborhood mask
    dists = torch.abs(tau_expanded - true_tau_exp)
    in_mask = dists <= neighborhood_radius_s
    out_mask = ~in_mask

    # Get true peak logit (max logit within neighborhood)
    in_logits = torch.where(in_mask, logits, torch.tensor(-1e9, device=logits.device))
    z_true = torch.max(in_logits, dim=-1)[0]  # (B,)

    # Get maximum competing logit outside neighborhood
    out_logits = torch.where(out_mask, logits, torch.tensor(-1e9, device=logits.device))
    z_competing = torch.max(out_logits, dim=-1)[0]  # (B,)

    # Hinge margin
    loss = F.relu(margin - (z_true - z_competing))
    return torch.mean(loss)


def compute_entropy_loss(
    probs: torch.Tensor,
) -> torch.Tensor:
    """Entropy of probability distribution: H = - sum p log(p)."""
    safe_p = torch.clamp(probs, min=1e-12)
    ent = -torch.sum(safe_p * torch.log(safe_p), dim=-1)
    return torch.mean(ent)


def compute_spatial_nll_loss(
    spatial_prob: torch.Tensor,
    true_xy: torch.Tensor,
    grid_points: torch.Tensor,
) -> torch.Tensor:
    """
    Spatial Negative Log-Likelihood (NLL) Loss:
    Finds closest spatial grid node to true source position and computes -log P(x_true).
    """
    B = true_xy.shape[0]
    # Find nearest grid node for each batch item
    # true_xy: (B, 2), grid_points: (N_grid, 2)
    dists = torch.cdist(true_xy, grid_points)  # (B, N_grid)
    nearest_idx = torch.argmin(dists, dim=-1)  # (B,)

    # Extract log probability at true grid node
    safe_prob = torch.clamp(spatial_prob, min=1e-12)
    log_prob = torch.log(safe_prob)
    # Gather nearest node log prob
    nll = -torch.gather(log_prob, dim=-1, index=nearest_idx.unsqueeze(-1)).squeeze(-1)
    return torch.mean(nll)


def compute_soft_argmax_position_loss(
    pred_xy: torch.Tensor,
    true_xy: torch.Tensor,
) -> torch.Tensor:
    """Euclidean distance loss between soft-argmax predicted position and ground truth."""
    err = torch.norm(pred_xy - true_xy, dim=-1)  # (B,)
    return torch.mean(err)


def compute_hybrid_experiment_04_5_loss(
    pred_probs: torch.Tensor,
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    true_tdoas: torch.Tensor,
    tau_axis: torch.Tensor,
    spatial_prob: torch.Tensor | None = None,
    pred_xy: torch.Tensor | None = None,
    true_xy: torch.Tensor | None = None,
    grid_points: torch.Tensor | None = None,
    lambda_delay: float = 1.0,
    lambda_contrast: float = 0.5,
    lambda_entropy: float = 0.1,
    lambda_spatial: float = 0.0,
    lambda_position: float = 0.0,
    contrast_margin: float = 0.10,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Master composite loss calculator supporting Methods M5, M8, M9, M10, and M11.
    """
    loss_delay = compute_delay_kl_loss(pred_probs, target_probs)
    loss_contrast = compute_peak_contrast_loss(logits, true_tdoas, tau_axis, margin=contrast_margin)
    loss_entropy = compute_entropy_loss(pred_probs)

    total_loss = lambda_delay * loss_delay + lambda_contrast * loss_contrast + lambda_entropy * loss_entropy
    loss_dict = {
        "loss_delay": float(loss_delay.item()),
        "loss_contrast": float(loss_contrast.item()),
        "loss_entropy": float(loss_entropy.item()),
        "loss_spatial": 0.0,
        "loss_position": 0.0,
    }

    if lambda_spatial > 0.0 and spatial_prob is not None and true_xy is not None and grid_points is not None:
        loss_sp = compute_spatial_nll_loss(spatial_prob, true_xy, grid_points)
        total_loss = total_loss + lambda_spatial * loss_sp
        loss_dict["loss_spatial"] = float(loss_sp.item())

    if lambda_position > 0.0 and pred_xy is not None and true_xy is not None:
        loss_pos = compute_soft_argmax_position_loss(pred_xy, true_xy)
        total_loss = total_loss + lambda_position * loss_pos
        loss_dict["loss_position"] = float(loss_pos.item())

    loss_dict["total_loss"] = float(total_loss.item())
    return total_loss, loss_dict
