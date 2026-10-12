"""
EchoLocate — Experiment 4.5 Training & Ablation Engine
======================================================
Trains and evaluates calibrated DeepGCC architectures and differentiable SRP models
with strict validation-only checkpoint selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyroomacoustics as pra
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

from src.models.acoustic_frontend import (
    extract_pairwise_gcc_features,
    generate_experiment_04_signal,
    generate_gaussian_target_distribution,
)
from src.models.deepgcc_calibrated import CalibratedDeepGCC
from src.models.differentiable_srp import DifferentiableSRPLayer
from src.training.likelihood_losses import compute_hybrid_experiment_04_5_loss

logger = logging.getLogger(__name__)


class MultiChannelSpatialDataset(Dataset):
    """
    Dataset containing pairwise GCC features, pairwise target distributions,
    true pairwise TDOAs, and ground-truth 2D source positions for spatial supervision.
    """

    def __init__(
        self,
        gcc_features: np.ndarray,  # (N_scenes, N_pairs, N_tau)
        target_distributions: np.ndarray,  # (N_scenes, N_pairs, N_tau)
        true_tdoas: np.ndarray,  # (N_scenes, N_pairs)
        source_positions_2d: np.ndarray,  # (N_scenes, 2)
        tau_axis: np.ndarray,  # (N_tau,)
    ) -> None:
        self.gcc_features = torch.from_numpy(gcc_features.astype(np.float32))
        self.target_distributions = torch.from_numpy(target_distributions.astype(np.float32))
        self.true_tdoas = torch.from_numpy(true_tdoas.astype(np.float32))
        self.source_positions_2d = torch.from_numpy(source_positions_2d.astype(np.float32))
        self.tau_axis = torch.from_numpy(tau_axis.astype(np.float32))

    def __len__(self) -> int:
        return len(self.gcc_features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "gcc_features": self.gcc_features[idx],
            "target_distributions": self.target_distributions[idx],
            "true_tdoas": self.true_tdoas[idx],
            "source_xy": self.source_positions_2d[idx],
        }


def generate_synthetic_spatial_dataset(
    num_scenes: int,
    room_dims: list[float] = [10.0, 6.0, 3.0],
    mics_2d: list[list[float]] = ((2.0, 1.5), (8.0, 1.5), (5.0, 4.5)),
    fs: int = 16000,
    base_seed: int = 4500,
    target_sigma_s: float = 0.002,
    max_tau_seconds: float = 0.030,
    num_tau_bins: int = 129,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic scenes with spatial (x, y) ground truth."""
    rng = np.random.RandomState(base_seed)
    signal_types = ["broadband_noise", "linear_chirp", "log_chirp", "speech_male_vowel", "speech_female_vowel"]

    gcc_list = []
    tgt_list = []
    tdoa_list = []
    xy_list = []

    mics_arr = np.array(mics_2d)
    n_mics = len(mics_arr)
    pairs = [(i, j) for i in range(n_mics) for j in range(i + 1, n_mics)]

    tau_axis = np.linspace(-max_tau_seconds, max_tau_seconds, num_tau_bins, dtype=np.float32)

    for s_idx in range(num_scenes):
        rt60 = float(rng.uniform(0.18, 0.45))
        snr_db = float(rng.uniform(15.0, 40.0))
        sig_type = signal_types[s_idx % len(signal_types)]

        src_x = float(rng.uniform(0.5, room_dims[0] - 0.5))
        src_y = float(rng.uniform(0.5, room_dims[1] - 0.5))
        src_pos = [src_x, src_y, 1.5]

        # Pyroomacoustics simulation
        try:
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dims)
            room = pra.ShoeBox(
                room_dims, fs=fs,
                materials=pra.Material(e_absorption),
                max_order=min(12, max_order),
                air_absorption=False
            )
            mic_positions_3d = np.array([[m[0], m[1], 1.5] for m in mics_2d]).T
            room.add_microphone_array(mic_positions_3d)

            src_sig, _ = generate_experiment_04_signal(sig_type, fs=fs, duration=0.1, seed=base_seed + s_idx)
            room.add_source(src_pos, signal=src_sig)
            room.simulate(snr=snr_db)

            mic_signals = room.mic_array.signals
            scene_gcc, _, _ = extract_pairwise_gcc_features(
                mic_signals, fs=fs, max_tau_seconds=max_tau_seconds, num_tau_bins=num_tau_bins
            )
        except Exception:
            scene_gcc = np.zeros((len(pairs), num_tau_bins), dtype=np.float32)

        # Theoretical TDOAs and continuous targets
        scene_tgts = []
        scene_tdoas = []
        for i, j in pairs:
            d_i = np.linalg.norm(np.array([src_x, src_y]) - mics_arr[i])
            d_j = np.linalg.norm(np.array([src_x, src_y]) - mics_arr[j])
            tau_true = float((d_i - d_j) / 343.0)
            tgt_curve = generate_gaussian_target_distribution(tau_true, tau_axis, sigma_seconds=target_sigma_s)
            scene_tgts.append(tgt_curve)
            scene_tdoas.append(tau_true)

        gcc_list.append(scene_gcc)
        tgt_list.append(np.array(scene_tgts, dtype=np.float32))
        tdoa_list.append(np.array(scene_tdoas, dtype=np.float32))
        xy_list.append(np.array([src_x, src_y], dtype=np.float32))

    return (
        np.array(gcc_list, dtype=np.float32),
        np.array(tgt_list, dtype=np.float32),
        np.array(tdoa_list, dtype=np.float32),
        np.array(xy_list, dtype=np.float32),
        tau_axis,
    )


def train_calibrated_model(
    config_path: str | Path = "configs/experiment_04_5.yaml",
    target_sigma_s: float = 0.002,
    enable_learned_temperature: bool = False,
    lambda_delay: float = 1.0,
    lambda_contrast: float = 0.5,
    lambda_entropy: float = 0.1,
    lambda_spatial: float = 0.0,
    lambda_position: float = 0.0,
    contrast_margin: float = 0.10,
    save_checkpoint_path: str | Path | None = None,
    smoke_test: bool = False,
) -> tuple[CalibratedDeepGCC, pd.DataFrame]:
    """
    Train a CalibratedDeepGCC model with specified loss configuration and validation selection.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    n_train = 20 if smoke_test else cfg["training"]["num_train_scenes"]
    n_val = 5 if smoke_test else cfg["training"]["num_val_scenes"]
    n_epochs = 3 if smoke_test else cfg["training"]["num_epochs"]
    batch_size = 8 if smoke_test else cfg["training"]["batch_size"]

    room_dims = cfg["physical_world"]["room_dimensions"]
    mics_2d = cfg["physical_world"]["mic_positions_2d"]
    fs = int(cfg["physical_world"]["fs"])
    max_tau = float(cfg["neural_model"]["max_tau_seconds"])
    n_tau = int(cfg["neural_model"]["num_tau_bins"])

    # Build datasets
    tr_gcc, tr_tgt, tr_tdoas, tr_xy, tau_axis = generate_synthetic_spatial_dataset(
        num_scenes=n_train, room_dims=room_dims, mics_2d=mics_2d, fs=fs,
        base_seed=cfg["reproducibility"]["train_seed"], target_sigma_s=target_sigma_s,
        max_tau_seconds=max_tau, num_tau_bins=n_tau
    )
    val_gcc, val_tgt, val_tdoas, val_xy, _ = generate_synthetic_spatial_dataset(
        num_scenes=n_val, room_dims=room_dims, mics_2d=mics_2d, fs=fs,
        base_seed=cfg["reproducibility"]["val_seed"], target_sigma_s=target_sigma_s,
        max_tau_seconds=max_tau, num_tau_bins=n_tau
    )

    train_ds = MultiChannelSpatialDataset(tr_gcc, tr_tgt, tr_tdoas, tr_xy, tau_axis)
    val_ds = MultiChannelSpatialDataset(val_gcc, val_tgt, val_tdoas, val_xy, tau_axis)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CalibratedDeepGCC(
        in_channels=int(cfg["neural_model"]["in_channels"]),
        hidden_channels=int(cfg["neural_model"]["hidden_channels"]),
        num_res_blocks=int(cfg["neural_model"]["num_res_blocks"]),
        kernel_size=int(cfg["neural_model"]["kernel_size"]),
        dilation_rates=list(cfg["neural_model"]["dilation_rates"]),
        num_tau_bins=n_tau,
        temperature=float(cfg["neural_model"]["temperature"]),
        enable_learned_temperature=enable_learned_temperature,
    )

    srp_layer = None
    if lambda_spatial > 0.0 or lambda_position > 0.0:
        srp_layer = DifferentiableSRPLayer(
            room_dimensions=room_dims,
            mic_positions_2d=mics_2d,
            speed_of_sound=float(cfg["physical_world"]["speed_of_sound"]),
            grid_resolution_m=float(cfg["spatial_grid"]["grid_resolution_m"]),
            search_margin_m=float(cfg["spatial_grid"]["search_margin_m"]),
            max_tau_seconds=max_tau,
            num_tau_bins=n_tau,
            temperature_alpha=float(cfg["spatial_grid"]["soft_argmax_temperature"]),
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"])
    )

    best_val_loss = float("inf")
    history_records = []
    tau_tensor = torch.from_numpy(tau_axis)

    for epoch in range(1, n_epochs + 1):
        model.train()
        tr_loss_total = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            gcc_feat = batch["gcc_features"]  # (B, N_pairs, N_tau)
            tgt_dist = batch["target_distributions"]  # (B, N_pairs, N_tau)
            true_tdoas = batch["true_tdoas"]  # (B, N_pairs)
            true_xy = batch["source_xy"]  # (B, 2)

            B, N_pairs, N_tau = gcc_feat.shape
            # Flatten pairs for frontend
            gcc_flat = gcc_feat.view(B * N_pairs, 1, N_tau)
            tgt_flat = tgt_dist.view(B * N_pairs, N_tau)
            tdoa_flat = true_tdoas.view(B * N_pairs)

            pred_probs_flat, aux = model(gcc_flat)
            logits_flat = aux["logits"]

            # Reshape for spatial layer
            pred_probs_3d = pred_probs_flat.view(B, N_pairs, N_tau)
            conf_3d = aux["confidence"].view(B, N_pairs) if enable_learned_temperature else None

            spatial_prob, pred_xy, grid_pts = None, None, None
            if srp_layer is not None:
                _, spatial_prob, pred_xy = srp_layer(pred_probs_3d, pair_weights=conf_3d)
                grid_pts = srp_layer.grid_points_2d

            loss, _ = compute_hybrid_experiment_04_5_loss(
                pred_probs=pred_probs_flat,
                logits=logits_flat,
                target_probs=tgt_flat,
                true_tdoas=tdoa_flat,
                tau_axis=tau_tensor,
                spatial_prob=spatial_prob,
                pred_xy=pred_xy,
                true_xy=true_xy if srp_layer is not None else None,
                grid_points=grid_pts,
                lambda_delay=lambda_delay,
                lambda_contrast=lambda_contrast,
                lambda_entropy=lambda_entropy,
                lambda_spatial=lambda_spatial,
                lambda_position=lambda_position,
                contrast_margin=contrast_margin,
            )

            loss.backward()
            optimizer.step()
            tr_loss_total += float(loss.item()) * B

        tr_loss_avg = tr_loss_total / len(train_ds)

        # Validation Step
        model.eval()
        val_loss_total = 0.0
        val_loc_err_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                gcc_feat = batch["gcc_features"]
                tgt_dist = batch["target_distributions"]
                true_tdoas = batch["true_tdoas"]
                true_xy = batch["source_xy"]

                B, N_pairs, N_tau = gcc_feat.shape
                gcc_flat = gcc_feat.view(B * N_pairs, 1, N_tau)
                tgt_flat = tgt_dist.view(B * N_pairs, N_tau)
                tdoa_flat = true_tdoas.view(B * N_pairs)

                pred_probs_flat, aux = model(gcc_flat)
                logits_flat = aux["logits"]
                pred_probs_3d = pred_probs_flat.view(B, N_pairs, N_tau)
                conf_3d = aux["confidence"].view(B, N_pairs) if enable_learned_temperature else None

                spatial_prob, pred_xy, grid_pts = None, None, None
                if srp_layer is not None:
                    _, spatial_prob, pred_xy = srp_layer(pred_probs_3d, pair_weights=conf_3d)
                    grid_pts = srp_layer.grid_points_2d
                    loc_err = torch.norm(pred_xy - true_xy, dim=-1)
                    val_loc_err_total += float(torch.sum(loc_err).item())

                v_loss, _ = compute_hybrid_experiment_04_5_loss(
                    pred_probs=pred_probs_flat,
                    logits=logits_flat,
                    target_probs=tgt_flat,
                    true_tdoas=tdoa_flat,
                    tau_axis=tau_tensor,
                    spatial_prob=spatial_prob,
                    pred_xy=pred_xy,
                    true_xy=true_xy if srp_layer is not None else None,
                    grid_points=grid_pts,
                    lambda_delay=lambda_delay,
                    lambda_contrast=lambda_contrast,
                    lambda_entropy=lambda_entropy,
                    lambda_spatial=lambda_spatial,
                    lambda_position=lambda_position,
                    contrast_margin=contrast_margin,
                )
                val_loss_total += float(v_loss.item()) * B

        val_loss_avg = val_loss_total / len(val_ds)
        val_loc_avg = val_loc_err_total / max(1, len(val_ds))

        history_records.append({
            "epoch": epoch,
            "train_loss": tr_loss_avg,
            "val_loss": val_loss_avg,
            "val_localization_error_m": val_loc_avg,
        })

        if val_loss_avg < best_val_loss and save_checkpoint_path is not None:
            best_val_loss = val_loss_avg
            Path(save_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "target_sigma_s": target_sigma_s,
                "val_loss": val_loss_avg,
            }, str(save_checkpoint_path))

    df_hist = pd.DataFrame(history_records)
    return model, df_hist
