"""
EchoLocate — DeepGCC Training & Dataset Generation Module
=========================================================
Generates synthetic multi-channel training scenes, constructs PyTorch Datasets,
computes composite losses (KL + TDOA + IPD), and executes the training loop.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyroomacoustics as pra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from src.experiments.environment_robustness import (
    generate_exact_snr_noise,
    safe_inverse_sabine,
)
from src.models.acoustic_frontend import (
    extract_pairwise_gcc_features,
    generate_experiment_04_signal,
    generate_gaussian_target_distribution,
)
from src.models.neural_gcc import DeepGCCRes1D

logger = logging.getLogger(__name__)


class SyntheticGCCDataset(Dataset):
    """PyTorch Dataset for pairwise GCC features and target Gaussian delay distributions."""

    def __init__(
        self,
        gcc_features: np.ndarray,
        target_distributions: np.ndarray,
        true_tdoas: np.ndarray,
        tau_axis: np.ndarray,
    ):
        self.gcc = torch.from_numpy(gcc_features.astype(np.float32))
        self.targets = torch.from_numpy(target_distributions.astype(np.float32))
        self.true_tdoas = torch.from_numpy(true_tdoas.astype(np.float32))
        self.tau_axis = torch.from_numpy(tau_axis.astype(np.float32))

    def __len__(self) -> int:
        return len(self.gcc)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "gcc": self.gcc[idx],
            "target": self.targets[idx],
            "true_tdoa": self.true_tdoas[idx],
            "tau_axis": self.tau_axis,
        }


def build_synthetic_training_dataset(
    num_scenes: int = 600,
    room_dims: list[float] = [10.0, 6.0, 3.0],
    mics_2d: list[list[float]] = [[1.0, 1.5], [9.0, 1.5], [5.0, 4.5]],
    fs: int = 16000,
    base_seed: int = 100,
    max_tau_seconds: float = 0.030,
    num_tau_bins: int = 129,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate randomized acoustic rooms and generate training tensors.
    """
    rng = np.random.default_rng(base_seed)
    mics_2d_arr = np.array(mics_2d, dtype=np.float64)
    mics_3d = np.column_stack([mics_2d_arr, np.full(len(mics_2d_arr), 1.2)]).T
    c = 343.0

    pairs = []
    for i in range(len(mics_2d_arr)):
        for j in range(i + 1, len(mics_2d_arr)):
            pairs.append((i, j))
    n_pairs = len(pairs)

    all_gcc = []
    all_targets = []
    all_tdoas = []

    signal_palette = ["broadband_noise", "linear_chirp", "log_chirp", "speech_male_vowel", "speech_female_vowel"]

    for sc_idx in range(num_scenes):
        s_seed = base_seed + sc_idx
        sc_rng = np.random.default_rng(s_seed)

        # Randomize acoustic parameters
        rt60 = sc_rng.uniform(0.18, 0.45)
        snr_db = sc_rng.uniform(15.0, 40.0)
        src_x = sc_rng.uniform(0.5, room_dims[0] - 0.5)
        src_y = sc_rng.uniform(0.5, room_dims[1] - 0.5)
        sig_type = sc_rng.choice(signal_palette)

        sig, _ = generate_experiment_04_signal(sig_type, fs=fs, duration=0.1, seed=s_seed)
        abs_mat, max_ord = safe_inverse_sabine(rt60, room_dims)

        room = pra.ShoeBox(room_dims, fs=fs, materials=pra.Material(abs_mat), max_order=max_ord)
        room.add_microphone_array(pra.MicrophoneArray(mics_3d, fs=fs))
        room.add_source([src_x, src_y, 1.2], signal=sig)
        room.simulate()
        clean_sig = room.mic_array.signals.copy()

        noisy_sig, _ = generate_exact_snr_noise(clean_sig, snr_db, seed=s_seed)

        # Feature extraction
        gcc_feat, _, tau_axis = extract_pairwise_gcc_features(
            noisy_sig, fs=fs, max_tau_seconds=max_tau_seconds, num_tau_bins=num_tau_bins
        )

        # Compute true theoretical TDOAs
        for p_idx, (i, j) in enumerate(pairs):
            d_i = np.linalg.norm(mics_2d_arr[i] - np.array([src_x, src_y]))
            d_j = np.linalg.norm(mics_2d_arr[j] - np.array([src_x, src_y]))
            true_tdoa = (d_i - d_j) / c

            target_dist = generate_gaussian_target_distribution(true_tdoa, tau_axis, sigma_seconds=0.0001)

            all_gcc.append(gcc_feat[p_idx])
            all_targets.append(target_dist)
            all_tdoas.append(true_tdoa)

    return (
        np.array(all_gcc, dtype=np.float32),
        np.array(all_targets, dtype=np.float32),
        np.array(all_tdoas, dtype=np.float32),
        tau_axis,
    )


def compute_composite_loss(
    pred_likelihood: torch.Tensor,
    target_distribution: torch.Tensor,
    true_tdoa: torch.Tensor,
    tau_axis: torch.Tensor,
    lambda_kl: float = 1.0,
    lambda_tdoa: float = 0.5,
    lambda_ipd: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Calculate composite loss combining KL divergence and expected TDOA error.
    """
    # 1. KL Divergence: KL(target || pred)
    # pred is a probability distribution (sum to 1.0)
    eps = 1e-10
    log_pred = torch.log(pred_likelihood + eps)
    kl_loss = F.kl_div(log_pred, target_distribution, reduction="batchmean")

    # 2. Expected Delay Bias
    # expected_tau = sum_k (tau_k * P(tau_k))
    expected_tau = torch.sum(pred_likelihood * tau_axis.unsqueeze(0), dim=-1)
    tdoa_loss = torch.mean(torch.abs(expected_tau - true_tdoa))

    # 3. Peak Entropy Loss (encourages sharp unimodal peak)
    entropy_loss = -torch.mean(torch.sum(pred_likelihood * torch.log(pred_likelihood + eps), dim=-1))

    total_loss = lambda_kl * kl_loss + lambda_tdoa * tdoa_loss + lambda_ipd * entropy_loss

    loss_dict = {
        "total_loss": float(total_loss.item()),
        "kl_loss": float(kl_loss.item()),
        "tdoa_loss": float(tdoa_loss.item()),
        "entropy_loss": float(entropy_loss.item()),
    }
    return total_loss, loss_dict


def train_deep_gcc_model(
    config_path: str | Path = "configs/experiment_04.yaml",
    save_checkpoint_path: str | Path = "results/experiment_04/deep_gcc_model.pt",
    history_csv_path: str | Path = "results/experiment_04/training_history.csv",
    use_ipd_loss: bool = True,
    smoke_test: bool = False,
) -> tuple[DeepGCCRes1D, pd.DataFrame]:
    """
    Train DeepGCC model and save checkpoint and training history.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    save_p = Path(save_checkpoint_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)

    n_train_scenes = 20 if smoke_test else cfg["training"]["num_train_scenes"]
    n_val_scenes = 5 if smoke_test else cfg["training"]["num_val_scenes"]
    n_epochs = 3 if smoke_test else cfg["training"]["num_epochs"]
    batch_size = 8 if smoke_test else cfg["training"]["batch_size"]
    lr = float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"]["weight_decay"])
    patience = int(cfg["training"]["early_stopping_patience"])

    room_dims = cfg["physical_world"]["room_dimensions"]
    mics_2d = cfg["physical_world"]["mic_positions_2d"]
    fs = int(cfg["physical_world"]["fs"])
    max_tau = float(cfg["neural_model"]["max_tau_seconds"])
    n_tau = int(cfg["neural_model"]["num_tau_bins"])

    logger.info("Generating synthetic training scenes (%d scenes)...", n_train_scenes)
    train_gcc, train_tgt, train_tdoas, tau_axis = build_synthetic_training_dataset(
        num_scenes=n_train_scenes, room_dims=room_dims, mics_2d=mics_2d, fs=fs,
        base_seed=cfg["reproducibility"]["train_seed"], max_tau_seconds=max_tau, num_tau_bins=n_tau
    )

    logger.info("Generating synthetic validation scenes (%d scenes)...", n_val_scenes)
    val_gcc, val_tgt, val_tdoas, _ = build_synthetic_training_dataset(
        num_scenes=n_val_scenes, room_dims=room_dims, mics_2d=mics_2d, fs=fs,
        base_seed=cfg["reproducibility"]["train_seed"] + 5000, max_tau_seconds=max_tau, num_tau_bins=n_tau
    )

    train_ds = SyntheticGCCDataset(train_gcc, train_tgt, train_tdoas, tau_axis)
    val_ds = SyntheticGCCDataset(val_gcc, val_tgt, val_tdoas, tau_axis)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DeepGCCRes1D(
        in_channels=int(cfg["neural_model"]["in_channels"]),
        hidden_channels=int(cfg["neural_model"]["hidden_channels"]),
        num_res_blocks=int(cfg["neural_model"]["num_res_blocks"]),
        kernel_size=int(cfg["neural_model"]["kernel_size"]),
        dilation_rates=list(cfg["neural_model"]["dilation_rates"]),
        temperature=float(cfg["neural_model"]["temperature"]),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    lambda_kl = float(cfg["training"]["loss_weights"]["lambda_kl"])
    lambda_tdoa = float(cfg["training"]["loss_weights"]["lambda_tdoa"])
    lambda_ipd = float(cfg["training"]["loss_weights"]["lambda_ipd"]) if use_ipd_loss else 0.0

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    logger.info("Starting DeepGCC training (%d epochs, %d training samples)...", n_epochs, len(train_ds))

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch["gcc"])
            loss, l_dict = compute_composite_loss(
                pred, batch["target"], batch["true_tdoa"], batch["tau_axis"][0],
                lambda_kl=lambda_kl, lambda_tdoa=lambda_tdoa, lambda_ipd=lambda_ipd
            )
            loss.backward()
            optimizer.step()
            train_losses.append(l_dict["total_loss"])

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        val_tdoa_errs = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["gcc"])
                loss, l_dict = compute_composite_loss(
                    pred, batch["target"], batch["true_tdoa"], batch["tau_axis"][0],
                    lambda_kl=lambda_kl, lambda_tdoa=lambda_tdoa, lambda_ipd=lambda_ipd
                )
                val_losses.append(l_dict["total_loss"])

                # Measure predicted peak TDOA error
                max_idx = torch.argmax(pred, dim=-1)
                tau_ax = batch["tau_axis"][0]
                pred_tau = tau_ax[max_idx]
                tdoa_err = torch.abs(pred_tau - batch["true_tdoa"])
                val_tdoa_errs.extend(tdoa_err.cpu().numpy().tolist())

        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        mean_val_tdoa_err_us = float(np.mean(val_tdoa_errs) * 1e6)

        history.append({
            "epoch": epoch,
            "train_loss": mean_train,
            "val_loss": mean_val,
            "val_mean_tdoa_error_us": mean_val_tdoa_err_us,
            "lr": float(scheduler.get_last_lr()[0]),
        })

        logger.info(
            "Epoch %02d/%02d | Train Loss: %.4f | Val Loss: %.4f | Val TDOA Error: %.2f us",
            epoch, n_epochs, mean_train, mean_val, mean_val_tdoa_err_us,
        )

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            patience_counter = 0
            torch.save(model.state_dict(), save_p)
        else:
            patience_counter += 1
            if patience_counter >= patience and not smoke_test:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    # Load best weights
    if save_p.exists():
        model.load_state_dict(torch.load(save_p))
    model.eval()

    df_hist = pd.DataFrame(history)
    df_hist.to_csv(history_csv_path, index=False)
    logger.info("Training complete. Best model checkpoint saved to: %s", save_p)
    return model, df_hist
