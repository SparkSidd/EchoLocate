"""
EchoLocate — Visualizations: Environmental Robustness (Figures A1–A9)
======================================================================
Generates all 9 required figures for Part A of Experiment 2.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Consistent publication styling
PALETTE_SNR = ["#2563eb", "#06b6d4", "#10b981", "#f59e0b", "#f97316", "#ef4444"]
PALETTE_RT60 = ["#1e293b", "#334155", "#475569", "#0284c7", "#0d9488", "#d97706", "#dc2626"]


def set_plot_theme():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.titlesize": 13,
        "figure.dpi": 300,
    })


def generate_all_robustness_figures(
    df_summary: pd.DataFrame,
    df_all_trials: pd.DataFrame,
    df_rt60_calib: pd.DataFrame,
    output_dir: Path,
):
    """
    Generate and save Figures A1 through A9.
    """
    set_plot_theme()
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating Environmental Robustness Figures A1–A9 into: %s", fig_dir)

    # --------------------------------------------------------------------------
    # Figure A1: Mean Localization Error vs. RT60 (lines per SNR)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    snr_levels = sorted(df_summary["snr_target_db"].unique(), reverse=True)
    
    for idx, snr in enumerate(snr_levels):
        df_sub = df_summary[df_summary["snr_target_db"] == snr].sort_values("rt60_target_s")
        color = PALETTE_SNR[idx % len(PALETTE_SNR)]
        ax.plot(
            df_sub["rt60_target_s"],
            df_sub["mean_localization_error_m"],
            marker="o",
            linewidth=2,
            markersize=6,
            label=f"SNR = {snr:.0f} dB",
            color=color,
        )
        
    ax.set_xlabel("Target Reverberation Time $RT_{60}$ (seconds)")
    ax.set_ylabel("Mean Localization Error (meters)")
    ax.set_title("Figure A1: Mean Localization Error vs. Reverberation Time ($RT_{60}$)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(title="Noise Level", frameon=True, facecolor="white", edgecolor="none")
    fig.tight_layout()
    fig.savefig(fig_dir / "figA1_error_vs_rt60.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure A2: Mean Localization Error vs. SNR (lines per RT60)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rt60_levels = sorted(df_summary["rt60_target_s"].unique())
    
    for idx, rt60 in enumerate(rt60_levels):
        df_sub = df_summary[df_summary["rt60_target_s"] == rt60].sort_values("snr_target_db")
        color = PALETTE_RT60[idx % len(PALETTE_RT60)]
        lbl = f"$RT_{{60}} = {rt60:.1f}$ s" if rt60 > 0 else "Anechoic"
        ax.plot(
            df_sub["snr_target_db"],
            df_sub["mean_localization_error_m"],
            marker="s",
            linewidth=2,
            markersize=6,
            label=lbl,
            color=color,
        )
        
    ax.set_xlabel("Signal-to-Noise Ratio SNR (dB)")
    ax.set_ylabel("Mean Localization Error (meters)")
    ax.set_title("Figure A2: Mean Localization Error vs. Signal-to-Noise Ratio (SNR)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(title="Reverberation", frameon=True, facecolor="white", edgecolor="none")
    fig.tight_layout()
    fig.savefig(fig_dir / "figA2_error_vs_snr.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure A3: Mean Localization Error Heatmap (RT60 x SNR)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    pivot_mean = df_summary.pivot(index="rt60_target_s", columns="snr_target_db", values="mean_localization_error_m")
    pivot_mean = pivot_mean.sort_index(ascending=False).sort_index(axis=1, ascending=False)
    
    sns.heatmap(
        pivot_mean,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Mean Localization Error (m)"},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Target Signal-to-Noise Ratio SNR (dB)")
    ax.set_ylabel("Target Reverberation Time $RT_{60}$ (s)")
    ax.set_title("Figure A3: Mean Localization Error Matrix ($RT_{60} \\times$ SNR)")
    fig.tight_layout()
    fig.savefig(fig_dir / "figA3_mean_error_heatmap.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure A4: Median Localization Error Heatmap (RT60 x SNR)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    pivot_med = df_summary.pivot(index="rt60_target_s", columns="snr_target_db", values="median_localization_error_m")
    pivot_med = pivot_med.sort_index(ascending=False).sort_index(axis=1, ascending=False)
    
    sns.heatmap(
        pivot_med,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Median Localization Error (m)"},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Target Signal-to-Noise Ratio SNR (dB)")
    ax.set_ylabel("Target Reverberation Time $RT_{60}$ (s)")
    ax.set_title("Figure A4: Median Localization Error Matrix ($RT_{60} \\times$ SNR)")
    fig.tight_layout()
    fig.savefig(fig_dir / "figA4_median_error_heatmap.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure A5: Mean Absolute TDOA Error Heatmap (RT60 x SNR)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    pivot_tdoa = df_summary.pivot(index="rt60_target_s", columns="snr_target_db", values="mean_tdoa_error_ms")
    pivot_tdoa = pivot_tdoa.sort_index(ascending=False).sort_index(axis=1, ascending=False)
    
    sns.heatmap(
        pivot_tdoa,
        annot=True,
        fmt=".2f",
        cmap="PuBuGn",
        cbar_kws={"label": "Mean Absolute TDOA Error (ms)"},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Target Signal-to-Noise Ratio SNR (dB)")
    ax.set_ylabel("Target Reverberation Time $RT_{60}$ (s)")
    ax.set_title("Figure A5: GCC-PHAT Mean Absolute TDOA Error Matrix (ms)")
    fig.tight_layout()
    fig.savefig(fig_dir / "figA5_tdoa_error_heatmap.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure A6: Localization Success Rate (% within 1.0 m)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    pivot_succ = df_summary.pivot(index="rt60_target_s", columns="snr_target_db", values="pct_within_1_0m")
    pivot_succ = pivot_succ.sort_index(ascending=False).sort_index(axis=1, ascending=False)
    
    sns.heatmap(
        pivot_succ,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar_kws={"label": "Success Rate (% $\\leq 1.0$ m)"},
        linewidths=0.5,
        vmin=0,
        vmax=100,
        ax=ax,
    )
    ax.set_xlabel("Target Signal-to-Noise Ratio SNR (dB)")
    ax.set_ylabel("Target Reverberation Time $RT_{60}$ (s)")
    ax.set_title("Figure A6: Localization Success Rate Matrix (% within 1.0 m)")
    fig.tight_layout()
    fig.savefig(fig_dir / "figA6_success_rate_1m_heatmap.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure A7: Error Distribution Boxplots across selected RT60 values
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    selected_rt60s = [0.0, 0.2, 0.4, 0.6]
    df_box = df_all_trials[df_all_trials["rt60_target_s"].isin(selected_rt60s)].copy()
    df_box["rt60_label"] = df_box["rt60_target_s"].apply(lambda r: "Anechoic" if r == 0 else f"$RT_{{60}}={r:.1f}$s")
    
    sns.boxplot(
        data=df_box,
        x="rt60_label",
        y="localization_error",
        hue="snr_target_db",
        palette="viridis",
        ax=ax,
        fliersize=2,
    )
    ax.set_xlabel("Reverberation Condition")
    ax.set_ylabel("Localization Error (meters)")
    ax.set_title("Figure A7: Localization Error Distributions across Selected $RT_{60}$ and SNR Levels")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(title="SNR (dB)", loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(fig_dir / "figA7_error_distribution_boxplots.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure A8: TDOA Error vs. Localization Error Scatter
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Compute mean TDOA error for each evaluation row in df_all_trials
    tdoa_mean_ms = (
        np.abs(df_all_trials["tdoa_01_estimated_s"] - df_all_trials["tdoa_01_theoretical_s"]) +
        np.abs(df_all_trials["tdoa_02_estimated_s"] - df_all_trials["tdoa_02_theoretical_s"]) +
        np.abs(df_all_trials["tdoa_12_estimated_s"] - df_all_trials["tdoa_12_theoretical_s"])
    ) / 3.0 * 1000.0
    
    scatter = ax.scatter(
        tdoa_mean_ms,
        df_all_trials["localization_error"],
        c=df_all_trials["rt60_target_s"],
        cmap="plasma",
        alpha=0.6,
        s=30,
        edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Target $RT_{60}$ (seconds)", rotation=270, labelpad=15)
    
    ax.set_xlabel("Mean Absolute TDOA Error (milliseconds)")
    ax.set_ylabel("Localization Error (meters)")
    ax.set_title("Figure A8: TDOA Estimation Error vs. Spatial Localization Error")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(fig_dir / "figA8_tdoa_vs_localization_error_scatter.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure A9: Measured RT60 vs. Target RT60 Calibration
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    df_rev_calib = df_rt60_calib[~df_rt60_calib["is_anechoic"]].sort_values("target_rt60_s")
    
    ax.plot([0.0, 0.7], [0.0, 0.7], "k--", linewidth=1.5, label="Ideal Calibration ($y = x$)")
    
    ax.errorbar(
        df_rev_calib["target_rt60_s"],
        df_rev_calib["measured_mean_rt60_s"],
        yerr=df_rev_calib["measured_std_rt60_s"],
        fmt="o-",
        color="#2563eb",
        linewidth=2,
        capsize=4,
        capthick=1.5,
        markersize=7,
        label="Simulated $RT_{60}$ (Mean $\\pm$ Std)",
    )
    ax.scatter(
        df_rev_calib["target_rt60_s"],
        df_rev_calib["measured_median_rt60_s"],
        color="#dc2626",
        marker="s",
        s=50,
        zorder=5,
        label="Simulated $RT_{60}$ (Median)",
    )
    
    ax.set_xlabel("Target $RT_{60}$ via Sabine Inversion (seconds)")
    ax.set_ylabel("Empirically Measured RIR $RT_{60}$ (seconds)")
    ax.set_title("Figure A9: RIR Schroeder Decay Calibration: Target vs. Measured $RT_{60}$")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=True, facecolor="white")
    ax.set_xlim(-0.02, 0.7)
    ax.set_ylim(-0.02, 0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "figA9_rt60_calibration.png")
    plt.close(fig)

    logger.info("Successfully generated and saved all 9 Part A figures.")
