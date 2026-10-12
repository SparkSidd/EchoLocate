"""
EchoLocate — Visualizations: Transition-Zone & Reverberation Threshold (Figures T1–T11)
========================================================================================
Generates all 11 required transition-zone figures and GCC-PHAT correlation diagnostic plots.
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


def set_plot_theme():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.titlesize": 13,
        "figure.dpi": 300,
    })


def generate_all_transition_zone_figures(
    df_summary: pd.DataFrame,
    df_evals: pd.DataFrame,
    df_rt60_summary: pd.DataFrame,
    df_per_source: pd.DataFrame,
    output_dir: Path,
):
    """
    Generate and save Figures T1 through T11 plus diagnostic plots.
    """
    set_plot_theme()
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating Transition-Zone Figures T1–T11 into: %s", fig_dir)

    df_sum_sorted = df_summary.sort_values("rt60_measured_mean_s").reset_index(drop=True)

    # --------------------------------------------------------------------------
    # Figure T1: Nominal Target RT60 vs. Mean Measured RT60
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5.5))
    x_tgt = df_rt60_summary["target_rt60"].values
    y_meas = df_rt60_summary["mean_measured_rt60"].values
    y_err = df_rt60_summary["std_measured_rt60"].values
    
    ax.errorbar(
        x_tgt,
        y_meas,
        yerr=y_err,
        fmt="o-",
        color="#2563eb",
        ecolor="#93c5fd",
        elinewidth=2,
        capsize=4,
        capthick=1.5,
        markersize=7,
        label="Schroeder Measured Mean $\\pm 1\\sigma$",
    )
    
    # Ideal y = x
    min_v = min(x_tgt.min(), y_meas.min()) * 0.9
    max_v = max(x_tgt.max(), y_meas.max()) * 1.1
    ax.plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.6, label="Ideal Target ($y = x$)")
    
    for x, y in zip(x_tgt, y_meas):
        ax.annotate(f"{y:.3f}s", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")
        
    ax.set_xlabel("Nominal Target $RT_{60}$ (seconds)")
    ax.set_ylabel("Measured Mean $RT_{60}$ (seconds)")
    ax.set_title("Figure T1: Nominal Target $RT_{60}$ vs. Empirical Schroeder Measured $RT_{60}$")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=True, facecolor="white")
    fig.tight_layout()
    fig.savefig(fig_dir / "T1_rt60_calibration.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure T2: Mean Localization Error vs. Target RT60
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(
        df_sum_sorted["rt60_target_s"],
        df_sum_sorted["mean_localization_error_m"],
        yerr=[
            df_sum_sorted["mean_localization_error_m"] - df_sum_sorted["ci95_low_m"],
            df_sum_sorted["ci95_high_m"] - df_sum_sorted["mean_localization_error_m"],
        ],
        fmt="s-",
        color="#d97706",
        ecolor="#fcd34d",
        elinewidth=2,
        capsize=4,
        markersize=6,
        label="Mean Error (95% CI)",
    )
    ax.set_xlabel("Nominal Target $RT_{60}$ (seconds)")
    ax.set_ylabel("Mean Localization Error (meters)")
    ax.set_title("Figure T2: Mean Localization Error vs. Nominal Target $RT_{60}$")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=True, facecolor="white")
    fig.tight_layout()
    fig.savefig(fig_dir / "T2_error_vs_target_rt60.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure T3: Mean Localization Error vs. Measured RT60 (Primary Axis)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(
        df_sum_sorted["rt60_measured_mean_s"],
        df_sum_sorted["mean_localization_error_m"],
        yerr=[
            df_sum_sorted["mean_localization_error_m"] - df_sum_sorted["ci95_low_m"],
            df_sum_sorted["ci95_high_m"] - df_sum_sorted["mean_localization_error_m"],
        ],
        fmt="o-",
        color="#dc2626",
        ecolor="#fca5a5",
        elinewidth=2,
        capsize=4,
        markersize=7,
        label="Mean Error (95% CI)",
    )
    for _, r in df_sum_sorted.iterrows():
        ax.annotate(f"{r['mean_localization_error_m']:.2f}m", (r["rt60_measured_mean_s"], r["mean_localization_error_m"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")
                    
    ax.set_xlabel("Schroeder Measured Mean $RT_{60}$ (seconds)")
    ax.set_ylabel("Mean Localization Error (meters)")
    ax.set_title("Figure T3: Mean Localization Error vs. Empirical Measured $RT_{60}$")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=True, facecolor="white")
    fig.tight_layout()
    fig.savefig(fig_dir / "T3_error_vs_measured_rt60.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure T4: Median Localization Error vs. Measured RT60
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(
        df_sum_sorted["rt60_measured_mean_s"],
        df_sum_sorted["median_localization_error_m"],
        "D-",
        color="#7c3aed",
        linewidth=2,
        markersize=7,
        label="Median Error",
    )
    ax.plot(
        df_sum_sorted["rt60_measured_mean_s"],
        df_sum_sorted["mean_localization_error_m"],
        "o--",
        color="#9ca3af",
        alpha=0.7,
        label="Mean Error (Reference)",
    )
    ax.set_xlabel("Schroeder Measured Mean $RT_{60}$ (seconds)")
    ax.set_ylabel("Localization Error (meters)")
    ax.set_title("Figure T4: Median vs. Mean Localization Error across Measured $RT_{60}$")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=True, facecolor="white")
    fig.tight_layout()
    fig.savefig(fig_dir / "T4_median_error_vs_measured_rt60.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure T5: TDOA RMSE vs. Measured RT60
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(
        df_sum_sorted["rt60_measured_mean_s"],
        df_sum_sorted["rmse_tdoa_error_ms"],
        "^-",
        color="#0891b2",
        linewidth=2,
        markersize=7,
        label="TDOA RMSE (ms)",
    )
    for _, r in df_sum_sorted.iterrows():
        ax.annotate(f"{r['rmse_tdoa_error_ms']:.2f}ms", (r["rt60_measured_mean_s"], r["rmse_tdoa_error_ms"]),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
                    
    ax.set_xlabel("Schroeder Measured Mean $RT_{60}$ (seconds)")
    ax.set_ylabel("TDOA RMSE (milliseconds)")
    ax.set_title("Figure T5: GCC-PHAT TDOA RMSE Degradation vs. Measured $RT_{60}$")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=True, facecolor="white")
    fig.tight_layout()
    fig.savefig(fig_dir / "T5_tdoa_error_vs_measured_rt60.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure T6: Success Rate (% within 1.0 m) vs. Measured RT60
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(
        df_sum_sorted["rt60_measured_mean_s"],
        df_sum_sorted["pct_within_1_0m"],
        "o-",
        color="#16a34a",
        linewidth=2.5,
        markersize=7,
        label="Success Rate (% $\\leq 1.0$ m)",
    )
    ax.axhline(80.0, color="#16a34a", linestyle=":", alpha=0.7, label="80% Usability Threshold")
    ax.axhline(50.0, color="#dc2626", linestyle=":", alpha=0.7, label="50% Collapse Threshold")
    
    for _, r in df_sum_sorted.iterrows():
        ax.annotate(f"{r['pct_within_1_0m']:.1f}%", (r["rt60_measured_mean_s"], r["pct_within_1_0m"]),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, fontweight="bold")
                    
    ax.set_xlabel("Schroeder Measured Mean $RT_{60}$ (seconds)")
    ax.set_ylabel("Success Rate (% $\\leq 1.0$ m)")
    ax.set_ylim(-5, 105)
    ax.set_title("Figure T6: Localization Reliability (% $\\leq 1.0$ m) vs. Measured $RT_{60}$")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=True, facecolor="white")
    fig.tight_layout()
    fig.savefig(fig_dir / "T6_success_rate_vs_measured_rt60.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure T7: Catastrophic Failure Rate (% > 2.0 m) vs. Measured RT60
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(
        df_sum_sorted["rt60_measured_mean_s"],
        df_sum_sorted["pct_catastrophic_failure"],
        "s-",
        color="#dc2626",
        linewidth=2.5,
        markersize=7,
        label="Catastrophic Failure Rate (% $> 2.0$ m)",
    )
    for _, r in df_sum_sorted.iterrows():
        ax.annotate(f"{r['pct_catastrophic_failure']:.1f}%", (r["rt60_measured_mean_s"], r["pct_catastrophic_failure"]),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, fontweight="bold")
                    
    ax.set_xlabel("Schroeder Measured Mean $RT_{60}$ (seconds)")
    ax.set_ylabel("Catastrophic Failure Rate (% $> 2.0$ m)")
    ax.set_ylim(-5, 105)
    ax.set_title("Figure T7: Catastrophic Localization Failure Rate vs. Measured $RT_{60}$")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=True, facecolor="white")
    fig.tight_layout()
    fig.savefig(fig_dir / "T7_failure_rate_vs_measured_rt60.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure T8: Boxplot of Localization Error Ordered by Measured RT60
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Order categories by measured RT60
    meas_map = dict(zip(df_sum_sorted["rt60_target_s"], df_sum_sorted["rt60_measured_mean_s"]))
    df_evals_copy = df_evals.copy()
    df_evals_copy["measured_lbl"] = df_evals_copy["target_rt60_s"].apply(lambda t: f"{meas_map.get(t, t):.3f}s\n(Tgt {t:.3f}s)")
    
    order_lbls = [f"{r['rt60_measured_mean_s']:.3f}s\n(Tgt {r['rt60_target_s']:.3f}s)" for _, r in df_sum_sorted.iterrows()]
    
    sns.boxplot(
        data=df_evals_copy,
        x="measured_lbl",
        y="localization_error",
        hue="measured_lbl",
        legend=False,
        order=order_lbls,
        palette="Reds",
        ax=ax,
        fliersize=3,
    )
    ax.set_xlabel("Condition (Ordered by Measured Mean $RT_{60}$)")
    ax.set_ylabel("Localization Error (meters)")
    ax.set_title("Figure T8: Error Distribution and Dispersion Ordered by Measured $RT_{60}$")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "T8_error_distribution.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure T9: Transition-Zone Per-Source Trajectory
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    src_piv = df_evals.groupby(["source_id", "measured_mean_rt60_s"])["localization_error"].mean().unstack()
    
    palette = sns.color_palette("tab20", len(src_piv))
    for idx, (s_id, row_vals) in enumerate(src_piv.iterrows()):
        ax.plot(row_vals.index, row_vals.values, "o-", label=s_id, color=palette[idx], alpha=0.8, linewidth=1.5)
        
    ax.set_xlabel("Schroeder Measured Mean $RT_{60}$ (seconds)")
    ax.set_ylabel("Mean Localization Error (meters)")
    ax.set_title("Figure T9: Per-Source Degradation Trajectories Across the Transition Zone")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "T9_transition_distribution.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure T10: TDOA Error vs. Localization Error Scatter (525 evaluations)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sc = ax.scatter(
        df_evals["mean_tdoa_error_ms"],
        df_evals["localization_error"],
        c=df_evals["measured_mean_rt60_s"],
        cmap="coolwarm",
        alpha=0.7,
        s=35,
        edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Measured Mean $RT_{60}$ (s)")
    
    ax.axhline(2.0, color="red", linestyle="--", alpha=0.6, label="Catastrophic Error Threshold ($2.0$ m)")
    ax.set_xlabel("Mean Absolute TDOA Error (milliseconds)")
    ax.set_ylabel("Localization Error (meters)")
    ax.set_title("Figure T10: Scatter Analysis: TDOA Error vs. Spatial Localization Error")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=True, facecolor="white")
    fig.tight_layout()
    fig.savefig(fig_dir / "T10_tdoa_vs_localization.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure T11: Transition-Zone Zoom Plot & Error Slope
    # --------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))
    
    color1 = "#dc2626"
    ax1.plot(df_sum_sorted["rt60_measured_mean_s"], df_sum_sorted["mean_localization_error_m"], "o-", color=color1, linewidth=2.5, label="Mean Error (m)")
    ax1.set_xlabel("Schroeder Measured Mean $RT_{60}$ (seconds)")
    ax1.set_ylabel("Mean Localization Error (m)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3, linestyle="--")
    
    # Calculate discrete derivative
    d_rt = np.diff(df_sum_sorted["rt60_measured_mean_s"].values)
    d_err = np.diff(df_sum_sorted["mean_localization_error_m"].values)
    slopes = d_err / np.maximum(1e-6, d_rt)
    mid_rt = 0.5 * (df_sum_sorted["rt60_measured_mean_s"].values[:-1] + df_sum_sorted["rt60_measured_mean_s"].values[1:])
    
    ax2 = ax1.twinx()
    color2 = "#2563eb"
    ax2.bar(mid_rt, slopes, width=0.015, color=color2, alpha=0.4, label="Degradation Slope $\\Delta E / \\Delta RT_{60}$ (m/s)")
    ax2.set_ylabel("Degradation Slope (m / s of $RT_{60}$)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    
    plt.title("Figure T11: Transition-Zone Zoom Analysis: Error vs. Local Degradation Slope")
    fig.tight_layout()
    fig.savefig(fig_dir / "T11_transition_zoom.png")
    plt.close(fig)

    logger.info("Successfully generated and saved all 11 Transition-Zone figures.")
