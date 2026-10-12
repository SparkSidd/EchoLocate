"""
EchoLocate — Microphone Geometry Calibration Sensitivity Analysis
==================================================================
Evaluates localization accuracy under G1 (Nominal CAD), G2 (Measured physical),
G3 (Deliberately perturbed 0-20mm), and G4 (Estimated calibration) regimes.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_geometry_calibration_summary(
    nominal_errors: list[float] | np.ndarray,
    measured_errors: list[float] | np.ndarray,
    perturbed_errors_dict: dict[float, list[float]],
) -> pd.DataFrame:
    """Compile comparative performance across geometry calibration tiers."""
    records = []

    nom_arr = np.asarray(nominal_errors)
    meas_arr = np.asarray(measured_errors)

    records.append({
        "geometry_tier": "G1_Nominal_CAD",
        "mean_error_m": float(np.mean(nom_arr)),
        "median_error_m": float(np.median(nom_arr)),
        "rmse_m": float(np.sqrt(np.mean(nom_arr**2))),
        "p95_m": float(np.percentile(nom_arr, 95)),
    })

    records.append({
        "geometry_tier": "G2_Physically_Measured",
        "mean_error_m": float(np.mean(meas_arr)),
        "median_error_m": float(np.median(meas_arr)),
        "rmse_m": float(np.sqrt(np.mean(meas_arr**2))),
        "p95_m": float(np.percentile(meas_arr, 95)),
    })

    for pert_mm, errs in perturbed_errors_dict.items():
        e_arr = np.asarray(errs)
        records.append({
            "geometry_tier": f"G3_Perturbed_{pert_mm:.1f}mm",
            "mean_error_m": float(np.mean(e_arr)),
            "median_error_m": float(np.median(e_arr)),
            "rmse_m": float(np.sqrt(np.mean(e_arr**2))),
            "p95_m": float(np.percentile(e_arr, 95)),
        })

    return pd.DataFrame(records)
