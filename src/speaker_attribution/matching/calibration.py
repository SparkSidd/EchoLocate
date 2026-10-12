"""
EchoLocate — Threshold Calibration & Verification Metrics
=========================================================
Calculates EER, FAR, FRR, ROC, and optimal decision boundaries
from genuine and impostor verification score distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CalibrationReport:
    """Quantitative threshold calibration analysis results."""
    optimal_threshold: float
    eer: float
    min_dcf: float
    far_at_threshold: float
    frr_at_threshold: float
    threshold_at_1pct_far: float
    threshold_at_1pct_frr: float
    auc_roc: float
    total_genuine_trials: int
    total_impostor_trials: int
    curves: Dict[str, List[float]]


class ThresholdCalibrator:
    """
    Calibrates verification decision boundaries using validation trial pairs.
    Prevents empirical guess-work by deriving data-driven thresholds.
    """

    def __init__(self, num_points: int = 500):
        self.num_points = num_points

    def calibrate(
        self,
        scores: np.ndarray | List[float],
        labels: np.ndarray | List[bool | int],
    ) -> CalibrationReport:
        """
        Calibrate threshold on score-label pairs.
        
        Args:
            scores: 1D array of cosine similarities
            labels: 1D array of booleans (True = Genuine / Same Speaker, False = Impostor / Different)
        """
        scores = np.asarray(scores, dtype=np.float32)
        labels = np.asarray(labels, dtype=bool)

        if len(scores) != len(labels) or len(scores) == 0:
            raise ValueError("Scores and labels must be non-empty and of identical length")

        genuine_scores = scores[labels]
        impostor_scores = scores[~labels]

        if len(genuine_scores) == 0 or len(impostor_scores) == 0:
            # Fallback if only one class exists
            return CalibrationReport(
                optimal_threshold=0.65,
                eer=0.0,
                min_dcf=0.0,
                far_at_threshold=0.0,
                frr_at_threshold=0.0,
                threshold_at_1pct_far=0.65,
                threshold_at_1pct_frr=0.65,
                auc_roc=1.0,
                total_genuine_trials=len(genuine_scores),
                total_impostor_trials=len(impostor_scores),
                curves={},
            )

        min_score = min(float(np.min(scores)), -1.0)
        max_score = max(float(np.max(scores)), 1.0)
        thresholds = np.linspace(min_score, max_score, self.num_points)

        far_list = []
        frr_list = []

        num_gen = len(genuine_scores)
        num_imp = len(impostor_scores)

        for th in thresholds:
            # FAR: Impostors accepted (score >= th)
            far = np.sum(impostor_scores >= th) / num_imp
            # FRR: Genuine rejected (score < th)
            frr = np.sum(genuine_scores < th) / num_gen
            far_list.append(float(far))
            frr_list.append(float(frr))

        far_arr = np.array(far_list)
        frr_arr = np.array(frr_list)

        # Equal Error Rate (EER) where |FAR - FRR| is minimal
        diff = np.abs(far_arr - frr_arr)
        eer_idx = int(np.argmin(diff))
        eer = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0)
        optimal_th = float(thresholds[eer_idx])

        # Min Detection Cost Function (minDCF with C_miss=1, C_fa=1, P_target=0.1)
        p_target = 0.1
        c_miss = 1.0
        c_fa = 1.0
        dcf = c_miss * p_target * frr_arr + c_fa * (1 - p_target) * far_arr
        min_dcf = float(np.min(dcf))

        # Operating points
        idx_1pct_far = np.where(far_arr <= 0.01)[0]
        th_1pct_far = float(thresholds[idx_1pct_far[0]]) if len(idx_1pct_far) > 0 else optimal_th

        idx_1pct_frr = np.where(frr_arr <= 0.01)[0]
        th_1pct_frr = float(thresholds[idx_1pct_frr[-1]]) if len(idx_1pct_frr) > 0 else optimal_th

        # AUC ROC
        # Trapezoidal numerical integration of ROC curve (TPR = 1 - FRR, FPR = FAR)
        tpr = 1.0 - frr_arr
        # Sort by FAR for clean integration
        sort_order = np.argsort(far_arr)
        auc = float(np.trapezoid(tpr[sort_order], far_arr[sort_order]))

        return CalibrationReport(
            optimal_threshold=optimal_th,
            eer=eer,
            min_dcf=min_dcf,
            far_at_threshold=float(far_arr[eer_idx]),
            frr_at_threshold=float(frr_arr[eer_idx]),
            threshold_at_1pct_far=th_1pct_far,
            threshold_at_1pct_frr=th_1pct_frr,
            auc_roc=max(0.0, min(1.0, auc)),
            total_genuine_trials=num_gen,
            total_impostor_trials=num_imp,
            curves={
                "thresholds": [float(t) for t in thresholds],
                "far": [float(f) for f in far_arr],
                "frr": [float(f) for f in frr_arr],
            },
        )
