"""
EchoLocate — Causal Temporal Smoothing & Trajectory Tracker
===========================================================
Implements causal exponential smoothing and 2D Kalman trajectory tracking
with strict isolation against future-frame lookahead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackedEstimate:
    """Smoothed localization output and tracking state."""
    raw_x: float
    raw_y: float
    smoothed_x: float
    smoothed_y: float
    velocity_x: float
    velocity_y: float
    tracking_mode: str  # "RAW", "CAUSAL_SMOOTHED", "KALMAN"
    is_causal: bool = True


class CausalTemporalTracker:
    """Strictly causal online trajectory tracking filter."""

    def __init__(
        self,
        mode: Literal["RAW", "CAUSAL_SMOOTHED", "KALMAN"] = "CAUSAL_SMOOTHED",
        alpha: float = 0.40,
        jump_rejection_m: float = 2.50,
    ) -> None:
        self.mode = mode
        self.alpha = alpha
        self.jump_rejection_m = jump_rejection_m
        self.state_x: float | None = None
        self.state_y: float | None = None
        self.vel_x: float = 0.0
        self.vel_y: float = 0.0

        # Simple 2D Kalman state: [x, y, vx, vy]^T
        self.kalman_state = np.zeros(4, dtype=np.float64)
        self.kalman_cov = np.eye(4, dtype=np.float64) * 0.5
        self.is_initialized = False

    def reset(self) -> None:
        """Reset internal filter states for deterministic replay."""
        self.state_x = None
        self.state_y = None
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.kalman_state = np.zeros(4, dtype=np.float64)
        self.kalman_cov = np.eye(4, dtype=np.float64) * 0.5
        self.is_initialized = False

    def update(self, raw_x: float, raw_y: float, confidence: float = 1.0) -> TrackedEstimate:
        """Update tracker state with new frame estimate strictly causally."""
        if not self.is_initialized or self.state_x is None or self.state_y is None:
            self.state_x = raw_x
            self.state_y = raw_y
            self.kalman_state[:2] = [raw_x, raw_y]
            self.is_initialized = True
            return TrackedEstimate(
                raw_x=raw_x, raw_y=raw_y,
                smoothed_x=raw_x, smoothed_y=raw_y,
                velocity_x=0.0, velocity_y=0.0,
                tracking_mode=self.mode, is_causal=True,
            )

        if self.mode == "RAW":
            return TrackedEstimate(
                raw_x=raw_x, raw_y=raw_y,
                smoothed_x=raw_x, smoothed_y=raw_y,
                velocity_x=0.0, velocity_y=0.0,
                tracking_mode="RAW", is_causal=True,
            )

        # Check for sudden spatial reflection jump
        dist = float(np.hypot(raw_x - self.state_x, raw_y - self.state_y))
        effective_alpha = self.alpha * max(0.2, confidence)

        if dist > self.jump_rejection_m and confidence < 0.65:
            # Dampen sudden low-confidence jump
            effective_alpha *= 0.15

        if self.mode == "CAUSAL_SMOOTHED":
            new_x = float(effective_alpha * raw_x + (1.0 - effective_alpha) * self.state_x)
            new_y = float(effective_alpha * raw_y + (1.0 - effective_alpha) * self.state_y)
            self.vel_x = new_x - self.state_x
            self.vel_y = new_y - self.state_y
            self.state_x = new_x
            self.state_y = new_y

            return TrackedEstimate(
                raw_x=raw_x, raw_y=raw_y,
                smoothed_x=new_x, smoothed_y=new_y,
                velocity_x=self.vel_x, velocity_y=self.vel_y,
                tracking_mode="CAUSAL_SMOOTHED", is_causal=True,
            )

        elif self.mode == "KALMAN":
            # Constant-velocity 2D Kalman step
            dt = 0.20  # approx frame duration
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float64)
            Q = np.eye(4) * 0.01
            R = np.eye(2) * (0.10 / max(0.1, confidence))

            # Predict
            pred_state = F @ self.kalman_state
            pred_cov = F @ self.kalman_cov @ F.T + Q

            # Measurement Update
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
            z = np.array([raw_x, raw_y])
            y = z - H @ pred_state
            S = H @ pred_cov @ H.T + R
            K = pred_cov @ H.T @ np.linalg.inv(S)

            self.kalman_state = pred_state + K @ y
            self.kalman_cov = (np.eye(4) - K @ H) @ pred_cov
            self.state_x = float(self.kalman_state[0])
            self.state_y = float(self.kalman_state[1])
            self.vel_x = float(self.kalman_state[2])
            self.vel_y = float(self.kalman_state[3])

            return TrackedEstimate(
                raw_x=raw_x, raw_y=raw_y,
                smoothed_x=self.state_x, smoothed_y=self.state_y,
                velocity_x=self.vel_x, velocity_y=self.vel_y,
                tracking_mode="KALMAN", is_causal=True,
            )

        return TrackedEstimate(
            raw_x=raw_x, raw_y=raw_y,
            smoothed_x=raw_x, smoothed_y=raw_y,
            velocity_x=0.0, velocity_y=0.0,
            tracking_mode="RAW", is_causal=True,
        )
