"""
EchoLocate — 2D Source Localization via TDOA Multilateration
=============================================================
Estimates the 2D (x, y) position of an acoustic source given:
  - Microphone positions in the horizontal plane
  - Estimated TDOAs between microphone pairs

Theory: Hyperbolic Localization
--------------------------------
Each TDOA measurement τ_ij between mics i and j defines a hyperbola:
    d_i - d_j = τ_ij · c
where d_i = ||source - mic_i|| and c = speed of sound.

The source lies at the intersection of these hyperbolas.
With 3 microphones (pairs: (0,1), (0,2), (1,2)), we have 3 TDOA equations
but only 2 unknowns (x, y). The system is over-determined → solved via
nonlinear least squares (NLS) minimisation.

Formulation:
    Residual for pair (i, j):
        f_ij(x, y) = (d_i(x,y) - d_j(x,y)) - τ_ij · c
    where d_k(x,y) = sqrt((x - x_k)^2 + (y - y_k)^2)

    Minimise: Σ_ij  f_ij(x, y)^2

This is solved by scipy.optimize.least_squares using the Levenberg–Marquardt
or Trust Region Reflective algorithm with bounds to constrain the solution
inside the room.

Initial Guess:
    The algorithm is sensitive to initial guess. We start from the centroid
    of the microphone array, which is always inside the convex hull of the mics.
    For robustness, multiple starting points can be tried.

Bounds:
    The room boundaries are used as hard constraints: 0 ≤ x ≤ Lx, 0 ≤ y ≤ Ly.
    This prevents the solver from returning physically impossible locations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)


@dataclass
class LocalizationResult:
    """
    Output of the 2D source localization solver.

    Attributes
    ----------
    predicted_position : np.ndarray
        Shape (2,). Estimated (x, y) in metres.
    true_position : np.ndarray or None
        Shape (2,). Ground truth (x, y) if available.
    localization_error : float or None
        Euclidean distance between true and predicted (metres), if true_position given.
    tdoa_residuals : np.ndarray
        Residuals at the solution (one per TDOA pair).
    converged : bool
        Whether the NLS solver converged successfully.
    solver_message : str
        Human-readable solver status message.
    """

    predicted_position: np.ndarray
    true_position: np.ndarray | None
    localization_error: float | None
    tdoa_residuals: np.ndarray
    converged: bool
    solver_message: str

    def __post_init__(self) -> None:
        if self.true_position is not None and self.localization_error is None:
            self.localization_error = float(
                np.linalg.norm(self.predicted_position - self.true_position)
            )


def _tdoa_residuals(
    xy: np.ndarray,
    mic_positions: np.ndarray,
    tdoa_values: np.ndarray,
    tdoa_pairs: list[tuple[int, int]],
    speed_of_sound: float,
) -> np.ndarray:
    """
    Compute residuals for the NLS hyperbolic TDOA system.

    For each pair (i, j):
        residual = (d_i - d_j) / c - τ̂_ij

    where d_k = Euclidean distance from candidate position xy to mic k.

    Parameters
    ----------
    xy : np.ndarray
        Shape (2,). Candidate source (x, y).
    mic_positions : np.ndarray
        Shape (n_mics, 2). Microphone positions.
    tdoa_values : np.ndarray
        Shape (n_pairs,). Estimated TDOAs in seconds.
    tdoa_pairs : list of (i, j)
        Microphone pair indices corresponding to tdoa_values.
    speed_of_sound : float
        Speed of sound in m/s.

    Returns
    -------
    np.ndarray
        Residuals, shape (n_pairs,).
    """
    x, y = xy
    residuals = []
    for (i, j), tau_ij in zip(tdoa_pairs, tdoa_values):
        d_i = np.sqrt((x - mic_positions[i, 0]) ** 2 + (y - mic_positions[i, 1]) ** 2)
        d_j = np.sqrt((x - mic_positions[j, 0]) ** 2 + (y - mic_positions[j, 1]) ** 2)
        # Predicted TDOA from geometry: (d_i - d_j) / c
        predicted_tdoa = (d_i - d_j) / speed_of_sound
        residuals.append(predicted_tdoa - tau_ij)
    return np.array(residuals)


def localize_source_2d(
    mic_positions_2d: np.ndarray,
    tdoa_dict: dict[tuple[int, int], float],
    speed_of_sound: float,
    room_dimensions: list[float],
    initial_guess: str | np.ndarray = "centroid",
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
    true_position: np.ndarray | None = None,
) -> LocalizationResult:
    """
    Estimate 2D source position from TDOA measurements using NLS.

    Parameters
    ----------
    mic_positions_2d : np.ndarray
        Shape (n_mics, 2). Mic (x, y) positions.
    tdoa_dict : dict
        Keys: (i, j) tuples. Values: estimated TDOA τ̂_ij in seconds.
    speed_of_sound : float
        Speed of sound in m/s.
    room_dimensions : list[float]
        [Lx, Ly, ...] room dimensions. Used to define bounds.
    initial_guess : str or np.ndarray
        'centroid' = start from the mean of mic positions.
        np.ndarray of shape (2,) = explicit starting point.
    max_iterations : int
        Maximum NLS iterations.
    tolerance : float
        Function-value tolerance for convergence.
    true_position : np.ndarray or None
        Ground truth (x, y) for error calculation.

    Returns
    -------
    LocalizationResult
    """
    n_mics = mic_positions_2d.shape[0]

    # --- Build sorted list of TDOA pairs and values -----------------------
    tdoa_pairs = sorted(tdoa_dict.keys())
    tdoa_values = np.array([tdoa_dict[k] for k in tdoa_pairs])

    if len(tdoa_pairs) < 2:
        raise ValueError(
            f"Need at least 2 TDOA measurements for 2D localization; got {len(tdoa_pairs)}."
        )

    # --- Initial guess ----------------------------------------------------
    if isinstance(initial_guess, str) and initial_guess == "centroid":
        x0 = mic_positions_2d.mean(axis=0)
    elif isinstance(initial_guess, np.ndarray):
        x0 = initial_guess.copy()
    else:
        raise ValueError(f"Unknown initial_guess type: {initial_guess}")

    logger.debug("NLS initial guess: (%.3f, %.3f)", x0[0], x0[1])

    # --- Define bounds (room boundaries) ----------------------------------
    Lx = room_dimensions[0]
    Ly = room_dimensions[1]
    lower_bounds = [0.0, 0.0]
    upper_bounds = [Lx, Ly]

    # --- Run NLS solver ---------------------------------------------------
    result = least_squares(
        fun=_tdoa_residuals,
        x0=x0,
        args=(mic_positions_2d, tdoa_values, tdoa_pairs, speed_of_sound),
        bounds=(lower_bounds, upper_bounds),
        method="trf",          # Trust Region Reflective — handles bounds
        max_nfev=max_iterations,
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
    )

    predicted = result.x.copy()
    converged = result.success or result.cost < 1e-6

    logger.info(
        "NLS result: predicted=(%.4f, %.4f), cost=%.4e, status=%d (%s)",
        predicted[0], predicted[1], result.cost, result.status, result.message,
    )

    # Compute localization error if ground truth is provided
    loc_error = None
    if true_position is not None:
        loc_error = float(np.linalg.norm(predicted - np.asarray(true_position)))
        logger.info("Localization error: %.4f m", loc_error)

    return LocalizationResult(
        predicted_position=predicted,
        true_position=np.asarray(true_position) if true_position is not None else None,
        localization_error=loc_error,
        tdoa_residuals=result.fun,
        converged=converged,
        solver_message=result.message,
    )


def localize_with_multiple_starts(
    mic_positions_2d: np.ndarray,
    tdoa_dict: dict[tuple[int, int], float],
    speed_of_sound: float,
    room_dimensions: list[float],
    n_restarts: int = 5,
    rng_seed: int = 42,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
    true_position: np.ndarray | None = None,
) -> LocalizationResult:
    """
    Run NLS localization with multiple random starting points and return the
    best result (lowest residual cost).

    This increases robustness against local minima in the cost landscape.

    Parameters
    ----------
    n_restarts : int
        Number of random starting points in addition to the centroid start.
    rng_seed : int
        Random seed for reproducible starting points.
    (other parameters same as localize_source_2d)

    Returns
    -------
    LocalizationResult
        The result with the lowest NLS cost.
    """
    rng = np.random.default_rng(rng_seed)
    Lx, Ly = room_dimensions[0], room_dimensions[1]

    # Always include centroid as first starting point
    starts = [mic_positions_2d.mean(axis=0)]
    # Add random starting points
    for _ in range(n_restarts):
        x0 = np.array([rng.uniform(0.1 * Lx, 0.9 * Lx), rng.uniform(0.1 * Ly, 0.9 * Ly)])
        starts.append(x0)

    best_result = None
    best_cost = np.inf

    for x0 in starts:
        try:
            res = localize_source_2d(
                mic_positions_2d=mic_positions_2d,
                tdoa_dict=tdoa_dict,
                speed_of_sound=speed_of_sound,
                room_dimensions=room_dimensions,
                initial_guess=x0,
                max_iterations=max_iterations,
                tolerance=tolerance,
                true_position=true_position,
            )
            cost = float(np.sum(res.tdoa_residuals ** 2))
            if cost < best_cost:
                best_cost = cost
                best_result = res
        except Exception as exc:
            logger.warning("NLS solver failed for start %s: %s", x0, exc)

    if best_result is None:
        raise RuntimeError("All NLS solver starts failed.")

    logger.info(
        "Best multi-start result: predicted=(%.4f, %.4f), cost=%.4e",
        best_result.predicted_position[0], best_result.predicted_position[1], best_cost,
    )
    return best_result
