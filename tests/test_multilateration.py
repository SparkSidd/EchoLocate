"""
EchoLocate — Tests: Multilateration Localization
==================================================
Tests for the 2D source localization solver.

Verified properties:
1. Solver recovers known positions from exact (noise-free) TDOAs
2. Room boundary constraints are respected
3. Multiple starts improve robustness
4. Residuals are near-zero at the true solution (with exact TDOAs)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.localization.multilateration import localize_source_2d, localize_with_multiple_starts


# Room and microphone configuration matching experiment_01.yaml
ROOM_DIMS = [10.0, 6.0, 3.0]
SPEED_OF_SOUND = 343.0
MIC_POSITIONS = np.array([
    [1.0, 1.5],  # M1
    [9.0, 1.5],  # M2
    [5.0, 4.5],  # M3
])


def compute_exact_tdoas(source_pos: np.ndarray) -> dict:
    """Compute exact (noise-free) TDOAs from geometry."""
    tdoa_dict = {}
    n_mics = MIC_POSITIONS.shape[0]
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            d_i = np.linalg.norm(source_pos - MIC_POSITIONS[i])
            d_j = np.linalg.norm(source_pos - MIC_POSITIONS[j])
            tdoa_dict[(i, j)] = (d_i - d_j) / SPEED_OF_SOUND
    return tdoa_dict


class TestLocalizationAccuracy:
    """Test that the solver recovers known positions from exact TDOAs."""

    TOLERANCE_M = 0.05  # Allow up to 5 cm error for NLS with exact TDOAs

    @pytest.mark.parametrize("source_pos", [
        [5.0, 3.0],   # Centre
        [2.0, 1.0],   # Near M1
        [8.0, 5.0],   # Far corner
        [5.0, 1.5],   # Midway between M1 and M2 (on mic baseline)
        [1.5, 4.5],   # Near M3
    ])
    def test_known_positions_exact_tdoa(self, source_pos):
        """With exact TDOAs, localization error should be < 5 cm."""
        src = np.array(source_pos)
        tdoa_dict = compute_exact_tdoas(src)

        result = localize_source_2d(
            mic_positions_2d=MIC_POSITIONS,
            tdoa_dict=tdoa_dict,
            speed_of_sound=SPEED_OF_SOUND,
            room_dimensions=ROOM_DIMS,
            true_position=src,
        )

        assert result.localization_error is not None
        assert result.localization_error < self.TOLERANCE_M, (
            f"Error {result.localization_error:.4f} m exceeds {self.TOLERANCE_M} m "
            f"for source at {source_pos}"
        )

    def test_centre_of_room(self):
        """Centre of room should localize with very low error."""
        src = np.array([5.0, 3.0])
        tdoa_dict = compute_exact_tdoas(src)
        result = localize_source_2d(
            mic_positions_2d=MIC_POSITIONS,
            tdoa_dict=tdoa_dict,
            speed_of_sound=SPEED_OF_SOUND,
            room_dimensions=ROOM_DIMS,
            true_position=src,
        )
        assert result.localization_error < self.TOLERANCE_M

    def test_predicted_position_inside_room(self):
        """Predicted position should always be inside the room."""
        for source_pos in [[2.0, 1.0], [8.0, 5.0], [5.0, 3.0]]:
            src = np.array(source_pos)
            tdoa_dict = compute_exact_tdoas(src)
            result = localize_source_2d(
                mic_positions_2d=MIC_POSITIONS,
                tdoa_dict=tdoa_dict,
                speed_of_sound=SPEED_OF_SOUND,
                room_dimensions=ROOM_DIMS,
            )
            x, y = result.predicted_position
            assert 0.0 <= x <= ROOM_DIMS[0], f"x={x} outside room"
            assert 0.0 <= y <= ROOM_DIMS[1], f"y={y} outside room"


class TestLocalizationEdgeCases:
    def test_residuals_near_zero_at_true_position(self):
        """At the true position, TDOA residuals should be essentially zero."""
        src = np.array([4.0, 2.5])
        tdoa_dict = compute_exact_tdoas(src)
        result = localize_source_2d(
            mic_positions_2d=MIC_POSITIONS,
            tdoa_dict=tdoa_dict,
            speed_of_sound=SPEED_OF_SOUND,
            room_dimensions=ROOM_DIMS,
            true_position=src,
        )
        # Residuals at the solution should be small
        max_residual = float(np.max(np.abs(result.tdoa_residuals)))
        assert max_residual < 0.01, f"Max residual {max_residual} too large"

    def test_multi_start_matches_single_start(self):
        """Multi-start should achieve at least as good a result as single start."""
        src = np.array([3.0, 4.0])
        tdoa_dict = compute_exact_tdoas(src)

        single = localize_source_2d(
            mic_positions_2d=MIC_POSITIONS,
            tdoa_dict=tdoa_dict,
            speed_of_sound=SPEED_OF_SOUND,
            room_dimensions=ROOM_DIMS,
            true_position=src,
        )
        multi = localize_with_multiple_starts(
            mic_positions_2d=MIC_POSITIONS,
            tdoa_dict=tdoa_dict,
            speed_of_sound=SPEED_OF_SOUND,
            room_dimensions=ROOM_DIMS,
            n_restarts=5,
            true_position=src,
        )
        # Multi-start should not be worse
        assert multi.localization_error <= single.localization_error + 0.01

    def test_insufficient_pairs_raises(self):
        """Fewer than 2 TDOA pairs should raise ValueError."""
        src = np.array([5.0, 3.0])
        tdoa_dict = compute_exact_tdoas(src)
        single_pair_dict = {(0, 1): tdoa_dict[(0, 1)]}

        with pytest.raises(ValueError):
            localize_source_2d(
                mic_positions_2d=MIC_POSITIONS,
                tdoa_dict=single_pair_dict,
                speed_of_sound=SPEED_OF_SOUND,
                room_dimensions=ROOM_DIMS,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
