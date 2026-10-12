"""
EchoLocate — Tests: Simulation Module
=======================================
Tests for the room simulator, source generator, and mic array builder.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulation.source_generator import generate_chirp, generate_click, generate_source_signal
from src.simulation.room_simulator import build_mic_array_3d, SimulationResult


class TestSourceGenerator:
    FS = 16000

    def test_chirp_length(self):
        """Chirp total length should equal pad_before + chirp + pad_after."""
        chirp = generate_chirp(
            fs=self.FS,
            duration=0.005,
            f_low=300, f_high=7000,
            pad_before=0.01, pad_after=0.1,
        )
        expected = int(0.01 * self.FS) + int(0.005 * self.FS) + int(0.1 * self.FS)
        assert len(chirp) == expected, f"Length {len(chirp)} ≠ {expected}"

    def test_chirp_amplitude(self):
        """Chirp peak amplitude should not exceed specified amplitude."""
        amp = 0.5
        chirp = generate_chirp(self.FS, 0.01, 100, 7000, amplitude=amp)
        assert np.max(np.abs(chirp)) <= amp + 1e-6

    def test_click_has_single_nonzero(self):
        """Click signal should have exactly one non-zero sample."""
        click = generate_click(self.FS, pad_before=0.01, total_duration=0.1)
        nonzero = np.sum(click != 0)
        assert nonzero == 1

    def test_generate_source_signal_chirp(self):
        """generate_source_signal should return float64 array."""
        flat_cfg = {
            "sampling_rate": self.FS,
            "source": {
                "type": "chirp",
                "duration": 0.005,
                "f_low": 300,
                "f_high": 7000,
                "amplitude": 1.0,
                "pad_before": 0.01,
                "pad_after": 0.1,
            },
        }
        signal, fs = generate_source_signal(flat_cfg)
        assert signal.dtype == np.float64
        assert fs == self.FS
        assert len(signal) > 0

    def test_unknown_source_type_raises(self):
        """Unknown source type should raise ValueError."""
        flat_cfg = {
            "sampling_rate": self.FS,
            "source": {"type": "foobar", "duration": 0.01,
                       "f_low": 100, "f_high": 5000,
                       "pad_before": 0.01, "pad_after": 0.1},
        }
        with pytest.raises(ValueError, match="Unknown source type"):
            generate_source_signal(flat_cfg)


class TestMicArrayBuilder:
    def test_build_three_mics(self):
        """build_mic_array_3d should produce (3, 3) array for 3 mics."""
        positions_2d = {
            "M1": [1.0, 1.5],
            "M2": [9.0, 1.5],
            "M3": [5.0, 4.5],
        }
        mic_array, labels = build_mic_array_3d(positions_2d, height=1.2)
        assert mic_array.shape == (3, 3), f"Expected (3,3), got {mic_array.shape}"
        assert labels == ["M1", "M2", "M3"]

    def test_height_set_correctly(self):
        """All mic z-coordinates should equal the specified height."""
        positions_2d = {"M1": [1.0, 2.0], "M2": [5.0, 3.0]}
        height = 1.5
        mic_array, _ = build_mic_array_3d(positions_2d, height=height)
        assert np.allclose(mic_array[2, :], height), (
            f"Heights {mic_array[2, :]} ≠ {height}"
        )

    def test_x_y_positions_preserved(self):
        """X and Y coordinates should match input."""
        positions_2d = {"M1": [2.0, 3.0], "M2": [7.0, 4.0]}
        mic_array, labels = build_mic_array_3d(positions_2d, height=1.0)
        assert np.isclose(mic_array[0, 0], 2.0)  # M1 x
        assert np.isclose(mic_array[1, 0], 3.0)  # M1 y
        assert np.isclose(mic_array[0, 1], 7.0)  # M2 x
        assert np.isclose(mic_array[1, 1], 4.0)  # M2 y


class TestSimulationResult:
    """Test SimulationResult helper methods."""

    def _make_result(self):
        """Create a mock SimulationResult."""
        # Source at (5, 3), mics at (1, 1.5), (9, 1.5), (5, 4.5)
        mic_3d = np.array([[1.0, 9.0, 5.0],
                            [1.5, 1.5, 4.5],
                            [1.2, 1.2, 1.2]])
        return SimulationResult(
            mic_signals=np.zeros((3, 1000)),
            mic_positions_3d=mic_3d,
            source_position_2d=np.array([5.0, 3.0]),
            source_signal=np.zeros(100),
            fs=16000,
            speed_of_sound=343.0,
            stage="anechoic",
            room_dimensions=[10.0, 6.0, 3.0],
        )

    def test_mic_position_2d(self):
        result = self._make_result()
        pos = result.mic_position_2d(0)
        assert np.allclose(pos, [1.0, 1.5])

    def test_all_mic_positions_2d_shape(self):
        result = self._make_result()
        positions = result.all_mic_positions_2d()
        assert positions.shape == (3, 2)

    def test_theoretical_tdoa_sign(self):
        """
        With source at (5,3):
          d_M1 = dist((5,3),(1,1.5)) = sqrt(16+2.25) = sqrt(18.25)
          d_M2 = dist((5,3),(9,1.5)) = sqrt(16+2.25) = sqrt(18.25)
          TDOA_12 = (d_M1 - d_M2) / c = 0 (symmetric)
        """
        result = self._make_result()
        tdoa_01 = result.theoretical_tdoa(0, 1)
        # Symmetric → should be near 0
        assert abs(tdoa_01) < 1e-9, f"Expected ~0, got {tdoa_01}"

    def test_theoretical_tdoa_nonzero(self):
        """
        Source closer to M1 → d_M1 < d_M2 → TDOA_12 < 0.
        """
        result = self._make_result()
        # Move source close to M1 at (1, 1.5)
        result.source_position_2d = np.array([1.5, 1.5])
        tdoa_01 = result.theoretical_tdoa(0, 1)
        # d_M1 < d_M2 → (d_M1 - d_M2) < 0 → TDOA < 0
        assert tdoa_01 < 0, f"Expected negative TDOA, got {tdoa_01}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
