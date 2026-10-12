"""
EchoLocate — Tests: GCC-PHAT Validation
=========================================
Unit and integration tests for the GCC-PHAT TDOA estimator.

These tests verify:
1. _next_power_of_2 helper
2. gcc_phat on synthetic signals with known delays
3. Sign convention: positive TDOA when signal arrives earlier at mic j
4. estimate_all_tdoas for 3 microphones
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.localization.gcc_phat import gcc_phat, estimate_all_tdoas, _next_power_of_2


class TestNextPowerOf2:
    def test_exact_powers(self):
        assert _next_power_of_2(1) == 1
        assert _next_power_of_2(2) == 2
        assert _next_power_of_2(4) == 4
        assert _next_power_of_2(1024) == 1024

    def test_non_powers(self):
        assert _next_power_of_2(3) == 4
        assert _next_power_of_2(5) == 8
        assert _next_power_of_2(100) == 128
        assert _next_power_of_2(1000) == 1024

    def test_zero_and_negative(self):
        assert _next_power_of_2(0) == 1
        assert _next_power_of_2(-5) == 1


class TestGccPhat:
    """Test GCC-PHAT on synthetic signals with known integer sample delays."""

    FS = 16000

    def _make_delayed_pair(self, delay_samples: int, n: int = 4000):
        """
        Create two signals with a known relative delay.

        Convention: TDOA_ij = (d_i - d_j) / c
          - If delay_samples > 0: x_i arrives delay_samples LATER than x_j
            → d_i > d_j → TDOA_ij > 0
          - If delay_samples < 0: x_j arrives |delay_samples| LATER than x_i
            → d_j > d_i → TDOA_ij < 0

        We construct this as:
          - delay_samples > 0: x_i = [zeros, src], x_j = [src, zeros]
            (x_i has a leading zero-pad, so it's delayed)
          - delay_samples < 0: x_j is delayed instead
        """
        rng = np.random.default_rng(0)
        src = rng.standard_normal(n)
        abs_d = abs(delay_samples)
        if delay_samples >= 0:
            # x_i is the delayed signal (arrives later → d_i > d_j → TDOA > 0)
            x_i = np.concatenate([np.zeros(abs_d), src[:n - abs_d]])
            x_j = src.copy()
        else:
            # x_j is the delayed signal (arrives later → d_j > d_i → TDOA < 0)
            x_i = src.copy()
            x_j = np.concatenate([np.zeros(abs_d), src[:n - abs_d]])
        return x_i, x_j

    def test_zero_delay(self):
        """Identical signals should give zero TDOA."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal(2000)
        result = gcc_phat(x, x.copy(), fs=self.FS, mic_i=0, mic_j=1)
        assert abs(result.tau_samples) <= 1.0, f"Expected ~0 samples, got {result.tau_samples}"

    def test_positive_delay_detection(self):
        """
        x_i is delayed by 40 samples → sound arrives at mic j before mic i
        → TDOA_ij = (d_i - d_j)/c > 0 → tau_samples should be +40.
        """
        delay = 40
        x_i, x_j = self._make_delayed_pair(delay_samples=delay)
        result = gcc_phat(x_i, x_j, fs=self.FS, mic_i=0, mic_j=1)
        # Allow ±1 sample tolerance (integer resolution)
        assert abs(result.tau_samples - delay) <= 1.0, (
            f"Expected +{delay} samples, got {result.tau_samples}"
        )

    def test_negative_delay_detection(self):
        """
        x_j is delayed by 30 samples → TDOA_ij < 0 (j further from source).
        """
        delay = -30
        x_i, x_j = self._make_delayed_pair(delay_samples=delay)
        result = gcc_phat(x_i, x_j, fs=self.FS, mic_i=0, mic_j=1)
        assert abs(result.tau_samples - delay) <= 1.0, (
            f"Expected {delay} samples, got {result.tau_samples}"
        )

    def test_tau_seconds_consistent_with_samples(self):
        """tau_seconds should equal tau_samples / fs."""
        x_i, x_j = self._make_delayed_pair(delay_samples=20)
        result = gcc_phat(x_i, x_j, fs=self.FS)
        assert abs(result.tau_seconds - result.tau_samples / self.FS) < 1e-9

    def test_output_shapes(self):
        """Check that output arrays have expected shapes."""
        n = 2000
        x_i = np.random.randn(n)
        x_j = np.random.randn(n)
        result = gcc_phat(x_i, x_j, fs=self.FS, fft_size=8192)
        assert result.gcc_phat_response.shape == (8192,)
        assert result.tau_axis_seconds.shape == (8192,)

    def test_max_tau_constraint(self):
        """With max_tau_samples constraint, peak must be within the window."""
        delay = 50
        x_i, x_j = self._make_delayed_pair(delay_samples=delay)
        # Set max_tau to 100 samples (wider than the actual delay)
        result = gcc_phat(x_i, x_j, fs=self.FS, max_tau_samples=100, mic_i=0, mic_j=1)
        assert abs(result.tau_samples) <= 100, (
            f"Peak outside max_tau window: {result.tau_samples}"
        )


class TestEstimateAllTdoas:
    FS = 16000

    def test_three_mic_all_pairs(self):
        """estimate_all_tdoas should return 3 pairs for 3-mic array."""
        rng = np.random.default_rng(5)
        mic_signals = rng.standard_normal((3, 3000))
        results = estimate_all_tdoas(mic_signals, fs=self.FS)
        assert set(results.keys()) == {(0, 1), (0, 2), (1, 2)}

    def test_custom_pairs(self):
        """Only requested pairs should be computed."""
        rng = np.random.default_rng(6)
        mic_signals = rng.standard_normal((3, 3000))
        results = estimate_all_tdoas(mic_signals, fs=self.FS, pairs=[(0, 1)])
        assert list(results.keys()) == [(0, 1)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
