"""
Unit tests for Experiment 4.5: Neural Likelihood Calibration & Spatial Training.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.localization_objective_analysis import (
    audit_dataset_split_isolation,
    compute_paired_statistical_comparison,
)
from src.evaluation.neural_likelihood_analysis import compute_fwhm, compute_local_curvature
from src.evaluation.spatial_contrast_analysis import analyze_spatial_likelihood_surface
from src.models.deepgcc_calibrated import CalibratedDeepGCC, apply_temperature_sharpening
from src.models.differentiable_srp import DifferentiableSRPLayer
from src.training.likelihood_losses import (
    compute_delay_kl_loss,
    compute_hybrid_experiment_04_5_loss,
    compute_peak_contrast_loss,
    compute_spatial_nll_loss,
)


class TestCalibratedDeepGCC:
    def test_calibrated_model_forward_and_learned_temperature(self):
        model = CalibratedDeepGCC(
            in_channels=1, hidden_channels=32, num_res_blocks=2, enable_learned_temperature=True
        )
        model.eval()

        dummy_input = torch.randn(3, 1, 129)
        with torch.no_grad():
            probs, aux = model(dummy_input)

        assert probs.shape == (3, 129)
        assert torch.all(probs >= 0.0)
        assert torch.allclose(torch.sum(probs, dim=-1), torch.ones(3), atol=1e-4)
        assert "temperature" in aux
        assert "confidence" in aux

    def test_temperature_sharpening_effect(self):
        base_probs = torch.tensor([[0.1, 0.2, 0.4, 0.2, 0.1]])
        sharp_probs = apply_temperature_sharpening(base_probs, temperature=0.25)

        assert torch.allclose(torch.sum(sharp_probs, dim=-1), torch.ones(1), atol=1e-4)
        # Peak probability should increase under T=0.25 sharpening
        assert float(sharp_probs[0, 2]) > float(base_probs[0, 2])


class TestDifferentiableSRPLayer:
    def test_differentiable_srp_forward_and_gradient_flow(self):
        srp = DifferentiableSRPLayer(
            room_dimensions=[10.0, 6.0, 3.0],
            mic_positions_2d=[[2.0, 1.5], [8.0, 1.5], [5.0, 4.5]],
            grid_resolution_m=0.20,  # Coarser grid for fast unit test
            num_tau_bins=129,
        )

        logits = torch.randn(2, 3, 129, requires_grad=True)
        input_likelihoods = torch.softmax(logits, dim=-1)
        spatial_map, spatial_prob, pred_xy = srp(input_likelihoods)

        assert spatial_map.shape == (2, srp.n_grid)
        assert spatial_prob.shape == (2, srp.n_grid)
        assert pred_xy.shape == (2, 2)

        # Test gradient flow back to input
        loss = torch.sum(pred_xy)
        loss.backward()
        assert logits.grad is not None
        assert torch.all(torch.isfinite(logits.grad))


class TestLikelihoodLosses:
    def test_peak_contrast_loss_margin(self):
        logits = torch.tensor([[5.0, 1.0, 0.0, 4.8, 0.0]])  # true=5.0, competing=4.8, diff=0.2
        tau_axis = torch.tensor([-0.02, -0.01, 0.0, 0.01, 0.02])
        true_tdoa = torch.tensor([-0.02])

        loss_loose = compute_peak_contrast_loss(logits, true_tdoa, tau_axis, margin=0.10)
        loss_strict = compute_peak_contrast_loss(logits, true_tdoa, tau_axis, margin=0.50)

        assert float(loss_loose) == 0.0  # diff 0.2 >= margin 0.10 -> 0 loss
        assert float(loss_strict) > 0.0  # diff 0.2 < margin 0.50 -> penalty > 0

    def test_fwhm_calculation(self):
        tau_axis = np.linspace(-0.010, 0.010, 201)
        # Gaussian with sigma = 0.001s -> theoretical FWHM = 2 * sqrt(2 * ln(2)) * sigma approx 2.355 ms
        curve = np.exp(-(tau_axis**2) / (2 * (0.001**2)))
        fwhm_s = compute_fwhm(curve, tau_axis)
        assert 0.0020 <= fwhm_s <= 0.0028


class TestSpatialAnalysisAndAudit:
    def test_spatial_likelihood_surface_metrics(self):
        grid_pts = np.array([[2.0, 3.0], [5.0, 3.0], [8.0, 3.0]])
        spatial_map = np.array([0.1, 0.9, 0.2])
        true_xy = np.array([5.0, 3.0])

        res = analyze_spatial_likelihood_surface(spatial_map, grid_pts, true_xy)
        assert res["true_spatial_rank"] == 1.0
        assert res["spatial_peak_contrast"] > 1.0
        assert res["localization_error_m"] == 0.0

    def test_dataset_split_audit(self):
        df = audit_dataset_split_isolation()
        assert len(df) >= 4
        assert (df["Isolation Status"] == "STRICT_PASS").all()
