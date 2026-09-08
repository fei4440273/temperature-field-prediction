from __future__ import annotations

import numpy as np
import pytest

from sic_cu.models.residual_interpolation import (
    ChebyshevSurfaceResidualGuide,
    PowerResidualInterpolator,
    ResidualTrajectory,
    fit_chebyshev_surface_residual_guide,
)
import torch


def _trajectory(power_w: float, scale: float) -> ResidualTrajectory:
    return ResidualTrajectory(
        power_w=power_w,
        times_s=np.array([5.0, 10.0]),
        radii_m=np.array([0.0, 0.01]),
        residual_k=scale * np.array([[1.0, 2.0], [2.0, 4.0]]),
    )


def test_trajectory_enforces_zero_initial_residual_and_holds_final_frame() -> None:
    trajectory = _trajectory(100.0, 1.0)
    values = trajectory.evaluate(
        np.array([0.0, 2.5, 20.0]), np.array([0.0, 0.005, 0.01])
    )
    assert values == pytest.approx([0.0, 0.75, 4.0])


def test_power_normalized_interpolation_recovers_power_scaled_residual() -> None:
    model = PowerResidualInterpolator(
        [_trajectory(100.0, 1.0), _trajectory(300.0, 3.0)]
    )
    values = model.predict(
        200.0,
        np.array([10.0, 10.0]),
        np.array([0.0, 0.01]),
        "linear_residual_per_watt",
    )
    assert values == pytest.approx([4.0, 8.0])


def test_power_interpolator_rejects_unknown_method() -> None:
    model = PowerResidualInterpolator(
        [_trajectory(100.0, 1.0), _trajectory(300.0, 3.0)]
    )
    with pytest.raises(ValueError, match="Unsupported"):
        model.predict(200.0, np.array([5.0]), np.array([0.0]), "cubic")


def test_global_mean_and_regression_use_power_normalized_trajectories() -> None:
    model = PowerResidualInterpolator(
        [
            _trajectory(100.0, 1.0),
            _trajectory(200.0, 2.0),
            _trajectory(300.0, 3.0),
        ]
    )
    for method in ("mean_residual_per_watt", "regression_residual_per_watt"):
        value = model.predict(250.0, np.array([10.0]), np.array([0.01]), method)
        assert value == pytest.approx([10.0])


def test_chebyshev_surface_guide_is_serializable_and_differentiable() -> None:
    interpolator = PowerResidualInterpolator(
        [_trajectory(100.0, 1.0), _trajectory(300.0, 3.0)]
    )
    guide = fit_chebyshev_surface_residual_guide(
        interpolator,
        degree_r=2,
        degree_t=2,
        radial_samples=5,
        time_samples=5,
        radius_max_m=0.01,
        time_max_s=10.0,
    )
    restored = ChebyshevSurfaceResidualGuide.from_spec(guide.to_spec())
    coordinates = torch.tensor(
        [[0.005, 0.0, 10.0, 200.0, 1.0]], requires_grad=True
    )
    value = restored(coordinates)
    gradient = torch.autograd.grad(value.sum(), coordinates, create_graph=True)[0]
    assert value.item() == pytest.approx(6.0, abs=0.2)
    assert torch.isfinite(gradient).all()
