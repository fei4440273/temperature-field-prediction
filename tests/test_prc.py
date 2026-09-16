from __future__ import annotations

import numpy as np
import pytest
import torch

from sic_cu.models import ModelScales, PRCMultifidelityModel
from sic_cu.prediction import Predictor
from sic_cu.train.prc import train_prc_multifidelity


def _coordinates(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0, 400.0, 1.0],
            [0.01, -0.006, 20.0, 400.0, 1.0],
            [0.028, -0.0175, 50.0, 300.0, 0.0],
            [0.05834, -0.010, 100.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )


def test_prc_training_entry_point_is_importable() -> None:
    assert callable(train_prc_multifidelity)


@pytest.mark.parametrize(
    "variant", ["temperature_residual", "amplitude", "amplitude_time"]
)
def test_prc_zero_correction_is_exact_lf_and_anchors_initial_state(variant: str) -> None:
    model = PRCMultifidelityModel(
        modes=4, width=8, depth=2, correction_variant=variant
    ).double()
    coordinates = _coordinates()

    low = model(coordinates, fidelity="low")
    high = model(coordinates, fidelity="high")

    assert low.shape == high.shape == (4, 1)
    assert torch.equal(low, high)
    assert high[0].item() == pytest.approx(295.15)
    assert high[3].item() == pytest.approx(295.15)


def test_prc_tau_is_positive_ordered_and_finite_at_long_time() -> None:
    model = PRCMultifidelityModel(modes=8, width=8, depth=2).double()
    coordinates = _coordinates()
    _, tau = model.response_parameters(coordinates, "high")
    long_time = coordinates.clone()
    long_time[:, 2] = 10_000.0

    assert torch.all(tau > 0.0)
    assert torch.all(torch.diff(tau, dim=1) > 0.0)
    assert torch.isfinite(model(long_time)).all()


def test_prc_analytic_time_derivative_matches_autograd() -> None:
    torch.manual_seed(4)
    model = PRCMultifidelityModel(
        modes=4, width=8, depth=2, correction_variant="amplitude_time"
    ).double()
    for parameter in model.parameters():
        if parameter.requires_grad:
            torch.nn.init.uniform_(parameter, -0.05, 0.05)
    coordinates = _coordinates().clone().requires_grad_(True)
    output = model(coordinates)
    automatic = torch.autograd.grad(output.sum(), coordinates, create_graph=True)[0][
        :, 2:3
    ]
    analytic = model.analytic_time_derivative(coordinates)

    assert torch.allclose(automatic, analytic, rtol=1e-7, atol=1e-8)


def test_prc_frozen_lf_keeps_first_and_second_coordinate_derivatives() -> None:
    model = PRCMultifidelityModel(modes=4, width=8, depth=2).double()
    model.freeze_low_fidelity(True)
    coordinates = _coordinates()[1:3].clone().requires_grad_(True)
    low = model(coordinates, fidelity="low")
    first = torch.autograd.grad(low.sum(), coordinates, create_graph=True)[0]
    second = torch.autograd.grad(first[:, 0].sum(), coordinates, create_graph=True)[0]

    assert not any(parameter.requires_grad for parameter in model.low_fidelity_parameters())
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()


def test_prc_copper_outer_boundary_has_continuous_inner_limit() -> None:
    model = PRCMultifidelityModel(modes=4, width=8, depth=2).double()
    with torch.no_grad():
        model.lf_amplitude[-1].bias.fill_(1.0)
    boundary = torch.tensor([[0.05834, -0.01, 50.0, 400.0, 0.0]], dtype=torch.float64)
    inner = boundary.clone()
    inner[:, 0] -= 1e-8

    boundary_temperature = model(boundary, fidelity="low")
    inner_temperature = model(inner, fidelity="low")

    assert boundary_temperature.item() == pytest.approx(295.15, abs=1e-10)
    assert abs(inner_temperature.item() - boundary_temperature.item()) < 1e-3


def test_prc_checkpoint_roundtrip_and_direct_point_query(tmp_path) -> None:
    model = PRCMultifidelityModel(
        modes=4, width=8, depth=2, correction_variant="amplitude"
    )
    checkpoint = tmp_path / "prc.pt"
    kwargs = {
        "modes": 4,
        "width": 8,
        "depth": 2,
        "correction_variant": "amplitude",
    }
    torch.save(
        {
            "method": "prc_multifidelity",
            "scales": ModelScales().__dict__,
            "model_kwargs": kwargs,
            "model_state": model.state_dict(),
            "material_passport": {"experiment_data_used": True},
        },
        checkpoint,
    )
    coordinates = _coordinates(torch.float32).numpy()
    expected = model(torch.from_numpy(coordinates)).detach().numpy()
    actual = Predictor(checkpoint, device="cpu").predict_points(coordinates)

    assert np.array_equal(actual, expected)
