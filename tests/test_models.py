from __future__ import annotations

import numpy as np
import pytest
import torch

from sic_cu.models import (
    ChebyshevSurfaceResidualGuide,
    DeepONetPINN,
    AdditiveCorrectionModel,
    GNOPINN,
    LSTMPINN,
    MLPPINN,
    MaterialWisePODPINN,
    ModelScales,
    PODPINN,
    fit_pod,
    load_pod_basis,
    save_pod_basis,
)
from sic_cu.models.common import parameter_count
from sic_cu.models.gno_pinn import knn_edges
from sic_cu.train.simulation import build_model


def _coordinates(count: int = 12, include_material: bool = False) -> torch.Tensor:
    coordinates = torch.rand(count, 4, dtype=torch.float32)
    coordinates[:, 0] *= 0.05834
    coordinates[:, 1] = -0.0175 * coordinates[:, 1]
    coordinates[:, 2] *= 200.0
    coordinates[:, 3] *= 800.0
    if include_material:
        coordinates = torch.cat(
            (coordinates, torch.randint(0, 2, (count, 1), dtype=torch.float32)), dim=1
        )
    return coordinates.requires_grad_(True)


@pytest.mark.parametrize(
    "model",
    [
        MLPPINN(width=16, depth=2),
        LSTMPINN(window=8, hidden_size=16),
        DeepONetPINN(width=16, latent_dim=16, blocks=1),
    ],
)
def test_coordinate_models_support_physics_gradients(model: torch.nn.Module) -> None:
    coordinates = _coordinates()
    output = model(coordinates)
    gradient = torch.autograd.grad(output.sum(), coordinates, create_graph=True)[0]
    assert output.shape == (coordinates.shape[0], 1)
    assert gradient.shape == coordinates.shape
    assert torch.isfinite(output).all()
    assert parameter_count(model) > 0


@pytest.mark.parametrize("method", ["lstm", "lstm_pinn", "deeponet", "deeponet_pinn"])
def test_data_and_physics_method_aliases_build(method: str) -> None:
    model = build_model(method, ModelScales(), width=8, hidden_size=8, latent_dim=8)
    assert model(_coordinates(4)).shape == (4, 1)


@pytest.mark.parametrize(
    "model",
    [
        MLPPINN(width=16, depth=2, include_material=True),
        LSTMPINN(window=8, hidden_size=16, include_material=True),
        DeepONetPINN(width=16, latent_dim=16, blocks=1, include_material=True),
    ],
)
def test_material_aware_coordinate_models_accept_explicit_labels(
    model: torch.nn.Module,
) -> None:
    coordinates = _coordinates(8, include_material=True)
    output = model(coordinates)
    gradient = torch.autograd.grad(output.sum(), coordinates, create_graph=True)[0]
    assert output.shape == (8, 1)
    assert gradient.shape == coordinates.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "model",
    [
        MLPPINN(width=16, depth=2, include_material=True),
        LSTMPINN(window=8, hidden_size=16, include_material=True),
        DeepONetPINN(width=16, latent_dim=16, blocks=1, include_material=True),
    ],
)
def test_material_aware_coordinate_models_require_labels(model: torch.nn.Module) -> None:
    with pytest.raises(ValueError, match="material_id"):
        model(_coordinates(8))


def test_lstm_unique_time_fast_path_matches_gradient_path() -> None:
    model = LSTMPINN(window=8, hidden_size=16, include_material=True)
    torch.manual_seed(9)
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.1, 0.1)
    coordinates = _coordinates(12, include_material=True).detach()
    coordinates[:, 2] = torch.tensor([0.0, 20.0, 40.0]).repeat_interleave(4)
    fast = model(coordinates)
    differentiable = model(coordinates.clone().requires_grad_(True))
    assert torch.allclose(fast, differentiable, atol=1e-6)


def test_pod_fit_and_model_forward(tmp_path) -> None:
    rng = np.random.default_rng(0)
    mesh = rng.random((20, 2), dtype=np.float32)
    snapshots = 300.0 + rng.normal(size=(12, 3)) @ rng.normal(size=(3, 20))
    basis = fit_pod(snapshots, mesh, max_modes=5)
    assert basis.modes.shape == (20, 5)
    assert np.all(np.diff(basis.cumulative_energy) >= 0)
    model = PODPINN(basis, modes=3, width=16)
    output = model(_coordinates(7))
    assert output.shape == (7, 1)
    assert torch.isfinite(output).all()
    path = tmp_path / "basis.npz"
    save_pod_basis(basis, path)
    loaded = load_pod_basis(path)
    assert np.array_equal(loaded.modes, basis.modes)
    assert model.predict_coefficients(torch.tensor([[100.0, 20.0]])).shape == (1, 3)


def test_material_wise_pod_routes_both_materials() -> None:
    rng = np.random.default_rng(3)
    copper_mesh = np.array([[0.03, -0.017], [0.04, -0.01], [0.05, 0.0]], dtype=np.float32)
    sic_mesh = np.array([[0.0, -0.01], [0.01, -0.005], [0.02, 0.0]], dtype=np.float32)
    copper = fit_pod(300.0 + rng.normal(size=(6, 3)), copper_mesh, max_modes=2)
    sic = fit_pod(350.0 + rng.normal(size=(6, 3)), sic_mesh, max_modes=2)
    model = MaterialWisePODPINN(copper, sic, modes=2, width=8, neighbors=2)
    coordinates = torch.tensor(
        [[0.04, -0.01, 20.0, 100.0], [0.01, -0.005, 20.0, 100.0]],
        dtype=torch.float32,
    )
    assert model(coordinates).shape == (2, 1)
    assert model(torch.tensor([[100.0, 20.0]]), True).shape == (1, 4)


def test_gno_uses_spatial_knn_graph() -> None:
    coordinates = _coordinates(16)
    materials = torch.randint(0, 2, (16,))
    edges = knn_edges(coordinates[:, :2], k=4)
    model = GNOPINN(width=16, layers=2, k=4)
    output = model.forward_graph(coordinates, materials, edges)
    assert edges.shape == (2, 64)
    assert output.shape == (16, 1)
    inferred = model(coordinates)
    assert inferred.shape == (16, 1)


def test_multifidelity_correction_starts_at_low_fidelity() -> None:
    low_fidelity = MLPPINN(width=16, depth=2)
    model = AdditiveCorrectionModel(low_fidelity, width=16, depth=2)
    coordinates = _coordinates(6, include_material=True)
    assert torch.allclose(model(coordinates), low_fidelity(coordinates))
    assert not any(parameter.requires_grad for parameter in model.low_fidelity_model.parameters())


def test_multifidelity_hard_initial_condition_and_power_gate() -> None:
    low_fidelity = MLPPINN(width=8, depth=1, include_material=True)
    model = AdditiveCorrectionModel(
        low_fidelity,
        width=8,
        depth=1,
        include_material=True,
        hard_initial_temperature_k=295.15,
        initial_ramp_time_s=0.05,
        correction_calibration_range_w=(115.2, 800.0),
    )
    with torch.no_grad():
        model.correction[-1].bias.fill_(1.0)
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0, 400.0, 1.0],
            [0.0, 0.0, 20.0, 0.0, 1.0],
            [0.0, 0.0, 20.0, 400.0, 1.0],
        ]
    )
    low = low_fidelity(coordinates)
    output = model(coordinates)
    assert output[0].item() == pytest.approx(295.15, abs=1e-5)
    assert output[1].item() == pytest.approx(low[1].item(), abs=1e-5)
    assert output[2].item() == pytest.approx(
        low[2].item() + model.scales.temperature_scale_k, abs=1e-4
    )


def test_multifidelity_hard_temperature_floor_and_cooling_boundary() -> None:
    low_fidelity = MLPPINN(width=8, depth=1, include_material=True)
    model = AdditiveCorrectionModel(
        low_fidelity,
        width=8,
        depth=1,
        include_material=True,
        hard_initial_temperature_k=295.15,
        hard_minimum_temperature_k=295.15,
        hard_cooling_radius_m=0.05834,
        hard_cooling_temperature_k=295.15,
        correction_calibration_range_w=(115.2, 800.0),
    )
    with torch.no_grad():
        model.correction[-1].bias.fill_(-10.0)
    coordinates = torch.tensor(
        [
            [0.02, -0.01, 20.0, 400.0, 0.0],
            [0.05834, -0.01, 20.0, 400.0, 0.0],
            [0.02, -0.01, 0.0, 400.0, 0.0],
        ]
    )
    output = model(coordinates)
    assert torch.all(output >= 295.15)
    assert output[1].item() == pytest.approx(295.15, abs=1e-5)
    assert output[2].item() == pytest.approx(295.15, abs=1e-5)


def test_multifidelity_linear_power_scaling_and_removed_direct_power_input() -> None:
    low_fidelity = MLPPINN(width=8, depth=1, include_material=True)
    model = AdditiveCorrectionModel(
        low_fidelity,
        width=8,
        depth=1,
        include_material=True,
        correction_power_scaling="linear",
        correction_power_reference_w=400.0,
        correction_direct_power_input=False,
    )
    assert model.correction[0].in_features == 5
    with torch.no_grad():
        model.correction[-1].bias.fill_(1.0)
    coordinates = torch.tensor([[0.01, -0.005, 20.0, 200.0, 1.0]])
    low = low_fidelity(coordinates)
    output = model(coordinates)
    assert output.item() == pytest.approx(
        low.item() + 0.5 * model.scales.temperature_scale_k, abs=1e-4
    )
    correction = model(coordinates, correction_only=True)
    assert correction.item() == pytest.approx(
        0.5 * model.scales.temperature_scale_k, abs=1e-4
    )


def test_multifidelity_surface_guide_is_hard_at_top_and_fades_to_network() -> None:
    low_fidelity = MLPPINN(width=8, depth=1, include_material=True)
    guide = ChebyshevSurfaceResidualGuide(
        mean_coefficients=np.array([[0.01]], dtype=np.float32),
        slope_coefficients=np.array([[0.0]], dtype=np.float32),
        power_center_w=200.0,
    )
    model = AdditiveCorrectionModel(
        low_fidelity,
        width=8,
        depth=1,
        include_material=True,
        surface_residual_guide=guide,
    )
    with torch.no_grad():
        model.correction[-1].bias.fill_(1.0)
    coordinates = torch.tensor(
        [
            [0.01, 0.0, 20.0, 200.0, 1.0],
            [0.01, -0.012, 20.0, 200.0, 1.0],
            [0.04, -0.012, 20.0, 200.0, 0.0],
        ]
    )
    correction = model(coordinates, correction_only=True).reshape(-1)
    assert correction[0].item() == pytest.approx(2.0, abs=1e-4)
    assert correction[1].item() == pytest.approx(250.0, abs=1e-4)
    assert correction[2].item() == pytest.approx(250.0, abs=1e-4)
