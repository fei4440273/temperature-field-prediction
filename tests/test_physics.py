from __future__ import annotations

import pytest
import torch
import yaml
from torch import nn

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.losses import PhysicsLossComputer
from sic_cu.physics.boundary import gaussian_laser_flux
from sic_cu.physics.collocation import sample_collocation
from sic_cu.physics.configuration import (
    BoundaryConditions,
    CoolingBoundary,
    LaserBoundary,
    load_boundary_conditions,
)
from sic_cu.physics.geometry import material_mask
from sic_cu.physics.heat_equation import axisymmetric_heat_residual
from sic_cu.physics.interface import interface_residuals
from sic_cu.physics.materials import (
    Material,
    MaterialProperty,
    PhysicsConfigurationError,
    load_materials,
)
from sic_cu.physics.trainable_parameters import TrainableBoundaryParameters


class AnalyticTemperature(nn.Module):
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        r, z, t = coordinates[:, 0:1], coordinates[:, 1:2], coordinates[:, 2:3]
        return 300.0 + r.pow(2) + z.pow(2) + t


def _constant_material(material_id: int) -> dict[int, Material]:
    return {
        material_id: Material(
            name="test",
            density_kg_m3=2.0,
            conductivity=MaterialProperty(kind="constant", value=3.0),
            heat_capacity=MaterialProperty(kind="constant", value=5.0),
        )
    }


def test_verified_production_materials_load_confirmed_constants() -> None:
    materials = load_materials(str(PROJECT_ROOT / "configs/materials.yaml"))
    assert materials[0].density_kg_m3 == 8900.0
    assert materials[0].conductivity.value == 401.0
    assert materials[0].heat_capacity.value == 400.0
    assert materials[1].density_kg_m3 == 3170.0
    assert materials[1].conductivity.value == 120.0
    assert materials[1].heat_capacity.value == 700.0


def test_unverified_production_boundaries_are_rejected() -> None:
    with pytest.raises(PhysicsConfigurationError, match="not verified"):
        load_boundary_conditions(str(PROJECT_ROOT / "configs/boundary_conditions.yaml"))


def test_identification_mode_accepts_only_declared_unidentified_values() -> None:
    boundaries = load_boundary_conditions(
        str(PROJECT_ROOT / "configs/boundary_conditions.yaml"),
        allow_unidentified=True,
    )
    assert boundaries.silicon_carbide_emissivity is None
    assert boundaries.copper_emissivity is None
    assert boundaries.contact_resistance_m2_k_w is None
    assert boundaries.contact_resistance_initialization_m2_k_w == pytest.approx(
        7.34072435302768e-5
    )


def test_identification_mode_rejects_unclassified_unverified_config(tmp_path) -> None:
    config = load_yaml(PROJECT_ROOT / "configs/boundary_conditions.yaml")
    config["interface"]["parameter_status"] = "unknown"
    path = tmp_path / "boundary_conditions.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(PhysicsConfigurationError, match="explicitly declared"):
        load_boundary_conditions(str(path), allow_unidentified=True)


def test_confirmed_boundary_semantics_are_parsed(tmp_path) -> None:
    config = load_yaml(PROJECT_ROOT / "configs/boundary_conditions.yaml")
    config["verified"] = True
    config["external_surface"]["silicon_carbide_emissivity"] = 0.8
    config["external_surface"]["copper_emissivity"] = 0.2
    config["interface"]["contact_resistance_m2_k_w"] = 1e-4
    path = tmp_path / "boundary_conditions.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    boundaries = load_boundary_conditions(str(path))

    assert boundaries.initial_temperature_k == 295.15
    assert boundaries.ambient_temperature_k == 295.15
    assert boundaries.laser.beam_radius_m == 0.02
    assert boundaries.laser.constant_during_heating is True
    assert boundaries.cooling.surface == "outer_radius"
    assert boundaries.cooling.fixed_temperature_k == 295.15
    assert boundaries.top_convection_coefficient_w_m2_k == 9.0
    assert boundaries.bottom_convection_coefficient_w_m2_k == 4.3
    assert boundaries.side_convection_coefficient_w_m2_k is None


def test_nonphysical_material_properties_are_rejected() -> None:
    with pytest.raises(PhysicsConfigurationError, match="positive"):
        MaterialProperty(kind="constant", value=0.0)


def test_axisymmetric_pde_residual_shape_and_value() -> None:
    coordinates = torch.tensor(
        [[0.0, -0.01, 1.0, 100.0], [0.01, -0.005, 2.0, 200.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    residual = axisymmetric_heat_residual(
        AnalyticTemperature(), coordinates, torch.zeros(2, dtype=torch.long), _constant_material(0)
    )
    # rho*cp*T_t - k*(T_rr + T_r/r + T_zz) = 2*5 - 3*(2+2+2) = -8.
    assert residual.shape == (2, 1)
    assert torch.allclose(residual, torch.full_like(residual, -8.0), atol=1e-8)


def test_material_mask_matches_embedded_geometry() -> None:
    points = torch.tensor(
        [
            [0.0, -0.006],
            [0.024, -0.011],
            [0.026, -0.011],
            [0.01, -0.013],
        ]
    )
    assert material_mask(points).tolist() == [1, 1, 0, 0]


def test_interface_conditions_support_perfect_and_resistive_contact() -> None:
    sic_t = torch.tensor([[310.0]])
    cu_t = torch.tensor([[300.0]])
    sic_q = torch.tensor([[1000.0]])
    cu_q = torch.tensor([[990.0]])
    perfect_t, flux = interface_residuals(sic_t, cu_t, sic_q, cu_q, None)
    resistive_t, _ = interface_residuals(sic_t, cu_t, sic_q, cu_q, 0.01)
    assert perfect_t.item() == 10.0
    assert flux.item() == 10.0
    assert resistive_t.item() == 0.0


def test_gaussian_laser_flux_integrates_to_absorbed_power() -> None:
    radius = torch.linspace(0.0, 0.1, 200_001, dtype=torch.float64)
    power = torch.full_like(radius, 200.0)
    flux = gaussian_laser_flux(radius, power, absorption_fraction=0.4, beam_radius_m=0.01)
    integral = torch.trapezoid(2.0 * torch.pi * radius * flux, radius)
    assert torch.isclose(integral, torch.tensor(80.0, dtype=torch.float64), rtol=1e-5)


def test_collocation_and_unified_physics_loss_are_finite() -> None:
    device = torch.device("cpu")
    batch = sample_collocation(8, device, seed=7)
    assert batch.interior.shape == (8, 5)
    assert set(batch.interior_material_ids.tolist()) == {0, 1}
    assert torch.equal(batch.interior[:, 4].long(), batch.interior_material_ids)
    assert torch.all(batch.interface_sic[:, 4] == 1)
    assert torch.all(batch.interface_copper[:, 4] == 0)
    boundaries = BoundaryConditions(
        ambient_temperature_k=295.15,
        initial_temperature_k=295.15,
        laser=LaserBoundary("gaussian", 0.4, 0.005, True),
        cooling=CoolingBoundary("fixed_temperature", 295.15, None, "outer_radius"),
        top_convection_coefficient_w_m2_k=10.0,
        bottom_convection_coefficient_w_m2_k=5.0,
        side_convection_coefficient_w_m2_k=None,
        radiation_enabled=False,
        silicon_carbide_emissivity=None,
        copper_emissivity=None,
        contact_resistance_m2_k_w=None,
    )
    materials = {
        0: _constant_material(0)[0],
        1: Material(
            name="test_sic",
            density_kg_m3=3.0,
            conductivity=MaterialProperty(kind="constant", value=4.0),
            heat_capacity=MaterialProperty(kind="constant", value=6.0),
        ),
    }
    components = PhysicsLossComputer(materials, boundaries)(AnalyticTemperature(), batch)
    assert set(components) == {"pde", "initial", "boundary", "interface", "physics_total"}
    assert all(torch.isfinite(value) for value in components.values())


def test_inverse_physics_parameters_receive_finite_gradients() -> None:
    device = torch.device("cpu")
    parameters = TrainableBoundaryParameters(0.5, 0.5, 7.34e-5)
    boundaries = load_boundary_conditions(
        str(PROJECT_ROOT / "configs/boundary_conditions.yaml"),
        allow_unidentified=True,
    )
    physics = PhysicsLossComputer(
        {
            0: _constant_material(0)[0],
            1: Material(
                name="test_sic",
                density_kg_m3=3.0,
                conductivity=MaterialProperty(kind="constant", value=4.0),
                heat_capacity=MaterialProperty(kind="constant", value=6.0),
            ),
        },
        boundaries,
        trainable_parameters=parameters,
    )
    components = physics(AnalyticTemperature(), sample_collocation(8, device, seed=13))
    components["physics_total"].backward()
    assert all(parameter.grad is not None for parameter in parameters.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters.parameters())
