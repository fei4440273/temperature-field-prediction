from __future__ import annotations

import numpy as np
import pytest
import torch

from sic_cu.config import load_yaml
from sic_cu.data.fields import load_processed_field
from sic_cu.eval.energy_v5 import (
    AxisymmetricGeometry,
    analytic_control_tests,
    axisymmetric_horizontal_quadrature,
    axisymmetric_lumped_nodal_weights,
    axisymmetric_rectangle_quadrature,
    axisymmetric_vertical_quadrature,
    hard_cooling_override_control,
)
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions


def _geometry() -> AxisymmetricGeometry:
    return AxisymmetricGeometry.from_config(load_yaml("configs/geometry.yaml"))


def test_axisymmetric_gauss_weights_match_analytic_measures() -> None:
    geometry = _geometry()
    device = torch.device("cpu")
    rz, volume_weights = axisymmetric_rectangle_quadrature(
        (0.0, geometry.silicon_carbide_radius_m),
        (geometry.silicon_carbide_bottom_z_m, 0.0),
        16,
        device,
    )
    radius, top_weights = axisymmetric_horizontal_quadrature(
        (0.0, geometry.silicon_carbide_radius_m), 16, device
    )
    axial, side_weights = axisymmetric_vertical_quadrature(
        geometry.copper_radius_m,
        (geometry.copper_bottom_z_m, 0.0),
        16,
        device,
    )
    assert rz.shape == (16**2, 2)
    assert len(radius) == len(axial) == 16
    assert volume_weights.sum().item() == pytest.approx(
        geometry.silicon_carbide_volume_m3, rel=1e-12
    )
    assert top_weights.sum().item() == pytest.approx(
        np.pi * geometry.silicon_carbide_radius_m**2, rel=1e-12
    )
    assert side_weights.sum().item() == pytest.approx(
        2
        * np.pi
        * geometry.copper_radius_m
        * abs(geometry.copper_bottom_z_m),
        rel=1e-12,
    )


def test_hard_temperature_override_exposes_zero_boundary_flux() -> None:
    geometry = _geometry()
    materials = load_materials()
    boundaries = load_resolved_boundary_conditions()
    result = hard_cooling_override_control(
        geometry.copper_radius_m,
        float(boundaries.cooling.fixed_temperature_k),
        float(materials[0].conductivity.value),
        1e-7,
        1e-6,
        torch.device("cpu"),
    )
    assert result["exposes_zero_gradient_failure"] is True
    assert result["boundary_radial_gradient_k_m"] == pytest.approx(0.0)
    assert result["inner_radial_gradient_k_m"] == pytest.approx(-100.0)
    assert result["inner_outward_heat_flux_w_m2"] == pytest.approx(40100.0)


def test_manufactured_and_isothermal_controls_close() -> None:
    rows = analytic_control_tests(
        _geometry(),
        load_resolved_boundary_conditions(),
        load_materials(),
        [16, 32],
        torch.device("cpu"),
    )
    assert rows
    assert all(row["passed"] for row in rows)
    assert {
        "manufactured_axisymmetric_quadratic_field",
        "finite_disc_laser_integral",
        "zero_power_isothermal_field",
    } == {row["control"] for row in rows}


def test_lumped_simulation_weights_recover_material_volumes() -> None:
    geometry = _geometry()
    field = load_processed_field(90.0)
    weights = axisymmetric_lumped_nodal_weights(
        field.coordinates_rz_m, field.material_ids, geometry
    )
    assert np.all(weights > 0.0)
    assert weights[field.material_ids == 0].sum() == pytest.approx(
        geometry.copper_volume_m3, rel=2e-7
    )
    assert weights[field.material_ids == 1].sum() == pytest.approx(
        geometry.silicon_carbide_volume_m3, rel=2e-7
    )
