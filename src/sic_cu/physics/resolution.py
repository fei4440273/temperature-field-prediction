from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.eval.protocol_checks import combined_file_sha256

from .configuration import BoundaryConditions, load_boundary_conditions


def resolve_physics_state(
    boundary_path: str = "configs/boundary_conditions.yaml",
) -> dict[str, Any]:
    raw = load_yaml(boundary_path)
    external = raw["external_surface"]
    nominal = external["nominal_modeling_values"]
    interface = raw["interface"]
    resolved = {
        "schema_version": 1,
        "source_config": boundary_path,
        "physics_config_sha256": combined_file_sha256(
            (
                "configs/geometry.yaml",
                "configs/materials.yaml",
                boundary_path,
            )
        ),
        "values": {
            "ambient_temperature_k": {
                "value": float(raw["ambient_temperature_k"]),
                "status": "confirmed",
            },
            "initial_temperature_k": {
                "value": float(raw["initial_temperature_k"]),
                "status": "confirmed",
            },
            "laser_absorption_fraction": {
                "value": float(raw["laser"]["absorption_fraction"]),
                "status": "confirmed",
            },
            "laser_beam_radius_m": {
                "value": float(raw["laser"]["beam_radius_m"]),
                "status": "confirmed",
            },
            "top_convection_coefficient_w_m2_k": {
                "value": float(external["top_convection_coefficient_w_m2_k"]),
                "status": "effective_initialization",
            },
            "bottom_convection_coefficient_w_m2_k": {
                "value": float(external["bottom_convection_coefficient_w_m2_k"]),
                "status": "effective_initialization",
            },
            "silicon_carbide_emissivity": {
                "value": float(nominal["silicon_carbide"]),
                "status": "assumed_scenario",
            },
            "copper_emissivity": {
                "value": float(nominal["copper"]),
                "status": "assumed_scenario",
            },
            "contact_resistance_m2_k_w": {
                "value": float(interface["contact_resistance_initialization_m2_k_w"]),
                "status": "effective_initialization",
            },
            "cooling_outer_radius_temperature_k": {
                "value": float(raw["cooling"]["fixed_temperature_k"]),
                "status": "confirmed",
            },
        },
        "geometry_convention": {
            "z_direction": "negative_depth",
            "sic_top_z_m": 0.0,
            "sic_bottom_z_m": -0.012,
            "copper_bottom_z_m": -0.0175,
            "cooling_surface": "copper_outer_cylindrical_surface",
        },
        "unresolved_claims": [
            "emissivities_are_not_material_measurements",
            "contact_resistance_is_not_a_unique_true_value",
            "gaussian_power_finite_disc_convention_requires_simulation_cross_check",
        ],
    }
    return resolved


def load_resolved_boundary_conditions(
    boundary_path: str = "configs/boundary_conditions.yaml",
) -> BoundaryConditions:
    state = resolve_physics_state(boundary_path)
    values = state["values"]
    boundaries = load_boundary_conditions(boundary_path, allow_unidentified=True)
    return replace(
        boundaries,
        silicon_carbide_emissivity=values["silicon_carbide_emissivity"]["value"],
        copper_emissivity=values["copper_emissivity"]["value"],
        contact_resistance_m2_k_w=values["contact_resistance_m2_k_w"]["value"],
    )


def write_resolved_physics(
    output_path: str | Path = "reports/current_protocol/resolved_physics.yaml",
    boundary_path: str = "configs/boundary_conditions.yaml",
) -> Path:
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = resolve_physics_state(boundary_path)
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination


def resolved_boundary_snapshot(boundaries: BoundaryConditions) -> dict[str, Any]:
    """Return a serializable runtime snapshot for checkpoints."""
    return asdict(boundaries)
