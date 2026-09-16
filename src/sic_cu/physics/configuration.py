from __future__ import annotations

from dataclasses import dataclass

from sic_cu.config import load_yaml

from .materials import PhysicsConfigurationError


def _positive(config: dict, key: str, scope: str, allow_zero: bool = False) -> float:
    value = config.get(key)
    if value is None:
        raise PhysicsConfigurationError(f"Missing {scope}.{key}")
    number = float(value)
    if number < 0 or (number == 0 and not allow_zero):
        raise PhysicsConfigurationError(f"Invalid {scope}.{key}: {number}")
    return number


@dataclass(frozen=True)
class LaserBoundary:
    profile: str
    absorption_fraction: float
    beam_radius_m: float
    constant_during_heating: bool


@dataclass(frozen=True)
class CoolingBoundary:
    kind: str
    fixed_temperature_k: float | None
    heat_transfer_coefficient_w_m2_k: float | None
    surface: str = "bottom"


@dataclass(frozen=True)
class BoundaryConditions:
    ambient_temperature_k: float
    initial_temperature_k: float
    laser: LaserBoundary
    cooling: CoolingBoundary
    top_convection_coefficient_w_m2_k: float
    bottom_convection_coefficient_w_m2_k: float | None
    side_convection_coefficient_w_m2_k: float | None
    radiation_enabled: bool
    silicon_carbide_emissivity: float | None
    copper_emissivity: float | None
    contact_resistance_m2_k_w: float | None
    contact_resistance_initialization_m2_k_w: float | None = None


def load_boundary_conditions(
    path: str = "configs/boundary_conditions.yaml",
    *,
    allow_unidentified: bool = False,
) -> BoundaryConditions:
    config = load_yaml(path)
    if config.get("verified") is not True:
        if not allow_unidentified:
            raise PhysicsConfigurationError(f"Boundary configuration is not verified: {path}")
        emissivity_status = config.get("external_surface", {}).get(
            "emissivity_parameter_status"
        )
        contact_status = config.get("interface", {}).get("parameter_status")
        if emissivity_status != "pending_joint_inverse_identification" or contact_status not in {
            "pending_inverse_identification",
            "low_fidelity_initialization_available",
        }:
            raise PhysicsConfigurationError(
                "Unverified boundary configuration is only allowed for explicitly declared "
                "inverse-identification parameters"
            )
    ambient = _positive(config, "ambient_temperature_k", "boundary")
    initial = _positive(config, "initial_temperature_k", "boundary")
    laser_cfg = config["laser"]
    profile = str(laser_cfg.get("profile", ""))
    if profile not in {"gaussian", "top_hat"}:
        raise PhysicsConfigurationError(f"Unsupported laser.profile: {profile}")
    absorption = _positive(laser_cfg, "absorption_fraction", "laser", allow_zero=True)
    if absorption > 1.0:
        raise PhysicsConfigurationError("laser.absorption_fraction must be in [0, 1]")
    if laser_cfg.get("beam_radius_definition") != "one_over_e_squared_intensity_radius":
        raise PhysicsConfigurationError(
            "laser.beam_radius_definition must match the implemented Gaussian convention"
        )
    constant = laser_cfg.get("constant_during_heating")
    if not isinstance(constant, bool):
        raise PhysicsConfigurationError("laser.constant_during_heating must be confirmed")
    if not constant:
        raise PhysicsConfigurationError(
            "Only a confirmed constant heating schedule is currently implemented"
        )
    cooling_cfg = config["cooling"]
    cooling_kind = str(cooling_cfg.get("kind", ""))
    cooling_surface = str(cooling_cfg.get("surface", ""))
    if cooling_surface not in {"bottom", "outer_radius"}:
        raise PhysicsConfigurationError(f"Unsupported cooling.surface: {cooling_surface}")
    if cooling_kind == "fixed_temperature":
        fixed_temperature = _positive(cooling_cfg, "fixed_temperature_k", "cooling")
        cooling_coefficient = None
    elif cooling_kind == "convection":
        fixed_temperature = None
        cooling_coefficient = _positive(
            cooling_cfg,
            "heat_transfer_coefficient_w_m2_k",
            "cooling",
            allow_zero=True,
        )
    else:
        raise PhysicsConfigurationError(f"Unsupported cooling.kind: {cooling_kind}")
    external_cfg = config["external_surface"]
    if external_cfg.get("convection_model") != (
        "effective_constant_from_horizontal_plate_natural_convection"
    ):
        raise PhysicsConfigurationError("Unsupported external_surface.convection_model")
    top_coefficient = _positive(
        external_cfg,
        "top_convection_coefficient_w_m2_k",
        "external_surface",
        allow_zero=True,
    )
    bottom_coefficient = None
    if cooling_surface != "bottom":
        bottom_coefficient = _positive(
            external_cfg,
            "bottom_convection_coefficient_w_m2_k",
            "external_surface",
            allow_zero=True,
        )
    side_coefficient = None
    if cooling_surface != "outer_radius":
        side_coefficient = _positive(
            external_cfg,
            "side_convection_coefficient_w_m2_k",
            "external_surface",
            allow_zero=True,
        )
    radiation_enabled = external_cfg.get("radiation_enabled")
    if not isinstance(radiation_enabled, bool):
        raise PhysicsConfigurationError("external_surface.radiation_enabled must be confirmed")
    sic_emissivity = None
    copper_emissivity = None
    if radiation_enabled:
        if external_cfg.get("silicon_carbide_emissivity") is not None:
            sic_emissivity = _positive(
                external_cfg,
                "silicon_carbide_emissivity",
                "external_surface",
                allow_zero=True,
            )
        if external_cfg.get("copper_emissivity") is not None:
            copper_emissivity = _positive(
                external_cfg,
                "copper_emissivity",
                "external_surface",
                allow_zero=True,
            )
        if not allow_unidentified and (sic_emissivity is None or copper_emissivity is None):
            raise PhysicsConfigurationError(
                "surface emissivities must be identified before production training"
            )
        if (sic_emissivity is not None and sic_emissivity > 1.0) or (
            copper_emissivity is not None and copper_emissivity > 1.0
        ):
            raise PhysicsConfigurationError("external surface emissivities must be in [0, 1]")
    interface_cfg = config["interface"]
    interface_kind = interface_cfg.get("kind")
    contact = interface_cfg.get("contact_resistance_m2_k_w")
    contact_initialization = interface_cfg.get(
        "contact_resistance_initialization_m2_k_w"
    )
    if contact is not None and float(contact) <= 0:
        raise PhysicsConfigurationError("interface.contact_resistance_m2_k_w must be positive")
    if contact_initialization is not None and float(contact_initialization) <= 0:
        raise PhysicsConfigurationError(
            "interface.contact_resistance_initialization_m2_k_w must be positive"
        )
    if (
        interface_kind == "identified_contact_resistance"
        and contact is None
        and not allow_unidentified
    ):
        raise PhysicsConfigurationError(
            "interface contact resistance must be identified before production training"
        )
    if interface_kind not in {"identified_contact_resistance", "perfect_contact"}:
        raise PhysicsConfigurationError(f"Unsupported interface.kind: {interface_kind}")
    if interface_kind == "perfect_contact" and contact is not None:
        raise PhysicsConfigurationError("perfect_contact cannot specify a contact resistance")
    return BoundaryConditions(
        ambient_temperature_k=ambient,
        initial_temperature_k=initial,
        laser=LaserBoundary(
            profile=profile,
            absorption_fraction=absorption,
            beam_radius_m=_positive(laser_cfg, "beam_radius_m", "laser"),
            constant_during_heating=constant,
        ),
        cooling=CoolingBoundary(
            kind=cooling_kind,
            fixed_temperature_k=fixed_temperature,
            heat_transfer_coefficient_w_m2_k=cooling_coefficient,
            surface=cooling_surface,
        ),
        top_convection_coefficient_w_m2_k=top_coefficient,
        bottom_convection_coefficient_w_m2_k=bottom_coefficient,
        side_convection_coefficient_w_m2_k=side_coefficient,
        radiation_enabled=radiation_enabled,
        silicon_carbide_emissivity=sic_emissivity,
        copper_emissivity=copper_emissivity,
        contact_resistance_m2_k_w=None if contact is None else float(contact),
        contact_resistance_initialization_m2_k_w=(
            None if contact_initialization is None else float(contact_initialization)
        ),
    )
