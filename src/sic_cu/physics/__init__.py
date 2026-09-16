"""Shared axisymmetric heat-transfer physics."""

from .heat_equation import axisymmetric_heat_residual
from .resolution import (
    load_resolved_boundary_conditions,
    resolve_physics_state,
    resolved_boundary_snapshot,
    write_resolved_physics,
)
from .configuration import BoundaryConditions, load_boundary_conditions
from .collocation import CollocationBatch, sample_collocation
from .materials import Material, MaterialProperty, PhysicsConfigurationError, load_materials
from .trainable_parameters import TrainableBoundaryParameters

__all__ = [
    "Material",
    "MaterialProperty",
    "PhysicsConfigurationError",
    "TrainableBoundaryParameters",
    "BoundaryConditions",
    "CollocationBatch",
    "axisymmetric_heat_residual",
    "load_resolved_boundary_conditions",
    "resolve_physics_state",
    "resolved_boundary_snapshot",
    "write_resolved_physics",
    "load_boundary_conditions",
    "load_materials",
    "sample_collocation",
]
