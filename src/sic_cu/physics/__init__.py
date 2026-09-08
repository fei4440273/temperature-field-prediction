"""Shared axisymmetric heat-transfer physics."""

from .heat_equation import axisymmetric_heat_residual
from .configuration import BoundaryConditions, load_boundary_conditions
from .collocation import CollocationBatch, sample_collocation
from .materials import Material, MaterialProperty, PhysicsConfigurationError
from .trainable_parameters import TrainableBoundaryParameters

__all__ = [
    "Material",
    "MaterialProperty",
    "PhysicsConfigurationError",
    "TrainableBoundaryParameters",
    "BoundaryConditions",
    "CollocationBatch",
    "axisymmetric_heat_residual",
    "load_boundary_conditions",
    "sample_collocation",
]
