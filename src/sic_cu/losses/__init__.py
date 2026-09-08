"""Loss functions shared by all temperature-field candidates."""

from .physics import PhysicsLossComputer, PhysicsLossWeights

__all__ = ["PhysicsLossComputer", "PhysicsLossWeights"]
