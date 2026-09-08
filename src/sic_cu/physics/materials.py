from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from sic_cu.config import load_yaml


class PhysicsConfigurationError(ValueError):
    """Raised when unverified or missing production physics would be used."""


@dataclass(frozen=True)
class MaterialProperty:
    kind: str
    value: float | None = None
    temperatures_k: tuple[float, ...] = ()
    values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "constant" and self.value is None:
            raise PhysicsConfigurationError("Constant property requires value")
        if self.kind == "constant" and float(self.value) <= 0:
            raise PhysicsConfigurationError("Material property values must be positive")
        if self.kind == "table":
            if len(self.temperatures_k) < 2 or len(self.temperatures_k) != len(self.values):
                raise PhysicsConfigurationError("Tabulated property requires aligned arrays")
            if any(b <= a for a, b in zip(self.temperatures_k, self.temperatures_k[1:])):
                raise PhysicsConfigurationError("Property temperatures must be strictly increasing")
            if any(value <= 0 for value in self.values):
                raise PhysicsConfigurationError("Material property values must be positive")
        if self.kind not in {"constant", "table"}:
            raise PhysicsConfigurationError(f"Unsupported/unverified property kind: {self.kind}")

    def __call__(self, temperature_k: Tensor) -> Tensor:
        if self.kind == "constant":
            return torch.full_like(temperature_k, float(self.value))
        temperatures = torch.as_tensor(
            self.temperatures_k, dtype=temperature_k.dtype, device=temperature_k.device
        )
        values = torch.as_tensor(self.values, dtype=temperature_k.dtype, device=temperature_k.device)
        indices = torch.searchsorted(temperatures, temperature_k.detach()).clamp(1, len(values) - 1)
        left_t, right_t = temperatures[indices - 1], temperatures[indices]
        left_v, right_v = values[indices - 1], values[indices]
        fraction = ((temperature_k - left_t) / (right_t - left_t)).clamp(0.0, 1.0)
        return left_v + fraction * (right_v - left_v)


@dataclass(frozen=True)
class Material:
    name: str
    density_kg_m3: float
    conductivity: MaterialProperty
    heat_capacity: MaterialProperty


def _property_from_config(config: dict[str, Any], name: str) -> MaterialProperty:
    kind = config.get("kind")
    if kind == "constant":
        return MaterialProperty(kind="constant", value=float(config["value"]))
    if kind == "table":
        table = config.get("table")
        if not isinstance(table, list):
            raise PhysicsConfigurationError(f"{name} table is missing")
        return MaterialProperty(
            kind="table",
            temperatures_k=tuple(float(row[0]) for row in table),
            values=tuple(float(row[1]) for row in table),
        )
    raise PhysicsConfigurationError(f"{name} is unverified")


def load_materials(path: str = "configs/materials.yaml") -> dict[int, Material]:
    config = load_yaml(path)
    if config.get("verified") is not True:
        raise PhysicsConfigurationError(f"Material configuration is not verified: {path}")
    output: dict[int, Material] = {}
    for material_id, key in ((0, "copper"), (1, "silicon_carbide")):
        item = config[key]
        density = item.get("density_kg_m3")
        if density is None or float(density) <= 0:
            raise PhysicsConfigurationError(f"Missing density for {key}")
        output[material_id] = Material(
            name=key,
            density_kg_m3=float(density),
            conductivity=_property_from_config(item["conductivity_w_m_k"], f"{key}.k"),
            heat_capacity=_property_from_config(item["heat_capacity_j_kg_k"], f"{key}.cp"),
        )
    return output
