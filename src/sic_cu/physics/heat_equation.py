from __future__ import annotations

import torch
from torch import Tensor, nn

from .materials import Material


def _gradient(output: Tensor, inputs: Tensor) -> Tensor:
    gradient = torch.autograd.grad(
        output,
        inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]
    return gradient


def axisymmetric_heat_residual(
    model: nn.Module,
    coordinates: Tensor,
    material_ids: Tensor,
    materials: dict[int, Material],
    axis_tolerance_m: float = 1e-7,
) -> Tensor:
    """Return rho*cp*T_t - div(k grad(T)) for [r,z,t,P,(material)] coordinates."""
    if coordinates.ndim != 2 or coordinates.shape[1] not in {4, 5}:
        raise ValueError("coordinates must have columns [r,z,t,P,(material_id)]")
    if material_ids.reshape(-1).shape[0] != coordinates.shape[0]:
        raise ValueError("material_ids must have one value per coordinate")
    x = coordinates.requires_grad_(True)
    temperature = model(x)
    if temperature.shape != (x.shape[0], 1):
        raise ValueError("model must return shape [N, 1]")
    first = _gradient(temperature, x)
    t_r = first[:, 0:1]
    t_z = first[:, 1:2]
    t_t = first[:, 2:3]
    residual = torch.empty_like(temperature)
    flat_ids = material_ids.reshape(-1)
    for material_id, material in materials.items():
        mask = flat_ids == material_id
        if not bool(mask.any()):
            continue
        conductivity = material.conductivity(temperature)
        heat_capacity = material.heat_capacity(temperature)
        radial_flux = conductivity * t_r
        axial_flux = conductivity * t_z
        radial_derivative = _gradient(radial_flux, x)[:, 0:1]
        axial_derivative = _gradient(axial_flux, x)[:, 1:2]
        r = x[:, 0:1]
        radial_term = torch.where(
            r.abs() <= axis_tolerance_m,
            2.0 * radial_derivative,
            radial_derivative + radial_flux / r.clamp_min(axis_tolerance_m),
        )
        value = material.density_kg_m3 * heat_capacity * t_t - radial_term - axial_derivative
        residual[mask] = value[mask]
    known_ids = torch.zeros_like(flat_ids, dtype=torch.bool)
    for material_id in materials:
        known_ids |= flat_ids == material_id
    if not bool(known_ids.all()):
        raise ValueError("Unknown material_id in collocation points")
    return residual
