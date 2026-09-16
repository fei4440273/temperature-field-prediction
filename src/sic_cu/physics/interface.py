from __future__ import annotations

import torch
from torch import Tensor


def interface_residuals(
    temperature_sic: Tensor,
    temperature_cu: Tensor,
    normal_flux_sic: Tensor,
    normal_flux_cu: Tensor,
    contact_resistance_m2_k_w: float | Tensor | None,
) -> tuple[Tensor, Tensor]:
    flux_residual = normal_flux_sic - normal_flux_cu
    if contact_resistance_m2_k_w is None:
        temperature_residual = temperature_sic - temperature_cu
    else:
        temperature_residual = (
            temperature_sic
            - temperature_cu
            - contact_resistance_m2_k_w * normal_flux_sic
        )
    return temperature_residual, flux_residual


def normal_heat_flux(conductivity: Tensor, temperature_gradient: Tensor, normal: Tensor) -> Tensor:
    return -conductivity * torch.sum(temperature_gradient * normal, dim=-1, keepdim=True)
