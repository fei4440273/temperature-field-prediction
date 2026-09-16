from __future__ import annotations

import math

import torch
from torch import Tensor


def gaussian_laser_flux(
    radius_m: Tensor,
    power_w: Tensor,
    absorption_fraction: float,
    beam_radius_m: float,
) -> Tensor:
    if not 0.0 <= absorption_fraction <= 1.0:
        raise ValueError("absorption_fraction must be in [0, 1]")
    if beam_radius_m <= 0:
        raise ValueError("beam_radius_m must be positive")
    scale = 2.0 * absorption_fraction * power_w / (math.pi * beam_radius_m**2)
    return scale * torch.exp(-2.0 * radius_m.pow(2) / beam_radius_m**2)


def top_hat_laser_flux(
    radius_m: Tensor,
    power_w: Tensor,
    absorption_fraction: float,
    beam_radius_m: float,
) -> Tensor:
    """Uniform circular heat flux whose area integral is absorbed power."""
    if not 0.0 <= absorption_fraction <= 1.0:
        raise ValueError("absorption_fraction must be in [0, 1]")
    if beam_radius_m <= 0:
        raise ValueError("beam_radius_m must be positive")
    value = absorption_fraction * power_w / (math.pi * beam_radius_m**2)
    return torch.where(radius_m <= beam_radius_m, value, torch.zeros_like(value))


def laser_flux(
    radius_m: Tensor,
    power_w: Tensor,
    profile: str,
    absorption_fraction: float,
    beam_radius_m: float,
) -> Tensor:
    if profile == "gaussian":
        return gaussian_laser_flux(radius_m, power_w, absorption_fraction, beam_radius_m)
    if profile == "top_hat":
        return top_hat_laser_flux(radius_m, power_w, absorption_fraction, beam_radius_m)
    raise ValueError(f"Unsupported laser profile: {profile}")


def convection_flux(
    temperature_k: Tensor,
    ambient_temperature_k: float,
    coefficient_w_m2_k: float,
) -> Tensor:
    if coefficient_w_m2_k < 0:
        raise ValueError("Convection coefficient cannot be negative")
    return coefficient_w_m2_k * (temperature_k - ambient_temperature_k)


def radiation_flux(
    temperature_k: Tensor,
    ambient_temperature_k: float,
    emissivity: float | Tensor,
) -> Tensor:
    emissivity_tensor = torch.as_tensor(
        emissivity, dtype=temperature_k.dtype, device=temperature_k.device
    )
    if bool(torch.any((emissivity_tensor < 0.0) | (emissivity_tensor > 1.0)).detach()):
        raise ValueError("emissivity must be in [0, 1]")
    stefan_boltzmann = 5.670374419e-8
    return emissivity_tensor * stefan_boltzmann * (
        temperature_k.pow(4) - float(ambient_temperature_k) ** 4
    )
