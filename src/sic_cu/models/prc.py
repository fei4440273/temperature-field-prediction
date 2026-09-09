from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch import Tensor, nn

from .common import CoordinateScaler, ModelScales, build_mlp


PRC_VARIANTS = {"temperature_residual", "amplitude", "amplitude_time"}


def _inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


class PRCMultifidelityModel(nn.Module):
    """Pole-response LF field with controlled high-fidelity corrections."""

    def __init__(
        self,
        scales: ModelScales = ModelScales(),
        modes: int = 8,
        width: int = 64,
        depth: int = 3,
        activation: str = "tanh",
        correction_variant: str = "amplitude_time",
        tau_min_s: float = 0.05,
        tau_max_correction_log_scale: float = 0.7,
        power_reference_w: float = 400.0,
        copper_radius_m: float = 0.05834,
        freeze_low_fidelity: bool = False,
    ) -> None:
        super().__init__()
        if modes <= 0:
            raise ValueError("modes must be positive")
        if correction_variant not in PRC_VARIANTS:
            raise ValueError(f"Unsupported PRC correction_variant: {correction_variant}")
        if tau_min_s <= 0.0 or power_reference_w <= 0.0 or copper_radius_m <= 0.0:
            raise ValueError("PRC time, power, and radius scales must be positive")
        if tau_max_correction_log_scale < 0.0:
            raise ValueError("tau_max_correction_log_scale must be nonnegative")
        self.scales = scales
        self.scaler = CoordinateScaler(scales)
        self.modes = int(modes)
        self.correction_variant = correction_variant
        self.tau_min_s = float(tau_min_s)
        self.tau_max_correction_log_scale = float(tau_max_correction_log_scale)
        self.power_reference_w = float(power_reference_w)
        self.copper_radius_m = float(copper_radius_m)

        self.lf_amplitude = build_mlp(4, modes, width, depth, activation)
        self.lf_tau_adjustment = build_mlp(1, modes, width, depth, activation)
        initial_tau = torch.logspace(
            math.log10(max(self.tau_min_s + 0.05, 0.1)),
            math.log10(scales.time_max_s),
            modes,
        )
        increments = torch.diff(
            torch.cat((torch.tensor([self.tau_min_s]), initial_tau))
        ).clamp_min(1e-4)
        self.lf_tau_increment_raw = nn.Parameter(_inverse_softplus(increments))

        self.amplitude_correction = build_mlp(
            4 + modes, modes, width, depth, activation
        )
        self.tau_correction = build_mlp(1, modes, width, depth, activation)
        self.temperature_correction = build_mlp(6, 1, width, depth, activation)
        self.freeze_low_fidelity(freeze_low_fidelity)

    def low_fidelity_parameters(self) -> Iterator[nn.Parameter]:
        yield self.lf_tau_increment_raw
        yield from self.lf_amplitude.parameters()
        yield from self.lf_tau_adjustment.parameters()

    def high_fidelity_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.amplitude_correction.parameters()
        yield from self.tau_correction.parameters()
        yield from self.temperature_correction.parameters()

    def active_high_fidelity_parameters(self) -> Iterator[nn.Parameter]:
        if self.correction_variant == "temperature_residual":
            yield from self.temperature_correction.parameters()
            return
        yield from self.amplitude_correction.parameters()
        if self.correction_variant == "amplitude_time":
            yield from self.tau_correction.parameters()

    def freeze_low_fidelity(self, freeze: bool = True) -> None:
        for parameter in self.low_fidelity_parameters():
            parameter.requires_grad_(not freeze)

    def _features(self, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        scaled = self.scaler(coordinates)
        radial_squared = 2.0 * (coordinates[:, 0:1] / self.scales.r_max_m).pow(2) - 1.0
        field_features = torch.cat(
            (radial_squared, scaled[:, 1:2], scaled[:, 3:4], scaled[:, 4:5]),
            dim=1,
        )
        return field_features, scaled[:, 3:4]

    def _lf_response_parameters(
        self, coordinates: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        features, power_feature = self._features(coordinates)
        amplitude = self.scales.temperature_scale_k * self.lf_amplitude(features)
        raw_increments = (
            self.lf_tau_increment_raw.unsqueeze(0)
            + self.lf_tau_adjustment(power_feature)
        )
        increments = torch.nn.functional.softplus(raw_increments) + 1e-6
        tau = self.tau_min_s + torch.cumsum(increments, dim=1)
        return amplitude, tau, increments

    def _envelope(self, coordinates: Tensor) -> Tensor:
        radius = coordinates[:, 0:1]
        material = coordinates[:, 4:5]
        copper = (1.0 - (radius / self.copper_radius_m).pow(2)).clamp_min(0.0)
        return torch.where(material >= 0.5, torch.ones_like(copper), copper)

    def _response(self, coordinates: Tensor, amplitude: Tensor, tau: Tensor) -> Tensor:
        time_s = coordinates[:, 2:3].clamp_min(0.0)
        power = coordinates[:, 3:4]
        kernel = -torch.expm1(-time_s / tau)
        increment = (
            (power / self.power_reference_w)
            * self._envelope(coordinates)
            * torch.sum(amplitude * kernel, dim=1, keepdim=True)
        )
        return self.scales.temperature_offset_k + increment

    def response_parameters(
        self, coordinates: Tensor, fidelity: str = "high"
    ) -> tuple[Tensor, Tensor]:
        self._validate_coordinates(coordinates)
        if fidelity not in {"low", "high"}:
            raise ValueError("fidelity must be 'low' or 'high'")
        amplitude_lf, tau_lf, increments_lf = self._lf_response_parameters(coordinates)
        if fidelity == "low" or self.correction_variant == "temperature_residual":
            return amplitude_lf, tau_lf
        features, power_feature = self._features(coordinates)
        normalized_amplitude = amplitude_lf / self.scales.temperature_scale_k
        delta_amplitude = self.scales.temperature_scale_k * self.amplitude_correction(
            torch.cat((features, normalized_amplitude), dim=1)
        )
        if self.correction_variant == "amplitude":
            return amplitude_lf + delta_amplitude, tau_lf
        log_scale = self.tau_max_correction_log_scale * torch.tanh(
            self.tau_correction(power_feature)
        )
        increments_hf = increments_lf * torch.exp(log_scale)
        tau_hf = self.tau_min_s + torch.cumsum(increments_hf, dim=1)
        return amplitude_lf + delta_amplitude, tau_hf

    @staticmethod
    def _validate_coordinates(coordinates: Tensor) -> None:
        if coordinates.ndim != 2 or coordinates.shape[1] != 5:
            raise ValueError(
                "PRC coordinates must have [r,z,t,power,material_id] columns"
            )

    def forward(self, coordinates: Tensor, fidelity: str = "high") -> Tensor:
        self._validate_coordinates(coordinates)
        amplitude_lf, tau_lf = self.response_parameters(coordinates, "low")
        low = self._response(coordinates, amplitude_lf, tau_lf)
        if fidelity == "low":
            return low
        if fidelity != "high":
            raise ValueError("fidelity must be 'low' or 'high'")
        if self.correction_variant == "temperature_residual":
            scaled = self.scaler(coordinates)
            normalized_lf = (
                low - self.scales.temperature_offset_k
            ) / self.scales.temperature_scale_k
            residual = self.scales.temperature_scale_k * self.temperature_correction(
                torch.cat((scaled, normalized_lf), dim=1)
            )
            time_gate = -torch.expm1(
                -coordinates[:, 2:3].clamp_min(0.0) / self.tau_min_s
            )
            power_gate = coordinates[:, 3:4] / self.power_reference_w
            return low + power_gate * time_gate * self._envelope(coordinates) * residual
        amplitude_hf, tau_hf = self.response_parameters(coordinates, "high")
        return self._response(coordinates, amplitude_hf, tau_hf)

    def forward_low(self, coordinates: Tensor) -> Tensor:
        return self.forward(coordinates, fidelity="low")

    def analytic_time_derivative(
        self, coordinates: Tensor, fidelity: str = "high"
    ) -> Tensor:
        if fidelity == "high" and self.correction_variant == "temperature_residual":
            raise ValueError(
                "temperature_residual has a time-dependent correction; use autograd"
            )
        amplitude, tau = self.response_parameters(coordinates, fidelity)
        time_s = coordinates[:, 2:3].clamp_min(0.0)
        return (
            (coordinates[:, 3:4] / self.power_reference_w)
            * self._envelope(coordinates)
            * torch.sum(amplitude / tau * torch.exp(-time_s / tau), dim=1, keepdim=True)
        )
