from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .common import CoordinateScaler, ModelScales, build_mlp


class AdditiveCorrectionModel(nn.Module):
    """T_HF = T_LF + delta T with a trainable coordinate/temperature correction."""

    def __init__(
        self,
        low_fidelity_model: nn.Module,
        scales: ModelScales = ModelScales(),
        width: int = 128,
        depth: int = 4,
        activation: str = "tanh",
        freeze_low_fidelity: bool = True,
        include_material: bool = True,
        hard_initial_temperature_k: float | None = None,
        initial_ramp_time_s: float = 0.05,
        hard_minimum_temperature_k: float | None = None,
        minimum_temperature_beta_per_k: float = 10.0,
        hard_cooling_radius_m: float | None = None,
        hard_cooling_temperature_k: float | None = None,
        hard_cooling_tolerance_m: float = 1e-7,
        correction_calibration_range_w: tuple[float, float] | list[float] | None = None,
        correction_support_range_w: tuple[float, float] | list[float] = (0.0, 800.0),
        correction_extrapolation_exponent: float = 2.0,
        correction_power_scaling: str = "none",
        correction_power_reference_w: float = 400.0,
        correction_direct_power_input: bool = True,
        surface_residual_guide: nn.Module | None = None,
        silicon_carbide_height_m: float = 0.012,
        surface_guide_output: str = "residual",
    ) -> None:
        super().__init__()
        self.low_fidelity_model = low_fidelity_model
        self.scaler = CoordinateScaler(scales)
        self.scales = scales
        self.include_material = include_material
        self.hard_initial_temperature_k = hard_initial_temperature_k
        self.initial_ramp_time_s = float(initial_ramp_time_s)
        self.hard_minimum_temperature_k = hard_minimum_temperature_k
        self.minimum_temperature_beta_per_k = float(minimum_temperature_beta_per_k)
        self.hard_cooling_radius_m = hard_cooling_radius_m
        self.hard_cooling_temperature_k = hard_cooling_temperature_k
        self.hard_cooling_tolerance_m = float(hard_cooling_tolerance_m)
        self.correction_calibration_range_w = (
            None
            if correction_calibration_range_w is None
            else tuple(float(value) for value in correction_calibration_range_w)
        )
        self.correction_support_range_w = tuple(
            float(value) for value in correction_support_range_w
        )
        self.correction_extrapolation_exponent = float(
            correction_extrapolation_exponent
        )
        self.correction_power_scaling = str(correction_power_scaling)
        self.correction_power_reference_w = float(correction_power_reference_w)
        self.correction_direct_power_input = bool(correction_direct_power_input)
        self.surface_residual_guide = surface_residual_guide
        self.silicon_carbide_height_m = float(silicon_carbide_height_m)
        self.surface_guide_output = str(surface_guide_output)
        if self.hard_initial_temperature_k is not None and self.initial_ramp_time_s <= 0:
            raise ValueError("initial_ramp_time_s must be positive")
        if (
            self.hard_minimum_temperature_k is not None
            and self.minimum_temperature_beta_per_k <= 0
        ):
            raise ValueError("minimum_temperature_beta_per_k must be positive")
        if (self.hard_cooling_radius_m is None) != (
            self.hard_cooling_temperature_k is None
        ):
            raise ValueError(
                "hard_cooling_radius_m and hard_cooling_temperature_k must be set together"
            )
        if self.hard_cooling_radius_m is not None:
            if self.hard_cooling_radius_m <= 0 or self.hard_cooling_tolerance_m <= 0:
                raise ValueError("Hard cooling radius and tolerance must be positive")
        if self.correction_calibration_range_w is not None:
            support_low, support_high = self.correction_support_range_w
            calibration_low, calibration_high = self.correction_calibration_range_w
            if not support_low < calibration_low <= calibration_high <= support_high:
                raise ValueError(
                    "Calibration range must lie inside the correction support range"
                )
            if self.correction_extrapolation_exponent <= 0:
                raise ValueError("correction_extrapolation_exponent must be positive")
        if self.correction_power_scaling not in {"none", "linear"}:
            raise ValueError("correction_power_scaling must be 'none' or 'linear'")
        if self.correction_power_reference_w <= 0.0:
            raise ValueError("correction_power_reference_w must be positive")
        if self.surface_residual_guide is not None:
            if not self.include_material:
                raise ValueError("Surface residual guide requires material labels")
            if self.silicon_carbide_height_m <= 0.0:
                raise ValueError("silicon_carbide_height_m must be positive")
            if self.surface_guide_output not in {"residual", "absolute_temperature"}:
                raise ValueError(
                    "surface_guide_output must be 'residual' or 'absolute_temperature'"
                )
            for parameter in self.surface_residual_guide.parameters():
                parameter.requires_grad_(False)
        correction_input_dim = 6 if include_material else 5
        if not self.correction_direct_power_input:
            correction_input_dim -= 1
        self.correction = build_mlp(
            correction_input_dim, 1, width, depth, activation
        )
        self.freeze_low_fidelity(freeze_low_fidelity)

    def freeze_low_fidelity(self, freeze: bool = True) -> None:
        for parameter in self.low_fidelity_model.parameters():
            parameter.requires_grad_(not freeze)

    def forward(self, coordinates: Tensor, correction_only: bool = False) -> Tensor:
        if self.include_material and coordinates.shape[-1] != 5:
            raise ValueError(
                "Material-aware correction requires [r,z,t,power,material_id]"
            )
        low_fidelity = self.low_fidelity_model(coordinates)
        normalized_lf = (
            low_fidelity - self.scales.temperature_offset_k
        ) / self.scales.temperature_scale_k
        scaled = self.scaler(coordinates)
        if not self.include_material:
            scaled = scaled[:, :4]
        if not self.correction_direct_power_input:
            scaled = torch.cat((scaled[:, :3], scaled[:, 4:]), dim=-1)
        correction = self.correction(torch.cat((scaled, normalized_lf), dim=-1))
        correction_k = self.scales.temperature_scale_k * correction
        if self.correction_power_scaling == "linear":
            correction_k = correction_k * (
                coordinates[:, 3:4] / self.correction_power_reference_w
            )
        correction_gate = torch.ones_like(correction_k)
        if self.correction_calibration_range_w is not None:
            support_low, support_high = self.correction_support_range_w
            calibration_low, calibration_high = self.correction_calibration_range_w
            power = coordinates[:, 3:4]
            lower_gate = torch.clamp(
                (power - support_low) / (calibration_low - support_low), 0.0, 1.0
            ).pow(self.correction_extrapolation_exponent)
            upper_gate = (
                torch.ones_like(power)
                if calibration_high == support_high
                else torch.clamp(
                    (support_high - power) / (support_high - calibration_high),
                    0.0,
                    1.0,
                ).pow(self.correction_extrapolation_exponent)
            )
            correction_gate = lower_gate * upper_gate
            correction_k = correction_k * correction_gate
        if self.surface_residual_guide is not None:
            silicon_carbide = coordinates[:, 4] > 0.5
            if bool(silicon_carbide.any()):
                sic_coordinates = coordinates[silicon_carbide]
                surface_guide = self.surface_residual_guide(sic_coordinates)
                if self.surface_guide_output == "absolute_temperature":
                    surface_guide = surface_guide - low_fidelity[silicon_carbide]
                surface_guide = surface_guide * correction_gate[silicon_carbide]
                depth_fraction = torch.clamp(
                    -sic_coordinates[:, 1:2] / self.silicon_carbide_height_m,
                    0.0,
                    1.0,
                )
                correction_k = correction_k.clone()
                correction_k[silicon_carbide] = (
                    (1.0 - depth_fraction) * surface_guide
                    + depth_fraction * correction_k[silicon_carbide]
                )
        if correction_only:
            return correction_k
        output = low_fidelity + correction_k
        if self.hard_minimum_temperature_k is not None:
            increment = output - self.hard_minimum_temperature_k
            output = self.hard_minimum_temperature_k + F.softplus(
                increment,
                beta=self.minimum_temperature_beta_per_k,
                threshold=20.0,
            )
        if self.hard_initial_temperature_k is not None:
            time_s = torch.clamp(coordinates[:, 2:3], min=0.0)
            time_gate = -torch.expm1(-time_s / self.initial_ramp_time_s)
            output = self.hard_initial_temperature_k + time_gate * (
                output - self.hard_initial_temperature_k
            )
        if self.hard_cooling_radius_m is not None:
            copper = coordinates[:, 4:5] < 0.5
            outer_radius = torch.isclose(
                coordinates[:, 0:1],
                torch.as_tensor(
                    self.hard_cooling_radius_m,
                    dtype=coordinates.dtype,
                    device=coordinates.device,
                ),
                rtol=0.0,
                atol=self.hard_cooling_tolerance_m,
            )
            cooling_temperature = torch.as_tensor(
                self.hard_cooling_temperature_k,
                dtype=output.dtype,
                device=output.device,
            )
            output = torch.where(copper & outer_radius, cooling_temperature, output)
        return output
