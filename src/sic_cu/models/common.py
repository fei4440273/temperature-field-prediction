from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelScales:
    r_max_m: float = 0.05834
    z_min_m: float = -0.0175
    time_max_s: float = 200.0
    power_max_w: float = 800.0
    temperature_offset_k: float = 295.15
    temperature_scale_k: float = 250.0


class CoordinateScaler(nn.Module):
    def __init__(self, scales: ModelScales) -> None:
        super().__init__()
        self.scales = scales

    def forward(self, coordinates: Tensor) -> Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] not in {4, 5}:
            raise ValueError("Coordinates must have columns [r,z,t,power,(material_id)]")
        r, z, t, power = coordinates[:, :4].unbind(dim=-1)
        scaled = torch.stack(
            (
                2.0 * r / self.scales.r_max_m - 1.0,
                2.0 * (z - self.scales.z_min_m) / (-self.scales.z_min_m) - 1.0,
                2.0 * t / self.scales.time_max_s - 1.0,
                2.0 * power / self.scales.power_max_w - 1.0,
            ),
            dim=-1,
        )
        if coordinates.shape[1] == 5:
            material = 2.0 * coordinates[:, 4:5] - 1.0
            scaled = torch.cat((scaled, material), dim=-1)
        return scaled


def activation_factory(name: str) -> Callable[[], nn.Module]:
    normalized = name.lower()
    if normalized == "tanh":
        return nn.Tanh
    if normalized == "silu":
        return nn.SiLU
    raise ValueError(f"Unsupported activation: {name}")


def build_mlp(
    input_dim: int,
    output_dim: int,
    width: int,
    depth: int,
    activation: str,
    zero_last: bool = True,
) -> nn.Sequential:
    make_activation = activation_factory(activation)
    layers: list[nn.Module] = [nn.Linear(input_dim, width), make_activation()]
    for _ in range(depth - 1):
        layers.extend((nn.Linear(width, width), make_activation()))
    final = nn.Linear(width, output_dim)
    if zero_last:
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, activation: str = "silu") -> None:
        super().__init__()
        make_activation = activation_factory(activation)
        self.net = nn.Sequential(
            nn.Linear(width, width),
            make_activation(),
            nn.Linear(width, width),
        )
        self.activation = make_activation()

    def forward(self, features: Tensor) -> Tensor:
        return self.activation(features + self.net(features))


class TemperatureOutput(nn.Module):
    def __init__(self, offset_k: float, scale_k: float) -> None:
        super().__init__()
        self.register_buffer("offset_k", torch.tensor(float(offset_k)))
        self.register_buffer("scale_k", torch.tensor(float(scale_k)))

    def forward(self, normalized_temperature: Tensor) -> Tensor:
        return self.offset_k + self.scale_k * normalized_temperature


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
