from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import ModelScales, build_mlp


class SurfaceResidualMLP(nn.Module):
    """Deterministic SiC-top correction baseline; it does not infer internal fields."""

    def __init__(
        self,
        scales: ModelScales = ModelScales(),
        width: int = 128,
        depth: int = 4,
        activation: str = "tanh",
        residual_scale_k: float = 100.0,
    ) -> None:
        super().__init__()
        self.scales = scales
        self.residual_scale_k = residual_scale_k
        self.network = build_mlp(3, 1, width, depth, activation)

    def forward(self, radius_time_power: Tensor) -> Tensor:
        if radius_time_power.ndim != 2 or radius_time_power.shape[1] != 3:
            raise ValueError("Expected [r, time, power]")
        normalized = torch.stack(
            (
                2.0 * radius_time_power[:, 0] / self.scales.r_max_m - 1.0,
                2.0 * radius_time_power[:, 1] / self.scales.time_max_s - 1.0,
                2.0 * radius_time_power[:, 2] / self.scales.power_max_w - 1.0,
            ),
            dim=-1,
        )
        return self.residual_scale_k * self.network(normalized)
