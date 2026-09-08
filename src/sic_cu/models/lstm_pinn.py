from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import CoordinateScaler, ModelScales, TemperatureOutput, build_mlp


class LSTMPINN(nn.Module):
    """Coordinate model with an explicit differentiable time-history encoder."""

    def __init__(
        self,
        scales: ModelScales = ModelScales(),
        window: int = 16,
        hidden_size: int = 96,
        activation: str = "silu",
        include_material: bool = False,
    ) -> None:
        super().__init__()
        if window < 2:
            raise ValueError("LSTM window must be at least 2")
        self.window = window
        self.include_material = include_material
        self.scaler = CoordinateScaler(scales)
        self.spatial_power_encoder = build_mlp(
            4 if include_material else 3,
            hidden_size,
            hidden_size,
            2,
            activation,
            False,
        )
        self.time_encoder = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fusion = build_mlp(hidden_size * 2, 1, hidden_size, 2, activation)
        self.output = TemperatureOutput(scales.temperature_offset_k, scales.temperature_scale_k)

    def forward(self, coordinates: Tensor) -> Tensor:
        if self.include_material and coordinates.shape[-1] != 5:
            raise ValueError("Material-aware LSTM requires [r,z,t,power,material_id]")
        scaled = self.scaler(coordinates)
        spatial_power = scaled[:, [0, 1, 3, 4]] if self.include_material else scaled[:, [0, 1, 3]]
        fractions = torch.linspace(
            0.0, 1.0, self.window, dtype=coordinates.dtype, device=coordinates.device
        ).view(1, self.window, 1)
        current_scaled_time = scaled[:, 2:3]
        if coordinates.requires_grad:
            encoded_times = current_scaled_time
            inverse = None
        else:
            encoded_times, inverse = torch.unique(
                current_scaled_time, sorted=True, return_inverse=True, dim=0
            )
        current_physical_fraction = (encoded_times + 1.0) / 2.0
        history_scaled = 2.0 * current_physical_fraction[:, None, :] * fractions - 1.0
        _, (hidden, _) = self.time_encoder(history_scaled)
        time_features = hidden[-1] if inverse is None else hidden[-1].index_select(0, inverse)
        fused = torch.cat((self.spatial_power_encoder(spatial_power), time_features), dim=-1)
        return self.output(self.fusion(fused))
