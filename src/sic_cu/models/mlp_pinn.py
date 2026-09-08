from __future__ import annotations

from torch import Tensor, nn

from .common import CoordinateScaler, ModelScales, TemperatureOutput, build_mlp


class MLPPINN(nn.Module):
    """Pointwise temperature model used with or without physics loss."""

    def __init__(
        self,
        scales: ModelScales = ModelScales(),
        width: int = 128,
        depth: int = 5,
        activation: str = "tanh",
        include_material: bool = False,
    ) -> None:
        super().__init__()
        self.scaler = CoordinateScaler(scales)
        self.include_material = include_material
        self.network = build_mlp(5 if include_material else 4, 1, width, depth, activation)
        self.output = TemperatureOutput(scales.temperature_offset_k, scales.temperature_scale_k)

    def forward(self, coordinates: Tensor) -> Tensor:
        if self.include_material and coordinates.shape[-1] != 5:
            raise ValueError("Material-aware MLP requires [r,z,t,power,material_id]")
        scaled = self.scaler(coordinates)
        if not self.include_material:
            scaled = scaled[:, :4]
        return self.output(self.network(scaled))
