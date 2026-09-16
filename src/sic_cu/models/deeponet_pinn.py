from __future__ import annotations

from torch import Tensor, nn

from .common import CoordinateScaler, ModelScales, ResidualBlock, TemperatureOutput


class _ResidualEncoder(nn.Module):
    def __init__(self, input_dim: int, width: int, blocks: int, activation: str) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.blocks = nn.Sequential(*(ResidualBlock(width, activation) for _ in range(blocks)))

    def forward(self, values: Tensor) -> Tensor:
        return self.blocks(self.input(values))


class DeepONetPINN(nn.Module):
    """Residual DeepONet with power branch and r-z-t trunk."""

    def __init__(
        self,
        scales: ModelScales = ModelScales(),
        width: int = 128,
        latent_dim: int = 128,
        blocks: int = 3,
        activation: str = "silu",
        include_material: bool = False,
    ) -> None:
        super().__init__()
        self.scaler = CoordinateScaler(scales)
        self.include_material = include_material
        self.branch = _ResidualEncoder(1, width, blocks, activation)
        self.trunk = _ResidualEncoder(4 if include_material else 3, width, blocks, activation)
        self.branch_projection = nn.Linear(width, latent_dim)
        self.trunk_projection = nn.Linear(width, latent_dim)
        self.bias = nn.Parameter(Tensor([0.0]))
        nn.init.zeros_(self.branch_projection.weight)
        nn.init.zeros_(self.branch_projection.bias)
        self.output = TemperatureOutput(scales.temperature_offset_k, scales.temperature_scale_k)

    def forward(self, coordinates: Tensor) -> Tensor:
        if self.include_material and coordinates.shape[-1] != 5:
            raise ValueError("Material-aware DeepONet requires [r,z,t,power,material_id]")
        scaled = self.scaler(coordinates)
        branch = self.branch_projection(self.branch(scaled[:, 3:4]))
        trunk_inputs = scaled[:, [0, 1, 2, 4]] if self.include_material else scaled[:, :3]
        trunk = self.trunk_projection(self.trunk(trunk_inputs))
        normalized = (branch * trunk).sum(dim=-1, keepdim=True) / branch.shape[-1] ** 0.5
        return self.output(normalized + self.bias)
