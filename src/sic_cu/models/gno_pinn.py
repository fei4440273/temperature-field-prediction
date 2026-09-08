from __future__ import annotations

import torch
from torch import Tensor, nn

from sic_cu.physics.geometry import material_mask

from .common import CoordinateScaler, ModelScales, ResidualBlock, TemperatureOutput


def knn_edges(coordinates_rz: Tensor, k: int = 8) -> Tensor:
    if coordinates_rz.ndim != 2 or coordinates_rz.shape[1] != 2:
        raise ValueError("coordinates_rz must have shape [N, 2]")
    if not 1 <= k < coordinates_rz.shape[0]:
        raise ValueError("k must be between 1 and N-1")
    distances = torch.cdist(coordinates_rz, coordinates_rz)
    neighbors = torch.topk(distances, k=k + 1, largest=False).indices[:, 1:]
    destination = torch.arange(coordinates_rz.shape[0], device=coordinates_rz.device)
    destination = destination[:, None].expand_as(neighbors)
    return torch.stack((neighbors.reshape(-1), destination.reshape(-1)), dim=0)


class GraphMessageLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.message = nn.Sequential(nn.Linear(width * 2 + 2, width), nn.SiLU(), nn.Linear(width, width))
        self.update = ResidualBlock(width, "silu")

    def forward(self, hidden: Tensor, coordinates_rz: Tensor, edge_index: Tensor) -> Tensor:
        source, destination = edge_index
        relative = coordinates_rz[source] - coordinates_rz[destination]
        messages = self.message(torch.cat((hidden[source], hidden[destination], relative), dim=-1))
        aggregate = torch.zeros_like(hidden)
        aggregate.index_add_(0, destination, messages)
        degree = torch.bincount(destination, minlength=hidden.shape[0]).clamp_min(1).to(hidden.dtype)
        return self.update(hidden + aggregate / degree[:, None])


class GNOPINN(nn.Module):
    """KNN graph neural operator for one fixed-power/time FEM graph."""

    def __init__(
        self,
        scales: ModelScales = ModelScales(),
        width: int = 128,
        layers: int = 4,
        k: int = 8,
    ) -> None:
        super().__init__()
        self.k = k
        self.scaler = CoordinateScaler(scales)
        self.input = nn.Linear(5, width)
        self.layers = nn.ModuleList(GraphMessageLayer(width) for _ in range(layers))
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.output = TemperatureOutput(scales.temperature_offset_k, scales.temperature_scale_k)

    def forward_graph(
        self,
        coordinates: Tensor,
        material_ids: Tensor,
        edge_index: Tensor | None = None,
    ) -> Tensor:
        scaled = self.scaler(coordinates[:, :4])
        hidden = self.input(torch.cat((scaled, material_ids.reshape(-1, 1).to(scaled)), dim=-1))
        edges = knn_edges(coordinates[:, :2], self.k) if edge_index is None else edge_index
        for layer in self.layers:
            hidden = layer(hidden, coordinates[:, :2], edges)
        return self.output(self.head(hidden))

    def forward(
        self,
        coordinates: Tensor,
        material_ids: Tensor | None = None,
        edge_index: Tensor | None = None,
    ) -> Tensor:
        if material_ids is None:
            material_ids = (
                coordinates[:, 4].long()
                if coordinates.shape[1] == 5
                else material_mask(coordinates[:, :2])
            )
        return self.forward_graph(coordinates, material_ids, edge_index)
