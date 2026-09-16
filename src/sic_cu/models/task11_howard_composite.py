"""Independent Howard-style three-block operator, not an author-code reproduction.

The affine-layer depth includes the output layer, following equations (2)-(6)
of arXiv:2204.09157v2. Query buffers contain fixed [r,z,t,material] points;
the operator condition is the scalar power, not a PDE differentiation axis.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .common import CoordinateScaler, ModelScales, parameter_count


def _affine_layers(input_dim: int, width: int, depth: int, latent_dim: int) -> nn.ModuleList:
    return nn.ModuleList([nn.Linear(input_dim, width),
                          *(nn.Linear(width, width) for _ in range(depth - 2)),
                          nn.Linear(width, latent_dim)])


class _ModifiedDeepONet(nn.Module):
    def __init__(self, branch_dim: int, trunk_dim: int, width: int, depth: int,
                 latent_dim: int, *, final_activation: bool) -> None:
        super().__init__()
        self.branch_encoder = nn.Linear(branch_dim, width)
        self.trunk_encoder = nn.Linear(trunk_dim, width)
        self.branch_layers = _affine_layers(branch_dim, width, depth, latent_dim)
        self.trunk_layers = _affine_layers(trunk_dim, width, depth, latent_dim)
        self.final_activation = final_activation

    def forward(self, branch: Tensor, trunk: Tensor) -> Tensor:
        encoded_u = torch.tanh(self.branch_encoder(branch))
        encoded_x = torch.tanh(self.trunk_encoder(trunk))
        h_u, h_x = branch, trunk
        for branch_layer, trunk_layer in zip(self.branch_layers[:-1], self.trunk_layers[:-1]):
            z_u = torch.tanh(branch_layer(h_u))
            z_x = torch.tanh(trunk_layer(h_x))
            h_u = (1.0 - z_u) * encoded_u + z_u * encoded_x
            h_x = (1.0 - z_x) * encoded_u + z_x * encoded_x
        branch_output = self.branch_layers[-1](h_u)
        trunk_output = self.trunk_layers[-1](h_x)
        if self.final_activation:
            branch_output, trunk_output = torch.tanh(branch_output), torch.tanh(trunk_output)
        return (branch_output * trunk_output).sum(dim=-1, keepdim=True)

    def branch_square_sum(self) -> Tensor:
        return sum(parameter.square().sum()
                   for parameter in (*self.branch_encoder.parameters(), *self.branch_layers.parameters()))


class _LinearDeepONet(nn.Module):
    def __init__(self, branch_dim: int, trunk_dim: int, width: int,
                 depth: int, latent_dim: int) -> None:
        super().__init__()
        self.branch_layers = _affine_layers(branch_dim, width, depth, latent_dim)
        self.trunk_layers = _affine_layers(trunk_dim, width, depth, latent_dim)

    def forward(self, branch: Tensor, trunk: Tensor) -> Tensor:
        for layer in self.branch_layers:
            branch = layer(branch)
        for layer in self.trunk_layers:
            trunk = layer(trunk)
        return (branch * trunk).sum(dim=-1, keepdim=True)


def _fixed_queries(points: Tensor | Sequence[Sequence[float]], name: str) -> Tensor:
    try:
        values = points if isinstance(points, Tensor) else torch.as_tensor(points, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name}查询点必须为有限数值的N乘4形状") from error
    if (values.ndim != 2 or values.shape[1] != 4 or len(values) == 0
            or values.is_complex() or values.dtype == torch.bool):
        raise ValueError(f"{name}查询点形状须为非空N乘4：[r,z,t,material]")
    values = values.detach().clone().to(dtype=torch.float64)
    if not torch.isfinite(values).all() or not ((values[:, 3] == 0) | (values[:, 3] == 1)).all():
        raise ValueError(f"{name}查询点必须有限，材料只能为0或1")
    return values


class Task11HowardComposite(nn.Module):
    """Three trainable DeepONets with one shared temperature reconstruction.

    ``subnet_outputs`` returns normalized LF, linear and nonlinear scalar fields.
    ``forward`` returns Kelvin: LF alone for ``low``, linear + nonlinear for ``high``.
    ``depth`` counts all affine layers, including the latent-output layer.
    ``query_chunk_size`` bounds total LF query rows, not rows per power group.
    """

    def __init__(
        self, *, linear_query_points: Tensor | Sequence[Sequence[float]],
        nonlinear_query_points: Tensor | Sequence[Sequence[float]],
        scales: ModelScales | Mapping[str, float] = ModelScales(),
        width: int = 128, depth: int = 4, latent_dim: int = 128,
        query_chunk_size: int = 16384,
    ) -> None:
        super().__init__()
        if (type(width) is not int or width < 1 or type(depth) is not int or depth < 2
                or type(latent_dim) is not int or latent_dim < 1):
            raise ValueError("width/latent_dim必须为正整数，depth含末层且至少为2")
        if type(query_chunk_size) is not int or query_chunk_size < 1:
            raise ValueError("query_chunk_size分块总行上限必须为正整数")
        if not isinstance(scales, ModelScales):
            scales = ModelScales(**dict(scales))
        self.scales = scales
        self.width, self.depth, self.latent_dim = width, depth, latent_dim
        self.query_chunk_size = query_chunk_size
        self.scaler = CoordinateScaler(scales)
        self.register_buffer("linear_query_points", _fixed_queries(linear_query_points, "linear"))
        self.register_buffer("nonlinear_query_points", _fixed_queries(nonlinear_query_points, "nonlinear"))
        self.low_fidelity_subnet = _ModifiedDeepONet(1, 4, width, depth, latent_dim, final_activation=False)
        self.linear_subnet = _LinearDeepONet(len(self.linear_query_points), 4, width, depth, latent_dim)
        self.nonlinear_subnet = _ModifiedDeepONet(1 + len(self.nonlinear_query_points), 4,
                                                 width, depth, latent_dim, final_activation=True)

    @staticmethod
    def _validate_coordinates(coordinates: Tensor) -> None:
        if (not isinstance(coordinates, Tensor) or coordinates.ndim != 2 or coordinates.shape[1] != 5
                or not coordinates.is_floating_point() or not torch.isfinite(coordinates).all()
                or not ((coordinates[:, 4] == 0) | (coordinates[:, 4] == 1)).all()):
            raise ValueError("coordinates须为有限五列[r,z,t,power,material]，材料只能为0或1")

    def _low_normalized(self, coordinates: Tensor) -> Tensor:
        scaled = self.scaler(coordinates)
        return self.low_fidelity_subnet(scaled[:, 3:4], scaled[:, [0, 1, 2, 4]])

    def _query_low_fidelity(self, grouped_power: Tensor, points: Tensor) -> Tensor:
        points = points.to(grouped_power)
        count, query_count = len(grouped_power), len(points)
        if count == 0:
            return self._low_normalized(grouped_power.new_empty((0, 5))).reshape(0, query_count)
        chunks = []
        # Flat indices preserve power-major/anchor-minor order while bounding
        # both coordinate construction and LF hidden activations across all powers.
        for start in range(0, count * query_count, self.query_chunk_size):
            indices = torch.arange(start, min(start + self.query_chunk_size, count * query_count),
                                   device=grouped_power.device)
            anchors = points.index_select(0, indices.remainder(query_count))
            powers = grouped_power.index_select(0, torch.div(indices, query_count, rounding_mode="floor"))
            coordinates = torch.cat((anchors[:, :3], powers, anchors[:, 3:4]), dim=-1)
            # Nonreentrant checkpoint retains LF parameter and higher-order graphs
            # even for observation batches whose input coordinates need no gradients.
            if torch.is_grad_enabled():
                chunks.append(checkpoint(self._low_normalized, coordinates, use_reentrant=False))
            else:
                chunks.append(self._low_normalized(coordinates))
        return torch.cat(chunks).reshape(count, query_count)

    def _live_query_vectors(self, power: Tensor) -> tuple[Tensor, Tensor]:
        # Only the group key is detached. LF weights/responses and query RZT stay
        # differentiable; row-wise power partials are outside the PDE contract.
        _, inverse = torch.unique(power.detach().reshape(-1), sorted=True, return_inverse=True)
        counts = torch.bincount(inverse)
        order = torch.argsort(inverse, stable=True)
        starts = torch.cumsum(counts, dim=0) - counts
        grouped_power = power.index_select(0, order.index_select(0, starts))
        linear = self._query_low_fidelity(grouped_power, self.linear_query_points)
        nonlinear = self._query_low_fidelity(grouped_power, self.nonlinear_query_points)
        return linear.index_select(0, inverse), nonlinear.index_select(0, inverse)

    def subnet_outputs(self, coordinates: Tensor) -> dict[str, Tensor]:
        """Return normalized fields, preserving the caller's row and anchor order."""
        self._validate_coordinates(coordinates)
        scaled = self.scaler(coordinates)
        branch, trunk = scaled[:, 3:4], scaled[:, [0, 1, 2, 4]]
        linear_queries, nonlinear_queries = self._live_query_vectors(coordinates[:, 3:4])
        return {"low_fidelity": self.low_fidelity_subnet(branch, trunk),
                "linear": self.linear_subnet(linear_queries, trunk),
                "nonlinear": self.nonlinear_subnet(torch.cat((branch, nonlinear_queries), dim=-1), trunk)}

    def forward(self, coordinates: Tensor, fidelity: str = "high") -> Tensor:
        if fidelity not in {"low", "high"}:
            raise ValueError("fidelity只能为low或high")
        if fidelity == "low":
            self._validate_coordinates(coordinates)
            normalized = self._low_normalized(coordinates)
        else:
            outputs = self.subnet_outputs(coordinates)
            normalized = outputs["linear"] + outputs["nonlinear"]
        return self.scales.temperature_offset_k + self.scales.temperature_scale_k * normalized

    def branch_regularization(self) -> dict[str, Tensor]:
        """Equation (10) raw branch sums, ready for separate lambda4/lambda3 weights."""
        return {"low_fidelity": self.low_fidelity_subnet.branch_square_sum(),
                "nonlinear": self.nonlinear_subnet.branch_square_sum()}

    @property
    def model_kwargs(self) -> dict[str, object]:
        return {"scales": asdict(self.scales), "width": self.width, "depth": self.depth,
                "latent_dim": self.latent_dim, "query_chunk_size": self.query_chunk_size,
                "linear_query_points": self.linear_query_points.detach().cpu().tolist(),
                "nonlinear_query_points": self.nonlinear_query_points.detach().cpu().tolist()}

    def parameter_count(self) -> int:
        return parameter_count(self)
