"""Board-only DeepONet and LF-conditioned correction for Task 10."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sic_cu.models.common import ResidualBlock


class _BoardEncoder(nn.Module):
    def __init__(self, input_dim: int, width: int, blocks: int):
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.blocks = nn.Sequential(*(ResidualBlock(width, "tanh") for _ in range(blocks)))

    def forward(self, values: Tensor) -> Tensor:
        return self.blocks(torch.tanh(self.input(values)))


class BoardDeepONet(nn.Module):
    """Input is [top depth m, time s, absorbed-equivalent q W/m2, SiC/Cu id]."""

    def __init__(self, *, width: int = 64, latent_dim: int = 64, blocks: int = 2,
                 with_lf_input: bool = False, thickness_m: float = .0175,
                 time_max_s: float = 200.0, flux_max_w_m2: float = 80000.0,
                 temperature_offset_k: float = 295.15, temperature_scale_k: float = 25.0):
        super().__init__()
        if min(width, latent_dim, thickness_m, time_max_s, flux_max_w_m2,
               temperature_scale_k) <= 0 or blocks < 0:
            raise ValueError("同板网络宽度、坐标/温度缩放均须正且残差块数非负")
        self.with_lf_input = with_lf_input
        self.thickness_m = float(thickness_m)
        self.time_max_s = float(time_max_s)
        self.flux_max_w_m2 = float(flux_max_w_m2)
        self.register_buffer("offset_k", torch.tensor(float(temperature_offset_k)))
        self.register_buffer("scale_k", torch.tensor(float(temperature_scale_k)))
        self.branch = _BoardEncoder(1, width, blocks)
        self.trunk = _BoardEncoder(4 if with_lf_input else 3, width, blocks)
        self.branch_projection = nn.Linear(width, latent_dim)
        self.trunk_projection = nn.Linear(width, latent_dim)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.branch_projection.weight)
        nn.init.zeros_(self.branch_projection.bias)

    def forward(self, coordinates: Tensor, low_temperature_k: Tensor | None = None) -> Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] != 4:
            raise ValueError("新一维板DeepONet只能接受四列[depth,time,q,material]，拒绝旧RZ")
        z, time, flux, material = coordinates.unbind(-1)
        if (not torch.isfinite(coordinates).all() or bool((z < 0).any())
                or bool((z > self.thickness_m).any()) or bool((time < 0).any())
                or bool((time > self.time_max_s).any()) or bool((flux < 20000).any())
                or bool((flux > self.flux_max_w_m2).any())
                or bool(((material != 0) & (material != 1)).any())):
            raise ValueError("同板深度/时刻/人为热通量q/材料坐标不合法")
        if self.with_lf_input != (low_temperature_k is not None):
            raise ValueError("LF校正DeepONet必须仅在有配对LF预测时提供该输入")
        branch_input = (2.0 * flux[:, None] / self.flux_max_w_m2 - 1.0)
        trunk_input = torch.stack((2.0 * z / self.thickness_m - 1.0,
                                   2.0 * time / self.time_max_s - 1.0,
                                   2.0 * material - 1.0), dim=1)
        if low_temperature_k is not None:
            if low_temperature_k.shape != (len(coordinates), 1):
                raise ValueError("LF条件须逐个同板位置返回一列温度K")
            trunk_input = torch.cat((trunk_input,
                                     (low_temperature_k - self.offset_k) / self.scale_k), 1)
        branch = self.branch_projection(self.branch(branch_input))
        trunk = self.trunk_projection(self.trunk(trunk_input))
        normalized = (branch * trunk).sum(1, keepdim=True) / branch.shape[1] ** .5
        return self.offset_k + self.scale_k * (normalized + self.bias)


class BoardMultifidelityDeepONet(nn.Module):
    """One frozen trained LF network plus a fresh HF DeepONet correction."""

    def __init__(self, low_model: BoardDeepONet, correction: BoardDeepONet, method: str):
        super().__init__()
        if method not in ("F2", "F3") or low_model.with_lf_input or not correction.with_lf_input:
            raise ValueError("只有F2/F3可用同板LF预测加HF校正，不得加载旧RZ权重")
        self.method = method
        self.low_model = low_model
        self.correction = correction
        for parameter in self.low_model.parameters():
            parameter.requires_grad_(False)

    def forward(self, coordinates: Tensor) -> Tensor:
        low = self.low_model(coordinates)
        corrected = self.correction(coordinates, low)
        return low + corrected - self.correction.offset_k
