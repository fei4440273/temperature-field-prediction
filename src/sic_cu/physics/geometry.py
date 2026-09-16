from __future__ import annotations

import torch
from torch import Tensor

from sic_cu.config import load_yaml


def material_mask(coordinates_rz_m: Tensor, path: str = "configs/geometry.yaml") -> Tensor:
    if coordinates_rz_m.ndim != 2 or coordinates_rz_m.shape[1] != 2:
        raise ValueError("coordinates_rz_m must have shape [N, 2]")
    cfg = load_yaml(path)
    radius = float(cfg["silicon_carbide"]["radius_m"])
    top = float(cfg["embedding"]["sic_top_z_m"])
    bottom = float(cfg["embedding"]["sic_bottom_z_m"])
    r, z = coordinates_rz_m[:, 0], coordinates_rz_m[:, 1]
    sic = (r <= radius) & (z >= bottom) & (z <= top)
    return sic.to(torch.int64)

