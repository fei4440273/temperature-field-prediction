from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from sic_cu.config import load_yaml


@dataclass(frozen=True)
class CollocationBatch:
    interior: Tensor
    interior_material_ids: Tensor
    initial: Tensor
    axis: Tensor
    sic_top: Tensor
    copper_top: Tensor
    outer_radius: Tensor
    bottom: Tensor
    interface_sic: Tensor
    interface_copper: Tensor
    interface_normals: Tensor


def _uniform(count: int, low: float, high: float, device: torch.device, generator: torch.Generator) -> Tensor:
    return low + (high - low) * torch.rand(count, 1, device=device, generator=generator)


def _pt(r: Tensor, z: Tensor, t: Tensor, power: Tensor, material_id: Tensor | float) -> Tensor:
    if isinstance(material_id, Tensor):
        material = material_id.to(dtype=r.dtype, device=r.device).reshape(-1, 1)
    else:
        material = torch.full_like(r, float(material_id))
    return torch.cat((r, z, t, power, material), dim=1)


def sample_collocation(
    count: int,
    device: torch.device,
    seed: int,
    geometry_path: str = "configs/geometry.yaml",
    time_max_s: float = 200.0,
    power_range_w: tuple[float, float] = (10.0, 800.0),
    interface_offset_m: float = 1e-6,
) -> CollocationBatch:
    """Sample a balanced, deterministic set for all PDE and boundary terms."""
    if count < 4:
        raise ValueError("count must be at least 4")
    geometry = load_yaml(geometry_path)
    copper_radius = float(geometry["copper"]["radius_m"])
    copper_bottom = float(geometry["embedding"]["copper_bottom_z_m"])
    sic_radius = float(geometry["silicon_carbide"]["radius_m"])
    sic_bottom = float(geometry["embedding"]["sic_bottom_z_m"])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    def time(n: int) -> Tensor:
        return _uniform(n, 0.0, time_max_s, device, generator)

    def power(n: int) -> Tensor:
        return _uniform(n, power_range_w[0], power_range_w[1], device, generator)

    sic_count = count // 2
    copper_count = count - sic_count
    sic_r = sic_radius * torch.sqrt(torch.rand(sic_count, 1, device=device, generator=generator))
    sic_z = _uniform(sic_count, sic_bottom, 0.0, device, generator)

    copper_r = copper_radius * torch.sqrt(torch.rand(copper_count, 1, device=device, generator=generator))
    copper_z = _uniform(copper_count, copper_bottom, 0.0, device, generator)
    inside_sic = (copper_r <= sic_radius) & (copper_z >= sic_bottom)
    while bool(inside_sic.any()):
        n = int(inside_sic.sum().item())
        copper_r[inside_sic] = copper_radius * torch.sqrt(
            torch.rand(n, device=device, generator=generator)
        )
        copper_z[inside_sic] = _uniform(n, copper_bottom, 0.0, device, generator).reshape(-1)
        inside_sic = (copper_r <= sic_radius) & (copper_z >= sic_bottom)
    interior = torch.cat(
        (
            _pt(sic_r, sic_z, time(sic_count), power(sic_count), 1.0),
            _pt(copper_r, copper_z, time(copper_count), power(copper_count), 0.0),
        )
    )
    material_ids = torch.cat(
        (
            torch.ones(sic_count, dtype=torch.long, device=device),
            torch.zeros(copper_count, dtype=torch.long, device=device),
        )
    )

    initial_r = copper_radius * torch.sqrt(torch.rand(count, 1, device=device, generator=generator))
    initial_z = _uniform(count, copper_bottom, 0.0, device, generator)
    initial_material = ((initial_r <= sic_radius) & (initial_z >= sic_bottom)).to(initial_r)
    initial = _pt(
        initial_r,
        initial_z,
        torch.zeros_like(initial_r),
        power(count),
        initial_material,
    )
    axis_z = _uniform(count, copper_bottom, 0.0, device, generator)
    axis_material = (axis_z >= sic_bottom).to(axis_z)
    axis = _pt(
        torch.zeros(count, 1, device=device),
        axis_z,
        time(count),
        power(count),
        axis_material,
    )
    sic_top = _pt(
        _uniform(count, 0.0, sic_radius, device, generator),
        torch.zeros(count, 1, device=device),
        time(count),
        power(count),
        1.0,
    )
    copper_top = _pt(
        _uniform(count, sic_radius, copper_radius, device, generator),
        torch.zeros(count, 1, device=device),
        time(count),
        power(count),
        0.0,
    )
    outer = _pt(
        torch.full((count, 1), copper_radius, device=device),
        _uniform(count, copper_bottom, 0.0, device, generator),
        time(count),
        power(count),
        0.0,
    )
    bottom = _pt(
        _uniform(count, 0.0, copper_radius, device, generator),
        torch.full((count, 1), copper_bottom, device=device),
        time(count),
        power(count),
        0.0,
    )

    bottom_count = count // 2
    side_count = count - bottom_count
    bottom_r = _uniform(bottom_count, 0.0, sic_radius, device, generator)
    bottom_z = torch.full((bottom_count, 1), sic_bottom, device=device)
    side_r = torch.full((side_count, 1), sic_radius, device=device)
    side_z = _uniform(side_count, sic_bottom, 0.0, device, generator)
    shared_t = time(count)
    shared_p = power(count)
    interface_sic = _pt(
        torch.cat((bottom_r, side_r - interface_offset_m)),
        torch.cat((bottom_z + interface_offset_m, side_z)),
        shared_t,
        shared_p,
        1.0,
    )
    interface_copper = _pt(
        torch.cat((bottom_r, side_r + interface_offset_m)),
        torch.cat((bottom_z - interface_offset_m, side_z)),
        shared_t,
        shared_p,
        0.0,
    )
    normals = torch.cat(
        (
            torch.tensor([[0.0, -1.0]], device=device).repeat(bottom_count, 1),
            torch.tensor([[1.0, 0.0]], device=device).repeat(side_count, 1),
        )
    )
    return CollocationBatch(
        interior=interior,
        interior_material_ids=material_ids,
        initial=initial,
        axis=axis,
        sic_top=sic_top,
        copper_top=copper_top,
        outer_radius=outer,
        bottom=bottom,
        interface_sic=interface_sic,
        interface_copper=interface_copper,
        interface_normals=normals,
    )
