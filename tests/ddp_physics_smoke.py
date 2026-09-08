"""Two-GPU optimizer smoke test with synthetic constants; not a scientific experiment."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from sic_cu.losses import PhysicsLossComputer
from sic_cu.models import MLPPINN
from sic_cu.physics import BoundaryConditions, sample_collocation
from sic_cu.physics.configuration import CoolingBoundary, LaserBoundary
from sic_cu.physics.materials import Material, MaterialProperty
from sic_cu.train.common import physics_optimizer_step


def material(name: str, density: float, conductivity: float, heat_capacity: float) -> Material:
    return Material(
        name=name,
        density_kg_m3=density,
        conductivity=MaterialProperty("constant", conductivity),
        heat_capacity=MaterialProperty("constant", heat_capacity),
    )


def main() -> None:
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model = DistributedDataParallel(
        MLPPINN(width=16, depth=2, include_material=True).to(device),
        device_ids=[local_rank],
    )
    physics = PhysicsLossComputer(
        {
            0: material("synthetic_copper", 2.0, 3.0, 5.0),
            1: material("synthetic_sic", 3.0, 4.0, 6.0),
        },
        BoundaryConditions(
            ambient_temperature_k=295.15,
            initial_temperature_k=295.15,
            laser=LaserBoundary("gaussian", 0.4, 0.005, True),
            cooling=CoolingBoundary("fixed_temperature", 295.15, None, "outer_radius"),
            top_convection_coefficient_w_m2_k=10.0,
            bottom_convection_coefficient_w_m2_k=5.0,
            side_convection_coefficient_w_m2_k=None,
            radiation_enabled=False,
            silicon_carbide_emissivity=None,
            copper_emissivity=None,
            contact_resistance_m2_k_w=None,
        ),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    components = physics_optimizer_step(
        model,
        optimizer,
        physics,
        sample_collocation(16, device, seed=rank),
    )
    checksum = torch.stack(
        [parameter.detach().double().sum() for parameter in model.parameters()]
    ).sum()
    gathered = [torch.zeros_like(checksum) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, checksum)
    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": dist.get_world_size(),
                    "finite": all(torch.isfinite(value) for value in components.values()),
                    "parameter_checksums_equal": bool(
                        torch.allclose(gathered[0], gathered[1], atol=1e-10)
                    ),
                    "scope": "synthetic_constants_software_test_only",
                },
                indent=2,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
