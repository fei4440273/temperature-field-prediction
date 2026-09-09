from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch.distributed as dist
from torch import nn

from sic_cu.config import PROJECT_ROOT
from sic_cu.physics.resolution import write_resolved_physics


CONFIG_FILES = (
    "configs/geometry.yaml",
    "configs/materials.yaml",
    "configs/boundary_conditions.yaml",
    "configs/data_metadata.yaml",
    "configs/splits.yaml",
    "configs/training.yaml",
    "configs/release_manifest.yaml",
)


def write_config_snapshot(output_directory: str | Path) -> Path:
    output = Path(output_directory)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    destination = output / "config_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for relative in CONFIG_FILES:
        source = PROJECT_ROOT / relative
        content = source.read_bytes()
        (destination / source.name).write_bytes(content)
        hashes[relative] = hashlib.sha256(content).hexdigest()
    plan = PROJECT_ROOT / "temperature_field_prediction_multifidelity_plan_v2.md"
    if not plan.exists():
        raise FileNotFoundError(f"Active multifidelity plan is missing: {plan}")
    hashes[plan.name] = hashlib.sha256(plan.read_bytes()).hexdigest()
    (destination / "sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    write_resolved_physics(destination / "resolved_physics.yaml")
    return destination


def physics_optimizer_step(
    model: nn.Module,
    optimizer,
    physics_loss,
    collocation,
    multiplier: float = 1.0,
) -> dict[str, Any]:
    """Run multi-forward physics loss and explicitly average its DDP gradients."""
    optimizer.zero_grad(set_to_none=True)
    underlying = model.module if hasattr(model, "module") else model
    components = physics_loss(underlying, collocation)
    (float(multiplier) * components["physics_total"]).backward()
    if dist.is_initialized():
        world_size = dist.get_world_size()
        optimizer_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        for parameter in optimizer_parameters:
            if parameter.grad is None:
                raise RuntimeError(
                    "An optimized parameter has no physics gradient; DDP reduction would be unsafe"
                )
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad /= world_size
    optimizer.step()
    return components
