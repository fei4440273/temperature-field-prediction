from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from sic_cu.config import PROJECT_ROOT
from sic_cu.physics.collocation import CollocationBatch
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


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    main_plan = PROJECT_ROOT / "多保真DeepONet预测精度优化总计划与执行台账.md"
    if main_plan.exists():
        content = main_plan.read_bytes()
        hashes[main_plan.name] = hashlib.sha256(content).hexdigest()
        (destination / main_plan.name).write_bytes(content)
    (destination / "sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    write_resolved_physics(destination / "resolved_physics.yaml")
    return destination


def save_training_state(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    stage: str,
    epoch: int,
    scheduler: Any = None,
    samplers: Mapping[str, Any] | None = None,
    budget: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "training_state_schema_version": 1,
        "stage": stage,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "parameter_requires_grad": {
            name: parameter.requires_grad for name, parameter in model.named_parameters()
        },
        "sampler_epochs": {
            name: sampler.epoch for name, sampler in (samplers or {}).items()
        },
        "random_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "budget": dict(budget or {}),
        "metadata": dict(metadata or {}),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_training_state(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Any = None,
    samplers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("training_state_schema_version") != 1:
        raise ValueError("Checkpoint has no complete training state; historical best.pt is model-only")
    named_parameters = dict(model.named_parameters())
    if named_parameters.keys() != payload["parameter_requires_grad"].keys():
        raise ValueError("Training state parameter names do not match the current model")
    model.load_state_dict(payload["model_state"])
    for name, frozen in payload["parameter_requires_grad"].items():
        named_parameters[name].requires_grad_(frozen)
    optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload["scheduler_state"] is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    sampler_map = samplers or {}
    for name, epoch in payload["sampler_epochs"].items():
        if name not in sampler_map:
            raise ValueError(f"Training sampler is missing: {name}")
        sampler_map[name].set_epoch(epoch)
    state = payload["random_state"]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    return payload


def split_collocation(batch: CollocationBatch, subpackages: int) -> list[tuple[CollocationBatch, float]]:
    count = len(batch.interior)
    if not 1 <= subpackages <= count // 2:
        raise ValueError("Number of collocation subpackages must not exceed paired points")
    names = [field.name for field in fields(batch)]
    if any(len(getattr(batch, name)) != count for name in names):
        raise ValueError("Collocation fields must have the same paired point count")
    sic_groups = torch.tensor_split(torch.arange(count // 2), subpackages)
    copper_groups = torch.tensor_split(torch.arange(count // 2, count), subpackages)
    result = []
    for sic, copper in zip(sic_groups, copper_groups):
        common_count = min(len(sic), len(copper))
        paired = torch.stack((sic[:common_count], copper[:common_count]), dim=1).reshape(-1)
        indices = torch.cat((paired, sic[common_count:], copper[common_count:]))
        part = CollocationBatch(**{
            name: getattr(batch, name).index_select(0, indices.to(batch.interior.device))
            for name in names
        })
        result.append((part, len(indices) / count))
    return result


def physics_backward(
    model: nn.Module,
    optimizer,
    physics_loss,
    collocation,
    multiplier: float = 1.0,
) -> dict[str, Any]:
    """Average only the new physics gradient; keep any existing data gradient."""
    underlying = model.module if hasattr(model, "module") else model
    components = physics_loss(underlying, collocation)
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    physics_gradients = torch.autograd.grad(
        float(multiplier) * components["physics_total"],
        optimizer_parameters,
        allow_unused=True,
    )
    if dist.is_initialized():
        world_size = dist.get_world_size()
        for gradient in physics_gradients:
            if gradient is None:
                raise RuntimeError(
                    "An optimized parameter has no physics gradient; DDP reduction would be unsafe"
                )
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient /= world_size
    for parameter, gradient in zip(optimizer_parameters, physics_gradients):
        if gradient is not None:
            parameter.grad = gradient.detach() if parameter.grad is None else parameter.grad + gradient.detach()
    return components


def physics_optimizer_step(
    model: nn.Module,
    optimizer,
    physics_loss,
    collocation,
    multiplier: float = 1.0,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    components = physics_backward(model, optimizer, physics_loss, collocation, multiplier)
    optimizer.step()
    return components
