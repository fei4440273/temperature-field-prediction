from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.fields import load_processed_field
from sic_cu.data.splits import build_power_splits
from sic_cu.models import GNOPINN, ModelScales
from sic_cu.models.common import parameter_count
from sic_cu.models.gno_pinn import knn_edges
from sic_cu.physics.collocation import sample_collocation
from sic_cu.physics.materials import PhysicsConfigurationError
from sic_cu.train.simulation import (
    VALIDATION_SAMPLING_SEED,
    _physics_ready,
    distributed_context,
    set_seed,
)
from sic_cu.train.common import physics_optimizer_step, write_config_snapshot


class GraphPhysicsNotImplementedError(PhysicsConfigurationError):
    """Raised when a graph model lacks a valid spatial differential operator."""


def _snapshot_dataset(
    powers: list[float],
    samples_per_power: int | None,
    seed: int,
) -> TensorDataset:
    rng = np.random.default_rng(seed)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for power in powers:
        field = load_processed_field(power)
        count = len(field.times_s) if samples_per_power is None else min(samples_per_power, len(field.times_s))
        indices = np.sort(rng.choice(len(field.times_s), count, replace=False))
        inputs.append(
            np.column_stack(
                (np.full(count, power), field.times_s[indices])
            ).astype(np.float32)
        )
        targets.append(field.temperature_k[indices].astype(np.float32))
    return TensorDataset(
        torch.from_numpy(np.concatenate(inputs)),
        torch.from_numpy(np.concatenate(targets)),
    )


def _graph_coordinates(mesh: Tensor, power_time: Tensor) -> Tensor:
    return torch.cat(
        (
            mesh,
            power_time[1].expand_as(mesh[:, :1]),
            power_time[0].expand_as(mesh[:, :1]),
        ),
        dim=1,
    )


def _batch_loss(
    model: nn.Module,
    power_time: Tensor,
    targets: Tensor,
    mesh: Tensor,
    material_ids: Tensor,
    edges: Tensor,
    temperature_scale_k: float,
) -> tuple[Tensor, Tensor]:
    batch_count = len(power_time)
    node_count = len(mesh)
    coordinates = torch.cat([_graph_coordinates(mesh, item) for item in power_time])
    materials = material_ids.repeat(batch_count)
    batched_edges = torch.cat(
        [edges + index * node_count for index in range(batch_count)], dim=1
    )
    predictions = model(coordinates, materials, batched_edges).reshape(batch_count, node_count)
    loss = ((predictions - targets) / temperature_scale_k).pow(2).mean()
    return loss, predictions


@torch.no_grad()
def _validation_sums(
    model: nn.Module,
    loader: DataLoader,
    mesh: Tensor,
    material_ids: Tensor,
    edges: Tensor,
    device: torch.device,
) -> Tensor:
    sums = torch.zeros(3, dtype=torch.float64, device=device)
    model.eval()
    for power_time, target in loader:
        target = target.to(device)
        _, prediction = _batch_loss(
            model,
            power_time.to(device),
            target,
            mesh,
            material_ids,
            edges,
            1.0,
        )
        error = prediction - target
        sums[0] += error.double().pow(2).sum()
        sums[1] += error.double().abs().sum()
        sums[2] += error.numel()
    if dist.is_initialized():
        dist.all_reduce(sums)
    return sums


@torch.no_grad()
def _predict_field(
    model: nn.Module,
    power: float,
    times: np.ndarray,
    mesh: Tensor,
    material_ids: Tensor,
    edges: Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output = []
    for time_s in times:
        power_time = torch.tensor([power, time_s], dtype=torch.float32, device=device)
        output.append(
            model(_graph_coordinates(mesh, power_time), material_ids, edges).squeeze(-1).cpu().numpy()
        )
    return np.stack(output)


def train_gno_model(
    output_directory: str,
    seed: int = 0,
    epochs: int = 2000,
    snapshots_per_power: int = 32,
    validation_snapshots_per_power: int = 32,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    width: int = 128,
    layers: int = 4,
    k: int = 8,
    physics_collocation: int = 256,
    use_physics: bool = True,
    patience: int = 200,
) -> dict[str, Any] | None:
    if use_physics:
        raise GraphPhysicsNotImplementedError(
            "GNO-PINN is disabled: the current message-passing output couples nodes, so the "
            "pointwise autograd heat residual is not a valid per-node PDE derivative. Use "
            "--data-only until a graph differential/Jacobian residual is implemented."
        )
    physics = None
    rank, local_rank, world_size = distributed_context()
    set_seed(seed)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        torch.cuda.reset_peak_memory_stats(device)
    splits = build_power_splits()
    reference = load_processed_field(10.0)
    mesh = torch.from_numpy(reference.coordinates_rz_m).to(device)
    material_ids = torch.from_numpy(reference.material_ids).to(device)
    edges = knn_edges(mesh, k)
    base_model = GNOPINN(ModelScales(), width=width, layers=layers, k=k).to(device)
    model: nn.Module = (
        DistributedDataParallel(base_model, device_ids=[local_rank])
        if world_size > 1
        else base_model
    )
    train_data = _snapshot_dataset(
        sorted(splits.simulation_train), snapshots_per_power, seed
    )
    validation_data = _snapshot_dataset(
        sorted(splits.simulation_validation),
        validation_snapshots_per_power,
        VALIDATION_SAMPLING_SEED,
    )
    train_sampler = DistributedSampler(train_data, world_size, rank, True, seed) if world_size > 1 else None
    validation_sampler = DistributedSampler(validation_data, world_size, rank, False) if world_size > 1 else None
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        sampler=validation_sampler,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    output = PROJECT_ROOT / output_directory
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "training.jsonl").write_text("", encoding="utf-8")
        write_config_snapshot(output)
    if dist.is_initialized():
        dist.barrier()
    best = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_sums = torch.zeros(2, dtype=torch.float64, device=device)
        for power_time, target in train_loader:
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, prediction = _batch_loss(
                model,
                power_time.to(device),
                target,
                mesh,
                material_ids,
                edges,
                base_model.output.scale_k.item(),
            )
            loss.backward()
            optimizer.step()
            train_sums[0] += (prediction.detach().double() - target.double()).pow(2).sum()
            train_sums[1] += target.numel()
        physics_values: dict[str, float] = {}
        if physics is not None:
            collocation = sample_collocation(
                physics_collocation,
                device,
                seed=seed * 1_000_000 + epoch * world_size + rank,
            )
            components = physics_optimizer_step(
                model,
                optimizer,
                physics,
                collocation,
            )
            physics_values = {name: float(value.detach()) for name, value in components.items()}
        if dist.is_initialized():
            dist.all_reduce(train_sums)
        validation = _validation_sums(
            model,
            validation_loader,
            mesh,
            material_ids,
            edges,
            device,
        )
        validation_rmse = float(torch.sqrt(validation[0] / validation[2]))
        if validation_rmse < best - 1e-4:
            best = validation_rmse
            best_epoch = epoch
            epochs_without_improvement = 0
            if rank == 0:
                torch.save(
                    {
                        "schema_version": 1,
                        "method": "gno_pinn" if use_physics else "gno",
                        "seed": seed,
                        "epoch": epoch,
                        "model_state": base_model.state_dict(),
                        "model_kwargs": {"width": width, "layers": layers, "k": k},
                        "scales": asdict(ModelScales()),
                        "train_powers_w": sorted(splits.simulation_train),
                        "validation_powers_w": sorted(splits.simulation_validation),
                        "validation_rmse_c": best,
                        "material_passport": {
                            "simulation_data_used": True,
                            "experiment_data_used": False,
                            "sensor_data_used": False,
                            "physics_loss_used": use_physics,
                            "internal_experiment_truth": "not available",
                        },
                    },
                    output / "best.pt",
                )
        else:
            epochs_without_improvement += 1
        if rank == 0:
            record = {
                "epoch": epoch,
                "train_rmse_c": float(torch.sqrt(train_sums[0] / train_sums[1])),
                "validation_rmse_c": validation_rmse,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"loss_{name}": value for name, value in physics_values.items()},
            }
            with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        stop = torch.tensor(
            int(epochs_without_improvement >= patience),
            dtype=torch.int32,
            device=device,
        )
        if dist.is_initialized():
            dist.broadcast(stop, src=0)
        if bool(stop.item()):
            break
    elapsed = time.perf_counter() - started
    result = None
    if rank == 0:
        checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
        base_model.load_state_dict(checkpoint["model_state"])
        result = {
            "method": checkpoint["method"],
            "seed": seed,
            "world_size": world_size,
            "epochs_completed": epoch,
            "best_epoch": best_epoch,
            "best_validation_rmse_c": best,
            "training_seconds": elapsed,
            "parameter_count": parameter_count(base_model),
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "test": None,
            "test_status": "sealed_until_frozen_release",
            "configuration": {
                "epochs": epochs,
                "snapshots_per_power": snapshots_per_power,
                "validation_snapshots_per_power": validation_snapshots_per_power,
                "validation_sampling_seed": VALIDATION_SAMPLING_SEED,
                "batch_size_per_rank": batch_size,
                "learning_rate": learning_rate,
                "patience": patience,
                "width": width,
                "layers": layers,
                "k": k,
                "physics_enabled": use_physics,
                "physics_collocation_per_rank": physics_collocation if use_physics else 0,
            },
            "material_passport": {
                "simulation_role": "low-fidelity graph training and frozen-power testing",
                "simulation_data_used": True,
                "experiment_data_used": False,
                "sensor_data_used": False,
                "physics_loss_used": use_physics,
                "internal_experiment_truth": "not available",
            },
        }
        (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fixed-mesh GNO candidate")
    parser.add_argument("--output", default="reports/runs/gno_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--snapshots-per-power", type=int, default=32)
    parser.add_argument("--validation-snapshots-per-power", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--physics-collocation", type=int, default=256)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--patience", type=int, default=200)
    args = parser.parse_args()
    try:
        result = train_gno_model(
            output_directory=args.output,
            seed=args.seed,
            epochs=args.epochs,
            snapshots_per_power=args.snapshots_per_power,
            validation_snapshots_per_power=args.validation_snapshots_per_power,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            width=args.width,
            layers=args.layers,
            k=args.k,
            physics_collocation=args.physics_collocation,
            use_physics=not args.data_only,
            patience=args.patience,
        )
    except GraphPhysicsNotImplementedError as error:
        parser.exit(2, f"BLOCKED_INVALID_GRAPH_PDE: {error}\n")
    except PhysicsConfigurationError as error:
        parser.exit(2, f"BLOCKED_UNVERIFIED_PHYSICS: {error}\n")
    if result is not None:
        print(json.dumps({"best_epoch": result["best_epoch"]}, indent=2))


if __name__ == "__main__":
    main()
