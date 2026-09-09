from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import TensorDataset

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.balanced_sampler import (
    balanced_material_time_indices,
    material_sample_counts,
)
from sic_cu.data.fields import load_processed_field
from sic_cu.data.splits import build_power_splits
from sic_cu.eval.protocol_checks import (
    checkpoint_provenance,
    current_protocol_fingerprints,
    validate_lf_checkpoint_provenance,
    validate_release_checkpoint,
)
from sic_cu.eval.metrics import aggregate_field_records, field_metrics
from sic_cu.losses import PhysicsLossComputer, PhysicsLossWeights
from sic_cu.models import (
    DeepONetPINN,
    LSTMPINN,
    MLPPINN,
    ModelScales,
    PRCMultifidelityModel,
)
from sic_cu.models.common import parameter_count
from sic_cu.physics.collocation import sample_collocation
from sic_cu.physics.materials import PhysicsConfigurationError, load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.train.common import physics_optimizer_step, write_config_snapshot


VALIDATION_SAMPLING_SEED = 10_000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def _physics_ready() -> PhysicsLossComputer:
    materials = load_materials()
    boundaries = load_resolved_boundary_conditions()
    training = load_yaml("configs/training.yaml")
    weights = training["loss_weights"]
    return PhysicsLossComputer(
        materials,
        boundaries,
        PhysicsLossWeights(
            pde=float(weights["pde"]),
            boundary=float(weights["boundary"]),
            initial=float(weights["initial"]),
            interface=float(weights["interface"]),
        ),
    )


def build_model(method: str, scales: ModelScales, **kwargs: Any) -> nn.Module:
    if method in {"mlp", "mlp_pinn"}:
        return MLPPINN(
            scales=scales,
            width=int(kwargs.get("width", 128)),
            depth=int(kwargs.get("depth", 5)),
            activation=str(kwargs.get("activation", "tanh")),
            include_material=bool(kwargs.get("include_material", False)),
        )
    if method in {"lstm", "lstm_pinn"}:
        return LSTMPINN(
            scales=scales,
            window=int(kwargs.get("window", 16)),
            hidden_size=int(kwargs.get("hidden_size", 96)),
            activation=str(kwargs.get("activation", "silu")),
            include_material=bool(kwargs.get("include_material", False)),
        )
    if method in {"deeponet", "deeponet_pinn"}:
        return DeepONetPINN(
            scales=scales,
            width=int(kwargs.get("width", 128)),
            latent_dim=int(kwargs.get("latent_dim", 128)),
            blocks=int(kwargs.get("blocks", 3)),
            activation=str(kwargs.get("activation", "silu")),
            include_material=bool(kwargs.get("include_material", False)),
        )
    if method == "prc_lf":
        model = PRCMultifidelityModel(
            scales=scales,
            modes=int(kwargs.get("modes", 8)),
            width=int(kwargs.get("width", 64)),
            depth=int(kwargs.get("depth", 3)),
            activation=str(kwargs.get("activation", "tanh")),
            correction_variant=str(
                kwargs.get("correction_variant", "amplitude_time")
            ),
            tau_min_s=float(kwargs.get("tau_min_s", 0.05)),
            tau_max_correction_log_scale=float(
                kwargs.get("tau_max_correction_log_scale", 0.7)
            ),
            power_reference_w=float(kwargs.get("power_reference_w", 400.0)),
            freeze_low_fidelity=False,
        )
        for parameter in model.high_fidelity_parameters():
            parameter.requires_grad_(False)
        return model
    raise ValueError(f"Pointwise trainer does not support method: {method}")


def _simulation_forward(model: nn.Module, coordinates: Tensor) -> Tensor:
    underlying = model.module if hasattr(model, "module") else model
    return (
        model(coordinates, fidelity="low")
        if isinstance(underlying, PRCMultifidelityModel)
        else model(coordinates)
    )


def load_sampled_points(
    powers: list[float],
    samples_per_power: int,
    seed: int,
    *,
    balanced: bool = True,
) -> TensorDataset:
    rng = np.random.default_rng(seed)
    coordinates: list[np.ndarray] = []
    temperatures: list[np.ndarray] = []
    for power in powers:
        path = PROJECT_ROOT / "data/processed/simulation" / f"{power:g}W.parquet"
        frame = pl.read_parquet(
            path,
            columns=[
                "r_m",
                "z_m",
                "time_s",
                "power_w",
                "material_id",
                "temperature_k",
            ],
        )
        sample_count = min(samples_per_power, frame.height)
        indices = (
            balanced_material_time_indices(
                frame["material_id"].to_numpy(),
                frame["time_s"].to_numpy(),
                sample_count,
                rng,
            )
            if balanced
            else rng.choice(frame.height, size=sample_count, replace=False)
        )
        sampled = frame[indices]
        coordinates.append(
            sampled.select("r_m", "z_m", "time_s", "power_w", "material_id").to_numpy()
        )
        temperatures.append(sampled["temperature_k"].to_numpy()[:, None])
    x = torch.from_numpy(np.concatenate(coordinates).astype(np.float32))
    y = torch.from_numpy(np.concatenate(temperatures).astype(np.float32))
    return TensorDataset(x, y)


def _rank_indices(
    size: int,
    device: torch.device,
    rank: int,
    world_size: int,
    *,
    shuffle: bool,
    seed: int = 0,
    equal_length: bool = False,
) -> Tensor:
    if shuffle:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        indices = torch.randperm(size, generator=generator, device=device)
    else:
        indices = torch.arange(size, device=device)
    if equal_length and world_size > 1:
        total_size = int(np.ceil(size / world_size)) * world_size
        if total_size > size:
            indices = torch.cat((indices, indices[: total_size - size]))
    return indices[rank::world_size]


@torch.no_grad()
def _validation_sums(
    model: nn.Module,
    coordinates: Tensor,
    target: Tensor,
    batch_size: int,
    device: torch.device,
    rank: int,
    world_size: int,
) -> Tensor:
    sums = torch.zeros(3, dtype=torch.float64, device=device)
    model.eval()
    indices = _rank_indices(
        len(coordinates), device, rank, world_size, shuffle=False, equal_length=False
    )
    for offset in range(0, len(indices), batch_size):
        batch_indices = indices[offset : offset + batch_size]
        error = _simulation_forward(
            model, coordinates.index_select(0, batch_indices)
        ) - target.index_select(
            0, batch_indices
        )
        sums[0] += error.double().pow(2).sum()
        sums[1] += error.double().abs().sum()
        sums[2] += error.numel()
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    return sums


@torch.no_grad()
def predict_field(
    model: nn.Module,
    power_w: float,
    device: torch.device,
    batch_size: int = 8192,
) -> tuple[np.ndarray, float]:
    field = load_processed_field(power_w)
    times = np.repeat(field.times_s, field.coordinates_rz_m.shape[0])
    mesh = np.tile(field.coordinates_rz_m, (len(field.times_s), 1))
    powers = np.full((len(times), 1), power_w, dtype=np.float32)
    materials = np.tile(field.material_ids, len(field.times_s))[:, None]
    coordinates = np.column_stack((mesh, times, powers, materials)).astype(np.float32)
    predictions: list[np.ndarray] = []
    model.eval()
    start = time.perf_counter()
    for offset in range(0, len(coordinates), batch_size):
        batch = torch.from_numpy(coordinates[offset : offset + batch_size]).to(device)
        predictions.append(_simulation_forward(model, batch).cpu().numpy())
    elapsed = time.perf_counter() - start
    return np.concatenate(predictions).reshape(field.temperature_k.shape), elapsed


def _evaluate_test_powers(
    model: nn.Module,
    powers: list[float],
    device: torch.device,
    batch_size: int = 8192,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for power in powers:
        field = load_processed_field(power)
        prediction, elapsed = predict_field(model, power, device, batch_size)
        records.append(
            {
                "power_w": power,
                "inference_seconds": elapsed,
                "metrics": field_metrics(
                    field.temperature_k,
                    prediction,
                    field.times_s,
                    field.material_ids,
                    field.coordinates_rz_m,
                ),
            }
        )
    return {"aggregate": aggregate_field_records(records), "per_power": records}


def train_simulation_model(
    method: str,
    seed: int,
    output_directory: str,
    epochs: int,
    samples_per_power: int,
    validation_samples_per_power: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    model_kwargs: dict[str, Any] | None = None,
    physics_collocation: int = 256,
    physics_weight: float = 1.0,
) -> dict[str, Any] | None:
    use_physics = method.endswith("_pinn")
    physics_loss = _physics_ready() if use_physics else None
    rank, local_rank, world_size = distributed_context()
    set_seed(seed)
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.cuda.reset_peak_memory_stats(device)
    splits = build_power_splits()
    fingerprints = current_protocol_fingerprints()
    train_powers = sorted(splits.simulation_train)
    validation_powers = sorted(splits.simulation_validation)
    train_dataset = load_sampled_points(train_powers, samples_per_power, seed)
    validation_dataset = load_sampled_points(
        validation_powers,
        validation_samples_per_power,
        VALIDATION_SAMPLING_SEED,
    )
    train_coordinates, train_target = (tensor.to(device) for tensor in train_dataset.tensors)
    validation_coordinates, validation_target = (
        tensor.to(device) for tensor in validation_dataset.tensors
    )
    scales = ModelScales()
    kwargs = dict(model_kwargs or {})
    if method == "prc_lf":
        kwargs.pop("include_material", None)
    else:
        kwargs.setdefault("include_material", True)
    base_model = build_model(method, scales, **kwargs).to(device)
    model: nn.Module = (
        DistributedDataParallel(base_model, device_ids=[local_rank]) if world_size > 1 else base_model
    )
    trainable_parameters = (
        list(base_model.low_fidelity_parameters())
        if isinstance(base_model, PRCMultifidelityModel)
        else list(model.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=learning_rate, weight_decay=1e-6
    )
    output_root = PROJECT_ROOT / output_directory
    checkpoint_path = output_root / "best.pt"
    log_path = output_root / "training.jsonl"
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        write_config_snapshot(output_root)
    if dist.is_initialized():
        dist.barrier()
    best_rmse = float("inf")
    epochs_without_improvement = 0
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        train_sse = torch.zeros(2, dtype=torch.float64, device=device)
        indices = _rank_indices(
            len(train_coordinates),
            device,
            rank,
            world_size,
            shuffle=True,
            seed=seed * 1_000_000 + epoch,
            equal_length=True,
        )
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset : offset + batch_size]
            coordinates = train_coordinates.index_select(0, batch_indices)
            target = train_target.index_select(0, batch_indices)
            optimizer.zero_grad(set_to_none=True)
            prediction = _simulation_forward(model, coordinates)
            loss = ((prediction - target) / scales.temperature_scale_k).pow(2).mean()
            loss.backward()
            optimizer.step()
            train_sse[0] += (prediction.detach().double() - target.double()).pow(2).sum()
            train_sse[1] += target.numel()
        physics_values: dict[str, float] = {}
        if physics_loss is not None:
            collocation = sample_collocation(
                physics_collocation,
                device,
                seed=seed * 1_000_000 + epoch * world_size + rank,
            )
            components = physics_optimizer_step(
                model,
                optimizer,
                physics_loss,
                collocation,
                physics_weight,
            )
            for name, value in components.items():
                reduced = value.detach().double()
                if dist.is_initialized():
                    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                    reduced /= world_size
                physics_values[name] = float(reduced.item())
        if dist.is_initialized():
            dist.all_reduce(train_sse, op=dist.ReduceOp.SUM)
        validation_sums = _validation_sums(
            model,
            validation_coordinates,
            validation_target,
            batch_size,
            device,
            rank,
            world_size,
        )
        train_rmse = float(torch.sqrt(train_sse[0] / train_sse[1]).item())
        validation_rmse = float(torch.sqrt(validation_sums[0] / validation_sums[2]).item())
        validation_mae = float((validation_sums[1] / validation_sums[2]).item())
        improved = validation_rmse < best_rmse - 1e-4
        if improved:
            best_rmse = validation_rmse
            epochs_without_improvement = 0
            if rank == 0:
                torch.save(
                    {
                        "schema_version": 1,
                        "method": method,
                        "seed": seed,
                        "epoch": epoch,
                        "model_state": (model.module if hasattr(model, "module") else model).state_dict(),
                        "model_kwargs": kwargs,
                        "scales": asdict(scales),
                        "validation_rmse_c": validation_rmse,
                        "train_powers_w": train_powers,
                        "validation_powers_w": validation_powers,
                        "provenance": checkpoint_provenance(
                            role="low_fidelity_simulation",
                            train_powers_w=train_powers,
                            validation_powers_w=validation_powers,
                            fingerprints=fingerprints,
                        ),
                        "material_passport": {
                            "simulation_data_used": True,
                            "experiment_data_used": False,
                            "sensor_data_used": False,
                            "physics_loss_used": use_physics,
                            "internal_experiment_truth": "not available",
                        },
                    },
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1
        if rank == 0:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "train_rmse_c": train_rmse,
                            "validation_rmse_c": validation_rmse,
                            "validation_mae_c": validation_mae,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            **{f"loss_{name}": value for name, value in physics_values.items()},
                        }
                    )
                    + "\n"
                )
        stop = torch.tensor(
            int(epochs_without_improvement >= patience), dtype=torch.int32, device=device
        )
        if dist.is_initialized():
            dist.broadcast(stop, src=0)
        if bool(stop.item()):
            break
    elapsed = time.perf_counter() - start_time
    elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        dist.barrier()
    result: dict[str, Any] | None = None
    if rank == 0:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        base_model.load_state_dict(checkpoint["model_state"])
        run_state = {
            "method": method,
            "seed": seed,
            "world_size": world_size,
            "epochs_completed": epoch,
            "best_epoch": checkpoint["epoch"],
            "best_validation_rmse_c": checkpoint["validation_rmse_c"],
            "training_seconds": float(elapsed_tensor.item()),
            "parameter_count": parameter_count(base_model),
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "configuration": {
                "epochs": epochs,
                "samples_per_power": samples_per_power,
                "validation_samples_per_power": validation_samples_per_power,
                "validation_sampling_seed": VALIDATION_SAMPLING_SEED,
                "batch_size_per_rank": batch_size,
                "learning_rate": learning_rate,
                "patience": patience,
                "model_kwargs": kwargs,
                "physics_enabled": use_physics,
                "physics_collocation_per_rank": physics_collocation if use_physics else 0,
                "physics_weight": physics_weight if use_physics else 0.0,
                "balanced_simulation_sampling": True,
            },
            "data_consumption": {
                "train_points": int(len(train_coordinates)),
                "validation_points": int(len(validation_coordinates)),
                "train_material_points": material_sample_counts(
                    train_coordinates.detach().cpu().numpy()
                ),
                "validation_material_points": material_sample_counts(
                    validation_coordinates.detach().cpu().numpy()
                ),
            },
            "material_passport": {
                "simulation_role": "low-fidelity training and frozen-power testing",
                "simulation_data_used": True,
                "experiment_data_used": False,
                "sensor_data_used": False,
                "physics_loss_used": use_physics,
                "internal_experiment_truth": "not available",
            },
        }
        (output_root / "run_state.json").write_text(
            json.dumps(run_state, indent=2), encoding="utf-8"
        )
        result = run_state | {
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "test": None,
            "test_status": "sealed_until_frozen_release",
        }
        (output_root / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return result


def evaluate_saved_simulation_model(
    checkpoint_path: str,
    output_directory: str,
    *,
    device_name: str = "cpu",
    batch_size: int = 8192,
    template_metrics_path: str | None = None,
    training_seconds: float | None = None,
    release_manifest_path: str,
) -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type == "cuda":
        device = torch.device("cuda", 0 if device.index is None else device.index)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint = torch.load(
        PROJECT_ROOT / checkpoint_path, map_location=device, weights_only=False
    )
    validate_lf_checkpoint_provenance(checkpoint)
    validate_release_checkpoint(release_manifest_path, checkpoint_path)
    method = str(checkpoint["method"])
    scales = ModelScales(**checkpoint["scales"])
    model = build_model(method, scales, **checkpoint.get("model_kwargs", {})).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    test = _evaluate_test_powers(
        model,
        sorted(build_power_splits().simulation_test),
        device,
        batch_size,
    )

    template: dict[str, Any] = {}
    if template_metrics_path is not None:
        template = json.loads(
            (PROJECT_ROOT / template_metrics_path).read_text(encoding="utf-8")
        )
    output_root = PROJECT_ROOT / output_directory
    log_path = output_root / "training.jsonl"
    epochs_completed = sum(1 for line in log_path.read_text(encoding="utf-8").splitlines() if line)
    configuration = dict(template.get("configuration", {}))
    configuration["evaluation_batch_size"] = batch_size
    passport = {
        "simulation_role": "low-fidelity training and frozen-power testing",
        **dict(checkpoint.get("material_passport", {})),
    }
    result = {
        "method": method,
        "seed": int(checkpoint["seed"]),
        "world_size": int(template.get("world_size", 1)),
        "epochs_completed": epochs_completed,
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_rmse_c": float(checkpoint["validation_rmse_c"]),
        "training_seconds": None if training_seconds is None else float(training_seconds),
        "parameter_count": parameter_count(model),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "test": test,
        "configuration": configuration,
        "recovered_evaluation": {
            "checkpoint_reused": True,
            "reason": "post-training full-field evaluation did not complete",
            "training_seconds_source": (
                "not_available"
                if training_seconds is None
                else "filesystem birth-to-final-training-log mtime, rounded to one second"
            ),
        },
        "material_passport": passport,
    }
    (output_root / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simulation-only candidate model")
    parser.add_argument(
        "--method",
        choices=(
            "mlp",
            "mlp_pinn",
            "lstm",
            "lstm_pinn",
            "deeponet",
            "deeponet_pinn",
            "prc_lf",
        ),
        default="mlp",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="reports/runs/mlp_seed0")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--samples-per-power", type=int, default=8192)
    parser.add_argument("--validation-samples-per-power", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--activation", choices=("tanh", "silu"), default="tanh")
    parser.add_argument("--window", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--modes", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument("--physics-collocation", type=int, default=256)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    args = parser.parse_args()
    try:
        model_kwargs: dict[str, Any] = {"activation": args.activation}
        if args.method in {"mlp", "mlp_pinn"}:
            model_kwargs.update(width=args.width, depth=args.depth)
        elif args.method in {"lstm", "lstm_pinn"}:
            model_kwargs.update(window=args.window, hidden_size=args.hidden_size)
        elif args.method in {"deeponet", "deeponet_pinn"}:
            model_kwargs.update(
                width=args.width,
                latent_dim=args.latent_dim,
                blocks=args.blocks,
            )
        elif args.method == "prc_lf":
            model_kwargs.update(
                modes=args.modes,
                width=args.width,
                depth=args.depth,
                correction_variant="amplitude_time",
            )
        result = train_simulation_model(
            method=args.method,
            seed=args.seed,
            output_directory=args.output,
            epochs=args.epochs,
            samples_per_power=args.samples_per_power,
            validation_samples_per_power=args.validation_samples_per_power,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            model_kwargs=model_kwargs,
            physics_collocation=args.physics_collocation,
            physics_weight=args.physics_weight,
        )
    except PhysicsConfigurationError as error:
        parser.exit(2, f"BLOCKED_UNVERIFIED_PHYSICS: {error}\n")
    if result is not None:
        print(
            json.dumps(
                {
                    "best_validation_rmse_c": result["best_validation_rmse_c"],
                    "test_status": result["test_status"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
