from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.splits import build_power_splits
from sic_cu.data.processed import load_processed_ir_observations
from sic_cu.models import ModelScales, SurfaceResidualMLP
from sic_cu.models.common import parameter_count
from sic_cu.models.interpolation import interpolate_simulation_power
from sic_cu.eval.protocol_checks import (
    checkpoint_provenance,
    current_protocol_fingerprints,
    validate_hf_checkpoint_provenance,
    validate_release_checkpoint,
)
from sic_cu.train.simulation import distributed_context, set_seed
from sic_cu.train.common import write_config_snapshot


def _lf_surface(frame: pl.DataFrame, simulation_powers: list[float]) -> np.ndarray:
    output = np.empty(frame.height, dtype=np.float32)
    powers = frame["power_w"].to_numpy()
    times = frame["time_s"].to_numpy()
    radii = frame["r_m"].to_numpy()
    for power in np.unique(powers):
        field = interpolate_simulation_power(float(power), simulation_powers, kind="linear")
        surface = (field.material_ids == 1) & np.isclose(
            field.coordinates_rz_m[:, 1], 0.0, atol=1e-8
        )
        order = np.argsort(field.coordinates_rz_m[surface, 0])
        surface_r = field.coordinates_rz_m[surface, 0][order]
        surface_temperature = field.temperature_k[:, surface][:, order]
        power_indices = np.flatnonzero(powers == power)
        for time_s in np.unique(times[power_indices]):
            indices = power_indices[times[power_indices] == time_s]
            high_index = int(np.searchsorted(field.times_s, time_s))
            if high_index == 0:
                profile = surface_temperature[0]
            elif high_index == len(field.times_s):
                profile = surface_temperature[-1]
            elif np.isclose(field.times_s[high_index], time_s):
                profile = surface_temperature[high_index]
            else:
                low_index = high_index - 1
                fraction = (time_s - field.times_s[low_index]) / (
                    field.times_s[high_index] - field.times_s[low_index]
                )
                profile = surface_temperature[low_index] + fraction * (
                    surface_temperature[high_index] - surface_temperature[low_index]
                )
            output[indices] = np.interp(radii[indices], surface_r, profile)
    return output


def _dataset(frame: pl.DataFrame, simulation_powers: list[float]) -> TensorDataset:
    frame = frame.with_columns(
        (
            pl.col("frame_weight")
            / pl.col("frame_weight").sum().over("power_w")
        )
        .cast(pl.Float32)
        .alias("condition_weight")
    )
    lf = _lf_surface(frame, simulation_powers)
    inputs = frame.select("r_m", "time_s", "power_w").to_numpy().astype(np.float32)
    target = frame["temperature_mean_k"].to_numpy().astype(np.float32)
    residual = target - lf
    weight = frame["condition_weight"].to_numpy().astype(np.float32)
    return TensorDataset(
        torch.from_numpy(inputs),
        torch.from_numpy(lf[:, None]),
        torch.from_numpy(target[:, None]),
        torch.from_numpy(residual[:, None]),
        torch.from_numpy(weight[:, None]),
    )


@torch.no_grad()
def _evaluation_sums(model: nn.Module, loader: DataLoader, device: torch.device) -> Tensor:
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    model.eval()
    for inputs, lf, target, _, weight in loader:
        prediction = lf.to(device) + model(inputs.to(device))
        error = prediction - target.to(device)
        weight = weight.to(device)
        sums[0] += (weight.double() * error.double().pow(2)).sum()
        sums[1] += (weight.double() * error.double().abs()).sum()
        sums[2] += weight.double().sum()
        sums[3] += error.numel()
    if dist.is_initialized():
        dist.all_reduce(sums)
    return sums


@torch.no_grad()
def _per_power_metrics(
    model: nn.Module,
    frame: pl.DataFrame,
    simulation_powers: list[float],
    device: torch.device,
) -> list[dict[str, float]]:
    records = []
    for power in sorted(frame["power_w"].unique().to_list()):
        power_frame = frame.filter(pl.col("power_w") == power)
        data = _dataset(power_frame, simulation_powers)
        inputs, lf, target, _, weight = data.tensors
        prediction = lf.to(device) + model(inputs.to(device))
        prediction_cpu = prediction.cpu()
        error = prediction_cpu - target
        denominator = float(weight.sum())
        times = power_frame["time_s"].to_numpy()
        radii = power_frame["r_m"].to_numpy()
        predicted_values = prediction_cpu.numpy().reshape(-1)
        target_values = target.numpy().reshape(-1)
        peak_errors = []
        peak_relative_errors = []
        gradient_absolute = []
        for time_s in np.unique(times):
            indices = np.flatnonzero(times == time_s)
            order = indices[np.argsort(radii[indices])]
            peak_error = float(
                predicted_values[indices].max() - target_values[indices].max()
            )
            peak_errors.append(peak_error)
            peak_relative_errors.append(
                100.0
                * abs(peak_error)
                / max(abs(float(target_values[indices].max()) - 273.15), 1e-12)
            )
            if len(order) > 1:
                predicted_gradient = np.gradient(
                    predicted_values[order], radii[order], edge_order=1
                )
                target_gradient = np.gradient(target_values[order], radii[order], edge_order=1)
                gradient_absolute.append(
                    float(np.mean(np.abs(predicted_gradient - target_gradient)) * 1e-3)
                )
        records.append(
            {
                "power_w": float(power),
                "rmse_c": float(torch.sqrt((weight * error.pow(2)).sum() / denominator)),
                "mae_c": float((weight * error.abs()).sum() / denominator),
                "peak_mae_c": float(np.mean(np.abs(peak_errors))),
                "peak_max_abs_error_c": float(np.max(np.abs(peak_errors))),
                "peak_mean_relative_error_percent": float(
                    np.mean(peak_relative_errors)
                ),
                "peak_max_relative_error_percent": float(np.max(peak_relative_errors)),
                "radial_gradient_mae_c_per_mm": float(np.mean(gradient_absolute)),
            }
        )
    return records


def train_surface_residual(
    output_directory: str,
    seed: int = 0,
    epochs: int = 1500,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    patience: int = 200,
    width: int = 128,
    depth: int = 4,
    evaluate_test: bool = False,
) -> dict[str, Any] | None:
    if evaluate_test:
        raise ValueError(
            "Training-time test evaluation is disabled; freeze a release before test access"
        )
    rank, local_rank, world_size = distributed_context()
    set_seed(seed)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        torch.cuda.reset_peak_memory_stats(device)
    splits = build_power_splits()
    fingerprints = current_protocol_fingerprints()
    simulation_powers = sorted(splits.simulation_train)
    ir = load_processed_ir_observations()
    train_frame = ir.filter(pl.col("split") == "train")
    validation_frame = ir.filter(pl.col("split") == "validation")
    train_data = _dataset(train_frame, simulation_powers)
    validation_data = _dataset(validation_frame, simulation_powers)
    train_sampler = DistributedSampler(train_data, world_size, rank, True, seed) if world_size > 1 else None
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
        pin_memory=device.type == "cuda",
    )
    scales = ModelScales()
    base_model = SurfaceResidualMLP(scales, width, depth).to(device)
    model: nn.Module = DistributedDataParallel(base_model, device_ids=[local_rank]) if world_size > 1 else base_model
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
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_weighted_sse = torch.zeros(2, dtype=torch.float64, device=device)
        for inputs, _, _, residual, weight in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            residual = residual.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            error = model(inputs) - residual
            loss = (weight * (error / base_model.residual_scale_k).pow(2)).sum() / weight.sum()
            loss.backward()
            optimizer.step()
            train_weighted_sse[0] += (weight.double() * error.detach().double().pow(2)).sum()
            train_weighted_sse[1] += weight.double().sum()
        if dist.is_initialized():
            dist.all_reduce(train_weighted_sse)
        validation = _evaluation_sums(model, validation_loader, device)
        validation_rmse = float(torch.sqrt(validation[0] / validation[2]))
        validation_mae = float(validation[1] / validation[2])
        if validation_rmse < best - 1e-4:
            best = validation_rmse
            best_epoch = epoch
            stale = 0
            if rank == 0:
                torch.save(
                    {
                        "schema_version": 1,
                        "method": "surface_residual_mlp",
                        "seed": seed,
                        "epoch": epoch,
                        "model_state": base_model.state_dict(),
                        "model_kwargs": {"width": width, "depth": depth, "activation": "tanh"},
                        "scales": asdict(scales),
                        "simulation_train_powers_w": simulation_powers,
                        "hf_train_powers_w": sorted(splits.hf_train),
                        "hf_validation_powers_w": sorted(splits.hf_validation),
                        "validation_rmse_c": best,
                        "provenance": checkpoint_provenance(
                            role="surface_multifidelity",
                            train_powers_w=splits.hf_train,
                            validation_powers_w=splits.hf_validation,
                            fingerprints=fingerprints,
                        ),
                        "scope": "SiC top surface only",
                        "material_passport": {
                            "simulation_data_used": True,
                            "experiment_data_used": True,
                            "sensor_data_used": False,
                            "physics_loss_used": False,
                            "internal_experiment_truth": "not available",
                        },
                    },
                    output / "best.pt",
                )
        else:
            stale += 1
        if rank == 0:
            record = {
                "epoch": epoch,
                "train_residual_rmse_c": float(
                    torch.sqrt(train_weighted_sse[0] / train_weighted_sse[1])
                ),
                "validation_surface_rmse_c": validation_rmse,
                "validation_surface_mae_c": validation_mae,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        stop = torch.tensor(int(stale >= patience), device=device)
        if dist.is_initialized():
            dist.broadcast(stop, src=0)
        if bool(stop):
            break
    result = None
    if rank == 0:
        checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
        base_model.load_state_dict(checkpoint["model_state"])
        validation_records = _per_power_metrics(
            base_model,
            validation_frame,
            simulation_powers,
            device,
        )
        result = {
            "method": "deterministic_multifidelity_surface_residual",
            "scope": "SiC top surface only; no internal-field claim",
            "seed": seed,
            "world_size": world_size,
            "epochs_completed": epoch,
            "best_epoch": best_epoch,
            "best_validation_rmse_c": best,
            "training_seconds": time.perf_counter() - started,
            "parameter_count": parameter_count(base_model),
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "validation": {
                "aggregate_rmse_c": float(
                    np.mean([item["rmse_c"] for item in validation_records])
                ),
                "aggregate_mae_c": float(
                    np.mean([item["mae_c"] for item in validation_records])
                ),
                "aggregate_peak_mae_c": float(
                    np.mean([item["peak_mae_c"] for item in validation_records])
                ),
                "aggregate_peak_relative_error_percent": float(
                    np.mean(
                        [
                            item["peak_mean_relative_error_percent"]
                            for item in validation_records
                        ]
                    )
                ),
                "aggregate_radial_gradient_mae_c_per_mm": float(
                    np.mean(
                        [
                            item["radial_gradient_mae_c_per_mm"]
                            for item in validation_records
                        ]
                    )
                ),
                "per_power": validation_records,
            },
            "test": None,
            "test_status": "sealed_until_frozen_release",
            "material_passport": {
                "simulation_role": "low-fidelity surface baseline",
                "experiment_role": "SiC top-surface correction",
                "simulation_data_used": True,
                "experiment_data_used": True,
                "sensor_data_used": False,
                "physics_loss_used": False,
                "internal_field_truth": "not available",
            },
        }
        (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return result


def recompute_surface_run_metrics(
    output_directory: str,
    release_manifest_path: str,
) -> dict[str, Any]:
    """Re-evaluate a saved run on the untouched test powers using current metrics."""
    output = PROJECT_ROOT / output_directory
    metrics_path = output / "metrics.json"
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    validate_hf_checkpoint_provenance(checkpoint)
    validate_release_checkpoint(release_manifest_path, output / "best.pt")
    scales = ModelScales(**checkpoint["scales"])
    kwargs = checkpoint["model_kwargs"]
    model = SurfaceResidualMLP(scales, **kwargs)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    splits = build_power_splits()
    checkpoint_train = {
        round(float(value), 4) for value in checkpoint.get("hf_train_powers_w", [])
    }
    checkpoint_validation = {
        round(float(value), 4)
        for value in checkpoint.get("hf_validation_powers_w", [])
    }
    if (
        checkpoint_train != set(splits.hf_train)
        or checkpoint_validation != set(splits.hf_validation)
    ):
        raise RuntimeError(
            "Surface checkpoint does not match the current fixed train/validation protocol"
        )
    test_frame = load_processed_ir_observations("test")
    records = _per_power_metrics(
        model,
        test_frame,
        sorted(splits.simulation_train),
        torch.device("cpu"),
    )
    result["test"] = {
        "aggregate_rmse_c": float(np.mean([item["rmse_c"] for item in records])),
        "aggregate_mae_c": float(np.mean([item["mae_c"] for item in records])),
        "aggregate_peak_mae_c": float(np.mean([item["peak_mae_c"] for item in records])),
        "aggregate_peak_relative_error_percent": float(
            np.mean([item["peak_mean_relative_error_percent"] for item in records])
        ),
        "aggregate_radial_gradient_mae_c_per_mm": float(
            np.mean([item["radial_gradient_mae_c_per_mm"] for item in records])
        ),
        "per_power": records,
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the deterministic MF surface baseline")
    parser.add_argument("--output", default="reports/runs/surface_residual_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args()
    result = train_surface_residual(
        args.output,
        args.seed,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.patience,
        args.width,
        args.depth,
        False,
    )
    if result is not None:
        print(json.dumps(result["validation"], indent=2))


if __name__ == "__main__":
    main()
