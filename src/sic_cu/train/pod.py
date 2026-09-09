from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import TensorDataset

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.fields import load_processed_field
from sic_cu.data.splits import build_power_splits
from sic_cu.eval.pod_analysis import run_pod_analysis
from sic_cu.models import MaterialWisePODPINN, ModelScales, PODPINN, load_pod_basis
from sic_cu.models.common import parameter_count
from sic_cu.physics.collocation import sample_collocation
from sic_cu.physics.materials import PhysicsConfigurationError
from sic_cu.train.simulation import _physics_ready, _rank_indices, distributed_context, set_seed
from sic_cu.train.common import physics_optimizer_step, write_config_snapshot


def _coefficient_dataset(
    powers: list[float],
    model: PODPINN | MaterialWisePODPINN,
) -> TensorDataset:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    truncation_sse: list[np.ndarray] = []
    for power in powers:
        field = load_processed_field(power)
        inputs.append(
            np.column_stack(
                (np.full(len(field.times_s), power), field.times_s)
            ).astype(np.float32)
        )
        if isinstance(model, PODPINN):
            mean = model.mean_k.cpu().numpy().reshape(-1)
            mode_shapes = model.modes.cpu().numpy()
            coefficients = (field.temperature_k - mean) @ mode_shapes
            reconstruction = mean[None, :] + coefficients @ mode_shapes.T
            residual_sse = np.sum((field.temperature_k - reconstruction) ** 2, axis=1)
        else:
            material_coefficients = []
            residual_sse = np.zeros(len(field.times_s), dtype=np.float64)
            for material_id, submodel in (
                (0, model.copper),
                (1, model.silicon_carbide),
            ):
                mask = field.material_ids == material_id
                mean = submodel.mean_k.cpu().numpy().reshape(-1)
                mode_shapes = submodel.modes.cpu().numpy()
                coefficients_for_material = (field.temperature_k[:, mask] - mean) @ mode_shapes
                material_coefficients.append(coefficients_for_material)
                reconstruction = mean[None, :] + coefficients_for_material @ mode_shapes.T
                residual_sse += np.sum(
                    (field.temperature_k[:, mask] - reconstruction) ** 2, axis=1
                )
            coefficients = np.concatenate(material_coefficients, axis=1)
        targets.append(
            (coefficients / model.scales.temperature_scale_k).astype(np.float32)
        )
        truncation_sse.append(residual_sse.astype(np.float64)[:, None])
    return TensorDataset(
        torch.from_numpy(np.concatenate(inputs)),
        torch.from_numpy(np.concatenate(targets)),
        torch.from_numpy(np.concatenate(truncation_sse)),
    )


@torch.no_grad()
def _coefficient_validation(
    model: nn.Module,
    power_time: Tensor,
    target: Tensor,
    truncation_sse: Tensor,
    node_count: int,
    batch_size: int,
    device: torch.device,
    rank: int,
    world_size: int,
) -> Tensor:
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    model.eval()
    indices = _rank_indices(
        len(power_time), device, rank, world_size, shuffle=False, equal_length=False
    )
    for offset in range(0, len(indices), batch_size):
        batch_indices = indices[offset : offset + batch_size]
        error = model(power_time.index_select(0, batch_indices), True) - target.index_select(
            0, batch_indices
        )
        sums[0] += error.double().pow(2).sum()
        sums[1] += error.numel()
        sums[2] += (
            error.double().pow(2).sum() * model.module.scales.temperature_scale_k**2
            if hasattr(model, "module")
            else error.double().pow(2).sum() * model.scales.temperature_scale_k**2
        ) + truncation_sse.index_select(0, batch_indices).sum()
        sums[3] += len(batch_indices) * node_count
    if dist.is_initialized():
        dist.all_reduce(sums)
    return sums


@torch.no_grad()
def _predict_mesh(
    model: PODPINN | MaterialWisePODPINN,
    power: float,
    times: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    power_time = torch.from_numpy(
        np.column_stack((np.full(len(times), power), times)).astype(np.float32)
    ).to(device)
    if isinstance(model, PODPINN):
        coefficients = model.predict_coefficients(power_time).cpu().numpy()
        mean = model.mean_k.cpu().numpy().reshape(-1)
        mode_shapes = model.modes.cpu().numpy()
        return mean[None, :] + model.scales.temperature_scale_k * coefficients @ mode_shapes.T
    reference = load_processed_field(10.0)
    output = np.empty((len(times), len(reference.material_ids)), dtype=np.float32)
    offset = 0
    for material_id, submodel in (
        (0, model.copper),
        (1, model.silicon_carbide),
    ):
        count = submodel.modes.shape[1]
        coefficients = model.predict_coefficients(power_time).cpu().numpy()[:, offset : offset + count]
        mean = submodel.mean_k.cpu().numpy().reshape(-1)
        mode_shapes = submodel.modes.cpu().numpy()
        output[:, reference.material_ids == material_id] = (
            mean[None, :]
            + model.scales.temperature_scale_k * coefficients @ mode_shapes.T
        )
        offset += count
    return output


def train_pod_model(
    output_directory: str,
    seed: int = 0,
    epochs: int = 2000,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    modes: int = 20,
    width: int = 128,
    physics_collocation: int = 256,
    use_physics: bool = True,
    basis_variant: str = "global",
    patience: int = 200,
) -> dict[str, Any] | None:
    physics = _physics_ready() if use_physics else None
    rank, local_rank, world_size = distributed_context()
    set_seed(seed)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        torch.cuda.reset_peak_memory_stats(device)
    basis_path = PROJECT_ROOT / "data/cache/pod/global.npz"
    if not basis_path.exists():
        if rank == 0:
            run_pod_analysis()
        if dist.is_initialized():
            dist.barrier()
    if basis_variant == "global":
        base_model: PODPINN | MaterialWisePODPINN = PODPINN(
            load_pod_basis(basis_path), modes=modes, width=width
        ).to(device)
        basis_paths: str | dict[str, str] = str(basis_path.relative_to(PROJECT_ROOT))
    elif basis_variant == "material_wise":
        copper_path = PROJECT_ROOT / "data/cache/pod/copper.npz"
        sic_path = PROJECT_ROOT / "data/cache/pod/silicon_carbide.npz"
        base_model = MaterialWisePODPINN(
            load_pod_basis(copper_path),
            load_pod_basis(sic_path),
            modes=modes,
            width=width,
        ).to(device)
        basis_paths = {
            "copper": str(copper_path.relative_to(PROJECT_ROOT)),
            "silicon_carbide": str(sic_path.relative_to(PROJECT_ROOT)),
        }
    else:
        raise ValueError("basis_variant must be global or material_wise")
    model: nn.Module = (
        DistributedDataParallel(base_model, device_ids=[local_rank])
        if world_size > 1
        else base_model
    )
    splits = build_power_splits()
    train_data = _coefficient_dataset(sorted(splits.simulation_train), base_model)
    validation_data = _coefficient_dataset(sorted(splits.simulation_validation), base_model)
    train_power_time, train_target, _ = (tensor.to(device) for tensor in train_data.tensors)
    validation_power_time, validation_target, validation_truncation_sse = (
        tensor.to(device) for tensor in validation_data.tensors
    )
    node_count = (
        int(base_model.mesh_rz_m.shape[0])
        if isinstance(base_model, PODPINN)
        else int(
            base_model.copper.mesh_rz_m.shape[0]
            + base_model.silicon_carbide.mesh_rz_m.shape[0]
        )
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
        model.train()
        train_sums = torch.zeros(2, dtype=torch.float64, device=device)
        indices = _rank_indices(
            len(train_power_time),
            device,
            rank,
            world_size,
            shuffle=True,
            seed=seed * 1_000_000 + epoch,
            equal_length=True,
        )
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset : offset + batch_size]
            power_time = train_power_time.index_select(0, batch_indices)
            target = train_target.index_select(0, batch_indices)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(power_time, True)
            loss = (prediction - target).pow(2).mean()
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
        validation = _coefficient_validation(
            model,
            validation_power_time,
            validation_target,
            validation_truncation_sse,
            node_count,
            batch_size,
            device,
            rank,
            world_size,
        )
        validation_coefficient_rmse = float(torch.sqrt(validation[0] / validation[1]))
        validation_field_rmse = float(torch.sqrt(validation[2] / validation[3]))
        if validation_field_rmse < best - 1e-4:
            best = validation_field_rmse
            best_epoch = epoch
            epochs_without_improvement = 0
            if rank == 0:
                torch.save(
                    {
                        "schema_version": 1,
                        "method": "pod_pinn" if use_physics else "pod",
                        "seed": seed,
                        "epoch": epoch,
                        "model_state": base_model.state_dict(),
                        "model_kwargs": {"modes": modes, "width": width},
                        "basis_variant": basis_variant,
                        "basis_path": basis_paths,
                        "scales": asdict(base_model.scales),
                        "train_powers_w": sorted(splits.simulation_train),
                        "validation_powers_w": sorted(splits.simulation_validation),
                        "validation_field_rmse_c": best,
                        "validation_coefficient_rmse": validation_coefficient_rmse,
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
                "train_coefficient_rmse": float(torch.sqrt(train_sums[0] / train_sums[1])),
                "validation_coefficient_rmse": validation_coefficient_rmse,
                "validation_field_rmse_c": validation_field_rmse,
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
            "best_validation_field_rmse_c": best,
            "best_validation_coefficient_rmse": checkpoint[
                "validation_coefficient_rmse"
            ],
            "training_seconds": elapsed,
            "parameter_count": parameter_count(base_model),
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "test": None,
            "test_status": "sealed_until_frozen_release",
            "configuration": {
                "epochs": epochs,
                "batch_size_per_rank": batch_size,
                "learning_rate": learning_rate,
                "patience": patience,
                "modes_per_material_or_global": modes,
                "width": width,
                "basis_variant": basis_variant,
                "physics_enabled": use_physics,
                "physics_collocation_per_rank": physics_collocation if use_physics else 0,
            },
            "material_passport": {
                "simulation_role": "POD fit, training, and frozen-power testing",
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
    parser = argparse.ArgumentParser(description="Train the POD coefficient candidate")
    parser.add_argument("--output", default="reports/runs/pod_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--physics-collocation", type=int, default=256)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument(
        "--basis-variant",
        choices=("global", "material_wise"),
        default="global",
    )
    args = parser.parse_args()
    try:
        result = train_pod_model(
            output_directory=args.output,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            modes=args.modes,
            width=args.width,
            physics_collocation=args.physics_collocation,
            use_physics=not args.data_only,
            basis_variant=args.basis_variant,
            patience=args.patience,
        )
    except PhysicsConfigurationError as error:
        parser.exit(2, f"BLOCKED_UNVERIFIED_PHYSICS: {error}\n")
    if result is not None:
        print(json.dumps({"best_epoch": result["best_epoch"]}, indent=2))


if __name__ == "__main__":
    main()
