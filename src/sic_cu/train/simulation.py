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
from sic_cu.data.common import sha256_file
from sic_cu.data.balanced_sampler import (
    balanced_material_time_indices,
    balanced_material_time_space_indices,
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
from sic_cu.train.common import (
    CONFIG_FILES, load_training_state, physics_optimizer_step, resolve_device,
    save_training_state, write_config_snapshot,
)


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


def validate_locked_lf_start(path: str | Path, *, seed: int) -> dict[str, Any]:
    manifest = load_yaml("reports/development_v4/baseline_manifest.yaml")
    entries = [item for item in manifest["checkpoints"] if item["seed"] == seed]
    if len(entries) != 1:
        raise ValueError("No uniquely locked B0 and LF checkpoint for this seed")
    from sic_cu.train.multifidelity import validate_locked_b0_start

    validate_locked_b0_start(
        PROJECT_ROOT / entries[0]["hf_checkpoint"], path, seed=seed,
    )
    return torch.load(path, map_location="cpu", weights_only=False)


def lf_physics_enabled(method: str, mode: str) -> bool:
    if mode not in {"original", "none"}:
        raise ValueError(f"Unknown LF physics mode: {mode}")
    return method.endswith("_pinn") and mode == "original"


def load_sampled_points(
    powers: list[float],
    samples_per_power: int,
    seed: int,
    *,
    balanced: bool = True,
    sampling_mode: str | None = None,
) -> TensorDataset:
    mode = sampling_mode or ("material_time" if balanced else "uniform")
    if mode not in {"uniform", "material_time", "material_time_space"}:
        raise ValueError(f"Unknown LF simulation sampling mode: {mode}")
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
        if mode == "material_time_space":
            indices = balanced_material_time_space_indices(
                frame["material_id"].to_numpy(), frame["time_s"].to_numpy(),
                frame["r_m"].to_numpy(), frame["z_m"].to_numpy(), sample_count, rng,
            )
        elif mode == "material_time":
            indices = balanced_material_time_indices(
                frame["material_id"].to_numpy(),
                frame["time_s"].to_numpy(),
                sample_count,
                rng,
            )
        else:
            indices = rng.choice(frame.height, size=sample_count, replace=False)
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
    initial_checkpoint: str | None = None,
    sampling_mode: str = "material_time",
    lf_physics_mode: str = "original",
    resume_training_checkpoint: str | None = None,
    session_epoch_limit: int | None = None,
) -> dict[str, Any] | None:
    if initial_checkpoint is not None and (method != "deeponet_pinn" or lf_physics_mode != "none"):
        raise ValueError("Locked LF continuation keeps the DeepONet and disables nominal HF physics")
    if (resume_training_checkpoint is not None or session_epoch_limit is not None) and initial_checkpoint is None:
        raise ValueError("LF training resume or session limit requires a locked LF start")
    if session_epoch_limit is not None and session_epoch_limit < 1:
        raise ValueError("LF session epoch limit must be positive")
    if sampling_mode not in {"material_time", "material_time_space"}:
        raise ValueError("LF sampling must use original material/time or space-balanced strata")
    use_physics = lf_physics_enabled(method, lf_physics_mode)
    physics_loss = _physics_ready() if use_physics else None
    rank, local_rank, world_size = distributed_context()
    if initial_checkpoint is not None and world_size > 1:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise ValueError("Locked LF continuation currently requires one GPU for full random-state recovery")
    set_seed(seed)
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.cuda.reset_peak_memory_stats(device)
    splits = build_power_splits()
    fingerprints = current_protocol_fingerprints()
    historical_lf = (
        None if initial_checkpoint is None else validate_locked_lf_start(initial_checkpoint, seed=seed)
    )
    train_powers = sorted(splits.simulation_train)
    validation_powers = sorted(splits.simulation_validation)
    train_dataset = load_sampled_points(
        train_powers, samples_per_power, seed, sampling_mode=sampling_mode,
    )
    validation_dataset = load_sampled_points(
        validation_powers,
        validation_samples_per_power,
        VALIDATION_SAMPLING_SEED,
    )
    train_coordinates, train_target = (tensor.to(device) for tensor in train_dataset.tensors)
    validation_coordinates, validation_target = (
        tensor.to(device) for tensor in validation_dataset.tensors
    )
    scales = (
        ModelScales() if historical_lf is None else ModelScales(**historical_lf["scales"])
    )
    kwargs = dict(
        model_kwargs or {}
        if historical_lf is None else historical_lf["model_kwargs"]
    )
    if historical_lf is not None and model_kwargs is not None and model_kwargs != kwargs:
        raise ValueError("Locked LF continuation cannot change pretrained DeepONet architecture")
    if method == "prc_lf":
        kwargs.pop("include_material", None)
    else:
        kwargs.setdefault("include_material", True)
    base_model = build_model(method, scales, **kwargs).to(device)
    if historical_lf is not None:
        base_model.load_state_dict(historical_lf["model_state"])
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
    if historical_lf is not None:
        if resume_training_checkpoint is None and output_root.exists():
            raise FileExistsError(f"Locked LF output already exists: {output_root}")
        if resume_training_checkpoint is not None and not output_root.is_dir():
            raise FileNotFoundError(f"Locked LF resume directory is missing: {output_root}")
    source_hashes = {
        name: sha256_file(PROJECT_ROOT / name) for name in (
            "src/sic_cu/train/simulation.py",
            "src/sic_cu/data/balanced_sampler.py",
            "src/sic_cu/train/common.py",
        )
    } if historical_lf is not None else {}
    origin_hash = None if historical_lf is None else sha256_file(initial_checkpoint)
    total_train_exposure = 0
    total_optimizer_steps = 0
    if rank == 0:
        if resume_training_checkpoint is None:
            output_root.mkdir(parents=True, exist_ok=True)
            log_path.write_text("", encoding="utf-8")
            write_config_snapshot(output_root)
            if historical_lf is not None:
                save_training_state(
                    output_root / "阶段_初始.pt", base_model, optimizer,
                    stage="low_fidelity", epoch=0, budget={"LF轮次": epochs},
                    metadata={
                        "起点LF哈希": origin_hash, "源码哈希": source_hashes,
                        "采样方式": sampling_mode, "物理模式": lf_physics_mode,
                        "LF最佳验证RMSE": None, "无改善轮次": 0,
                        "累计训练点": 0, "累计优化步": 0,
                    },
                )
    if dist.is_initialized():
        dist.barrier()
    best_rmse = float("inf")
    epochs_without_improvement = 0
    resume_epoch = 0
    if resume_training_checkpoint is not None:
        checkpoint_file = Path(resume_training_checkpoint)
        if not checkpoint_file.is_absolute():
            checkpoint_file = PROJECT_ROOT / checkpoint_file
        if checkpoint_file.resolve().parent != output_root.resolve() or checkpoint_file.name not in (
            "阶段_初始.pt", "阶段_最近.pt"
        ):
            raise ValueError("LF resume must use initial or latest state in its original run")
        hashes = json.loads((output_root / "config_snapshot/sha256.json").read_text(encoding="utf-8"))
        if any(sha256_file(PROJECT_ROOT / name) != hashes[name] for name in CONFIG_FILES):
            raise ValueError("LF continuation configuration changed after the run started")
        saved = load_training_state(checkpoint_file, base_model, optimizer)
        if (
            saved["stage"] != "low_fidelity" or saved["budget"] != {"LF轮次": epochs}
            or saved["metadata"].get("起点LF哈希") != origin_hash
            or saved["metadata"].get("源码哈希") != source_hashes
            or saved["metadata"].get("采样方式") != sampling_mode
            or saved["metadata"].get("物理模式") != lf_physics_mode
        ):
            raise ValueError("LF resume state differs from its fixed budget or checkpoint")
        from sic_cu.train.multifidelity import reconcile_correction_resume_log

        resume_epoch = int(saved["epoch"])
        reconcile_correction_resume_log(
            output_root, epoch=resume_epoch, state_file=checkpoint_file,
        )
        previous_best = saved["metadata"]["LF最佳验证RMSE"]
        best_rmse = float("inf") if previous_best is None else float(previous_best)
        epochs_without_improvement = int(saved["metadata"]["无改善轮次"])
        total_train_exposure = int(saved["metadata"]["累计训练点"])
        total_optimizer_steps = int(saved["metadata"]["累计优化步"])
        if resume_epoch >= epochs:
            raise ValueError("LF run has already reached its registered budget")
    session_last_epoch = (
        epochs if session_epoch_limit is None else min(epochs, resume_epoch + session_epoch_limit)
    )
    start_time = time.perf_counter()
    early_stopped = False
    for epoch in range(resume_epoch + 1, session_last_epoch + 1):
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
        epoch_optimizer_steps = 0
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset : offset + batch_size]
            coordinates = train_coordinates.index_select(0, batch_indices)
            target = train_target.index_select(0, batch_indices)
            optimizer.zero_grad(set_to_none=True)
            prediction = _simulation_forward(model, coordinates)
            loss = ((prediction - target) / scales.temperature_scale_k).pow(2).mean()
            loss.backward()
            optimizer.step()
            epoch_optimizer_steps += 1
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
            epoch_optimizer_steps += 1
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
                        **({"lf_lineage": {
                            "locked_lf_start_sha256": origin_hash,
                            "simulation_sampling_mode": sampling_mode,
                            "lf_physics_mode": lf_physics_mode,
                            "logical_budget_epochs": epochs,
                            "source_hashes": source_hashes,
                        }} if historical_lf is not None else {}),
                    },
                    checkpoint_path,
                )
                if historical_lf is not None:
                    save_training_state(
                        output_root / "阶段_LF观测最佳.pt", base_model, optimizer,
                        stage="low_fidelity", epoch=epoch, budget={"LF轮次": epochs},
                        metadata={
                            "起点LF哈希": origin_hash, "源码哈希": source_hashes,
                            "采样方式": sampling_mode, "物理模式": lf_physics_mode,
                            "最佳LF验证RMSE": best_rmse,
                            "累计训练点": total_train_exposure + len(indices),
                            "累计优化步": total_optimizer_steps + epoch_optimizer_steps,
                        },
                    )
        else:
            epochs_without_improvement += 1
        if rank == 0:
            total_train_exposure += len(indices)
            total_optimizer_steps += epoch_optimizer_steps
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "train_rmse_c": train_rmse,
                            "validation_rmse_c": validation_rmse,
                            "validation_mae_c": validation_mae,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            **({
                                "LF采样方式": sampling_mode,
                                "LF物理模式": lf_physics_mode,
                                "训练样本暴露": len(indices),
                                "观测优化步": epoch_optimizer_steps - int(use_physics),
                                "物理优化步": int(use_physics),
                                "累计训练点": total_train_exposure,
                                "累计优化步": total_optimizer_steps,
                            } if historical_lf is not None else {}),
                            **{f"loss_{name}": value for name, value in physics_values.items()},
                        }
                    )
                    + "\n"
                )
            if historical_lf is not None:
                save_training_state(
                    output_root / "阶段_最近.pt", base_model, optimizer,
                    stage="low_fidelity", epoch=epoch, budget={"LF轮次": epochs},
                    metadata={
                        "起点LF哈希": origin_hash, "源码哈希": source_hashes,
                        "采样方式": sampling_mode, "物理模式": lf_physics_mode,
                        "LF最佳验证RMSE": best_rmse,
                        "无改善轮次": epochs_without_improvement,
                        "累计训练点": total_train_exposure,
                        "累计优化步": total_optimizer_steps,
                    },
                )
        stop = torch.tensor(
            int(epochs_without_improvement >= patience), dtype=torch.int32, device=device
        )
        if dist.is_initialized():
            dist.broadcast(stop, src=0)
        if bool(stop.item()):
            early_stopped = True
            break
    elapsed = time.perf_counter() - start_time
    if historical_lf is not None and rank == 0 and (
        early_stopped or epoch == epochs
    ):
        save_training_state(
            output_root / "阶段_LF训练末.pt", base_model, optimizer,
            stage="low_fidelity", epoch=epoch, budget={"LF轮次": epochs},
            metadata={
                "实际结束轮次": epoch,
                "终止原因": "验证耐心提前停止" if early_stopped else "预算轮次执行完毕",
                "最佳LF验证RMSE": best_rmse,
                "累计训练点": total_train_exposure,
                "累计优化步": total_optimizer_steps,
            },
        )
    if historical_lf is not None and epoch < epochs and not early_stopped:
        paused = {
            "status": "paused_with_complete_training_state",
            "seed": seed, "last_epoch": epoch, "planned_epochs": epochs,
            "last_checkpoint": str(output_root / "阶段_最近.pt"),
            "training_seconds_this_session": elapsed,
        }
        return paused if rank == 0 else None
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
            "stopping_reason": "validation_patience" if early_stopped else "planned_budget_completed",
            "best_epoch": checkpoint["epoch"],
            "best_validation_rmse_c": checkpoint["validation_rmse_c"],
            **({"best_checkpoint_sha256": sha256_file(checkpoint_path)}
               if historical_lf is not None else {}),
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
                **({
                    "locked_lf_start_sha256": origin_hash,
                    "simulation_sampling_mode": sampling_mode,
                    "lf_physics_mode": lf_physics_mode,
                    "historical_optimizer_restored": False,
                } if historical_lf is not None else {}),
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
                **({
                    "total_train_points_exposed": total_train_exposure,
                    "total_optimizer_steps": total_optimizer_steps,
                } if historical_lf is not None else {}),
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
    device_name: str | None = None,
    batch_size: int = 8192,
    template_metrics_path: str | None = None,
    training_seconds: float | None = None,
    release_manifest_path: str,
) -> dict[str, Any]:
    device = resolve_device(device_name)
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
