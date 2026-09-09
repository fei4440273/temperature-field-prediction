from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.sensors import load_canonical_sensor_observations
from sic_cu.data.common import sha256_file
from sic_cu.data.processed import load_processed_ir_observations
from sic_cu.data.splits import assert_no_hf_leakage, build_power_splits
from sic_cu.eval.protocol_checks import (
    checkpoint_provenance,
    current_protocol_fingerprints,
    validate_hf_checkpoint_provenance,
    validate_lf_checkpoint_provenance,
    validate_release_checkpoint,
)
from sic_cu.eval.metrics import curve_metrics, macro_metric_summary, weighted_metrics
from sic_cu.eval.validation import (
    build_three_power_comparison,
    write_three_power_comparison,
)
from sic_cu.eval.residual_interpolation import (
    fit_residual_interpolator,
    fit_surface_temperature_interpolator,
)
from sic_cu.losses import PhysicsLossComputer, PhysicsLossWeights
from sic_cu.models import AdditiveCorrectionModel, ModelScales, PRCMultifidelityModel
from sic_cu.models.residual_interpolation import (
    ChebyshevSurfaceResidualGuide,
    fit_chebyshev_surface_residual_guide,
)
from sic_cu.models.common import parameter_count
from sic_cu.physics import (
    load_boundary_conditions,
    load_resolved_boundary_conditions,
    resolved_boundary_snapshot,
    sample_collocation,
)
from sic_cu.physics.materials import PhysicsConfigurationError, load_materials
from sic_cu.physics.trainable_parameters import TrainableBoundaryParameters
from sic_cu.train.simulation import build_model, distributed_context, load_sampled_points, set_seed
from sic_cu.train.common import physics_optimizer_step, write_config_snapshot


def _load_low_fidelity(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    method = str(checkpoint["method"])
    scales = ModelScales(**checkpoint["scales"])
    model = build_model(method, scales, **checkpoint.get("model_kwargs", {})).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def _filter_observations(
    frame: pl.DataFrame,
    split: str | None,
    powers_w: Iterable[float] | None,
) -> pl.DataFrame:
    if split is None and powers_w is None:
        raise ValueError("At least one of split or powers_w must be supplied")
    result = frame if split is None else frame.filter(pl.col("split") == split)
    if powers_w is not None:
        powers = sorted({round(float(value), 4) for value in powers_w})
        if not powers:
            raise ValueError("Power selection cannot be empty")
        power_keys = [int(round(value * 10_000)) for value in powers]
        result = result.filter(
            (pl.col("power_w").cast(pl.Float64) * 10_000)
            .round(0)
            .cast(pl.Int64)
            .is_in(power_keys)
        )
    return result


def _ir_dataset(
    split: str | None = None,
    powers_w: Iterable[float] | None = None,
) -> TensorDataset:
    frame = _filter_observations(
        load_processed_ir_observations(split),
        split,
        powers_w,
    )
    if frame.is_empty():
        raise ValueError("IR selection is empty")
    frame = frame.with_columns(
        (
            pl.col("frame_weight")
            / pl.col("frame_weight").sum().over("power_w")
        )
        .cast(pl.Float32)
        .alias("condition_weight")
    )
    coordinates = np.column_stack(
        (
            frame["r_m"].to_numpy(),
            np.zeros(frame.height),
            frame["time_s"].to_numpy(),
            frame["power_w"].to_numpy(),
            np.ones(frame.height),
        )
    ).astype(np.float32)
    target = frame["temperature_mean_k"].to_numpy().astype(np.float32)[:, None]
    weight = frame["condition_weight"].to_numpy().astype(np.float32)[:, None]
    return TensorDataset(
        torch.from_numpy(coordinates),
        torch.from_numpy(target),
        torch.from_numpy(weight),
    )


def _surface_teacher_dataset(
    training_powers_w: Iterable[float],
    simulation_powers_w: list[float],
    method: str = "regression_residual_per_watt",
    power_count: int = 24,
    time_step_s: float = 5.0,
    radial_count: int = 51,
) -> TensorDataset:
    if power_count < 2 or radial_count < 2 or time_step_s <= 0.0:
        raise ValueError("Invalid surface teacher grid configuration")
    frame = load_processed_ir_observations()
    training = sorted(round(float(value), 4) for value in training_powers_w)
    predictor = fit_residual_interpolator(frame, training, simulation_powers_w)
    powers = np.linspace(training[0], training[-1], power_count, dtype=np.float32)
    times = np.arange(time_step_s, 200.0 + time_step_s / 2.0, time_step_s, dtype=np.float32)
    radii = np.linspace(0.0, 0.025, radial_count, dtype=np.float32)
    coordinate_blocks = []
    residual_blocks = []
    time_grid, radius_grid = np.meshgrid(times, radii, indexing="ij")
    flat_times = time_grid.reshape(-1)
    flat_radii = radius_grid.reshape(-1)
    for power in powers:
        residual = predictor.predict(
            float(power), flat_times, flat_radii, method
        )
        coordinate_blocks.append(
            np.column_stack(
                (
                    flat_radii,
                    np.zeros_like(flat_radii),
                    flat_times,
                    np.full_like(flat_times, power),
                    np.ones_like(flat_times),
                )
            )
        )
        residual_blocks.append(residual[:, None])
    return TensorDataset(
        torch.from_numpy(np.concatenate(coordinate_blocks).astype(np.float32)),
        torch.from_numpy(np.concatenate(residual_blocks).astype(np.float32)),
    )


def _sensor_tensors(
    device: torch.device,
    split: str | None = "train",
    powers_w: Iterable[float] | None = None,
) -> tuple[Tensor, ...]:
    frame = _filter_observations(
        load_canonical_sensor_observations(split=split), split, powers_w
    )
    geometry = load_yaml("configs/geometry.yaml")
    bottom = float(geometry["embedding"]["copper_bottom_z_m"])
    coordinates = np.column_stack(
        (
            frame["r_m"].to_numpy(),
            np.full(frame.height, bottom),
            frame["time_s"].to_numpy(),
            frame["power_w"].to_numpy(),
            np.zeros(frame.height),
        )
    ).astype(np.float32)
    group_keys = list(zip(frame["power_w"].to_list(), frame["sensor_type"].to_list()))
    first_by_group: dict[tuple[float, str], int] = {}
    baseline_indices = []
    for index, key in enumerate(group_keys):
        first_by_group.setdefault(key, index)
        baseline_indices.append(first_by_group[key])
    return (
        torch.from_numpy(coordinates).to(device),
        torch.from_numpy(frame["temperature_k"].to_numpy().astype(np.float32)[:, None]).to(device),
        torch.from_numpy(frame["delta_temperature_k"].to_numpy().astype(np.float32)[:, None]).to(device),
        torch.tensor(baseline_indices, dtype=torch.long, device=device),
    )


def _macro_sensor_training_losses(
    prediction: Tensor,
    target: Tensor,
    delta: Tensor,
    baseline: Tensor,
    coordinates: Tensor,
    temperature_scale_k: float,
) -> tuple[Tensor, Tensor]:
    """Return equal-curve MSEs instead of record-count-weighted losses."""
    predicted_delta = prediction - prediction[baseline]
    group_keys = torch.stack(
        (
            torch.round(coordinates[:, 3] * 10_000),
            torch.round(coordinates[:, 0] * 1_000_000),
        ),
        dim=1,
    )
    absolute_losses = []
    delta_losses = []
    for key in torch.unique(group_keys, dim=0):
        mask = (group_keys == key).all(dim=1)
        absolute_losses.append(
            ((prediction[mask] - target[mask]) / temperature_scale_k).pow(2).mean()
        )
        delta_losses.append(
            ((predicted_delta[mask] - delta[mask]) / temperature_scale_k)
            .pow(2)
            .mean()
        )
    if not absolute_losses:
        raise RuntimeError("Sensor selection contains no complete curves")
    return torch.stack(absolute_losses).mean(), torch.stack(delta_losses).mean()


@torch.no_grad()
def _sensor_validation(
    model: nn.Module,
    sensor: tuple[Tensor, ...],
) -> dict[str, float]:
    model.eval()
    coordinates, target, delta, baseline = sensor
    prediction = model(coordinates)
    predicted_delta = prediction - prediction[baseline]
    records = []
    group_keys = torch.stack(
        (
            torch.round(coordinates[:, 3] * 10_000),
            torch.round(coordinates[:, 0] * 1_000_000),
        ),
        dim=1,
    )
    for key in torch.unique(group_keys, dim=0):
        mask = (group_keys == key).all(dim=1)
        absolute_error = prediction[mask] - target[mask]
        delta_error = predicted_delta[mask] - delta[mask]
        records.append(
            {
                "absolute_rmse_c": float(torch.sqrt(absolute_error.pow(2).mean())),
                "absolute_mae_c": float(absolute_error.abs().mean()),
                "delta_rmse_c": float(torch.sqrt(delta_error.pow(2).mean())),
                "delta_mae_c": float(delta_error.abs().mean()),
            }
        )
    return {
        name: float(np.mean([record[name] for record in records]))
        for name in (
            "absolute_rmse_c",
            "absolute_mae_c",
            "delta_rmse_c",
            "delta_mae_c",
        )
    }


@torch.no_grad()
def _weighted_validation(model: nn.Module, loader: DataLoader, device: torch.device) -> Tensor:
    """Return rank-0 full-dataset macro metrics, broadcast to every rank."""
    model.eval()
    by_power: dict[float, list[float]] = {}
    if not dist.is_initialized() or dist.get_rank() == 0:
        for coordinates, target, weight in loader:
            coordinates = coordinates.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            error = model(coordinates) - target
            for power in torch.unique(coordinates[:, 3]):
                mask = torch.isclose(coordinates[:, 3], power, atol=1e-4, rtol=0.0)
                key = round(float(power), 4)
                sums = by_power.setdefault(key, [0.0, 0.0, 0.0])
                sums[0] += float((weight[mask].double() * error[mask].double().pow(2)).sum())
                sums[1] += float((weight[mask].double() * error[mask].double().abs()).sum())
                sums[2] += float(weight[mask].double().sum())
    result = torch.zeros(3, dtype=torch.float64, device=device)
    if not dist.is_initialized() or dist.get_rank() == 0:
        if not by_power or any(values[2] <= 0.0 for values in by_power.values()):
            raise RuntimeError("Validation contains an empty or zero-weight power condition")
        per_power_rmse = [(values[0] / values[2]) ** 0.5 for values in by_power.values()]
        per_power_mae = [values[1] / values[2] for values in by_power.values()]
        result[0] = float(np.mean(per_power_rmse))
        result[1] = float(np.mean(per_power_mae))
        result[2] = float(np.sqrt(np.mean(np.square(per_power_rmse))))
    if dist.is_initialized():
        dist.broadcast(result, src=0)
    return result


@torch.no_grad()
def _predict_batches(model: nn.Module, coordinates: Tensor, device: torch.device) -> np.ndarray:
    output = []
    model.eval()
    for offset in range(0, len(coordinates), 65536):
        output.append(model(coordinates[offset : offset + 65536].to(device)).cpu().numpy())
    return np.concatenate(output).reshape(-1)


def _gradient_l2(parameters: Iterable[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().double().pow(2).sum())
    return squared**0.5


@torch.no_grad()
def _evaluate_ir_model(
    model: nn.Module,
    split: str,
    device: torch.device,
    powers_w: Iterable[float] | None = None,
) -> dict[str, Any]:
    frame = _filter_observations(
        load_processed_ir_observations(split),
        split,
        powers_w,
    )
    dataset = _ir_dataset(split, powers_w)
    coordinates, target, weight = dataset.tensors
    prediction = _predict_batches(model, coordinates, device)
    target_values = target.numpy().reshape(-1)
    weights = weight.numpy().reshape(-1)
    error = prediction - target_values
    powers = frame["power_w"].to_numpy()
    times = frame["time_s"].to_numpy()
    records = []
    for power in np.unique(powers):
        power_mask = np.isclose(powers, power, atol=1e-4)
        condition_metrics = weighted_metrics(
            target_values[power_mask], prediction[power_mask], weights[power_mask]
        )
        peak_errors = []
        peak_rise_relative_errors = []
        condition_times = np.unique(times[power_mask])
        first_peak_k = float(target_values[power_mask & np.isclose(times, condition_times[0], atol=1e-5)].max())
        for time_s in condition_times:
            frame_mask = power_mask & np.isclose(times, time_s, atol=1e-5)
            peak_error = float(
                prediction[frame_mask].max() - target_values[frame_mask].max()
            )
            peak_errors.append(peak_error)
            peak_rise_k = float(target_values[frame_mask].max()) - first_peak_k
            if peak_rise_k >= 1.0:
                peak_rise_relative_errors.append(abs(peak_error) / peak_rise_k * 100.0)
        time_windows: dict[str, Any] = {}
        for name, lower, upper in (
            ("time_0_30_s", 0.0, 30.0),
            ("time_30_100_s", 30.0, 100.0),
            ("time_100_200_s", 100.0, 200.0 + np.finfo(float).eps),
        ):
            mask = power_mask & (times >= lower) & (times < upper)
            time_windows[name] = (
                weighted_metrics(target_values[mask], prediction[mask], weights[mask])
                if bool(mask.any())
                else None
            )
        records.append(
            {
                "power_w": float(power),
                **condition_metrics,
                "peak_mae_c": float(np.mean(np.abs(peak_errors))),
                "peak_max_abs_error_c": float(np.max(np.abs(peak_errors))),
                "peak_rise_relative_error_percent": (
                    float(np.mean(peak_rise_relative_errors))
                    if peak_rise_relative_errors
                    else None
                ),
                "time_windows": time_windows,
            }
        )
    macro = macro_metric_summary(records)
    return {
        "split": split,
        "selection_metric_version": "macro_v1",
        **macro,
        "peak_mae_c": float(np.mean([record["peak_mae_c"] for record in records])),
        "peak_max_abs_error_c": float(
            max(record["peak_max_abs_error_c"] for record in records)
        ),
        "per_power": records,
    }


@torch.no_grad()
def _evaluate_sensor_model(
    model: nn.Module,
    split: str,
    device: torch.device,
    powers_w: Iterable[float] | None = None,
) -> dict[str, Any]:
    frame = _filter_observations(
        load_canonical_sensor_observations(split=split),
        split,
        powers_w,
    )
    geometry = load_yaml("configs/geometry.yaml")
    bottom = float(geometry["embedding"]["copper_bottom_z_m"])
    records = []
    for power in sorted(frame["power_w"].unique().to_list()):
        for sensor_type in ("hot", "cold"):
            group = frame.filter(
                (pl.col("power_w") == power) & (pl.col("sensor_type") == sensor_type)
            ).sort("time_s")
            coordinates = torch.from_numpy(
                np.column_stack(
                    (
                        group["r_m"].to_numpy(),
                        np.full(group.height, bottom),
                        group["time_s"].to_numpy(),
                        group["power_w"].to_numpy(),
                        np.zeros(group.height),
                    )
                ).astype(np.float32)
            )
            prediction = _predict_batches(model, coordinates, device)
            target = group["temperature_k"].to_numpy()
            records.append(
                {
                    "power_w": float(power),
                    "sensor_type": sensor_type,
                    "absolute": curve_metrics(target, prediction),
                    "delta": curve_metrics(
                        target - target[0],
                        prediction - prediction[0],
                    ),
                }
            )
    return {
        "split": split,
        "selection_metric_version": "macro_v1",
        "absolute_rmse_c": float(
            np.mean([record["absolute"]["rmse_c"] for record in records])
        ),
        "absolute_mae_c": float(
            np.mean([record["absolute"]["mae_c"] for record in records])
        ),
        "delta_rmse_c": float(
            np.mean([record["delta"]["rmse_c"] for record in records])
        ),
        "delta_mae_c": float(
            np.mean([record["delta"]["mae_c"] for record in records])
        ),
        "per_curve": records,
    }


def train_multifidelity(
    low_fidelity_checkpoint: str,
    output_directory: str,
    seed: int = 0,
    correction_epochs: int = 1500,
    joint_epochs: int = 500,
    batch_size: int = 2048,
    physics_collocation: int = 256,
    width: int = 128,
    depth: int = 4,
    skip_sensors: bool = False,
    patience: int = 200,
    identify_physics_parameters: bool = False,
    silicon_carbide_emissivity_initial: float = 0.5,
    copper_emissivity_initial: float = 0.5,
    contact_resistance_initial_m2_k_w: float | None = None,
    evaluate_test: bool = False,
    freeze_physics_parameters: bool = False,
    hard_deployment_constraints: bool = True,
    sensor_absolute_weight: float | None = None,
    sensor_delta_weight: float | None = None,
    hf_train_powers_w: Iterable[float] | None = None,
    hf_validation_powers_w: Iterable[float] | None = None,
    hf_test_powers_w: Iterable[float] | None = None,
    correction_power_scaling: str = "none",
    correction_power_reference_w: float = 400.0,
    correction_direct_power_input: bool = True,
    surface_teacher_weight: float = 0.0,
    surface_teacher_method: str = "regression_residual_per_watt",
    checkpoint_selection: str = "validation",
    hard_surface_residual_guide: bool = False,
    surface_guide_degree_r: int = 20,
    surface_guide_degree_t: int = 20,
    simulation_supervision_target: str = "low_fidelity",
) -> dict[str, Any] | None:
    if correction_epochs < 0 or joint_epochs < 0 or correction_epochs + joint_epochs < 1:
        raise ValueError("At least one correction or joint epoch is required")
    if evaluate_test:
        raise ValueError(
            "Training-time test evaluation is disabled; freeze a release and use "
            "scripts/06_evaluate_test_data.py"
        )
    if freeze_physics_parameters and not identify_physics_parameters:
        raise ValueError(
            "freeze_physics_parameters requires identify_physics_parameters so that "
            "unverified sensitivity values are explicit"
        )
    if surface_teacher_weight < 0.0:
        raise ValueError("surface_teacher_weight must be nonnegative")
    if checkpoint_selection not in {"validation", "final_epoch"}:
        raise ValueError("checkpoint_selection must be 'validation' or 'final_epoch'")
    if simulation_supervision_target not in {"low_fidelity", "high_fidelity_pullback"}:
        raise ValueError(
            "simulation_supervision_target must be 'low_fidelity' or "
            "'high_fidelity_pullback'"
        )
    materials = load_materials()
    boundaries = (
        load_boundary_conditions(allow_unidentified=True)
        if identify_physics_parameters
        else load_resolved_boundary_conditions()
    )
    rank, local_rank, world_size = distributed_context()
    set_seed(seed)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
    trainable_physics = None
    if identify_physics_parameters:
        contact_initial = contact_resistance_initial_m2_k_w
        if contact_initial is None:
            contact_initial = (
                boundaries.contact_resistance_m2_k_w
                or boundaries.contact_resistance_initialization_m2_k_w
                or 1e-6
            )
        trainable_physics = TrainableBoundaryParameters(
            silicon_carbide_emissivity_initial,
            copper_emissivity_initial,
            contact_initial,
        ).to(device)
        if freeze_physics_parameters:
            trainable_physics.requires_grad_(False)
    splits = build_power_splits()
    power_overrides = (
        hf_train_powers_w,
        hf_validation_powers_w,
        hf_test_powers_w,
    )
    if all(values is None for values in power_overrides):
        hf_train_powers = splits.hf_train
        hf_validation_powers = splits.hf_validation
        hf_test_powers = splits.hf_test
        hf_split_source = "fixed_experiment_train_validation_and_test_data_test"
    elif any(values is None for values in power_overrides):
        raise ValueError(
            "hf_train_powers_w, hf_validation_powers_w, and hf_test_powers_w "
            "must be supplied together"
        )
    else:
        hf_train_powers = frozenset(
            round(float(value), 4) for value in hf_train_powers_w or ()
        )
        hf_validation_powers = frozenset(
            round(float(value), 4) for value in hf_validation_powers_w or ()
        )
        hf_test_powers = frozenset(
            round(float(value), 4) for value in hf_test_powers_w or ()
        )
        groups = (hf_train_powers, hf_validation_powers, hf_test_powers)
        if not hf_train_powers:
            raise ValueError("Explicit high-fidelity training powers cannot be empty")
        if checkpoint_selection == "validation" and not hf_validation_powers:
            raise ValueError("Validation checkpoint selection requires validation powers")
        if evaluate_test and not hf_test_powers:
            raise ValueError("Test evaluation requires held-out test powers")
        if any(left & right for i, left in enumerate(groups) for right in groups[i + 1 :]):
            raise ValueError("Explicit high-fidelity power groups overlap")
        if (
            hf_train_powers != splits.hf_train
            or hf_validation_powers != splits.hf_validation
            or hf_test_powers != splits.hf_test
        ):
            raise ValueError(
                "Explicit high-fidelity groups must exactly match the fixed train/validation/test protocol"
            )
        hf_split_source = "explicit_fixed_train_validation_test_protocol"
    assert_no_hf_leakage(
        {"ir": hf_train_powers, "hot_cold": () if skip_sensors else hf_train_powers},
        hf_validation_powers | hf_test_powers | splits.external_sensor_test,
    )
    training = load_yaml("configs/training.yaml")
    loss_cfg = dict(training["loss_weights"])
    if sensor_absolute_weight is not None:
        if sensor_absolute_weight < 0:
            raise ValueError("sensor_absolute_weight must be nonnegative")
        loss_cfg["sensor_absolute"] = float(sensor_absolute_weight)
    if sensor_delta_weight is not None:
        if sensor_delta_weight < 0:
            raise ValueError("sensor_delta_weight must be nonnegative")
        loss_cfg["sensor_delta"] = float(sensor_delta_weight)
    selection_weights = training["multifidelity_selection_weights"]
    physics = PhysicsLossComputer(
        materials,
        boundaries,
        PhysicsLossWeights(
            pde=float(loss_cfg["pde"]),
            boundary=float(loss_cfg["boundary"]),
            initial=float(loss_cfg["initial"]),
            interface=float(loss_cfg["interface"]),
        ),
        trainable_parameters=trainable_physics,
    )
    low_fidelity, lf_checkpoint = _load_low_fidelity(low_fidelity_checkpoint, device)
    fingerprints = current_protocol_fingerprints()
    validate_lf_checkpoint_provenance(lf_checkpoint, splits, fingerprints)
    expected_lf_train = sorted(splits.simulation_train)
    if sorted(float(value) for value in lf_checkpoint.get("train_powers_w", [])) != expected_lf_train:
        raise RuntimeError("Low-fidelity checkpoint does not use the fixed simulation training powers")
    scales = ModelScales(**lf_checkpoint["scales"])
    surface_residual_guide = None
    if hard_surface_residual_guide:
        ir_frame = load_processed_ir_observations()
        surface_predictor = fit_surface_temperature_interpolator(
            ir_frame,
            hf_train_powers,
            boundaries.initial_temperature_k,
        )
        surface_residual_guide = fit_chebyshev_surface_residual_guide(
            surface_predictor,
            degree_r=surface_guide_degree_r,
            degree_t=surface_guide_degree_t,
            output_offset_k=boundaries.initial_temperature_k,
        ).to(device)
    base_model = AdditiveCorrectionModel(
        low_fidelity,
        scales,
        width=width,
        depth=depth,
        freeze_low_fidelity=True,
        hard_initial_temperature_k=(
            boundaries.initial_temperature_k if hard_deployment_constraints else None
        ),
        correction_calibration_range_w=(
            (min(splits.experiment_powers), 800.0) if hard_deployment_constraints else None
        ),
        correction_power_scaling=correction_power_scaling,
        correction_power_reference_w=correction_power_reference_w,
        correction_direct_power_input=correction_direct_power_input,
        surface_residual_guide=surface_residual_guide,
        surface_guide_output=(
            "absolute_temperature" if hard_surface_residual_guide else "residual"
        ),
    ).to(device)
    model: nn.Module = (
        DistributedDataParallel(base_model, device_ids=[local_rank])
        if world_size > 1
        else base_model
    )
    train_data = _ir_dataset(split="train", powers_w=hf_train_powers)
    validation_data = (
        _ir_dataset(split="validation", powers_w=hf_validation_powers)
        if hf_validation_powers
        else None
    )
    train_sampler = (
        DistributedSampler(train_data, world_size, rank, shuffle=True, seed=seed)
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=device.type == "cuda",
    )
    validation_loader = (
        None
        if validation_data is None
        else DataLoader(
            validation_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
    )
    teacher_data = (
        None
        if surface_teacher_weight == 0.0
        else _surface_teacher_dataset(
            hf_train_powers,
            sorted(splits.simulation_train),
            surface_teacher_method,
        )
    )
    teacher_sampler = (
        DistributedSampler(teacher_data, world_size, rank, shuffle=True, seed=seed)
        if teacher_data is not None and world_size > 1
        else None
    )
    teacher_loader = (
        None
        if teacher_data is None
        else DataLoader(
            teacher_data,
            batch_size=batch_size,
            shuffle=teacher_sampler is None,
            sampler=teacher_sampler,
            pin_memory=device.type == "cuda",
        )
    )
    teacher_iterator = None if teacher_loader is None else iter(teacher_loader)
    sensor = (
        None
        if skip_sensors
        else _sensor_tensors(device, split="train", powers_w=hf_train_powers)
    )
    validation_sensor = (
        None
        if skip_sensors or not hf_validation_powers
        else _sensor_tensors(
            device, split="validation", powers_w=hf_validation_powers
        )
    )
    correction_parameters = list(base_model.correction.parameters())
    physics_parameters = (
        []
        if trainable_physics is None
        else [parameter for parameter in trainable_physics.parameters() if parameter.requires_grad]
    )
    weight_decay = float(training["optimizer"]["weight_decay"])
    parameter_learning_rate = float(
        training["inverse_identification"].get(
            "learning_rate", training["optimizer"]["learning_rate"]
        )
    )

    def make_optimizer(model_parameters, learning_rate: float):
        parameter_groups = [
            {
                "params": list(model_parameters),
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        ]
        if physics_parameters:
            parameter_groups.append(
                {
                    "params": physics_parameters,
                    "lr": parameter_learning_rate,
                    "weight_decay": 0.0,
                }
            )
        return torch.optim.AdamW(parameter_groups)

    optimizer = make_optimizer(
        correction_parameters,
        float(training["optimizer"]["learning_rate"]),
    )
    output = PROJECT_ROOT / output_directory
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "training.jsonl").write_text("", encoding="utf-8")
        write_config_snapshot(output)
    if dist.is_initialized():
        dist.barrier()
    best: float | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    data_consumption = {
        "hf_ir_points": 0,
        "hf_sensor_points": 0,
        "surface_teacher_points": 0,
        "lf_simulation_points": 0,
        "lf_simulation_material_points": {"copper": 0, "silicon_carbide": 0},
    }
    total_epochs = correction_epochs + joint_epochs
    simulation_iterator = None
    simulation_loader = None
    started = time.perf_counter()
    for epoch in range(1, total_epochs + 1):
        stage = "correction" if epoch <= correction_epochs else "joint"
        if epoch == correction_epochs + 1:
            epochs_without_improvement = 0
            base_model.freeze_low_fidelity(False)
            if world_size > 1:
                model = DistributedDataParallel(base_model, device_ids=[local_rank])
            optimizer = make_optimizer(
                [parameter for parameter in base_model.parameters() if parameter.requires_grad],
                float(training["optimizer"]["joint_learning_rate"]),
            )
            simulation_data = load_sampled_points(
                sorted(splits.simulation_train),
                samples_per_power=2048,
                seed=50_000 + seed,
            )
            simulation_sampler = (
                DistributedSampler(simulation_data, world_size, rank, shuffle=True, seed=seed)
                if world_size > 1
                else None
            )
            simulation_loader = DataLoader(
                simulation_data,
                batch_size=batch_size,
                shuffle=simulation_sampler is None,
                sampler=simulation_sampler,
                pin_memory=device.type == "cuda",
            )
            simulation_iterator = iter(simulation_loader)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if teacher_sampler is not None:
            teacher_sampler.set_epoch(epoch)
            teacher_iterator = iter(teacher_loader)
        model.train()
        sums = {"ir": 0.0, "sensor": 0.0, "teacher": 0.0, "low_fidelity": 0.0}
        batches = 0
        for coordinates, target, weight in train_loader:
            coordinates = coordinates.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(coordinates)
            ir_loss = (weight * ((prediction - target) / scales.temperature_scale_k).pow(2)).sum() / weight.sum()
            total = float(loss_cfg["ir"]) * ir_loss
            sensor_loss = torch.zeros((), device=device)
            if sensor is not None:
                sensor_coordinates, sensor_target, sensor_delta, baseline = sensor
                sensor_prediction = model(sensor_coordinates)
                sensor_absolute_loss, sensor_delta_loss = _macro_sensor_training_losses(
                    sensor_prediction,
                    sensor_target,
                    sensor_delta,
                    baseline,
                    sensor_coordinates,
                    scales.temperature_scale_k,
                )
                sensor_loss = (
                    float(loss_cfg["sensor_absolute"]) * sensor_absolute_loss
                    + float(loss_cfg["sensor_delta"]) * sensor_delta_loss
                )
                total = total + sensor_loss
            teacher_loss = torch.zeros((), device=device)
            if teacher_loader is not None:
                try:
                    teacher_coordinates, teacher_residual = next(teacher_iterator)
                except StopIteration:
                    teacher_iterator = iter(teacher_loader)
                    teacher_coordinates, teacher_residual = next(teacher_iterator)
                teacher_coordinates = teacher_coordinates.to(device, non_blocking=True)
                teacher_residual = teacher_residual.to(device, non_blocking=True)
                predicted_residual = model(teacher_coordinates, correction_only=True)
                teacher_loss = (
                    (predicted_residual - teacher_residual)
                    / scales.temperature_scale_k
                ).pow(2).mean()
                total = total + float(surface_teacher_weight) * teacher_loss
            lf_loss = torch.zeros((), device=device)
            if stage == "joint" and simulation_loader is not None:
                try:
                    lf_coordinates, lf_target = next(simulation_iterator)
                except StopIteration:
                    simulation_iterator = iter(simulation_loader)
                    lf_coordinates, lf_target = next(simulation_iterator)
                lf_coordinates = lf_coordinates.to(device, non_blocking=True)
                lf_target = lf_target.to(device, non_blocking=True)
                lf_prediction = (
                    model(lf_coordinates, fidelity="low")
                    if simulation_supervision_target == "low_fidelity"
                    else model(lf_coordinates, fidelity="high")
                )
                lf_loss = (
                    (lf_prediction - lf_target) / scales.temperature_scale_k
                ).pow(2).mean()
                total = total + float(loss_cfg["low_fidelity"]) * lf_loss
            total.backward()
            lf_gradient_norm = _gradient_l2(base_model.low_fidelity_model.parameters())
            hf_gradient_norm = _gradient_l2(base_model.correction.parameters())
            optimizer.step()
            sums["ir"] += float(ir_loss.detach())
            sums["sensor"] += float(sensor_loss.detach())
            sums["teacher"] += float(teacher_loss.detach())
            sums["low_fidelity"] += float(lf_loss.detach())
            data_consumption["hf_ir_points"] += int(len(coordinates))
            if sensor is not None:
                data_consumption["hf_sensor_points"] += int(len(sensor[0]))
            if teacher_loader is not None:
                data_consumption["surface_teacher_points"] += int(
                    len(teacher_coordinates)
                )
            if stage == "joint" and simulation_loader is not None:
                data_consumption["lf_simulation_points"] += int(len(lf_coordinates))
                data_consumption["lf_simulation_material_points"]["copper"] += int(
                    (lf_coordinates[:, 4] < 0.5).sum()
                )
                data_consumption["lf_simulation_material_points"][
                    "silicon_carbide"
                ] += int((lf_coordinates[:, 4] >= 0.5).sum())
            batches += 1
        collocation = sample_collocation(
            physics_collocation,
            device,
            seed=seed * 1_000_000 + epoch * world_size + rank,
        )
        physics_components = physics_optimizer_step(
            model,
            optimizer,
            physics,
            collocation,
        )
        validation_rmse = None
        validation_mae = None
        validation_sensor_metrics = None
        validation_selection_score = None
        save_checkpoint = checkpoint_selection == "final_epoch" and epoch == total_epochs
        if validation_loader is not None:
            validation = _weighted_validation(base_model, validation_loader, device)
            validation_rmse = float(validation[0])
            validation_mae = float(validation[1])
            validation_sensor_metrics = (
                None
                if validation_sensor is None
                else _sensor_validation(base_model, validation_sensor)
            )
            selection_numerator = (
                float(selection_weights["ir_rmse"]) * validation_rmse
            )
            selection_denominator = float(selection_weights["ir_rmse"])
            if validation_sensor_metrics is not None:
                selection_numerator += float(selection_weights["sensor_absolute_rmse"]) * (
                    validation_sensor_metrics["absolute_rmse_c"]
                )
                selection_numerator += float(selection_weights["sensor_delta_rmse"]) * (
                    validation_sensor_metrics["delta_rmse_c"]
                )
                selection_denominator += float(selection_weights["sensor_absolute_rmse"])
                selection_denominator += float(selection_weights["sensor_delta_rmse"])
            validation_selection_score = selection_numerator / selection_denominator
            if checkpoint_selection == "validation":
                save_checkpoint = best is None or validation_selection_score < best - 1e-4
        if save_checkpoint:
            best = validation_selection_score
            best_epoch = epoch
            epochs_without_improvement = 0
            if rank == 0:
                torch.save(
                    {
                        "schema_version": 1,
                        "method": "multifidelity_correction",
                        "seed": seed,
                        "epoch": epoch,
                        "model_state": base_model.state_dict(),
                        "physics_parameter_state": (
                            None
                            if trainable_physics is None
                            else trainable_physics.state_dict()
                        ),
                        "identified_physics_parameters": (
                            None
                            if trainable_physics is None
                            else trainable_physics.snapshot()
                        ),
                        "low_fidelity_method": lf_checkpoint["method"],
                        "low_fidelity_model_kwargs": lf_checkpoint.get("model_kwargs", {}),
                        "correction_model_kwargs": {
                            "width": width,
                            "depth": depth,
                            "activation": "tanh",
                            "include_material": True,
                            "hard_initial_temperature_k": (
                                boundaries.initial_temperature_k
                                if hard_deployment_constraints
                                else None
                            ),
                            "initial_ramp_time_s": 0.05,
                            "correction_calibration_range_w": (
                                [min(splits.experiment_powers), 800.0]
                                if hard_deployment_constraints
                                else None
                            ),
                            "correction_support_range_w": [0.0, 800.0],
                            "correction_extrapolation_exponent": 2.0,
                            "correction_power_scaling": correction_power_scaling,
                            "correction_power_reference_w": correction_power_reference_w,
                            "correction_direct_power_input": correction_direct_power_input,
                            "silicon_carbide_height_m": 0.012,
                            "surface_guide_output": (
                                "absolute_temperature"
                                if hard_surface_residual_guide
                                else "residual"
                            ),
                        },
                        "surface_residual_guide_spec": (
                            None
                            if surface_residual_guide is None
                            else surface_residual_guide.to_spec()
                        ),
                        "scales": asdict(scales),
                        "hf_train_powers_w": sorted(hf_train_powers),
                        "hf_validation_powers_w": sorted(hf_validation_powers),
                        "hf_test_powers_w": sorted(hf_test_powers),
                        "provenance": checkpoint_provenance(
                            role="high_fidelity_multifidelity",
                            train_powers_w=hf_train_powers,
                            validation_powers_w=hf_validation_powers,
                            fingerprints=fingerprints,
                        )
                        | {
                            "lf_checkpoint_sha256": sha256_file(low_fidelity_checkpoint),
                            "lf_checkpoint_provenance": lf_checkpoint["provenance"],
                        },
                        "resolved_physics": resolved_boundary_snapshot(boundaries),
                        "sensors_used": not skip_sensors,
                        "validation_rmse_c": validation_rmse,
                        "validation_selection_score_c": validation_selection_score,
                        "validation_sensor": validation_sensor_metrics,
                        "material_passport": {
                            "simulation_data_used": True,
                            "experiment_data_used": True,
                            "sensor_data_used": not skip_sensors,
                            "physics_loss_used": True,
                            "physics_parameters_trainable": (
                                identify_physics_parameters
                                and not freeze_physics_parameters
                            ),
                            "physics_parameter_status": (
                                "fixed_sensitivity_scenario"
                                if freeze_physics_parameters
                                else "preliminary_joint_optimization"
                                if identify_physics_parameters
                                else "fixed_from_verified_configuration"
                            ),
                            "internal_experiment_truth": "not available",
                            "surface_teacher_used": surface_teacher_weight > 0.0,
                            "hard_surface_residual_guide": hard_surface_residual_guide,
                        },
                    },
                    output / "best.pt",
                )
        elif checkpoint_selection == "validation":
            epochs_without_improvement += 1
        if rank == 0:
            record = {
                "epoch": epoch,
                "stage": stage,
                "validation_ir_rmse_c": validation_rmse,
                "validation_ir_mae_c": validation_mae,
                "validation_selection_score_c": validation_selection_score,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "lf_gradient_l2": lf_gradient_norm,
                "hf_gradient_l2": hf_gradient_norm,
                "lf_parameters_frozen": stage == "correction",
                "epoch_hf_ir_points": int(len(train_data)),
                "epoch_lf_simulation_points": int(
                    0
                    if stage != "joint" or simulation_loader is None
                    else len(simulation_loader.dataset)
                ),
                **{f"loss_{name}": value / max(batches, 1) for name, value in sums.items()},
                **{f"loss_{name}": float(value.detach()) for name, value in physics_components.items()},
            }
            if validation_sensor_metrics is not None:
                record.update(
                    {
                        "validation_sensor_absolute_rmse_c": validation_sensor_metrics[
                            "absolute_rmse_c"
                        ],
                        "validation_sensor_delta_rmse_c": validation_sensor_metrics[
                            "delta_rmse_c"
                        ],
                    }
                )
            if trainable_physics is not None:
                record.update(
                    {
                        f"identified_{name}": value
                        for name, value in trainable_physics.snapshot().items()
                    }
                )
            with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        may_stop = (
            checkpoint_selection == "validation"
            and (stage == "joint" or joint_epochs == 0)
        )
        stop = torch.tensor(
            int(may_stop and epochs_without_improvement >= patience),
            dtype=torch.int32,
            device=device,
        )
        if dist.is_initialized():
            dist.broadcast(stop, src=0)
        if bool(stop.item()):
            break
    training_seconds = time.perf_counter() - started
    if joint_epochs > 0:
        material_points = data_consumption["lf_simulation_material_points"]
        if (
            data_consumption["lf_simulation_points"] == 0
            or material_points["copper"] == 0
            or material_points["silicon_carbide"] == 0
        ):
            raise RuntimeError(
                "Joint training declared LF replay but did not consume both materials"
            )
    elapsed = torch.tensor(training_seconds, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        dist.barrier()
    result = None
    if rank == 0:
        checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
        base_model.load_state_dict(checkpoint["model_state"])
        if trainable_physics is not None:
            trainable_physics.load_state_dict(checkpoint["physics_parameter_state"])
        test_ir = (
            _evaluate_ir_model(base_model, "test", device, hf_test_powers)
            if evaluate_test and hf_test_powers
            else None
        )
        test_sensor = (
            _evaluate_sensor_model(base_model, "test", device, hf_test_powers)
            if evaluate_test and hf_test_powers and not skip_sensors
            else None
        )
        final_validation_ir = (
            _evaluate_ir_model(base_model, "validation", device, hf_validation_powers)
            if hf_validation_powers
            else None
        )
        final_validation_sensor = (
            _evaluate_sensor_model(
                base_model, "validation", device, hf_validation_powers
            )
            if hf_validation_powers and not skip_sensors
            else None
        )
        validation_comparison = (
            build_three_power_comparison(
                final_validation_ir,
                final_validation_sensor,
                splits.hf_validation,
                "validation",
            )
            if final_validation_ir is not None
            and final_validation_sensor is not None
            and frozenset(hf_validation_powers) == splits.hf_validation
            else None
        )
        if validation_comparison is not None:
            write_three_power_comparison(
                validation_comparison,
                output / "validation_three_power_comparison.json",
                output / "validation_three_power_comparison.csv",
            )
        test_comparison = (
            build_three_power_comparison(
                test_ir,
                test_sensor,
                splits.test_only,
                "test",
            )
            if test_ir is not None
            and test_sensor is not None
            and frozenset(hf_test_powers) == splits.test_only
            else None
        )
        if test_comparison is not None:
            write_three_power_comparison(
                test_comparison,
                output / "test_three_power_comparison.json",
                output / "test_three_power_comparison.csv",
            )
        result = {
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_selection_score_c": best,
            "best_validation_ir_rmse_c": checkpoint["validation_rmse_c"],
            "best_validation_sensor": checkpoint.get("validation_sensor"),
            "world_size": world_size,
            "epochs_completed": epoch,
            "training_seconds": float(elapsed.item()),
            "parameter_count": parameter_count(base_model),
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "sensors_used": not skip_sensors,
            "lf_checkpoint_sha256": sha256_file(low_fidelity_checkpoint),
            "data_consumption_rank0": data_consumption,
            "identified_physics_parameters": (
                None if trainable_physics is None else trainable_physics.snapshot()
            ),
            "test_ir": test_ir,
            "test_sensor": test_sensor,
            "validation_ir": final_validation_ir,
            "validation_sensor": final_validation_sensor,
            "validation_three_power_comparison": validation_comparison,
            "test_three_power_comparison": test_comparison,
            "configuration": {
                "seed": seed,
                "correction_epochs": correction_epochs,
                "joint_epochs": joint_epochs,
                "batch_size_per_rank": batch_size,
                "physics_collocation_per_rank": physics_collocation,
                "patience": patience,
                "correction_width": width,
                "correction_depth": depth,
                "identify_physics_parameters": identify_physics_parameters,
                "silicon_carbide_emissivity_initial": silicon_carbide_emissivity_initial,
                "copper_emissivity_initial": copper_emissivity_initial,
                "contact_resistance_initial_m2_k_w": contact_resistance_initial_m2_k_w,
                "test_evaluation_enabled": evaluate_test,
                "physics_parameters_frozen": freeze_physics_parameters,
                "hard_deployment_constraints": hard_deployment_constraints,
                "hf_split_source": hf_split_source,
                "hf_train_powers_w": sorted(hf_train_powers),
                "hf_validation_powers_w": sorted(hf_validation_powers),
                "hf_test_powers_w": sorted(hf_test_powers),
                "sensor_absolute_weight": float(loss_cfg["sensor_absolute"]),
                "sensor_delta_weight": float(loss_cfg["sensor_delta"]),
                "validation_selection_weights": selection_weights,
                "correction_power_scaling": correction_power_scaling,
                "correction_power_reference_w": correction_power_reference_w,
                "correction_direct_power_input": correction_direct_power_input,
                "surface_teacher_weight": surface_teacher_weight,
                "surface_teacher_method": surface_teacher_method,
                "checkpoint_selection": checkpoint_selection,
                "hard_surface_residual_guide": hard_surface_residual_guide,
                "surface_guide_degree_r": surface_guide_degree_r,
                "surface_guide_degree_t": surface_guide_degree_t,
                "simulation_supervision_target": simulation_supervision_target,
            },
            "status": "completed",
            "material_passport": {
                "simulation_role": "low-fidelity pretraining and joint regularization",
                "experiment_role": "SiC top-surface high-fidelity correction",
                "sensor_role": "Cu bottom-ring supervision" if not skip_sensors else "not used",
                "simulation_data_used": True,
                "experiment_data_used": True,
                "sensor_data_used": not skip_sensors,
                "physics_loss_used": True,
                "physics_parameters_trainable": (
                    identify_physics_parameters and not freeze_physics_parameters
                ),
                "physics_parameter_status": (
                    "fixed_sensitivity_scenario"
                    if freeze_physics_parameters
                    else "preliminary_joint_optimization"
                    if identify_physics_parameters
                    else "fixed_from_verified_configuration"
                ),
                "internal_experiment_truth": "not available",
                "surface_teacher_used": surface_teacher_weight > 0.0,
                "hard_surface_residual_guide": hard_surface_residual_guide,
            },
        }
        (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return result


def _load_multifidelity_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[Path, dict[str, Any], nn.Module]:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_absolute():
        checkpoint_file = PROJECT_ROOT / checkpoint_file
    payload = torch.load(checkpoint_file, map_location=device, weights_only=False)
    if payload.get("method") not in {"multifidelity_correction", "prc_multifidelity"}:
        raise ValueError("Checkpoint is not a recognized multifidelity model")
    validate_hf_checkpoint_provenance(payload)
    scales = ModelScales(**payload["scales"])
    if payload["method"] == "prc_multifidelity":
        model = PRCMultifidelityModel(
            scales, **payload["model_kwargs"]
        ).to(device)
    else:
        low_fidelity = build_model(
            str(payload["low_fidelity_method"]),
            scales,
            **payload.get("low_fidelity_model_kwargs", {}),
        )
        surface_guide_spec = payload.get("surface_residual_guide_spec")
        surface_guide = (
            None
            if surface_guide_spec is None
            else ChebyshevSurfaceResidualGuide.from_spec(surface_guide_spec)
        )
        model = AdditiveCorrectionModel(
            low_fidelity,
            scales,
            freeze_low_fidelity=False,
            surface_residual_guide=surface_guide,
            **payload["correction_model_kwargs"],
        ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return checkpoint_file, payload, model


def evaluate_multifidelity_checkpoint(
    checkpoint_path: str | Path,
    release_manifest_path: str | Path,
    powers_w: Iterable[float] | None = None,
    output_path: str | Path | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate a selected MF checkpoint only on its declared held-out powers."""
    validate_release_checkpoint(release_manifest_path, checkpoint_path)
    device = torch.device(
        device_name if device_name is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_file, payload, model = _load_multifidelity_model(
        checkpoint_path, device
    )
    declared = frozenset(
        round(float(value), 4) for value in payload.get("hf_test_powers_w", [])
    )
    splits = build_power_splits()
    configured_test = splits.test_only
    if declared != configured_test:
        raise ValueError(
            "Checkpoint test powers do not match the current fixed test_Data protocol"
        )
    checkpoint_train = frozenset(
        round(float(value), 4) for value in payload.get("hf_train_powers_w", [])
    )
    checkpoint_validation = frozenset(
        round(float(value), 4)
        for value in payload.get("hf_validation_powers_w", [])
    )
    if (
        checkpoint_train != splits.hf_train
        or checkpoint_validation != splits.hf_validation
    ):
        raise ValueError(
            "Checkpoint train/validation powers do not match the current fixed protocol"
        )
    requested = declared if powers_w is None else frozenset(
        round(float(value), 4) for value in powers_w
    )
    if requested != declared:
        raise ValueError(
            "Evaluation powers must exactly match the checkpoint's declared held-out powers"
        )
    result = {
        "schema_version": 1,
        "temperature_error_unit": "℃",
        "checkpoint": str(checkpoint_file.relative_to(PROJECT_ROOT)),
        "seed": int(payload["seed"]),
        "test_powers_w": sorted(requested),
        "test_ir": _evaluate_ir_model(model, "test", device, requested),
        "test_sensor": (
            _evaluate_sensor_model(model, "test", device, requested)
            if payload.get("sensors_used", False)
            else None
        ),
        "selection_was_completed_before_test_evaluation": True,
        "internal_experiment_truth": "not available",
    }
    if output_path is not None:
        destination = Path(output_path)
        if not destination.is_absolute():
            destination = PROJECT_ROOT / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_multifidelity_validation(
    checkpoint_path: str | Path,
    output_json: str | Path = "reports/validation_three_power_comparison.json",
    output_csv: str | Path = "reports/validation_three_power_comparison.csv",
    device_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen model on the Experiment_data validation subset."""
    device = torch.device(
        device_name if device_name is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_file, payload, model = _load_multifidelity_model(
        checkpoint_path, device
    )
    splits = build_power_splits()
    validation_powers = splits.hf_validation
    trained = {
        round(float(value), 4) for value in payload.get("hf_train_powers_w", [])
    }
    declared = {
        round(float(value), 4)
        for value in payload.get("hf_validation_powers_w", [])
    }
    declared_test = {
        round(float(value), 4) for value in payload.get("hf_test_powers_w", [])
    }
    if (
        trained != splits.hf_train
        or declared != validation_powers
        or declared_test != splits.test_only
    ):
        raise RuntimeError(
            "Checkpoint powers do not match the current fixed train/validation/test protocol"
        )
    overlap = sorted(trained & validation_powers)
    if overlap:
        raise RuntimeError(
            f"Validation powers were present in checkpoint training data: {overlap}"
        )
    if not payload.get("sensors_used", False):
        raise RuntimeError(
            "Three-power validation requires a checkpoint trained with the Hot/Cold branch"
        )
    validation_ir = _evaluate_ir_model(
        model, "validation", device, validation_powers
    )
    validation_sensor = _evaluate_sensor_model(
        model, "validation", device, validation_powers
    )
    comparison = build_three_power_comparison(
        validation_ir, validation_sensor, validation_powers, "validation"
    )
    comparison.update(
        {
            "checkpoint": str(checkpoint_file.relative_to(PROJECT_ROOT)),
            "checkpoint_training_powers_w": sorted(trained),
        }
    )
    write_three_power_comparison(comparison, output_json, output_csv)
    return comparison


def evaluate_multifidelity_test_data(
    checkpoint_path: str | Path,
    release_manifest_path: str | Path,
    output_json: str | Path = "reports/test_three_power_comparison.json",
    output_csv: str | Path = "reports/test_three_power_comparison.csv",
    device_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen, protocol-compatible model once on test_Data."""
    device = torch.device(
        device_name if device_name is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    validate_release_checkpoint(release_manifest_path, checkpoint_path)
    checkpoint_file, payload, model = _load_multifidelity_model(
        checkpoint_path, device
    )
    splits = build_power_splits()
    test_powers = splits.test_only
    trained = {
        round(float(value), 4) for value in payload.get("hf_train_powers_w", [])
    }
    validated = {
        round(float(value), 4)
        for value in payload.get("hf_validation_powers_w", [])
    }
    declared_test = {
        round(float(value), 4) for value in payload.get("hf_test_powers_w", [])
    }
    if trained != splits.hf_train or validated != splits.hf_validation:
        raise RuntimeError(
            "Checkpoint train/validation powers do not match the current fixed protocol"
        )
    if declared_test != test_powers:
        raise RuntimeError(
            "Checkpoint test powers do not match the current fixed test_Data protocol"
        )
    overlap = sorted((trained | validated) & test_powers)
    if overlap:
        raise RuntimeError(
            f"test_Data powers were used for training or model selection: {overlap}"
        )
    if not payload.get("sensors_used", False):
        raise RuntimeError(
            "Three-power test requires a checkpoint trained with the Hot/Cold branch"
        )
    training = load_yaml("configs/training.yaml")
    evaluation_config = training["evaluation"]
    boundaries = load_resolved_boundary_conditions()
    loss_weights = training["loss_weights"]
    physics = PhysicsLossComputer(
        load_materials(),
        boundaries,
        PhysicsLossWeights(
            pde=float(loss_weights["pde"]),
            boundary=float(loss_weights["boundary"]),
            initial=float(loss_weights["initial"]),
            interface=float(loss_weights["interface"]),
        ),
    )
    collocation = sample_collocation(
        int(evaluation_config["physics_collocation"]),
        device,
        seed=int(evaluation_config["physics_seed"]),
    )
    physics_diagnostics = {
        name: float(value.detach())
        for name, value in physics(model, collocation).items()
    }
    test_ir = _evaluate_ir_model(model, "test", device, test_powers)
    test_sensor = _evaluate_sensor_model(model, "test", device, test_powers)
    comparison = build_three_power_comparison(
        test_ir, test_sensor, test_powers, "test"
    )
    comparison.update(
        {
            "checkpoint": str(checkpoint_file.relative_to(PROJECT_ROOT)),
            "seed": int(payload["seed"]),
            "checkpoint_training_powers_w": sorted(trained),
            "checkpoint_validation_powers_w": sorted(validated),
            "physics_diagnostics": {
                "used_for_model_selection": False,
                "collocation_count_per_component": int(
                    evaluation_config["physics_collocation"]
                ),
                "seed": int(evaluation_config["physics_seed"]),
                "losses": physics_diagnostics,
            },
        }
    )
    write_three_power_comparison(comparison, output_json, output_csv)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the common additive multi-fidelity corrector")
    parser.add_argument("--lf-checkpoint", required=True)
    parser.add_argument("--output", default="reports/runs/multifidelity_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--correction-epochs", type=int, default=1500)
    parser.add_argument("--joint-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--physics-collocation", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--sensor-absolute-weight", type=float)
    parser.add_argument("--sensor-delta-weight", type=float)
    parser.add_argument("--hf-train-powers", type=float, nargs="+")
    parser.add_argument("--hf-validation-powers", type=float, nargs="+")
    parser.add_argument("--hf-test-powers", type=float, nargs="+")
    parser.add_argument(
        "--correction-power-scaling", choices=("none", "linear"), default="none"
    )
    parser.add_argument("--correction-power-reference-w", type=float, default=400.0)
    parser.add_argument("--no-correction-direct-power-input", action="store_true")
    parser.add_argument("--surface-teacher-weight", type=float, default=0.0)
    parser.add_argument(
        "--surface-teacher-method",
        default="regression_residual_per_watt",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=("validation", "final_epoch"),
        default="validation",
    )
    parser.add_argument("--hard-surface-residual-guide", action="store_true")
    parser.add_argument("--surface-guide-degree-r", type=int, default=20)
    parser.add_argument("--surface-guide-degree-t", type=int, default=20)
    parser.add_argument("--identify-physics-parameters", action="store_true")
    parser.add_argument("--silicon-carbide-emissivity-initial", type=float, default=0.5)
    parser.add_argument("--copper-emissivity-initial", type=float, default=0.5)
    parser.add_argument("--contact-resistance-initial", type=float)
    parser.add_argument(
        "--simulation-supervision-target",
        choices=("low_fidelity", "high_fidelity_pullback"),
        default="low_fidelity",
        help="Use high_fidelity_pullback only as the named ablation.",
    )
    parser.add_argument(
        "--freeze-physics-parameters",
        action="store_true",
        help=(
            "Use supplied inverse-identification values as a fixed sensitivity scenario "
            "instead of optimizing them."
        ),
    )
    parser.add_argument(
        "--skip-sensors",
        action="store_true",
        help="IR-only ablation; never label its output as the complete protocol.",
    )
    args = parser.parse_args()
    try:
        result = train_multifidelity(
            low_fidelity_checkpoint=args.lf_checkpoint,
            output_directory=args.output,
            seed=args.seed,
            correction_epochs=args.correction_epochs,
            joint_epochs=args.joint_epochs,
            batch_size=args.batch_size,
            physics_collocation=args.physics_collocation,
            width=args.width,
            depth=args.depth,
            skip_sensors=args.skip_sensors,
            patience=args.patience,
            identify_physics_parameters=args.identify_physics_parameters,
            silicon_carbide_emissivity_initial=(
                args.silicon_carbide_emissivity_initial
            ),
            copper_emissivity_initial=args.copper_emissivity_initial,
            contact_resistance_initial_m2_k_w=args.contact_resistance_initial,
            evaluate_test=False,
            freeze_physics_parameters=args.freeze_physics_parameters,
            sensor_absolute_weight=args.sensor_absolute_weight,
            sensor_delta_weight=args.sensor_delta_weight,
            hf_train_powers_w=args.hf_train_powers,
            hf_validation_powers_w=args.hf_validation_powers,
            hf_test_powers_w=args.hf_test_powers,
            correction_power_scaling=args.correction_power_scaling,
            correction_power_reference_w=args.correction_power_reference_w,
            correction_direct_power_input=not args.no_correction_direct_power_input,
            surface_teacher_weight=args.surface_teacher_weight,
            surface_teacher_method=args.surface_teacher_method,
            checkpoint_selection=args.checkpoint_selection,
            hard_surface_residual_guide=args.hard_surface_residual_guide,
            surface_guide_degree_r=args.surface_guide_degree_r,
            surface_guide_degree_t=args.surface_guide_degree_t,
            simulation_supervision_target=args.simulation_supervision_target,
        )
    except PhysicsConfigurationError as error:
        parser.exit(2, f"BLOCKED_UNVERIFIED_METADATA: {error}\n")
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
