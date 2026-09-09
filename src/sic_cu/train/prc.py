from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import build_power_splits
from sic_cu.eval.metrics import macro_v1_selection_score
from sic_cu.eval.protocol_checks import (
    checkpoint_provenance,
    current_protocol_fingerprints,
    validate_lf_checkpoint_provenance,
)
from sic_cu.losses import PhysicsLossComputer, PhysicsLossWeights
from sic_cu.models import ModelScales, PRCMultifidelityModel
from sic_cu.models.common import parameter_count
from sic_cu.physics import (
    load_materials,
    load_resolved_boundary_conditions,
    resolved_boundary_snapshot,
    sample_collocation,
)
from sic_cu.train.common import physics_optimizer_step, write_config_snapshot
from sic_cu.train.multifidelity import (
    _evaluate_ir_model,
    _evaluate_sensor_model,
    _gradient_l2,
    _ir_dataset,
    _macro_sensor_training_losses,
    _sensor_tensors,
)
from sic_cu.train.simulation import (
    distributed_context,
    load_sampled_points,
    set_seed,
    train_simulation_model,
)


def _load_prc_lf_checkpoint(
    checkpoint_path: str | Path, device: torch.device, variant: str
) -> tuple[PRCMultifidelityModel, dict[str, Any], Path]:
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("method") != "prc_lf":
        raise RuntimeError("PRC correction requires a prc_lf checkpoint")
    validate_lf_checkpoint_provenance(payload)
    kwargs = dict(payload["model_kwargs"])
    kwargs["correction_variant"] = variant
    model = PRCMultifidelityModel(
        ModelScales(**payload["scales"]), **kwargs
    ).to(device)
    model.load_state_dict(payload["model_state"])
    return model, payload, path


def _set_active_correction_parameters(model: PRCMultifidelityModel) -> list[nn.Parameter]:
    for parameter in model.high_fidelity_parameters():
        parameter.requires_grad_(False)
    active = list(model.active_high_fidelity_parameters())
    for parameter in active:
        parameter.requires_grad_(True)
    return active


def _broadcast_validation(value: dict[str, Any] | None) -> dict[str, Any]:
    if not dist.is_initialized():
        if value is None:
            raise RuntimeError("Rank-zero validation result is missing")
        return value
    items = [value]
    dist.broadcast_object_list(items, src=0)
    if items[0] is None:
        raise RuntimeError("Distributed validation broadcast failed")
    return items[0]


def train_prc_multifidelity(
    low_fidelity_checkpoint: str,
    output_directory: str,
    *,
    correction_variant: str = "amplitude_time",
    seed: int = 0,
    correction_epochs: int = 1500,
    joint_epochs: int = 0,
    batch_size: int = 2048,
    physics_collocation: int = 256,
    patience: int = 200,
) -> dict[str, Any] | None:
    if correction_epochs < 1 or joint_epochs < 0:
        raise ValueError("correction_epochs must be positive and joint_epochs nonnegative")
    rank, local_rank, world_size = distributed_context()
    set_seed(seed)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        torch.cuda.reset_peak_memory_stats(device)

    splits = build_power_splits()
    fingerprints = current_protocol_fingerprints()
    base_model, lf_payload, lf_path = _load_prc_lf_checkpoint(
        low_fidelity_checkpoint, device, correction_variant
    )
    base_model.freeze_low_fidelity(True)
    correction_parameters = _set_active_correction_parameters(base_model)
    model: nn.Module = (
        DistributedDataParallel(base_model, device_ids=[local_rank])
        if world_size > 1
        else base_model
    )

    train_data = _ir_dataset(split="train", powers_w=splits.hf_train)
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
    sensor = _sensor_tensors(device, split="train", powers_w=splits.hf_train)
    training = load_yaml("configs/training.yaml")
    weights = training["loss_weights"]
    boundaries = load_resolved_boundary_conditions()
    physics = PhysicsLossComputer(
        load_materials(),
        boundaries,
        PhysicsLossWeights(
            pde=float(weights["pde"]),
            boundary=float(weights["boundary"]),
            initial=float(weights["initial"]),
            interface=float(weights["interface"]),
        ),
    )

    def make_optimizer(parameters: list[nn.Parameter], learning_rate: float):
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=float(training["optimizer"]["weight_decay"]),
        )

    optimizer = make_optimizer(
        correction_parameters, float(training["optimizer"]["learning_rate"])
    )
    output = PROJECT_ROOT / output_directory
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "training.jsonl").write_text("", encoding="utf-8")
        write_config_snapshot(output)
    if dist.is_initialized():
        dist.barrier()

    best_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    total_epochs = correction_epochs + joint_epochs
    simulation_loader = None
    simulation_iterator = None
    consumed = {
        "hf_ir_points": 0,
        "hf_sensor_points": 0,
        "lf_simulation_points": 0,
        "lf_copper_points": 0,
        "lf_silicon_carbide_points": 0,
    }
    started = time.perf_counter()
    for epoch in range(1, total_epochs + 1):
        stage = "correction" if epoch <= correction_epochs else "joint"
        if epoch == correction_epochs + 1:
            base_model.freeze_low_fidelity(False)
            if world_size > 1:
                model = DistributedDataParallel(base_model, device_ids=[local_rank])
            optimizer = make_optimizer(
                [
                    parameter
                    for parameter in base_model.parameters()
                    if parameter.requires_grad
                ],
                float(training["optimizer"]["joint_learning_rate"]),
            )
            simulation_data = load_sampled_points(
                sorted(splits.simulation_train),
                int(training["multifidelity"]["joint_simulation_samples_per_power"]),
                70_000 + seed,
                balanced=True,
            )
            simulation_sampler = (
                DistributedSampler(
                    simulation_data, world_size, rank, shuffle=True, seed=seed
                )
                if world_size > 1
                else None
            )
            simulation_loader = DataLoader(
                simulation_data,
                batch_size=batch_size,
                shuffle=simulation_sampler is None,
                sampler=simulation_sampler,
            )
            simulation_iterator = iter(simulation_loader)
            stale_epochs = 0
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_losses = {"ir": 0.0, "sensor": 0.0, "lf": 0.0}
        batches = 0
        lf_gradient = 0.0
        hf_gradient = 0.0
        for coordinates, target, weight in train_loader:
            coordinates = coordinates.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(coordinates, fidelity="high")
            ir_loss = (
                weight
                * ((prediction - target) / base_model.scales.temperature_scale_k).pow(2)
            ).sum() / weight.sum()
            sensor_coordinates, sensor_target, sensor_delta, baseline = sensor
            sensor_prediction = model(sensor_coordinates, fidelity="high")
            sensor_absolute_loss, sensor_delta_loss = _macro_sensor_training_losses(
                sensor_prediction,
                sensor_target,
                sensor_delta,
                baseline,
                sensor_coordinates,
                base_model.scales.temperature_scale_k,
            )
            sensor_loss = (
                float(weights["sensor_absolute"]) * sensor_absolute_loss
                + float(weights["sensor_delta"]) * sensor_delta_loss
            )
            total = float(weights["ir"]) * ir_loss + sensor_loss
            lf_loss = torch.zeros((), device=device)
            if stage == "joint" and simulation_iterator is not None:
                try:
                    lf_coordinates, lf_target = next(simulation_iterator)
                except StopIteration:
                    simulation_iterator = iter(simulation_loader)
                    lf_coordinates, lf_target = next(simulation_iterator)
                lf_coordinates = lf_coordinates.to(device)
                lf_target = lf_target.to(device)
                lf_prediction = model(lf_coordinates, fidelity="low")
                lf_loss = (
                    (lf_prediction - lf_target)
                    / base_model.scales.temperature_scale_k
                ).pow(2).mean()
                total = total + float(weights["low_fidelity"]) * lf_loss
                consumed["lf_simulation_points"] += int(len(lf_coordinates))
                consumed["lf_copper_points"] += int((lf_coordinates[:, 4] < 0.5).sum())
                consumed["lf_silicon_carbide_points"] += int(
                    (lf_coordinates[:, 4] >= 0.5).sum()
                )
            total.backward()
            lf_gradient = _gradient_l2(base_model.low_fidelity_parameters())
            hf_gradient = _gradient_l2(base_model.active_high_fidelity_parameters())
            optimizer.step()
            epoch_losses["ir"] += float(ir_loss.detach())
            epoch_losses["sensor"] += float(sensor_loss.detach())
            epoch_losses["lf"] += float(lf_loss.detach())
            consumed["hf_ir_points"] += int(len(coordinates))
            consumed["hf_sensor_points"] += int(len(sensor_coordinates))
            batches += 1

        collocation = sample_collocation(
            physics_collocation,
            device,
            seed=seed * 1_000_000 + epoch * world_size + rank,
        )
        physics_components = physics_optimizer_step(model, optimizer, physics, collocation)
        validation = None
        if rank == 0:
            validation_ir = _evaluate_ir_model(
                base_model, "validation", device, splits.hf_validation
            )
            validation_sensor = _evaluate_sensor_model(
                base_model, "validation", device, splits.hf_validation
            )
            score = macro_v1_selection_score(
                validation_ir["rmse_c"],
                validation_sensor["absolute_rmse_c"],
                validation_sensor["delta_rmse_c"],
            )
            validation = {
                "ir": validation_ir,
                "sensor": validation_sensor,
                "score_c": score,
            }
        validation = _broadcast_validation(validation)
        score = float(validation["score_c"])
        improved = score < best_score - 1e-4
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            if rank == 0:
                model_kwargs = dict(lf_payload["model_kwargs"])
                model_kwargs["correction_variant"] = correction_variant
                torch.save(
                    {
                        "schema_version": 1,
                        "method": "prc_multifidelity",
                        "seed": seed,
                        "epoch": epoch,
                        "model_state": base_model.state_dict(),
                        "model_kwargs": model_kwargs,
                        "scales": asdict(base_model.scales),
                        "validation_selection_score_c": score,
                        "validation_ir": validation["ir"],
                        "validation_sensor": validation["sensor"],
                        "hf_train_powers_w": sorted(splits.hf_train),
                        "hf_validation_powers_w": sorted(splits.hf_validation),
                        "hf_test_powers_w": sorted(splits.hf_test),
                        "sensors_used": True,
                        "provenance": checkpoint_provenance(
                            role="prc_multifidelity",
                            train_powers_w=splits.hf_train,
                            validation_powers_w=splits.hf_validation,
                            fingerprints=fingerprints,
                        )
                        | {
                            "lf_checkpoint_sha256": sha256_file(lf_path),
                            "lf_checkpoint_provenance": lf_payload["provenance"],
                        },
                        "resolved_physics": resolved_boundary_snapshot(boundaries),
                        "material_passport": {
                            "simulation_data_used": True,
                            "experiment_data_used": True,
                            "sensor_data_used": True,
                            "physics_loss_used": True,
                            "internal_experiment_truth": "not available",
                        },
                    },
                    output / "best.pt",
                )
        else:
            stale_epochs += 1
        if rank == 0:
            with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "stage": stage,
                            "selection_metric_version": "macro_v1",
                            "validation_selection_score_c": score,
                            "lf_parameters_frozen": stage == "correction",
                            "lf_gradient_l2": lf_gradient,
                            "hf_gradient_l2": hf_gradient,
                            **{
                                f"loss_{name}": value / max(batches, 1)
                                for name, value in epoch_losses.items()
                            },
                            **{
                                f"loss_{name}": float(value.detach())
                                for name, value in physics_components.items()
                            },
                        }
                    )
                    + "\n"
                )
        stop = torch.tensor(
            int(stale_epochs >= patience and (stage == "joint" or joint_epochs == 0)),
            device=device,
        )
        if dist.is_initialized():
            dist.broadcast(stop, src=0)
        if bool(stop.item()):
            break

    if joint_epochs > 0 and (
        consumed["lf_simulation_points"] == 0
        or consumed["lf_copper_points"] == 0
        or consumed["lf_silicon_carbide_points"] == 0
    ):
        raise RuntimeError("PRC joint stage failed to consume both LF materials")
    elapsed = time.perf_counter() - started
    result = None
    if rank == 0:
        checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
        base_model.load_state_dict(checkpoint["model_state"])
        result = {
            "status": "completed_validation_only",
            "method": "prc_multifidelity",
            "correction_variant": correction_variant,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_selection_score_c": best_score,
            "selection_metric_version": "macro_v1",
            "epochs_completed": epoch,
            "training_seconds": elapsed,
            "parameter_count": parameter_count(base_model),
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "data_consumption_rank0": consumed,
            "validation_ir": checkpoint["validation_ir"],
            "validation_sensor": checkpoint["validation_sensor"],
            "test": None,
            "test_status": "sealed_until_frozen_release",
        }
        (output / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PRC multifidelity candidate")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    lf = subparsers.add_parser("lf", help="Fit LF response on simulation train")
    lf.add_argument("--output", default="reports/runs/prc_lf_k8_seed0")
    lf.add_argument("--seed", type=int, default=0)
    lf.add_argument("--modes", type=int, choices=(8, 16, 32), default=8)
    lf.add_argument("--epochs", type=int, default=2000)
    lf.add_argument("--samples-per-power", type=int, default=8192)
    lf.add_argument("--validation-samples-per-power", type=int, default=8192)
    lf.add_argument("--batch-size", type=int, default=8192)
    lf.add_argument("--learning-rate", type=float, default=1e-3)
    lf.add_argument("--patience", type=int, default=200)
    lf.add_argument("--width", type=int, default=64)
    lf.add_argument("--depth", type=int, default=3)

    correction = subparsers.add_parser("correct", help="Fit HF correction on train/validation")
    correction.add_argument("--lf-checkpoint", required=True)
    correction.add_argument("--output", default="reports/runs/prc_at_seed0")
    correction.add_argument(
        "--variant",
        choices=("temperature_residual", "amplitude", "amplitude_time"),
        default="amplitude_time",
    )
    correction.add_argument("--seed", type=int, default=0)
    correction.add_argument("--correction-epochs", type=int, default=1500)
    correction.add_argument("--joint-epochs", type=int, default=0)
    correction.add_argument("--batch-size", type=int, default=2048)
    correction.add_argument("--physics-collocation", type=int, default=256)
    correction.add_argument("--patience", type=int, default=200)
    args = parser.parse_args()

    if args.stage == "lf":
        result = train_simulation_model(
            method="prc_lf",
            seed=args.seed,
            output_directory=args.output,
            epochs=args.epochs,
            samples_per_power=args.samples_per_power,
            validation_samples_per_power=args.validation_samples_per_power,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            model_kwargs={
                "modes": args.modes,
                "width": args.width,
                "depth": args.depth,
                "activation": "tanh",
                "correction_variant": "amplitude_time",
            },
            physics_collocation=0,
            physics_weight=0.0,
        )
    else:
        result = train_prc_multifidelity(
            args.lf_checkpoint,
            args.output,
            correction_variant=args.variant,
            seed=args.seed,
            correction_epochs=args.correction_epochs,
            joint_epochs=args.joint_epochs,
            batch_size=args.batch_size,
            physics_collocation=args.physics_collocation,
            patience=args.patience,
        )
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
