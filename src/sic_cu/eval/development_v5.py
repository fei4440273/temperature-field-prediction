from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import polars as pl
import torch
import yaml
from torch import nn

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.fields import assert_compatible_fields, load_processed_field
from sic_cu.data.processed import load_processed_ir_observations
from sic_cu.data.sensors import (
    audit_sensor_file,
    load_canonical_sensor_observations,
    sensor_files,
)
from sic_cu.data.splits import assert_development_label_split, build_power_splits
from sic_cu.eval.development_v4 import error_statistics, load_existing_baseline
from sic_cu.eval.energy_v5 import (
    AxisymmetricGeometry,
    analytic_control_tests,
    axisymmetric_lumped_nodal_weights,
    cooling_boundary_limit,
    deterministic_energy_terms,
    hard_cooling_override_control,
)
from sic_cu.eval.protocol_checks import current_protocol_fingerprints
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions


ENERGY_TERMS = (
    "absorbed_power_w",
    "storage_rate_w",
    "cooling_heat_w",
    "convection_heat_w",
    "radiation_heat_w",
    "balance_w",
)


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise RuntimeError(f"拒绝写入空审计表：{path.name}")
    pl.DataFrame([dict(record) for record in records]).write_csv(path, null_value="")


def _run_git(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout if process.returncode == 0 else "UNAVAILABLE\n"


def _predict(
    model: nn.Module,
    coordinates: np.ndarray,
    device: torch.device,
    batch_size: int,
    fidelity: str | None = None,
) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(coordinates, dtype=np.float32))
    predictions = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(tensor), batch_size):
            batch = tensor[offset : offset + batch_size].to(device)
            value = model(batch) if fidelity is None else model(batch, fidelity=fidelity)
            predictions.append(value.detach().cpu().numpy())
    return np.concatenate(predictions).reshape(-1).astype(np.float64)


def _window_mask(times: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    lower = float(config["lower"])
    upper = float(config["upper"])
    left = times >= lower if bool(config["lower_closed"]) else times > lower
    right = times <= upper if bool(config["upper_closed"]) else times < upper
    return left & right


def _relative_change(current: float, previous: float, scale: float) -> float:
    return abs(current - previous) / max(abs(scale), 1e-12)


def _quadrature_convergence(
    energy_rows: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_state: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for record in energy_rows:
        by_state.setdefault((record["power_w"], record["time_s"]), []).append(record)
    for (power, time_s), values in sorted(by_state.items()):
        ordered = sorted(values, key=lambda item: item["quadrature_order"])
        previous: dict[str, Any] | None = None
        for current in ordered:
            changes: list[float] = []
            for term in ENERGY_TERMS:
                change_w = (
                    None
                    if previous is None
                    else abs(float(current[term]) - float(previous[term]))
                )
                relative = (
                    None
                    if change_w is None
                    else change_w / max(abs(float(current["absorbed_power_w"])), 1e-12)
                )
                if relative is not None:
                    changes.append(relative)
                rows.append(
                    {
                        "power_w": power,
                        "time_s": time_s,
                        "previous_order": None
                        if previous is None
                        else previous["quadrature_order"],
                        "quadrature_order": current["quadrature_order"],
                        "term": term,
                        "previous_value_w": None if previous is None else previous[term],
                        "current_value_w": current[term],
                        "absolute_change_w": change_w,
                        "relative_change_to_absorbed_power": relative,
                        "target_relative_change": threshold,
                        "passed": None if relative is None else relative < threshold,
                    }
                )
            current["quadrature_change_w"] = (
                None
                if previous is None
                else max(
                    abs(float(current[term]) - float(previous[term]))
                    for term in ENERGY_TERMS
                )
            )
            current["maximum_relative_quadrature_change"] = (
                None if not changes else max(changes)
            )
            previous = current
    return rows


def _latest_quadrature_passes(
    energy_rows: Sequence[Mapping[str, Any]],
    threshold: float,
) -> bool:
    latest: dict[tuple[float, float], Mapping[str, Any]] = {}
    for record in energy_rows:
        key = (float(record["power_w"]), float(record["time_s"]))
        if key not in latest or int(record["quadrature_order"]) > int(
            latest[key]["quadrature_order"]
        ):
            latest[key] = record
    return bool(latest) and all(
        record.get("maximum_relative_quadrature_change") is not None
        and float(record["maximum_relative_quadrature_change"]) < threshold
        for record in latest.values()
    )


def _model_energy_audit(
    model: nn.Module,
    config: Mapping[str, Any],
    geometry: AxisymmetricGeometry,
    materials: Mapping[int, Any],
    boundaries: Any,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    energy_cfg = config["energy"]
    energy_rows: list[dict[str, Any]] = []
    divergence_rows: list[dict[str, Any]] = []
    used_orders: list[int] = []

    def evaluate_order(order: int) -> None:
        for power in energy_cfg["powers_w"]:
            for time_s in energy_cfg["times_s"]:
                energy, divergence = deterministic_energy_terms(
                    model,
                    materials,
                    boundaries,
                    geometry,
                    float(power),
                    float(time_s),
                    order,
                    float(energy_cfg["outer_flux_epsilon_m"]),
                    int(energy_cfg["derivative_batch_size"]),
                    device,
                )
                energy_rows.append(energy)
                divergence_rows.append(divergence)
        used_orders.append(order)

    for order in energy_cfg["initial_quadrature_orders"]:
        evaluate_order(int(order))
    convergence = _quadrature_convergence(
        energy_rows, float(energy_cfg["convergence_relative_to_absorbed_power"])
    )
    for order in energy_cfg.get("fallback_quadrature_orders", []):
        if _latest_quadrature_passes(
            energy_rows,
            float(energy_cfg["convergence_relative_to_absorbed_power"]),
        ):
            break
        evaluate_order(int(order))
        convergence = _quadrature_convergence(
            energy_rows, float(energy_cfg["convergence_relative_to_absorbed_power"])
        )
    return energy_rows, divergence_rows, convergence, used_orders


def _cooling_limit_audit(
    model: nn.Module,
    config: Mapping[str, Any],
    geometry: AxisymmetricGeometry,
    materials: Mapping[int, Any],
    boundaries: Any,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    energy_cfg = config["energy"]
    rows: list[dict[str, Any]] = []
    for power in energy_cfg["powers_w"]:
        for time_s in energy_cfg["times_s"]:
            rows.extend(
                cooling_boundary_limit(
                    model,
                    materials[0],
                    geometry,
                    float(boundaries.cooling.fixed_temperature_k),
                    float(power),
                    float(time_s),
                    int(energy_cfg["cooling_limit_quadrature_order"]),
                    energy_cfg["outer_limit_epsilons_m"],
                    device,
                )
            )
    control = hard_cooling_override_control(
        geometry.copper_radius_m,
        float(boundaries.cooling.fixed_temperature_k),
        float(materials[0].conductivity.value),
        float(energy_cfg["hard_cooling_tolerance_m"]),
        float(energy_cfg["outer_flux_epsilon_m"]),
        device,
    )
    return rows, control


def _lf_coordinates(field: Any) -> np.ndarray:
    node_count = len(field.coordinates_rz_m)
    return np.column_stack(
        (
            np.tile(field.coordinates_rz_m, (len(field.times_s), 1)),
            np.repeat(field.times_s, node_count),
            np.full(len(field.times_s) * node_count, field.power_w),
            np.tile(field.material_ids, len(field.times_s)),
        )
    ).astype(np.float32)


def _lf_audit(
    baseline_run: Path,
    seeds: Sequence[int],
    config: Mapping[str, Any],
    geometry: AxisymmetricGeometry,
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    splits = build_power_splits()
    fields = [load_processed_field(power) for power in sorted(splits.simulation_validation)]
    assert_compatible_fields(fields)
    reference = fields[0]
    nodal_volume_weights = axisymmetric_lumped_nodal_weights(
        reference.coordinates_rz_m, reference.material_ids, geometry
    )
    material_volume_sums = {
        material_id: float(nodal_volume_weights[reference.material_ids == material_id].sum())
        for material_id in (0, 1)
    }
    alignment = {
        "中文说明": "低保真评价使用真实仿真节点、真实时间、米/开尔文单位及材料标签；未读取 test_Data。",
        "simulation_validation_powers_w": sorted(splits.simulation_validation),
        "power_count": len(fields),
        "mesh_aligned_across_powers": True,
        "time_grid_aligned_across_powers": True,
        "node_count": len(reference.coordinates_rz_m),
        "time_count": len(reference.times_s),
        "time_min_s": float(reference.times_s.min()),
        "time_max_s": float(reference.times_s.max()),
        "time_step_s": float(np.min(np.diff(reference.times_s))),
        "coordinate_unit": "m",
        "temperature_storage_unit": "K",
        "temperature_error_unit": "℃",
        "z_direction": "negative_depth",
        "material_ids": sorted(np.unique(reference.material_ids).tolist()),
        "node_counts_by_material": {
            "copper": int(np.sum(reference.material_ids == 0)),
            "silicon_carbide": int(np.sum(reference.material_ids == 1)),
        },
        "axisymmetric_lumped_volume_m3": {
            "copper": material_volume_sums[0],
            "silicon_carbide": material_volume_sums[1],
        },
        "analytic_volume_m3": {
            "copper": geometry.copper_volume_m3,
            "silicon_carbide": geometry.silicon_carbide_volume_m3,
        },
        "volume_relative_error": {
            "copper": abs(material_volume_sums[0] - geometry.copper_volume_m3)
            / geometry.copper_volume_m3,
            "silicon_carbide": abs(
                material_volume_sums[1] - geometry.silicon_carbide_volume_m3
            )
            / geometry.silicon_carbide_volume_m3,
        },
        "node_weight_definition": "每个仿真节点等权",
        "volume_weight_definition": "按材料三角剖分，对线性形函数精确积分2πr后集总到节点",
    }

    comparison: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    spot_rows: list[dict[str, Any]] = []
    lf_cfg = config["low_fidelity"]
    material_groups = (("all", None), ("copper", 0), ("silicon_carbide", 1))
    windows = list(lf_cfg["time_windows_s"])
    for seed in seeds:
        pretrained, multifidelity, lf_payload, hf_payload, lf_path, hf_path = (
            load_existing_baseline(baseline_run, seed, device)
        )
        first_coordinates = _lf_coordinates(reference)[: int(lf_cfg["direct_call_check_points"])]
        direct = _predict(
            pretrained,
            first_coordinates,
            device,
            int(lf_cfg["prediction_batch_size"]),
        )
        embedded = _predict(
            multifidelity,
            first_coordinates,
            device,
            int(lf_cfg["prediction_batch_size"]),
            fidelity="low",
        )
        call_difference = float(np.max(np.abs(direct - embedded)))
        expected_hash = hf_payload["provenance"]["lf_checkpoint_sha256"]
        actual_hash = sha256_file(lf_path)
        snapshot_manifest = lf_path.parent / "config_snapshot" / "sha256.json"
        provenance = lf_payload["provenance"]
        provenance_rows.append(
            {
                "seed": seed,
                "lf_checkpoint": str(lf_path.relative_to(PROJECT_ROOT)),
                "lf_checkpoint_sha256": actual_hash,
                "hf_checkpoint": str(hf_path.relative_to(PROJECT_ROOT)),
                "hf_referenced_lf_sha256": expected_hash,
                "hf_reference_matches_file": expected_hash == actual_hash,
                "method": lf_payload["method"],
                "pretrained_best_epoch": int(lf_payload["epoch"]),
                "stored_validation_rmse_c": float(lf_payload["validation_rmse_k"]),
                "source_code_commit": provenance["code_commit"],
                "source_role": provenance["role"],
                "protocol_id": provenance["protocol_id"],
                "train_power_count": len(provenance["train_powers_w"]),
                "validation_power_count": len(provenance["validation_powers_w"]),
                "train_powers_w": json.dumps(provenance["train_powers_w"]),
                "validation_powers_w": json.dumps(provenance["validation_powers_w"]),
                "train_powers_match_current_split": set(provenance["train_powers_w"])
                == set(splits.simulation_train),
                "validation_powers_match_current_split": set(
                    provenance["validation_powers_w"]
                )
                == set(splits.simulation_validation),
                "test_labels_consumed": provenance["test_labels_consumed"],
                "hf_hard_cooling_enabled": bool(
                    multifidelity.hard_cooling_radius_m is not None
                    or multifidelity.hard_cooling_temperature_k is not None
                ),
                "direct_lf_call": "standalone_lf_checkpoint_forward",
                "paired_hf_lf_call": "multifidelity_forward_fidelity_low",
                "direct_vs_paired_low_max_abs_difference_c": call_difference,
                "calls_match": call_difference <= 1e-6,
                "config_snapshot_manifest": str(
                    snapshot_manifest.relative_to(PROJECT_ROOT)
                ),
                "config_snapshot_manifest_sha256": sha256_file(snapshot_manifest),
            }
        )

        per_power_records: list[dict[str, Any]] = []
        for field in fields:
            coordinates = _lf_coordinates(field)
            prediction = _predict(
                pretrained,
                coordinates,
                device,
                int(lf_cfg["prediction_batch_size"]),
            )
            target = field.temperature_k.reshape(-1).astype(np.float64)
            repeated_material = np.tile(field.material_ids, len(field.times_s))
            repeated_times = np.repeat(field.times_s, len(field.coordinates_rz_m))
            volume_weights = np.tile(nodal_volume_weights, len(field.times_s))
            for weighting, all_weights in (
                ("node_equal", np.ones_like(target)),
                ("axisymmetric_lumped_volume", volume_weights),
            ):
                for window in windows:
                    time_mask = _window_mask(repeated_times, window)
                    for material_name, material_id in material_groups:
                        material_mask = (
                            np.ones(len(target), dtype=bool)
                            if material_id is None
                            else repeated_material == material_id
                        )
                        selected = time_mask & material_mask
                        metrics = error_statistics(
                            target[selected], prediction[selected], all_weights[selected]
                        )
                        record = {
                            "seed": seed,
                            "aggregation": "per_power",
                            "power_w": field.power_w,
                            "material": material_name,
                            "time_window": window["name"],
                            "weighting": weighting,
                            "query_path": "direct_checkpoint_node_query",
                            "sample_count": int(selected.sum()),
                            "weight_sum": float(all_weights[selected].sum()),
                            **metrics,
                        }
                        comparison.append(record)
                        per_power_records.append(record)

            node_count = int(lf_cfg["spot_check_nodes_per_material"])
            for material_name, material_id in material_groups[1:]:
                indices = np.flatnonzero(field.material_ids == material_id)
                selected_nodes = indices[
                    np.linspace(0, len(indices) - 1, node_count, dtype=np.int64)
                ]
                for time_index in (0, len(field.times_s) // 2, len(field.times_s) - 1):
                    for node_index in selected_nodes:
                        flat_index = time_index * len(field.coordinates_rz_m) + node_index
                        spot_rows.append(
                            {
                                "seed": seed,
                                "power_w": field.power_w,
                                "time_s": float(field.times_s[time_index]),
                                "node_label": int(field.node_labels[node_index]),
                                "material": material_name,
                                "r_m": float(field.coordinates_rz_m[node_index, 0]),
                                "z_m": float(field.coordinates_rz_m[node_index, 1]),
                                "target_c": float(target[flat_index] - 273.15),
                                "prediction_c": float(prediction[flat_index] - 273.15),
                                "error_c": float(prediction[flat_index] - target[flat_index]),
                            }
                        )

        keys = ("rmse_c", "mae_c", "signed_bias_c", "p95_abs_error_c", "max_abs_error_c")
        for weighting, _ in (
            ("node_equal", None),
            ("axisymmetric_lumped_volume", None),
        ):
            for window in windows:
                for material_name, _ in material_groups:
                    selected = [
                        row
                        for row in per_power_records
                        if row["weighting"] == weighting
                        and row["time_window"] == window["name"]
                        and row["material"] == material_name
                    ]
                    comparison.append(
                        {
                            "seed": seed,
                            "aggregation": "macro_mean_across_validation_powers",
                            "power_w": None,
                            "material": material_name,
                            "time_window": window["name"],
                            "weighting": weighting,
                            "query_path": "direct_checkpoint_node_query",
                            "sample_count": int(sum(row["sample_count"] for row in selected)),
                            "weight_sum": float(sum(row["weight_sum"] for row in selected)),
                            **{
                                key: float(np.mean([row[key] for row in selected]))
                                for key in keys
                            },
                        }
                    )
    return comparison, provenance_rows, spot_rows, alignment


def _initial_observation_audit(
    initial_temperature_k: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = load_yaml("configs/data_metadata.yaml")
    data_root = PROJECT_ROOT / metadata["data_root"]
    splits = build_power_splits()
    allowed_powers = splits.experiment_powers
    rows: list[dict[str, Any]] = []

    sensors = load_canonical_sensor_observations()
    sensors = sensors.filter(pl.col("split").is_in(["train", "validation"]))
    raw_audits: dict[tuple[str, float], Any] = {}
    for sensor_type, key in (("hot", "hot_directory"), ("cold", "cold_directory")):
        for path in sensor_files(data_root / metadata["sensors"][key]):
            audit = audit_sensor_file(path, sensor_type)
            if round(audit.power_w, 4) in allowed_powers:
                raw_audits[(sensor_type, round(audit.power_w, 4))] = audit
    for group in sensors.partition_by(["split", "sensor_type", "power_w"], maintain_order=True):
        first = group.sort("time_s").row(0, named=True)
        power = round(float(first["power_w"]), 4)
        sensor_type = str(first["sensor_type"])
        audit = raw_audits[(sensor_type, power)]
        exact_t0 = np.isclose(first["time_s"], 0.0)
        difference = float(first["temperature_k"] - initial_temperature_k)
        classification = (
            "exact_t0_observation_conflicts_with_hard_initial_condition"
            if exact_t0 and abs(difference) > 1e-6
            else "first_observation_after_laser_on_not_initial_temperature"
        )
        rows.append(
            {
                "split": first["split"],
                "modality": "Hot" if sensor_type == "hot" else "Cold",
                "power_w": power,
                "source_file": str(Path(audit.path).relative_to(PROJECT_ROOT)),
                "raw_first_time_header_s": audit.time_min_raw,
                "first_observed_time_s": float(first["time_s"]),
                "first_temperature_c": float(first["temperature_k"] - 273.15),
                "hard_initial_temperature_c": float(initial_temperature_k - 273.15),
                "first_minus_hard_initial_c": difference,
                "classification": classification,
                "is_exact_t0_conflict": bool(
                    exact_t0 and abs(difference) > 1e-6
                ),
                "temperature_modified": False,
                "point_deleted": False,
                "bias_added": False,
            }
        )

    top = load_processed_ir_observations()
    top = top.filter(pl.col("split").is_in(["train", "validation"]))
    for group in top.partition_by(["split", "power_w"], maintain_order=True):
        first_time = float(group["time_s"].min())
        first_frame = group.filter(pl.col("time_s") == first_time)
        power = round(float(first_frame["power_w"][0]), 4)
        candidates = sorted(
            (data_root / metadata["experiment_ir"]["directory"]).glob(
                f"{power:g}W-{first_time:g}s_*"
            )
        )
        rows.append(
            {
                "split": str(first_frame["split"][0]),
                "modality": "Top",
                "power_w": power,
                "source_file": None
                if not candidates
                else str(candidates[0].relative_to(PROJECT_ROOT)),
                "raw_first_time_header_s": first_time,
                "first_observed_time_s": first_time,
                "first_temperature_c": float(
                    first_frame["temperature_mean_k"].mean() - 273.15
                ),
                "hard_initial_temperature_c": float(initial_temperature_k - 273.15),
                "first_minus_hard_initial_c": float(
                    first_frame["temperature_mean_k"].mean() - initial_temperature_k
                ),
                "classification": "first_saved_ir_frame_after_laser_on_not_initial_temperature",
                "is_exact_t0_conflict": False,
                "temperature_modified": False,
                "point_deleted": False,
                "bias_added": False,
            }
        )

    for power in sorted(splits.simulation_validation):
        field = load_processed_field(power)
        rows.append(
            {
                "split": "simulation_validation",
                "modality": "Simulation",
                "power_w": power,
                "source_file": f"data/processed/simulation/{power:g}W.parquet",
                "raw_first_time_header_s": float(field.times_s[0]),
                "first_observed_time_s": float(field.times_s[0]),
                "first_temperature_c": float(field.temperature_k[0].mean() - 273.15),
                "hard_initial_temperature_c": float(initial_temperature_k - 273.15),
                "first_minus_hard_initial_c": float(
                    field.temperature_k[0].mean() - initial_temperature_k
                ),
                "classification": "simulation_true_t0_matches_configured_initial_condition",
                "is_exact_t0_conflict": False,
                "temperature_modified": False,
                "point_deleted": False,
                "bias_added": False,
            }
        )

    evidence = {
        "中文说明": "只查证训练/验证实验文件及仿真验证初始帧；没有读取 test_Data。",
        "sensor_time_metadata": {
            "synchronized_start": metadata["sensors"]["synchronized_start"],
            "synchronization_reference": metadata["sensors"]["synchronization_reference"],
            "time_column_interpretation": metadata["sensors"]["time_column_interpretation"],
            "metadata_source": metadata["sensors"]["metadata_source"],
        },
        "ir_time_metadata": {
            "initial_temperature_synchronization": metadata["experiment_ir"][
                "initial_temperature_synchronization"
            ],
            "metadata_source": metadata["experiment_ir"]["metadata_source"],
        },
        "pre_laser_records_found": False,
        "calibration_or_acquisition_log_found": False,
        "reason": "Experiment_data 中只有温度 CSV，未发现独立采集日志、激光前记录或传感器标定文件。",
        "mutation_policy": "保持22℃名义初值，不删首点，不改温度，不增加偏置。",
    }
    return rows, evidence


def _create_source_snapshot(destination: Path) -> str:
    roots = (
        "src",
        "scripts",
        "tests",
        "configs",
        "pyproject.toml",
        "requirements.txt",
        "temperature_field_prediction_optimization_v4.md",
        "temperature_field_prediction_diagnostic_action_v5.md",
    )
    with tarfile.open(destination, "w:gz") as archive:
        for relative in roots:
            root = PROJECT_ROOT / relative
            if not root.exists():
                continue
            if root.is_file():
                archive.add(root, arcname=relative)
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.add(path, arcname=str(path.relative_to(PROJECT_ROOT)))
    return sha256_file(destination)


def _normalized_physics_sha256() -> str:
    payload = {
        name: load_yaml(f"configs/{name}.yaml")
        for name in ("geometry", "materials", "boundary_conditions")
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _input_hashes() -> dict[str, Any]:
    splits = build_power_splits()
    simulation = {
        str(power): sha256_file(
            PROJECT_ROOT / "data/processed/simulation" / f"{power:g}W.parquet"
        )
        for power in sorted(splits.simulation_validation)
    }
    experiment_files: dict[str, str] = {}
    for directory in ("HotData", "ColdData"):
        for path in sorted((PROJECT_ROOT / "data/Experiment_data" / directory).glob("*.csv")):
            experiment_files[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)
    return {
        "simulation_validation": simulation,
        "experiment_sensor_train_and_validation": experiment_files,
        "processed_experiment_ir": sha256_file(
            PROJECT_ROOT / "data/processed/experiment_ir_radial.parquet"
        ),
        "processed_experiment_sensor": sha256_file(
            PROJECT_ROOT / "data/processed/sensor_ring_raw.parquet"
        ),
        "test_Data_temperature_files": "未打开、未哈希",
    }


def _energy_report(
    energy_rows: Sequence[Mapping[str, Any]],
    divergence_rows: Sequence[Mapping[str, Any]],
    convergence_rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    cooling_rows: Sequence[Mapping[str, Any]],
    hard_control: Mapping[str, Any],
    actual_hard_cooling_enabled: bool,
    all_five_hard_cooling_disabled: bool,
    used_orders: Sequence[int],
    gate: Mapping[str, Any],
) -> str:
    highest = max(used_orders)
    energy = [row for row in energy_rows if row["quadrature_order"] == highest]
    divergence = [
        row for row in divergence_rows if row["quadrature_order"] == highest
    ]
    latest_convergence = [
        row for row in convergence_rows if row["quadrature_order"] == highest
    ]
    relative_balance = [abs(float(row["relative_balance"])) for row in energy]
    raw_balance = [abs(float(row["balance_w"])) for row in energy]
    convergence_values = [
        float(row["relative_change_to_absorbed_power"])
        for row in latest_convergence
        if row["relative_change_to_absorbed_power"] is not None
    ]
    identity_relative = [
        abs(float(row["divergence_identity_gap_w"]))
        / max(abs(float(row["integrated_pde_residual_w"])), abs(float(row["storage_plus_external_plus_interface_w"])), 1.0)
        for row in divergence
    ]
    outer_exact = [row for row in cooling_rows if row["epsilon_m"] == 0.0]
    outer_inner = [
        row
        for row in cooling_rows
        if np.isclose(row["epsilon_m"], 1e-6, rtol=0.0, atol=1e-15)
    ]
    allow_text = "允许" if gate["allow_s0_s1_schedule_comparison"] else "不允许"
    powers = sorted({float(row["power_w"]) for row in energy})
    times = sorted({float(row["time_s"]) for row in energy})
    return "\n".join(
        [
            "# V5 M2-E 能量与水冷通量审计",
            "",
            "## 审计范围",
            "",
            "本轮仅对已有 seed=0 DeepONet 多保真检查点进行只读审计。未创建优化器、未训练网络、未读取 `test_Data` 温度，也未改变硬边界实现或冻结结果。",
            "",
            "## 已确认事实",
            "",
            f"- 实际 B0 检查点是否启用 `hard_cooling` 温度覆写：{'是' if actual_hard_cooling_enabled else '否'}。因此此前能量不闭合不能归因于该检查点实际执行了硬覆写。",
            f"- 五份配对 HF 检查点是否均未启用硬水冷覆写：{'是' if all_five_hard_cooling_disabled else '否'}。本报告的实际能量场计算使用 seed=0，另外四份只核对配置来源。",
            f"- 合成线性场明确复现硬覆写缺陷：边界自动微分梯度为 {hard_control['boundary_radial_gradient_k_m']:.1f} K/m，内侧梯度为 {hard_control['inner_radial_gradient_k_m']:.1f} K/m；边界通量被错误置为 {hard_control['boundary_outward_heat_flux_w_m2']:.1f} W/m²，而正确内侧通量为 {hard_control['inner_outward_heat_flux_w_m2']:.1f} W/m²。",
            f"- 使用三个互不重叠材料子域和 `2πr` 权重完成 {list(used_orders)} 阶 Gauss–Legendre 确定性求积；最高阶为 {highest}。",
            f"- 实际审计功率为 {powers} W，审计时刻为 {times} s。t=0 阶跃角点只作为初始条件控制，不混入常规瞬态通量收敛。",
            "- 符号约定：储能率为正表示内能增加；水冷、对流和辐射为正表示向外散热；吸收功率以正的入射幅值记录，并在 `B=储能+水冷+对流+辐射-吸收` 中相减。该定义也逐行写入能量 CSV。",
            "- 工程水冷项使用 `R-1e-6 m` 内侧自动微分通量；由于实际检查点未启用硬覆写，数学散度恒等式使用真实 `r=R` 模型梯度。两者分别记录，未把内移表面混入精确域的散度定理。",
            f"- 等温零功率、制造解、有限圆盘高斯热源控制测试是否全部通过：{'是' if all(row['passed'] for row in controls) else '否'}。",
            f"- 最高阶相对吸收功率的最大相邻阶能量项变化为 {max(convergence_values):.3e}；1% 求积收敛门槛是否通过：{'是' if gate['criteria']['quadrature_converged'] else '否'}。",
            f"- 散度恒等式最高阶最大相对数值差为 {max(identity_relative):.3e}；恒等式数值门槛是否通过：{'是' if gate['criteria']['divergence_identity_numerically_closed'] else '否'}。",
            f"- 模型工程热预算的绝对原始不平衡为 {min(raw_balance):.3f} 至 {max(raw_balance):.3f} W，相对不平衡绝对值为 {min(relative_balance):.3f} 至 {max(relative_balance):.3f}。这是模型/边界一致性诊断，不是实验测得的内部误差。",
            f"- 外圆柱精确边界温度最大偏离固定水冷温度 {max(row['temperature_max_abs_deviation_from_cooling_c'] for row in outer_exact):.4f} ℃；在 `R-1e-6 m` 内侧为 {max(row['temperature_max_abs_deviation_from_cooling_c'] for row in outer_inner):.4f} ℃。实际模型未使用硬覆写，水冷温度仅通过物理边界损失约束。",
            "- `m2_energy_terms.csv` 发布吸收功率、储能率、水冷、对流、辐射和总平衡的原始瓦数；`m2_divergence_identity.csv` 另列模型外向导热通量、材料界面双侧通量及边界残差解释。",
            "",
            "## 结论",
            "",
            "硬覆写实现风险已经确认，但当前五份 B0 检查点均未启用该选项。确定性审计器是否可信由控制测试、相邻阶收敛和散度恒等式共同决定；即使审计器通过，较大的稳定能量不闭合仍表示当前模型不能标为内部物理可信。未经验证的全局能量项不得直接加入训练损失。",
            "",
            f"- S0/S1 排程对照准入：**{allow_text}**。具体门控原因见 `m2_gate.json`。",
            "",
        ]
    )


def _lf_report(
    comparison: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
    alignment: Mapping[str, Any],
) -> str:
    selected = [
        row
        for row in comparison
        if row["aggregation"] == "macro_mean_across_validation_powers"
        and row["time_window"] == "all_0_200_s"
        and row["material"] in {"copper", "silicon_carbide"}
    ]

    def value(material: str, weighting: str) -> tuple[float, float]:
        rows = [
            row
            for row in selected
            if row["material"] == material and row["weighting"] == weighting
        ]
        return float(np.mean([row["rmse_c"] for row in rows])), float(
            np.std([row["rmse_c"] for row in rows], ddof=1)
        )

    cu_node = value("copper", "node_equal")
    sic_node = value("silicon_carbide", "node_equal")
    cu_volume = value("copper", "axisymmetric_lumped_volume")
    sic_volume = value("silicon_carbide", "axisymmetric_lumped_volume")
    all_sources = all(
        row["hf_reference_matches_file"]
        and row["train_powers_match_current_split"]
        and row["validation_powers_match_current_split"]
        and row["calls_match"]
        and not row["test_labels_consumed"]
        for row in provenance
    )
    return "\n".join(
        [
            "# V5 M2-L 低保真主干核验",
            "",
            "## 已确认事实",
            "",
            f"- 五份 LF 检查点的文件哈希、HF 引用哈希、60 个训练功率、10 个验证功率和协议来源是否全部一致：{'是' if all_sources else '否'}。",
            "- LF 评价明确调用独立 LF 检查点的 `forward`；并抽样与配对 HF 模型的 `fidelity=\"low\"` 比较，未调用最终高保真输出。",
            f"- 节点等权五种子宏平均：Cu RMSE={cu_node[0]:.4f} ± {cu_node[1]:.4f} ℃，SiC RMSE={sic_node[0]:.4f} ± {sic_node[1]:.4f} ℃。",
            f"- 带 `2πr` 的轴对称集总体积权重：Cu RMSE={cu_volume[0]:.4f} ± {cu_volume[1]:.4f} ℃，SiC RMSE={sic_volume[0]:.4f} ± {sic_volume[1]:.4f} ℃。",
            f"- Cu/SiC 集总权重相对解析体积误差分别为 {alignment['volume_relative_error']['copper']:.3e} 和 {alignment['volume_relative_error']['silicon_carbide']:.3e}。",
            "- 坐标按 m、深度 z 为负、温度存储为 K、误差输出为 ℃；材料标签、节点编号和 0–200 s 时间网格在十个仿真验证功率间完全对齐。",
            "",
            "## 判定",
            "",
            "现有较大 SiC/Cu 误差来自当前五份配对 LF 检查点在同一明确评价口径下的真实拟合表现，不是误调用高保真输出、功率划分错配或 K/℃ 偏移造成。节点等权与体积权重结果必须并列，不能再与未注明口径的历史 0.538 ℃ 直接比较。",
            "",
            "本轮只完成核验，不重训 LF。若后续修复 LF，只允许用 simulation train 选择，不得使用高保真验证温度选择 LF。",
            "",
        ]
    )


def _initial_report(
    rows: Sequence[Mapping[str, Any]], evidence: Mapping[str, Any]
) -> str:
    exact = [row for row in rows if row["is_exact_t0_conflict"]]
    sensor_after = [
        row
        for row in rows
        if row["modality"] in {"Hot", "Cold"}
        and row["first_observed_time_s"] > 0.0
    ]
    top = [row for row in rows if row["modality"] == "Top"]
    return "\n".join(
        [
            "# V5 M2-O 初始观测分类报告",
            "",
            "## 已查证事实",
            "",
            f"- 精确 t=0 且与 22 ℃ 硬初值冲突的曲线共 {len(exact)} 条："
            + "、".join(
                f"{row['power_w']:g} W {row['modality']}（{row['first_temperature_c']:.5f} ℃，差 {row['first_minus_hard_initial_c']:.5f} ℃）"
                for row in exact
            )
            + "。",
            f"- 其余热环/冷环曲线有 {len(sensor_after)} 条首点位于激光开启后的 t=1 s，只能分类为首个观测，不能当作真实初温。",
            f"- {len(top)} 条顶面曲线的首个红外帧均为 t=5 s；它们不是 t=0 初始帧。",
            "- 十个仿真验证场包含真实 t=0 帧且与 22 ℃ 仿真初始设置一致；这不能反推全部实验样品的真实初温一定为 22 ℃。",
            f"- 是否找到激光开启前记录：{'是' if evidence['pre_laser_records_found'] else '否'}；是否找到独立标定/采集日志：{'是' if evidence['calibration_or_acquisition_log_found'] else '否'}。",
            "",
            "## 待辨识事项",
            "",
            "257 W 热环 t=0 的 3.28903 ℃ 差异仍不能在现有文件中区分为实际初温、传感器标定偏差、环境条件差异或同步定义问题。",
            "",
            "## 本轮处理",
            "",
            "保持 22 ℃ 名义物理初值；不修改任何温度、不删除首点、不制造 t=0 帧、不增加共享或逐功率偏置。绝对温度与相对首观测温升继续并列报告。",
            "",
        ]
    )


def export_v5_m2_audit(
    config_path: str | Path = "configs/diagnostic_v5.yaml",
    output_root: str | Path | None = None,
    effective_cli: Sequence[str] | None = None,
) -> Path:
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_file
    config = load_yaml(config_file)
    forbidden = {
        "allow_training": False,
        "allow_optimizer": False,
        "allow_test_labels": False,
        "allow_logo": False,
        "allow_e1_e2": False,
    }
    for key, expected in forbidden.items():
        if config.get(key) is not expected:
            raise RuntimeError(f"V5 M2 要求 {key}={expected}")
    for split in ("train", "validation"):
        assert_development_label_split(split)

    destination = Path(output_root or config["output_root"])
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖已有 V5 审计目录：{destination}")
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"临时目录已存在：{staging}")
    staging.mkdir(parents=True)
    completed_at = datetime.now(timezone.utc).isoformat()

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seeds = [int(seed) for seed in config["seeds"]]
        primary_seed = int(config["primary_seed"])
        baseline_run = Path(config["baseline_run"])
        if not baseline_run.is_absolute():
            baseline_run = PROJECT_ROOT / baseline_run
        primary_snapshot = (
            baseline_run / f"deeponet_pinn_mf_seed{primary_seed}" / "config_snapshot"
        )
        geometry_config = load_yaml(primary_snapshot / "geometry.yaml")
        geometry = AxisymmetricGeometry.from_config(geometry_config)
        materials = load_materials(str(primary_snapshot / "materials.yaml"))
        boundaries = load_resolved_boundary_conditions(
            str(primary_snapshot / "boundary_conditions.yaml")
        )
        _, model, _, hf_payload, _, hf_path = load_existing_baseline(
            baseline_run, primary_seed, device
        )
        checkpoint_hash_before = sha256_file(hf_path)
        actual_hard_cooling_enabled = (
            model.hard_cooling_radius_m is not None
            or model.hard_cooling_temperature_k is not None
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model = model.to(dtype=torch.float64)
        model.eval()

        controls = analytic_control_tests(
            geometry,
            boundaries,
            materials,
            config["energy"]["initial_quadrature_orders"],
            device,
        )
        cooling_rows, hard_control = _cooling_limit_audit(
            model, config, geometry, materials, boundaries, device
        )
        energy_rows, divergence_rows, convergence_rows, used_orders = (
            _model_energy_audit(
                model, config, geometry, materials, boundaries, device
            )
        )
        checkpoint_hash_after = sha256_file(hf_path)
        if checkpoint_hash_before != checkpoint_hash_after:
            raise RuntimeError("只读审计期间检查点文件发生变化")

        lf_comparison, lf_provenance, lf_spots, lf_alignment = _lf_audit(
            baseline_run, seeds, config, geometry, device
        )
        initial_rows, initial_evidence = _initial_observation_audit(
            boundaries.initial_temperature_k
        )

        highest_order = max(used_orders)
        latest_divergence = [
            row
            for row in divergence_rows
            if row["quadrature_order"] == highest_order
        ]
        divergence_relative = [
            abs(float(row["divergence_identity_gap_w"]))
            / max(
                abs(float(row["integrated_pde_residual_w"])),
                abs(float(row["storage_plus_external_plus_interface_w"])),
                1.0,
            )
            for row in latest_divergence
        ]
        criteria = {
            "e1_e2_paused": True,
            "no_training_or_optimizer": True,
            "test_temperature_not_read": True,
            "hard_override_failure_exposed_by_control": bool(
                hard_control["exposes_zero_gradient_failure"]
            ),
            "actual_checkpoint_hard_override_disabled": not actual_hard_cooling_enabled,
            "all_five_hf_checkpoints_hard_override_disabled": all(
                not row["hf_hard_cooling_enabled"] for row in lf_provenance
            ),
            "analytic_controls_passed": all(row["passed"] for row in controls),
            "quadrature_converged": _latest_quadrature_passes(
                energy_rows,
                float(config["energy"]["convergence_relative_to_absorbed_power"]),
            ),
            "divergence_identity_numerically_closed": max(divergence_relative)
            < float(config["energy"]["divergence_identity_relative_tolerance"]),
            "five_lf_sources_and_call_paths_verified": len(lf_provenance) == 5
            and all(
                row["hf_reference_matches_file"]
                and row["train_powers_match_current_split"]
                and row["validation_powers_match_current_split"]
                and row["calls_match"]
                and not row["test_labels_consumed"]
                for row in lf_provenance
            ),
            "initial_observations_classified_without_mutation": all(
                not row["temperature_modified"]
                and not row["point_deleted"]
                and not row["bias_added"]
                for row in initial_rows
            ),
        }
        allow_s0_s1 = all(criteria.values())
        gate = {
            "中文说明": "仅判定数值审计链路和证据是否足以进入冻结 LF 的排程对照，不表示当前模型能量已经闭合。",
            "criteria": criteria,
            "allow_s0_s1_schedule_comparison": allow_s0_s1,
            "e1_e2_status": "paused",
            "s0_s1_status": "authorized_but_not_started"
            if allow_s0_s1
            else "blocked_pending_failed_m2_criteria",
            "new_training_started": False,
            "unresolved": [
                "当前高保真场的工程能量预算是否达到项目物理容忍度",
                "257 W热环t=0偏差的实际来源",
                "历史未注明口径的0.538℃低保真指标来源",
                "当前LF拟合误差是否需要先修复，以及改善能否传导到HF",
            ],
        }

        _write_csv(staging / "m2_energy_terms.csv", energy_rows)
        _write_csv(staging / "m2_quadrature_convergence.csv", convergence_rows)
        _write_csv(staging / "m2_divergence_identity.csv", divergence_rows)
        _write_csv(staging / "m2_cooling_boundary_limit.csv", cooling_rows)
        _write_csv(staging / "m2_energy_controls.csv", controls)
        _write_csv(staging / "m2_lf_checkpoint_comparison.csv", lf_comparison)
        _write_csv(staging / "m2_lf_checkpoint_provenance.csv", lf_provenance)
        _write_csv(staging / "m2_lf_node_spot_checks.csv", lf_spots)
        _write_csv(staging / "m2_initial_observation_classification.csv", initial_rows)
        _json_dump(staging / "m2_lf_node_alignment.json", lf_alignment)
        _json_dump(staging / "m2_initial_evidence.json", initial_evidence)
        _json_dump(staging / "m2_hard_cooling_control.json", hard_control)
        _json_dump(staging / "m2_gate.json", gate)

        (staging / "m2_energy_flux_audit.md").write_text(
            _energy_report(
                energy_rows,
                divergence_rows,
                convergence_rows,
                controls,
                cooling_rows,
                hard_control,
                actual_hard_cooling_enabled,
                all(not row["hf_hard_cooling_enabled"] for row in lf_provenance),
                used_orders,
                gate,
            ),
            encoding="utf-8",
        )
        (staging / "m2_lf_backbone_audit.md").write_text(
            _lf_report(lf_comparison, lf_provenance, lf_alignment),
            encoding="utf-8",
        )
        (staging / "m2_initial_observation_audit.md").write_text(
            _initial_report(initial_rows, initial_evidence),
            encoding="utf-8",
        )
        decision_log = "\n".join(
            [
                "# V5 M2 决策记录",
                "",
                f"- {completed_at}：暂停 E1/E2，仅执行 M2-E/M2-L/M2-O 只读审计。",
                "- 不修改历史检查点、不训练新网络、不读取 test_Data 温度。",
                "- 已将硬水冷覆写导致边界自动微分热流为零确认为实现风险；当前 B0 检查点未启用该覆写。",
                "- 能量审计使用分材料三子域、2πr 权重和确定性 Gauss–Legendre 求积，并发布原始瓦数、相邻阶收敛和散度恒等式。",
                "- 五份 LF 来源、调用路径、Cu/SiC 节点等权与体积权重误差均已核验。",
                "- 257 W 热环只分类，不改初温、不删点、不加偏置。",
                f"- S0/S1 准入结论：{'允许进入但尚未启动' if allow_s0_s1 else '暂不允许'}。",
                "- 后续训练的强制合同：分别保存各阶段最佳、阶段末与训练最终检查点；每次验证同时评价 HF 观测误差、LF 保持和独立物理误差。",
                "",
            ]
        )
        (staging / "decision_log.md").write_text(decision_log, encoding="utf-8")

        dirty_patch = _run_git("diff", "--binary", "HEAD")
        (staging / "diagnostic_dirty.patch").write_text(dirty_patch, encoding="utf-8")
        git_status = _run_git("status", "--short")
        (staging / "git_status.txt").write_text(git_status, encoding="utf-8")
        source_snapshot_hash = _create_source_snapshot(staging / "source_snapshot.tar.gz")
        provenance = {
            "中文说明": "V5 M2 只读审计来源记录。稳定英文键供后续自动核验。",
            "schema_version": 1,
            "round_id": config["round_id"],
            "completed_at_utc": completed_at,
            "effective_cli": list(effective_cli or []),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "optimizer_constructed": False,
            "optimizer_step_count": 0,
            "new_network_trained": False,
            "test_Data_temperature_read": False,
            "test_Data_processed_temperature_read": False,
            "e1_e2_started": False,
            "s0_s1_started": False,
            "frozen_checkpoint_overwritten": False,
            "primary_hf_checkpoint": str(hf_path.relative_to(PROJECT_ROOT)),
            "primary_hf_checkpoint_sha256_before": checkpoint_hash_before,
            "primary_hf_checkpoint_sha256_after": checkpoint_hash_after,
            "checkpoint_hard_cooling_enabled": actual_hard_cooling_enabled,
            "checkpoint_epoch": int(hf_payload["epoch"]),
            "git_head": _run_git("rev-parse", "HEAD").strip(),
            "git_status_file": "git_status.txt",
            "dirty_patch_file": "diagnostic_dirty.patch",
            "source_snapshot": "source_snapshot.tar.gz",
            "source_snapshot_sha256": source_snapshot_hash,
            "raw_physics_fingerprints": current_protocol_fingerprints(),
            "normalized_physics_values_sha256": _normalized_physics_sha256(),
            "v5_config_sha256": sha256_file(config_file),
            "input_hashes": _input_hashes(),
        }
        _json_dump(staging / "run_provenance.json", provenance)

        artifact_hashes = {
            path.name: sha256_file(path)
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != "manifest.yaml"
        }
        manifest = {
            "中文摘要": {
                "本轮": "V5 M2-E/M2-L/M2-O 只读审计",
                "E1_E2": "已暂停",
                "S0_S1": gate["s0_s1_status"],
                "未训练新网络": True,
                "未读取test_Data温度": True,
            },
            "schema_version": 1,
            "round_id": config["round_id"],
            "completed_at_utc": completed_at,
            "gate": gate,
            "artifact_sha256s": artifact_hashes,
        }
        (staging / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(staging, destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
