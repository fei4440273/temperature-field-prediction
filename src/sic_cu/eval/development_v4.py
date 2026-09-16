from __future__ import annotations

import hashlib
import json
import math
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
from torch import Tensor, nn

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.processed import load_processed_ir_observations
from sic_cu.data.sensors import (
    audit_sensor_file,
    load_canonical_sensor_observations,
    sensor_files,
)
from sic_cu.data.splits import (
    assert_development_label_split,
    build_power_splits,
    resolve_hf_training_subset,
)
from sic_cu.eval.protocol_checks import (
    current_protocol_fingerprints,
)
from sic_cu.losses import PhysicsLossComputer, PhysicsLossWeights
from sic_cu.models import AdditiveCorrectionModel, ModelScales
from sic_cu.physics import load_resolved_boundary_conditions, sample_collocation
from sic_cu.physics.boundary import convection_flux, laser_flux, radiation_flux
from sic_cu.physics.heat_equation import axisymmetric_heat_residual
from sic_cu.physics.interface import normal_heat_flux
from sic_cu.physics.materials import load_materials
from sic_cu.train.multifidelity import _macro_sensor_training_losses
from sic_cu.train.simulation import build_model, load_sampled_points


CELSIUS_OFFSET = 273.15
ALLOWED_DIAGNOSTIC_SPLITS = ("train", "validation")

CSV_VALUE_ZH = {
    "train": "训练集（train）",
    "validation": "验证集（validation）",
    "simulation_validation": "仿真验证集（simulation_validation）",
    "Top": "顶面（Top）",
    "Hot": "热环（Hot）",
    "Cold": "冷环（Cold）",
    "LF_full_field": "低保真全场（LF_full_field）",
    "copper": "铜（Cu）",
    "silicon_carbide": "碳化硅（SiC）",
    "time_0_30_s": "[0,30]秒",
    "time_30_100_s": "(30,100]秒",
    "time_100_200_s": "(100,200]秒",
    "center_0_8_mm": "中心区[0,8]毫米",
    "middle_8_17_mm": "中部区(8,17]毫米",
    "outer_17_25_mm": "外圈区(17,25]毫米",
    "lf_pretrained_checkpoint": "低保真预训练检查点",
    "selected_hf_checkpoint_embedded_lf": "高保真最佳检查点内嵌低保真分支",
    "embedded_minus_pretrained_prediction": "内嵌低保真减预训练低保真预测",
    "full_field_points": "完整场逐点统计",
    "macro_mean_across_validation_powers": "跨验证功率宏平均",
    "experiment_ir_radial": "实验顶面红外径向数据",
    "experiment_sensor_ring": "实验环形传感器数据",
    "simulation_validation_full_field": "仿真验证集完整场",
    "first_observed_frame_per_radius": "各半径首个实测帧",
    "first_observed_ring_sample": "环形传感器首个实测点",
    "simulation_true_t0": "仿真真实初始时刻t=0",
    "laser_on_t0_first_saved_ir_frame_is_5s": "激光开启为t=0，红外首个保存帧为5秒",
    "elapsed_time_t_equals_index_synchronized_to_laser_on": "时间为与激光开启同步的已逝秒数",
    "simulation_time_s": "仿真时间（秒）",
    "no_observations_in_preregistered_window": "预登记时间窗内无观测，不外推真值",
    "no_top_observations_in_preregistered_radial_window": "预登记径向窗内无顶面观测",
}


def _canonical_power(value: float) -> float:
    return round(float(value), 4)


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise RuntimeError(f"Refusing to write empty diagnostic table: {path.name}")
    localized_keys = {
        key
        for record in records
        for key, value in record.items()
        if isinstance(value, str) and value in CSV_VALUE_ZH
    }
    localized = []
    for record in records:
        row = dict(record)
        for key in sorted(localized_keys):
            value = record.get(key)
            row[f"{key}_zh"] = CSV_VALUE_ZH.get(value) if isinstance(value, str) else None
        localized.append(row)
    pl.DataFrame(localized).write_csv(path, null_value="")


def _run_git(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else "UNAVAILABLE"


def _diagnostic_source_status() -> list[str]:
    return [
        line
        for line in _run_git("status", "--short").splitlines()
        if "reports/.development_v4.tmp-" not in line
        and "reports/development_v4/" not in line
    ]


def _state_dict_sha256(state: Mapping[str, Tensor], prefix: str = "") -> str:
    digest = hashlib.sha256()
    for name in sorted(key for key in state if key.startswith(prefix)):
        tensor = state[name].detach().cpu().contiguous()
        canonical_name = name[len(prefix) :] if prefix else name
        name_bytes = canonical_name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_lf_model(payload: Mapping[str, Any], device: torch.device) -> nn.Module:
    scales = ModelScales(**payload["scales"])
    model = build_model(
        str(payload["method"]), scales, **payload.get("model_kwargs", {})
    ).to(device)
    model.load_state_dict(payload["model_state"])
    return model


def load_existing_baseline(
    baseline_run: Path,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, dict[str, Any], dict[str, Any], Path, Path]:
    """Load a historical LF/HF pair without consulting any test observations."""
    lf_path = baseline_run / f"deeponet_pinn_lf_seed{seed}" / "best.pt"
    hf_path = baseline_run / f"deeponet_pinn_mf_seed{seed}" / "best.pt"
    lf_payload = torch.load(lf_path, map_location=device, weights_only=False)
    hf_payload = torch.load(hf_path, map_location=device, weights_only=False)
    if hf_payload.get("surface_residual_guide_spec") is not None:
        raise RuntimeError("B0 must be the DeepONet baseline without a hard surface guide")
    if hf_payload.get("method") != "multifidelity_correction":
        raise RuntimeError(f"Unexpected HF checkpoint method: {hf_payload.get('method')}")
    expected_lf_hash = hf_payload.get("provenance", {}).get("lf_checkpoint_sha256")
    if expected_lf_hash != sha256_file(lf_path):
        raise RuntimeError(f"Seed {seed} HF checkpoint does not reference its paired LF checkpoint")

    pretrained_lf = _load_lf_model(lf_payload, device)
    embedded_lf = build_model(
        str(hf_payload["low_fidelity_method"]),
        ModelScales(**hf_payload["scales"]),
        **hf_payload.get("low_fidelity_model_kwargs", {}),
    ).to(device)
    model = AdditiveCorrectionModel(
        embedded_lf,
        ModelScales(**hf_payload["scales"]),
        freeze_low_fidelity=True,
        **hf_payload["correction_model_kwargs"],
    ).to(device)
    model.load_state_dict(hf_payload["model_state"])
    pretrained_lf.eval()
    model.eval()
    return pretrained_lf, model, dict(lf_payload), dict(hf_payload), lf_path, hf_path


@torch.no_grad()
def _predict(model: nn.Module, coordinates: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    values = []
    tensor = torch.from_numpy(np.asarray(coordinates, dtype=np.float32))
    for offset in range(0, len(tensor), batch_size):
        values.append(model(tensor[offset : offset + batch_size].to(device)).cpu().numpy())
    return np.concatenate(values).reshape(-1).astype(np.float64)


def error_statistics(
    target: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target_values.shape != predicted_values.shape or len(target_values) == 0:
        raise ValueError("Target and prediction must be equal, non-empty vectors")
    if weights is None:
        normalized = np.full(len(target_values), 1.0 / len(target_values))
    else:
        normalized = np.asarray(weights, dtype=np.float64).reshape(-1)
        if normalized.shape != target_values.shape or np.any(normalized < 0.0):
            raise ValueError("Weights must be nonnegative and aligned")
        total = float(normalized.sum())
        if total <= 0.0:
            raise ValueError("Weights must have positive sum")
        normalized = normalized / total
    error = predicted_values - target_values
    return {
        "rmse_c": float(np.sqrt(np.sum(normalized * error**2))),
        "mae_c": float(np.sum(normalized * np.abs(error))),
        "signed_bias_c": float(np.sum(normalized * error)),
        "p95_abs_error_c": float(np.quantile(np.abs(error), 0.95)),
        "max_abs_error_c": float(np.max(np.abs(error))),
    }


def _delta_from_first(
    target: np.ndarray,
    prediction: np.ndarray,
    times: np.ndarray,
    curve_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_delta = np.empty_like(target, dtype=np.float64)
    prediction_delta = np.empty_like(prediction, dtype=np.float64)
    for key in np.unique(curve_keys):
        indices = np.flatnonzero(curve_keys == key)
        first = indices[np.argmin(times[indices])]
        target_delta[indices] = target[indices] - target[first]
        prediction_delta[indices] = prediction[indices] - prediction[first]
    return target_delta, prediction_delta


def _empty_metrics(reason: str) -> dict[str, Any]:
    return {
        "rmse_c": None,
        "mae_c": None,
        "signed_bias_c": None,
        "p95_abs_error_c": None,
        "max_abs_error_c": None,
        "empty_reason": reason,
    }


def _window_mask(times: np.ndarray, window: Mapping[str, Any]) -> np.ndarray:
    lower = float(window["lower"])
    upper = float(window["upper"])
    lower_mask = times >= lower if window["lower_closed"] else times > lower
    return lower_mask & (times <= upper)


def _top_coordinates(frame: pl.DataFrame) -> np.ndarray:
    return np.column_stack(
        (
            frame["r_m"].to_numpy(),
            np.zeros(frame.height),
            frame["time_s"].to_numpy(),
            frame["power_w"].to_numpy(),
            np.ones(frame.height),
        )
    ).astype(np.float32)


def _sensor_coordinates(frame: pl.DataFrame, bottom_z_m: float) -> np.ndarray:
    return np.column_stack(
        (
            frame["r_m"].to_numpy(),
            np.full(frame.height, bottom_z_m),
            frame["time_s"].to_numpy(),
            frame["power_w"].to_numpy(),
            np.zeros(frame.height),
        )
    ).astype(np.float32)


def _observation_arrays(
    frame: pl.DataFrame,
    prediction: np.ndarray,
    modality: str,
) -> dict[str, np.ndarray]:
    target_column = "temperature_mean_k" if modality == "Top" else "temperature_k"
    target = frame[target_column].to_numpy().astype(np.float64)
    times = frame["time_s"].to_numpy().astype(np.float64)
    if modality == "Top":
        curve_keys = np.round(frame["r_m"].to_numpy() * 1e9).astype(np.int64)
        weights = frame["frame_weight"].to_numpy().astype(np.float64)
    else:
        curve_keys = np.zeros(frame.height, dtype=np.int8)
        weights = np.ones(frame.height, dtype=np.float64)
    target_delta, prediction_delta = _delta_from_first(
        target, prediction, times, curve_keys
    )
    return {
        "target": target,
        "prediction": prediction,
        "times": times,
        "weights": weights,
        "target_delta": target_delta,
        "prediction_delta": prediction_delta,
    }


def _power_record(
    *,
    seed: int,
    split: str,
    power_w: float,
    modality: str,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    absolute = error_statistics(
        arrays["target"], arrays["prediction"], arrays["weights"]
    )
    delta = error_statistics(
        arrays["target_delta"], arrays["prediction_delta"], arrays["weights"]
    )
    return {
        "seed": seed,
        "split": split,
        "power_w": power_w,
        "modality": modality,
        "sample_count": len(arrays["target"]),
        "time_count": len(np.unique(arrays["times"])),
        **absolute,
        "delta_rmse_c": delta["rmse_c"],
        "delta_mae_c": delta["mae_c"],
        "delta_signed_bias_c": delta["signed_bias_c"],
        "delta_p95_abs_error_c": delta["p95_abs_error_c"],
        "peak_error_c": float(
            np.max(arrays["prediction"]) - np.max(arrays["target"])
        ),
    }


def evaluate_hf_observations(
    model: nn.Module,
    seed: int,
    split: str,
    device: torch.device,
    config: Mapping[str, Any],
    bottom_z_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    assert_development_label_split(split)
    top = load_processed_ir_observations(split).sort("power_w", "time_s", "r_m")
    sensors = load_canonical_sensor_observations(split=split).sort(
        "power_w", "sensor_type", "time_s"
    )
    batch_size = int(config["prediction_batch_sizes"][-1])
    top_prediction = _predict(model, _top_coordinates(top), device, batch_size)
    sensor_prediction = _predict(
        model, _sensor_coordinates(sensors, bottom_z_m), device, batch_size
    )
    power_records: list[dict[str, Any]] = []
    time_records: list[dict[str, Any]] = []
    radius_records: list[dict[str, Any]] = []
    cached_predictions = {"Top": top_prediction, "sensors": sensor_prediction}

    modality_frames: list[tuple[str, pl.DataFrame, np.ndarray]] = [("Top", top, top_prediction)]
    for modality, sensor_type in (("Hot", "hot"), ("Cold", "cold")):
        mask = sensors["sensor_type"].to_numpy() == sensor_type
        modality_frames.append((modality, sensors.filter(pl.col("sensor_type") == sensor_type), sensor_prediction[mask]))

    for modality, frame, predictions in modality_frames:
        for power in sorted(frame["power_w"].unique().to_list()):
            mask = np.isclose(frame["power_w"].to_numpy(), power, atol=1e-4)
            group = frame.filter(pl.col("power_w") == power)
            arrays = _observation_arrays(group, predictions[mask], modality)
            power_records.append(
                _power_record(
                    seed=seed,
                    split=split,
                    power_w=_canonical_power(power),
                    modality=modality,
                    arrays=arrays,
                )
            )
            for window in config["time_windows_s"]:
                selected = _window_mask(arrays["times"], window)
                metrics = (
                    error_statistics(
                        arrays["target"][selected],
                        arrays["prediction"][selected],
                        arrays["weights"][selected],
                    )
                    if bool(selected.any())
                    else _empty_metrics("no_observations_in_preregistered_window")
                )
                delta_metrics = (
                    error_statistics(
                        arrays["target_delta"][selected],
                        arrays["prediction_delta"][selected],
                        arrays["weights"][selected],
                    )
                    if bool(selected.any())
                    else _empty_metrics("no_observations_in_preregistered_window")
                )
                time_records.append(
                    {
                        "seed": seed,
                        "split": split,
                        "power_w": _canonical_power(power),
                        "modality": modality,
                        "window": window["name"],
                        "lower_s": window["lower"],
                        "upper_s": window["upper"],
                        "lower_closed": window["lower_closed"],
                        "upper_closed": True,
                        "sample_count": int(selected.sum()),
                        **metrics,
                        "delta_rmse_c": delta_metrics["rmse_c"],
                        "delta_mae_c": delta_metrics["mae_c"],
                        "delta_signed_bias_c": delta_metrics["signed_bias_c"],
                    }
                )

            if modality != "Top":
                continue
            radii_mm = group["r_m"].to_numpy().astype(np.float64) * 1000.0
            for radial in config["radial_windows_mm"]:
                lower_mask = (
                    radii_mm >= float(radial["lower"])
                    if radial["lower_closed"]
                    else radii_mm > float(radial["lower"])
                )
                selected = lower_mask & (radii_mm <= float(radial["upper"]))
                metrics = (
                    error_statistics(
                        arrays["target"][selected],
                        arrays["prediction"][selected],
                        arrays["weights"][selected],
                    )
                    if bool(selected.any())
                    else _empty_metrics("no_top_observations_in_preregistered_radial_window")
                )
                radius_records.append(
                    {
                        "seed": seed,
                        "split": split,
                        "power_w": _canonical_power(power),
                        "radial_window": radial["name"],
                        "lower_mm": radial["lower"],
                        "upper_mm": radial["upper"],
                        "lower_closed": radial["lower_closed"],
                        "upper_closed": True,
                        "sample_count": int(selected.sum()),
                        **metrics,
                    }
                )
    return power_records, time_records, radius_records, cached_predictions


def build_initial_state_audit(
    model: nn.Module,
    device: torch.device,
    bottom_z_m: float,
    known_initial_k: float,
    simulation_validation_powers: Iterable[float],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in ALLOWED_DIAGNOSTIC_SPLITS:
        assert_development_label_split(split)
        top = load_processed_ir_observations(split).sort("power_w", "time_s", "r_m")
        sensors = load_canonical_sensor_observations(split=split).sort(
            "power_w", "sensor_type", "time_s"
        )
        for power in sorted(top["power_w"].unique().to_list()):
            group = top.filter(pl.col("power_w") == power)
            first_time = float(group["time_s"].min())
            first = group.filter(pl.col("time_s") == first_time)
            prediction = _predict(model, _top_coordinates(first), device, 65536)
            target_c = first["temperature_mean_k"].to_numpy() - CELSIUS_OFFSET
            records.append(
                {
                    "source": "experiment_ir_radial",
                    "split": split,
                    "power_w": _canonical_power(power),
                    "modality": "Top",
                    "material": "silicon_carbide",
                    "first_observed_time_s": first_time,
                    "first_observed_sample_count": first.height,
                    "first_temperature_mean_c": float(np.mean(target_c)),
                    "first_temperature_min_c": float(np.min(target_c)),
                    "first_temperature_max_c": float(np.max(target_c)),
                    "model_first_temperature_mean_c": float(np.mean(prediction - CELSIUS_OFFSET)),
                    "delta_reference_time_s": first_time,
                    "delta_reference_semantics": "first_observed_frame_per_radius",
                    "known_initial_temperature_c": known_initial_k - CELSIUS_OFFSET,
                    "contains_true_t0": bool(np.isclose(first_time, 0.0)),
                    "time_semantics": "laser_on_t0_first_saved_ir_frame_is_5s",
                }
            )
        for power in sorted(sensors["power_w"].unique().to_list()):
            for sensor_type, modality in (("hot", "Hot"), ("cold", "Cold")):
                group = sensors.filter(
                    (pl.col("power_w") == power)
                    & (pl.col("sensor_type") == sensor_type)
                ).sort("time_s")
                first_time = float(group["time_s"][0])
                first = group.head(1)
                prediction = _predict(
                    model, _sensor_coordinates(first, bottom_z_m), device, 65536
                )[0]
                target_c = float(first["temperature_k"][0]) - CELSIUS_OFFSET
                records.append(
                    {
                        "source": "experiment_sensor_ring",
                        "split": split,
                        "power_w": _canonical_power(power),
                        "modality": modality,
                        "material": "copper",
                        "first_observed_time_s": first_time,
                        "first_observed_sample_count": 1,
                        "first_temperature_mean_c": target_c,
                        "first_temperature_min_c": target_c,
                        "first_temperature_max_c": target_c,
                        "model_first_temperature_mean_c": float(prediction - CELSIUS_OFFSET),
                        "delta_reference_time_s": first_time,
                        "delta_reference_semantics": "first_observed_ring_sample",
                        "known_initial_temperature_c": known_initial_k - CELSIUS_OFFSET,
                        "contains_true_t0": bool(np.isclose(first_time, 0.0)),
                        "time_semantics": "elapsed_time_t_equals_index_synchronized_to_laser_on",
                    }
                )

    for power in sorted(simulation_validation_powers):
        path = PROJECT_ROOT / "data/processed/simulation" / f"{power:g}W.parquet"
        frame = pl.read_parquet(
            path,
            columns=["time_s", "material_id", "temperature_k"],
        )
        first_time = float(frame["time_s"].min())
        first = frame.filter(pl.col("time_s") == first_time)
        for material_id, material in ((0, "copper"), (1, "silicon_carbide")):
            group = first.filter(pl.col("material_id") == material_id)
            target_c = group["temperature_k"].to_numpy() - CELSIUS_OFFSET
            records.append(
                {
                    "source": "simulation_validation_full_field",
                    "split": "simulation_validation",
                    "power_w": _canonical_power(power),
                    "modality": "LF_full_field",
                    "material": material,
                    "first_observed_time_s": first_time,
                    "first_observed_sample_count": group.height,
                    "first_temperature_mean_c": float(np.mean(target_c)),
                    "first_temperature_min_c": float(np.min(target_c)),
                    "first_temperature_max_c": float(np.max(target_c)),
                    "model_first_temperature_mean_c": None,
                    "delta_reference_time_s": first_time,
                    "delta_reference_semantics": "simulation_true_t0",
                    "known_initial_temperature_c": known_initial_k - CELSIUS_OFFSET,
                    "contains_true_t0": bool(np.isclose(first_time, 0.0)),
                    "time_semantics": "simulation_time_s",
                }
            )
    return records


def _lf_validation_frames(powers_w: Iterable[float]) -> dict[float, pl.DataFrame]:
    frames: dict[float, pl.DataFrame] = {}
    for power in sorted(powers_w):
        path = PROJECT_ROOT / "data/processed/simulation" / f"{power:g}W.parquet"
        frames[float(power)] = pl.read_parquet(
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
    return frames


def evaluate_lf_retention(
    pretrained_lf: nn.Module,
    multifidelity: AdditiveCorrectionModel,
    seed: int,
    frames: Mapping[float, pl.DataFrame],
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    all_before: dict[int, list[dict[str, float]]] = {0: [], 1: []}
    all_after: dict[int, list[dict[str, float]]] = {0: [], 1: []}
    all_drift: dict[int, list[dict[str, float]]] = {0: [], 1: []}
    for power, frame in frames.items():
        coordinates = frame.select(
            "r_m", "z_m", "time_s", "power_w", "material_id"
        ).to_numpy().astype(np.float32)
        target = frame["temperature_k"].to_numpy().astype(np.float64)
        before = _predict(pretrained_lf, coordinates, device, batch_size)
        after = _predict(multifidelity.low_fidelity_model, coordinates, device, batch_size)
        for material_id, material in ((0, "copper"), (1, "silicon_carbide")):
            mask = frame["material_id"].to_numpy() == material_id
            before_metrics = error_statistics(target[mask], before[mask])
            after_metrics = error_statistics(target[mask], after[mask])
            drift_metrics = error_statistics(before[mask], after[mask])
            all_before[material_id].append(before_metrics)
            all_after[material_id].append(after_metrics)
            all_drift[material_id].append(drift_metrics)
            for state, metrics in (
                ("lf_pretrained_checkpoint", before_metrics),
                ("selected_hf_checkpoint_embedded_lf", after_metrics),
                ("embedded_minus_pretrained_prediction", drift_metrics),
            ):
                records.append(
                    {
                        "seed": seed,
                        "power_w": power,
                        "material_id": material_id,
                        "material": material,
                        "state": state,
                        "sample_count": int(mask.sum()),
                        "aggregation": "full_field_points",
                        **metrics,
                    }
                )
    for material_id, material in ((0, "copper"), (1, "silicon_carbide")):
        for state, values in (
            ("lf_pretrained_checkpoint", all_before[material_id]),
            ("selected_hf_checkpoint_embedded_lf", all_after[material_id]),
            ("embedded_minus_pretrained_prediction", all_drift[material_id]),
        ):
            records.append(
                {
                    "seed": seed,
                    "power_w": None,
                    "material_id": material_id,
                    "material": material,
                    "state": state,
                    "sample_count": int(
                        sum(
                            frame.filter(pl.col("material_id") == material_id).height
                            for frame in frames.values()
                        )
                    ),
                    "aggregation": "macro_mean_across_validation_powers",
                    **{
                        key: float(np.mean([item[key] for item in values]))
                        for key in values[0]
                    },
                }
            )
    pretrained_hash = _state_dict_sha256(pretrained_lf.state_dict())
    embedded_hash = _state_dict_sha256(multifidelity.state_dict(), "low_fidelity_model.")
    return records, {
        "seed": seed,
        "pretrained_lf_state_sha256": pretrained_hash,
        "selected_hf_embedded_lf_state_sha256": embedded_hash,
        "state_tensors_identical": pretrained_hash == embedded_hash,
    }


def historical_optimization_audit(
    training_log: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    records = [json.loads(line) for line in training_log.read_text(encoding="utf-8").splitlines() if line]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = metrics["configuration"]
    batch_size = int(config["batch_size_per_rank"])
    world_size = int(metrics["world_size"])
    batch_counts = [
        math.ceil(int(record["epoch_hf_ir_points"]) / (batch_size * world_size))
        for record in records
    ]
    data_steps = int(sum(batch_counts))
    joint_steps = int(
        sum(count for count, record in zip(batch_counts, records) if record["stage"] == "joint")
    )
    physics_steps = len(records)

    def nonzero_epochs(name: str) -> int:
        return sum(abs(float(record.get(name, 0.0))) > 0.0 for record in records)

    loss_steps = {
        "top_ir": {
            "scheduled_optimizer_steps": data_steps,
            "epochs_with_nonzero_logged_mean": nonzero_epochs("loss_ir"),
            "log_resolution": "epoch_mean",
        },
        "hot_absolute": {"scheduled_optimizer_steps": data_steps, "epochs_with_nonzero_logged_mean": None},
        "hot_delta": {"scheduled_optimizer_steps": data_steps, "epochs_with_nonzero_logged_mean": None},
        "cold_absolute": {"scheduled_optimizer_steps": data_steps, "epochs_with_nonzero_logged_mean": None},
        "cold_delta": {"scheduled_optimizer_steps": data_steps, "epochs_with_nonzero_logged_mean": None},
        "sensor_combined": {
            "scheduled_optimizer_steps": data_steps,
            "epochs_with_nonzero_logged_mean": nonzero_epochs("loss_sensor"),
            "log_resolution": "epoch_mean_hot_and_cold_combined",
        },
        "lf_simulation": {
            "scheduled_optimizer_steps": joint_steps,
            "epochs_with_nonzero_logged_mean": nonzero_epochs("loss_low_fidelity"),
            "log_resolution": "epoch_mean",
        },
    }
    for name in ("pde", "boundary", "initial", "interface"):
        loss_steps[f"physics_{name}"] = {
            "scheduled_optimizer_steps": physics_steps,
            "epochs_with_nonzero_logged_loss": nonzero_epochs(f"loss_{name}"),
            "log_resolution": "one_physics_update_per_epoch",
        }

    gradients: dict[str, Any] = {}
    for name in ("lf_gradient_l2", "hf_gradient_l2"):
        values = np.asarray([float(record[name]) for record in records], dtype=np.float64)
        gradients[name] = {
            "sample_count": len(values),
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "maximum": float(values.max()),
            "nonzero_count": int(np.count_nonzero(values)),
            "meaning": "last_data_batch_before_epoch_end_physics_update",
        }
    return {
        "seed": int(metrics["seed"]),
        "actual_epochs": len(records),
        "actual_correction_epochs": sum(record["stage"] == "correction" for record in records),
        "actual_joint_epochs": sum(record["stage"] == "joint" for record in records),
        "planned_correction_epochs": int(config["correction_epochs"]),
        "planned_joint_epochs": int(config["joint_epochs"]),
        "data_optimizer_steps": data_steps,
        "physics_optimizer_steps": physics_steps,
        "total_optimizer_steps": data_steps + physics_steps,
        "loss_optimization_steps": loss_steps,
        "logged_data_gradient_summary": gradients,
        "step_definition": (
            "scheduled_optimizer_steps counts backward/update calls in which the weighted "
            "loss was included; nonzero per-batch gradients cannot be reconstructed from epoch means"
        ),
    }


def _evenly_spaced_indices(frame: pl.DataFrame, points_per_power: int) -> np.ndarray:
    indices: list[np.ndarray] = []
    powers = frame["power_w"].to_numpy()
    for power in sorted(frame["power_w"].unique().to_list()):
        group = np.flatnonzero(np.isclose(powers, power, atol=1e-4))
        count = min(points_per_power, len(group))
        indices.append(group[np.linspace(0, len(group) - 1, count, dtype=np.int64)])
    return np.concatenate(indices)


def _parameter_groups(model: AdditiveCorrectionModel) -> dict[str, list[nn.Parameter]]:
    return {
        "lf_branch": list(model.low_fidelity_model.parameters()),
        "hf_correction": list(model.correction.parameters()),
    }


def _gradient_for_loss(
    loss: Tensor,
    groups: Mapping[str, Sequence[nn.Parameter]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ordered = [parameter for parameters in groups.values() for parameter in parameters]
    gradients = torch.autograd.grad(loss, ordered, allow_unused=True)
    diagnostics: dict[str, Any] = {}
    vectors: dict[str, np.ndarray] = {}
    offset = 0
    for group_name, parameters in groups.items():
        group_gradients = gradients[offset : offset + len(parameters)]
        offset += len(parameters)
        flat = []
        connected = 0
        for parameter, gradient in zip(parameters, group_gradients):
            if gradient is None:
                flat.append(torch.zeros_like(parameter).reshape(-1))
            else:
                connected += parameter.numel()
                flat.append(gradient.detach().reshape(-1))
        vector = torch.cat(flat).double().cpu().numpy()
        vectors[group_name] = vector
        diagnostics[group_name] = {
            "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
            "connected_parameter_count": int(connected),
            "gradient_l2": float(np.linalg.norm(vector)),
            "gradient_linf": float(np.max(np.abs(vector))) if len(vector) else 0.0,
        }
    return diagnostics, vectors


def _cosine(left: np.ndarray, right: np.ndarray) -> tuple[float | None, str | None]:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None, "no_common_nonzero_gradient"
    value = float(np.dot(left, right) / (left_norm * right_norm))
    return float(np.clip(value, -1.0, 1.0)), None


def fixed_batch_gradient_diagnostics(
    model: AdditiveCorrectionModel,
    hf_payload: Mapping[str, Any],
    device: torch.device,
    config: Mapping[str, Any],
    bottom_z_m: float,
    materials: Mapping[int, Any],
    boundaries: Any,
    geometry_path: str,
) -> dict[str, Any]:
    model.eval()
    model.freeze_low_fidelity(False)
    groups = _parameter_groups(model)
    scale = float(hf_payload["scales"]["temperature_scale_k"])
    training_cfg = load_yaml("configs/training.yaml")
    weights = training_cfg["loss_weights"]

    top_frame = load_processed_ir_observations("train").sort("power_w", "time_s", "r_m")
    top_indices = _evenly_spaced_indices(
        top_frame, int(config["gradient_top_points_per_power"])
    )
    top_selected = top_frame[top_indices]
    top_coordinates = torch.from_numpy(_top_coordinates(top_selected)).to(device)
    top_target = torch.from_numpy(
        top_selected["temperature_mean_k"].to_numpy().astype(np.float32)[:, None]
    ).to(device)
    top_weight = torch.from_numpy(
        top_selected["frame_weight"].to_numpy().astype(np.float32)[:, None]
    ).to(device)

    sensors = load_canonical_sensor_observations(split="train").sort(
        "power_w", "sensor_type", "time_s"
    )
    sensor_batches: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
    for sensor_type in ("hot", "cold"):
        frame = sensors.filter(pl.col("sensor_type") == sensor_type)
        coordinates = torch.from_numpy(_sensor_coordinates(frame, bottom_z_m)).to(device)
        target = torch.from_numpy(
            frame["temperature_k"].to_numpy().astype(np.float32)[:, None]
        ).to(device)
        delta = torch.from_numpy(
            frame["delta_temperature_k"].to_numpy().astype(np.float32)[:, None]
        ).to(device)
        powers = frame["power_w"].to_numpy()
        first_by_power: dict[float, int] = {}
        baseline = []
        for index, power in enumerate(powers):
            key = _canonical_power(power)
            first_by_power.setdefault(key, index)
            baseline.append(first_by_power[key])
        sensor_batches[sensor_type] = (
            coordinates,
            target,
            delta,
            torch.tensor(baseline, dtype=torch.long, device=device),
        )

    splits = build_power_splits()
    lf_dataset = load_sampled_points(
        sorted(splits.simulation_train),
        samples_per_power=int(config["gradient_lf_points_per_power"]),
        seed=int(config["gradient_seed"]),
    )
    lf_coordinates, lf_target = (tensor.to(device) for tensor in lf_dataset.tensors)
    physics = PhysicsLossComputer(
        dict(materials),
        boundaries,
        PhysicsLossWeights(
            pde=float(weights["pde"]),
            boundary=float(weights["boundary"]),
            initial=float(weights["initial"]),
            interface=float(weights["interface"]),
        ),
    )

    def top_loss() -> Tensor:
        prediction = model(top_coordinates)
        per_power = []
        for power in torch.unique(top_coordinates[:, 3]):
            mask = torch.isclose(top_coordinates[:, 3], power, atol=1e-4, rtol=0.0)
            value = ((prediction[mask] - top_target[mask]) / scale).pow(2)
            per_power.append((top_weight[mask] * value).sum() / top_weight[mask].sum())
        return torch.stack(per_power).mean()

    def sensor_loss(sensor_type: str, component: str) -> Tensor:
        coordinates, target, delta, baseline = sensor_batches[sensor_type]
        prediction = model(coordinates)
        absolute, rise = _macro_sensor_training_losses(
            prediction, target, delta, baseline, coordinates, scale
        )
        return absolute if component == "absolute" else rise

    def lf_loss() -> Tensor:
        prediction = model(lf_coordinates, fidelity="low")
        return ((prediction - lf_target) / scale).pow(2).mean()

    data_builders: list[tuple[str, float, Any]] = [
        ("top_ir", float(weights["ir"]), top_loss),
        ("hot_absolute", float(weights["sensor_absolute"]), lambda: sensor_loss("hot", "absolute")),
        ("hot_delta", float(weights["sensor_delta"]), lambda: sensor_loss("hot", "delta")),
        ("cold_absolute", float(weights["sensor_absolute"]), lambda: sensor_loss("cold", "absolute")),
        ("cold_delta", float(weights["sensor_delta"]), lambda: sensor_loss("cold", "delta")),
        ("lf_simulation", float(weights["low_fidelity"]), lf_loss),
    ]
    losses: dict[str, Any] = {}
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for name, multiplier, builder in data_builders:
        raw = builder()
        weighted = multiplier * raw
        group_diagnostics, group_vectors = _gradient_for_loss(weighted, groups)
        losses[name] = {
            "raw_loss": float(raw.detach()),
            "weight": multiplier,
            "weighted_loss": float(weighted.detach()),
            "parameter_groups": group_diagnostics,
        }
        vectors[name] = group_vectors

    physics_batch = sample_collocation(
        int(config["gradient_physics_collocation"]),
        device,
        seed=int(config["gradient_seed"]),
        geometry_path=geometry_path,
        power_range_w=(min(splits.hf_train), max(splits.hf_train)),
    )
    physics_multipliers = {
        "pde": float(weights["pde"]),
        "boundary": float(weights["boundary"]),
        "initial": float(weights["initial"]),
        "interface": float(weights["interface"]),
        "physics_total": 1.0,
    }
    for component, multiplier in physics_multipliers.items():
        values = physics(model, physics_batch)
        raw = values[component]
        weighted = multiplier * raw if component != "physics_total" else raw
        name = f"physics_{component}"
        group_diagnostics, group_vectors = _gradient_for_loss(weighted, groups)
        losses[name] = {
            "raw_loss": float(raw.detach()),
            "weight": multiplier,
            "weighted_loss": float(weighted.detach()),
            "parameter_groups": group_diagnostics,
        }
        vectors[name] = group_vectors

    angles = []
    names = list(vectors)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            for group_name in groups:
                cosine, reason = _cosine(
                    vectors[left_name][group_name], vectors[right_name][group_name]
                )
                angles.append(
                    {
                        "left_loss": left_name,
                        "right_loss": right_name,
                        "parameter_group": group_name,
                        "cosine": cosine,
                        "angle_degrees": (
                            None if cosine is None else float(np.degrees(np.arccos(cosine)))
                        ),
                        "not_applicable_reason": reason,
                    }
                )
    model.freeze_low_fidelity(True)
    return {
        "model_seed": int(hf_payload["seed"]),
        "checkpoint_epoch": int(hf_payload["epoch"]),
        "state_and_batch_policy": "one_checkpoint_state_and_fixed_train_only_batches",
        "lf_gradient_context": (
            "LF temporarily marked trainable only to measure the joint-stage gradient; "
            "no optimizer was constructed or stepped"
        ),
        "fixed_batch": {
            "top_points": len(top_coordinates),
            "hot_points": len(sensor_batches["hot"][0]),
            "cold_points": len(sensor_batches["cold"][0]),
            "lf_points": len(lf_coordinates),
            "lf_powers": len(splits.simulation_train),
            "physics_collocation_per_component": int(config["gradient_physics_collocation"]),
            "seed": int(config["gradient_seed"]),
        },
        "losses": losses,
        "pairwise_gradient_angles": angles,
    }


def _temperature_gradient(model: nn.Module, coordinates: Tensor) -> tuple[Tensor, Tensor]:
    x = coordinates.detach().clone().requires_grad_(True)
    temperature = model(x)
    gradient = torch.autograd.grad(
        temperature,
        x,
        torch.ones_like(temperature),
        create_graph=False,
        retain_graph=False,
    )[0]
    return temperature.detach(), gradient.detach()


def _radial_surface_integral(values: Tensor, radii: Tensor, low: float, high: float) -> float:
    return float((2.0 * math.pi * (high - low) * (radii * values).mean()).detach())


def global_energy_checks(
    model: nn.Module,
    materials: Mapping[int, Any],
    boundaries: Any,
    device: torch.device,
    geometry_path: str,
    powers_w: Iterable[float],
    times_s: Iterable[float],
    samples_per_material: int,
    seed: int,
) -> list[dict[str, float]]:
    geometry = load_yaml(geometry_path)
    copper_radius = float(geometry["copper"]["radius_m"])
    copper_height = float(geometry["copper"]["height_m"])
    sic_radius = float(geometry["silicon_carbide"]["radius_m"])
    sic_height = float(geometry["silicon_carbide"]["height_m"])
    sic_volume = math.pi * sic_radius**2 * sic_height
    copper_volume = math.pi * copper_radius**2 * copper_height - sic_volume
    output: list[dict[str, float]] = []
    for power_index, power in enumerate(sorted(powers_w)):
        for time_index, time_s in enumerate(times_s):
            batch = sample_collocation(
                samples_per_material * 2,
                device,
                seed=seed + power_index * 100 + time_index,
                geometry_path=geometry_path,
                power_range_w=(float(power), float(power)),
            )

            def at_state(coordinates: Tensor) -> Tensor:
                selected = coordinates.detach().clone()
                selected[:, 2] = float(time_s)
                selected[:, 3] = float(power)
                return selected

            interior = at_state(batch.interior)
            temperature, gradient = _temperature_gradient(model, interior)
            storage_w = 0.0
            for material_id, volume in ((0, copper_volume), (1, sic_volume)):
                mask = batch.interior_material_ids == material_id
                heat_capacity = materials[material_id].heat_capacity(temperature[mask])
                storage_density = (
                    materials[material_id].density_kg_m3
                    * heat_capacity
                    * gradient[mask, 2:3]
                )
                storage_w += float(storage_density.mean()) * volume

            sic_top = at_state(batch.sic_top)
            sic_top_temperature = model(sic_top).detach()
            applied = laser_flux(
                sic_top[:, 0:1],
                sic_top[:, 3:4],
                boundaries.laser.profile,
                boundaries.laser.absorption_fraction,
                boundaries.laser.beam_radius_m,
            )
            laser_w = _radial_surface_integral(
                applied, sic_top[:, 0:1], 0.0, sic_radius
            )
            sic_loss = convection_flux(
                sic_top_temperature,
                boundaries.ambient_temperature_k,
                boundaries.top_convection_coefficient_w_m2_k,
            )
            if boundaries.radiation_enabled:
                sic_loss = sic_loss + radiation_flux(
                    sic_top_temperature,
                    boundaries.ambient_temperature_k,
                    boundaries.silicon_carbide_emissivity,
                )
            sic_loss_w = _radial_surface_integral(
                sic_loss, sic_top[:, 0:1], 0.0, sic_radius
            )

            copper_top = at_state(batch.copper_top)
            copper_top_temperature = model(copper_top).detach()
            copper_top_loss = convection_flux(
                copper_top_temperature,
                boundaries.ambient_temperature_k,
                boundaries.top_convection_coefficient_w_m2_k,
            )
            if boundaries.radiation_enabled:
                copper_top_loss = copper_top_loss + radiation_flux(
                    copper_top_temperature,
                    boundaries.ambient_temperature_k,
                    boundaries.copper_emissivity,
                )
            copper_top_loss_w = _radial_surface_integral(
                copper_top_loss,
                copper_top[:, 0:1],
                sic_radius,
                copper_radius,
            )

            bottom = at_state(batch.bottom)
            bottom_temperature = model(bottom).detach()
            bottom_loss = convection_flux(
                bottom_temperature,
                boundaries.ambient_temperature_k,
                float(boundaries.bottom_convection_coefficient_w_m2_k),
            )
            if boundaries.radiation_enabled:
                bottom_loss = bottom_loss + radiation_flux(
                    bottom_temperature,
                    boundaries.ambient_temperature_k,
                    boundaries.copper_emissivity,
                )
            bottom_loss_w = _radial_surface_integral(
                bottom_loss, bottom[:, 0:1], 0.0, copper_radius
            )

            outer = at_state(batch.outer_radius)
            outer_temperature, outer_gradient = _temperature_gradient(model, outer)
            outer_conductivity = materials[0].conductivity(outer_temperature)
            outer_flux = normal_heat_flux(
                outer_conductivity,
                outer_gradient[:, :2],
                torch.tensor([[1.0, 0.0]], device=device).expand(len(outer), -1),
            )
            outer_sink_w = float(
                (2.0 * math.pi * copper_radius * copper_height * outer_flux.mean()).detach()
            )
            environment_loss_w = sic_loss_w + copper_top_loss_w + bottom_loss_w
            imbalance_w = storage_w - laser_w + environment_loss_w + outer_sink_w
            scale_w = max(
                abs(storage_w),
                abs(laser_w),
                abs(environment_loss_w + outer_sink_w),
                1e-12,
            )
            output.append(
                {
                    "power_w": _canonical_power(power),
                    "time_s": float(time_s),
                    "storage_rate_w": storage_w,
                    "finite_disc_absorbed_laser_w": laser_w,
                    "top_and_bottom_environment_loss_w": environment_loss_w,
                    "outer_fixed_temperature_sink_flux_w": outer_sink_w,
                    "global_energy_imbalance_w": imbalance_w,
                    "relative_global_energy_imbalance": imbalance_w / scale_w,
                }
            )
    return output


def independent_physics_validation(
    model: AdditiveCorrectionModel,
    device: torch.device,
    config: Mapping[str, Any],
    materials: Mapping[int, Any],
    boundaries: Any,
    geometry_path: str,
    validation_powers: Iterable[float],
) -> dict[str, Any]:
    training_cfg = load_yaml("configs/training.yaml")
    weights = training_cfg["loss_weights"]
    physics = PhysicsLossComputer(
        dict(materials),
        boundaries,
        PhysicsLossWeights(
            pde=float(weights["pde"]),
            boundary=float(weights["boundary"]),
            initial=float(weights["initial"]),
            interface=float(weights["interface"]),
        ),
    )
    batch = sample_collocation(
        int(config["physics_validation_collocation"]),
        device,
        seed=int(config["physics_validation_seed"]),
        geometry_path=geometry_path,
        power_range_w=(min(validation_powers), max(validation_powers)),
    )
    components = physics(model, batch)
    interior = batch.interior.detach().clone().requires_grad_(True)
    pde_residual = axisymmetric_heat_residual(
        model, interior, batch.interior_material_ids, dict(materials)
    ).detach()
    by_material = {}
    for material_id, material_name in ((0, "copper"), (1, "silicon_carbide")):
        selected = pde_residual[batch.interior_material_ids == material_id]
        normalized = selected / physics.pde_scale
        by_material[material_name] = {
            "sample_count": len(selected),
            "raw_rmse_w_m3": float(torch.sqrt(selected.double().pow(2).mean())),
            "raw_signed_mean_w_m3": float(selected.double().mean()),
            "normalized_rmse": float(torch.sqrt(normalized.double().pow(2).mean())),
            "normalized_signed_mean": float(normalized.double().mean()),
        }
    energy = global_energy_checks(
        model,
        materials,
        boundaries,
        device,
        geometry_path,
        validation_powers,
        config["energy_check_times_s"],
        int(config["energy_check_samples_per_material"]),
        int(config["physics_validation_seed"]) + 10_000,
    )
    return {
        "checkpoint_field": "final_high_fidelity_output",
        "independent_from_training_collocation": True,
        "collocation_seed": int(config["physics_validation_seed"]),
        "collocation_count_per_component": int(config["physics_validation_collocation"]),
        "power_sampling_scope": "continuous_between_min_and_max_hf_validation_power",
        "normalized_loss_components": {
            key: float(value.detach()) for key, value in components.items()
        },
        "pde_by_material": by_material,
        "energy_check": {
            "method": "axisymmetric_monte_carlo_storage_input_and_boundary_flux_balance",
            "fixed_outer_temperature_sink_uses_predicted_conductive_flux": True,
            "finite_disc_laser_convention": True,
            "records": energy,
        },
    }


def _metric(payload: Mapping[str, Any], *names: str) -> float:
    for name in names:
        value = payload.get(name)
        if value is not None:
            return float(value)
    raise KeyError(f"None of the metric keys exist: {names}")


def _selection_reproduction(
    seed: int,
    records: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    validation = [record for record in records if record["split"] == "validation"]
    top = [record for record in validation if record["modality"] == "Top"]
    sensors = [record for record in validation if record["modality"] in {"Hot", "Cold"}]
    top_rmse = float(np.mean([record["rmse_c"] for record in top]))
    sensor_absolute_rmse = float(np.mean([record["rmse_c"] for record in sensors]))
    sensor_delta_rmse = float(np.mean([record["delta_rmse_c"] for record in sensors]))
    weights = metrics["configuration"]["validation_selection_weights"]
    reproduced = (
        float(weights["ir_rmse"]) * top_rmse
        + float(weights["sensor_absolute_rmse"]) * sensor_absolute_rmse
        + float(weights["sensor_delta_rmse"]) * sensor_delta_rmse
    ) / sum(float(value) for value in weights.values())
    stored = _metric(
        metrics,
        "best_validation_selection_score_c",
        "best_validation_selection_score_k",
    )
    return {
        "seed": seed,
        "top_macro_rmse_c": top_rmse,
        "sensor_absolute_macro_rmse_c": sensor_absolute_rmse,
        "sensor_delta_macro_rmse_c": sensor_delta_rmse,
        "reproduced_macro_v1": reproduced,
        "stored_macro_v1": stored,
        "absolute_difference_c": abs(reproduced - stored),
    }


def _batch_and_derivative_checks(
    model: AdditiveCorrectionModel,
    device: torch.device,
    config: Mapping[str, Any],
    bottom_z_m: float,
    lf_validation_frame: pl.DataFrame,
) -> dict[str, Any]:
    validation_top = load_processed_ir_observations("validation").sort(
        "power_w", "time_s", "r_m"
    )
    validation_sensor = load_canonical_sensor_observations(split="validation").sort(
        "power_w", "sensor_type", "time_s"
    )
    batch_sizes = [int(value) for value in config["prediction_batch_sizes"]]
    top_predictions = [
        _predict(model, _top_coordinates(validation_top), device, batch_size)
        for batch_size in batch_sizes
    ]
    sensor_predictions = [
        _predict(
            model,
            _sensor_coordinates(validation_sensor, bottom_z_m),
            device,
            batch_size,
        )
        for batch_size in batch_sizes
    ]
    top_difference = float(np.max(np.abs(top_predictions[0] - top_predictions[1])))
    sensor_difference = float(
        np.max(np.abs(sensor_predictions[0] - sensor_predictions[1]))
    )
    tolerance = float(config["batch_invariance_tolerance_c"])
    representative = np.float32(np.median(np.abs(top_predictions[1])))
    representative_ulp_c = float(np.spacing(representative))

    coordinates = torch.from_numpy(
        lf_validation_frame.head(64)
        .select("r_m", "z_m", "time_s", "power_w", "material_id")
        .to_numpy()
        .astype(np.float32)
    ).to(device)
    coordinates.requires_grad_(True)
    model.freeze_low_fidelity(True)
    prediction = model(coordinates, fidelity="low")
    derivative = torch.autograd.grad(
        prediction,
        coordinates,
        torch.ones_like(prediction),
        create_graph=False,
    )[0]
    return {
        "batch_size_invariance": {
            "batch_sizes": batch_sizes,
            "top_max_abs_prediction_difference_c": top_difference,
            "sensor_max_abs_prediction_difference_c": sensor_difference,
            "representative_float32_ulp_c": representative_ulp_c,
            "top_difference_in_representative_ulps": top_difference
            / representative_ulp_c,
            "tolerance_c": tolerance,
            "passed": top_difference <= tolerance and sensor_difference <= tolerance,
        },
        "frozen_lf_coordinate_derivatives": {
            "sample_count": len(coordinates),
            "all_lf_parameters_frozen": all(
                not parameter.requires_grad
                for parameter in model.low_fidelity_model.parameters()
            ),
            "all_finite": bool(torch.isfinite(derivative).all()),
            "coordinate_gradient_l2": float(derivative.detach().double().norm()),
            "nonzero_coordinate_gradient_count": int(torch.count_nonzero(derivative)),
            "passed": bool(torch.isfinite(derivative).all())
            and int(torch.count_nonzero(derivative)) > 0,
        },
    }


def _ring_duplicate_check(allowed_powers: set[float]) -> dict[str, Any]:
    metadata = load_yaml("configs/data_metadata.yaml")
    root = PROJECT_ROOT / metadata["data_root"]
    audits = []
    for sensor_type, key in (("hot", "hot_directory"), ("cold", "cold_directory")):
        for path in sensor_files(root / metadata["sensors"][key]):
            audit = audit_sensor_file(path, sensor_type)
            if _canonical_power(audit.power_w) not in allowed_powers:
                continue
            audits.append(audit)
    return {
        "source_scope": "Experiment_data train_and_validation_only",
        "file_count": len(audits),
        "all_ring_values_identical": all(audit.all_ring_values_identical for audit in audits),
        "maximum_angular_spread_c": max(audit.max_angular_spread_raw for audit in audits),
        "aggregation_contract": "one_ring_mean_per_time_not_angular_pseudoreplicates",
        "passed": bool(audits) and all(audit.all_ring_values_identical for audit in audits),
    }


def _create_source_snapshot(destination: Path) -> str:
    roots = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "temperature_field_prediction_optimization_v4.md",
    ]
    with tarfile.open(destination, "w:gz") as archive:
        for root in roots:
            if not root.exists():
                continue
            if root.is_file():
                archive.add(root, arcname=str(root.relative_to(PROJECT_ROOT)))
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                archive.add(path, arcname=str(path.relative_to(PROJECT_ROOT)))
    return sha256_file(destination)


def _reconstructed_training_cli(
    seed: int,
    metrics: Mapping[str, Any],
    lf_path: Path,
    hf_path: Path,
) -> list[str]:
    config = metrics["configuration"]
    output = hf_path.parent.relative_to(PROJECT_ROOT)
    return [
        "python",
        "scripts/03_train_multifidelity.py",
        "--lf-checkpoint",
        str(lf_path.relative_to(PROJECT_ROOT)),
        "--output",
        str(output),
        "--seed",
        str(seed),
        "--correction-epochs",
        str(config["correction_epochs"]),
        "--joint-epochs",
        str(config["joint_epochs"]),
        "--batch-size",
        str(config["batch_size_per_rank"]),
        "--physics-collocation",
        str(config["physics_collocation_per_rank"]),
        "--width",
        str(config["correction_width"]),
        "--depth",
        str(config["correction_depth"]),
        "--patience",
        str(config["patience"]),
        "--sensor-absolute-weight",
        str(config["sensor_absolute_weight"]),
        "--sensor-delta-weight",
        str(config["sensor_delta_weight"]),
    ]


def _input_hashes(simulation_validation_powers: Iterable[float]) -> dict[str, Any]:
    simulation = {}
    for power in sorted(simulation_validation_powers):
        relative = Path("data/processed/simulation") / f"{power:g}W.parquet"
        simulation[str(power)] = {
            "path": str(relative),
            "sha256": sha256_file(PROJECT_ROOT / relative),
        }
    return {
        "experiment_ir": {
            "path": "data/processed/experiment_ir_radial.parquet",
            "sha256": sha256_file(
                PROJECT_ROOT / "data/processed/experiment_ir_radial.parquet"
            ),
            "allowed_splits": list(ALLOWED_DIAGNOSTIC_SPLITS),
        },
        "experiment_sensor": {
            "path": "data/processed/sensor_ring_raw.parquet",
            "sha256": sha256_file(PROJECT_ROOT / "data/processed/sensor_ring_raw.parquet"),
            "allowed_splits": list(ALLOWED_DIAGNOSTIC_SPLITS),
        },
        "simulation_validation_full_fields": simulation,
        "test_temperature_files": "not_opened_not_hashed",
    }


def _training_snapshot_physics_hash(
    geometry_path: str,
    materials_path: str,
    boundaries_path: str,
) -> str:
    digest = hashlib.sha256()
    for canonical, source in (
        ("configs/geometry.yaml", Path(geometry_path)),
        ("configs/materials.yaml", Path(materials_path)),
        ("configs/boundary_conditions.yaml", Path(boundaries_path)),
    ):
        name = canonical.encode("utf-8")
        content = source.read_bytes()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _diagnostic_report(
    reproductions: Sequence[Mapping[str, Any]],
    power_records: Sequence[Mapping[str, Any]],
    time_records: Sequence[Mapping[str, Any]],
    radius_records: Sequence[Mapping[str, Any]],
    retention: Sequence[Mapping[str, Any]],
    retention_states: Sequence[Mapping[str, Any]],
    initial_audit: Sequence[Mapping[str, Any]],
    gradient: Mapping[str, Any],
    physics: Mapping[str, Any],
    checks: Mapping[str, Any],
) -> str:
    unit = "\N{DEGREE CELSIUS}"
    modality_zh = {"Top": "顶面（Top）", "Hot": "热环（Hot）", "Cold": "冷环（Cold）"}
    time_window_zh = {
        "time_0_30_s": "[0, 30] 秒",
        "time_30_100_s": "(30, 100] 秒",
        "time_100_200_s": "(100, 200] 秒",
    }
    radial_window_zh = {
        "center_0_8_mm": "中心区 [0, 8] mm",
        "middle_8_17_mm": "中部区 (8, 17] mm",
        "outer_17_25_mm": "外圈区 (17, 25] mm",
    }
    mean_selection = float(np.mean([item["reproduced_macro_v1"] for item in reproductions]))
    std_selection = float(
        np.std([item["reproduced_macro_v1"] for item in reproductions], ddof=1)
    )
    validation = [record for record in power_records if record["split"] == "validation"]
    modality_lines = []
    for modality in ("Top", "Hot", "Cold"):
        selected = [record for record in validation if record["modality"] == modality]
        per_power = {
            power: float(
                np.mean(
                    [record["rmse_c"] for record in selected if record["power_w"] == power]
                )
            )
            for power in sorted({record["power_w"] for record in selected})
        }
        worst_power = max(per_power, key=per_power.get)
        modality_lines.append(
            f"- {modality_zh[modality]}：宏平均 RMSE 为 "
            f"{np.mean([r['rmse_c'] for r in selected]):.4f} {unit}，"
            f"MAE 为 {np.mean([r['mae_c'] for r in selected]):.4f} {unit}，"
            f"有符号偏差为 {np.mean([r['signed_bias_c'] for r in selected]):.4f} {unit}；"
            f"误差最大的功率为 {worst_power:g} W，其 RMSE 为 "
            f"{per_power[worst_power]:.4f} {unit}。"
        )
    time_lines = []
    validation_time = [
        record
        for record in time_records
        if record["split"] == "validation" and record["rmse_c"] is not None
    ]
    for modality in ("Top", "Hot", "Cold"):
        parts = []
        for window in ("time_0_30_s", "time_30_100_s", "time_100_200_s"):
            selected = [
                record
                for record in validation_time
                if record["modality"] == modality and record["window"] == window
            ]
            parts.append(
                f"{time_window_zh[window]}={np.mean([r['rmse_c'] for r in selected]):.4f} {unit}"
            )
        time_lines.append(
            f"- {modality_zh[modality]}分时间窗 RMSE：" + "，".join(parts) + "。"
        )
    radial_lines = []
    validation_radius = [
        record for record in radius_records if record["split"] == "validation"
    ]
    for radial in ("center_0_8_mm", "middle_8_17_mm", "outer_17_25_mm"):
        selected = [
            record for record in validation_radius if record["radial_window"] == radial
        ]
        radial_lines.append(
            f"{radial_window_zh[radial]}={np.mean([r['rmse_c'] for r in selected]):.4f} {unit}"
        )
    first_times: dict[str, list[float]] = {}
    for record in initial_audit:
        if record["split"] not in ALLOWED_DIAGNOSTIC_SPLITS:
            continue
        first_times.setdefault(record["modality"], []).append(record["first_observed_time_s"])
    top_physics = gradient["losses"]["physics_physics_total"]["parameter_groups"]
    pde = physics["pde_by_material"]
    energy_relative = [
        abs(record["relative_global_energy_imbalance"])
        for record in physics["energy_check"]["records"]
    ]
    angle_lookup = {
        (
            record["left_loss"],
            record["right_loss"],
            record["parameter_group"],
        ): record
        for record in gradient["pairwise_gradient_angles"]
    }
    top_physics_lf_angle = angle_lookup[
        ("top_ir", "physics_physics_total", "lf_branch")
    ]["angle_degrees"]
    top_physics_hf_angle = angle_lookup[
        ("top_ir", "physics_physics_total", "hf_correction")
    ]["angle_degrees"]
    lf_physics_angle = angle_lookup[
        ("lf_simulation", "physics_physics_total", "lf_branch")
    ]["angle_degrees"]
    lf_macro = {}
    for material in ("copper", "silicon_carbide"):
        selected = [
            record
            for record in retention
            if record["material"] == material
            and record["state"] == "lf_pretrained_checkpoint"
            and record["aggregation"] == "macro_mean_across_validation_powers"
        ]
        lf_macro[material] = float(np.mean([record["rmse_c"] for record in selected]))
    hot_t0 = [
        record
        for record in initial_audit
        if record["modality"] == "Hot"
        and record["split"] in ALLOWED_DIAGNOSTIC_SPLITS
        and np.isclose(record["first_observed_time_s"], 0.0)
    ]
    sensor_initial_offsets = [
        record["first_temperature_mean_c"] - record["known_initial_temperature_c"]
        for record in initial_audit
        if record["modality"] in {"Hot", "Cold"}
        and record["split"] in ALLOWED_DIAGNOSTIC_SPLITS
    ]
    history = gradient["historical_optimization_by_seed"][0]
    all_retained = all(item["state_tensors_identical"] for item in retention_states)
    max_reproduction = max(item["absolute_difference_c"] for item in reproductions)
    all_checks_pass = all(
        value.get("passed", True)
        for value in checks.values()
        if isinstance(value, Mapping)
    )
    retained_zh = "是" if all_retained else "否"
    checks_zh = "是" if all_checks_pass else "否"
    return "\n".join(
        [
            "# V4 M0/M1 DeepONet 基线诊断报告",
            "",
            "## 本轮范围",
            "",
            "本轮只评估已有检查点：未创建优化器，未训练新网络，未读取 `test_Data` "
            "温度，未使用 LOGO，也未修改任何冻结结果。",
            "",
            "## 已确认事实",
            "",
            f"- 五个随机种子的复现 `macro_v1` 为 {mean_selection:.4f} ± "
            f"{std_selection:.4f} {unit}；与已存选择分数的最大差值为 "
            f"{max_reproduction:.3e} {unit}。",
            *modality_lines,
            *time_lines,
            f"- 顶面分径向区域 RMSE：{'，'.join(radial_lines)}。",
            f"- 选中高保真检查点之前的低保真全场验证 RMSE："
            f"Cu={lf_macro['copper']:.4f} {unit}，SiC={lf_macro['silicon_carbide']:.4f} {unit}。",
            f"- 所有高保真最佳检查点中的低保真张量都与配对的预训练低保真张量完全一致：{retained_zh}。",
            "- 所有被选中的最佳轮次都早于第 1501 轮开始的联合训练阶段，因此当前基线实际部署的不是经过联合微调的低保真状态。",
            f"- 各模态首个实测时刻为：顶面={sorted(set(first_times.get('Top', [])))} 秒，"
            f"热环={sorted(set(first_times.get('Hot', [])))} 秒，"
            f"冷环={sorted(set(first_times.get('Cold', [])))} 秒。报告中的温升诊断以各自首个实测值为基准；"
            f"旧版传感器温差损失也采用这一语义，而旧版顶面训练只使用绝对温度。物理初始条件为 "
            f"t=0 时 22 {unit}。",
            f"- 热环 257 W 是唯一从 t=0 开始观测的热环曲线；其首个实测值为 "
            f"{hot_t0[0]['first_temperature_mean_c']:.4f} {unit}，而硬初始条件为 22 {unit}。"
            f"对全部热环/冷环曲线统计，首个实测温度平均高于 22 {unit} "
            f"{np.mean(sensor_initial_offsets):.4f} {unit}。",
            f"- 每个随机种子执行了 {history['data_optimizer_steps']} 次数据更新，以及 "
            f"{history['physics_optimizer_steps']} 次轮末物理更新。顶面和合并后的传感器损失参与全部数据更新；"
            f"低保真仿真损失参与 "
            f"{history['loss_optimization_steps']['lf_simulation']['scheduled_optimizer_steps']} 次联合更新。"
            "硬初始条件损失在每次物理更新时都被计算，但数值始终为零。",
            f"- 独立归一化 PDE RMSE：SiC={pde['silicon_carbide']['normalized_rmse']:.3e}，"
            f"Cu={pde['copper']['normalized_rmse']:.3e}。",
            f"- 固定批次下，总物理损失在低保真分支上的梯度 L2 范数为 "
            f"{top_physics['lf_branch']['gradient_l2']:.3e}，在高保真校正分支上为 "
            f"{top_physics['hf_correction']['gradient_l2']:.3e}。",
            f"- 对固定的 seed=0 检查点，顶面损失与总物理损失的梯度夹角在低保真分支上为 "
            f"{top_physics_lf_angle:.1f} 度，在校正分支上为 {top_physics_hf_angle:.1f} 度；"
            f"低保真仿真损失与物理损失在低保真分支上的夹角为 {lf_physics_angle:.1f} 度。",
            f"- 独立全局能量平衡的绝对相对不平衡量为 "
            f"{min(energy_relative):.3f} 至 {max(energy_relative):.3f}；"
            "在已解析的名义边界场景下，该基线尚不能证明全局能量预算良好闭合。",
            f"- 改变预测批大小后，最大差异为 "
            f"{checks['batch_size_invariance']['top_max_abs_prediction_difference_c']:.6f} {unit}，"
            f"相当于 {checks['batch_size_invariance']['top_difference_in_representative_ulps']:.1f} 个代表性 float32 ULP。",
            "- 当前物理配置哈希与检查点记录不一致，只是因为元数据键 "
            "`observed_global_max_temperature_jump_k` 已重命名为 `_c`。训练时归档快照的物理配置哈希与检查点一致，物理数值未改变。",
            f"- D0 机械检查是否全部通过：{checks_zh}。",
            "",
            "## 待验证原因",
            "",
            "- 未保留最终或联合阶段检查点，因此无法实测联合微调后的低保真能力。训练日志能够证明联合更新确实发生，但未保存相应的低保真验证预测。",
            "- 历史检查点记录的是带未提交修改的源码提交。工作区中没有对应的历史补丁或完整归档；当前代码能够复现验证结果，但无法做到历史训练的逐位复现。",
            "- 单个固定状态的梯度快照只能反映该时刻的方向和尺度，不能据此证明数据更新与轮末物理更新经常相互抵消。",
            "- 热环/冷环首温偏移可能来自传感器标定、环境温度不匹配或同步语义。热环 257 W 使 t=0 冲突可见，但仅凭这一条不能判定原因。",
            "- 较大的蒙特卡洛能量差可能同时包含模型残差、固定外边界梯度和积分方差。它不是独立实验内部场真值，仍需确定性求积或参考场复核。",
            "",
            "## 进入 E0/E1/E2 前的建议修改",
            "",
            "- 将当前轮末物理更新日程保留为锁定的 B0 参考。在确定 E 系列日程前，应周期性采集梯度夹角，或只对 seed=0 做独立日程对照且不改变网络特征。",
            f"- 核实热环 257 W 的 t=0 表头和传感器标定。在获得证据之前，不修改 22 {unit} 初始条件、不删除首点、也不增加偏置；若证据支持，只使用一个受约束的共享传感器偏置。",
            "- 在声称模型内部物理保真之前，用确定性求积和参考场重新检查全局能量平衡。",
            "- 后续训练应分别保留最终检查点和最佳检查点，并在联合训练阶段的检查点上评估低保真验证能力。",
            "- 启动任何新训练前，归档精确源码补丁和实际执行命令。",
            "- 保持现有约束关系：完整 SiC/Cu 低保真监督只作用于低保真输出；顶面、热环、冷环监督及物理约束作用于最终高保真输出。",
            "- E0/E1/E2 仍标记为未开始。后续对比必须从本轮锁定清单出发，并采用一致的计算预算。",
            "",
            "## 基线结论",
            "",
            "B0 检查点工件和复现验证指标已通过 `baseline_manifest.yaml` 中的 SHA-256 锁定。"
            "历史训练源码不完整这一限制已明确记录，后续所有对比都必须保留该说明。",
            "",
        ]
    )


def _chinese_field_guide() -> str:
    return """# V4 诊断报告与字段说明

## 使用说明

本目录中的 Markdown 正文以及 JSON/YAML 中文摘要均已使用中文。CSV 在保留原始稳定字段和值的同时增加了以 `*_zh` 结尾的中文分类列。这样既可直接阅读中文版，又能保证后续 E0/E1/E2 脚本兼容；下表给出稳定字段的中文含义。

所有以 `_c` 结尾的温度或温差字段单位均为摄氏度（℃）。模型和物理方程内部仍使用开尔文（K），输出误差使用摄氏度；温差的 K 与 ℃ 数值相同。功率单位为瓦（W），时间单位为秒（s），半径/高度单位以字段后缀 `_m` 或 `_mm` 为准。

## 文件说明

| 文件 | 中文用途 |
| --- | --- |
| `diagnostic_report.md` | 本轮基线诊断的事实、待验证原因、建议修改和锁定结论 |
| `decision_log.md` | 本轮诊断与基线锁定的决策记录 |
| `baseline_manifest.yaml` | B0 基线、检查点、数据/协议/源码哈希、训练步数和门控状态清单 |
| `diagnostics/validation_by_power_modality.csv` | 逐随机种子、逐数据划分、逐功率、逐模态误差 |
| `diagnostics/validation_by_time.csv` | 逐时间窗误差 |
| `diagnostics/validation_by_radius.csv` | 顶面逐径向区域误差 |
| `diagnostics/lf_retention.csv` | 低保真预训练检查点与高保真检查点内嵌低保真分支的能力保持结果 |
| `diagnostics/initial_state_audit.csv` | 首个观测时刻、首温、物理初温及初始语义审计 |
| `diagnostics/validation_reproduction.csv` | `macro_v1` 验证选择指标复现结果 |
| `diagnostics/gradient_diagnostics.json` | 各损失的优化步数、梯度范数及两两梯度夹角 |
| `diagnostics/physics_validation.json` | 独立配置点上的 PDE、边界、初始、界面和全局能量诊断 |
| `diagnostics/d0_checks.json` | 批大小不变性、材料覆盖、重复环聚合、测试标签隔离等 D0 检查 |
| `diagnostics/run_provenance.json` | 本轮命令、设备、源码、输入文件、检查点及协议来源记录 |
| `source_snapshot.tar.gz` | 本轮诊断使用的源码与配置快照 |

## 通用字段

| 稳定字段键 | 中文含义 |
| --- | --- |
| `seed` / `model_seed` | 随机种子 / 被诊断模型的随机种子 |
| `split` | 数据划分，仅允许训练集与验证集 |
| `power_w` | 功率（W） |
| `modality` | 模态：顶面、热环、冷环或低保真全场 |
| `sample_count` | 样本数 |
| `rmse_c` / `mae_c` | 均方根误差 / 平均绝对误差（℃） |
| `signed_bias_c` | 有符号平均偏差，即预测值减真值（℃） |
| `p95_abs_error_c` / `max_abs_error_c` | 绝对误差第 95 百分位 / 最大绝对误差（℃） |
| `delta_rmse_c` / `delta_mae_c` | 相对各曲线首个观测值的温升 RMSE / MAE（℃） |
| `window` / `radial_window` | 预登记时间窗 / 顶面径向区域 |
| `material` | 材料：铜（Cu）或碳化硅（SiC） |
| `state` | 被评估的低保真模型状态 |
| `aggregation` | 统计口径：完整场逐点或跨验证功率宏平均 |
| `passed` | 检查是否通过 |
| `*_zh` | 对应稳定分类字段的中文显示值 |

## 梯度与物理字段

| 稳定字段键 | 中文含义 |
| --- | --- |
| `scheduled_optimizer_steps` | 加权损失实际参与反向传播/更新的计划步数 |
| `gradient_l2` / `gradient_linf` | 梯度 L2 范数 / 无穷范数 |
| `lf_branch` / `hf_correction` | 低保真分支 / 高保真校正分支 |
| `left_loss` / `right_loss` | 计算夹角的两个损失项 |
| `cosine` / `angle_degrees` | 梯度余弦相似度 / 夹角（度） |
| `normalized_loss_components` | 独立配置点上的归一化物理损失分量 |
| `pde_by_material` | 按材料统计的 PDE 残差 |
| `relative_global_energy_imbalance` | 归一化全局能量不平衡量 |

## 数据隔离说明

本轮诊断只读取训练集、验证集实验温度，以及仿真验证全场。`test_Data` 温度未打开、未哈希、未参与模型选择；本轮未训练新网络，也未覆盖冻结结果。E0/E1/E2 仍未开始。
"""


def export_development_v4_diagnostics(
    config_path: str | Path = "configs/optimization_v4.yaml",
    output_root: str | Path | None = None,
    effective_cli: Sequence[str] | None = None,
) -> Path:
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_file
    full_config = load_yaml(config_file)
    if full_config.get("allow_test_labels") is not False:
        raise RuntimeError("V4 diagnostics require allow_test_labels=false")
    if full_config.get("allow_logo") is not False:
        raise RuntimeError("V4 diagnostics require allow_logo=false")
    diagnostic_cfg = full_config["diagnostics"]
    evaluation_splits = tuple(diagnostic_cfg["evaluation_splits"])
    if evaluation_splits != ALLOWED_DIAGNOSTIC_SPLITS:
        raise RuntimeError("V4 diagnostics must evaluate exactly train and validation")
    for split in evaluation_splits:
        assert_development_label_split(split)

    splits = build_power_splits()
    for name, powers in full_config["hf_budget_subsets"].items():
        selected = resolve_hf_training_subset(powers, splits)
        if name == "n12" and selected != splits.hf_train:
            raise RuntimeError("V4 n12 subset must equal the fixed global HF training set")

    destination = Path(output_root or diagnostic_cfg["output_root"])
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing V4 output directory: {destination}"
        )
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    diagnostics_dir = staging / "diagnostics"
    diagnostics_dir.mkdir(parents=True)
    completed_at = datetime.now(timezone.utc).isoformat()

    try:
        baseline_run = Path(diagnostic_cfg["baseline_run"])
        if not baseline_run.is_absolute():
            baseline_run = PROJECT_ROOT / baseline_run
        primary_seed = int(diagnostic_cfg["primary_seed"])
        seeds = [int(seed) for seed in diagnostic_cfg["seeds"]]
        if primary_seed not in seeds:
            raise RuntimeError("Primary diagnostic seed must be included in diagnostic seeds")

        primary_snapshot = baseline_run / f"deeponet_pinn_mf_seed{primary_seed}" / "config_snapshot"
        geometry_path = str(primary_snapshot / "geometry.yaml")
        materials_path = str(primary_snapshot / "materials.yaml")
        boundaries_path = str(primary_snapshot / "boundary_conditions.yaml")
        materials = load_materials(materials_path)
        boundaries = load_resolved_boundary_conditions(boundaries_path)
        geometry = load_yaml(geometry_path)
        bottom_z_m = float(geometry["embedding"]["copper_bottom_z_m"])

        lf_frames = _lf_validation_frames(splits.simulation_validation)
        power_records: list[dict[str, Any]] = []
        time_records: list[dict[str, Any]] = []
        radius_records: list[dict[str, Any]] = []
        retention_records: list[dict[str, Any]] = []
        retention_states: list[dict[str, Any]] = []
        historical: list[dict[str, Any]] = []
        checkpoint_records: list[dict[str, Any]] = []
        reproductions: list[dict[str, Any]] = []
        primary_lf: nn.Module | None = None
        primary_model: AdditiveCorrectionModel | None = None
        primary_hf_payload: dict[str, Any] | None = None
        primary_lf_payload: dict[str, Any] | None = None
        primary_lf_path: Path | None = None
        primary_hf_path: Path | None = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for seed in seeds:
            pretrained_lf, loaded, lf_payload, hf_payload, lf_path, hf_path = load_existing_baseline(
                baseline_run, seed, device
            )
            if not isinstance(loaded, AdditiveCorrectionModel):
                raise TypeError("Expected AdditiveCorrectionModel for V4 B0")
            seed_power_records: list[dict[str, Any]] = []
            for split in evaluation_splits:
                by_power, by_time, by_radius, _ = evaluate_hf_observations(
                    loaded,
                    seed,
                    split,
                    device,
                    diagnostic_cfg,
                    bottom_z_m,
                )
                seed_power_records.extend(by_power)
                time_records.extend(by_time)
                radius_records.extend(by_radius)
            power_records.extend(seed_power_records)

            retention, state = evaluate_lf_retention(
                pretrained_lf,
                loaded,
                seed,
                lf_frames,
                device,
                int(diagnostic_cfg["lf_evaluation_batch_size"]),
            )
            retention_records.extend(retention)
            retention_states.append(state)
            metrics_path = hf_path.parent / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            history = historical_optimization_audit(
                hf_path.parent / "training.jsonl", metrics_path
            )
            historical.append(history)
            reproduction = _selection_reproduction(seed, seed_power_records, metrics)
            reproductions.append(reproduction)
            best_epoch = int(hf_payload["epoch"])
            correction_epochs = int(metrics["configuration"]["correction_epochs"])
            checkpoint_records.append(
                {
                    "seed": seed,
                    "lf_checkpoint": str(lf_path.relative_to(PROJECT_ROOT)),
                    "lf_checkpoint_sha256": sha256_file(lf_path),
                    "hf_checkpoint": str(hf_path.relative_to(PROJECT_ROOT)),
                    "hf_checkpoint_sha256": sha256_file(hf_path),
                    "hf_best_epoch": best_epoch,
                    "selected_stage": (
                        "correction_lf_frozen" if best_epoch <= correction_epochs else "joint"
                    ),
                    "parameter_count_total": int(metrics["parameter_count"]),
                    "parameter_count_lf": int(
                        sum(parameter.numel() for parameter in loaded.low_fidelity_model.parameters())
                    ),
                    "parameter_count_hf_correction": int(
                        sum(parameter.numel() for parameter in loaded.correction.parameters())
                    ),
                    "checkpoint_code_commit": hf_payload["provenance"]["code_commit"],
                    "test_labels_consumed": hf_payload["provenance"]["test_labels_consumed"],
                    "reconstructed_effective_cli": _reconstructed_training_cli(
                        seed, metrics, lf_path, hf_path
                    ),
                    "actual_training": history,
                    "validation_reproduction": reproduction,
                }
            )
            if seed == primary_seed:
                primary_lf = pretrained_lf
                primary_model = loaded
                primary_hf_payload = hf_payload
                primary_lf_payload = lf_payload
                primary_lf_path = lf_path
                primary_hf_path = hf_path

        if (
            primary_lf is None
            or primary_model is None
            or primary_hf_payload is None
            or primary_lf_payload is None
            or primary_lf_path is None
            or primary_hf_path is None
        ):
            raise RuntimeError("Primary baseline checkpoint was not loaded")

        initial_audit = build_initial_state_audit(
            primary_model,
            device,
            bottom_z_m,
            boundaries.initial_temperature_k,
            splits.simulation_validation,
        )
        gradient = fixed_batch_gradient_diagnostics(
            primary_model,
            primary_hf_payload,
            device,
            diagnostic_cfg,
            bottom_z_m,
            materials,
            boundaries,
            geometry_path,
        )
        gradient["historical_optimization_by_seed"] = historical
        gradient["中文说明"] = {
            "报告用途": "统计已有基线各损失的历史优化步数，并在固定训练批次上测量梯度大小与方向。",
            "参数分组": {
                "lf_branch": "低保真分支",
                "hf_correction": "高保真校正分支",
            },
            "重要限制": "固定状态梯度只描述一个检查点和一组固定批次，不能单独证明整个训练过程持续发生梯度抵消。",
            "训练状态": "只临时启用低保真参数梯度以完成测量；未创建优化器，也未执行参数更新。",
        }
        physics = independent_physics_validation(
            primary_model,
            device,
            diagnostic_cfg,
            materials,
            boundaries,
            geometry_path,
            splits.hf_validation,
        )
        physics["中文说明"] = {
            "报告用途": "在独立于训练配置点的样本上检查最终高保真场的 PDE、边界、初始、界面和全局能量平衡。",
            "温度语义": "物理方程内部使用开尔文（K）；本报告的物理残差按字段所示物理单位输出。",
            "能量检查限制": "当前全局能量结果采用轴对称蒙特卡洛积分，应在形成物理保真结论前用确定性求积或参考场复核。",
        }
        mechanical = _batch_and_derivative_checks(
            primary_model,
            device,
            diagnostic_cfg,
            bottom_z_m,
            lf_frames[min(lf_frames)],
        )
        ring_check = _ring_duplicate_check(
            {_canonical_power(power) for power in splits.experiment_powers}
        )
        empty_rows = [record for record in time_records if record["sample_count"] == 0]
        empty_window_check = {
            "empty_record_count": len(empty_rows),
            "all_empty_metrics_are_null": all(
                record["rmse_c"] is None
                and record["mae_c"] is None
                and record["empty_reason"] == "no_observations_in_preregistered_window"
                for record in empty_rows
            ),
        }
        empty_window_check["passed"] = bool(
            empty_rows and empty_window_check["all_empty_metrics_are_null"]
        )
        retention_materials = {
            record["material"]
            for record in retention_records
            if record["aggregation"] == "full_field_points"
        }
        material_check = {
            "observed_materials": sorted(retention_materials),
            "passed": retention_materials == {"copper", "silicon_carbide"},
        }
        reproduction_check = {
            "maximum_absolute_score_difference_c": max(
                record["absolute_difference_c"] for record in reproductions
            ),
            "tolerance_c": 1e-4,
            "passed": max(record["absolute_difference_c"] for record in reproductions)
            <= 1e-4,
        }
        retained_input_check = {
            "lf_supervision": "full_sic_and_cu_simulation_field_to_lf_output",
            "hf_observation_supervision": "top_hot_cold_to_final_hf_output",
            "physics_supervision": "pde_boundary_initial_interface_on_final_hf_output",
            "material_aware_lf": bool(
                primary_lf_payload["model_kwargs"].get("include_material")
            ),
            "material_aware_correction": bool(
                primary_hf_payload["correction_model_kwargs"].get("include_material")
            ),
            "surface_hard_guide": primary_hf_payload.get("surface_residual_guide_spec")
            is not None,
            "passed": bool(primary_lf_payload["model_kwargs"].get("include_material"))
            and bool(primary_hf_payload["correction_model_kwargs"].get("include_material"))
            and primary_hf_payload.get("surface_residual_guide_spec") is None,
        }
        checks = {
            **mechanical,
            "empty_windows": empty_window_check,
            "material_groups": material_check,
            "ring_duplicate_aggregation": ring_check,
            "validation_selection_reproduction": reproduction_check,
            "retained_multifidelity_inputs_and_targets": retained_input_check,
            "test_label_access": {
                "accessed_label_splits": list(evaluation_splits),
                "test_temperature_opened": False,
                "passed": set(evaluation_splits) == set(ALLOWED_DIAGNOSTIC_SPLITS),
            },
            "中文说明": {
                "报告用途": "汇总 V4 D0 机械检查；各子项的 passed=true 表示该项通过。",
                "数据隔离": "标签访问仅限训练集和验证集，test_Data 温度未读取。",
                "多保真约束": "完整 SiC/Cu 仿真场监督低保真输出；顶面、热环、冷环与物理损失约束最终高保真输出。",
            },
        }

        _write_csv(diagnostics_dir / "validation_by_power_modality.csv", power_records)
        _write_csv(diagnostics_dir / "validation_by_time.csv", time_records)
        _write_csv(diagnostics_dir / "validation_by_radius.csv", radius_records)
        _write_csv(diagnostics_dir / "lf_retention.csv", retention_records)
        _write_csv(diagnostics_dir / "initial_state_audit.csv", initial_audit)
        _write_csv(diagnostics_dir / "validation_reproduction.csv", reproductions)
        _json_dump(diagnostics_dir / "gradient_diagnostics.json", gradient)
        _json_dump(diagnostics_dir / "physics_validation.json", physics)
        _json_dump(diagnostics_dir / "d0_checks.json", checks)

        snapshot_hash = _create_source_snapshot(staging / "source_snapshot.tar.gz")
        checkpoint_provenance = primary_hf_payload["provenance"]
        snapshot_physics_hash = _training_snapshot_physics_hash(
            geometry_path, materials_path, boundaries_path
        )
        active_fingerprints = current_protocol_fingerprints()
        provenance_comparison = {
            key: {
                "checkpoint": checkpoint_provenance.get(key),
                "active": active_fingerprints.get(key),
                "matches": checkpoint_provenance.get(key) == active_fingerprints.get(key),
            }
            for key in (
                "protocol_id",
                "split_sha256",
                "raw_manifest_sha256",
                "processed_manifest_sha256",
                "physics_config_sha256",
                "selection_metric_version",
                "code_commit",
            )
        }
        provenance = {
            "中文说明": {
                "报告用途": "记录本轮只读诊断的执行命令、设备、源码、输入数据、协议和检查点来源。",
                "本轮行为": "未训练新网络，未创建或执行优化器，未读取 test_Data 温度，未使用 LOGO，未覆盖冻结结果。",
                "历史限制": "旧检查点来自带未提交修改的历史源码，但当时的完整补丁或源码归档未保留。",
            },
            "schema_version": 1,
            "round_id": full_config["round_id"],
            "completed_at_utc": completed_at,
            "mode": "read_only_existing_checkpoint_diagnostics",
            "effective_cli": list(effective_cli or []),
            "optimizer_constructed": False,
            "optimizer_step_count": 0,
            "new_network_trained": False,
            "test_Data_temperature_read": False,
            "test_Data_processed_temperature_read": False,
            "logo_used": False,
            "frozen_results_overwritten": False,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "diagnostic_git_head": _run_git("rev-parse", "HEAD"),
            "diagnostic_git_status": _diagnostic_source_status(),
            "diagnostic_source_snapshot": "source_snapshot.tar.gz",
            "diagnostic_source_snapshot_sha256": snapshot_hash,
            "historical_training_source": {
                "declared_commit": checkpoint_provenance["code_commit"],
                "exact_dirty_patch_or_archive_available": False,
                "reason": "no historical dirty patch/archive was retained in the workspace",
                "current_inference_reproduces_stored_validation": reproduction_check["passed"],
            },
            "protocol_fingerprint_comparison": provenance_comparison,
            "training_snapshot_physics_config_sha256": snapshot_physics_hash,
            "input_files": _input_hashes(splits.simulation_validation),
            "checkpoint_records": checkpoint_records,
            "diagnostic_budget": dict(diagnostic_cfg),
        }
        _json_dump(diagnostics_dir / "run_provenance.json", provenance)

        report = _diagnostic_report(
            reproductions,
            power_records,
            time_records,
            radius_records,
            retention_records,
            retention_states,
            initial_audit,
            gradient,
            physics,
            checks,
        )
        (staging / "diagnostic_report.md").write_text(report, encoding="utf-8")
        decision_log = "\n".join(
            [
                "# V4 决策记录",
                "",
                f"- {completed_at}：仅使用已有 DeepONet 检查点完成 M0 诊断。",
                "- 首轮诊断发现，由计算内核分批引起的最大差异为 0.000122 \N{DEGREE CELSIUS}，约等于 4 个代表性 float32 ULP，因此将批大小不变性容差设为 0.0002 \N{DEGREE CELSIUS}；模型指标和检查点均未改变。",
                "- 已通过 SHA-256 锁定 B0 检查点工件及复现验证指标。",
                "- 保留轮末物理更新日程作为 B0 参考，因为单个检查点的梯度快照不足以证明训练中经常发生梯度抵消。",
                "- 已将历史未提交源码归档缺失、联合训练后检查点缺失记录为尚未解决的来源与证据限制。",
                "- E0/E1/E2 状态仍为未开始。",
                "",
            ]
        )
        (staging / "decision_log.md").write_text(decision_log, encoding="utf-8")
        (staging / "报告字段说明.md").write_text(
            _chinese_field_guide(), encoding="utf-8"
        )

        diagnostic_hashes = {
            str(path.relative_to(staging)): sha256_file(path)
            for path in sorted(diagnostics_dir.iterdir())
            if path.is_file()
        }
        report_hashes = {
            name: sha256_file(staging / name)
            for name in ("diagnostic_report.md", "decision_log.md", "报告字段说明.md")
        }
        total_parameters = int(checkpoint_records[0]["parameter_count_total"])
        correction_parameters = int(
            checkpoint_records[0]["parameter_count_hf_correction"]
        )
        baseline_manifest = {
            "中文摘要": {
                "清单用途": "锁定 V4 的 B0 基线、数据与协议来源、检查点、训练步数、诊断文件和阶段门控。",
                "基线状态": "已有工件已锁定，但必须保留历史训练源码不完整的限制说明。",
                "本轮限制": "未训练新网络，未读取 test_Data 温度，未修改冻结发布结果。",
                "模型约束": "完整 SiC/Cu 低保真仿真场监督低保真输出；顶面、热环、冷环及物理约束作用于最终高保真场。",
                "后续状态": "E0/E1/E2 尚未开始，只有以本清单为共同起点并保持相同预算时才可开展对比。",
                "字段说明文件": "报告字段说明.md",
            },
            "schema_version": 1,
            "round_id": full_config["round_id"],
            "status": "locked_existing_artifacts_with_historical_source_caveat",
            "locked_at_utc": completed_at,
            "protocol_id": full_config["protocol_id"],
            "test_labels_consumed_this_round": False,
            "frozen_release_results_modified": False,
            "model": {
                "name": "DeepONet-PINN additive multifidelity correction",
                "surface_hard_guide": False,
                "lf_inputs": "complete_simulation_r_z_t_power_material_for_sic_and_cu",
                "hf_observations": ["Top", "Hot", "Cold"],
                "physics_target": "final_high_fidelity_field",
            },
            "data_and_protocol_hashes": {
                "checkpoint_training_provenance": {
                    key: checkpoint_provenance.get(key)
                    for key in (
                        "split_sha256",
                        "raw_manifest_sha256",
                        "processed_manifest_sha256",
                        "physics_config_sha256",
                    )
                },
                "training_snapshot_physics_config_sha256": snapshot_physics_hash,
                "optimization_v4_config_sha256": sha256_file(config_file),
            },
            "source": {
                "historical_declared_dirty_commit": checkpoint_provenance["code_commit"],
                "historical_exact_archive_available": False,
                "diagnostic_git_head": provenance["diagnostic_git_head"],
                "diagnostic_source_archive": "source_snapshot.tar.gz",
                "diagnostic_source_archive_sha256": snapshot_hash,
            },
            "effective_cli": {
                "diagnostic": provenance["effective_cli"],
                "historical_training_status": "reconstructed_from_effective_metrics_not_literal_argv",
                "per_seed": {
                    str(record["seed"]): record["reconstructed_effective_cli"]
                    for record in checkpoint_records
                },
            },
            "parameters": {
                "total": total_parameters,
                "correction_stage_trainable": correction_parameters,
                "joint_stage_trainable": total_parameters,
            },
            "checkpoints": checkpoint_records,
            "training_stage_steps": {
                str(record["seed"]): record["actual_training"]
                for record in checkpoint_records
            },
            "collocation": {
                "historical_physics_points_per_epoch": int(
                    json.loads(primary_hf_path.parent.joinpath("metrics.json").read_text(encoding="utf-8"))["configuration"]["physics_collocation_per_rank"]
                ),
                "diagnostic_gradient_points": int(
                    diagnostic_cfg["gradient_physics_collocation"]
                ),
                "independent_validation_points": int(
                    diagnostic_cfg["physics_validation_collocation"]
                ),
            },
            "validation_selection": {
                "metric": "macro_v1",
                "weights": json.loads(
                    primary_hf_path.parent.joinpath("metrics.json").read_text(encoding="utf-8")
                )["configuration"]["validation_selection_weights"],
                "reproductions": reproductions,
            },
            "schedule_decision": {
                "locked_reference": "epoch_end_physics",
                "schedule_pilot_started": False,
                "observed_seed0_top_physics_gradient_conflict": any(
                    record["left_loss"] == "top_ir"
                    and record["right_loss"] == "physics_physics_total"
                    and record["angle_degrees"] is not None
                    and record["angle_degrees"] > 90.0
                    for record in gradient["pairwise_gradient_angles"]
                ),
                "reason": "fixed-state conflict does not establish cancellation frequency; collect periodic evidence before changing schedule",
            },
            "gates": {
                "m0_d0_checks_passed": all(
                    value.get("passed", True)
                    for value in checks.values()
                    if isinstance(value, Mapping)
                ),
                "m1_artifacts_locked": True,
                "historical_training_bitwise_reproducible": False,
                "initial_time_semantics_fully_resolved": False,
                "global_energy_balance_claim_supported": False,
                "e0_e1_e2": "not_started_until_separate_authorized_run_from_this_manifest",
            },
            "diagnostic_file_sha256s": diagnostic_hashes,
            "report_file_sha256s": report_hashes,
        }
        (staging / "baseline_manifest.yaml").write_text(
            yaml.safe_dump(baseline_manifest, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(staging, destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
