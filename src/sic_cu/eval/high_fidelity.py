from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.splits import build_power_splits
from sic_cu.data.processed import load_processed_ir_observations
from sic_cu.eval.metrics import weighted_metrics
from sic_cu.eval.protocol_checks import validate_release_checkpoint
from sic_cu.prediction import Predictor


def _surface_profile(prediction, time_index: int) -> tuple[np.ndarray, np.ndarray]:
    coordinates = prediction.coordinates_rz_m
    mask = (prediction.material_ids == 1) & np.isclose(coordinates[:, 1], 0.0, atol=1e-8)
    order = np.argsort(coordinates[mask, 0])
    return (
        coordinates[mask, 0][order],
        prediction.mean_temperature_k[time_index, mask][order],
    )


def _power_metrics(frame: pl.DataFrame, predictor: Predictor) -> dict[str, Any]:
    power = float(frame["power_w"][0])
    times = np.sort(frame["time_s"].unique().to_numpy())
    prediction = predictor.predict(power, times_s=times)
    weighted_sse = 0.0
    weighted_absolute = 0.0
    weight_sum = 0.0
    peak_errors = []
    peak_relative_errors = []
    frame_records = []
    all_errors: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    all_times: list[np.ndarray] = []
    direct_grid_differences: list[np.ndarray] = []
    first_observed_peak_k: float | None = None
    for index, time_s in enumerate(times):
        observed = frame.filter(pl.col("time_s") == float(time_s)).sort("r_m")
        surface_r, surface_temperature = _surface_profile(prediction, index)
        grid_interpolated = np.interp(
            observed["r_m"].to_numpy(), surface_r, surface_temperature
        )
        if predictor.supports_point_queries:
            coordinates = np.column_stack(
                (
                    observed["r_m"].to_numpy(),
                    np.zeros(observed.height),
                    np.full(observed.height, time_s),
                    np.full(observed.height, power),
                    np.ones(observed.height),
                )
            ).astype(np.float32)
            predicted = predictor.predict_points(coordinates).reshape(-1)
            direct_grid_differences.append(predicted - grid_interpolated)
        else:
            predicted = grid_interpolated
        target = observed["temperature_mean_k"].to_numpy()
        weights = observed["frame_weight"].to_numpy()
        error = predicted - target
        frame_rmse = float(np.sqrt(np.sum(weights * error**2) / np.sum(weights)))
        frame_mae = float(np.sum(weights * np.abs(error)) / np.sum(weights))
        peak_error = float(predicted.max() - target.max())
        if first_observed_peak_k is None:
            first_observed_peak_k = float(target.max())
        peak_rise_k = float(target.max()) - first_observed_peak_k
        peak_temperature_c = max(float(target.max() - 273.15), np.finfo(float).eps)
        peak_relative_error = abs(peak_error) / peak_temperature_c * 100.0
        peak_rise_relative_error = (
            abs(peak_error) / peak_rise_k * 100.0 if peak_rise_k >= 1.0 else None
        )
        weighted_sse += float(np.sum(weights * error**2))
        weighted_absolute += float(np.sum(weights * np.abs(error)))
        weight_sum += float(np.sum(weights))
        peak_errors.append(peak_error)
        peak_relative_errors.append(peak_relative_error)
        all_errors.append(error)
        all_weights.append(weights)
        all_times.append(np.full(len(error), time_s))
        frame_records.append(
            {
                "time_s": float(time_s),
                "rmse_c": frame_rmse,
                "mae_c": frame_mae,
                "peak_error_c": peak_error,
                "peak_rise_relative_error_percent": peak_rise_relative_error,
                "legacy_peak_celsius_relative_error_percent": peak_relative_error,
            }
        )
    error_values = np.concatenate(all_errors)
    weight_values = np.concatenate(all_weights)
    time_values = np.concatenate(all_times)
    target_proxy = np.zeros_like(error_values)
    overall = weighted_metrics(target_proxy, error_values, weight_values)
    time_windows: dict[str, Any] = {}
    for name, lower, upper in (
        ("time_0_30_s", 0.0, 30.0),
        ("time_30_100_s", 30.0, 100.0),
        ("time_100_200_s", 100.0, 200.0 + np.finfo(float).eps),
    ):
        mask = (time_values >= lower) & (time_values < upper)
        time_windows[name] = (
            weighted_metrics(target_proxy[mask], error_values[mask], weight_values[mask])
            if bool(mask.any())
            else None
        )
    direct_grid = (
        np.concatenate(direct_grid_differences)
        if direct_grid_differences
        else np.empty(0, dtype=np.float64)
    )
    return {
        "power_w": power,
        "query_mode": "direct_coordinates" if predictor.supports_point_queries else "explicit_grid_interpolation",
        "radial_rmse_c": overall["rmse_c"],
        "radial_mae_c": overall["mae_c"],
        "mean_error_c": overall["mean_error_c"],
        "max_abs_error_c": overall["max_abs_error_c"],
        "peak_mae_c": float(np.mean(np.abs(peak_errors))),
        "peak_max_abs_error_c": float(np.max(np.abs(peak_errors))),
        "legacy_peak_celsius_relative_error_percent": float(np.mean(peak_relative_errors)),
        "direct_vs_grid": {
            "rmse_c": float(np.sqrt(np.mean(direct_grid**2))) if len(direct_grid) else None,
            "max_abs_error_c": float(np.max(np.abs(direct_grid))) if len(direct_grid) else None,
        },
        "time_windows": time_windows,
        "frames": frame_records,
        "prediction_metadata": {
            "source": prediction.metadata.source,
            "warnings": prediction.metadata.warnings,
        },
    }


def evaluate_ir_surface(
    checkpoint: str | None = None,
    split: str = "test",
    output_path: str = "reports/ir_surface_evaluation.json",
    device: str | None = None,
    release_manifest_path: str | None = None,
) -> dict[str, Any]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if split == "test":
        if checkpoint is None or release_manifest_path is None:
            raise RuntimeError(
                "Test IR evaluation requires a registered checkpoint and frozen release manifest"
            )
        validate_release_checkpoint(release_manifest_path, checkpoint)
    splits = build_power_splits()
    expected = {
        "train": splits.hf_train,
        "validation": splits.hf_validation,
        "test": splits.hf_test,
    }[split]
    frame = load_processed_ir_observations(split)
    observed = {round(float(value), 4) for value in frame["power_w"].unique()}
    if observed != set(expected):
        raise RuntimeError(f"IR evaluation split mismatch: observed={observed}, expected={expected}")
    predictor = Predictor(checkpoint=checkpoint, device=device)
    records = [
        _power_metrics(frame.filter(pl.col("power_w") == power), predictor)
        for power in sorted(frame["power_w"].unique().to_list())
    ]
    result = {
        "schema_version": 1,
        "temperature_error_unit": "℃",
        "split": split,
        "checkpoint": checkpoint,
        "powers_w": sorted(expected),
        "aggregate": {
            name: {
                "mean": float(np.mean([record[name] for record in records])),
                "std": float(np.std([record[name] for record in records])),
            }
            for name in (
                "radial_rmse_c",
                "radial_mae_c",
                "peak_mae_c",
                "peak_max_abs_error_c",
                "legacy_peak_celsius_relative_error_percent",
            )
        },
        "per_power": records,
        "material_passport": {
            "ir_truth_scope": "measured SiC top surface only",
            "internal_field_truth": "not available",
            "pixel_independence_claimed": False,
            "frame_weights_capped": True,
        },
    }
    destination = PROJECT_ROOT / output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
