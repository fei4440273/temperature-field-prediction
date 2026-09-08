from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.splits import build_power_splits
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
    for index, time_s in enumerate(times):
        observed = frame.filter(pl.col("time_s") == float(time_s)).sort("r_m")
        surface_r, surface_temperature = _surface_profile(prediction, index)
        predicted = np.interp(observed["r_m"].to_numpy(), surface_r, surface_temperature)
        target = observed["temperature_mean_k"].to_numpy()
        weights = observed["frame_weight"].to_numpy()
        error = predicted - target
        frame_rmse = float(np.sqrt(np.sum(weights * error**2) / np.sum(weights)))
        frame_mae = float(np.sum(weights * np.abs(error)) / np.sum(weights))
        peak_error = float(predicted.max() - target.max())
        peak_temperature_c = max(float(target.max() - 273.15), np.finfo(float).eps)
        peak_relative_error = abs(peak_error) / peak_temperature_c * 100.0
        weighted_sse += float(np.sum(weights * error**2))
        weighted_absolute += float(np.sum(weights * np.abs(error)))
        weight_sum += float(np.sum(weights))
        peak_errors.append(peak_error)
        peak_relative_errors.append(peak_relative_error)
        frame_records.append(
            {
                "time_s": float(time_s),
                "rmse_k": frame_rmse,
                "mae_k": frame_mae,
                "peak_error_k": peak_error,
                "peak_relative_error_percent": peak_relative_error,
            }
        )
    return {
        "power_w": power,
        "radial_rmse_k": float(np.sqrt(weighted_sse / weight_sum)),
        "radial_mae_k": weighted_absolute / weight_sum,
        "peak_mae_k": float(np.mean(np.abs(peak_errors))),
        "peak_max_abs_error_k": float(np.max(np.abs(peak_errors))),
        "peak_mean_relative_error_percent": float(np.mean(peak_relative_errors)),
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
) -> dict[str, Any]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    splits = build_power_splits()
    expected = {
        "train": splits.hf_train,
        "validation": splits.hf_validation,
        "test": splits.hf_test,
    }[split]
    frame = pl.read_parquet(
        PROJECT_ROOT / "data/processed/experiment_ir_radial.parquet"
    ).filter(pl.col("split") == split)
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
        "split": split,
        "checkpoint": checkpoint,
        "powers_w": sorted(expected),
        "aggregate": {
            name: {
                "mean": float(np.mean([record[name] for record in records])),
                "std": float(np.std([record[name] for record in records])),
            }
            for name in (
                "radial_rmse_k",
                "radial_mae_k",
                "peak_mae_k",
                "peak_max_abs_error_k",
                "peak_mean_relative_error_percent",
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
