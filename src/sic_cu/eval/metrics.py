from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _basic_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - target
    mse = float(np.mean(error**2))
    denominator = float(np.sum((target - target.mean()) ** 2))
    relative_denominator = float(np.linalg.norm(target))
    return {
        "rmse_c": mse**0.5,
        "mae_c": float(np.mean(np.abs(error))),
        "mean_error_c": float(np.mean(error)),
        "max_abs_error_c": float(np.max(np.abs(error))),
        "r2": 1.0 - float(np.sum(error**2)) / denominator if denominator > 0 else float("nan"),
        "relative_l2": float(np.linalg.norm(error)) / relative_denominator
        if relative_denominator > 0
        else float("nan"),
    }


def field_metrics(
    target_k: np.ndarray,
    prediction_k: np.ndarray,
    times_s: np.ndarray,
    material_ids: np.ndarray,
    coordinates_rz_m: np.ndarray | None = None,
) -> dict[str, Any]:
    target = np.asarray(target_k, dtype=np.float64)
    prediction = np.asarray(prediction_k, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    materials = np.asarray(material_ids)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target and prediction must have shape [time, node]")
    if times.shape != (target.shape[0],) or materials.shape != (target.shape[1],):
        raise ValueError("time/material dimensions do not match field")
    result: dict[str, Any] = {"full_field": _basic_metrics(target, prediction)}
    for name, mask in (("copper", materials == 0), ("silicon_carbide", materials == 1)):
        result[name] = _basic_metrics(target[:, mask], prediction[:, mask])
    for name, lower, upper in (
        ("time_0_30_s", 0.0, 30.0),
        ("time_30_100_s", 30.0, 100.0),
        ("time_100_200_s", 100.0, np.inf),
    ):
        mask = (times >= lower) & (times <= upper if np.isfinite(upper) else True)
        result[name] = _basic_metrics(target[mask], prediction[mask])
    target_max = target.max(axis=1)
    prediction_max = prediction.max(axis=1)
    result["tmax"] = {
        "mae_c": float(np.mean(np.abs(prediction_max - target_max))),
        "max_abs_error_c": float(np.max(np.abs(prediction_max - target_max))),
        "final_error_c": float(prediction_max[-1] - target_max[-1]),
    }
    if coordinates_rz_m is not None:
        coordinates = np.asarray(coordinates_rz_m, dtype=np.float64)
        target_index = target.argmax(axis=1)
        prediction_index = prediction.argmax(axis=1)
        distance = np.linalg.norm(
            coordinates[target_index] - coordinates[prediction_index], axis=1
        )
        result["tmax"]["location_mae_m"] = float(distance.mean())
        result["tmax"]["location_max_error_m"] = float(distance.max())
    return result


def curve_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    result = _basic_metrics(np.asarray(target), np.asarray(prediction))
    target_centered = np.asarray(target) - np.mean(target)
    prediction_centered = np.asarray(prediction) - np.mean(prediction)
    denominator = np.linalg.norm(target_centered) * np.linalg.norm(prediction_centered)
    result["correlation"] = (
        float(np.dot(target_centered, prediction_centered) / denominator)
        if denominator > 0
        else float("nan")
    )
    result["max_rise_error_c"] = float(
        (np.max(prediction) - prediction[0]) - (np.max(target) - target[0])
    )
    return result


def weighted_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction_values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    weight_values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not (
        target_values.shape == prediction_values.shape == weight_values.shape
        and len(target_values) > 0
    ):
        raise ValueError("target, prediction, and weights must be aligned nonempty arrays")
    if not np.isfinite(weight_values).all() or np.any(weight_values < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    denominator = float(weight_values.sum())
    if denominator <= 0.0:
        raise ValueError("weights must have a positive sum")
    error = prediction_values - target_values
    return {
        "rmse_c": float(np.sqrt(np.sum(weight_values * error**2) / denominator)),
        "mae_c": float(np.sum(weight_values * np.abs(error)) / denominator),
        "mean_error_c": float(np.sum(weight_values * error) / denominator),
        "max_abs_error_c": float(np.max(np.abs(error))),
    }


def macro_metric_summary(
    records: Iterable[dict[str, Any]],
    metric_names: Iterable[str] = ("rmse_c", "mae_c", "mean_error_c"),
) -> dict[str, float]:
    values = list(records)
    if not values:
        raise ValueError("At least one per-condition metric record is required")
    summary = {
        name: float(np.mean([float(record[name]) for record in values]))
        for name in metric_names
    }
    if all("max_abs_error_c" in record for record in values):
        summary["max_abs_error_c"] = float(
            max(float(record["max_abs_error_c"]) for record in values)
        )
    if all("rmse_c" in record for record in values):
        summary["equal_power_mse_rmse_c"] = float(
            np.sqrt(np.mean([float(record["rmse_c"]) ** 2 for record in values]))
        )
    return summary


def macro_v1_selection_score(
    ir_macro_rmse_c: float,
    ring_absolute_macro_rmse_c: float,
    ring_delta_macro_rmse_c: float,
) -> float:
    values = np.asarray(
        [ir_macro_rmse_c, ring_absolute_macro_rmse_c, ring_delta_macro_rmse_c],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("macro_v1 selection inputs must be finite")
    return float((values[0] + 0.2 * values[1] + values[2]) / 2.2)


def aggregate_field_records(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not records:
        raise ValueError("At least one field-metric record is required")
    names = ("rmse_c", "mae_c", "r2", "relative_l2")
    return {
        name: {
            "mean": float(
                np.mean([record["metrics"]["full_field"][name] for record in records])
            ),
            "std": float(
                np.std([record["metrics"]["full_field"][name] for record in records])
            ),
        }
        for name in names
    }
