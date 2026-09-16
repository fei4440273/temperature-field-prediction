from __future__ import annotations

import json
from typing import Any

import numpy as np
import polars as pl

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.sensors import load_canonical_sensor_observations
from sic_cu.data.splits import build_power_splits
from sic_cu.eval.metrics import curve_metrics
from sic_cu.prediction import Predictor


def _bottom_curve(prediction, radius_m: float) -> np.ndarray:
    coordinates = prediction.coordinates_rz_m
    bottom_z = coordinates[prediction.material_ids == 0, 1].min()
    mask = (prediction.material_ids == 0) & np.isclose(
        coordinates[:, 1], bottom_z, atol=1e-8
    )
    order = np.argsort(coordinates[mask, 0])
    radius = coordinates[mask, 0][order]
    values = prediction.mean_temperature_k[:, mask][:, order]
    return np.asarray([np.interp(radius_m, radius, row) for row in values])


def evaluate_external_sensors(
    checkpoint: str | None = None,
    output_path: str = "reports/external_sensor_evaluation.json",
    device: str | None = None,
) -> dict[str, Any]:
    frame = load_canonical_sensor_observations().filter(pl.col("split") == "external_test")
    splits = build_power_splits()
    observed = {round(float(value), 4) for value in frame["power_w"].unique()}
    if observed != set(splits.external_sensor_test):
        raise RuntimeError("External sensor powers do not match the frozen protocol")
    predictor = Predictor(checkpoint=checkpoint, device=device)
    records = []
    for power in sorted(observed):
        power_frame = frame.filter(pl.col("power_w") == power)
        times = np.sort(power_frame["time_s"].unique().to_numpy())
        prediction = predictor.predict(power, times_s=times)
        for sensor_type in ("hot", "cold"):
            sensor = power_frame.filter(pl.col("sensor_type") == sensor_type).sort("time_s")
            radius = float(sensor["r_m"][0])
            predicted = _bottom_curve(prediction, radius)
            target = sensor["temperature_k"].to_numpy()
            records.append(
                {
                    "power_w": power,
                    "sensor_type": sensor_type,
                    "radius_m": radius,
                    "absolute": curve_metrics(target, predicted),
                    "delta": curve_metrics(target - target[0], predicted - predicted[0]),
                }
            )
    result = {
        "schema_version": 1,
        "temperature_error_unit": "℃",
        "checkpoint": checkpoint,
        "training_use_of_external_powers": False,
        "powers_w": sorted(observed),
        "per_curve": records,
        "aggregate": {
            "absolute_rmse_c": float(np.mean([item["absolute"]["rmse_c"] for item in records])),
            "delta_rmse_c": float(np.mean([item["delta"]["rmse_c"] for item in records])),
        },
        "material_passport": {
            "measurement_scope": "two Cu bottom rings",
            "angular_duplicates_collapsed": True,
            "internal_field_truth": "not available",
        },
    }
    destination = PROJECT_ROOT / output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
