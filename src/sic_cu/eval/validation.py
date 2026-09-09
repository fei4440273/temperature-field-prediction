from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl

from sic_cu.config import PROJECT_ROOT
from sic_cu.eval.metrics import macro_metric_summary, macro_v1_selection_score


def _power_key(value: float) -> float:
    return round(float(value), 4)


def build_three_power_comparison(
    ir_metrics: dict[str, Any],
    sensor_metrics: dict[str, Any],
    expected_powers_w: Iterable[float],
    split: str,
) -> dict[str, Any]:
    """Combine top/Hot/Cold validation errors with equal modality weighting."""
    if split not in {"validation", "test"}:
        raise ValueError("Three-power comparison split must be validation or test")
    expected = sorted({_power_key(value) for value in expected_powers_w})
    if len(expected) != 3:
        raise ValueError(f"Validation comparison requires exactly three powers, got {expected}")
    top_by_power = {
        _power_key(item["power_w"]): item for item in ir_metrics["per_power"]
    }
    sensor_by_key = {
        (_power_key(item["power_w"]), str(item["sensor_type"])): item
        for item in sensor_metrics["per_curve"]
    }
    if sorted(top_by_power) != expected:
        raise RuntimeError(
            f"{split.title()} IR powers mismatch: observed={sorted(top_by_power)}, expected={expected}"
        )

    per_power = []
    for power in expected:
        missing = [
            sensor_type
            for sensor_type in ("hot", "cold")
            if (power, sensor_type) not in sensor_by_key
        ]
        if missing:
            raise RuntimeError(f"Missing {split} sensors for {power:g} W: {missing}")
        top = top_by_power[power]
        hot = sensor_by_key[(power, "hot")]["absolute"]
        cold = sensor_by_key[(power, "cold")]["absolute"]
        modalities = {
            "top_surface": {
                name: float(top[name])
                for name in ("mean_error_c", "mae_c", "rmse_c", "max_abs_error_c")
            },
            "hot": {
                name: float(hot[name])
                for name in ("mean_error_c", "mae_c", "rmse_c", "max_abs_error_c")
            },
            "cold": {
                name: float(cold[name])
                for name in ("mean_error_c", "mae_c", "rmse_c", "max_abs_error_c")
            },
        }
        values = list(modalities.values())
        combined = {
            "mean_error_c": float(np.mean([item["mean_error_c"] for item in values])),
            "mae_c": float(np.mean([item["mae_c"] for item in values])),
            "rmse_c": float(np.mean([item["rmse_c"] for item in values])),
            "max_abs_error_c": float(max(item["max_abs_error_c"] for item in values)),
        }
        per_power.append(
            {"power_w": power, "modalities": modalities, "combined": combined}
        )

    aggregate = macro_metric_summary([item["combined"] for item in per_power])
    ir_macro_rmse = float(np.mean([top_by_power[power]["rmse_c"] for power in expected]))
    ring_absolute_macro_rmse = float(
        np.mean(
            [
                sensor_by_key[(power, sensor_type)]["absolute"]["rmse_c"]
                for power in expected
                for sensor_type in ("hot", "cold")
            ]
        )
    )
    delta_values = [
        sensor_by_key[(power, sensor_type)].get("delta", {}).get("rmse_c")
        for power in expected
        for sensor_type in ("hot", "cold")
    ]
    selection = None
    if split == "validation" and all(value is not None for value in delta_values):
        ring_delta_macro_rmse = float(np.mean(delta_values))
        selection = {
            "version": "macro_v1",
            "ir_macro_rmse_c": ir_macro_rmse,
            "ring_absolute_macro_rmse_c": ring_absolute_macro_rmse,
            "ring_delta_macro_rmse_c": ring_delta_macro_rmse,
            "score_c": macro_v1_selection_score(
                ir_macro_rmse,
                ring_absolute_macro_rmse,
                ring_delta_macro_rmse,
            ),
        }
    return {
        "schema_version": 1,
        "temperature_error_unit": "℃",
        "split": split,
        "powers_w": expected,
        "used_for_gradient_updates": False,
        "used_for_model_selection": split == "validation",
        "selection_metric_version": "macro_v1",
        "weighting": "arithmetic mean of per-condition metrics; equal top/hot/cold and power weighting",
        "per_power": per_power,
        "aggregate": aggregate,
        "selection": selection,
    }


def write_three_power_comparison(
    result: dict[str, Any],
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    json_destination = Path(json_path)
    csv_destination = Path(csv_path)
    if not json_destination.is_absolute():
        json_destination = PROJECT_ROOT / json_destination
    if not csv_destination.is_absolute():
        csv_destination = PROJECT_ROOT / csv_destination
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(json.dumps(result, indent=2), encoding="utf-8")

    rows = []
    metric_names = ("mean_error_c", "mae_c", "rmse_c", "max_abs_error_c")
    for item in result["per_power"]:
        row: dict[str, float] = {"power_w": float(item["power_w"])}
        for modality, metrics in item["modalities"].items():
            row.update(
                {f"{modality}_{name}": float(metrics[name]) for name in metric_names}
            )
        row.update(
            {f"combined_{name}": float(item["combined"][name]) for name in metric_names}
        )
        rows.append(row)
    pl.DataFrame(rows).write_csv(csv_destination)
