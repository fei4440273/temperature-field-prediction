from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl

from sic_cu.config import PROJECT_ROOT


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
                for name in ("mean_error_k", "mae_k", "rmse_k", "max_abs_error_k")
            },
            "hot": {
                name: float(hot[name])
                for name in ("mean_error_k", "mae_k", "rmse_k", "max_abs_error_k")
            },
            "cold": {
                name: float(cold[name])
                for name in ("mean_error_k", "mae_k", "rmse_k", "max_abs_error_k")
            },
        }
        values = list(modalities.values())
        combined = {
            "mean_error_k": float(np.mean([item["mean_error_k"] for item in values])),
            "mae_k": float(np.mean([item["mae_k"] for item in values])),
            "rmse_k": float(np.sqrt(np.mean([item["rmse_k"] ** 2 for item in values]))),
            "max_abs_error_k": float(max(item["max_abs_error_k"] for item in values)),
        }
        per_power.append(
            {"power_w": power, "modalities": modalities, "combined": combined}
        )

    aggregate = {
        "mean_error_k": float(
            np.mean([item["combined"]["mean_error_k"] for item in per_power])
        ),
        "mae_k": float(np.mean([item["combined"]["mae_k"] for item in per_power])),
        "rmse_k": float(
            np.sqrt(np.mean([item["combined"]["rmse_k"] ** 2 for item in per_power]))
        ),
        "max_abs_error_k": float(
            max(item["combined"]["max_abs_error_k"] for item in per_power)
        ),
    }
    return {
        "schema_version": 1,
        "split": split,
        "powers_w": expected,
        "used_for_gradient_updates": False,
        "used_for_model_selection": split == "validation",
        "weighting": "equal top_surface/hot/cold weighting within power; equal power weighting",
        "per_power": per_power,
        "aggregate": aggregate,
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
    metric_names = ("mean_error_k", "mae_k", "rmse_k", "max_abs_error_k")
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
