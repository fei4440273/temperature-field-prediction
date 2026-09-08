from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.fields import load_processed_field
from sic_cu.data.splits import build_power_splits
from sic_cu.models.interpolation import interpolate_simulation_power

from .metrics import field_metrics


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["rmse_k", "mae_k", "r2", "relative_l2"]
    return {
        key: {
            "mean": float(np.mean([record["metrics"]["full_field"][key] for record in records])),
            "std": float(np.std([record["metrics"]["full_field"][key] for record in records])),
        }
        for key in keys
    }


def run_interpolation_benchmark(
    kind: str = "linear",
    output: str = "reports/interpolation_benchmark.json",
) -> dict[str, Any]:
    splits = build_power_splits()
    available = sorted(splits.simulation_train)
    records: list[dict[str, Any]] = []
    for power in sorted(splits.simulation_test):
        target = load_processed_field(power)
        start = time.perf_counter()
        prediction = interpolate_simulation_power(power, available, kind)
        elapsed = time.perf_counter() - start
        records.append(
            {
                "power_w": power,
                "method": f"fem_{kind}_power_interpolation",
                "source_powers_w": _source_powers(power, available, kind),
                "inference_seconds": elapsed,
                "metrics": field_metrics(
                    target.temperature_k,
                    prediction.temperature_k,
                    target.times_s,
                    target.material_ids,
                    target.coordinates_rz_m,
                ),
            }
        )
    result = {
        "method": f"fem_{kind}_power_interpolation",
        "training_powers_w": available,
        "test_powers_w": sorted(splits.simulation_test),
        "aggregate": _aggregate(records),
        "per_power": records,
        "material_passport": {
            "simulation_role": "low-fidelity reference and target",
            "experiment_data_used": False,
            "sensor_data_used": False,
            "physics_parameters_used": False,
            "internal_experiment_truth": "not available",
        },
    }
    path = PROJECT_ROOT / output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _source_powers(query: float, available: list[float], kind: str) -> list[float]:
    if kind == "linear":
        return [max(p for p in available if p < query), min(p for p in available if p > query)]
    return sorted(sorted(available, key=lambda p: abs(p - query))[:4])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FEM power interpolation")
    parser.add_argument("--kind", choices=("linear", "cubic"), default="linear")
    parser.add_argument("--output", default="reports/interpolation_benchmark.json")
    args = parser.parse_args()
    result = run_interpolation_benchmark(args.kind, args.output)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
