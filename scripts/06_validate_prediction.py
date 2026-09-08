#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _vtk_point_count(path: Path) -> int:
    with path.open(encoding="ascii") as handle:
        for line in handle:
            if line.startswith("POINTS "):
                return int(line.split()[1])
    raise ValueError(f"Missing POINTS declaration in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an exported temperature prediction")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initial-temperature-c", type=float, default=22.0)
    parser.add_argument("--cooling-radius-m", type=float, default=0.05834)
    parser.add_argument("--cooling-radius-tolerance-m", type=float, default=1e-7)
    parser.add_argument("--temperature-tolerance-k", type=float, default=1e-5)
    parser.add_argument("--theta-resolution", type=int, default=72)
    args = parser.parse_args()

    field_path = args.input / "field_rzt.npz"
    metadata_path = args.input / "metadata.json"
    hot_cold_path = args.input / "hot_cold.csv"
    with np.load(field_path) as field:
        times = field["times_s"]
        coordinates = field["coordinates_rz_m"]
        materials = field["material_ids"]
        temperature = field["mean_temperature_k"]
        q05 = field["q05_temperature_k"]
        q95 = field["q95_temperature_k"]
        maximum = field["max_temperature_k"]

    expected_k = args.initial_temperature_c + 273.15
    outer_copper = (materials == 0) & np.isclose(
        coordinates[:, 0],
        args.cooling_radius_m,
        atol=args.cooling_radius_tolerance_m,
        rtol=0.0,
    )
    vtk_paths = sorted((args.input / "vtk").glob("*.vtk"))
    vtk_counts = {path.name: _vtk_point_count(path) for path in vtk_paths}
    with hot_cold_path.open(encoding="utf-8", newline="") as handle:
        hot_cold_rows = sum(1 for _ in csv.reader(handle)) - 1

    checks = {
        "shape_consistent": bool(
            temperature.shape == (len(times), len(coordinates))
            and materials.shape == (len(coordinates),)
            and q05.shape == temperature.shape
            and q95.shape == temperature.shape
            and maximum.shape == (len(times),)
        ),
        "all_finite": bool(np.isfinite(temperature).all()),
        "strictly_increasing_times": bool(np.all(np.diff(times) > 0)),
        "initial_condition": bool(
            np.max(np.abs(temperature[0] - expected_k)) <= args.temperature_tolerance_k
        ),
        "temperature_floor": bool(
            temperature.min() >= expected_k - args.temperature_tolerance_k
        ),
        "cooling_boundary_found": bool(outer_copper.any()),
        "cooling_boundary": bool(
            outer_copper.any()
            and np.max(np.abs(temperature[:, outer_copper] - expected_k))
            <= args.temperature_tolerance_k
        ),
        "quantiles_ordered": bool(np.all(q05 <= temperature) and np.all(temperature <= q95)),
        "maximum_curve_consistent": bool(
            np.allclose(maximum, temperature.max(axis=1), atol=args.temperature_tolerance_k)
        ),
        "hot_cold_row_count": bool(hot_cold_rows == len(times)),
        "vtk_point_counts": bool(
            vtk_paths
            and all(
                count == len(coordinates) * args.theta_resolution
                for count in vtk_counts.values()
            )
        ),
    }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "input_directory": str(args.input),
        "passed": bool(all(checks.values())),
        "checks": checks,
        "summary": {
            "power_w": float(metadata["power_w"]),
            "time_count": int(len(times)),
            "node_count": int(len(coordinates)),
            "temperature_c_min": float(temperature.min() - 273.15),
            "temperature_c_max": float(temperature.max() - 273.15),
            "initial_max_abs_error_k": float(np.max(np.abs(temperature[0] - expected_k))),
            "cooling_node_count": int(outer_copper.sum()),
            "cooling_max_abs_error_k": float(
                np.max(np.abs(temperature[:, outer_copper] - expected_k))
            ),
            "last_tmax_change_k": float(abs(maximum[-1] - maximum[-2])),
            "stable_time_s": metadata["stable_time_s"],
            "stable_time_display": (
                f">{times[-1]:g} s" if metadata["stable_time_s"] is None else metadata["stable_time_s"]
            ),
            "vtk_point_counts": vtk_counts,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
