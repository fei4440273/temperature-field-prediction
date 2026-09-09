from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sic_cu.config import PROJECT_ROOT
from sic_cu.train.surface_residual import recompute_surface_run_metrics


def _sample_statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    result = {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }
    if len(array) == 5:
        result["ci95_half_width"] = 2.776 * result["std"] / np.sqrt(5)
    return result


def aggregate_simulation_runs(
    run_directories: list[str],
    output_path: str = "reports/simulation_model_5seed_summary.json",
) -> dict[str, Any]:
    if not run_directories:
        raise ValueError("At least one run directory is required")
    records: list[dict[str, Any]] = []
    for directory in run_directories:
        path = PROJECT_ROOT / directory / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        records.append(json.loads(path.read_text(encoding="utf-8")))
    methods = {str(record["method"]) for record in records}
    if len(methods) != 1:
        raise ValueError(f"Runs use different methods: {sorted(methods)}")
    seeds = [int(record["seed"]) for record in records]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seeds cannot be aggregated")
    test_power_sets = [
        tuple(float(item["power_w"]) for item in record["test"]["per_power"])
        for record in records
    ]
    if any(powers != test_power_sets[0] for powers in test_power_sets[1:]):
        raise ValueError("Runs do not use the same ordered simulation test powers")

    field_metric_names = ("rmse_c", "mae_c", "r2", "relative_l2")
    aggregate = {
        name: _sample_statistics(
            [float(record["test"]["aggregate"][name]["mean"]) for record in records]
        )
        for name in field_metric_names
    }
    aggregate["tmax_mae_c"] = _sample_statistics(
        [
            float(
                np.mean(
                    [item["metrics"]["tmax"]["mae_c"] for item in record["test"]["per_power"]]
                )
            )
            for record in records
        ]
    )
    aggregate["training_seconds"] = _sample_statistics(
        [float(record["training_seconds"]) for record in records]
    )
    aggregate["total_test_inference_seconds"] = _sample_statistics(
        [
            float(sum(item["inference_seconds"] for item in record["test"]["per_power"]))
            for record in records
        ]
    )

    per_power = []
    for index, power in enumerate(test_power_sets[0]):
        power_records = [record["test"]["per_power"][index] for record in records]
        per_power.append(
            {
                "power_w": power,
                "full_field": {
                    name: _sample_statistics(
                        [float(item["metrics"]["full_field"][name]) for item in power_records]
                    )
                    for name in field_metric_names
                },
                "tmax_mae_c": _sample_statistics(
                    [float(item["metrics"]["tmax"]["mae_c"]) for item in power_records]
                ),
            }
        )
    result = {
        "schema_version": 1,
        "method": next(iter(methods)),
        "scope": "low-fidelity simulation frozen-power test; no experiment claim",
        "seeds": sorted(seeds),
        "run_count": len(records),
        "test_powers_w": list(test_power_sets[0]),
        "aggregate_across_seeds": aggregate,
        "per_power_across_seeds": per_power,
        "per_seed": [
            {
                "seed": record["seed"],
                "best_epoch": record["best_epoch"],
                "best_validation": {
                    key: value
                    for key, value in record.items()
                    if key.startswith("best_validation_")
                },
                "training_seconds": record["training_seconds"],
                "test": record["test"]["aggregate"],
            }
            for record in sorted(records, key=lambda item: int(item["seed"]))
        ],
        "material_passport": records[0]["material_passport"],
    }
    destination = PROJECT_ROOT / output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def aggregate_interface_runs(
    result_paths: list[str],
    output_path: str,
) -> dict[str, Any]:
    if not result_paths:
        raise ValueError("At least one interface result is required")
    records = [
        json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))
        for path in result_paths
    ]
    signatures = [
        (record["method"], record["material_aware"], tuple(record["test_powers_w"]))
        for record in records
    ]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Interface results do not share method, material mode, and test powers")
    seeds = [int(record["seed"]) for record in records]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate interface-evaluation seeds cannot be aggregated")
    metric_names = (
        "mae_c",
        "rmse_c",
        "max_abs_error_c",
        "target_mean_abs_jump_c",
        "prediction_mean_abs_jump_c",
    )
    result = {
        "schema_version": 1,
        "method": signatures[0][0],
        "material_aware": signatures[0][1],
        "scope": "locked simulation test powers, same-coordinate material-side pairs",
        "test_powers_w": list(signatures[0][2]),
        "seeds": sorted(seeds),
        "run_count": len(records),
        "aggregate_across_seeds": {
            name: _sample_statistics(
                [float(record["aggregate"][name]) for record in records]
            )
            for name in metric_names
        },
        "per_seed": [
            {
                "seed": record["seed"],
                "checkpoint": record["checkpoint"],
                "aggregate": record["aggregate"],
            }
            for record in sorted(records, key=lambda item: int(item["seed"]))
        ],
        "interpretation": records[0]["interpretation"],
        "material_passport": records[0]["material_passport"],
    }
    destination = PROJECT_ROOT / output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def aggregate_ir_pixel_runs(
    result_paths: list[str],
    output_path: str = "reports/ir_pixel_surface_residual_5seed_summary.json",
) -> dict[str, Any]:
    if not result_paths:
        raise ValueError("At least one IR pixel result is required")
    records = [
        json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8")) for path in result_paths
    ]
    seeds = [record.get("seed") for record in records]
    if any(seed is None for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("IR pixel results require unique non-null seeds")
    signatures = [
        (record["split"], tuple(record["powers_w"]), record["aggregation"])
        for record in records
    ]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("IR pixel results do not share the same split, powers, and aggregation")
    metric_names = (
        "pixel_rmse_c",
        "pixel_mae_c",
        "axisymmetric_floor_rmse_c",
        "radial_profile_rmse_c",
        "peak_mae_c",
    )
    aggregate = {
        name: _sample_statistics(
            [float(record["aggregate"][name]["mean"]) for record in records]
        )
        for name in metric_names
    }
    per_power = []
    for index, power in enumerate(signatures[0][1]):
        items = [record["per_power"][index] for record in records]
        if any(float(item["power_w"]) != float(power) for item in items):
            raise ValueError("IR pixel per-power order is inconsistent")
        per_power.append(
            {
                "power_w": float(power),
                **{
                    name: _sample_statistics([float(item[name]) for item in items])
                    for name in metric_names
                },
            }
        )
    result = {
        "schema_version": 1,
        "method": "deterministic_multifidelity_surface_residual",
        "scope": "raw SiC top-surface pixels only; no internal-field claim",
        "split": signatures[0][0],
        "powers_w": list(signatures[0][1]),
        "seeds": sorted(int(seed) for seed in seeds),
        "run_count": len(records),
        "aggregation": signatures[0][2],
        "aggregate_across_seeds": aggregate,
        "per_power_across_seeds": per_power,
        "per_seed": [
            {
                "seed": record["seed"],
                "surface_checkpoint": record["surface_checkpoint"],
                "aggregate": record["aggregate"],
            }
            for record in sorted(records, key=lambda item: int(item["seed"]))
        ],
        "material_passport": records[0]["material_passport"],
    }
    destination = PROJECT_ROOT / output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def aggregate_surface_residual_runs(
    run_directories: list[str],
    output_path: str = "reports/surface_residual_5seed_summary.json",
    release_manifest_path: str | None = None,
) -> dict[str, Any]:
    if release_manifest_path is None:
        raise ValueError("A frozen release manifest is required for test aggregation")
    records = []
    for directory in run_directories:
        path = PROJECT_ROOT / directory / "metrics.json"
        result = recompute_surface_run_metrics(directory, release_manifest_path)
        if result["method"] != "deterministic_multifidelity_surface_residual":
            raise ValueError(f"Unexpected method in {path}")
        records.append(result)
    seeds = [int(record["seed"]) for record in records]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seeds cannot be aggregated")
    metric_names = (
        "aggregate_rmse_c",
        "aggregate_mae_c",
        "aggregate_peak_mae_c",
        "aggregate_peak_relative_error_percent",
        "aggregate_radial_gradient_mae_c_per_mm",
    )
    aggregate = {
        name: {
            "mean": float(np.mean([record["test"][name] for record in records])),
            "std": float(np.std([record["test"][name] for record in records], ddof=1))
            if len(records) > 1
            else 0.0,
        }
        for name in metric_names
    }
    if len(records) > 1:
        for value in aggregate.values():
            value["ci95_half_width"] = 2.776 * value["std"] / np.sqrt(len(records))
    powers = sorted(
        {item["power_w"] for record in records for item in record["test"]["per_power"]}
    )
    per_power = []
    for power in powers:
        power_records = [
            next(item for item in record["test"]["per_power"] if item["power_w"] == power)
            for record in records
        ]
        power_summary: dict[str, Any] = {"power_w": power}
        power_summary.update(
            {
                name: {
                    "mean": float(np.mean([item[name] for item in power_records])),
                    "std": float(np.std([item[name] for item in power_records], ddof=1)),
                }
                for name in ("rmse_c", "mae_c", "peak_mean_relative_error_percent")
            }
        )
        per_power.append(power_summary)
    result = {
        "schema_version": 1,
        "method": "deterministic_multifidelity_surface_residual",
        "scope": "SiC top surface only; no internal-field claim",
        "seeds": sorted(seeds),
        "run_count": len(records),
        "aggregate": aggregate,
        "per_seed": [
            {"seed": record["seed"], "test": record["test"]} for record in records
        ],
        "per_power_across_seeds": per_power,
        "acceptance": {
            "surface_mae_le_5c": aggregate["aggregate_mae_c"]["mean"] <= 5.0,
            "every_test_power_mean_mae_le_5c": all(
                item["mae_c"]["mean"] <= 5.0 for item in per_power
            ),
            "mean_peak_relative_error_le_5pct": aggregate[
                "aggregate_peak_relative_error_percent"
            ]["mean"]
            <= 5.0,
            "internal_field_acceptance_evaluable": False,
        },
        "material_passport": records[0]["material_passport"],
    }
    destination = PROJECT_ROOT / output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def aggregate_multifidelity_runs(
    run_directories: list[str],
    ir_evaluation_paths: list[str],
    output_path: str = "reports/mf_pinn_nominal_5seed_summary.json",
) -> dict[str, Any]:
    if len(run_directories) != len(ir_evaluation_paths) or not run_directories:
        raise ValueError("Run directories and IR evaluations must be nonempty and aligned")
    records = []
    ir_records = []
    for directory, evaluation_path in zip(
        run_directories, ir_evaluation_paths, strict=True
    ):
        record = json.loads(
            (PROJECT_ROOT / directory / "metrics.json").read_text(encoding="utf-8")
        )
        evaluation = json.loads(
            (PROJECT_ROOT / evaluation_path).read_text(encoding="utf-8")
        )
        expected_checkpoint = str(PROJECT_ROOT / directory / "best.pt")
        actual_checkpoint = str(Path(evaluation["checkpoint"]).resolve())
        if actual_checkpoint != str(Path(expected_checkpoint).resolve()):
            raise ValueError("IR evaluation/checkpoint alignment mismatch")
        if record["test_ir"] is None or record["test_sensor"] is None:
            raise ValueError("Formal multi-fidelity runs must contain locked test results")
        if record["configuration"].get("physics_parameters_frozen") is not True:
            raise ValueError("Formal runs must use the locked physics scenario")
        records.append(record)
        ir_records.append(evaluation)
    seeds = [int(record["seed"]) for record in records]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seeds cannot be aggregated")
    configurations = [
        (
            record["configuration"]["silicon_carbide_emissivity_initial"],
            record["configuration"]["copper_emissivity_initial"],
            record["configuration"]["contact_resistance_initial_m2_k_w"],
        )
        for record in records
    ]
    if any(configuration != configurations[0] for configuration in configurations[1:]):
        raise ValueError("Formal runs do not share the locked physics scenario")

    aggregate = {
        "best_validation_selection_score_c": _sample_statistics(
            [
                float(
                    record.get(
                        "best_validation_selection_score_c",
                        record["best_validation_ir_rmse_c"],
                    )
                )
                for record in records
            ]
        ),
        "best_validation_ir_rmse_c": _sample_statistics(
            [float(record["best_validation_ir_rmse_c"]) for record in records]
        ),
        "test_ir_rmse_c": _sample_statistics(
            [float(record["test_ir"]["rmse_c"]) for record in records]
        ),
        "test_ir_mae_c": _sample_statistics(
            [float(record["test_ir"]["mae_c"]) for record in records]
        ),
        "test_peak_mae_c": _sample_statistics(
            [float(record["test_ir"]["peak_mae_c"]) for record in records]
        ),
        "test_peak_relative_error_percent": _sample_statistics(
            [
                float(evaluation["aggregate"]["peak_mean_relative_error_percent"]["mean"])
                for evaluation in ir_records
            ]
        ),
        "test_sensor_absolute_rmse_c": _sample_statistics(
            [float(record["test_sensor"]["absolute_rmse_c"]) for record in records]
        ),
        "test_sensor_delta_rmse_c": _sample_statistics(
            [float(record["test_sensor"]["delta_rmse_c"]) for record in records]
        ),
        "training_seconds": _sample_statistics(
            [float(record["training_seconds"]) for record in records]
        ),
    }
    test_powers = [
        float(item["power_w"]) for item in records[0]["test_ir"]["per_power"]
    ]
    per_power = []
    for index, power in enumerate(test_powers):
        items = [record["test_ir"]["per_power"][index] for record in records]
        evaluations = [evaluation["per_power"][index] for evaluation in ir_records]
        if any(float(item["power_w"]) != power for item in items):
            raise ValueError("Test-power order mismatch")
        per_power.append(
            {
                "power_w": power,
                "rmse_c": _sample_statistics([float(item["rmse_c"]) for item in items]),
                "mae_c": _sample_statistics([float(item["mae_c"]) for item in items]),
                "peak_mae_c": _sample_statistics(
                    [float(item["peak_mae_c"]) for item in items]
                ),
                "peak_relative_error_percent": _sample_statistics(
                    [
                        float(item["peak_mean_relative_error_percent"])
                        for item in evaluations
                    ]
                ),
            }
        )
    result = {
        "schema_version": 1,
        "method": "deterministic_multifidelity_deeponet_pinn_correction",
        "scope": (
            "SiC top-surface experiment test and Cu ring sensor test; internal field is "
            "physics-guided inference without internal experiment truth"
        ),
        "seeds": sorted(seeds),
        "run_count": len(records),
        "test_powers_w": test_powers,
        "locked_physics_scenario": {
            "silicon_carbide_emissivity": configurations[0][0],
            "copper_emissivity": configurations[0][1],
            "contact_resistance_m2_k_w": configurations[0][2],
            "emissivity_interpretation": "neutral sensitivity values, not identified truth",
        },
        "aggregate_across_seeds": aggregate,
        "per_power_across_seeds": per_power,
        "per_seed": [
            {
                "seed": int(record["seed"]),
                "best_epoch": int(record["best_epoch"]),
                "best_validation_ir_rmse_c": float(
                    record["best_validation_ir_rmse_c"]
                ),
                "best_validation_selection_score_c": float(
                    record.get(
                        "best_validation_selection_score_c",
                        record["best_validation_ir_rmse_c"],
                    )
                ),
                "best_validation_sensor": record.get("best_validation_sensor"),
                "test_ir": record["test_ir"],
                "test_sensor": {
                    "absolute_rmse_c": record["test_sensor"]["absolute_rmse_c"],
                    "delta_rmse_c": record["test_sensor"]["delta_rmse_c"],
                },
            }
            for record in sorted(records, key=lambda item: int(item["seed"]))
        ],
        "acceptance": {
            "mean_surface_mae_le_5c": aggregate["test_ir_mae_c"]["mean"] <= 5.0,
            "every_test_power_mean_mae_le_5c": all(
                item["mae_c"]["mean"] <= 5.0 for item in per_power
            ),
            "mean_peak_relative_error_le_5pct": aggregate[
                "test_peak_relative_error_percent"
            ]["mean"]
            <= 5.0,
            "internal_field_acceptance_evaluable": False,
        },
        "material_passport": records[0]["material_passport"],
    }
    destination = PROJECT_ROOT / output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def aggregate_external_sensor_runs(
    result_paths: list[str],
    output_path: str = "reports/mf_pinn_nominal_external_sensor_5seed_summary.json",
) -> dict[str, Any]:
    if not result_paths:
        raise ValueError("At least one external-sensor result is required")
    records = [
        json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))
        for path in result_paths
    ]
    checkpoints = [str(record["checkpoint"]) for record in records]
    seeds = []
    for checkpoint in checkpoints:
        stem = Path(checkpoint).parent.name
        if "seed" not in stem:
            raise ValueError(f"Cannot recover seed from checkpoint path: {checkpoint}")
        seeds.append(int(stem.rsplit("seed", 1)[1]))
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate external-sensor seeds cannot be aggregated")
    signatures = [
        (
            tuple(float(value) for value in record["powers_w"]),
            tuple(
                (float(item["power_w"]), str(item["sensor_type"]))
                for item in record["per_curve"]
            ),
        )
        for record in records
    ]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("External-sensor results do not use the same frozen curves")
    per_curve = []
    for index, (power, sensor_type) in enumerate(signatures[0][1]):
        items = [record["per_curve"][index] for record in records]
        per_curve.append(
            {
                "power_w": power,
                "sensor_type": sensor_type,
                "absolute_rmse_c": _sample_statistics(
                    [float(item["absolute"]["rmse_c"]) for item in items]
                ),
                "delta_rmse_c": _sample_statistics(
                    [float(item["delta"]["rmse_c"]) for item in items]
                ),
                "absolute_mae_c": _sample_statistics(
                    [float(item["absolute"]["mae_c"]) for item in items]
                ),
                "delta_mae_c": _sample_statistics(
                    [float(item["delta"]["mae_c"]) for item in items]
                ),
            }
        )
    result = {
        "schema_version": 1,
        "method": "deterministic_multifidelity_deeponet_pinn_correction",
        "scope": "frozen external powers; two Cu bottom-ring observations",
        "seeds": sorted(seeds),
        "run_count": len(records),
        "powers_w": list(signatures[0][0]),
        "aggregate_across_seeds": {
            "absolute_rmse_c": _sample_statistics(
                [float(record["aggregate"]["absolute_rmse_c"]) for record in records]
            ),
            "delta_rmse_c": _sample_statistics(
                [float(record["aggregate"]["delta_rmse_c"]) for record in records]
            ),
        },
        "per_curve_across_seeds": per_curve,
        "per_seed": [
            {
                "seed": seed,
                "checkpoint": record["checkpoint"],
                "aggregate": record["aggregate"],
            }
            for seed, record in sorted(zip(seeds, records, strict=True))
        ],
        "interpretation": (
            "Absolute error includes an unresolved copper-field offset. Delta metrics test "
            "heating-curve shape. These ring measurements do not constitute internal-field truth."
        ),
        "material_passport": records[0]["material_passport"],
    }
    destination = PROJECT_ROOT / output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
