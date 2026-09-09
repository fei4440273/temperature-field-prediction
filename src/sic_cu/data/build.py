from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from sic_cu.config import PROJECT_ROOT, load_yaml, resolve_data_root
from sic_cu.eval.protocol_checks import canonical_json_sha256, current_code_commit

from .common import sha256_file
from .experiment import experiment_files, radial_observations
from .sensors import ring_average_raw, sensor_files
from .simulation import read_simulation, simulation_files
from .splits import assert_no_hf_leakage, build_power_splits


def _split_name(power: float, mapping: dict[str, set[float]]) -> str:
    canonical = round(float(power), 4)
    matches = [name for name, values in mapping.items() if canonical in values]
    if len(matches) != 1:
        raise RuntimeError(f"Power {power} maps to {matches}, expected exactly one split")
    return matches[0]


def _write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd", statistics=True)


def build_processed_data(
    metadata_path: str = "configs/data_metadata.yaml",
    training_path: str = "configs/training.yaml",
    output_directory: str = "data/processed",
) -> dict[str, Any]:
    metadata = load_yaml(metadata_path)
    training = load_yaml(training_path)
    data_root = resolve_data_root(metadata)
    output_root = PROJECT_ROOT / output_directory
    splits = build_power_splits()
    sim_mapping = {
        "train": set(splits.simulation_train),
        "validation": set(splits.simulation_validation),
        "test": set(splits.simulation_test),
    }
    experiment_mapping = {
        "train": set(splits.hf_train),
        "validation": set(splits.hf_validation),
    }
    test_mapping = {"test": set(splits.test_only)}
    test_cfg = metadata["test"]
    if test_cfg.get("training_allowed") is not False:
        raise RuntimeError("test_Data must be configured with training_allowed: false")
    if test_cfg.get("model_selection_allowed") is not False:
        raise RuntimeError("test_Data must be configured with model_selection_allowed: false")
    sensor_cfg = metadata["sensors"]
    sensor_metadata_verified = (
        sensor_cfg.get("coordinate_unit_status") == "verified"
        and sensor_cfg.get("value_unit_status") == "verified"
        and sensor_cfg.get("time_unit_status") == "verified"
        and sensor_cfg.get("synchronized_start") is True
    )
    warnings = [
        "The 590 W simulation time near 6 s is tolerance-aligned while time_s_raw is retained."
    ]
    if not sensor_metadata_verified:
        warnings.insert(
            0,
            "Sensor coordinate/time/value units are unverified; processed sensor columns remain raw.",
        )

    manifest: dict[str, Any] = {
        "schema_version": 3,
        "source_metadata": metadata_path,
        "raw_data_modified": False,
        "fingerprints": {
            "split_sha256": sha256_file(PROJECT_ROOT / "configs/splits.yaml"),
            "data_metadata_sha256": sha256_file(PROJECT_ROOT / metadata_path),
            "training_config_sha256": sha256_file(PROJECT_ROOT / training_path),
            "code_commit": current_code_commit(),
        },
        "simulation": [],
        "experiment_ir": [],
        "test_ir": [],
        "sensors": [],
        "warnings": warnings,
    }

    sim_dir = data_root / metadata["simulation"]["directory"]
    for path in simulation_files(sim_dir):
        frame = read_simulation(path)
        power = float(frame["power_w"][0])
        frame = frame.rename({"time_s": "time_s_raw"}).with_columns(
            ((pl.col("time_s_raw") / 2.0).round() * 2.0)
            .cast(pl.Float32)
            .alias("time_s"),
            pl.lit(_split_name(power, sim_mapping)).alias("split"),
        )
        max_alignment_error = frame.select(
            (pl.col("time_s_raw") - pl.col("time_s")).abs().max()
        ).item()
        if max_alignment_error > 0.01:
            raise RuntimeError(f"Unsafe time alignment in {path}: {max_alignment_error}")
        output = output_root / "simulation" / f"{power:g}W.parquet"
        _write_parquet(frame, output)
        manifest["simulation"].append(
            {
                "source": str(path.relative_to(PROJECT_ROOT)),
                "source_sha256": sha256_file(path),
                "power_w": power,
                "split": _split_name(power, sim_mapping),
                "rows": frame.height,
                "max_time_alignment_error_s": float(max_alignment_error),
                "path": str(output.relative_to(PROJECT_ROOT)),
            }
        )

    ir_cfg = training["ir"]
    radial_frames: dict[str, list[pl.DataFrame]] = {"experiment": [], "test": []}
    for source_dataset, directory, mapping, manifest_key in (
        (
            "experiment",
            data_root / metadata["experiment_ir"]["directory"],
            experiment_mapping,
            "experiment_ir",
        ),
        (
            "test",
            data_root / test_cfg["experiment_ir_directory"],
            test_mapping,
            "test_ir",
        ),
    ):
        for path in experiment_files(directory):
            radial = radial_observations(
                path,
                bin_width_mm=float(ir_cfg["radial_bin_width_mm"]),
                inverse_variance_epsilon=float(ir_cfg["inverse_variance_epsilon"]),
                weight_clip=tuple(float(value) for value in ir_cfg["weight_clip"]),
            )
            power = float(radial["power_w"][0])
            split = _split_name(power, mapping)
            radial = radial.with_columns(
                pl.lit(split).alias("split"),
                pl.lit(source_dataset).alias("source_dataset"),
            )
            radial_frames[source_dataset].append(radial)
            manifest[manifest_key].append(
                {
                    "source": str(path.relative_to(PROJECT_ROOT)),
                    "source_sha256": sha256_file(path),
                    "source_dataset": source_dataset,
                    "power_w": power,
                    "time_s": float(radial["time_s"][0]),
                    "split": split,
                    "radial_observations": radial.height,
                    "frame_weight_sum": float(radial["frame_weight"].sum()),
                }
            )
    ir_output = output_root / "experiment_ir_radial.parquet"
    test_ir_output = output_root / "test_ir_radial.parquet"
    _write_parquet(pl.concat(radial_frames["experiment"]), ir_output)
    _write_parquet(pl.concat(radial_frames["test"]), test_ir_output)

    sensor_frames: dict[str, list[pl.DataFrame]] = {"experiment": [], "test": []}
    for source_dataset, sensor_type, directory, mapping in (
        (
            "experiment",
            "hot",
            data_root / metadata["sensors"]["hot_directory"],
            experiment_mapping,
        ),
        (
            "experiment",
            "cold",
            data_root / metadata["sensors"]["cold_directory"],
            experiment_mapping,
        ),
        (
            "test",
            "hot",
            data_root / test_cfg["hot_directory"],
            test_mapping,
        ),
        (
            "test",
            "cold",
            data_root / test_cfg["cold_directory"],
            test_mapping,
        ),
    ):
        for path in sensor_files(directory):
            ring = ring_average_raw(path, sensor_type)
            power = float(ring["power_w"][0])
            split = _split_name(power, mapping)
            ring = ring.with_columns(
                pl.lit(split).alias("split"),
                pl.lit(source_dataset).alias("source_dataset"),
            )
            sensor_frames[source_dataset].append(ring)
            manifest["sensors"].append(
                {
                    "source": str(path.relative_to(PROJECT_ROOT)),
                    "source_sha256": sha256_file(path),
                    "source_dataset": source_dataset,
                    "sensor_type": sensor_type,
                    "power_w": power,
                    "split": split,
                    "samples": ring.height,
                    "angular_rows_collapsed": int(ring["n_angular_samples"][0]),
                }
            )
    sensor_output = output_root / "sensor_ring_raw.parquet"
    test_sensor_output = output_root / "test_sensor_ring_raw.parquet"
    _write_parquet(pl.concat(sensor_frames["experiment"]), sensor_output)
    _write_parquet(pl.concat(sensor_frames["test"]), test_sensor_output)

    assert_no_hf_leakage(
        {
            "ir": [entry["power_w"] for entry in manifest["experiment_ir"] if entry["split"] == "train"],
            "hot_cold": [entry["power_w"] for entry in manifest["sensors"] if entry["split"] == "train"],
        },
        splits.hf_validation | splits.hf_test | splits.external_sensor_test,
    )
    expected_experiment = set(splits.experiment_powers)
    observed_experiment_ir = {
        round(float(entry["power_w"]), 4) for entry in manifest["experiment_ir"]
    }
    observed_test_ir = {
        round(float(entry["power_w"]), 4) for entry in manifest["test_ir"]
    }
    if observed_experiment_ir != expected_experiment:
        raise RuntimeError(
            f"Experiment_data power mismatch: observed={sorted(observed_experiment_ir)}, "
            f"expected={sorted(expected_experiment)}"
        )
    if observed_test_ir != set(splits.test_only):
        raise RuntimeError(
            f"test_Data power mismatch: observed={sorted(observed_test_ir)}, "
            f"expected={sorted(splits.test_only)}"
        )
    forbidden_test_rows = [
        entry
        for entry in manifest["test_ir"] + manifest["sensors"]
        if entry["source_dataset"] == "test" and entry["split"] != "test"
    ]
    if forbidden_test_rows:
        raise RuntimeError("test_Data leakage detected outside the final test split")
    manifest["outputs"] = {
        "simulation_files": len(manifest["simulation"]),
        "experiment_ir_rows": sum(
            item["radial_observations"] for item in manifest["experiment_ir"]
        ),
        "test_ir_rows": sum(
            item["radial_observations"] for item in manifest["test_ir"]
        ),
        "sensor_rows": sum(item["samples"] for item in manifest["sensors"]),
        "experiment_sensor_rows": sum(
            item["samples"]
            for item in manifest["sensors"]
            if item["source_dataset"] == "experiment"
        ),
        "test_sensor_rows": sum(
            item["samples"]
            for item in manifest["sensors"]
            if item["source_dataset"] == "test"
        ),
        "ir_path": str(ir_output.relative_to(PROJECT_ROOT)),
        "test_ir_path": str(test_ir_output.relative_to(PROJECT_ROOT)),
        "sensor_path": str(sensor_output.relative_to(PROJECT_ROOT)),
        "test_sensor_path": str(test_sensor_output.relative_to(PROJECT_ROOT)),
    }
    manifest["processed_manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical processed data")
    parser.add_argument("--metadata", default="configs/data_metadata.yaml")
    parser.add_argument("--training", default="configs/training.yaml")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()
    result = build_processed_data(args.metadata, args.training, args.output)
    print(json.dumps(result["outputs"], indent=2))


if __name__ == "__main__":
    main()
