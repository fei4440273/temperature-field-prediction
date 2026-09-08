from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import parse_ir_power_time, parse_power
from sic_cu.data.experiment import radial_observations
from sic_cu.data.sensors import (
    audit_sensor_file,
    load_canonical_sensor_observations,
    ring_average_raw,
)
from sic_cu.data.simulation import read_simulation
from sic_cu.data.splits import (
    assert_no_hf_leakage,
    build_power_splits,
)


DATA_ROOT = PROJECT_ROOT / "data"


def test_filename_parsing() -> None:
    assert parse_power("590W_temperature_coordinates.csv") == 590.0
    assert parse_power("HotData-115.2W.csv") == 115.2
    assert parse_ir_power_time("309W-135s_50mm_temperature.csv") == (309.0, 135.0)


def test_simulation_coordinate_and_temperature_conversion() -> None:
    frame = read_simulation(DATA_ROOT / "Simulation_data/10W_temperature_coordinates.csv")
    assert frame.columns == [
        "power_w",
        "time_s",
        "r_m",
        "z_m",
        "material_id",
        "node_label",
        "temperature_k",
    ]
    assert np.isclose(frame["r_m"].max(), 0.05834)
    assert np.isclose(frame["z_m"].min(), -0.0175)
    assert np.isclose(frame["temperature_k"].min(), 295.15)
    assert set(frame["material_id"].unique().to_list()) == {0, 1}


def test_power_splits_are_complete_and_disjoint() -> None:
    splits = build_power_splits()
    assert len(splits.simulation_train) == 60
    assert len(splits.simulation_validation) == 10
    assert len(splits.simulation_test) == 10
    assert len(splits.hf_train) == 12
    assert splits.hf_validation == frozenset({115.2, 403.0, 630.5})
    assert len(splits.hf_test) == 3
    assert splits.hf_test == frozenset({169.0, 339.0, 634.0})
    assert splits.external_sensor_test == frozenset()
    assert splits.experiment_powers == splits.hf_train | splits.hf_validation
    assert splits.test_only == splits.hf_test
    assert not (
        splits.hf_train & splits.hf_validation
        or splits.hf_train & splits.hf_test
        or splits.hf_validation & splits.hf_test
    )


def test_hf_leakage_is_fatal() -> None:
    with pytest.raises(RuntimeError, match="leakage"):
        assert_no_hf_leakage({"ir": [115.2, 309.0], "hot": [254.5]}, [309.0, 593.5])
    assert_no_hf_leakage({"ir": [115.2, 254.5], "hot": [364.3]}, [309.0, 593.5])


def test_explicit_power_filter_handles_float32_sensor_keys() -> None:
    from sic_cu.train.multifidelity import _sensor_tensors

    coordinates, target, delta, baseline = _sensor_tensors(
        torch.device("cpu"), split="train", powers_w=[216.8, 364.3]
    )
    assert len(coordinates) == len(target) == len(delta) == len(baseline) > 0
    observed = {round(float(value), 4) for value in coordinates[:, 3].unique()}
    assert observed == {216.8, 364.3}


@pytest.mark.parametrize(
    ("sensor_type", "path", "expected_radius", "expected_rows"),
    [
        ("hot", "Experiment_data/HotData/HotData-55W.csv", 0.028, 148),
        ("cold", "Experiment_data/ColdData/ColdData-55W.csv", 0.0415, 218),
    ],
)
def test_sensor_ring_is_not_counted_as_independent_samples(
    sensor_type: str, path: str, expected_radius: float, expected_rows: int
) -> None:
    source = DATA_ROOT / path
    audit = audit_sensor_file(source, sensor_type)
    ring = ring_average_raw(source, sensor_type)
    assert audit.rows == expected_rows
    assert np.isclose(audit.radius_mean_raw, expected_radius, atol=1e-6)
    assert audit.all_ring_values_identical
    assert audit.max_angular_spread_raw == 0.0
    assert ring.height == audit.time_columns
    assert ring["n_angular_samples"].unique().to_list() == [expected_rows]
    assert np.isclose(ring["delta_value_raw"][0], 0.0)


def test_verified_sensor_units_are_converted_to_canonical_units() -> None:
    observations = load_canonical_sensor_observations()
    first_hot = observations.filter(
        (pl.col("power_w") == 55.0)
        & (pl.col("sensor_type") == "hot")
        & (pl.col("time_s") == 1.0)
    )
    assert first_hot.height == 1
    assert np.isclose(first_hot["r_m"][0], 0.028, atol=1e-6)
    assert np.isclose(first_hot["temperature_k"][0], 25.29215 + 273.15, atol=1e-5)
    assert np.isclose(first_hot["delta_temperature_k"][0], 0.0, atol=1e-6)


def test_ir_frame_weights_are_capped() -> None:
    source = DATA_ROOT / "Experiment_data/Topdata/115.2W-5s_50mm_temperature.csv"
    radial = radial_observations(source, bin_width_mm=0.25)
    assert radial.height == 101
    assert np.isclose(radial["frame_weight"].sum(), 1.0, atol=1e-6)
    assert radial["reliability_weight_raw"].min() >= 0.1
    assert radial["reliability_weight_raw"].max() <= 10.0


def test_processed_manifest_has_no_hf_leakage() -> None:
    manifest_path = DATA_ROOT / "processed/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["raw_data_modified"] is False
    assert manifest["outputs"]["simulation_files"] == 80
    forbidden = {115.2, 403.0, 630.5, 169.0, 339.0, 634.0}
    train_sensor = {
        round(item["power_w"], 4)
        for item in manifest["sensors"]
        if item["split"] == "train"
    }
    assert not train_sensor & forbidden
    assert all(
        item["split"] == "test" and item["source_dataset"] == "test"
        for item in manifest["test_ir"]
    )
    ir = pl.read_parquet(DATA_ROOT / "processed/experiment_ir_radial.parquet")
    frame_weights = ir.group_by("power_w", "time_s").agg(pl.col("frame_weight").sum())
    assert np.allclose(frame_weights["frame_weight"].to_numpy(), 1.0, atol=1e-6)


def test_audit_report_is_complete() -> None:
    audit = json.loads((PROJECT_ROOT / "reports/data_audit.json").read_text())
    assert audit["gate"]["structural_status"] == "PASS"
    assert audit["gate"]["physics_training_status"] == "BLOCKED_UNVERIFIED_METADATA"
    assert audit["gate"]["parameter_identification_status"] == "READY"
    assert audit["gate"]["user_input_status"] == "COMPLETE"
    assert audit["gate"]["experiment_ir_metadata_status"] == "VERIFIED"
    assert audit["gate"]["user_input_unknowns"] == []
    assert set(audit["gate"]["pending_model_parameters"]) == {
        "configs/boundary_conditions.yaml.external_surface.silicon_carbide_emissivity",
        "configs/boundary_conditions.yaml.external_surface.copper_emissivity",
        "configs/boundary_conditions.yaml.interface.contact_resistance_m2_k_w",
    }
    assert audit["inventory"]["file_count"] == 543
    assert audit["inventory"]["sha256_computed"] is True
    assert all(record["sha256"] for record in audit["inventory"]["files"])
