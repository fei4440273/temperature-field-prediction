from __future__ import annotations

import polars as pl
import pytest

from sic_cu.data.sensors import load_canonical_sensor_observations
from sic_cu.data.processed import load_processed_ir_observations
from sic_cu.data.splits import build_power_splits
from sic_cu.eval.validation import build_three_power_comparison


def test_processed_train_validation_and_test_sources_are_isolated() -> None:
    experiment_ir = load_processed_ir_observations()
    test_ir = load_processed_ir_observations("test")
    experiment_sensors = load_canonical_sensor_observations()
    test_sensors = load_canonical_sensor_observations(split="test")
    splits = build_power_splits()

    for experiment_frame, test_frame in (
        (experiment_ir, test_ir),
        (experiment_sensors, test_sensors),
    ):
        assert set(experiment_frame["source_dataset"].unique()) == {"experiment"}
        assert set(experiment_frame["split"].unique()) == {"train", "validation"}
        assert set(test_frame["source_dataset"].unique()) == {"test"}
        assert set(test_frame["split"].unique()) == {"test"}
        for split, expected in (("train", splits.hf_train), ("validation", splits.hf_validation)):
            selected = experiment_frame.filter(pl.col("split") == split)
            observed = {
                round(float(value), 4) for value in selected["power_w"].unique()
            }
            assert observed == set(expected)
        observed_test = {
            round(float(value), 4) for value in test_frame["power_w"].unique()
        }
        assert observed_test == set(splits.hf_test)


def test_sensor_loader_preserves_header_times() -> None:
    sensors = load_canonical_sensor_observations(split="test").filter(
        pl.col("power_w") == 169.0
    )
    hot = sensors.filter(pl.col("sensor_type") == "hot")
    cold = sensors.filter(pl.col("sensor_type") == "cold")

    assert hot["time_s"].min() == pytest.approx(0.0)
    assert cold["time_s"].min() == pytest.approx(1.0)


def test_three_power_test_summary_contains_error_mae_and_rmse() -> None:
    powers = [169.0, 339.0, 634.0]
    ir = {
        "per_power": [
            {
                "power_w": power,
                "mean_error_c": 1.0,
                "mae_c": 2.0,
                "rmse_c": 3.0,
                "max_abs_error_c": 4.0,
            }
            for power in powers
        ]
    }
    sensors = {
        "per_curve": [
            {
                "power_w": power,
                "sensor_type": sensor,
                "absolute": {
                    "mean_error_c": 1.0,
                    "mae_c": 2.0,
                    "rmse_c": 3.0,
                    "max_abs_error_c": 4.0,
                },
                "delta": {
                    "mean_error_c": 0.5,
                    "mae_c": 1.0,
                    "rmse_c": 1.5,
                    "max_abs_error_c": 2.0,
                },
            }
            for power in powers
            for sensor in ("hot", "cold")
        ]
    }

    result = build_three_power_comparison(ir, sensors, powers, "test")

    assert result["powers_w"] == powers
    assert result["temperature_error_unit"] == "℃"
    assert result["used_for_gradient_updates"] is False
    assert result["used_for_model_selection"] is False
    assert len(result["per_power"]) == 3
    for item in result["per_power"]:
        assert set(item["modalities"]) == {"top_surface", "hot", "cold"}
        assert item["combined"]["mae_c"] == pytest.approx(2.0)
        assert item["combined"]["rmse_c"] == pytest.approx(3.0)
        assert item["combined"]["mean_error_c"] == pytest.approx(1.0)
    assert result["selection"] is None


def test_three_power_test_aggregate_uses_global_metric_definitions() -> None:
    powers = [169.0, 339.0, 634.0]
    ir = {
        "per_power": [
            {
                "power_w": power,
                "mean_error_c": index,
                "mae_c": index,
                "rmse_c": index,
                "max_abs_error_c": index,
            }
            for index, power in enumerate(powers, start=1)
        ]
    }
    sensors = {
        "per_curve": [
            {
                "power_w": power,
                "sensor_type": sensor,
                "absolute": {
                    "mean_error_c": index,
                    "mae_c": index,
                    "rmse_c": index,
                    "max_abs_error_c": index,
                },
                "delta": {
                    "mean_error_c": index,
                    "mae_c": index,
                    "rmse_c": index,
                    "max_abs_error_c": index,
                },
            }
            for index, power in enumerate(powers, start=1)
            for sensor in ("hot", "cold")
        ]
    }

    aggregate = build_three_power_comparison(ir, sensors, powers, "test")["aggregate"]

    assert aggregate["mean_error_c"] == pytest.approx(2.0)
    assert aggregate["mae_c"] == pytest.approx(2.0)
    assert aggregate["rmse_c"] == pytest.approx(2.0)
    assert aggregate["equal_power_mse_rmse_c"] == pytest.approx((14.0 / 3.0) ** 0.5)
    assert aggregate["max_abs_error_c"] == pytest.approx(3.0)
