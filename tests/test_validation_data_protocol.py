from __future__ import annotations

import polars as pl
import pytest

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.sensors import load_canonical_sensor_observations
from sic_cu.data.splits import build_power_splits
from sic_cu.eval.validation import build_three_power_comparison


def test_processed_train_validation_and_test_sources_are_isolated() -> None:
    ir = pl.read_parquet(PROJECT_ROOT / "data/processed/experiment_ir_radial.parquet")
    sensors = load_canonical_sensor_observations()
    splits = build_power_splits()

    for frame in (ir, sensors):
        observed_pairs = set(
            frame.select("source_dataset", "split").unique().iter_rows()
        )
        assert observed_pairs == {
            ("experiment", "train"),
            ("experiment", "validation"),
            ("test", "test"),
        }
        for split, expected in (
            ("train", splits.hf_train),
            ("validation", splits.hf_validation),
            ("test", splits.hf_test),
        ):
            selected = frame.filter(pl.col("split") == split)
            observed = {
                round(float(value), 4) for value in selected["power_w"].unique()
            }
            assert observed == set(expected)


def test_sensor_loader_preserves_header_times() -> None:
    sensors = load_canonical_sensor_observations().filter(
        (pl.col("source_dataset") == "test") & (pl.col("power_w") == 169.0)
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
                "mean_error_k": 1.0,
                "mae_k": 2.0,
                "rmse_k": 3.0,
                "max_abs_error_k": 4.0,
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
                    "mean_error_k": 1.0,
                    "mae_k": 2.0,
                    "rmse_k": 3.0,
                    "max_abs_error_k": 4.0,
                },
            }
            for power in powers
            for sensor in ("hot", "cold")
        ]
    }

    result = build_three_power_comparison(ir, sensors, powers, "test")

    assert result["powers_w"] == powers
    assert result["used_for_gradient_updates"] is False
    assert result["used_for_model_selection"] is False
    assert len(result["per_power"]) == 3
    for item in result["per_power"]:
        assert set(item["modalities"]) == {"top_surface", "hot", "cold"}
        assert item["combined"]["mae_k"] == pytest.approx(2.0)
        assert item["combined"]["rmse_k"] == pytest.approx(3.0)
        assert item["combined"]["mean_error_k"] == pytest.approx(1.0)


def test_three_power_test_aggregate_uses_global_metric_definitions() -> None:
    powers = [169.0, 339.0, 634.0]
    ir = {
        "per_power": [
            {
                "power_w": power,
                "mean_error_k": index,
                "mae_k": index,
                "rmse_k": index,
                "max_abs_error_k": index,
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
                    "mean_error_k": index,
                    "mae_k": index,
                    "rmse_k": index,
                    "max_abs_error_k": index,
                },
            }
            for index, power in enumerate(powers, start=1)
            for sensor in ("hot", "cold")
        ]
    }

    aggregate = build_three_power_comparison(ir, sensors, powers, "test")["aggregate"]

    assert aggregate["mean_error_k"] == pytest.approx(2.0)
    assert aggregate["mae_k"] == pytest.approx(2.0)
    assert aggregate["rmse_k"] == pytest.approx((14.0 / 3.0) ** 0.5)
    assert aggregate["max_abs_error_k"] == pytest.approx(3.0)
