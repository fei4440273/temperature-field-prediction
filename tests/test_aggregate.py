from __future__ import annotations

import json

import pytest

from sic_cu.eval import aggregate_runs


def _run_record(seed: int, powers: tuple[float, ...]) -> dict:
    per_power = []
    for index, power in enumerate(powers):
        value = float(seed + index + 1)
        per_power.append(
            {
                "power_w": power,
                "inference_seconds": value / 10,
                "metrics": {
                    "full_field": {
                        "rmse_k": value,
                        "mae_k": value / 2,
                        "r2": 1 - value / 100,
                        "relative_l2": value / 100,
                    },
                    "tmax": {"mae_k": value * 2},
                },
            }
        )
    return {
        "method": "mlp",
        "seed": seed,
        "best_epoch": 3,
        "best_validation_rmse_k": 1.0,
        "training_seconds": 2.0 + seed,
        "test": {
            "aggregate": {
                name: {
                    "mean": sum(item["metrics"]["full_field"][name] for item in per_power)
                    / len(per_power),
                    "std": 0.0,
                }
                for name in ("rmse_k", "mae_k", "r2", "relative_l2")
            },
            "per_power": per_power,
        },
        "material_passport": {"simulation_role": "synthetic unit test"},
    }


def test_aggregate_simulation_runs_and_reject_mismatched_powers(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(aggregate_runs, "PROJECT_ROOT", tmp_path)
    directories = []
    for seed in range(2):
        directory = tmp_path / f"run{seed}"
        directory.mkdir()
        (directory / "metrics.json").write_text(
            json.dumps(_run_record(seed, (50.0, 130.0))), encoding="utf-8"
        )
        directories.append(directory.name)
    result = aggregate_runs.aggregate_simulation_runs(directories, "summary.json")
    assert result["run_count"] == 2
    assert result["test_powers_w"] == [50.0, 130.0]
    assert result["aggregate_across_seeds"]["rmse_k"]["mean"] == pytest.approx(2.0)
    assert (tmp_path / "summary.json").exists()

    mismatch = tmp_path / "mismatch"
    mismatch.mkdir()
    (mismatch / "metrics.json").write_text(
        json.dumps(_run_record(2, (50.0, 210.0))), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="same ordered simulation test powers"):
        aggregate_runs.aggregate_simulation_runs([directories[0], mismatch.name])


def test_aggregate_ir_pixel_runs_requires_unique_seeds(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(aggregate_runs, "PROJECT_ROOT", tmp_path)
    paths = []
    names = (
        "pixel_rmse_k",
        "pixel_mae_k",
        "axisymmetric_floor_rmse_k",
        "radial_profile_rmse_k",
        "peak_mae_k",
    )
    for seed in range(2):
        record = {
            "split": "test",
            "powers_w": [309.0],
            "aggregation": "frame_then_power",
            "seed": seed,
            "surface_checkpoint": f"seed{seed}/best.pt",
            "aggregate": {name: {"mean": float(seed + 1)} for name in names},
            "per_power": [
                {"power_w": 309.0, **{name: float(seed + 1) for name in names}}
            ],
            "material_passport": {"truth_scope": "synthetic unit test"},
        }
        path = tmp_path / f"pixel{seed}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        paths.append(path.name)
    result = aggregate_runs.aggregate_ir_pixel_runs(paths, "pixel_summary.json")
    assert result["aggregate_across_seeds"]["pixel_rmse_k"]["mean"] == pytest.approx(1.5)

    duplicate = json.loads((tmp_path / paths[1]).read_text(encoding="utf-8"))
    duplicate["seed"] = 0
    (tmp_path / paths[1]).write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="unique non-null seeds"):
        aggregate_runs.aggregate_ir_pixel_runs(paths)
