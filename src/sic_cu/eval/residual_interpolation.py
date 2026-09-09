from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
import polars as pl

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.splits import build_power_splits
from sic_cu.data.processed import load_processed_ir_observations
from sic_cu.eval.protocol_checks import validate_release_manifest
from sic_cu.models.residual_interpolation import (
    SUPPORTED_RESIDUAL_INTERPOLATION_METHODS,
    PowerResidualInterpolator,
    ResidualTrajectory,
)
from sic_cu.train.surface_residual import _lf_surface


LOW_FIDELITY_COLUMN = "_low_fidelity_temperature_k"


def _canonical_set(values: Iterable[float]) -> frozenset[float]:
    return frozenset(round(float(value), 4) for value in values)


def _power_frame(frame: pl.DataFrame, power_w: float) -> pl.DataFrame:
    key = round(float(power_w) * 10_000)
    return frame.filter((pl.col("power_w") * 10_000).round().cast(pl.Int64) == key)


def build_residual_trajectory(
    frame: pl.DataFrame,
    power_w: float,
    simulation_powers_w: list[float],
) -> ResidualTrajectory:
    selected = _power_frame(frame, power_w)
    if selected.is_empty():
        raise ValueError(f"No experiment observations for {power_w:g} W")
    low_fidelity = (
        selected[LOW_FIDELITY_COLUMN].to_numpy()
        if LOW_FIDELITY_COLUMN in selected.columns
        else _lf_surface(selected, simulation_powers_w)
    )
    residual = selected["temperature_mean_k"].to_numpy() - low_fidelity
    times = np.asarray(sorted(selected["time_s"].unique().to_list()), dtype=np.float64)
    first = selected.filter(pl.col("time_s") == times[0]).sort("r_m")
    radii = first["r_m"].to_numpy().astype(np.float64)
    profiles = []
    for time_s in times:
        current = selected.filter(pl.col("time_s") == time_s).sort("r_m")
        current_radii = current["r_m"].to_numpy().astype(np.float64)
        current_residual = residual[
            np.flatnonzero(np.isclose(selected["time_s"].to_numpy(), time_s))
        ]
        order = np.argsort(
            selected.filter(pl.col("time_s") == time_s)["r_m"].to_numpy()
        )
        current_residual = current_residual[order]
        profiles.append(np.interp(radii, current_radii, current_residual))
    return ResidualTrajectory(
        power_w=float(power_w),
        times_s=times,
        radii_m=radii,
        residual_k=np.asarray(profiles),
    )


def fit_residual_interpolator(
    frame: pl.DataFrame,
    training_powers_w: Iterable[float],
    simulation_powers_w: list[float],
) -> PowerResidualInterpolator:
    available = _canonical_set(frame["power_w"].unique().to_list())
    requested = _canonical_set(training_powers_w)
    missing = requested - available
    if missing:
        raise ValueError(f"Missing training powers: {sorted(missing)}")
    return PowerResidualInterpolator(
        build_residual_trajectory(frame, power, simulation_powers_w)
        for power in sorted(requested)
    )


def fit_surface_temperature_interpolator(
    frame: pl.DataFrame,
    training_powers_w: Iterable[float],
    initial_temperature_k: float = 295.15,
) -> PowerResidualInterpolator:
    available = _canonical_set(frame["power_w"].unique().to_list())
    requested = _canonical_set(training_powers_w)
    missing = requested - available
    if missing:
        raise ValueError(f"Missing training powers: {sorted(missing)}")
    trajectories = []
    for power in sorted(requested):
        selected = _power_frame(frame, power)
        times = np.asarray(
            sorted(selected["time_s"].unique().to_list()), dtype=np.float64
        )
        first = selected.filter(pl.col("time_s") == times[0]).sort("r_m")
        radii = first["r_m"].to_numpy().astype(np.float64)
        profiles = []
        for time_s in times:
            current = selected.filter(pl.col("time_s") == time_s).sort("r_m")
            profiles.append(
                np.interp(
                    radii,
                    current["r_m"].to_numpy(),
                    current["temperature_mean_k"].to_numpy()
                    - initial_temperature_k,
                )
            )
        trajectories.append(
            ResidualTrajectory(power, times, radii, np.asarray(profiles))
        )
    return PowerResidualInterpolator(trajectories)


def _metrics_for_power(
    frame: pl.DataFrame,
    prediction_k: np.ndarray,
) -> dict[str, float]:
    target = frame["temperature_mean_k"].to_numpy()
    weights = frame["frame_weight"].to_numpy()
    error = prediction_k - target
    denominator = float(weights.sum())
    times = frame["time_s"].to_numpy()
    radii = frame["r_m"].to_numpy()
    peak_errors = []
    peak_relative_errors = []
    gradient_absolute = []
    for time_s in np.unique(times):
        indices = np.flatnonzero(times == time_s)
        order = indices[np.argsort(radii[indices])]
        peak_error = float(prediction_k[indices].max() - target[indices].max())
        peak_errors.append(peak_error)
        peak_relative_errors.append(
            100.0
            * abs(peak_error)
            / max(abs(float(target[indices].max()) - 273.15), 1e-12)
        )
        if len(order) > 1:
            predicted_gradient = np.gradient(prediction_k[order], radii[order])
            target_gradient = np.gradient(target[order], radii[order])
            gradient_absolute.append(
                float(np.mean(np.abs(predicted_gradient - target_gradient)) * 1e-3)
            )
    return {
        "power_w": float(frame["power_w"][0]),
        "rmse_c": float(np.sqrt(np.sum(weights * error**2) / denominator)),
        "mae_c": float(np.sum(weights * np.abs(error)) / denominator),
        "peak_mae_c": float(np.mean(np.abs(peak_errors))),
        "peak_max_abs_error_c": float(np.max(np.abs(peak_errors))),
        "peak_mean_relative_error_percent": float(np.mean(peak_relative_errors)),
        "peak_max_relative_error_percent": float(np.max(peak_relative_errors)),
        "radial_gradient_mae_c_per_mm": float(np.mean(gradient_absolute)),
    }


def evaluate_residual_interpolator(
    frame: pl.DataFrame,
    training_powers_w: Iterable[float],
    test_powers_w: Iterable[float],
    simulation_powers_w: list[float],
    method: str,
) -> dict[str, Any]:
    training = _canonical_set(training_powers_w)
    testing = _canonical_set(test_powers_w)
    if training & testing:
        raise RuntimeError("Training and test powers overlap")
    predictor = fit_residual_interpolator(frame, training, simulation_powers_w)
    per_power = []
    for power in sorted(testing):
        selected = _power_frame(frame, power)
        low_fidelity = (
            selected[LOW_FIDELITY_COLUMN].to_numpy()
            if LOW_FIDELITY_COLUMN in selected.columns
            else _lf_surface(selected, simulation_powers_w)
        )
        correction = predictor.predict(
            power,
            selected["time_s"].to_numpy(),
            selected["r_m"].to_numpy(),
            method,
        )
        per_power.append(_metrics_for_power(selected, low_fidelity + correction))
    return {
        "method": method,
        "training_powers_w": sorted(training),
        "test_powers_w": sorted(testing),
        "aggregate": {
            key: float(np.mean([record[key] for record in per_power]))
            for key in (
                "rmse_c",
                "mae_c",
                "peak_mae_c",
                "peak_mean_relative_error_percent",
                "radial_gradient_mae_c_per_mm",
            )
        },
        "per_power": per_power,
    }


def _method_comparison(
    frame: pl.DataFrame,
    training: Iterable[float],
    testing: Iterable[float],
    simulation_powers: list[float],
) -> dict[str, dict[str, Any]]:
    return {
        method: evaluate_residual_interpolator(
            frame, training, testing, simulation_powers, method
        )
        for method in SUPPORTED_RESIDUAL_INTERPOLATION_METHODS
    }


def evaluate_all_protocols(
    output_path: str = "reports/residual_interpolation_cv.json",
    release_manifest_path: str | None = None,
) -> dict[str, Any]:
    splits = build_power_splits()
    frame = load_processed_ir_observations()
    if release_manifest_path is not None:
        validate_release_manifest(release_manifest_path)
        frame = pl.concat((frame, load_processed_ir_observations("test")))
    simulation_powers = sorted(splits.simulation_train)
    frame = frame.with_columns(
        pl.Series(LOW_FIDELITY_COLUMN, _lf_surface(frame, simulation_powers))
    )

    fixed_validation = _method_comparison(
        frame, splits.hf_train, splits.hf_validation, simulation_powers
    )
    selected_method = min(
        fixed_validation,
        key=lambda name: fixed_validation[name]["aggregate"]["rmse_c"],
    )
    fixed_test = None
    if release_manifest_path is not None:
        fixed_test = evaluate_residual_interpolator(
            frame,
            splits.hf_train,
            splits.hf_test,
            simulation_powers,
            selected_method,
        )

    result = {
        "schema_version": 1,
        "temperature_error_unit": "℃",
        "method_family": "complete-power surface residual interpolation",
        "scope": "SiC top surface only; no internal-field claim",
        "fixed_split": {
            "selection_metric": "validation aggregate RMSE",
            "selected_method": selected_method,
            "validation_method_comparison": fixed_validation,
            "test": fixed_test,
            "test_status": (
                "evaluated_from_frozen_release"
                if fixed_test is not None
                else "sealed_until_frozen_release"
            ),
        },
        "cross_validation": "disabled_by_fixed_split_protocol",
        "material_passport": {
            "simulation_role": "low-fidelity surface baseline",
            "experiment_role": "complete-power radial residual trajectories",
            "physics_assumption": "constant-property temperature rise and dominant model discrepancy are approximately power-scaled",
            "experiment_internal_field_truth": "not available",
            "pixel_level_random_split": False,
        },
    }
    destination = PROJECT_ROOT / output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
