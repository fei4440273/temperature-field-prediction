from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import parse_ir_power_time
from sic_cu.data.experiment import experiment_files
from sic_cu.data.splits import build_power_splits
from sic_cu.eval.protocol_checks import validate_release_checkpoint
from sic_cu.models import ModelScales, SurfaceResidualMLP
from sic_cu.prediction import Predictor


def pixel_frame_metrics(
    target_temperature_c: np.ndarray,
    predicted_temperature_k: np.ndarray,
    radial_bins: np.ndarray,
) -> dict[str, float]:
    target_k = np.asarray(target_temperature_c, dtype=np.float64) + 273.15
    prediction_k = np.asarray(predicted_temperature_k, dtype=np.float64)
    bins = np.asarray(radial_bins)
    if target_k.shape != prediction_k.shape or bins.shape != target_k.shape:
        raise ValueError("Pixel target, prediction, and radial bins must have equal 1D shapes")
    _, inverse = np.unique(bins, return_inverse=True)
    counts = np.bincount(inverse)
    radial_mean = np.bincount(inverse, weights=target_k) / counts
    predicted_radial_mean = np.bincount(inverse, weights=prediction_k) / counts
    floor_error = target_k - radial_mean[inverse]
    error = prediction_k - target_k
    mse = float(np.mean(error**2))
    floor_mse = float(np.mean(floor_error**2))
    return {
        "pixel_rmse_c": mse**0.5,
        "pixel_mae_c": float(np.mean(np.abs(error))),
        "axisymmetric_floor_rmse_c": floor_mse**0.5,
        "radial_profile_rmse_c": float(
            np.sqrt(np.sum(counts * (predicted_radial_mean - radial_mean) ** 2) / len(target_k))
        ),
        "peak_error_c": float(np.max(prediction_k) - np.max(target_k)),
    }


def _grid(values: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray) -> tuple[np.ndarray, tuple[float, ...]]:
    x = np.unique(x_mm)
    y = np.unique(y_mm)
    image = np.full((len(y), len(x)), np.nan, dtype=np.float64)
    image[np.searchsorted(y, y_mm), np.searchsorted(x, x_mm)] = values
    half_dx = float(np.min(np.diff(x))) / 2 if len(x) > 1 else 0.5
    half_dy = float(np.min(np.diff(y))) / 2 if len(y) > 1 else 0.5
    return image, (x[0] - half_dx, x[-1] + half_dx, y[0] - half_dy, y[-1] + half_dy)


def _save_comparison(
    frame: pl.DataFrame,
    prediction_k: np.ndarray,
    destination: Path,
    power_w: float,
    time_s: float,
) -> None:
    x = frame["x_mm"].to_numpy()
    y = frame["y_mm"].to_numpy()
    target_c = frame["temperature_c"].to_numpy()
    prediction_c = prediction_k - 273.15
    error = prediction_c - target_c
    target_image, extent = _grid(target_c, x, y)
    prediction_image, _ = _grid(prediction_c, x, y)
    error_image, _ = _grid(error, x, y)
    low = float(min(np.nanmin(target_image), np.nanmin(prediction_image)))
    high = float(max(np.nanmax(target_image), np.nanmax(prediction_image)))
    error_limit = max(float(np.nanmax(np.abs(error_image))), 1e-6)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, image, title in zip(
        axes[:2], (target_image, prediction_image), ("Experiment", "Prediction"), strict=True
    ):
        artist = axis.imshow(
            image, origin="lower", extent=extent, cmap="inferno", vmin=low, vmax=high
        )
        figure.colorbar(artist, ax=axis, label="Temperature (℃)")
        axis.set_title(title)
    artist = axes[2].imshow(
        error_image,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-error_limit,
        vmax=error_limit,
    )
    figure.colorbar(artist, ax=axes[2], label="Prediction - experiment (℃)")
    axes[2].set_title("Error")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    figure.suptitle(f"SiC top surface: {power_w:g} W, {time_s:g} s")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def evaluate_ir_pixels(
    checkpoint: str | None = None,
    surface_checkpoint: str | None = None,
    split: str = "test",
    output_path: str = "reports/ir_pixel_evaluation.json",
    figure_directory: str = "reports/figures/ir_pixels",
    device: str | None = None,
    release_manifest_path: str | None = None,
) -> dict[str, Any]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if checkpoint is not None and surface_checkpoint is not None:
        raise ValueError("checkpoint and surface_checkpoint are mutually exclusive")
    if split == "test":
        frozen_checkpoint = checkpoint or surface_checkpoint
        if frozen_checkpoint is None or release_manifest_path is None:
            raise RuntimeError(
                "Test pixel evaluation requires a registered checkpoint "
                "and frozen release manifest"
            )
        validate_release_checkpoint(release_manifest_path, frozen_checkpoint)
    splits = build_power_splits()
    expected = {
        "train": splits.hf_train,
        "validation": splits.hf_validation,
        "test": splits.hf_test,
    }[split]
    metadata = load_yaml("configs/data_metadata.yaml")
    directory_key = (
        metadata["test"]["experiment_ir_directory"]
        if split == "test"
        else metadata["experiment_ir"]["directory"]
    )
    directory = PROJECT_ROOT / metadata["data_root"] / directory_key
    grouped: dict[float, list[Path]] = {float(power): [] for power in expected}
    for path in experiment_files(directory):
        power, _ = parse_ir_power_time(path)
        for expected_power in expected:
            if np.isclose(power, expected_power, atol=1e-6):
                grouped[float(expected_power)].append(path)
                break
    if any(not paths for paths in grouped.values()):
        missing = [power for power, paths in grouped.items() if not paths]
        raise RuntimeError(f"Missing raw IR files for powers: {missing}")

    predictor = Predictor(checkpoint=checkpoint, device=device)
    surface_model: SurfaceResidualMLP | None = None
    surface_seed: int | None = None
    surface_device = predictor.device
    if surface_checkpoint is not None:
        payload = torch.load(
            PROJECT_ROOT / surface_checkpoint,
            map_location=surface_device,
            weights_only=False,
        )
        if payload.get("method") != "surface_residual_mlp":
            raise ValueError("surface_checkpoint must contain a surface_residual_mlp model")
        surface_seed = int(payload["seed"])
        surface_model = SurfaceResidualMLP(
            ModelScales(**payload["scales"]), **payload["model_kwargs"]
        ).to(surface_device)
        surface_model.load_state_dict(payload["model_state"])
        surface_model.eval()
    power_records: list[dict[str, Any]] = []
    figure_root = PROJECT_ROOT / figure_directory
    for power, paths in sorted(grouped.items()):
        ordered = sorted(paths, key=lambda path: parse_ir_power_time(path)[1])
        times = np.asarray([parse_ir_power_time(path)[1] for path in ordered], dtype=np.float32)
        prediction = predictor.predict(power, times_s=times)
        frame_records: list[dict[str, float]] = []
        last_frame: pl.DataFrame | None = None
        last_prediction: np.ndarray | None = None
        for index, path in enumerate(ordered):
            frame = pl.read_csv(
                path,
                columns=["x_mm", "y_mm", "r_mm", "temperature_c", "radial_bin"],
                low_memory=True,
            )
            radius_m = frame["r_mm"].to_numpy() / 1000.0
            if predictor.supports_point_queries:
                coordinates = np.column_stack(
                    (
                        radius_m,
                        np.zeros(frame.height),
                        np.full(frame.height, times[index]),
                        np.full(frame.height, power),
                        np.ones(frame.height),
                    )
                ).astype(np.float32)
                predicted = predictor.predict_points(coordinates).reshape(-1)
            else:
                surface_r, surface_temperature = _surface_profile(prediction, index)
                predicted = np.interp(radius_m, surface_r, surface_temperature)
            if surface_model is not None:
                inputs = np.column_stack(
                    (
                        frame["r_mm"].to_numpy() / 1000.0,
                        np.full(frame.height, times[index]),
                        np.full(frame.height, power),
                    )
                ).astype(np.float32)
                residuals = []
                with torch.no_grad():
                    for offset in range(0, len(inputs), 65536):
                        residuals.append(
                            surface_model(
                                torch.from_numpy(inputs[offset : offset + 65536]).to(
                                    surface_device
                                )
                            )
                            .cpu()
                            .numpy()
                        )
                predicted = predicted + np.concatenate(residuals).reshape(-1)
            frame_records.append(
                {
                    "time_s": float(times[index]),
                    **pixel_frame_metrics(
                        frame["temperature_c"].to_numpy(),
                        predicted,
                        frame["radial_bin"].to_numpy(),
                    ),
                }
            )
            last_frame = frame
            last_prediction = predicted
        assert last_frame is not None and last_prediction is not None
        figure_path = figure_root / f"{power:g}W_last_frame.png"
        _save_comparison(
            last_frame,
            last_prediction,
            figure_path,
            power,
            float(times[-1]),
        )
        means = {
            name: float(np.mean([record[name] for record in frame_records]))
            for name in (
                "pixel_rmse_c",
                "pixel_mae_c",
                "axisymmetric_floor_rmse_c",
                "radial_profile_rmse_c",
            )
        }
        means["peak_mae_c"] = float(
            np.mean([abs(record["peak_error_c"]) for record in frame_records])
        )
        power_records.append(
            {
                "power_w": power,
                "frame_count": len(frame_records),
                **means,
                "last_frame_figure": str(figure_path.relative_to(PROJECT_ROOT)),
                "frames": frame_records,
                "prediction_metadata": {
                    "source": (
                        f"surface_residual_mlp:{surface_checkpoint}"
                        if surface_model is not None
                        else prediction.metadata.source
                    ),
                    "warnings": (
                        (
                            "This model is corrected only on the SiC top surface; it does not "
                            "predict an experiment-validated internal field.",
                        )
                        if surface_model is not None
                        else prediction.metadata.warnings
                    ),
                },
            }
        )
    aggregate_names = (
        "pixel_rmse_c",
        "pixel_mae_c",
        "axisymmetric_floor_rmse_c",
        "radial_profile_rmse_c",
        "peak_mae_c",
    )
    result = {
        "schema_version": 1,
        "temperature_error_unit": "℃",
        "split": split,
        "checkpoint": checkpoint,
        "surface_checkpoint": surface_checkpoint,
        "seed": surface_seed,
        "powers_w": sorted(expected),
        "aggregation": "pixels within frame, then equal frame means, then equal power means",
        "query_mode": (
            "direct_coordinates"
            if predictor.supports_point_queries
            else "explicit_grid_interpolation"
        ),
        "aggregate": {
            name: {
                "mean": float(np.mean([record[name] for record in power_records])),
                "std": float(np.std([record[name] for record in power_records])),
            }
            for name in aggregate_names
        },
        "per_power": power_records,
        "material_passport": {
            "truth_scope": "raw measured SiC top-surface pixels",
            "internal_field_truth": "not available",
            "pixel_independence_claimed": False,
            "each_frame_total_weight_capped": True,
            "all_source_pixels_marked_recovered": True,
            "experiment_data_used": surface_model is not None,
            "prediction_scope": (
                "SiC top surface only" if surface_model is not None else "predictor field surface"
            ),
        },
    }
    destination = PROJECT_ROOT / output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _surface_profile(prediction: Any, time_index: int) -> tuple[np.ndarray, np.ndarray]:
    coordinates = prediction.coordinates_rz_m
    mask = (prediction.material_ids == 1) & np.isclose(coordinates[:, 1], 0.0, atol=1e-8)
    order = np.argsort(coordinates[mask, 0])
    return coordinates[mask, 0][order], prediction.mean_temperature_k[time_index, mask][order]
