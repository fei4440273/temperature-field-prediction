from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .common import parse_ir_power_time


EXPECTED_COLUMNS = (
    "x_mm",
    "y_mm",
    "r_mm",
    "r_norm",
    "temperature_c",
    "radial_bin",
    "radial_bin_samples",
    "is_max_temperature_anchor",
    "is_recovered",
)

SCHEMA_OVERRIDES = {
    "x_mm": pl.Float64,
    "y_mm": pl.Float64,
    "r_mm": pl.Float64,
    "r_norm": pl.Float64,
    "temperature_c": pl.Float64,
    "radial_bin": pl.Int64,
    "radial_bin_samples": pl.Int64,
    "is_max_temperature_anchor": pl.Boolean,
    "is_recovered": pl.Boolean,
}


@dataclass(frozen=True)
class ExperimentFileAudit:
    path: str
    power_w: float
    time_s: float
    rows: int
    x_range_mm: tuple[float, float]
    y_range_mm: tuple[float, float]
    r_range_mm: tuple[float, float]
    temperature_range_c: tuple[float, float]
    radial_bins: int
    radial_coordinate_max_error_mm: float
    recovered_true: int
    recovered_false: int
    max_temperature_anchors: int
    null_values: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def experiment_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*W-*s_50mm_temperature.csv"), key=parse_ir_power_time)


def audit_experiment_file(path: Path) -> ExperimentFileAudit:
    frame = pl.read_csv(
        path,
        low_memory=True,
        rechunk=False,
        schema_overrides=SCHEMA_OVERRIDES,
    )
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected experiment schema in {path}: {frame.columns}")
    power, time = parse_ir_power_time(path)
    radial_error = frame.select(
        (
            (pl.col("x_mm").pow(2) + pl.col("y_mm").pow(2)).sqrt()
            - pl.col("r_mm")
        )
        .abs()
        .max()
    ).item()
    return ExperimentFileAudit(
        path=str(path),
        power_w=power,
        time_s=time,
        rows=frame.height,
        x_range_mm=(float(frame["x_mm"].min()), float(frame["x_mm"].max())),
        y_range_mm=(float(frame["y_mm"].min()), float(frame["y_mm"].max())),
        r_range_mm=(float(frame["r_mm"].min()), float(frame["r_mm"].max())),
        temperature_range_c=(
            float(frame["temperature_c"].min()),
            float(frame["temperature_c"].max()),
        ),
        radial_bins=int(frame["radial_bin"].n_unique()),
        radial_coordinate_max_error_mm=float(radial_error),
        recovered_true=int(frame["is_recovered"].sum()),
        recovered_false=int((~frame["is_recovered"]).sum()),
        max_temperature_anchors=int(frame["is_max_temperature_anchor"].sum()),
        null_values=sum(frame.null_count().row(0)),
    )


def radial_observations(
    path: Path,
    bin_width_mm: float = 0.25,
    inverse_variance_epsilon: float = 0.01,
    weight_clip: tuple[float, float] = (0.1, 10.0),
) -> pl.DataFrame:
    if bin_width_mm <= 0:
        raise ValueError("bin_width_mm must be positive")
    frame = pl.read_csv(
        path,
        columns=["r_mm", "temperature_c"],
        low_memory=True,
        schema_overrides=SCHEMA_OVERRIDES,
    )
    power, time = parse_ir_power_time(path)
    result = (
        frame.with_columns(
            (pl.col("r_mm") / bin_width_mm).floor().cast(pl.Int32).alias("radial_bin")
        )
        .group_by("radial_bin")
        .agg(
            pl.col("r_mm").mean().alias("r_mm"),
            pl.col("temperature_c").mean().alias("temperature_mean_c"),
            pl.col("temperature_c").std(ddof=0).fill_null(0.0).alias("temperature_std_c"),
            pl.len().alias("n_pixels"),
        )
        .sort("radial_bin")
        .with_columns(
            pl.lit(power).cast(pl.Float32).alias("power_w"),
            pl.lit(time).cast(pl.Float32).alias("time_s"),
            (
                1.0
                / (pl.col("temperature_std_c").pow(2) + inverse_variance_epsilon)
            )
            .clip(*weight_clip)
            .cast(pl.Float32)
            .alias("reliability_weight_raw"),
        )
        .with_columns(
            (
                pl.col("reliability_weight_raw")
                / pl.col("reliability_weight_raw").sum()
            )
            .cast(pl.Float32)
            .alias("frame_weight")
        )
    )
    return result.select(
        "power_w",
        "time_s",
        (pl.col("r_mm") / 1000.0).cast(pl.Float32).alias("r_m"),
        (pl.col("temperature_mean_c") + 273.15)
        .cast(pl.Float32)
        .alias("temperature_mean_k"),
        pl.col("temperature_std_c").cast(pl.Float32),
        pl.col("n_pixels").cast(pl.Int32),
        "reliability_weight_raw",
        "frame_weight",
    )


def angular_asymmetry_c(path: Path, radial_bins: int = 100) -> float:
    frame = pl.read_csv(
        path,
        columns=["r_mm", "temperature_c"],
        low_memory=True,
        schema_overrides=SCHEMA_OVERRIDES,
    )
    width = max(float(frame["r_mm"].max()) / radial_bins, np.finfo(float).eps)
    radial_std = (
        frame.with_columns((pl.col("r_mm") / width).floor().alias("bin"))
        .group_by("bin")
        .agg(pl.col("temperature_c").std(ddof=0).alias("std"))
    )
    return float(radial_std["std"].mean())
