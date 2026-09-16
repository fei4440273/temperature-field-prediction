from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.physics.materials import PhysicsConfigurationError

from .common import parse_power


@dataclass(frozen=True)
class SensorFileAudit:
    path: str
    sensor_type: str
    power_w: float
    rows: int
    time_columns: int
    time_min_raw: float
    time_max_raw: float
    x_range_raw: tuple[float, float]
    y_range_raw: tuple[float, float]
    radius_mean_raw: float
    radius_std_raw: float
    radius_min_raw: float
    radius_max_raw: float
    value_range_raw: tuple[float, float]
    initial_value_mean_raw: float
    initial_value_std_raw: float
    max_angular_spread_raw: float
    all_ring_values_identical: bool
    nan_values: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sensor_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.csv"), key=parse_power)


TIME_COLUMN_RE = re.compile(r"t=(\d+(?:\.\d+)?)")


def _read_sensor_matrix(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [[float(value) if value else np.nan for value in row] for row in reader]
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(header):
        raise ValueError(f"Inconsistent sensor row width in {path}")
    time_matches = [TIME_COLUMN_RE.fullmatch(name) for name in header[2:]]
    if header[:2] != ["X", "Y"] or any(match is None for match in time_matches):
        raise ValueError(f"Unexpected sensor schema in {path}")
    times = np.asarray([float(match.group(1)) for match in time_matches], dtype=np.float64)
    if len(times) == 0 or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"Sensor time columns must be strictly increasing in {path}")
    return header, matrix, times


def audit_sensor_file(path: Path, sensor_type: str) -> SensorFileAudit:
    header, matrix, times = _read_sensor_matrix(path)
    radius = np.hypot(matrix[:, 0], matrix[:, 1])
    values = matrix[:, 2:]
    angular_spread = np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
    return SensorFileAudit(
        path=str(path),
        sensor_type=sensor_type,
        power_w=parse_power(path),
        rows=matrix.shape[0],
        time_columns=len(header) - 2,
        time_min_raw=float(times[0]),
        time_max_raw=float(times[-1]),
        x_range_raw=(float(np.nanmin(matrix[:, 0])), float(np.nanmax(matrix[:, 0]))),
        y_range_raw=(float(np.nanmin(matrix[:, 1])), float(np.nanmax(matrix[:, 1]))),
        radius_mean_raw=float(np.nanmean(radius)),
        radius_std_raw=float(np.nanstd(radius)),
        radius_min_raw=float(np.nanmin(radius)),
        radius_max_raw=float(np.nanmax(radius)),
        value_range_raw=(float(np.nanmin(values)), float(np.nanmax(values))),
        initial_value_mean_raw=float(np.nanmean(values[:, 0])),
        initial_value_std_raw=float(np.nanstd(values[:, 0])),
        max_angular_spread_raw=float(np.nanmax(angular_spread)),
        all_ring_values_identical=bool(np.nanmax(angular_spread) <= 1e-12),
        nan_values=int(np.isnan(matrix).sum()),
    )


def ring_average_raw(path: Path, sensor_type: str) -> pl.DataFrame:
    """Collapse angular duplicates without assigning unverified units."""
    header, matrix, times = _read_sensor_matrix(path)
    values = matrix[:, 2:]
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    return pl.DataFrame(
        {
            "power_w": np.full(len(mean), parse_power(path), dtype=np.float32),
            "sample_index": np.arange(1, len(mean) + 1, dtype=np.int32),
            "time_raw": times,
            "sensor_type": [sensor_type] * len(mean),
            "radius_raw": np.full(
                len(mean), np.mean(np.hypot(matrix[:, 0], matrix[:, 1])), dtype=np.float64
            ),
            "value_mean_raw": mean,
            "value_std_raw": std,
            "n_angular_samples": np.full(len(mean), matrix.shape[0], dtype=np.int32),
            "delta_value_raw": mean - mean[0],
            "source_last_header": [header[-1]] * len(mean),
        }
    )


def load_canonical_sensor_observations(
    metadata_path: str = "configs/data_metadata.yaml",
    processed_path: str = "data/processed/sensor_ring_raw.parquet",
    *,
    split: str | None = None,
    test_processed_path: str = "data/processed/test_sensor_ring_raw.parquet",
) -> pl.DataFrame:
    """Convert the ring table only after coordinate, value, and time semantics are verified."""
    metadata = load_yaml(metadata_path)["sensors"]
    required_statuses = (
        "coordinate_unit_status",
        "value_unit_status",
        "time_unit_status",
    )
    if any(metadata.get(key) != "verified" for key in required_statuses):
        raise PhysicsConfigurationError(
            "Hot/Cold coordinate, value, and time units must all be verified before use"
        )
    if metadata.get("synchronized_start") is not True:
        raise PhysicsConfigurationError("Hot/Cold synchronized_start must be confirmed true")
    coordinate_unit = metadata.get("coordinate_unit")
    coordinate_scale = {"m": 1.0, "mm": 1e-3}.get(coordinate_unit)
    if coordinate_scale is None:
        raise PhysicsConfigurationError(f"Unsupported sensor coordinate unit: {coordinate_unit}")
    time_unit = metadata.get("time_unit")
    time_scale = {"s": 1.0, "ms": 1e-3}.get(time_unit)
    if time_scale is None:
        raise PhysicsConfigurationError(f"Unsupported sensor time unit: {time_unit}")
    interpretation = metadata.get("time_column_interpretation")
    if interpretation != "elapsed_time_t_equals_index":
        raise PhysicsConfigurationError(
            "Sensor time_column_interpretation must be elapsed_time_t_equals_index"
        )
    value_unit = metadata.get("value_unit")
    if value_unit not in {"degC", "K"}:
        raise PhysicsConfigurationError(f"Unsupported sensor value unit: {value_unit}")
    if split not in {None, "train", "validation", "test", "external_test"}:
        raise ValueError("Unsupported sensor split")
    selected_path = test_processed_path if split == "test" else processed_path
    source = Path(selected_path)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    frame = pl.read_parquet(source)
    temperature = pl.col("value_mean_raw")
    if value_unit == "degC":
        temperature = temperature + 273.15
    time_column = "time_raw" if "time_raw" in frame.columns else "sample_index"
    if "source_dataset" not in frame.columns:
        frame = frame.with_columns(pl.lit("legacy").alias("source_dataset"))
    canonical = frame.with_columns(
        (pl.col(time_column) * time_scale).cast(pl.Float32).alias("time_s"),
        (pl.col("radius_raw") * coordinate_scale).cast(pl.Float32).alias("r_m"),
        temperature.cast(pl.Float32).alias("temperature_k"),
        pl.col("delta_value_raw").cast(pl.Float32).alias("delta_temperature_k"),
    ).select(
        "power_w",
        "time_s",
        "sensor_type",
        "r_m",
        "temperature_k",
        "delta_temperature_k",
        "split",
        "source_dataset",
    )
    return canonical if split is None else canonical.filter(pl.col("split") == split)
