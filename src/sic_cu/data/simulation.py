from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .common import parse_power


EXPECTED_COLUMNS = (
    "power_W",
    "time_s",
    "part_instance",
    "node_label",
    "x",
    "y",
    "temperature",
)

SCHEMA_OVERRIDES = {
    "power_W": pl.Float64,
    "time_s": pl.Float64,
    "part_instance": pl.String,
    "node_label": pl.Int64,
    "x": pl.Float64,
    "y": pl.Float64,
    "temperature": pl.Float64,
}


@dataclass(frozen=True)
class SimulationFileAudit:
    path: str
    filename_power_w: float
    column_power_w: float
    rows: int
    time_frames: int
    time_min_s: float
    time_max_s: float
    max_time_grid_deviation_s: float
    nodes_per_frame_min: int
    nodes_per_frame_max: int
    copper_nodes: int
    sic_nodes: int
    r_range_raw: tuple[float, float]
    z_range_raw: tuple[float, float]
    temperature_range_c: tuple[float, float]
    initial_temperature_range_c: tuple[float, float]
    initial_temperature_std_c: float
    null_values: int
    mesh_signature: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def simulation_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*W_temperature_coordinates.csv"), key=parse_power)


def _mesh_signature(frame: pl.DataFrame) -> str:
    ordered = frame.select("part_instance", "node_label", "x", "y").sort(
        "part_instance", "node_label"
    )
    return hashlib.sha256(ordered.write_csv().encode("utf-8")).hexdigest()


def audit_simulation_file(path: Path) -> SimulationFileAudit:
    frame = pl.read_csv(
        path,
        low_memory=True,
        rechunk=False,
        schema_overrides=SCHEMA_OVERRIDES,
    )
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected simulation schema in {path}: {frame.columns}")
    times = np.sort(frame["time_s"].unique().to_numpy().astype(float))
    expected_times = np.arange(len(times), dtype=float) * 2.0
    max_deviation = float(np.max(np.abs(times - expected_times)))
    counts = frame.group_by("time_s").len()["len"]
    first = frame.filter(pl.col("time_s") == float(times[0]))
    material_counts = dict(
        first.group_by("part_instance").len().iter_rows()
    )
    powers = frame["power_W"].unique().to_list()
    if len(powers) != 1:
        raise ValueError(f"Multiple power values in {path}: {powers}")
    return SimulationFileAudit(
        path=str(path),
        filename_power_w=parse_power(path),
        column_power_w=float(powers[0]),
        rows=frame.height,
        time_frames=len(times),
        time_min_s=float(times.min()),
        time_max_s=float(times.max()),
        max_time_grid_deviation_s=max_deviation,
        nodes_per_frame_min=int(counts.min()),
        nodes_per_frame_max=int(counts.max()),
        copper_nodes=int(material_counts.get("CU-1", 0)),
        sic_nodes=int(material_counts.get("SIC-1", 0)),
        r_range_raw=(float(frame["x"].min()), float(frame["x"].max())),
        z_range_raw=(float(frame["y"].min()), float(frame["y"].max())),
        temperature_range_c=(
            float(frame["temperature"].min()),
            float(frame["temperature"].max()),
        ),
        initial_temperature_range_c=(
            float(first["temperature"].min()),
            float(first["temperature"].max()),
        ),
        initial_temperature_std_c=float(first["temperature"].std()),
        null_values=sum(frame.null_count().row(0)),
        mesh_signature=_mesh_signature(first),
    )


def read_simulation(path: Path) -> pl.DataFrame:
    """Read one raw simulation trajectory into the canonical SI/K schema."""
    frame = pl.read_csv(path, low_memory=True, schema_overrides=SCHEMA_OVERRIDES)
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected simulation schema in {path}")
    return frame.select(
        pl.col("power_W").cast(pl.Float32).alias("power_w"),
        pl.col("time_s").cast(pl.Float32),
        (pl.col("x") / 1000.0).cast(pl.Float32).alias("r_m"),
        (pl.col("y") / 1000.0).cast(pl.Float32).alias("z_m"),
        pl.when(pl.col("part_instance") == "SIC-1")
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("material_id"),
        pl.col("node_label").cast(pl.Int32),
        (pl.col("temperature") + 273.15).cast(pl.Float32).alias("temperature_k"),
    )
