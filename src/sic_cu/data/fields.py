from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from sic_cu.config import PROJECT_ROOT


@dataclass(frozen=True)
class SimulationField:
    power_w: float
    times_s: np.ndarray
    coordinates_rz_m: np.ndarray
    material_ids: np.ndarray
    node_labels: np.ndarray
    temperature_k: np.ndarray


def load_processed_field(power_w: float, root: Path | None = None) -> SimulationField:
    base = PROJECT_ROOT / "data/processed/simulation" if root is None else Path(root)
    path = base / f"{float(power_w):g}W.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pl.read_parquet(path).sort("time_s", "material_id", "node_label")
    times = np.sort(frame["time_s"].unique().to_numpy())
    first = frame.filter(pl.col("time_s") == float(times[0]))
    nodes = first.height
    if frame.height != len(times) * nodes:
        raise ValueError(f"Nonrectangular trajectory in {path}")
    temperature = frame["temperature_k"].to_numpy().reshape(len(times), nodes)
    return SimulationField(
        power_w=float(power_w),
        times_s=times.astype(np.float32),
        coordinates_rz_m=first.select("r_m", "z_m").to_numpy().astype(np.float32),
        material_ids=first["material_id"].to_numpy().astype(np.int64),
        node_labels=first["node_label"].to_numpy().astype(np.int64),
        temperature_k=temperature.astype(np.float32),
    )


def assert_compatible_fields(fields: list[SimulationField]) -> None:
    if not fields:
        raise ValueError("At least one field is required")
    reference = fields[0]
    for field in fields[1:]:
        if not np.array_equal(field.times_s, reference.times_s):
            raise ValueError("Simulation time grids do not match")
        if not np.array_equal(field.material_ids, reference.material_ids):
            raise ValueError("Simulation material masks do not match")
        if not np.array_equal(field.node_labels, reference.node_labels):
            raise ValueError("Simulation node labels do not match")
        if not np.allclose(field.coordinates_rz_m, reference.coordinates_rz_m, atol=1e-9):
            raise ValueError("Simulation meshes do not match")

