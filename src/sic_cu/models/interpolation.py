from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from sic_cu.data.fields import SimulationField, assert_compatible_fields, load_processed_field


def _select_powers(query: float, available: list[float], kind: str) -> list[float]:
    powers = sorted(float(value) for value in available)
    if query in powers:
        return [query]
    lower = [value for value in powers if value < query]
    upper = [value for value in powers if value > query]
    if not lower or not upper:
        raise ValueError("Interpolation query is outside available power support")
    if kind == "linear":
        return [lower[-1], upper[0]]
    if kind == "cubic":
        nearest = sorted(powers, key=lambda value: abs(value - query))[:4]
        if len(nearest) < 4:
            raise ValueError("Cubic interpolation requires four powers")
        return sorted(nearest)
    raise ValueError(f"Unknown interpolation kind: {kind}")


def interpolate_simulation_power(
    power_w: float,
    available_powers_w: list[float],
    kind: str = "linear",
    root: Path | None = None,
) -> SimulationField:
    selected = _select_powers(float(power_w), available_powers_w, kind)
    fields = [load_processed_field(power, root) for power in selected]
    assert_compatible_fields(fields)
    reference = fields[0]
    if len(fields) == 1:
        temperature = reference.temperature_k.copy()
    elif kind == "linear":
        low, high = selected
        fraction = (float(power_w) - low) / (high - low)
        temperature = fields[0].temperature_k + fraction * (
            fields[1].temperature_k - fields[0].temperature_k
        )
    else:
        stacked = np.stack([field.temperature_k for field in fields], axis=0)
        temperature = CubicSpline(np.asarray(selected), stacked, axis=0)(float(power_w))
    return SimulationField(
        power_w=float(power_w),
        times_s=reference.times_s.copy(),
        coordinates_rz_m=reference.coordinates_rz_m.copy(),
        material_ids=reference.material_ids.copy(),
        node_labels=reference.node_labels.copy(),
        temperature_k=np.asarray(temperature, dtype=np.float32),
    )

