from __future__ import annotations

import json
from typing import Any

import numpy as np

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.fields import load_processed_field


def _paired_interface_nodes(coordinates: np.ndarray, material_ids: np.ndarray) -> list[tuple[int, int]]:
    copper = {
        (round(float(r), 8), round(float(z), 8)): index
        for index, (r, z) in enumerate(coordinates)
        if material_ids[index] == 0
    }
    return [
        (index, copper[key])
        for index, (r, z) in enumerate(coordinates)
        if material_ids[index] == 1
        and (key := (round(float(r), 8), round(float(z), 8))) in copper
    ]


def audit_simulation_interface(
    output_path: str = "reports/simulation_interface_audit.json",
) -> dict[str, Any]:
    records = []
    pair_count = None
    global_max = (-1.0, None)
    for power in range(10, 801, 10):
        field = load_processed_field(float(power))
        pairs = _paired_interface_nodes(field.coordinates_rz_m, field.material_ids)
        pair_count = len(pairs) if pair_count is None else pair_count
        if len(pairs) != pair_count:
            raise RuntimeError("Interface pair count changes by power")
        jumps = np.stack(
            [field.temperature_k[:, sic] - field.temperature_k[:, copper] for sic, copper in pairs],
            axis=1,
        )
        absolute = np.abs(jumps)
        flat_index = int(np.argmax(absolute))
        time_index, pair_index = np.unravel_index(flat_index, absolute.shape)
        sic_index, copper_index = pairs[pair_index]
        record = {
            "power_w": float(power),
            "mean_abs_jump_k": float(absolute.mean()),
            "max_abs_jump_k": float(absolute[time_index, pair_index]),
            "signed_jump_k_at_max": float(jumps[time_index, pair_index]),
            "time_s_at_max": float(field.times_s[time_index]),
            "coordinate_rz_m_at_max": field.coordinates_rz_m[sic_index].tolist(),
            "sic_node_label": int(field.node_labels[sic_index]),
            "copper_node_label": int(field.node_labels[copper_index]),
        }
        records.append(record)
        if record["max_abs_jump_k"] > global_max[0]:
            global_max = (record["max_abs_jump_k"], record)
    result = {
        "schema_version": 1,
        "paired_same_coordinate_nodes": pair_count,
        "power_count": len(records),
        "global_max_abs_jump_k": global_max[0],
        "global_max_record": global_max[1],
        "per_power": records,
        "interpretation": (
            "A nonzero same-coordinate material-side temperature jump is evidence that perfect "
            "temperature continuity must not be assumed. Contact resistance or the original "
            "interface formulation must be checked; resistance is not identifiable from jumps alone."
        ),
        "material_passport": {
            "source": "processed low-fidelity simulation fields",
            "experiment_data_used": False,
            "contact_resistance_inferred": False,
            "heat_flux_available": False,
        },
    }
    destination = PROJECT_ROOT / output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
