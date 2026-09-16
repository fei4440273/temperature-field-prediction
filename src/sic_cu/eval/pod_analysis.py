from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.fields import assert_compatible_fields, load_processed_field
from sic_cu.data.splits import build_power_splits
from sic_cu.models.pod_pinn import PODBasis, fit_pod, save_pod_basis


def _threshold_modes(cumulative_energy: np.ndarray) -> dict[str, int | None]:
    return {
        f"{target * 100:g}%": (
            int(np.searchsorted(cumulative_energy, target) + 1)
            if cumulative_energy[-1] >= target
            else None
        )
        for target in (0.95, 0.99, 0.999)
    }


def _summary(basis: PODBasis) -> dict[str, Any]:
    candidates = (5, 10, 20, 30, 50)
    return {
        "nodes": int(basis.mesh_rz_m.shape[0]),
        "stored_modes": int(basis.modes.shape[1]),
        "modes_at_energy_threshold": _threshold_modes(basis.cumulative_energy),
        "candidate_cumulative_energy": {
            str(count): float(basis.cumulative_energy[min(count, len(basis.cumulative_energy)) - 1])
            for count in candidates
        },
    }


def run_pod_analysis(
    output_directory: str = "data/cache/pod",
    report_path: str = "reports/pod_energy.json",
    max_modes: int = 50,
    plot_path: str = "reports/pod_energy.png",
) -> dict[str, Any]:
    splits = build_power_splits()
    powers = sorted(splits.simulation_train)
    fields = [load_processed_field(power) for power in powers]
    assert_compatible_fields(fields)
    snapshots = np.concatenate([field.temperature_k for field in fields], axis=0)
    reference = fields[0]
    output = PROJECT_ROOT / output_directory
    if output.exists() and not output.is_dir():
        raise ValueError(
            f"POD output must be a directory, but an existing file was provided: {output}"
        )

    bases: dict[str, PODBasis] = {
        "global": fit_pod(
            snapshots,
            reference.coordinates_rz_m,
            max_modes=max_modes,
            material_ids=reference.material_ids,
        )
    }
    for name, material_id in (("copper", 0), ("silicon_carbide", 1)):
        mask = reference.material_ids == material_id
        bases[name] = fit_pod(
            snapshots[:, mask],
            reference.coordinates_rz_m[mask],
            max_modes=max_modes,
            material_ids=reference.material_ids[mask],
        )
    for name, basis in bases.items():
        save_pod_basis(basis, output / f"{name}.npz")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for name, basis in bases.items():
        shown = min(max_modes, len(basis.cumulative_energy))
        axis.plot(
            np.arange(1, shown + 1),
            basis.cumulative_energy[:shown],
            label=name.replace("_", " "),
        )
    for threshold in (0.95, 0.99, 0.999):
        axis.axhline(threshold, color="0.6", linewidth=0.8, linestyle="--")
    axis.set(
        xlabel="Number of POD modes",
        ylabel="Cumulative energy",
        xlim=(1, max_modes),
        ylim=(0.94, 1.0001),
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure_destination = PROJECT_ROOT / plot_path
    figure_destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_destination, dpi=200)
    plt.close(figure)
    result = {
        "schema_version": 1,
        "fit_scope": "simulation_training_powers_only",
        "training_powers_w": powers,
        "snapshots": int(snapshots.shape[0]),
        "variants": {name: _summary(basis) for name, basis in bases.items()},
        "warning": "POD bases must be fitted on training powers only; validation and test powers remain held out.",
        "energy_plot": str(figure_destination.relative_to(PROJECT_ROOT)),
    }
    destination = PROJECT_ROOT / report_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
