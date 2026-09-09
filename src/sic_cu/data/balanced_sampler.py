from __future__ import annotations

import numpy as np


TIME_BINS_S = ((0.0, 30.0), (30.0, 100.0), (100.0, np.inf))


def balanced_material_time_indices(
    material_ids: np.ndarray,
    times_s: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a condition with equal material/time-stratum targets."""
    materials = np.asarray(material_ids).reshape(-1)
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    if materials.shape != times.shape:
        raise ValueError("material_ids and times_s must have the same shape")
    if count <= 0 or count > len(materials):
        raise ValueError("count must be in [1, number of rows]")
    unique_materials = sorted(int(value) for value in np.unique(materials))
    if unique_materials != [0, 1]:
        raise ValueError("Balanced simulation sampling requires both Cu=0 and SiC=1")

    strata: list[np.ndarray] = []
    for material_id in unique_materials:
        for lower, upper in TIME_BINS_S:
            mask = (materials == material_id) & (times >= lower)
            if np.isfinite(upper):
                mask &= times < upper
            indices = np.flatnonzero(mask)
            if len(indices):
                strata.append(indices)
    if len(strata) < 2:
        raise ValueError("Simulation condition does not cover material/time strata")

    base, remainder = divmod(count, len(strata))
    selected: list[np.ndarray] = []
    for index, candidates in enumerate(strata):
        requested = base + int(index < remainder)
        take = min(requested, len(candidates))
        if take:
            selected.append(rng.choice(candidates, size=take, replace=False))
    chosen = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    if len(chosen) < count:
        remaining = np.setdiff1d(np.arange(len(materials)), chosen, assume_unique=False)
        chosen = np.concatenate(
            (chosen, rng.choice(remaining, size=count - len(chosen), replace=False))
        )
    rng.shuffle(chosen)
    return chosen.astype(np.int64, copy=False)


def material_sample_counts(coordinates: np.ndarray) -> dict[str, int]:
    values = np.asarray(coordinates)[:, 4]
    return {
        "copper": int(np.count_nonzero(values < 0.5)),
        "silicon_carbide": int(np.count_nonzero(values >= 0.5)),
    }
