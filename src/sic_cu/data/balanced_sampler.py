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


def balanced_material_time_space_indices(
    material_ids: np.ndarray,
    times_s: np.ndarray,
    radial_m: np.ndarray,
    depth_m: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    materials = np.asarray(material_ids).reshape(-1)
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    radial = np.asarray(radial_m, dtype=np.float64).reshape(-1)
    depth = np.asarray(depth_m, dtype=np.float64).reshape(-1)
    if any(len(field) != len(materials) for field in (times, radial, depth)):
        raise ValueError("Balanced material/time/space fields must align")
    if not np.isfinite(radial).all() or not np.isfinite(depth).all():
        raise ValueError("Spatial coordinates must be finite")
    if count <= 0 or count > len(materials):
        raise ValueError("count must be in [1, number of rows]")
    if sorted(int(value) for value in np.unique(materials)) != [0, 1]:
        raise ValueError("Balanced simulation sampling requires both Cu=0 and SiC=1")

    strata: list[np.ndarray] = []
    for material_id in (0, 1):
        for lower, upper in TIME_BINS_S:
            mask = (materials == material_id) & (times >= lower)
            if np.isfinite(upper):
                mask &= times < upper
            candidates = np.flatnonzero(mask)
            if len(candidates):
                strata.append(candidates)
    if len(strata) < 2:
        raise ValueError("Simulation condition does not cover material/time strata")

    base, remainder = divmod(count, len(strata))
    selected: list[np.ndarray] = []
    for index, candidates in enumerate(strata):
        requested = min(base + int(index < remainder), len(candidates))
        radial_mid = np.median(radial[candidates])
        depth_mid = np.median(depth[candidates])
        quadrants = (
            (radial[candidates] >= radial_mid).astype(np.int8) * 2
            + (depth[candidates] >= depth_mid).astype(np.int8)
        )
        groups = [candidates[quadrants == quadrant] for quadrant in range(4)]
        groups = [group for group in groups if len(group)]
        group_base, extra = divmod(requested, len(groups))
        choices = [
            rng.choice(group, size=min(group_base + int(position < extra), len(group)), replace=False)
            for position, group in enumerate(groups)
        ]
        chosen = np.concatenate(choices)
        if len(chosen) < requested:
            remaining = np.setdiff1d(candidates, chosen, assume_unique=False)
            chosen = np.concatenate((
                chosen, rng.choice(remaining, size=requested - len(chosen), replace=False),
            ))
        selected.append(chosen)
    chosen = np.concatenate(selected)
    if len(chosen) < count:
        remaining = np.setdiff1d(np.arange(len(materials)), chosen, assume_unique=False)
        chosen = np.concatenate((
            chosen, rng.choice(remaining, size=count - len(chosen), replace=False),
        ))
    rng.shuffle(chosen)
    return chosen.astype(np.int64, copy=False)


def material_sample_counts(coordinates: np.ndarray) -> dict[str, int]:
    values = np.asarray(coordinates)[:, 4]
    return {
        "copper": int(np.count_nonzero(values < 0.5)),
        "silicon_carbide": int(np.count_nonzero(values >= 0.5)),
    }
