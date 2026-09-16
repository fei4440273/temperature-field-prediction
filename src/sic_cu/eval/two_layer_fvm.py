"""Independent conservative one-dimensional SiC/Cu plate reference solver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from sic_cu.config import PROJECT_ROOT, load_yaml


@dataclass(frozen=True)
class TwoLayerFvmResult:
    time_s: np.ndarray
    node_depth_m: np.ndarray
    material_id: np.ndarray
    temperature_k: np.ndarray
    top_surface_temperature_k: np.ndarray
    bottom_outward_flux_w_m2: np.ndarray
    storage_rate_per_area_w_m2: np.ndarray
    balance_per_area_w_m2: np.ndarray
    interface_flux_w_m2: np.ndarray
    interface_temperature_jump_k: np.ndarray


def solve_two_layer_fvm(
    registration_path: str | Path, *, sic_cells: int, cu_cells: int,
    dt_s: float, end_s: float, contact_multiplier: float = 1.0,
) -> TwoLayerFvmResult:
    if sic_cells < 2 or cu_cells < 2 or dt_s <= 0 or end_s <= 0:
        raise ValueError("Each material needs at least two cells and positive time steps")
    steps = round(end_s / dt_s)
    if not np.isclose(steps * dt_s, end_s, atol=1e-12, rtol=0):
        raise ValueError("The registered end time must be on the implicit time grid")
    registration = Path(registration_path)
    if not registration.is_absolute():
        registration = PROJECT_ROOT / registration
    setup = load_yaml(registration)
    if setup["stage"].split("；")[0] != "任-10独立求解器纯数值控制":
        raise ValueError("The reference must use the registered independent numerical case")
    synthetic_mismatch = float(setup["thermal_conditions"]["coarse_mismatch_contact_multiplier"])
    if contact_multiplier not in (1.0, synthetic_mismatch):
        raise ValueError("Contact variation is limited to the two preregistered synthetic cases")

    material_path = setup["materials_source"]
    materials = load_yaml(material_path)
    geometry = setup["geometry"]
    conditions = setup["thermal_conditions"]
    height_sic = float(geometry["silicon_carbide_thickness_m"])
    height_cu = float(geometry["copper_thickness_m"])
    initial = float(conditions["initial_temperature_k"])
    bottom_temperature = float(conditions["bottom_fixed_temperature_k"])
    flux = float(conditions["top_constant_heat_flux_w_m2"])
    contact = float(conditions["benchmark_contact_resistance_m2_k_w"]) * contact_multiplier
    if min(height_sic, height_cu, flux, contact) < 0 or height_sic == 0 or height_cu == 0:
        raise ValueError("Registered thickness, heat flux and contact resistance must be physical")

    widths = np.concatenate((
        np.full(sic_cells, height_sic / sic_cells),
        np.full(cu_cells, height_cu / cu_cells),
    ))
    si = materials["silicon_carbide"]
    cu = materials["copper"]
    conductivities = np.concatenate((
        np.full(sic_cells, float(si["conductivity_w_m_k"]["value"])),
        np.full(cu_cells, float(cu["conductivity_w_m_k"]["value"])),
    ))
    heat_capacity = np.concatenate((
        np.full(sic_cells, float(si["density_kg_m3"]) * float(si["heat_capacity_j_kg_k"]["value"])),
        np.full(cu_cells, float(cu["density_kg_m3"]) * float(cu["heat_capacity_j_kg_k"]["value"])),
    )) * widths
    if np.any(conductivities <= 0) or np.any(heat_capacity <= 0):
        raise ValueError("A material conductivity or volumetric heat capacity is not positive")
    face_resistance = (
        widths[:-1] / (2 * conductivities[:-1])
        + widths[1:] / (2 * conductivities[1:])
    )
    face_resistance[sic_cells - 1] += contact
    face_conductance = 1 / face_resistance
    bottom_conductance = 2 * conductivities[-1] / widths[-1]

    diagonal = heat_capacity / dt_s
    diagonal[:-1] += face_conductance
    diagonal[1:] += face_conductance
    diagonal[-1] += bottom_conductance
    solver = splu(diags(
        [-face_conductance, diagonal, -face_conductance],
        offsets=[-1, 0, 1], shape=(len(widths), len(widths)), format="csc",
    ))
    forcing = np.zeros(len(widths))
    forcing[0] = flux
    forcing[-1] += bottom_conductance * bottom_temperature
    times = np.linspace(0.0, end_s, steps + 1)
    field = np.empty((steps + 1, len(widths)))
    field[0] = initial
    top = np.full(steps + 1, initial)
    bottom_flux = np.zeros(steps + 1)
    storage = np.zeros(steps + 1)
    balance = np.zeros(steps + 1)
    interface_flux = np.zeros(steps + 1)
    interface_jump = np.zeros(steps + 1)
    for n in range(1, steps + 1):
        field[n] = solver.solve(heat_capacity / dt_s * field[n - 1] + forcing)
        storage[n] = np.dot(heat_capacity, field[n] - field[n - 1]) / dt_s
        bottom_flux[n] = bottom_conductance * (field[n, -1] - bottom_temperature)
        balance[n] = storage[n] + bottom_flux[n] - flux
        top[n] = field[n, 0] + flux * widths[0] / (2 * conductivities[0])
        interface_flux[n] = face_conductance[sic_cells - 1] * (
            field[n, sic_cells - 1] - field[n, sic_cells]
        )
        interface_jump[n] = (
            field[n, sic_cells - 1] - interface_flux[n] * widths[sic_cells - 1] / (2 * conductivities[sic_cells - 1])
            - field[n, sic_cells] - interface_flux[n] * widths[sic_cells] / (2 * conductivities[sic_cells])
        )
    centers = np.cumsum(widths) - widths / 2
    return TwoLayerFvmResult(
        time_s=times, node_depth_m=centers,
        material_id=np.concatenate((np.ones(sic_cells, dtype=np.int8), np.zeros(cu_cells, dtype=np.int8))),
        temperature_k=field, top_surface_temperature_k=top,
        bottom_outward_flux_w_m2=bottom_flux,
        storage_rate_per_area_w_m2=storage,
        balance_per_area_w_m2=balance, interface_flux_w_m2=interface_flux,
        interface_temperature_jump_k=interface_jump,
    )


def check_registered_fvm_controls(registration_path: str | Path) -> dict[str, object]:
    registration = Path(registration_path)
    if not registration.is_absolute():
        registration = PROJECT_ROOT / registration
    setup = load_yaml(registration)
    pairs = setup["reference_controls"]["mesh_time_pairs"]
    if len(pairs) != 4:
        raise ValueError("The registered numerical control requires four frozen refinements")
    grids = [(int(pair["silicon_carbide_cells"]), int(pair["copper_cells"]),
              float(pair["time_step_s"])) for pair in pairs]
    end = float(setup["reference_controls"]["end_time_s"])
    results = [solve_two_layer_fvm(
        registration, sic_cells=sic, cu_cells=cu, dt_s=dt, end_s=end,
    ) for sic, cu, dt in grids]
    times = list(setup["reference_controls"]["comparison_times_s"])
    if not times or any(time <= 0 or time > end for time in times):
        raise ValueError("The registered comparison times must be within the reference horizon")
    depths = [0.003, 0.009, 0.0119, 0.0121]
    depths.extend(float(value) for value in setup["geometry"]["virtual_probe_depths_from_top_m"])
    interface = float(setup["geometry"]["silicon_carbide_thickness_m"])
    samples = []
    for result, (_, _, dt) in zip(results[-2:], grids[-2:]):
        at_times = []
        for time in times:
            index = round(time / dt)
            if not np.isclose(result.time_s[index], time, atol=1e-12, rtol=0):
                raise ValueError("Every independent numerical comparison time must be on both grids")
            values = [float(result.top_surface_temperature_k[index])]
            for depth in depths:
                material = 1 if depth < interface else 0
                chosen = result.material_id == material
                values.append(float(np.interp(
                    depth, result.node_depth_m[chosen], result.temperature_k[index, chosen],
                )))
            at_times.append(values)
        samples.append(np.asarray(at_times, dtype=np.float64))
    difference = np.abs(samples[0] - samples[1])
    final = results[-1]
    conditions = setup["thermal_conditions"]
    geometry = setup["geometry"]
    materials = load_yaml(setup["materials_source"])
    steady = float(conditions["bottom_fixed_temperature_k"]) + float(
        conditions["top_constant_heat_flux_w_m2"]
    ) * (
        float(geometry["silicon_carbide_thickness_m"])
        / float(materials["silicon_carbide"]["conductivity_w_m_k"]["value"])
        + float(conditions["benchmark_contact_resistance_m2_k_w"])
        + float(geometry["copper_thickness_m"])
        / float(materials["copper"]["conductivity_w_m_k"]["value"])
    )
    return {
        "grid_steps": grids,
        "checked_comparison_times_s": [int(value) if int(value) == value else float(value) for value in times],
        "sampling_depths_m": [0.0, *depths],
        "fine_grid_max_difference_c": float(difference.max()),
        "fine_grid_max_step_balance_w_m2": float(np.max(np.abs(final.balance_per_area_w_m2[1:]))),
        "fine_grid_steady_top_error_c": float(abs(final.top_surface_temperature_k[-1] - steady)),
        "refinement_target_c": float(setup["reference_controls"]["refinement_target_max_difference_c"]),
        "refinement_target_met": bool(difference.max() < float(
            setup["reference_controls"]["refinement_target_max_difference_c"]
        )),
    }
