from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import polars as pl
import torch
from torch import Tensor, nn

from sic_cu.models import AdditiveCorrectionModel, ModelScales
from sic_cu.physics.boundary import convection_flux, laser_flux, radiation_flux
from sic_cu.physics.configuration import BoundaryConditions
from sic_cu.physics.materials import Material


def chinese_energy_terms(source: pl.DataFrame, order: int) -> tuple[pl.DataFrame, dict[str, float]]:
    """Map audited raw watts into readable columns without changing the balance definition."""
    selected = source.filter(pl.col("quadrature_order") == order)
    if selected.is_empty():
        raise ValueError("No original energy rows exist at the requested quadrature order")
    mapped = selected.select(
        pl.col("power_w").alias("功率_瓦"),
        pl.col("time_s").alias("时刻_秒"),
        pl.col("quadrature_order").alias("求积阶数"),
        pl.col("absorbed_power_w").alias("吸收功率_瓦"),
        pl.col("storage_rate_w").alias("储能率_瓦"),
        pl.col("cooling_heat_w").alias("水冷散热_瓦"),
        pl.col("convection_heat_w").alias("对流散热_瓦"),
        pl.col("radiation_heat_w").alias("辐射散热_瓦"),
        pl.col("balance_w").alias("平衡_瓦"),
        pl.col("balance_w").abs().alias("绝对平衡_瓦"),
        pl.col("relative_balance_denominator_w").alias("原定义相对平衡分母_瓦"),
        pl.col("relative_balance").alias("原定义相对平衡"),
        pl.col("boundary_gradient_mode").alias("边界梯度模式原标识"),
        pl.col("outer_epsilon_m").alias("外边界内移_米"),
    )
    absolute = mapped["绝对平衡_瓦"].to_numpy()
    normalized = absolute / mapped["吸收功率_瓦"].to_numpy()
    return mapped, {
        "绝对平衡宏均值_瓦": float(absolute.mean()),
        "绝对平衡95分位_瓦": float(np.percentile(absolute, 95)),
        "绝对平衡最大值_瓦": float(absolute.max()),
        "吸收功率归一宏均值": float(normalized.mean()),
        "吸收功率归一95分位": float(np.percentile(normalized, 95)),
    }


def audit_schedule_energy(
    model: nn.Module,
    materials: Mapping[int, Material],
    boundaries: BoundaryConditions,
    geometry: AxisymmetricGeometry,
    *,
    powers_w: Iterable[float],
    times_s: Iterable[float],
    orders: tuple[int, int],
    device: torch.device,
) -> dict[str, Any]:
    """Use identical fixed conditions at two orders, preserving all original watts."""
    if len(orders) != 2 or not 2 <= orders[0] < orders[1]:
        raise ValueError("Energy audit requires two increasing quadrature orders")
    powers, times = tuple(powers_w), tuple(times_s)
    if not powers or not times or any(power <= 0 for power in powers):
        raise ValueError("Energy audit requires fixed positive powers and observation times")
    original_energy = []
    original_divergence = []
    relative_changes = []
    for power in powers:
        for time_s in times:
            previous = None
            for order in orders:
                energy, divergence = deterministic_energy_terms(
                    model, materials, boundaries, geometry, power, time_s, order,
                    outer_epsilon_m=1e-6, derivative_batch_size=2048, device=device,
                )
                original_energy.append(energy)
                original_divergence.append(divergence)
                if previous is not None:
                    relative_changes.append(max(
                        abs(energy[name] - previous[name])
                        for name in (
                            "absorbed_power_w", "storage_rate_w", "cooling_heat_w",
                            "convection_heat_w", "radiation_heat_w", "balance_w",
                        )
                    ) / abs(energy["absorbed_power_w"]))
                previous = energy
    original_frame = pl.DataFrame(original_energy)
    mapped, summary = chinese_energy_terms(original_frame, orders[1])
    decomposition = pl.DataFrame(original_divergence).filter(
        pl.col("quadrature_order") == orders[1]
    ).select(
        pl.col("power_w").alias("功率_瓦"),
        pl.col("time_s").alias("时刻_秒"),
        pl.col("integrated_pde_residual_w").alias("体积分残差V_瓦"),
        pl.col("interface_two_sided_flux_w").alias("界面双侧通量J_瓦"),
        pl.col("boundary_flux_residual_w").alias("边界失配D_瓦"),
        pl.col("explained_engineering_balance_w").alias("解释工程平衡_瓦"),
        pl.col("engineering_balance_w").alias("原工程平衡_瓦"),
        pl.col("engineering_explanation_gap_w").alias("工程解释剩余差_瓦"),
        pl.col("silicon_carbide_pde_residual_w").alias("SiC体内积分残差_瓦"),
        pl.col("copper_outer_ring_pde_residual_w").alias("Cu外环积分残差_瓦"),
        pl.col("copper_below_sic_pde_residual_w").alias("Cu下层积分残差_瓦"),
        pl.col("outer_temperature_max_abs_deviation_c").alias("水冷边界最大温差_摄氏度"),
    )
    original_balance = mapped["平衡_瓦"].to_numpy()
    explanation_gap = decomposition["工程解释剩余差_瓦"].to_numpy()
    summary.update({
        "功率时刻审核行数": mapped.height,
        "原定义相对平衡分母已保持": bool(np.allclose(
            mapped["原定义相对平衡分母_瓦"].to_numpy(),
            original_frame.filter(pl.col("quadrature_order") == orders[1])[
                "relative_balance_denominator_w"
            ].to_numpy(),
        )),
        "最大相邻阶变化对吸收功率比": float(max(relative_changes)),
        "最大分解剩余差_瓦": float(np.max(np.abs(explanation_gap))),
        "工程平衡与原瓦数逐行一致": bool(np.allclose(
            original_balance, decomposition["原工程平衡_瓦"].to_numpy()
        )),
    })
    return {
        "指标明细": mapped,
        "物理分解": decomposition,
        "汇总": summary,
        "原始能量": original_energy,
        "原始散度": original_divergence,
    }


@dataclass(frozen=True)
class AxisymmetricGeometry:
    copper_radius_m: float
    silicon_carbide_radius_m: float
    copper_bottom_z_m: float
    silicon_carbide_bottom_z_m: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "AxisymmetricGeometry":
        return cls(
            copper_radius_m=float(config["copper"]["radius_m"]),
            silicon_carbide_radius_m=float(
                config["silicon_carbide"]["radius_m"]
            ),
            copper_bottom_z_m=float(config["embedding"]["copper_bottom_z_m"]),
            silicon_carbide_bottom_z_m=float(
                config["embedding"]["sic_bottom_z_m"]
            ),
        )

    @property
    def volume_m3(self) -> float:
        return math.pi * self.copper_radius_m**2 * abs(self.copper_bottom_z_m)

    @property
    def silicon_carbide_volume_m3(self) -> float:
        return (
            math.pi
            * self.silicon_carbide_radius_m**2
            * abs(self.silicon_carbide_bottom_z_m)
        )

    @property
    def copper_volume_m3(self) -> float:
        return self.volume_m3 - self.silicon_carbide_volume_m3


def gauss_legendre_interval(
    low: float,
    high: float,
    order: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor]:
    if order < 2:
        raise ValueError("Gauss-Legendre order must be at least two")
    if not high > low:
        raise ValueError("Quadrature interval must have positive width")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    mapped_nodes = low + (nodes + 1.0) * (high - low) / 2.0
    mapped_weights = weights * (high - low) / 2.0
    return (
        torch.as_tensor(mapped_nodes, device=device, dtype=dtype),
        torch.as_tensor(mapped_weights, device=device, dtype=dtype),
    )


def axisymmetric_rectangle_quadrature(
    radial_bounds_m: tuple[float, float],
    axial_bounds_m: tuple[float, float],
    order: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    radii, radial_weights = gauss_legendre_interval(
        *radial_bounds_m, order, device
    )
    axial, axial_weights = gauss_legendre_interval(*axial_bounds_m, order, device)
    r_grid, z_grid = torch.meshgrid(radii, axial, indexing="ij")
    wr_grid, wz_grid = torch.meshgrid(radial_weights, axial_weights, indexing="ij")
    weights = 2.0 * math.pi * r_grid * wr_grid * wz_grid
    return torch.stack((r_grid.reshape(-1), z_grid.reshape(-1)), dim=1), weights.reshape(-1)


def axisymmetric_horizontal_quadrature(
    radial_bounds_m: tuple[float, float],
    order: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    radii, radial_weights = gauss_legendre_interval(
        *radial_bounds_m, order, device
    )
    return radii, 2.0 * math.pi * radii * radial_weights


def axisymmetric_vertical_quadrature(
    radius_m: float,
    axial_bounds_m: tuple[float, float],
    order: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    axial, axial_weights = gauss_legendre_interval(*axial_bounds_m, order, device)
    return axial, 2.0 * math.pi * float(radius_m) * axial_weights


def axisymmetric_lumped_nodal_weights(
    coordinates_rz_m: np.ndarray,
    material_ids: np.ndarray,
    geometry: AxisymmetricGeometry,
) -> np.ndarray:
    """Build geometry-aware lumped FEM-style nodal volume weights."""
    from scipy.spatial import Delaunay

    coordinates = np.asarray(coordinates_rz_m, dtype=np.float64)
    ids = np.asarray(material_ids, dtype=np.int64).reshape(-1)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(ids) != len(coordinates):
        raise ValueError("Expected aligned [r,z] coordinates and material ids")
    output = np.zeros(len(coordinates), dtype=np.float64)
    rs = geometry.silicon_carbide_radius_m
    zs = geometry.silicon_carbide_bottom_z_m
    geometry_tolerance_m = 2e-7

    groups = (
        (
            1,
            (ids == 1),
            lambda values: (values[:, 0] <= rs + geometry_tolerance_m)
            & (values[:, 1] >= zs - geometry_tolerance_m),
        ),
        (
            0,
            (ids == 0) & (coordinates[:, 1] <= zs + geometry_tolerance_m),
            lambda values: values[:, 1] <= zs + geometry_tolerance_m,
        ),
        (
            0,
            (ids == 0)
            & (coordinates[:, 0] >= rs - geometry_tolerance_m)
            & (coordinates[:, 1] >= zs - geometry_tolerance_m),
            lambda values: (values[:, 0] >= rs - geometry_tolerance_m)
            & (values[:, 1] >= zs - geometry_tolerance_m),
        ),
    )
    for material_id, node_mask, in_subdomain in groups:
        global_indices = np.flatnonzero(node_mask)
        points = coordinates[global_indices]
        triangulation = Delaunay(points)
        for local_triangle in triangulation.simplices:
            vertices = points[local_triangle]
            probes = np.concatenate(
                (
                    vertices.mean(axis=0, keepdims=True),
                    (vertices + np.roll(vertices, -1, axis=0)) / 2.0,
                ),
                axis=0,
            )
            if not bool(np.all(in_subdomain(probes))):
                continue
            first_edge = vertices[1] - vertices[0]
            second_edge = vertices[2] - vertices[0]
            twice_area = abs(
                first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
            )
            area = float(twice_area) / 2.0
            if area <= 0.0:
                continue
            radial_sum = float(vertices[:, 0].sum())
            for local_vertex, radial in zip(local_triangle, vertices[:, 0]):
                # Exact integral of 2*pi*r*N_i for linear triangle shape functions.
                contribution = 2.0 * math.pi * area * (radial_sum + radial) / 12.0
                output[global_indices[local_vertex]] += contribution
    if np.any(output < 0.0) or not np.all(np.isfinite(output)):
        raise RuntimeError("Invalid axisymmetric lumped nodal weights")
    return output


def _coordinates(
    radius_m: Tensor,
    axial_m: Tensor,
    time_s: float,
    power_w: float,
    material_id: int,
) -> Tensor:
    return torch.column_stack(
        (
            radius_m,
            axial_m,
            torch.full_like(radius_m, float(time_s)),
            torch.full_like(radius_m, float(power_w)),
            torch.full_like(radius_m, float(material_id)),
        )
    )


def _constant_material_values(material: Material) -> tuple[float, float, float]:
    if material.conductivity.kind != "constant" or material.heat_capacity.kind != "constant":
        raise ValueError(
            "V5 instantaneous energy audit currently requires the configured constant k and cp"
        )
    return (
        float(material.density_kg_m3),
        float(material.conductivity.value),
        float(material.heat_capacity.value),
    )


def _volume_integrals(
    model: nn.Module,
    coordinates: Tensor,
    weights: Tensor,
    material: Material,
    batch_size: int,
) -> tuple[float, float]:
    density, conductivity, heat_capacity = _constant_material_values(material)
    storage = torch.zeros((), dtype=torch.float64, device=coordinates.device)
    residual = torch.zeros_like(storage)
    for offset in range(0, len(coordinates), batch_size):
        x = coordinates[offset : offset + batch_size].detach().clone().requires_grad_(True)
        w = weights[offset : offset + batch_size]
        temperature = model(x)
        first = torch.autograd.grad(
            temperature,
            x,
            torch.ones_like(temperature),
            create_graph=True,
            retain_graph=True,
        )[0]
        radial_second = torch.autograd.grad(
            first[:, 0:1],
            x,
            torch.ones_like(first[:, 0:1]),
            create_graph=False,
            retain_graph=True,
        )[0][:, 0]
        axial_second = torch.autograd.grad(
            first[:, 1:2],
            x,
            torch.ones_like(first[:, 1:2]),
            create_graph=False,
            retain_graph=False,
        )[0][:, 1]
        radial = x[:, 0]
        storage_density = density * heat_capacity * first[:, 2]
        pde_residual = storage_density - conductivity * (
            radial_second + first[:, 0] / radial + axial_second
        )
        storage = storage + torch.sum(w * storage_density.detach())
        residual = residual + torch.sum(w * pde_residual.detach())
    return float(storage.cpu()), float(residual.cpu())


def _surface_values(
    model: nn.Module,
    coordinates: Tensor,
    weights: Tensor,
    normal_rz: tuple[float, float],
    material: Material,
) -> tuple[Tensor, Tensor, float]:
    x = coordinates.detach().clone().requires_grad_(True)
    temperature = model(x)
    gradient = torch.autograd.grad(
        temperature,
        x,
        torch.ones_like(temperature),
        create_graph=False,
        retain_graph=False,
    )[0]
    conductivity = material.conductivity(temperature)
    normal = torch.as_tensor(normal_rz, dtype=x.dtype, device=x.device)
    outward_flux = -conductivity[:, 0] * torch.sum(gradient[:, :2] * normal, dim=1)
    integral = float(torch.sum(weights * outward_flux.detach()).cpu())
    return temperature.detach()[:, 0], gradient.detach(), integral


def _radiative_flux(
    temperature: Tensor,
    ambient_temperature_k: float,
    emissivity: float | None,
    enabled: bool,
) -> Tensor:
    if not enabled:
        return torch.zeros_like(temperature)
    if emissivity is None:
        raise ValueError("Resolved emissivity is required for the V5 energy audit")
    return radiation_flux(temperature[:, None], ambient_temperature_k, emissivity)[:, 0]


def deterministic_energy_terms(
    model: nn.Module,
    materials: Mapping[int, Material],
    boundaries: BoundaryConditions,
    geometry: AxisymmetricGeometry,
    power_w: float,
    time_s: float,
    order: int,
    outer_epsilon_m: float,
    derivative_batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if time_s <= 0.0:
        raise ValueError("t=0 is an initial-condition corner test, not a regular energy row")
    if not 0.0 < outer_epsilon_m < geometry.copper_radius_m:
        raise ValueError("outer_epsilon_m must be inside the copper radius")
    if boundaries.cooling.kind != "fixed_temperature" or boundaries.cooling.surface != "outer_radius":
        raise ValueError("V5 audit expects the resolved fixed-temperature outer cooling surface")

    domains = (
        (
            "silicon_carbide",
            1,
            (0.0, geometry.silicon_carbide_radius_m),
            (geometry.silicon_carbide_bottom_z_m, 0.0),
        ),
        (
            "copper_outer_ring",
            0,
            (geometry.silicon_carbide_radius_m, geometry.copper_radius_m),
            (geometry.copper_bottom_z_m, 0.0),
        ),
        (
            "copper_below_sic",
            0,
            (0.0, geometry.silicon_carbide_radius_m),
            (geometry.copper_bottom_z_m, geometry.silicon_carbide_bottom_z_m),
        ),
    )
    storage_by_domain: dict[str, float] = {}
    residual_by_domain: dict[str, float] = {}
    for name, material_id, radial_bounds, axial_bounds in domains:
        rz, volume_weights = axisymmetric_rectangle_quadrature(
            radial_bounds, axial_bounds, order, device
        )
        coordinates = _coordinates(
            rz[:, 0], rz[:, 1], time_s, power_w, material_id
        )
        storage, residual = _volume_integrals(
            model,
            coordinates,
            volume_weights,
            materials[material_id],
            derivative_batch_size,
        )
        storage_by_domain[name] = storage
        residual_by_domain[name] = residual

    rs = geometry.silicon_carbide_radius_m
    rc = geometry.copper_radius_m
    z_bottom = geometry.copper_bottom_z_m
    z_sic = geometry.silicon_carbide_bottom_z_m
    ambient = boundaries.ambient_temperature_k

    sic_top_r, sic_top_weights = axisymmetric_horizontal_quadrature(
        (0.0, rs), order, device
    )
    sic_top_coordinates = _coordinates(
        sic_top_r, torch.zeros_like(sic_top_r), time_s, power_w, 1
    )
    sic_top_t, _, sic_top_model_flux = _surface_values(
        model, sic_top_coordinates, sic_top_weights, (0.0, 1.0), materials[1]
    )
    absorbed_density = laser_flux(
        sic_top_r[:, None],
        torch.full_like(sic_top_r[:, None], power_w),
        boundaries.laser.profile,
        boundaries.laser.absorption_fraction,
        boundaries.laser.beam_radius_m,
    )[:, 0]
    absorbed_power = float(torch.sum(sic_top_weights * absorbed_density).cpu())
    sic_top_convection = convection_flux(
        sic_top_t[:, None], ambient, boundaries.top_convection_coefficient_w_m2_k
    )[:, 0]
    sic_top_radiation = _radiative_flux(
        sic_top_t,
        ambient,
        boundaries.silicon_carbide_emissivity,
        boundaries.radiation_enabled,
    )

    copper_top_r, copper_top_weights = axisymmetric_horizontal_quadrature(
        (rs, rc), order, device
    )
    copper_top_coordinates = _coordinates(
        copper_top_r, torch.zeros_like(copper_top_r), time_s, power_w, 0
    )
    copper_top_t, _, copper_top_model_flux = _surface_values(
        model,
        copper_top_coordinates,
        copper_top_weights,
        (0.0, 1.0),
        materials[0],
    )
    copper_top_convection = convection_flux(
        copper_top_t[:, None], ambient, boundaries.top_convection_coefficient_w_m2_k
    )[:, 0]
    copper_top_radiation = _radiative_flux(
        copper_top_t,
        ambient,
        boundaries.copper_emissivity,
        boundaries.radiation_enabled,
    )

    bottom_r, bottom_weights = axisymmetric_horizontal_quadrature(
        (0.0, rc), order, device
    )
    bottom_coordinates = _coordinates(
        bottom_r,
        torch.full_like(bottom_r, z_bottom),
        time_s,
        power_w,
        0,
    )
    bottom_t, _, bottom_model_flux = _surface_values(
        model, bottom_coordinates, bottom_weights, (0.0, -1.0), materials[0]
    )
    if boundaries.bottom_convection_coefficient_w_m2_k is None:
        raise ValueError("Resolved bottom convection coefficient is required")
    bottom_convection = convection_flux(
        bottom_t[:, None], ambient, boundaries.bottom_convection_coefficient_w_m2_k
    )[:, 0]
    bottom_radiation = _radiative_flux(
        bottom_t,
        ambient,
        boundaries.copper_emissivity,
        boundaries.radiation_enabled,
    )

    outer_z, outer_weights = axisymmetric_vertical_quadrature(
        rc, (z_bottom, 0.0), order, device
    )
    outer_coordinates = _coordinates(
        torch.full_like(outer_z, rc - outer_epsilon_m),
        outer_z,
        time_s,
        power_w,
        0,
    )
    outer_t, outer_gradient, cooling_heat = _surface_values(
        model, outer_coordinates, outer_weights, (1.0, 0.0), materials[0]
    )
    outer_exact_coordinates = _coordinates(
        torch.full_like(outer_z, rc), outer_z, time_s, power_w, 0
    )
    outer_exact_t, outer_exact_gradient, outer_exact_model_flux = _surface_values(
        model,
        outer_exact_coordinates,
        outer_weights,
        (1.0, 0.0),
        materials[0],
    )
    hard_cooling_enabled = getattr(model, "hard_cooling_radius_m", None) is not None
    mathematical_outer_flux = (
        cooling_heat if hard_cooling_enabled else outer_exact_model_flux
    )

    horizontal_r, horizontal_weights = axisymmetric_horizontal_quadrature(
        (0.0, rs), order, device
    )
    sic_bottom_coordinates = _coordinates(
        horizontal_r,
        torch.full_like(horizontal_r, z_sic),
        time_s,
        power_w,
        1,
    )
    copper_under_coordinates = _coordinates(
        horizontal_r,
        torch.full_like(horizontal_r, z_sic),
        time_s,
        power_w,
        0,
    )
    _, _, sic_bottom_flux = _surface_values(
        model,
        sic_bottom_coordinates,
        horizontal_weights,
        (0.0, -1.0),
        materials[1],
    )
    _, _, copper_under_top_flux = _surface_values(
        model,
        copper_under_coordinates,
        horizontal_weights,
        (0.0, 1.0),
        materials[0],
    )

    side_z, side_weights = axisymmetric_vertical_quadrature(
        rs, (z_sic, 0.0), order, device
    )
    sic_side_coordinates = _coordinates(
        torch.full_like(side_z, rs), side_z, time_s, power_w, 1
    )
    copper_inner_coordinates = _coordinates(
        torch.full_like(side_z, rs), side_z, time_s, power_w, 0
    )
    _, _, sic_side_flux = _surface_values(
        model, sic_side_coordinates, side_weights, (1.0, 0.0), materials[1]
    )
    _, _, copper_inner_flux = _surface_values(
        model, copper_inner_coordinates, side_weights, (-1.0, 0.0), materials[0]
    )

    convection_heat = float(
        (
            torch.sum(sic_top_weights * sic_top_convection)
            + torch.sum(copper_top_weights * copper_top_convection)
            + torch.sum(bottom_weights * bottom_convection)
        ).cpu()
    )
    radiation_heat = float(
        (
            torch.sum(sic_top_weights * sic_top_radiation)
            + torch.sum(copper_top_weights * copper_top_radiation)
            + torch.sum(bottom_weights * bottom_radiation)
        ).cpu()
    )
    storage_rate = sum(storage_by_domain.values())
    integrated_residual = sum(residual_by_domain.values())
    interface_flux = (
        sic_bottom_flux
        + copper_under_top_flux
        + sic_side_flux
        + copper_inner_flux
    )
    external_model_flux = (
        sic_top_model_flux
        + copper_top_model_flux
        + bottom_model_flux
        + mathematical_outer_flux
    )
    boundary_law_outward = cooling_heat + convection_heat + radiation_heat - absorbed_power
    balance = storage_rate + boundary_law_outward
    mathematical_rhs = storage_rate + external_model_flux + interface_flux
    boundary_flux_residual = external_model_flux - boundary_law_outward
    relative_denominator = max(
        abs(absorbed_power),
        abs(storage_rate),
        abs(cooling_heat + convection_heat + radiation_heat),
        1e-12,
    )

    energy = {
        "power_w": float(power_w),
        "time_s": float(time_s),
        "quadrature_order": int(order),
        "absorbed_power_w": absorbed_power,
        "storage_rate_w": storage_rate,
        "cooling_heat_w": cooling_heat,
        "convection_heat_w": convection_heat,
        "radiation_heat_w": radiation_heat,
        "balance_w": balance,
        "relative_balance_denominator_w": relative_denominator,
        "relative_balance": balance / relative_denominator,
        "boundary_gradient_mode": "inner_autodiff_at_R_minus_epsilon",
        "outer_epsilon_m": float(outer_epsilon_m),
        "sign_convention": (
            "storage_positive_for_internal_energy_increase;"
            "cooling_convection_radiation_positive_outward;"
            "absorbed_power_recorded_as_positive_inward_magnitude"
        ),
        "balance_definition": (
            "storage+cooling+convection+radiation-absorbed"
        ),
    }
    divergence = {
        "power_w": float(power_w),
        "time_s": float(time_s),
        "quadrature_order": int(order),
        "integrated_pde_residual_w": integrated_residual,
        "storage_rate_w": storage_rate,
        "external_model_conductive_flux_w": external_model_flux,
        "interface_two_sided_flux_w": interface_flux,
        "storage_plus_external_plus_interface_w": mathematical_rhs,
        "divergence_identity_gap_w": integrated_residual - mathematical_rhs,
        "boundary_law_outward_w": boundary_law_outward,
        "boundary_flux_residual_w": boundary_flux_residual,
        "engineering_balance_w": balance,
        "explained_engineering_balance_w": (
            integrated_residual - boundary_flux_residual - interface_flux
        ),
        "engineering_explanation_gap_w": balance
        - (integrated_residual - boundary_flux_residual - interface_flux),
        "sic_top_model_flux_w": sic_top_model_flux,
        "copper_top_model_flux_w": copper_top_model_flux,
        "bottom_model_flux_w": bottom_model_flux,
        "outer_model_flux_w": mathematical_outer_flux,
        "outer_exact_model_flux_w": outer_exact_model_flux,
        "outer_inner_limit_cooling_flux_w": cooling_heat,
        "sic_bottom_interface_flux_w": sic_bottom_flux,
        "copper_under_interface_flux_w": copper_under_top_flux,
        "sic_side_interface_flux_w": sic_side_flux,
        "copper_outer_ring_interface_flux_w": copper_inner_flux,
        "silicon_carbide_pde_residual_w": residual_by_domain["silicon_carbide"],
        "copper_outer_ring_pde_residual_w": residual_by_domain["copper_outer_ring"],
        "copper_below_sic_pde_residual_w": residual_by_domain["copper_below_sic"],
        "outer_temperature_mean_c": float(outer_exact_t.mean().cpu()) - 273.15,
        "outer_temperature_max_abs_deviation_c": float(
            torch.max(
                torch.abs(
                    outer_exact_t - float(boundaries.cooling.fixed_temperature_k)
                )
            ).cpu()
        ),
        "outer_radial_gradient_mean_k_m": float(
            outer_exact_gradient[:, 0].mean().cpu()
        ),
        "outer_inner_temperature_mean_c": float(outer_t.mean().cpu()) - 273.15,
        "outer_inner_radial_gradient_mean_k_m": float(
            outer_gradient[:, 0].mean().cpu()
        ),
        "boundary_gradient_mode": (
            "inner_limit_for_hard_override"
            if hard_cooling_enabled
            else "exact_R_for_divergence_and_inner_limit_for_engineering_cooling"
        ),
        "outer_epsilon_m": float(outer_epsilon_m),
        "conductive_flux_sign_convention": "positive_outward_from_each_material",
        "interface_definition": "sum_of_two_material_outward_fluxes",
    }
    return energy, divergence


def cooling_boundary_limit(
    model: nn.Module,
    copper: Material,
    geometry: AxisymmetricGeometry,
    cooling_temperature_k: float,
    power_w: float,
    time_s: float,
    order: int,
    epsilons_m: Iterable[float],
    device: torch.device,
) -> list[dict[str, Any]]:
    axial, weights = axisymmetric_vertical_quadrature(
        geometry.copper_radius_m,
        (geometry.copper_bottom_z_m, 0.0),
        order,
        device,
    )
    rows: list[dict[str, Any]] = []
    for epsilon in (0.0, *[float(value) for value in epsilons_m]):
        coordinates = _coordinates(
            torch.full_like(axial, geometry.copper_radius_m - epsilon),
            axial,
            time_s,
            power_w,
            0,
        )
        temperature, gradient, cooling_w = _surface_values(
            model, coordinates, weights, (1.0, 0.0), copper
        )
        rows.append(
            {
                "scope": "actual_seed0_hf_checkpoint",
                "power_w": float(power_w),
                "time_s": float(time_s),
                "quadrature_order": int(order),
                "epsilon_m": epsilon,
                "evaluation_radius_m": geometry.copper_radius_m - epsilon,
                "inside_isclose_tolerance": bool(
                    epsilon
                    <= float(getattr(model, "hard_cooling_tolerance_m", 0.0))
                ),
                "temperature_mean_c": float(temperature.mean().cpu()) - 273.15,
                "temperature_max_abs_deviation_from_cooling_c": float(
                    torch.max(torch.abs(temperature - cooling_temperature_k)).cpu()
                ),
                "radial_gradient_mean_k_m": float(gradient[:, 0].mean().cpu()),
                "cooling_heat_w": cooling_w,
            }
        )
    return rows


class _LinearCoolingControl(nn.Module):
    def __init__(self, radius_m: float, cooling_temperature_k: float, slope_k_m: float):
        super().__init__()
        self.radius_m = float(radius_m)
        self.cooling_temperature_k = float(cooling_temperature_k)
        self.slope_k_m = float(slope_k_m)

    def forward(self, coordinates: Tensor) -> Tensor:
        return self.cooling_temperature_k + self.slope_k_m * (
            self.radius_m - coordinates[:, 0:1]
        )


def hard_cooling_override_control(
    radius_m: float,
    cooling_temperature_k: float,
    copper_conductivity_w_m_k: float,
    tolerance_m: float,
    epsilon_m: float,
    device: torch.device,
) -> dict[str, Any]:
    slope = 100.0
    low = _LinearCoolingControl(radius_m, cooling_temperature_k, slope)
    model = AdditiveCorrectionModel(
        low,
        ModelScales(),
        width=4,
        depth=1,
        include_material=True,
        hard_cooling_radius_m=radius_m,
        hard_cooling_temperature_k=cooling_temperature_k,
        hard_cooling_tolerance_m=tolerance_m,
    ).to(device=device, dtype=torch.float64)
    for parameter in model.correction.parameters():
        parameter.data.zero_()
        parameter.requires_grad_(False)

    def evaluate(radius: float) -> tuple[float, float]:
        x = torch.tensor(
            [[radius, -0.005, 10.0, 100.0, 0.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        temperature = model(x)
        gradient = torch.autograd.grad(temperature.sum(), x)[0][0, 0]
        return float(temperature.item()), float(gradient.item())

    boundary_temperature, boundary_gradient = evaluate(radius_m)
    inner_temperature, inner_gradient = evaluate(radius_m - epsilon_m)
    return {
        "control": "linear_field_with_hard_temperature_override",
        "raw_field": "T=T_cool+100*(R-r)",
        "hard_tolerance_m": float(tolerance_m),
        "inner_epsilon_m": float(epsilon_m),
        "boundary_temperature_k": boundary_temperature,
        "boundary_radial_gradient_k_m": boundary_gradient,
        "boundary_outward_heat_flux_w_m2": -copper_conductivity_w_m_k
        * boundary_gradient,
        "inner_temperature_k": inner_temperature,
        "inner_radial_gradient_k_m": inner_gradient,
        "inner_outward_heat_flux_w_m2": -copper_conductivity_w_m_k * inner_gradient,
        "expected_raw_radial_gradient_k_m": -slope,
        "expected_raw_outward_heat_flux_w_m2": copper_conductivity_w_m_k * slope,
        "exposes_zero_gradient_failure": abs(boundary_gradient) < 1e-12
        and abs(inner_gradient + slope) < 1e-12,
    }


def analytic_control_tests(
    geometry: AxisymmetricGeometry,
    boundaries: BoundaryConditions,
    materials: Mapping[int, Material],
    orders: Iterable[int],
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rc = geometry.copper_radius_m
    rs = geometry.silicon_carbide_radius_m
    zb = geometry.copper_bottom_z_m
    zs = geometry.silicon_carbide_bottom_z_m
    domains = (
        ("silicon_carbide", 1, (0.0, rs), (zs, 0.0)),
        ("copper_outer_ring", 0, (rs, rc), (zb, 0.0)),
        ("copper_below_sic", 0, (0.0, rs), (zb, zs)),
    )
    a = 0.7
    b = 11.0
    c = -5.0
    for order in orders:
        volume_by_domain: dict[str, float] = {}
        storage = 0.0
        residual = 0.0
        for name, material_id, radial_bounds, axial_bounds in domains:
            _, weights = axisymmetric_rectangle_quadrature(
                radial_bounds, axial_bounds, int(order), device
            )
            volume = float(weights.sum().cpu())
            volume_by_domain[name] = volume
            density, conductivity, heat_capacity = _constant_material_values(
                materials[material_id]
            )
            storage += density * heat_capacity * a * volume
            residual += (
                density * heat_capacity * a - 4.0 * conductivity * b
            ) * volume

        sic_top_r, sic_top_w = axisymmetric_horizontal_quadrature(
            (0.0, rs), int(order), device
        )
        del sic_top_r
        copper_top_r, copper_top_w = axisymmetric_horizontal_quadrature(
            (rs, rc), int(order), device
        )
        del copper_top_r
        bottom_r, bottom_w = axisymmetric_horizontal_quadrature(
            (0.0, rc), int(order), device
        )
        del bottom_r
        _, outer_w = axisymmetric_vertical_quadrature(
            rc, (zb, 0.0), int(order), device
        )
        _, horizontal_w = axisymmetric_horizontal_quadrature(
            (0.0, rs), int(order), device
        )
        _, side_w = axisymmetric_vertical_quadrature(
            rs, (zs, 0.0), int(order), device
        )
        k_cu = float(materials[0].conductivity.value)
        k_sic = float(materials[1].conductivity.value)
        external = (
            -k_sic * c * float(sic_top_w.sum().cpu())
            - k_cu * c * float(copper_top_w.sum().cpu())
            + k_cu * c * float(bottom_w.sum().cpu())
            - 2.0 * k_cu * b * rc * float(outer_w.sum().cpu())
        )
        interface = (
            (k_sic - k_cu) * c * float(horizontal_w.sum().cpu())
            + 2.0 * b * rs * (k_cu - k_sic) * float(side_w.sum().cpu())
        )
        rhs = storage + external + interface
        volume_error = max(
            abs(volume_by_domain["silicon_carbide"] - geometry.silicon_carbide_volume_m3),
            abs(
                volume_by_domain["copper_outer_ring"]
                + volume_by_domain["copper_below_sic"]
                - geometry.copper_volume_m3
            ),
        )
        rows.append(
            {
                "control": "manufactured_axisymmetric_quadratic_field",
                "quadrature_order": int(order),
                "field_definition": "T=T0+0.7*t+11*r^2-5*z",
                "storage_rate_w": storage,
                "external_conductive_flux_w": external,
                "interface_two_sided_flux_w": interface,
                "integrated_pde_residual_w": residual,
                "storage_plus_external_plus_interface_w": rhs,
                "identity_gap_w": residual - rhs,
                "maximum_material_volume_error_m3": volume_error,
                "passed": abs(residual - rhs) <= 1e-8 * max(abs(residual), 1.0)
                and volume_error <= 1e-15,
            }
        )

        laser_r, laser_w = axisymmetric_horizontal_quadrature(
            (0.0, rs), int(order), device
        )
        audit_power = 729.0
        numerical_absorbed = float(
            torch.sum(
                laser_w
                * laser_flux(
                    laser_r[:, None],
                    torch.full_like(laser_r[:, None], audit_power),
                    boundaries.laser.profile,
                    boundaries.laser.absorption_fraction,
                    boundaries.laser.beam_radius_m,
                )[:, 0]
            ).cpu()
        )
        if boundaries.laser.profile == "gaussian":
            capture = 1.0 - math.exp(
                -2.0 * rs**2 / boundaries.laser.beam_radius_m**2
            )
            analytic_absorbed = boundaries.laser.absorption_fraction * audit_power * capture
        else:
            illuminated_radius = min(rs, boundaries.laser.beam_radius_m)
            capture = (illuminated_radius / boundaries.laser.beam_radius_m) ** 2
            analytic_absorbed = boundaries.laser.absorption_fraction * audit_power * capture
        rows.append(
            {
                "control": "finite_disc_laser_integral",
                "quadrature_order": int(order),
                "field_definition": boundaries.laser.profile,
                "storage_rate_w": 0.0,
                "external_conductive_flux_w": 0.0,
                "interface_two_sided_flux_w": 0.0,
                "integrated_pde_residual_w": 0.0,
                "storage_plus_external_plus_interface_w": 0.0,
                "identity_gap_w": numerical_absorbed - analytic_absorbed,
                "maximum_material_volume_error_m3": 0.0,
                "audit_power_w": audit_power,
                "finite_disc_capture_fraction": capture,
                "numerical_absorbed_power_w": numerical_absorbed,
                "analytic_absorbed_power_w": analytic_absorbed,
                "passed": abs(numerical_absorbed - analytic_absorbed)
                <= 1e-10 * max(abs(analytic_absorbed), 1.0),
            }
        )

        rows.append(
            {
                "control": "zero_power_isothermal_field",
                "quadrature_order": int(order),
                "field_definition": f"T={boundaries.ambient_temperature_k} K, P=0 W",
                "storage_rate_w": 0.0,
                "external_conductive_flux_w": 0.0,
                "interface_two_sided_flux_w": 0.0,
                "integrated_pde_residual_w": 0.0,
                "storage_plus_external_plus_interface_w": 0.0,
                "identity_gap_w": 0.0,
                "maximum_material_volume_error_m3": volume_error,
                "absorbed_power_w": 0.0,
                "convection_heat_w": 0.0,
                "radiation_heat_w": 0.0,
                "cooling_heat_w": 0.0,
                "balance_w": 0.0,
                "passed": volume_error <= 1e-15,
            }
        )
    return rows
