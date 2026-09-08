from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.fields import SimulationField, load_processed_field
from sic_cu.data.splits import build_power_splits


STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8


@dataclass(frozen=True)
class ScalarFit:
    value: float
    unconstrained_value: float
    sample_count: int
    power_count: int
    residual_rmse: float
    normalized_residual_rmse: float
    power_estimate_p05: float
    power_estimate_median: float
    power_estimate_p95: float


@dataclass(frozen=True)
class BoundarySamples:
    power_w: np.ndarray
    predictor: np.ndarray
    response: np.ndarray

    def filtered(self, mask: np.ndarray) -> BoundarySamples:
        return BoundarySamples(
            power_w=self.power_w[mask],
            predictor=self.predictor[mask],
            response=self.response[mask],
        )


def derivative_weights(coordinates: np.ndarray, evaluation_coordinate: float) -> np.ndarray:
    """Return finite-difference weights for a first derivative at one coordinate."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 1 or len(coordinates) < 2:
        raise ValueError("At least two one-dimensional coordinates are required")
    if len(np.unique(coordinates)) != len(coordinates):
        raise ValueError("Derivative stencil coordinates must be unique")
    scale = float(np.max(np.abs(coordinates - evaluation_coordinate)))
    if scale <= 0:
        raise ValueError("Derivative stencil has zero extent")
    normalized = (coordinates - evaluation_coordinate) / scale
    powers = np.arange(len(coordinates), dtype=np.int64)
    system = normalized[np.newaxis, :] ** powers[:, np.newaxis]
    target = np.zeros(len(coordinates), dtype=np.float64)
    target[1] = 1.0 / scale
    return np.linalg.solve(system, target)


def _key(value: float) -> float:
    return round(float(value), 8)


def _boundary_stencils(
    field: SimulationField,
    material_id: int,
    derivative_axis: int,
    boundary_coordinate: float,
    interior_direction: int,
    *,
    exclude_tangent_endpoints: bool = False,
    stencil_size: int = 3,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    coordinates = field.coordinates_rz_m.astype(np.float64)
    material_mask = field.material_ids == material_id
    boundary_mask = material_mask & np.isclose(
        coordinates[:, derivative_axis], boundary_coordinate, atol=2e-7
    )
    boundary_indices = np.flatnonzero(boundary_mask)
    tangent_axis = 1 - derivative_axis
    if exclude_tangent_endpoints and len(boundary_indices) > 2:
        tangents = coordinates[boundary_indices, tangent_axis]
        lower, upper = float(tangents.min()), float(tangents.max())
        boundary_indices = boundary_indices[
            (~np.isclose(tangents, lower, atol=2e-7))
            & (~np.isclose(tangents, upper, atol=2e-7))
        ]
    stencils: list[tuple[int, np.ndarray, np.ndarray]] = []
    for boundary_index in boundary_indices:
        tangent = coordinates[boundary_index, tangent_axis]
        same_line = material_mask & np.isclose(
            coordinates[:, tangent_axis], tangent, atol=2e-7
        )
        offsets = coordinates[:, derivative_axis] - boundary_coordinate
        inside = same_line & (interior_direction * offsets >= -2e-7)
        candidates = np.flatnonzero(inside)
        candidates = candidates[np.argsort(np.abs(offsets[candidates]))][:stencil_size]
        if len(candidates) != stencil_size:
            continue
        weights = derivative_weights(
            coordinates[candidates, derivative_axis], boundary_coordinate
        )
        stencils.append((int(boundary_index), candidates, weights))
    if not stencils:
        raise ValueError(
            f"No derivative stencils for material={material_id}, axis={derivative_axis}, "
            f"coordinate={boundary_coordinate}"
        )
    return stencils


def _evaluate_stencils(
    field: SimulationField,
    stencils: list[tuple[int, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundary_indices = np.asarray([item[0] for item in stencils], dtype=np.int64)
    temperature = field.temperature_k[:, boundary_indices].astype(np.float64)
    derivative = np.column_stack(
        [
            field.temperature_k[:, indices].astype(np.float64) @ weights
            for _, indices, weights in stencils
        ]
    )
    coordinates = field.coordinates_rz_m[boundary_indices].astype(np.float64)
    return temperature, derivative, coordinates


def _radiation_basis(temperature_k: np.ndarray, ambient_temperature_k: float) -> np.ndarray:
    return STEFAN_BOLTZMANN_W_M2_K4 * (
        temperature_k**4 - float(ambient_temperature_k) ** 4
    )


def _gaussian_flux(
    radius_m: np.ndarray,
    power_w: float,
    absorption_fraction: float,
    beam_radius_m: float,
) -> np.ndarray:
    scale = 2.0 * absorption_fraction * power_w / (np.pi * beam_radius_m**2)
    return scale * np.exp(-2.0 * radius_m**2 / beam_radius_m**2)


def _flatten_samples(
    power_w: float,
    predictor: np.ndarray,
    response: np.ndarray,
    valid: np.ndarray,
) -> BoundarySamples:
    return BoundarySamples(
        power_w=np.full(int(valid.sum()), float(power_w), dtype=np.float64),
        predictor=predictor[valid].astype(np.float64),
        response=response[valid].astype(np.float64),
    )


def _concat(samples: Iterable[BoundarySamples]) -> BoundarySamples:
    values = list(samples)
    if not values:
        raise ValueError("No parameter-identification samples were produced")
    return BoundarySamples(
        power_w=np.concatenate([item.power_w for item in values]),
        predictor=np.concatenate([item.predictor for item in values]),
        response=np.concatenate([item.response for item in values]),
    )


def fit_scalar_through_origin(
    samples: BoundarySamples,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> ScalarFit:
    x = samples.predictor
    y = samples.response
    if len(x) < 2 or float(np.dot(x, x)) <= np.finfo(np.float64).eps:
        raise ValueError("Insufficient sensitivity for scalar parameter fit")
    unconstrained = float(np.dot(x, y) / np.dot(x, x))
    value = unconstrained
    if lower is not None:
        value = max(value, lower)
    if upper is not None:
        value = min(value, upper)
    residual = y - value * x
    scale = max(float(np.sqrt(np.mean(y**2))), np.finfo(np.float64).eps)
    per_power = []
    for power in np.unique(samples.power_w):
        mask = np.isclose(samples.power_w, power, atol=1e-6)
        denominator = float(np.dot(x[mask], x[mask]))
        if denominator > np.finfo(np.float64).eps:
            per_power.append(float(np.dot(x[mask], y[mask]) / denominator))
    quantiles = np.quantile(np.asarray(per_power), [0.05, 0.5, 0.95])
    return ScalarFit(
        value=value,
        unconstrained_value=unconstrained,
        sample_count=len(x),
        power_count=len(per_power),
        residual_rmse=float(np.sqrt(np.mean(residual**2))),
        normalized_residual_rmse=float(np.sqrt(np.mean(residual**2)) / scale),
        power_estimate_p05=float(quantiles[0]),
        power_estimate_median=float(quantiles[1]),
        power_estimate_p95=float(quantiles[2]),
    )


def evaluate_scalar(samples: BoundarySamples, value: float) -> dict[str, float | int]:
    residual = samples.response - float(value) * samples.predictor
    scale = max(
        float(np.sqrt(np.mean(samples.response**2))), np.finfo(np.float64).eps
    )
    return {
        "sample_count": len(residual),
        "power_count": len(np.unique(samples.power_w)),
        "residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "normalized_residual_rmse": float(np.sqrt(np.mean(residual**2)) / scale),
    }


def _fit_record(
    name: str,
    fit: ScalarFit,
    validation: dict[str, float | int],
) -> dict[str, Any]:
    if name.endswith("emissivity"):
        lower, upper = 0.0, 1.0
        within_bounds = lower <= fit.unconstrained_value <= upper
        status = (
            "PRELIMINARY_LOW_FIDELITY_ESTIMATE"
            if within_bounds
            else "NOT_IDENTIFIABLE_BOUNDARY_MODEL_MISMATCH"
        )
        recommendation = (
            "Use as an initialization only; verify by high-fidelity sensitivity analysis."
            if within_bounds
            else "Do not use the constrained zero value as a physical estimate; retain this "
            "quantity as a nuisance parameter in sensitivity analysis."
        )
        physical_bounds: list[float | None] = [lower, upper]
    else:
        within_bounds = fit.unconstrained_value > 0.0
        validation_error = float(validation["normalized_residual_rmse"])
        stable_across_powers = fit.power_estimate_p05 > 0.0 and (
            fit.power_estimate_p95 / fit.power_estimate_p05 < 1.05
        )
        status = (
            "IDENTIFIED_LOW_FIDELITY_EFFECTIVE_INITIALIZATION"
            if within_bounds and stable_across_powers and validation_error < 0.01
            else "PRELIMINARY_LOW_FIDELITY_ESTIMATE"
        )
        recommendation = (
            "Use as the effective low-fidelity initialization; refine or profile it only on "
            "training/validation powers before the held-out test is opened."
        )
        physical_bounds = [0.0, None]
    return {
        "train": asdict(fit),
        "validation": validation,
        "physical_bounds": physical_bounds,
        "unconstrained_value_within_physical_bounds": within_bounds,
        "identification_status": status,
        "recommendation": recommendation,
    }


def _render_markdown(result: dict[str, Any]) -> str:
    sic = result["fits"]["silicon_carbide_emissivity"]
    copper = result["fits"]["copper_emissivity"]
    contact = result["fits"]["contact_resistance_m2_k_w"]
    return f"""# 物理参数初步辨识报告

生成状态：`{result['status']}`  
数据隔离：仅使用 Simulation 训练功率拟合、验证功率检查；测试功率未访问。

## 结论

| 参数 | 拟合值 | 验证归一化残差 RMSE | 判定 |
|---|---:|---:|---|
| SiC 热辐射率 | {sic['train']['value']:.6g}（无约束值 {sic['train']['unconstrained_value']:.6g}） | {sic['validation']['normalized_residual_rmse']:.6g} | `{sic['identification_status']}` |
| Cu 热辐射率 | {copper['train']['value']:.6g}（无约束值 {copper['train']['unconstrained_value']:.6g}） | {copper['validation']['normalized_residual_rmse']:.6g} | `{copper['identification_status']}` |
| SiC-Cu 接触热阻 | {contact['train']['value']:.12g} m²·K/W | {contact['validation']['normalized_residual_rmse']:.6g} | `{contact['identification_status']}` |

SiC 与 Cu 辐射率的无约束解均为负值，违反 `[0,1]` 物理范围。这说明当前低保真场的
表面法向梯度与已声明的 Gaussian 热源、自然对流和辐射边界不能闭合，约束优化得到的
`0` 只是边界投影，不是可接受的材料辐射率。后续不得把它写入生产配置，而应将辐射率
作为敏感性/干扰参数处理。

接触热阻在训练功率间的 5%/中位数/95% 分位数为
`{contact['train']['power_estimate_p05']:.12g}` / `{contact['train']['power_estimate_median']:.12g}` /
`{contact['train']['power_estimate_p95']:.12g} m²·K/W`，验证集归一化残差 RMSE 为
`{contact['validation']['normalized_residual_rmse']:.4%}`。因此可把
`{contact['train']['value']:.12g} m²·K/W` 用作低保真有效初值，但它不是内部实验真值，
仍需在高保真训练折内做固定值/可训练值对照。

## 方法

- 表面和界面法向梯度：三点单侧二阶有限差分。
- 参数拟合：过原点最小二乘；辐射率施加 `[0,1]` 物理范围，接触热阻要求非负。
- 数据范围：{len(result['train_powers_w'])} 个训练功率拟合，{len(result['validation_powers_w'])} 个验证功率检查。
- 用途：只作为后续多保真 PINN 的参数初始化与可辨识性证据。

## Material Passport

- 使用低保真仿真场：是。
- 使用实验 IR：否。
- 使用 Hot/Cold：否。
- 使用内部实验真值：否（不存在）。
- 生产验证状态：未通过；本报告不会自动打开生产训练门禁。
"""


def _surface_samples(
    field: SimulationField,
    *,
    material_id: int,
    surface: str,
    conductivity_w_m_k: float,
    ambient_temperature_k: float,
    convection_coefficient_w_m2_k: float,
    absorption_fraction: float,
    beam_radius_m: float,
) -> BoundarySamples:
    if surface == "top":
        stencils = _boundary_stencils(
            field,
            material_id,
            derivative_axis=1,
            boundary_coordinate=0.0,
            interior_direction=-1,
            exclude_tangent_endpoints=True,
        )
        temperature, derivative, coordinates = _evaluate_stencils(field, stencils)
        outward_flux = -conductivity_w_m_k * derivative
        laser = (
            _gaussian_flux(
                coordinates[:, 0],
                field.power_w,
                absorption_fraction,
                beam_radius_m,
            )[None, :]
            if material_id == 1
            else np.zeros((1, len(coordinates)), dtype=np.float64)
        )
    elif surface == "bottom" and material_id == 0:
        bottom = float(field.coordinates_rz_m[:, 1].min())
        stencils = _boundary_stencils(
            field,
            material_id,
            derivative_axis=1,
            boundary_coordinate=bottom,
            interior_direction=1,
            exclude_tangent_endpoints=True,
        )
        temperature, derivative, coordinates = _evaluate_stencils(field, stencils)
        outward_flux = conductivity_w_m_k * derivative
        laser = np.zeros((1, len(coordinates)), dtype=np.float64)
    else:
        raise ValueError(f"Unsupported surface/material combination: {surface}/{material_id}")
    predictor = _radiation_basis(temperature, ambient_temperature_k)
    response = (
        outward_flux
        + laser
        - convection_coefficient_w_m2_k * (temperature - ambient_temperature_k)
    )
    valid = (
        (field.times_s[:, None] > 0.0)
        & (temperature - ambient_temperature_k > 3.0)
        & np.isfinite(predictor)
        & np.isfinite(response)
    )
    return _flatten_samples(field.power_w, predictor, response, valid)


def _interface_samples(
    field: SimulationField,
    conductivity_sic_w_m_k: float,
) -> tuple[BoundarySamples, dict[str, float]]:
    coordinates = field.coordinates_rz_m.astype(np.float64)
    sic_mask = field.material_ids == 1
    copper_lookup = {
        (_key(r), _key(z)): index
        for index, ((r, z), material) in enumerate(
            zip(coordinates, field.material_ids, strict=True)
        )
        if material == 0
    }
    sic_bottom = float(coordinates[sic_mask, 1].min())
    sic_radius = float(coordinates[sic_mask, 0].max())
    definitions = (
        (1, sic_bottom, 1, 1.0, True),
        (0, sic_radius, -1, -1.0, True),
    )
    sample_parts: list[BoundarySamples] = []
    paired_points = 0
    for axis, boundary, interior_direction, flux_sign, exclude_endpoints in definitions:
        stencils = _boundary_stencils(
            field,
            1,
            derivative_axis=axis,
            boundary_coordinate=boundary,
            interior_direction=interior_direction,
            exclude_tangent_endpoints=exclude_endpoints,
        )
        temperature_sic, derivative, boundary_coordinates = _evaluate_stencils(
            field, stencils
        )
        sic_indices = np.asarray([item[0] for item in stencils], dtype=np.int64)
        copper_indices = np.asarray(
            [copper_lookup[(_key(r), _key(z))] for r, z in boundary_coordinates],
            dtype=np.int64,
        )
        del sic_indices
        temperature_copper = field.temperature_k[:, copper_indices].astype(np.float64)
        outward_flux = flux_sign * conductivity_sic_w_m_k * derivative
        jump = temperature_sic - temperature_copper
        valid = (
            (field.times_s[:, None] > 0.0)
            & (outward_flux > 100.0)
            & (jump > 0.01)
            & np.isfinite(outward_flux)
            & np.isfinite(jump)
        )
        sample_parts.append(_flatten_samples(field.power_w, outward_flux, jump, valid))
        paired_points += len(boundary_coordinates)
    samples = _concat(sample_parts)
    return samples, {"paired_interface_points_per_frame": paired_points}


def collect_parameter_samples(powers_w: Iterable[float]) -> dict[str, BoundarySamples]:
    boundary = load_yaml("configs/boundary_conditions.yaml")
    materials = load_yaml("configs/materials.yaml")
    ambient = float(boundary["ambient_temperature_k"])
    top_h = float(boundary["external_surface"]["top_convection_coefficient_w_m2_k"])
    bottom_h = float(
        boundary["external_surface"]["bottom_convection_coefficient_w_m2_k"]
    )
    absorption = float(boundary["laser"]["absorption_fraction"])
    beam_radius = float(boundary["laser"]["beam_radius_m"])
    sic_k = float(materials["silicon_carbide"]["conductivity_w_m_k"]["value"])
    copper_k = float(materials["copper"]["conductivity_w_m_k"]["value"])
    sic_surface: list[BoundarySamples] = []
    copper_surface: list[BoundarySamples] = []
    interface: list[BoundarySamples] = []
    for power in sorted(float(value) for value in powers_w):
        field = load_processed_field(power)
        sic_surface.append(
            _surface_samples(
                field,
                material_id=1,
                surface="top",
                conductivity_w_m_k=sic_k,
                ambient_temperature_k=ambient,
                convection_coefficient_w_m2_k=top_h,
                absorption_fraction=absorption,
                beam_radius_m=beam_radius,
            )
        )
        copper_surface.extend(
            (
                _surface_samples(
                    field,
                    material_id=0,
                    surface="top",
                    conductivity_w_m_k=copper_k,
                    ambient_temperature_k=ambient,
                    convection_coefficient_w_m2_k=top_h,
                    absorption_fraction=absorption,
                    beam_radius_m=beam_radius,
                ),
                _surface_samples(
                    field,
                    material_id=0,
                    surface="bottom",
                    conductivity_w_m_k=copper_k,
                    ambient_temperature_k=ambient,
                    convection_coefficient_w_m2_k=bottom_h,
                    absorption_fraction=absorption,
                    beam_radius_m=beam_radius,
                ),
            )
        )
        interface_samples, _ = _interface_samples(field, sic_k)
        interface.append(interface_samples)
    return {
        "silicon_carbide_emissivity": _concat(sic_surface),
        "copper_emissivity": _concat(copper_surface),
        "contact_resistance_m2_k_w": _concat(interface),
    }


def identify_physics_parameters(
    output_json: str | Path = "reports/physics_parameter_identification.json",
    output_markdown: str | Path | None = "reports/physics_parameter_identification.md",
) -> dict[str, Any]:
    splits = build_power_splits()
    train_powers = sorted(splits.simulation_train)
    validation_powers = sorted(splits.simulation_validation)
    train = collect_parameter_samples(train_powers)
    validation = collect_parameter_samples(validation_powers)
    fits = {
        "silicon_carbide_emissivity": fit_scalar_through_origin(
            train["silicon_carbide_emissivity"], lower=0.0, upper=1.0
        ),
        "copper_emissivity": fit_scalar_through_origin(
            train["copper_emissivity"], lower=0.0, upper=1.0
        ),
        "contact_resistance_m2_k_w": fit_scalar_through_origin(
            train["contact_resistance_m2_k_w"], lower=0.0
        ),
    }
    result: dict[str, Any] = {
        "schema_version": 2,
        "status": "PRELIMINARY_LOW_FIDELITY_INITIALIZATION",
        "train_powers_w": train_powers,
        "validation_powers_w": validation_powers,
        "test_powers_accessed": False,
        "method": {
            "temperature_source": "processed low-fidelity simulation fields",
            "normal_gradient": "three-point one-sided second-order finite difference",
            "fit": "least squares through physical zero intercept",
            "role": "initialization for later multi-fidelity PINN joint identification",
        },
        "fits": {
            name: _fit_record(
                name,
                fit,
                evaluate_scalar(validation[name], fit.value),
            )
            for name, fit in fits.items()
        },
        "material_passport": {
            "simulation_data_used": True,
            "experiment_data_used": False,
            "sensor_data_used": False,
            "internal_experiment_truth": "not available",
            "verification_status": "ANALYZED_NOT_PRODUCTION_VERIFIED",
        },
    }
    destination = Path(output_json)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if output_markdown is not None:
        markdown_destination = Path(output_markdown)
        if not markdown_destination.is_absolute():
            markdown_destination = PROJECT_ROOT / markdown_destination
        markdown_destination.parent.mkdir(parents=True, exist_ok=True)
        markdown_destination.write_text(_render_markdown(result), encoding="utf-8")
    return result
