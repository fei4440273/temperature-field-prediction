"""Conservative one-dimensional plate PINN conditions from locked Task 10 data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from sic_cu.config import load_yaml
from sic_cu.eval.task10_plate_benchmark import load_benchmark_registration


@dataclass(frozen=True)
class BoardCollocation:
    initial: Tensor
    top: Tensor
    bottom: Tensor
    interface_sic: Tensor
    interface_cu: Tensor
    interior: Tensor


def registered_board_collocation(registration_path: str | Path, flux_w_m2: int, *,
                                 points: int, time_s: float,
                                 dtype: torch.dtype = torch.float64) -> BoardCollocation:
    setup = load_benchmark_registration(registration_path)
    if flux_w_m2 not in setup["flux_splits_w_m2"]["train"]:
        raise PermissionError("方法物理训练配点仅接受已登记训练热流，禁止42/58数值测试")
    if points < 2 or not 0 < time_s <= setup["reference_controls"]["end_time_s"]:
        raise ValueError("同板体内和边界配点数至少2、时间须在(0,200]秒")
    sic_length = float(setup["geometry"]["silicon_carbide_thickness_m"])
    end_depth = sic_length + float(setup["geometry"]["copper_thickness_m"])
    sic_depths = torch.linspace(sic_length / (points + 1),
                                sic_length * points / (points + 1), points, dtype=dtype)
    cu_depths = torch.linspace(sic_length + (end_depth - sic_length) / (points + 1),
                               sic_length + (end_depth - sic_length) * points / (points + 1),
                               points, dtype=dtype)

    def rows(depths: Tensor, at: float, material: float) -> Tensor:
        n = len(depths)
        return torch.column_stack((depths, torch.full((n,), at, dtype=dtype),
                                   torch.full((n,), flux_w_m2, dtype=dtype),
                                   torch.full((n,), material, dtype=dtype)))

    interior = torch.cat((rows(sic_depths, time_s, 1.0),
                          rows(cu_depths, time_s, 0.0)), dim=0)
    return BoardCollocation(
        initial=torch.cat((rows(sic_depths, 0.0, 1.0),
                           rows(cu_depths, 0.0, 0.0)), 0),
        top=rows(torch.zeros(points, dtype=dtype), time_s, 1.0),
        bottom=rows(torch.full((points,), end_depth, dtype=dtype), time_s, 0.0),
        interface_sic=rows(torch.full((points,), sic_length, dtype=dtype), time_s, 1.0),
        interface_cu=rows(torch.full((points,), sic_length, dtype=dtype), time_s, 0.0),
        interior=interior,
    )


class RegisteredBoardPhysics:
    """F2 omits interior PDE entirely; F1/F3 keep it with the same other terms."""

    def __init__(self, registration_path: str | Path, *, method: str):
        if method not in ("F1", "F2", "F3"):
            raise ValueError("独立一维板物理仅支持F1/F2/F3")
        setup = load_benchmark_registration(registration_path)
        materials = load_yaml(setup["materials_source"])
        self.method = method
        self.compute_pde = method != "F2"
        self.interior_pde_calls = 0
        self.sic_length = float(setup["geometry"]["silicon_carbide_thickness_m"])
        self.end_depth = self.sic_length + float(setup["geometry"]["copper_thickness_m"])
        self.training_probe_times_s = tuple(setup["observations"]["train_times_s"])
        self.training_probe_depths_m = tuple(setup["observations"]["allowed_depths_from_top_m"])
        self.training_fluxes_w_m2 = tuple(setup["flux_splits_w_m2"]["train"])
        self.initial_k = float(setup["thermal_conditions"]["initial_temperature_k"])
        self.bottom_k = float(setup["thermal_conditions"]["bottom_fixed_temperature_k"])
        self.contact_m2_k_w = float(setup["thermal_conditions"]["benchmark_contact_resistance_m2_k_w"])
        self.conductivity = {
            1: float(materials["silicon_carbide"]["conductivity_w_m_k"]["value"]),
            0: float(materials["copper"]["conductivity_w_m_k"]["value"]),
        }
        self.capacity = {
            1: float(materials["silicon_carbide"]["density_kg_m3"])
               * float(materials["silicon_carbide"]["heat_capacity_j_kg_k"]["value"]),
            0: float(materials["copper"]["density_kg_m3"])
               * float(materials["copper"]["heat_capacity_j_kg_k"]["value"]),
        }
        self.temperature_scale_k = 25.0
        self.flux_scale_w_m2 = float(max(max(group) for group in setup["flux_splits_w_m2"].values()))
        self.pde_scale_w_m3 = max(
            max(self.capacity.values()) * self.temperature_scale_k / 200.0,
            max(self.conductivity.values()) * self.temperature_scale_k / self.end_depth ** 2,
        )

    @staticmethod
    def _at_model_dtype(model: nn.Module, rows: Tensor) -> Tensor:
        parameter = next(model.parameters(), None)
        return rows.to(device=parameter.device, dtype=parameter.dtype) if parameter is not None else rows

    def _first_derivatives(self, model: nn.Module, rows: Tensor) -> tuple[Tensor, Tensor]:
        coords = self._at_model_dtype(model, rows).detach().clone().requires_grad_(True)
        temperature = model(coords)
        gradient = torch.autograd.grad(temperature.sum(), coords, create_graph=True)[0]
        return temperature, gradient

    def _interior_pde(self, model: nn.Module, rows: Tensor) -> Tensor:
        self.interior_pde_calls += 1
        coords = self._at_model_dtype(model, rows).detach().clone().requires_grad_(True)
        temperature = model(coords)
        gradient = torch.autograd.grad(temperature.sum(), coords, create_graph=True)[0]
        depth_gradient = gradient[:, 0]
        if depth_gradient.requires_grad:
            second = torch.autograd.grad(depth_gradient.sum(), coords,
                                         create_graph=True, allow_unused=True)[0]
            depth_second = (second[:, 0] if second is not None
                            else torch.zeros_like(depth_gradient))
        else:
            depth_second = torch.zeros_like(depth_gradient)
        material = coords[:, 3] > .5
        conductivity = torch.where(material, self.conductivity[1], self.conductivity[0])
        capacity = torch.where(material, self.capacity[1], self.capacity[0])
        return (capacity * gradient[:, 1] - conductivity * depth_second) / self.pde_scale_w_m3

    @staticmethod
    def _square_mean(residual: Tensor) -> Tensor:
        return residual.square().mean()

    def components(self, model: nn.Module, batch: BoardCollocation) -> dict[str, Tensor | None]:
        if not all(rows.ndim == 2 and rows.shape[1] == 4 for rows in
                   (batch.initial, batch.top, batch.bottom, batch.interface_sic,
                    batch.interface_cu, batch.interior)):
            raise ValueError("同板配点绝不能含旧圆柱r-z坐标第五列")
        pde = self._square_mean(self._interior_pde(model, batch.interior)) if self.compute_pde else None
        initial = model(self._at_model_dtype(model, batch.initial))
        top_temperature, top_gradient = self._first_derivatives(model, batch.top)
        del top_temperature
        top_flux = -self.conductivity[1] * top_gradient[:, 0:1]
        bottom = model(self._at_model_dtype(model, batch.bottom))
        sic_temp, sic_grad = self._first_derivatives(model, batch.interface_sic)
        cu_temp, cu_grad = self._first_derivatives(model, batch.interface_cu)
        sic_flux = -self.conductivity[1] * sic_grad[:, 0:1]
        cu_flux = -self.conductivity[0] * cu_grad[:, 0:1]
        return {
            "pde": pde,
            "initial": self._square_mean((initial - self.initial_k) / self.temperature_scale_k),
            "top": self._square_mean((top_flux - batch.top[:, 2:3].to(device=top_flux.device,
                                                                        dtype=top_flux.dtype))
                                     / self.flux_scale_w_m2),
            "bottom": self._square_mean((bottom - self.bottom_k) / self.temperature_scale_k),
            "interface_flux": self._square_mean((sic_flux - cu_flux) / self.flux_scale_w_m2),
            "interface_jump": self._square_mean((sic_temp - cu_temp
                                                   - self.contact_m2_k_w * sic_flux)
                                                  / self.temperature_scale_k),
        }
