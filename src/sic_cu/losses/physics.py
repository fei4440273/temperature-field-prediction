from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sic_cu.physics.boundary import convection_flux, laser_flux, radiation_flux
from sic_cu.physics.collocation import CollocationBatch
from sic_cu.physics.configuration import BoundaryConditions
from sic_cu.physics.heat_equation import axisymmetric_heat_residual
from sic_cu.physics.interface import interface_residuals, normal_heat_flux
from sic_cu.physics.materials import Material
from sic_cu.physics.trainable_parameters import TrainableBoundaryParameters


def _gradient(temperature: Tensor, coordinates: Tensor) -> Tensor:
    return torch.autograd.grad(
        temperature,
        coordinates,
        torch.ones_like(temperature),
        create_graph=True,
        retain_graph=True,
    )[0][:, :2]


def _mse(value: Tensor, scale: float) -> Tensor:
    return (value / max(float(scale), torch.finfo(value.dtype).eps)).pow(2).mean()


@dataclass(frozen=True)
class PhysicsLossWeights:
    pde: float = 1.0
    boundary: float = 1.0
    initial: float = 1.0
    interface: float = 1.0


class PhysicsLossComputer:
    """Dimensionless loss components for the common SiC-Cu physics protocol."""

    def __init__(
        self,
        materials: dict[int, Material],
        boundaries: BoundaryConditions,
        weights: PhysicsLossWeights = PhysicsLossWeights(),
        trainable_parameters: TrainableBoundaryParameters | None = None,
        temperature_scale_k: float = 250.0,
        length_scale_m: float = 0.0175,
        time_scale_s: float = 200.0,
    ) -> None:
        self.materials = materials
        self.boundaries = boundaries
        self.weights = weights
        self.trainable_parameters = trainable_parameters
        self.temperature_scale_k = temperature_scale_k
        conductivity = max(
            float(material.conductivity.value or max(material.conductivity.values))
            for material in materials.values()
        )
        volumetric_capacity = max(
            material.density_kg_m3
            * float(material.heat_capacity.value or max(material.heat_capacity.values))
            for material in materials.values()
        )
        self.pde_scale = max(
            volumetric_capacity * temperature_scale_k / time_scale_s,
            conductivity * temperature_scale_k / length_scale_m**2,
        )
        self.flux_scale = conductivity * temperature_scale_k / length_scale_m

    @staticmethod
    def _temperature_gradient(model: nn.Module, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        x = coordinates.detach().clone().requires_grad_(True)
        temperature = model(x)
        return temperature, _gradient(temperature, x)

    def _outward_flux(self, model: nn.Module, coordinates: Tensor, normal: Tensor, material_id: int) -> tuple[Tensor, Tensor]:
        temperature, gradient = self._temperature_gradient(model, coordinates)
        conductivity = self.materials[material_id].conductivity(temperature)
        return temperature, normal_heat_flux(conductivity, gradient, normal)

    def __call__(self, model: nn.Module, batch: CollocationBatch) -> dict[str, Tensor]:
        pde = axisymmetric_heat_residual(
            model,
            batch.interior.detach().clone().requires_grad_(True),
            batch.interior_material_ids,
            self.materials,
        )
        initial = model(batch.initial) - self.boundaries.initial_temperature_k

        axis_t, axis_grad = self._temperature_gradient(model, batch.axis)
        del axis_t
        axis = axis_grad[:, 0:1]

        top_normal = torch.tensor([[0.0, 1.0]], device=batch.sic_top.device).expand(len(batch.sic_top), -1)
        sic_top_t, sic_top_flux = self._outward_flux(model, batch.sic_top, top_normal, 1)
        applied = laser_flux(
            batch.sic_top[:, 0:1],
            batch.sic_top[:, 3:4],
            self.boundaries.laser.profile,
            self.boundaries.laser.absorption_fraction,
            self.boundaries.laser.beam_radius_m,
        )
        surface_loss = convection_flux(
            sic_top_t,
            self.boundaries.ambient_temperature_k,
            self.boundaries.top_convection_coefficient_w_m2_k,
        )
        if self.boundaries.radiation_enabled:
            sic_emissivity = (
                self.trainable_parameters.silicon_carbide_emissivity
                if self.trainable_parameters is not None
                else self.boundaries.silicon_carbide_emissivity
            )
            if sic_emissivity is None:
                raise ValueError("Missing SiC emissivity outside identification mode")
            surface_loss = surface_loss + radiation_flux(
                sic_top_t,
                self.boundaries.ambient_temperature_k,
                sic_emissivity,
            )
        # Outward conductive flux equals environmental loss minus inward laser flux.
        sic_top = sic_top_flux + applied - surface_loss

        exposed_residuals: list[Tensor] = []
        copper_top_normal = torch.tensor(
            [[0.0, 1.0]], device=batch.copper_top.device
        ).expand(len(batch.copper_top), -1)
        copper_top_t, copper_top_flux = self._outward_flux(
            model, batch.copper_top, copper_top_normal, 0
        )
        copper_top_loss = convection_flux(
            copper_top_t,
            self.boundaries.ambient_temperature_k,
            self.boundaries.top_convection_coefficient_w_m2_k,
        )
        if self.boundaries.radiation_enabled:
            copper_emissivity = (
                self.trainable_parameters.copper_emissivity
                if self.trainable_parameters is not None
                else self.boundaries.copper_emissivity
            )
            if copper_emissivity is None:
                raise ValueError("Missing copper emissivity outside identification mode")
            copper_top_loss = copper_top_loss + radiation_flux(
                copper_top_t,
                self.boundaries.ambient_temperature_k,
                copper_emissivity,
            )
        exposed_residuals.append(copper_top_flux - copper_top_loss)

        external_surfaces = {
            "bottom": (
                batch.bottom,
                (0.0, -1.0),
                self.boundaries.bottom_convection_coefficient_w_m2_k,
            ),
            "outer_radius": (
                batch.outer_radius,
                (1.0, 0.0),
                self.boundaries.side_convection_coefficient_w_m2_k,
            ),
        }
        cooling: Tensor | None = None
        for surface, (coordinates, normal, coefficient) in external_surfaces.items():
            normal_tensor = torch.tensor([normal], device=coordinates.device).expand(
                len(coordinates), -1
            )
            temperature, flux = self._outward_flux(model, coordinates, normal_tensor, 0)
            if surface == self.boundaries.cooling.surface:
                if self.boundaries.cooling.kind == "fixed_temperature":
                    cooling = (
                        temperature - float(self.boundaries.cooling.fixed_temperature_k)
                    ) * (self.flux_scale / self.temperature_scale_k)
                else:
                    cooling = flux - convection_flux(
                        temperature,
                        self.boundaries.ambient_temperature_k,
                        float(self.boundaries.cooling.heat_transfer_coefficient_w_m2_k),
                    )
                continue
            if coefficient is None:
                raise ValueError(f"Missing convection coefficient for exposed {surface}")
            expected = convection_flux(
                temperature,
                self.boundaries.ambient_temperature_k,
                coefficient,
            )
            if self.boundaries.radiation_enabled:
                copper_emissivity = (
                    self.trainable_parameters.copper_emissivity
                    if self.trainable_parameters is not None
                    else self.boundaries.copper_emissivity
                )
                if copper_emissivity is None:
                    raise ValueError("Missing copper emissivity outside identification mode")
                expected = expected + radiation_flux(
                    temperature,
                    self.boundaries.ambient_temperature_k,
                    copper_emissivity,
                )
            exposed_residuals.append(flux - expected)
        if cooling is None:
            raise ValueError(f"Unsupported cooling surface: {self.boundaries.cooling.surface}")

        sic_t, sic_gradient = self._temperature_gradient(model, batch.interface_sic)
        copper_t, copper_gradient = self._temperature_gradient(model, batch.interface_copper)
        sic_flux = normal_heat_flux(
            self.materials[1].conductivity(sic_t), sic_gradient, batch.interface_normals
        )
        copper_flux = normal_heat_flux(
            self.materials[0].conductivity(copper_t),
            copper_gradient,
            batch.interface_normals,
        )
        contact_resistance = (
            self.trainable_parameters.contact_resistance_m2_k_w
            if self.trainable_parameters is not None
            else self.boundaries.contact_resistance_m2_k_w
        )
        interface_temperature, interface_flux = interface_residuals(
            sic_t,
            copper_t,
            sic_flux,
            copper_flux,
            contact_resistance,
        )
        components = {
            "pde": _mse(pde, self.pde_scale),
            "initial": _mse(initial, self.temperature_scale_k),
            "boundary": (
                _mse(axis, self.temperature_scale_k / 0.0175)
                + _mse(sic_top, self.flux_scale)
                + sum(_mse(value, self.flux_scale) for value in exposed_residuals)
                + _mse(cooling, self.flux_scale)
            ),
            "interface": _mse(interface_temperature, self.temperature_scale_k)
            + _mse(interface_flux, self.flux_scale),
        }
        components["physics_total"] = (
            self.weights.pde * components["pde"]
            + self.weights.initial * components["initial"]
            + self.weights.boundary * components["boundary"]
            + self.weights.interface * components["interface"]
        )
        return components
