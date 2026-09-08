from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from sic_cu.physics.geometry import material_mask

from .common import ModelScales, TemperatureOutput, build_mlp


@dataclass(frozen=True)
class PODBasis:
    mesh_rz_m: np.ndarray
    mean_k: np.ndarray
    modes: np.ndarray
    singular_values: np.ndarray
    cumulative_energy: np.ndarray
    material_ids: np.ndarray | None = None


def save_pod_basis(basis: PODBasis, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        mesh_rz_m=basis.mesh_rz_m,
        mean_k=basis.mean_k,
        modes=basis.modes,
        singular_values=basis.singular_values,
        cumulative_energy=basis.cumulative_energy,
        material_ids=np.asarray([], dtype=np.int64)
        if basis.material_ids is None
        else basis.material_ids,
    )


def load_pod_basis(path: str | Path) -> PODBasis:
    with np.load(Path(path)) as values:
        material_ids = values["material_ids"]
        return PODBasis(
            mesh_rz_m=values["mesh_rz_m"],
            mean_k=values["mean_k"],
            modes=values["modes"],
            singular_values=values["singular_values"],
            cumulative_energy=values["cumulative_energy"],
            material_ids=None if material_ids.size == 0 else material_ids,
        )


def fit_pod(
    snapshots_k: np.ndarray,
    mesh_rz_m: np.ndarray,
    max_modes: int = 50,
    material_ids: np.ndarray | None = None,
) -> PODBasis:
    if snapshots_k.ndim != 2:
        raise ValueError("snapshots_k must have shape [snapshots, nodes]")
    if mesh_rz_m.shape != (snapshots_k.shape[1], 2):
        raise ValueError("mesh shape does not match snapshot nodes")
    working = np.asarray(snapshots_k, dtype=np.float64)
    mean = working.mean(axis=0)
    centered = working - mean
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    mode_count = min(max_modes, right.shape[0])
    energy = singular_values**2
    cumulative = np.clip(
        np.cumsum(energy, dtype=np.float64)
        / max(float(np.sum(energy, dtype=np.float64)), np.finfo(float).eps),
        0.0,
        1.0,
    )
    return PODBasis(
        mesh_rz_m=np.asarray(mesh_rz_m, dtype=np.float32),
        mean_k=np.asarray(mean, dtype=np.float32),
        modes=np.asarray(right[:mode_count].T, dtype=np.float32),
        singular_values=np.asarray(singular_values[:mode_count], dtype=np.float32),
        cumulative_energy=np.asarray(cumulative, dtype=np.float64),
        material_ids=None if material_ids is None else np.asarray(material_ids, dtype=np.int64),
    )


class PODPINN(nn.Module):
    """Predict POD coefficients and interpolate fixed FEM modes at query coordinates."""

    def __init__(
        self,
        basis: PODBasis,
        scales: ModelScales = ModelScales(),
        modes: int = 20,
        width: int = 128,
        neighbors: int = 4,
    ) -> None:
        super().__init__()
        mode_count = min(modes, basis.modes.shape[1])
        if mode_count < 1:
            raise ValueError("PODPINN requires at least one mode")
        self.neighbors = min(neighbors, basis.mesh_rz_m.shape[0])
        self.scales = scales
        self.register_buffer("mesh_rz_m", torch.from_numpy(basis.mesh_rz_m))
        self.register_buffer("mean_k", torch.from_numpy(basis.mean_k).view(-1, 1))
        self.register_buffer("modes", torch.from_numpy(basis.modes[:, :mode_count]))
        self.register_buffer(
            "material_ids",
            torch.empty(0, dtype=torch.long)
            if basis.material_ids is None
            else torch.from_numpy(basis.material_ids).long(),
        )
        self.coefficient_net = build_mlp(2, mode_count, width, 3, "tanh")
        self.output = TemperatureOutput(0.0, scales.temperature_scale_k)

    def _interpolate_basis(
        self,
        query_rz: Tensor,
        query_material_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        distances = torch.cdist(query_rz, self.mesh_rz_m.to(query_rz))
        if self.material_ids.numel() and query_material_ids is not None:
            same_material = query_material_ids.reshape(-1, 1).long() == self.material_ids.reshape(
                1, -1
            )
            distances = distances.masked_fill(~same_material, torch.inf)
        nearest_distance, nearest_index = torch.topk(
            distances, k=self.neighbors, dim=1, largest=False
        )
        weights = 1.0 / nearest_distance.clamp_min(1e-8)
        weights = weights / weights.sum(dim=1, keepdim=True)
        mean = (self.mean_k[nearest_index].squeeze(-1) * weights).sum(dim=1, keepdim=True)
        modes = (self.modes[nearest_index] * weights.unsqueeze(-1)).sum(dim=1)
        return mean, modes

    def predict_coefficients(self, power_time: Tensor) -> Tensor:
        if power_time.ndim != 2 or power_time.shape[1] != 2:
            raise ValueError("power_time must have shape [N, 2] ordered as power,time")
        normalized = torch.stack(
            (
                2.0 * power_time[:, 0] / self.scales.power_max_w - 1.0,
                2.0 * power_time[:, 1] / self.scales.time_max_s - 1.0,
            ),
            dim=-1,
        )
        return self.coefficient_net(normalized)

    def forward(self, coordinates: Tensor, coefficient_only: bool = False) -> Tensor:
        if coefficient_only:
            return self.predict_coefficients(coordinates)
        coefficients = self.predict_coefficients(coordinates[:, [3, 2]])
        query_material_ids = (
            coordinates[:, 4].long()
            if coordinates.shape[1] == 5
            else material_mask(coordinates[:, :2])
        )
        mean, modes = self._interpolate_basis(coordinates[:, :2], query_material_ids)
        return mean + self.output((modes * coefficients).sum(dim=-1, keepdim=True))


class MaterialWisePODPINN(nn.Module):
    """Independent Cu/SiC POD bases with a geometry-routed continuous decoder."""

    def __init__(
        self,
        copper_basis: PODBasis,
        silicon_carbide_basis: PODBasis,
        scales: ModelScales = ModelScales(),
        modes: int = 20,
        width: int = 128,
        neighbors: int = 4,
    ) -> None:
        super().__init__()
        self.scales = scales
        self.copper = PODPINN(copper_basis, scales, modes, width, neighbors)
        self.silicon_carbide = PODPINN(
            silicon_carbide_basis,
            scales,
            modes,
            width,
            neighbors,
        )

    def predict_coefficients(self, power_time: Tensor) -> Tensor:
        return torch.cat(
            (
                self.copper.predict_coefficients(power_time),
                self.silicon_carbide.predict_coefficients(power_time),
            ),
            dim=-1,
        )

    def forward(self, coordinates: Tensor, coefficient_only: bool = False) -> Tensor:
        if coefficient_only:
            return self.predict_coefficients(coordinates)
        materials = (
            coordinates[:, 4].long()
            if coordinates.shape[1] == 5
            else material_mask(coordinates[:, :2])
        )
        output = torch.empty((len(coordinates), 1), dtype=coordinates.dtype, device=coordinates.device)
        copper = materials == 0
        silicon_carbide = materials == 1
        if bool(copper.any()):
            output[copper] = self.copper(coordinates[copper])
        if bool(silicon_carbide.any()):
            output[silicon_carbide] = self.silicon_carbide(coordinates[silicon_carbide])
        return output
