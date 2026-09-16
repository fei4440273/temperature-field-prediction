from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from numpy.polynomial.chebyshev import chebvander2d
from torch import Tensor, nn


SUPPORTED_RESIDUAL_INTERPOLATION_METHODS = (
    "nearest_residual",
    "linear_residual",
    "linear_residual_per_watt",
    "mean_residual_per_watt",
    "regression_residual_per_watt",
)


@dataclass(frozen=True)
class ResidualTrajectory:
    """Radial surface-residual trajectory observed at one laser power."""

    power_w: float
    times_s: np.ndarray
    radii_m: np.ndarray
    residual_k: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=np.float64)
        radii = np.asarray(self.radii_m, dtype=np.float64)
        residual = np.asarray(self.residual_k, dtype=np.float64)
        if self.power_w <= 0.0:
            raise ValueError("Residual trajectory power must be positive")
        if times.ndim != 1 or radii.ndim != 1:
            raise ValueError("Trajectory times and radii must be one-dimensional")
        if residual.shape != (len(times), len(radii)):
            raise ValueError("Residual array must have shape (time, radius)")
        if len(times) == 0 or len(radii) == 0:
            raise ValueError("Residual trajectory cannot be empty")
        if np.any(np.diff(times) <= 0.0) or np.any(np.diff(radii) <= 0.0):
            raise ValueError("Trajectory times and radii must be strictly increasing")
        if not np.isfinite(residual).all():
            raise ValueError("Residual trajectory contains non-finite values")

    def evaluate(self, times_s: np.ndarray, radii_m: np.ndarray) -> np.ndarray:
        """Interpolate in radius/time, use zero at t=0, and hold the final frame."""
        query_times = np.asarray(times_s, dtype=np.float64)
        query_radii = np.asarray(radii_m, dtype=np.float64)
        if query_times.shape != query_radii.shape:
            raise ValueError("Query times and radii must have equal shapes")
        if np.any(query_times < 0.0):
            raise ValueError("Query time cannot be negative")

        times = np.asarray(self.times_s, dtype=np.float64)
        values = np.asarray(self.residual_k, dtype=np.float64)
        if times[0] > 0.0:
            times = np.concatenate(([0.0], times))
            values = np.vstack((np.zeros((1, values.shape[1])), values))

        clipped_times = np.minimum(query_times.reshape(-1), times[-1])
        flat_radii = query_radii.reshape(-1)
        radial_values = np.vstack(
            [
                np.interp(
                    flat_radii,
                    self.radii_m,
                    profile,
                    left=profile[0],
                    right=profile[-1],
                )
                for profile in values
            ]
        )
        high = np.searchsorted(times, clipped_times, side="right")
        high = np.minimum(high, len(times) - 1)
        low = np.maximum(high - 1, 0)
        interval = times[high] - times[low]
        fraction = np.divide(
            clipped_times - times[low],
            interval,
            out=np.zeros_like(clipped_times),
            where=interval > 0.0,
        )
        columns = np.arange(len(clipped_times))
        result = radial_values[low, columns] + fraction * (
            radial_values[high, columns] - radial_values[low, columns]
        )
        return result.reshape(query_times.shape).astype(np.float32)


class PowerResidualInterpolator:
    """Interpolate complete-power experimental residual trajectories without leakage."""

    def __init__(self, trajectories: Iterable[ResidualTrajectory]) -> None:
        ordered = sorted(trajectories, key=lambda item: item.power_w)
        powers = np.asarray([item.power_w for item in ordered], dtype=np.float64)
        if len(ordered) < 2:
            raise ValueError("At least two source powers are required")
        if np.any(np.diff(powers) <= 0.0):
            raise ValueError("Source powers must be unique")
        self.trajectories = tuple(ordered)
        self.powers_w = powers

    def _source_pair(self, power_w: float) -> tuple[int, int, float]:
        high = int(np.searchsorted(self.powers_w, power_w, side="right"))
        if high == 0:
            low, high = 0, 1
        elif high == len(self.powers_w):
            low, high = len(self.powers_w) - 2, len(self.powers_w) - 1
        else:
            low = high - 1
        fraction = (power_w - self.powers_w[low]) / (
            self.powers_w[high] - self.powers_w[low]
        )
        return low, high, float(fraction)

    def predict(
        self,
        power_w: float,
        times_s: np.ndarray,
        radii_m: np.ndarray,
        method: str = "linear_residual_per_watt",
    ) -> np.ndarray:
        if method not in SUPPORTED_RESIDUAL_INTERPOLATION_METHODS:
            raise ValueError(f"Unsupported residual interpolation method: {method}")
        if power_w <= 0.0:
            return np.zeros_like(np.asarray(times_s), dtype=np.float32)
        low, high, fraction = self._source_pair(float(power_w))
        low_values = self.trajectories[low].evaluate(times_s, radii_m)
        high_values = self.trajectories[high].evaluate(times_s, radii_m)
        if method == "nearest_residual":
            return low_values if fraction <= 0.5 else high_values
        if method == "linear_residual":
            return (low_values + fraction * (high_values - low_values)).astype(np.float32)
        if method == "linear_residual_per_watt":
            low_normalized = low_values / self.powers_w[low]
            high_normalized = high_values / self.powers_w[high]
            return (
                power_w
                * (low_normalized + fraction * (high_normalized - low_normalized))
            ).astype(np.float32)

        normalized = np.stack(
            [
                trajectory.evaluate(times_s, radii_m) / trajectory.power_w
                for trajectory in self.trajectories
            ]
        )
        mean = normalized.mean(axis=0)
        if method == "mean_residual_per_watt":
            return (power_w * mean).astype(np.float32)
        centered_power = self.powers_w - self.powers_w.mean()
        slope = np.tensordot(centered_power, normalized, axes=(0, 0)) / np.sum(
            centered_power**2
        )
        predicted_normalized = mean + (power_w - self.powers_w.mean()) * slope
        return (power_w * predicted_normalized).astype(np.float32)


def _chebyshev_basis(values: Tensor, degree: int) -> Tensor:
    terms = [torch.ones_like(values)]
    if degree > 0:
        terms.append(values)
    for _ in range(2, degree + 1):
        terms.append(2.0 * values * terms[-1] - terms[-2])
    return torch.cat(terms, dim=-1)


class ChebyshevSurfaceResidualGuide(nn.Module):
    """Smooth, differentiable regression guide for the SiC top-surface residual."""

    def __init__(
        self,
        mean_coefficients: np.ndarray | Tensor,
        slope_coefficients: np.ndarray | Tensor,
        power_center_w: float,
        radius_max_m: float = 0.025,
        time_max_s: float = 200.0,
        initial_ramp_time_s: float = 0.05,
        output_offset_k: float | None = None,
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(mean_coefficients, dtype=torch.float32)
        slope = torch.as_tensor(slope_coefficients, dtype=torch.float32)
        if mean.ndim != 2 or slope.shape != mean.shape:
            raise ValueError("Surface guide coefficients must be equal 2D arrays")
        if radius_max_m <= 0.0 or time_max_s <= 0.0 or initial_ramp_time_s <= 0.0:
            raise ValueError("Surface guide scales must be positive")
        self.register_buffer("mean_coefficients", mean)
        self.register_buffer("slope_coefficients", slope)
        self.power_center_w = float(power_center_w)
        self.radius_max_m = float(radius_max_m)
        self.time_max_s = float(time_max_s)
        self.initial_ramp_time_s = float(initial_ramp_time_s)
        self.output_offset_k = (
            None if output_offset_k is None else float(output_offset_k)
        )

    @property
    def degree_r(self) -> int:
        return self.mean_coefficients.shape[0] - 1

    @property
    def degree_t(self) -> int:
        return self.mean_coefficients.shape[1] - 1

    def forward(self, coordinates: Tensor) -> Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] < 4:
            raise ValueError("Surface guide expects [r,z,t,power,...] coordinates")
        radius = 2.0 * coordinates[:, 0:1] / self.radius_max_m - 1.0
        time = 2.0 * coordinates[:, 2:3] / self.time_max_s - 1.0
        power = coordinates[:, 3:4]
        radial_basis = _chebyshev_basis(radius, self.degree_r)
        time_basis = _chebyshev_basis(time, self.degree_t)
        mean = torch.einsum(
            "ni,ij,nj->n", radial_basis, self.mean_coefficients, time_basis
        )[:, None]
        slope = torch.einsum(
            "ni,ij,nj->n", radial_basis, self.slope_coefficients, time_basis
        )[:, None]
        initial_gate = -torch.expm1(
            -torch.clamp(coordinates[:, 2:3], min=0.0) / self.initial_ramp_time_s
        )
        value = initial_gate * power * (
            mean + (power - self.power_center_w) * slope
        )
        return value if self.output_offset_k is None else self.output_offset_k + value

    def to_spec(self) -> dict[str, Any]:
        return {
            "kind": "chebyshev_power_regression",
            "mean_coefficients": self.mean_coefficients.detach().cpu().tolist(),
            "slope_coefficients": self.slope_coefficients.detach().cpu().tolist(),
            "power_center_w": self.power_center_w,
            "radius_max_m": self.radius_max_m,
            "time_max_s": self.time_max_s,
            "initial_ramp_time_s": self.initial_ramp_time_s,
            "output_offset_k": self.output_offset_k,
        }

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> "ChebyshevSurfaceResidualGuide":
        if spec.get("kind") != "chebyshev_power_regression":
            raise ValueError("Unsupported surface guide specification")
        return cls(
            spec["mean_coefficients"],
            spec["slope_coefficients"],
            float(spec["power_center_w"]),
            float(spec["radius_max_m"]),
            float(spec["time_max_s"]),
            float(spec["initial_ramp_time_s"]),
            (
                None
                if spec.get("output_offset_k") is None
                else float(spec["output_offset_k"])
            ),
        )


def fit_chebyshev_surface_residual_guide(
    predictor: PowerResidualInterpolator,
    degree_r: int = 20,
    degree_t: int = 20,
    radial_samples: int = 101,
    time_samples: int = 101,
    radius_max_m: float = 0.025,
    time_max_s: float = 200.0,
    output_offset_k: float | None = None,
) -> ChebyshevSurfaceResidualGuide:
    if min(degree_r, degree_t) < 0:
        raise ValueError("Chebyshev degrees must be nonnegative")
    if radial_samples <= degree_r or time_samples <= degree_t:
        raise ValueError("Fit grid must contain more samples than polynomial degree")
    radii = np.linspace(0.0, radius_max_m, radial_samples)
    times = np.linspace(0.0, time_max_s, time_samples)
    time_grid, radius_grid = np.meshgrid(times, radii, indexing="ij")
    flat_times = time_grid.reshape(-1)
    flat_radii = radius_grid.reshape(-1)
    normalized = np.stack(
        [
            trajectory.evaluate(flat_times, flat_radii) / trajectory.power_w
            for trajectory in predictor.trajectories
        ]
    ).astype(np.float64)
    powers = predictor.powers_w.astype(np.float64)
    power_center = float(powers.mean())
    centered = powers - power_center
    mean = normalized.mean(axis=0)
    slope = np.tensordot(centered, normalized, axes=(0, 0)) / np.sum(centered**2)
    scaled_r = 2.0 * flat_radii / radius_max_m - 1.0
    scaled_t = 2.0 * flat_times / time_max_s - 1.0
    vandermonde = chebvander2d(
        scaled_r, scaled_t, [degree_r, degree_t]
    ).reshape(len(flat_radii), -1)
    mean_coefficients = np.linalg.lstsq(vandermonde, mean, rcond=1e-8)[0]
    slope_coefficients = np.linalg.lstsq(vandermonde, slope, rcond=1e-8)[0]
    shape = (degree_r + 1, degree_t + 1)
    return ChebyshevSurfaceResidualGuide(
        mean_coefficients.reshape(shape),
        slope_coefficients.reshape(shape),
        power_center,
        radius_max_m,
        time_max_s,
        output_offset_k=output_offset_k,
    )
