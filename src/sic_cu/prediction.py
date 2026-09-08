from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.interpolate import interp1d

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.fields import SimulationField, load_processed_field
from sic_cu.models import (
    AdditiveCorrectionModel,
    ChebyshevSurfaceResidualGuide,
    GNOPINN,
    MaterialWisePODPINN,
    ModelScales,
    PODPINN,
    load_pod_basis,
)
from sic_cu.models.gno_pinn import knn_edges
from sic_cu.models.interpolation import interpolate_simulation_power
from sic_cu.train.simulation import build_model


SIMULATION_MIN_POWER_W = 10.0
SIMULATION_MAX_POWER_W = 800.0
PREDICTION_MIN_POWER_W = 0.0
IR_MIN_POWER_W = 55.0
IR_MAX_POWER_W = 729.0


@dataclass(frozen=True)
class PredictionMetadata:
    source: str
    support_domain: str
    ir_coverage: str
    deterministic: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class Prediction:
    power_w: float
    times_s: np.ndarray
    coordinates_rz_m: np.ndarray
    material_ids: np.ndarray
    mean_temperature_k: np.ndarray
    q05_temperature_k: np.ndarray
    q95_temperature_k: np.ndarray
    max_temperature_k: np.ndarray
    stable_time_s: float | None
    metadata: PredictionMetadata

    @property
    def mean_temperature_c(self) -> np.ndarray:
        return self.mean_temperature_k - 273.15

    def rotate(
        self,
        time_index: int,
        theta_resolution: int = 72,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return rotate_axisymmetric(
            self.coordinates_rz_m,
            self.mean_temperature_k[time_index],
            self.material_ids,
            theta_resolution,
        )


def _validate_power(power_w: float) -> float:
    power = float(power_w)
    if not np.isfinite(power):
        raise ValueError("power_w must be finite")
    if not PREDICTION_MIN_POWER_W <= power <= SIMULATION_MAX_POWER_W:
        raise ValueError("power_w must be within the supported range 0-800 W")
    return power


def _validate_times(times_s: Iterable[float] | None, source_times: np.ndarray) -> np.ndarray:
    if times_s is None:
        return source_times.astype(np.float32, copy=True)
    times = np.asarray(list(times_s), dtype=np.float64)
    if times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all():
        raise ValueError("times_s must be a nonempty sequence of finite values")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times_s must be strictly increasing")
    if times[0] < source_times[0] or times[-1] > source_times[-1]:
        raise ValueError(f"times_s must lie within {source_times[0]:g}-{source_times[-1]:g} s")
    return times.astype(np.float32)


def _time_interpolate(field: SimulationField, times_s: np.ndarray) -> np.ndarray:
    if np.array_equal(times_s, field.times_s):
        return field.temperature_k.copy()
    interpolator = interp1d(field.times_s, field.temperature_k, axis=0, kind="linear")
    return np.asarray(interpolator(times_s), dtype=np.float32)


def stable_time_from_maximum(
    times_s: np.ndarray,
    maximum_temperature_k: np.ndarray,
    tolerance_k: float = 0.01,
) -> float | None:
    """Earliest time after which every adjacent Tmax change stays within tolerance."""
    times = np.asarray(times_s, dtype=np.float64)
    maximum = np.asarray(maximum_temperature_k, dtype=np.float64)
    if times.ndim != 1 or maximum.shape != times.shape or len(times) == 0:
        raise ValueError("times and maximum temperature must be aligned 1D arrays")
    if len(times) == 1:
        return None
    stable_transitions = np.abs(np.diff(maximum)) <= float(tolerance_k)
    suffix = np.logical_and.accumulate(stable_transitions[::-1])[::-1]
    indices = np.flatnonzero(suffix)
    return float(times[indices[0]]) if len(indices) else None


def rotate_axisymmetric(
    coordinates_rz_m: np.ndarray,
    temperature_k: np.ndarray,
    material_ids: np.ndarray,
    theta_resolution: int = 72,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if theta_resolution < 3:
        raise ValueError("theta_resolution must be at least 3")
    coordinates = np.asarray(coordinates_rz_m, dtype=np.float64)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    materials = np.asarray(material_ids, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates_rz_m must have shape [nodes, 2]")
    if temperature.shape != (len(coordinates),) or materials.shape != (len(coordinates),):
        raise ValueError("temperature/material arrays must match node count")
    theta = np.linspace(0.0, 2.0 * np.pi, theta_resolution, endpoint=False)
    r = np.repeat(coordinates[:, 0], theta_resolution)
    z = np.repeat(coordinates[:, 1], theta_resolution)
    angles = np.tile(theta, len(coordinates))
    xyz = np.column_stack((r * np.cos(angles), r * np.sin(angles), z)).astype(np.float32)
    return (
        xyz,
        np.repeat(temperature, theta_resolution).astype(np.float32),
        np.repeat(materials, theta_resolution),
    )


class Predictor:
    """Deterministic LF interpolation or trained coordinate-model inference."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        interpolation_kind: str = "linear",
        interpolation_powers_w: Iterable[float] = range(10, 801, 10),
        device: str | torch.device | None = None,
    ) -> None:
        self.interpolation_kind = interpolation_kind
        self.interpolation_powers_w = [float(value) for value in interpolation_powers_w]
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: torch.nn.Module | None = None
        self.checkpoint_path: Path | None = None
        self.method: str | None = None
        self.material_passport: dict[str, object] = {}
        if checkpoint is not None:
            self.checkpoint_path = Path(checkpoint)
            payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.material_passport = dict(payload.get("material_passport", {}))
            method = str(payload["method"])
            scales = ModelScales(**payload["scales"])
            if method in {
                "mlp",
                "mlp_pinn",
                "lstm",
                "lstm_pinn",
                "deeponet",
                "deeponet_pinn",
            }:
                self.model = build_model(
                    method,
                    scales,
                    **payload.get("model_kwargs", {}),
                ).to(self.device)
            elif method in {"pod", "pod_pinn"}:
                if payload.get("basis_variant", "global") == "material_wise":
                    paths = payload["basis_path"]
                    copper_path = PROJECT_ROOT / paths["copper"]
                    sic_path = PROJECT_ROOT / paths["silicon_carbide"]
                    self.model = MaterialWisePODPINN(
                        load_pod_basis(copper_path),
                        load_pod_basis(sic_path),
                        scales,
                        **payload["model_kwargs"],
                    ).to(self.device)
                else:
                    basis_path = Path(payload["basis_path"])
                    if not basis_path.is_absolute():
                        basis_path = PROJECT_ROOT / basis_path
                    self.model = PODPINN(
                        load_pod_basis(basis_path),
                        scales,
                        **payload["model_kwargs"],
                    ).to(self.device)
            elif method in {"gno", "gno_pinn"}:
                self.model = GNOPINN(scales, **payload["model_kwargs"]).to(self.device)
            elif method == "multifidelity_correction":
                low_fidelity = build_model(
                    str(payload["low_fidelity_method"]),
                    scales,
                    **payload.get("low_fidelity_model_kwargs", {}),
                )
                surface_guide_spec = payload.get("surface_residual_guide_spec")
                surface_guide = (
                    None
                    if surface_guide_spec is None
                    else ChebyshevSurfaceResidualGuide.from_spec(surface_guide_spec)
                )
                self.model = AdditiveCorrectionModel(
                    low_fidelity,
                    scales,
                    freeze_low_fidelity=False,
                    surface_residual_guide=surface_guide,
                    **payload["correction_model_kwargs"],
                ).to(self.device)
            else:
                raise ValueError(f"Unsupported deployment checkpoint method: {method}")
            self.model.load_state_dict(payload["model_state"])
            self.model.eval()
            self.method = method

    @torch.no_grad()
    def _model_field(
        self,
        power_w: float,
        reference: SimulationField,
        times_s: np.ndarray | None = None,
    ) -> np.ndarray:
        evaluation_times = reference.times_s if times_s is None else times_s
        if isinstance(self.model, PODPINN):
            power_time = torch.from_numpy(
                np.column_stack(
                    (np.full(len(evaluation_times), power_w), evaluation_times)
                ).astype(np.float32)
            ).to(self.device)
            coefficients = self.model.predict_coefficients(power_time).cpu().numpy()
            mean = self.model.mean_k.cpu().numpy().reshape(-1)
            modes = self.model.modes.cpu().numpy()
            return (
                mean[None, :]
                + self.model.scales.temperature_scale_k * coefficients @ modes.T
            ).astype(np.float32)
        if isinstance(self.model, MaterialWisePODPINN):
            power_time = torch.from_numpy(
                np.column_stack(
                    (np.full(len(evaluation_times), power_w), evaluation_times)
                ).astype(np.float32)
            ).to(self.device)
            all_coefficients = self.model.predict_coefficients(power_time).cpu().numpy()
            result = np.empty(
                (len(evaluation_times), len(reference.material_ids)), dtype=np.float32
            )
            offset = 0
            for material_id, submodel in (
                (0, self.model.copper),
                (1, self.model.silicon_carbide),
            ):
                count = submodel.modes.shape[1]
                coefficients = all_coefficients[:, offset : offset + count]
                mean = submodel.mean_k.cpu().numpy().reshape(-1)
                modes = submodel.modes.cpu().numpy()
                result[:, reference.material_ids == material_id] = (
                    mean[None, :]
                    + self.model.scales.temperature_scale_k * coefficients @ modes.T
                )
                offset += count
            return result
        if isinstance(self.model, GNOPINN):
            mesh = torch.from_numpy(reference.coordinates_rz_m).to(self.device)
            material_ids = torch.from_numpy(reference.material_ids).to(self.device)
            edges = knn_edges(mesh, self.model.k)
            output = []
            for time_s in evaluation_times:
                coordinates = torch.cat(
                    (
                        mesh,
                        torch.full_like(mesh[:, :1], float(time_s)),
                        torch.full_like(mesh[:, :1], power_w),
                    ),
                    dim=1,
                )
                output.append(
                    self.model(coordinates, material_ids, edges).squeeze(-1).cpu().numpy()
                )
            return np.stack(output).astype(np.float32)
        coordinates = np.concatenate(
            [
                np.column_stack(
                    (
                        reference.coordinates_rz_m,
                        np.full(len(reference.coordinates_rz_m), time_s),
                        np.full(len(reference.coordinates_rz_m), power_w),
                        reference.material_ids,
                    )
                )
                for time_s in evaluation_times
            ]
        ).astype(np.float32)
        output: list[np.ndarray] = []
        for offset in range(0, len(coordinates), 65536):
            batch = torch.from_numpy(coordinates[offset : offset + 65536]).to(self.device)
            output.append(self.model(batch).cpu().numpy())
        return np.concatenate(output).reshape(len(evaluation_times), -1).astype(np.float32)

    def predict(
        self,
        power_w: float,
        times_s: Iterable[float] | None = None,
        n_samples: int = 1,
        theta_resolution: int = 72,
    ) -> Prediction:
        power = _validate_power(power_w)
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        if theta_resolution < 3:
            raise ValueError("theta_resolution must be at least 3")
        warnings: list[str] = []
        support_domain = "inside_10_800W_simulation_domain"
        if power == 0.0:
            reference = load_processed_field(SIMULATION_MIN_POWER_W)
            field = SimulationField(
                power_w=power,
                times_s=reference.times_s,
                coordinates_rz_m=reference.coordinates_rz_m,
                material_ids=reference.material_ids,
                node_labels=reference.node_labels,
                temperature_k=np.full_like(reference.temperature_k, 295.15),
            )
            source = "zero_power_uniform_observed_initial_field"
            support_domain = "zero_power_physics_anchor"
        elif self.model is None and power < SIMULATION_MIN_POWER_W:
            reference = load_processed_field(SIMULATION_MIN_POWER_W)
            fraction = power / SIMULATION_MIN_POWER_W
            temperature = 295.15 + fraction * (reference.temperature_k - 295.15)
            field = SimulationField(
                power_w=power,
                times_s=reference.times_s,
                coordinates_rz_m=reference.coordinates_rz_m,
                material_ids=reference.material_ids,
                node_labels=reference.node_labels,
                temperature_k=temperature.astype(np.float32),
            )
            source = "low_fidelity_linear_zero_to_10W_physics_anchor"
            support_domain = "between_zero_anchor_and_10W_simulation"
            warnings.append(
                "Power below 10 W is linearly anchored between the uniform 0 W field and the "
                "10 W simulation; no direct simulation or experiment exists in this interval."
            )
        elif self.model is None:
            field = interpolate_simulation_power(
                power,
                self.interpolation_powers_w,
                kind=self.interpolation_kind,
            )
            source = f"low_fidelity_fem_{self.interpolation_kind}_power_interpolation"
            warnings.append(
                "This is a low-fidelity simulation interpolation, not a high-fidelity PINN result."
            )
        else:
            reference = load_processed_field(10.0)
            model_times = _validate_times(times_s, reference.times_s)
            field = SimulationField(
                power_w=power,
                times_s=model_times,
                coordinates_rz_m=reference.coordinates_rz_m,
                material_ids=reference.material_ids,
                node_labels=reference.node_labels,
                temperature_k=self._model_field(power, reference, model_times),
            )
            source = f"{self.method}:{self.checkpoint_path}"
            if power < SIMULATION_MIN_POWER_W:
                support_domain = "below_checkpoint_training_power_range"
                warnings.append(
                    "Power is below the checkpoint training range of 10-800 W; this is model "
                    "extrapolation."
                )
            if not bool(self.material_passport.get("experiment_data_used", False)):
                warnings.append(
                    "This checkpoint was trained without experiment data and remains a "
                    "low-fidelity simulation model."
                )
            else:
                warnings.append(
                    "The internal field is a physics-guided inference constrained by surface or "
                    "sparse experiment observations, not internal experiment ground truth."
                )
        requested_times = _validate_times(times_s, field.times_s)
        temperature = _time_interpolate(field, requested_times)
        maximum = temperature.max(axis=1)
        ir_coverage = (
            "inside_ir_power_range"
            if IR_MIN_POWER_W <= power <= IR_MAX_POWER_W
            else "outside_ir_power_range"
        )
        if ir_coverage == "outside_ir_power_range":
            warnings.append("Power is outside the IR experiment range 115.2-630.5 W.")
        if n_samples != 1:
            warnings.append("The selected predictor is deterministic; requested samples are identical.")
        metadata = PredictionMetadata(
            source=source,
            support_domain=support_domain,
            ir_coverage=ir_coverage,
            deterministic=True,
            warnings=tuple(warnings),
        )
        return Prediction(
            power_w=power,
            times_s=requested_times,
            coordinates_rz_m=field.coordinates_rz_m,
            material_ids=field.material_ids,
            mean_temperature_k=temperature,
            q05_temperature_k=temperature.copy(),
            q95_temperature_k=temperature.copy(),
            max_temperature_k=maximum,
            stable_time_s=stable_time_from_maximum(requested_times, maximum),
            metadata=metadata,
        )


def predict(
    power_w: float,
    times_s: Iterable[float] | None = None,
    n_samples: int = 1,
    theta_resolution: int = 72,
    checkpoint: str | Path | None = None,
) -> Prediction:
    return Predictor(checkpoint=checkpoint).predict(power_w, times_s, n_samples, theta_resolution)


def metadata_json(prediction: Prediction) -> str:
    payload = asdict(prediction.metadata) | {
        "power_w": prediction.power_w,
        "stable_time_s": prediction.stable_time_s,
        "time_count": int(len(prediction.times_s)),
        "node_count": int(len(prediction.coordinates_rz_m)),
    }
    return json.dumps(payload, indent=2)
