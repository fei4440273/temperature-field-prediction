from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
import torch
from torch import nn

from .fields import SimulationField
from .splits import build_power_splits


@dataclass(frozen=True)
class SelectedNode:
    material_id: int
    node_label: int
    r_m: float
    z_m: float


@dataclass(frozen=True)
class SpatialSelection:
    nodes: tuple[SelectedNode, ...]
    excluded_copper_count: int
    copper_outer_radius_m: float


@dataclass(frozen=True)
class CurveMatrix:
    times_s: np.ndarray
    temperature_deltas_k: np.ndarray
    normalized_deltas: np.ndarray
    scales_k: np.ndarray
    power_w: np.ndarray
    material_ids: np.ndarray
    node_labels: np.ndarray
    initial_abs_delta_k: np.ndarray


@dataclass(frozen=True)
class FitCandidate:
    initial_tau_seconds: tuple[float, ...]
    tau_seconds: np.ndarray
    training_normalized_rmse: float
    regularized_objective: float
    converged: bool
    optimizer_message: str
    iterations: int
    function_evaluations: int
    feature_condition_number: float
    ridge_condition_number: float


@dataclass(frozen=True)
class DictionaryFit:
    candidates: tuple[FitCandidate, ...]
    selected_candidate_index: int | None
    training_amplitudes: np.ndarray


def validate_training_power_contract(powers_w: Iterable[float]) -> list[float]:
    selected = sorted(float(power) for power in powers_w)
    allowed = sorted(build_power_splits().simulation_train)
    if len(selected) != 60 or selected != allowed:
        raise ValueError("Dictionary fitting requires exactly 60 LF simulation training powers")
    return selected


def select_spatial_nodes(field: SimulationField, per_material: int = 32) -> SpatialSelection:
    if per_material < 1:
        raise ValueError("Select at least one node per material")
    coords = np.asarray(field.coordinates_rz_m, dtype=np.float64)
    labels = np.asarray(field.node_labels, dtype=np.int64)
    material_ids = np.asarray(field.material_ids, dtype=np.int64)
    if coords.shape != (len(labels), 2) or len(material_ids) != len(labels):
        raise ValueError("Invalid material-labeled LF mesh")
    if not np.isfinite(coords).all():
        raise ValueError("Nonfinite LF node coordinate")
    outer = float(coords[material_ids == 0, 0].max())
    outer_mask = (material_ids == 0) & (coords[:, 0] == outer)
    selected: list[SelectedNode] = []
    for material in (0, 1):
        indices = np.flatnonzero((material_ids == material) & ~outer_mask)
        if len(indices) < per_material:
            raise ValueError(f"Insufficient non-Dirichlet LF nodes for material {material}")
        xy = coords[indices]
        minima = xy.min(axis=0)
        spans = xy.max(axis=0) - minima
        normalized = np.divide(xy - minima, spans, out=np.zeros_like(xy), where=spans > 0)
        distances = np.linalg.norm(normalized - np.array([0.0, 1.0]), axis=1)
        first = min(range(len(indices)), key=lambda i: (float(distances[i]), int(labels[indices[i]])))
        order = [first]
        while len(order) < per_material:
            nearest = np.min(np.linalg.norm(
                normalized[:, None, :] - normalized[np.asarray(order)][None, :, :], axis=2,
            ), axis=1)
            nearest[order] = -np.inf
            next_index = min(range(len(indices)), key=lambda i: (-float(nearest[i]),
                                                               int(labels[indices[i]])))
            order.append(next_index)
        selected.extend(SelectedNode(material, int(labels[indices[i]]),
                                     float(coords[indices[i], 0]), float(coords[indices[i], 1]))
                        for i in order)
    return SpatialSelection(tuple(selected), int(outer_mask.sum()), outer)


def build_curve_matrix(fields: Sequence[SimulationField], selection: SpatialSelection) -> CurveMatrix:
    if not fields:
        raise ValueError("LF trajectories cannot be empty")
    reference = fields[0]
    if len(selection.nodes) % 2 or not selection.nodes:
        raise ValueError("The selected materials need equal nonzero node counts")
    expected_selection = select_spatial_nodes(reference, per_material=len(selection.nodes) // 2)
    if selection != expected_selection:
        raise ValueError("Frozen LF node selection differs from its source mesh")
    times = np.arange(0, 201, 2, dtype=np.float32)
    if not np.array_equal(reference.times_s, times):
        raise ValueError("LF time grid must be the aligned 0..200 s grid, every 2 s")
    index_lookup = {(int(mat), int(label)): index for index, (mat, label) in enumerate(
        zip(reference.material_ids, reference.node_labels)
    )}
    if len(index_lookup) != len(reference.material_ids):
        raise ValueError("Repeated material/node labels in LF mesh")
    indices = [index_lookup[(node.material_id, node.node_label)] for node in selection.nodes]
    delta_groups = []
    scales_groups = []
    powers = []
    materials = []
    labels = []
    initial_deltas = []
    for field in fields:
        if not np.array_equal(field.times_s, reference.times_s):
            raise ValueError("LF aligned time meshes differ")
        if (not np.array_equal(field.material_ids, reference.material_ids)
                or not np.array_equal(field.node_labels, reference.node_labels)
                or not np.array_equal(field.coordinates_rz_m, reference.coordinates_rz_m)):
            raise ValueError("LF material-labeled node meshes or coordinates differ")
        if field.temperature_k.shape != (101, len(index_lookup)):
            raise ValueError("Incomplete LF material/time mesh")
        temperatures = np.asarray(field.temperature_k[:, indices], dtype=np.float64)
        if not np.isfinite(temperatures).all():
            raise ValueError("Nonfinite LF temperature prevents time dictionary fitting")
        delta = temperatures - temperatures[0:1, :]
        tail = delta[1:, :].T
        scales = np.maximum(0.1, np.max(np.abs(tail), axis=1))
        delta_groups.append(tail)
        scales_groups.append(scales)
        initial_deltas.append(delta[0, :])
        powers.extend([float(field.power_w)] * len(indices))
        materials.extend(node.material_id for node in selection.nodes)
        labels.extend(node.node_label for node in selection.nodes)
    raw = np.concatenate(delta_groups, axis=0)
    scale = np.concatenate(scales_groups)
    return CurveMatrix(
        times_s=np.asarray(times[1:], dtype=np.float64), temperature_deltas_k=raw,
        normalized_deltas=raw / scale[:, None], scales_k=scale,
        power_w=np.asarray(powers, dtype=np.float64),
        material_ids=np.asarray(materials, dtype=np.int64),
        node_labels=np.asarray(labels, dtype=np.int64),
        initial_abs_delta_k=np.abs(np.concatenate(initial_deltas)),
    )


def _validate_tau(tau_seconds: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau_seconds, dtype=np.float64)
    if tau.shape != (4,) or not np.isfinite(tau).all() or np.any(tau <= 0) or np.any(np.diff(tau) <= 0):
        raise ValueError("Time dictionary needs four strictly ordered positive finite tau values")
    return tau


def _project(curves: np.ndarray, times: np.ndarray, tau: np.ndarray, ridge: float):
    features = -np.expm1(-times[:, None] / tau[None, :])
    gram = features.T @ features / len(times) + ridge * np.eye(4, dtype=np.float64)
    amplitudes = np.linalg.solve(gram, (curves @ features / len(times)).T).T
    residual = curves - amplitudes @ features.T
    loss = float(np.mean(residual * residual) + ridge * np.mean(np.sum(amplitudes * amplitudes, axis=1)))
    return features, gram, amplitudes, residual, loss


def project_fixed_dictionary(
    normalized_curves: np.ndarray, times_s: np.ndarray, tau_seconds: np.ndarray,
    ridge_lambda: float = 1e-6,
) -> dict[str, float | int | np.ndarray]:
    curves = np.asarray(normalized_curves, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    tau = _validate_tau(tau_seconds)
    if curves.ndim != 2 or curves.shape[1] != len(times) or not np.isfinite(curves).all():
        raise ValueError("Nonfinite or incompatible normalized LF curves")
    if np.any(times < 0) or not np.isfinite(times).all() or ridge_lambda <= 0:
        raise ValueError("Invalid LF time grid or ridge coefficient")
    features, gram, amplitude, residual, objective = _project(curves, times, tau, ridge_lambda)
    return {
        "normalized_rmse": float(np.sqrt(np.mean(residual * residual))),
        "regularized_objective": objective,
        "curve_count": curves.shape[0],
        "feature_condition_number": float(np.linalg.cond(features)),
        "ridge_condition_number": float(np.linalg.cond(gram)),
        "signed_normalized_amplitudes": amplitude,
        "residual": residual,
    }


def fit_time_dictionary(
    normalized_curves: np.ndarray, times_s: np.ndarray,
    initial_groups: Sequence[Sequence[float]], ridge_lambda: float = 1e-6,
) -> DictionaryFit:
    curves = np.asarray(normalized_curves, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    if curves.ndim != 2 or not len(curves) or curves.shape[1] != len(times) or not np.isfinite(curves).all():
        raise ValueError("Fitting needs finite normalized LF training curves")
    if len(initial_groups) != 2 or ridge_lambda <= 0:
        raise ValueError("Fit exactly two registered starting tau groups with positive ridge")
    first, second = (_validate_tau(np.asarray(group)) for group in initial_groups)
    candidates = []
    for start in (first, second):
        increments = np.diff(np.r_[0.0, start])
        initial = np.log(increments)

        def objective(log_increments: np.ndarray):
            steps = np.exp(log_increments)
            tau = np.cumsum(steps)
            feature, _, amplitude, residual, value = _project(curves, times, tau, ridge_lambda)
            derivative = -(times[:, None] / (tau[None, :] ** 2)) * np.exp(-times[:, None] / tau[None, :])
            tau_gradient = -2.0 / (len(curves) * len(times)) * np.einsum(
                "nt,nj,tj->j", residual, amplitude, derivative,
            )
            step_gradient = steps * np.cumsum(tau_gradient[::-1])[::-1]
            return value, step_gradient

        solution = minimize(objective, initial, method="L-BFGS-B", jac=True,
                            bounds=[(-10.0, 9.0)] * 4,
                            options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9})
        fitted = np.cumsum(np.exp(solution.x))
        feature, gram, amplitude, residual, value = _project(curves, times, fitted, ridge_lambda)
        candidates.append(FitCandidate(
            initial_tau_seconds=tuple(float(x) for x in start), tau_seconds=fitted,
            training_normalized_rmse=float(np.sqrt(np.mean(residual * residual))),
            regularized_objective=value, converged=bool(solution.success),
            optimizer_message=str(solution.message), iterations=int(solution.nit),
            function_evaluations=int(solution.nfev),
            feature_condition_number=float(np.linalg.cond(feature)),
            ridge_condition_number=float(np.linalg.cond(gram)),
        ))
    admissible = [i for i, result in enumerate(candidates) if result.converged
                  and np.isfinite(result.regularized_objective)]
    selected = min(admissible, key=lambda i: (candidates[i].regularized_objective, i)) if admissible else None
    amplitudes = (_project(curves, times, candidates[selected].tau_seconds, ridge_lambda)[2]
                  if selected is not None else np.empty((0, 4), dtype=np.float64))
    return DictionaryFit(tuple(candidates), selected, amplitudes)


class TimeResponseFeatures(nn.Module):
    def __init__(self, tau_seconds: Sequence[float]) -> None:
        super().__init__()
        tau = _validate_tau(np.asarray(tau_seconds, dtype=np.float64))
        self.register_buffer("tau_seconds", torch.as_tensor(tau))

    def forward(self, time_s: torch.Tensor) -> torch.Tensor:
        if time_s.shape[-1] != 1:
            raise ValueError("Time input needs a final singleton column")
        return -torch.expm1(-time_s / self.tau_seconds.to(dtype=time_s.dtype))
