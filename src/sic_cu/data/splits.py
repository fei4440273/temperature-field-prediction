from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from sic_cu.config import load_yaml


def _canonical(power: float) -> float:
    return round(float(power), 4)


@dataclass(frozen=True)
class PowerSplits:
    simulation_train: frozenset[float]
    simulation_validation: frozenset[float]
    simulation_test: frozenset[float]
    hf_train: frozenset[float]
    hf_validation: frozenset[float]
    hf_test: frozenset[float]
    external_sensor_test: frozenset[float]

    @property
    def experiment_powers(self) -> frozenset[float]:
        """Experiment_data powers assigned to training or model validation."""
        return self.hf_train | self.hf_validation

    @property
    def test_only(self) -> frozenset[float]:
        """Powers sourced exclusively from test_Data."""
        return self.hf_test


def _as_set(values: Iterable[float]) -> frozenset[float]:
    return frozenset(_canonical(value) for value in values)


def build_power_splits(path: str = "configs/splits.yaml") -> PowerSplits:
    cfg = load_yaml(path)
    all_sim = _as_set(range(10, 801, 10))
    sim_val = _as_set(cfg["simulation"]["validation_powers_w"])
    sim_test = _as_set(cfg["simulation"]["test_powers_w"])
    sim_train = all_sim - sim_val - sim_test
    hf = cfg["high_fidelity"]
    external = cfg["external_sensor_test"]
    result = PowerSplits(
        simulation_train=sim_train,
        simulation_validation=sim_val,
        simulation_test=sim_test,
        hf_train=_as_set(hf["training_powers_w"]),
        hf_validation=_as_set(hf["validation_powers_w"]),
        hf_test=_as_set(hf["test_powers_w"]),
        external_sensor_test=_as_set(
            external["interpolation_powers_w"]
            + external["extrapolation_relative_to_ir_powers_w"]
        ),
    )
    validate_splits(result)
    return result


def validate_splits(splits: PowerSplits) -> None:
    sim_sets = [splits.simulation_train, splits.simulation_validation, splits.simulation_test]
    if any(a & b for i, a in enumerate(sim_sets) for b in sim_sets[i + 1 :]):
        raise ValueError("Simulation power splits overlap")
    if set().union(*sim_sets) != set(_as_set(range(10, 801, 10))):
        raise ValueError("Simulation power splits do not cover 10-800 W")
    hf_sets = [splits.hf_train, splits.hf_validation, splits.hf_test]
    if any(a & b for i, a in enumerate(hf_sets) for b in hf_sets[i + 1 :]):
        raise ValueError("High-fidelity power splits overlap")
    if len(splits.hf_train) != 12 or len(splits.hf_validation) != 3:
        raise ValueError("Experiment_data must use the fixed 12-train/3-validation split")
    if len(splits.hf_test) != 3:
        raise ValueError("test_Data must contain exactly three held-out powers")
    if splits.test_only & splits.experiment_powers:
        raise ValueError("test_Data powers overlap Experiment_data powers")
    if splits.external_sensor_test & set().union(*hf_sets):
        raise ValueError("External sensor powers overlap IR powers")


def assert_no_hf_leakage(
    training_powers_by_modality: Mapping[str, Iterable[float]],
    forbidden_powers: Iterable[float],
) -> None:
    forbidden = _as_set(forbidden_powers)
    leaked: dict[str, list[float]] = {}
    for modality, powers in training_powers_by_modality.items():
        overlap = sorted(_as_set(powers) & forbidden)
        if overlap:
            leaked[modality] = overlap
    if leaked:
        raise RuntimeError(f"High-fidelity power leakage detected: {leaked}")


def logo_folds(*_args, **_kwargs):
    raise RuntimeError("LOGO is disabled; use the fixed train/validation/test protocol")


def grouped_five_fold_splits(*_args, **_kwargs):
    raise RuntimeError(
        "Grouped cross-validation is disabled; use the fixed train/validation/test protocol"
    )
