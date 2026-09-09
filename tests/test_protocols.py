from __future__ import annotations

import inspect

import pytest
import polars as pl

from sic_cu.data.splits import (
    build_power_splits,
    grouped_five_fold_splits,
    logo_folds,
)
from sic_cu.train.multifidelity import _filter_observations, train_multifidelity
from sic_cu.train.surface_residual import train_surface_residual
from sic_cu.eval.high_fidelity import evaluate_ir_surface
from sic_cu.eval.interpolation_benchmark import run_interpolation_benchmark
from sic_cu.eval.ir_pixels import evaluate_ir_pixels


def test_fixed_protocol_has_three_independent_power_sets() -> None:
    splits = build_power_splits()
    hf_groups = (splits.hf_train, splits.hf_validation, splits.hf_test)
    simulation_groups = (
        splits.simulation_train,
        splits.simulation_validation,
        splits.simulation_test,
    )

    assert [len(group) for group in hf_groups] == [12, 3, 3]
    assert [len(group) for group in simulation_groups] == [60, 10, 10]
    for groups in (hf_groups, simulation_groups):
        assert all(
            not left & right
            for index, left in enumerate(groups)
            for right in groups[index + 1 :]
        )


def test_logo_and_grouped_cross_validation_are_disabled() -> None:
    with pytest.raises(RuntimeError, match="LOGO is disabled"):
        logo_folds([])
    with pytest.raises(RuntimeError, match="cross-validation is disabled"):
        grouped_five_fold_splits()


def test_explicit_power_cannot_bypass_split_filter() -> None:
    frame = pl.DataFrame(
        {
            "power_w": [55.0, 169.0],
            "split": ["train", "test"],
        }
    )

    selected = _filter_observations(frame, "train", [169.0])

    assert selected.is_empty()


def test_multifidelity_training_does_not_evaluate_test_by_default() -> None:
    parameter = inspect.signature(train_multifidelity).parameters["evaluate_test"]
    assert parameter.default is False


def test_training_entry_points_reject_test_evaluation() -> None:
    with pytest.raises(ValueError, match="Training-time test evaluation is disabled"):
        train_multifidelity("unused.pt", "unused", evaluate_test=True)
    with pytest.raises(ValueError, match="Training-time test evaluation is disabled"):
        train_surface_residual("unused", evaluate_test=True)


def test_test_evaluators_require_a_frozen_release() -> None:
    with pytest.raises(RuntimeError, match="frozen release"):
        evaluate_ir_surface(split="test")
    with pytest.raises(RuntimeError, match="frozen release"):
        evaluate_ir_pixels(split="test")
    with pytest.raises(RuntimeError, match="frozen release"):
        run_interpolation_benchmark(split="test")
