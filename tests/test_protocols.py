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


def test_fixed_protocol_has_three_independent_power_sets() -> None:
    splits = build_power_splits()
    groups = (splits.hf_train, splits.hf_validation, splits.hf_test)

    assert [len(group) for group in groups] == [12, 3, 3]
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
