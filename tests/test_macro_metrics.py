from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sic_cu.eval.metrics import (
    macro_metric_summary,
    macro_v1_selection_score,
    weighted_metrics,
)
from sic_cu.train.multifidelity import _weighted_validation


class _PowerOffset(nn.Module):
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return coordinates[:, 3:4]


def test_macro_rmse_does_not_weight_a_long_condition_more() -> None:
    records = [
        {"rmse_c": 1.0, "mae_c": 1.0, "mean_error_c": 1.0},
        {"rmse_c": 9.0, "mae_c": 9.0, "mean_error_c": 9.0},
    ]
    result = macro_metric_summary(records)

    assert result["rmse_c"] == pytest.approx(5.0)
    assert result["equal_power_mse_rmse_c"] == pytest.approx(np.sqrt(41.0))


def test_weighted_metrics_are_temperature_offset_invariant() -> None:
    target = np.array([300.0, 310.0, 320.0])
    prediction = np.array([301.0, 308.0, 324.0])
    weights = np.array([0.2, 0.3, 0.5])

    kelvin = weighted_metrics(target, prediction, weights)
    celsius = weighted_metrics(target - 273.15, prediction - 273.15, weights)

    assert kelvin == pytest.approx(celsius)


def test_macro_validation_is_batch_size_invariant() -> None:
    power = torch.tensor([1.0] * 2 + [9.0] * 8)
    coordinates = torch.zeros((10, 5))
    coordinates[:, 3] = power
    target = torch.zeros((10, 1))
    weights = torch.ones((10, 1))
    dataset = TensorDataset(coordinates, target, weights)
    model = _PowerOffset()

    small = _weighted_validation(
        model, DataLoader(dataset, batch_size=3), torch.device("cpu")
    )
    large = _weighted_validation(
        model, DataLoader(dataset, batch_size=10), torch.device("cpu")
    )

    assert torch.equal(small, large)
    assert small[0].item() == pytest.approx(5.0)
    assert small[2].item() == pytest.approx(np.sqrt(41.0))


def test_macro_v1_selection_formula_is_frozen() -> None:
    assert macro_v1_selection_score(2.0, 5.0, 3.0) == pytest.approx(6.0 / 2.2)
