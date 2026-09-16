from __future__ import annotations

import numpy as np

from sic_cu.eval.development_v4 import _window_mask, error_statistics


def test_v4_error_statistics_are_signed_and_in_celsius_differences() -> None:
    metrics = error_statistics(
        np.array([10.0, 20.0]),
        np.array([12.0, 18.0]),
        np.array([1.0, 3.0]),
    )
    assert np.isclose(metrics["rmse_c"], 2.0)
    assert np.isclose(metrics["mae_c"], 2.0)
    assert np.isclose(metrics["signed_bias_c"], -1.0)
    assert np.isclose(metrics["max_abs_error_c"], 2.0)


def test_v4_time_windows_have_preregistered_boundary_semantics() -> None:
    times = np.array([0.0, 30.0, 30.0001, 100.0, 100.0001, 200.0])
    first = _window_mask(
        times,
        {"lower": 0.0, "upper": 30.0, "lower_closed": True},
    )
    second = _window_mask(
        times,
        {"lower": 30.0, "upper": 100.0, "lower_closed": False},
    )
    third = _window_mask(
        times,
        {"lower": 100.0, "upper": 200.0, "lower_closed": False},
    )
    assert times[first].tolist() == [0.0, 30.0]
    assert times[second].tolist() == [30.0001, 100.0]
    assert times[third].tolist() == [100.0001, 200.0]
