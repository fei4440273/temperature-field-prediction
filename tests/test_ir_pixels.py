from __future__ import annotations

import numpy as np
import pytest

from sic_cu.eval.ir_pixels import pixel_frame_metrics


def test_pixel_metrics_separate_axisymmetric_floor() -> None:
    target_c = np.array([20.0, 22.0, 30.0, 34.0])
    bins = np.array([0, 0, 1, 1])
    radial_mean_k = np.array([21.0, 21.0, 32.0, 32.0]) + 273.15
    result = pixel_frame_metrics(target_c, radial_mean_k, bins)
    expected_floor = np.sqrt((1.0 + 1.0 + 4.0 + 4.0) / 4)
    assert result["pixel_rmse_k"] == pytest.approx(expected_floor)
    assert result["axisymmetric_floor_rmse_k"] == pytest.approx(expected_floor)
    assert result["radial_profile_rmse_k"] == pytest.approx(0.0, abs=1e-7)
