from __future__ import annotations

import numpy as np

from sic_cu.physics.parameter_identification import (
    BoundarySamples,
    derivative_weights,
    fit_scalar_through_origin,
)


def test_one_sided_derivative_weights_are_exact_for_quadratic() -> None:
    coordinates = np.asarray([0.0, -0.001, -0.002])
    values = 4.0 + 3.0 * coordinates + 2.0 * coordinates**2
    weights = derivative_weights(coordinates, 0.0)
    assert np.isclose(np.dot(weights, values), 3.0, atol=1e-10)


def test_scalar_fit_tracks_unconstrained_and_physical_value() -> None:
    samples = BoundarySamples(
        power_w=np.asarray([10.0, 10.0, 20.0, 20.0]),
        predictor=np.asarray([1.0, 2.0, 3.0, 4.0]),
        response=np.asarray([1.2, 2.4, 3.6, 4.8]),
    )
    fit = fit_scalar_through_origin(samples, lower=0.0, upper=1.0)
    assert np.isclose(fit.unconstrained_value, 1.2)
    assert fit.value == 1.0
    assert fit.sample_count == 4
    assert fit.power_count == 2

