# Metric definition: macro_v1

The primary unit of high-fidelity evaluation is one complete power condition.

- IR RMSE is computed per power after radial weights within each frame sum to one,
  then averaged arithmetically across powers.
- `equal_power_mse_rmse_c` is the square root of the mean per-power MSE and is
  reported under that distinct name.
- Hot and Cold errors are computed per curve, then averaged equally across sensor
  type and power. Repeated angular rows do not increase sample weight.
- The validation selection score is
  `(IR_macro_RMSE + 0.2 * ring_absolute_macro_RMSE +
  ring_delta_macro_RMSE) / 2.2`.
- Missing time intervals are `N/A` (`null` in JSON), never zero.
- Peak absolute error is reported in ℃. Peak-relative error uses the predefined
  temperature-rise denominator and is retained only as a legacy diagnostic.
- Temperature-rise curves use each curve's first real observation, not an implied
  measurement at `t=0`.
