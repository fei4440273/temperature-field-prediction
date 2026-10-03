# ASL Temperature Bias Optimization Plan

> **For agentic workers:** Execute the authorized optimization inline and verify the real end-to-end result before completing the active goal.

**Goal:** Explain ASL's power-dependent signed temperature errors and reduce experimental curve errors, especially the late near-center top response, without restoring oscillations.

**Architecture:** Keep all learned ASL modules and history preprocessing. Add measured near-center supervision and per-radius late temperature, endpoint and trend constraints. Fit each radius separately so opposite central/rim errors cannot cancel. Select the configuration on training/validation data; retain the same initialization and actual 1000-update budget.

**Tech Stack:** Existing PyTorch, NumPy, Table/BatchTable, YAML and matplotlib.

## Evidence And Constraints

- Published baseline: V5 commit `9ba8f101f48f918dd79c9a64160030500760d35b`, `Sequential_DeepONet_1000epochs_ASL_thermal`.
- Baseline final near-center test biases: 169 W +3.930 K, 339 W -8.286 K, 634 W -5.596 K. At 634 W, late whole-top mean bias is +3.869 K despite the negative center bias.
- Training-power late central/rim errors also differ in sign; 729 W central-region mean bias -16.400 K vs outer-region +2.442 K. This is already training underfit, not only test extrapolation.
- One seeded diagnostic gives shared-gradient cosine -0.277 between all-top and center-tail losses. This supports objective competition; it does not establish a unique physical cause.
- No per-test-power offset fitting, current/future input labels, output filtering, forced monotonicity, or new learned modules.
- Existing four-method comparison and source snapshots stay archived. Experimental changes and model selection must be disclosed, including the previously inspected test set.

## Task 1: Measured Curve Objectives And Regression Tests

**Files:** `scripts/sequential_temperature_objective.py`, `scripts/train_sequential_deeponet.py`, `tests/test_sequential_temperature_objective.py`, `configs/sequential_deeponet_tail.yaml`.

**Interfaces:** `center_tail_indices(table, fraction)` selects the actual smallest-radius point per frame in each power's own late interval. `TopCurveConstraints(table, provider, device, window_s)` exposes `losses(model)` with normalized late-temperature, per-radius slope-change and measured-endpoint losses. `center_table(table, radius_m)` builds training-only central observations.

- [x] Write failing tests for own-power tail support, measured minimum-radius selection, opposing radial slopes, absolute bias versus trend bias, usable-window validation and differentiable raw-model losses.
- [x] Implement the objectives using actual observed timestamps and cached past-only histories; preserve old-config behavior when weights are zero.
- [x] Add independent center-sampling RNG state and optional center-tail validation scoring; keep base observation and temporal RNG schedules unchanged.
- [x] Freeze first candidate: center radius5 mm, last20 s measured constraints; top-center weight4, top-tail2, slope10 and endpoint2. Keep all original losses and 1000 updates.
- [x] Run focused tests and a five-update real-data smoke experiment. Independent review confirms exact RNG restoration, unchanged modules, and compatible legacy validation scoring. Align all command defaults after selection.

## Task 2: Validation-Guided Optimization And Final Comparison

**Files:** New candidate/final result directories and a reproducible signed-error analysis entry point.

- [x] Train the first ASL candidate for actual1000 updates and compare validation full-top, near-center late and sensor accuracy with the published baseline. Center-tail7.3873 ->4.3636K, whole-top7.8109 ->8.9312K, selection score6.6622 ->6.7039K. Candidate1 is not promoted.
- [x] If evidence requires a revision, change one diagnosed factor and keep every run/config/source snapshot. Choose using validation metrics only. Candidate2: validation score5.4845K, whole-top7.3084K, center-tail3.5275K. Selected on2026-10-03 16:15:25UTC before final training/test evaluation; selection.json records both candidates and the baseline.

Candidate2 changes only the full-top observation weight2 ->8. Center/tail weights, inputs, architecture, sampling and1000 updates are identical. Its target is to retain the central tail improvement while correcting the whole-curve regression. Configuration: `configs/sequential_deeponet_temperature.yaml`.
- [x] Lock the selected protocol and complete four same-protocol1000-update runs for the final comparison. All checkpoints were verified before opening final test labels.
- [x] Evaluate held-out experimental curves, including signed center/rim errors, each power's late support, final offsets and late slopes. ASL whole-top6.7109 ->6.1333K, center-tail6.1361 ->3.6970K. Per-power center-tail169W3.5420 ->0.8061K;339W7.9046 ->5.7943K;634W5.6227 ->2.0941K. All3 final absolute central offsets decrease. Hot/cold0.6834/0.7034 ->0.9489/0.9259K; disclose these increases. Residual late slope errors and339W final-6.260K remain. Final formal validation score5.4652K vs candidate2's5.4845K; same seeded protocol does not imply bit-identical GPU trajectories.
- [x] Preserve raw dense predictions; audit early/mid/late sensor updates and top curves. Export revised Figure8 and useful before/after error diagnostics. Actual36dense cases cover24sensor+12center-top curves. ASL late incoming jumps0.014282K(sensor),0.085693K(top); post3s top0.202148K, initial2s top4.239655K disclosed. Manually inspect Figure8 and120scloud,17PNG+PDF exported.

## Task 3: Verification, Review And Publication

**Files:** Scoped audit/plot/verification, README/release notes and isolated V5 worktree.

- [x] Independently recompute new bias/trend metrics, actual optimizer steps, source/data/model/figure hashes and unchanged ASL learned structure. Full verifier passed, including72temperature metric groups, all radial/aggregate metrics, candidate validation archives and both historical/raw thermal baseline comparisons. Main38focused tests passed. Isolated full suite275passed/1reference skip/2missing V4 artifact failures; unchanged failing source and absent previous-tree fixtures confirmed, disclosed in V5-release.md.
- [x] Review the implementation and real numerical/visual results under requesting-code-review; resolve material findings. Independent final review reports no material issue:48experimental groups,304signed-bias groups,36dense cases,20cloud maxima and unchanged initial ASL tensors verified. Command-default, reproduction-metadata and verifier-coverage findings resolved.
- [ ] Update user-facing default results and publish only validated source/results to the authorized V5 branch.
- [ ] Verify remote state and audit the complete user objective before marking the active goal complete.
