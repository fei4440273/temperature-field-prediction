# ASL balanced and smooth response

Goal: improve ASL below each frozen FNN/GRU/LSTM experimental RMSE while removing
the artificial early gate corner. Preserve the SSM -> LSTM architecture, causal
histories, physical equations, and experimental copper initial temperature of 25 C.

Architecture: opt-in differentiable maturity and local-window transitions; balanced
observational training and validation selection that checks every measured region.
Legacy defaults preserve existing checkpoints. Baseline checkpoints and predictions
remain immutable. The comparison combines these fixed baselines with new ASL output.

Technology: PyTorch, NumPy, Polars, pytest, Matplotlib; real CUDA training only.
Execute in this session, with recorded configurations and frozen experiment sources.

- [x] Diagnose training-objective regression and gate derivative discontinuity.
- [x] Add failing tests for continuous ASL response and legacy compatibility.
- [x] Implement opt-in smooth ASL transitions and balanced validation selection.
- [x] Run and record ASL-only candidates; select using training/validation evidence.
- [x] Verify top < 5.939946568 K, hot < 0.642608036 K, cold < 0.645021947 K,
      combined < 4.270692359 K on the existing experimental benchmark.
- [x] Verify raw dense early and late curves, gate derivatives, causality, and t=0.
- [x] Export PNG-only figures and metrics with frozen baseline provenance.
- [x] Independently review the final code and artifacts; verify baseline identity.
- [x] Publish the verified change to V5.

The existing test benchmark has already been inspected during prior development.
This is refinement on that known benchmark, not evidence of an untouched blind test.
Record all candidates and selection choices; do not smooth output curves or change
baseline models to meet the acceptance targets. A run must finish all declared
updates before its validation-selected checkpoint is released.

Candidate 1: Gaussian maturity, smooth window, hot/cold weights 50, 1000 fresh
updates. Best validation ratio 1.0644; top validation 8.2992 K. Rejected on validation.
Candidate 2: softplus-smoothed original gate, smooth window, balanced losses,
1000 additional updates from original ASL (2000 total completed). Validation chose
update 750: top 6.4130 K, hot 0.89757 K, cold 0.68967 K, ratio 0.96983. Locked before
evaluation on the known existing test benchmark. No other baseline was retrained.

Candidate 2 locked-test RMSE: top 5.091397 K, hot 0.467239 K, cold 0.438634 K,
combined 3.614394 K, simulation 7.779938 K; all five beat each fixed baseline.
All 18 ASL prediction files were independently recomputed from the selected model.
Each non-ASL array was compared exactly to its baseline; original checkpoint
identities match the previously published evaluation provenance. All 17 PNGs and
20 cloud maxima passed verification. Maximum frozen-history slope change across
18 +/- 0.001 s is 0.00006667 K/s (previously up to 0.47885 K/s).
Late incoming sensor changes are below 0.019 K; no output smoothing is applied.
Independent review confirmed no remaining substantive issues. Development tests:
52 passed. Release tests: 51 passed, 1 optional upstream-reference test skipped.
Published code, checkpoints, predictions, and 17 PNGs to the GitHub V5 branch in
commit d95a579. The original V5 tag and all three baseline artifacts are unchanged.
