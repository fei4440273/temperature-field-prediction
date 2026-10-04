# Corrected cold sensor data and 25 C initialization

## Authorized Scope

- Refresh the processed 169 W cold test sensor from the user's corrected CSV; preserve all other observations and splits.
- Set the experimental copper hot/cold initial temperature to exactly 25 C in all four operators, histories, and physical initial conditions. Preserve simulation and SiC initial temperatures at 22 C.
- Preserve ASL learned blocks, architecture, loss weights, seed, validation selection, and the 1000-epoch training budget.
- Generate only PNG figures now and in future exports. Retain historical result archives as provenance.
- Recalculate results against current observations without comparing incompatible historical cached test metrics.
- Verify results and synchronize the authorized V5 GitHub release branch.

## Execution

- [x] Add regression tests for initial conditions, inference consistency, history normalization, and PNG exports.
- [x] Refresh the corrected sensor cache with source hashes and a revision record.
- [x] Implement the shared initial-temperature contract and PNG export policy.
- [x] Run focused tests and a short real-data smoke experiment (44 tests; all four methods completed five real-data updates).
- [x] Train all four methods for 1000 epochs using a frozen configuration and sources.
- [x] Evaluate current measurements, audit dense temporal predictions, and regenerate PNG figures (17 PNGs; 36 dense curves; current-only diagnostics).
- [x] Verify and independently review the implementation and artifacts (46 focused tests; full current-data artifact audit; 133 release-worktree tests passed, 1 optional upstream-reference test skipped).
- [x] Prepare the verified V5 publication and synchronized result directory; confirm the remote branch after pushing.

## Version Folder

`../temperature-field-prediction-V5` is a Git worktree for branch V5, created for publication on 2026-10-03 around 17:21 +0800. The main directory remains the V4 development worktree and contains unrelated uncommitted work. The prior published sequential implementation matches the development sequential implementation before this revision.
