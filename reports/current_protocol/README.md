# Current fixed protocol

This directory contains small, versioned evidence for the active fixed-split
protocol. Generated entries must describe real local inputs or completed runs;
missing experiments remain explicitly marked as not run.

- High-fidelity train/validation/test powers are fixed at 12/3/3.
- Low-fidelity simulation train/validation/test powers are fixed at 60/10/10.
- Test temperatures are excluded from training, preprocessing fitting, checkpoint
  selection, and uncertainty calibration.
- Processed Experiment and test IR/ring observations are stored in separate files;
  non-test loaders do not open the test files.
- `data_inventory.json` records file identity and provenance only. It deliberately
  excludes test-temperature distribution statistics.
- `resolved_physics.yaml` distinguishes confirmed values, effective
  initializations, assumed scenarios, and train-only learned values.
- A release under `reports/releases/` must be frozen before the test entry point
  accepts its checkpoints.

Large data, checkpoints, predictions, and run logs remain outside Git.
