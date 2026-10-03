# Sequential DeepONet Implementation Plan

**Goal:** Implement and execute four comparable 1000-epoch runs and produce
PNG figures with a Chinese result report.

**Architecture:** New scripts use the existing data/physics interfaces.
Branch encoders implement FNN, GRU, LSTM, and selective SSM plus LSTM;
all share a trunk and multi-fidelity output structure.

**Tech stack:** PyTorch 2.5.1, NumPy, Polars, Matplotlib, existing PINN environment.

- [x] Read authenticated ASL source and record the exact encoder adaptation.
- [x] Add `tests/test_sequential_deeponet.py`; run to establish missing functionality.
- [x] Add `scripts/sequential_deeponet_data.py` for causal histories and grouped batches.
- [x] Add `scripts/sequential_deeponet_core.py` for branches and common operator network.
- [x] Verify causal histories, physical gradients, and checkpoint round trips with
  `/home/phl/anaconda3/envs/PINN/bin/python -m pytest tests/test_sequential_deeponet.py -q`.
- [x] Add `configs/sequential_deeponet.yaml` and
  `scripts/train_sequential_deeponet.py` with atomic, resumable checkpoints.
- [x] Run an isolated short real-data smoke experiment to measure runtime and validate losses.
- [x] Execute four full 1000-epoch runs using the same saved configuration and seed.
- [x] Evaluate final and validation-best models on held-out data; save complete predictions.
- [x] Use `scripts/plot_sequential_deeponet.py` for 300-DPI PNG/PDF figures and a report.
- [x] Inspect rendered figures and independently verify all histories/checkpoints/metrics.

Formal artifacts are in `研究记录/Sequential_DeepONet_1000epochs`.
The independent audit verified 1000 actual AdamW updates for every method,
matching source/data hashes, recomputed prediction metrics, and 16 PNG/PDF pairs.
The 19 focused tests passed. All 16 figures were inspected in a contact sheet;
accuracy/cost, histograms, ASL percentile fields, the 120-second surface fields,
and temporal responses were also inspected individually.

Execution is inline in the existing session. Existing unrelated experiments and
the previously retained model are not inputs to these runs.
