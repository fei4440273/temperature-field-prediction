# V5: balanced ASL and smooth early response

The latest result is `研究记录/ASL_balanced_smooth_candidate_2_20261004/`, using
`configs/asl_balanced_smooth_2.yaml`. Only ASL changed. FNN, GRU, and LSTM retain
their exact previous checkpoints, histories, metrics, and pointwise prediction arrays.

The earlier objective emphasized top-center and late-time trends, improving the
top while increasing sensor error. Its ASL maturity gate also had derivative
corners at 18 and 72 seconds. The adaptation preserves the two selective SSM
blocks, LSTM, state initialization, and 76,227 parameters. A softplus transition
smooths the original gate; a smooth local window removes the 21-second window
corner. Balanced supervision and the worst normalized regional validation RMSE
prevent top improvements from hiding sensor regressions.

| Method | Top RMSE / K | Hot RMSE / K | Cold RMSE / K | Combined RMSE / K |
|---|---:|---:|---:|---:|
| FNN | 5.9399 | 1.3498 | 0.7533 | 4.2707 |
| GRU | 6.1591 | 0.6426 | 0.6450 | 4.3789 |
| LSTM | 7.0958 | 1.1419 | 0.8475 | 5.0677 |
| ASL | **5.0914** | **0.4672** | **0.4386** | **3.6144** |

ASL simulation-test RMSE is 7.7799 K, also below all three fixed baselines.
These are aggregate errors on an already inspected benchmark; they do not imply
zero error or superiority on every individual observation. ASL initialized from
its previous 1000-update checkpoint and completed 1000 additional updates, with
validation selecting update 750. The fixed baselines completed 1000 updates.
This release does not claim an equal-budget architecture comparison.

The new verifier recomputes all ASL experimental, dense temporal, simulation, and
surface prediction arrays from the selected model, checks executing-source hashes,
and verifies exact identity of all non-ASL arrays with the published baseline.
Frozen-history derivative probes at 18/21/72 seconds confirm the switch corners
are removed. Copper initial temperature remains 25 C; SiC and simulation remain
22 C. All 17 updated figures are 300-DPI PNGs, with no output filtering.
Figure 8 uses dense raw sensor predictions and actual-time top predictions.

```bash
python scripts/train_sequential_deeponet.py --config configs/asl_balanced_smooth_2.yaml --methods asl --output '研究记录/ASL_new'
python scripts/inspect_asl_iteration.py '研究记录/ASL_new'
python scripts/evaluate_asl_iteration.py '研究记录/ASL_new' --checkpoint best.pt --figures
python scripts/verify_asl_iteration.py '研究记录/ASL_new'
```

`python scripts/plot_sequential_deeponet.py` now regenerates the latest ASL result
from saved predictions. The full local dataset is required for training or raw
model verification. Candidate 1 was rejected on validation; the plan and both
candidate validation records preserve the iteration evidence.

## Previous Sensor Revision

The 2026-10-04 revision uses the user's corrected `colddata169W.csv`. The 169 W cold test sensor cache has been refreshed; all other processed inputs are unchanged and checked by SHA256. The corrected first measurement is 25.29476 C at 1 s. Measurements are preserved without shifting them to the model's initial temperature.

Experimental copper, including both hot/cold sensors, starts at exactly 25 C. The model output anchor, physical initial-condition target, and experimental sensor history anchor/normalization agree. SiC and simulation initial conditions stay at 22 C. This is an explicit material-dependent initial profile. The ASL learned modules, parameter count (76,227), established thermal-trend inputs, losses, and training budget are retained.

All four operators completed 1000 actual AdamW updates with seed 123. Validation selects the supplemental best checkpoint; the main figures consistently use epoch 1000. The corrected test measurements were not used for a new hyperparameter search. Old cached metrics contain different sensor observations and initial conditions, so they are not presented as directly comparable baselines.

## Current Results

| Method | Top RMSE / K | Hot RMSE / K | Cold RMSE / K | Combined RMSE / K |
|---|---:|---:|---:|---:|
| FNN | 5.9399 | 1.3498 | 0.7533 | 4.2707 |
| GRU | 6.1591 | 0.6426 | 0.6450 | 4.3789 |
| LSTM | 7.0958 | 1.1419 | 0.8475 | 5.0677 |
| ASL | 6.1948 | 0.9917 | 0.8248 | 4.4276 |

For ASL hot/cold predictions at t >= 75 s, the maximum 0.1 s change is 0.01413 K and the maximum incoming observation-boundary change is 0.01282 K. The top-center late incoming change is 0.09540 K; an early change of 0.38596 K remains at 2 s. No output smoothing or monotonic projection is used. Measured errors remain, including the ASL 339 W final center bias of -5.7993 K. The report preserves current raw prediction diagnostics.

## Artifacts

- Current configuration: `configs/sequential_deeponet_sensor25.yaml`.
- Current results: `研究记录/Sequential_DeepONet_1000epochs_sensor25_20261004/`.
- Data revision: `data_revision/data_revision.json`, corrected CSV snapshot, and previous test cache/manifest.
- Training protocol: `revision_protocol.json`, configuration, splits, input hashes, and source snapshots.
- Each method: initial, latest, epoch-1000, and validation-best checkpoints with full history/logs.
- Evaluation: saved pointwise predictions, metrics, CSV summaries, 36 dense temporal curves, and cloud maxima.
- Figures: 17 PNGs at 300 DPI, with 11 experimental main figures and 6 appendix figures. Current and future export commands produce PNG only.
- Verification: `verification.json`, including exact sensor anchors, raw/cache agreement, actual update counts, hashes, recomputed metrics, and PNG checks.

The complete raw/processed `data/` directory and third-party reference repositories are not included. The corrected sensor source and previous cache are included as evidence for this revision. Retraining and full input verification require the rest of the local project dataset; regenerating published images uses saved predictions and diagnostics.

Historical result directories retain their original observations, model assumptions, and file formats: `Sequential_DeepONet_1000epochs_temperature_tail`, `Sequential_DeepONet_1000epochs_ASL_thermal`, `Sequential_DeepONet_1000epochs_history_fix_v3`, and `Sequential_DeepONet_1000epochs`. They are historical evidence, not the current figures.

## Verification And Reproduction

The development workspace passed 46 sequential tests, 15 data tests, 110 joint-model tests, and the full artifact audit. The release worktree passed 133 relevant tests; one optional upstream ASL alignment test was skipped because its reference repository is not included. Independent review found no remaining actionable issues.

For a fresh output directory and a complete local dataset:

```bash
python scripts/refresh_sequential_sensor_data.py --destination '研究记录/Sequential_DeepONet_sensor25_new/data_revision'
python scripts/train_sequential_deeponet.py --config configs/sequential_deeponet_sensor25.yaml --device cuda --output '研究记录/Sequential_DeepONet_sensor25_new'
python scripts/evaluate_sequential_deeponet.py --device cuda --output '研究记录/Sequential_DeepONet_sensor25_new'
python scripts/analyze_sequential_temperature_bias.py --output '研究记录/Sequential_DeepONet_sensor25_new' --methods fnn gru lstm asl --splits train validation test
python scripts/audit_sequential_temporal.py --output '研究记录/Sequential_DeepONet_sensor25_new'
python scripts/plot_sequential_deeponet.py --output '研究记录/Sequential_DeepONet_sensor25_new'
python scripts/verify_sequential_deeponet.py --output '研究记录/Sequential_DeepONet_sensor25_new'
```

Use `--resume` only to resume the same training configuration and directory. Evaluation archives the corrected source automatically and rejects changed raw data or caches. The model remains a reconstruction conditioned on past hot/cold measurements, with a 1 s sensor lag; it is not an autonomous prediction without sensor inputs. Physics coordinate derivatives hold histories fixed.

## V5 Worktree

`../temperature-field-prediction-V5` was created on 2026-10-03 at 17:21:51 +0800 for publishing branch V5. Git's worktree HEAD log and branch reflog record its creation. It is another working directory of the same repository, not a different algorithm or an automatic backup. The development worktree also contains unrelated, unpublished work. The V5 branch receives verified sequential code/results; the original annotated V5 tag retains the first release snapshot.
