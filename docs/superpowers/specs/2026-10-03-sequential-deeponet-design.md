# Four sequential DeepONet methods

The requested comparison consists of FNN-, GRU-, LSTM-, and ASL-DeepONet,
each trained from scratch for 1000 epochs without boundary attention.
The ASL source is the authenticated GitHub repository at commit
4477c971007624133eec92bd7eeb5bb58d3b9c26. The S-DeepONet source is
Jasiuk-Research-Group/S-DeepONet, commit f502cfa322c0e1d619ee091f4889ee84c5d561eb.

This extends the existing SiC-Cu multi-fidelity reconstruction task. Each
method uses the same history input, spatial/time trunk, low/high fidelity
heads, data split, physical losses, seed, optimizer, and update budget.
Inputs contain normalized time, power, past hot/cold temperature and validity.
Long history has 32 samples, local history has 8 samples over 20 seconds;
history ends at max(0, query time minus 1 second). ASL uses the source's
selective state-space blocks and its SSM/LSTM fusion. No boundary-attention
module is imported or constructed.

The existing simulation split (60/10/10 powers) and experimental split
(12/3/3 powers) are retained for this new comparison. Simulation histories
come from the nearest bottom copper nodes to the nominal sensor radii.
Experimental histories come from measured sensor curves. Held-out histories
are inference inputs; current/future sensor temperatures and top-surface
labels are excluded from those inputs. Thus the task is conditional
reconstruction, and the test sensor error is not an independent assessment
of load-only forecasting. Physics evaluates coordinate derivatives holding
history fixed, as in the ASL source.

All four runs retain final-epoch and validation-best checkpoints. Primary
comparison reports epoch 1000; validation-best results are supplementary.
No early stopping. Save training histories, predictions, parameter counts,
runtime, RMSE, MAE, relative L2 (explicit Kelvin and temperature-rise
denominators), and R2. Test labels are opened only after all runs finish.

Generate 300-DPI PNGs with training/validation curves, error histograms,
best/90th-percentile/worst fields, shared-case field/error panels, radial
and temporal profiles, metric/runtime comparisons. Clearly label simulated
interior reference fields and experimental surface measurements. Do not
claim variable-load validation or measured interior ground truth.
