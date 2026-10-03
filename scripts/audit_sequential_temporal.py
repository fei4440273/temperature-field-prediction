"""Trace history masks and compare unfiltered sensor responses with the V5 run."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from joint_temperature_core import Geometry, Table, fixed_splits
from sequential_deeponet_core import METHODS, load_checkpoint
from sequential_deeponet_data import HistoryProvider, experiment_history, metrics
from train_sequential_deeponet import ROOT, digest, table_predict, write_json


def curve_statistics(times, reference, prediction, late_start=75.):
    times = np.asarray(times, dtype=float)
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    order = np.argsort(times)
    times, reference, prediction = times[order], reference[order], prediction[order]
    if len(times) < 2 or (np.diff(times) <= 0).any():
        raise ValueError("Sensor curves need distinct, increasing timestamps.")
    steps = np.diff(prediction)
    selected = np.flatnonzero(times[:-1] >= late_start)
    result = metrics(reference, prediction)
    result["negative_steps_over_0_01k"] = int((steps < -.01).sum())
    if len(selected):
        index = selected[np.argmax(np.abs(steps[selected]))]
        result.update(late_max_abs_step_k=float(abs(steps[index])),
            late_peak_step_k=float(steps[index]), late_peak_from_s=float(times[index]),
            late_peak_to_s=float(times[index+1]),
            late_max_abs_rate_k_per_s=float(np.max(np.abs(steps[selected]/np.diff(times)[selected]))),
            late_total_downward_variation_k=float(-np.minimum(steps[selected], 0.).sum()))
    return result


def audit_masks(cfg, baseline):
    spec = importlib.util.spec_from_file_location("v5_history", baseline/"source_snapshot/sequential_deeponet_data.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    geometry, splits = Geometry.from_project(ROOT), fixed_splits(ROOT)
    rows = []
    for split in ("train", "validation", "test"):
        tables, provider = experiment_history(ROOT, split, splits, geometry, cfg["model"])
        previous = original.HistoryProvider(provider.curves, geometry, cfg["model"])
        for power, (times, values) in provider.curves.items():
            observed = np.unique(tables["热端"].x[np.isclose(tables["热端"].x[:,3], power),2])
            knots = np.arange(2., times[-1])
            query = np.unique(np.r_[observed, np.arange(0., times[-1]+.01, .1),
                                   knots-.001, knots+.001])
            single = HistoryProvider({power: (times, values)}, geometry, cfg["model"])
            actual, expected = provider.build(np.full(len(query), power), query), single.build(np.full(len(query), power), query)
            if any(not np.array_equal(a, b) for a, b in zip(actual, expected)):
                raise AssertionError(f"Cross-power history influence: {split}, {power}")
            old = previous.build(np.full(len(query), power), query)
            rows.append(dict(split=split, power_w=power, observation_end_s=float(times[-1]),
                queries=len(query), exact_power_neighbor_invariance=True,
                original_invalid_local_tokens=int((old[1][...,4] == 0).sum()),
                corrected_invalid_local_tokens=int((actual[1][...,4] == 0).sum())))
    return rows


def dense_statistics(times, prediction):
    order = np.argsort(times)
    times, prediction = times[order], prediction[order]
    steps = np.diff(prediction)
    eligible = np.flatnonzero(times[:-1] >= 75.)
    peak = eligible[np.argmax(np.abs(steps[eligible]))]
    full_peak = int(np.argmax(np.abs(steps)))
    return dict(late_max_abs_step_k=float(abs(steps[peak])),
        late_peak_from_s=float(times[peak]), late_peak_to_s=float(times[peak+1]),
        max_abs_step_k=float(abs(steps[full_peak])),
        peak_from_s=float(times[full_peak]),peak_to_s=float(times[full_peak+1]),
        late_max_abs_rate_k_per_s=float(np.max(np.abs(steps[eligible]/np.diff(times)[eligible]))))


def audit_dense_predictions(output, cfg):
    tables, provider = experiment_history(ROOT, "test", fixed_splits(ROOT),
                                           Geometry.from_project(ROOT), cfg["model"])
    models = {m:load_checkpoint(output/m/"epoch_1000.pt", "cpu")[0] for m in METHODS}
    rows, files = [], {}
    for sensor, name in (("hot", "热端"), ("cold", "冷端")):
        observed = tables[name]
        for power in np.unique(observed.x[:,3]):
            sample = observed.x[np.isclose(observed.x[:,3], power)]
            end = sample[:,2].max()
            grid = np.arange(0., end+.01, .1, dtype=np.float32)
            knots = np.arange(2., end, dtype=np.float32)
            epsilon_times = np.stack((knots-.001, knots, knots+.001), axis=1).reshape(-1)
            query = np.r_[grid, epsilon_times]
            x = np.repeat(sample[:1], len(query), axis=0)
            x[:,2] = query
            table = Table(x, np.zeros(len(x)), np.ones(len(x)), np.zeros(len(x))).validate()
            pack = dict(x=x,grid_count=np.array(len(grid)),epsilon_s=np.array(.001))
            for method, model in models.items():
                prediction = table_predict(model, table, provider, "high")
                pack[method] = prediction
                dense = dense_statistics(query[:len(grid)], prediction[:len(grid)])
                epsilon = prediction[len(grid):].reshape(-1,3)
                outgoing = np.abs(epsilon[:,2]-epsilon[:,1])
                incoming = np.abs(epsilon[:,1]-epsilon[:,0])
                late = knots >= 75.
                rows.append(dict(sensor=sensor,power_w=float(power),method=method,
                    query_count=len(query),time_step_s=.1,epsilon_s=.001,**dense,
                    outgoing_integer_max_jump_k=float(outgoing.max()),
                    incoming_integer_max_jump_k=float(incoming.max()),
                    incoming_peak_time_s=float(knots[np.argmax(incoming)]),
                    outgoing_peak_time_s=float(knots[np.argmax(outgoing)]),
                    late_outgoing_integer_max_jump_k=float(outgoing[late].max()),
                    late_incoming_integer_max_jump_k=float(incoming[late].max())))
            path = output/"evaluation"/f"dense_{sensor}_{power:g}W_predictions.npz"
            np.savez_compressed(path, **pack)
            files[path.name] = digest(path)
    return rows, files


def audit(output, baseline, *, counterfactual=False):
    cfg = json.loads((output/"config.json").read_text(encoding="utf-8"))
    rows = []
    providers = None
    if counterfactual:
        _, providers = experiment_history(ROOT, "test", fixed_splits(ROOT), Geometry(), cfg["model"])
    for tag in ("hot", "cold"):
        previous = np.load(baseline/"evaluation"/f"{tag}_predictions.npz")
        if counterfactual:
            x, y = previous["x"], previous["y"]
            table = Table(x, y, np.ones(len(y)), np.zeros(len(y))).validate()
            current = {}
            for method in METHODS:
                model, _ = load_checkpoint(baseline/method/"epoch_1000.pt", "cpu")
                current[method] = table_predict(model, table, providers, "high")
        else:
            current = np.load(output/"evaluation"/f"{tag}_predictions.npz")
            x, y = current["x"], current["y"]
            np.testing.assert_array_equal(x, previous["x"])
            np.testing.assert_array_equal(y, previous["y"])
        for power in np.unique(x[:,3]):
            selected = np.flatnonzero(np.isclose(x[:,3], power))
            for method in METHODS:
                rows.append(dict(sensor=tag, power_w=float(power), method=method,
                    original=curve_statistics(x[selected,2], y[selected], previous[method][selected]),
                    corrected=curve_statistics(x[selected,2], y[selected], current[method][selected])))
    result = dict(mode="old_weights_corrected_input_only" if counterfactual else "four_retrained_models",
        baseline=str(baseline.relative_to(ROOT)) if baseline.is_relative_to(ROOT) else str(baseline),
        output=str(output), late_start_s=75.,
        no_prediction_smoothing=True, history_masks=audit_masks(cfg, baseline), curves=rows,
        baseline_checkpoint_sha256={m:digest(baseline/m/"epoch_1000.pt") for m in METHODS},
        audit_script_sha256=digest(Path(__file__)))
    if not counterfactual:
        result["corrected_checkpoint_sha256"] = {m:digest(output/m/"epoch_1000.pt") for m in METHODS}
        result["dense_curves"], result["dense_prediction_sha256"] = audit_dense_predictions(output, cfg)
        result["history_mask_semantics"] = "known recording-window coverage"
        result["endpoint_reconstruction"] = "last-two-past-observation linear trend, three-point warmup, one-cadence cap, within recording coverage"
    path = output/("temporal_counterfactual.json" if counterfactual else "temporal_audit.json")
    write_json(path, result)
    for row in rows:
        before, after = row["original"]["late_max_abs_step_k"], row["corrected"]["late_max_abs_step_k"]
        print(f"{row['sensor']} {row['power_w']:g} W {row['method']}: max late step {before:.4f} -> {after:.4f} K")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"研究记录/Sequential_DeepONet_1000epochs_history_fix_v3")
    parser.add_argument("--baseline", type=Path, default=ROOT/"研究记录/Sequential_DeepONet_1000epochs")
    parser.add_argument("--counterfactual", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit(args.output.resolve(), args.baseline.resolve(), counterfactual=args.counterfactual)


if __name__ == "__main__":
    main()
