"""Evaluate locked epoch-1000 operators and preserve prediction-level evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import polars as pl
import torch

from joint_temperature_core import Geometry,Table,fixed_splits,load_simulation
from sequential_deeponet_core import METHODS,LABELS,load_checkpoint
from sequential_deeponet_data import experiment_history,simulation_history,metrics
from train_sequential_deeponet import ROOT,write_json,digest,table_predict


def per_case_metrics(table,prediction,*,exclude_initial=False):
    pairs,inverse = np.unique(table.x[:,[3,2]],axis=0,return_inverse=True)
    rows = []
    for i,(power,time_s) in enumerate(pairs):
        if exclude_initial and time_s<=0:
            continue
        mask = inverse==i
        row = dict(power_w=float(power),time_s=float(time_s))
        row.update(metrics(table.y[mask],prediction[mask]))
        rows.append(row)
    return rows


def composite(high):
    return float(np.sqrt(.5*high["顶部"]["rmse_k"]**2+.25*high["热端"]["rmse_k"]**2
                         +.25*high["冷端"]["rmse_k"]**2))


def evaluate(output,device):
    cfg = json.loads((output/"config.json").read_text(encoding="utf-8"))
    geometry = Geometry.from_project(ROOT)
    splits = fixed_splits(ROOT,"legacy_split")
    checkpoint_paths = {}
    for method in METHODS:
        info = json.loads((output/method/"final_info.json").read_text(encoding="utf-8"))
        checkpoint = output/method/"epoch_1000.pt"
        model,state = load_checkpoint(checkpoint,"cpu")
        if (info["epochs_completed"]!=1000 or info["history_length"]!=1000 or state["epoch"]!=1000
                or state["method"]!=method or info["method"]!=method
                or state["seed"]!=cfg["experiment"]["seed"] or info["seed"]!=cfg["experiment"]["seed"]
                or [row["epoch"] for row in state["history"]]!=list(range(1,1001))
                or state["config"]!=cfg or state["model_config"]["boundary_attention"]):
            raise ValueError(f"Incomplete or inconsistent formal run: {method}")
        checkpoint_paths[method] = checkpoint
    print("Four 1000-epoch checkpoints verified; opening held-out labels for evaluation.",flush=True)
    high_tables,high_provider = experiment_history(ROOT,"test",splits,geometry,cfg["model"])
    low_provider,nodes = simulation_history(ROOT,splits["low"]["test"],geometry,cfg["model"])
    low_table = load_simulation(ROOT,splits["low"]["test"])
    pairs = np.unique(low_table.x[:,[3,2]],axis=0)
    low_provider.build(pairs[:,0],pairs[:,1])
    evaluation = output/"evaluation"
    evaluation.mkdir(exist_ok=True)
    write_json(evaluation/"simulation_sensor_nodes.json",nodes)
    low_predictions = {}
    high_predictions = {name:{} for name in high_tables}
    results = {}
    case_results = {}
    summary_rows = []
    for method in METHODS:
        model,state = load_checkpoint(checkpoint_paths[method],device)
        model.eval()
        info = json.loads((output/method/"final_info.json").read_text(encoding="utf-8"))
        warm_size = min(128,len(low_table.x))
        warm = Table(low_table.x[:warm_size],low_table.y[:warm_size],np.ones(warm_size),np.zeros(warm_size)).validate()
        table_predict(model,warm,low_provider,"low")
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        low = table_predict(model,low_table,low_provider,"low")
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter()-start
        high = {name:table_predict(model,table,high_provider,"high") for name,table in high_tables.items()}
        low_predictions[method] = low
        for name,pred in high.items():
            high_predictions[name][method] = pred
        high_metrics = {name:metrics(table.y,high[name]) for name,table in high_tables.items()}
        low_metrics = metrics(low_table.y,low)
        cases = dict(simulation=per_case_metrics(low_table,low,exclude_initial=True),
                     top=per_case_metrics(high_tables["顶部"],high["顶部"]))
        case_results[method] = cases
        best,best_state = load_checkpoint(output/method/"best.pt",device)
        if best_state["method"]!=method or best_state["seed"]!=cfg["experiment"]["seed"]:
            raise ValueError("Validation-best checkpoint identity does not match its method.")
        best_metrics = {name:metrics(table.y,table_predict(best,table,high_provider,"high"))
                        for name,table in high_tables.items()}
        results[method] = dict(final_epoch=1000,high_test=high_metrics,low_test=low_metrics,
            low_test_by_material={tag:metrics(low_table.y[low_table.x[:,4]==material],low[low_table.x[:,4]==material])
                                  for tag,material in (("copper",0),("sic",1))},
            combined_test_rmse_k=composite(high_metrics),parameters=info["parameters"],
            training_seconds=info["training_seconds"],inference_low_test_seconds=inference_seconds,
            inference_low_test_points=len(low_table.x),best_epoch=best_state["epoch"],
            best_high_test=best_metrics,best_combined_test_rmse_k=composite(best_metrics))
        for kind,rows in cases.items():
            errors = np.array([row["relative_l2_kelvin_pct"] for row in rows])
            results[method][kind+"_case_statistics"] = dict(cases=len(rows),
                mean_l2_kelvin_pct=float(errors.mean()),std_across_cases_l2_kelvin_pct=float(errors.std(ddof=1)),
                p90_l2_kelvin_pct=float(np.percentile(errors,90)),max_l2_kelvin_pct=float(errors.max()))
        row = dict(method=method,label=LABELS[method],epoch=1000,seed=info["seed"],parameters=info["parameters"],
            top_rmse_k=high_metrics["顶部"]["rmse_k"],hot_rmse_k=high_metrics["热端"]["rmse_k"],
            cold_rmse_k=high_metrics["冷端"]["rmse_k"],combined_rmse_k=composite(high_metrics),
            simulation_rmse_k=low_metrics["rmse_k"],simulation_r2=low_metrics["r2"],
            simulation_l2_kelvin_pct=low_metrics["relative_l2_kelvin_pct"],
            simulation_l2_rise_pct=low_metrics["relative_l2_rise_pct"],
            top_r2=high_metrics["顶部"]["r2"],top_l2_kelvin_pct=high_metrics["顶部"]["relative_l2_kelvin_pct"],
            top_l2_rise_pct=high_metrics["顶部"]["relative_l2_rise_pct"],
            training_seconds=info["training_seconds"],best_epoch=best_state["epoch"],
            best_combined_rmse_k=composite(best_metrics))
        summary_rows.append(row)
        print(f"{LABELS[method]} epoch 1000: top={row['top_rmse_k']:.4f} K, "
              f"combined={row['combined_rmse_k']:.4f} K, simulation={row['simulation_rmse_k']:.4f} K",flush=True)
        del model,best
    np.savez_compressed(evaluation/"simulation_predictions.npz",x=low_table.x,y=low_table.y.reshape(-1),**low_predictions)
    for name,table in high_tables.items():
        tag = {"顶部":"top","热端":"hot","冷端":"cold"}[name]
        np.savez_compressed(evaluation/f"{tag}_predictions.npz",x=table.x,y=table.y.reshape(-1),**high_predictions[name])
    # Retain the supplied two-dimensional IR-derived field, including angular
    # deviations from the radial means used during fitting.
    for time_s in (5,30,60,90,120):
        path = ROOT/f"data/test_Data/topdata/634W-{time_s}s_50mm_temperature.csv"
        raw = pl.read_csv(path,columns=["x_mm","y_mm","r_mm","temperature_c","is_recovered"])
        x = np.column_stack((raw["r_mm"].to_numpy()/1000,np.zeros(raw.height),np.full(raw.height,time_s),
                             np.full(raw.height,634.),np.ones(raw.height))).astype(np.float32)
        table = Table(x,raw["temperature_c"].to_numpy()+273.15,np.ones(raw.height),np.zeros(raw.height)).validate()
        pred = {}
        for method in METHODS:
            model,_ = load_checkpoint(checkpoint_paths[method],device)
            model.eval()
            pred[method] = table_predict(model,table,high_provider,"high")
        np.savez_compressed(evaluation/f"surface_634W_{time_s}s.npz",xy_mm=raw.select("x_mm","y_mm").to_numpy(),
                            x=table.x,y=table.y.reshape(-1),is_recovered=raw["is_recovered"].to_numpy(),**pred)
    write_json(evaluation/"metrics.json",results)
    write_json(evaluation/"case_metrics.json",case_results)
    with (evaluation/"summary.csv").open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    for method,cases in case_results.items():
        for tag,rows in cases.items():
            with (evaluation/f"{method}_{tag}_cases.csv").open("w",newline="",encoding="utf-8") as handle:
                writer = csv.DictWriter(handle,list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    write_json(evaluation/"provenance.json",dict(epoch=1000,seed=cfg["experiment"]["seed"],
        checkpoint_sha256={method:digest(path) for method,path in checkpoint_paths.items()},
        evaluation_script_sha256=digest(Path(__file__)),
        training_checkpoints_verified_before_test_labels=True,
        inference_timing_definition="one complete simulation-test prediction after common history-cache prefill and per-model warmup; same GPU, no repeated benchmark",
        input_history_policy="same held-out past sensor measurements for each method; current and future labels excluded",
        percentile_policy="nearest ranked field by per-case relative L2 in Kelvin; initial t=0 excluded",
        raw_ir_fields="supplied is_recovered flag preserved; not independent unprocessed radiometric measurements"))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"研究记录/Sequential_DeepONet_1000epochs")
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.output.resolve(),args.device)


if __name__=="__main__":
    main()
