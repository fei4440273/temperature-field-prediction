"""Measure signed experimental errors on each power's observed time support."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from joint_temperature_core import Geometry,fixed_splits
from sequential_deeponet_core import METHODS,load_checkpoint
from sequential_deeponet_data import experiment_history,metrics
from train_sequential_deeponet import ROOT,digest,table_predict,validate,write_json


def signed_metrics(y,prediction):
    result = metrics(y,prediction)
    result["mean_bias_k"] = float(np.mean(np.asarray(prediction).reshape(-1)-np.asarray(y).reshape(-1)))
    return result


def curve_drift(times,errors):
    times = np.asarray(times,dtype=np.float64)
    errors = np.asarray(errors,dtype=np.float64)
    dt = times-times.mean()
    variance = float(np.dot(dt,dt))
    return float(np.dot(dt,errors-errors.mean())/variance) if variance>0 else None


def bias_statistics(table,prediction,*,top=False,tail_fraction=.7):
    if not 0<tail_fraction<1:
        raise ValueError("Tail fraction must be strictly between zero and one.")
    prediction = np.asarray(prediction).reshape(-1)
    y = table.y.reshape(-1)
    result = dict(full=signed_metrics(y,prediction),by_power={})
    all_tail,all_center,all_center_tail = [],[],[]
    for power in np.unique(table.x[:,3]):
        ids = np.flatnonzero(table.x[:,3]==power)
        times = table.x[ids,2]
        end = float(times.max())
        tail = ids[times>=tail_fraction*end]
        all_tail.extend(tail)
        row = dict(end_time_s=end,tail_start_s=float(table.x[tail,2].min()),
                   full=signed_metrics(y[ids],prediction[ids]),
                   tail=signed_metrics(y[tail],prediction[tail]))
        if top:
            center = np.array([frame[np.argmin(table.x[frame,0])]
                for time in np.unique(times)
                for frame in [ids[times==time]]],dtype=np.int64)
            center_tail = center[table.x[center,2]>=tail_fraction*end]
            all_center.extend(center)
            all_center_tail.extend(center_tail)
            row["center"] = signed_metrics(y[center],prediction[center])
            row["center_tail"] = signed_metrics(y[center_tail],prediction[center_tail])
            row["center_tail"]["error_slope_k_per_s"] = curve_drift(
                table.x[center_tail,2],prediction[center_tail]-y[center_tail])
            row["center_final_bias_k"] = float(prediction[center[-1]]-y[center[-1]])
            row["center_observations"] = dict(time_s=table.x[center,2].tolist(),
                radius_mm=(table.x[center,0]*1000).tolist(),observed_k=y[center].tolist(),
                predicted_k=prediction[center].tolist())
            r = table.x[tail,0]
            row["radial_tail"] = {name:signed_metrics(y[tail[mask]],prediction[tail[mask]])
                for name,mask in (("0_to_5_mm",r<=.005),("5_to_15_mm",(r>.005)&(r<=.015)),
                                  ("15_to_25_mm",r>.015)) if mask.any()}
        else:
            row["tail"]["error_slope_k_per_s"] = curve_drift(table.x[tail,2],prediction[tail]-y[tail])
            last = ids[np.argmax(times)]
            row["final_bias_k"] = float(prediction[last]-y[last])
        result["by_power"][f"{power:g}"] = row
    result["tail"] = signed_metrics(y[all_tail],prediction[all_tail])
    if top:
        result["center"] = signed_metrics(y[all_center],prediction[all_center])
        result["center_tail"] = signed_metrics(y[all_center_tail],prediction[all_center_tail])
    return result


def analyze(output,baseline,methods,splits_requested,device,destination):
    cfg = json.loads((output/"config.json").read_text(encoding="utf-8"))
    geometry = Geometry.from_project(ROOT)
    splits = fixed_splits(ROOT,"legacy_split")
    roots = {"current":output}
    if baseline:
        roots["baseline"] = baseline
    selection = cfg.get("validation_selection")
    results = dict(tail_fraction=.7,selection_policy="training/validation only; test set previously inspected",
                   script_sha256=digest(Path(__file__)),runs={})
    for label,root in roots.items():
        run_cfg = json.loads((root/"config.json").read_text(encoding="utf-8"))
        block = dict(result_directory=str(root.relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root),
                     config_sha256=digest(root/"config.json"),checkpoint_sha256={},splits={})
        for split in splits_requested:
            tables,provider = experiment_history(ROOT,split,splits,geometry,run_cfg["model"])
            block["splits"][split] = {}
            for method in methods:
                checkpoint = root/method/"epoch_1000.pt"
                model,state = load_checkpoint(checkpoint,device)
                if state["epoch"]!=1000 or state["config"]!=run_cfg or state["method"]!=method:
                    raise ValueError("Signed-error analysis requires a matching actual epoch-1000 checkpoint.")
                block["checkpoint_sha256"][method] = digest(checkpoint)
                model.eval()
                predictions = {name:table_predict(model,table,provider,"high") for name,table in tables.items()}
                measured = {name:bias_statistics(table,predictions[name],top=name=="顶部")
                            for name,table in tables.items()}
                if split=="validation":
                    score,_ = validate(model,tables,provider,selection)
                    measured["selection_score_k"] = score
                block["splits"][split][method] = measured
                top = measured["顶部"]
                print(f"{label} {method} {split}: top={top['full']['rmse_k']:.4f} K, "
                      f"center-tail={top['center_tail']['rmse_k']:.4f} K",flush=True)
                del model
        results["runs"][label] = block
    write_json(destination/"temperature_bias.json",results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--baseline",type=Path)
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=["asl"])
    parser.add_argument("--splits",nargs="+",choices=("train","validation","test"),default=["train","validation"])
    parser.add_argument("--destination",type=Path)
    parser.add_argument("--selection-from",type=Path)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    destination = args.destination or args.output/"diagnostics"
    if args.selection_from:
        selected = json.loads(args.selection_from.read_text(encoding="utf-8"))
        if selected["selected_config_json_sha256"]!=digest(args.output/"config.json"):
            raise ValueError("Selected protocol metadata does not match this experiment configuration.")
        write_json(destination/"selection.json",selected)
        for tag in ("1","2"):
            source = args.selection_from.parent/f"candidate_{tag}_validation_bias.json"
            target = destination/source.name
            if source.resolve()!=target.resolve():
                shutil.copy2(source,target)
    cfg = json.loads((args.output/"config.json").read_text(encoding="utf-8"))
    if cfg["experiment"].get("data_revision") and args.baseline:
        parser.error("This data revision requires current-data diagnostics; omit --baseline.")
    baseline = args.baseline
    if not cfg["experiment"].get("data_revision") and baseline is None:
        baseline = ROOT/"研究记录/Sequential_DeepONet_1000epochs_ASL_thermal"
    analyze(args.output.resolve(),baseline.resolve() if baseline else None,args.methods,args.splits,args.device,
            destination)


if __name__=="__main__":
    main()
