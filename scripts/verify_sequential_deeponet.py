"""Independently audit completed training and the delivered comparison artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from sequential_deeponet_core import METHODS
from sequential_deeponet_data import metrics
from train_sequential_deeponet import ROOT,digest,write_json


def close(actual,expected,label):
    if actual is None or expected is None:
        if actual!=expected:
            raise AssertionError(label)
    elif not np.isclose(actual,expected,rtol=1e-9,atol=1e-9):
        raise AssertionError(f"{label}: {actual} != {expected}")


def audit(output):
    cfg = json.loads((output/"config.json").read_text(encoding="utf-8"))
    provenance = json.loads((output/"provenance.json").read_text(encoding="utf-8"))
    evaluation = json.loads((output/"evaluation/provenance.json").read_text(encoding="utf-8"))
    result = json.loads((output/"evaluation/metrics.json").read_text(encoding="utf-8"))
    checks = {}
    for name,expected in provenance["source_hashes"].items():
        snapshot = output/"source_snapshot"/name
        current = ROOT/("configs" if name.endswith(".yaml") else "scripts")/name
        if digest(snapshot)!=expected or digest(current)!=expected:
            raise AssertionError(f"Source changed after training started: {name}")
    for name,expected in provenance["data_hashes"].items():
        if digest(ROOT/name)!=expected:
            raise AssertionError(f"Input data changed after training: {name}")
    initial_shared = None
    for method in METHODS:
        path = output/method/"epoch_1000.pt"
        state = torch.load(path,map_location="cpu",weights_only=False)
        initial = torch.load(output/method/"initial.pt",map_location="cpu",weights_only=False)
        best = torch.load(output/method/"best.pt",map_location="cpu",weights_only=False)
        if (state["method"]!=method or state["seed"]!=cfg["experiment"]["seed"]
                or state["config"]!=cfg or state["epoch"]!=1000 or initial["epoch"]!=0
                or state["model_config"].get("boundary_attention") is not False):
            raise AssertionError(f"Checkpoint identity/budget mismatch for {method}")
        history = state["history"]
        if [row["epoch"] for row in history]!=list(range(1,1001)):
            raise AssertionError(f"Missing or duplicate training epochs for {method}")
        if not all(np.isfinite(row["loss"]) and row["loss"]>=0 for row in history):
            raise AssertionError("Invalid training losses.")
        steps = [int(float(value["step"])) for value in state["optimizer"]["state"].values()]
        if not steps or set(steps)!={1000}:
            raise AssertionError(f"Actual optimizer steps differ from 1000: {method}, {set(steps)}")
        if not any(not torch.equal(value,initial["model_state"][name])
                   for name,value in state["model_state"].items()):
            raise AssertionError(f"Weights did not change: {method}")
        shared = {name:value for name,value in initial["model_state"].items() if not name.startswith("branch.")}
        if initial_shared is None:
            initial_shared = shared
        elif any(not torch.equal(initial_shared[name],value) for name,value in shared.items()):
            raise AssertionError("Common operator initialization differs across methods.")
        score = min(row["validation_score_k"] for row in history if "validation_score_k" in row)
        close(best["extra"]["validation_score_k"],score,"Best checkpoint selection")
        if digest(path)!=evaluation["checkpoint_sha256"][method]:
            raise AssertionError("Evaluated checkpoint hash changed.")
        with (output/method/"history.csv").open(encoding="utf-8") as handle:
            csv_history = list(csv.DictReader(handle))
        if [int(row["epoch"]) for row in csv_history]!=list(range(1,1001)):
            raise AssertionError("CSV history is incomplete.")
        for left,right in zip(history,csv_history):
            close(left["loss"],float(right["loss"]),"Saved training loss")
        checks[method] = dict(epoch=1000,optimizer_steps=1000,history_rows=1000,
                              best_epoch=best["epoch"],checkpoint_sha256=digest(path))
    for tag,name in (("simulation",None),("top","顶部"),("hot","热端"),("cold","冷端")):
        pack = np.load(output/"evaluation"/f"{tag}_predictions.npz")
        y,x = pack["y"],pack["x"]
        if len(y)!=len(x) or not np.isfinite(x).all():
            raise AssertionError("Prediction coordinate data is invalid.")
        for method in METHODS:
            actual = metrics(y,pack[method])
            expected = result[method]["low_test"] if name is None else result[method]["high_test"][name]
            for key,value in actual.items():
                close(value,expected[key],f"{method}/{tag}/{key}")
            if tag=="simulation":
                for region,material in (("copper",0),("sic",1)):
                    mask = x[:,4]==material
                    actual_region = metrics(y[mask],pack[method][mask])
                    for key,value in actual_region.items():
                        close(value,result[method]["low_test_by_material"][region][key],f"{method}/{region}/{key}")
    for method in METHODS:
        high = result[method]["high_test"]
        score = np.sqrt(.5*high["顶部"]["rmse_k"]**2+.25*high["热端"]["rmse_k"]**2+.25*high["冷端"]["rmse_k"]**2)
        close(score,result[method]["combined_test_rmse_k"],"Combined test score")
    manifest = json.loads((output/"figures/manifest.json").read_text(encoding="utf-8"))
    expected_names = {"Figure_1_training_curves.png","Figure_2_accuracy_and_cost.png",
        "Figure_3_relative_error_histograms.png","Figure_5_common_full_field_and_errors.png",
        "Figure_7_top_radial_profiles.png","Figure_8_temporal_response.png","Figure_9_actual_vs_predicted.png"}
    expected_names.update(f"Figure_4_{method}_percentile_fields.png" for method in METHODS)
    expected_names.update(f"Figure_6_surface_634W_{t}s.png" for t in (5,30,60,90,120))
    if set(manifest["figures"])!=expected_names:
        raise AssertionError("Required figure types/cases are missing.")
    if digest(ROOT/"scripts/plot_sequential_deeponet.py")!=manifest["plot_script_sha256"]:
        raise AssertionError("Plot source changed after figure export.")
    images = []
    for name in sorted(expected_names):
        path = output/"figures"/name
        if digest(path)!=manifest["png_sha256"][name] or not path.with_suffix(".pdf").is_file():
            raise AssertionError(f"Figure provenance or PDF missing: {name}")
        with Image.open(path) as im:
            im.load()
            dpi = im.info.get("dpi",(0.,0.))
            size = im.size
            if im.format!="PNG" or min(dpi)<299 or min(size)<800:
                raise AssertionError(f"PNG format/resolution invalid: {name}")
            preview = im.convert("RGB")
            preview.thumbnail((128,128))
            if np.asarray(preview,dtype=float).std()<10:
                raise AssertionError(f"Blank figure: {name}")
        images.append(dict(name=name,width_px=size[0],height_px=size[1],dpi=dpi))
    report = output/"对比总结.md"
    if not report.is_file() or report.stat().st_size<2000:
        raise AssertionError("Comparison report is missing.")
    audit_result = dict(status="passed",methods=checks,figures=images,
        training_source_and_input_hashes_verified=True,all_prediction_metrics_recomputed=True,
        no_boundary_attention=True,report=str(report),
        note="Source/metric/image checks supplement manual visual inspection and the 19 focused tests.")
    write_json(output/"verification.json",audit_result)
    print("Verified four actual 1000-update runs, exact shared initialization, source/data hashes, "
          "recomputed test metrics, 16 nonblank 300-DPI PNGs/PDFs and the Chinese report.")
    return audit_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"研究记录/Sequential_DeepONet_1000epochs")
    args = parser.parse_args()
    audit(args.output.resolve())


if __name__=="__main__":
    main()
