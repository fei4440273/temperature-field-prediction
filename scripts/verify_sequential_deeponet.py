"""Independently audit completed training and the delivered comparison artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from joint_temperature_core import Geometry,Table,fixed_splits
from sequential_deeponet_core import METHODS,load_checkpoint
from sequential_deeponet_data import experiment_history,metrics
from train_sequential_deeponet import ROOT,digest,table_predict,write_json


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
    temporal_rng_state = None
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
        if cfg["loss_weights"].get("temporal",0)>0:
            if not all("temporal" in row and np.isfinite(row["temporal"]) and row["temporal"]>=0 for row in history):
                raise AssertionError("Shared temporal regularization was not trained in every update.")
            current_rng = state["extra"]["temporal_rng"]
            if temporal_rng_state is None:
                temporal_rng_state = current_rng
            elif temporal_rng_state!=current_rng:
                raise AssertionError("Temporal sampling RNG differs across methods.")
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
            if name is not None:
                for power in np.unique(x[:,3]):
                    mask = np.isclose(x[:,3],power)
                    group = metrics(y[mask],pack[method][mask])
                    for key,value in group.items():
                        close(value,result[method]["high_test_by_power"][f"{power:g}"][name][key],
                              f"{method}/{tag}/{power}/{key}")
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
    expected_names = {"Figure_1_training_curves.png","Figure_2_test_accuracy.png",
        "Figure_3_test_errors_by_power.png","appendix/Figure_3_relative_error_histograms.png",
        "appendix/Figure_5_common_full_field_and_errors.png",
        "Figure_7_top_radial_profiles.png","Figure_8_temporal_response.png","Figure_9_actual_vs_predicted.png"}
    expected_names.update(f"appendix/Figure_4_{method}_percentile_fields.png" for method in METHODS)
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
    maxima = json.loads((output/"evaluation/cloud_maxima.json").read_text(encoding="utf-8"))
    if {(row["time_s"],row["method"]) for row in maxima}!={(t,m) for t in (5,30,60,90,120) for m in METHODS} or len(maxima)!=20:
        raise AssertionError("Missing or duplicate experimental cloud maxima.")
    for row in maxima:
        path = output/"evaluation"/f"surface_634W_{row['time_s']}s.npz"
        if digest(path)!=row["source_npz_sha256"]:
            raise AssertionError("Annotated cloud values came from different predictions.")
        pack = np.load(path)
        error = np.abs(pack[row["method"]]-pack["y"])
        index = np.argmax(error)
        close(error[index],row["max_abs_error_k"],"Cloud maximum")
        close(np.sqrt(np.mean(error**2)),row["rmse_k"],"Cloud RMSE")
        for coordinate,key in zip(pack["xy_mm"][index],("x_mm","y_mm")):
            close(coordinate,row[key],"Cloud maximum position")
    temporal = json.loads((output/"temporal_audit.json").read_text(encoding="utf-8"))
    if (temporal["mode"]!="four_retrained_models" or not temporal["no_prediction_smoothing"]
            or len(temporal["curves"])!=24
            or digest(ROOT/"scripts/audit_sequential_temporal.py")!=temporal["audit_script_sha256"]):
        raise AssertionError("Temporal comparison is incomplete or uses changed audit source.")
    for method in METHODS:
        if temporal["corrected_checkpoint_sha256"][method]!=checks[method]["checkpoint_sha256"]:
            raise AssertionError("Temporal audit used different corrected weights.")
    baseline_directory = Path(temporal["baseline"])
    if not baseline_directory.is_absolute():
        baseline_directory = ROOT/baseline_directory
    for row in temporal["curves"]:
        for version,directory in (("original",baseline_directory),("corrected",output)):
            pack = np.load(directory/"evaluation"/f"{row['sensor']}_predictions.npz")
            ids = np.flatnonzero(np.isclose(pack["x"][:,3],row["power_w"]))
            ids = ids[np.argsort(pack["x"][ids,2])]
            times = pack["x"][ids,2]
            steps = np.diff(pack[row["method"]][ids])
            eligible = np.flatnonzero(times[:-1]>=temporal["late_start_s"])
            peak = eligible[np.argmax(np.abs(steps[eligible]))]
            for value,key in ((abs(steps[peak]),"late_max_abs_step_k"),
                              (times[peak],"late_peak_from_s"),(times[peak+1],"late_peak_to_s")):
                close(value,row[version][key],"Temporal jump measurement")
    if not all(row["exact_power_neighbor_invariance"] and row["corrected_invalid_local_tokens"]==0
               for row in temporal["history_masks"]):
        raise AssertionError("History masks still change within actual measured time ranges.")
    expected_dense = {(s,p,m) for s in ("hot","cold") for p in (169.,339.,634.) for m in METHODS}
    if {(r["sensor"],r["power_w"],r["method"]) for r in temporal["dense_curves"]}!=expected_dense or len(temporal["dense_curves"])!=24:
        raise AssertionError("Dense-time audit is incomplete.")
    dense_tables,provider = experiment_history(ROOT,"test",fixed_splits(ROOT),Geometry.from_project(ROOT),cfg["model"])
    models = {m:load_checkpoint(output/m/"epoch_1000.pt","cpu")[0] for m in METHODS}
    for sensor in ("hot","cold"):
        for power in (169.,339.,634.):
            name = f"dense_{sensor}_{power:g}W_predictions.npz"
            path = output/"evaluation"/name
            if digest(path)!=temporal["dense_prediction_sha256"][name]:
                raise AssertionError("Dense-time prediction source changed.")
            pack = np.load(path)
            x,n = pack["x"],int(pack["grid_count"])
            observed = dense_tables["热端" if sensor=="hot" else "冷端"].x
            sample = observed[np.isclose(observed[:,3],power)]
            horizon = sample[:,2].max()
            grid = np.arange(0.,horizon+.01,.1,dtype=np.float32)
            knots = np.arange(2.,horizon,dtype=np.float32)
            epsilon_times = np.stack((knots-.001,knots,knots+.001),axis=1).reshape(-1)
            if n!=len(grid):
                raise AssertionError("Dense grid does not cover the full measured sensor range.")
            np.testing.assert_array_equal(x[:,2],np.r_[grid,epsilon_times])
            np.testing.assert_array_equal(x[:,[0,1,3,4]],np.repeat(sample[:1,[0,1,3,4]],len(x),axis=0))
            times = x[:n,2]
            np.testing.assert_allclose(np.diff(times),.1,rtol=0,atol=1e-5)
            table = Table(x,np.zeros(len(x)),np.ones(len(x)),np.zeros(len(x))).validate()
            for method,model in models.items():
                actual = table_predict(model,table,provider,"high")
                np.testing.assert_allclose(actual,pack[method],rtol=0,atol=2e-4)
                row = next(r for r in temporal["dense_curves"] if (r["sensor"],r["power_w"],r["method"])==(sensor,power,method))
                steps = np.diff(pack[method][:n])
                late = times[:-1]>=75.
                close(float(np.abs(steps[late]).max()),row["late_max_abs_step_k"],"Dense-time maximum step")
                close(float(np.abs(steps).max()),row["max_abs_step_k"],"All-time dense maximum step")
                peak = int(np.argmax(np.abs(steps)))
                close(float(times[peak]),row["peak_from_s"],"All-time dense peak start")
                close(float(times[peak+1]),row["peak_to_s"],"All-time dense peak end")
                epsilon = pack[method][n:].reshape(-1,3)
                knots = x[n:,2].reshape(-1,3)[:,1]
                for delta,tag in ((epsilon[:,2]-epsilon[:,1],"outgoing"),(epsilon[:,1]-epsilon[:,0],"incoming")):
                    close(float(np.abs(delta).max()),row[f"{tag}_integer_max_jump_k"],"Integer boundary jump")
                    close(float(knots[np.argmax(np.abs(delta))]),row[f"{tag}_peak_time_s"],"Integer peak time")
                    close(float(np.abs(delta[knots>=75.]).max()),row[f"late_{tag}_integer_max_jump_k"],"Late integer boundary jump")
    if cfg["model"].get("asl_rate_mode")=="thermal_trend":
        adaptation = temporal["asl_adaptation"]
        previous = Path(adaptation["previous_result_directory"])
        if not previous.is_absolute():
            previous = ROOT/previous
        if (digest(previous/"asl/epoch_1000.pt")!=adaptation["previous_checkpoint_sha256"]
                or digest(previous/"evaluation/metrics.json")!=adaptation["previous_metrics_sha256"]):
            raise AssertionError("Previous adaptation baseline changed.")
        previous_state = torch.load(previous/"asl/epoch_1000.pt",map_location="cpu",weights_only=False)
        current_state = torch.load(output/"asl/epoch_1000.pt",map_location="cpu",weights_only=False)
        if previous_state["model_config"].get("asl_rate_mode","original")!="original":
            raise AssertionError("Adaptation baseline already uses thermal rate processing.")
        if (previous_state["model_state"].keys()!=current_state["model_state"].keys()
                or any(v.shape!=current_state["model_state"][k].shape for k,v in previous_state["model_state"].items())):
            raise AssertionError("ASL learned architecture changed during input adaptation.")
        previous_model,_ = load_checkpoint(previous/"asl/epoch_1000.pt","cpu")
        _,previous_provider = experiment_history(ROOT,"test",fixed_splits(ROOT),
            Geometry.from_project(ROOT),previous_state["model_config"])
        for tag,name in (("top","顶部"),("hot","热端"),("cold","冷端")):
            for version,directory in (("previous",previous),("adapted",output)):
                pack = np.load(directory/"evaluation"/f"{tag}_predictions.npz")
                actual = metrics(pack["y"],pack["asl"])
                for key,value in actual.items():
                    close(value,adaptation[version]["high_test"][name][key],"Adaptation comparison metric")
        for row in adaptation["curves"]:
            name = f"dense_{row['sensor']}_{row['power_w']:g}W_predictions.npz"
            if digest(previous/"evaluation"/name)!=adaptation["previous_dense_prediction_sha256"][name]:
                raise AssertionError("Previous adaptation dense predictions changed.")
            for version,directory in (("previous",previous),("adapted",output)):
                pack = np.load(directory/"evaluation"/name)
                n = int(pack["grid_count"])
                if version=="previous":
                    table = Table(pack["x"],np.zeros(len(pack["x"])),
                        np.ones(len(pack["x"])),np.zeros(len(pack["x"]))).validate()
                    actual = table_predict(previous_model,table,previous_provider,"high")
                    np.testing.assert_allclose(actual,pack["asl"],rtol=0,atol=2e-4)
                steps = np.diff(pack["asl"][:n])
                times = pack["x"][:n,2]
                epsilon = pack["asl"][n:].reshape(-1,3)
                values = dict(max_abs_step_k=float(np.abs(steps).max()),
                    late_max_abs_step_k=float(np.abs(steps[times[:-1]>=75.]).max()),
                    incoming_integer_max_jump_k=float(np.abs(epsilon[:,1]-epsilon[:,0]).max()),
                    outgoing_integer_max_jump_k=float(np.abs(epsilon[:,2]-epsilon[:,1]).max()))
                for key,value in values.items():
                    close(value,row[version][key],"Adaptation raw-curve comparison")
    audit_result = dict(status="passed",methods=checks,figures=images,
        training_source_and_input_hashes_verified=True,all_prediction_metrics_recomputed=True,
        no_boundary_attention=True,report=str(report),experimental_cloud_maxima_verified=True,
        original_and_corrected_temporal_jumps_verified=True,
        dense_predictions_recomputed_from_all_four_models=True,
        shared_temporal_loss_verified=temporal_rng_state is not None,
        asl_thermal_adaptation_verified=cfg["model"].get("asl_rate_mode")=="thermal_trend",
        previous_asl_dense_predictions_recomputed=cfg["model"].get("asl_rate_mode")=="thermal_trend",
        note="Source/metric/image checks supplement manual visual inspection and the 31 focused tests.")
    write_json(output/"verification.json",audit_result)
    print("Verified four actual 1000-update runs, exact shared initialization, source/data hashes, "
          "recomputed test metrics, 17 nonblank 300-DPI PNGs/PDFs, cloud maxima and raw dense-time predictions.")
    return audit_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"研究记录/Sequential_DeepONet_1000epochs_ASL_thermal")
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit(args.output.resolve())


if __name__=="__main__":
    main()
