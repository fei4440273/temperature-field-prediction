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


def verify_temperature_bias(output):
    diagnostic = json.loads((output/"diagnostics/temperature_bias.json").read_text(encoding="utf-8"))
    if digest(ROOT/"scripts/analyze_sequential_temperature_bias.py")!=diagnostic["script_sha256"]:
        raise AssertionError("Temperature diagnostic source changed.")
    states = {}
    groups = 0
    for version,run in diagnostic["runs"].items():
        directory = Path(run["result_directory"])
        if not directory.is_absolute():
            directory = ROOT/directory
        if digest(directory/"config.json")!=run["config_sha256"]:
            raise AssertionError("Temperature diagnostic configuration changed.")
        for method,expected_hash in run["checkpoint_sha256"].items():
            checkpoint = directory/method/"epoch_1000.pt"
            if digest(checkpoint)!=expected_hash:
                raise AssertionError("Temperature diagnostic checkpoint changed.")
            if method=="asl":
                states[version] = torch.load(checkpoint,map_location="cpu",weights_only=False)
            measured = run["splits"]["test"][method]
            for tag,name in (("top","顶部"),("hot","热端"),("cold","冷端")):
                pack = np.load(directory/"evaluation"/f"{tag}_predictions.npz")
                x,y,prediction = pack["x"],pack["y"],pack[method]
                def check_subset(ids,expected,label):
                    actual = metrics(y[ids],prediction[ids])
                    actual["mean_bias_k"] = float(np.mean(prediction[ids]-y[ids]))
                    for key,value in actual.items():
                        close(value,expected[key],label+"/"+key)
                all_tail,all_center,all_center_tail = [],[],[]
                check_subset(np.arange(len(y)),measured[name]["full"],"Measured whole-curve metrics")
                for power in np.unique(x[:,3]):
                    ids = np.flatnonzero(x[:,3]==power)
                    times = x[ids,2]
                    tail = ids[times>=diagnostic["tail_fraction"]*times.max()]
                    all_tail.extend(tail)
                    row = measured[name]["by_power"][f"{power:g}"]
                    close(float(times.max()),row["end_time_s"],"Measured power end time")
                    close(float(x[tail,2].min()),row["tail_start_s"],"Measured power tail support")
                    check_subset(ids,row["full"],"Measured per-power metrics")
                    expected = row["tail"]
                    check_subset(tail,expected,"Measured late metrics")
                    if tag=="top":
                        center = np.array([frame[np.argmin(x[frame,0])]
                            for t in np.unique(times) for frame in [ids[times==t]]])
                        ct = center[x[center,2]>=diagnostic["tail_fraction"]*times.max()]
                        all_center.extend(center)
                        all_center_tail.extend(ct)
                        check_subset(center,row["center"],"Measured center metrics")
                        expected = row["center_tail"]
                        check_subset(ct,expected,"Measured center-tail metrics")
                        close(float(prediction[center[-1]]-y[center[-1]]),row["center_final_bias_k"],"Measured final bias")
                        t = x[ct,2].astype(np.float64)
                        e = (prediction[ct]-y[ct]).astype(np.float64)
                        slope = float(np.sum((t-t.mean())*(e-e.mean()))/np.sum((t-t.mean())**2))
                        close(slope,expected["error_slope_k_per_s"],"Measured late error slope")
                        np.testing.assert_array_equal(row["center_observations"]["time_s"],x[center,2])
                        np.testing.assert_array_equal(row["center_observations"]["radius_mm"],x[center,0]*1000)
                        np.testing.assert_array_equal(row["center_observations"]["observed_k"],y[center])
                        np.testing.assert_array_equal(row["center_observations"]["predicted_k"],prediction[center])
                        r = x[tail,0]
                        regions = {label:tail[mask] for label,mask in
                            (("0_to_5_mm",r<=.005),("5_to_15_mm",(r>.005)&(r<=.015)),("15_to_25_mm",r>.015))
                            if mask.any()}
                        if regions.keys()!=row["radial_tail"].keys():
                            raise AssertionError("Measured radial diagnostic support changed.")
                        for label,radial_ids in regions.items():
                            check_subset(radial_ids,row["radial_tail"][label],"Measured radial-tail metrics")
                    else:
                        last = ids[np.argmax(times)]
                        close(float(prediction[last]-y[last]),row["final_bias_k"],"Measured sensor final bias")
                        t = x[tail,2].astype(np.float64)
                        e = (prediction[tail]-y[tail]).astype(np.float64)
                        slope = float(np.sum((t-t.mean())*(e-e.mean()))/np.sum((t-t.mean())**2))
                        close(slope,row["tail"]["error_slope_k_per_s"],"Measured sensor error slope")
                    groups += 1
                check_subset(all_tail,measured[name]["tail"],"Measured aggregate tail metrics")
                if tag=="top":
                    check_subset(all_center,measured[name]["center"],"Measured aggregate center metrics")
                    check_subset(all_center_tail,measured[name]["center_tail"],"Measured aggregate center-tail metrics")
    if states["current"]["config"]["experiment"].get("data_revision"):
        if set(diagnostic["runs"])!={"current"} or groups!=36:
            raise AssertionError("Revised measurements require all four current-data diagnostics only.")
        return dict(metric_groups=groups,current_measurements_only=True,
                    diagnostic_sha256=digest(output/"diagnostics/temperature_bias.json"))
    before,after = states["baseline"],states["current"]
    if (before["model_config"]!=after["model_config"] or before["model_state"].keys()!=after["model_state"].keys()
            or any(v.shape!=after["model_state"][k].shape for k,v in before["model_state"].items())):
        raise AssertionError("Temperature optimization changed ASL's learned architecture or inputs.")
    selection = json.loads((output/"diagnostics/selection.json").read_text(encoding="utf-8"))
    if (selection["selection_data"]!=["train","validation"] or
            selection["selected_config_json_sha256"]!=digest(output/"config.json")):
        raise AssertionError("Temperature selection did not lock the delivered configuration on train/validation.")
    for tag in ("1","2"):
        candidate = json.loads((output/f"diagnostics/candidate_{tag}_validation_bias.json").read_text(encoding="utf-8"))
        run = candidate["runs"]["current"]
        recorded = selection[f"candidate_{tag}"]
        if (set(run["splits"])!={"train","validation"} or
                run["config_sha256"]!=recorded["config_json_sha256"] or
                run["checkpoint_sha256"]["asl"]!=recorded["checkpoint_sha256"]):
            raise AssertionError("Candidate validation archive does not match selection metadata.")
        measured = run["splits"]["validation"]["asl"]
        for key,value in (("validation_score_k",measured["selection_score_k"]),
                          ("validation_top_rmse_k",measured["顶部"]["full"]["rmse_k"]),
                          ("validation_center_tail_rmse_k",measured["顶部"]["center_tail"]["rmse_k"]),
                          ("validation_hot_rmse_k",measured["热端"]["full"]["rmse_k"]),
                          ("validation_cold_rmse_k",measured["冷端"]["full"]["rmse_k"])):
            close(value,recorded[key],"Archived candidate selection metric")
    temporal = json.loads((output/"temporal_audit.json").read_text(encoding="utf-8"))
    comparison = temporal["temperature_optimization"]
    baseline = Path(diagnostic["runs"]["baseline"]["result_directory"])
    if not baseline.is_absolute():
        baseline = ROOT/baseline
    stated_baseline = Path(comparison["previous_result_directory"])
    if not stated_baseline.is_absolute():
        stated_baseline = ROOT/stated_baseline
    if (stated_baseline.resolve()!=baseline.resolve() or
            digest(baseline/"asl/epoch_1000.pt")!=comparison["previous_checkpoint_sha256"] or
            digest(baseline/"evaluation/metrics.json")!=comparison["previous_metrics_sha256"] or
            comparison["comparison_scope"]!="same learned architecture and inputs; measured central/tail objectives and full-top reweighting"):
        raise AssertionError("Temperature raw-curve baseline or comparison scope changed.")
    for version,label in (("previous","baseline"),("adapted","current")):
        measured = diagnostic["runs"][label]["splits"]["test"]["asl"]
        for name in ("顶部","热端","冷端"):
            for key,value in measured[name]["full"].items():
                if key!="mean_bias_k":
                    close(value,comparison[version]["high_test"][name][key],"Temperature baseline comparison metric")
    old_model,_ = load_checkpoint(baseline/"asl/epoch_1000.pt","cpu")
    _,provider = experiment_history(ROOT,"test",fixed_splits(ROOT),Geometry.from_project(ROOT),before["model_config"])
    expected_curves = {(s,p) for s in ("hot","cold") for p in (169.,339.,634.)}
    if {(r["sensor"],r["power_w"]) for r in comparison["curves"]}!=expected_curves or len(comparison["curves"])!=6:
        raise AssertionError("Temperature baseline comparison omitted raw sensor curves.")
    for row in comparison["curves"]:
        name = f"dense_{row['sensor']}_{row['power_w']:g}W_predictions.npz"
        if digest(baseline/"evaluation"/name)!=comparison["previous_dense_prediction_sha256"][name]:
            raise AssertionError("Temperature baseline dense predictions changed.")
        for version,directory in (("previous",baseline),("adapted",output)):
            pack = np.load(directory/"evaluation"/name)
            n = int(pack["grid_count"])
            if version=="previous":
                table = Table(pack["x"],np.zeros(len(pack["x"])),np.ones(len(pack["x"])),np.zeros(len(pack["x"]))).validate()
                np.testing.assert_allclose(table_predict(old_model,table,provider,"high"),pack["asl"],rtol=0,atol=2e-4)
            steps = np.diff(pack["asl"][:n])
            epsilon = pack["asl"][n:].reshape(-1,3)
            values = dict(max_abs_step_k=float(np.abs(steps).max()),
                late_max_abs_step_k=float(np.abs(steps[pack["x"][:n,2][:-1]>=75.]).max()),
                incoming_integer_max_jump_k=float(np.abs(epsilon[:,1]-epsilon[:,0]).max()),
                outgoing_integer_max_jump_k=float(np.abs(epsilon[:,2]-epsilon[:,1]).max()))
            for key,value in values.items():
                close(value,row[version][key],"Temperature raw-curve comparison")
    return dict(metric_groups=groups,unchanged_asl_architecture_and_input_config=True,
                thermal_baseline_dense_predictions_recomputed=True,validation_selection_config_verified=True,
                candidate_validation_archives_verified=True,
                diagnostic_sha256=digest(output/"diagnostics/temperature_bias.json"))


def verify_sensor_revision(output,cfg,provenance):
    import polars as pl
    from sic_cu.data.sensors import ring_average_raw
    revision = json.loads((output/"data_revision/data_revision.json").read_text(encoding="utf-8"))
    protocol = json.loads((output/"revision_protocol.json").read_text(encoding="utf-8"))
    if (protocol["mode"]!="user_requested_data_and_initial_temperature_revision"
            or protocol["data_revision"]!=cfg["experiment"]["data_revision"]
            or protocol["config_json_sha256"]!=digest(output/"config.json")
            or protocol["selection_data"]!=["train","validation"]
            or protocol["hyperparameter_search"] or protocol["test_labels_for_selection"]
            or protocol["experimental_copper_initial_c"]!=25.):
        raise AssertionError("The user-requested data/initial-temperature revision protocol changed.")
    source = ROOT/revision["source"]
    if (digest(source)!=revision["source_sha256"]
            or digest(output/"data_revision/corrected_colddata169W.csv")!=digest(source)
            or digest(ROOT/revision["processed_path"])!=revision["processed_after_sha256"]
            or digest(ROOT/"data/processed/manifest.json")!=revision["manifest_after_sha256"]
            or revision["raw_measurements_shifted"] or revision["split"]!="test"
            or revision["training_allowed"] or revision["model_selection_allowed"]):
        raise AssertionError("Corrected sensor data provenance changed.")
    for name,expected in revision["unchanged_processed_sha256"].items():
        if digest(ROOT/name)!=expected:
            raise AssertionError("Another processed input changed during the sensor refresh.")
    ring = ring_average_raw(source,"cold").sort("time_raw")
    processed = pl.read_parquet(ROOT/revision["processed_path"]).filter(
        (pl.col("power_w")==169.) & (pl.col("sensor_type")=="cold")).sort("time_raw")
    for column in ring.columns:
        np.testing.assert_array_equal(ring[column].to_numpy(),processed[column].to_numpy())
    pack = np.load(output/"evaluation/cold_predictions.npz")
    ids = np.flatnonzero(pack["x"][:,3]==169.)
    ids = ids[np.argsort(pack["x"][ids,2])]
    np.testing.assert_array_equal(pack["x"][ids,2],ring["time_raw"].to_numpy())
    np.testing.assert_array_equal(pack["y"][ids],(ring["value_mean_raw"].to_numpy()+273.15).astype(np.float32))
    for name,expected in provenance["raw_source_hashes"].items():
        if digest(ROOT/name)!=expected or protocol["raw_source_sha256"][name]!=expected:
            raise AssertionError("Training raw-source hash differs from the delivered data.")
    _,provider = experiment_history(ROOT,"test",fixed_splits(ROOT),Geometry.from_project(ROOT),cfg["model"])
    if provider.initial_temperature_k!=298.15:
        raise AssertionError("Experimental history normalization still uses the old initial temperature.")
    for power,(times,values) in provider.curves.items():
        if times[0]!=0 or not np.all(values[0]==298.15):
            raise AssertionError("Experimental history lacks the requested 25 C initial anchor.")
        for sensor in ("hot","cold"):
            dense = np.load(output/"evaluation"/f"dense_{sensor}_{power:g}W_predictions.npz")
            for method in METHODS:
                if dense["x"][0,2]!=0 or dense[method][0]!=np.float32(298.15):
                    raise AssertionError("Saved raw sensor prediction does not start at 25 C.")
    g = Geometry.from_project(ROOT)
    x = torch.tensor([[.028,g.bottom_cu,0.,169.,0.],[.0415,g.bottom_cu,0.,634.,0.],[0.,0.,0.,169.,1.]])
    for method in METHODS:
        model,_ = load_checkpoint(output/method/"epoch_1000.pt","cpu")
        model.set_history_providers(low=provider,high=provider)
        with torch.no_grad():
            torch.testing.assert_close(model(x,"high")[:,0],torch.tensor([298.15,298.15,295.15]),rtol=0,atol=0)
            torch.testing.assert_close(model(x,"low")[:,0],torch.full((3,),295.15),rtol=0,atol=0)
    return dict(corrected_raw_and_processed_sensor_verified=True,unchanged_other_inputs_verified=True,
                sensor_initial_25c_verified=True,sic_and_simulation_initial_22c_verified=True,
                user_requested_protocol_verified=True)


def audit(output):
    cfg = json.loads((output/"config.json").read_text(encoding="utf-8"))
    revised = bool(cfg["experiment"].get("data_revision"))
    provenance = json.loads((output/"provenance.json").read_text(encoding="utf-8"))
    evaluation = json.loads((output/"evaluation/provenance.json").read_text(encoding="utf-8"))
    result = json.loads((output/"evaluation/metrics.json").read_text(encoding="utf-8"))
    checks = {}
    for name,expected in provenance["source_hashes"].items():
        snapshot = output/"source_snapshot"/name
        current = ROOT/("configs" if name.endswith(".yaml") else "scripts")/name
        if name=="sensors.py":
            current = ROOT/"src/sic_cu/data/sensors.py"
        if digest(snapshot)!=expected or digest(current)!=expected:
            raise AssertionError(f"Source changed after training started: {name}")
    for name,expected in provenance["data_hashes"].items():
        if digest(ROOT/name)!=expected:
            raise AssertionError(f"Input data changed after training: {name}")
    initial_shared = None
    temporal_rng_state = None
    objective_rng_state = None
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
        for name in ("top_center","top_tail","top_tail_slope","top_tail_endpoint"):
            if cfg["loss_weights"].get(name,0)>0 and not all(
                    name in row and np.isfinite(row[name]) and row[name]>=0 for row in history):
                raise AssertionError(f"Measured temperature objective missing: {method}/{name}")
        if cfg["loss_weights"].get("top_center",0)>0:
            current_rng = state["extra"]["objective_rng"]
            if objective_rng_state is None:
                objective_rng_state = current_rng
            elif objective_rng_state!=current_rng:
                raise AssertionError("Center observation sampling differs across methods.")
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
        if digest(path)!=manifest["png_sha256"][name]:
            raise AssertionError(f"Figure provenance changed: {name}")
        if manifest.get("formats")==["png"]:
            if path.with_suffix(".pdf").exists():
                raise AssertionError(f"PNG-only export includes an obsolete PDF: {name}")
        elif not path.with_suffix(".pdf").is_file():
            raise AssertionError(f"Historical paired-format export is incomplete: {name}")
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
    expected_mode = "four_models_updated_measurements" if revised else "four_retrained_models"
    if (temporal["mode"]!=expected_mode or not temporal["no_prediction_smoothing"]
            or len(temporal["curves"])!=24
            or digest(ROOT/"scripts/audit_sequential_temporal.py")!=temporal["audit_script_sha256"]):
        raise AssertionError("Temporal comparison is incomplete or uses changed audit source.")
    for method in METHODS:
        if temporal["corrected_checkpoint_sha256"][method]!=checks[method]["checkpoint_sha256"]:
            raise AssertionError("Temporal audit used different corrected weights.")
    versions = (("corrected",output),)
    if not revised:
        baseline_directory = Path(temporal["baseline"])
        if not baseline_directory.is_absolute():
            baseline_directory = ROOT/baseline_directory
        versions = (("original",baseline_directory),("corrected",output))
    for row in temporal["curves"]:
        for version,directory in versions:
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
                if "post_warmup_incoming_integer_max_jump_k" in row:
                    close(float(np.abs(epsilon[knots>=3.,1]-epsilon[knots>=3.,0]).max()),
                          row["post_warmup_incoming_integer_max_jump_k"],"Post-warmup arrival jump")
    if cfg["loss_weights"].get("top_center",0)>0:
        expected_top = {(p,m) for p in (169.,339.,634.) for m in METHODS}
        top_rows = temporal["dense_top_curves"]
        if {(r["power_w"],r["method"]) for r in top_rows}!=expected_top or len(top_rows)!=12:
            raise AssertionError("Dense center-top audit is incomplete.")
        for power in (169.,339.,634.):
            name = f"dense_top_{power:g}W_predictions.npz"
            path = output/"evaluation"/name
            if digest(path)!=temporal["dense_top_prediction_sha256"][name]:
                raise AssertionError("Dense center-top predictions changed.")
            pack = np.load(path)
            x,n = pack["x"],int(pack["grid_count"])
            observed = dense_tables["顶部"].x
            sample = observed[observed[:,3]==power]
            coordinate = sample[[np.argmin(sample[:,0])]]
            grid = np.arange(0.,sample[:,2].max()+.01,.1,dtype=np.float32)
            knots = np.arange(2.,sample[:,2].max(),dtype=np.float32)
            epsilon_times = np.stack((knots-.001,knots,knots+.001),axis=1).reshape(-1)
            if n!=len(grid):
                raise AssertionError("Dense center-top grid misses observed time support.")
            np.testing.assert_array_equal(x[:,2],np.r_[grid,epsilon_times])
            np.testing.assert_array_equal(x[:,[0,1,3,4]],np.repeat(coordinate[:,[0,1,3,4]],len(x),axis=0))
            table = Table(x,np.zeros(len(x)),np.ones(len(x)),np.zeros(len(x))).validate()
            for method,model in models.items():
                np.testing.assert_allclose(table_predict(model,table,provider,"high"),pack[method],rtol=0,atol=2e-4)
                row = next(r for r in top_rows if (r["power_w"],r["method"])==(power,method))
                steps = np.diff(pack[method][:n])
                close(float(np.abs(steps).max()),row["max_abs_step_k"],"Dense top all-time step")
                close(float(np.abs(steps[grid[:-1]>=75.]).max()),row["late_max_abs_step_k"],"Dense top late step")
                epsilon = pack[method][n:].reshape(-1,3)
                for delta,tag in ((epsilon[:,2]-epsilon[:,1],"outgoing"),(epsilon[:,1]-epsilon[:,0],"incoming")):
                    close(float(np.abs(delta).max()),row[f"{tag}_integer_max_jump_k"],"Dense top arrival jump")
                    close(float(np.abs(delta[knots>=75.]).max()),row[f"late_{tag}_integer_max_jump_k"],"Dense top late arrival jump")
                close(float(np.abs(epsilon[knots>=3.,1]-epsilon[knots>=3.,0]).max()),
                      row["post_warmup_incoming_integer_max_jump_k"],"Dense top post-warmup jump")
    if not revised and cfg["model"].get("asl_rate_mode")=="thermal_trend":
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
        original_and_corrected_temporal_jumps_verified=not revised,
        dense_predictions_recomputed_from_all_four_models=True,
        shared_temporal_loss_verified=temporal_rng_state is not None,
        asl_thermal_adaptation_verified=not revised and cfg["model"].get("asl_rate_mode")=="thermal_trend",
        previous_asl_dense_predictions_recomputed=not revised and cfg["model"].get("asl_rate_mode")=="thermal_trend",
        note="Source/metric/image checks supplement manual visual inspection and the focused regression suite.")
    if cfg["loss_weights"].get("top_center",0)>0:
        audit_result["temperature_optimization"] = verify_temperature_bias(output)
    if revised:
        if manifest.get("formats")!=["png"] or any(p.suffix.lower() in (".pdf",".svg",".jpg",".jpeg",".tif",".tiff") for p in (output/"figures").rglob("*")):
            raise AssertionError("The current release must contain PNG figures only.")
        audit_result["sensor_data_revision"] = verify_sensor_revision(output,cfg,provenance)
    write_json(output/"verification.json",audit_result)
    print("Verified four actual 1000-update runs, exact shared initialization, source/data hashes, "
          "recomputed test metrics, 17 nonblank 300-DPI PNGs, cloud maxima and raw dense-time predictions.")
    return audit_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"研究记录/Sequential_DeepONet_1000epochs_sensor25_20261004")
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit(args.output.resolve())


if __name__=="__main__":
    main()
