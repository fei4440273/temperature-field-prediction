"""Verify revised ASL predictions, early response, and immutable comparison baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from joint_temperature_core import Geometry,Table,fixed_splits
from sequential_deeponet_core import load_checkpoint
from sequential_deeponet_data import experiment_history,simulation_history,metrics
from train_sequential_deeponet import ROOT,digest,table_predict,write_json
from evaluate_asl_iteration import BASELINES,as_table
from evaluate_sequential_deeponet import composite


def early_gate_derivatives(model,provider,tables):
    rows = []
    device = next(model.parameters()).device
    for name in ("热端","冷端"):
        table = tables[name]
        for power in np.unique(table.x[:,3]):
            coordinate = table.x[np.isclose(table.x[:,3],power)][0]
            for corner in (18.,21.,72.):
                long,local = provider.build([power],[corner])
                x = np.repeat(coordinate[None,:],2,axis=0)
                x[:,2] = [corner-.001,corner+.001]
                query = torch.tensor(x,device=device,requires_grad=True)
                prediction = model.forward_explicit(query,torch.tensor(long,device=device),
                    torch.tensor(local,device=device),torch.zeros(2,device=device,dtype=torch.long),"high")
                slopes = torch.autograd.grad(prediction.sum(),query)[0][:,2].detach().cpu().numpy()
                rows.append(dict(sensor=name,power_w=float(power),time_s=corner,
                    left_rate_k_per_s=float(slopes[0]),right_rate_k_per_s=float(slopes[1]),
                    abs_rate_change_k_per_s=float(abs(slopes[1]-slopes[0]))))
    return rows


def verify(output,*,require_targets=True):
    cfg = json.loads((output/"config.json").read_text(encoding="utf-8"))
    selection = json.loads((output/"selection.json").read_text(encoding="utf-8"))
    baseline = ROOT/cfg["experiment"]["baseline_output"]
    provenance = json.loads((output/"evaluation/provenance.json").read_text(encoding="utf-8"))
    result = json.loads((output/"evaluation/metrics.json").read_text(encoding="utf-8"))
    before = json.loads((baseline/"evaluation/metrics.json").read_text(encoding="utf-8"))
    original_identity = json.loads((baseline/"evaluation/provenance.json").read_text(encoding="utf-8"))
    for m in BASELINES:
        assert result[m]==before[m],f"Baseline reported metrics changed: {m}"
        assert digest(baseline/m/"epoch_1000.pt")==selection["baseline_sources"][m]["checkpoint_sha256"]
        assert selection["baseline_sources"][m]["checkpoint_sha256"]==original_identity["checkpoint_sha256"][m]
        assert digest(baseline/m/"history.csv")==selection["baseline_sources"][m]["history_sha256"]
    _,final = load_checkpoint(output/"asl/epoch_1000.pt","cpu")
    assert final["epoch"]==1000 and [r["epoch"] for r in final["history"]]==list(range(1,1001))
    checkpoint = output/"asl"/selection["checkpoint"]
    assert digest(checkpoint)==selection["checkpoint_sha256"]==provenance["asl_checkpoint_sha256"]
    model,state = load_checkpoint(checkpoint,"cpu")
    model.eval()
    prediction_device = "cuda" if torch.cuda.is_available() else "cpu"
    prediction_model,_ = load_checkpoint(checkpoint,prediction_device)
    prediction_model.eval()
    tables,provider = experiment_history(ROOT,"test",fixed_splits(ROOT),Geometry.from_project(ROOT),cfg["model"])
    low_provider,_ = simulation_history(ROOT,fixed_splits(ROOT)["low"]["test"],Geometry.from_project(ROOT),cfg["model"])
    assert state["config"]==cfg and state["epoch"]==selection["selected_epoch"]
    assert model.model_config["asl_maturity_mode"]!="legacy"
    assert sum(p.numel() for p in model.parameters())==result["asl"]["parameters"]==76227
    training = json.loads((output/"provenance.json").read_text(encoding="utf-8"))
    for name,sha in training["source_hashes"].items():
        assert digest(output/"source_snapshot"/name)==sha,f"Frozen source changed: {name}"
    for name,sha in training["data_hashes"].items():
        assert digest(ROOT/name)==sha,f"Input data changed: {name}"
    for name,sha in training["raw_source_hashes"].items():
        assert digest(ROOT/name)==sha,f"Raw sensor changed: {name}"
    if training.get("initial_checkpoint_sha256"):
        assert digest(ROOT/cfg["training"]["initial_checkpoint"])==training["initial_checkpoint_sha256"]
    for name,sha in provenance["prediction_source_sha256"].items():
        assert digest(ROOT/"scripts"/name)==sha,f"Prediction implementation changed: {name}"
    checked,dense_checks = [],[]
    for filename,sha in provenance["unchanged_prediction_files"].items():
        source = baseline/"evaluation"/filename
        assert digest(source)==sha,f"Baseline cache changed: {filename}"
        with np.load(source) as old,np.load(output/"evaluation"/filename) as new:
            assert old.files==new.files
            for key in old.files:
                if key!="asl":
                    np.testing.assert_array_equal(old[key],new[key],err_msg=f"Changed {filename}:{key}")
            fidelity = "low" if filename=="simulation_predictions.npz" else "high"
            raw = table_predict(prediction_model,as_table(new),low_provider if fidelity=="low" else provider,fidelity)
            np.testing.assert_allclose(raw,new["asl"],rtol=1e-6,atol=5e-5,
                                       err_msg=f"ASL cache differs from raw selected-model output: {filename}")
            if filename.startswith(("dense_hot_","dense_cold_")):
                n = int(new["grid_count"])
                t,p = new["x"][:n,2],new["asl"][:n]
                early = (t>=12.)&(t<=28.)
                old_p = old["asl"][:n]
                late = t>=75.
                triples = new["asl"][n:].reshape(-1,3)
                arrivals = new["x"][n:,2].reshape(-1,3)[:,1]
                late_jumps = np.abs(triples[arrivals>=75.,1]-triples[arrivals>=75.,0])
                incoming = float(late_jumps.max())
                assert incoming<.03,f"Late observation-arrival jump exceeds 0.03 K: {filename}"
                dense_checks.append(dict(file=filename,
                    early_previous_max_second_difference_k=float(abs(np.diff(old_p[early],n=2)).max()),
                    early_new_max_second_difference_k=float(abs(np.diff(p[early],n=2)).max()),
                    late_max_0_1s_step_k=float(abs(np.diff(p[late])).max()),
                    late_max_incoming_arrival_jump_k=incoming))
            if filename in ("top_predictions.npz","hot_predictions.npz","cold_predictions.npz","simulation_predictions.npz"):
                tag = filename.split("_")[0]
                name = {"top":"顶部","hot":"热端","cold":"冷端"}.get(tag)
                measured = metrics(new["y"],new["asl"])
                expected = result["asl"]["high_test"][name] if name else result["asl"]["low_test"]
                for key in measured:
                    np.testing.assert_allclose(measured[key],expected[key],rtol=1e-9,atol=1e-9)
            checked.append(filename)
    targets = {name:min(before[m]["high_test"][name]["rmse_k"] for m in BASELINES)
               for name in ("顶部","热端","冷端")}
    acceptance = {name:result["asl"]["high_test"][name]["rmse_k"]<limit for name,limit in targets.items()}
    acceptance["综合"] = composite(result["asl"]["high_test"])<min(before[m]["combined_test_rmse_k"] for m in BASELINES)
    if require_targets:
        assert all(acceptance.values()),f"ASL has not met all requested accuracy targets: {acceptance}"
    derivatives = early_gate_derivatives(model,provider,tables)
    assert max(row["abs_rate_change_k_per_s"] for row in derivatives)<.001,"Artificial gate corner remains."
    for name in ("热端","冷端"):
        x = tables[name].x[:3].copy()
        x[:,2] = 0.
        zero = Table(x,np.zeros(len(x)),np.ones(len(x)),np.zeros(len(x))).validate()
        np.testing.assert_array_equal(table_predict(model,zero,provider,"high"),np.full(len(x),298.15,np.float32))
    if (output/"figures/manifest.json").exists():
        manifest = json.loads((output/"figures/manifest.json").read_text(encoding="utf-8"))
        assert manifest["formats"]==["png"] and len(manifest["figures"])==17
        for name in manifest["figures"]:
            path = output/"figures"/name
            assert digest(path)==manifest["png_sha256"][name]
            with Image.open(path) as img:
                assert min(img.size)>500 and min(img.info["dpi"])>=299
                assert np.asarray(img.convert("RGB")).std()>10
        assert not any(p.suffix.lower() in (".pdf",".svg",".jpg",".tif",".tiff") for p in (output/"figures").rglob("*"))
        maxima = json.loads((output/"evaluation/cloud_maxima.json").read_text(encoding="utf-8"))
        assert len(maxima)==20 and {(r["time_s"],r["method"]) for r in maxima}=={
            (t,m) for t in (5,30,60,90,120) for m in (*BASELINES,"asl")}
        for row in maxima:
            path = output/"evaluation"/f"surface_634W_{row['time_s']}s.npz"
            assert digest(path)==row["source_npz_sha256"]
            with np.load(path) as pack:
                errors = np.abs(pack[row["method"]]-pack["y"])
                index = np.argmax(errors)
                np.testing.assert_allclose(errors[index],row["max_abs_error_k"])
                np.testing.assert_allclose(pack["xy_mm"][index],[row["x_mm"],row["y_mm"]])
    verification = dict(status="passed" if require_targets else "diagnostic_only_passed",
        acceptance_targets_enforced=require_targets,accuracy_acceptance=acceptance,
        all_three_baseline_models_histories_metrics_and_prediction_arrays_unchanged=True,
        all_asl_prediction_arrays_recomputed_from_selected_model=True,
        checked_prediction_files=checked,early_gate_derivatives=derivatives,dense_sensor_checks=dense_checks,
        verification_script_sha256=digest(Path(__file__)),
        no_prediction_smoothing=True,experimental_copper_initial_c=25.,
        actual_asl_parameters=result["asl"]["parameters"],selected_epoch=state["epoch"],
        completed_new_updates=1000,total_asl_training_updates=selection["total_training_updates"])
    if cfg["experiment"].get("sensor_plateau_revision"):
        from asl_sensor_shape import audit
        shape = audit(output,ROOT/cfg["experiment"]["previous_asl_output"])
        recorded = json.loads((output/"sensor_shape_audit.json").read_text(encoding="utf-8"))
        assert shape==recorded,"Sensor shape audit does not match raw predictions."
        if require_targets:
            assert shape["all_six_rate_errors_improved"],"Some sensor late-rate errors did not improve."
            assert shape["all_six_waviness_metrics_improved"],"Some raw sensor waviness did not improve."
            assert max(abs(r["rate_error_k_per_s"]) for r in shape["curves"])<.003,"Sensor late-rate mismatch remains excessive."
            assert max(r["late_max_incoming_jump_k"] for r in shape["curves"])<.0005,"Sensor arrival discontinuity remains."
        verification["sensor_shape"] = shape
    write_json(output/"verification.json",verification)
    print(json.dumps(verification,ensure_ascii=False),flush=True)
    return verification


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--diagnostic-only",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    verify(args.output.resolve(),require_targets=not args.diagnostic_only)
