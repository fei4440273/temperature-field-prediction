"""Run the four-method, 1000-epoch comparison on the project's real data."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import yaml

from joint_temperature_core import Geometry,PhysicalSettings,read_yaml,fixed_splits,load_simulation,physics_loss
from sequential_deeponet_core import METHODS,LABELS,SequentialDeepONet,load_checkpoint,save_checkpoint
from sequential_deeponet_data import BatchTable,experiment_history,simulation_history,metrics
from sequential_temperature_objective import (TopCurveConstraints,SensorCurveConstraints,
                                             center_table,center_tail_indices,sensor_tail_metrics)

ROOT = Path(__file__).resolve().parents[1]


def write_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def write_history(path,history):
    keys = list(dict.fromkeys(key for row in history for key in row))
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,keys)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def training_updates_in_checkpoint(state,*,root=ROOT,_seen=None):
    initial = state.get("extra",{}).get("initial_training_updates")
    if initial is None:
        initial = 0
        previous = state.get("config",{}).get("training",{}).get("initial_checkpoint")
        if previous:
            path = (root/previous).resolve()
            seen = set() if _seen is None else _seen
            if path in seen:
                raise ValueError("Cyclic checkpoint initialization ancestry.")
            parent = torch.load(path,map_location="cpu",weights_only=False)
            initial = training_updates_in_checkpoint(parent,root=root,_seen=seen|{path})
    return int(state["epoch"])+int(initial)


@torch.no_grad()
def table_predict(model,table,provider,fidelity,batch_size=16384):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    pairs,inverse = np.unique(table.x[:,[3,2]],axis=0,return_inverse=True)
    long,local = provider.build(pairs[:,0],pairs[:,1])
    branch = []
    for start in range(0,len(pairs),256):
        end = start+256
        branch.append(model.branch(torch.as_tensor(long[start:end],device=device),
            torch.as_tensor(local[start:end],device=device),
            torch.as_tensor(pairs[start:end,1]/model.geometry.time_max,device=device)))
    encoded = torch.cat(branch)
    predictions = []
    g = model.geometry
    for start in range(0,len(table.x),batch_size):
        end = start+batch_size
        x = torch.as_tensor(table.x[start:end],device=device)
        z = torch.stack(((x[:,0]/g.radius_cu).square(),
            (x[:,1]-g.bottom_cu)/(-g.bottom_cu),x[:,2]/g.time_max,x[:,4]),1)
        trunk = model.trunk(z)
        features = encoded[torch.as_tensor(inverse[start:end],device=device)]
        result = (model.low_projection(features)*trunk).sum(-1)/math.sqrt(model.rank)+model.low_bias
        if fidelity=="high":
            result += (model.high_projection(features)*trunk).sum(-1)/math.sqrt(model.rank)+model.high_bias
        initial = model.initial_temperature(x,fidelity)[:,0]
        anchor = 1-torch.exp(-x[:,2]/float(model.model_config.get("initial_tau_s",5.)))
        predictions.append((initial+model.temperature_scale*anchor*result).cpu().numpy())
    model.train(was_training)
    return np.concatenate(predictions)


def validate(model,tables,provider,selection=None):
    predictions = {name:table_predict(model,table,provider,"high") for name,table in tables.items()}
    result = {name:metrics(table.y,predictions[name]) for name,table in tables.items()}
    if selection and selection.get("criterion")=="max_reference_rmse_ratio":
        references = selection["reference_rmse_k"]
        if (set(references)!={"顶部","热端","冷端"}
                or any(not math.isfinite(float(v)) or float(v)<=0 for v in references.values())):
            raise ValueError("Balanced selection requires positive references for all three measured regions.")
        score = max(result[name]["rmse_k"]/float(value) for name,value in references.items())
        rate_scale = selection.get("sensor_tail_rate_scale_k_per_s")
        if rate_scale is not None:
            if not math.isfinite(float(rate_scale)) or rate_scale<=0:
                raise ValueError("Sensor tail validation rate scale must be finite and positive.")
            rates = []
            for name in ("热端","冷端"):
                tail = sensor_tail_metrics(tables[name],predictions[name],float(selection.get("sensor_tail_window_s",20.)))
                result[name]["tail_rate_rmse_k_per_s"] = tail["rate_rmse_k_per_s"]
                rates.extend(r["rate_error_k_per_s"] for r in tail["curves"])
            score = max(score,float(np.sqrt(np.mean(np.square(rates))))/float(rate_scale))
        return score,result
    weights = {"顶部":.5,"热端":.25,"冷端":.25}
    if selection:
        ids = center_tail_indices(tables["顶部"],float(selection.get("tail_fraction",.7)))
        result["中心后段"] = metrics(tables["顶部"].y[ids],predictions["顶部"][ids])
        weights = selection["weights"]
    if (not weights or any(k not in result or not math.isfinite(float(v)) or v<0 for k,v in weights.items())
            or sum(weights.values())<=0):
        raise ValueError("Validation selection requires finite nonnegative observed-metric weights.")
    score = math.sqrt(sum(float(w)*result[k]["rmse_k"]**2 for k,w in weights.items())/sum(weights.values()))
    return score,result


def temporal_sensor_curvature(model,provider,sensor_tables,rng,delta_s):
    """Penalize raw prediction curvature across training-sensor arrivals."""
    if not math.isfinite(delta_s) or delta_s<=0:
        raise ValueError("Temporal probe spacing must be finite and positive.")
    queries = []
    for power in sorted(provider.curves):
        times,_ = provider.curves[power]
        arrivals = times[1:]+provider.lag
        arrivals = arrivals[(arrivals>delta_s)&(arrivals<=times[-1]-delta_s)]
        if not len(arrivals):
            raise ValueError(f"No interior sensor arrival for training power {power}.")
        center = rng.choice(arrivals)
        for name in ("热端","冷端"):
            table = sensor_tables[name]
            coordinates = table.x[np.isclose(table.x[:,3],power,atol=1e-3,rtol=0)][0]
            triple = np.repeat(coordinates[None,:],3,axis=0)
            triple[:,2] = center+np.array([-delta_s,0.,delta_s])
            queries.append(triple)
    x = torch.as_tensor(np.concatenate(queries),device=next(model.parameters()).device)
    pairs,inverse = torch.unique(x[:,[3,2]],dim=0,return_inverse=True)
    long,local = provider.build(pairs[:,0].cpu().numpy(),pairs[:,1].cpu().numpy())
    prediction = model.forward_explicit(x,torch.as_tensor(long,device=x.device),
        torch.as_tensor(local,device=x.device),inverse,"high").reshape(-1,3)
    curvature = (prediction[:,0]-2*prediction[:,1]+prediction[:,2])/(model.temperature_scale*delta_s**2)
    return curvature.square().mean()


def run_method(method,output,cfg,geometry,settings,pools,providers,val_tables,val_provider,
               *,device,resume=False):
    output.mkdir(parents=True,exist_ok=True)
    logger = logging.getLogger(method)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (logging.StreamHandler(),logging.FileHandler(output/"training.log",encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    seed = int(cfg["experiment"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    temporal_rng = np.random.default_rng(seed+20000)
    objective_rng = np.random.default_rng(seed+30000)
    generator = torch.Generator(device=device).manual_seed(seed+10000)
    model = SequentialDeepONet(method,geometry,settings,cfg["model"]).to(device)
    initial_checkpoint = cfg["training"].get("initial_checkpoint")
    initial_updates = 0
    if initial_checkpoint:
        if method!="asl" or cfg["experiment"]["methods"]!=["asl"]:
            raise ValueError("Warm-start adaptation is restricted to ASL-only experiments.")
        initial_model,initial_state = load_checkpoint(ROOT/initial_checkpoint,device)
        if (initial_state["method"]!=method or initial_state["geometry"]!=asdict(geometry)
                or initial_state["physical_settings"]!=asdict(settings)):
            raise ValueError("ASL initialization must use the same physical problem.")
        model.load_state_dict(initial_model.state_dict(),strict=True)
        initial_updates = training_updates_in_checkpoint(initial_state)
        del initial_model
    model.set_history_providers(low=providers["low"],high=providers["high"])
    train = cfg["training"]
    epochs = int(train["epochs"])
    optimizer = torch.optim.AdamW(model.parameters(),lr=float(train["learning_rate"]),
                                  weight_decay=float(train["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,
                                                          eta_min=float(train["final_learning_rate"]))
    history,best_score,best_epoch,start_epoch,elapsed = [],float("inf"),0,1,0.
    latest = output/"latest.pt"
    if latest.exists():
        if not resume:
            raise FileExistsError(f"Use --resume for existing run {output}")
        restored,state = load_checkpoint(latest,device)
        if state["method"]!=method or state["config"]!=cfg:
            raise ValueError("Checkpoint/config mismatch; do not change an active experiment.")
        model.load_state_dict(restored.state_dict())
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        start_epoch = state["epoch"]+1
        rng.bit_generator.state = state["numpy_generator"]
        if "temporal_rng" in state["extra"]:
            temporal_rng.bit_generator.state = state["extra"]["temporal_rng"]
        elif cfg["loss_weights"].get("temporal",0)>0:
            raise ValueError("Temporal regularization checkpoint lacks its sampling RNG.")
        if "objective_rng" in state["extra"]:
            objective_rng.bit_generator.state = state["extra"]["objective_rng"]
        elif cfg["loss_weights"].get("top_center",0)>0:
            raise ValueError("Central observation checkpoint lacks its sampling RNG.")
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all([state_.cpu() for state_ in state["cuda_rng"]])
        generator.set_state(state["extra"]["physics_rng"].cpu())
        best_score = state["extra"]["best_score"]
        best_epoch = state["extra"]["best_epoch"]
        elapsed = state["extra"]["training_seconds"]
    elif resume and (output/"final_info.json").exists():
        raise ValueError("Completed run lacks its recovery checkpoint.")
    else:
        save_checkpoint(output/"initial.pt",model,epoch=0,config=cfg,history=[],seed=seed,
                        extra=dict(initial_training_updates=initial_updates))
    logger.info("%s: same data/physics, no boundary attention, %d total epochs, seed=%d, parameters=%d",
                LABELS[method],epochs,seed,sum(p.numel() for p in model.parameters()))
    score_unit = "ratio" if cfg.get("validation_selection",{}).get("criterion")=="max_reference_rmse_ratio" else "K"
    weights = cfg["loss_weights"]
    center_pool = None
    tail_constraints = None
    sensor_constraints = None
    if float(weights.get("top_center",0.))>0:
        center_pool = BatchTable(center_table(pools["top"].table,float(train["top_center_radius_m"])),
                                 providers["high"],device)
    if any(float(weights.get(k,0.))>0 for k in ("top_tail","top_tail_slope","top_tail_endpoint")):
        tail_constraints = TopCurveConstraints(pools["top"].table,providers["high"],device,
                                                float(train["top_tail_window_s"]))
    if any(float(weights.get(k,0.))>0 for k in ("sensor_tail","sensor_tail_slope","sensor_tail_endpoint","sensor_shape")):
        if method!="asl" or cfg["experiment"]["methods"]!=["asl"]:
            raise ValueError("Sensor plateau adaptation is restricted to ASL-only experiments.")
        sensor_constraints = SensorCurveConstraints(
            {"热端":pools["hot"].table,"冷端":pools["cold"].table},providers["high"],device,
            float(train.get("sensor_tail_window_s",20.)),shape_points=int(train.get("sensor_shape_points_per_power",12)),
            shape_delta_s=float(train.get("sensor_shape_delta_s",2.)))
    names = {"传热方程":"pde","边界条件":"boundary","初始条件":"initial", "材料界面":"interface",
             "仿真水冷":"low_cooling","热阻正则":"contact_prior"}
    observation_counts = {"simulation":int(train["simulation_points_per_power_material"]),
        "top":int(train["top_points_per_power"]),"hot":int(train["sensor_points_per_power"]),
        "cold":int(train["sensor_points_per_power"])}
    base_epoch = start_epoch
    timer = time.perf_counter()
    base_elapsed = elapsed
    for epoch in range(start_epoch,epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pieces = {}
        for name,pool in pools.items():
            ids = pool.sample(observation_counts[name],rng)
            x,long,local,inverse = pool.inputs(ids)
            yhat = model.forward_explicit(x,long,local,inverse,"low" if name=="simulation" else "high")
            pieces[name] = ((yhat-pool.y[ids])/model.temperature_scale).square().mean()
        if center_pool is not None:
            ids = center_pool.sample(int(train["top_center_points_per_power"]),objective_rng)
            x,long,local,inverse = center_pool.inputs(ids)
            prediction = model.forward_explicit(x,long,local,inverse,"high")
            pieces["top_center"] = ((prediction-center_pool.y[ids])/model.temperature_scale).square().mean()
        if tail_constraints is not None:
            pieces.update({k:v for k,v in tail_constraints.losses(model).items() if float(weights.get(k,0.))>0})
        if sensor_constraints is not None:
            pieces.update({k:v for k,v in sensor_constraints.losses(model).items() if float(weights.get(k,0.))>0})
        physical = physics_loss(model,int(train["physics_points_per_region"]),generator)
        pieces.update({names[name]:value for name,value in physical.items()})
        if float(weights.get("temporal",0.))>0:
            pieces["temporal"] = temporal_sensor_curvature(model,providers["high"],
                {"热端":pools["hot"].table,"冷端":pools["cold"].table},temporal_rng,
                float(train["temporal_probe_delta_s"]))
        loss = sum(float(weights[name])*value for name,value in pieces.items())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss in {method}, epoch {epoch}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),float(train["gradient_clip"]),
                                             error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        elapsed = base_elapsed+time.perf_counter()-timer
        row = dict(epoch=epoch,loss=float(loss.detach()),learning_rate=float(scheduler.get_last_lr()[0]),
                   training_seconds=elapsed,gradient_norm=float(norm))
        row.update({name:float(value.detach()) for name,value in pieces.items()})
        selection = None
        if epoch==1 or epoch%int(train["validation_every"])==0 or epoch==epochs:
            selection,val_metrics = validate(model,val_tables,val_provider,cfg.get("validation_selection"))
            row["validation_score_k"] = selection
            row.update({"validation_"+name+"_rmse_k":value["rmse_k"] for name,value in val_metrics.items()})
            row.update({"validation_"+name+"_tail_rate_rmse_k_per_s":value["tail_rate_rmse_k_per_s"]
                        for name,value in val_metrics.items() if "tail_rate_rmse_k_per_s" in value})
            if selection<best_score:
                best_score,best_epoch = selection,epoch
                save_checkpoint(output/"best.pt",model,epoch=epoch,config=cfg,history=history+[row],seed=seed,
                                extra=dict(validation_metrics=val_metrics,validation_score_k=selection,
                                           initial_training_updates=initial_updates))
            logger.info("epoch=%d/%d loss=%.6g val=%.4f %s best=%.4f %s @%d elapsed=%.1fs",
                        epoch,epochs,row["loss"],selection,score_unit,best_score,score_unit,best_epoch,elapsed)
        history.append(row)
        if selection is not None or epoch%int(train["checkpoint_every"])==0:
            extra = dict(best_score=best_score,best_epoch=best_epoch,training_seconds=elapsed,initial_training_updates=initial_updates,
                         physics_rng=generator.get_state(),temporal_rng=temporal_rng.bit_generator.state,
                         objective_rng=objective_rng.bit_generator.state)
            save_checkpoint(latest,model,epoch=epoch,config=cfg,history=history,seed=seed,
                            optimizer=optimizer,scheduler=scheduler,numpy_generator=rng,extra=extra)
            write_history(output/"history.csv",history)
            write_json(output/"progress.json",dict(method=method,pid=os.getpid(),epoch=epoch,total_epochs=epochs,
                status="running",loss=row["loss"],best_epoch=best_epoch,best_validation_rmse_k=best_score,
                training_seconds=elapsed,seconds_per_epoch=(elapsed-base_elapsed)/(epoch-base_epoch+1)))
    extra = dict(best_score=best_score,best_epoch=best_epoch,training_seconds=elapsed,initial_training_updates=initial_updates,
                 physics_rng=generator.get_state(),temporal_rng=temporal_rng.bit_generator.state,
                 objective_rng=objective_rng.bit_generator.state)
    save_checkpoint(output/"epoch_1000.pt" if epochs==1000 else output/"final.pt",model,
        epoch=epochs,config=cfg,history=history,seed=seed,optimizer=optimizer,scheduler=scheduler,
        numpy_generator=rng,extra=extra)
    write_history(output/"history.csv",history)
    info = dict(method=method,label=LABELS[method],epochs_completed=epochs,seed=seed,
        parameters=sum(p.numel() for p in model.parameters()),training_seconds=elapsed,
        best_epoch=best_epoch,best_validation_rmse_k=best_score,device=str(device),
        history_length=len(history),boundary_attention=False,protocol="conditional_past_sensor_history")
    if initial_checkpoint:
        info.update(initial_checkpoint=initial_checkpoint,initial_checkpoint_sha256=digest(ROOT/initial_checkpoint),
                    initial_training_updates=initial_updates,total_training_updates=initial_updates+epochs)
    write_json(output/"final_info.json",info)
    write_json(output/"progress.json",dict(info,status="completed",pid=os.getpid()))
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/sequential_deeponet_sensor25.yaml")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--methods",nargs="+",choices=METHODS)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--smoke-epochs",type=int,default=0)
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    if args.smoke_epochs:
        if not args.output:
            parser.error("Smoke runs must use a separate --output directory.")
        cfg["training"]["epochs"] = args.smoke_epochs
        cfg["training"]["validation_every"] = min(5,args.smoke_epochs)
    elif cfg["training"]["epochs"]!=1000:
        parser.error("Formal comparison requires exactly 1000 epochs per method.")
    output = (args.output or ROOT/cfg["experiment"]["output"]).resolve()
    output.mkdir(parents=True,exist_ok=True)
    configuration = output/"config.json"
    if configuration.exists() and json.loads(configuration.read_text(encoding="utf-8"))!=cfg:
        raise ValueError("Existing result directory contains a different experiment configuration.")
    write_json(configuration,cfg)
    torch.set_num_threads(int(cfg["training"]["cpu_threads"]))
    torch.backends.cudnn.benchmark = False
    geometry = Geometry.from_project(ROOT)
    settings = PhysicalSettings.from_project(ROOT,cfg)
    splits = fixed_splits(ROOT,"legacy_split")
    write_json(output/"splits.json",splits)
    snapshots = output/"source_snapshot"
    snapshots.mkdir(exist_ok=True)
    paths = [Path(__file__),ROOT/"scripts/sequential_deeponet_core.py",ROOT/"scripts/sequential_deeponet_data.py",
             ROOT/"scripts/joint_temperature_core.py",ROOT/"scripts/sequential_temperature_objective.py",args.config]
    raw_hashes = {}
    if cfg["experiment"].get("data_revision"):
        source = ROOT/"data/test_Data/colddata/colddata169W.csv"
        manifest = json.loads((ROOT/"data/processed/manifest.json").read_text(encoding="utf-8"))
        entry = next(e for e in manifest["sensors"] if e["source"]==str(source.relative_to(ROOT)))
        if entry["source_sha256"]!=digest(source):
            raise ValueError("Corrected cold sensor cache is stale; run refresh_sequential_sensor_data.py first.")
        raw_hashes[str(source.relative_to(ROOT))] = digest(source)
        paths.extend((ROOT/"scripts/refresh_sequential_sensor_data.py",ROOT/"src/sic_cu/data/sensors.py"))
        iteration = cfg["experiment"].get("asl_iteration",False)
        write_json(output/"revision_protocol.json",dict(mode="asl_only_balanced_smooth_iteration" if iteration else "user_requested_data_and_initial_temperature_revision",
            data_revision=cfg["experiment"]["data_revision"],config_json_sha256=digest(configuration),
            selection_data=["train","validation"],hyperparameter_search=bool(iteration),test_labels_for_selection=False,
            known_test_benchmark=bool(iteration),
            experimental_copper_initial_c=cfg["model"]["experiment_copper_initial_c"],
            simulation_initial_c=settings.initial_simulation_k-273.15,
            experimental_sic_initial_c=settings.initial_experiment_k-273.15,
            raw_source_sha256=raw_hashes))
    for path in paths:
        target = snapshots/path.name
        if target.exists() and digest(target)!=digest(path):
            raise ValueError(f"Active experiment source changed: {path.name}")
        if not target.exists():
            shutil.copy2(path,target)
    initial_path = cfg["training"].get("initial_checkpoint")
    write_json(output/"provenance.json",dict(source_hashes={p.name:digest(p) for p in paths},
        initial_checkpoint_sha256=digest(ROOT/initial_path) if initial_path else None,
        torch_version=torch.__version__,device=args.device,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        data_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/"data/processed").rglob("*.parquet")},
        raw_source_hashes=raw_hashes,
        epoch_definition="one balanced sampled-data update including every training power and physics collocation",
        history_derivative="held fixed for PDE coordinate derivatives, as in ASL-PINN source",
        temporal_regularization="raw +/- delta_s predictions at training sensor arrivals; shared by all four methods"))
    print("Preparing real training and validation data; test labels remain unopened.",flush=True)
    low_provider,nodes = simulation_history(ROOT,splits["low"]["training"],geometry,cfg["model"])
    low_table = load_simulation(ROOT,splits["low"]["training"])
    high_tables,high_provider = experiment_history(ROOT,"train",splits,geometry,cfg["model"])
    val_tables,val_provider = experiment_history(ROOT,"validation",splits,geometry,cfg["model"])
    write_json(output/"simulation_sensor_nodes.json",nodes)
    providers = {"low":low_provider,"high":high_provider}
    device = torch.device(args.device)
    pools = {"simulation":BatchTable(low_table,low_provider,device)}
    pools.update({tag:BatchTable(high_tables[name],high_provider,device)
                  for tag,name in (("top","顶部"),("hot","热端"),("cold","冷端"))})
    write_json(output/"data_counts.json",dict(simulation_points=len(low_table.x),
        high_training={name:len(table.x) for name,table in high_tables.items()},
        high_validation={name:len(table.x) for name,table in val_tables.items()}))
    for method in (args.methods or cfg["experiment"]["methods"]):
        run_method(method,output/method,cfg,geometry,settings,pools,providers,val_tables,val_provider,
                   device=device,resume=args.resume)
    print("All requested runs completed. Evaluation and PNG export use plot_sequential_deeponet.py.",flush=True)


if __name__=="__main__":
    main()
