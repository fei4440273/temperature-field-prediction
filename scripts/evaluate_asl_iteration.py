"""Evaluate only revised ASL, retaining the three published baseline predictions."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from joint_temperature_core import Geometry,Table,fixed_splits
from sequential_deeponet_core import METHODS,LABELS,load_checkpoint
from sequential_deeponet_data import experiment_history,simulation_history,metrics
from train_sequential_deeponet import ROOT,digest,table_predict,write_json
from evaluate_sequential_deeponet import composite,per_case_metrics
from audit_sequential_temporal import dense_statistics

BASELINES = ("fnn","gru","lstm")


def replace_asl(pack,prediction):
    result = {key:pack[key] for key in pack.files}
    prediction = np.asarray(prediction)
    if prediction.shape!=result["asl"].shape or not np.isfinite(prediction).all():
        raise ValueError("ASL replacement must match the cached prediction shape and be finite.")
    result["asl"] = prediction
    return result


def as_table(pack):
    x = pack["x"]
    y = pack["y"] if "y" in pack.files else np.zeros(len(x))
    return Table(x,y,np.ones(len(x)),np.zeros(len(x))).validate()


def evaluate(output,checkpoint_name,device):
    cfg = json.loads((output/"config.json").read_text(encoding="utf-8"))
    baseline = ROOT/cfg["experiment"]["baseline_output"]
    info = json.loads((output/"asl/final_info.json").read_text(encoding="utf-8"))
    completed,completion = load_checkpoint(output/"asl/epoch_1000.pt","cpu")
    if (info["epochs_completed"]!=1000 or len(completion["history"])!=1000
            or [r["epoch"] for r in completion["history"]]!=list(range(1,1001))
            or completion["config"]!=cfg or cfg["experiment"]["methods"]!=["asl"]):
        raise ValueError("The declared ASL training must complete before evaluation.")
    checkpoint = output/"asl"/checkpoint_name
    model,state = load_checkpoint(checkpoint,device)
    if state["config"]!=cfg or state["method"]!="asl":
        raise ValueError("Selected ASL checkpoint identity mismatch.")
    model.eval()
    selection = json.loads((output/"candidate_validation.json").read_text(encoding="utf-8"))
    if selection["checkpoints"][checkpoint_name]["sha256"]!=digest(checkpoint):
        raise ValueError("Selected checkpoint must be inspected on train/validation first.")
    baseline_sources = {m:dict(checkpoint_sha256=digest(baseline/m/"epoch_1000.pt"),
        history_sha256=digest(baseline/m/"history.csv")) for m in BASELINES}
    write_json(output/"selection.json",dict(checkpoint=checkpoint_name,checkpoint_sha256=digest(checkpoint),
        selected_epoch=state["epoch"],completed_updates=1000,
        total_training_updates=info.get("total_training_updates",1000),
        selection_data=["train","validation"],known_test_benchmark=True,
        selection_rule=cfg["validation_selection"],baseline_sources=baseline_sources))
    geometry,splits = Geometry.from_project(ROOT),fixed_splits(ROOT)
    high_tables,high_provider = experiment_history(ROOT,"test",splits,geometry,cfg["model"])
    low_provider,_ = simulation_history(ROOT,splits["low"]["test"],geometry,cfg["model"])
    destination = output/"evaluation"
    destination.mkdir(exist_ok=True)
    results = copy.deepcopy(json.loads((baseline/"evaluation/metrics.json").read_text(encoding="utf-8")))
    cases = copy.deepcopy(json.loads((baseline/"evaluation/case_metrics.json").read_text(encoding="utf-8")))
    record = dict(baseline_directory=str(baseline.relative_to(ROOT)),baseline_sources=baseline_sources,
                  asl_checkpoint_sha256=digest(checkpoint),asl_checkpoint=checkpoint_name,
                  selected_epoch=state["epoch"],completed_updates=1000,unchanged_prediction_files={})
    record["prediction_source_sha256"] = {name:digest(ROOT/"scripts"/name) for name in (
        "sequential_deeponet_core.py","sequential_deeponet_data.py","joint_temperature_core.py",
        "train_sequential_deeponet.py","evaluate_asl_iteration.py")}
    measured = {}
    predictions = {}
    for tag,name in (("top","顶部"),("hot","热端"),("cold","冷端"),("simulation",None)):
        source = baseline/"evaluation"/f"{tag}_predictions.npz"
        with np.load(source) as pack:
            table = as_table(pack)
            if name:
                np.testing.assert_array_equal(table.x,high_tables[name].x)
                np.testing.assert_array_equal(table.y,high_tables[name].y)
            pred = table_predict(model,table,high_provider if name else low_provider,"high" if name else "low")
            np.savez_compressed(destination/source.name,**replace_asl(pack,pred))
            measured[tag] = metrics(table.y,pred)
            predictions[tag] = table,pred
        record["unchanged_prediction_files"][source.name] = digest(source)
    high = {name:measured[tag] for tag,name in (("top","顶部"),("hot","热端"),("cold","冷端"))}
    low_table,low_pred = predictions["simulation"]
    cases["asl"] = dict(simulation=per_case_metrics(low_table,low_pred,exclude_initial=True),
                        top=per_case_metrics(*predictions["top"]))
    asl = dict(final_epoch=state["epoch"],completed_updates=1000,
        total_training_updates=info.get("total_training_updates",1000),selected_epoch=state["epoch"],high_test=high,
        high_test_by_power={f"{power:g}":{name:metrics(table.y[np.isclose(table.x[:,3],power)],
            predictions[tag][1][np.isclose(table.x[:,3],power)])
            for tag,name in (("top","顶部"),("hot","热端"),("cold","冷端"))
            for table in [predictions[tag][0]]} for power in np.unique(high_tables["热端"].x[:,3])},
        low_test=measured["simulation"],
        low_test_by_material={name:metrics(low_table.y[low_table.x[:,4]==material],low_pred[low_table.x[:,4]==material])
                             for name,material in (("copper",0),("sic",1))},
        combined_test_rmse_k=composite(high),parameters=info["parameters"],training_seconds=info["training_seconds"],
        best_epoch=info["best_epoch"])
    for kind,rows in cases["asl"].items():
        errors = np.array([r["relative_l2_kelvin_pct"] for r in rows])
        asl[kind+"_case_statistics"] = dict(cases=len(rows),mean_l2_kelvin_pct=float(errors.mean()),
            std_across_cases_l2_kelvin_pct=float(errors.std(ddof=1)),p90_l2_kelvin_pct=float(np.percentile(errors,90)),
            max_l2_kelvin_pct=float(errors.max()))
    results["asl"] = asl
    for source in sorted((baseline/"evaluation").glob("surface_*s.npz")):
        with np.load(source) as pack:
            pred = table_predict(model,as_table(pack),high_provider,"high")
            np.savez_compressed(destination/source.name,**replace_asl(pack,pred))
        record["unchanged_prediction_files"][source.name] = digest(source)
    dense_rows = []
    for source in sorted((baseline/"evaluation").glob("dense_*_predictions.npz")):
        with np.load(source) as pack:
            pred = table_predict(model,as_table(pack),high_provider,"high")
            np.savez_compressed(destination/source.name,**replace_asl(pack,pred))
            n = int(pack["grid_count"])
            dense_rows.append(dict(file=source.name,**dense_statistics(pack["x"][:n,2],pred[:n])))
        record["unchanged_prediction_files"][source.name] = digest(source)
    write_json(output/"temporal_audit.json",dict(no_prediction_smoothing=True,dense_curves=dense_rows))
    write_json(destination/"metrics.json",results)
    write_json(destination/"case_metrics.json",cases)
    write_json(destination/"provenance.json",record)
    with (destination/"summary.csv").open("w",encoding="utf-8",newline="") as handle:
        writer = csv.DictWriter(handle,["method","selected_epoch","top_rmse_k","hot_rmse_k","cold_rmse_k","combined_rmse_k","simulation_rmse_k"])
        writer.writeheader()
        for m in METHODS:
            r = results[m]
            writer.writerow(dict(method=m,selected_epoch=r.get("selected_epoch",1000),
                top_rmse_k=r["high_test"]["顶部"]["rmse_k"],hot_rmse_k=r["high_test"]["热端"]["rmse_k"],
                cold_rmse_k=r["high_test"]["冷端"]["rmse_k"],combined_rmse_k=r["combined_test_rmse_k"],
                simulation_rmse_k=r["low_test"]["rmse_k"]))
    print(json.dumps(dict(selected_epoch=state["epoch"],high_test=high,combined_test_rmse_k=composite(high),
                         low_test=measured["simulation"]),ensure_ascii=False),flush=True)
    return results


def export(output):
    import plot_sequential_deeponet as plot
    cfg = json.loads((output/"config.json").read_text(encoding="utf-8"))
    baseline = ROOT/cfg["experiment"]["baseline_output"]
    evaluation,directory = output/"evaluation",output/"figures"
    directory.mkdir(exist_ok=True)
    appendix = directory/"appendix"
    appendix.mkdir(exist_ok=True)
    plot.FIGURES.clear()
    plot.SURFACE_MAXIMA.clear()
    plot.plt.rcParams.update({"font.size":10,"axes.titlesize":11,"axes.labelsize":10,
        "font.family":["Noto Sans CJK JP","DejaVu Sans"],"axes.spines.top":False,"axes.spines.right":False})
    results = json.loads((evaluation/"metrics.json").read_text(encoding="utf-8"))
    cases = json.loads((evaluation/"case_metrics.json").read_text(encoding="utf-8"))
    with np.load(evaluation/"simulation_predictions.npz") as simulation:
        histories = {m:(output if m=="asl" else baseline)/m/"history.csv" for m in METHODS}
        plot.training_figures(output,directory,history_sources=histories,comparable_validation=True)
        plot.metric_figures(results,directory)
        plot.test_power_figures(results,directory)
        plot.histograms(cases,appendix)
        write_json(evaluation/"percentile_field_selections.json",plot.percentile_fields(simulation,cases,appendix))
        plot.shared_simulation(simulation,appendix)
    selection = json.loads((output/"selection.json").read_text(encoding="utf-8"))
    plot.surface_fields(evaluation,directory,model_caption=f"固定基线；ASL 本轮第 {selection['selected_epoch']} 次更新")
    plot.profile_figures(evaluation,directory)
    plot.parity_figure(evaluation,results,directory)
    write_json(evaluation/"cloud_maxima.json",plot.SURFACE_MAXIMA)
    write_json(directory/"manifest.json",dict(dpi=300,formats=["png"],figures=plot.FIGURES,
        png_sha256={name:digest(directory/name) for name in plot.FIGURES},
        export_script_sha256=digest(Path(__file__)),plot_script_sha256=digest(ROOT/"scripts/plot_sequential_deeponet.py")))
    selection = json.loads((output/"selection.json").read_text(encoding="utf-8"))
    lines = ["# ASL 专项改造与固定基线对比","",
        "FNN、GRU、LSTM 使用上一版原始模型及预测，未重新训练。ASL 保留 SSM→LSTM 主体及参数规模，",
        "取消 18 秒硬切换门控，平滑局部窗口过渡，并平衡三个实验区域的监督。",
        f"本轮训练完成 1000 次更新，ASL 包含初始化训练的累计预算为 {selection['total_training_updates']} 次更新，另外三种基线仍为原来的 1000 次。",
        f"依据训练/验证记录选取本轮第 {selection['selected_epoch']} 次更新的检查点。选型和预算见 selection.json。",
        "这是已检查过的现有测试基准上的改进，不能视为未接触测试集的盲测结论。","",
        "| 方法 | 顶部 RMSE / K | 热端 RMSE / K | 冷端 RMSE / K | 综合 RMSE / K |",
        "|---|---:|---:|---:|---:|"]
    for m in METHODS:
        r = results[m]
        lines.append(f"| {m.upper()} | {r['high_test']['顶部']['rmse_k']:.6f} | {r['high_test']['热端']['rmse_k']:.6f} | {r['high_test']['冷端']['rmse_k']:.6f} | {r['combined_test_rmse_k']:.6f} |")
    lines += ["","Figure 8 热端、冷端使用未做滤波或平滑处理的密集时间预测，顶部使用实际观测时刻的原始预测。实验铜冷端、热端在 t=0 时为 25 ℃；",
              "仿真与实验 SiC 初温继续为 22 ℃。云图标出最大绝对误差，仅输出 PNG。","",
              "Figure 3 显示各测试功率的三类实验误差，用于检查平均分是否掩盖某一功率的退步。",
              "主图直接对照实验数据；仿真全场诊断放在 appendix。","",
              "原模型与预测不变的逐数组核验记录见 verification.json，来源见 evaluation/provenance.json。","",
              "## 图片","",*[f"- [{name}](figures/{name})" for name in plot.FIGURES]]
    (output/"对比总结.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--checkpoint",choices=("best.pt","epoch_1000.pt"),default="best.pt")
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--export-only",action="store_true")
    parser.add_argument("--figures",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not args.export_only:
        evaluate(args.output.resolve(),args.checkpoint,args.device)
    if args.figures or args.export_only:
        export(args.output.resolve())
