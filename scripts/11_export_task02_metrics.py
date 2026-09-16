#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.eval.metrics import chinese_schedule_observation_rows
from sic_cu.train.multifidelity import (
    _evaluate_ir_model, _evaluate_sensor_model, _load_multifidelity_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="合法HF验证的中文多臂分窗重算")
    parser.add_argument("--s0")
    parser.add_argument("--s1")
    parser.add_argument("--arm", action="append", help="可重复的中文标签=运行目录，替代--s0/--s1")
    parser.add_argument("--task", choices=("任-02", "任-03"), default="任-02")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.arm:
        if args.s0 or args.s1:
            parser.error("--arm不能与--s0/--s1混用")
        arms = []
        for item in args.arm:
            label, separator, path = item.partition("=")
            if not separator or not label or not path:
                parser.error("--arm须为中文标签=运行目录")
            arms.append((label, path))
        if len({label for label, _ in arms}) != len(arms):
            parser.error("报告中的运行臂标签不能重复")
    elif args.s0 and args.s1:
        arms = [("S0", args.s0), ("S1", args.s1)]
    else:
        parser.error("指定成对--s0/--s1，或至少一条--arm中文标签=目录")
    destination = Path(args.output)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    if destination.exists():
        raise FileExistsError("Validation report cannot overwrite an earlier artifact")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "中文说明": "同口径复核30/100秒端点，只从合法HF验证温度重算；训练选择分数和模型状态不变。",
        "来源任务": args.task, "候选": {},
    }
    frames = []
    raw = {}
    print(f"本次只复核{len(arms)}臂高保真合法验证观测的窗口口径，不读取旧测试温度。")
    for arm, path in arms:
        run = Path(path)
        if not run.is_absolute():
            run = PROJECT_ROOT / run
        original = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        checkpoint_file, _, model = _load_multifidelity_model(run / "best.pt", device)
        revised_ir = _evaluate_ir_model(model, "validation", device)
        revised_sensor = _evaluate_sensor_model(model, "validation", device)
        if abs(original["validation_ir"]["rmse_c"] - revised_ir["rmse_c"]) > 1e-4:
            raise RuntimeError("Full-period IR RMSE changed; reevaluate comparable baseline before reporting")
        if abs(original["validation_sensor"]["absolute_rmse_c"] - revised_sensor["absolute_rmse_c"]) > 1e-4:
            raise RuntimeError("Full-period sensor RMSE changed while only time windows were corrected")
        frames.append(chinese_schedule_observation_rows(
            arm, original["validation_three_power_comparison"], revised_ir, revised_sensor,
        ))
        raw[arm] = {"顶部验证": revised_ir, "环温验证": revised_sensor}
        report["候选"][arm] = {
            "模型检查点SHA256": sha256_file(checkpoint_file),
            "旧顶部完整时段RMSE_摄氏度": original["validation_ir"]["rmse_c"],
            "新顶部完整时段RMSE_摄氏度": revised_ir["rmse_c"],
            "旧环温完整时段RMSE_摄氏度": original["validation_sensor"]["absolute_rmse_c"],
            "新环温完整时段RMSE_摄氏度": revised_sensor["absolute_rmse_c"],
            "原检查点选分_摄氏度": original["best_validation_selection_score_c"],
        }
    import polars as pl

    table = pl.concat(frames)
    destination.mkdir(parents=True)
    csv_tmp = destination / "观测指标明细.csv.tmp"
    table.write_csv(csv_tmp)
    os.replace(csv_tmp, destination / "观测指标明细.csv")
    raw_tmp = destination / "验证窗口原始重算.json.tmp"
    raw_tmp.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(raw_tmp, destination / "验证窗口原始重算.json")
    report["报告行数"] = table.height
    report["指标CSV_SHA256"] = sha256_file(destination / "观测指标明细.csv")
    summary_tmp = destination / "窗口复核.json.tmp"
    summary_tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(summary_tmp, destination / "窗口复核.json")
    print(json.dumps({"报告行数": table.height, "原完整时段分数未变": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
