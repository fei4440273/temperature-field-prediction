#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.processed import load_processed_ir_observations
from sic_cu.eval.metrics import configured_radial_mask, weighted_metrics
from sic_cu.train.multifidelity import _ir_dataset, _load_multifidelity_model, _predict_batches


def main() -> None:
    parser = argparse.ArgumentParser(description="任-03合法HF验证顶部径向三区与绝对误差95分位")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = PROJECT_ROOT / args.output
    if destination.exists():
        raise FileExistsError(f"Top-region report already exists: {destination}")
    root = PROJECT_ROOT / "研究记录/任务03_低保真精度修复"
    runs = {
        "旧LF同源": root / "任务03_HF旧LF同源_种子0_20260915T184956+0800",
        "空间LF同源": root / "任务03_HF新LF同源_种子0_20260915T185317+0800",
        "拟合LF同源": root / "任务03_HF拟合LF同源_种子0_20260915T185928+0800",
    }
    radial = load_yaml("configs/optimization_v4.yaml")["diagnostics"]["radial_windows_mm"]
    windows = {
        "center_0_8_mm": "中心[0,8]毫米",
        "middle_8_17_mm": "中部(8,17]毫米",
        "outer_17_25_mm": "外圈(17,25]毫米",
    }
    frame = load_processed_ir_observations("validation")
    coordinates, target, weights = _ir_dataset("validation").tensors
    if frame.height != len(coordinates):
        raise ValueError("HF top radial observations and the training dataset must align")
    radii = frame["r_m"].to_numpy().astype(np.float64) * 1000.0
    powers = frame["power_w"].to_numpy()
    targets = target.numpy().reshape(-1)
    radial_weights = weights.numpy().reshape(-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regions, tails, sources = [], [], {}
    for label, run in runs.items():
        if not (run / "metrics.json").is_file():
            raise FileNotFoundError(f"Only a completed HF validation arm can enter this report: {run}")
        checkpoint, _, model = _load_multifidelity_model(run / "best.pt", device)
        estimated = _predict_batches(model, coordinates, device)
        sources[label] = {"路径": str(checkpoint.relative_to(PROJECT_ROOT)), "SHA256": sha256_file(checkpoint)}
        for power in np.unique(powers):
            power_mask = np.isclose(powers, power, atol=1e-4)
            abs_error = np.abs(estimated[power_mask] - targets[power_mask])
            tails.append({
                "运行臂": label, "功率_瓦": round(float(power), 4),
                "顶部绝对误差95分位_摄氏度": float(np.percentile(abs_error, 95)),
                "最大绝对误差_摄氏度": float(abs_error.max()),
                "合法顶面观测数": int(power_mask.sum()),
                "统计权重": "原径向观测点等权分位数；不与面积/径向RMSE混淆",
            })
            for definition in radial:
                region = definition["name"]
                selected = power_mask & configured_radial_mask(radii, definition)
                result = weighted_metrics(
                    targets[selected], estimated[selected], radial_weights[selected],
                ) if selected.any() else None
                regions.append({
                    "运行臂": label, "功率_瓦": round(float(power), 4),
                    "径向区域": windows[region],
                    "RMSE_摄氏度": None if result is None else result["rmse_c"],
                    "MAE_摄氏度": None if result is None else result["mae_c"],
                    "偏差_摄氏度": None if result is None else result["mean_error_c"],
                    "合法顶面观测数": int(selected.sum()),
                    "数据状态": "无数据" if result is None else "实际观测",
                    "指标权重": "原HF顶部径向观测权重",
                })
    region_frame = pl.DataFrame(regions)
    tail_frame = pl.DataFrame(tails)
    if region_frame.height != 27 or tail_frame.height != 9:
        raise ValueError("HF validation region and percentile rows do not match the fixed three powers")
    macros = region_frame.group_by("运行臂", "径向区域").agg(
        pl.col("RMSE_摄氏度").mean().alias("逐功率等权径向宏RMSE_摄氏度"),
        pl.col("RMSE_摄氏度").max().alias("最差功率径向RMSE_摄氏度"),
        pl.col("功率_瓦").n_unique().alias("验证功率数"),
    )
    destination.mkdir(parents=True)
    for name, data in (
        ("顶部径向逐功率.csv", region_frame),
        ("顶部径向宏指标.csv", macros),
        ("顶部绝对误差95分位逐功率.csv", tail_frame),
    ):
        temporary = destination / (name + ".tmp")
        data.write_csv(temporary)
        os.replace(temporary, destination / name)
    report = {
        "中文说明": "仅三种HF合法验证功率；径向RMSE保留旧HF权重，95分位为原径向点的等权绝对误差。",
        "模型来源": sources,
        "径向窗配置_SHA256": sha256_file(PROJECT_ROOT / "configs/optimization_v4.yaml"),
        "合法HF顶部处理数据_SHA256": sha256_file(PROJECT_ROOT / "data/processed/experiment_ir_radial.parquet"),
        "宏径向指标": macros.to_dicts(),
        "95分位逐功率": tails,
    }
    temporary = destination / "径向与长尾核对.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination / "径向与长尾核对.json")
    print(json.dumps(macros.to_dicts(), ensure_ascii=False))


if __name__ == "__main__":
    main()
