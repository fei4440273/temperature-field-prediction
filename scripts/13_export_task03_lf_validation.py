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
from sic_cu.data.fields import assert_compatible_fields, load_processed_field
from sic_cu.data.splits import build_power_splits
from sic_cu.eval.energy_v5 import AxisymmetricGeometry, axisymmetric_lumped_nodal_weights
from sic_cu.eval.metrics import observation_time_window_mask, weighted_metrics
from sic_cu.eval.protocol_checks import validate_lf_checkpoint_provenance
from sic_cu.train.simulation import build_model, predict_field, validate_locked_lf_start
from sic_cu.models import ModelScales


def main() -> None:
    parser = argparse.ArgumentParser(description="任-03合法LF验证完整场节点/体积/时间窗明细")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    if destination.exists():
        raise FileExistsError(f"LF validation output already exists: {destination}")
    locked = load_yaml("研究记录/任务03_低保真精度修复/有效运行配置.yaml")
    root = PROJECT_ROOT / "研究记录/任务03_低保真精度修复"
    checkpoints = {
        "原始LF_B0": PROJECT_ROOT / locked["旧LF起点"],
        "原采样接续": root / "任务03_LF控制_种子0_20260915T184117+0800/best.pt",
        "空间平衡接续": root / "任务03_LF空间_种子0_20260915T184423+0800/best.pt",
    }
    fields = [load_processed_field(power) for power in sorted(build_power_splits().simulation_validation)]
    assert len(fields) == 10
    assert_compatible_fields(fields)
    reference = fields[0]
    geometry = AxisymmetricGeometry.from_config(load_yaml("configs/geometry.yaml"))
    nodal_weights = axisymmetric_lumped_nodal_weights(
        reference.coordinates_rz_m, reference.material_ids, geometry,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    source_index = {}
    for arm, checkpoint_path in checkpoints.items():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if arm == "原始LF_B0":
            validate_locked_lf_start(checkpoint_path, seed=0)
        else:
            validate_lf_checkpoint_provenance(payload)
            if payload["provenance"]["test_labels_consumed"] is not False:
                raise ValueError("LF validation cannot use sealed fixed test temperatures")
        source_index[arm] = {
            "路径": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "SHA256": sha256_file(checkpoint_path), "最佳轮次": payload["epoch"],
        }
        model = build_model(
            payload["method"], ModelScales(**payload["scales"]),
            **payload["model_kwargs"],
        ).to(device)
        model.load_state_dict(payload["model_state"], strict=True)
        for field in fields:
            predictions, elapsed = predict_field(model, field.power_w, device)
            target = field.temperature_k.reshape(-1)
            estimated = predictions.reshape(-1)
            time_by_row = np.repeat(field.times_s, len(field.material_ids))
            material_by_row = np.tile(field.material_ids, len(field.times_s))
            volume_by_row = np.tile(nodal_weights, len(field.times_s))
            for material_label, material_id in (("Cu", 0), ("SiC", 1)):
                material_mask = material_by_row == material_id
                for window_name, label in (
                    ("全部", "全时段"), ("time_0_30_s", "[0,30]秒"),
                    ("time_30_100_s", "(30,100]秒"),
                    ("time_100_200_s", "(100,200]秒"),
                ):
                    window_mask = (
                        np.ones(len(time_by_row), dtype=bool) if window_name == "全部"
                        else observation_time_window_mask(time_by_row, window_name)
                    )
                    selected = material_mask & window_mask
                    if not np.any(selected):
                        continue
                    for name, weights in (
                        ("节点等权", np.ones(len(target), dtype=np.float64)),
                        ("轴对称有限元集总真实体积", volume_by_row),
                    ):
                        result = weighted_metrics(
                            target[selected], estimated[selected], weights[selected],
                        )
                        rows.append({
                            "运行臂": arm, "功率_瓦": field.power_w,
                            "材料": material_label, "时间窗": label, "权重口径": name,
                            "RMSE_摄氏度": result["rmse_c"],
                            "MAE_摄氏度": result["mae_c"],
                            "偏差_摄氏度": result["mean_error_c"],
                            "最大绝对误差_摄氏度": result["max_abs_error_c"],
                            "验证点数": int(selected.sum()),
                            "完整场推理秒": elapsed,
                        })
    frame = pl.DataFrame(rows)
    expected = 3 * 10 * 2 * 4 * 2
    if frame.height != expected:
        raise ValueError(f"Unexpected LF validation detail count: {frame.height} != {expected}")
    macro = frame.group_by("运行臂", "材料", "时间窗", "权重口径").agg(
        pl.col("RMSE_摄氏度").mean().alias("逐功率等权宏RMSE_摄氏度"),
        pl.col("RMSE_摄氏度").max().alias("最差功率RMSE_摄氏度"),
        pl.col("功率_瓦").n_unique().alias("验证功率数"),
    ).sort("运行臂", "材料", "时间窗", "权重口径")
    destination.mkdir(parents=True)
    for name, data in (("LF验证逐功率指标.csv", frame), ("LF验证材料时间宏指标.csv", macro)):
        temporary = destination / (name + ".tmp")
        data.write_csv(temporary)
        os.replace(temporary, destination / name)
    summary = {
        "中文说明": "只用10个合法LF验证功率的仿真真实全场；各材料真实集总体积权重复用V5审计方法。",
        "起点与候选检查点": source_index,
        "总明细行数": expected, "验证功率数": 10,
        "时间窗": ["全时段", "[0,30]秒", "(30,100]秒", "(100,200]秒"],
        "物理求积几何来源SHA256": sha256_file(PROJECT_ROOT / "configs/geometry.yaml"),
        "仅LF抽样验证选择的候选": "空间平衡接续",
    }
    temporary = destination / "来源与选取口径.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination / "来源与选取口径.json")
    print(macro.filter(pl.col("时间窗") == "全时段").to_dicts())


if __name__ == "__main__":
    main()
