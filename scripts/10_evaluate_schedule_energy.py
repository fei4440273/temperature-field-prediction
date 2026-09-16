#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.eval.energy_v5 import AxisymmetricGeometry, audit_schedule_energy
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.train.multifidelity import _load_multifidelity_model


def main() -> None:
    parser = argparse.ArgumentParser(description="只读审核任-02最佳或末态的完整能量")
    parser.add_argument("--run", required=True)
    parser.add_argument("--state", choices=("best", "final"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="研究记录/任务02_训练排程/有效运行配置.yaml")
    args = parser.parse_args()
    run = Path(args.run)
    destination = Path(args.output)
    if not run.is_absolute():
        run = PROJECT_ROOT / run
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    if destination.exists():
        raise FileExistsError(f"Physical audit output already exists: {destination}")
    if not (run / "metrics.json").is_file():
        raise FileNotFoundError("Only a completed 300-epoch arm can enter the final audit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_file, _, model = _load_multifidelity_model(run / "best.pt", device)
    chosen_file = best_file
    if args.state == "final":
        chosen_file = run / "阶段_训练末.pt"
        payload = torch.load(chosen_file, map_location=device, weights_only=False)
        if payload["epoch"] != 300:
            raise ValueError("This task requires the actual 300-epoch final state")
        model.load_state_dict(payload["model_state"], strict=True)
    model.requires_grad_(False)
    model.to(dtype=torch.float64).eval()
    snapshot = run / "config_snapshot"
    geometry = AxisymmetricGeometry.from_config(load_yaml(snapshot / "geometry.yaml"))
    materials = load_materials(str(snapshot / "materials.yaml"))
    boundaries = load_resolved_boundary_conditions(str(snapshot / "boundary_conditions.yaml"))
    audit_config = load_yaml(args.config)
    fixed = audit_config.get("最终能量审核", audit_config.get("HF最终工程能量审核"))
    if fixed is None:
        raise ValueError("指定的中文运行配置缺少预登记的30点工程能量审核设定")
    orders = fixed.get("求积阶数", [16, 64])
    if orders != [16, 64]:
        raise ValueError("此审核要求先16阶后64阶对照")
    print("本次只读计算两材料模型场的完整工程能量和V-J-D分解，不加载测试温度。")
    audit = audit_schedule_energy(
        model, materials, boundaries, geometry,
        powers_w=fixed["功率_瓦"], times_s=fixed["时刻_秒"],
        orders=tuple(orders), device=device,
    )
    destination.mkdir(parents=True)
    files = {
        "指标明细.csv": audit["指标明细"],
        "物理分解.csv": audit["物理分解"],
    }
    for name, frame in files.items():
        temporary = destination / (name + ".tmp")
        frame.write_csv(temporary)
        os.replace(temporary, destination / name)
    for name, rows in (
        ("原始能量.jsonl", audit["原始能量"]),
        ("原始散度.jsonl", audit["原始散度"]),
    ):
        temporary = destination / (name + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        os.replace(temporary, destination / name)
    summary = {
        "中文说明": "16/64阶相同30审核点；数值能量失配不等于真实内部温度误差。",
        "运行目录": str(run.relative_to(PROJECT_ROOT)),
        "审核状态": args.state,
        "模型检查点SHA256": sha256_file(chosen_file),
        "最佳检查点SHA256": sha256_file(best_file),
        "审核源码SHA256": sha256_file(PROJECT_ROOT / "src/sic_cu/eval/energy_v5.py"),
        "审核中文配置": args.config,
        "审核配置_SHA256": sha256_file(PROJECT_ROOT / args.config),
        "审核配置功率_瓦": fixed["功率_瓦"],
        "审核配置时刻_秒": fixed["时刻_秒"],
        "审核求积阶数": orders,
        **audit["汇总"],
    }
    temporary = destination / "汇总指标.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination / "汇总指标.json")
    print(json.dumps(audit["汇总"], ensure_ascii=False))


if __name__ == "__main__":
    main()
