#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.eval.guardrails import (
    task03_hf_guardrails, task03_volume_guardrails, validate_task03_hf_budget,
)


ROOT = PROJECT_ROOT / "研究记录/任务03_低保真精度修复"
LF_VALIDATION = ROOT / "任务03_LF完整验证_20260915T184950+0800"
HF_RUNS = {
    "旧LF同源": "任务03_HF旧LF同源_种子0_20260915T184956+0800",
    "空间LF同源": "任务03_HF新LF同源_种子0_20260915T185317+0800",
    "拟合LF同源": "任务03_HF拟合LF同源_种子0_20260915T185928+0800",
}
ENERGY_RUNS = {
    "旧LF同源": "任务03_HF旧LF最佳_能量审核_20260915T190937+0800",
    "空间LF同源": "任务03_HF空间LF最佳_能量审核_20260915T190425+0800",
    "拟合LF同源": "任务03_HF拟合LF最佳_能量审核_20260915T190224+0800",
}


def _lf_material_metrics(frame: pl.DataFrame, arm: str) -> dict[str, dict[str, float]]:
    rows = frame.filter((pl.col("运行臂") == arm) & (pl.col("时间窗") == "全时段"))
    if rows.height != 4 or rows["验证功率数"].to_list() != [10] * 4:
        raise ValueError("任-03 LF验证材料/权重行不完整")
    values: dict[str, dict[str, float]] = {"Cu": {}, "SiC": {}}
    for row in rows.to_dicts():
        metric = "node" if row["权重口径"] == "节点等权" else "volume"
        values[row["材料"]][metric] = row["逐功率等权宏RMSE_摄氏度"]
    if any(set(values[material]) != {"node", "volume"} for material in values):
        raise ValueError("任-03 LF材料完整场缺少节点或真实体积指标")
    return values


def _hf_modalities(frame: pl.DataFrame, arm: str) -> dict[str, float]:
    rows = frame.filter(
        (pl.col("运行臂") == arm) & (pl.col("指标") == "绝对温度")
        & (pl.col("时间窗") == "全部合法观测")
    )
    names = {"顶部": "top", "热端": "hot", "冷端": "cold"}
    values = {}
    for label, name in names.items():
        selected = rows.filter(pl.col("模态") == label)
        if selected.height != 3 or selected["RMSE_摄氏度"].null_count():
            raise ValueError(f"任-03 HF {arm}/{label}缺少三个验证功率的温度记录")
        values[name] = float(selected["RMSE_摄氏度"].mean())
    return values


def main() -> None:
    destination = ROOT / "验收判定_预算门禁复核.json"
    if destination.exists():
        raise FileExistsError(f"任-03机器验收判定已存在，不覆盖：{destination}")
    lf_path = LF_VALIDATION / "LF验证材料时间宏指标.csv"
    hf_path = ROOT / "任务03_HF三臂观测复核_20260915T190745+0800/观测指标明细.csv"
    lf_frame = pl.read_csv(lf_path)
    hf_frame = pl.read_csv(hf_path)
    if hf_frame.height != 180:
        raise ValueError("任-03 HF三臂合法验证窗口行数须为180")
    locked = load_yaml("研究记录/任务03_低保真精度修复/有效运行配置.yaml")
    lf = {
        arm: _lf_material_metrics(lf_frame, arm)
        for arm in ("原始LF_B0", "原采样接续", "空间平衡接续")
    }
    lf_guard = task03_volume_guardrails(lf["原采样接续"], lf["空间平衡接续"])
    registered = json.loads((LF_VALIDATION / "来源与选取口径.json").read_text(encoding="utf-8"))
    for label, recorded in registered["起点与候选检查点"].items():
        if sha256_file(PROJECT_ROOT / recorded["路径"]) != recorded["SHA256"]:
            raise ValueError(f"任-03 LF {label}先于HF注册的检查点SHA已变更")
    hf: dict[str, dict] = {}
    for arm in HF_RUNS:
        run_path = ROOT / HF_RUNS[arm]
        metric_path = run_path / "metrics.json"
        audit_path = ROOT / ENERGY_RUNS[arm] / "汇总指标.json"
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        training_path = run_path / "training.jsonl"
        training_rows = [json.loads(line) for line in training_path.read_text(
            encoding="utf-8"
        ).splitlines()]
        consumption = validate_task03_hf_budget(metrics, training_rows, locked)
        best_path = run_path / "best.pt"
        best_hash = sha256_file(best_path)
        if (
            metrics["seed"] != 0 or metrics["epochs_completed"] != 300
            or metrics["test_ir"] is not None or metrics["test_sensor"] is not None
            or audit["审核状态"] != "best"
            or audit["模型检查点SHA256"] != best_hash
            or audit["功率时刻审核行数"] != 30
            or audit["审核求积阶数"] != [16, 64]
        ):
            raise ValueError(f"任-03 HF {arm}训练/合法验证/能量来源不一致")
        hf[arm] = {
            "最佳模型检查点": str(best_path.relative_to(PROJECT_ROOT)),
            "最佳模型SHA256": best_hash,
            "训练指标SHA256": sha256_file(metric_path),
            "逐轮训练日志SHA256": sha256_file(training_path),
            "同预算实际训练消耗": consumption,
            "能量审核SHA256": sha256_file(audit_path),
            "LF检查点SHA256": metrics["lf_checkpoint_sha256"],
            "最佳轮次": metrics["best_epoch"],
            "已执行轮次": metrics["epochs_completed"],
            "训练秒": metrics["training_seconds"],
            "选择分数_摄氏度": metrics["best_validation_selection_score_c"],
            "顶部热端冷端逐功率宏RMSE_摄氏度": _hf_modalities(hf_frame, arm),
            "绝对能量宏均值_瓦": audit["绝对平衡宏均值_瓦"],
            "绝对能量95分位_瓦": audit["绝对平衡95分位_瓦"],
            "最大求积相邻阶变化对吸收功率比": audit["最大相邻阶变化对吸收功率比"],
        }
    expected_lf_hash = {
        "旧LF同源": registered["起点与候选检查点"]["原始LF_B0"]["SHA256"],
        "空间LF同源": registered["起点与候选检查点"]["空间平衡接续"]["SHA256"],
        "拟合LF同源": registered["起点与候选检查点"]["原采样接续"]["SHA256"],
    }
    if any(hf[arm]["LF检查点SHA256"] != expected_lf_hash[arm] for arm in hf):
        raise ValueError("任-03 HF三臂LF检查点来源不是先验登记版本")
    if len({
        tuple(hf[arm]["同预算实际训练消耗"].items()) for arm in hf
    }) != 1:
        raise ValueError("任-03 HF三臂训练消耗不相等，不可判定同预算收益")
    old = hf["旧LF同源"]
    combination = {}
    for arm in ("空间LF同源", "拟合LF同源"):
        new = hf[arm]
        combination[arm] = task03_hf_guardrails(
            old_score=old["选择分数_摄氏度"], new_score=new["选择分数_摄氏度"],
            old_modalities=old["顶部热端冷端逐功率宏RMSE_摄氏度"],
            new_modalities=new["顶部热端冷端逐功率宏RMSE_摄氏度"],
            old_energy={"mean_w": old["绝对能量宏均值_瓦"],
                        "p95_w": old["绝对能量95分位_瓦"]},
            new_energy={"mean_w": new["绝对能量宏均值_瓦"],
                        "p95_w": new["绝对能量95分位_瓦"]},
        )
    history = load_yaml("reports/development_v4/baseline_manifest.yaml")
    historical_seed0 = next(entry for entry in history["checkpoints"] if entry["seed"] == 0)
    deployment_reference = historical_seed0["validation_reproduction"]["reproduced_macro_v1"]
    old_hf_root = ROOT / HF_RUNS["旧LF同源"]
    payload = {
        "结论": "任-03已验收／LF拟合收益明确但空间机制及新LF-HF替换均未达采用门槛",
        "HF标签用途": "仅使用固定12训练与3合法验证；无旧test_Data温度或独立外部实测作为开发依据",
        "标准差": "只执行seed0同源先导，不报告五种子标准差，任-07负责正式重复性",
        "HF预算门禁": "仅将同训练起点、每臂完成300轮、4500次观测步、300次256点物理步且功率划分相同的三臂纳入采用判定；较短诊断不可用于此判定",
        "LF完整场": lf,
        "LF空间机制预设体积护栏": lf_guard,
        "HF三臂": hf,
        "新LF-HF组合采用护栏": combination,
        "LF选择": "原始LF_B0（仅HF组合/任-04阶段对照）；原采样拟合LF作为合法LF收益证据保留，空间采样机制回退",
        "任04共同阶段起点": {
            "检查点": str((old_hf_root / "阶段_校正末.pt").relative_to(PROJECT_ROOT)),
            "SHA256": sha256_file(old_hf_root / "阶段_校正末.pt"),
            "只作两臂同起点": True,
        },
        "历史部署B0参考": {
            "检查点": historical_seed0["hf_checkpoint"],
            "SHA256": historical_seed0["hf_checkpoint_sha256"],
            "原始合法验证macro_v1_摄氏度": deployment_reference,
            "不得被本次更差的300轮续训模型替换": (
                old["选择分数_摄氏度"] > deployment_reference
            ),
        },
        "输入SHA256": {
            str(lf_path.relative_to(PROJECT_ROOT)): sha256_file(lf_path),
            str(hf_path.relative_to(PROJECT_ROOT)): sha256_file(hf_path),
            str((LF_VALIDATION / "来源与选取口径.json").relative_to(PROJECT_ROOT)): sha256_file(
                LF_VALIDATION / "来源与选取口径.json"
            ),
        },
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"结论": payload["结论"], "LF空间机制": lf_guard,
                      "组合采用": combination}, ensure_ascii=False))


if __name__ == "__main__":
    main()
