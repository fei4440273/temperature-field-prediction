#!/usr/bin/env python
"""Task 10 one-dimensional board F1 or paired F2/F3, never old RZ weights."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import resource
import tarfile
from time import perf_counter

import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.eval.task10_plate_benchmark import REGISTRATION, seal_evaluation_model
from sic_cu.eval.task10_plate_training_source import Task10PlateTrainingSource
from sic_cu.train.task10_plate_methods import (
    audit_training_source_code, fit_f1_method, fit_same_lf_pair_methods,
)


LEDGER = PROJECT_ROOT / "多保真DeepONet预测精度优化总计划与执行台账.md"
METHOD_NAMES = (
    "研究记录/任务10_独立双层场基准/正式人为多热流入场前登记.yaml",
    "研究记录/任务10_独立双层场基准/正式同板F1_F2_F3重训方法预算前登记_v2.yaml",
    "configs/materials.yaml",
    "src/sic_cu/eval/task10_plate_training_source.py",
    "src/sic_cu/models/task10_plate_deeponet.py",
    "src/sic_cu/losses/task10_plate_physics.py",
    "src/sic_cu/train/task10_plate_methods.py",
    "scripts/任务10_同板F1_F2_F3重训.py",
    "tests/test_task10_plate_methods.py",
    "src/sic_cu/eval/task10_plate_group_gate.py",
    "tests/test_task10_plate_group_gate.py",
)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise ValueError("任10正式方法的预算/原件/源码归档必须位于项目目录内")
    return resolved


def _check_tar(path: Path, expected_sha: str, members: tuple[str, ...]) -> None:
    if len(expected_sha) != 64 or sha256_file(path) != expected_sha:
        raise ValueError("任10方法训练的事前源码tar SHA未锁定或已漂移")
    try:
        with tarfile.open(path, "r:gz") as snapshot:
            if not set(members) <= set(snapshot.getnames()):
                raise ValueError("任10事前源码tar缺数值/来源/方法预算所需成员")
            for name in members:
                archived = snapshot.extractfile(name)
                if (archived is None or sha256(archived.read()).hexdigest()
                        != sha256_file(PROJECT_ROOT / name)):
                    raise ValueError(f"任10事前源码成员SHA不同于生效源码：{name}")
    except tarfile.TarError as exc:
        raise ValueError("任10事前源码tar不可解析，不得开始训练") from exc


def _validate_budget(budget: dict, archive: Path, archive_sha: str,
                     old_source: Path, old_source_sha: str) -> None:
    if (budget.get("schema_version") != 1
            or budget.get("阶段") != "任10一维板F1及同LF来源配对F2/F3正式重训前登记"
            or budget.get("训练随机种子") != [0, 1, 2, 3, 4]
            or budget.get("一维数值登记_SHA256") != sha256_file(REGISTRATION)
            or budget.get("已封数值六源tar_SHA256") != old_source_sha
            or budget.get("已封来源入口两源tar_SHA256")
               != sha256_file(_project_path(budget["已封来源入口两源tar"]))
            or budget.get("受限源清单_SHA256")
               != sha256_file(_project_path(budget["正式九档受限源"]) / "探针与源场SHA清单.json")
            or budget.get("受限源逐热流控制CSV_SHA256")
               != sha256_file(_project_path(budget["正式九档受限源"]) / "九热流数值门禁原始明细.csv")
            or budget.get("方法十一源tar成员") != list(METHOD_NAMES)
            or budget.get("同LF成对PDE对照") is not True
            or budget.get("F2体内PDE计算") is not False
            or budget.get("F1或F3体内PDE计算") is not True
            or budget.get("允许LF来源")
               != ["lf_same_physics_coarse", "lf_contact_mismatch_coarse"]
            or budget.get("训练来源仅五折探针且验证仅两折探针") is not True
            or budget.get("短先导不可修订正式预算") is not True
            or budget.get("完整HF后验仅全部25模型冻结后") is not True
            or budget.get("单模型SHA锁不足以提前开启完整HF后验") is not True
            or budget.get("能源及内部场审核延后至全部25模型及合法验证锁定") is not True
            or budget.get("机器优化器与四类随机态归档") is not True):
        raise ValueError("任10方法预算、机器受限源或数值物理来源不符事前登记")
    if (old_source != _project_path(budget["已封数值六源tar"])
            or archive_sha != sha256_file(archive)):
        raise ValueError("方法tar或独立数值六源tar未与事前预算绑定")
    registration = load_yaml(REGISTRATION)
    manifest = json.loads((_project_path(budget["正式九档受限源"]) /
                           "探针与源场SHA清单.json").read_text(encoding="utf-8"))
    controls = manifest.get("逐热流数值控制", {})
    expected_fluxes = {str(flux) for group in registration["flux_splits_w_m2"].values()
                       for flux in group}
    allowed_grids = [[item["silicon_carbide_cells"], item["copper_cells"],
                      item["time_step_s"]] for item in registration["reference_controls"]["mesh_time_pairs"]]
    threshold = float(registration["reference_controls"]["per_flux_max_adjacent_difference_c"])
    if (manifest.get("登记SHA256") != budget["一维数值登记_SHA256"]
            or manifest.get("事前六源tar_SHA256") != old_source_sha
            or set(controls) != expected_fluxes
            or any(item["细网格"] not in allowed_grids
                   or not math.isfinite(item["相邻差_摄氏度"])
                   or not 0 <= item["相邻差_摄氏度"] < threshold
                   or not math.isfinite(item["最大单步平衡_瓦每平方米"])
                   or item["最大单步平衡_瓦每平方米"] < 0
                   for item in controls.values())):
        raise ValueError("九热流独立数值门禁或每功率能量平衡原始凭据不符合事前登记")
    expected = budget.get("方法生效源码_SHA256", {})
    if (set(expected) != set(METHOD_NAMES) - {
            "研究记录/任务10_独立双层场基准/正式同板F1_F2_F3重训方法预算前登记_v2.yaml"}
            or any(sha256_file(PROJECT_ROOT / name) != digest for name, digest in expected.items())):
        raise ValueError("正式方法核心/脚本/配置/回归源码与训练前预算字节SHA不符")
    settings = budget.get("共同训练预算", {})
    if (settings.get("网络") != {"width": 32, "latent_dim": 32, "blocks": 1}
            or settings.get("HF每臂最大真实优化步") != 800
            or settings.get("共享LF每同seed每来源最大真实优化步") != 300
            or settings.get("合法验证间隔_步") != 20
            or settings.get("LF单步采样上限_点") != 512
            or settings.get("每材料体内配点数") != 8
            or settings.get("AdamW学习率") != .001
            or settings.get("AdamW权重衰减") != .000001
            or settings.get("正式优化步合计上限") != 23000
            or settings.get("预登记模型身份总数") != 25
            or settings.get("已吸收顶部热流单位") != "W/m²"):
        raise ValueError("实际网络/种子/优化/配点预算不可在测试来源或结果后改写")
    if archive_sha not in LEDGER.read_text(encoding="utf-8"):
        raise ValueError("正式方法源码tar SHA尚未实际进入唯一总台账事前行，禁止训练")


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("没有真实优化逐行记录，禁止生成方法成绩")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_fit(output: Path, fit, arm: str, seed: int, source_root: Path,
              budget_sha: str, snapshot_sha: str, lf_source: str | None) -> dict:
    folder = output / arm
    folder.mkdir(exist_ok=False)
    checkpoint = folder / "合法验证选定板模型_state_dict.pt"
    torch.save(fit.best_state, checkpoint)
    evidence = {
        "最佳HF模型状态": ("最佳HF模型_state_dict.pt", fit.best_state),
        "最佳HF_AdamW机器状态": ("最佳HF_AdamW机器状态.pt", fit.best_optimizer_state),
        "终态HF_AdamW机器状态": ("终态HF_AdamW机器状态.pt", fit.final_optimizer_state),
        "终态HF模型状态": ("终态HF模型_state_dict.pt", fit.final_model_state),
        "最佳HF四类随机态": ("最佳HF四类随机态.pt", fit.best_rng_state),
        "终态HF四类随机态": ("终态HF四类随机态.pt", fit.final_rng_state),
    }
    if arm in ("F2", "F3"):
        evidence["共享LF终态_AdamW机器状态"] = (
            "共享LF终态_AdamW机器状态.pt", fit.lf_optimizer_state)
        evidence["共享LF四类采样随机态"] = (
            "共享LF四类采样随机态.pt", fit.lf_rng_state)
        if fit.lf_optimizer_state is None or fit.lf_rng_state is None:
            raise ValueError("同LF配对两方法必须记录共享LF真实优化器及四类随机态")
    machine_shas = {}
    for name, (filename, values) in evidence.items():
        machine_path = folder / filename
        torch.save(values, machine_path)
        machine_shas[name + "_SHA256"] = sha256_file(machine_path)
    train = dict(fit.training)
    hf_rows = train.pop("逐步HF真实优化明细")
    hf_csv = folder / "逐步HF实际优化原始明细.csv"
    _write_rows(hf_csv, hf_rows)
    if arm in ("F2", "F3"):
        lf_rows = train.pop("共享LF逐步真实优化明细")
        lf_csv = folder / "同seed单源LF预训真实优化明细.csv"
        _write_rows(lf_csv, lf_rows)
        train["共享LF逐步CSV_SHA256"] = sha256_file(lf_csv)
        train["LF终态可更新参数实测AdamW步数"] = train["同seed共享LF真实优化步数"]
        train["LF可更新参数状态数"] = len(fit.lf_optimizer_state["state"])
    train.update({
        "登记SHA256": sha256_file(REGISTRATION), "模型_SHA256": sha256_file(checkpoint),
        "方法预算_SHA256": budget_sha, "方法源码tar_SHA256": snapshot_sha,
        "正式受限数值源清单_SHA256": sha256_file(source_root / "探针与源场SHA清单.json"),
        "随机种子": seed, "LF来源": lf_source, "HF逐步CSV_SHA256": sha256_file(hf_csv),
        "设备": next(fit.model.parameters()).device.type,
        "完整HF内部真值进入训练": False,
        "HF最佳可更新参数实测AdamW步数": fit.validation.get("最佳步数"),
        "HF终态可更新参数实测AdamW步数": train["实际优化步数"],
        "HF可更新参数状态数": len(fit.final_optimizer_state["state"]),
        **machine_shas,
    })
    validation = dict(fit.validation)
    validation.update({
        "登记SHA256": sha256_file(REGISTRATION), "模型_SHA256": sha256_file(checkpoint),
        "合法验证温度只来自三探针": True,
    })
    train_log = folder / "真实训练日志.json"
    val_report = folder / "合法探针验证.json"
    train_log.write_text(json.dumps(train, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    val_report.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lock_file = folder / "合法训练验证后模型SHA锁.json"
    seal_evaluation_model(REGISTRATION, source_root, checkpoint, train_log, val_report, lock_file)
    return {"方法": arm, "随机种子": seed, "LF来源": lf_source,
            "实际HF优化步数": train["实际优化步数"],
            "共享LF真实优化步数": train.get("同seed共享LF真实优化步数", 0),
            "体内PDE实际调用": train["体内PDE实际调用"],
            "合法两档三探针RMSE_K": validation["两档探针RMSE_K"],
            "选定模型_SHA256": sha256_file(checkpoint),
            "真实训练日志_SHA256": sha256_file(train_log),
            "合法探针验证_SHA256": sha256_file(val_report),
            "真实HF逐步CSV_SHA256": sha256_file(hf_csv),
            "已锁模型_SHA256文件_SHA256": sha256_file(lock_file)}


def main() -> None:
    parser = argparse.ArgumentParser(description="任10同一维板F1或同LF配对F2/F3受限方法重训")
    parser.add_argument("--arm", choices=["F1", "PAIR"], required=True)
    parser.add_argument("--lf-source", choices=["lf_same_physics_coarse",
                                                 "lf_contact_mismatch_coarse"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True, choices=["cpu", "cuda"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", required=True)
    parser.add_argument("--budget-sha256", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--methods-archive", required=True)
    parser.add_argument("--methods-sha256", required=True)
    options = parser.parse_args()
    output = _project_path(options.output)
    if output.exists():
        raise FileExistsError(f"任10方法训练目录绝不覆盖：{output}")
    budget_path = _project_path(options.budget)
    if len(options.budget_sha256) != 64 or sha256_file(budget_path) != options.budget_sha256:
        raise ValueError("任10正式方法预算YAML SHA与事前冻结值不符")
    source_archive = _project_path(options.source_archive)
    snapshot = _project_path(options.methods_archive)
    if options.arm == "F1" and options.lf_source is not None:
        raise ValueError("F1 HF-only禁止LF预训或复用任一LF来源")
    if options.arm == "PAIR" and options.lf_source is None:
        raise ValueError("F2/F3单PDE配对必须固定同一已登记LF来源")
    budget = load_yaml(budget_path)
    if options.seed not in budget.get("训练随机种子", []):
        raise ValueError("仅允许训练前冻结的五个随机种子")
    _check_tar(source_archive, options.source_sha256, (
        "研究记录/任务10_独立双层场基准/正式人为多热流入场前登记.yaml",
        "src/sic_cu/eval/task10_plate_benchmark.py",
        "scripts/任务10_检查人为多热流入场.py",
        "scripts/任务10_生成多热流受限数值源.py",
        "tests/test_task10_plate_benchmark.py", "configs/materials.yaml",
    ))
    input_tar = _project_path(budget["已封来源入口两源tar"])
    _check_tar(input_tar, budget["已封来源入口两源tar_SHA256"], (
        "src/sic_cu/eval/task10_plate_training_source.py",
        "tests/test_task10_plate_training_source.py",
    ))
    _check_tar(snapshot, options.methods_sha256, METHOD_NAMES)
    _validate_budget(budget, snapshot, options.methods_sha256,
                     source_archive, options.source_sha256)
    audit_training_source_code(PROJECT_ROOT / "scripts/任务10_同板F1_F2_F3重训.py",
                               expected_sha256=budget["方法生效源码_SHA256"][
                                   "scripts/任务10_同板F1_F2_F3重训.py"])
    source_root = _project_path(budget["正式九档受限源"])
    if options.arm == "F1":
        f1 = Task10PlateTrainingSource(REGISTRATION, source_root, "F1",
                                       manifest_sha256=budget["受限源清单_SHA256"])
    else:
        f2 = Task10PlateTrainingSource(REGISTRATION, source_root, "F2", options.lf_source,
                                       manifest_sha256=budget["受限源清单_SHA256"])
        f3 = Task10PlateTrainingSource(REGISTRATION, source_root, "F3", options.lf_source,
                                       manifest_sha256=budget["受限源清单_SHA256"])
    device = torch.device(options.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("指定CUDA不可用，禁止把CPU短诊断伪装正式GPU重训")
    output.mkdir(parents=True, exist_ok=False)
    (output / "开始训练源码与受限源凭据.json").write_text(json.dumps({
        "事前预算_SHA256": options.budget_sha256,
        "数值六源tar_SHA256": options.source_sha256,
        "方法源码tar_SHA256": options.methods_sha256,
        "受限源清单_SHA256": budget["受限源清单_SHA256"],
        "运行方法": options.arm, "单LF来源": options.lf_source,
        "随机种子": options.seed, "设备": options.device,
        "起点时刻": datetime.now().astimezone().isoformat(),
        "完整内部场后验评价已执行": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    begin = perf_counter()
    settings = budget["共同训练预算"]
    kwargs = {
        "seed": options.seed,
        "hf_steps": settings["HF每臂最大真实优化步"],
        "validation_every": settings["合法验证间隔_步"],
        "collocation_points": settings["每材料体内配点数"],
        "learning_rate": settings["AdamW学习率"],
        "device": device,
        **settings["网络"],
    }
    if options.arm == "F1":
        fitted = [("F1", fit_f1_method(f1, **kwargs))]
    else:
        fitted2, fitted3 = fit_same_lf_pair_methods(
            f2, f3, lf_steps=settings["共享LF每同seed每来源最大真实优化步"],
            lf_batch_size=settings["LF单步采样上限_点"], **kwargs)
        fitted = [("F2", fitted2), ("F3", fitted3)]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = perf_counter() - begin
    entries = [_save_fit(output, fit, arm, options.seed, source_root,
                         options.budget_sha256, options.methods_sha256,
                         options.lf_source) for arm, fit in fitted]
    summary = {
        "阶段": "同板新模型受限训练及合法探针验证，无内部HF评价",
        "方法成绩仅合法三探针": entries,
        "方法预算_SHA256": options.budget_sha256,
        "方法源码tar_SHA256": options.methods_sha256,
        "真正训练耗时_秒": wall_seconds,
        "进程峰值RSS_kB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "CUDA峰值分配MiB": (torch.cuda.max_memory_allocated(device) / 2**20
                         if device.type == "cuda" else None),
        "源场清单_SHA256": budget["受限源清单_SHA256"],
        "完整内部场评价已计算": False,
    }
    (output / "合法模型训练与资源运行摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "同板方法训练入场受限记录.md").write_text(
        "# 任10同板方法：仅合法训练/验证探针\n\n"
        f"数值来源清单SHA256 `{budget['受限源清单_SHA256']}`；正式方法源码tar SHA256 "
        f"`{options.methods_sha256}`；方法预算YAML SHA256 `{options.budget_sha256}`。"
        "F1无LF且使用公开常物性一维PDE；F2/F3在同seed同一种LF来源起点逐位一致，"
        "只F3计算体内二阶PDE，初值、顶底边界及界面仍是共同条件。"
        "本目录仅允许30/70千W/m²三个虚拟探针选模型；尚未读取数值隐藏42/58千温度，"
        "没有未观测内部场模型误差、真实装置几何泛化或工程B0物理可信结论。"
        "同LF配对的PDE差异与Rc0.7 LF失配实验必须分别分析，绝不混求贡献。"
        "源码/API门禁不构成操作系统对任意恶意Python脚本的读权限隔离。\n",
        encoding="utf-8",
    )
    print(json.dumps({"阶段": summary["阶段"], "受限训练": options.arm,
                      "两档合法三探针选分与原件SHA": entries,
                      "内部HF评价": False,
                      "运行摘要": str(output / "合法模型训练与资源运行摘要.json")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
