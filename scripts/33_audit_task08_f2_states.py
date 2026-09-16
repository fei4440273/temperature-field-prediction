#!/usr/bin/env python
"""任08 F2五种子完整状态、合法HF观测和全项工程能源独立审计。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.processed import load_processed_ir_observations
from sic_cu.eval.development_v4 import (
    _observation_arrays, _predict, _top_coordinates, error_statistics,
    evaluate_hf_observations,
)
from sic_cu.eval.energy_v5 import AxisymmetricGeometry, audit_schedule_energy
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.train.common import CONFIG_FILES
from sic_cu.train.multifidelity import _ir_dataset, _sensor_tensors
from sic_cu.train.task04_joint import (
    _lf_material_validation, _validation_selection, task04_lf_keep_guardrail,
)
from sic_cu.train.task07_formal import _equal, _score_guardrails, _snapshot_hashes
from sic_cu.train.task07_source import validate_task07_sources
from sic_cu.train.task08_f2_formal import (
    CORRECTION_STAGE, JOINT_STAGE, TASK08_F2_REGISTRATION,
    _check_model, _check_snapshot_identity, _committed_best, _lf_sha,
    _physics_pair, _project_output, _qualification, _require_formal_registration,
    fork_f2_e0,
)


TASK04_CONFIG = PROJECT_ROOT / "研究记录/任务04_联合微调/有效运行配置.yaml"
TASK04_CONFIG_SHA256 = "b464eddd752000694a513da0e4a1b4b0f7447e31b85d19726e243396a0864710"
TASK06_CONFIG = PROJECT_ROOT / "研究记录/任务06_时间响应特征/HF三臂先导有效配置.yaml"
TASK06_CONFIG_SHA256 = "7a418d94493ddafd9c78e5070bca0739ab2de30aecdc22ef322ef69bc7e03b19"
OPTIMIZATION_SHA256 = "59c1e7f569ba846effc5b3fd7f1776d55f4c2341b25f012809e60b4c6b913334"
AUDIT04_SCRIPT = PROJECT_ROOT / "scripts/21_audit_task04_energy.py"


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _old_auditor():
    spec = importlib.util.spec_from_file_location("task08_f2_energy_writer", AUDIT04_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError("事前六功率工程能源原瓦数输出源码缺失")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_schedule() -> tuple[list[float], list[float], list[int]]:
    if (sha256_file(TASK04_CONFIG) != TASK04_CONFIG_SHA256 or
        sha256_file(TASK06_CONFIG) != TASK06_CONFIG_SHA256):
        raise ValueError("任08 F2沿用的任04/06能源预登记SHA已漂移")
    old = _old_auditor()
    powers, times, orders = old.FIXED_POWERS, old.FIXED_TIMES, [16, 64]
    task04 = load_yaml(TASK04_CONFIG)["独立能量审核"]
    task06 = load_yaml(TASK06_CONFIG)["独立30点能量"]
    if (task04["功率_瓦"] != powers or task04["时刻_秒"] != times or
        task04["求积阶数"] != orders or task04["功率时刻点数"] != 30 or
        task06["功率_瓦"] != powers or task06["时刻_秒"] != times or
        task06["阶数"] != orders or len(powers) * len(times) != 30):
        raise ValueError("F2须复用F3同一30功率时刻点及16/64阶，不得另定能量分母")
    return powers, times, orders


def verify_physics_distinction(row: Mapping[str, Any]) -> None:
    nominal, independent = (row.get("名义物理训练分项"),
                            row.get("独立局部物理损失"))
    if (not isinstance(nominal, dict) or nominal.get("pde", "非法") is not None or
        any(not isinstance(nominal.get(name), (int, float)) or
            not math.isfinite(nominal[name]) for name in
            ("initial", "boundary", "interface", "physics_total")) or
        independent is not None and (
            not isinstance(independent, dict) or
            any(not isinstance(independent.get(name), (int, float)) or
                not math.isfinite(independent[name]) for name in
                ("pde", "initial", "boundary", "interface", "physics_total")))):
        raise ValueError("F2训练PDE必须pde=None未计算，独立全项PDE则须有限真实值")


def verify_initial_origin(initial: Mapping[str, Any], sources: Mapping[int, Any],
                          seed: int, device: torch.device) -> None:
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))
                               if device.type == "cuda" else []):
        origin = fork_f2_e0(sources, seed, device)
    if (not _equal(initial.get("model_state"), origin.model.state_dict()) or
        not _equal(initial.get("optimizer_state"), origin.optimizer.state_dict()) or
        not _equal(initial.get("random_state"), origin.random_state) or
        initial.get("epoch") != 0 or initial.get("stage") != CORRECTION_STAGE):
        raise ValueError("任08 F2第0轮初态HF/LF全张量、空AdamW或四类随机源不同seed E0")


def five_seed_score_statistics(scores: Mapping[int, float]) -> dict[str, Any]:
    if set(scores) != set(range(5)) or any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("F2五seed样本方差必须来自五份真实有限合法HF选分")
    values = [float(scores[seed]) for seed in range(5)]
    return {"五种子宏平均选择分_摄氏度": float(np.mean(values)),
            "五种子宏选择分样本标准差_摄氏度": float(np.std(values, ddof=1)),
            "五种子样本标准差分母": 4}


def _verify_log(rows: list[dict[str, Any]], initial: Mapping[str, Any],
                final: Mapping[str, Any], source: Any) -> dict[str, Any]:
    c_epoch = int(final["metadata"]["校正实际轮次"])
    j_epoch = int(final["metadata"]["联合实际轮次"])
    if (not 200 <= c_epoch <= 1500 or c_epoch % 10 or
        not 200 <= j_epoch <= 500 or j_epoch % 10 or
        len(rows) != c_epoch + j_epoch):
        raise ValueError("F2每seed正式校正/联合真截止应各200轮起、十轮验证及完整日志")
    best_score = float(initial["metadata"]["观测最佳选分_摄氏度"])
    physical_score = float(initial["metadata"]["物理最佳独立全项损失"])
    best_epoch = physical_epoch = 0
    phase_score = best_score
    phase_epoch = 0
    last_lf_keep = None
    reference = initial["metadata"]["LF合法验证初态逐材料节点与体积RMSE_摄氏度"]
    for epoch, row in enumerate(rows, 1):
        correction = epoch <= c_epoch
        local = epoch if correction else epoch - c_epoch
        played = 0 if correction else 60
        cumulative = {
            "HF训练观测点": 29593 * epoch,
            "HF训练传感器点": 44775 * epoch,
            "LF真实回放训练点": (0 if correction else local * 60 * 2048),
            "物理配点": 256 * epoch,
            "HF观测优化步": 15 * epoch,
            "物理优化步": epoch,
            "LF联合回放batch": 0 if correction else 60 * local,
        }
        if (row.get("epoch") != epoch or row.get("全局实际轮次") != epoch or
            row.get("运行种子") != source.seed or row.get("运行臂") != "F2" or
            row.get("训练阶段") != (CORRECTION_STAGE if correction else JOINT_STAGE) or
            row.get("阶段实际轮次") != local or
            row.get("HF训练观测点") != 29593 or
            row.get("HF训练传感器点") != 44775 or
            row.get("HF观测优化步") != 15 or row.get("物理优化步") != 1 or
            row.get("物理配点") != 256 or row.get("LF联合回放batch") != played or
            row.get("LF真实回放训练点") != played * 2048 or
            row.get("累计实际消耗") != cumulative or
            correction and row.get("LF当前真实张量SHA256") != source.lf_tensor_sha256 or
            not correction and (row.get("LF_Cu真实回放点", 0) <= 0 or
                                row.get("LF_SiC真实回放点", 0) <= 0 or
                                row.get("LF_Cu真实回放点") + row.get("LF_SiC真实回放点") != 60 * 2048)):
            raise ValueError(f"任08 F2第{epoch}轮HF15+1、LF60及四口径真实源预算不自洽")
        verify_physics_distinction(row)
        due = local % 10 == 0
        if not due:
            if (row.get("独立局部物理损失") is not None or
                row.get("HF合法验证选分_摄氏度") is not None or
                row.get("LF合法验证逐材料节点与体积RMSE_摄氏度") is not None):
                raise ValueError("F2非十轮非法注入验证分数或完整物理评分")
            continue
        independent = row["独立局部物理损失"]["physics_total"]
        score = row.get("HF合法验证选分_摄氏度")
        if not isinstance(score, (float, int)) or not math.isfinite(score):
            raise ValueError("F2合法HF选分必须真实有限值")
        if correction:
            reported = row.get("LF合法验证初态逐材料节点与体积RMSE_摄氏度")
            if reported != reference:
                raise ValueError("F2校正LF真初态逐材料节点/体积漂移")
            if score < best_score - 0.0001:
                best_score, best_epoch = score, epoch
            if independent < physical_score:
                physical_score, physical_epoch = independent, epoch
            if local == c_epoch:
                phase_score, phase_epoch = score, 0
        else:
            materials = row.get("LF合法验证逐材料节点与体积RMSE_摄氏度")
            lf_keep = row.get("LF逐材料节点和真实体积5%护栏")
            if not isinstance(materials, dict) or not isinstance(lf_keep, dict):
                raise ValueError("F2联合十轮缺逐材料节点/真实体积及LF守护")
            derived = task04_lf_keep_guardrail(reference, materials)
            if lf_keep != derived:
                raise ValueError("F2联合LF验证护栏声称与真实Cu/SiC四口径不符")
            last_lf_keep = lf_keep
            if lf_keep["LF两材料节点与真实体积均守住5%护栏"]:
                if score < phase_score - 0.0001:
                    phase_score, phase_epoch = score, local
                if score < best_score - 0.0001:
                    best_score, best_epoch = score, epoch
                if independent < physical_score:
                    physical_score, physical_epoch = independent, epoch
    meta = final["metadata"]
    if (not math.isclose(float(meta["观测最佳选分_摄氏度"]), best_score, abs_tol=1e-4) or
        not math.isclose(float(meta["物理最佳独立全项损失"]), physical_score, abs_tol=1e-4) or
        meta["观测最佳全局轮次"] != best_epoch or
        meta["物理最佳全局轮次"] != physical_epoch or
        meta["本阶段早停最佳轮次"] != phase_epoch or
        meta["LF逐材料节点和真实体积5%护栏"] != last_lf_keep or
        meta["累计实际消耗"] != rows[-1]["累计实际消耗"]):
        raise ValueError("F2全量历史重推双轨最佳或真实LF最终护栏与最终阶段不同")
    return {"观测最佳选分_摄氏度": best_score,
            "观测最佳全局轮次": best_epoch,
            "物理最佳全项损失": physical_score,
            "物理最佳全局轮次": physical_epoch,
            "联合末LF5%护栏": last_lf_keep,
            "校正真实轮次": c_epoch, "联合真实轮次": j_epoch}


def verify_run(run_directory: str | Path, *, registry_sha256: str,
               device: torch.device) -> dict[str, Any]:
    _require_formal_registration(TASK08_F2_REGISTRATION, registry_sha256)
    run = _path(run_directory)
    if not run.is_dir():
        raise FileNotFoundError("任08 F2真实五seed运行目录尚不存在")
    report_path = run / "阶段报告.json"
    if not report_path.is_file():
        raise FileNotFoundError("任08 F2正式训练报告原件缺失")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("运行资格") != _qualification(False) or
        report.get("真实最后状态") != "阶段_训练末.pt"):
        raise ValueError("任08 F2诊断或正式半程不得作五seed已完成审核")
    seed = report.get("运行种子")
    sources = validate_task07_sources()
    _require_formal_registration(TASK08_F2_REGISTRATION, registry_sha256, sources)
    if seed not in sources or report.get("运行臂") != "F2":
        raise ValueError("任08 F2报告种子或运行臂与锁定五配对来源不符")
    source = sources[seed]
    _snapshot_hashes(run)
    c_epoch, j_epoch = int(report["校正实际轮次"]), int(report["联合实际轮次"])
    if (not 200 <= c_epoch <= 1500 or c_epoch % 10 or
        not 200 <= j_epoch <= 500 or j_epoch % 10):
        raise ValueError("任08 F2独立能源仅审核校正、联合真早停/上限完成的种子")
    paths = {
        "initial": "阶段_初始.pt", "best": "阶段_观测最佳.pt",
        "physical": "阶段_物理最佳.pt", "correction": "阶段_校正末.pt",
        "joint_initial": "阶段_联合初始.pt", "latest": "阶段_最近.pt",
        "joint_final": "阶段_联合末.pt", "final": "阶段_训练末.pt",
    }
    stages: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for name, filename in paths.items():
        path = run / filename
        if not path.is_file():
            raise FileNotFoundError(f"任08 F2完整正式阶段原件缺失：{filename}")
        stages[name] = torch.load(path, map_location="cpu", weights_only=False)
        hashes[name] = sha256_file(path)
        _check_snapshot_identity(stages[name], source, registry_sha256, False)
    initial, latest, terminal = stages["initial"], stages["latest"], stages["final"]
    verify_initial_origin(initial, sources, seed, device)
    if (initial["metadata"]["全局累计实际轮次"] != 0 or
        stages["correction"]["metadata"]["全局累计实际轮次"] != c_epoch or
        stages["joint_initial"]["metadata"]["全局累计实际轮次"] != c_epoch or
        stages["joint_initial"]["stage"] != JOINT_STAGE or
        terminal["metadata"]["全局累计实际轮次"] != c_epoch + j_epoch or
        any(not _equal(terminal[key], stages[label][key])
            for label in ("latest", "joint_final") for key in
            ("model_state", "optimizer_state", "random_state",
             "parameter_requires_grad", "metadata"))):
        raise ValueError("F2真校正/联合及final/latest三份末态模型、AdamW和四RNG不一致")
    correction, joint_start = stages["correction"], stages["joint_initial"]
    if (not _equal(correction["model_state"], joint_start["model_state"]) or
        not _equal(correction["optimizer_state"]["state"],
                   joint_start["optimizer_state"]["state"]) or
        joint_start["metadata"].get("已提交正式校正末原件SHA256") != hashes["correction"] or
        terminal["metadata"].get("已提交正式校正末原件SHA256") != hashes["correction"] or
        any(not _equal(correction["random_state"][key],
                       joint_start["random_state"][key])
            for key in ("python", "numpy", "torch_cuda"))):
        raise ValueError("F2联合起点重置了LF/HF或新建原HF动量，非同seed冻结承接")
    rows = [json.loads(line) for line in (run / "training.jsonl").read_text(
        encoding="utf-8").splitlines()]
    derived = _verify_log(rows, initial, terminal, source)
    if (report.get("最近一轮真实入场证据") != rows[-1] or
        report.get("累计实际消耗") != terminal["metadata"]["累计实际消耗"] or
        report.get("观测最佳全局轮次") != derived["观测最佳全局轮次"] or
        report.get("物理最佳全局轮次") != derived["物理最佳全局轮次"] or
        not math.isclose(float(report.get("观测最佳HF合法选分_摄氏度", math.inf)),
                         derived["观测最佳选分_摄氏度"], abs_tol=1e-4) or
        not math.isclose(float(report.get("物理最佳独立全项损失", math.inf)),
                         derived["物理最佳全项损失"], abs_tol=1e-4)):
        raise ValueError("任08 F2正式报告与双阶段真消费/双最佳观测物理分数不符")
    architecture = torch.load(source.hf_checkpoint_path, map_location="cpu", weights_only=False)
    selections = load_yaml("configs/training.yaml")["multifidelity_selection_weights"]
    validation_loader = DataLoader(_ir_dataset("validation", source.hf_validation_powers_w),
                                   batch_size=2048, shuffle=False)
    validation_sensor = _sensor_tensors(device, "validation", source.hf_validation_powers_w)
    materials, boundaries = load_materials(), load_resolved_boundary_conditions()
    _train, full_audit = _physics_pair(device)
    for name in ("阶段_观测最佳.pt", "阶段_物理最佳.pt"):
        _snapshot, _model, _validation, repair = _committed_best(
            run, terminal, name, source, registry_sha256, False,
            validation_loader, validation_sensor, selections,
            full_audit, materials, boundaries, device)
        if repair:
            raise ValueError("任08 F2旧最佳完整原件丢失或尚未提交，独立审计不得修改原运行目录")
    best_view = run / "best.pt"
    if not best_view.is_file():
        raise FileNotFoundError("任08 F2历史架构模型视图缺失")
    view = torch.load(best_view, map_location="cpu", weights_only=False)
    lineage = view.get("任08F2模型视图来源", {})
    best = stages["best"]
    if (view.get("epoch") != derived["观测最佳全局轮次"] or
        view.get("correction_model_kwargs") != source.correction_model_kwargs or
        view.get("low_fidelity_model_kwargs") != source.lf_model_kwargs or
        view.get("scales") != source.scales or
        lineage.get("运行臂") != "F2" or lineage.get("训练种子") != seed or
        lineage.get("正式预登记配置SHA256") != registry_sha256 or
        lineage.get("训练体内PDE", "非法") is not None or
        not _equal(view.get("model_state"), best["model_state"])):
        raise ValueError("任08 F2观测最佳视图kwargs/PDE声明与完整状态不一致")
    hashes["best_view"] = sha256_file(best_view)
    score = float(derived["观测最佳选分_摄氏度"])
    is_correction_only = best["stage"] == CORRECTION_STAGE
    final_keep = derived["联合末LF5%护栏"]
    if not is_correction_only and not final_keep["LF两材料节点与真实体积均守住5%护栏"]:
        # Last joint candidate may fail while an older qualified best is valid.
        qualified = best["metadata"].get("LF逐材料节点和真实体积5%护栏")
        if not qualified or not qualified["LF两材料节点与真实体积均守住5%护栏"]:
            raise ValueError("任08 F2不合LF护栏联合best不得进入观测采用")
    return {
        "运行目录": run, "运行种子": seed,
        "配对来源": source, "正式配置SHA256": registry_sha256,
        "校正真实轮次": c_epoch, "联合真实轮次": j_epoch,
        "观测合法选分_摄氏度": score,
        "联合最后LF5%护栏": final_keep,
        "联合末候选不可采用": not final_keep["LF两材料节点与真实体积均守住5%护栏"],
        "最佳仅校正阶段": is_correction_only,
        "阶段文件": {key: run / file for key, file in paths.items()},
        "阶段SHA256": hashes,
        "日志SHA256": sha256_file(run / "training.jsonl"),
        "报告SHA256": sha256_file(report_path),
        "best_view": best_view,
        "日志重推": derived,
    }


def _stage_model(info: Mapping[str, Any], state: str,
                 device: torch.device) -> torch.nn.Module:
    source = info["配对来源"]
    stage_file = info["阶段文件"][state]
    snapshot = torch.load(stage_file, map_location="cpu", weights_only=False)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))
                               if device.type == "cuda" else []):
        sources = validate_task07_sources()
        if sources[source.seed].source_metadata_sha256 != source.source_metadata_sha256:
            raise ValueError("任08 F2重载时完整五seed源与阶段本seed血缘已变化")
        model = fork_f2_e0(sources, source.seed, device).model
    model.load_state_dict(snapshot["model_state"], strict=True)
    if snapshot["stage"] == JOINT_STAGE:
        from sic_cu.train.task04_joint import PROJECTION_NAMES

        for name, parameter in model.named_parameters():
            if name in PROJECTION_NAMES:
                parameter.requires_grad_(True)
    _check_model(model, source, snapshot["stage"])
    if _lf_sha(model) != snapshot["metadata"]["当前真实LF张量SHA256"]:
        raise ValueError("F2独立模型重载和完整阶段LF原张量不同")
    model.eval()
    return model


def audit_state_energy(
    run_directory: str | Path, *, state: str, output_directory: str | Path,
    registry_sha256: str, device_name: str = "cuda",
) -> dict[str, Any]:
    if state not in ("best", "physical", "final"):
        raise ValueError("F2仅分别审核观测最佳、全项物理最佳和真末态三个独立模型状态")
    run, output = _path(run_directory), _project_output(output_directory)
    if output.exists() or run.resolve() == output.resolve() or run.resolve() in output.resolve().parents:
        raise FileExistsError("任08 F2能源输出不能覆盖或嵌入原训练目录")
    _require_formal_registration(TASK08_F2_REGISTRATION, registry_sha256)
    if not (run / "阶段_训练末.pt").is_file():
        raise ValueError("任08 F2诊断/半程缺真实联合训练末，拒绝启动能源积分")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("任08 F2正式能源只许本次真实CUDA双阶全项独立积分")
    info = verify_run(run, registry_sha256=registry_sha256, device=device)
    model = _stage_model(info, state, device)
    powers, times, orders = fixed_schedule()
    geometry = AxisymmetricGeometry.from_config(load_yaml("configs/geometry.yaml"))
    evidence = audit_schedule_energy(
        model, load_materials(), load_resolved_boundary_conditions(), geometry,
        powers_w=powers, times_s=times, orders=(orders[0], orders[1]), device=device)
    payload = {
        "状态": "任08 F2仅独立工程能源审计；不能用作训练梯度或直接称物理通过",
        "运行臂": "F2", "真实种子": info["运行种子"], "模型状态": state,
        "源正式预登记SHA256": registry_sha256,
        "本seed源LF原件SHA256": info["配对来源"].lf_checkpoint_sha256,
        "本seed本状态完整阶段SHA256": info["阶段SHA256"][state],
        "本seed训练日志SHA256": info["日志SHA256"],
        "本seed阶段报告SHA256": info["报告SHA256"],
        "原工程能源计算与F3同六功率五时刻两阶": {
            "功率_瓦": powers, "时刻_秒": times, "求积阶数": orders,
            "功率时刻点数": len(powers) * len(times)},
        "最佳合法HF选分_摄氏度": info["观测合法选分_摄氏度"],
        "校正真实轮次": info["校正真实轮次"],
        "联合真实轮次": info["联合真实轮次"],
        "联合末LF护栏未过不可采用": info["联合末候选不可采用"],
        "来源训练PDE标记": None,
        "来源训练PDE意义": "未计算；不是能源体内积分或真实HF内部实验真值",
        "原定义相对平衡分母已保持": evidence["汇总"]["原定义相对平衡分母已保持"],
        "原始能量双阶行数": len(evidence["原始能量"]),
        "原始散度双阶行数": len(evidence["原始散度"]),
        "瓦数/体内V_界面J_边界D同源汇总": evidence["汇总"],
        "旧test_Data温度标签读取": False,
    }
    if (not payload["原定义相对平衡分母已保持"] or
        not evidence["汇总"]["工程平衡与原瓦数逐行一致"] or
        len(evidence["原始能量"]) != 60 or len(evidence["原始散度"]) != 60):
        raise ValueError("F2 30功率时间点双阶原瓦数/散度V-J-D缺失或混口径")
    hashes = _old_auditor().write_energy_evidence(output, evidence, payload)
    return payload | {"输出": str(output), "六工件真实SHA256": hashes}


def _native_f32_radius_records(
    model: torch.nn.Module, seed: int, device: torch.device,
    config: Mapping[str, Any], validation_powers: tuple[float, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = load_processed_ir_observations("validation").sort("power_w", "time_s", "r_m")
    predictions = _predict(model, _top_coordinates(frame), device,
                           int(config["prediction_batch_sizes"][-1]))
    if len(predictions) != frame.height:
        raise ValueError("F2修订径向须来自完整合法HF顶部真实点与逐点模型预测")
    windows = config["radial_windows_mm"]
    old_rows, new_rows = [], []
    counts = {115.2: (1919, 19), 403.0: (2424, 24), 630.5: (2929, 29)}
    for power in validation_powers:
        selected = np.isclose(frame["power_w"].to_numpy(), power, atol=1e-4)
        group = frame.filter(pl.col("power_w") == power)
        powers_prediction = predictions[selected]
        radii = group["r_m"].to_numpy()
        if (radii.dtype != np.float32 or group.height != counts[power][0] or
            group.height != len(powers_prediction)):
            raise ValueError("F2每验证功率径向须消费原生米制float32所有顶部实测行")
        arrays = _observation_arrays(group, powers_prediction, "Top")
        old_masks, new_masks = [], []
        for window in windows:
            lower, upper = float(window["lower"]), float(window["upper"])
            old_mm = radii.astype(np.float64) * 1000.0
            old_lower = old_mm >= lower if window["lower_closed"] else old_mm > lower
            native_lower = radii >= np.float32(lower / 1000.0) if window[
                "lower_closed"] else radii > np.float32(lower / 1000.0)
            old_masks.append(old_lower & (old_mm <= upper))
            new_masks.append(native_lower & (radii <= np.float32(upper / 1000.0)))
        if (len(old_masks) != 3 or not np.all(sum(mask.astype(np.int8)
                                                for mask in new_masks) == 1) or
            any(np.count_nonzero(old_masks[i] ^ new_masks[i]) for i in (0, 1)) or
            np.count_nonzero(old_masks[2] ^ new_masks[2]) != counts[power][1] or
            sum(int(mask.sum()) for mask in new_masks) != group.height):
            raise ValueError("F2闭端点更正只能补上原V4遗漏的最外25mm点，不许改中内窗")
        for target, masks in ((old_rows, old_masks), (new_rows, new_masks)):
            for window, mask in zip(windows, masks):
                metrics = error_statistics(
                    arrays["target"][mask], arrays["prediction"][mask],
                    arrays["weights"][mask])
                target.append({
                    "seed": seed, "split": "validation", "power_w": float(power),
                    "radial_window": window["name"], "lower_mm": window["lower"],
                    "upper_mm": window["upper"],
                    "lower_closed": window["lower_closed"], "upper_closed": True,
                    "sample_count": int(mask.sum()), **metrics,
                })
    return old_rows, new_rows


def audit_five_observations(
    seed_directories: Mapping[int, str | Path], *, output_directory: str | Path,
    registry_sha256: str, device_name: str = "cuda",
) -> dict[str, Any]:
    if set(seed_directories) != set(range(5)):
        raise ValueError("任08 F2五种子合法HF观测须有0到4各一完整真实末态")
    output = _project_output(output_directory)
    if (output.exists() or any(_path(run).resolve() in output.resolve().parents
                               or _path(run).resolve() == output.resolve()
                               for run in seed_directories.values())):
        raise FileExistsError("任08 F2五种子观测目录不可覆盖或嵌在训练原件目录内")
    _require_formal_registration(TASK08_F2_REGISTRATION, registry_sha256)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("任08 F2正式观测须真实CUDA五seed模型合法重载")
    registered = {seed: verify_run(seed_directories[seed],
                                   registry_sha256=registry_sha256, device=device)
                  for seed in range(5)}
    if any(info["运行种子"] != seed for seed, info in registered.items()):
        raise ValueError("任08 F2五seed目录出现重复种子或跨seed阶段")
    optimization_path = PROJECT_ROOT / "configs/optimization_v4.yaml"
    if sha256_file(optimization_path) != OPTIMIZATION_SHA256:
        raise ValueError("任08 F2原V4合法时间/径向窗配置SHA已变")
    configuration = load_yaml(optimization_path)
    if (configuration.get("allow_test_labels") is not False or
        configuration["diagnostics"].get("evaluation_splits") !=
        ["train", "validation"]):
        raise ValueError("任08 F2不可动用旧测试温度标签")
    diagnostic = configuration["diagnostics"]
    bottom_z = float(load_yaml("configs/geometry.yaml")["embedding"]["copper_bottom_z_m"])
    by_power, by_time, by_radius_old, by_radius_fixed = [], [], [], []
    scores = {}
    for seed in range(5):
        info = registered[seed]
        source = info["配对来源"]
        model = _stage_model(info, "best", device)
        powers, windows, _old, _cached = evaluate_hf_observations(
            model, seed, "validation", device, diagnostic, bottom_z)
        radial_old, radial_fixed = _native_f32_radius_records(
            model, seed, device, diagnostic, source.hf_validation_powers_w)
        if (len(powers) != 9 or len(windows) != 27 or
            len(radial_old) != 9 or len(radial_fixed) != 9 or
            {row["power_w"] for row in powers} != set(source.hf_validation_powers_w)):
            raise ValueError(f"任08 F2 seed{seed}合法三功率三模态/三时窗/三径向缺行")
        by_power.extend(powers)
        by_time.extend(windows)
        by_radius_old.extend(radial_old)
        by_radius_fixed.extend(radial_fixed)
        scores[seed] = info["观测合法选分_摄氏度"]
    if (len(by_power) != 45 or len(by_time) != 135 or
        len(by_radius_old) != 45 or len(by_radius_fixed) != 45):
        raise ValueError("任08 F2五seed观测原/修同身份45+135+45行未完整")
    raw = {"逐功率三模态": by_power, "逐功率分时窗口": by_time,
           "顶部径向原V4表示": by_radius_old,
           "顶部径向原生float32闭端点": by_radius_fixed}
    summary = {
        "状态": "F2五种子已真实重载；合法HF观测统计独立复核，能源仍另行审核",
        "正式来源预登记SHA256": registry_sha256,
        **five_seed_score_statistics(scores),
        "逐seed真实合法选择分_摄氏度": scores,
        "逐seed校正和联合实际轮次": {
            seed: [info["校正真实轮次"], info["联合真实轮次"]]
            for seed, info in registered.items()},
        "逐seed联合末护栏不可采用": {
            seed: info["联合末候选不可采用"]
            for seed, info in registered.items()},
        "径向与历史B0_任07F3比较版本": (
            "须仅与十模型原生float32闭端点修版45行对比；原V4旧径向表保留不能混用"),
        "当前资格": "任07乙：只准讨论合法观测；未知内部HF真值和工程能源不能凭观测推断",
        "旧test_Data温度标签读取": False,
        "逐seed原件来源SHA256": {
            seed: info["阶段SHA256"] for seed, info in registered.items()},
    }
    output.mkdir(parents=True)
    for name, rows in raw.items():
        pl.DataFrame(rows).write_csv(output / f"{name}.csv", null_value="")
    (output / "原始合法验证指标.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "五种子合法观测摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    files = {file.name: sha256_file(file) for file in sorted(output.iterdir()) if file.is_file()}
    (output / "审计工件SHA256.json").write_text(
        json.dumps(files, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary | {"工件真实SHA256": files, "输出": str(output),
                      "明细行数": {name: len(rows) for name, rows in raw.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description="任08 F2五种子合法HF与三状态独立能源审核")
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    sub = parser.add_subparsers(dest="audit", required=True)
    state = sub.add_parser("energy", help="单seed的best、physical或final单独六件能源")
    state.add_argument("--run", required=True)
    state.add_argument("--state", choices=("best", "physical", "final"), required=True)
    state.add_argument("--output", required=True)
    five = sub.add_parser("observations", help="五seed同合法功率45模态/135窗/45修版径向")
    for seed in range(5):
        five.add_argument(f"--seed{seed}", required=True)
    five.add_argument("--output", required=True)
    args = parser.parse_args()
    report = (audit_state_energy(args.run, state=args.state,
                                 output_directory=args.output,
                                 registry_sha256=args.registry_sha256,
                                 device_name=args.device) if args.audit == "energy" else
              audit_five_observations(
                  {seed: getattr(args, f"seed{seed}") for seed in range(5)},
                  output_directory=args.output,
                  registry_sha256=args.registry_sha256, device_name=args.device))
    print(json.dumps(report, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
