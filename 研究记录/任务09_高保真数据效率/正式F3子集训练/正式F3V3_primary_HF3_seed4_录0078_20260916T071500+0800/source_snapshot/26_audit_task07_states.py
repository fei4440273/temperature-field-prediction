#!/usr/bin/env python
"""任-07五种子观测最佳与真实联合末态独立能源审核。"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.eval.energy_v5 import AxisymmetricGeometry, audit_schedule_energy
from sic_cu.losses import PhysicsLossComputer, PhysicsLossWeights
from sic_cu.physics import sample_collocation
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.physics.resolution import resolve_physics_state
from sic_cu.train.task04_joint import PROJECTION_NAMES, task04_lf_keep_guardrail
from sic_cu.train.common import CONFIG_FILES
from sic_cu.train.task07_source import (
    MANIFEST_PATH, MANIFEST_SHA256, _lf_tensor_sha256, fork_task07_initialization,
    validate_task07_sources,
)
from sic_cu.train.task07_formal import (
    TASK07_LEDGER, TASK07_REGISTRATION, _expected_registration,
)
from sic_cu.train.multifidelity import _ir_dataset, _sensor_tensors
from sic_cu.train.task04_joint import _lf_material_validation, _validation_selection


TASK04_CONFIG = PROJECT_ROOT / "研究记录/任务04_联合微调/有效运行配置.yaml"
TASK04_CONFIG_SHA256 = "b464eddd752000694a513da0e4a1b4b0f7447e31b85d19726e243396a0864710"
TASK06_CONFIG = PROJECT_ROOT / "研究记录/任务06_时间响应特征/HF三臂先导有效配置.yaml"
TASK06_CONFIG_SHA256 = "7a418d94493ddafd9c78e5070bca0739ab2de30aecdc22ef322ef69bc7e03b19"
TASK04_AUDITOR_FILE = PROJECT_ROOT / "scripts/21_audit_task04_energy.py"
TASK07_RUN_ROOT = PROJECT_ROOT / "研究记录/任务07_正式五种子重训"
OFFICIAL = "五种子正式E0候选；须五seed完整归档与能源审核后决定采用"


def _old_auditor() -> Any:
    spec = importlib.util.spec_from_file_location("task04_energy_audit_task07", TASK04_AUDITOR_FILE)
    if spec is None or spec.loader is None:
        raise ImportError("任07依赖的旧任务04工程能源源码文件不可加载")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixed_schedule() -> tuple[list[float], list[float], list[int]]:
    old = _old_auditor()
    if (sha256_file(TASK04_CONFIG) != TASK04_CONFIG_SHA256
            or sha256_file(TASK06_CONFIG) != TASK06_CONFIG_SHA256):
        raise ValueError("任07独立能源继承的任04/06事前六功率五时刻配置SHA已经改变")
    task04 = load_yaml(TASK04_CONFIG)["独立能量审核"]
    task06 = load_yaml(TASK06_CONFIG)["独立30点能量"]
    powers, times, orders = old.FIXED_POWERS, old.FIXED_TIMES, [16, 64]
    if (task04["功率_瓦"] != powers or task04["时刻_秒"] != times
            or task04["求积阶数"] != orders or task04["功率时刻点数"] != 30
            or task06["功率_瓦"] != powers or task06["时刻_秒"] != times
            or task06["阶数"] != orders or len(powers) * len(times) != 30):
        raise ValueError("任07必须沿用任04/06事前相同30点与16/64阶原能源定义")
    return powers, times, orders


def _metrics_valid(values: Any, names: tuple[str, ...]) -> bool:
    return (isinstance(values, dict) and set(values) == set(names)
            and all(type(values[name]) in (float, int)
                    and math.isfinite(values[name]) and values[name] >= 0
                    for name in names))


def _materials_valid(values: Any) -> bool:
    return (isinstance(values, dict) and set(values) == {"Cu", "SiC"}
            and all(_metrics_valid(values[name], ("node", "volume"))
                    for name in ("Cu", "SiC")))


def _verify_history(
    rows: list[dict[str, Any]], *, seed: int, correction_epoch: int,
    joint_epoch: int, source_lf_sha: str, terminal_lf_sha: str,
    initial_score: float, initial_physical: float,
) -> dict[str, Any]:
    total = correction_epoch + joint_epoch
    if (not 200 <= correction_epoch <= 1500 or correction_epoch % 10
            or not 200 <= joint_epoch <= 500 or joint_epoch % 10
            or len(rows) != total or not all(math.isfinite(float(value)) for value in
                                          (initial_score, initial_physical))):
        raise ValueError("任07正式日志须真实两阶段校正/联合截止且至少各200轮")
    best_epoch = physical_epoch = correction_phase_epoch = joint_phase_epoch = 0
    best_score, physical_score = initial_score, initial_physical
    correction_phase_score = initial_score
    joint_phase_score = None
    initial_lf = last_materials = last_keep = None
    last_lf_sha = source_lf_sha
    for epoch, row in enumerate(rows, 1):
        correction = epoch <= correction_epoch
        local = epoch if correction else epoch - correction_epoch
        replay = 0 if correction else 60
        expected_consumption = {
            "HF训练观测点": 29593 * epoch,
            "HF训练传感器点": 44775 * epoch,
            "LF真实回放训练点": max(epoch - correction_epoch, 0) * 60 * 2048,
            "物理配点": 256 * epoch,
            "HF观测优化步": 15 * epoch,
            "物理优化步": epoch,
            "LF联合回放batch": 60 * max(epoch - correction_epoch, 0),
        }
        lf_sha = row.get("LF当前真实张量SHA256")
        if (
            row.get("epoch") != epoch or row.get("全局实际轮次") != epoch
            or row.get("运行种子") != seed or row.get("运行臂") != "E0"
            or row.get("训练阶段") != ("task07_correction" if correction else
                                           "task07_restricted_joint")
            or row.get("阶段实际轮次") != local
            or row.get("HF训练观测点") != 29593
            or row.get("HF训练传感器点") != 44775
            or row.get("HF观测优化步") != 15 or row.get("物理优化步") != 1
            or row.get("物理配点") != 256
            or row.get("LF真实回放训练点") != replay * 2048
            or row.get("LF联合回放batch") != replay
            or row.get("累计实际消耗") != expected_consumption
            or not isinstance(lf_sha, str) or len(lf_sha) != 64
            or (correction and lf_sha != source_lf_sha)
            or (not correction and (
                row.get("LF_Cu真实回放点", 0) <= 0
                or row.get("LF_SiC真实回放点", 0) <= 0
                or row["LF_Cu真实回放点"] + row["LF_SiC真实回放点"] != 60 * 2048
            ))
            or (correction and (row.get("LF_Cu真实回放点") != 0
                                or row.get("LF_SiC真实回放点") != 0))
        ):
            raise ValueError(f"任07第{epoch}轮HF/物理/LF实际点数、来源或正式阶段日志异常")
        last_lf_sha = lf_sha
        due = local % 10 == 0
        score, modalities = (row.get("HF合法验证选分_摄氏度"),
                             row.get("HF合法验证分模态_摄氏度"))
        physical = row.get("独立局部物理损失")
        if not due:
            if (score is not None or modalities is not None or physical is not None
                    or row.get("LF合法验证逐材料节点与体积RMSE_摄氏度") is not None):
                raise ValueError(f"任07第{epoch}轮非10倍次不能冒用合法HF/物理/LF验证选择")
            continue
        if (type(score) not in (int, float) or not math.isfinite(score)
                or not _metrics_valid(modalities, ("top", "hot", "cold"))
                or not isinstance(physical, dict)
                or not all(type(physical.get(key)) in (int, float)
                           and math.isfinite(physical[key])
                           for key in ("pde", "initial", "boundary", "interface", "physics_total"))):
            raise ValueError(f"任07第{epoch}轮合法HF三模态或独立物理诊断没有有限原数")
        if correction:
            raw_lf = row.get("LF合法验证初态逐材料节点与体积RMSE_摄氏度")
            if not _materials_valid(raw_lf):
                raise ValueError("任07校正每10轮须具LF独立来源四口径验证基准")
            if initial_lf is None:
                initial_lf = raw_lf
            if any(not math.isclose(raw_lf[name][key], initial_lf[name][key], abs_tol=1e-4)
                   for name in ("Cu", "SiC") for key in ("node", "volume")):
                raise ValueError("任07校正冻结LF初态节点与真实体积四口径漂移")
            if score < best_score - 1e-4:
                best_epoch, best_score = epoch, score
                correction_phase_epoch = epoch
            if physical["physics_total"] < physical_score:
                physical_epoch, physical_score = epoch, physical["physics_total"]
            if epoch == correction_epoch:
                joint_phase_score = score
        else:
            material = row.get("LF合法验证逐材料节点与体积RMSE_摄氏度")
            if not _materials_valid(material) or initial_lf is None:
                raise ValueError(f"任07第{epoch}轮联合缺Cu/SiC节点或真实体积四口径")
            derived = task04_lf_keep_guardrail(initial_lf, material)
            reported = row.get("LF逐材料节点和真实体积5%护栏")
            if (not isinstance(reported, dict) or
                    reported.get("LF两材料节点与真实体积均守住5%护栏") is not
                    derived["LF两材料节点与真实体积均守住5%护栏"]):
                raise ValueError(f"任07第{epoch}轮LF四口径5%保持资格与真实材料误差不一致")
            last_materials, last_keep = material, reported
            if derived["LF两材料节点与真实体积均守住5%护栏"]:
                if joint_phase_score is None:
                    raise ValueError("任07联合缺少真实校正末第10轮阶段初始分")
                if score < joint_phase_score - 1e-4:
                    joint_phase_score, joint_phase_epoch = score, local
                if score < best_score - 1e-4:
                    best_epoch, best_score = epoch, score
                if physical["physics_total"] < physical_score:
                    physical_epoch, physical_score = epoch, physical["physics_total"]
    if (last_lf_sha != terminal_lf_sha or initial_lf is None or last_keep is None
            or (correction_epoch != 1500 and correction_epoch - correction_phase_epoch < 200)
            or (joint_epoch != 500 and joint_epoch - joint_phase_epoch < 200)):
        raise ValueError("任07两阶段真实末状态、早停耐心或LF完整源SHA与全程日志不一致")
    return {
        "观测最佳全局轮次": best_epoch, "观测最佳选分_摄氏度": best_score,
        "物理最佳全局轮次": physical_epoch, "物理最佳独立损失": physical_score,
        "校正实际截止轮次": correction_epoch, "联合实际截止轮次": joint_epoch,
        "LF初始四口径": initial_lf, "LF末态四口径": last_materials,
        "LF末次保持资格": last_keep,
        "累计实际消耗": expected_consumption,
    }


def _verify_stage(
    stage: dict[str, Any], sources: dict[int, Any], *, seed: int,
    registry_sha: str, expected_stage: str, correction_epoch: int,
    joint_epoch: int,
) -> None:
    if seed not in sources or expected_stage not in ("task07_correction", "task07_restricted_joint"):
        raise ValueError("任07仅接受五seed配对来源的旧六列E0校正/受限联合完整状态")
    source, meta = sources[seed], stage.get("metadata", {})
    joint = expected_stage == "task07_restricted_joint"
    current_epoch = joint_epoch if joint else correction_epoch
    total = correction_epoch + joint_epoch
    expected_usage = {
        "HF训练观测点": 29593 * total,
        "HF训练传感器点": 44775 * total,
        "LF真实回放训练点": 60 * 2048 * joint_epoch,
        "物理配点": 256 * total,
        "HF观测优化步": 15 * total,
        "物理优化步": total,
        "LF联合回放batch": 60 * joint_epoch,
    }
    if (
        stage.get("training_state_schema_version") != 1
        or stage.get("stage") != expected_stage or stage.get("epoch") != current_epoch
        or stage.get("budget") != {"HF校正上限轮次": 1500, "受限联合上限轮次": 500}
        or not 0 <= correction_epoch <= 1500 or not 0 <= joint_epoch <= 500
        or (not joint and joint_epoch != 0)
        or (joint and (correction_epoch < 200 or meta.get("校正实际截止轮次") != correction_epoch
                       or not meta.get("已提交正式校正末原件SHA256")))
        or meta.get("任07运行种子") != seed or meta.get("任07运行臂") != "E0"
        or meta.get("运行资格") != "五种子正式E0候选；须五seed完整归档与能源审核后决定采用"
        or meta.get("正式预登记配置SHA256") != registry_sha
        or meta.get("V4_B0来源清单SHA256") != MANIFEST_SHA256
        or meta.get("本seed源LF检查点SHA256") != source.lf_checkpoint_sha256
        or meta.get("本seed历史HF架构视图SHA256") != source.hf_checkpoint_sha256
        or meta.get("本seed真实LF初始张量SHA256") != source.lf_tensor_sha256
        or meta.get("本seed来源协议SHA256") != source.source_metadata_sha256
        or meta.get("当前阶段") != expected_stage
        or meta.get("校正实际轮次") != correction_epoch
        or meta.get("联合实际轮次") != joint_epoch
        or meta.get("全局累计实际轮次") != total
        or meta.get("累计实际消耗") != expected_usage
        or meta.get("旧test_Data温度标签读取") is not False
        or not _materials_valid(meta.get("LF合法验证初态逐材料节点与体积RMSE_摄氏度"))
    ):
        raise ValueError("任07完整阶段本seed LF来源、CUDA正式资格、两段预算或原始累计日志血缘无效")
    rng = stage.get("random_state", {})
    cuda = rng.get("torch_cuda")
    if (
        set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}
        or any(rng.get(key) is None for key in ("python", "numpy", "torch_cpu"))
        or not isinstance(rng.get("torch_cpu"), torch.Tensor)
        or not isinstance(cuda, (list, tuple)) or len(cuda) != 1
        or any(not isinstance(state, torch.Tensor) or state.dtype != torch.uint8
               or state.device.type != "cpu" or tuple(state.shape) != (16,)
               for state in cuda)
    ):
        raise ValueError("任07正式完整阶段四类随机源与真实CUDA种子状态形状必须可恢复")
    fresh = fork_task07_initialization(sources, seed, "E0", torch.device("cpu"))
    model = fresh.model
    actual = stage.get("model_state", {})
    canonical = model.state_dict()
    if (
        set(actual) != set(canonical)
        or any(not isinstance(actual[name], torch.Tensor)
               or actual[name].shape != original.shape or actual[name].dtype != original.dtype
               for name, original in canonical.items())
        or tuple(actual["correction.0.weight"].shape) != (128, 6)
        or any(name.startswith("response_features.") for name in actual)
    ):
        raise ValueError("任07本seed正式HF旧六列和源LF模型不能借历史视图或改架构")
    expected_lf = source.lf_state
    prefix = "low_fidelity_model."
    for name, source_value in expected_lf.items():
        full = prefix + name
        if full not in actual:
            raise ValueError("任07LF来源张量缺失")
        if joint and full in PROJECTION_NAMES:
            continue
        if not torch.equal(actual[full].cpu(), source_value.cpu()):
            raise ValueError("任07本seed原LF来源非四投影张量改变或校正LF未全冻结")
    lf_sha = _lf_tensor_sha256({name: actual[prefix + name] for name in expected_lf})
    if meta.get("当前真实LF张量SHA256") != lf_sha or (
        not joint and lf_sha != source.lf_tensor_sha256
    ):
        raise ValueError("任07本seed真实LF原张量SHA与全阶段来源不一致")
    if total == 0 and any(not torch.equal(actual[name].cpu(), original.cpu())
                          for name, original in canonical.items()):
        raise ValueError("任07新HF第0轮必须来自配对LF和仅本seed新随机初态，不可套历史HF权重")
    if joint and joint_epoch > 0 and any(torch.equal(
        actual[name].cpu(), expected_lf[name[len(prefix):]].cpu(),
    ) for name in PROJECTION_NAMES):
        raise ValueError("任07联合LF四末投影必须分别实际变化，不能只有伪优化器动量")
    named = dict(model.named_parameters())
    frozen = stage.get("parameter_requires_grad", {})
    correctable = [name for name in named if name.startswith("correction.")]
    projected = [name for name in named if name in PROJECTION_NAMES]
    if (
        set(frozen) != set(named) or len(projected) != 4
        or {name for name, flag in frozen.items() if flag} !=
        (set(correctable) | (set(projected) if joint else set()))
    ):
        raise ValueError("任07LF只有四个末投影可训练，HF校正器名单须与独立模型相同")
    optimizer = stage.get("optimizer_state", {})
    groups = optimizer.get("param_groups", [])
    moment = optimizer.get("state", {})
    first = fresh.optimizer.state_dict()["param_groups"][0]
    expected_ids = [first["params"]]
    if joint:
        expected_ids.append(list(range(len(correctable), len(correctable) + 4)))
    if not isinstance(moment, dict) or len(groups) != len(expected_ids):
        raise ValueError("任07真实HF或LF四投影AdamW参数组/动量不完整")
    for index, (group, ids) in enumerate(zip(groups, expected_ids)):
        reference = first.copy()
        reference["params"] = ids
        reference["lr"] = 0.00001 if index else (0.0001 if joint else 0.001)
        if group != reference:
            raise ValueError("任07HF/LF AdamW组参数ID、其他超参或阶段学习率与新初态不一致")
        names = projected if index else correctable
        required = 16 * (joint_epoch if index else total)
        for identifier, name in zip(ids, names):
            record = moment.get(identifier)
            if required == 0:
                if record is not None:
                    raise ValueError("任07未优化的新HF/LF不能继承历史AdamW动量")
                continue
            if (not isinstance(record, dict)
                    or not {"step", "exp_avg", "exp_avg_sq"} <= set(record)
                    or type(record["step"]) not in (float, int, torch.Tensor)
                    or not math.isfinite(float(record["step"]))
                    or float(record["step"]) != required
                    or any(not isinstance(record[key], torch.Tensor)
                           or record[key].shape != actual[name].shape
                           or not torch.isfinite(record[key]).all()
                           for key in ("exp_avg", "exp_avg_sq"))):
                raise ValueError("任07HF15＋物理1与LF四投影逐参数AdamW真实动量或步数不足/偷步")
    actual_momentum_ids = set(expected_ids[0]) if total else set()
    if joint and joint_epoch:
        actual_momentum_ids.update(expected_ids[1])
    if set(moment) != actual_momentum_ids:
        raise ValueError("任07完整阶段AdamW未知参数动量或第0轮伪动量")


def _verify_registry(sources: dict[int, Any], reported_sha: str) -> dict[str, Any]:
    if (not re.fullmatch(r"[0-9a-f]{64}", reported_sha or "")
            or not TASK07_REGISTRATION.is_file()
            or sha256_file(TASK07_REGISTRATION) != reported_sha):
        raise ValueError("任07修复后固定正式预登记真实字节SHA不存在或与完整阶段报告不符")
    recorded = load_yaml(TASK07_REGISTRATION)
    if (recorded.get("事前台账记录编号") != "录-0036"
            or recorded != _expected_registration(sources)):
        raise ValueError("任07录-0036五种子配对LF/HF来源、HF预算或训练入口源码SHA逐字段不符")
    lines = TASK07_LEDGER.read_text(encoding="utf-8").splitlines()
    ninth = next((index for index, line in enumerate(lines) if line.startswith("## 九、")), None)
    tenth = next((index for index, line in enumerate(lines) if line.startswith("## 十、")), None)
    if ninth is None or tenth is None or tenth <= ninth:
        raise ValueError("任07总台账第九节录-0036事前预登记位置缺失")
    entry = [line for line in lines[ninth + 1:tenth]
             if re.search(r"\|\s*录-0036\s*\|", line)]
    if (len(entry) != 1 or reported_sha not in entry[0]
            or TASK07_REGISTRATION.name not in entry[0]):
        raise ValueError("任07修复后总台账录-0036文件名与预登记SHA必须唯一且匹配")
    return {"sha256": reported_sha, "config": recorded, "ledger_row": entry[0]}


def _verify_snapshot(run: Path) -> dict[str, Any]:
    directory = run / "config_snapshot"
    manifest_file = directory / "sha256.json"
    recorded = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(recorded, dict):
        raise ValueError("任07七份运营配置快照SHA清单缺失")
    originals = {}
    for filename in CONFIG_FILES:
        copied = directory / Path(filename).name
        trusted = PROJECT_ROOT / filename
        if (not copied.is_file() or not trusted.is_file()
                or recorded.get(filename) != sha256_file(trusted)
                or recorded[filename] != sha256_file(copied)):
            raise ValueError(f"任07七份运行运营配置快照或真实原件SHA失配：{filename}")
        originals[filename] = recorded[filename]
    ledger_copy = directory / TASK07_LEDGER.name
    if (not ledger_copy.is_file() or recorded.get(TASK07_LEDGER.name) !=
            sha256_file(ledger_copy)):
        raise ValueError("任07总台账运行时复制原件的本地SHA已改变；不要求追加后的活台账逐字同")
    physics_file = directory / "resolved_physics.yaml"
    if (not physics_file.is_file()
            or load_yaml(physics_file) != resolve_physics_state()):
        raise ValueError("任07运行快照解析几何/材料/边界物理状态与可信结构化源不一致")
    return {
        "七份运营配置当前与复制双SHA": originals,
        "运行快照SHA256清单": sha256_file(manifest_file),
        "总台账运行时副本SHA256": sha256_file(ledger_copy),
        "运行解析物理SHA256": sha256_file(physics_file),
    }


def validate_completed_run(run: str | Path, *, seed: int) -> dict[str, Any]:
    path = Path(run)
    run = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if (seed not in range(5) or run.parent != TASK07_RUN_ROOT.resolve()
            or not run.name.startswith(f"修复后正式_E0_种子{seed}_")):
        raise ValueError("任07只读审核仅准入修复后录-0036的五个独立正式E0 seed目录")
    _fixed_schedule()
    report_file = run / "阶段报告.json"
    if not report_file.is_file():
        raise FileNotFoundError("任07完整正式五seed阶段报告缺失")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    c_epoch, j_epoch = report.get("校正实际轮次"), report.get("联合实际轮次")
    if (report.get("运行种子") != seed or report.get("运行臂") != "E0"
            or report.get("运行资格") != OFFICIAL
            or type(c_epoch) is not int or type(j_epoch) is not int
            or not 200 <= c_epoch <= 1500 or c_epoch % 10
            or not 200 <= j_epoch <= 500 or j_epoch % 10
            or report.get("原校正预算上限") != 1500
            or report.get("原受限联合预算上限") != 500
            or "正式受限联合已达同组截止" not in report.get("状态", "")
            or report.get("真实最后状态") != "阶段_训练末.pt"
            or report.get("旧test_Data温度标签读取") is not False):
        raise ValueError("任07CPU短诊断、缺真实联合末状态或两阶段未到合法早停/预算不可审核")
    sources = validate_task07_sources()
    source = sources[seed]
    registry_sha = report.get("正式预登记配置SHA256")
    registry = _verify_registry(sources, registry_sha)
    snapshot = _verify_snapshot(run)
    if (
        report.get("本seed源LF检查点SHA256") != source.lf_checkpoint_sha256
        or report.get("本seed真实LF初始张量SHA256") != source.lf_tensor_sha256
        or report.get("本seed历史HF架构视图SHA256") != source.hf_checkpoint_sha256
        or not _materials_valid(report.get("LF合法验证初态Cu_SiC节点及真实体积RMSE_摄氏度"))
    ):
        raise ValueError("任07训练阶段报告本seed配对V4原LF/HF来源和合法LF初态四口径不符")
    initial_file = run / "阶段_初始.pt"
    if not initial_file.is_file():
        raise FileNotFoundError("任07真实本seed第0轮完整状态缺失")
    initial = torch.load(initial_file, map_location="cpu", weights_only=False)
    _verify_stage(initial, sources, seed=seed, registry_sha=registry_sha,
                  expected_stage="task07_correction", correction_epoch=0, joint_epoch=0)
    initial_meta = initial["metadata"]
    if (
        not math.isclose(float(initial_meta.get("初始合法HF选分_摄氏度", math.inf)),
                         float(report.get("初始HF合法选分_摄氏度", math.inf)), abs_tol=1e-4)
        or not math.isclose(float(initial_meta.get("物理最佳独立损失", math.inf)),
                           float(report.get("物理最佳独立损失", math.inf)), abs_tol=1e-4)
        and report.get("物理最佳全局轮次") == 0
    ):
        raise ValueError("任07正式第0轮HF初分或原物理初分与合法报告源不一致")
    log_file = run / "training.jsonl"
    if not log_file.is_file():
        raise FileNotFoundError("任07未完成两段HF/LF训练JSONL真实日志")
    try:
        rows = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    except json.JSONDecodeError as exc:
        raise ValueError("任07训练日志尾行损坏或混入未提交的非法JSON") from exc
    if len(rows) != c_epoch + j_epoch:
        raise ValueError("任07训练日志缺行或包含未提交尾行，不能掩盖真实阶段末轮次")
    stages = {
        "观测最佳完整状态": ("阶段_观测最佳.pt", report.get("观测最佳全局轮次")),
        "物理最佳完整状态": ("阶段_物理最佳.pt", report.get("物理最佳全局轮次")),
        "校正真实末": ("阶段_校正末.pt", c_epoch),
        "联合初始": ("阶段_联合初始.pt", c_epoch),
        "最近完整状态": ("阶段_最近.pt", c_epoch + j_epoch),
        "联合专名真实末": ("阶段_联合末.pt", c_epoch + j_epoch),
        "真实训练末": ("阶段_训练末.pt", c_epoch + j_epoch),
    }
    artifacts, hashes = {"初始完整状态": initial}, {
        "初始完整状态": sha256_file(initial_file),
    }
    for label, (filename, global_epoch) in stages.items():
        if type(global_epoch) is not int or not 0 <= global_epoch <= c_epoch + j_epoch:
            raise ValueError("任07最佳和末状态报告全局轮次未落在真实已完成阶段")
        complete = run / filename
        if not complete.is_file():
            raise FileNotFoundError(f"任07真实完整阶段工件缺失：{filename}")
        value = torch.load(complete, map_location="cpu", weights_only=False)
        which = "task07_correction" if global_epoch <= c_epoch and label != "联合初始" else "task07_restricted_joint"
        _verify_stage(value, sources, seed=seed, registry_sha=registry_sha,
                      expected_stage=which, correction_epoch=min(global_epoch, c_epoch),
                      joint_epoch=max(global_epoch - c_epoch, 0))
        artifacts[label], hashes[label] = value, sha256_file(complete)
    terminal = artifacts["真实训练末"]
    derived = _verify_history(
        rows, seed=seed, correction_epoch=c_epoch, joint_epoch=j_epoch,
        source_lf_sha=source.lf_tensor_sha256,
        terminal_lf_sha=terminal["metadata"]["当前真实LF张量SHA256"],
        initial_score=initial_meta["初始合法HF选分_摄氏度"],
        initial_physical=initial_meta["物理最佳独立损失"],
    )
    if (
        derived["观测最佳全局轮次"] != report.get("观测最佳全局轮次")
        or derived["物理最佳全局轮次"] != report.get("物理最佳全局轮次")
        or report.get("累计实际消耗") != derived["累计实际消耗"]
        or report.get("最近一轮真实入场证据") != rows[-1]
        or not math.isclose(float(report.get("观测最佳HF合法选分_摄氏度", math.inf)),
                           derived["观测最佳选分_摄氏度"], abs_tol=1e-4)
        or not math.isclose(float(report.get("物理最佳独立损失", math.inf)),
                           derived["物理最佳独立损失"], abs_tol=1e-4)
        or any(not math.isclose(
            report["LF合法验证初态Cu_SiC节点及真实体积RMSE_摄氏度"][material][metric],
            derived["LF初始四口径"][material][metric], abs_tol=1e-4,
        ) for material in ("Cu", "SiC") for metric in ("node", "volume"))
        or any(not math.isclose(
            report["LF合法验证本轮Cu_SiC节点及真实体积RMSE_摄氏度"][material][metric],
            derived["LF末态四口径"][material][metric], abs_tol=1e-4,
        ) for material in ("Cu", "SiC") for metric in ("node", "volume"))
        or report["LF逐材料节点和真实体积5%护栏"].get(
            "LF两材料节点与真实体积均守住5%护栏"
        ) is not derived["LF末次保持资格"]["LF两材料节点与真实体积均守住5%护栏"]
    ):
        raise ValueError("任07历史合法观测/物理最佳或LF四口径资格与两阶段全量日志不符")
    old = _old_auditor()
    identical = ("model_state", "optimizer_state", "random_state",
                 "parameter_requires_grad", "metadata")
    if any(not old._same(terminal[key], artifacts[label][key])
           for label in ("最近完整状态", "联合专名真实末") for key in identical):
        raise ValueError("任07最近态、联合专名真末及训练真末模型/AdamW/四RNG非同一完整状态")
    correction = artifacts["校正真实末"]
    joined = artifacts["联合初始"]
    if (
        correction["metadata"]["校正实际截止轮次"] != c_epoch
        or not old._same(correction["model_state"], joined["model_state"])
        or any(not old._same(correction["random_state"][rng],
                             joined["random_state"][rng])
               for rng in ("python", "numpy", "torch_cuda"))
        or not old._same(correction["optimizer_state"]["state"],
                         joined["optimizer_state"]["state"])
        or joined["metadata"].get("已提交正式校正末原件SHA256") != hashes["校正真实末"]
        or terminal["metadata"].get("已提交正式校正末原件SHA256") != hashes["校正真实末"]
    ):
        raise ValueError("任07校正真实末至联合初始不得重建HF动量、LF原张量或非CPU预加载随机源")
    for label in ("观测最佳完整状态", "物理最佳完整状态"):
        record = artifacts[label]["metadata"]
        epoch = report["观测最佳全局轮次" if label.startswith("观测") else "物理最佳全局轮次"]
        prefix = "观测最佳" if label.startswith("观测") else "物理最佳"
        if (
            record[f"{prefix}全局轮次"] != epoch
            or terminal["metadata"][f"{prefix}全局轮次"] != epoch
            or not math.isclose(
                float(record["观测最佳选分_摄氏度" if label.startswith("观测") else "物理最佳独立损失"]),
                report["观测最佳HF合法选分_摄氏度" if label.startswith("观测") else "物理最佳独立损失"],
                abs_tol=1e-4,
            )
            or (epoch < c_epoch + j_epoch and terminal["metadata"].get(
                f"已提交旧{stages[label][0]}SHA256"
            ) != hashes[label])
        ):
            raise ValueError("任07历史最佳完整状态与末态所引用的原件SHA/得分/轮次不同")
    view_file = run / "best.pt"
    if not view_file.is_file():
        raise FileNotFoundError("任07旧HF架构的新最佳轻量视图缺失")
    view = torch.load(view_file, map_location="cpu", weights_only=False)
    lineage = view.get("任07新HF模型视图来源", {})
    if (
        view.get("epoch") != derived["观测最佳全局轮次"]
        or view.get("correction_model_kwargs") != source.correction_model_kwargs
        or view.get("low_fidelity_model_kwargs") != source.lf_model_kwargs
        or view.get("scales") != source.scales
        or lineage.get("训练种子") != seed or lineage.get("运行臂") != "E0"
        or lineage.get("正式预登记配置SHA256") != registry_sha
        or lineage.get("V4_B0来源清单SHA256") != MANIFEST_SHA256
        or lineage.get("本seed真实LF检查点SHA256") != source.lf_checkpoint_sha256
        or lineage.get("本seed历史HF架构视图SHA256") != source.hf_checkpoint_sha256
        or lineage.get("运行资格") != OFFICIAL
        or lineage.get("旧test_Data温度标签读取") is not False
        or not old._same(view.get("model_state"), artifacts["观测最佳完整状态"]["model_state"])
        or not math.isclose(float(view.get("validation_selection_score_c", math.inf)),
                            derived["观测最佳选分_摄氏度"], abs_tol=1e-4)
    ):
        raise ValueError("任07最佳HF模型视图与五源SHA、校正架构、合法HF观测最佳原件非同模型")
    hashes["观测最佳模型视图"] = sha256_file(view_file)
    return {
        "运行目录": run, "运行种子": seed,
        "校正实际轮次": c_epoch, "联合实际轮次": j_epoch,
        "运行报告": report, "预登记": registry,
        "配对来源": source, "快照SHA256": snapshot,
        "训练日志SHA256": sha256_file(log_file),
        "阶段报告SHA256": sha256_file(report_file),
        "状态文件": {"best": view_file, "final": run / "阶段_训练末.pt"},
        "完整状态文件": {
            "best": run / "阶段_观测最佳.pt",
            "physical": run / "阶段_物理最佳.pt",
            "final": run / "阶段_训练末.pt",
        },
        "阶段SHA256": hashes, "日志重推": derived,
    }


def _verify_energy_raw(audit: dict[str, Any]) -> dict[str, float]:
    powers, times, orders = _fixed_schedule()
    energy, divergence = audit.get("原始能量"), audit.get("原始散度")
    mapped, decomposition, summary = (audit.get("指标明细"), audit.get("物理分解"),
                                      audit.get("汇总"))
    if (not isinstance(energy, list) or not isinstance(divergence, list)
            or len(energy) != 60 or len(divergence) != 60
            or mapped is None or decomposition is None
            or mapped.height != 30 or decomposition.height != 30
            or not isinstance(summary, dict)):
        raise ValueError("任07能源必须保留固定30点双阶各60条原始能量/散度以及30行汇总")
    expected = [(p, t, q) for p in powers for t in times for q in orders]
    watt_keys = ("absorbed_power_w", "storage_rate_w", "cooling_heat_w",
                 "convection_heat_w", "radiation_heat_w", "balance_w",
                 "relative_balance_denominator_w", "relative_balance")
    vjd_keys = ("integrated_pde_residual_w", "interface_two_sided_flux_w",
                "boundary_flux_residual_w", "explained_engineering_balance_w",
                "engineering_balance_w", "engineering_explanation_gap_w")

    def same(a: float, b: float, *, denominator: bool = False) -> bool:
        return math.isclose(a, b, rel_tol=1e-12 if denominator else 1e-10,
                            abs_tol=1e-9 if denominator else 1e-7)

    lower_gap, higher_gap, neighbors = [], [], []
    high_energy, high_divergence = [], []
    for index, (power, time_s, order) in enumerate(expected):
        raw, terms = energy[index], divergence[index]
        if (not isinstance(raw, dict) or not isinstance(terms, dict)
                or any(row.get("power_w") != power or row.get("time_s") != time_s
                       or row.get("quadrature_order") != order for row in (raw, terms))
                or any(type(raw.get(key)) not in (float, int)
                       or not math.isfinite(raw[key]) for key in watt_keys)
                or any(type(terms.get(key)) not in (float, int)
                       or not math.isfinite(terms[key]) for key in vjd_keys)):
            raise ValueError("任07原能源/散度30点双阶网格来源、原瓦数或有限值不一致")
        absorbed, storage = raw["absorbed_power_w"], raw["storage_rate_w"]
        cooling = raw["cooling_heat_w"] + raw["convection_heat_w"] + raw["radiation_heat_w"]
        denominator = max(abs(absorbed), abs(storage), abs(cooling), 1e-12)
        v, j, d = (terms["integrated_pde_residual_w"],
                   terms["interface_two_sided_flux_w"], terms["boundary_flux_residual_w"])
        explained = v - j - d
        if (absorbed <= 0 or raw["relative_balance_denominator_w"] <= 0
                or not same(raw["balance_w"], storage + cooling - absorbed)
                or not same(raw["relative_balance_denominator_w"], denominator,
                            denominator=True)
                or not same(raw["relative_balance"], raw["balance_w"] / denominator)
                or not same(terms["engineering_balance_w"], raw["balance_w"])
                or not same(terms["explained_engineering_balance_w"], explained)
                or not same(terms["engineering_explanation_gap_w"],
                            raw["balance_w"] - explained)):
            raise ValueError("任07原瓦数B、原定义相对分母及两阶散度V-J-D必须逐点一致")
        if order == 16:
            lower_gap.append(abs(terms["engineering_explanation_gap_w"]))
        else:
            higher_gap.append(abs(terms["engineering_explanation_gap_w"]))
            high_energy.append(raw)
            high_divergence.append(terms)
            previous = energy[index - 1]
            neighbors.append(max(abs(raw[key] - previous[key]) for key in watt_keys[:6]) /
                             abs(absorbed))
    if len(lower_gap) != 30 or len(higher_gap) != 30:
        raise ValueError("任07第16阶散度积分剩余差及64阶V-J-D均须原样记录30点")
    csv_rows, vjd_rows = mapped.iter_rows(named=True), decomposition.iter_rows(named=True)
    for raw, terms, line, vjd in zip(high_energy, high_divergence, csv_rows, vjd_rows):
        for key, name in (
            ("power_w", "功率_瓦"), ("time_s", "时刻_秒"),
            ("absorbed_power_w", "吸收功率_瓦"), ("storage_rate_w", "储能率_瓦"),
            ("cooling_heat_w", "水冷散热_瓦"), ("convection_heat_w", "对流散热_瓦"),
            ("radiation_heat_w", "辐射散热_瓦"), ("balance_w", "平衡_瓦"),
            ("relative_balance_denominator_w", "原定义相对平衡分母_瓦"),
            ("relative_balance", "原定义相对平衡"),
        ):
            if name not in line or not same(float(line[name]), raw[key],
                                            denominator=key == "relative_balance_denominator_w"):
                raise ValueError("任07三十行工程指标CSV与第64阶原瓦数/相对分母不一致")
        for key, name in (
            ("power_w", "功率_瓦"), ("time_s", "时刻_秒"),
            ("integrated_pde_residual_w", "体积分残差V_瓦"),
            ("interface_two_sided_flux_w", "界面双侧通量J_瓦"),
            ("boundary_flux_residual_w", "边界失配D_瓦"),
            ("engineering_balance_w", "原工程平衡_瓦"),
            ("explained_engineering_balance_w", "解释工程平衡_瓦"),
            ("engineering_explanation_gap_w", "工程解释剩余差_瓦"),
        ):
            if name not in vjd or not same(float(vjd[name]), terms[key]):
                raise ValueError("任07三十行物理分解CSV与第64阶原始散度V-J-D不一致")
    computed = {
        "绝对平衡宏均值_瓦": float(np.mean([abs(r["balance_w"]) for r in high_energy])),
        "绝对平衡95分位_瓦": float(np.percentile(
            [abs(r["balance_w"]) for r in high_energy], 95)),
        "最大相邻阶变化对吸收功率比": max(neighbors),
        "最大分解剩余差_瓦": max(higher_gap),
    }
    if (summary.get("原定义相对平衡分母已保持") is not True
            or summary.get("工程平衡与原瓦数逐行一致") is not True
            or summary.get("功率时刻审核行数", 30) != 30
            or any(not same(float(summary.get(name, math.inf)), value)
                   for name, value in computed.items())):
        raise ValueError("任07能源汇总/30点双阶求积原瓦数或相对分母与原件不一致")
    return {
        "16阶V-J-D散度积分剩余差最大_瓦": max(lower_gap),
        "16阶V-J-D散度积分剩余差宏均值_瓦": sum(lower_gap) / 30,
        "64阶V-J-D散度积分剩余差最大_瓦": max(higher_gap),
        "64阶V-J-D散度积分剩余差宏均值_瓦": sum(higher_gap) / 30,
    }


def _recompute_physical_best(
    qualified: dict[str, Any], selected_model: torch.nn.Module, device: torch.device,
) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("任07正式物理最佳必须在原GPU CUDA同设备seed260908的256配点重算；CPU同种子点不同")
    physical = torch.load(qualified["完整状态文件"]["physical"],
                          map_location="cpu", weights_only=False)
    physical_model = copy.deepcopy(selected_model)
    physical_model.load_state_dict(physical["model_state"], strict=True)
    physical_model.eval()
    snapshot = qualified["运行目录"] / "config_snapshot"
    weights = load_yaml(snapshot / "training.yaml")["loss_weights"]
    materials = load_materials(str(snapshot / "materials.yaml"))
    boundaries = load_resolved_boundary_conditions(str(snapshot / "boundary_conditions.yaml"))
    physics = PhysicsLossComputer(materials, boundaries, PhysicsLossWeights(
        pde=float(weights["pde"]), boundary=float(weights["boundary"]),
        initial=float(weights["initial"]), interface=float(weights["interface"]),
    ))
    actual = physics(physical_model, sample_collocation(256, device, seed=260908))
    components = {name: float(value.detach()) for name, value in actual.items()}
    expected = float(qualified["运行报告"]["物理最佳独立损失"])
    if (not math.isfinite(components["physics_total"])
            or not math.isclose(components["physics_total"], expected,
                                rel_tol=0.0, abs_tol=1e-4)):
        raise ValueError("任07物理最佳独立原件固定CUDA物理配点损失与训练报告不相等")
    return components


def _recompute_selected_hf_lf(
    qualified: dict[str, Any], model: torch.nn.Module, device: torch.device,
    *, state: str,
) -> dict[str, Any]:
    if state not in ("best", "final"):
        raise ValueError("任07合法宏HF/LF须指定历史最佳或真实训练末态")
    source, report, run = qualified["配对来源"], qualified["运行报告"], qualified["运行目录"]
    if (list(source.hf_validation_powers_w) != [115.2, 403.0, 630.5]
            or list(source.lf_validation_powers_w) !=
            [90.0, 170.0, 250.0, 330.0, 410.0, 490.0, 570.0, 650.0, 730.0, 800.0]):
        raise ValueError("任07合法HF三功率与LF十功率须来自固定V4原划分")
    loader_data = _ir_dataset("validation", source.hf_validation_powers_w)
    legal_sensor = _sensor_tensors(device, split="validation",
                                   powers_w=source.hf_validation_powers_w)
    if len(loader_data) != 7272 or len(legal_sensor[0]) != 752:
        raise ValueError("任07合法HF验证7272个IR样本及752个传感器点不得缩水")
    selected, validation = _validation_selection(
        model, DataLoader(loader_data, batch_size=2048, shuffle=False),
        legal_sensor, device,
        load_yaml(run / "config_snapshot/training.yaml")["multifidelity_selection_weights"],
    )
    if state == "best":
        best_epoch = qualified["日志重推"]["观测最佳全局轮次"]
        expected_score = qualified["日志重推"]["观测最佳选分_摄氏度"]
        if best_epoch == 0 or best_epoch <= qualified["校正实际轮次"]:
            expected_lf = qualified["日志重推"]["LF初始四口径"]
            legal_keep = True
        else:
            record = json.loads((run / "training.jsonl").read_text(encoding="utf-8")
                                .splitlines()[best_epoch - 1])
            expected_lf = record["LF合法验证逐材料节点与体积RMSE_摄氏度"]
            legal_keep = record["LF逐材料节点和真实体积5%护栏"][
                "LF两材料节点与真实体积均守住5%护栏"
            ]
            if legal_keep is not True:
                raise ValueError("任07历史最佳合法HF选分不得引用联合LF四口径护栏失败轮次")
    else:
        last = report["最近一轮真实入场证据"]
        expected_score = last.get("HF合法验证选分_摄氏度")
        expected_lf = report["LF合法验证本轮Cu_SiC节点及真实体积RMSE_摄氏度"]
        legal_keep = report["LF逐材料节点和真实体积5%护栏"][
            "LF两材料节点与真实体积均守住5%护栏"
        ]
    if (type(selected) not in (float, int) or type(expected_score) not in (float, int)
            or not math.isfinite(selected) or not math.isclose(
                selected, expected_score, rel_tol=0.0, abs_tol=1e-4,
            )):
        raise ValueError("任07选中模型合法HF macro_v1独立重算选分与最佳/末态真实报告不一致")
    actual_lf = _lf_material_validation(model, list(source.lf_validation_powers_w), device)
    if (not _materials_valid(actual_lf) or not _materials_valid(expected_lf)
            or any(not math.isclose(actual_lf[name][key], expected_lf[name][key],
                                    rel_tol=0.0, abs_tol=1e-4)
                   for name in ("Cu", "SiC") for key in ("node", "volume"))):
        raise ValueError("任07选中模型Cu/SiC真实LF十功率逐材料node与轴对称volume重算不等于历史原数")
    independently_qualified = task04_lf_keep_guardrail(
        qualified["日志重推"]["LF初始四口径"], actual_lf,
    )["LF两材料节点与真实体积均守住5%护栏"]
    if (independently_qualified is not legal_keep):
        raise ValueError("任07选中模型LF逐材料节点和真实体积5%保持资格与独立四口径重算不一致")
    return {
        "合法HF验证功率_瓦": list(source.hf_validation_powers_w),
        "合法HF样本点数": len(loader_data), "合法HF传感器点数": len(legal_sensor[0]),
        "合法HF原macro_v1选分独立重算_摄氏度": selected,
        "合法HF验证原分模态": validation,
        "合法LF验证功率数": len(source.lf_validation_powers_w),
        "合法LF逐材料节点和真实体积独立重算_摄氏度": actual_lf,
        "LF逐材料节点和真实体积5%原护栏实际资格": independently_qualified,
    }


def _load_selected_model(
    qualified: dict[str, Any], device: torch.device, *, state: str,
) -> torch.nn.Module:
    if state not in ("best", "final"):
        raise ValueError("任07仅加载合格best或真实训练末完整状态")
    stage = qualified["完整状态文件"][state]
    expected_sha = qualified["阶段SHA256"][
        "观测最佳完整状态" if state == "best" else "真实训练末"
    ]
    if sha256_file(stage) != expected_sha:
        raise ValueError("任07完整状态静态核验之后模型原件被更改，不能写名义能源")
    original = torch.load(stage, map_location="cpu", weights_only=False)
    model = fork_task07_initialization(
        validate_task07_sources(), qualified["运行种子"], "E0", torch.device("cpu"),
    ).model
    model.load_state_dict(original["model_state"], strict=True)
    model.eval()
    return model.to(device)


def audit_completed_state(run: str | Path, *, seed: int, state: str,
                          output: str | Path, device_name: str | None = None) -> dict[str, Any]:
    if state not in ("best", "final"):
        raise ValueError("任07仅分别审核合法观测最佳和真实最终末态")
    destination = Path(output)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    if destination.exists():
        raise FileExistsError(f"任07已有独立审核工件不可覆盖：{destination}")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("任07正式30点双阶能源及256固定物理配点仅准原GPU CUDA，CPU诊断不可冒充")
    qualified = validate_completed_run(run, seed=seed)
    model = _load_selected_model(qualified, device, state=state)
    selected_evidence = _recompute_selected_hf_lf(qualified, model, device, state=state)
    physical_components = _recompute_physical_best(qualified, model, device)
    model.requires_grad_(False).to(dtype=torch.float64).eval()
    snapshot = qualified["运行目录"] / "config_snapshot"
    powers, times, orders = _fixed_schedule()
    audit = audit_schedule_energy(
        model, load_materials(str(snapshot / "materials.yaml")),
        load_resolved_boundary_conditions(str(snapshot / "boundary_conditions.yaml")),
        AxisymmetricGeometry.from_config(load_yaml(snapshot / "geometry.yaml")),
        powers_w=powers, times_s=times, orders=tuple(orders), device=device,
    )
    computed_gaps = _verify_energy_raw(audit)
    stage_originals = {
        "初始完整状态": "阶段_初始.pt",
        "观测最佳完整状态": "阶段_观测最佳.pt",
        "物理最佳完整状态": "阶段_物理最佳.pt",
        "校正真实末": "阶段_校正末.pt",
        "联合初始": "阶段_联合初始.pt",
        "最近完整状态": "阶段_最近.pt",
        "联合专名真实末": "阶段_联合末.pt",
        "真实训练末": "阶段_训练末.pt",
        "观测最佳模型视图": "best.pt",
    }
    if (
        any(sha256_file(qualified["运行目录"] / name) != qualified["阶段SHA256"][label]
            for label, name in stage_originals.items())
        or sha256_file(MANIFEST_PATH) != MANIFEST_SHA256
        or sha256_file(qualified["配对来源"].lf_checkpoint_path) !=
        qualified["配对来源"].lf_checkpoint_sha256
        or sha256_file(qualified["配对来源"].hf_checkpoint_path) !=
        qualified["配对来源"].hf_checkpoint_sha256
        or sha256_file(qualified["运行目录"] / "training.jsonl") !=
        qualified["训练日志SHA256"]
        or sha256_file(qualified["运行目录"] / "阶段报告.json") !=
        qualified["阶段报告SHA256"]
        or sha256_file(TASK07_REGISTRATION) != qualified["预登记"]["sha256"]
        or _verify_snapshot(qualified["运行目录"]) != qualified["快照SHA256"]
    ):
        raise ValueError("任07完整状态、历史最佳、七源快照或报告在审核过程中被改动，不得写盘")
    summary = audit["汇总"]
    chosen_sha = qualified["阶段SHA256"][
        "观测最佳模型视图" if state == "best" else "真实训练末"
    ]
    payload = {
        "中文说明": (
            "任07修复后录-0036正式五种子E0中本seed真实校正与受限联合已到预登记早停/轮次截止；"
            "历史合法HF观测最佳与独立物理最佳分别溯源，选中最佳或真实训练末原件单独审核。"
            "仅验证当前名义模型场固定30功率时刻点16/64阶原工程瓦数及V-J-D数值分解；"
            "原相对平衡分母不变，16阶/64阶散度积分剩余差逐点原值保留。"
            "此模型数值自洽不等于原FEM盘原始热预算已修复，也不含内部真实温度证明。"
        ),
        "运行种子": seed, "运行臂": "E0", "审核状态": state,
        "运行目录": str(qualified["运行目录"]),
        "选择模型SHA256": chosen_sha,
        "V4_B0五种子配对来源清单SHA256": MANIFEST_SHA256,
        "本seed源LF模型检查点SHA256": qualified["配对来源"].lf_checkpoint_sha256,
        "本seed历史HF架构视图SHA256": qualified["配对来源"].hf_checkpoint_sha256,
        "本seedLF初态全张量SHA256": qualified["配对来源"].lf_tensor_sha256,
        "本seed来源协议SHA256": qualified["配对来源"].source_metadata_sha256,
        "任07修复后录-0036正式预登记SHA256": qualified["预登记"]["sha256"],
        "继承任04独立能源预登记SHA256": TASK04_CONFIG_SHA256,
        "继承任06独立能源预登记SHA256": TASK06_CONFIG_SHA256,
        "校正实际截止轮次": qualified["校正实际轮次"],
        "受限联合实际截止轮次": qualified["联合实际轮次"],
        "训练日志SHA256": qualified["训练日志SHA256"],
        "阶段报告SHA256": qualified["阶段报告SHA256"],
        "完整阶段原件及最佳视图SHA256": qualified["阶段SHA256"],
        "七运营配置及解析物理运行快照SHA256": qualified["快照SHA256"],
        "本审核源码SHA256": {
            filename: sha256_file(PROJECT_ROOT / filename) for filename in (
                "src/sic_cu/eval/energy_v5.py", "scripts/21_audit_task04_energy.py",
                "scripts/26_audit_task07_states.py", "src/sic_cu/train/task07_source.py",
                "src/sic_cu/train/task07_formal.py", "src/sic_cu/models/multifidelity.py",
            )
        },
        "本次合法HF/LF独立重算": selected_evidence,
        "物理最佳固定GPU_CUDA种子260908及256配点独立重算损失": physical_components,
        "历史最佳合法HF全局轮次": qualified["日志重推"]["观测最佳全局轮次"],
        "历史物理最佳全局轮次": qualified["日志重推"]["物理最佳全局轮次"],
        "功率_瓦": powers, "时刻_秒": times, "求积阶数": orders,
        "物理审核点数": len(powers) * len(times),
        "LF末态四口径护栏不通过时绝不判采用": True,
        "本审核模型LF逐材料节点和真实体积5%保持资格": selected_evidence[
            "LF逐材料节点和真实体积5%原护栏实际资格"
        ],
        "名义能量不证明原FEM热预算或内部温度真值": True,
        **computed_gaps,
        **summary,
    }
    _old_auditor().write_energy_evidence(destination, audit, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="任07五种子E0真实最佳/末态30点16/64阶原瓦数独立审核")
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=range(5))
    parser.add_argument("--state", required=True, choices=("best", "final"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="仅CPU只读验证完整阶段来源，不产出名义能源")
    arguments = parser.parse_args()
    if arguments.diagnostic_only:
        if arguments.device not in (None, "cpu"):
            raise ValueError("任07CPU静态诊断不得指定GPU CUDA或写入正式能源")
        qualified = validate_completed_run(arguments.run, seed=arguments.seed)
        print(json.dumps({
            "运行种子": qualified["运行种子"],
            "运行资格": "仅CPU静态完整阶段诊断，未独立审核HF/LF标签与名义能源",
            "校正实际轮次": qualified["校正实际轮次"],
            "联合实际轮次": qualified["联合实际轮次"],
            "观测最佳全局轮次": qualified["日志重推"]["观测最佳全局轮次"],
            "物理最佳全局轮次": qualified["日志重推"]["物理最佳全局轮次"],
        }, ensure_ascii=False))
        return
    result = audit_completed_state(arguments.run, seed=arguments.seed,
                                   state=arguments.state, output=arguments.output,
                                   device_name=arguments.device)
    print(json.dumps({key: result[key] for key in (
        "运行种子", "审核状态", "绝对平衡宏均值_瓦", "绝对平衡95分位_瓦",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
