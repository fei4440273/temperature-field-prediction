#!/usr/bin/env python
"""任-06真实600轮三臂最佳/末态的独立名义工程能量审核。"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import build_power_splits
from sic_cu.eval.energy_v5 import AxisymmetricGeometry, audit_schedule_energy
from sic_cu.losses import PhysicsLossComputer, PhysicsLossWeights
from sic_cu.physics import sample_collocation
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.train.multifidelity import _ir_dataset, _sensor_tensors
from sic_cu.train.task04_joint import _lf_material_validation, _validation_selection
from sic_cu.train.task06_time_features import (
    ARMS, CONFIG_SHA, _actual_lf_tensor_sha256, _registered_tau,
    fork_task06_model, load_task06_model_view, task06_model_kwargs, validate_response_contract,
    validate_task06_source,
)


_task04_file = PROJECT_ROOT / "scripts/21_audit_task04_energy.py"
_task04_spec = importlib.util.spec_from_file_location("task04_energy_audit", _task04_file)
if _task04_spec is None or _task04_spec.loader is None:
    raise ImportError(f"项目内任04能量审核器文件不可加载：{_task04_file}")
TASK04_ENERGY = importlib.util.module_from_spec(_task04_spec)
_task04_spec.loader.exec_module(TASK04_ENERGY)
OFFICIAL = "单种子正式同源600轮先导候选；须三臂审计后再决定采用"
HF_POINTS = 29_593
SENSOR_POINTS = 44_775


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _budget(epoch: int) -> dict[str, int]:
    return {
        "HF训练观测点": epoch * HF_POINTS,
        "HF训练传感器点": epoch * SENSOR_POINTS,
        "LF仿真温度训练点": 0,
        "物理配点": epoch * 256,
        "HF观测优化步": epoch * 15,
        "物理优化步": epoch,
    }


def _conditions(config: dict[str, Any]) -> tuple[list[float], list[float], list[int]]:
    fixed = config.get("独立30点能量", {})
    powers, times, orders = fixed.get("功率_瓦"), fixed.get("时刻_秒"), fixed.get("阶数")
    if (
        config.get("任务") != "任-06" or config.get("每臂追加HF校正轮数") != 600
        or config.get("HF学习率") != 1e-4 or config.get("LF学习率") != 0.0
        or powers != TASK04_ENERGY.FIXED_POWERS or times != TASK04_ENERGY.FIXED_TIMES
        or orders != [16, 64] or len(powers or []) * len(times or []) != 30
        or fixed.get("三臂最佳和末态分别审计") is not True
    ):
        raise ValueError("任-06能量审核只能采用事前登记的30点16/64阶与600轮三臂双状态")
    return powers, times, orders


def _check_log(run: Path, arm: str, lf_sha: str, lf_count: int,
               report: dict[str, Any], *, initial_score: float,
               initial_physical: float) -> str:
    log = run / "training.jsonl"
    if not log.is_file():
        raise FileNotFoundError("任-06真实600轮training.jsonl训练日志缺失")
    best_epoch, physical_epoch = (
        report["观测最佳任06轮次"], report["物理最佳任06轮次"],
    )
    frozen_lf = report["LF合法验证逐材料节点及真实体积RMSE_摄氏度"]
    derived_best, derived_physical = 0, 0
    derived_score, derived_physical_score = initial_score, initial_physical
    last = 0
    with log.open(encoding="utf-8") as handle:
        for epoch, line in enumerate(handle, 1):
            last = epoch
            if epoch > 600:
                raise ValueError("任-06训练日志超过原600轮预算")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"任-06第{epoch}轮训练日志不是完整JSON") from exc
            if (
                row.get("epoch") != epoch or row.get("任06轮次") != epoch
                or row.get("运行臂") != arm or row.get("源任04完整实际阶段轮次") != 120
                or row.get("HF观测优化步") != 15 or row.get("物理优化步") != 1
                or row.get("物理配点") != 256
                or row.get("HF训练观测点") != HF_POINTS
                or row.get("HF训练传感器点") != SENSOR_POINTS
                or row.get("LF仿真训练来源功率数") != 60
                or row.get("LF仿真温度训练点") != 0
                or row.get("LF实际冻结参数数") != lf_count
                or row.get("LF共同真实张量SHA256") != lf_sha
                or row.get("HF学习率") != 1e-4 or row.get("LF学习率") != 0.0
                or row.get("累计实际消耗") != _budget(epoch)
            ):
                raise ValueError(f"任-06第{epoch}轮HF样本缩水、物理配点或LF冻结预算异常")
            due = epoch % 10 == 0
            score = row.get("HF合法验证选分_摄氏度")
            modalities = row.get("HF合法验证分模态RMSE_摄氏度")
            materials = row.get("LF合法验证材料RMSE_摄氏度")
            physics = row.get("独立局部物理损失")
            if due:
                if (
                    type(score) not in (float, int) or not math.isfinite(score)
                    or not isinstance(modalities, dict)
                    or set(modalities) != {"top", "hot", "cold"}
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           or value < 0 for value in modalities.values())
                    or not isinstance(materials, dict) or set(materials) != {"Cu", "SiC"}
                    or any(
                        not isinstance(materials[m], dict)
                        or set(materials[m]) != {"node", "volume"}
                        or any(type(materials[m][key]) not in (int, float)
                               or not math.isfinite(materials[m][key])
                               or materials[m][key] < 0 for key in ("node", "volume"))
                        for m in ("Cu", "SiC")
                    )
                    or not isinstance(physics, dict)
                    or not {"pde", "initial", "boundary", "interface", "physics_total"} <= set(physics)
                    or type(physics.get("physics_total")) not in (int, float)
                    or not math.isfinite(physics["physics_total"])
                    or any(type(physics[part]) not in (int, float)
                           or not math.isfinite(physics[part])
                           for part in ("pde", "initial", "boundary", "interface"))
                    or any(
                        not math.isclose(materials[m][key], frozen_lf[m][key], abs_tol=1e-4)
                        for m in ("Cu", "SiC") for key in ("node", "volume")
                    )
                ):
                    raise ValueError(f"任-06第{epoch}轮合法HF/LF冻结材料与物理验证或报告不一致")
                if score < derived_score - 1e-4:
                    derived_score, derived_best = score, epoch
                if physics["physics_total"] < derived_physical_score:
                    derived_physical_score, derived_physical = physics["physics_total"], epoch
            elif score is not None or materials is not None:
                raise ValueError("任-06未预登记的非10轮验证不得选择最佳")
            if epoch == 600 and report.get("最后一轮门禁证据") != row:
                raise ValueError("任-06完成报告末轮门禁证据与真实日志不同")
    if (
        last != 600 or derived_best != best_epoch or derived_physical != physical_epoch
        or not math.isclose(derived_score, report["观测最佳HF合法选分_摄氏度"], abs_tol=1e-4)
        or not math.isclose(derived_physical_score, report["物理最佳独立损失"], abs_tol=1e-4)
    ):
        raise ValueError("任-06真实600轮日志/全程观测与物理最佳合法轮次或分数不符")
    if report.get("累计实际消耗") != _budget(600):
        raise ValueError("任-06报告中的HF观察或物理实际样本累计缩水")
    return sha256_file(log)


def _check_state(path: Path, *, epoch: int, arm: str, config: dict[str, Any],
                 source: dict[str, Any], lf_sha: str, kwargs: dict[str, Any],
                 expected_group: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"任-06真实完整最佳或末状态原件缺失：{path.name}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    metadata = state.get("metadata", {})
    tau = None if arm == "E0" else list(_registered_tau(config, arm))
    if (
        state.get("training_state_schema_version") != 1
        or state.get("stage") != "correction_time_features"
        or state.get("epoch") != epoch
        or state.get("budget") != {"任06HF校正先导轮次": 600}
        or metadata.get("任06运行臂") != arm
        or metadata.get("任06预登记配置SHA256") != CONFIG_SHA
        or metadata.get("源任04第120轮完整状态SHA256") != config["共同完整训练状态_SHA256"]
        or metadata.get("源任04模型视图SHA256") != config["共同观测最佳模型视图_SHA256"]
        or metadata.get("LF共同真实张量SHA256") != lf_sha
        or metadata.get("运行资格") != OFFICIAL
        or metadata.get("响应tau秒") != tau
        or metadata.get("LF全部参数冻结") is not True
        or metadata.get("LF仿真温度进入HF监督") is not False
        or metadata.get("旧test_Data温度标签读取") is not False
        or metadata.get("累计实际消耗") != _budget(epoch)
    ):
        raise ValueError(f"任-06{path.name}的来源、tau、轮次或正式预算不符")
    model_state = state.get("model_state", {})
    validate_response_contract(model_state, kwargs, config, arm)
    source_lf = {
        name: value for name, value in source["model_state"].items()
        if name.startswith("low_fidelity_model.")
    }
    candidate_lf = {
        name: value for name, value in model_state.items()
        if name.startswith("low_fidelity_model.")
    }
    if (
        not source_lf or source_lf.keys() != candidate_lf.keys()
        or any(not TASK04_ENERGY._same(value, candidate_lf[name])
               for name, value in source_lf.items())
        or _actual_lf_tensor_sha256(model_state) != lf_sha
    ):
        raise ValueError("任-06 LF真实张量偏离任04联合最佳来源，不得冒用原LF SHA")
    named = state.get("parameter_requires_grad", {})
    if (
        set(named) != set(source["parameter_requires_grad"])
        or not set(named) <= set(model_state)
        or not all(name.startswith(("low_fidelity_model.", "correction.")) for name in named)
        or any(named[name] for name in named if name.startswith("low_fidelity_model."))
        or any(not named[name] for name in named if name.startswith("correction."))
    ):
        raise ValueError("任-06阶段LF非全部冻结或HF可训练参数名单异常")
    optimizer = state.get("optimizer_state", {})
    groups = optimizer.get("param_groups", [])
    hf_names = [name for name in named if name.startswith("correction.")]
    if (
        len(groups) != 1 or len(groups[0].get("params", [])) != len(hf_names)
        or len(set(groups[0]["params"])) != len(hf_names)
        or groups[0] != expected_group
    ):
        raise ValueError("任-06 HF AdamW完整参数组须等于训练器同源新建优化器")
    moments = optimizer.get("state", {})
    ids = groups[0]["params"]
    if (epoch == 0 and moments) or (epoch > 0 and set(moments) != set(ids)):
        raise ValueError("任-06第0轮新HF AdamW应无旧动量，真实后续步不能缺失")
    for ident, name in zip(ids, hf_names):
        if epoch == 0:
            continue
        value = moments[ident]
        if (
            not {"step", "exp_avg", "exp_avg_sq"} <= set(value)
            or float(value["step"]) != 16 * epoch
            or not isinstance(value["exp_avg"], torch.Tensor)
            or not isinstance(value["exp_avg_sq"], torch.Tensor)
            or value["exp_avg"].shape != model_state[name].shape
            or value["exp_avg_sq"].shape != model_state[name].shape
        ):
            raise ValueError("任-06 HF AdamW动量不是逐轮HF15加物理1步或与真实参数形状不符")
    rng = state.get("random_state", {})
    source_cuda = source["random_state"]["torch_cuda"]
    if (
        set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}
        or any(rng[name] is None for name in ("python", "numpy", "torch_cpu"))
        or not isinstance(rng["torch_cuda"], list)
        or len(rng["torch_cuda"]) != len(source_cuda)
        or not all(isinstance(item, torch.Tensor) and item.numel() for item in rng["torch_cuda"])
    ):
        raise ValueError("任-06四源RNG/CUDA随机状态缺失，CPU短诊断不得转正式状态")
    if epoch == 0:
        for name, old in source["model_state"].items():
            fresh = model_state.get(name)
            if name == "correction.0.weight" and arm != "E0":
                if (
                    fresh is None or fresh.shape != (128, 10)
                    or not TASK04_ENERGY._same(fresh[:, :6], old)
                    or torch.count_nonzero(fresh[:, 6:]).item() != 0
                ):
                    raise ValueError("任-06新增4响应通道应零初值、旧6通道应同任04真实源")
            elif not TASK04_ENERGY._same(old, fresh):
                raise ValueError("任-06共同第0轮模型初态不等于任04真实联合最佳")
    return state


def validate_completed_run(run: str | Path, *, arm: str) -> dict[str, Any]:
    """Reject incomplete 600-epoch arms before model differentiation."""
    if arm not in ARMS:
        raise ValueError("任-06独立能源审核仅接受E0/E1/E2三臂")
    run = _resolve(run)
    if not (run / "阶段报告.json").is_file():
        raise FileNotFoundError("任-06正式600轮阶段报告缺失")
    source, architecture, config = validate_task06_source()
    powers, times, orders = _conditions(config)
    source_path = _resolve(config["共同实际全模型起点"])
    TASK04_ENERGY.validate_completed_run(source_path.parent, arm="有限解冻")
    lf_sha = _actual_lf_tensor_sha256(source["model_state"])
    kwargs = task06_model_kwargs(architecture, config, arm)
    report_file = run / "阶段报告.json"
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if (
        report.get("运行臂") != arm or report.get("运行资格") != OFFICIAL
        or report.get("本臂实际完成轮次") != 600
        or report.get("原预登记预算") != 600
        or "真实完成单种子三臂原预算600轮" not in report.get("状态", "")
        or report.get("任06预登记配置SHA256") != CONFIG_SHA
        or report.get("源任04第120轮完整状态SHA256") != config["共同完整训练状态_SHA256"]
        or report.get("LF共同真实张量SHA256") != lf_sha
        or report.get("LF仿真训练温度进入HF监督") is not False
        or report.get("旧test_Data温度标签读取") is not False
        or report.get("真实最后状态") != "阶段_HF先导末.pt"
    ):
        raise ValueError("任-06只接受同源600轮正式候选，短诊断不可冒充")
    best_epoch = report.get("观测最佳任06轮次")
    physical_epoch = report.get("物理最佳任06轮次")
    if (
        type(best_epoch) is not int or not 0 <= best_epoch <= 600 or best_epoch % 10
        or type(physical_epoch) is not int or not 0 <= physical_epoch <= 600
        or physical_epoch % 10
        or any(type(report.get(name)) not in (int, float) or not math.isfinite(report[name])
               for name in ("观测最佳HF合法选分_摄氏度", "物理最佳独立损失"))
    ):
        raise ValueError("任-06最佳轮次须是真实第0轮或每10轮合法评估点")
    lf_metrics = report.get("LF合法验证逐材料节点及真实体积RMSE_摄氏度", {})
    if not isinstance(lf_metrics, dict) or set(lf_metrics) != {"Cu", "SiC"} or any(
        not isinstance(lf_metrics[material], dict)
        or set(lf_metrics[material]) != {"node", "volume"}
        for material in ("Cu", "SiC")
    ):
        raise ValueError("任-06必须报告LF合法验证两材料node和真实volume误差")
    count = sum(name.startswith("low_fidelity_model.")
                for name in source["parameter_requires_grad"])
    _, expected_optimizer = fork_task06_model(
        source, architecture, config, arm, torch.device("cpu"),
    )
    expected_group = expected_optimizer.state_dict()["param_groups"][0]
    initial_file = run / "阶段_初始.pt"
    initial_state = _check_state(initial_file, epoch=0, arm=arm, config=config,
                                 source=source, lf_sha=lf_sha, kwargs=kwargs,
                                 expected_group=expected_group)
    initial_meta = initial_state["metadata"]
    initial_score = initial_meta.get("初始合法HF选分_摄氏度")
    initial_physical = initial_meta.get("物理最佳独立损失")
    if (
        any(type(value) not in (int, float) or not math.isfinite(value)
            for value in (initial_score, initial_physical, report.get("初始HF合法选分_摄氏度")))
        or not math.isclose(initial_score, report["初始HF合法选分_摄氏度"], abs_tol=1e-4)
        or not math.isclose(initial_meta.get("观测最佳选分_摄氏度", math.inf),
                            initial_score, abs_tol=1e-4)
    ):
        raise ValueError("任-06第0轮合法HF与独立物理初值不等于原完整初态与报告")
    log_sha = _check_log(run, arm, lf_sha, count, report,
                         initial_score=initial_score, initial_physical=initial_physical)
    snapshots = {
        "初始完整状态": ("阶段_初始.pt", 0),
        "观测最佳完整状态": ("阶段_观测最佳.pt", best_epoch),
        "物理最佳完整状态": ("阶段_物理最佳.pt", physical_epoch),
        "真实阶段末": ("阶段_训练末.pt", 600),
        "专名HF先导真实阶段末": ("阶段_HF先导末.pt", 600),
    }
    states, hashes = {"初始完整状态": initial_state}, {
        "初始完整状态": sha256_file(initial_file),
    }
    for label, (filename, epoch) in snapshots.items():
        if label == "初始完整状态":
            continue
        path = run / filename
        states[label] = _check_state(path, epoch=epoch, arm=arm, config=config,
                                     source=source, lf_sha=lf_sha, kwargs=kwargs,
                                     expected_group=expected_group)
        hashes[label] = sha256_file(path)
    initial, best, physical, terminal, named = (
        states[label] for label in snapshots
    )
    if (
        initial["metadata"].get("观测最佳任06轮次") != 0
        or initial["metadata"].get("物理最佳任06轮次") != 0
        or best["metadata"].get("观测最佳任06轮次") != best_epoch
        or physical["metadata"].get("物理最佳任06轮次") != physical_epoch
        or terminal["metadata"].get("观测最佳任06轮次") != best_epoch
        or terminal["metadata"].get("物理最佳任06轮次") != physical_epoch
        or not math.isclose(best["metadata"].get("观测最佳选分_摄氏度", math.inf),
                            report["观测最佳HF合法选分_摄氏度"], abs_tol=1e-4)
        or not math.isclose(terminal["metadata"].get("观测最佳选分_摄氏度", math.inf),
                            report["观测最佳HF合法选分_摄氏度"], abs_tol=1e-4)
        or not math.isclose(physical["metadata"].get("物理最佳独立损失", math.inf),
                            report["物理最佳独立损失"], abs_tol=1e-4)
        or not math.isclose(terminal["metadata"].get("物理最佳独立损失", math.inf),
                            report["物理最佳独立损失"], abs_tol=1e-4)
        or any(not TASK04_ENERGY._same(terminal[field], named[field])
               for field in ("model_state", "optimizer_state", "random_state",
                             "parameter_requires_grad", "metadata"))
    ):
        raise ValueError("任-06观测/物理最佳轮次或两个真实末态模型AdamW及四源RNG不一致")
    for label, filename, epoch in (
        ("观测最佳完整状态", "阶段_观测最佳.pt", best_epoch),
        ("物理最佳完整状态", "阶段_物理最佳.pt", physical_epoch),
    ):
        if epoch < 600 and terminal["metadata"].get(
            f"已提交旧{filename}SHA256"
        ) != hashes[label]:
            raise ValueError(f"任-06历史最佳原件{filename}SHA与真实第600轮阶段承诺不符")
    best_path = run / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError("任-06合法HF最佳模型视图best.pt缺失")
    view, _ = load_task06_model_view(best_path, arm=arm, device=torch.device("cpu"))
    ancestry = view["任06新HF模型视图来源"]
    if (
        ancestry.get("运行资格") != OFFICIAL
        or ancestry.get("任06本臂实际轮次") != best_epoch
        or view.get("epoch") != 420 + best_epoch
        or not math.isclose(view.get("validation_selection_score_c", math.inf),
                            report["观测最佳HF合法选分_摄氏度"], abs_tol=1e-4)
        or not TASK04_ENERGY._same(view["model_state"], best["model_state"])
    ):
        raise ValueError("任-06最佳视图与原最佳完整状态张量和合法HF选分不一致")
    hashes["观测最佳模型视图"] = sha256_file(best_path)
    snapshot = TASK04_ENERGY._snapshot_hashes(run, source_checkpoint=source_path)
    snapshot["任04源运营配置快照SHA256清单"] = snapshot.pop(
        "任03源运营配置快照SHA256清单"
    )
    return {
        "运行目录": run, "运行臂": arm,
        "状态文件": {"best": best_path, "final": run / "阶段_训练末.pt"},
        "完整状态": {
            "best": run / "阶段_观测最佳.pt",
            "physical": run / "阶段_物理最佳.pt",
            "final": run / "阶段_训练末.pt",
        },
        "训练报告": report,
        "源任04完整状态SHA256": config["共同完整训练状态_SHA256"],
        "源任04最佳模型视图SHA256": config["共同观测最佳模型视图_SHA256"],
        "任06预登记配置SHA256": CONFIG_SHA,
        "LF共同真实张量SHA256": lf_sha,
        "响应tau秒": None if arm == "E0" else list(_registered_tau(config, arm)),
        "功率": powers, "时刻": times, "阶数": orders, "审核点数": 30,
        "训练日志SHA256": log_sha, "阶段报告SHA256": sha256_file(report_file),
        "物理快照SHA256": snapshot, "检查点哈希": hashes,
    }


def _compare_physical_score(measured: float, reported: float) -> None:
    if not math.isfinite(measured) or not math.isfinite(reported) or not math.isclose(
        measured, reported, rel_tol=0.0, abs_tol=1e-4,
    ):
        raise ValueError("任-06物理最佳真实模型固定CUDA配点损失独立重算不等于历史报告")


def _recompute_physical_best(
    qualified: dict[str, Any], selected_model: torch.nn.Module, device: torch.device,
) -> dict[str, float]:
    # CPU and CUDA generators draw different coordinates from the same seed.
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("任-06正式物理最佳须CUDA同设备固定256配点重算；CPU同seed不能冒充GPU配点")
    stage = torch.load(qualified["完整状态"]["physical"], map_location="cpu", weights_only=False)
    physical_model = copy.deepcopy(selected_model)
    physical_model.load_state_dict(stage["model_state"], strict=True)
    physical_model.eval()
    snapshot = qualified["运行目录"] / "config_snapshot"
    weights = load_yaml(str(snapshot / "training.yaml"))["loss_weights"]
    materials = load_materials(str(snapshot / "materials.yaml"))
    boundaries = load_resolved_boundary_conditions(str(snapshot / "boundary_conditions.yaml"))
    physics = PhysicsLossComputer(materials, boundaries, PhysicsLossWeights(
        pde=float(weights["pde"]), boundary=float(weights["boundary"]),
        initial=float(weights["initial"]), interface=float(weights["interface"]),
    ))
    components = physics(physical_model, sample_collocation(
        256, device, seed=260908, geometry_path=str(snapshot / "geometry.yaml"),
    ))
    result = {name: float(value.detach()) for name, value in components.items()}
    _compare_physical_score(
        result["physics_total"], qualified["训练报告"]["物理最佳独立损失"],
    )
    return result


def audit_completed_state(
    run: str | Path, *, arm: str, state: str, output: str | Path,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Audit a selected original best or true final, retaining all original W terms."""
    destination = _resolve(output)
    if destination.exists():
        raise FileExistsError(f"任-06独立名义能量结果已存在，不允许覆盖：{destination}")
    if state not in ("best", "final"):
        raise ValueError("任-06只能分别审核best与真实第600轮final状态")
    qualified = validate_completed_run(run, arm=arm)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    _, model = load_task06_model_view(
        qualified["状态文件"]["best"], arm=arm, device=device,
    )
    if state == "final":
        full = torch.load(qualified["状态文件"]["final"], map_location="cpu", weights_only=False)
        model.load_state_dict(full["model_state"], strict=True)
    splits = build_power_splits()
    legal_hf = sorted(splits.hf_validation)
    selected_score, _ = _validation_selection(
        model,
        DataLoader(_ir_dataset("validation", splits.hf_validation),
                   batch_size=2048, shuffle=False),
        _sensor_tensors(device, split="validation", powers_w=splits.hf_validation),
        device, load_yaml("configs/training.yaml")["multifidelity_selection_weights"],
    )
    reference = (
        qualified["训练报告"]["观测最佳HF合法选分_摄氏度"] if state == "best"
        else qualified["训练报告"]["最后一轮门禁证据"]["HF合法验证选分_摄氏度"]
    )
    if type(reference) not in (int, float) or not math.isclose(
        selected_score, reference, abs_tol=1e-4,
    ):
        raise ValueError("任-06选定真实模型的合法HF macro_v1独立重算选分与原始日志不同")
    lf_measured = _lf_material_validation(
        model, sorted(splits.simulation_validation), device,
    )
    lf_reference = qualified["训练报告"]["LF合法验证逐材料节点及真实体积RMSE_摄氏度"]
    if any(not math.isclose(lf_measured[m][metric], lf_reference[m][metric], abs_tol=1e-4)
           for m in ("Cu", "SiC") for metric in ("node", "volume")):
        raise ValueError("任-06真实冻结LF合法验证材料node/volume独立重算不等于候选报告")
    physical_components = _recompute_physical_best(qualified, model, device)
    model.requires_grad_(False).to(dtype=torch.float64).eval()
    snapshot = qualified["运行目录"] / "config_snapshot"
    audit = audit_schedule_energy(
        model,
        load_materials(str(snapshot / "materials.yaml")),
        load_resolved_boundary_conditions(str(snapshot / "boundary_conditions.yaml")),
        AxisymmetricGeometry.from_config(load_yaml(str(snapshot / "geometry.yaml"))),
        powers_w=qualified["功率"], times_s=qualified["时刻"],
        orders=tuple(qualified["阶数"]), device=device,
    )
    summary = audit["汇总"]
    if (
        audit["指标明细"].height != 30 or audit["物理分解"].height != 30
        or len(audit["原始能量"]) != 60 or len(audit["原始散度"]) != 60
        or summary.get("原定义相对平衡分母已保持") is not True
        or summary.get("工程平衡与原瓦数逐行一致") is not True
        or not math.isfinite(summary["最大相邻阶变化对吸收功率比"])
        or not math.isfinite(summary["最大分解剩余差_瓦"])
    ):
        raise ValueError("任-06独立审计缺30×16/64原瓦数、V-J-D或原相对平衡分母")
    lower_gaps = [
        abs(row["engineering_explanation_gap_w"]) for row in audit["原始散度"]
        if row["quadrature_order"] == 16
    ]
    if len(lower_gaps) != 30 or not all(math.isfinite(value) for value in lower_gaps):
        raise ValueError("任-06第16阶V-J-D散度积分剩余差必须逐点原值记录")
    payload = {
        "中文说明": (
            "仅对任04联合最佳第120轮真实模型同源分叉的任06正式600轮最佳/末态做名义模型热预算"
            "30点16/64阶独立数值审核；原瓦数、原相对分母和V-J-D逐项保留。"
            "64阶分解剩余差与16阶散度积分剩余差分别报告；没有内部实验真值，名义能量不等于真实物理验证。"
        ),
        "运行臂": arm, "审核状态": state,
        "运行目录": str(qualified["运行目录"]),
        "选择模型SHA256": qualified["检查点哈希"][
            "观测最佳模型视图" if state == "best" else "真实阶段末"
        ],
        "源任04完整状态SHA256": qualified["源任04完整状态SHA256"],
        "源任04最佳模型视图SHA256": qualified["源任04最佳模型视图SHA256"],
        "任06预登记配置SHA256": qualified["任06预登记配置SHA256"],
        "LF共同真实张量SHA256": qualified["LF共同真实张量SHA256"],
        "响应tau秒": qualified["响应tau秒"],
        "训练日志SHA256": qualified["训练日志SHA256"],
        "阶段报告SHA256": qualified["阶段报告SHA256"],
        "检查点SHA256": qualified["检查点哈希"],
        "物理来源配置快照SHA256": qualified["物理快照SHA256"],
        "本审核源码SHA256": {
            name: sha256_file(PROJECT_ROOT / name) for name in (
                "src/sic_cu/eval/energy_v5.py", "scripts/21_audit_task04_energy.py",
                "scripts/24_audit_task06_energy.py",
                "src/sic_cu/train/task06_time_features.py",
                "src/sic_cu/models/multifidelity.py",
            )
        },
        "HF合法验证功率_瓦": legal_hf,
        "本次合法HF原macro_v1独立重算选分_摄氏度": selected_score,
        "本次合法LF逐材料node及真实volume独立重算RMSE_摄氏度": lf_measured,
        "物理最佳原件固定CUDA种子260908及256配点独立重算损失": physical_components,
        "功率_瓦": qualified["功率"], "时刻_秒": qualified["时刻"],
        "求积阶数": qualified["阶数"], "物理审核点数": qualified["审核点数"],
        "16阶V-J-D散度积分剩余差最大_瓦": max(lower_gaps),
        "16阶V-J-D散度积分剩余差宏均值_瓦": sum(lower_gaps) / len(lower_gaps),
        "名义物理资格仍待独立判定": True,
        **summary,
    }
    TASK04_ENERGY.write_energy_evidence(destination, audit, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="任-06真实600轮三臂best/final名义热预算只读审核")
    parser.add_argument("--run", required=True)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--state", required=True, choices=("best", "final"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    result = audit_completed_state(
        args.run, arm=args.arm, state=args.state, output=args.output,
        device_name=args.device,
    )
    print(json.dumps({
        "运行臂": result["运行臂"], "审核状态": result["审核状态"],
        "绝对平衡宏均值_瓦": result["绝对平衡宏均值_瓦"],
        "绝对平衡95分位_瓦": result["绝对平衡95分位_瓦"],
        "16阶V-J-D散度积分剩余差最大_瓦": result["16阶V-J-D散度积分剩余差最大_瓦"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
