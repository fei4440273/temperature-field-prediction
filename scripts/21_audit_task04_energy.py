#!/usr/bin/env python
"""任-04完成臂的只读30点双阶工程能量审计。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.eval.energy_v5 import AxisymmetricGeometry, audit_schedule_energy
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions, resolve_physics_state
from sic_cu.train.common import CONFIG_FILES
from sic_cu.train.multifidelity import _load_multifidelity_model


CONFIG_PATH = PROJECT_ROOT / "研究记录/任务04_联合微调/有效运行配置.yaml"
TASK03_DECISION = PROJECT_ROOT / "研究记录/任务03_低保真精度修复/验收判定_预算门禁复核.json"
OFFICIAL = "正式同源200轮候选；仍须两臂验收后决定采用"
TERMINALS = {"冻结": "阶段_HF续训冻结LF末.pt", "有限解冻": "阶段_联合末.pt"}
PROJECTIONS = frozenset({
    "low_fidelity_model.branch_projection.weight",
    "low_fidelity_model.branch_projection.bias",
    "low_fidelity_model.trunk_projection.weight",
    "low_fidelity_model.trunk_projection.bias",
})
CONSUMPTION = (
    "HF训练观测点", "HF训练传感器点", "LF仿真训练点", "物理配点",
    "HF观测优化步", "物理优化步",
)
FIXED_POWERS = [55.0, 115.2, 364.3, 403.0, 630.5, 729.0]
FIXED_TIMES = [1.0, 10.0, 50.0, 100.0, 200.0]


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _fixed_conditions(config: dict[str, Any]) -> tuple[list[float], list[float], list[int]]:
    fixed = config.get("独立能量审核", {})
    powers, times, orders = fixed.get("功率_瓦"), fixed.get("时刻_秒"), fixed.get("求积阶数")
    if (
        config.get("任务") != "任-04" or config.get("先导后续轮次_每臂") != 200
        or powers != FIXED_POWERS or times != FIXED_TIMES or orders != [16, 64]
        or fixed.get("功率时刻点数") != 30
    ):
        raise ValueError("任-04独立能量审核必须使用先于候选锁定的30点及16/64阶预登记配置")
    return powers, times, orders


def _check_log(run: Path, arm: str, report: dict[str, Any]) -> str:
    log_file = run / "training.jsonl"
    if not log_file.is_file():
        raise FileNotFoundError("任-04完成臂缺少真实200轮training.jsonl日志")
    cumulative = {name: 0 for name in CONSUMPTION}
    observations: int | None = None
    sensors: int | None = None
    last_epoch = 0
    with log_file.open(encoding="utf-8") as handle:
        for epoch, line in enumerate(handle, 1):
            last_epoch = epoch
            if epoch > 200:
                raise ValueError("任-04真实训练日志轮次超过正式200轮预算")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"任-04日志第{epoch}行不是完整JSON训练记录") from exc
            if (
                row.get("epoch") != epoch or row.get("任04轮次") != epoch
                or row.get("源校正实际轮次") != 300 or row.get("运行臂") != arm
            ):
                raise ValueError(f"任-04训练日志轮次/来源或运行臂不连续：第{epoch}行")
            if row.get("HF观测优化步") != 15 or row.get("物理优化步") != 1:
                raise ValueError(f"任-04第{epoch}轮HF观测优化步或物理优化步不满足15+1")
            if row.get("物理配点") != 256:
                raise ValueError(f"任-04第{epoch}轮独立物理整包必须真实消费256点")
            ir, replay = row.get("HF训练观测点"), row.get("LF仿真回放训练点")
            cu, sic = row.get("LF仿真回放Cu点"), row.get("LF仿真回放SiC点")
            if (
                type(ir) is not int or ir <= 0 or type(replay) is not int
                or replay != 60 * 2048 or type(cu) is not int or type(sic) is not int
                or cu <= 0 or sic <= 0 or cu + sic != replay
                or row.get("LF回放监督模式") != "low"
                or row.get("LF回放反向更新") is not (arm == "有限解冻")
                or row.get("LF冻结权重符合源状态") is not True
                or row.get("LF其余权重实际变化张量数") != 0
            ):
                raise ValueError(f"任-04第{epoch}轮HF/LF真实点数、两材料或LF冻结范围异常")
            if observations is None:
                observations = ir
            elif observations != ir:
                raise ValueError("任-04真实HF观测训练点数须在200轮保持相同")
            increments = {
                "HF训练观测点": ir, "LF仿真训练点": replay,
                "物理配点": 256, "HF观测优化步": 15, "物理优化步": 1,
            }
            reported = row.get("累计实际消耗", {})
            sensor_total = reported.get("HF训练传感器点")
            sensor_increment = sensor_total - cumulative["HF训练传感器点"] if type(sensor_total) is int else -1
            if sensor_increment <= 0 or (sensors is not None and sensor_increment != sensors):
                raise ValueError(f"任-04第{epoch}轮HF传感器点数或累计实际消耗不一致")
            sensors = sensor_increment
            increments["HF训练传感器点"] = sensor_increment
            for key, delta in increments.items():
                cumulative[key] += delta
                if type(reported.get(key)) is not int or reported[key] != cumulative[key]:
                    raise ValueError(f"任-04第{epoch}轮{key}累计实际消耗与训练行不一致")
    if last_epoch != 200:
        raise ValueError(f"任-04训练日志只含{last_epoch}轮，正式工程审核要求真实200轮")
    if report.get("累计实际消耗") != cumulative:
        raise ValueError("任-04阶段报告实际消费与200轮日志累计不一致")
    return sha256_file(log_file)


def _training_state(path: Path, *, epoch: int, arm: str, source_sha: str, config_sha: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"任-04完整阶段训练状态缺失：{path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    metadata = state.get("metadata", {})
    if (
        state.get("training_state_schema_version") != 1 or state.get("stage") != "joint"
        or state.get("epoch") != epoch or state.get("budget") != {"任04联合续训轮次": 200}
        or metadata.get("源校正末SHA256") != source_sha
        or metadata.get("任04预登记配置SHA256") != config_sha
        or metadata.get("源校正末实际轮次") != 300 or metadata.get("任04运行臂") != arm
        or metadata.get("运行资格") != OFFICIAL
        or metadata.get("观测最佳任04轮次", -1) not in range(epoch + 1)
    ):
        raise ValueError("任-04最佳/真实末态不是同源且同200轮预算的完整正式训练状态")
    parameters = state.get("parameter_requires_grad", {})
    model_state = state.get("model_state", {})
    low_names = {name for name in parameters if name.startswith("low_fidelity_model.")}
    hf_names = {name for name in parameters if name.startswith("correction.")}
    allowed = PROJECTIONS if arm == "有限解冻" else frozenset()
    if (
        not low_names or not hf_names or not set(parameters) <= set(model_state)
        or not all(parameters[name] for name in hf_names)
        or {name for name in low_names if parameters[name]} != allowed
        or metadata.get("LF允许训练投影参数") != sorted(allowed)
        or metadata.get("LF冻结参数") != sorted(low_names - allowed)
    ):
        raise ValueError("任-04完整训练状态LF末投影/其余冻结与HF更新名单不符")
    optimizer = state.get("optimizer_state", {})
    groups = optimizer.get("param_groups", [])
    if (
        len(groups) != (2 if arm == "有限解冻" else 1)
        or len(groups[0].get("params", [])) != len(hf_names)
        or groups[0].get("lr") != 1e-4
        or (arm == "有限解冻" and (
            len(groups[1].get("params", [])) != len(PROJECTIONS)
            or groups[1].get("lr") != 1e-5
        ))
    ):
        raise ValueError("任-04完整训练状态HF/LF AdamW参数组或学习率与预登记不符")
    hf_ids = groups[0]["params"]
    all_ids = {item for group in groups for item in group["params"]}
    momentum = optimizer.get("state", {})
    if not set(hf_ids) <= set(momentum) or not set(momentum) <= all_ids or any(
        not {"step", "exp_avg", "exp_avg_sq"} <= set(momentum[item])
        or float(momentum[item]["step"]) < 4800 + epoch * 16
        for item in hf_ids
    ):
        raise ValueError("任-04 HF AdamW动量步数或完整优化器状态不足；不得用模型视图伪造真实末态")
    if arm == "有限解冻" and epoch > 0 and not set(groups[1]["params"]) <= set(momentum):
        raise ValueError("任-04有限解冻投影层实际更新的AdamW动量缺失")
    rng = state.get("random_state", {})
    if set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"} or any(
        rng[key] is None for key in ("python", "numpy", "torch_cpu")
    ):
        raise ValueError("任-04完整阶段状态缺少Python/NumPy/Torch CPU/CUDA随机状态")
    return state


def _snapshot_hashes(run: Path, *, source_checkpoint: Path) -> dict[str, str]:
    snapshot = run / "config_snapshot"
    source = source_checkpoint.parent / "config_snapshot"
    manifest = json.loads((snapshot / "sha256.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((source / "sha256.json").read_text(encoding="utf-8"))
    found = {
        "运行配置快照SHA256清单": sha256_file(snapshot / "sha256.json"),
        "任03源运营配置快照SHA256清单": sha256_file(source / "sha256.json"),
    }
    for relative in CONFIG_FILES:
        filename = Path(relative).name
        current = PROJECT_ROOT / relative
        original = source / filename
        copied = snapshot / filename
        trusted = sha256_file(current)
        if (
            source_manifest.get(relative) != trusted
            or manifest.get(relative) != trusted
            or sha256_file(original) != trusted
            or sha256_file(copied) != trusted
        ):
            raise ValueError(
                f"任-04模型来源运营配置{filename}偏离任03真实源/当前configs独立SHA，"
                "自持快照及manifest不能独立定义物理或数据协议"
            )
        found[filename] = trusted
    resolved = snapshot / "resolved_physics.yaml"
    source_resolved = source / "resolved_physics.yaml"
    if not resolved.is_file() or not source_resolved.is_file():
        raise FileNotFoundError("任-04模型来源或任03真实源缺少名义物理已解析快照")
    if (
        sha256_file(resolved) != sha256_file(source_resolved)
        or load_yaml(str(source_resolved)) != resolve_physics_state()
    ):
        raise ValueError("任-04解析名义物理resolved_physics.yaml与任03真实源或当前边界解析不一致")
    found["resolved_physics.yaml"] = sha256_file(source_resolved)
    return found


def validate_lf_state_against_source(
    source: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor],
    *, arm: str, require_joint_effect: bool,
) -> None:
    if arm not in TERMINALS:
        raise ValueError("任-04仅支持冻结或有限解冻LF来源核对")
    source_lf = {name: tensor for name, tensor in source.items() if name.startswith("low_fidelity_model.")}
    candidate_lf = {name: tensor for name, tensor in candidate.items() if name.startswith("low_fidelity_model.")}
    if not source_lf or source_lf.keys() != candidate_lf.keys():
        raise ValueError("任-04LF来源权重张量键名必须与同一校正末完整一致")
    changed = {name for name in source_lf if not _same(source_lf[name], candidate_lf[name])}
    if arm == "冻结" and changed:
        raise ValueError("任-04冻结LF臂相对原真实校正末出现逐张量权重漂移")
    if arm == "有限解冻" and changed - PROJECTIONS:
        raise ValueError("任-04有限解冻臂的其余LF权重相对原真实校正末出现漂移")
    if arm == "有限解冻" and require_joint_effect and not changed:
        raise ValueError("任-04有限解冻臂第200轮末投影张量实际上没有更新")


def validate_completed_run(
    run: str | Path, *, arm: str, config_path: str | Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Reject incomplete or provenance-incompatible arms before energy model derivatives."""
    if arm not in TERMINALS:
        raise ValueError("任-04独立能量审核仅接受冻结/有限解冻预登记臂")
    run = _resolve(run).resolve()
    config_path = _resolve(config_path).resolve()
    if config_path != CONFIG_PATH.resolve():
        raise ValueError("任-04正式能量审核不能通过CLI改动原预登记运行配置")
    config = load_yaml(str(config_path))
    powers, times, orders = _fixed_conditions(config)
    source = _resolve(config["起点状态"]).resolve()
    origin = json.loads(TASK03_DECISION.read_text(encoding="utf-8"))["任04共同阶段起点"]
    if (
        source != _resolve(origin["检查点"]).resolve()
        or sha256_file(source) != origin["SHA256"]
        or config["起点状态_SHA256"] != origin["SHA256"]
    ):
        raise ValueError("任-04能量审核登记的旧LF同源真实校正末来源SHA256发生漂移")
    config_sha = sha256_file(config_path)
    report_file = run / "阶段报告.json"
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if (
        report.get("运行臂") != arm or report.get("本臂实际完成轮次") != 200
        or report.get("原预登记预算") != 200 or report.get("运行资格") != OFFICIAL
        or "完成同预算200轮" not in report.get("状态", "")
        or report.get("源校正末SHA256") != origin["SHA256"]
        or report.get("任04预登记配置SHA256") != config_sha
        or report.get("真实最后状态") != TERMINALS[arm]
        or report.get("旧test_Data温度读取") is not False
    ):
        raise ValueError("任-04能量审核只接受同源/同预登记配置的正式200轮完成臂，短诊断不得转正")
    training_sha = _check_log(run, arm, report)
    best_epoch = report.get("观测最佳任04轮次")
    physical_epoch = report.get("物理最佳任04轮次")
    if type(best_epoch) is not int or not 0 <= best_epoch <= 200:
        raise ValueError("任-04缺少合法观测最佳本阶段真实轮次")
    if type(physical_epoch) is not int or not 0 <= physical_epoch <= 200:
        raise ValueError("任-04缺少合法物理最佳本阶段真实轮次")
    stage_best = run / "阶段_观测最佳.pt"
    physical_best = run / "阶段_物理最佳.pt"
    terminal = run / "阶段_训练末.pt"
    arm_terminal = run / TERMINALS[arm]
    best_full = _training_state(stage_best, epoch=best_epoch, arm=arm,
                                source_sha=origin["SHA256"], config_sha=config_sha)
    physical_full = _training_state(physical_best, epoch=physical_epoch, arm=arm,
                                    source_sha=origin["SHA256"], config_sha=config_sha)
    final_full = _training_state(terminal, epoch=200, arm=arm,
                                 source_sha=origin["SHA256"], config_sha=config_sha)
    named_full = _training_state(arm_terminal, epoch=200, arm=arm,
                                 source_sha=origin["SHA256"], config_sha=config_sha)
    source_full = torch.load(source, map_location="cpu", weights_only=False)
    for stage in (best_full, physical_full, final_full, named_full):
        validate_lf_state_against_source(
            source_full["model_state"], stage["model_state"], arm=arm,
            require_joint_effect=arm == "有限解冻" and stage["epoch"] == 200,
        )
    if (
        best_full["metadata"]["观测最佳任04轮次"] != best_epoch
        or physical_full["metadata"].get("物理最佳任04轮次") != physical_epoch
        or final_full["metadata"].get("物理最佳任04轮次") != physical_epoch
        or final_full["metadata"]["观测最佳任04轮次"] != best_epoch
        or not _same(final_full["model_state"], named_full["model_state"])
        or not _same(final_full["optimizer_state"], named_full["optimizer_state"])
        or not _same(final_full["random_state"], named_full["random_state"])
    ):
        raise ValueError("任-04观测最佳与阶段报告轮次不同，或真实末态和本臂专名末态不是同一完整状态")
    best_path = run / "best.pt"
    best_view = torch.load(best_path, map_location="cpu", weights_only=False)
    lineage = best_view.get("任04新模型视图来源", {})
    allowed = PROJECTIONS if arm == "有限解冻" else frozenset()
    if (
        best_view.get("method") != "multifidelity_correction"
        or best_view.get("epoch") != 300 + best_epoch
        or best_view.get("provenance", {}).get("test_labels_consumed") is not False
        or lineage.get("旧test_Data温度标签读取") is not False
        or lineage.get("源校正末SHA256") != origin["SHA256"]
        or lineage.get("任04预登记配置SHA256") != config_sha
        or lineage.get("任04本阶段实际轮次") != best_epoch
        or lineage.get("运行资格") != OFFICIAL
        or lineage.get("LF权重更新名单") != sorted(allowed)
        or lineage.get("LF权重冻结名单") != sorted(
            name for name in best_full["parameter_requires_grad"]
            if name.startswith("low_fidelity_model.") and name not in allowed
        )
    ):
        raise ValueError("任-04观测最佳模型视图未证明同源、合法功率、LF冻结或旧测试温度隔离")
    if not _same(best_view.get("model_state"), best_full["model_state"]):
        raise ValueError("任-04best.pt模型张量与阶段_观测最佳.pt完整训练状态不相同")
    return {
        "运行目录": run, "运行臂": arm,
        "状态文件": {"best": best_path, "final": terminal},
        "完整状态": {"best": stage_best, "final": terminal},
        "训练报告": report,
        "源校正末SHA256": origin["SHA256"],
        "任04预登记配置SHA256": config_sha,
        "审核配置": config_path,
        "功率": powers, "时刻": times, "阶数": orders, "审核点数": 30,
        "训练日志SHA256": training_sha,
        "阶段报告SHA256": sha256_file(report_file),
        "物理快照SHA256": _snapshot_hashes(run, source_checkpoint=source),
        "检查点哈希": {
            "观测最佳模型视图": sha256_file(best_path),
            "观测最佳完整状态": sha256_file(stage_best),
            "物理最佳完整状态": sha256_file(physical_best),
            "真实阶段末": sha256_file(terminal),
            "本臂专名真实阶段末": sha256_file(arm_terminal),
        },
    }


def write_energy_evidence(
    output: str | Path, audit: dict[str, Any], payload: dict[str, Any],
) -> dict[str, str]:
    """Persist unmodified audited watts and an independently checkable artifact digest."""
    destination = _resolve(output).resolve()
    if destination.exists():
        raise FileExistsError(f"任-04工程能量审计产物已存在，不能覆盖：{destination}")
    if (
        audit["指标明细"].height != 30 or audit["物理分解"].height != 30
        or len(audit["原始能量"]) != 60 or len(audit["原始散度"]) != 60
        or payload.get("原定义相对平衡分母已保持") is not True
    ):
        raise ValueError("任-04输出必须是同一30点双阶原工程瓦数、V-J-D和未经更改的原相对分母")
    high_order = [row for row in audit["原始能量"] if row["quadrature_order"] == 64]
    frame = audit["指标明细"]
    if len(high_order) != 30 or not np.allclose(
        frame["原定义相对平衡分母_瓦"].to_numpy(),
        [row["relative_balance_denominator_w"] for row in high_order],
        rtol=1e-12, atol=1e-9,
    ):
        raise ValueError("任-04汇总CSV相对能量分母与原始64阶瓦数定义不一致")
    destination.mkdir(parents=True)
    for name, data in (("指标明细.csv", frame), ("物理分解.csv", audit["物理分解"])):
        temporary = destination / (name + ".tmp")
        data.write_csv(temporary)
        os.replace(temporary, destination / name)
    for name, rows in (("原始能量.jsonl", audit["原始能量"]), ("原始散度.jsonl", audit["原始散度"])):
        temporary = destination / (name + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
        )
        os.replace(temporary, destination / name)
    summary_file = destination / "汇总指标.json"
    temporary = summary_file.with_name(summary_file.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, summary_file)
    files = {path.name: sha256_file(path) for path in sorted(destination.iterdir()) if path.is_file()}
    manifest = destination / "审计工件SHA256.json"
    manifest.write_text(json.dumps(files, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return files


def audit_completed_state(
    run: str | Path, *, arm: str, state: str, output: str | Path,
    device_name: str | None = None, config_path: str | Path = CONFIG_PATH,
) -> dict[str, Any]:
    destination = _resolve(output).resolve()
    if destination.exists():
        raise FileExistsError(f"任-04工程能量审计产物已存在，不能覆盖：{destination}")
    if state not in ("best", "final"):
        raise ValueError("任-04真实模型状态须分别审核best和final，不可混为监测时刻或物理最佳")
    qualified = validate_completed_run(run, arm=arm, config_path=config_path)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    _, _, model = _load_multifidelity_model(qualified["状态文件"]["best"], device)
    if state == "final":
        full = torch.load(qualified["状态文件"]["final"], map_location="cpu", weights_only=False)
        model.load_state_dict(full["model_state"], strict=True)
    model.requires_grad_(False)
    model.to(dtype=torch.float64).eval()
    snapshot = qualified["运行目录"] / "config_snapshot"
    geometry = AxisymmetricGeometry.from_config(load_yaml(str(snapshot / "geometry.yaml")))
    materials = load_materials(str(snapshot / "materials.yaml"))
    boundaries = load_resolved_boundary_conditions(str(snapshot / "boundary_conditions.yaml"))
    audit = audit_schedule_energy(
        model, materials, boundaries, geometry,
        powers_w=qualified["功率"], times_s=qualified["时刻"],
        orders=tuple(qualified["阶数"]), device=device,
    )
    summary = audit["汇总"]
    if (
        audit["指标明细"].height != 30 or audit["物理分解"].height != 30
        or len(audit["原始能量"]) != 60 or len(audit["原始散度"]) != 60
        or not summary["原定义相对平衡分母已保持"]
        or not summary["工程平衡与原瓦数逐行一致"]
        or not math.isfinite(summary["最大相邻阶变化对吸收功率比"])
    ):
        raise ValueError("任-04能量结果缺少30点双阶、原始相对分母或逐行B=V-J-D原瓦数")
    payload = {
        "中文说明": (
            "源于同一真实任03校正末的任04正式200轮；只读审核当前模型场30点16/64阶原工程瓦数，"
            "保留原定义相对平衡分母和B=V-J-D各项；只报告阶间数值变化，不把未锁定阈值的求积或"
            "名义边界与真实内部温度声称为已验证。"
        ),
        "运行臂": arm, "审核状态": state,
        "运行目录": str(qualified["运行目录"]),
        "选择模型SHA256": qualified["检查点哈希"][
            "观测最佳模型视图" if state == "best" else "真实阶段末"
        ],
        "源校正末SHA256": qualified["源校正末SHA256"],
        "任04预登记配置SHA256": qualified["任04预登记配置SHA256"],
        "训练日志SHA256": qualified["训练日志SHA256"],
        "阶段报告SHA256": qualified["阶段报告SHA256"],
        "检查点SHA256": qualified["检查点哈希"],
        "物理来源配置快照SHA256": qualified["物理快照SHA256"],
        "本审核源码SHA256": {
            name: sha256_file(PROJECT_ROOT / name) for name in (
                "src/sic_cu/eval/energy_v5.py", "scripts/21_audit_task04_energy.py",
            )
        },
        "功率_瓦": qualified["功率"], "时刻_秒": qualified["时刻"],
        "求积阶数": qualified["阶数"], "物理审核点数": qualified["审核点数"],
        "求积阶间变化仅记录且名义物理资格仍待独立判定": True,
        **summary,
    }
    write_energy_evidence(destination, audit, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="任-04真实200轮完成臂的30点16/64阶原定义工程能量审计")
    parser.add_argument("--run", required=True)
    parser.add_argument("--arm", required=True, choices=tuple(TERMINALS))
    parser.add_argument("--state", required=True, choices=("best", "final"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    print("本次只读评估两材料名义模型场，不访问HF旧测试温度；最佳与第200轮真实末态须独立审核。")
    result = audit_completed_state(
        args.run, arm=args.arm, state=args.state, output=args.output,
        config_path=args.config, device_name=args.device,
    )
    print(json.dumps({
        "运行臂": result["运行臂"], "审核状态": result["审核状态"],
        "绝对平衡宏均值_瓦": result["绝对平衡宏均值_瓦"],
        "绝对平衡95分位_瓦": result["绝对平衡95分位_瓦"],
        "最大相邻阶变化对吸收功率比": result["最大相邻阶变化对吸收功率比"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
