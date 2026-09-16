"""Read-only Task-09 CUDA audit of new subset models and original energy watts."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.eval.energy_v5 import AxisymmetricGeometry, audit_schedule_energy
from sic_cu.losses import PhysicsLossComputer, PhysicsLossWeights
from sic_cu.physics import sample_collocation
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.train.multifidelity import _ir_dataset, _sensor_tensors
from sic_cu.train.task04_joint import _lf_material_validation, _validation_selection, task04_lf_keep_guardrail
from sic_cu.train.task07_source import MANIFEST_SHA256, fork_task07_initialization
from sic_cu.train.task09_subset_formal import (
    JOINT_STAGE, TASK09_FORMAL_ROOT, _task09_checkpoint_preflight,
    check_task09_resume_payload, require_task09_cuda,
    require_task09_registration, validate_task09_output,
)
from sic_cu.train.task09_subset_gate import audit_task09_observations, load_task09_registry, load_task09_sources


TASK04_CONFIG = PROJECT_ROOT / "研究记录/任务04_联合微调/有效运行配置.yaml"
TASK04_CONFIG_SHA256 = "b464eddd752000694a513da0e4a1b4b0f7447e31b85d19726e243396a0864710"
TASK06_CONFIG = PROJECT_ROOT / "研究记录/任务06_时间响应特征/HF三臂先导有效配置.yaml"
TASK06_CONFIG_SHA256 = "7a418d94493ddafd9c78e5070bca0739ab2de30aecdc22ef322ef69bc7e03b19"
TASK04_ENERGY_CLI = PROJECT_ROOT / "scripts/21_audit_task04_energy.py"
TASK07_ENERGY_CLI = PROJECT_ROOT / "scripts/26_audit_task07_states.py"
TASK07_LEGAL_S = PROJECT_ROOT / (
    "研究记录/任务07_正式五种子重训/"
    "正式五种子HF合法验证逐窗明细_20260916T022125+0800/五种子观测验证摘要.json"
)
TASK07_LEGAL_S_SHA256 = "f73f8a42bbdb7393770358809f28f59aedcb031658dd76641cc9d6e89ebac9c9"
TASK07_BEST_ENERGY = {
    0: ("独立能源_修版seed0_观测最佳_20260916T015810+0800",
        "28addc8bcf3ffbe60eb935dd6ece76df318378d56a5b72efc949d4ef77eeea39"),
    1: ("独立能源_修版seed1_观测最佳_20260916T020040+0800",
        "d5f0956424bc3dcd7c2cdfc4bb903dab07132a98adef630a21218583d2f93524"),
    2: ("独立能源_修版seed2_观测最佳_20260916T020040+0800",
        "ce9abfa828d449a0b48e71501833a609dc14756aad52af7aa89c13d6c23899ee"),
    3: ("独立能源_修版seed3_观测最佳_20260916T020040+0800",
        "f915b3ff760f2947d89b3e8eb6f6c66526ad7240b356ece40631d3a6d25e6552"),
    4: ("独立能源_修版seed4_观测最佳_20260916T020244+0800",
        "2b3d017ec8fa79ad64dd40972dfd0ae97078cc0b7c4acd17fc0d37cf7144d8a2"),
}
FIXED_POWERS = [55.0, 115.2, 364.3, 403.0, 630.5, 729.0]
FIXED_TIMES = [1.0, 10.0, 50.0, 100.0, 200.0]


def _frozen_auditor(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("任09继承的只读原能源核验API缺席")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task09_fixed_energy_schedule() -> tuple[list[float], list[float], list[int]]:
    """Both older preregistrations must agree on all 30 original Watt conditions."""
    if (sha256_file(TASK04_CONFIG) != TASK04_CONFIG_SHA256
            or sha256_file(TASK06_CONFIG) != TASK06_CONFIG_SHA256):
        raise ValueError("任09能源审计继承的任04/06先于候选固定配置SHA已漂移")
    first = load_yaml(TASK04_CONFIG)["独立能量审核"]
    second = load_yaml(TASK06_CONFIG)["独立30点能量"]
    if (first["功率_瓦"] != FIXED_POWERS or second["功率_瓦"] != FIXED_POWERS
            or first["时刻_秒"] != FIXED_TIMES or second["时刻_秒"] != FIXED_TIMES
            or first["求积阶数"] != [16, 64] or second["阶数"] != [16, 64]
            or first["功率时刻点数"] != 30):
        raise ValueError("任09须使用原六功率五时刻及16/64阶60原瓦数/散度记录")
    return list(FIXED_POWERS), list(FIXED_TIMES), [16, 64]


def task09_locked_task07_observed_S() -> dict[int, float]:
    """Observational comparator only: never load Task-07 HF best or fixed TEST."""
    if sha256_file(TASK07_LEGAL_S) != TASK07_LEGAL_S_SHA256:
        raise ValueError("任09观测12HF冻结任07乙原件合法S摘要SHA发生变化")
    report = json.loads(TASK07_LEGAL_S.read_text(encoding="utf-8"))
    raw = report["本轮合法macro_v1"]["逐种子选分_摄氏度"]
    scores = {int(seed): float(value) for seed, value in raw.items()}
    if (set(scores) != set(range(5)) or not all(math.isfinite(value) and value > 0
                                             for value in scores.values())
            or report["旧test_Data温度标签读取"] is not False
            or report["V4历史五种子来源清单SHA256"] != MANIFEST_SHA256):
        raise ValueError("任09固定12HF只可引用已冻任07乙合法验证S五seed观测，非能源或内部真值")
    return scores


def task09_locked_task07_energy_best() -> dict[int, dict[str, Any]]:
    """Frozen nominal Watt references; source never touches old HF weights/TEST."""
    task09_fixed_energy_schedule()
    output = {}
    root = PROJECT_ROOT / "研究记录/任务07_正式五种子重训"
    for seed, (folder, original_sha) in TASK07_BEST_ENERGY.items():
        directory = root / folder
        manifest = directory / "审计工件SHA256.json"
        if sha256_file(manifest) != original_sha:
            raise ValueError("任09旧12HF乙名义能源原审计工件清单被改变")
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if (set(record) != {"原始能量.jsonl", "原始散度.jsonl",
                           "指标明细.csv", "物理分解.csv", "汇总指标.json"}
                or any(sha256_file(directory / filename) != sha
                       for filename, sha in record.items())):
            raise ValueError("任09旧12HF能源原瓦数/散度双阶60条原件SHA不吻合")
        summary = json.loads((directory / "汇总指标.json").read_text(encoding="utf-8"))
        if (summary["运行种子"] != seed or summary["审核状态"] != "best"
                or summary["运行臂"] != "E0"
                or summary["功率_瓦"] != FIXED_POWERS
                or summary["时刻_秒"] != FIXED_TIMES
                or summary["求积阶数"] != [16, 64]
                or summary["继承任04独立能源预登记SHA256"] != TASK04_CONFIG_SHA256
                or summary["继承任06独立能源预登记SHA256"] != TASK06_CONFIG_SHA256
                or summary["原定义相对平衡分母已保持"] is not True
                or summary["功率时刻审核行数"] != 30
                or summary["名义能量不证明原FEM热预算或内部温度真值"] is not True):
            raise ValueError("任09旧12HF能源乙参照只能同条件、仅名义且不得冒HF内部真值")
        output[seed] = {
            "绝对平衡宏均值_瓦": summary["绝对平衡宏均值_瓦"],
            "绝对平衡95分位_瓦": summary["绝对平衡95分位_瓦"],
            "吸收功率归一宏均值": summary["吸收功率归一宏均值"],
            "原定义相对平衡分母已保持": True,
            "功率时刻审核行数": 30,
            "清单原件SHA256": original_sha,
        }
    return output


def task09_energy_output(run: str | Path, output: str | Path, state: str) -> Path:
    if state not in ("best", "final"):
        raise ValueError("任09独审模型状态只能best或final")
    run, destination = Path(run).resolve(), Path(output).resolve()
    label = "观测最佳" if state == "best" else "训练末"
    if (run.parent != TASK09_FORMAL_ROOT.resolve()
            or destination.parent != run
            or not re.fullmatch(rf"独立原能源_{label}_\d{{8}}T\d{{6}}\+0800",
                                destination.name)):
        raise ValueError("任09独立审核目录须本seed正式子集运行目录下唯一全新best/final子目录")
    if destination.exists():
        raise FileExistsError("任09独立能源原件已存在，不覆盖任09或任何旧工件")
    return destination


def qualified_task09_completed_run(
    run: str | Path, *, name: str, size: int, seed: int,
    registry_path: str | Path, registry_sha256: str,
) -> dict[str, Any]:
    """Do not open HF validation until finished *own* true LF60/CUDA history."""
    run = validate_task09_output(run, name, size, seed)
    registry_sha = require_task09_registration(registry_path, registry_sha256)
    task09_fixed_energy_schedule()
    locked = task09_locked_task07_observed_S()
    old_energy = task09_locked_task07_energy_best()
    report_file, recent_file = run / "任09F3阶段报告.json", run / "阶段_最近.pt"
    if not report_file.is_file() or not recent_file.is_file():
        raise ValueError("任09能源仅允许已有真实CUDA完成两阶段与提交报告的本seed子集状态")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    corr, joint = report.get("校正实际轮次"), report.get("受限联合实际轮次")
    if (report.get("运行种子") != seed or report.get("序列") != name
            or report.get("HF子集功率数") != size
            or report.get("正式登记SHA256") != registry_sha
            or report.get("旧固定测试温度读取") is not False
            or type(corr) is not int or type(joint) is not int
            or not 200 <= corr <= 1500 or corr % 10
            or not 200 <= joint <= 500 or joint % 10
            or not (run / "阶段_训练末.pt").is_file()
            or "受限联合真实截止" not in report.get("状态", "")):
        raise ValueError("任09联合至少200轮且达到预登记早停/上限，缺席真实末态不能作能源成绩")
    sources, arm = load_task09_sources(), load_task09_registry()[name][size]
    source = sources[seed]
    sample = {**audit_task09_observations(load_task09_registry())["arms"][name][size],
              "arm": name}
    if report["真实HF子集功率"] != list(arm):
        raise ValueError("任09能源报告功率与原三序列嵌套前登记原件不一致")
    recent = torch.load(recent_file, map_location="cpu", weights_only=False)
    _task09_checkpoint_preflight(run, recent, source, sample,
                                 registry_sha256=registry_sha)
    meta = recent["metadata"]
    if (recent["stage"] != JOINT_STAGE
            or (meta["校正实际轮次"], meta["联合实际轮次"]) != (corr, joint)
            or report["逐真实轮次累计训练预算"] != meta["累计实际消耗"]
            or not math.isclose(float(report["观测最佳合法HF选分_摄氏度"]),
                                float(meta["合法HF观测最佳选分_摄氏度"]), abs_tol=1e-4)
            or report["观测最佳真实全局轮次"] != meta["观测最佳全局轮次"]):
        raise ValueError("任09能源最近完整状态、合法最佳和报告真实消费不吻合")
    stages = {"best": run / "阶段_观测最佳.pt",
              "final": run / "阶段_训练末.pt",
              "physical": run / "阶段_物理最佳.pt"}
    for stage_file in stages.values():
        if not stage_file.is_file():
            raise ValueError("任09双轨最佳和真实训练末完整状态原件都须提交")
        payload = torch.load(stage_file, map_location="cpu", weights_only=False)
        check_task09_resume_payload(payload, source, sample,
                                    registry_sha256=registry_sha)
    terminal = torch.load(stages["final"], map_location="cpu", weights_only=False)
    if (terminal["metadata"]["完整全局实际轮次"] != corr + joint
            or terminal["model_state"].keys() != recent["model_state"].keys()
            or any(not torch.equal(terminal["model_state"][key], recent["model_state"][key])
                   for key in recent["model_state"])):
        raise ValueError("任09真实训练末状态不能冒名未完成的最近状态")
    return {"run": run, "source": source, "sample": sample, "report": report,
            "meta": meta, "stages": stages, "registration": registry_sha,
            "locked_12hf_S": locked, "locked_12hf_energy": old_energy,
            "log_sha256": sha256_file(run / "training.jsonl"),
            "report_sha256": sha256_file(report_file),
            "stage_sha256": {name: sha256_file(path) for name, path in stages.items()},
            "snapshot_sha256": sha256_file(run / "源码与登记事前快照SHA256.json")}


def audit_task09_formal_energy(
    run: str | Path, *, name: str, size: int, seed: int, state: str,
    output: str | Path, device_name: str = "cuda",
    registry_path: str | Path | None = None,
    registry_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit new F3 best/final; a CUDA result is still no interior HF truth proof."""
    device = require_task09_cuda(device_name)
    destination = task09_energy_output(run, output, state)
    qualified = qualified_task09_completed_run(
        run, name=name, size=size, seed=seed,
        registry_path=registry_path, registry_sha256=registry_sha256,
    )
    source, snapshot = qualified["source"], qualified["run"] / "config_snapshot"
    payload = torch.load(qualified["stages"][state], map_location="cpu", weights_only=False)
    model = fork_task07_initialization(load_task09_sources(), seed, "E0",
                                       torch.device("cpu")).model
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    validation = _ir_dataset("validation", source.hf_validation_powers_w)
    sensors = _sensor_tensors(device, "validation", source.hf_validation_powers_w)
    if len(validation) != 7272 or len(sensors[0]) != 752:
        raise ValueError("任09独审合法3HF验证7272IR+752Hot/Cold不准缩水")
    chosen, modalities = _validation_selection(
        model, DataLoader(validation, batch_size=2048, shuffle=False),
        sensors, device,
        load_yaml(snapshot / "training.yaml")["multifidelity_selection_weights"],
    )
    expected = (qualified["meta"]["合法HF观测最佳选分_摄氏度"]
                if state == "best" else json.loads((qualified["run"] / "training.jsonl")
                       .read_text(encoding="utf-8").splitlines()[-1])[
                           "HF合法验证选分_摄氏度"])
    if (type(expected) not in (float, int) or not math.isfinite(chosen)
            or not math.isclose(chosen, expected, abs_tol=1e-4, rel_tol=0.0)):
        raise ValueError("任09独审模型完整合法macro_v1重算须等于真实最佳/末轮选分")
    lf = _lf_material_validation(model, list(source.lf_validation_powers_w), device)
    reference = qualified["meta"]["LF初始十验证功率两材料节点及体积"]
    keep = task04_lf_keep_guardrail(reference, lf)
    if (state == "best" and payload["metadata"]["观测最佳全局轮次"] >
            qualified["meta"]["校正实际轮次"]
            and keep["LF两材料节点与真实体积均守住5%护栏"] is not True):
        raise ValueError("任09合法最佳不得脱离同seed LF两材料节点与真实体积5%资格")
    materials = load_materials(str(snapshot / "materials.yaml"))
    boundaries = load_resolved_boundary_conditions(str(snapshot / "boundary_conditions.yaml"))
    weights = load_yaml(snapshot / "training.yaml")["loss_weights"]
    physics = PhysicsLossComputer(materials, boundaries, PhysicsLossWeights(
        pde=float(weights["pde"]), boundary=float(weights["boundary"]),
        initial=float(weights["initial"]), interface=float(weights["interface"]),
    ))
    physical_payload = torch.load(qualified["stages"]["physical"],
                                  map_location="cpu", weights_only=False)
    physical_model = fork_task07_initialization(load_task09_sources(), seed, "E0",
                                                torch.device("cpu")).model
    physical_model.load_state_dict(physical_payload["model_state"], strict=True)
    physical_model.to(device).eval()
    actual_physical = physics(
        physical_model, sample_collocation(256, device, seed=260908),
    )
    physical_components = {name: float(value.detach())
                           for name, value in actual_physical.items()}
    if (not math.isfinite(physical_components["physics_total"])
            or not math.isclose(physical_components["physics_total"],
                                qualified["meta"]["独立名义全项物理最佳损失"],
                                abs_tol=1e-4, rel_tol=0.0)):
        raise ValueError("任09物理最佳真实256名义全项固定配点重算与训练登记分数不同")
    model.requires_grad_(False).to(dtype=torch.float64).eval()
    powers, times, orders = task09_fixed_energy_schedule()
    audit = audit_schedule_energy(
        model, materials, boundaries,
        AxisymmetricGeometry.from_config(load_yaml(snapshot / "geometry.yaml")),
        powers_w=powers, times_s=times, orders=tuple(orders), device=device,
    )
    # A locked verifier preserves original B, denominator and both V-J-D rows.
    energy_checker = _frozen_auditor(TASK07_ENERGY_CLI, "task07_locked_raw_energy_task09")
    computed = energy_checker._verify_energy_raw(audit)
    task09_fixed_energy_schedule()
    current = qualified_task09_completed_run(
        run, name=name, size=size, seed=seed,
        registry_path=registry_path, registry_sha256=registry_sha256,
    )
    for name_key in ("log_sha256", "report_sha256", "stage_sha256", "snapshot_sha256"):
        if current[name_key] != qualified[name_key]:
            raise ValueError("任09能源审计期间来源/双轨最佳/报告原件变更，拒绝写盘")
    summary = {
        "状态": "任09真实CUDA独立30点16/64阶名义能源；不证明内部HF真值或FEM热预算修复",
        "本seed": seed, "序列": name, "HF子集功率数": size, "选中状态": state,
        "原LF来源全张量SHA256": source.lf_tensor_sha256,
        "本模型阶段SHA256": qualified["stage_sha256"][state],
        "正式事前登记SHA256": qualified["registration"],
        "来源事前源码归档清单SHA256": qualified["snapshot_sha256"],
        "训练日志SHA256": qualified["log_sha256"],
        "真实阶段报告SHA256": qualified["report_sha256"],
        "任04能源预登记SHA256": TASK04_CONFIG_SHA256,
        "任06能源预登记SHA256": TASK06_CONFIG_SHA256,
        "任07乙版合法观测S摘要SHA256": TASK07_LEGAL_S_SHA256,
        "任07本seed十二HF合法S_摄氏度_只读观测": qualified["locked_12hf_S"][seed],
        "任07本seed十二HF乙名义原瓦数只读参照": qualified[
            "locked_12hf_energy"
        ][seed],
        "任09子集合法HF选分独立重算_摄氏度": chosen,
        "仅观测差值_非同轨迹因果_摄氏度": chosen - qualified["locked_12hf_S"][seed],
        "任09与任07乙同网格原平衡瓦数名义差值_非修复因果_瓦": (
            audit["汇总"]["绝对平衡宏均值_瓦"]
            - qualified["locked_12hf_energy"][seed]["绝对平衡宏均值_瓦"]
        ),
        "HF合法验证三功率_瓦": list(source.hf_validation_powers_w),
        "HF合法验证IR行": len(validation), "HotCold合法验证行": len(sensors[0]),
        "HF分模态独立重算": modalities,
        "LF十合法功率Cu_SiC节点和轴对称真实体积独立重算": lf,
        "LF本模型逐材料节点与真实体积5%保持资格": keep,
        "物理最佳固定CUDA256名义全项独立重算": physical_components,
        "功率_瓦": powers, "时刻_秒": times, "求积阶数": orders,
        "两阶原瓦数与原散度各60行": True,
        "旧固定测试温度读取": False,
        "旧任07HF最佳张量与优化器复用": False,
        "LF60x2048联合按N实批分组_不能等同HF旧训练轨迹": True,
        **computed, **audit["汇总"],
    }
    original_writer = _frozen_auditor(TASK04_ENERGY_CLI, "task04_locked_raw_writer_task09")
    original_writer.write_energy_evidence(destination, audit, summary)
    return summary
