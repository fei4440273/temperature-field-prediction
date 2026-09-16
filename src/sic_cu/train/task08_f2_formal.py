"""任08 F2: independent five-seed HF correction and restricted LF joint training."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import tarfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import assert_no_hf_leakage, build_power_splits
from sic_cu.losses.physics import PhysicsLossComputer, PhysicsLossWeights
from sic_cu.physics.collocation import sample_collocation
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.train.common import (
    CONFIG_FILES, load_training_state, physics_optimizer_step, save_training_state,
    write_config_snapshot,
)
from sic_cu.train.multifidelity import (
    _ir_dataset, _macro_sensor_training_losses, _sensor_tensors,
)
from sic_cu.train.task04_joint import (
    PROJECTION_NAMES, _hf_modalities, _lf_material_validation,
    _validation_selection, task04_lf_keep_guardrail,
)
from sic_cu.train.task07_formal import (
    _equal, _joint_best_update, _log_budget, _rng_restore,
    _score_guardrails, _simulation_replay, _snapshot_hashes,
)
from sic_cu.train.task07_source import (
    MANIFEST_SHA256, SEEDS, Task07Initialization, Task07Source,
    _lf_tensor_sha256, fork_task07_initialization, validate_task07_sources,
)
from sic_cu.train.task08_f2_no_pde import _verify_shared_protocol


STAGE_BUDGET = {"HF校正上限轮次": 1500, "受限联合上限轮次": 500}
CORRECTION_STAGE = "task08_f2_correction"
JOINT_STAGE = "task08_f2_restricted_joint"
TASK08_F2_REGISTRATION = (PROJECT_ROOT / "研究记录/任务08_贡献消融/"
    "F2_正式五种子有效运行配置_事务修订后.yaml")
TASK08_F2_LEDGER = PROJECT_ROOT / "多保真DeepONet预测精度优化总计划与执行台账.md"
TASK08_F2_CLI = PROJECT_ROOT / "scripts/32_run_task08_f2_formal.py"
TASK08_F2_STATE_AUDIT = PROJECT_ROOT / "scripts/33_audit_task08_f2_states.py"
TASK08_F2_SOURCE_TAR = (PROJECT_ROOT / "研究记录/任务08_贡献消融/"
    "F2_正式五种子事前源码快照_事务修订后_20260916T040112+0800.tar.gz")
F3_LOCKED_IR_MANIFEST = (PROJECT_ROOT / "研究记录/任务07_正式五种子重训/"
    "版本化顶部径向端点_事前独立源码登记_20260916T024008+0800/"
    "事前独立径向审计登记SHA256.json")
F3_LOCKED_IR_MANIFEST_SHA256 = "e5c4795952abb02662fe3172d5a32b5a8024926a32bd63db01447051d14e3c2d"
SOURCE_TAR_MEMBERS = (
    "src/sic_cu/train/task08_f2_formal.py",
    "scripts/32_run_task08_f2_formal.py",
    "scripts/33_audit_task08_f2_states.py",
    "tests/test_task08_f2_formal.py",
    "tests/test_task08_f2_states_audit.py",
    "src/sic_cu/train/task08_f2_no_pde.py",
    "src/sic_cu/train/task07_formal.py",
    "src/sic_cu/train/task07_source.py",
    "src/sic_cu/losses/physics.py",
    "src/sic_cu/eval/development_v4.py",
    "src/sic_cu/eval/energy_v5.py",
    "scripts/21_audit_task04_energy.py",
    "configs/geometry.yaml",
    "configs/materials.yaml",
    "configs/boundary_conditions.yaml",
    "configs/data_metadata.yaml",
    "configs/splits.yaml",
    "configs/training.yaml",
    "configs/release_manifest.yaml",
    "configs/optimization_v4.yaml",
    "研究记录/任务04_联合微调/有效运行配置.yaml",
    "研究记录/任务06_时间响应特征/HF三臂先导有效配置.yaml",
    "研究记录/任务07_正式五种子重训/修复版五种子合法验证稳定性与双轨能源验收.md",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _project_output(path: str | Path) -> Path:
    resolved = _resolve(path).resolve()
    if PROJECT_ROOT.resolve() not in resolved.parents:
        raise ValueError("任08 F2任何新训练、观测或审计写入只能位于项目内严格子目录")
    return resolved


def _lf_sha(model: nn.Module) -> str:
    return _lf_tensor_sha256({name[len("low_fidelity_model."):]: value
                              for name, value in model.state_dict().items()
                              if name.startswith("low_fidelity_model.")})


def _lf_state_sha(state: Mapping[str, Tensor]) -> str:
    return _lf_tensor_sha256({name[len("low_fidelity_model."):]: value
                              for name, value in state.items()
                              if name.startswith("low_fidelity_model.")})


def _check_model(model: nn.Module, source: Task07Source, stage: str) -> None:
    named = dict(model.named_parameters())
    movable = {name for name, parameter in named.items() if parameter.requires_grad}
    expected = {name for name in named if name.startswith("correction.")}
    if stage == JOINT_STAGE:
        expected |= PROJECTION_NAMES
    if (stage not in (CORRECTION_STAGE, JOINT_STAGE) or movable != expected
        or tuple(model.state_dict()["correction.0.weight"].shape) != (128, 6)
        or any(name.startswith("response_features.") for name in model.state_dict())):
        raise ValueError("任08 F2只许旧六列新HF校正；联合LF仅四个末投影")
    for name, tensor in source.lf_state.items():
        full = f"low_fidelity_model.{name}"
        if stage == JOINT_STAGE and full in PROJECTION_NAMES:
            continue
        if not torch.equal(tensor, model.state_dict()[full].detach().cpu()):
            raise ValueError("任08 F2已冻结LF非末投影或缓冲区被改变")
    if stage == CORRECTION_STAGE and _lf_sha(model) != source.lf_tensor_sha256:
        raise ValueError("任08 F2校正阶段LF全张量须与同seed源精确相等")


def fork_f2_e0(sources: Mapping[int, Task07Source], seed: int,
               device: torch.device) -> Task07Initialization:
    initial = fork_task07_initialization(sources, seed, "E0", device)
    if (initial.optimizer.state_dict()["state"] or initial.model.correction[0].in_features != 6
        or _lf_sha(initial.model) != sources[seed].lf_tensor_sha256):
        raise ValueError("任08 F2须独立起步于本seed原LF全张量与新HF E0空AdamW")
    _check_model(initial.model, sources[seed], CORRECTION_STAGE)
    return initial


def _physics_pair(device: torch.device) -> tuple[PhysicsLossComputer, PhysicsLossComputer]:
    materials = load_materials()
    boundaries = load_resolved_boundary_conditions()
    train = PhysicsLossComputer(
        materials, boundaries,
        PhysicsLossWeights(pde=0.0, initial=1.0, boundary=1.0, interface=1.0),
        compute_pde=False,
    )
    audit = PhysicsLossComputer(
        materials, boundaries,
        PhysicsLossWeights(pde=1.0, initial=1.0, boundary=1.0, interface=1.0),
        compute_pde=True,
    )
    if any(getattr(train.weights, name) != 1.0 or getattr(audit.weights, name) != 1.0
           for name in ("initial", "boundary", "interface")):
        raise ValueError("任08 F2其它物理软约束与冻结E0不一致")
    return train, audit


def _qualification(diagnostic_only: bool) -> str:
    return ("CPU短诊断；不得列入正式消融" if diagnostic_only else
            "F2正式五种子候选；能源与合法观测审核前不得采用")


def _empty_consumption() -> dict[str, int]:
    return {"HF训练观测点": 0, "HF训练传感器点": 0,
            "LF真实回放训练点": 0, "物理配点": 0,
            "HF观测优化步": 0, "物理优化步": 0,
            "LF联合回放batch": 0}


def _input_hashes() -> dict[str, Any]:
    splits = build_power_splits()
    if len(splits.simulation_train) != 60 or len(splits.simulation_validation) != 10:
        raise ValueError("F2须锁60份真LF训练场和10份合法LF验证场")

    def power_hashes(powers: frozenset[float]) -> dict[str, str]:
        paths = [f"data/processed/simulation/{power:g}W.parquet"
                 for power in sorted(powers)]
        if len(set(paths)) != len(powers):
            raise ValueError("F2 LF真功率与处理parquet文件对应不唯一")
        return {path: sha256_file(PROJECT_ROOT / path) for path in paths}

    return {
        "合法HF顶部IR原件字节SHA256": sha256_file(
            PROJECT_ROOT / "data/processed/experiment_ir_radial.parquet"),
        "合法HF热冷环原件字节SHA256": sha256_file(
            PROJECT_ROOT / "data/processed/sensor_ring_raw.parquet"),
        "处理manifest原件字节SHA256": sha256_file(
            PROJECT_ROOT / "data/processed/manifest.json"),
        "LF60训练真场parquet逐文件SHA256": power_hashes(splits.simulation_train),
        "LF10验证真场parquet逐文件SHA256": power_hashes(splits.simulation_validation),
    }


def _verify_registered_inputs(registration: Mapping[str, Any]) -> None:
    actual = _input_hashes()
    if sha256_file(F3_LOCKED_IR_MANIFEST) != F3_LOCKED_IR_MANIFEST_SHA256:
        raise ValueError("任07原合法HF顶部来源事前登记SHA已漂移")
    former = json.loads(F3_LOCKED_IR_MANIFEST.read_text(encoding="utf-8"))
    if (actual["合法HF顶部IR原件字节SHA256"] != former.get("原合法顶部IR数据SHA256")
        or actual["处理manifest原件字节SHA256"] != former.get("原处理manifest_SHA256")
        or registration.get("训练与合法验证真实输入原件SHA256") != actual):
        raise ValueError("F2真实IR/HotCold及LF70原输入逐文件SHA与冻结F3或正式登记不符")


def _source_archive_identity(path: str | Path = TASK08_F2_SOURCE_TAR) -> dict[str, Any]:
    archive = _resolve(path)
    if not archive.is_file():
        raise FileNotFoundError("F2正式源码tar原件不存在；不得启动正式CUDA")
    names = set(SOURCE_TAR_MEMBERS)
    with tarfile.open(archive, mode="r:gz") as package:
        members = package.getmembers()
        if (len(members) != len(names) or {member.name for member in members} != names
            or any(not member.isfile() for member in members)):
            raise ValueError("F2正式源码tar成员集合/类型与事前承诺不一致")
        identities: dict[str, str] = {}
        for member in members:
            stream = package.extractfile(member)
            if stream is None:
                raise ValueError("F2正式源码tar成员原件缺失")
            fingerprint = hashlib.sha256(stream.read()).hexdigest()
            if sha256_file(PROJECT_ROOT / member.name) != fingerprint:
                raise ValueError(f"F2源码tar成员已与真实源码字节漂移：{member.name}")
            identities[member.name] = fingerprint
    return {"路径": str(archive.relative_to(PROJECT_ROOT)),
            "源码tar实际字节SHA256": sha256_file(archive),
            "源码tar成员逐文件SHA256": dict(sorted(identities.items()))}


def _metadata(
    source: Task07Source, model: nn.Module, stage: str, registry_sha: str | None,
    diagnostic_only: bool, *, correction_epoch: int, joint_epoch: int,
    correction_completed: int | None, initial_score: float, best_score: float,
    best_global_epoch: int, best_stage: str, best_stage_epoch: int,
    physical_score: float, physical_global_epoch: int, physical_stage: str,
    physical_stage_epoch: int, phase_best_epoch: int,
    lf_reference: dict[str, dict[str, float]], consumption: dict[str, int],
    lf_keep: dict[str, Any] | None = None, correction_terminal_sha: str | None = None,
    diagnostic_switch_sha: str | None = None, phase_best_score: float | None = None,
) -> dict[str, Any]:
    return {
        "任08F2运行种子": source.seed, "任08F2运行臂": "F2",
        "运行资格": _qualification(diagnostic_only),
        "正式预登记配置SHA256": registry_sha,
        "V4_B0来源清单SHA256": MANIFEST_SHA256,
        "本seed源LF检查点SHA256": source.lf_checkpoint_sha256,
        "本seed历史HF架构视图SHA256": source.hf_checkpoint_sha256,
        "本seed真实LF初始张量SHA256": source.lf_tensor_sha256,
        "本seed来源协议SHA256": source.source_metadata_sha256,
        "当前真实LF张量SHA256": _lf_sha(model),
        "当前阶段": stage, "校正实际轮次": correction_epoch,
        "联合实际轮次": joint_epoch, "校正实际截止轮次": correction_completed,
        "已提交正式校正末原件SHA256": correction_terminal_sha,
        "已提交诊断校正切换原件SHA256": diagnostic_switch_sha,
        "诊断提前联合不代表正式校正末": bool(
            diagnostic_only and stage == JOINT_STAGE and correction_completed is None),
        "全局累计实际轮次": correction_epoch + joint_epoch,
        "HF校正学习率": 0.001, "联合HF学习率": 0.0001,
        "联合LF四末投影学习率": 0.00001,
        "初始合法HF选分_摄氏度": float(initial_score),
        "观测最佳选分_摄氏度": float(best_score),
        "观测最佳全局轮次": best_global_epoch,
        "观测最佳阶段": best_stage,
        "观测最佳阶段轮次": best_stage_epoch,
        "物理最佳独立全项损失": float(physical_score),
        "物理最佳全局轮次": physical_global_epoch,
        "物理最佳阶段": physical_stage,
        "物理最佳阶段轮次": physical_stage_epoch,
        "本阶段早停最佳轮次": phase_best_epoch,
        "本阶段最低合格HF选分_摄氏度": (
            best_score if phase_best_score is None else phase_best_score),
        "LF合法验证初态逐材料节点与体积RMSE_摄氏度": copy.deepcopy(lf_reference),
        "LF逐材料节点和真实体积5%护栏": copy.deepcopy(lf_keep),
        "累计实际消耗": consumption.copy(),
        "HF真实训练功率数": 12, "HF合法验证功率数": 3,
        "LF真仿真回放功率数": 60,
        "训练体内PDE残差": None, "训练体内PDE意义": "未计算；不代表误差为零",
        "独立物理评分体内PDE": "已计算，不参与训练梯度",
        "旧test_Data温度标签读取": False,
    }


def _save_state(path: Path, model: nn.Module, optimizer: torch.optim.AdamW,
                source: Task07Source, *, stage: str, epoch: int,
                metadata: dict[str, Any]) -> None:
    _check_model(model, source, stage)
    if metadata["当前真实LF张量SHA256"] != _lf_sha(model):
        raise ValueError("任08 F2真实LF张量与待提交元数据不一致")
    if metadata["运行资格"] == _qualification(False) and (
        not torch.cuda.is_available() or not torch.cuda.get_rng_state_all()):
        raise ValueError("任08 F2正式CUDA四随机源不存在，不提交完整状态")
    save_training_state(path, model, optimizer, stage=stage, epoch=epoch,
                        budget=STAGE_BUDGET, metadata=metadata)


def _model_view(path: Path, architecture: dict[str, Any], model: nn.Module,
                source: Task07Source, *, global_epoch: int, correction_epoch: int,
                joint_epoch: int, score: float, validation: dict[str, float],
                registry_sha: str | None, diagnostic_only: bool) -> None:
    stage = JOINT_STAGE if joint_epoch else CORRECTION_STAGE
    _check_model(model, source, stage)
    view = copy.deepcopy(architecture)
    view["model_state"] = {name: value.detach().cpu().clone()
                           for name, value in model.state_dict().items()}
    view["epoch"] = global_epoch
    view["validation_rmse_c"] = float(validation["顶部"])
    view["validation_selection_score_c"] = float(score)
    view["validation_sensor"] = {key: float(value) for key, value in validation.items()
                                  if key != "顶部"}
    view["任08F2模型视图来源"] = {
        "训练种子": source.seed, "运行臂": "F2",
        "本轮校正轮次": correction_epoch, "本轮联合轮次": joint_epoch,
        "本轮全局实际轮次": global_epoch,
        "V4_B0来源清单SHA256": MANIFEST_SHA256,
        "本seed真实LF检查点SHA256": source.lf_checkpoint_sha256,
        "本seed历史HF架构视图SHA256": source.hf_checkpoint_sha256,
        "本seed来源协议SHA256": source.source_metadata_sha256,
        "本seed真实LF初始张量SHA256": source.lf_tensor_sha256,
        "当前真实LF张量SHA256": _lf_sha(model),
        "正式预登记配置SHA256": registry_sha,
        "运行资格": _qualification(diagnostic_only),
        "HF训练功率": list(source.hf_train_powers_w),
        "HF合法验证功率": list(source.hf_validation_powers_w),
        "LF训练回放功率": list(source.lf_train_powers_w),
        "训练体内PDE": None, "训练体内PDE意义": "未计算；不代表误差为零",
        "旧test_Data温度标签读取": False,
    }
    view["续跑资格"] = "仅模型视图，没有本轮AdamW和四RNG，不得作为训练续跑输入"
    temporary = path.with_name(path.name + ".tmp")
    torch.save(view, temporary)
    temporary.replace(path)


def _joint_optimizer(model: nn.Module, optimizer: torch.optim.AdamW,
                     source: Task07Source, *, loading_committed_state: bool = False) -> None:
    if len(optimizer.param_groups) != 1:
        raise ValueError("任08 F2联合须接续HF单组动量")
    optimizer.param_groups[0]["lr"] = 0.0001
    model.freeze_low_fidelity(True)
    named = dict(model.named_parameters())
    for name in PROJECTION_NAMES:
        named[name].requires_grad_(True)
    optimizer.add_param_group({"params": [named[name] for name in named if name in PROJECTION_NAMES],
                               "lr": 0.00001,
                               "weight_decay": optimizer.param_groups[0]["weight_decay"]})
    _check_model(model, source, JOINT_STAGE)
    if not loading_committed_state and not optimizer.state_dict()["state"]:
        raise ValueError("任08 F2联合不得从无HF校正动量的伪初态开始")


def _hf_epoch(
    model: nn.Module, optimizer: torch.optim.AdamW, train_data: Any,
    sensor: tuple[Tensor, ...], train_physics: PhysicsLossComputer,
    device: torch.device, seed: int, global_epoch: int,
    simulation_data: Any | None,
) -> dict[str, Any]:
    if train_physics.compute_pde is not False or train_physics.weights.pde != 0:
        raise ValueError("F2训练物理唯一PDE项须显式跳过二阶导数")
    loader = DataLoader(train_data, batch_size=2048, shuffle=True,
                        generator=torch.Generator().manual_seed(
                            7_070_000 + seed * 100_000 + global_epoch))
    if len(loader) != 15 or len(train_data) != 29593:
        raise ValueError("F2每轮15批HF12合法真实IR29593点，不接受短数据")
    lf_batches = None
    if simulation_data is not None:
        lf_loader = DataLoader(simulation_data, batch_size=2048, shuffle=True,
                               generator=torch.Generator().manual_seed(
                                   7_080_000 + seed * 100_000 + global_epoch))
        if len(lf_loader) != 60:
            raise ValueError("F2联合每轮须60功率×2048真LF场回放")
        lf_batches = iter(lf_loader)
    model.train()
    ir_losses, sensor_losses, lf_losses = [], [], []
    exposed = replayed = copper = carbide = 0
    sx, sy, sensor_delta, baseline = sensor
    for x, y, weight in loader:
        x, y, weight = (value.to(device) for value in (x, y, weight))
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, fidelity="high")
        ir_loss = (weight * ((prediction - y) / model.scales.temperature_scale_k).square()).sum() / weight.sum()
        sensor_prediction = model(sx, fidelity="high")
        absolute, change = _macro_sensor_training_losses(
            sensor_prediction, sy, sensor_delta, baseline, sx,
            model.scales.temperature_scale_k,
        )
        sensor_loss = 5.0 * absolute + change
        objective = ir_loss + sensor_loss
        if lf_batches is not None:
            replay_x, replay_y = zip(*(next(lf_batches) for _ in range(4)))
            replay_x = torch.cat(replay_x).to(device)
            replay_y = torch.cat(replay_y).to(device)
            lf_prediction = model(replay_x, fidelity="low")
            replay_loss = ((lf_prediction - replay_y) / model.scales.temperature_scale_k).square().mean()
            objective = objective + replay_loss
            lf_losses.append(float(replay_loss.detach()))
            replayed += len(replay_x)
            copper += int((replay_x[:, 4] < 0.5).sum())
            carbide += int((replay_x[:, 4] >= 0.5).sum())
        if not torch.isfinite(objective):
            raise ValueError("F2真HF/LF观测训练损失非有限")
        objective.backward()
        optimizer.step()
        exposed += len(x)
        ir_losses.append(float(ir_loss.detach()))
        sensor_losses.append(float(sensor_loss.detach()))
    if lf_batches is not None:
        try:
            next(lf_batches)
        except StopIteration:
            pass
        else:
            raise ValueError("F2联合LF60批没有完整消费")
        if replayed != 60 * 2048 or copper == 0 or carbide == 0:
            raise ValueError("F2真LF回放缺SiC/Cu或缺工况")
    components = physics_optimizer_step(
        model, optimizer, train_physics,
        sample_collocation(256, device, seed=7_090_000 + seed * 100_000 + global_epoch),
    )
    if components["pde"] is not None or not torch.isfinite(components["physics_total"]):
        raise ValueError("F2物理训练PDE必须标未计算并保留三项有限梯度")
    return {
        "HF训练观测点": exposed, "HF训练传感器点": 15 * len(sx),
        "LF真实回放训练点": replayed,
        "LF联合回放batch": 60 if lf_batches is not None else 0,
        "LF_Cu真实回放点": copper, "LF_SiC真实回放点": carbide,
        "HF观测优化步": 15, "物理优化步": 1, "物理配点": 256,
        "HF顶部训练损失": float(np.mean(ir_losses)),
        "HF环温训练损失": float(np.mean(sensor_losses)),
        "LF仅低保真训练损失": float(np.mean(lf_losses)) if lf_losses else None,
        "名义物理训练分项": {name: None if value is None else float(value.detach())
                              for name, value in components.items()},
        "训练体内PDE说明": "未计算；不代表误差为零",
    }


def _expected_registration(sources: Mapping[int, Task07Source]) -> dict[str, Any]:
    return {
        "schema_version": 1, "事前台账记录编号": "录-0058",
        "运行臂": "F2_仅移除HF体内PDE软损失", "种子": list(range(5)),
        "V4_B0来源清单SHA256": MANIFEST_SHA256,
        "五种子配对来源": {
            seed: {
                "LF检查点SHA256": source.lf_checkpoint_sha256,
                "历史HF架构视图SHA256": source.hf_checkpoint_sha256,
                "LF初始张量SHA256": source.lf_tensor_sha256,
                "来源协议SHA256": source.source_metadata_sha256,
            } for seed, source in sorted(sources.items())
        },
        "入口源码SHA256": {
            "来源": sha256_file(PROJECT_ROOT / "src/sic_cu/train/task07_source.py"),
            "F2训练": sha256_file(Path(__file__)),
            "F2命令行": sha256_file(TASK08_F2_CLI),
            "F2状态和能源独立审核": sha256_file(TASK08_F2_STATE_AUDIT),
            "F2训练TDD回归": sha256_file(PROJECT_ROOT / "tests/test_task08_f2_formal.py"),
            "F2独立审核TDD回归": sha256_file(
                PROJECT_ROOT / "tests/test_task08_f2_states_audit.py"),
            "共享物理默认True及F2 False": sha256_file(
                PROJECT_ROOT / "src/sic_cu/losses/physics.py"),
            "F3冻结入口仅作血缘": sha256_file(
                PROJECT_ROOT / "src/sic_cu/train/task07_formal.py"),
            "F2 CPU入场只作来源": sha256_file(
                PROJECT_ROOT / "src/sic_cu/train/task08_f2_no_pde.py"),
            "F3工程能源原瓦数写出": sha256_file(
                PROJECT_ROOT / "scripts/21_audit_task04_energy.py"),
            "合法HF分窗核心": sha256_file(
                PROJECT_ROOT / "src/sic_cu/eval/development_v4.py"),
            "独立全项能源积分": sha256_file(
                PROJECT_ROOT / "src/sic_cu/eval/energy_v5.py"),
        },
        "共同配置SHA256": {file: sha256_file(PROJECT_ROOT / file)
                              for file in CONFIG_FILES},
        "冻结任07乙报告SHA256": sha256_file(PROJECT_ROOT /
            "研究记录/任务07_正式五种子重训/修复版五种子合法验证稳定性与双轨能源验收.md"),
        "固定HF展示分窗原V4配置SHA256": sha256_file(
            PROJECT_ROOT / "configs/optimization_v4.yaml"),
        "独立双阶能源沿用事前配置SHA256": {
            "任04": sha256_file(PROJECT_ROOT /
                "研究记录/任务04_联合微调/有效运行配置.yaml"),
            "任06": sha256_file(PROJECT_ROOT /
                "研究记录/任务06_时间响应特征/HF三臂先导有效配置.yaml"),
        },
        "训练与合法验证真实输入原件SHA256": _input_hashes(),
        "任07原合法HF顶部IR与处理manifest来源登记SHA256": F3_LOCKED_IR_MANIFEST_SHA256,
        "F3旧HF热冷环与LF70逐文件历史SHA未留存_单PDE因果限界": True,
        "事前源码tar与成员SHA256": _source_archive_identity(),
        "固定训练合同": {
            "HF训练功率数": 12, "HF合法验证功率数": 3,
            "LF低保真回放功率数": 60, "LF合法验证功率数": 10,
            "HF观测每轮batch数": 15, "HF观测batch大小": 2048,
            "HF每轮真实观测点": 29593,
            "每轮独立物理优化步": 1, "每轮独立物理配点": 256,
            "LF每训练功率回放点": 2048,
            "真实传感器绝对权重": 5.0, "真实传感器差分权重": 1.0,
            "校正阶段上限": 1500, "联合阶段上限": 500,
            "HF校正学习率": 0.001, "联合HF学习率": 0.0001,
            "联合LF末投影学习率": 0.00001,
            "受限联合仅LF四末投影": sorted(PROJECTION_NAMES),
            "选分": "macro_v1", "合法验证周期轮数": 10,
            "早停耐心轮数": 200, "早停改善阈值": 0.0001,
            "LF两材料节点与体积四口径恶化百分比上限": 5.0,
            "唯一训练软约束差异": {
                "体内PDE权重": 0.0, "体内PDE实际计算": False,
                "初值权重": 1.0, "边界权重": 1.0, "界面权重": 1.0,
                "训练PDE分项": None,
                "独立全项物理审核PDE权重": 1.0,
                "独立全项物理参与优化梯度": False,
                "全局能源参与优化梯度": False,
            },
            "六能源固定功率五时刻两阶原瓦数独立审核": True,
            "五种子合法观测模态行数": 45,
            "五种子合法观测窗口行数": 135,
            "旧test_Data温度标签读取": False,
            "F3冻结资格": "乙；只能比较合法观测，不能声称原能源或内部物理可信",
        },
    }


def _require_formal_registration(
    registry_path: str | Path | None, registry_sha256: str | None,
    sources: Mapping[int, Task07Source] | None = None,
) -> None:
    if (registry_path is None or not registry_sha256 or
        _resolve(registry_path).resolve() != TASK08_F2_REGISTRATION.resolve() or
        not re.fullmatch(r"[0-9a-f]{64}", registry_sha256)):
        raise ValueError("任08 F2只能使用固定项目内预登记和真实64位SHA")
    if not TASK08_F2_REGISTRATION.is_file() or sha256_file(TASK08_F2_REGISTRATION) != registry_sha256:
        raise ValueError("任08 F2专属事前登记实际字节SHA不存在或不符")
    lines = TASK08_F2_LEDGER.read_text(encoding="utf-8").splitlines()
    ninth = next((i for i, line in enumerate(lines) if line.startswith("## 九、")), None)
    tenth = next((i for i, line in enumerate(lines) if line.startswith("## 十、")), None)
    if ninth is None or tenth is None or tenth <= ninth:
        raise ValueError("任08 F2总台账第九节预登记范围不存在")
    rows = [line for line in lines[ninth + 1:tenth]
            if re.search(r"\|\s*录-0058\s*\|", line)]
    if (len(rows) != 1 or registry_sha256 not in rows[0] or
        "研究记录/任务08_贡献消融/F2_正式五种子有效运行配置_事务修订后.yaml" not in rows[0]):
        raise ValueError("任08 F2总台账录-0058事前登记路径或真实SHA尚未一致")
    registered = load_yaml(TASK08_F2_REGISTRATION)
    _verify_registered_inputs(registered)
    source_tar = _source_archive_identity()
    if (registered.get("事前源码tar与成员SHA256") != source_tar or
        source_tar["源码tar实际字节SHA256"] not in rows[0]):
        raise ValueError("任08 F2台账录-0058还必须承诺事前源码tar字节与成员原SHA")
    if sources is not None and registered != _expected_registration(sources):
        raise ValueError("任08 F2五seed真配对源、版本SHA或唯一PDE软约束登记不一致")


def _validate_latest_and_best_positions(meta: Mapping[str, Any], latest_global: int) -> None:
    if any(not isinstance(meta.get(key), int) or not 0 <= meta[key] <= latest_global
           for key in ("观测最佳全局轮次", "物理最佳全局轮次")):
        raise ValueError("任08 F2两种最佳完整阶段不可超前最近真实提交轮次")


def _check_snapshot_identity(
    state: Mapping[str, Any], source: Task07Source, registry_sha: str | None,
    diagnostic_only: bool,
) -> None:
    meta = state.get("metadata", {})
    stage = state.get("stage")
    epoch = state.get("epoch")
    if (state.get("training_state_schema_version") != 1
        or state.get("budget") != STAGE_BUDGET
        or stage not in (CORRECTION_STAGE, JOINT_STAGE)
        or not isinstance(epoch, int) or not 0 <= epoch <= (1500 if stage == CORRECTION_STAGE else 500)
        or meta.get("任08F2运行种子") != source.seed
        or meta.get("任08F2运行臂") != "F2"
        or meta.get("运行资格") != _qualification(diagnostic_only)
        or meta.get("正式预登记配置SHA256") != registry_sha
        or meta.get("V4_B0来源清单SHA256") != MANIFEST_SHA256
        or meta.get("本seed源LF检查点SHA256") != source.lf_checkpoint_sha256
        or meta.get("本seed历史HF架构视图SHA256") != source.hf_checkpoint_sha256
        or meta.get("本seed来源协议SHA256") != source.source_metadata_sha256
        or meta.get("本seed真实LF初始张量SHA256") != source.lf_tensor_sha256
        or meta.get("当前阶段") != stage
        or meta.get("训练体内PDE残差", "非法") is not None
        or meta.get("独立物理评分体内PDE") != "已计算，不参与训练梯度"
        or meta.get("全局累计实际轮次") != meta.get("校正实际轮次", -1) + meta.get("联合实际轮次", -1)
        or (stage == CORRECTION_STAGE and (
            epoch != meta.get("校正实际轮次") or meta.get("联合实际轮次") != 0))
        or (stage == JOINT_STAGE and (
            epoch != meta.get("联合实际轮次")
            or (diagnostic_only and (
                meta.get("校正实际截止轮次") is not None
                or not meta.get("已提交诊断校正切换原件SHA256")
                or meta.get("已提交正式校正末原件SHA256") is not None
                or meta.get("诊断提前联合不代表正式校正末") is not True))
            or (not diagnostic_only and (
                meta.get("校正实际截止轮次") != meta.get("校正实际轮次")
                or not meta.get("已提交正式校正末原件SHA256")
                or meta.get("已提交诊断校正切换原件SHA256") is not None
                or meta.get("诊断提前联合不代表正式校正末") is not False))))
        or meta.get("当前真实LF张量SHA256") != _lf_state_sha(state.get("model_state", {}))
        or tuple(state.get("model_state", {}).get("correction.0.weight", torch.empty(0)).shape) != (128, 6)
        or set(state.get("random_state", {})) != {"python", "numpy", "torch_cpu", "torch_cuda"}
        or not isinstance(meta.get("本阶段早停最佳轮次"), int)
        or not 0 <= meta["本阶段早停最佳轮次"] <= epoch
        or not math.isfinite(float(meta.get("观测最佳选分_摄氏度", math.inf)))
        or not math.isfinite(float(meta.get("物理最佳独立全项损失", math.inf)))
        or any(name.startswith("response_features.") for name in state.get("model_state", {}))):
        raise ValueError("任08 F2完整状态来源、六列、实际轮次或训练PDE标识不符")
    _validate_latest_and_best_positions(meta, meta["全局累计实际轮次"])
    cuda_rng = state["random_state"]["torch_cuda"]
    if not diagnostic_only and (
        not isinstance(cuda_rng, (tuple, list)) or not cuda_rng
        or len(cuda_rng) != torch.cuda.device_count()
        or any(not isinstance(item, Tensor) or item.dtype != current.dtype
               or item.device.type != "cpu" or item.shape != current.shape
               for item, current in zip(cuda_rng, torch.cuda.get_rng_state_all()))):
        raise ValueError("任08 F2正式状态须具逐设备可恢复Torch CUDA随机源；合法全零可通过")
    names = list(state.get("parameter_requires_grad", {}))
    movable = {name for name in names if state["parameter_requires_grad"][name]}
    correction = [name for name in names if name.startswith("correction.")]
    lf_group = [name for name in names if name in PROJECTION_NAMES]
    if movable != set(correction) | (set(PROJECTION_NAMES) if stage == JOINT_STAGE else set()):
        raise ValueError("任08 F2冻结参数或联合LF四投影组已漂移")
    for name, original in source.lf_state.items():
        full = f"low_fidelity_model.{name}"
        if stage != JOINT_STAGE or full not in PROJECTION_NAMES:
            if not torch.equal(state["model_state"][full].detach().cpu(), original):
                raise ValueError("任08 F2完整状态LF冻结原张量已被篡改")
    groups = state.get("optimizer_state", {}).get("param_groups", [])
    momentum = state["optimizer_state"].get("state", {})
    if (len(groups) != (2 if stage == JOINT_STAGE else 1)
        or len(groups[0]["params"]) != len(correction)
        or not math.isclose(float(groups[0]["lr"]), 0.0001 if stage == JOINT_STAGE else 0.001)
        or stage == JOINT_STAGE and (
            len(groups[1]["params"]) != 4 or not math.isclose(float(groups[1]["lr"]), 0.00001))):
        raise ValueError("任08 F2真实HF/LF AdamW组数、学习率或参数个数不符")
    expected = [(groups[0]["params"], correction, 16 * meta["全局累计实际轮次"])]
    if stage == JOINT_STAGE:
        expected.append((groups[1]["params"], lf_group, 16 * meta["联合实际轮次"]))
    all_ids = [identifier for ids, _names, _step in expected for identifier in ids]
    if set(momentum) - set(all_ids):
        raise ValueError("任08 F2 AdamW中出现来源不明参数ID")
    for ids, group_names, steps in expected:
        for identifier, name in zip(ids, group_names):
            if steps == 0:
                if identifier in momentum:
                    raise ValueError("任08 F2未训练的HF/LF组不许历史AdamW动量")
                continue
            record = momentum.get(identifier, {})
            if (not {"step", "exp_avg", "exp_avg_sq"} <= record.keys()
                or float(record["step"]) != steps
                or any(not isinstance(record[key], Tensor)
                       or record[key].shape != state["model_state"][name].shape
                       or not torch.isfinite(record[key]).all()
                       for key in ("exp_avg", "exp_avg_sq"))):
                raise ValueError("任08 F2 AdamW每轮15+1步骤或HF/LF四投影真实动量不完整")


def _committed_best(
    output: Path, latest: dict[str, Any], name: str, source: Task07Source,
    registry_sha: str | None, diagnostic_only: bool,
    validation_loader: DataLoader, validation_sensor: tuple[Tensor, ...],
    selections: Mapping[str, float], audit_physics: PhysicsLossComputer,
    materials: Any, boundaries: Any, device: torch.device,
) -> tuple[dict[str, Any], nn.Module, dict[str, float] | None, bool]:
    meta = latest["metadata"]
    observed = name == "阶段_观测最佳.pt"
    keys = (("观测最佳全局轮次", "观测最佳阶段", "观测最佳阶段轮次",
             "观测最佳选分_摄氏度") if observed else
            ("物理最佳全局轮次", "物理最佳阶段", "物理最佳阶段轮次",
             "物理最佳独立全项损失"))
    best_global, latest_global = int(meta[keys[0]]), int(meta["全局累计实际轮次"])
    _validate_latest_and_best_positions(meta, latest_global)
    file = output / name
    old_sha = meta.get(f"已提交旧{name}SHA256")
    if best_global < latest_global and (
        not file.is_file() or not old_sha or sha256_file(file) != old_sha):
        raise ValueError("任08 F2先前历史最佳完整状态承诺SHA缺失，不得伪造动量或RNG")
    existing = torch.load(file, map_location="cpu", weights_only=False) if file.is_file() else None
    if existing is not None:
        _check_snapshot_identity(existing, source, registry_sha, diagnostic_only)
        if existing["metadata"]["全局累计实际轮次"] > latest_global:
            raise ValueError("任08 F2最佳完整模型超前最近完整状态")
    if existing is None or existing["metadata"]["全局累计实际轮次"] < best_global:
        if best_global != latest_global:
            raise ValueError("任08 F2丢失历史最佳原件，不得从末态重建")
        best, repair = latest, True
    else:
        best, repair = existing, False
    stage_meta = best["metadata"]
    if (best["stage"] != meta[keys[1]] or best["epoch"] != meta[keys[2]]
        or stage_meta["全局累计实际轮次"] != best_global
        or stage_meta[keys[0]] != best_global
        or not math.isclose(float(stage_meta[keys[3]]), float(meta[keys[3]]), abs_tol=1e-4)
        or (best_global == latest_global and best["stage"] == latest["stage"]
            and best["epoch"] == latest["epoch"] and any(
                not _equal(best[key], latest[key]) for key in
                ("model_state", "optimizer_state", "parameter_requires_grad",
                 "random_state", "metadata")))):
        raise ValueError("任08 F2最佳完整状态与最近状态模型、真实轮次、动量或评分不自洽")
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if device.type == "cuda" else []):
        model = fork_f2_e0(validate_task07_sources(), source.seed, device).model
    model.load_state_dict(best["model_state"], strict=True)
    if best["stage"] == JOINT_STAGE:
        for name, parameter in model.named_parameters():
            if name in PROJECTION_NAMES:
                parameter.requires_grad_(True)
    _check_model(model, source, best["stage"])
    if observed:
        score, validation = _validation_selection(model, validation_loader,
                                                  validation_sensor, device, selections)
    else:
        score, _ = _score_guardrails(model, audit_physics, materials, boundaries, device)
        validation = None
    if not math.isclose(float(score), float(meta[keys[3]]), abs_tol=1e-4):
        raise ValueError("任08 F2最佳模型合法HF或独立全PDE物理评分与完整状态不符")
    return best, model, validation, repair


def _resume_preflight(
    output: Path, latest: dict[str, Any], source: Task07Source,
    architecture: dict[str, Any], registry_sha: str | None, diagnostic_only: bool,
    sensor_count: int, validation_loader: DataLoader,
    validation_sensor: tuple[Tensor, ...], selections: Mapping[str, float],
    audit_physics: PhysicsLossComputer, materials: Any, boundaries: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    global_epoch = int(latest["metadata"]["全局累计实际轮次"])
    rows = [json.loads(line) for line in (output / "training.jsonl").read_text(
        encoding="utf-8").splitlines()]
    _log_budget(rows, global_epoch, sensor_count, latest["metadata"])
    observed = None
    repairs: list[Path] = []
    for name in ("阶段_观测最佳.pt", "阶段_物理最佳.pt"):
        best, model, validation, repair = _committed_best(
            output, latest, name, source, registry_sha, diagnostic_only,
            validation_loader, validation_sensor, selections, audit_physics,
            materials, boundaries, device)
        if repair:
            repairs.append(output / name)
        if name == "阶段_观测最佳.pt":
            observed = (best, model, validation)
    if observed is None or observed[2] is None:
        raise ValueError("任08 F2观测最佳全状态和合法HF验证不可恢复")
    best_state, best_model, validation = observed
    view_path = output / "best.pt"
    view = torch.load(view_path, map_location="cpu", weights_only=False) if view_path.is_file() else None
    best_epoch = latest["metadata"]["观测最佳全局轮次"]
    if view is not None and view.get("epoch", -1) > global_epoch:
        raise ValueError("任08 F2模型best.pt超前最近已提交轮次")
    if view is not None and view.get("epoch") == best_epoch:
        lineage = view.get("任08F2模型视图来源", {})
        if (view.get("correction_model_kwargs") != source.correction_model_kwargs
            or view.get("scales") != source.scales
            or view.get("low_fidelity_model_kwargs") != source.lf_model_kwargs
            or lineage.get("训练种子") != source.seed
            or lineage.get("运行臂") != "F2"
            or lineage.get("本轮全局实际轮次") != best_epoch
            or lineage.get("本轮校正轮次") != best_state["metadata"]["校正实际轮次"]
            or lineage.get("本轮联合轮次") != best_state["metadata"]["联合实际轮次"]
            or lineage.get("正式预登记配置SHA256") != registry_sha
            or lineage.get("V4_B0来源清单SHA256") != MANIFEST_SHA256
            or lineage.get("本seed真实LF检查点SHA256") != source.lf_checkpoint_sha256
            or lineage.get("本seed历史HF架构视图SHA256") != source.hf_checkpoint_sha256
            or lineage.get("本seed来源协议SHA256") != source.source_metadata_sha256
            or lineage.get("本seed真实LF初始张量SHA256") != source.lf_tensor_sha256
            or lineage.get("当前真实LF张量SHA256") != best_state["metadata"]["当前真实LF张量SHA256"]
            or lineage.get("运行资格") != _qualification(diagnostic_only)
            or lineage.get("训练体内PDE", "非法") is not None
            or not _equal(view.get("model_state"), best_state["model_state"])
            or not math.isclose(float(view.get("validation_selection_score_c", math.inf)),
                                float(latest["metadata"]["观测最佳选分_摄氏度"]), abs_tol=1e-4)
            or not math.isclose(float(view.get("validation_rmse_c", math.inf)),
                                float(validation["顶部"]), abs_tol=1e-4)
            or any(not math.isclose(float(view.get("validation_sensor", {}).get(key, math.inf)),
                                    float(score), abs_tol=1e-4)
                   for key, score in validation.items() if key != "顶部")):
            raise ValueError("任08 F2观测最佳视图、来源kwargs、实际张量或合法评分伪装")
    # Commit no repairs until all old full snapshots and model view pass.
    if len(rows) > global_epoch:
        suffix = time.strftime("%Y%m%dT%H%M%S")
        archived = output / f"training_中断尾行_{suffix}.jsonl"
        if archived.exists():
            raise FileExistsError("任08 F2旧中断日志归档同名已存在")
        archived.write_text("\n".join(json.dumps(row, ensure_ascii=False)
                                   for row in rows[global_epoch:]) + "\n", encoding="utf-8")
        trimmed = output / "training.jsonl.tmp"
        trimmed.write_text("\n".join(json.dumps(row, ensure_ascii=False)
                                  for row in rows[:global_epoch]) + "\n", encoding="utf-8")
        trimmed.replace(output / "training.jsonl")
    for path in repairs:
        temporary = path.with_name(path.name + ".tmp")
        torch.save(latest, temporary)
        temporary.replace(path)
    if view is None or view.get("epoch") != best_epoch:
        _model_view(view_path, architecture, best_model, source,
                    global_epoch=best_epoch,
                    correction_epoch=best_state["metadata"]["校正实际轮次"],
                    joint_epoch=best_state["metadata"]["联合实际轮次"],
                    score=latest["metadata"]["观测最佳选分_摄氏度"],
                    validation=validation, registry_sha=registry_sha,
                    diagnostic_only=diagnostic_only)
    return rows[:global_epoch]


def run_task08_f2_formal(
    *, seed: int, output_directory: str | Path,
    session_epoch_limit: int | None = None, diagnostic_only: bool = False,
    diagnostic_joint_preview: bool = False,
    resume_training_checkpoint: str | Path | None = None,
    formal_registry_path: str | Path | None = None,
    formal_registry_sha256: str | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError("任08 F2必须分别用V4 B0同seed来源0—4")
    if diagnostic_only and session_epoch_limit not in (1, 2):
        raise ValueError("任08 F2短CPU诊断只许会话1/2轮")
    if diagnostic_joint_preview and (not diagnostic_only or resume_training_checkpoint is None):
        raise ValueError("任08 F2短轮联合预览仅是不可采用的CPU诊断")
    if session_epoch_limit is not None and not 1 <= session_epoch_limit <= 2000:
        raise ValueError("任08 F2每次会话不得超出同组1500＋500上限")
    if resume_training_checkpoint is not None and session_epoch_limit is None:
        raise ValueError("任08 F2续跑须写明本次增加轮数")
    device = torch.device(device_name or ("cpu" if diagnostic_only else "cuda"))
    if diagnostic_only and device.type != "cpu":
        raise ValueError("任08 F2不可采用的短诊断仅限CPU，不得占CUDA")
    if not diagnostic_only and (device.type != "cuda" or not torch.cuda.is_available()):
        raise ValueError("任08 F2正式五种子必须真实CUDA并具完整四类随机源")
    output = _project_output(output_directory)
    if not diagnostic_only:
        _require_formal_registration(formal_registry_path, formal_registry_sha256)
    elif formal_registry_path is not None or formal_registry_sha256 is not None:
        raise ValueError("任08 F2 CPU短诊断不许借正式事前登记资格")
    registry_sha = None if diagnostic_only else formal_registry_sha256
    _verify_shared_protocol()
    sources = validate_task07_sources()
    if not diagnostic_only:
        _require_formal_registration(formal_registry_path, formal_registry_sha256, sources)
    source = sources[seed]
    initial = fork_f2_e0(sources, seed, device)
    if not diagnostic_only and not initial.random_state["torch_cuda"]:
        raise ValueError("任08 F2正式E0起点无本seed原CUDA随机源")
    model, optimizer = initial.model, initial.optimizer
    if resume_training_checkpoint is None and output.exists():
        raise FileExistsError(f"任08 F2目标已有用户资料，绝不覆盖：{output}")
    if resume_training_checkpoint is not None and not output.is_dir():
        raise FileNotFoundError(f"任08 F2不得跨seed无原目录续跑：{output}")
    splits = build_power_splits()
    assert_no_hf_leakage(
        {"F2顶部训练": splits.hf_train, "F2环温训练": splits.hf_train},
        splits.hf_validation | splits.hf_test | splits.external_sensor_test,
    )
    if (splits.hf_train != frozenset(source.hf_train_powers_w)
        or splits.hf_validation != frozenset(source.hf_validation_powers_w)
        or splits.simulation_train != frozenset(source.lf_train_powers_w)
        or splits.simulation_validation != frozenset(source.lf_validation_powers_w)):
        raise ValueError("任08 F2 LF/HF合法训练与验证功率来源已漂移")
    training = load_yaml("configs/training.yaml")
    selections = training["multifidelity_selection_weights"]
    if dict(selections) != {"ir_rmse": 1.0, "sensor_absolute_rmse": 0.2,
                            "sensor_delta_rmse": 1.0}:
        raise ValueError("任08 F2同F3选择分数1/.2/1已漂移")
    materials = load_materials()
    boundaries = load_resolved_boundary_conditions()
    train_physics, audit_physics = _physics_pair(device)
    train_data = _ir_dataset("train", source.hf_train_powers_w)
    validation_loader = DataLoader(
        _ir_dataset("validation", source.hf_validation_powers_w),
        batch_size=2048, shuffle=False)
    sensor = _sensor_tensors(device, split="train", powers_w=source.hf_train_powers_w)
    validation_sensor = _sensor_tensors(
        device, split="validation", powers_w=source.hf_validation_powers_w)
    if len(train_data) != 29593 or len(sensor[0]) != 2985:
        raise ValueError("任08 F2每轮HF12须真实顶部IR29593和Hot/Cold2985个训练环温")
    architecture = torch.load(source.hf_checkpoint_path, map_location="cpu", weights_only=False)
    consumption = _empty_consumption()
    correction_epoch = joint_epoch = global_epoch = 0
    correction_completed = None
    correction_terminal_sha = diagnostic_switch_sha = None
    phase_best_epoch = best_global_epoch = best_stage_epoch = 0
    physical_global_epoch = physical_stage_epoch = 0
    best_stage = physical_stage = stage = CORRECTION_STAGE
    lf_keep: dict[str, Any] | None = None
    began = time.perf_counter()
    if resume_training_checkpoint is None:
        output.mkdir(parents=True)
        write_config_snapshot(output)
        (output / "training.jsonl").write_text("", encoding="utf-8")
        initial_score, initial_validation = _validation_selection(
            model, validation_loader, validation_sensor, device, selections)
        physical_score, _ = _score_guardrails(
            model, audit_physics, materials, boundaries, device)
        lf_reference = _lf_material_validation(
            model, list(source.lf_validation_powers_w), device)
        initial_modalities = _hf_modalities(
            model, device, list(source.hf_validation_powers_w))
        best_score = phase_best_score = initial_score
        _rng_restore(initial.random_state)
        metadata = _metadata(
            source, model, stage, registry_sha, diagnostic_only,
            correction_epoch=0, joint_epoch=0, correction_completed=None,
            initial_score=initial_score, best_score=best_score,
            best_global_epoch=0, best_stage=stage, best_stage_epoch=0,
            physical_score=physical_score, physical_global_epoch=0,
            physical_stage=stage, physical_stage_epoch=0,
            phase_best_epoch=0, lf_reference=lf_reference,
            consumption=consumption)
        for name in ("阶段_初始.pt", "阶段_观测最佳.pt", "阶段_物理最佳.pt"):
            _save_state(output / name, model, optimizer, source, stage=stage,
                        epoch=0, metadata=metadata)
        _model_view(output / "best.pt", architecture, model, source,
                    global_epoch=0, correction_epoch=0, joint_epoch=0,
                    score=initial_score, validation=initial_validation,
                    registry_sha=registry_sha, diagnostic_only=diagnostic_only)
    else:
        resume = _resolve(resume_training_checkpoint).resolve()
        if resume.parent != output.resolve() or resume.name not in ("阶段_初始.pt", "阶段_最近.pt"):
            raise ValueError("F2仅本seed初始或最近完整阶段可续跑，旧best.pt不可冒用")
        _snapshot_hashes(output)
        candidate = torch.load(resume, map_location="cpu", weights_only=False)
        _check_snapshot_identity(candidate, source, registry_sha, diagnostic_only)
        stage = candidate["stage"]
        if stage == JOINT_STAGE:
            _joint_optimizer(model, optimizer, source, loading_committed_state=True)
        loaded = load_training_state(resume, model, optimizer)
        _check_model(model, source, stage)
        metadata = loaded["metadata"]
        correction_epoch = int(metadata["校正实际轮次"])
        joint_epoch = int(metadata["联合实际轮次"])
        global_epoch = correction_epoch + joint_epoch
        correction_completed = metadata["校正实际截止轮次"]
        correction_terminal_sha = metadata.get("已提交正式校正末原件SHA256")
        diagnostic_switch_sha = metadata.get("已提交诊断校正切换原件SHA256")
        if stage == JOINT_STAGE:
            terminal = output / ("阶段_诊断校正切换.pt" if diagnostic_only
                                 else "阶段_校正末.pt")
            committed = diagnostic_switch_sha if diagnostic_only else correction_terminal_sha
            if not terminal.is_file() or sha256_file(terminal) != committed:
                raise ValueError("F2联合必须保留同seed校正真实截止原件SHA")
        elif correction_completed is not None and not diagnostic_only:
            terminal = output / "阶段_校正末.pt"
            if not terminal.is_file():
                raise ValueError("F2已截止校正原件不在，不得开始联合")
            terminal_state = torch.load(terminal, map_location="cpu", weights_only=False)
            _check_snapshot_identity(terminal_state, source, registry_sha, diagnostic_only)
            if terminal_state["epoch"] != correction_epoch or any(
                not _equal(terminal_state[key], loaded[key]) for key in
                ("model_state", "optimizer_state", "random_state",
                 "parameter_requires_grad", "metadata")):
                raise ValueError("F2校正真实末原件与最近阶段非同源")
            correction_terminal_sha = sha256_file(terminal)
        if joint_epoch == 500:
            raise ValueError("F2联合已到500轮上限，不许续加隐形轮次")
        if diagnostic_joint_preview and (stage != CORRECTION_STAGE or correction_epoch < 2):
            raise ValueError("F2联合CPU短预览须有同seed真实校正至少2轮")
        _resume_preflight(
            output, loaded, source, architecture, registry_sha, diagnostic_only,
            len(sensor[0]), validation_loader, validation_sensor, selections,
            audit_physics, materials, boundaries, device)
        _rng_restore(loaded["random_state"])
        initial_score = float(metadata["初始合法HF选分_摄氏度"])
        best_score = float(metadata["观测最佳选分_摄氏度"])
        best_global_epoch = int(metadata["观测最佳全局轮次"])
        best_stage = str(metadata["观测最佳阶段"])
        best_stage_epoch = int(metadata["观测最佳阶段轮次"])
        physical_score = float(metadata["物理最佳独立全项损失"])
        physical_global_epoch = int(metadata["物理最佳全局轮次"])
        physical_stage = str(metadata["物理最佳阶段"])
        physical_stage_epoch = int(metadata["物理最佳阶段轮次"])
        phase_best_epoch = int(metadata["本阶段早停最佳轮次"])
        phase_best_score = float(metadata["本阶段最低合格HF选分_摄氏度"])
        lf_reference = metadata["LF合法验证初态逐材料节点与体积RMSE_摄氏度"]
        lf_keep = metadata.get("LF逐材料节点和真实体积5%护栏")
        consumption = metadata["累计实际消耗"].copy()
        initial_modalities = _hf_modalities(
            model, device, list(source.hf_validation_powers_w))
        _rng_restore(loaded["random_state"])

    last_row: dict[str, Any] = {}
    more = 2000 if session_epoch_limit is None else session_epoch_limit
    session_end = min(2000, global_epoch + more)
    correction_end = min(1500, session_end)
    if stage == CORRECTION_STAGE and correction_completed is None and not diagnostic_joint_preview:
        for epoch in range(correction_epoch + 1, correction_end + 1):
            started = time.perf_counter()
            global_epoch = correction_epoch = epoch
            actual = _hf_epoch(model, optimizer, train_data, sensor, train_physics,
                               device, seed, global_epoch, None)
            _check_model(model, source, CORRECTION_STAGE)
            for key in consumption:
                consumption[key] += actual[key]
            due = epoch % 10 == 0 or epoch == 1500 or (diagnostic_only and epoch == correction_end)
            score = validation = modalities = guardrails = physics_now = None
            if due:
                score, validation = _validation_selection(
                    model, validation_loader, validation_sensor, device, selections)
                modalities = _hf_modalities(model, device, list(source.hf_validation_powers_w))
                physics_now, guardrails = _score_guardrails(
                    model, audit_physics, materials, boundaries, device)
                if score < best_score - 0.0001:
                    best_score, best_global_epoch, best_stage, best_stage_epoch = (
                        score, global_epoch, CORRECTION_STAGE, epoch)
                    phase_best_epoch = epoch
                if physics_now < physical_score:
                    physical_score, physical_global_epoch, physical_stage, physical_stage_epoch = (
                        physics_now, global_epoch, CORRECTION_STAGE, epoch)
            boundary_stop = not diagnostic_only and due and (
                epoch == 1500 or epoch - phase_best_epoch >= 200)
            if boundary_stop:
                correction_completed = epoch
            last_row = {
                "epoch": global_epoch, "全局实际轮次": global_epoch,
                "运行种子": seed, "运行臂": "F2",
                "训练阶段": CORRECTION_STAGE, "阶段实际轮次": epoch,
                "HF合法验证选分_摄氏度": score,
                "HF合法验证分模态_摄氏度": modalities,
                "LF合法验证初态逐材料节点与体积RMSE_摄氏度": lf_reference if due else None,
                "LF当前真实张量SHA256": _lf_sha(model),
                "本轮包含验证墙钟秒": time.perf_counter() - started,
                "累计实际消耗": consumption.copy(), **actual,
            }
            if guardrails is not None:
                last_row.update(guardrails)
            with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(last_row, ensure_ascii=False) + "\n")
            metadata = _metadata(
                source, model, CORRECTION_STAGE, registry_sha, diagnostic_only,
                correction_epoch=epoch, joint_epoch=0,
                correction_completed=correction_completed,
                initial_score=initial_score, best_score=best_score,
                best_global_epoch=best_global_epoch, best_stage=best_stage,
                best_stage_epoch=best_stage_epoch,
                physical_score=physical_score,
                physical_global_epoch=physical_global_epoch,
                physical_stage=physical_stage,
                physical_stage_epoch=physical_stage_epoch,
                phase_best_epoch=phase_best_epoch, lf_reference=lf_reference,
                consumption=consumption)
            for name, selected in (("阶段_观测最佳.pt", best_global_epoch),
                                   ("阶段_物理最佳.pt", physical_global_epoch)):
                if selected < global_epoch:
                    previous = output / name
                    if not previous.is_file():
                        raise ValueError("F2历史双最佳原件缺失，不提交最新状态")
                    metadata[f"已提交旧{name}SHA256"] = sha256_file(previous)
            _save_state(output / "阶段_最近.pt", model, optimizer, source,
                        stage=CORRECTION_STAGE, epoch=epoch, metadata=metadata)
            for name, selected in (("阶段_观测最佳.pt", best_global_epoch),
                                   ("阶段_物理最佳.pt", physical_global_epoch)):
                if selected == global_epoch:
                    _save_state(output / name, model, optimizer, source,
                                stage=CORRECTION_STAGE, epoch=epoch, metadata=metadata)
            if best_global_epoch == global_epoch:
                _model_view(output / "best.pt", architecture, model, source,
                            global_epoch=global_epoch, correction_epoch=epoch,
                            joint_epoch=0, score=best_score, validation=validation,
                            registry_sha=registry_sha, diagnostic_only=diagnostic_only)
            if boundary_stop:
                _save_state(output / "阶段_校正末.pt", model, optimizer, source,
                            stage=CORRECTION_STAGE, epoch=epoch, metadata=metadata)
                break

    if stage == CORRECTION_STAGE and global_epoch < session_end and (
        diagnostic_joint_preview or correction_completed is not None and not diagnostic_only):
        if diagnostic_joint_preview:
            _save_state(output / "阶段_诊断校正切换.pt", model, optimizer,
                        source, stage=CORRECTION_STAGE, epoch=correction_epoch,
                        metadata=metadata)
            diagnostic_switch_sha = sha256_file(output / "阶段_诊断校正切换.pt")
        else:
            terminal = output / "阶段_校正末.pt"
            if not terminal.is_file():
                raise ValueError("F2缺校正真末态原件，不能切换联合")
            correction_terminal_sha = sha256_file(terminal)
        replay = _simulation_replay(source)
        phase_best_score, _ = _validation_selection(
            model, validation_loader, validation_sensor, device, selections)
        _joint_optimizer(model, optimizer, source)
        stage = JOINT_STAGE
        joint_epoch = phase_best_epoch = 0
        lf_keep = None
        metadata = _metadata(
            source, model, JOINT_STAGE, registry_sha, diagnostic_only,
            correction_epoch=correction_epoch, joint_epoch=0,
            correction_completed=correction_completed,
            initial_score=initial_score, best_score=best_score,
            best_global_epoch=best_global_epoch, best_stage=best_stage,
            best_stage_epoch=best_stage_epoch,
            physical_score=physical_score,
            physical_global_epoch=physical_global_epoch,
            physical_stage=physical_stage,
            physical_stage_epoch=physical_stage_epoch,
            phase_best_epoch=0, phase_best_score=phase_best_score,
            lf_reference=lf_reference, consumption=consumption,
            correction_terminal_sha=correction_terminal_sha,
            diagnostic_switch_sha=diagnostic_switch_sha)
        for name, selected in (("阶段_观测最佳.pt", best_global_epoch),
                               ("阶段_物理最佳.pt", physical_global_epoch)):
            if selected < global_epoch:
                previous = output / name
                if not previous.is_file():
                    raise ValueError("F2转联合第0轮仍须保留历史双最佳完整原件")
                metadata[f"已提交旧{name}SHA256"] = sha256_file(previous)
        _save_state(output / "阶段_最近.pt", model, optimizer, source,
                    stage=JOINT_STAGE, epoch=0, metadata=metadata)
        _save_state(output / "阶段_联合初始.pt", model, optimizer, source,
                    stage=JOINT_STAGE, epoch=0, metadata=metadata)
    elif stage == JOINT_STAGE:
        replay = _simulation_replay(source)
    else:
        replay = None

    if stage == JOINT_STAGE and global_epoch < session_end:
        joint_end = min(500, joint_epoch + session_end - global_epoch)
        for local_epoch in range(joint_epoch + 1, joint_end + 1):
            started = time.perf_counter()
            joint_epoch = local_epoch
            global_epoch = correction_epoch + joint_epoch
            actual = _hf_epoch(model, optimizer, train_data, sensor, train_physics,
                               device, seed, global_epoch, replay)
            _check_model(model, source, JOINT_STAGE)
            for key in consumption:
                consumption[key] += actual[key]
            due = joint_epoch % 10 == 0 or joint_epoch == 500 or (
                diagnostic_only and joint_epoch == joint_end)
            score = validation = modalities = guardrails = lf_values = physics_now = None
            if due:
                score, validation = _validation_selection(
                    model, validation_loader, validation_sensor, device, selections)
                modalities = _hf_modalities(model, device, list(source.hf_validation_powers_w))
                lf_values = _lf_material_validation(
                    model, list(source.lf_validation_powers_w), device)
                lf_keep = task04_lf_keep_guardrail(lf_reference, lf_values)
                physics_now, guardrails = _score_guardrails(
                    model, audit_physics, materials, boundaries, device)
                if lf_keep["LF两材料节点与真实体积均守住5%护栏"]:
                    phase_best_score, phase_best_epoch, _, global_improved = _joint_best_update(
                        score=score, global_best_score=best_score,
                        phase_best_score=phase_best_score,
                        phase_best_epoch=phase_best_epoch, joint_epoch=joint_epoch,
                        eligible=True)
                    if global_improved:
                        best_score, best_global_epoch, best_stage, best_stage_epoch = (
                            score, global_epoch, JOINT_STAGE, joint_epoch)
                    if physics_now < physical_score:
                        physical_score, physical_global_epoch, physical_stage, physical_stage_epoch = (
                            physics_now, global_epoch, JOINT_STAGE, joint_epoch)
            last_row = {
                "epoch": global_epoch, "全局实际轮次": global_epoch,
                "运行种子": seed, "运行臂": "F2",
                "训练阶段": JOINT_STAGE, "阶段实际轮次": local_epoch,
                "HF合法验证选分_摄氏度": score,
                "HF合法验证分模态_摄氏度": modalities,
                "LF合法验证逐材料节点与体积RMSE_摄氏度": lf_values,
                "LF逐材料节点和真实体积5%护栏": lf_keep,
                "LF当前真实张量SHA256": _lf_sha(model),
                "本轮包含验证墙钟秒": time.perf_counter() - started,
                "累计实际消耗": consumption.copy(), **actual,
            }
            if guardrails is not None:
                last_row.update(guardrails)
            with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(last_row, ensure_ascii=False) + "\n")
            metadata = _metadata(
                source, model, JOINT_STAGE, registry_sha, diagnostic_only,
                correction_epoch=correction_epoch, joint_epoch=joint_epoch,
                correction_completed=correction_completed,
                initial_score=initial_score, best_score=best_score,
                best_global_epoch=best_global_epoch, best_stage=best_stage,
                best_stage_epoch=best_stage_epoch,
                physical_score=physical_score,
                physical_global_epoch=physical_global_epoch,
                physical_stage=physical_stage,
                physical_stage_epoch=physical_stage_epoch,
                phase_best_epoch=phase_best_epoch, phase_best_score=phase_best_score,
                lf_reference=lf_reference, consumption=consumption,
                lf_keep=lf_keep, correction_terminal_sha=correction_terminal_sha,
                diagnostic_switch_sha=diagnostic_switch_sha)
            for name, selected in (("阶段_观测最佳.pt", best_global_epoch),
                                   ("阶段_物理最佳.pt", physical_global_epoch)):
                if selected < global_epoch:
                    previous = output / name
                    if not previous.is_file():
                        raise ValueError("F2历史最佳全状态缺失，不提交联合最新")
                    metadata[f"已提交旧{name}SHA256"] = sha256_file(previous)
            _save_state(output / "阶段_最近.pt", model, optimizer, source,
                        stage=JOINT_STAGE, epoch=joint_epoch, metadata=metadata)
            for name, selected in (("阶段_观测最佳.pt", best_global_epoch),
                                   ("阶段_物理最佳.pt", physical_global_epoch)):
                if selected == global_epoch:
                    _save_state(output / name, model, optimizer, source,
                                stage=JOINT_STAGE, epoch=joint_epoch,
                                metadata=metadata)
            if best_global_epoch == global_epoch:
                _model_view(output / "best.pt", architecture, model, source,
                            global_epoch=global_epoch,
                            correction_epoch=correction_epoch,
                            joint_epoch=joint_epoch, score=best_score,
                            validation=validation, registry_sha=registry_sha,
                            diagnostic_only=diagnostic_only)
            boundary_stop = not diagnostic_only and due and (
                joint_epoch == 500 or joint_epoch - phase_best_epoch >= 200)
            if boundary_stop:
                for name in ("阶段_联合末.pt", "阶段_训练末.pt"):
                    _save_state(output / name, model, optimizer, source,
                                stage=JOINT_STAGE, epoch=joint_epoch,
                                metadata=metadata)
                break

    if diagnostic_only:
        status = "CPU短诊断；只证实通路，不得入正式五种子、模型精度或消融排名"
    elif (output / "阶段_训练末.pt").is_file():
        status = "F2正式两阶段已真实截止；仍须五seed合法HF及六模型独立能源审核"
    else:
        status = "F2正式会话暂停，可从本seed最近真AdamW与四RNG状态续跑；绝不算正式完成"
    if stage == JOINT_STAGE and lf_keep is not None and not lf_keep[
        "LF两材料节点与真实体积均守住5%护栏"]:
        status += "；联合当前LF四口径护栏未过，不可采用该联合候选"
    report = {
        "状态": status, "运行资格": _qualification(diagnostic_only),
        "运行种子": seed, "运行臂": "F2",
        "正式预登记配置SHA256": registry_sha,
        "本seed源LF检查点SHA256": source.lf_checkpoint_sha256,
        "本seed真实LF初始张量SHA256": source.lf_tensor_sha256,
        "本seed历史HF架构视图SHA256": source.hf_checkpoint_sha256,
        "校正实际轮次": correction_epoch, "联合实际轮次": joint_epoch,
        "原校正预算上限": 1500, "原受限联合预算上限": 500,
        "初始HF合法选分_摄氏度": initial_score,
        "观测最佳HF合法选分_摄氏度": best_score,
        "观测最佳全局轮次": best_global_epoch,
        "物理最佳独立全项损失": physical_score,
        "物理最佳全局轮次": physical_global_epoch,
        "LF合法验证初态Cu_SiC节点及体积RMSE_摄氏度": lf_reference,
        "LF逐材料节点和真实体积5%护栏": lf_keep,
        "HF验证顶部热端冷端RMSE_摄氏度": (
            last_row.get("HF合法验证分模态_摄氏度") or initial_modalities),
        "训练PDE": None,
        "训练PDE意义": "未计算；不代表误差为零",
        "独立物理分项": "含PDE的完整物理筛查；不反传",
        "能源资格": "尚未独立审核；不作训练梯度或工程物理通过宣称",
        "旧test_Data温度标签读取": False,
        "真实最后状态": ("阶段_训练末.pt" if (output / "阶段_训练末.pt").is_file()
                       else "阶段_最近.pt"),
        "累计实际消耗": consumption.copy(),
        "最近一轮真实入场证据": last_row,
        "本会话耗时秒": time.perf_counter() - began,
    }
    temporary = output / "阶段报告.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(output / "阶段报告.json")
    return report
