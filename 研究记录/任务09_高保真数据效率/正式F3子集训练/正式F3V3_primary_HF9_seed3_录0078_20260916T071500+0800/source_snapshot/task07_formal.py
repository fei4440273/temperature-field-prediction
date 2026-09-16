"""Task-07 fresh E0 five-seed correction and restricted projection-joint training."""

from __future__ import annotations

import copy
import json
import math
import random
import re
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
from sic_cu.losses import PhysicsLossComputer, PhysicsLossWeights
from sic_cu.physics import sample_collocation
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions, resolve_physics_state
from sic_cu.train.common import (
    CONFIG_FILES, load_training_state, physics_optimizer_step, save_training_state,
    write_config_snapshot,
)
from sic_cu.train.multifidelity import (
    _ir_dataset, _macro_sensor_training_losses, _sensor_tensors,
    evaluate_schedule_guardrails,
)
from sic_cu.train.simulation import load_sampled_points
from sic_cu.train.task04_joint import (
    PROJECTION_NAMES, _hf_modalities, _lf_material_validation,
    _validation_selection, task04_lf_keep_guardrail,
)
from sic_cu.train.task07_source import (
    MANIFEST_SHA256, SEEDS, Task07Initialization, Task07Source,
    _lf_tensor_sha256, fork_task07_initialization, validate_task07_sources,
)


STAGE_BUDGET = {"HF校正上限轮次": 1500, "受限联合上限轮次": 500}
CORRECTION_STAGE = "task07_correction"
JOINT_STAGE = "task07_restricted_joint"
TASK07_REGISTRATION = PROJECT_ROOT / "研究记录/任务07_正式五种子重训/有效运行配置_实现修复后.yaml"
TASK07_LEDGER = PROJECT_ROOT / "多保真DeepONet预测精度优化总计划与执行台账.md"
TASK07_CLI = PROJECT_ROOT / "scripts/25_run_task07_formal.py"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _rng_restore(state: Mapping[str, Any]) -> None:
    if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("任-07必须保留Python/NumPy/Torch CPU/CUDA四类随机状态")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("任-07完整CUDA随机状态不可在没有GPU的机器上静默忽略")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _lf_sha(model: nn.Module) -> str:
    prefix = "low_fidelity_model."
    state = {name[len(prefix):]: value for name, value in model.state_dict().items()
             if name.startswith(prefix)}
    return _lf_tensor_sha256(state)


def _check_model(model: nn.Module, source: Task07Source, stage: str) -> None:
    named = dict(model.named_parameters())
    movable = {name for name, parameter in named.items() if parameter.requires_grad}
    expected = {name for name in named if name.startswith("correction.")}
    if stage == JOINT_STAGE:
        expected |= PROJECTION_NAMES
    if (
        stage not in (CORRECTION_STAGE, JOINT_STAGE) or movable != expected
        or tuple(model.state_dict()["correction.0.weight"].shape) != (128, 6)
        or any(name.startswith("response_features.") for name in model.state_dict())
    ):
        raise ValueError("任-07只许E0旧6列和HF校正器；联合仅LF四末投影可训练")
    for name, tensor in source.lf_state.items():
        full_name = f"low_fidelity_model.{name}"
        if stage == JOINT_STAGE and full_name in PROJECTION_NAMES:
            continue
        if not torch.equal(tensor, model.state_dict()[full_name].detach().cpu()):
            raise ValueError("任-07来源LF其余参数/输出buffer在校正或联合过程中被改变")
    if stage == CORRECTION_STAGE and _lf_sha(model) != source.lf_tensor_sha256:
        raise ValueError("任-07 HF校正阶段须完整冻结配对源LF")


def _qualification(diagnostic_only: bool) -> str:
    return (
        "短诊断；不参与任07正式五种子采用" if diagnostic_only else
        "五种子正式E0候选；须五seed完整归档与能源审核后决定采用"
    )


def _lf_keep_qualification(stage: str, lf_keep: dict[str, Any] | None) -> str:
    if stage == CORRECTION_STAGE:
        return "正式尚无联合验证，不能判通过"
    if stage != JOINT_STAGE:
        raise ValueError("任-07不存在的阶段不可报告LF资格")
    if lf_keep is None:
        return "联合合法LF逐材料节点和真实体积四口径尚未检查，不能判通过/不可采用"
    if not lf_keep["LF两材料节点与真实体积均守住5%护栏"]:
        return "未通过，不得采用这份联合模型"
    return "最近完成的合法LF四口径检查守住；仍须真实阶段完成和独立审核"


def _metadata(
    source: Task07Source, model: nn.Module, stage: str, registry_sha: str | None,
    diagnostic_only: bool, *, correction_epoch: int, joint_epoch: int,
    correction_completed: int | None, initial_score: float, best_score: float,
    best_global_epoch: int, best_stage: str, best_stage_epoch: int,
    physical_score: float, physical_global_epoch: int, physical_stage: str,
    physical_stage_epoch: int, phase_best_epoch: int,
    lf_reference: dict[str, dict[str, float]],
    consumption: dict[str, int], lf_keep: dict[str, Any] | None = None,
    correction_terminal_sha: str | None = None,
    diagnostic_switch_sha: str | None = None,
    phase_best_score: float | None = None,
) -> dict[str, Any]:
    return {
        "任07运行种子": source.seed, "任07运行臂": "E0",
        "运行资格": _qualification(diagnostic_only),
        "正式预登记配置SHA256": registry_sha,
        "V4_B0来源清单SHA256": MANIFEST_SHA256,
        "本seed源LF检查点SHA256": source.lf_checkpoint_sha256,
        "本seed历史HF架构视图SHA256": source.hf_checkpoint_sha256,
        "本seed真实LF初始张量SHA256": source.lf_tensor_sha256,
        "本seed来源协议SHA256": source.source_metadata_sha256,
        "当前真实LF张量SHA256": _lf_sha(model),
        "当前阶段": stage, "校正实际轮次": correction_epoch,
        "联合实际轮次": joint_epoch,
        "校正实际截止轮次": correction_completed,
        "已提交正式校正末原件SHA256": correction_terminal_sha,
        "已提交诊断校正切换原件SHA256": diagnostic_switch_sha,
        "诊断提前联合不代表正式校正末": bool(
            diagnostic_only and stage == JOINT_STAGE and correction_completed is None
        ),
        "全局累计实际轮次": correction_epoch + joint_epoch,
        "HF校正学习率": 0.001, "联合HF学习率": 0.0001,
        "联合LF四末投影学习率": 0.00001,
        "初始合法HF选分_摄氏度": initial_score,
        "观测最佳选分_摄氏度": best_score,
        "观测最佳全局轮次": best_global_epoch,
        "观测最佳阶段": best_stage, "观测最佳阶段轮次": best_stage_epoch,
        "物理最佳独立损失": physical_score,
        "物理最佳全局轮次": physical_global_epoch,
        "物理最佳阶段": physical_stage, "物理最佳阶段轮次": physical_stage_epoch,
        "本阶段早停最佳轮次": phase_best_epoch,
        "本阶段最低合格HF选分_摄氏度": (
            best_score if phase_best_score is None else phase_best_score
        ),
        "LF合法验证初态逐材料节点与体积RMSE_摄氏度": copy.deepcopy(lf_reference),
        "LF逐材料节点和真实体积5%护栏": copy.deepcopy(lf_keep),
        "累计实际消耗": consumption.copy(),
        "历史HF best只作架构不续用权重优化器RNG": True,
        "LF低保真训练功率数": 60,
        "HF训练功率数": 12, "HF合法验证功率数": 3,
        "旧test_Data温度标签读取": False,
    }


def _save_state(
    path: Path, model: nn.Module, optimizer: torch.optim.AdamW, source: Task07Source,
    *, stage: str, epoch: int, metadata: dict[str, Any],
) -> None:
    _check_model(model, source, stage)
    if metadata["当前真实LF张量SHA256"] != _lf_sha(model):
        raise ValueError("任-07阶段保存前实际LF张量与来源登记不一致")
    if metadata["运行资格"] == _qualification(False) and (
        not torch.cuda.is_available() or not torch.cuda.get_rng_state_all()
    ):
        raise ValueError("任-07正式完整状态保存前必须真实包含CUDA随机源")
    save_training_state(path, model, optimizer, stage=stage, epoch=epoch,
                        budget=STAGE_BUDGET, metadata=metadata)


def _model_view(
    path: Path, architecture: dict[str, Any], model: nn.Module, source: Task07Source,
    *, global_epoch: int, correction_epoch: int, joint_epoch: int,
    score: float, validation: dict[str, float], registry_sha: str | None,
    diagnostic_only: bool,
) -> None:
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
    view["任07新HF模型视图来源"] = {
        "训练种子": source.seed, "运行臂": "E0",
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
        "仅合法HF温度训练功率": list(source.hf_train_powers_w),
        "HF合法验证功率": list(source.hf_validation_powers_w),
        "LF真实温度只作低保真联合回放功率": list(source.lf_train_powers_w),
        "旧test_Data温度标签读取": False,
        "历史provenance解释": "历史HF权重不使用；仅复制旧架构与已认证数据协议",
    }
    view["续跑资格"] = "模型视图无本轮AdamW与随机源，不得作训练续跑输入"
    temporary = path.with_name(path.name + ".tmp")
    torch.save(view, temporary)
    temporary.replace(path)


def fork_task07_e0(
    sources: Mapping[int, Task07Source], seed: int,
    device: torch.device = torch.device("cpu"),
) -> Task07Initialization:
    initial = fork_task07_initialization(sources, seed, "E0", device)
    if (
        initial.arm != "E0" or initial.model.correction[0].in_features != 6
        or any(name.startswith("response_features.") for name in initial.model.state_dict())
        or initial.optimizer.state_dict()["state"]
    ):
        raise ValueError("任-07 E0必须来自各自V4 B0真实LF与全新空AdamW、严格旧6列")
    return initial


def _expected_registration(sources: Mapping[int, Task07Source]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "事前台账记录编号": "录-0036",
        "运行臂": "E0",
        "种子": list(range(5)),
        "V4_B0来源清单SHA256": MANIFEST_SHA256,
        "五种子配对来源": {
            seed: {
                "LF检查点SHA256": source.lf_checkpoint_sha256,
                "历史HF架构视图SHA256": source.hf_checkpoint_sha256,
                "LF初始张量SHA256": source.lf_tensor_sha256,
                "来源协议SHA256": source.source_metadata_sha256,
            }
            for seed, source in sorted(sources.items())
        },
        "入口源码SHA256": {
            "来源": sha256_file(PROJECT_ROOT / "src/sic_cu/train/task07_source.py"),
            "训练": sha256_file(Path(__file__)),
            "命令行": sha256_file(TASK07_CLI),
        },
        "固定训练合同": {
            "HF训练功率数": 12, "HF合法验证功率数": 3,
            "LF低保真回放功率数": 60, "LF合法验证功率数": 10,
            "HF观测每轮batch数": 15, "HF观测batch大小": 2048,
            "HF每轮真实观测点": 29593,
            "独立物理每轮优化步": 1, "独立物理每轮配点": 256,
            "LF每训练功率回放点": 2048,
            "传感器绝对权重": 5.0, "传感器差分权重": 1.0,
            "HF校正预算上限": 1500, "受限联合预算上限": 500,
            "HF校正学习率": 0.001, "联合HF学习率": 0.0001,
            "联合LF四末投影学习率": 0.00001,
            "受限联合仅LF四末投影": sorted(PROJECTION_NAMES),
            "HF合法选分": "macro_v1", "每10轮合法验证": True,
            "早停耐心": 200, "早停最小改善": 0.0001,
            "LF逐材料节点与轴对称真实体积恶化护栏百分比": 5.0,
            "LF回放保真度": "low", "旧test_Data温度标签读取": False,
        },
    }


def _require_formal_registration(
    registry_path: str | Path | None, registry_sha256: str | None,
    sources: Mapping[int, Task07Source] | None = None,
) -> None:
    if registry_path is None or not registry_sha256 or (
        _resolve(registry_path).resolve() != TASK07_REGISTRATION.resolve()
        or not re.fullmatch(r"[0-9a-f]{64}", registry_sha256)
    ):
        raise ValueError("任-07正式配置仅可采用项目内固定专属预登记路径及64位审计SHA")
    if not TASK07_REGISTRATION.is_file() or sha256_file(TASK07_REGISTRATION) != registry_sha256:
        raise ValueError("任-07专属正式预登记尚未落盘或真实字节SHA与登记不一致")
    ledger_lines = TASK07_LEDGER.read_text(encoding="utf-8").splitlines()
    ninth = next((index for index, line in enumerate(ledger_lines)
                  if line.startswith("## 九、")), None)
    tenth = next((index for index, line in enumerate(ledger_lines)
                  if line.startswith("## 十、")), None)
    if ninth is None or tenth is None or tenth <= ninth:
        raise ValueError("任-07唯一总计划第九节事前台账位置不存在")
    lines = [line for line in ledger_lines[ninth + 1:tenth]
             if re.search(r"\|\s*录-0036\s*\|", line)]
    if len(lines) != 1 or registry_sha256 not in lines[0] or (
        "研究记录/任务07_正式五种子重训/有效运行配置_实现修复后.yaml" not in lines[0]
    ):
        raise ValueError("任-07总计划第九节录-0036事前审计SHA与专属配置不一致")
    if sources is not None and load_yaml(TASK07_REGISTRATION) != _expected_registration(sources):
        raise ValueError("任-07五seed源SHA、源码、HF/LF合同或预算与正式专属登记逐字段不一致")


def _snapshot_hashes(output: Path) -> dict[str, str]:
    snapshot = output / "config_snapshot"
    hashes = json.loads((snapshot / "sha256.json").read_text(encoding="utf-8"))
    if not isinstance(hashes, dict) or any(
        hashes.get(filename) != sha256_file(PROJECT_ROOT / filename)
        or hashes.get(filename) != sha256_file(snapshot / Path(filename).name)
        for filename in CONFIG_FILES
    ):
        raise ValueError("任-07已复制的几何/材料/边界/划分/训练等七份配置快照SHA无效")
    main_plan = TASK07_LEDGER.name
    if main_plan in hashes and hashes[main_plan] != sha256_file(snapshot / main_plan):
        raise ValueError("任-07总计划物理与事前台账快照字节SHA无效")
    if load_yaml(snapshot / "resolved_physics.yaml") != resolve_physics_state():
        raise ValueError("任-07快照解析物理状态与真实几何/材料/边界来源签名不一致")
    return hashes


def _joint_best_update(
    *, score: float, global_best_score: float, phase_best_score: float,
    phase_best_epoch: int, joint_epoch: int, eligible: bool,
) -> tuple[float, int, float, bool]:
    if not eligible or not math.isfinite(score):
        return phase_best_score, phase_best_epoch, global_best_score, False
    if score < phase_best_score - 0.0001:
        phase_best_score, phase_best_epoch = score, joint_epoch
    global_improved = score < global_best_score - 0.0001
    if global_improved:
        global_best_score = score
    return phase_best_score, phase_best_epoch, global_best_score, global_improved


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return isinstance(left, Tensor) and isinstance(right, Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys()
                and all(_equal(left[key], right[key]) for key in left))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_equal(a, b) for a, b in zip(left, right)))
    return left == right


def _log_budget(rows: list[dict[str, Any]], global_epoch: int,
                sensor_count: int, metadata: dict[str, Any]) -> None:
    if len(rows) < global_epoch or any(
        row.get("全局实际轮次") != index or row.get("epoch") != index
        for index, row in enumerate(rows[:global_epoch], 1)
    ):
        raise ValueError("任-07最近完整阶段与训练日志出现缺行、跨阶段或非法轮次")
    committed = rows[:global_epoch]
    cumulative = {
        "HF训练观测点": sum(row["HF训练观测点"] for row in committed),
        "HF训练传感器点": 15 * sensor_count * global_epoch,
        "LF真实回放训练点": sum(row["LF真实回放训练点"] for row in committed),
        "物理配点": 256 * global_epoch,
        "HF观测优化步": 15 * global_epoch,
        "物理优化步": global_epoch,
        "LF联合回放batch": sum(row["LF联合回放batch"] for row in committed),
    }
    if cumulative != metadata.get("累计实际消耗") or any(
        row["HF训练观测点"] != 29593
        or row["累计实际消耗"]["HF训练观测点"] != 29593 * row["epoch"]
        or row["HF训练传感器点"] != 15 * sensor_count
        or row["HF观测优化步"] != 15 or row["物理优化步"] != 1
        or row["物理配点"] != 256 or row["HF训练传感器点"] != 15 * sensor_count
        or row["LF真实回放训练点"] != 2048 * row["LF联合回放batch"]
        or row["LF联合回放batch"] not in (0, 60)
        or row["累计实际消耗"]["HF观测优化步"] != 15 * row["epoch"]
        for row in committed
    ):
        raise ValueError("任-07 HF15＋物理1及LF60真实回放累计预算与已提交阶段不一致")


def _check_snapshot_identity(
    state: dict[str, Any], source: Task07Source, registry_sha: str | None,
    diagnostic_only: bool,
) -> None:
    meta = state.get("metadata", {})
    stage = state.get("stage")
    cuda_rng = state.get("random_state", {}).get("torch_cuda")
    if not diagnostic_only and (
        not isinstance(cuda_rng, (tuple, list))
        or len(cuda_rng) != torch.cuda.device_count() or not cuda_rng
        or any(not isinstance(value, Tensor) or value.numel() == 0 for value in cuda_rng)
        or any(value.dtype != current.dtype or value.device.type != "cpu"
               or value.shape != current.shape
               for value, current in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    ):
        raise ValueError("任-07正式完整状态CUDA随机源逐设备不存在或不可恢复")
    if (
        state.get("training_state_schema_version") != 1
        or state.get("budget") != STAGE_BUDGET
        or stage not in (CORRECTION_STAGE, JOINT_STAGE)
        or not isinstance(state.get("epoch"), int)
        or not 0 <= state["epoch"] <= (1500 if stage == CORRECTION_STAGE else 500)
        or meta.get("任07运行种子") != source.seed
        or meta.get("任07运行臂") != "E0"
        or meta.get("运行资格") != _qualification(diagnostic_only)
        or meta.get("正式预登记配置SHA256") != registry_sha
        or meta.get("V4_B0来源清单SHA256") != MANIFEST_SHA256
        or meta.get("本seed源LF检查点SHA256") != source.lf_checkpoint_sha256
        or meta.get("本seed历史HF架构视图SHA256") != source.hf_checkpoint_sha256
        or meta.get("本seed真实LF初始张量SHA256") != source.lf_tensor_sha256
        or meta.get("本seed来源协议SHA256") != source.source_metadata_sha256
        or meta.get("当前阶段") != stage
        or (stage == JOINT_STAGE and not meta.get(
            "已提交诊断校正切换原件SHA256" if meta.get("运行资格") == _qualification(True)
            else "已提交正式校正末原件SHA256"
        ))
        or meta.get("全局累计实际轮次") != meta.get("校正实际轮次", -1) + meta.get("联合实际轮次", -1)
        or meta.get("当前真实LF张量SHA256") != _lf_tensor_state_sha(state.get("model_state", {}))
        or set(state.get("random_state", {})) != {"python", "numpy", "torch_cpu", "torch_cuda"}
        or not math.isfinite(float(meta.get("本阶段最低合格HF选分_摄氏度", math.inf)))
        or not 0 <= meta.get("本阶段早停最佳轮次", -1) <= state["epoch"]
        or not state.get("optimizer_state", {}).get("param_groups")
        or tuple(state.get("model_state", {}).get("correction.0.weight", torch.empty(0)).shape) != (128, 6)
        or any(name.startswith("response_features.") for name in state.get("model_state", {}))
    ):
        raise ValueError("任-07完整阶段种子、E0六列、LF来源/预算、真实轮次或四RNG不一致")
    movable = {name for name, enabled in state.get("parameter_requires_grad", {}).items() if enabled}
    expected = {name for name in state["parameter_requires_grad"] if name.startswith("correction.")}
    if stage == JOINT_STAGE:
        expected |= PROJECTION_NAMES
    groups = state["optimizer_state"]["param_groups"]
    if (
        movable != expected
        or len(groups) != (1 if stage == CORRECTION_STAGE else 2)
        or not math.isclose(float(groups[0]["lr"]), 0.001 if stage == CORRECTION_STAGE else 0.0001)
        or (stage == JOINT_STAGE and not math.isclose(float(groups[1]["lr"]), 0.00001))
    ):
        raise ValueError("任-07阶段冻结名单、AdamW组数或HF/LF学习率与实际阶段不一致")
    group_names = [name for name in state["parameter_requires_grad"]
                   if name.startswith("correction.")]
    expected_group_ids = [groups[0]["params"]]
    expected_group_names = [group_names]
    if stage == JOINT_STAGE:
        expected_group_ids.append(groups[1]["params"])
        expected_group_names.append([name for name in state["parameter_requires_grad"]
                                     if name in PROJECTION_NAMES])
    if any(len(ids) != len(names) for ids, names in zip(expected_group_ids, expected_group_names)):
        raise ValueError("任-07真实HF AdamW及LF四投影组ID与冻结参数数目不一致")
    all_ids = [identifier for ids in expected_group_ids for identifier in ids]
    momentum = state["optimizer_state"]["state"]
    if set(momentum) - set(all_ids) or (meta["全局累计实际轮次"] == 0 and momentum):
        raise ValueError("任-07真初始AdamW必须空状态，后续动量参数ID必须属于当前HF/LF组")
    for index, (ids, names) in enumerate(zip(expected_group_ids, expected_group_names)):
        required_step = 16 * (
            meta["全局累计实际轮次"] if index == 0 else meta["联合实际轮次"]
        )
        for identifier, name in zip(ids, names):
            if required_step == 0:
                if identifier in momentum:
                    raise ValueError("任-07未实际执行的LF四投影不能携带历史AdamW动量")
                continue
            record = momentum.get(identifier, {})
            expected_shape = state["model_state"][name].shape
            if (
                not {"step", "exp_avg", "exp_avg_sq"} <= record.keys()
                or not math.isfinite(float(record["step"]))
                or float(record["step"]) != required_step
                or any(not isinstance(record[key], Tensor)
                       or record[key].shape != expected_shape
                       or not torch.isfinite(record[key]).all()
                       for key in ("exp_avg", "exp_avg_sq"))
            ):
                raise ValueError("任-07已提交轮次的HF AdamW或LF四投影真实动量/步数/张量不完整")


def _lf_tensor_state_sha(state: Mapping[str, Tensor]) -> str:
    prefix = "low_fidelity_model."
    return _lf_tensor_sha256({name[len(prefix):]: value for name, value in state.items()
                              if name.startswith(prefix)})


def _score_guardrails(model: nn.Module, physics: PhysicsLossComputer,
                      materials: Any, boundaries: Any, device: torch.device) -> tuple[float, dict[str, Any]]:
    evidence = evaluate_schedule_guardrails(model, physics, materials, boundaries, device)
    value = float(evidence["独立局部物理损失"]["physics_total"])
    if not math.isfinite(value):
        raise ValueError("任-07独立物理诊断非有限，不能保存最佳完整状态")
    return value, evidence


def _hf_epoch(
    model: nn.Module, optimizer: torch.optim.AdamW, train_data: Any,
    sensor: tuple[Tensor, ...], physics: PhysicsLossComputer,
    weights: Mapping[str, float], device: torch.device, seed: int,
    global_epoch: int, simulation_data: Any | None,
) -> dict[str, Any]:
    hf_loader = DataLoader(train_data, batch_size=2048, shuffle=True,
                           generator=torch.Generator().manual_seed(7_070_000 + seed * 100_000 + global_epoch))
    if len(hf_loader) != 15:
        raise RuntimeError("任-07一轮须真实消费HF12训练工况的15批2048观测")
    lf_batches = None
    if simulation_data is not None:
        lf_loader = DataLoader(simulation_data, batch_size=2048, shuffle=True,
                               generator=torch.Generator().manual_seed(
                                   7_080_000 + seed * 100_000 + global_epoch,
                               ))
        if len(lf_loader) != 60:
            raise RuntimeError("任-07受限联合须真实消费60功率各2048低保真仿真点")
        lf_batches = iter(lf_loader)
    model.train()
    ir_losses, sensor_losses, replay_losses = [], [], []
    exposed = replayed = copper = silicon_carbide = 0
    sensor_x, sensor_target, sensor_delta, baseline = sensor
    for x, y, weight in hf_loader:
        x, y, weight = (item.to(device) for item in (x, y, weight))
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, fidelity="high")
        ir_loss = (weight * ((prediction - y) / model.scales.temperature_scale_k).square()).sum() / weight.sum()
        sensor_prediction = model(sensor_x, fidelity="high")
        absolute, delta = _macro_sensor_training_losses(
            sensor_prediction, sensor_target, sensor_delta, baseline,
            sensor_x, model.scales.temperature_scale_k,
        )
        sensor_loss = float(weights["sensor_absolute"]) * absolute + float(weights["sensor_delta"]) * delta
        objective = float(weights["ir"]) * ir_loss + sensor_loss
        if lf_batches is not None:
            replay_x, replay_y = zip(*(next(lf_batches) for _ in range(4)))
            replay_x = torch.cat(replay_x).to(device)
            replay_y = torch.cat(replay_y).to(device)
            lf_prediction = model(replay_x, fidelity="low")
            replay_loss = ((lf_prediction - replay_y) / model.scales.temperature_scale_k).square().mean()
            objective = objective + float(weights["low_fidelity"]) * replay_loss
            replay_losses.append(float(replay_loss.detach()))
            replayed += len(replay_x)
            copper += int((replay_x[:, 4] < 0.5).sum())
            silicon_carbide += int((replay_x[:, 4] >= 0.5).sum())
        objective.backward()
        optimizer.step()
        ir_losses.append(float(ir_loss.detach()))
        sensor_losses.append(float(sensor_loss.detach()))
        exposed += len(x)
    if lf_batches is not None:
        try:
            next(lf_batches)
        except StopIteration:
            pass
        else:
            raise RuntimeError("任-07真实LF60回放批次未完整消费")
        if replayed != 60 * 2048 or copper == 0 or silicon_carbide == 0:
            raise RuntimeError("任-07受限联合LF60功率各2048回放未同时覆盖Cu和SiC")
    components = physics_optimizer_step(
        model, optimizer, physics,
        sample_collocation(256, device, seed=7_090_000 + seed * 100_000 + global_epoch),
    )
    return {
        "HF训练观测点": exposed, "HF训练传感器点": 15 * len(sensor_x),
        "LF真实回放训练点": replayed, "LF联合回放batch": 60 if lf_batches is not None else 0,
        "LF_Cu真实回放点": copper, "LF_SiC真实回放点": silicon_carbide,
        "HF顶部训练损失": float(np.mean(ir_losses)),
        "HF环温训练损失": float(np.mean(sensor_losses)),
        "LF仅低保真训练损失": float(np.mean(replay_losses)) if replay_losses else None,
        "名义物理训练分项": {key: float(value.detach()) for key, value in components.items()},
    }


def _verify_snapshot_best(
    output: Path, latest: dict[str, Any], name: str, source: Task07Source,
    registry_sha: str | None, diagnostic_only: bool,
    validation_loader: DataLoader, validation_sensor: tuple[Tensor, ...],
    selection_weights: Mapping[str, float], physics: PhysicsLossComputer,
    materials: Any, boundaries: Any, device: torch.device,
) -> tuple[dict[str, Any], nn.Module, dict[str, float] | None]:
    meta = latest["metadata"]
    observed = name == "阶段_观测最佳.pt"
    score_key, global_key, stage_key, stage_epoch_key = (
        ("观测最佳选分_摄氏度", "观测最佳全局轮次", "观测最佳阶段", "观测最佳阶段轮次")
        if observed else
        ("物理最佳独立损失", "物理最佳全局轮次", "物理最佳阶段", "物理最佳阶段轮次")
    )
    best_global = int(meta[global_key])
    latest_global = int(meta["全局累计实际轮次"])
    if not 0 <= best_global <= latest_global:
        raise ValueError("任-07两类最佳轮次不可超前已提交最近完整阶段")
    file = output / name
    committed_sha = meta.get(f"已提交旧{name}SHA256")
    if best_global < latest_global and (
        not file.is_file() or not committed_sha or sha256_file(file) != committed_sha
    ):
        raise ValueError("任-07历史最佳完整阶段SHA/原件异常，不得伪造历史AdamW或随机源")
    existed = torch.load(file, map_location="cpu", weights_only=False) if file.is_file() else None
    if existed is not None:
        _check_snapshot_identity(existed, source, registry_sha, diagnostic_only)
        if existed["metadata"]["全局累计实际轮次"] > latest_global:
            raise ValueError("任-07最佳模型超前最近已提交状态，不可使用未提交轮次")
    if existed is None or int(existed["metadata"]["全局累计实际轮次"]) < best_global:
        if best_global != latest_global:
            raise ValueError("任-07旧最佳原件丢失，无法恢复其真实优化器和随机源")
        snapshot = latest
    else:
        snapshot = existed
    stage_meta = snapshot["metadata"]
    if (
        snapshot["stage"] != meta[stage_key]
        or snapshot["epoch"] != meta[stage_epoch_key]
        or stage_meta["全局累计实际轮次"] != best_global
        or stage_meta[global_key] != best_global
        or not math.isclose(float(stage_meta[score_key]), float(meta[score_key]), abs_tol=1e-4)
        or (best_global == latest_global and snapshot["stage"] == latest["stage"]
            and snapshot["epoch"] == latest["epoch"] and any(
            not _equal(snapshot[key], latest[key])
            for key in ("model_state", "optimizer_state", "random_state",
                        "parameter_requires_grad", "metadata")
        ))
    ):
        raise ValueError("任-07最佳完整阶段来源、真实轮次、评分、AdamW或四类随机源与最近阶段不一致")
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if device.type == "cuda" else []):
        model = fork_task07_e0(validate_task07_sources(), source.seed, device).model
    model.load_state_dict(snapshot["model_state"], strict=True)
    if snapshot["stage"] == JOINT_STAGE:
        for parameter_name, parameter in model.named_parameters():
            if parameter_name in PROJECTION_NAMES:
                parameter.requires_grad_(True)
    _check_model(model, source, snapshot["stage"])
    if observed:
        score, validation = _validation_selection(model, validation_loader,
                                                  validation_sensor, device, selection_weights)
    else:
        score, _ = _score_guardrails(model, physics, materials, boundaries, device)
        validation = None
    if not math.isclose(float(meta[score_key]), float(score), abs_tol=1e-4):
        raise ValueError("任-07最佳完整模型与合法HF选分或独立物理真实评分不一致")
    return snapshot, model, validation


def _resume_preflight(
    output: Path, latest: dict[str, Any], source: Task07Source, architecture: dict[str, Any],
    registry_sha: str | None, diagnostic_only: bool, sensor_count: int,
    validation_loader: DataLoader, validation_sensor: tuple[Tensor, ...],
    selection_weights: Mapping[str, float], physics: PhysicsLossComputer,
    materials: Any, boundaries: Any, device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], nn.Module, dict[str, float]]:
    meta = latest["metadata"]
    global_epoch = int(meta["全局累计实际轮次"])
    records = [json.loads(line) for line in (output / "training.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()]
    _log_budget(records, global_epoch, sensor_count, meta)
    to_repair = []
    observed = None
    for name in ("阶段_观测最佳.pt", "阶段_物理最佳.pt"):
        full, best_model, validation = _verify_snapshot_best(
            output, latest, name, source, registry_sha, diagnostic_only,
            validation_loader, validation_sensor, selection_weights,
            physics, materials, boundaries, device,
        )
        sidecar = output / name
        old = torch.load(sidecar, map_location="cpu", weights_only=False) if sidecar.is_file() else None
        selected_global = meta["观测最佳全局轮次" if name == "阶段_观测最佳.pt"
                               else "物理最佳全局轮次"]
        if selected_global == global_epoch and (
            old is None or old["metadata"]["全局累计实际轮次"] != global_epoch
        ):
            to_repair.append(sidecar)
        if name == "阶段_观测最佳.pt":
            observed = (full, best_model, validation)
    if observed is None or observed[2] is None:
        raise ValueError("任-07真实观测最佳完整阶段及合法HF验证不可恢复")
    best_state, best_model, validation = observed
    view_file = output / "best.pt"
    view = torch.load(view_file, map_location="cpu", weights_only=False) if view_file.is_file() else None
    best_epoch = int(meta["观测最佳全局轮次"])
    if view is not None and view.get("epoch", -1) > global_epoch:
        raise ValueError("任-07 best.pt超前最近真实轮次，不得冒用未提交模型")
    if view is not None and view.get("epoch") == best_epoch:
        lineage = view.get("任07新HF模型视图来源", {})
        if (
            view.get("correction_model_kwargs") != source.correction_model_kwargs
            or view.get("scales") != source.scales
            or view.get("low_fidelity_model_kwargs") != source.lf_model_kwargs
            or lineage.get("训练种子") != source.seed
            or lineage.get("运行臂") != "E0"
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
            or not _equal(view.get("model_state"), best_state["model_state"])
            or not math.isclose(float(view.get("validation_selection_score_c", math.inf)),
                                float(meta["观测最佳选分_摄氏度"]), abs_tol=1e-4)
            or not math.isclose(float(view.get("validation_rmse_c", math.inf)),
                                float(validation["顶部"]), abs_tol=1e-4)
            or any(not math.isclose(float(view.get("validation_sensor", {}).get(key, math.inf)),
                                    float(value), abs_tol=1e-4)
                   for key, value in validation.items() if key != "顶部")
        ):
            raise ValueError("任-07当前best.pt与历史HF架构、新E0六列、合法HF评分或真实完整阶段不一致")
    # No log or stage write occurs until the old best, physics state and view all pass.
    if len(records) > global_epoch:
        suffix = time.strftime("%Y%m%dT%H%M%S")
        archive = output / f"training_中断尾行_{suffix}.jsonl"
        if archive.exists():
            raise FileExistsError("任-07中断日志归档已有同名文件，不得覆盖旧行")
        archive.write_text("\n".join(json.dumps(row, ensure_ascii=False)
                                     for row in records[global_epoch:]) + "\n", encoding="utf-8")
        temporary = (output / "training.jsonl").with_name("training.jsonl.tmp")
        temporary.write_text("\n".join(json.dumps(row, ensure_ascii=False)
                                      for row in records[:global_epoch]) + "\n", encoding="utf-8")
        temporary.replace(output / "training.jsonl")
    for file in to_repair:
        temporary = file.with_name(file.name + ".tmp")
        torch.save(latest, temporary)
        temporary.replace(file)
    if view is None or view.get("epoch") != best_epoch:
        _model_view(view_file, architecture, best_model, source,
                    global_epoch=best_epoch,
                    correction_epoch=best_state["metadata"]["校正实际轮次"],
                    joint_epoch=best_state["metadata"]["联合实际轮次"],
                    score=meta["观测最佳选分_摄氏度"], validation=validation,
                    registry_sha=registry_sha, diagnostic_only=diagnostic_only)
    return records[:global_epoch], best_state, best_model, validation


def _joint_optimizer(model: nn.Module, optimizer: torch.optim.AdamW,
                     source: Task07Source, *, loading_committed_state: bool = False) -> None:
    if len(optimizer.param_groups) != 1:
        raise ValueError("任-07联合切换必须续用真实HF单组AdamW而非重建历史动量")
    optimizer.param_groups[0]["lr"] = 0.0001
    model.freeze_low_fidelity(True)
    named = dict(model.named_parameters())
    for name in PROJECTION_NAMES:
        named[name].requires_grad_(True)
    optimizer.add_param_group({
        "params": [parameter for name, parameter in named.items() if name in PROJECTION_NAMES],
        "lr": 0.00001, "weight_decay": optimizer.param_groups[0]["weight_decay"],
    })
    _check_model(model, source, JOINT_STAGE)
    if not loading_committed_state and len(optimizer.state_dict()["state"]) == 0:
        raise ValueError("任-07联合作业必须从真实校正步骤续用已训练HF AdamW动量")


def _simulation_replay(source: Task07Source) -> Any:
    data = load_sampled_points(
        list(source.lf_train_powers_w), 2048, seed=7_100_000 + source.seed,
        sampling_mode="material_time",
    )
    if len(data) != 60 * 2048:
        raise ValueError("任-07每个LF仿真训练功率必须实取2048点，不允许缺功率或短样")
    coordinates = data.tensors[0]
    actual = {round(float(value), 4): int((torch.isclose(
        coordinates[:, 3], value, atol=1e-4, rtol=0.0,
    )).sum()) for value in torch.unique(coordinates[:, 3])}
    if set(actual) != set(source.lf_train_powers_w) or set(actual.values()) != {2048}:
        raise ValueError("任-07 LF每轮60真实功率各2048低保真标签来源错误")
    return data


def run_task07_formal(
    *, seed: int, output_directory: str | Path,
    session_epoch_limit: int | None = None, diagnostic_only: bool = False,
    diagnostic_joint_preview: bool = False,
    resume_training_checkpoint: str | Path | None = None,
    formal_registry_path: str | Path | None = None,
    formal_registry_sha256: str | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError("任-07五种子只允许0到4的锁定来源")
    if diagnostic_only and session_epoch_limit not in (1, 2):
        raise ValueError("任-07无预登记只允许每次1或2轮诊断，不得冒充1500＋500轮正式结果")
    if diagnostic_joint_preview and (not diagnostic_only or resume_training_checkpoint is None):
        raise ValueError("任-07未满1500的联合短预览只能是不可采用的诊断")
    if session_epoch_limit is not None and not 1 <= session_epoch_limit <= 2000:
        raise ValueError("任-07会话轮数只能在同组1500＋500总预算内")
    if resume_training_checkpoint is not None and session_epoch_limit is None:
        raise ValueError("任-07续跑须写明本次增加轮数与原阶段上限")
    device = torch.device(device_name or ("cpu" if diagnostic_only else "cuda"))
    if diagnostic_only and device.type != "cpu":
        raise ValueError("任-07不可采用的短诊断仅允许CPU，不能占用CUDA正式设备")
    if not diagnostic_only and (device.type != "cuda" or not torch.cuda.is_available()):
        raise ValueError("任-07正式五种子完整训练必须真实CUDA且保存四类随机源")
    if not diagnostic_only:
        _require_formal_registration(formal_registry_path, formal_registry_sha256)
    if diagnostic_only and (formal_registry_path is not None or formal_registry_sha256 is not None):
        raise ValueError("任-07短诊断不复用将来的正式预登记资格")
    registry_sha = None if diagnostic_only else formal_registry_sha256
    sources = validate_task07_sources()
    if not diagnostic_only:
        _require_formal_registration(formal_registry_path, formal_registry_sha256, sources)
    source = sources[seed]
    initial = fork_task07_e0(sources, seed, device)
    if not diagnostic_only and (
        not isinstance(initial.random_state["torch_cuda"], (tuple, list))
        or not initial.random_state["torch_cuda"]
    ):
        raise ValueError("任-07正式模型初态必须有CUDA随机源，不得以CPU短诊断续跑")
    model, optimizer = initial.model, initial.optimizer
    _check_model(model, source, CORRECTION_STAGE)
    output = _resolve(output_directory)
    if resume_training_checkpoint is None and output.exists():
        raise FileExistsError(f"任-07训练目录已有产物，禁止覆盖：{output}")
    if resume_training_checkpoint is not None and not output.is_dir():
        raise FileNotFoundError(f"任-07原seed目录不存在，不得跨seed续跑：{output}")
    training = load_yaml("configs/training.yaml")
    optimizer_config = training["optimizer"]
    losses = training["loss_weights"]
    selections = training["multifidelity_selection_weights"]
    stopping = training["early_stopping"]
    splits = build_power_splits()
    assert_no_hf_leakage(
        {"HF训练顶部": splits.hf_train, "HF训练环温": splits.hf_train},
        splits.hf_validation | splits.hf_test | splits.external_sensor_test,
    )
    if (
        optimizer_config["name"] != "adamw"
        or float(optimizer_config["learning_rate"]) != 0.001
        or float(optimizer_config["joint_learning_rate"]) != 0.0001
        or float(stopping["patience"]) != 200
        or float(stopping["min_delta"]) != 0.0001
        or training["selection_metric_version"] != "macro_v1"
        or training["epochs"]["high_fidelity"] != 1500
        or training["epochs"]["joint"] != 500
        or [losses["sensor_absolute"], losses["sensor_delta"]] != [5.0, 1.0]
        or [losses[key] for key in ("ir", "pde", "boundary", "initial", "interface")]
        != [1.0] * 5
        or float(losses["low_fidelity"]) != 1.0
        or training["multifidelity"]["joint_simulation_samples_per_power"] != 2048
        or splits.hf_train != frozenset(source.hf_train_powers_w)
        or splits.hf_validation != frozenset(source.hf_validation_powers_w)
        or splits.simulation_train != frozenset(source.lf_train_powers_w)
        or splits.simulation_validation != frozenset(source.lf_validation_powers_w)
    ):
        raise ValueError("任-07五seed实际LF/HF固定划分、HF旧学习率、早停或名义损失配置发生漂移")
    materials = load_materials()
    boundaries = load_resolved_boundary_conditions()
    physics = PhysicsLossComputer(materials, boundaries, PhysicsLossWeights(
        pde=float(losses["pde"]), boundary=float(losses["boundary"]),
        initial=float(losses["initial"]), interface=float(losses["interface"]),
    ))
    train_data = _ir_dataset("train", source.hf_train_powers_w)
    validation_data = _ir_dataset("validation", source.hf_validation_powers_w)
    validation_loader = DataLoader(validation_data, batch_size=2048, shuffle=False)
    sensor = _sensor_tensors(device, split="train", powers_w=source.hf_train_powers_w)
    validation_sensor = _sensor_tensors(device, split="validation",
                                        powers_w=source.hf_validation_powers_w)
    if len(train_data) != 29593 or len(sensor[0]) == 0:
        raise RuntimeError("任-07本seed真实HF训练必须为15批29593观测点的合法12工况顶部/环温")
    architecture = torch.load(source.hf_checkpoint_path, map_location="cpu", weights_only=False)
    consumption = {
        "HF训练观测点": 0, "HF训练传感器点": 0,
        "LF真实回放训练点": 0, "物理配点": 0,
        "HF观测优化步": 0, "物理优化步": 0,
        "LF联合回放batch": 0,
    }
    correction_epoch = joint_epoch = global_epoch = 0
    correction_completed = None
    correction_terminal_sha = diagnostic_switch_sha = None
    phase_best_epoch = best_global_epoch = best_stage_epoch = 0
    physical_global_epoch = physical_stage_epoch = 0
    best_stage = physical_stage = stage = CORRECTION_STAGE
    lf_keep: dict[str, Any] | None = None
    lf_reference: dict[str, dict[str, float]]
    began = time.perf_counter()
    if resume_training_checkpoint is None:
        output.mkdir(parents=True)
        write_config_snapshot(output)
        (output / "training.jsonl").write_text("", encoding="utf-8")
        initial_score, initial_validation = _validation_selection(
            model, validation_loader, validation_sensor, device, selections,
        )
        physical_score, _ = _score_guardrails(model, physics, materials, boundaries, device)
        lf_reference = _lf_material_validation(model, list(source.lf_validation_powers_w), device)
        initial_modalities = _hf_modalities(model, device, list(source.hf_validation_powers_w))
        best_score = initial_score
        _rng_restore(initial.random_state)
        metadata = _metadata(
            source, model, stage, registry_sha, diagnostic_only,
            correction_epoch=0, joint_epoch=0, correction_completed=None,
            initial_score=initial_score, best_score=best_score,
            best_global_epoch=0, best_stage=stage, best_stage_epoch=0,
            physical_score=physical_score, physical_global_epoch=0,
            physical_stage=stage, physical_stage_epoch=0, phase_best_epoch=0,
            lf_reference=lf_reference, consumption=consumption,
        )
        for name in ("阶段_初始.pt", "阶段_观测最佳.pt", "阶段_物理最佳.pt"):
            _save_state(output / name, model, optimizer, source,
                        stage=CORRECTION_STAGE, epoch=0, metadata=metadata)
        _model_view(
            output / "best.pt", architecture, model, source, global_epoch=0,
            correction_epoch=0, joint_epoch=0, score=initial_score,
            validation=initial_validation, registry_sha=registry_sha,
            diagnostic_only=diagnostic_only,
        )
    else:
        resume = _resolve(resume_training_checkpoint).resolve()
        if resume.parent != output.resolve() or resume.name not in ("阶段_初始.pt", "阶段_最近.pt"):
            raise ValueError("任-07仅本seed阶段_初始或阶段_最近完整状态可真实续跑，旧best不能冒用")
        _snapshot_hashes(output)
        candidate = torch.load(resume, map_location="cpu", weights_only=False)
        _check_snapshot_identity(candidate, source, registry_sha, diagnostic_only)
        stage = candidate["stage"]
        if stage == JOINT_STAGE:
            _joint_optimizer(model, optimizer, source, loading_committed_state=True)
        _check_model(model, source, stage)
        payload = load_training_state(resume, model, optimizer)
        _check_model(model, source, stage)
        if stage == JOINT_STAGE and not optimizer.state_dict()["state"]:
            raise ValueError("任-07联合已提交阶段缺真实HF校正AdamW动量")
        metadata = payload["metadata"]
        correction_epoch = int(metadata["校正实际轮次"])
        joint_epoch = int(metadata["联合实际轮次"])
        global_epoch = correction_epoch + joint_epoch
        correction_completed = metadata["校正实际截止轮次"]
        correction_terminal_sha = metadata.get("已提交正式校正末原件SHA256")
        diagnostic_switch_sha = metadata.get("已提交诊断校正切换原件SHA256")
        if stage == CORRECTION_STAGE and (
            payload["epoch"] != correction_epoch
            or (correction_epoch == 1500 and correction_completed != 1500)
        ):
            raise ValueError("任-07校正阶段真实已完/阶段上限不能被最新状态伪装")
        if stage == JOINT_STAGE and (
            payload["epoch"] != joint_epoch
            or (not diagnostic_only and correction_completed != correction_epoch)
        ):
            raise ValueError("任-07联合阶段须承接同seed正式校正真实截止原件")
        if stage == JOINT_STAGE:
            name = "阶段_诊断校正切换.pt" if diagnostic_only else "阶段_校正末.pt"
            previous_sha = diagnostic_switch_sha if diagnostic_only else correction_terminal_sha
            if not (output / name).is_file() or sha256_file(output / name) != previous_sha:
                raise ValueError("任-07受限联合的原校正真实末态/诊断切换原件SHA失配")
        elif correction_completed is not None and not diagnostic_only:
            terminal = output / "阶段_校正末.pt"
            if not terminal.is_file():
                raise ValueError("任-07已截止的校正真实末态原件缺失，不得创建联合阶段")
            correction_terminal_sha = sha256_file(terminal)
            terminal_state = torch.load(terminal, map_location="cpu", weights_only=False)
            _check_snapshot_identity(terminal_state, source, registry_sha, diagnostic_only)
            if terminal_state["epoch"] != correction_epoch or any(
                not _equal(terminal_state[key], payload[key])
                for key in ("model_state", "optimizer_state", "random_state",
                            "parameter_requires_grad", "metadata")
            ):
                raise ValueError("任-07校正已截止但正式末态与最近阶段非同源完整状态")
        if joint_epoch == 500:
            raise ValueError("任-07联合已达到原500轮上限，不得追加隐形训练")
        if diagnostic_joint_preview and (stage != CORRECTION_STAGE or correction_epoch < 2):
            raise ValueError("任-07诊断联合只能从真实已训练至少2轮校正最近状态切换")
        _resume_preflight(
            output, payload, source, architecture, registry_sha, diagnostic_only,
            len(sensor[0]), validation_loader, validation_sensor, selections,
            physics, materials, boundaries, device,
        )
        _rng_restore(payload["random_state"])
        initial_score = float(metadata["初始合法HF选分_摄氏度"])
        best_score = float(metadata["观测最佳选分_摄氏度"])
        best_global_epoch = int(metadata["观测最佳全局轮次"])
        best_stage = str(metadata["观测最佳阶段"])
        best_stage_epoch = int(metadata["观测最佳阶段轮次"])
        physical_score = float(metadata["物理最佳独立损失"])
        physical_global_epoch = int(metadata["物理最佳全局轮次"])
        physical_stage = str(metadata["物理最佳阶段"])
        physical_stage_epoch = int(metadata["物理最佳阶段轮次"])
        phase_best_epoch = int(metadata["本阶段早停最佳轮次"])
        phase_best_score = float(metadata["本阶段最低合格HF选分_摄氏度"])
        lf_reference = metadata["LF合法验证初态逐材料节点与体积RMSE_摄氏度"]
        lf_keep = metadata.get("LF逐材料节点和真实体积5%护栏")
        consumption = metadata["累计实际消耗"].copy()
        initial_modalities = _hf_modalities(model, device, list(source.hf_validation_powers_w))
        _rng_restore(payload["random_state"])
    last_row: dict[str, Any] = {}
    more = (2000 if session_epoch_limit is None else session_epoch_limit)
    session_global_end = min(2000, global_epoch + more)
    session_end = min(1500, session_global_end)
    for epoch in range(correction_epoch + 1 if stage == CORRECTION_STAGE
                       and correction_completed is None and not diagnostic_joint_preview
                       else session_end + 1, session_end + 1):
        started_epoch = time.perf_counter()
        global_epoch = correction_epoch = epoch
        actual = _hf_epoch(model, optimizer, train_data, sensor, physics, losses,
                           device, seed, global_epoch, simulation_data=None)
        _check_model(model, source, CORRECTION_STAGE)
        for key in consumption:
            consumption[key] += actual[key] if key in actual else (15 if key == "HF观测优化步" else
                                                             1 if key == "物理优化步" else
                                                             256 if key == "物理配点" else 0)
        due = epoch % 10 == 0 or epoch == 1500 or (diagnostic_only and epoch == session_end)
        score = validation = modalities = guardrails = None
        physics_now = None
        if due:
            score, validation = _validation_selection(model, validation_loader,
                                                       validation_sensor, device, selections)
            modalities = _hf_modalities(model, device, list(source.hf_validation_powers_w))
            physics_now, guardrails = _score_guardrails(
                model, physics, materials, boundaries, device,
            )
            if score < best_score - 0.0001:
                best_score, best_global_epoch, best_stage, best_stage_epoch = (
                    score, epoch, CORRECTION_STAGE, epoch,
                )
                phase_best_epoch = epoch
            if physics_now < physical_score:
                physical_score, physical_global_epoch, physical_stage, physical_stage_epoch = (
                    physics_now, epoch, CORRECTION_STAGE, epoch,
                )
        boundary_stop = not diagnostic_only and due and (
            epoch == 1500 or epoch - phase_best_epoch >= 200
        )
        if boundary_stop:
            correction_completed = epoch
        last_row = {
            "epoch": epoch, "全局实际轮次": epoch,
            "运行种子": seed, "运行臂": "E0", "训练阶段": CORRECTION_STAGE,
            "阶段实际轮次": epoch,
            "HF观测优化步": 15, "物理优化步": 1, "物理配点": 256,
            "HF合法验证选分_摄氏度": score, "HF合法验证分模态_摄氏度": modalities,
            "LF合法验证初态逐材料节点与体积RMSE_摄氏度": lf_reference if due else None,
            "LF当前真实张量SHA256": _lf_sha(model),
            "LF真实回放训练点": 0, "LF联合回放batch": 0,
            "本轮包含验证墙钟秒": time.perf_counter() - started_epoch,
            "累计实际消耗": consumption.copy(),
            **actual,
        }
        if guardrails is not None:
            last_row.update(guardrails)
        with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(last_row, ensure_ascii=False) + "\n")
        metadata = _metadata(
            source, model, CORRECTION_STAGE, registry_sha, diagnostic_only,
            correction_epoch=epoch, joint_epoch=0, correction_completed=correction_completed,
            initial_score=initial_score, best_score=best_score,
            best_global_epoch=best_global_epoch, best_stage=best_stage,
            best_stage_epoch=best_stage_epoch, physical_score=physical_score,
            physical_global_epoch=physical_global_epoch, physical_stage=physical_stage,
            physical_stage_epoch=physical_stage_epoch,
            phase_best_epoch=phase_best_epoch, lf_reference=lf_reference,
            consumption=consumption,
        )
        for name, selected in (("阶段_观测最佳.pt", best_global_epoch),
                               ("阶段_物理最佳.pt", physical_global_epoch)):
            if selected < epoch:
                previous = output / name
                if not previous.is_file():
                    raise ValueError("任-07历史最佳完整原件丢失，不得提交新最近状态")
                metadata[f"已提交旧{name}SHA256"] = sha256_file(previous)
        _save_state(output / "阶段_最近.pt", model, optimizer, source,
                    stage=CORRECTION_STAGE, epoch=epoch, metadata=metadata)
        if best_global_epoch == epoch:
            _save_state(output / "阶段_观测最佳.pt", model, optimizer, source,
                        stage=CORRECTION_STAGE, epoch=epoch, metadata=metadata)
        if physical_global_epoch == epoch:
            _save_state(output / "阶段_物理最佳.pt", model, optimizer, source,
                        stage=CORRECTION_STAGE, epoch=epoch, metadata=metadata)
        if best_global_epoch == epoch:
            _model_view(
                output / "best.pt", architecture, model, source,
                global_epoch=epoch, correction_epoch=epoch, joint_epoch=0,
                score=best_score, validation=validation,
                registry_sha=registry_sha, diagnostic_only=diagnostic_only,
            )
        if boundary_stop:
            _save_state(output / "阶段_校正末.pt", model, optimizer, source,
                        stage=CORRECTION_STAGE, epoch=epoch, metadata=metadata)
            break
    if stage == CORRECTION_STAGE and global_epoch < session_global_end and (
        diagnostic_joint_preview or (not diagnostic_only and correction_completed is not None)
    ):
        if diagnostic_joint_preview:
            _save_state(output / "阶段_诊断校正切换.pt", model, optimizer, source,
                        stage=CORRECTION_STAGE, epoch=correction_epoch,
                        metadata=metadata)
            diagnostic_switch_sha = sha256_file(output / "阶段_诊断校正切换.pt")
        else:
            formal_end = output / "阶段_校正末.pt"
            if not formal_end.is_file():
                raise ValueError("任-07正式校正实际截止原件未提交，联合不可开始")
            correction_terminal_sha = sha256_file(formal_end)
        replay = _simulation_replay(source)
        phase_best_score, _ = _validation_selection(
            model, validation_loader, validation_sensor, device, selections,
        )
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
            best_stage_epoch=best_stage_epoch, physical_score=physical_score,
            physical_global_epoch=physical_global_epoch, physical_stage=physical_stage,
            physical_stage_epoch=physical_stage_epoch, phase_best_epoch=0,
            phase_best_score=phase_best_score,
            lf_reference=lf_reference, consumption=consumption,
            correction_terminal_sha=correction_terminal_sha,
            diagnostic_switch_sha=diagnostic_switch_sha,
        )
        _save_state(output / "阶段_最近.pt", model, optimizer, source,
                    stage=JOINT_STAGE, epoch=0, metadata=metadata)
        _save_state(output / "阶段_联合初始.pt", model, optimizer, source,
                    stage=JOINT_STAGE, epoch=0, metadata=metadata)
    elif stage == JOINT_STAGE:
        replay = _simulation_replay(source)
    else:
        replay = None
    if stage == JOINT_STAGE and global_epoch < session_global_end:
        joint_end = min(500, joint_epoch + session_global_end - global_epoch)
        for local_epoch in range(joint_epoch + 1, joint_end + 1):
            started_epoch = time.perf_counter()
            joint_epoch = local_epoch
            global_epoch = correction_epoch + joint_epoch
            actual = _hf_epoch(model, optimizer, train_data, sensor, physics, losses,
                               device, seed, global_epoch, simulation_data=replay)
            _check_model(model, source, JOINT_STAGE)
            consumption["HF训练观测点"] += actual["HF训练观测点"]
            consumption["HF训练传感器点"] += actual["HF训练传感器点"]
            consumption["LF真实回放训练点"] += actual["LF真实回放训练点"]
            consumption["物理配点"] += 256
            consumption["HF观测优化步"] += 15
            consumption["物理优化步"] += 1
            consumption["LF联合回放batch"] += 60
            due = (joint_epoch % 10 == 0 or joint_epoch == 500
                   or (diagnostic_only and joint_epoch == joint_end))
            score = validation = modalities = guardrails = lf_values = None
            if due:
                score, validation = _validation_selection(
                    model, validation_loader, validation_sensor, device, selections,
                )
                modalities = _hf_modalities(model, device, list(source.hf_validation_powers_w))
                lf_values = _lf_material_validation(
                    model, list(source.lf_validation_powers_w), device,
                )
                lf_keep = task04_lf_keep_guardrail(lf_reference, lf_values)
                physics_now, guardrails = _score_guardrails(
                    model, physics, materials, boundaries, device,
                )
                if lf_keep["LF两材料节点与真实体积均守住5%护栏"]:
                    phase_best_score, phase_best_epoch, _, global_improved = _joint_best_update(
                        score=score, global_best_score=best_score,
                        phase_best_score=phase_best_score,
                        phase_best_epoch=phase_best_epoch, joint_epoch=joint_epoch,
                        eligible=True,
                    )
                    if global_improved:
                        best_score, best_global_epoch, best_stage, best_stage_epoch = (
                            score, global_epoch, JOINT_STAGE, joint_epoch,
                        )
                    if physics_now < physical_score:
                        physical_score, physical_global_epoch, physical_stage, physical_stage_epoch = (
                            physics_now, global_epoch, JOINT_STAGE, joint_epoch,
                        )
            last_row = {
                "epoch": global_epoch, "全局实际轮次": global_epoch,
                "运行种子": seed, "运行臂": "E0", "训练阶段": JOINT_STAGE,
                "阶段实际轮次": joint_epoch,
                "HF观测优化步": 15, "物理优化步": 1, "物理配点": 256,
                "HF合法验证选分_摄氏度": score,
                "HF合法验证分模态_摄氏度": modalities,
                "LF合法验证逐材料节点与体积RMSE_摄氏度": lf_values,
                "LF逐材料节点和真实体积5%护栏": lf_keep,
                "LF当前真实张量SHA256": _lf_sha(model),
                "本轮包含验证墙钟秒": time.perf_counter() - started_epoch,
                "累计实际消耗": consumption.copy(),
                **actual,
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
                best_stage_epoch=best_stage_epoch, physical_score=physical_score,
                physical_global_epoch=physical_global_epoch, physical_stage=physical_stage,
                physical_stage_epoch=physical_stage_epoch, phase_best_epoch=phase_best_epoch,
                phase_best_score=phase_best_score,
                lf_reference=lf_reference, consumption=consumption, lf_keep=lf_keep,
                correction_terminal_sha=correction_terminal_sha,
                diagnostic_switch_sha=diagnostic_switch_sha,
            )
            for name, selected in (("阶段_观测最佳.pt", best_global_epoch),
                                   ("阶段_物理最佳.pt", physical_global_epoch)):
                if selected < global_epoch:
                    previous = output / name
                    if not previous.is_file():
                        raise ValueError("任-07两类历史最佳真实完整阶段缺失，拒绝联合最新提交")
                    metadata[f"已提交旧{name}SHA256"] = sha256_file(previous)
            _save_state(output / "阶段_最近.pt", model, optimizer, source,
                        stage=JOINT_STAGE, epoch=joint_epoch, metadata=metadata)
            if best_global_epoch == global_epoch:
                _save_state(output / "阶段_观测最佳.pt", model, optimizer, source,
                            stage=JOINT_STAGE, epoch=joint_epoch, metadata=metadata)
            if physical_global_epoch == global_epoch:
                _save_state(output / "阶段_物理最佳.pt", model, optimizer, source,
                            stage=JOINT_STAGE, epoch=joint_epoch, metadata=metadata)
            if best_global_epoch == global_epoch:
                _model_view(
                    output / "best.pt", architecture, model, source,
                    global_epoch=global_epoch, correction_epoch=correction_epoch,
                    joint_epoch=joint_epoch, score=best_score, validation=validation,
                    registry_sha=registry_sha, diagnostic_only=diagnostic_only,
                )
            boundary_stop = not diagnostic_only and due and (
                joint_epoch == 500 or joint_epoch - phase_best_epoch >= 200
            )
            if boundary_stop:
                for name in ("阶段_联合末.pt", "阶段_训练末.pt"):
                    _save_state(output / name, model, optimizer, source,
                                stage=JOINT_STAGE, epoch=joint_epoch, metadata=metadata)
                break
    status = (
        "短诊断提前联合只验证通路；未满正式1500＋500，不得计正式五种子" if diagnostic_joint_preview else
        "短诊断暂停；不得作任07正式1500＋500阶段或五种子结论" if diagnostic_only else
        "正式受限联合已达同组截止，仍须五种子能源审计" if (output / "阶段_联合末.pt").is_file() else
        "正式受限联合会话暂停，需续跑真实阶段" if stage == JOINT_STAGE else
        "正式HF校正实际截止，待同seed校正末原件进入受限联合" if correction_completed is not None else
        "正式HF校正会话暂停，可据最近完整状态接续"
    )
    if stage == JOINT_STAGE and lf_keep is not None and not lf_keep["LF两材料节点与真实体积均守住5%护栏"]:
        status += "；真实LF验证未守住Cu/SiC逐材料节点与真实体积5%护栏，本联合模型不可采用"
    result = {
        "状态": status, "运行资格": _qualification(diagnostic_only),
        "运行种子": seed, "运行臂": "E0",
        "正式预登记配置SHA256": registry_sha,
        "本seed源LF检查点SHA256": source.lf_checkpoint_sha256,
        "本seed真实LF初始张量SHA256": source.lf_tensor_sha256,
        "本seed历史HF架构视图SHA256": source.hf_checkpoint_sha256,
        "校正实际轮次": correction_epoch, "联合实际轮次": joint_epoch,
        "原校正预算上限": 1500, "原受限联合预算上限": 500,
        "初始HF合法选分_摄氏度": initial_score,
        "观测最佳HF合法选分_摄氏度": best_score,
        "观测最佳全局轮次": best_global_epoch,
        "受限联合本阶段最低合格HF选分_摄氏度": (
            phase_best_score if stage == JOINT_STAGE else None
        ),
        "物理最佳独立损失": physical_score,
        "物理最佳全局轮次": physical_global_epoch,
        "LF合法验证初态Cu_SiC节点及真实体积RMSE_摄氏度": lf_reference,
        "LF合法验证本轮Cu_SiC节点及真实体积RMSE_摄氏度": (
            last_row.get("LF合法验证逐材料节点与体积RMSE_摄氏度") if stage == JOINT_STAGE
            else lf_reference
        ),
        "LF逐材料节点和真实体积5%护栏": lf_keep,
        "LF保持资格": _lf_keep_qualification(stage, lf_keep),
        "LF血缘说明": (
            "五seed分别用V4 B0配对真实LF与同seed历史HF架构；seed0 LF张量与任04冻结臂"
            "原LF逐键相同，SiC源基准节点9.356303℃/真实体积7.533923℃；其联合后"
            "误差变化须按真实四投影更新及同口径合法验证归因，不能称来源LF先验变化"
            if seed == 0 else
            "本seed用独立V4 B0配对真实LF，不可把seed0任04校正末的绝对验证值"
            "直接充作本seed LF初始误差"
        ),
        "HF合法验证顶部热端冷端RMSE_摄氏度": last_row.get("HF合法验证分模态_摄氏度") or initial_modalities,
        "旧test_Data温度标签读取": False,
        "物理资格": "名义独立物理损失只供双轨选择；能源审计由后续独立完成",
        "真实最后状态": (
            "阶段_训练末.pt" if (output / "阶段_训练末.pt").is_file() else "阶段_最近.pt"
        ),
        "累计实际消耗": consumption.copy(), "最近一轮真实入场证据": last_row,
        "本会话耗时秒": time.perf_counter() - began,
    }
    temporary = output / "阶段报告.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "阶段报告.json")
    return result
