"""Registered Task-06 same-source HF response-feature comparison."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import assert_no_hf_leakage, build_power_splits
from sic_cu.eval.protocol_checks import validate_hf_checkpoint_provenance
from sic_cu.losses import PhysicsLossComputer, PhysicsLossWeights
from sic_cu.models import AdditiveCorrectionModel, DeepONetPINN, ModelScales
from sic_cu.models.residual_interpolation import ChebyshevSurfaceResidualGuide
from sic_cu.physics import sample_collocation
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.train.common import (
    CONFIG_FILES, load_training_state, physics_optimizer_step, save_training_state,
    write_config_snapshot,
)
from sic_cu.train.multifidelity import (
    _ir_dataset, _macro_sensor_training_losses, _sensor_tensors,
    evaluate_schedule_guardrails, reconcile_correction_resume_log,
)
from sic_cu.train.simulation import build_model
from sic_cu.train.task04_joint import (
    _hf_modalities, _lf_material_validation, _validation_selection,
)


CONFIG = PROJECT_ROOT / "研究记录/任务06_时间响应特征/HF三臂先导有效配置.yaml"
CONFIG_SHA = "7a418d94493ddafd9c78e5070bca0739ab2de30aecdc22ef322ef69bc7e03b19"
ARMS = ("E0", "E1", "E2")
E1_TAU = (2.0, 10.0, 50.0, 200.0)
STAGE_BUDGET = {"任06HF校正先导轮次": 600}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _actual_lf_tensor_sha256(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(key for key in state if key.startswith("low_fidelity_model.")):
        value = state[name].detach().cpu().contiguous()
        header = f"{name}|{value.dtype}|{tuple(value.shape)}|".encode("ascii")
        digest.update(header)
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _registered_tau(config: dict[str, Any], arm: str) -> tuple[float, ...] | None:
    if arm == "E0":
        return None
    if arm == "E1":
        return E1_TAU
    if arm != "E2":
        raise ValueError("任-06只允许E0/E1/E2三臂")
    evidence = config["E2仅LF原始训练曲线字典"]
    directory = PROJECT_ROOT / "研究记录/任务06_时间响应特征"
    registration = _resolve(evidence["预登记文件"])
    diagnostic = _resolve(evidence["首次机器诊断"])
    nodes = diagnostic.parent / "节点冻结清单.json"
    if (
        registration != directory / "字典拟合前登记.yaml"
        or diagnostic != directory / "LF字典准备_20260915T202534+0800/时间字典诊断.json"
        or sha256_file(registration) != evidence["预登记_SHA256"]
        or sha256_file(diagnostic) != evidence["首次机器诊断_SHA256"]
        or sha256_file(nodes) != evidence["节点冻结清单_SHA256"]
    ):
        raise ValueError("任-06 E2纯LF TRAIN字典登记、首次诊断或冻结节点SHA不一致")
    source = json.loads(diagnostic.read_text(encoding="utf-8"))
    tau = tuple(float(value) for value in evidence["tau秒"])
    if (
        tuple(source["冻结共同tau_秒"]) != tau
        or source["预登记SHA256"] != evidence["预登记_SHA256"]
        or source["节点冻结清单SHA256"] != evidence["节点冻结清单_SHA256"]
        or source["LF训练曲线数"] != 3840
        or source["选用TRAIN正则目标最小且收敛的注册初值组"] != 2
    ):
        raise ValueError("任-06 E2响应tau不属于登记的60个LF TRAIN冻结拟合")
    return tau


def validate_task06_source() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(CONFIG) != CONFIG_SHA:
        raise ValueError("任-06事前HF三臂有效配置字节SHA改变，不得创建训练输出")
    config = load_yaml(str(CONFIG))
    stage = _resolve(config["共同实际全模型起点"]).resolve()
    view_file = stage.parent / "best.pt"
    if (
        config.get("任务") != "任-06" or config.get("共同先导种子") != 0
        or config.get("每臂追加HF校正轮数") != 600
        or config.get("HF学习率") != 1e-4 or config.get("LF学习率") != 0.0
        or config.get("HF训练功率数") != 12
        or sorted(config.get("HF合法验证功率_瓦", [])) != sorted(build_power_splits().hf_validation)
        or config.get("LF仿真训练功率数") != 60
        or config.get("LF仿真合法验证功率数") != 10
        or any(part not in str(config.get("观察最佳选择", "")) for part in (
            "每10轮", "第600轮", "macro_v1", "早停耐心601轮",
        ))
        or stage.name != "阶段_观测最佳.pt"
        or sha256_file(stage) != config["共同完整训练状态_SHA256"]
        or sha256_file(view_file) != config["共同观测最佳模型视图_SHA256"]
    ):
        raise ValueError("任-06三臂同源登记、600轮预算或合法12训练/3验证配置不一致")
    state = torch.load(stage, map_location="cpu", weights_only=False)
    architecture = torch.load(view_file, map_location="cpu", weights_only=False)
    if (
        state.get("training_state_schema_version") != 1
        or state.get("stage") != "joint" or state.get("epoch") != 120
        or state.get("budget") != {"任04联合续训轮次": 200}
        or state.get("metadata", {}).get("任04运行臂") != "有限解冻"
        or architecture.get("epoch") != 420
        or architecture.get("method") != "multifidelity_correction"
        or architecture.get("low_fidelity_method") != "deeponet_pinn"
        or architecture.get("任04新模型视图来源", {}).get("任04本阶段实际轮次") != 120
        or set(architecture["model_state"]) != set(state["model_state"])
        or any(not torch.equal(value, architecture["model_state"][name])
               for name, value in state["model_state"].items())
        or tuple(state["model_state"]["correction.0.weight"].shape) != (128, 6)
        or any(name.startswith("response_features.") for name in state["model_state"])
        or set(state.get("random_state", {})) != {"python", "numpy", "torch_cpu", "torch_cuda"}
        or state["random_state"]["torch_cuda"] is None
    ):
        raise ValueError("任-06共同源必须是真实任04有限联合第120轮完整6列模型，不得使用旧best优化器")
    lf_parameters = {name for name in state["parameter_requires_grad"]
                     if name.startswith("low_fidelity_model.")}
    if (
        {name for name in lf_parameters if state["parameter_requires_grad"][name]}
        != {"low_fidelity_model.branch_projection.weight", "low_fidelity_model.branch_projection.bias",
            "low_fidelity_model.trunk_projection.weight", "low_fidelity_model.trunk_projection.bias"}
        or len(state["optimizer_state"]["param_groups"]) != 2
    ):
        raise ValueError("任-06 LF必须从任04HF已参与更新的真实末投影来源复制，而非原始LF B0")
    validate_hf_checkpoint_provenance(architecture)
    splits = build_power_splits()
    assert_no_hf_leakage(
        {"顶部HF": splits.hf_train, "环温HF": splits.hf_train},
        splits.hf_validation | splits.hf_test | splits.external_sensor_test,
    )
    for arm in ("E1", "E2"):
        if tuple(config["E2仅LF原始训练曲线字典"]["tau秒"]) == E1_TAU:
            raise ValueError("任-06 E2字典不能伪装为固定E1")
        _registered_tau(config, arm)
    return state, architecture, config


def task06_model_kwargs(architecture: dict[str, Any], config: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("任-06运行臂必须为E0、E1或E2")
    kwargs = copy.deepcopy(architecture["correction_model_kwargs"])
    if "response_tau_seconds" in kwargs:
        raise ValueError("任-06共同来源原校正器不得事先含时间响应特征")
    tau = _registered_tau(config, arm)
    if tau is not None:
        kwargs["response_tau_seconds"] = list(tau)
    return kwargs


def validate_response_contract(
    model_state: dict[str, Tensor], kwargs: dict[str, Any], config: dict[str, Any], arm: str,
) -> None:
    tau = _registered_tau(config, arm)
    key = "response_features.tau_seconds"
    actual = model_state.get(key)
    first = model_state.get("correction.0.weight")
    old_input = tau is None
    if (
        first is None or tuple(first.shape) != (128, 6 if old_input else 10)
        or (key in model_state) != (not old_input)
        or ("response_tau_seconds" in kwargs) != (not old_input)
        or any(name.startswith("response_features.") and name != key for name in model_state)
    ):
        raise ValueError("任-06 E0须严格旧6列且无buffer；E1/E2须10列并具有唯一真tau buffer")
    if tau is not None and (
        actual is None or actual.dtype != torch.float64
        or not torch.equal(actual.cpu(), torch.tensor(tau, dtype=torch.float64))
        or tuple(kwargs["response_tau_seconds"]) != tau
    ):
        raise ValueError("任-06响应tau的真实buffer、预登记源和可重载kwargs不一致")


def _make_model(
    architecture: dict[str, Any], kwargs: dict[str, Any], device: torch.device,
) -> AdditiveCorrectionModel:
    scales = ModelScales(**architecture["scales"])
    low = build_model(
        architecture["low_fidelity_method"], scales,
        **architecture.get("low_fidelity_model_kwargs", {}),
    )
    if not isinstance(low, DeepONetPINN):
        raise ValueError("任-06只能使用同一实际DeepONet LF")
    guide_spec = architecture.get("surface_residual_guide_spec")
    guide = None if guide_spec is None else ChebyshevSurfaceResidualGuide.from_spec(guide_spec)
    return AdditiveCorrectionModel(
        low, scales, freeze_low_fidelity=True, surface_residual_guide=guide,
        **kwargs,
    ).to(device)


def fork_task06_model(
    source: dict[str, Any], architecture: dict[str, Any], config: dict[str, Any],
    arm: str, device: torch.device,
) -> tuple[AdditiveCorrectionModel, torch.optim.AdamW]:
    kwargs = task06_model_kwargs(architecture, config, arm)
    model = _make_model(architecture, kwargs, device)
    copied = {name: value.detach().clone() for name, value in source["model_state"].items()}
    if arm != "E0":
        first = copied["correction.0.weight"]
        copied["correction.0.weight"] = torch.cat((first, torch.zeros(128, 4, dtype=first.dtype)), dim=1)
        copied["response_features.tau_seconds"] = torch.tensor(
            _registered_tau(config, arm), dtype=torch.float64,
        )
    validate_response_contract(copied, kwargs, config, arm)
    model.load_state_dict(copied, strict=True)
    model.freeze_low_fidelity(True)
    if (
        any(parameter.requires_grad for parameter in model.low_fidelity_model.parameters())
        or any(not parameter.requires_grad for parameter in model.correction.parameters())
    ):
        raise ValueError("任-06三个HF臂均必须完全冻结真实LF，仅重新训练HF校正器")
    optimizer = torch.optim.AdamW(
        model.correction.parameters(), lr=1e-4,
        weight_decay=float(source["optimizer_state"]["param_groups"][0]["weight_decay"]),
    )
    if optimizer.state_dict()["state"]:
        raise ValueError("任-06不得把任04旧6列首层AdamW动量伪装成新10列优化器")
    return model, optimizer


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _deep_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return isinstance(left, Tensor) and isinstance(right, Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys()
                and all(_deep_equal(left[key], right[key]) for key in left))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_deep_equal(a, b) for a, b in zip(left, right)))
    return left == right


def _stage_metadata(
    config: dict[str, Any], arm: str, lf_sha: str, initial_score: float,
    best_score: float, best_epoch: int, physical_score: float, physical_epoch: int,
    consumption: dict[str, int], *, diagnostic_only: bool,
) -> dict[str, Any]:
    return {
        "任06运行臂": arm, "任06预登记配置SHA256": CONFIG_SHA,
        "源任04第120轮完整状态SHA256": config["共同完整训练状态_SHA256"],
        "源任04模型视图SHA256": config["共同观测最佳模型视图_SHA256"],
        "LF共同真实张量SHA256": lf_sha,
        "LF真实张量SHA方法": "按LF状态键排序，每键追加name|dtype|shape|与CPU contiguous原始bytes的SHA256",
        "LF全部参数冻结": True,
        "HF新优化器来源": "从任04第120轮完整模型权重复制；三臂均新建相同AdamW、HF首层旧动量不继承",
        "四类随机源": "从同一个任04完整状态分别恢复Python/NumPy/Torch CPU/CUDA；各臂相同专用shuffle种子",
        "HF学习率": 1e-4, "LF学习率": 0.0,
        "运行资格": (
            "短诊断；不参与任-06正式600轮采用" if diagnostic_only else
            "单种子正式同源600轮先导候选；须三臂审计后再决定采用"
        ),
        "响应tau来源": (
            "原6通道不添加响应" if arm == "E0" else
            "预登记固定E1" if arm == "E1" else "仅60 LF TRAIN字典登记22b/原始诊断7bb及冻结节点705"
        ),
        "响应tau秒": None if arm == "E0" else list(_registered_tau(config, arm)),
        "初始合法HF选分_摄氏度": initial_score,
        "观测最佳选分_摄氏度": best_score, "观测最佳任06轮次": best_epoch,
        "物理最佳独立损失": physical_score, "物理最佳任06轮次": physical_epoch,
        "累计实际消耗": consumption.copy(), "旧test_Data温度标签读取": False,
        "LF仿真温度进入HF监督": False,
    }


def _check_live_model(
    model: AdditiveCorrectionModel, kwargs: dict[str, Any], config: dict[str, Any],
    arm: str, lf_sha: str,
) -> None:
    validate_response_contract(model.state_dict(), kwargs, config, arm)
    if (
        _actual_lf_tensor_sha256(model.state_dict()) != lf_sha
        or any(parameter.requires_grad for parameter in model.low_fidelity_model.parameters())
        or any(not parameter.requires_grad for parameter in model.correction.parameters())
    ):
        raise ValueError("任-06保存前真实LF权重改变或HF/LF参数冻结列表异常")


def _save_state(
    path: Path, model: AdditiveCorrectionModel, optimizer: torch.optim.AdamW,
    *, epoch: int, metadata: dict[str, Any], kwargs: dict[str, Any],
    config: dict[str, Any], arm: str, lf_sha: str,
) -> None:
    _check_live_model(model, kwargs, config, arm, lf_sha)
    save_training_state(
        path, model, optimizer, stage="correction_time_features", epoch=epoch,
        budget=STAGE_BUDGET, metadata=metadata,
    )


def _model_view(
    path: Path, architecture: dict[str, Any], model: AdditiveCorrectionModel,
    kwargs: dict[str, Any], config: dict[str, Any], arm: str, lf_sha: str,
    *, epoch: int, score: float, validation: dict[str, float], diagnostic_only: bool,
) -> None:
    _check_live_model(model, kwargs, config, arm, lf_sha)
    view = copy.deepcopy(architecture)
    view["correction_model_kwargs"] = copy.deepcopy(kwargs)
    view["model_state"] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    validate_response_contract(view["model_state"], view["correction_model_kwargs"], config, arm)
    view["epoch"] = 420 + epoch
    view["validation_rmse_c"] = float(validation["顶部"])
    view["validation_selection_score_c"] = float(score)
    view["validation_sensor"] = {name: float(value) for name, value in validation.items() if name != "顶部"}
    view["任06新HF模型视图来源"] = {
        "运行臂": arm, "任06本臂实际轮次": epoch,
        "源第120轮完整状态SHA256": config["共同完整训练状态_SHA256"],
        "任06预登记配置SHA256": CONFIG_SHA, "LF共同真实张量SHA256": lf_sha,
        "响应tau秒": None if arm == "E0" else list(_registered_tau(config, arm)),
        "响应tau来源": (
            "无响应旧6列" if arm == "E0" else "固定E1预登记tau" if arm == "E1"
            else "仅LF训练登记22b、原诊断7bb和冻结节点705的四tau，不加载LF验证幅值"
        ),
        "运行资格": (
            "短诊断；不参与任-06正式600轮采用" if diagnostic_only else
            "单种子正式同源600轮先导候选；须三臂审计后再决定采用"
        ),
        "HF训练功率": sorted(build_power_splits().hf_train),
        "HF仅合法验证功率": sorted(build_power_splits().hf_validation),
        "LF仿真训练功率仅作来源": sorted(build_power_splits().simulation_train),
        "LF仿真温度进入HF监督": False, "旧test_Data温度标签读取": False,
        "任06本轮源码SHA256": {
            filename: sha256_file(PROJECT_ROOT / filename) for filename in (
                "src/sic_cu/train/task06_time_features.py", "scripts/23_run_task06_time_features.py",
            )
        },
        "旧任04provenance解释": "继承字段仅表示旧源数据协议；本字典与新真实模型张量登记任06训练",
    }
    view["续跑资格"] = "模型视图无AdamW与四类RNG，不得作为续跑状态"
    temporary = path.with_name(path.name + ".tmp")
    torch.save(view, temporary)
    temporary.replace(path)


def load_task06_model_view(
    path: str | Path, *, arm: str, device: torch.device = torch.device("cpu"),
) -> tuple[dict[str, Any], AdditiveCorrectionModel]:
    source, architecture, config = validate_task06_source()
    view = torch.load(_resolve(path), map_location="cpu", weights_only=False)
    ancestry = view.get("任06新HF模型视图来源", {})
    kwargs = task06_model_kwargs(architecture, config, arm)
    validate_response_contract(view["model_state"], view["correction_model_kwargs"], config, arm)
    if (
        view["correction_model_kwargs"] != kwargs
        or ancestry.get("运行臂") != arm
        or ancestry.get("源第120轮完整状态SHA256") != config["共同完整训练状态_SHA256"]
        or ancestry.get("任06预登记配置SHA256") != CONFIG_SHA
        or ancestry.get("LF共同真实张量SHA256") != _actual_lf_tensor_sha256(source["model_state"])
        or ancestry.get("响应tau秒") != (None if arm == "E0" else list(_registered_tau(config, arm)))
        or not 0 <= int(ancestry.get("任06本臂实际轮次", -1)) <= 600
        or view.get("epoch") != 420 + ancestry["任06本臂实际轮次"]
    ):
        raise ValueError("任-06模型视图来源、可重载kwargs或真实tau buffer与运行臂不一致")
    validate_hf_checkpoint_provenance(view)
    model = _make_model(view, kwargs, device)
    model.load_state_dict(view["model_state"], strict=True)
    model.freeze_low_fidelity(True)
    _check_live_model(model, kwargs, config, arm, ancestry["LF共同真实张量SHA256"])
    return view, model


def _verify_best_snapshot(
    output: Path, name: str, expected_epoch: int, latest: dict[str, Any],
    architecture: dict[str, Any], kwargs: dict[str, Any], config: dict[str, Any], arm: str,
    lf_sha: str, validation_loader: DataLoader, validation_sensor: tuple[Tensor, ...],
    weights: dict[str, float], physics: PhysicsLossComputer, materials: Any,
    boundaries: Any, device: torch.device,
) -> tuple[dict[str, Any], AdditiveCorrectionModel, dict[str, float] | None]:
    file = output / name
    metadata = latest["metadata"]
    old_sha = metadata.get(f"已提交旧{name}SHA256")
    if expected_epoch < latest["epoch"] and (not file.is_file() or not old_sha):
        raise ValueError("任-06历史最佳完整状态缺失，不得伪造旧AdamW动量或历史最佳")
    snapshot = torch.load(file, map_location="cpu", weights_only=False) if file.is_file() else None
    if snapshot is not None and snapshot.get("epoch", -1) > latest["epoch"]:
        raise ValueError("任-06最佳完整状态超前于最近已提交轮次，拒绝未提交模型")
    if snapshot is not None and expected_epoch < latest["epoch"] and sha256_file(file) != old_sha:
        raise ValueError("任-06过去的真实最佳完整状态SHA改变，不得伪造优化器")
    if snapshot is None or snapshot.get("epoch") < expected_epoch:
        if expected_epoch != latest["epoch"]:
            raise ValueError("任-06历史最佳完整状态已丢失，最近状态无法无损重建")
        snapshot = latest
    selected, selected_epoch = (
        ("观测最佳选分_摄氏度", "观测最佳任06轮次") if name == "阶段_观测最佳.pt" else
        ("物理最佳独立损失", "物理最佳任06轮次")
    )
    stage_meta = snapshot.get("metadata", {})
    if (
        snapshot.get("training_state_schema_version") != 1
        or snapshot.get("stage") != "correction_time_features"
        or snapshot.get("epoch") != expected_epoch
        or snapshot.get("budget") != STAGE_BUDGET
        or stage_meta.get(selected_epoch) != expected_epoch
        or stage_meta.get(selected) != metadata[selected]
        or stage_meta.get("任06运行臂") != arm
        or stage_meta.get("任06预登记配置SHA256") != CONFIG_SHA
        or stage_meta.get("源任04第120轮完整状态SHA256") != config["共同完整训练状态_SHA256"]
        or stage_meta.get("LF共同真实张量SHA256") != lf_sha
        or stage_meta.get("运行资格") != metadata["运行资格"]
        or stage_meta.get("响应tau秒") != metadata["响应tau秒"]
        or not snapshot.get("optimizer_state", {}).get("param_groups")
        or set(snapshot.get("random_state", {})) != {"python", "numpy", "torch_cpu", "torch_cuda"}
        or not snapshot.get("model_state")
    ):
        raise ValueError("任-06最佳完整状态来源、预登记tau、优化器或真实轮次不一致")
    validate_response_contract(snapshot["model_state"], kwargs, config, arm)
    if expected_epoch == latest["epoch"] and any(
        not _deep_equal(snapshot[key], latest[key])
        for key in ("model_state", "optimizer_state", "random_state", "parameter_requires_grad", "metadata")
    ):
        raise ValueError("任-06同轮最佳与最近状态模型、AdamW、RNG或元数据不一致")
    devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        model = _make_model(architecture, kwargs, device)
    model.load_state_dict(snapshot["model_state"], strict=True)
    model.freeze_low_fidelity(True)
    _check_live_model(model, kwargs, config, arm, lf_sha)
    if name == "阶段_观测最佳.pt":
        score, validation = _validation_selection(model, validation_loader,
                                                   validation_sensor, device, weights)
    else:
        validation = None
        score = float(evaluate_schedule_guardrails(
            model, physics, materials, boundaries, device,
        )["独立局部物理损失"]["physics_total"])
    if not math.isclose(score, float(metadata[selected]), abs_tol=1e-4):
        raise ValueError("任-06最佳真实模型与合法HF观测或独立物理评分不一致")
    return snapshot, model, validation


def _resume_best_and_budget(
    output: Path, latest: dict[str, Any], architecture: dict[str, Any],
    kwargs: dict[str, Any], config: dict[str, Any], arm: str, lf_sha: str,
    validation_loader: DataLoader, validation_sensor: tuple[Tensor, ...],
    weights: dict[str, float], physics: PhysicsLossComputer, materials: Any,
    boundaries: Any, device: torch.device, diagnostic_only: bool, sensor_count: int,
) -> None:
    epoch = latest["epoch"]
    meta = latest["metadata"]
    records = [json.loads(line) for line in (output / "training.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()]
    if len(records) != epoch or any(record.get("epoch") != index for index, record in enumerate(records, 1)):
        raise ValueError("任-06最近状态与连续训练日志轮次不一致")
    budget = {
        "HF训练观测点": sum(row["HF训练观测点"] for row in records),
        "HF训练传感器点": sensor_count * 15 * epoch,
        "LF仿真温度训练点": 0,
        "物理配点": 256 * epoch,
        "HF观测优化步": 15 * epoch,
        "物理优化步": epoch,
    }
    if meta.get("累计实际消耗") != budget or any(
        row["HF观测优化步"] != 15 or row["物理优化步"] != 1
        or row["物理配点"] != 256 or row["LF仿真温度训练点"] != 0
        or row["累计实际消耗"]["HF观测优化步"] != 15 * row["epoch"]
        for row in records
    ):
        raise ValueError("任-06连续完整状态与日志累计HF15＋物理1步预算不一致")
    best_epoch = int(meta["观测最佳任06轮次"])
    physical_epoch = int(meta["物理最佳任06轮次"])
    if not 0 <= best_epoch <= epoch or not 0 <= physical_epoch <= epoch:
        raise ValueError("任-06观测/物理最佳轮次超出真实连续阶段")
    # Preflight all earlier commitments before writing any recovered current sidecar.
    for name, selected in (("阶段_观测最佳.pt", best_epoch), ("阶段_物理最佳.pt", physical_epoch)):
        if selected < epoch and (
            not (output / name).is_file()
            or not meta.get(f"已提交旧{name}SHA256")
            or sha256_file(output / name) != meta[f"已提交旧{name}SHA256"]
        ):
            raise ValueError("任-06已提交历史最佳原件缺失或SHA异常，拒绝伪造历史AdamW")
    observed_snapshot = None
    verified = []
    for name, selected in (("阶段_观测最佳.pt", best_epoch), ("阶段_物理最佳.pt", physical_epoch)):
        file = output / name
        existed = torch.load(file, map_location="cpu", weights_only=False) if file.is_file() else None
        snapshot, best_model, validation = _verify_best_snapshot(
            output, name, selected, latest, architecture, kwargs, config, arm,
            lf_sha, validation_loader, validation_sensor, weights,
            physics, materials, boundaries, device,
        )
        verified.append((file, selected, existed))
        if validation is not None:
            observed_snapshot = (snapshot, best_model, validation)
    if observed_snapshot is None:
        raise ValueError("任-06观测最佳完整状态不能用于合法模型视图")
    best_state, best_model, validation = observed_snapshot
    view_file = output / "best.pt"
    view = torch.load(view_file, map_location="cpu", weights_only=False) if view_file.is_file() else None
    if view is not None and view.get("epoch", -1) > 420 + epoch:
        raise ValueError("任-06best.pt超前于最近提交阶段，不允许未提交最佳")
    if view is not None and view.get("epoch") == 420 + best_epoch:
        ancestry = view.get("任06新HF模型视图来源", {})
        if (
            ancestry.get("运行臂") != arm
            or ancestry.get("任06本臂实际轮次") != best_epoch
            or ancestry.get("源第120轮完整状态SHA256") != config["共同完整训练状态_SHA256"]
            or ancestry.get("任06预登记配置SHA256") != CONFIG_SHA
            or ancestry.get("LF共同真实张量SHA256") != lf_sha
            or ancestry.get("响应tau秒") != (None if arm == "E0" else list(_registered_tau(config, arm)))
            or ancestry.get("运行资格") != meta["运行资格"]
            or view.get("correction_model_kwargs") != kwargs
            or view.get("validation_selection_score_c") != meta["观测最佳选分_摄氏度"]
            or not math.isclose(view.get("validation_rmse_c", math.inf),
                                validation["顶部"], abs_tol=1e-4)
            or any(not math.isclose(view.get("validation_sensor", {}).get(key, math.inf),
                                    value, abs_tol=1e-4)
                   for key, value in validation.items() if key != "顶部")
            or not _deep_equal(view.get("model_state"), best_state["model_state"])
        ):
            raise ValueError("任-06 best.pt与合法HF验证、观测最佳真实张量或来源不一致")
        validate_response_contract(view["model_state"], view["correction_model_kwargs"], config, arm)
    for file, selected, existed in verified:
        if selected == epoch and (existed is None or existed.get("epoch") != epoch):
            temporary = file.with_name(file.name + ".tmp")
            torch.save(latest, temporary)
            temporary.replace(file)
    if view is None or view.get("epoch") != 420 + best_epoch:
        _model_view(view_file, architecture, best_model, kwargs, config, arm, lf_sha,
                    epoch=best_epoch, score=meta["观测最佳选分_摄氏度"],
                    validation=validation, diagnostic_only=diagnostic_only)


def run_task06_time_features(
    *, arm: str, output_directory: str | Path, budget_epochs: int = 600,
    session_epoch_limit: int | None = None, diagnostic_only: bool = False,
    resume_training_checkpoint: str | Path | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    if arm not in ARMS or budget_epochs != 600 or (
        session_epoch_limit is not None and not 1 <= session_epoch_limit <= 600
    ):
        raise ValueError("任-06 E0/E1/E2各臂只能同一个正式600轮原预算")
    if session_epoch_limit is not None and session_epoch_limit < 600 and not diagnostic_only:
        raise ValueError("任-06未完成600轮的短会话必须显式标记诊断，不能作为正式候选")
    if resume_training_checkpoint is not None and session_epoch_limit is None:
        raise ValueError("任-06续跑必须明确本次会话轮数与原600轮预算")
    source, architecture, config = validate_task06_source()
    splits = build_power_splits()
    training = load_yaml("configs/training.yaml")
    losses = training["loss_weights"]
    weights = training["multifidelity_selection_weights"]
    if (
        [losses["sensor_absolute"], losses["sensor_delta"]] != [5.0, 1.0]
        or [losses[name] for name in ("ir", "pde", "boundary", "initial", "interface")]
        != [1.0, 1.0, 1.0, 1.0, 1.0]
        or sorted(splits.simulation_train) != sorted(build_power_splits().simulation_train)
        or len(splits.simulation_train) != 60
        or len(splits.simulation_validation) != 10
    ):
        raise ValueError("任-06不能改变任02名义物理/HF环温5比1与LF60训练血缘")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    kwargs = task06_model_kwargs(architecture, config, arm)
    model, optimizer = fork_task06_model(source, architecture, config, arm, device)
    lf_sha = _actual_lf_tensor_sha256(source["model_state"])
    _check_live_model(model, kwargs, config, arm, lf_sha)
    output = _resolve(output_directory)
    if resume_training_checkpoint is None and output.exists():
        raise FileExistsError(f"任-06已有训练输出，禁止覆盖：{output}")
    if resume_training_checkpoint is not None and not output.is_dir():
        raise FileNotFoundError(f"任-06原臂续跑目录不存在：{output}")
    materials = load_materials()
    boundaries = load_resolved_boundary_conditions()
    if architecture["resolved_physics"] != asdict(boundaries):
        raise ValueError("任-06第120轮源模型名义物理解析参数与当前设置不一致")
    physics = PhysicsLossComputer(materials, boundaries, PhysicsLossWeights(
        pde=float(losses["pde"]), boundary=float(losses["boundary"]),
        initial=float(losses["initial"]), interface=float(losses["interface"]),
    ))
    train_data = _ir_dataset("train", splits.hf_train)
    validation_data = _ir_dataset("validation", splits.hf_validation)
    validation_loader = DataLoader(validation_data, batch_size=2048, shuffle=False)
    sensor = _sensor_tensors(device, split="train", powers_w=splits.hf_train)
    validation_sensor = _sensor_tensors(device, split="validation", powers_w=splits.hf_validation)
    if not 14 * 2048 < len(train_data) <= 15 * 2048:
        raise RuntimeError("任-06 HF真实训练资料必须每轮暴露15个2048 batch")
    consumption = {
        "HF训练观测点": 0, "HF训练传感器点": 0, "LF仿真温度训练点": 0,
        "物理配点": 0, "HF观测优化步": 0, "物理优化步": 0,
    }
    resumed_epoch = 0
    started = time.perf_counter()
    if resume_training_checkpoint is None:
        output.mkdir(parents=True)
        write_config_snapshot(output)
        (output / "training.jsonl").write_text("", encoding="utf-8")
        _restore_rng(source["random_state"])
        score, validation = _validation_selection(
            model, validation_loader, validation_sensor, device, weights,
        )
        initial_score = best_score = score
        best_epoch = physical_epoch = 0
        physical_score = float(evaluate_schedule_guardrails(
            model, physics, materials, boundaries, device,
        )["独立局部物理损失"]["physics_total"])
        initial_lf = _lf_material_validation(model, sorted(splits.simulation_validation), device)
        initial_hf = _hf_modalities(model, device, sorted(splits.hf_validation))
        metadata = _stage_metadata(config, arm, lf_sha, initial_score, best_score,
                                   best_epoch, physical_score, physical_epoch,
                                   consumption, diagnostic_only=diagnostic_only)
        for name in ("阶段_初始.pt", "阶段_观测最佳.pt", "阶段_物理最佳.pt"):
            _save_state(output / name, model, optimizer, epoch=0, metadata=metadata,
                        kwargs=kwargs, config=config, arm=arm, lf_sha=lf_sha)
        _model_view(output / "best.pt", architecture, model, kwargs, config, arm, lf_sha,
                    epoch=0, score=score, validation=validation, diagnostic_only=diagnostic_only)
    else:
        resume = _resolve(resume_training_checkpoint).resolve()
        if resume.name not in ("阶段_初始.pt", "阶段_最近.pt") or resume.parent != output.resolve():
            raise ValueError("任-06只可从本臂阶段_初始.pt/阶段_最近.pt重载真实AdamW及四RNG")
        snapshot_hashes = json.loads((output / "config_snapshot/sha256.json").read_text(
            encoding="utf-8",
        ))
        if any(sha256_file(PROJECT_ROOT / filename) != snapshot_hashes[filename] for filename in CONFIG_FILES):
            raise ValueError("任-06本臂有效源configs字节快照漂移，不得续跑")
        candidate = torch.load(resume, map_location="cpu", weights_only=False)
        validate_response_contract(candidate["model_state"], kwargs, config, arm)
        payload = load_training_state(resume, model, optimizer)
        metadata = payload["metadata"]
        qualification = (
            "短诊断；不参与任-06正式600轮采用" if diagnostic_only else
            "单种子正式同源600轮先导候选；须三臂审计后再决定采用"
        )
        if (
            payload.get("stage") != "correction_time_features"
            or payload.get("budget") != STAGE_BUDGET
            or metadata.get("任06运行臂") != arm
            or metadata.get("源任04第120轮完整状态SHA256") != config["共同完整训练状态_SHA256"]
            or metadata.get("源任04模型视图SHA256") != config["共同观测最佳模型视图_SHA256"]
            or metadata.get("任06预登记配置SHA256") != CONFIG_SHA
            or metadata.get("LF共同真实张量SHA256") != lf_sha
            or metadata.get("运行资格") != qualification
            or metadata.get("响应tau秒") != (None if arm == "E0" else list(_registered_tau(config, arm)))
        ):
            raise ValueError("任-06恢复状态、实际LF/tau来源或诊断与正式运行资格不一致")
        _check_live_model(model, kwargs, config, arm, lf_sha)
        resumed_epoch = int(payload["epoch"])
        if resumed_epoch >= 600:
            raise ValueError("任-06已经训练原600轮，不得超额续跑或诊断污染")
        reconcile_correction_resume_log(output, epoch=resumed_epoch, state_file=resume)
        _resume_best_and_budget(
            output, payload, architecture, kwargs, config, arm, lf_sha,
            validation_loader, validation_sensor, weights, physics, materials,
            boundaries, device, diagnostic_only, len(sensor[0]),
        )
        _restore_rng(payload["random_state"])
        initial_score = float(metadata["初始合法HF选分_摄氏度"])
        best_score = float(metadata["观测最佳选分_摄氏度"])
        best_epoch = int(metadata["观测最佳任06轮次"])
        physical_score = float(metadata["物理最佳独立损失"])
        physical_epoch = int(metadata["物理最佳任06轮次"])
        consumption = metadata["累计实际消耗"]
        initial_lf = _lf_material_validation(model, sorted(splits.simulation_validation), device)
        initial_hf = _hf_modalities(model, device, sorted(splits.hf_validation))
    session_end = min(600, resumed_epoch + (session_epoch_limit or 600))
    last_record: dict[str, Any] = {}
    for epoch in range(resumed_epoch + 1, session_end + 1):
        began = time.perf_counter()
        hf_loader = DataLoader(
            train_data, batch_size=2048, shuffle=True,
            generator=torch.Generator().manual_seed(601_000 + epoch),
        )
        if len(hf_loader) != 15:
            raise RuntimeError("任-06必须每轮真实消费15批HF训练观测")
        model.train()
        ir_losses, sensor_losses = [], []
        exposed = 0
        for coordinates, target, weight in hf_loader:
            coordinates, target, weight = (value.to(device) for value in (coordinates, target, weight))
            optimizer.zero_grad(set_to_none=True)
            prediction = model(coordinates, fidelity="high")
            ir_loss = (weight * ((prediction - target) / model.scales.temperature_scale_k).square()).sum() / weight.sum()
            sensor_x, sensor_target, sensor_delta, baseline = sensor
            sensor_prediction = model(sensor_x, fidelity="high")
            absolute, delta = _macro_sensor_training_losses(
                sensor_prediction, sensor_target, sensor_delta, baseline,
                sensor_x, model.scales.temperature_scale_k,
            )
            sensor_loss = float(losses["sensor_absolute"]) * absolute + float(losses["sensor_delta"]) * delta
            (float(losses["ir"]) * ir_loss + sensor_loss).backward()
            optimizer.step()
            ir_losses.append(float(ir_loss.detach()))
            sensor_losses.append(float(sensor_loss.detach()))
            exposed += len(coordinates)
        components = physics_optimizer_step(
            model, optimizer, physics, sample_collocation(256, device, seed=720 + epoch),
        )
        _check_live_model(model, kwargs, config, arm, lf_sha)
        consumption["HF训练观测点"] += exposed
        consumption["HF训练传感器点"] += len(sensor[0]) * 15
        consumption["物理配点"] += 256
        consumption["HF观测优化步"] += 15
        consumption["物理优化步"] += 1
        due = epoch % 10 == 0 or epoch == 600 or (diagnostic_only and epoch == session_end)
        stage_score, hf_modalities, lf_material, guardrails = None, None, None, None
        validation = None
        if due:
            stage_score, validation = _validation_selection(
                model, validation_loader, validation_sensor, device, weights,
            )
            hf_modalities = _hf_modalities(model, device, sorted(splits.hf_validation))
            lf_material = _lf_material_validation(model, sorted(splits.simulation_validation), device)
            guardrails = evaluate_schedule_guardrails(model, physics, materials, boundaries, device)
            if stage_score < best_score - 1e-4:
                best_score, best_epoch = stage_score, epoch
            now_physical = float(guardrails["独立局部物理损失"]["physics_total"])
            if now_physical < physical_score:
                physical_score, physical_epoch = now_physical, epoch
        last_record = {
            "epoch": epoch, "任06轮次": epoch, "运行臂": arm,
            "源任04完整实际阶段轮次": 120,
            "HF观测优化步": 15, "物理优化步": 1, "物理配点": 256,
            "HF训练观测点": exposed, "HF训练传感器点": len(sensor[0]) * 15,
            "LF仿真训练来源功率数": len(splits.simulation_train),
            "LF仿真温度训练点": 0, "LF实际冻结参数数": sum(
                not parameter.requires_grad for parameter in model.low_fidelity_model.parameters()
            ),
            "LF共同真实张量SHA256": lf_sha,
            "HF顶部训练损失": float(np.mean(ir_losses)),
            "HF环温训练损失": float(np.mean(sensor_losses)),
            "本轮名义物理训练分项": {name: float(value.detach()) for name, value in components.items()},
            "HF合法验证选分_摄氏度": stage_score,
            "HF合法验证分模态RMSE_摄氏度": hf_modalities,
            "LF合法验证材料RMSE_摄氏度": lf_material,
            "HF学习率": 1e-4, "LF学习率": 0.0,
            "本轮含评估墙钟秒": time.perf_counter() - began,
            "累计实际消耗": consumption.copy(),
        }
        if guardrails is not None:
            last_record.update(guardrails)
        with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(last_record, ensure_ascii=False) + "\n")
        metadata = _stage_metadata(
            config, arm, lf_sha, initial_score, best_score, best_epoch,
            physical_score, physical_epoch, consumption, diagnostic_only=diagnostic_only,
        )
        for name, selected in (("阶段_观测最佳.pt", best_epoch),
                               ("阶段_物理最佳.pt", physical_epoch)):
            if selected < epoch:
                previous = output / name
                if not previous.is_file():
                    raise ValueError("任-06历史最佳完整副本缺失，不得提交最近真实状态")
                metadata[f"已提交旧{name}SHA256"] = sha256_file(previous)
        _save_state(output / "阶段_最近.pt", model, optimizer, epoch=epoch, metadata=metadata,
                    kwargs=kwargs, config=config, arm=arm, lf_sha=lf_sha)
        if best_epoch == epoch:
            _save_state(output / "阶段_观测最佳.pt", model, optimizer, epoch=epoch,
                        metadata=metadata, kwargs=kwargs, config=config, arm=arm, lf_sha=lf_sha)
        if physical_epoch == epoch:
            _save_state(output / "阶段_物理最佳.pt", model, optimizer, epoch=epoch,
                        metadata=metadata, kwargs=kwargs, config=config, arm=arm, lf_sha=lf_sha)
        if best_epoch == epoch:
            _model_view(
                output / "best.pt", architecture, model, kwargs, config, arm, lf_sha,
                epoch=epoch, score=best_score, validation=validation,
                diagnostic_only=diagnostic_only,
            )
    if session_end == 600 and not diagnostic_only:
        for name in ("阶段_HF先导末.pt", "阶段_训练末.pt"):
            _save_state(output / name, model, optimizer, epoch=600, metadata=metadata,
                        kwargs=kwargs, config=config, arm=arm, lf_sha=lf_sha)
    status = (
        "真实完成单种子三臂原预算600轮；仍须独立能量审计和任07多种子确认"
        if session_end == 600 and not diagnostic_only else
        "短诊断累计600轮仍无正式采用资格，不生成正式阶段末状态"
        if session_end == 600 else
        "短诊断暂停，不得参与正式600轮采用" if diagnostic_only else
        "正式会话暂停，从本臂阶段_最近.pt续跑"
    )
    summary = {
        "状态": status, "运行臂": arm, "运行资格": metadata["运行资格"],
        "本臂实际完成轮次": session_end, "原预登记预算": 600,
        "任06预登记配置SHA256": CONFIG_SHA,
        "源任04第120轮完整状态SHA256": config["共同完整训练状态_SHA256"],
        "LF共同真实张量SHA256": lf_sha,
        "LF仿真训练温度进入HF监督": False,
        "LF合法验证逐材料节点及真实体积RMSE_摄氏度": initial_lf,
        "HF合法验证各模态RMSE_摄氏度": last_record.get("HF合法验证分模态RMSE_摄氏度") or initial_hf,
        "初始HF合法选分_摄氏度": initial_score,
        "观测最佳HF合法选分_摄氏度": best_score, "观测最佳任06轮次": best_epoch,
        "物理最佳独立损失": physical_score, "物理最佳任06轮次": physical_epoch,
        "真实最后状态": "阶段_HF先导末.pt" if session_end == 600 and not diagnostic_only
                     else "阶段_最近.pt",
        "旧test_Data温度标签读取": False,
        "名义物理资格": "独立审核前不得称内部物理可信；任05问题未解决",
        "累计实际消耗": consumption.copy(),
        "最后一轮门禁证据": last_record,
        "本会话实际训练秒": time.perf_counter() - started,
        "机制采用": "三臂各600轮及独立能量审计后按原规则判定；先导非任07五种子结论",
    }
    report = output / "阶段报告.json"
    temporary = report.with_name(report.name + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report)
    return summary
