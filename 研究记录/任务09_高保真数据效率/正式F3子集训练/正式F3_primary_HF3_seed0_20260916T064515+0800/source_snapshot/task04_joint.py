"""Task-04 bounded continuation from the registered Task-03 terminal correction state."""

from __future__ import annotations

import json
import copy
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import PowerSplits, build_power_splits, assert_no_hf_leakage
from sic_cu.data.fields import assert_compatible_fields, load_processed_field
from sic_cu.eval.energy_v5 import AxisymmetricGeometry, axisymmetric_lumped_nodal_weights
from sic_cu.eval.metrics import weighted_metrics
from sic_cu.eval.protocol_checks import validate_hf_checkpoint_provenance
from sic_cu.losses import PhysicsLossComputer, PhysicsLossWeights
from sic_cu.models import AdditiveCorrectionModel, ModelScales
from sic_cu.models.deeponet_pinn import DeepONetPINN
from sic_cu.models.residual_interpolation import ChebyshevSurfaceResidualGuide
from sic_cu.physics import sample_collocation
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.train.common import (
    CONFIG_FILES, load_training_state, physics_optimizer_step, save_training_state,
    write_config_snapshot,
)
from sic_cu.train.multifidelity import (
    _evaluate_ir_model, _evaluate_sensor_model, _ir_dataset, _macro_sensor_training_losses,
    _sensor_tensors, _sensor_validation, _weighted_validation, evaluate_schedule_guardrails,
    reconcile_correction_resume_log,
)
from sic_cu.train.simulation import build_model, load_sampled_points


TASK03_DECISION = PROJECT_ROOT / "研究记录/任务03_低保真精度修复/验收判定_预算门禁复核.json"
TASK04_CONFIG = PROJECT_ROOT / "研究记录/任务04_联合微调/有效运行配置.yaml"
PROJECTION_NAMES = frozenset({
    "low_fidelity_model.branch_projection.weight", "low_fidelity_model.branch_projection.bias",
    "low_fidelity_model.trunk_projection.weight", "low_fidelity_model.trunk_projection.bias",
})
ARMS = ("冻结", "有限解冻")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def validate_task04_source(
    checkpoint_path: str | Path, decision_path: str | Path = TASK03_DECISION,
    *, require_registration: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _resolve(checkpoint_path).resolve()
    registered = json.loads(_resolve(decision_path).read_text(encoding="utf-8"))
    origin = registered["任04共同阶段起点"]
    expected = (PROJECT_ROOT / origin["检查点"]).resolve()
    if source.name != "阶段_校正末.pt" or (
        require_registration and (source != expected or sha256_file(source) != origin["SHA256"])
    ):
        raise ValueError("任-04起点必须是验收判定登记的同一真实校正末状态")
    state = torch.load(source, map_location="cpu", weights_only=False)
    if (
        state.get("training_state_schema_version") != 1
        or state.get("stage") != "correction" or state.get("epoch") != 300
        or state.get("budget") != {"校正轮次": 300, "联合轮次": 0}
    ):
        raise ValueError("任-04源状态必须是完整300轮真实校正末，历史最佳不能伪造续跑状态")
    names = state.get("parameter_requires_grad", {})
    correction = [name for name in names if name.startswith("correction.")]
    if (
        not correction or not all(names[name] for name in correction)
        or any(names[name] for name in names if name.startswith("low_fidelity_model."))
    ):
        raise ValueError("任-04源状态的HF校正器与冻结LF参数列表不符合校正阶段")
    optimizer = state.get("optimizer_state", {})
    groups = optimizer.get("param_groups", [])
    if (
        len(groups) != 1 or len(groups[0].get("params", [])) != len(correction)
        or set(optimizer.get("state", {})) != set(groups[0]["params"])
        or groups[0].get("lr") != 1e-4
        or any(
            not {"step", "exp_avg", "exp_avg_sq"} <= set(momentum)
            or float(momentum["step"]) <= 0
            for momentum in optimizer["state"].values()
        )
    ):
        raise ValueError("任-04源HF AdamW优化器动量须完整、真实续用，不得重新创建历史最佳动量")
    rng = state.get("random_state", {})
    if set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"} or any(
        rng[key] is None for key in ("python", "numpy", "torch_cpu")
    ):
        raise ValueError("任-04源Python/NumPy/CPU/CUDA随机状态必须完整")
    architecture_file = source.parent / "best.pt"
    if require_registration:
        registered_best = registered["HF三臂"]["旧LF同源"]
        if (
            architecture_file.resolve() != (PROJECT_ROOT / registered_best["最佳模型检查点"]).resolve()
            or sha256_file(architecture_file) != registered_best["最佳模型SHA256"]
        ):
            raise ValueError("任-04源校正阶段best.pt架构元数据须独立匹配任-03机器登记SHA")
    architecture = torch.load(architecture_file, map_location="cpu", weights_only=False)
    # The ledger is append-only; only operational YAML semantics are inherited from Task-03.
    snapshot = source.parent / "config_snapshot"
    for name, label in (
        ("geometry", "几何"), ("materials", "材料"),
        ("boundary_conditions", "物理边界"), ("splits", "功率划分"),
        ("training", "训练预算"), ("data_metadata", "数据协议"),
    ):
        if load_yaml(f"configs/{name}.yaml") != load_yaml(str(snapshot / f"{name}.yaml")):
            raise ValueError(f"任-04源快照{label}配置语义与当前配置不一致；总计划事后追加不参与此比较")
    if (
        architecture.get("method") != "multifidelity_correction"
        or architecture.get("low_fidelity_method") != "deeponet_pinn"
        or set(architecture["model_state"]) != set(state["model_state"])
        or sorted(architecture.get("hf_train_powers_w", [])) != sorted(build_power_splits().hf_train)
        or sorted(architecture.get("hf_validation_powers_w", [])) != sorted(build_power_splits().hf_validation)
    ):
        raise ValueError("任-04源结构与合法HF校正器/12训练3验证协议不一致")
    validate_hf_checkpoint_provenance(architecture)
    return state, architecture


def validate_task04_splits(splits: PowerSplits) -> dict[str, list[float]]:
    canonical = build_power_splits()
    if (
        splits.hf_train != canonical.hf_train
        or splits.hf_validation != canonical.hf_validation
        or splits.hf_test != canonical.hf_test
        or splits.simulation_train != canonical.simulation_train
        or len(splits.simulation_train) != 60
    ):
        raise RuntimeError("任-04功率必须严格沿用固定HF训练/合法验证与60个LF仿真训练功率")
    assert_no_hf_leakage(
        {"IR": splits.hf_train, "Hot/Cold": splits.hf_train},
        splits.hf_validation | splits.hf_test | splits.external_sensor_test,
    )
    return {
        "HF训练功率": sorted(splits.hf_train),
        "HF合法验证功率": sorted(splits.hf_validation),
        "LF仿真回放功率": sorted(splits.simulation_train),
    }


def task04_lf_keep_guardrail(
    frozen: dict[str, dict[str, float]], joint: dict[str, dict[str, float]],
) -> dict[str, Any]:
    rates: dict[str, dict[str, float]] = {}
    if set(frozen) != {"Cu", "SiC"} or set(joint) != {"Cu", "SiC"}:
        raise ValueError("任-04 LF保持护栏必须独立比较Cu和SiC材料")
    for material in ("Cu", "SiC"):
        if set(frozen[material]) != {"node", "volume"} or set(joint[material]) != {"node", "volume"}:
            raise ValueError("任-04 LF保持护栏缺少材料节点或真实轴对称体积RMSE")
        rates[material] = {}
        for metric in ("node", "volume"):
            old, new = float(frozen[material][metric]), float(joint[material][metric])
            if not math.isfinite(old) or not math.isfinite(new) or old <= 0 or new < 0:
                raise ValueError("任-04 LF保持护栏需要有限非负且源值正的验证误差")
            rates[material][metric] = 100.0 * (new - old) / old
    return {
        "逐材料逐口径恶化率_百分比": rates,
        "LF两材料节点与真实体积均守住5%护栏": all(
            rate <= 5.0 for material in rates.values() for rate in material.values()
        ),
        "资格": "只作为两臂完成相同200轮后由HF及物理共同决策的LF保持护栏",
    }


def validate_task04_registration(
    config: dict[str, Any], decision: dict[str, Any], splits: PowerSplits,
) -> None:
    source = decision["任04共同阶段起点"]
    old_lf = decision["HF三臂"]["旧LF同源"]["LF检查点SHA256"]
    training = load_yaml("configs/training.yaml")
    if (
        config.get("任务") != "任-04" or config.get("种子") != 0
        or config.get("起点状态") != source["检查点"]
        or config.get("起点状态_SHA256") != source["SHA256"]
        or config.get("原LF_B0_SHA256") != old_lf
        or config.get("原HF部署B0_SHA256") != decision["历史部署B0参考"]["SHA256"]
        or config.get("先导后续轮次_每臂") != 200
        or config.get("每卡HF观测batch") != 2048
        or config.get("校正器学习率") != 1e-4
        or config.get("LF仅末投影学习率") != 1e-5
        or config.get("LF可解冻模块") != ["branch_projection", "trunk_projection"]
        or config.get("LF其余参数与输出bias") != "保持冻结"
        or config.get("物理整包配点_每轮") != 256
        or "15次HF数据" not in str(config.get("HF原S0排程", ""))
        or "另1次独立物理" not in str(config.get("HF原S0排程", ""))
        or config.get("HF训练功率数") != len(splits.hf_train) or len(splits.hf_train) != 12
        or sorted(config.get("HF合法验证功率_瓦", [])) != sorted(splits.hf_validation)
        or config.get("LF仿真回放训练功率数") != len(splits.simulation_train)
        or len(splits.simulation_train) != 60
        or config.get("LF仿真验证功率数") != len(splits.simulation_validation)
        or len(splits.simulation_validation) != 10
        or not str(config.get("LF仿真回放目标", "")).startswith('fidelity="low"')
        or config.get("HF传感器绝对与温升损失权重") != [5.0, 1.0]
        or [training["loss_weights"]["sensor_absolute"],
            training["loss_weights"]["sensor_delta"]] != [5.0, 1.0]
        or "每10轮" not in str(config.get("HF验证选择", ""))
        or "macro_v1" not in str(config.get("HF验证选择", ""))
        or config.get("早停耐心轮数") != 201
        or training["multifidelity"]["joint_simulation_samples_per_power"] != 2048
    ):
        raise ValueError("任-04预登记运行配置与源SHA、200轮预算、投影范围、HF/LF协议或优化排程不一致")


def validate_task04_resume_name(path: str | Path) -> None:
    if Path(path).name not in ("阶段_初始.pt", "阶段_最近.pt"):
        raise ValueError("任-04只允许从本臂阶段_初始.pt或阶段_最近.pt连续续跑；最佳/联合末不得当优化器来源")


def validate_task04_resume_qualification(metadata: dict[str, Any], *, diagnostic_only: bool) -> None:
    expected = (
        "短诊断；不参与任-04正式200轮采用" if diagnostic_only
        else "正式同源200轮候选；仍须两臂验收后决定采用"
    )
    if metadata.get("运行资格") != expected:
        raise ValueError("任-04短诊断不得转正式候选，正式中断亦不得被诊断状态续跑污染")


def _model_from_architecture(architecture: dict[str, Any], device: torch.device) -> nn.Module:
    scales = ModelScales(**architecture["scales"])
    low = build_model(
        architecture["low_fidelity_method"], scales,
        **architecture.get("low_fidelity_model_kwargs", {}),
    )
    if not isinstance(low, DeepONetPINN):
        raise ValueError("任-04限定DeepONet LF最后两个投影层")
    guide_spec = architecture.get("surface_residual_guide_spec")
    guide = None if guide_spec is None else ChebyshevSurfaceResidualGuide.from_spec(guide_spec)
    return AdditiveCorrectionModel(
        low, scales, freeze_low_fidelity=True, surface_residual_guide=guide,
        **architecture["correction_model_kwargs"],
    ).to(device)


def fork_task04_model(
    state: dict[str, Any], architecture: dict[str, Any], arm: str, device: torch.device,
    source_path: str | Path | None = None,
) -> tuple[nn.Module, torch.optim.AdamW]:
    if arm not in ARMS:
        raise ValueError("任-04只允许冻结或有限解冻两臂")
    model = _model_from_architecture(architecture, device)
    named = dict(model.named_parameters())
    if set(named) != set(state["parameter_requires_grad"]):
        raise ValueError("任-04源状态参数名称不匹配实际DeepONet+HF校正器")
    weight_decay = float(state["optimizer_state"]["param_groups"][0]["weight_decay"])
    optimizer = torch.optim.AdamW([
        {"params": list(model.correction.parameters()), "lr": 1e-4, "weight_decay": weight_decay}
    ])
    path = TASK03_DECISION if source_path is None else source_path
    if source_path is None:
        registered = json.loads(TASK03_DECISION.read_text(encoding="utf-8"))
        path = PROJECT_ROOT / registered["任04共同阶段起点"]["检查点"]
    restored = load_training_state(path, model, optimizer)
    if set(restored["optimizer_state"]["state"]) != set(state["optimizer_state"]["state"]):
        raise ValueError("任-04 HF优化器续用验证失败")
    source_ids = state["optimizer_state"]["param_groups"][0]["params"]
    for item, source_id in zip(optimizer.param_groups[0]["params"], source_ids):
        momentum = state["optimizer_state"]["state"][source_id]
        if any(not torch.equal(optimizer.state[item][key].cpu(), value) for key, value in momentum.items()):
            raise ValueError("任-04 HF优化器动量非同源")
    if arm == "有限解冻":
        for name in PROJECTION_NAMES:
            named[name].requires_grad_(True)
        optimizer.add_param_group({
            "params": [parameter for name, parameter in named.items() if name in PROJECTION_NAMES],
            "lr": 1e-5, "weight_decay": weight_decay,
        })
    actual = {name for name, parameter in named.items() if parameter.requires_grad}
    expected = {name for name in named if name.startswith("correction.")}
    if arm == "有限解冻":
        expected |= PROJECTION_NAMES
    if actual != expected:
        raise ValueError("任-04冻结名单异常：不可解冻全部LF或HF校正器之外的额外参数")
    return model, optimizer


def low_replay_loss(model: nn.Module, coordinates: Tensor, temperature_k: Tensor) -> Tensor:
    trainable = any(param.requires_grad for param in model.low_fidelity_model.parameters())
    with torch.enable_grad() if trainable else torch.no_grad():
        prediction = model(coordinates, fidelity="low")
        return ((prediction - temperature_k) / model.scales.temperature_scale_k).square().mean()


def _validation_selection(
    model: nn.Module, loader: DataLoader, sensor: tuple[Tensor, ...],
    device: torch.device, weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    ir_rmse = float(_weighted_validation(model, loader, device)[0])
    sensor_metrics = _sensor_validation(model, sensor)
    numerator = (
        float(weights["ir_rmse"]) * ir_rmse
        + float(weights["sensor_absolute_rmse"]) * sensor_metrics["absolute_rmse_c"]
        + float(weights["sensor_delta_rmse"]) * sensor_metrics["delta_rmse_c"]
    )
    denominator = sum(float(value) for value in weights.values())
    return numerator / denominator, {"顶部": ir_rmse, **sensor_metrics}


@torch.no_grad()
def _lf_material_validation(model: nn.Module, powers: list[float], device: torch.device) -> dict[str, dict[str, float]]:
    geometry = AxisymmetricGeometry.from_config(load_yaml("configs/geometry.yaml"))
    fields = [load_processed_field(power) for power in powers]
    try:
        assert_compatible_fields(fields)
    except ValueError as exc:
        raise ValueError("任-04 LF合法验证网格/材料/节点不兼容，不得复用首功率集总体积权重") from exc
    nodal_weights: np.ndarray | None = None
    records: dict[str, dict[str, list[float]]] = {
        material: {"node": [], "volume": []} for material in ("Cu", "SiC")
    }
    model.eval()
    for power, field in zip(powers, fields):
        if nodal_weights is None:
            nodal_weights = axisymmetric_lumped_nodal_weights(
                field.coordinates_rz_m, field.material_ids, geometry,
            )
        size = len(field.material_ids)
        coordinates = np.column_stack((
            np.tile(field.coordinates_rz_m, (len(field.times_s), 1)),
            np.repeat(field.times_s, size),
            np.full(len(field.times_s) * size, power, dtype=np.float32),
            np.tile(field.material_ids, len(field.times_s)),
        )).astype(np.float32)
        x = torch.from_numpy(coordinates)
        predictions = []
        for offset in range(0, len(x), 8192):
            predictions.append(model(x[offset:offset + 8192].to(device), fidelity="low").cpu().numpy())
        predicted = np.concatenate(predictions).reshape(-1)
        actual = field.temperature_k.reshape(-1)
        tiled_material = np.tile(field.material_ids, len(field.times_s))
        tiled_weight = np.tile(nodal_weights, len(field.times_s))
        for material, material_id in (("Cu", 0), ("SiC", 1)):
            chosen = tiled_material == material_id
            for metric, weights in (("node", np.ones(int(chosen.sum()))),
                                    ("volume", tiled_weight[chosen])):
                records[material][metric].append(weighted_metrics(
                    actual[chosen], predicted[chosen], weights,
                )["rmse_c"])
    return {
        material: {metric: float(np.mean(values)) for metric, values in per_material.items()}
        for material, per_material in records.items()
    }


def _hf_modalities(model: nn.Module, device: torch.device, powers: list[float]) -> dict[str, float]:
    ir = _evaluate_ir_model(model, "validation", device, powers)
    sensor = _evaluate_sensor_model(model, "validation", device, powers)
    return {
        "top": float(ir["rmse_c"]),
        **{
            name: float(np.mean([
                curve["absolute"]["rmse_c"] for curve in sensor["per_curve"]
                if curve["sensor_type"] == name
            ])) for name in ("hot", "cold")
        },
    }


def _real_model_view(
    path: Path, architecture: dict[str, Any], model: nn.Module,
    *, epoch: int, score: float, arm: str, source_sha: str, config_sha: str,
    top_rmse: float, sensor_validation: dict[str, float], diagnostic_only: bool = False,
) -> None:
    payload = copy.deepcopy(architecture)
    payload["epoch"] = 300 + epoch
    payload["model_state"] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    payload["validation_rmse_c"] = top_rmse
    payload["validation_selection_score_c"] = score
    payload["validation_sensor"] = sensor_validation
    payload["任04新模型视图来源"] = {
        "源校正末SHA256": source_sha,
        "任04预登记配置SHA256": config_sha,
        "原校正阶段旧best轮次": int(architecture["epoch"]),
        "任04本阶段实际轮次": epoch,
        "运行资格": (
            "短诊断；不参与任-04正式200轮采用" if diagnostic_only
            else "正式同源200轮候选；仍须两臂验收后决定采用"
        ),
        "本轮实际模型轮次": 300 + epoch,
        "运行臂及LF事实": (
            "冻结：只更新HF校正器，LF仅监测" if arm == "冻结"
            else "有限解冻：更新HF校正器及DeepONet两套最后投影层"
        ),
        "HF温度标签训练功率": sorted(build_power_splits().hf_train),
        "HF温度标签只作验证功率": sorted(build_power_splits().hf_validation),
        "LF仿真仅低保真监督功率": sorted(build_power_splits().simulation_train),
        "旧test_Data温度标签读取": False,
        "LF权重冻结名单": sorted(
            name for name, param in model.named_parameters()
            if name.startswith("low_fidelity_model.") and not param.requires_grad
        ),
        "LF权重更新名单": sorted(
            name for name, param in model.named_parameters()
            if name.startswith("low_fidelity_model.") and param.requires_grad
        ),
        "任04新源码SHA256": {
            filename: sha256_file(PROJECT_ROOT / filename) for filename in (
                "src/sic_cu/train/task04_joint.py", "scripts/19_run_task04_joint.py",
            )
        },
        "继承provenance解释": "历史字段只描述任03源数据协议；本字典登记任04新续训，旧best模型状态未继承",
    }
    payload["续跑资格"] = "模型视图不含优化器和随机状态，不得据此续跑"
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def task04_terminal_name(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError("任-04阶段末名必须对应明确冻结或有限解冻运行臂")
    return "阶段_HF续训冻结LF末.pt" if arm == "冻结" else "阶段_联合末.pt"


def _stage_metadata(
    origin: dict[str, Any], arm: str, initial_score: float, best: float,
    best_epoch: int, physical: float, physical_epoch: int,
    cumulative: dict[str, int], source_hf_step: float,
    source_parameter_names: dict[str, bool], config_sha: str, diagnostic_only: bool = False,
) -> dict[str, Any]:
    frozen_names = sorted(
        name for name in source_parameter_names
        if name.startswith("low_fidelity_model.")
        and (arm == "冻结" or name not in PROJECTION_NAMES)
    )
    return {
        "源校正末路径": origin["检查点"], "源校正末SHA256": origin["SHA256"],
        "任04预登记配置SHA256": config_sha,
        "源校正末实际轮次": 300, "源HF优化器步": source_hf_step,
        "两臂随机源说明": (
            "两臂完整恢复同一源Python/NumPy/Torch CPU/CUDA随机状态；"
            "本轮HF与LF shuffle使用两臂相同的新专用确定性生成器，"
            "不声称恢复任03原数据顺序或历史最佳随机序列"
        ),
        "任04运行臂": arm,
        "运行资格": (
            "短诊断；不参与任-04正式200轮采用" if diagnostic_only
            else "正式同源200轮候选；仍须两臂验收后决定采用"
        ),
        "阶段实际策略": (
            "HF校正器继续训练+LF全程冻结，未发生LF联合更新" if arm == "冻结"
            else "HF校正器继续训练+仅LF branch/trunk最后投影层更新"
        ),
        "LF冻结参数": frozen_names,
        "LF允许训练投影参数": [] if arm == "冻结" else sorted(PROJECTION_NAMES),
        "HF学习率": 1e-4, "LF学习率": 0.0 if arm == "冻结" else 1e-5,
        "初始验证选分_摄氏度": initial_score,
        "观测最佳选分_摄氏度": best, "观测最佳任04轮次": best_epoch,
        "物理最佳独立损失": physical, "物理最佳任04轮次": physical_epoch,
        "累计实际消耗": cumulative,
        "冻结臂LF仅计算监测且不反传": arm == "冻结",
    }


def _assert_resume_log_budget(output: Path, epoch: int, consumption: dict[str, int], sensor_size: int) -> None:
    records = [json.loads(line) for line in (output / "training.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()]
    if len(records) != epoch or any(record.get("epoch") != index for index, record in enumerate(records, 1)):
        raise ValueError("任-04最近检查点与已提交训练日志轮次不一致")
    expected = {
        "HF训练观测点": sum(record["HF训练观测点"] for record in records),
        "HF训练传感器点": sensor_size * 15 * epoch,
        "LF仿真训练点": sum(record["LF仿真回放训练点"] for record in records),
        "物理配点": sum(record["物理配点"] for record in records),
        "HF观测优化步": sum(record["HF观测优化步"] for record in records),
        "物理优化步": sum(record["物理优化步"] for record in records),
    }
    if consumption != expected or any(
        record["HF观测优化步"] != 15 or record["物理优化步"] != 1
        or record["LF仿真回放训练点"] != 60 * 2048
        or record["物理配点"] != 256
        for record in records
    ):
        raise ValueError("任-04已提交日志与完整优化器状态累计实际预算不一致")


def _atomic_complete_state_copy(source: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(source, temporary)
    temporary.replace(destination)


def _deep_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return isinstance(left, Tensor) and isinstance(right, Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict)
                and left.keys() == right.keys()
                and all(_deep_equal(left[key], right[key]) for key in left))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_deep_equal(a, b) for a, b in zip(left, right)))
    return left == right


def _assert_complete_best_snapshot(
    snapshot: dict[str, Any], payload: dict[str, Any], name: str, expected: int,
    origin: dict[str, Any], config_sha: str, arm: str,
) -> None:
    metadata = snapshot.get("metadata", {})
    committed = payload["metadata"]
    score_name, epoch_name = (
        ("观测最佳选分_摄氏度", "观测最佳任04轮次") if name == "阶段_观测最佳.pt" else
        ("物理最佳独立损失", "物理最佳任04轮次")
    )
    if (
        snapshot.get("training_state_schema_version") != 1 or snapshot.get("stage") != "joint"
        or snapshot.get("epoch") != expected or snapshot.get("budget") != payload["budget"]
        or not snapshot.get("model_state") or not snapshot.get("optimizer_state", {}).get("state")
        or set(snapshot["model_state"]) != set(payload["model_state"])
        or set(snapshot.get("parameter_requires_grad", {})) != set(payload["parameter_requires_grad"])
        or set(snapshot.get("random_state", {})) != {"python", "numpy", "torch_cpu", "torch_cuda"}
        or metadata.get("源校正末SHA256") != origin["SHA256"]
        or metadata.get("任04预登记配置SHA256") != config_sha
        or metadata.get("任04运行臂") != arm
        or metadata.get("运行资格") != committed["运行资格"]
        or metadata.get("LF冻结参数") != committed["LF冻结参数"]
        or metadata.get("LF允许训练投影参数") != committed["LF允许训练投影参数"]
        or metadata.get(epoch_name) != expected
        or metadata.get(score_name) != committed[score_name]
    ):
        raise ValueError("任-04历史最佳完整状态轮次、优化器、来源或原分数不一致")
    if expected == payload["epoch"] and any(
        not _deep_equal(snapshot[name], payload[name])
        for name in ("model_state", "optimizer_state", "random_state", "parameter_requires_grad", "metadata")
    ):
        raise ValueError("任-04同轮最佳与最近完整模型、优化器或随机状态不一致")


def _verify_earlier_best_score(
    snapshot: dict[str, Any], name: str, architecture: dict[str, Any],
    validation_loader: DataLoader, validation_sensor: tuple[Tensor, ...],
    weights: dict[str, float], physics: PhysicsLossComputer, materials: Any,
    boundaries: Any, device: torch.device, committed_score: float,
) -> tuple[dict[str, float] | None, nn.Module]:
    # Constructing a validation-only model must not advance the committed Torch random stream.
    devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        best_model = _model_from_architecture(architecture, device)
    best_model.load_state_dict(snapshot["model_state"])
    for parameter_name, parameter in best_model.named_parameters():
        parameter.requires_grad_(snapshot["parameter_requires_grad"][parameter_name])
    if name == "阶段_观测最佳.pt":
        measured, validation = _validation_selection(
            best_model, validation_loader, validation_sensor, device, weights,
        )
    else:
        validation = None
        measured = float(evaluate_schedule_guardrails(
            best_model, physics, materials, boundaries, device,
        )["独立局部物理损失"]["physics_total"])
    if not math.isclose(measured, committed_score, abs_tol=1e-4):
        raise ValueError("任-04历史最佳真实张量与合法观测验证或物理最佳分数不一致")
    return validation, best_model


def _reconcile_committed_best(
    output: Path, payload: dict[str, Any], architecture: dict[str, Any], model: nn.Module,
    validation_loader: DataLoader, validation_sensor: tuple[Tensor, ...],
    weights: dict[str, float], physics: PhysicsLossComputer, materials: Any,
    boundaries: Any, device: torch.device, arm: str, origin: dict[str, Any],
    config_sha: str, diagnostic_only: bool,
) -> None:
    metadata = payload["metadata"]
    epoch = int(payload["epoch"])
    observed_epoch = int(metadata["观测最佳任04轮次"])
    physical_epoch = int(metadata["物理最佳任04轮次"])
    if not 0 <= observed_epoch <= epoch or not 0 <= physical_epoch <= epoch:
        raise ValueError("任-04最佳轮次超出真实最近检查点，拒绝伪造历史最佳")
    earlier_observed: tuple[dict[str, Any], dict[str, float], nn.Module] | None = None
    for name, expected in (("阶段_观测最佳.pt", observed_epoch),
                           ("阶段_物理最佳.pt", physical_epoch)):
        sidecar = output / name
        state = torch.load(sidecar, map_location="cpu", weights_only=False) if sidecar.exists() else None
        if state is not None and int(state.get("epoch", -1)) > epoch:
            raise ValueError("任-04最佳副本超前于完整最近状态，不得恢复未提交模型")
        if expected != epoch:
            commitment = metadata.get(f"已提交旧{name}SHA256")
            if state is None or state.get("epoch") != expected or not commitment:
                raise ValueError("任-04历史最佳完整状态缺失且无法从最近状态无损恢复")
            if sha256_file(sidecar) != commitment:
                raise ValueError("任-04历史最佳完整状态SHA不匹配，不得伪造历史优化器")
            _assert_complete_best_snapshot(state, payload, name, expected, origin, config_sha, arm)
            selected = ("观测最佳选分_摄氏度" if name == "阶段_观测最佳.pt" else
                        "物理最佳独立损失")
            validation, verified_model = _verify_earlier_best_score(
                state, name, architecture, validation_loader, validation_sensor,
                weights, physics, materials, boundaries, device, float(metadata[selected]),
            )
            if validation is not None:
                earlier_observed = (state, validation, verified_model)
            continue
        if name == "阶段_观测最佳.pt":
            measured, _ = _validation_selection(model, validation_loader, validation_sensor, device, weights)
            if not math.isclose(measured, float(metadata["观测最佳选分_摄氏度"]), abs_tol=1e-4):
                raise ValueError("任-04最近状态观测最佳分数与合法HF验证温度不一致")
        else:
            measured = float(evaluate_schedule_guardrails(
                model, physics, materials, boundaries, device,
            )["独立局部物理损失"]["physics_total"])
            if not math.isclose(measured, float(metadata["物理最佳独立损失"]), abs_tol=1e-4):
                raise ValueError("任-04最近状态物理最佳分数与独立诊断不一致")
        if state is None or state["epoch"] != epoch:
            _atomic_complete_state_copy(payload, sidecar)
        else:
            _assert_complete_best_snapshot(state, payload, name, expected, origin, config_sha, arm)
    if observed_epoch != epoch:
        if earlier_observed is None:
            raise ValueError("任-04历史观测最佳完整模型缺失，不得声称最佳模型视图")
        observed, validation, _ = earlier_observed
        view = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        ancestry = view.get("任04新模型视图来源", {})
        if (
            view.get("epoch") != 300 + observed_epoch
            or ancestry.get("任04本阶段实际轮次") != observed_epoch
            or ancestry.get("源校正末SHA256") != origin["SHA256"]
            or ancestry.get("任04预登记配置SHA256") != config_sha
            or ancestry.get("运行资格") != metadata["运行资格"]
            or view.get("validation_selection_score_c") != metadata["观测最佳选分_摄氏度"]
            or not math.isclose(view.get("validation_rmse_c", math.inf), validation["顶部"], abs_tol=1e-4)
            or any(not math.isclose(view.get("validation_sensor", {}).get(key, math.inf),
                                    value, abs_tol=1e-4)
                   for key, value in validation.items() if key != "顶部")
            or not _deep_equal(view.get("model_state"), observed["model_state"])
        ):
            raise ValueError("任-04历史best.pt与合法HF验证、完整观测最佳张量或来源不一致")
        return
    score, validation = _validation_selection(
        model, validation_loader, validation_sensor, device, weights,
    )
    view_path = output / "best.pt"
    view = torch.load(view_path, map_location="cpu", weights_only=False) if view_path.exists() else None
    if view is not None and view.get("epoch", -1) > 300 + epoch:
        raise ValueError("任-04模型视图超前于完整最近状态，不得使用未提交最佳")
    if view is not None and view.get("epoch") == 300 + epoch and (
        view.get("任04新模型视图来源", {}).get("任04本阶段实际轮次") != epoch
        or view.get("任04新模型视图来源", {}).get("源校正末SHA256") != origin["SHA256"]
        or view.get("validation_selection_score_c") != metadata["观测最佳选分_摄氏度"]
        or view.get("任04新模型视图来源", {}).get("任04预登记配置SHA256") != config_sha
        or view.get("任04新模型视图来源", {}).get("运行资格") != metadata["运行资格"]
        or not math.isclose(view.get("validation_rmse_c", math.inf), validation["顶部"], abs_tol=1e-4)
        or any(not math.isclose(view.get("validation_sensor", {}).get(key, math.inf),
                                value, abs_tol=1e-4)
               for key, value in validation.items() if key != "顶部")
        or set(view["model_state"]) != set(payload["model_state"])
        or any(not torch.equal(value, view["model_state"][name])
               for name, value in payload["model_state"].items())
    ):
        raise ValueError("任-04同轮模型视图张量或预登记来源与真实观测最佳不一致")
    if view is None or view["epoch"] < 300 + epoch:
        _real_model_view(
            view_path, architecture, model, epoch=epoch, score=score,
            arm=arm, source_sha=origin["SHA256"], config_sha=config_sha,
            top_rmse=validation["顶部"],
            sensor_validation={name: value for name, value in validation.items() if name != "顶部"},
            diagnostic_only=diagnostic_only,
        )


def run_task04_joint(
    *, arm: str, output_directory: str | Path, budget_epochs: int = 200,
    session_epoch_limit: int | None = None, diagnostic_only: bool = False,
    source_checkpoint: str | Path | None = None, decision_path: str | Path = TASK03_DECISION,
    resume_training_checkpoint: str | Path | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    if arm not in ARMS or budget_epochs != 200 or (
        session_epoch_limit is not None and not 1 <= session_epoch_limit <= 200
    ):
        raise ValueError("任-04只允许同200轮预算的冻结/有限解冻两臂，门禁会话至多200轮")
    if session_epoch_limit is not None and session_epoch_limit < budget_epochs and not diagnostic_only:
        raise ValueError("任-04短CPU门禁须显式标识diagnostic_only，不得参与正式采用")
    if resume_training_checkpoint is not None and session_epoch_limit is None:
        raise ValueError("任-04续跑须显式设定本次会话上限，并使用同一原预算")
    decision = json.loads(_resolve(decision_path).read_text(encoding="utf-8"))
    origin = decision["任04共同阶段起点"]
    source = source_checkpoint or origin["检查点"]
    source_state, architecture = validate_task04_source(source, decision_path)
    splits = build_power_splits()
    contract = validate_task04_splits(splits)
    registry = load_yaml(str(TASK04_CONFIG))
    validate_task04_registration(registry, decision, splits)
    config_sha = sha256_file(TASK04_CONFIG)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, optimizer = fork_task04_model(source_state, architecture, arm, device, source)
    source_hf_step = float(next(iter(source_state["optimizer_state"]["state"].values()))["step"])
    base_state = {name: value.detach().clone() for name, value in source_state["model_state"].items()}
    output = _resolve(output_directory)
    if resume_training_checkpoint is None and output.exists():
        raise FileExistsError(f"任-04训练产物已存在，禁止覆盖：{output}")
    if resume_training_checkpoint is not None and not output.is_dir():
        raise FileNotFoundError(f"任-04续跑目录不存在：{output}")
    training = load_yaml("configs/training.yaml")
    losses = training["loss_weights"]
    weights = training["multifidelity_selection_weights"]
    materials = load_materials()
    boundaries = load_resolved_boundary_conditions()
    physics = PhysicsLossComputer(materials, boundaries, PhysicsLossWeights(
        pde=float(losses["pde"]), boundary=float(losses["boundary"]),
        initial=float(losses["initial"]), interface=float(losses["interface"]),
    ))
    train_data = _ir_dataset("train", splits.hf_train)
    validation_data = _ir_dataset("validation", splits.hf_validation)
    validation_loader = DataLoader(validation_data, batch_size=2048, shuffle=False)
    sensor = _sensor_tensors(device, split="train", powers_w=splits.hf_train)
    validation_sensor = _sensor_tensors(device, split="validation", powers_w=splits.hf_validation)
    simulation_data = load_sampled_points(
        contract["LF仿真回放功率"], 2048, 50_000 + int(architecture["seed"]),
        sampling_mode="material_time",
    )
    if len(simulation_data) != 60 * 2048 or len(train_data) // 2048 != 14:
        raise RuntimeError("任-04 HF每轮15步/LF每轮60功率完整真实仿真回放预算改变")
    budget = {"任04联合续训轮次": 200}
    consumption = {"HF训练观测点": 0, "HF训练传感器点": 0,
                   "LF仿真训练点": 0, "物理配点": 0, "HF观测优化步": 0,
                   "物理优化步": 0}
    previous_correction_best = float(architecture["validation_selection_score_c"])
    resume_epoch = 0
    started = time.perf_counter()
    if resume_training_checkpoint is None:
        output.mkdir(parents=True)
        write_config_snapshot(output)
        (output / "training.jsonl").write_text("", encoding="utf-8")
        score, initial_validation = _validation_selection(model, validation_loader,
                                                           validation_sensor, device, weights)
        diagnostic = evaluate_schedule_guardrails(model, physics, materials, boundaries, device)
        physical = float(diagnostic["独立局部物理损失"]["physics_total"])
        best, best_epoch, physical_epoch, initial_score = score, 0, 0, score
        metadata = _stage_metadata(origin, arm, initial_score, best, best_epoch,
                                   physical, physical_epoch, consumption, source_hf_step,
                                   source_state["parameter_requires_grad"], config_sha,
                                   diagnostic_only)
        metadata["原校正阶段观测最佳选分_摄氏度"] = previous_correction_best
        for name in ("阶段_初始.pt", "阶段_观测最佳.pt", "阶段_物理最佳.pt"):
            save_training_state(output / name, model, optimizer, stage="joint", epoch=0,
                                budget=budget, metadata=metadata)
        _real_model_view(
            output / "best.pt", architecture, model, epoch=0, score=best,
            arm=arm, source_sha=origin["SHA256"], config_sha=config_sha,
            top_rmse=initial_validation["顶部"],
            sensor_validation={name: value for name, value in initial_validation.items()
                               if name != "顶部"}, diagnostic_only=diagnostic_only,
        )
    else:
        resume = _resolve(resume_training_checkpoint).resolve()
        validate_task04_resume_name(resume)
        if resume.parent != output.resolve():
            raise ValueError("任-04续跑状态必须位于本臂原目录，不得跨臂续跑")
        snapshot = json.loads((output / "config_snapshot/sha256.json").read_text(encoding="utf-8"))
        if any(sha256_file(PROJECT_ROOT / name) != snapshot[name] for name in CONFIG_FILES):
            raise ValueError("任-04续跑源配置改变，不得假装同预算连续训练")
        payload = load_training_state(resume, model, optimizer)
        metadata = payload["metadata"]
        validate_task04_resume_qualification(metadata, diagnostic_only=diagnostic_only)
        if (
            payload["stage"] != "joint" or payload["budget"] != budget
            or metadata["源校正末SHA256"] != origin["SHA256"]
            or metadata.get("任04预登记配置SHA256") != config_sha
            or metadata["任04运行臂"] != arm
        ):
            raise ValueError("任-04续跑权重/冻结列表/源SHA和原预算不一致")
        resume_epoch = int(payload["epoch"])
        if resume_epoch >= 200:
            raise ValueError("任-04已达到原200轮预算，不得超额续跑")
        reconcile_correction_resume_log(output, epoch=resume_epoch, state_file=resume)
        initial_score = metadata["初始验证选分_摄氏度"]
        best = metadata["观测最佳选分_摄氏度"]
        best_epoch = metadata["观测最佳任04轮次"]
        physical = metadata["物理最佳独立损失"]
        physical_epoch = metadata["物理最佳任04轮次"]
        consumption = metadata["累计实际消耗"]
        _assert_resume_log_budget(output, resume_epoch, consumption, len(sensor[0]))
        _reconcile_committed_best(
            output, payload, architecture, model, validation_loader, validation_sensor,
            weights, physics, materials, boundaries, device, arm, origin, config_sha,
            diagnostic_only,
        )
        committed_random = payload["random_state"]
        random.setstate(committed_random["python"])
        np.random.set_state(committed_random["numpy"])
        torch.set_rng_state(committed_random["torch_cpu"])
        if committed_random["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(committed_random["torch_cuda"])
    session_end = min(200, resume_epoch + (session_epoch_limit or 200))
    latest_record: dict[str, Any] = {}
    for epoch in range(resume_epoch + 1, session_end + 1):
        epoch_started = time.perf_counter()
        # Dedicated generators make the two arm permutations identical independently of execution order.
        hf_loader = DataLoader(train_data, batch_size=2048, shuffle=True,
                               generator=torch.Generator().manual_seed(301_000 + epoch))
        lf_loader = DataLoader(simulation_data, batch_size=2048, shuffle=True,
                               generator=torch.Generator().manual_seed(501_000 + epoch))
        if len(hf_loader) != 15 or len(lf_loader) != 60:
            raise RuntimeError("任-04一轮必须确实暴露15个HF batch和60个LF replay batch")
        lf_batches = iter(lf_loader)
        model.train()
        ir_losses, sensor_losses, replay_losses = [], [], []
        replay_forward_seconds = 0.0
        cu_points, sic_points, ir_points, replay_points = 0, 0, 0, 0
        for coordinates, target, weight in hf_loader:
            coordinates, target, weight = (value.to(device) for value in (coordinates, target, weight))
            optimizer.zero_grad(set_to_none=True)
            prediction = model(coordinates)
            ir_loss = (weight * ((prediction - target) / model.scales.temperature_scale_k).square()).sum() / weight.sum()
            sensor_x, sensor_target, sensor_delta, baseline = sensor
            sensor_prediction = model(sensor_x)
            absolute, delta = _macro_sensor_training_losses(
                sensor_prediction, sensor_target, sensor_delta, baseline,
                sensor_x, model.scales.temperature_scale_k,
            )
            sensor_loss = float(losses["sensor_absolute"]) * absolute + float(losses["sensor_delta"]) * delta
            replay_x, replay_target = zip(*(next(lf_batches) for _ in range(4)))
            replay_x = torch.cat(replay_x).to(device)
            replay_target = torch.cat(replay_target).to(device)
            replay_started = time.perf_counter()
            replay_loss = low_replay_loss(model, replay_x, replay_target)
            replay_forward_seconds += time.perf_counter() - replay_started
            total = float(losses["ir"]) * ir_loss + sensor_loss
            if arm == "有限解冻":
                total = total + float(losses["low_fidelity"]) * replay_loss
            total.backward()
            optimizer.step()
            ir_losses.append(float(ir_loss.detach()))
            sensor_losses.append(float(sensor_loss.detach()))
            replay_losses.append(float(replay_loss.detach()))
            ir_points += len(coordinates)
            replay_points += len(replay_x)
            cu_points += int((replay_x[:, 4] < 0.5).sum())
            sic_points += int((replay_x[:, 4] >= 0.5).sum())
        try:
            next(lf_batches)
        except StopIteration:
            pass
        else:
            raise RuntimeError("任-04 LF回放本轮未消耗完60功率训练数据")
        if replay_points != len(simulation_data) or cu_points == 0 or sic_points == 0:
            raise RuntimeError("任-04 LF回放没有完整暴露60功率及Cu/SiC真实仿真点")
        collocation = sample_collocation(256, device, seed=300 + epoch)
        components = physics_optimizer_step(model, optimizer, physics, collocation)
        changed_lf = {
            name for name, value in model.state_dict().items()
            if name.startswith("low_fidelity_model.")
            and not torch.equal(value.detach().cpu(), base_state[name])
        }
        outside_changes = changed_lf - (PROJECTION_NAMES if arm == "有限解冻" else set())
        if outside_changes:
            raise RuntimeError("任-04 LF冻结参数改变：不可解冻整个低保真网络")
        consumption["HF训练观测点"] += ir_points
        consumption["HF训练传感器点"] += len(sensor[0]) * len(hf_loader)
        consumption["LF仿真训练点"] += replay_points
        consumption["物理配点"] += 256
        consumption["HF观测优化步"] += len(hf_loader)
        consumption["物理优化步"] += 1
        eval_due = epoch % 10 == 0 or epoch == 200 or (
            diagnostic_only and epoch == session_end
        )
        stage_score, guardrails = None, None
        modality, lf_material = None, None
        if eval_due:
            stage_score, stage_validation = _validation_selection(
                model, validation_loader, validation_sensor, device, weights,
            )
            modality = _hf_modalities(model, device, contract["HF合法验证功率"])
            lf_material = _lf_material_validation(model, sorted(splits.simulation_validation), device)
            guardrails = evaluate_schedule_guardrails(model, physics, materials, boundaries, device)
            if stage_score < best - 1e-4:
                best, best_epoch = stage_score, epoch
            physical_now = float(guardrails["独立局部物理损失"]["physics_total"])
            if physical_now < physical:
                physical, physical_epoch = physical_now, epoch
        latest_record = {
            "epoch": epoch, "任04轮次": epoch, "源校正实际轮次": 300, "运行臂": arm,
            "HF观测优化步": len(hf_loader), "物理优化步": 1, "物理配点": 256,
            "HF训练观测点": ir_points, "LF仿真回放训练点": replay_points,
            "LF仿真回放Cu点": cu_points, "LF仿真回放SiC点": sic_points,
            "LF回放监督模式": "low", "LF回放反向更新": arm == "有限解冻",
            "LF冻结权重符合源状态": not outside_changes,
            "LF最后投影实际变化张量数": len(changed_lf & PROJECTION_NAMES),
            "LF其余权重实际变化张量数": len(changed_lf - PROJECTION_NAMES),
            "LF回放计算墙钟秒": replay_forward_seconds,
            "LF回放计算与反传差异": (
                "冻结臂仅无梯度forward监测，无LF反传" if arm == "冻结"
                else "有限解冻臂有梯度forward并入HF反传，仅更新LF最后投影"
            ),
            "本轮含评估墙钟秒": time.perf_counter() - epoch_started,
            "HF学习率": 1e-4, "LF投影学习率": 0.0 if arm == "冻结" else 1e-5,
            "HF顶部训练损失": float(np.mean(ir_losses)),
            "HF环温训练损失": float(np.mean(sensor_losses)),
            "LF仿真回放损失": float(np.mean(replay_losses)),
            "HF合法验证选分_摄氏度": stage_score,
            "HF合法验证分模态RMSE_摄氏度": modality,
            "LF合法验证材料RMSE_摄氏度": lf_material,
            "本轮物理训练损失": {name: float(value.detach()) for name, value in components.items()},
            "累计实际消耗": consumption.copy(),
        }
        if guardrails is not None:
            latest_record.update(guardrails)
        with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(latest_record, ensure_ascii=False) + "\n")
        metadata = _stage_metadata(origin, arm, initial_score, best, best_epoch,
                                   physical, physical_epoch, consumption.copy(), source_hf_step,
                                   source_state["parameter_requires_grad"], config_sha,
                                   diagnostic_only)
        metadata["原校正阶段观测最佳选分_摄氏度"] = previous_correction_best
        for name, selected_epoch in (("阶段_观测最佳.pt", best_epoch),
                                     ("阶段_物理最佳.pt", physical_epoch)):
            if selected_epoch < epoch:
                previous = output / name
                if not previous.is_file():
                    raise ValueError("任-04上一轮历史最佳完整状态缺失，不得写入最近状态")
                metadata[f"已提交旧{name}SHA256"] = sha256_file(previous)
        save_training_state(output / "阶段_最近.pt", model, optimizer,
                            stage="joint", epoch=epoch, budget=budget, metadata=metadata)
        if best_epoch == epoch:
            save_training_state(output / "阶段_观测最佳.pt", model, optimizer,
                                stage="joint", epoch=epoch, budget=budget, metadata=metadata)
        if physical_epoch == epoch:
            save_training_state(output / "阶段_物理最佳.pt", model, optimizer,
                                stage="joint", epoch=epoch, budget=budget, metadata=metadata)
        if best_epoch == epoch:
            _real_model_view(
                output / "best.pt", architecture, model,
                epoch=epoch, score=best, arm=arm, source_sha=origin["SHA256"],
                config_sha=config_sha,
                top_rmse=stage_validation["顶部"],
                sensor_validation={name: value for name, value in stage_validation.items()
                                   if name != "顶部"}, diagnostic_only=diagnostic_only,
            )
    status = (
        "完成同预算200轮，策略采用须与另一臂比较且主代理核对"
        if session_end == 200 and not diagnostic_only else
        "短诊断累计达到200轮仍无正式采用资格，不生成正式阶段末状态"
        if session_end == 200 and diagnostic_only else
        "短门禁暂停，不得用于200轮正式采用" if diagnostic_only else "会话暂停，从阶段_最近.pt续跑"
    )
    if session_end == 200 and not diagnostic_only:
        for name in (task04_terminal_name(arm), "阶段_训练末.pt"):
            save_training_state(output / name, model, optimizer, stage="joint",
                                epoch=200, budget=budget, metadata=metadata)
    summary = {
        "状态": status, "运行臂": arm, "本臂实际完成轮次": session_end,
        "运行资格": metadata["运行资格"],
        "原预登记预算": 200, "源校正末SHA256": origin["SHA256"],
        "任04预登记配置SHA256": config_sha,
        "源校正阶段观测最佳选分_摄氏度": previous_correction_best,
        "历史部署B0参考": copy.deepcopy(decision["历史部署B0参考"]),
        "同一源状态初始验证选分_摄氏度": initial_score,
        "观测最佳验证选分_摄氏度": best, "观测最佳任04轮次": best_epoch,
        "物理最佳独立损失": physical, "物理最佳任04轮次": physical_epoch,
        "真实最后状态": (
            task04_terminal_name(arm) if session_end == 200 and not diagnostic_only
            else "阶段_最近.pt"
        ),
        "旧test_Data温度读取": False,
        "LF源权重只允许更新投影": arm == "有限解冻",
        "两臂共同HF优化器动量来源": "同一任03校正末AdamW，逐张量核验",
        "本臂LF监测计算不同于更新": arm == "冻结",
        "累计实际消耗": consumption.copy(), "本会话训练秒": time.perf_counter() - started,
        "最后一轮门禁证据": latest_record,
        "策略采用": "待双方同预算200轮与历史B0参照核对；本轮不做采用声明",
    }
    temporary = output / "阶段报告.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "阶段报告.json")
    return summary
