"""任08 F1 独立HF观测入场；物理来源缺口未闭合前不得作正式PINN消融。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import assert_no_hf_leakage, build_power_splits
from sic_cu.models import ModelScales
from sic_cu.models.deeponet_pinn import DeepONetPINN
from sic_cu.physics.resolution import resolve_physics_state
from sic_cu.train.multifidelity import (
    _ir_dataset, _macro_sensor_training_losses, _sensor_tensors,
)


F1_KWARGS = {
    "width": 128, "latent_dim": 128, "blocks": 3,
    "activation": "tanh", "include_material": True,
}


def _path(value: str | Path) -> Path:
    result = Path(value)
    return result if result.is_absolute() else PROJECT_ROOT / result


def audit_f1_physics_inputs() -> dict[str, Any]:
    """Current shared resolved parameters contain LF-derived h and contact; never bless F1."""
    physics = resolve_physics_state()
    values = physics["values"]
    raw = load_yaml("configs/boundary_conditions.yaml")
    if (
        raw["interface"].get("parameter_status") != "low_fidelity_initialization_available"
        or values["contact_resistance_m2_k_w"]["status"] != "effective_initialization"
        or values["top_convection_coefficient_w_m2_k"]["status"] != "effective_initialization"
        or values["bottom_convection_coefficient_w_m2_k"]["status"] != "effective_initialization"
        or values["silicon_carbide_emissivity"]["status"] != "assumed_scenario"
        or values["copper_emissivity"]["status"] != "assumed_scenario"
    ):
        raise ValueError("任08 F1共同物理来源状态改变；未经新独立证明与事前登记不得沿用旧来源审计")
    documents = (
        "configs/boundary_conditions.yaml", "configs/geometry.yaml", "configs/materials.yaml",
        "reports/natural_convection_parameter_basis.md", "reports/physics_sensitivity.md",
    )
    return {
        "可作为纯HF-only物理输入": False,
        "限制": "物理输入仍源于LF，无法称纯HF-only；需独立h/辐射率与接触处理事前证据",
        "阻断参数": {
            "LF辨识接触热阻": values["contact_resistance_m2_k_w"]["value"],
            "LF表面温差所选顶部对流": values["top_convection_coefficient_w_m2_k"]["value"],
            "LF表面温差所选底部对流": values["bottom_convection_coefficient_w_m2_k"]["value"],
            "名义未测SiC辐射率": values["silicon_carbide_emissivity"]["value"],
            "名义未测Cu辐射率": values["copper_emissivity"]["value"],
        },
        "模型限制": (
            "可以只由HF顶部和Hot/Cold观测检验模型初态和观测优化通路；未独立确定"
            "物理输入时不计算体内PDE、边界、初值或界面训练损失，不称正式F1 PINN。"
            "完美接触(None)是可选建模假设，但不等于含LF辨识热阻的F3界面物理。"
        ),
        "来源工件SHA256": {relative: sha256_file(PROJECT_ROOT / relative)
                            for relative in documents},
    }


def require_independent_f1_physics_inputs() -> None:
    physics = audit_f1_physics_inputs()
    raise ValueError("任08 F1纯HF-only PINN物理入口已阻断：" + physics["限制"])


def initialize_f1_hf_only(
    seed: int, device: torch.device = torch.device("cpu"), *,
    model_kwargs: Mapping[str, Any] | None = None,
) -> tuple[DeepONetPINN, torch.optim.AdamW]:
    if seed not in range(5):
        raise ValueError("任08 F1诊断只允许五个预定随机种子0到4")
    if model_kwargs is not None and dict(model_kwargs) != F1_KWARGS:
        raise ValueError("任08 F1只允许固定新DeepONet架构；LF预训练/字典/伪标签来源禁止进入模型")
    if device.type != "cpu":
        raise ValueError("任08 F1入场阶段只许CPU建模；正式五seed GPU需等任07冻结")
    geometry = load_yaml("configs/geometry.yaml")
    scales = ModelScales()
    if (
        float(geometry["copper"]["radius_m"]) != scales.r_max_m
        or float(geometry["embedding"]["copper_bottom_z_m"]) != scales.z_min_m
    ):
        raise ValueError("任08 F1几何参数与独立模型坐标缩放器不一致")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8_080_000 + seed)
        model = DeepONetPINN(scales=scales, **F1_KWARGS).to(device)
    training = load_yaml("configs/training.yaml")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["optimizer"]["learning_rate"]),
        weight_decay=float(training["optimizer"]["weight_decay"]),
    )
    if optimizer.state_dict()["state"] or any(
        key.startswith("low_fidelity") for key in model.state_dict()
    ):
        raise ValueError("任08 F1从全新HF-only空AdamW及无任何LF参数开始")
    return model, optimizer


def load_f1_hf_observations(device: torch.device = torch.device("cpu")) -> dict[str, Any]:
    if device.type != "cpu":
        raise ValueError("任08 F1来源入场只许CPU；不提前占用任07正式CUDA设备")
    splits = build_power_splits()
    if len(splits.hf_train) != 12 or len(splits.hf_validation) != 3:
        raise ValueError("任08 F1必须固定HF12训练、3合法验证")
    assert_no_hf_leakage(
        {"HF顶部": splits.hf_train, "HF热端": splits.hf_train,
         "HF冷端": splits.hf_train},
        splits.hf_validation | splits.hf_test | splits.external_sensor_test,
    )
    observed = {
        "train_ir": _ir_dataset("train", splits.hf_train),
        "validation_ir": _ir_dataset("validation", splits.hf_validation),
        "train_sensor": _sensor_tensors(device, split="train", powers_w=splits.hf_train),
        "validation_sensor": _sensor_tensors(device, split="validation",
                                             powers_w=splits.hf_validation),
    }
    if len(observed["train_ir"]) != 29593:
        raise ValueError("任08 F1 HF12训练顶部IR真实行数应为29593，不可短样或伪标签")
    for split, allowed in (("train", splits.hf_train), ("validation", splits.hf_validation)):
        ir = observed[f"{split}_ir"].tensors[0]
        sensor = observed[f"{split}_sensor"][0]
        ir_powers = {round(float(power), 4) for power in torch.unique(ir[:, 3])}
        sensor_powers = {round(float(power), 4) for power in torch.unique(sensor[:, 3])}
        if ir_powers != allowed or sensor_powers != allowed or not torch.all(ir[:, 4] == 1):
            raise ValueError(f"任08 F1 {split} 顶部IR/Hot/Cold与HF合法同折功率不同步")
        for power in allowed:
            chosen = torch.isclose(sensor[:, 3], torch.tensor(power), rtol=0.0, atol=1e-4)
            radii = {round(float(radius), 4) for radius in torch.unique(sensor[chosen][:, 0])}
            if radii != {0.028, 0.0415}:
                raise ValueError(f"任08 F1 {split} 工况{power:g}W缺真实Hot或Cold环温")
    return observed


def run_task08_f1_formal(*, seed: int, output_directory: str | Path) -> None:
    del seed, output_directory
    raise ValueError(
        "任08 F1正式训练被阻断：任07五seed未冻结且F1独立h/辐射率/接触物理"
        "来源未事前登记；物理输入仍源于LF，无法称纯HF-only"
    )


def _subset_sensor_power(sensor: tuple[torch.Tensor, ...], power: float) -> tuple[torch.Tensor, ...]:
    coords, target, delta, baseline = sensor
    ids = torch.nonzero(torch.isclose(coords[:, 3], torch.tensor(power),
                                      rtol=0.0, atol=1e-4), as_tuple=False).flatten()
    positions = torch.full((len(coords),), -1, dtype=torch.long)
    positions[ids] = torch.arange(len(ids))
    selected_baseline = positions[baseline[ids]]
    if len(ids) == 0 or (selected_baseline < 0).any():
        raise ValueError("任08 F1 Hot/Cold诊断须同时保留所选HF训练工况两条完整基线曲线")
    return coords[ids], target[ids], delta[ids], selected_baseline


def run_task08_f1_cpu_diagnostic(
    *, seed: int, output_directory: str | Path,
) -> dict[str, Any]:
    if seed not in range(5):
        raise ValueError("任08 F1入场CPU短诊断只许五seed0到4")
    output = _path(output_directory)
    if output.exists():
        raise FileExistsError(f"任08 F1诊断已有原件，不得覆盖：{output}")
    from sic_cu.train.task07_formal import TASK07_REGISTRATION
    output_path = output.resolve()
    protected = (
        PROJECT_ROOT / "reports/runs", PROJECT_ROOT / "reports/development_v4",
        PROJECT_ROOT / "data", TASK07_REGISTRATION.parent,
    )
    if any(folder.resolve() == output_path or folder.resolve() in output_path.parents
           for folder in protected):
        raise ValueError("任08 F1诊断不可写入旧B0来源、任07冻结训练或数据只读目录")
    physics_audit = audit_f1_physics_inputs()
    observations = load_f1_hf_observations(torch.device("cpu"))
    model, optimizer = initialize_f1_hf_only(seed, torch.device("cpu"))
    splits = build_power_splits()
    training = load_yaml("configs/training.yaml")
    power = float(min(splits.hf_train))
    x, target, weight = observations["train_ir"].tensors
    subset = torch.nonzero(torch.isclose(x[:, 3], torch.tensor(power),
                                           rtol=0.0, atol=1e-4), as_tuple=False).flatten()[:64]
    sensor_x, sensor_target, delta, baseline = _subset_sensor_power(
        observations["train_sensor"], power,
    )
    if len(subset) != 64 or len(sensor_x) < 2:
        raise ValueError("任08 F1一优化步只许真实HF训练顶部64点及真实Hot/Cold同步基线")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    ir_prediction = model(x[subset])
    ir_loss = (weight[subset] * ((ir_prediction - target[subset]) /
             model.scaler.scales.temperature_scale_k).square()).sum() / weight[subset].sum()
    sensor_prediction = model(sensor_x)
    absolute, change = _macro_sensor_training_losses(
        sensor_prediction, sensor_target, delta, baseline, sensor_x,
        model.scaler.scales.temperature_scale_k,
    )
    weights = training["loss_weights"]
    total = (float(weights["ir"]) * ir_loss
             + float(weights["sensor_absolute"]) * absolute
             + float(weights["sensor_delta"]) * change)
    if not torch.isfinite(total):
        raise ValueError("任08 F1合法HF观测一优化步训练目标非有限")
    total.backward()
    optimizer.step()
    output.mkdir(parents=True)
    metadata = {
        "正式资格": False, "独立物理来源已解决": False,
        "资格": "CPU仅HF观测一优化步诊断；无PINN物理，不可作正式F1",
        "训练种子": seed, "合法HF训练功率": sorted(map(float, splits.hf_train)),
        "HF合法验证功率": sorted(map(float, splits.hf_validation)),
        "真实诊断顶部HF点数": len(subset), "真实诊断环温HF点数": len(sensor_x),
        "HF观测优化步": 1, "物理损失优化步": 0,
        "从LF检查点或字典初始化": False, "旧test_Data温度标签读取": False,
    }
    torch.save({
        "model_state": {name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "random_state": {"torch_cpu": torch.get_rng_state(), "torch_cuda": None},
        "metadata": metadata,
    }, output / "诊断训练状态.pt")
    report = {
        "资格": metadata["资格"], "运行种子": seed,
        "HF合法训练功率数": len(splits.hf_train),
        "HF合法验证功率数": len(splits.hf_validation),
        "HF训练IR全合同真实点数": len(observations["train_ir"]),
        "诊断实际顶部HF点数": len(subset), "诊断实际HotCold真实点数": len(sensor_x),
        "HF观测优化步": 1, "物理损失优化步": 0,
        "诊断HF观测归一目标": float(total.detach()),
        "从LF检查点或字典初始化": False,
        "物理来源限制": physics_audit["限制"],
        "物理缺口来源审计": physics_audit,
        "旧test_Data温度标签读取": False,
        "源码与合同来源SHA256": {
            relative: sha256_file(PROJECT_ROOT / relative) for relative in (
                "configs/splits.yaml", "configs/training.yaml",
                "src/sic_cu/models/deeponet_pinn.py",
            )
        },
    }
    (output / "入场摘要.json").write_text(json.dumps(report, ensure_ascii=False,
                                              indent=2) + "\n", encoding="utf-8")
    hashes = {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}
    (output / "审计工件SHA256.json").write_text(json.dumps(hashes, ensure_ascii=False,
                                                    indent=2) + "\n", encoding="utf-8")
    report["逐工件SHA256"] = hashes
    return report
