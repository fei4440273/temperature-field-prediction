"""任08 F2只去HF体内PDE软损失的来源入场与非正式CPU一轮诊断。"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import assert_no_hf_leakage, build_power_splits
from sic_cu.losses.physics import PhysicsLossComputer, PhysicsLossWeights
from sic_cu.physics.collocation import sample_collocation
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.train.common import physics_optimizer_step
from sic_cu.train.multifidelity import (
    _ir_dataset, _macro_sensor_training_losses, _sensor_tensors,
)
from sic_cu.train.task07_source import (
    MANIFEST_SHA256, Task07Initialization, Task07Source,
    _lf_tensor_sha256, fork_task07_initialization, validate_task07_sources,
)


F2_TRAINING_WEIGHTS = {
    "ir": 1.0, "sensor_absolute": 5.0, "sensor_delta": 1.0,
    "low_fidelity": 1.0, "pde": 0.0, "initial": 1.0,
    "boundary": 1.0, "interface": 1.0,
}


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _verify_shared_protocol() -> None:
    training = load_yaml("configs/training.yaml")
    splits = build_power_splits()
    shared = training["loss_weights"]
    if (
        training["optimizer"]["name"] != "adamw"
        or float(training["optimizer"]["learning_rate"]) != 0.001
        or float(training["optimizer"]["joint_learning_rate"]) != 0.0001
        or float(training["optimizer"]["weight_decay"]) != 0.000001
        or training["epochs"]["high_fidelity"] != 1500
        or training["epochs"]["joint"] != 500
        or training["selection_metric_version"] != "macro_v1"
        or float(training["early_stopping"]["patience"]) != 200.0
        or float(training["early_stopping"]["min_delta"]) != 0.0001
        or dict(training["multifidelity_selection_weights"]) != {
            "ir_rmse": 1.0, "sensor_absolute_rmse": 0.2,
            "sensor_delta_rmse": 1.0,
        }
        or training["multifidelity"]["joint_simulation_samples_per_power"] != 2048
        or any(float(shared[key]) != (1.0 if key == "pde" else value)
               for key, value in F2_TRAINING_WEIGHTS.items())
        or set(shared) & {"energy", "global_energy", "energy_balance"}
        or len(splits.simulation_train) != 60
        or len(splits.simulation_validation) != 10
        or len(splits.hf_train) != 12
        or len(splits.hf_validation) != 3
    ):
        raise ValueError(
            "任08 F2须沿用任07 HF12/3 LF60/10、同预算宏平均选分/早停和除PDE外"
            "全部共同配置；不可改共享YAML"
        )


def load_f2_sources() -> dict[int, Task07Source]:
    """只读已锁的五份V4 B0同seed LF与历史HF架构，绝不读旧测试温度。"""
    _verify_shared_protocol()
    return validate_task07_sources()


def fork_f2_start(
    sources: Mapping[int, Task07Source], seed: int,
    device: torch.device = torch.device("cpu"),
) -> Task07Initialization:
    if device.type != "cpu":
        raise ValueError("任08 F2入场未获任07冻结，禁止CUDA正式重训或从CUDA诊断起步")
    _verify_shared_protocol()
    initial = fork_task07_initialization(sources, seed, "E0", device)
    source = sources[seed]
    if (
        initial.arm != "E0" or initial.selected_for_training
        or initial.optimizer.state_dict()["state"]
        or initial.model.correction[0].in_features != 6
        or _lf_tensor_sha256(initial.model.low_fidelity_model.state_dict())
        != source.lf_tensor_sha256
        or any(parameter.requires_grad for parameter in
               initial.model.low_fidelity_model.parameters())
        or any(not parameter.requires_grad for parameter in
               initial.model.correction.parameters())
    ):
        raise ValueError("任08 F2必须从同seed V4 LF全张量及新HF E0空AdamW开始")
    return initial


def load_f2_observations(
    source: Task07Source, device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    if device.type != "cpu":
        raise ValueError("任08 F2冻结前HF来源审计只允许CPU，不占任07正式GPU")
    splits = build_power_splits()
    if (
        source.hf_train_powers_w != tuple(sorted(splits.hf_train))
        or source.hf_validation_powers_w != tuple(sorted(splits.hf_validation))
        or source.lf_train_powers_w != tuple(sorted(splits.simulation_train))
        or source.lf_validation_powers_w != tuple(sorted(splits.simulation_validation))
    ):
        raise ValueError("任08 F2同seed LF/HF来源与当前合法折或预算不符")
    assert_no_hf_leakage(
        {"HF训练顶部": splits.hf_train, "HF训练热端": splits.hf_train,
         "HF训练冷端": splits.hf_train},
        splits.hf_validation | splits.hf_test | splits.external_sensor_test,
    )
    observed = {
        "train_ir": _ir_dataset("train", source.hf_train_powers_w),
        "validation_ir": _ir_dataset("validation", source.hf_validation_powers_w),
        "train_sensor": _sensor_tensors(device, "train", source.hf_train_powers_w),
        "validation_sensor": _sensor_tensors(device, "validation",
                                               source.hf_validation_powers_w),
    }
    if len(observed["train_ir"]) != 29593:
        raise ValueError("任08 F2须消费12工况全部真实HF顶部IR29593行，不许截短/伪标签")
    for split, allowed in (
        ("train", source.hf_train_powers_w),
        ("validation", source.hf_validation_powers_w),
    ):
        ir_x = observed[f"{split}_ir"].tensors[0]
        sensor_x, sensor_y, sensor_delta, sensor_baseline = observed[f"{split}_sensor"]
        ir_powers = {round(float(power), 4) for power in torch.unique(ir_x[:, 3])}
        sensor_powers = {round(float(power), 4) for power in torch.unique(sensor_x[:, 3])}
        if (
            ir_powers != set(allowed) or sensor_powers != set(allowed)
            or not torch.all(ir_x[:, 4] == 1)
            or not torch.all(sensor_x[:, 4] == 0)
            or not len(sensor_x) == len(sensor_y) == len(sensor_delta) == len(sensor_baseline)
            or not len(sensor_x) or (sensor_baseline < 0).any()
            or not (sensor_baseline < len(sensor_x)).all()
            or not torch.all(sensor_x[sensor_baseline, 3] == sensor_x[:, 3])
            or not torch.all(sensor_x[sensor_baseline, 0] == sensor_x[:, 0])
        ):
            raise ValueError(f"任08 F2 {split} IR与Hot/Cold必须同步于同一合法HF工况")
        for power in allowed:
            selected = torch.isclose(sensor_x[:, 3], torch.tensor(power), atol=1e-4,
                                      rtol=0.0)
            radii = {round(float(r), 4) for r in torch.unique(sensor_x[selected][:, 0])}
            if radii != {0.028, 0.0415}:
                raise ValueError(f"任08 F2 {split} 工况{power:g}W缺Hot/Cold两条环温")
    return observed


def _require_explicit_skip() -> None:
    if "compute_pde" not in inspect.signature(PhysicsLossComputer).parameters:
        raise ValueError(
            "任08 F2入场尚无真实PDE跳过开关；零权重仍求二阶导数且0乘NaN污染，"
            "不可运行诊断或正式F2"
        )


def make_f2_training_physics(device: torch.device) -> PhysicsLossComputer:
    if device.type != "cpu":
        raise ValueError("任08 F2未冻结前只许CPU来源诊断")
    _require_explicit_skip()
    return PhysicsLossComputer(
        load_materials(), load_resolved_boundary_conditions(),
        PhysicsLossWeights(pde=0.0, boundary=1.0, initial=1.0, interface=1.0),
        compute_pde=False,
    )


def make_f2_independent_audit_physics(device: torch.device) -> PhysicsLossComputer:
    if device.type != "cpu":
        raise ValueError("任08 F2冻结前只许CPU独立全项来源审核，不得抢CUDA正式算力")
    _require_explicit_skip()
    return PhysicsLossComputer(
        load_materials(), load_resolved_boundary_conditions(),
        PhysicsLossWeights(pde=1.0, boundary=1.0, initial=1.0, interface=1.0),
        compute_pde=True,
    )


def _f2_one_correction_epoch(
    initial: Task07Initialization, source: Task07Source, observed: Mapping[str, Any],
    physics: PhysicsLossComputer,
) -> dict[str, Any]:
    model, optimizer = initial.model, initial.optimizer
    if (
        physics.compute_pde is not False or physics.weights.pde != 0.0
        or any(getattr(physics.weights, name) != 1.0
               for name in ("initial", "boundary", "interface"))
    ):
        raise ValueError("任08 F2一轮物理必须只跳过体内PDE并保留初值/边界/界面梯度")
    hf_loader = DataLoader(
        observed["train_ir"], batch_size=2048, shuffle=True,
        generator=torch.Generator().manual_seed(7_070_000 + source.seed * 100_000 + 1),
    )
    if len(hf_loader) != 15 or len(observed["train_ir"]) != 29593:
        raise ValueError("任08 F2一轮须HF12真实顶部29593点，2048一批共15批")
    sx, sy, delta, baseline = observed["train_sensor"]
    model.train()
    ir_sum = sensor_sum = points = steps = 0
    for x, y, weight in hf_loader:
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, fidelity="high")
        ir_loss = (
            weight * ((prediction - y) / model.scales.temperature_scale_k).square()
        ).sum() / weight.sum()
        sensor_prediction = model(sx, fidelity="high")
        absolute, change = _macro_sensor_training_losses(
            sensor_prediction, sy, delta, baseline, sx,
            model.scales.temperature_scale_k,
        )
        sensor_loss = (F2_TRAINING_WEIGHTS["sensor_absolute"] * absolute
                       + F2_TRAINING_WEIGHTS["sensor_delta"] * change)
        objective = F2_TRAINING_WEIGHTS["ir"] * ir_loss + sensor_loss
        if not torch.isfinite(objective):
            raise ValueError("任08 F2真实HF观测训练目标出现非有限值")
        objective.backward()
        optimizer.step()
        ir_sum += float(ir_loss.detach())
        sensor_sum += float(sensor_loss.detach())
        points += len(x)
        steps += 1
    collocation = sample_collocation(
        256, torch.device("cpu"), seed=7_090_000 + source.seed * 100_000 + 1,
    )
    components = physics_optimizer_step(model, optimizer, physics, collocation)
    if (
        points != 29593 or steps != 15 or components["pde"] is not None
        or not torch.isfinite(components["physics_total"])
    ):
        raise ValueError("任08 F2一轮需15真实HF观测优化＋1仅非PDE物理优化")
    if any(
        float(record["step"]) != 16
        for record in optimizer.state_dict()["state"].values()
    ) or len(optimizer.state_dict()["state"]) != len(list(model.correction.parameters())):
        raise ValueError("任08 F2同seed新HF空AdamW必须实际提交15＋1共16步")
    return {
        "HF训练观测点": points, "HF训练传感器点": steps * len(sx),
        "HF观测优化步": steps, "物理优化步": 1, "物理配点": 256,
        "LF真实回放训练点": 0, "LF联合实际轮次": 0,
        "HF顶部训练损失": ir_sum / steps,
        "HF环温训练损失": sensor_sum / steps,
        "三项非PDE物理训练损失": {
            name: float(components[name].detach())
            for name in ("initial", "boundary", "interface", "physics_total")
        },
        "训练PDE残差": "未计算；不表示误差为零",
    }


def run_task08_f2_formal(*, seed: int, output_directory: str | Path) -> None:
    del seed, output_directory
    raise ValueError(
        "任08 F2正式五seed须等任07甲/乙真实冻结及独立事前登记；"
        "当前仅CPU来源通路，不宣称PDE贡献或跨模型排名"
    )


def run_task08_f2_cpu_diagnostic(
    *, seed: int, output_directory: str | Path,
) -> dict[str, Any]:
    if seed not in range(5):
        raise ValueError("任08 F2非正式CPU诊断只许事前指定seed0到4")
    output = _path(output_directory)
    if output.exists():
        raise FileExistsError(f"任08 F2已有诊断原件不可覆盖：{output}")
    target = output.resolve()
    protected = (
        PROJECT_ROOT / "reports/runs", PROJECT_ROOT / "reports/development_v4",
        PROJECT_ROOT / "data", PROJECT_ROOT / "configs",
        PROJECT_ROOT / "研究记录/任务07_正式五种子重训",
    )
    if any(folder.resolve() == target or folder.resolve() in target.parents
           for folder in protected):
        raise ValueError("任08 F2不能将诊断写入V4旧B0、任07冻结、数据或共享配置树")
    _require_explicit_skip()  # Before opening even the legal training observations.
    sources = load_f2_sources()
    source = sources[seed]
    initial = fork_f2_start(sources, seed)
    physics = make_f2_training_physics(torch.device("cpu"))
    observed = load_f2_observations(source)
    actual = _f2_one_correction_epoch(initial, source, observed, physics)
    low_after = _lf_tensor_sha256(initial.model.low_fidelity_model.state_dict())
    if low_after != source.lf_tensor_sha256:
        raise ValueError("任08 F2校正诊断不得改变整份同seed低保真源张量")
    metadata = {
        "正式资格": False, "仅CPU诊断": True, "运行种子": seed,
        "训练阶段": "HF校正一轮；联合未开始", "训练PDE残差": actual["训练PDE残差"],
        "源LF检查点SHA256": source.lf_checkpoint_sha256,
        "源LF张量SHA256": source.lf_tensor_sha256,
        "历史HF检查点仅作架构验证SHA256": source.hf_checkpoint_sha256,
        "V4_B0来源清单SHA256": MANIFEST_SHA256,
        "独立能源仅用于审计不反向传播": True,
    }
    output.mkdir(parents=True)
    torch.save({
        "model_state": {name: value.detach().cpu().clone()
                        for name, value in initial.model.state_dict().items()},
        "optimizer_state": initial.optimizer.state_dict(),
        "random_state": dict(initial.random_state, torch_cpu=torch.get_rng_state().clone()),
        "metadata": metadata,
    }, output / "诊断训练状态.pt")
    report = {
        "资格": "只验证合法来源及CPU一轮HF通路；绝非任08 F2正式五seed消融",
        **actual, **metadata,
        "HF合法训练功率数": len(source.hf_train_powers_w),
        "HF合法验证功率数": len(source.hf_validation_powers_w),
        "LF训练功率数": len(source.lf_train_powers_w),
        "LF验证功率数": len(source.lf_validation_powers_w),
        "冻结LF张量未变": True,
        "旧test_Data温度标签读取": False,
        "训练共享YAML的PDE权重仍为1": True,
        "物理保留项": ["初值", "轴及表面对流/辐射/激光/冷却边界", "SiC-Cu界面"],
        "物理来源限制": (
            "F2与F3共用LF辨识接触热阻及LF依据选取的表面对流参数，"
            "LF预训练曾用完整物理；不能称完全无物理。"
        ),
        "源码与来源SHA256": {relative: sha256_file(PROJECT_ROOT / relative)
                           for relative in (
                               "configs/splits.yaml", "configs/training.yaml",
                               "configs/boundary_conditions.yaml", "configs/materials.yaml",
                               "src/sic_cu/losses/physics.py",
                               "src/sic_cu/train/task07_source.py",
                               "src/sic_cu/train/task08_f2_no_pde.py",
                           )},
    }
    (output / "入场摘要.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    hashes = {file.name: sha256_file(file) for file in output.iterdir() if file.is_file()}
    (output / "审计工件SHA256.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    report["逐工件SHA256"] = hashes
    return report
