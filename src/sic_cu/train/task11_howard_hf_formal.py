"""Byte-locked Howard three-network HF adaptation under Task-11's finite budget.

All three networks train together from epoch one. This is neither LF plus a
residual nor an exact author-code/Adam reproduction. CPU fixtures grant no HF
permission; genuine training has a separate four-source ROOT gate.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import pickle
import random
import re
import tarfile
import time
from contextlib import contextmanager
from dataclasses import asdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import assert_no_hf_leakage, build_power_splits
from sic_cu.eval.protocol_checks import canonical_json_sha256, checkpoint_provenance, current_protocol_fingerprints
from sic_cu.losses import PhysicsLossComputer, PhysicsLossWeights
from sic_cu.models.common import ModelScales
from sic_cu.models.task11_howard_composite import Task11HowardComposite
from sic_cu.physics.collocation import sample_collocation
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.train.common import CONFIG_FILES, load_training_state, save_training_state, write_config_snapshot
from sic_cu.train.multifidelity import _ir_dataset, _macro_sensor_training_losses, _sensor_tensors
from sic_cu.train.simulation import load_sampled_points, set_seed
from sic_cu.train.task04_joint import _validation_selection
from sic_cu.train.task11_howard_hf_initialization import (
    initialize_task11_howard_hf, build_task11_howard_hf_adamw, _state_content_sha as _cpu_state_content_sha,
)
from sic_cu.train.task11_howard_lf_formal import (
    SOURCE_MEMBERS as HOWARD_LF_SOURCE_MEMBERS, METHOD as LF_METHOD,
    RUN_DIRECTORY as LF_RUN_DIRECTORY, _same, qualify_task11_howard_lf_source,
)
from sic_cu.train.task11_mlp_hf_formal import (
    SOURCE_MEMBERS as MLP_HF_SOURCE_MEMBERS, collect_task11_mlp_hf_data_catalog,
)


ROOT_LEDGER = "多保真DeepONet预测精度优化总计划与执行台账.md"
ROOT_TOKEN = "TASK11_HOWARD_HF_GATE:v1"
REGISTRY = "研究记录/任务11_外部对照/Howard适配三网HF联合预算前登记.yaml"
SOURCE_TAR = "研究记录/任务11_外部对照/Howard适配三网HF联合源码事前冻结.tar.gz"
LF_CATALOG = "研究记录/任务11_外部对照/Howard适配五LF完训身份清单.json"
HF_DATA_CATALOG = "研究记录/任务11_外部对照/任11_新MLP_HF12_3开发结构来源清单_20260916T153147.json"
HF_DATA_CATALOG_SHA = "7665603abb31fa225c5ef36394ddf7c3b97461340b8b3307863766cbae252e18"
LF_CATALOG_SHA = "b3ad1acc8e56654a935d07d5275d98fb82648a39cf70d28a70814e2184e419de"
RUN_DIRECTORY = LF_RUN_DIRECTORY
METHOD = "task11_howard_three_network_joint"
STAGE1 = "task11_howard_hf_all_three_stage1"
STAGE2 = "task11_howard_hf_all_three_stage2"
STAGE_BUDGET = {"阶段1上限轮次": 1500, "阶段2上限轮次": 500}
ADAMW_CONTRACT = {"lr": .001, "weight_decay": 1e-6, "betas": [.9, .999],
    "eps": 1e-8, "amsgrad": False, "maximize": False, "capturable": False,
    "differentiable": False, "foreach": None, "fused": None}
LF_SOURCE_IDENTITY = {
    "registry": "研究记录/任务11_外部对照/Howard适配独立LF预算前登记.yaml",
    "registry_sha": "0de2aa7b3e78d92d76ff90dfb2e9a1c6f5b4ba9f98ead462bd24e1f23dcc8c6c",
    "source_tar": "研究记录/任务11_外部对照/任11_Howard独立LF35普通源事前冻结_20260916T171319+0800.tar.gz",
    "source_tar_sha": "42e1c376358f915e38625d9a197ae99606f24ff4af0e51a59e176fcba3d258bc",
    "catalog": "研究记录/任务11_外部对照/任11_Howard独立LF共同70真实源目录_20260916T171319+0800.json",
    "catalog_sha": "573550239566a5f1f0e7f6889fa40f99dde83379e7c2ebf7e89eae1c375da367",
}
EPOCH_CONSUMPTION = {"HF观测点": 29593, "HF传感器点": 44775, "HF混合优化步": 15,
    "物理优化步": 1, "物理配点": 256, "LF回放点": 30720, "LF回放小批": 60,
    "LF混合梯度更新": 15, "AdamW优化步": 16, "正则显式次数": 16}
HF_BUDGET = {
    "seeds": [0, 1, 2, 3, 4], "method": METHOD, "stage1_epochs": 1500, "stage2_epochs": 500,
    "stage1_learning_rate": .001, "stage2_learning_rate": .0001,
    "all_three_trainable_from_epoch1": True, "parameter_tensors": 56,
    "stage_boundary": "latest_actual_state_keep_all56_momentum_only_lower_lr",
    "hf_train_powers": 12, "hf_validation_powers": 3, "lf_train_powers": 60,
    "lf_validation_powers": 10, "hf_ir_train_points": 29593, "hf_ir_validation_points": 7272,
    "hf_sensor_train_points": 2985, "hf_sensor_validation_points": 752,
    "hf_batch_size": 2048, "hf_batches_per_epoch": 15, "sensor_repetitions_per_epoch": 15,
    "physics_collocation": 256, "physics_steps_per_epoch": 1,
    "lf_replay_samples_per_power": 512, "lf_replay_points_per_epoch": 30720,
    "lf_microbatch_size": 512, "lf_microbatches_per_epoch": 60,
    "lf_microbatches_per_mixed_step": 4, "lf_mixed_size": 2048,
    "lf_sampling": "material_time_fixed_seed_pool_then_epoch_shuffle",
    "lf_extra_optimizer_steps": 0, "lf_supervision_max_per_seed": 61440000,
    "lf_microbatches_max_per_seed": 120000, "lf_mixed_updates_max_per_seed": 30000,
    "adamw_max_steps_per_seed": 32000, "strict_update_conditions_equal": False,
    "patience": 200, "min_delta": .0001, "validation_interval": 10,
    "phase_best_and_patience_reset": True, "global_best_across_stages": True,
    "epoch0_formal_best_allowed": False, "early_stop_no_extra_budget": True,
    "selection_metric": "macro_v1", "selection_weights": {
        "ir_rmse": 1.0, "sensor_absolute_rmse": .2, "sensor_delta_rmse": 1.0},
    "loss_weights": {"low_fidelity": 1.0, "ir": 1.0, "sensor_absolute": 5.0,
        "sensor_delta": 1.0, "pde": 1.0, "boundary": 1.0, "initial": 1.0, "interface": 1.0},
    "branch_square_sum_weights": {"low_fidelity": 1e-6, "nonlinear": 1e-6},
    "branch_square_sum_frequency_per_epoch": 16, "branch_regularization": "raw_branch_sum_squared_not_trunk_or_linear",
    "branch_coefficient_basis": "paper_PI_Burgers_example_not_universal_recommendation",
    "adamw": ADAMW_CONTRACT, "author_exact_adam_reproduction": False,
    "hf_formula": "Kelvin=295.15+250*(Fl+Fnl);not_LF_plus_residual",
    "model_kwargs": {"width": 128, "depth": 4, "latent_dim": 128, "query_chunk_size": 16384},
    "lf_final_activation": False, "linear_all_affine": True, "nonlinear_final_activation": True,
    "linear_branch_features": 24, "nonlinear_branch_features": 25,
    "scales": asdict(ModelScales()), "query_buffer_dtype": "torch.float64", "parameter_dtype": "torch.float32",
    "formal_default_dtype": "torch.float32",
    "query_response_graph": "live_LF_parameter_and_RZT_higher_order_graph",
    "epoch_consumption": EPOCH_CONSUMPTION, "session_epoch_limit": 200,
    "device": "cuda", "world_size": 1,
}
LIMITATIONS = {
    "严格更新条件一致": False, "仅architecture因果归因": False, "原文精确三网联合训练复现": False,
    "LF回放差异": "旧联合500轮每轮60个2048小批四拼8192到15混合步；本两阶段每轮60个512小批四拼2048到15混合步；最大LF监督点同为61440000，但小批/混合更新次数和参数动量排程不同",
    "正则差异": "保留原LF/NL branch全部参数平方sum各1e-6，每轮16次；系数来自PI Burgers特例；另有全参数AdamW decay，不宣称作者exact Adam复现",
    "数据限制": "仅LF TRAIN60与合法VAL10、HF TRAIN12与合法VAL3；不读取旧固定TEST或模拟TEST温度",
}
SOURCE_MEMBERS = tuple(sorted(set(MLP_HF_SOURCE_MEMBERS) | set(HOWARD_LF_SOURCE_MEMBERS) | {
    "src/sic_cu/train/task11_howard_hf_formal.py", "scripts/55_run_task11_howard_hf_formal.py",
    "tests/test_task11_howard_hf_formal.py", "src/sic_cu/train/task11_howard_hf_initialization.py",
    "tests/test_task11_howard_hf_initialization.py", "tests/test_task11_howard_composite.py",
    "pyproject.toml", "requirements.txt", "temperature_field_prediction_multifidelity_plan_v2.md",
    "src/sic_cu/data/time_dictionary.py", "src/sic_cu/train/task06_time_features.py",
}))
IDENTITY_FIELDS = ("YAML_SHA256", "TAR_SHA256", "LF_CATALOG_SHA256", "HF_DATA_CATALOG_SHA256")
ISOLATION_FLAGS = ("旧固定TEST温度读取", "模拟测试功率温度读取", "原文精确三网联合训练复现")


def source_files() -> tuple[str, ...]:
    return SOURCE_MEMBERS


def _state_content_sha(state: Mapping[str, Tensor]) -> str:
    # The frozen initializer's numpy hash is CPU-only; formal snapshots are CUDA.
    return _cpu_state_content_sha({name: value.detach().cpu() for name, value in state.items()})


def _path(value: str | Path, root: Path, *, file: bool = True) -> Path:
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else root / candidate
    resolved = candidate.resolve()
    if root not in resolved.parents:
        raise ValueError("Howard HF所有来源/输出只能在本项目内")
    for parent in (candidate, *candidate.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Howard HF来源/输出不得经过符号链接")
    if file and not resolved.is_file():
        raise ValueError("Howard HF来源必须为项目内真实普通文件")
    return resolved


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _number(value: Any, *, positive: bool = False) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and (not positive or value > 0)


class _StrictLoader(yaml.SafeLoader):
    def construct_mapping(self, node: Any, deep: bool = False) -> dict[str, Any]:
        if not isinstance(node, yaml.MappingNode):
            raise ValueError("Howard HF YAML映射结构不合法")
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if type(key) is not str or key in result:
                raise ValueError("Howard HF YAML键必须是唯一字符串，拒绝重复覆盖合同")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _load_yaml(path: Path) -> Any:
    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
    except yaml.YAMLError as error:
        raise ValueError("Howard HF固定YAML无法严格解析") from error


def _fixed_points() -> Tensor:
    rc, hc, rs, hs = .05834, .0175, .025, .012
    space = [(rc*.25, -hc, 0.), (rc*.25, -(hc+hs)/2, 0.),
             (rc*.75, -hs/2, 0.), (rc, 0., 0.), (0., 0., 1.),
             (rs/2, -hs/2, 1.), (rs, -hs, 1.), (rs, 0., 1.)]
    return torch.tensor([[r, z, t, material] for r, z, material in space
                         for t in (5.0, 100.0, 200.0)], dtype=torch.float64, device="cpu")


def _query_metadata(geometry_sha: str) -> dict[str, Any]:
    return {"schema_version": 1, "方法": "fixed_unlabeled_geometry_PHQH_24_v1",
        "geometry": "configs/geometry.yaml", "geometry_SHA256": geometry_sha,
        "几何_米": {"Rc": .05834, "Hc": .0175, "Rs": .025, "Hs": .012},
        "空间定义": ["Cu(.25*Rc,-Hc,m0)", "Cu(.25*Rc,-(Hc+Hs)/2,m0)",
            "Cu(.75*Rc,-Hs/2,m0)", "Cu(Rc,0,m0)", "SiC(0,0,m1)",
            "SiC(Rs/2,-Hs/2,m1)", "SiC(Rs,-Hs,m1)", "SiC(Rs,0,m1)"],
        "时刻_秒": [5.0, 100.0, 200.0], "顺序": "space_outer_time_inner",
        "列": ["r", "z", "t", "material"], "PH点数": 24, "QH点数": 24,
        "PH与QH相同": True, "no_temperature_selection": True, "无power列": True,
        "buffer_dtype": "torch.float64", "query_chunk_size": 16384,
        "三网适配差异": "P标量条件；linear使用24 LF响应，NL使用P加24 LF响应；固定无标签几何PH=QH，不声称原文数据域查询复现"}


def build_task11_howard_queries(*, project_root: str | Path = PROJECT_ROOT) -> tuple[Tensor, dict[str, Any]]:
    root = Path(project_root).resolve()
    geometry = _path("configs/geometry.yaml", root)
    expected = {"coordinate_system": "axisymmetric_rz", "units": "m",
        "copper": {"radius_m": .05834, "height_m": .0175},
        "silicon_carbide": {"radius_m": .025, "height_m": .012},
        "embedding": {"sic_top_z_m": 0.0, "sic_bottom_z_m": -.012, "copper_bottom_z_m": -.0175}}
    if not _same(_load_yaml(geometry), expected):
        raise ValueError("Howard PH/QH只使用冻结的实际SiC-Cu几何和固定ModelScales")
    return _fixed_points(), _query_metadata(sha256_file(geometry))


def _validate_queries(points: Tensor, metadata: Mapping[str, Any]) -> None:
    if (not isinstance(points, Tensor) or points.device.type != "cpu" or points.dtype != torch.float64
            or points.layout != torch.strided or points.requires_grad or not torch.equal(points, _fixed_points())
            or not isinstance(metadata, dict) or not _sha(metadata.get("geometry_SHA256"))
            or not _same(metadata, _query_metadata(metadata["geometry_SHA256"]))):
        raise ValueError("Howard HF PH/QH必须为固定24行CPU float64无标签几何点及完整方法元数据")


def build_task11_howard_hf_initialization(
    lf_view: Mapping[str, Any], *, seed: int, query_points: Tensor, query_metadata: Mapping[str, Any],
) -> tuple[Task11HowardComposite, torch.optim.AdamW, dict[str, Any]]:
    _validate_queries(query_points, query_metadata)
    if type(seed) is not int or seed not in range(5) or type(lf_view.get("seed")) is not int or lf_view["seed"] != seed:
        raise ValueError("Howard HF只能严格迁移本人seed的Howard LF best，不得借旧MLP或跨seed")
    model, identity = initialize_task11_howard_hf(lf_view, seed=seed,
        linear_query_points=query_points, nonlinear_query_points=query_points,
        query_metadata=query_metadata, query_chunk_size=16384)
    optimizer = build_task11_howard_hf_adamw(model, **copy.deepcopy(ADAMW_CONTRACT))
    audit_task11_howard_hf_adamw(model, optimizer, stage=STAGE1, expected_steps=0)
    return model, optimizer, identity


def _validate_model(model: Task11HowardComposite) -> list[tuple[str, Tensor]]:
    if type(model) is not Task11HowardComposite:
        raise ValueError("Howard HF必须为冻结的真实三网类")
    reference_kwargs = {**HF_BUDGET["model_kwargs"], "scales": HF_BUDGET["scales"],
                        "linear_query_points": _fixed_points().tolist(), "nonlinear_query_points": _fixed_points().tolist()}
    if (not _same(model.model_kwargs, reference_kwargs) or model.low_fidelity_subnet.final_activation is not False
            or model.nonlinear_subnet.final_activation is not True or type(model.scaler.scales) is not ModelScales
            or not _same(asdict(model.scaler.scales), HF_BUDGET["scales"])):
        raise ValueError("Howard HF数学方法/激活/尺度/query合同漂移")
    named = list(model.named_parameters())
    if (len(named) != 56 or len({id(p) for _, p in named}) != 56
            or any(p.requires_grad is not True or p.dtype != torch.float32 or p.layout != torch.strided
                   or not torch.isfinite(p).all() for _, p in named)
            or any(p.device != named[0][1].device for _, p in named)):
        raise ValueError("Howard HF全部56个独立有限float32参数从开始必须真正可训练")
    ranges = sorted((p.untyped_storage().data_ptr(), p.untyped_storage().data_ptr()+p.untyped_storage().nbytes()) for _, p in named)
    if any(a <= 0 or b <= a for a, b in ranges) or any(c < b for (_, b), (c, _) in zip(ranges, ranges[1:])):
        raise ValueError("Howard HF参数不能别名或共享底层存储")
    for name, value in model.named_buffers():
        if (name not in ("linear_query_points", "nonlinear_query_points") or value.dtype != torch.float64
                or not torch.equal(value.detach().cpu(), _fixed_points()) or value.requires_grad):
            raise ValueError("Howard HF PH/QH float64 buffer不能被.float()或换点破坏")
    return named


def audit_task11_howard_hf_adamw(
    model: Task11HowardComposite, optimizer: torch.optim.AdamW, *, stage: str, expected_steps: int,
) -> None:
    named = _validate_model(model)
    if (type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1
            or stage not in (STAGE1, STAGE2) or type(expected_steps) is not int or expected_steps < 0):
        raise ValueError("Howard HF必须使用真实单组全56 AdamW与合法阶段步数")
    group = optimizer.param_groups[0]
    contract = {**ADAMW_CONTRACT, "lr": .001 if stage == STAGE1 else .0001}
    actual = {key: list(value) if key == "betas" and isinstance(value, tuple) else value
              for key, value in group.items() if key != "params"}
    if (not _same(actual, contract) or len(group["params"]) != 56
            or any(actual is not expected for actual, (_, expected) in zip(group["params"], named))
            or set(optimizer.state) != (set(group["params"]) if expected_steps else set())):
        raise ValueError("Howard HF AdamW全部参数、唯一引用、标准默认选项与状态来源必须完整")
    for parameter in group["params"]:
        if not expected_steps:
            continue
        state = optimizer.state[parameter]
        step = state.get("step")
        if (set(state) != {"step", "exp_avg", "exp_avg_sq"} or not isinstance(step, Tensor)
                or step.dtype != torch.float32 or step.shape != torch.Size([]) or float(step) != expected_steps
                or any(not isinstance(state[key], Tensor) or state[key].shape != parameter.shape
                       or state[key].dtype != parameter.dtype or state[key].device != parameter.device
                       or not torch.isfinite(state[key]).all() for key in ("exp_avg", "exp_avg_sq"))
                or bool((state["exp_avg_sq"] < 0).any())):
            raise ValueError("Howard HF必须保存全部56真实一致AdamW step与有限同dtype/shape动量")


def transition_task11_howard_hf_stage2(model: Task11HowardComposite, optimizer: torch.optim.AdamW) -> None:
    if not optimizer.state:
        raise ValueError("Howard HF阶段边界必须已有真实全56动量，不能重新初始化")
    steps = {float(value["step"]) for value in optimizer.state.values()}
    if len(steps) != 1 or not next(iter(steps)).is_integer():
        raise ValueError("Howard HF阶段边界全56真实步数不一致")
    audit_task11_howard_hf_adamw(model, optimizer, stage=STAGE1, expected_steps=int(next(iter(steps))))
    optimizer.param_groups[0]["lr"] = .0001


def _regularization(model: Task11HowardComposite) -> Tensor:
    components = model.branch_regularization()
    if set(components) != {"low_fidelity", "nonlinear"}:
        raise ValueError("Howard原branch平方sum必须仅完整LF/NL，不能包含trunk或linear")
    return sum(HF_BUDGET["branch_square_sum_weights"][key] * value for key, value in components.items())


def _backward_step(model: Task11HowardComposite, optimizer: torch.optim.AdamW, objective: Tensor) -> None:
    if not torch.isfinite(objective):
        raise ValueError("Howard HF真实训练目标必须有限")
    objective.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("Howard HF每步全部三网56参数必须具有有限真实梯度")
    optimizer.step()


def howard_hf_mixed_step(
    model: Task11HowardComposite, optimizer: torch.optim.AdamW, hf_batch: tuple[Tensor, Tensor, Tensor],
    sensor: tuple[Tensor, ...], lf_batch: tuple[Tensor, Tensor],
) -> dict[str, Any]:
    device = next(model.parameters()).device
    x, y, weight = (value.to(device) for value in hf_batch)
    sx, sy, sd, baseline = (value.to(device) for value in sensor)
    lx, ly = (value.to(device) for value in lf_batch)
    optimizer.zero_grad(set_to_none=True)
    ir = (weight * ((model(x, fidelity="high")-y)/250.).square()).sum()/weight.sum()
    absolute, delta = _macro_sensor_training_losses(model(sx, fidelity="high"), sy, sd, baseline, sx, 250.)
    low = ((model(lx, fidelity="low")-ly)/250.).square().mean()
    regularization = _regularization(model)
    weights = HF_BUDGET["loss_weights"]
    objective = weights["ir"]*ir + weights["sensor_absolute"]*absolute + weights["sensor_delta"]*delta + weights["low_fidelity"]*low + regularization
    _backward_step(model, optimizer, objective)
    return {"HF观测点": len(x), "HF传感器点": len(sx), "LF回放点": len(lx), "正则显式次数": 1,
        "HF顶部损失": float(ir.detach()), "sensor_absolute": float(absolute.detach()),
        "sensor_delta": float(delta.detach()), "LF损失": float(low.detach()), "原branch正则": float(regularization.detach()),
        "LF_Cu点": int((lx[:, 4] == 0).sum()), "LF_SiC点": int((lx[:, 4] == 1).sum())}


def howard_hf_physics_step(
    model: Task11HowardComposite, optimizer: torch.optim.AdamW, physics: Any, collocation: Any,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    components = physics(model, collocation)
    regularization = _regularization(model)
    _backward_step(model, optimizer, components["physics_total"] + regularization)
    return {"名义物理训练分项": {key: float(value.detach()) for key, value in components.items() if isinstance(value, Tensor)},
            "正则显式次数": 1, "原branch正则": float(regularization.detach())}


def howard_hf_epoch(
    model: Task11HowardComposite, optimizer: torch.optim.AdamW, train_data: Any, sensor: tuple[Tensor, ...],
    physics: Any, replay: Any, *, seed: int, global_epoch: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    hf_loader = DataLoader(train_data, batch_size=2048, shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(7_070_000+seed*100_000+global_epoch))
    lf_loader = DataLoader(replay, batch_size=512, shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(7_080_000+seed*100_000+global_epoch))
    if len(train_data) != 29593 or len(sensor[0]) != 2985 or len(replay) != 30720 or len(hf_loader) != 15 or len(lf_loader) != 60:
        raise ValueError("Howard HF每轮必须15 HF观测批、全2985 sensor重复和60×512 LF回放")
    lf_batches = iter(lf_loader)
    model.train()
    actual = dict.fromkeys(EPOCH_CONSUMPTION, 0)
    losses, copper, sic = [], 0, 0
    for batch in hf_loader:
        low_x, low_y = zip(*(next(lf_batches) for _ in range(4)))
        values = howard_hf_mixed_step(model, optimizer, batch, sensor, (torch.cat(low_x), torch.cat(low_y)))
        losses.append(values)
        for key in ("HF观测点", "HF传感器点", "LF回放点", "正则显式次数"):
            actual[key] += values[key]
        for key, value in (("HF混合优化步", 1), ("LF回放小批", 4), ("LF混合梯度更新", 1), ("AdamW优化步", 1)):
            actual[key] += value
        copper += values["LF_Cu点"]; sic += values["LF_SiC点"]
    if next(lf_batches, None) is not None or not copper or not sic:
        raise ValueError("Howard HF LF回放必须全消费且真实覆盖Cu和SiC")
    physical = howard_hf_physics_step(model, optimizer, physics,
        sample_collocation(256, device, seed=7_090_000+seed*100_000+global_epoch))
    for key in ("物理优化步", "AdamW优化步", "正则显式次数"):
        actual[key] += 1
    actual["物理配点"] += 256
    if not _same(actual, EPOCH_CONSUMPTION):
        raise ValueError("Howard HF真实每轮监督/正则/AdamW消费算术不闭合")
    return {**physical, **actual, "混合损失均值": {key: float(np.mean([row[key] for row in losses]))
            for key in ("HF顶部损失", "sensor_absolute", "sensor_delta", "LF损失", "原branch正则")},
            "LF_Cu点": copper, "LF_SiC点": sic}


def make_task11_howard_hf_log_row(
    *, seed: int, global_epoch: int, stage: str, local_epoch: int, actual: Mapping[str, Any],
    score: float | None, validation: Mapping[str, float] | None,
) -> dict[str, Any]:
    return {"seed": seed, "epoch": global_epoch, "阶段": stage, "阶段轮次": local_epoch,
            **dict(actual), "合法验证选分_摄氏度": score, "合法验证分模态": copy.deepcopy(validation),
            **dict.fromkeys(ISOLATION_FLAGS, False)}


def replay_task11_howard_hf_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    global_best = phase_best = recent = None
    global_epoch = best_local = phase_best_epoch = 0
    best_stage = None
    stage1 = stage2 = 0
    deadline = None
    stage = STAGE1
    finished = False
    validation = None
    global_history, phase_history = [], {STAGE1: [], STAGE2: []}
    for number, row in enumerate(rows, 1):
        if finished:
            raise ValueError("Howard HF日志不能越过首次正式预算/耐心截止")
        current = row.get("阶段")
        if current == STAGE2 and stage == STAGE1:
            if deadline != stage1:
                raise ValueError("Howard HF阶段2只能继承阶段1首次合法截止的最近状态")
            stage = STAGE2
            phase_best, phase_best_epoch = None, 0
        elif current != stage or (stage == STAGE1 and deadline is not None):
            raise ValueError("Howard HF阶段顺序/首次截止不闭合")
        if stage == STAGE1: stage1 += 1
        else: stage2 += 1
        local = stage1 if stage == STAGE1 else stage2
        score, modes = row.get("合法验证选分_摄氏度"), row.get("合法验证分模态")
        if (type(row.get("epoch")) is not int or row["epoch"] != number
                or type(row.get("阶段轮次")) is not int or row["阶段轮次"] != local
                or ((local % 10 == 0) != (score is not None))
                or (score is None and modes is not None)
                or (score is not None and (not _number(score) or score < 0 or not isinstance(modes, dict)
                    or not {"顶部", "absolute_rmse_c", "delta_rmse_c"} <= modes.keys()
                    or any(not _number(value) or value < 0 for value in modes.values())))):
            raise ValueError("Howard HF仅每10实际轮固定macro有限合法验证，不能把epoch0作最佳")
        if score is not None:
            recent, validation = float(score), copy.deepcopy(modes)
            macro = (modes["顶部"]+.2*modes["absolute_rmse_c"]+modes["delta_rmse_c"])/2.2
            if not math.isclose(score, macro, rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError("Howard HF固定macro验证分数/模态权重不闭合")
            if phase_best is None or score < phase_best-.0001:
                phase_best, phase_best_epoch = float(score), local
                phase_history[stage].append(number)
            if global_best is None or score < global_best-.0001:
                global_best, global_epoch, best_stage, best_local = float(score), number, stage, local
                global_history.append(number)
        limit = 1500 if stage == STAGE1 else 500
        boundary = local == limit or (local % 10 == 0 and local-phase_best_epoch >= 200)
        if local > limit:
            raise ValueError("Howard HF正式阶段预算越界")
        if boundary:
            if stage == STAGE1: deadline = local
            else: finished = True
    return {"阶段": stage, "阶段1实际轮次": stage1, "阶段2实际轮次": stage2,
        "全局实际轮次": len(rows), "阶段1截止轮次": deadline, "阶段最佳分数": phase_best,
        "阶段最佳轮次": phase_best_epoch, "全局最佳分数": global_best, "全局最佳轮次": global_epoch,
        "全局最佳阶段": best_stage, "全局最佳阶段轮次": best_local,
        "最近合法分数": recent, "最近合法分模态": validation, "已完成": finished,
        "全局最佳历史轮次": global_history, "阶段1最佳历史轮次": phase_history[STAGE1],
        "阶段2最佳历史轮次": phase_history[STAGE2]}


def audit_task11_howard_hf_log(rows: list[dict[str, Any]], *, seed: int) -> dict[str, int]:
    if type(seed) is not int or seed not in range(5):
        raise ValueError("Howard HF日志只认可本人五seed")
    replay_task11_howard_hf_selection(rows)
    totals = dict.fromkeys(EPOCH_CONSUMPTION, 0)
    for row in rows:
        if (type(row.get("seed")) is not int or row["seed"] != seed
                or any(row.get(key) is not False for key in ISOLATION_FLAGS)
                or any(type(row.get(key)) is not int or row[key] != value for key, value in EPOCH_CONSUMPTION.items())):
            raise ValueError("Howard HF逐轮真实观测/传感器/LF/物理/正则/AdamW消费闭合失败")
        for key, value in EPOCH_CONSUMPTION.items(): totals[key] += value
    return totals


def registration_contract(
    *, source_hashes: Mapping[str, str], source_tar_sha: str, lf_catalog_sha: str,
    hf_data_catalog_sha: str, query_metadata: Mapping[str, Any], registered_at: str,
) -> dict[str, Any]:
    """Return the immutable schema; this function writes no YAML, gate or archive."""
    _validate_queries(_fixed_points(), query_metadata)
    if (set(source_hashes) != set(SOURCE_MEMBERS) or any(not _sha(value) for value in source_hashes.values())
            or any(not _sha(value) for value in (source_tar_sha, lf_catalog_sha, hf_data_catalog_sha))
            or not isinstance(registered_at, str) or not registered_at.strip()):
        raise ValueError("Howard HF登记必须明确完整普通源码集合和四来源SHA")
    return {"schema_version": 1, "登记时间": registered_at, "阶段": "fresh_howard_adapted_three_network_hf",
        "ROOT门禁标签": ROOT_TOKEN, "实施性质": "Howard数学三网独立适配；两阶段同时全56参数训练；非作者exact复现",
        **dict.fromkeys(ISOLATION_FLAGS, False), "HF训练许可": True,
        "旧LF/HF权重读取": False, "仅本人HowardLF最佳权重读取": True,
        "三网从开始全参联合更新": True, "源码冻结tar": SOURCE_TAR, "源码冻结tarSHA256": source_tar_sha,
        "LF五seed目录": LF_CATALOG, "LF五seed目录SHA256": lf_catalog_sha,
        "HF开发数据目录": HF_DATA_CATALOG, "HF开发数据目录SHA256": hf_data_catalog_sha,
        "HowardLF三身份": copy.deepcopy(LF_SOURCE_IDENTITY), "正式预算": copy.deepcopy(HF_BUDGET),
        "AdamW固定参数": copy.deepcopy(ADAMW_CONTRACT), "固定PHQH查询": copy.deepcopy(dict(query_metadata)),
        "固定协议指纹": current_protocol_fingerprints(), "限制": copy.deepcopy(LIMITATIONS),
        "源码普通成员SHA256": dict(source_hashes)}


class _RootHTMLGuard(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.raw_tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in {"script", "pre", "style", "textarea"}: self.raw_tags.append(tag)

    def handle_startendtag(self, tag: str, attrs: Any) -> None:
        # These non-void HTML tags do not become safe merely by a self-closing slash.
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag in self.raw_tags:
            del self.raw_tags[len(self.raw_tags)-1-self.raw_tags[::-1].index(tag)]

    @property
    def hidden(self) -> bool:
        # HTMLParser buffers incomplete comments/tags instead of emitting callbacks.
        return bool(self.raw_tags) or bool(self.rawdata.strip())

    def continuation_end(self) -> str | None:
        if self.raw_tags: return rf"</{self.raw_tags[-1]}\s*>"
        return r"-->" if self.rawdata.lstrip().startswith("<!--") else None


def _root_active(ledger: Path, hashes: Mapping[str, str]) -> bool:
    expected = f"{ROOT_TOKEN}; status=active; " + "; ".join(f"{key}={hashes[key]}" for key in IDENTITY_FIELDS)
    rows = []
    numbers: dict[int, int] = {}
    fence = None
    html_block, html_end = False, None
    html = _RootHTMLGuard()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if html_block:
            try: html.feed(line+"\n")
            except (AssertionError, ValueError): return False
            if html_end is None:
                if not line.strip() and not html.hidden: html_block = False
            else:
                closing = re.search(html_end, line, re.I)
                if closing:
                    html_block = html.hidden
                    if html_block: html_end = html.continuation_end()
            continue
        match = re.match(r" {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence is not None:
            if (match and match[1][0] == fence[0] and len(match[1]) >= fence[1]
                    and not match[2].strip()):
                fence = None
            continue
        if match and (match[1][0] != "`" or "`" not in match[2]):
            fence = (match[1][0], len(match[1]))
            continue
        # Hidden HTML is not a plain ROOT row; ordinary HTML blocks end at a blank line.
        raw_tag = re.match(r" {0,3}<(script|pre|style|textarea)(?:\s|>|$)", line, re.I)
        if "<!--" in line or re.match(r" {0,3}<", line):
            try: html.feed(line+"\n")
            except (AssertionError, ValueError): return False
            if raw_tag: html_end = rf"</{raw_tag[1]}\s*>"
            elif re.match(r" {0,3}<!--", line): html_end = r"-->"
            elif re.match(r" {0,3}<\?", line): html_end = r"\?>"
            elif re.match(r" {0,3}<!\[CDATA\[", line): html_end = r"\]\]>"
            elif re.match(r" {0,3}<![A-Z]", line): html_end = r">"
            elif re.match(r" {0,3}<", line): html_end = None
            else: html_end = r"-->"
            closing = re.search(html_end, line, re.I) if html_end is not None else None
            html_block = True
            if closing:
                html_block = html.hidden
                if html_block: html_end = html.continuation_end()
            continue
        if not line.startswith("|"):
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) >= 4 and re.fullmatch(r"录-\d{4}", fields[1]):
            number = int(fields[1][2:])
            numbers[number] = numbers.get(number, 0)+1
            if fields[2].startswith(ROOT_TOKEN): rows.append((number, fields[2]))
    return (len(rows) == 1 and rows[0][0] > 107 and numbers[rows[0][0]] == 1
            and rows[0][1] == expected)


def _source_unchanged(prereg: Mapping[str, Any]) -> None:
    root = Path(prereg["项目根"])
    for key, digest_key in (("登记原件", "YAML_SHA256"), ("源码归档原件", "TAR_SHA256"),
                            ("LF目录原件", "LF_CATALOG_SHA256"), ("HF数据目录原件", "HF_DATA_CATALOG_SHA256")):
        if sha256_file(_path(prereg[key], root)) != prereg[digest_key]:
            raise ValueError("Howard HF事前来源原件在会话/审查期间漂移")
    for name, digest in prereg["登记"]["源码普通成员SHA256"].items():
        if sha256_file(_path(name, root)) != digest:
            raise ValueError("Howard HF冻结普通源码在会话/审查期间漂移")
    if (not _root_active(_path(ROOT_LEDGER, root), {key: prereg[key] for key in IDENTITY_FIELDS})
            or not _same(current_protocol_fingerprints(), prereg["登记"]["固定协议指纹"])):
        raise ValueError("Howard HF活动独立ROOT门禁或原协议现场漂移")


def _audit_source_tar(archive_file: Path, hashes: Mapping[str, str], root: Path) -> None:
    try:
        with tarfile.open(archive_file, "r:gz") as archive:
            members = archive.getmembers()
            if (len(members) != len(SOURCE_MEMBERS) or len({m.name for m in members}) != len(members)
                    or {m.name for m in members} != set(SOURCE_MEMBERS) or any(not m.isfile() for m in members)):
                raise ValueError("Howard HF源码tar必须无目录/link/重复/额外/缺失的完整普通成员")
            for member in members:
                stream = archive.extractfile(member)
                if (stream is None or hashlib.sha256(stream.read()).hexdigest() != hashes[member.name]
                        or sha256_file(_path(member.name, root)) != hashes[member.name]):
                    raise ValueError("Howard HF源码tar/登记/现场普通文件SHA不一致")
    except (tarfile.TarError, OSError) as error:
        raise ValueError("Howard HF事前普通源码归档无法核验") from error


def _collect_qualified_lf(recorded: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    if (not isinstance(recorded, dict) or type(recorded.get("schema_version")) is not int
            or recorded["schema_version"] != 1 or not _same(recorded.get("HowardLF三身份"), LF_SOURCE_IDENTITY)
            or not isinstance(recorded.get("用途"), str) or not recorded["用途"]
            or type(recorded.get("审核退出码")) is not int or recorded["审核退出码"] != 0
            or not isinstance(recorded.get("审核命令"), str) or not recorded["审核命令"]
            or not _sha(recorded.get("ROOT审核时SHA256")) or not isinstance(recorded.get("限制"), dict)
            or not isinstance(recorded.get("五seed描述统计"), dict)
            or not isinstance(recorded.get("HowardLF固定源码35SHA256"), dict)
            or set(recorded["HowardLF固定源码35SHA256"]) != set(HOWARD_LF_SOURCE_MEMBERS)
            or not isinstance(recorded.get("五LF真实完整资格"), list) or len(recorded["五LF真实完整资格"]) != 5):
        raise ValueError("Howard HF只接受本人五LF完整资格catalog，不得借MLP catalog字段或资格")
    for name, digest in recorded["HowardLF固定源码35SHA256"].items():
        if not _sha(digest) or sha256_file(_path(name, root)) != digest:
            raise ValueError("Howard HF本人LF35固定源码身份不一致")
    actual = []
    for seed in range(5):
        q = qualify_task11_howard_lf_source(**LF_SOURCE_IDENTITY, seed=seed,
            output=root / LF_RUN_DIRECTORY / f"正式Howard_LF_seed{seed}", project_root=root)
        if (not isinstance(q, dict) or type(q.get("seed")) is not int or q["seed"] != seed
                or not _same(q, recorded["五LF真实完整资格"][seed])):
            raise ValueError("Howard HF五LF完整资格必须由本人冻结Howard API重推且逐字段深相同")
        actual.append(q)
    return actual


def _load_view(path: Path) -> dict[str, Any]:
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError("Howard HF本人LF/model视图普通原件无法CPU核验") from error
    if not isinstance(saved, dict):
        raise ValueError("Howard HF模型视图必须为完整字典")
    return saved


def preflight_task11_howard_hf(
    *, registry: str | Path, registry_sha: str, source_tar: str | Path, source_tar_sha: str,
    lf_catalog: str | Path, lf_catalog_sha: str, hf_data_catalog: str | Path, hf_data_catalog_sha: str,
    output: str | Path, seed: int, resume_checkpoint: str | Path | None = None,
    qualified_source_only: bool = False, project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Gate and CPU source/state checks precede CUDA, output creation and labels."""
    root = Path(project_root).resolve()
    files = {"登记原件": _path(registry, root), "源码归档原件": _path(source_tar, root),
             "LF目录原件": _path(lf_catalog, root), "HF数据目录原件": _path(hf_data_catalog, root)}
    if tuple(files.values()) != tuple(root / name for name in (REGISTRY, SOURCE_TAR, LF_CATALOG, HF_DATA_CATALOG)):
        raise ValueError("Howard HF四来源只认可独立固定正式原件，不接受替身登记或旧MLP门禁")
    hashes = dict(zip(IDENTITY_FIELDS, (registry_sha, source_tar_sha, lf_catalog_sha, hf_data_catalog_sha)))
    if (any(not _sha(value) for value in hashes.values())
            or any(sha256_file(path) != digest for path, digest in zip(files.values(), hashes.values()))
            or lf_catalog_sha != LF_CATALOG_SHA or hf_data_catalog_sha != HF_DATA_CATALOG_SHA):
        raise ValueError("Howard HF四独立SHA原件缺失/漂移或不是冻结本人LF与共用HF12/3 catalog")
    if not _root_active(_path(ROOT_LEDGER, root), hashes):
        raise ValueError("Howard HF主ROOT需唯一plain普通表行录号>107同单格四独立SHA活动门禁")
    if (type(seed) is not int or seed not in range(5) or any(os.environ.get(key, default) != default
            for key, default in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))):
        raise ValueError("Howard HF仅登记seed0..4单卡单进程")
    destination = _path(output, root, file=False)
    if destination != root / RUN_DIRECTORY / f"正式Howard_HF_seed{seed}":
        raise ValueError("Howard HF仅本人同seed固定正式输出，不覆盖他人或跨seed目录")
    if qualified_source_only:
        if resume_checkpoint is not None or not destination.is_dir():
            raise ValueError("Howard HF只读资格需实际存在本人同seed目录")
    elif resume_checkpoint is None:
        if destination.exists():
            raise ValueError("Howard HF新会话禁止覆盖已有正式seed目录")
    elif not destination.is_dir() or _path(resume_checkpoint, root) != destination / "阶段_最近.pt":
        raise ValueError("Howard HF只能从本人seed阶段_最近.pt恢复，不重新初始化或补预算")
    points, query = build_task11_howard_queries(project_root=root)
    recorded = _load_yaml(files["登记原件"])
    source_hashes = recorded.get("源码普通成员SHA256") if isinstance(recorded, dict) else None
    if not isinstance(source_hashes, dict):
        raise ValueError("Howard HF登记缺完整普通源码映射")
    expected = registration_contract(source_hashes=source_hashes, source_tar_sha=source_tar_sha,
        lf_catalog_sha=lf_catalog_sha, hf_data_catalog_sha=hf_data_catalog_sha, query_metadata=query,
        registered_at=recorded.get("登记时间"))
    if not _same(recorded, expected):
        raise ValueError("Howard HF冻结schema/数值bool/dtype/Query/来源/预算/损失/三网method合同不符")
    _audit_source_tar(files["源码归档原件"], source_hashes, root)
    lf = json.loads(files["LF目录原件"].read_text(encoding="utf-8"))
    hf = json.loads(files["HF数据目录原件"].read_text(encoding="utf-8"))
    # Structural collector decodes no temperature; the Howard qualification is independent.
    if not _same(hf, collect_task11_mlp_hf_data_catalog()):
        raise ValueError("Howard HF共用12/3开发结构目录/原件SHA/原协议现场漂移")
    qualifications = _collect_qualified_lf(lf, root)
    q = qualifications[seed]
    best_path = _path(root / LF_RUN_DIRECTORY / f"正式Howard_LF_seed{seed}" / "best.pt", root)
    if sha256_file(best_path) != q.get("最佳LF检查点SHA256"):
        raise ValueError("Howard HF本人LF best普通原件与完整资格SHA不一致")
    view = _load_view(best_path)
    model, empty_cpu_optimizer, initialization = build_task11_howard_hf_initialization(
        view, seed=seed, query_points=points, query_metadata=query)
    del empty_cpu_optimizer
    identity = {**hashes, "seed": seed, "method": METHOD,
        "LF最佳检查点SHA256": q["最佳LF检查点SHA256"],
        "LF完整资格内容SHA256": canonical_json_sha256(q),
        "LF最佳轮次": q["最佳轮次"], "LF事前三SHA": copy.deepcopy(view["任11Howard事前来源"]),
        "CPU初态完整内容SHA256": _state_content_sha(model.state_dict()),
        "LF初始子网内容SHA256": _state_content_sha(model.low_fidelity_subnet.state_dict()),
        "固定PHQH查询": query, "固定协议指纹": expected["固定协议指纹"]}
    result = {"状态": "CPU Howard HF自身门禁与五LF资格PASS；未探测CUDA或创建目录", **hashes,
        **{key: str(value) for key, value in files.items()}, "项目根": str(root), "输出": str(destination),
        "seed": seed, "身份": identity, "登记": recorded, "预算": HF_BUDGET,
        "LF来源资格": q, "LF最佳原件": str(best_path), "查询点": points, "查询元数据": query,
        "CPU初始化身份": initialization, "HF训练许可": False}
    if resume_checkpoint is not None:
        result["续跑状态"] = verify_task11_howard_hf_resume(destination, seed=seed, identity=identity, project_root=root)
    return result


def run_task11_howard_hf_formal(
    *, session_epoch_limit: int = 200, device_name: str = "cuda", **arguments: Any,
) -> dict[str, Any]:
    session_start = time.perf_counter()
    prereg = preflight_task11_howard_hf(**arguments)
    if device_name != "cuda" or type(session_epoch_limit) is not int or not 1 <= session_epoch_limit <= 200:
        raise ValueError("Howard HF正式只许真CUDA及每会话1到200实际轮次")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Howard HF正式默认dtype必须float32；拒绝物理采样Double与参数Float混用，PH/QH仍保留float64")
    if Path(prereg["项目根"]) != PROJECT_ROOT:
        raise ValueError("Howard HF隔离CPU合成项目不许可正式CUDA训练")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Howard HF必须一张真实可见CUDA，禁止静默CPU回退或伪CUDA")
    with _run_claim(Path(prereg["输出"])):
        prereg = preflight_task11_howard_hf(**arguments)
        return _run_session(prereg, session_epoch_limit, arguments, session_start)


def _selection_for_state(rows: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    selected = replay_task11_howard_hf_selection(rows)
    if stage == STAGE2 and selected["阶段"] == STAGE1:
        if selected["阶段1截止轮次"] != len(rows):
            raise ValueError("Howard HF阶段2零轮完整状态需阶段1本人真实首次截止")
        selected.update({"阶段": STAGE2, "阶段最佳分数": None, "阶段最佳轮次": 0})
    elif selected["阶段"] != stage:
        raise ValueError("Howard HF完整状态与提交历史阶段不一致")
    return selected


def _architecture(seed: int, model: Task11HowardComposite) -> dict[str, Any]:
    splits = build_power_splits()
    return {"schema_version": 1, "method": METHOD, "seed": seed,
        "model_kwargs": copy.deepcopy(model.model_kwargs), "scales": asdict(ModelScales()),
        "hf_train_powers_w": sorted(splits.hf_train), "hf_validation_powers_w": sorted(splits.hf_validation),
        "sensors_used": True, "provenance": checkpoint_provenance(role="high_fidelity_multifidelity",
            train_powers_w=splits.hf_train, validation_powers_w=splits.hf_validation),
        "原文精确三网联合训练复现": False}


def _state_metadata(
    identity: Mapping[str, Any], model: Task11HowardComposite, rows: list[dict[str, Any]], stage: str,
    *, log_sha: str, initialization: Mapping[str, Any],
) -> dict[str, Any]:
    return {"seed": identity["seed"], "身份": copy.deepcopy(dict(identity)), "method": METHOD,
        "选模重推": _selection_for_state(rows, stage), "消费累计": audit_task11_howard_hf_log(rows, seed=identity["seed"]),
        "日志SHA256": log_sha, "模型构造参数": copy.deepcopy(model.model_kwargs),
        "当前LF子网内容SHA256": _state_content_sha(model.low_fidelity_subnet.state_dict()),
        "CPU初始化身份": copy.deepcopy(dict(initialization)),
        "执行设备": str(next(model.parameters()).device), "HF训练许可": False,
        **dict.fromkeys(ISOLATION_FLAGS, False)}


def _audit_rng(state: Any, *, formal: bool) -> None:
    if not isinstance(state, dict) or set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Howard HF完整断点必须包含四类真实RNG")
    try:
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        cpu = state["torch_cpu"]
        if not isinstance(cpu, Tensor) or cpu.dtype != torch.uint8 or cpu.ndim != 1:
            raise ValueError("CPU RNG不是实际随机字节")
        torch.Generator(device="cpu").set_state(cpu.cpu())
        cuda = state["torch_cuda"]
        if formal:
            if (not isinstance(cuda, list) or len(cuda) != 1 or not isinstance(cuda[0], Tensor)
                    or cuda[0].dtype != torch.uint8 or cuda[0].shape != (16,)):
                raise ValueError("真单CUDA RNG必须是完整Philox随机字节，CPU不能伪装CUDA")
        elif cuda is not None:
            raise ValueError("CPU合成状态不得编造CUDA RNG")
    except (ValueError, RuntimeError, TypeError, KeyError, IndexError) as error:
        raise ValueError("Howard HF四RNG格式必须真实可恢复且执行设备不可伪装") from error


def _fresh_cpu_model() -> Task11HowardComposite:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.default_generator.manual_seed(0)
            return Task11HowardComposite(linear_query_points=_fixed_points(), nonlinear_query_points=_fixed_points(),
                scales=ModelScales(), **HF_BUDGET["model_kwargs"])
    finally:
        torch.set_default_dtype(previous)


def audit_task11_howard_hf_state(
    saved: Mapping[str, Any], rows: list[dict[str, Any]], *, seed: int, identity: Mapping[str, Any], formal: bool = True,
) -> dict[str, Any]:
    """CPU audit of actual tensors; formal=False only recognizes CPU fixture states."""
    if not isinstance(saved, dict) or not isinstance(saved.get("metadata"), dict):
        raise ValueError("Howard HF完整阶段原件缺结构metadata")
    metadata, stage = saved["metadata"], saved.get("stage")
    selected = _selection_for_state(rows, stage)
    local = selected["阶段1实际轮次"] if stage == STAGE1 else selected["阶段2实际轮次"]
    totals = audit_task11_howard_hf_log(rows, seed=seed)
    if (type(saved.get("training_state_schema_version")) is not int or saved["training_state_schema_version"] != 1
            or type(saved.get("epoch")) is not int or saved["epoch"] != local
            or not _same(saved.get("budget"), STAGE_BUDGET) or saved.get("scheduler_state") is not None
            or saved.get("sampler_epochs") != {} or type(metadata.get("seed")) is not int or metadata["seed"] != seed
            or not _same(metadata.get("身份"), dict(identity)) or metadata.get("method") != METHOD
            or not _same(metadata.get("选模重推"), selected) or not _same(metadata.get("消费累计"), totals)
            or not _sha(metadata.get("日志SHA256")) or metadata.get("HF训练许可") is not False
            or any(metadata.get(key) is not False for key in ISOLATION_FLAGS)
            or metadata.get("执行设备") != ("cuda:0" if formal else "cpu")
            or not isinstance(metadata.get("CPU初始化身份"), dict)
            or metadata["CPU初始化身份"].get("LF来源seed") != seed
            or metadata["CPU初始化身份"].get("HF初始化seed") != seed
            or metadata["CPU初始化身份"].get("LF来源方法") != LF_METHOD
            or type(metadata["CPU初始化身份"].get("LF来源epoch")) is not int
            or metadata["CPU初始化身份"]["LF来源epoch"] != identity.get("LF最佳轮次")
            or not _same(metadata["CPU初始化身份"].get("LF来源三SHA"), identity.get("LF事前三SHA"))
            or metadata["CPU初始化身份"].get("LF子网权重内容SHA256") != identity.get("LF初始子网内容SHA256")
            or metadata["CPU初始化身份"].get("三网初始化状态内容SHA256") != identity.get("CPU初态完整内容SHA256")
            or metadata["CPU初始化身份"].get("HF训练许可") is not False
            or not _same(metadata["CPU初始化身份"].get("调用者查询元数据"), identity.get("固定PHQH查询"))):
        raise ValueError("Howard HF完整状态的本人身份/方法/选模/预算/Query/真实设备不闭合")
    _audit_rng(saved.get("random_state"), formal=formal)
    model = _fresh_cpu_model()
    reference = model.state_dict()
    states, flags = saved.get("model_state"), saved.get("parameter_requires_grad")
    if (not isinstance(states, Mapping) or states.keys() != reference.keys()
            or any(not isinstance(value, Tensor) or value.dtype != reference[name].dtype
                   or value.shape != reference[name].shape or value.layout != torch.strided
                   or value.device.type != "cpu" or not torch.isfinite(value).all() for name, value in states.items())
            or not isinstance(flags, dict) or flags.keys() != dict(model.named_parameters()).keys()
            or any(value is not True for value in flags.values())):
        raise ValueError("Howard HF完整状态必须为全部56实际float32参数及两个原float64 buffer")
    if any(not torch.equal(states[name], _fixed_points()) for name in ("linear_query_points", "nonlinear_query_points")):
        raise ValueError("Howard HF完整断点PH/QH buffer内容漂移")
    model.load_state_dict(states, strict=True)
    if (not _same(metadata.get("模型构造参数"), model.model_kwargs)
            or metadata.get("当前LF子网内容SHA256") != _state_content_sha(model.low_fidelity_subnet.state_dict())):
        raise ValueError("Howard HF当前LF真实权重/query构造内容SHA不闭合")
    state = saved.get("optimizer_state")
    if (not isinstance(state, dict) or set(state) != {"state", "param_groups"}
            or not isinstance(state["param_groups"], list) or len(state["param_groups"]) != 1
            or not _same(state["param_groups"][0].get("params"), list(range(56)))
            or not isinstance(state["state"], dict) or any(type(key) is not int for key in state["state"])):
        raise ValueError("Howard HF真实AdamW完整状态须顺序唯一全部56参数，不能遗漏/别名")
    optimizer = torch.optim.AdamW(model.parameters(), **copy.deepcopy(ADAMW_CONTRACT))
    try:
        ranges = []
        for record in state["state"].values():
            if not isinstance(record, dict): raise ValueError("Howard HF原动量记录须真实字典")
            for key in ("exp_avg", "exp_avg_sq"):
                value = record.get(key)
                if not isinstance(value, Tensor): raise ValueError("Howard HF原动量必须是全部真实张量")
                storage = value.untyped_storage()
                ranges.append((storage.data_ptr(), storage.data_ptr()+storage.nbytes()))
        ranges.sort()
        if any(a <= 0 or b <= a for a, b in ranges) or any(c < b for (_, b), (c, _) in zip(ranges, ranges[1:])):
            raise ValueError("Howard HF全56原AdamW动量必须独立存储，不能共享/别名或切片复用")
        optimizer.load_state_dict(state)
        # Loading can cast corrupt moments: inspect the original checkpoint first.
        for identifier, (_, parameter) in zip(range(56), model.named_parameters()):
            record = state["state"].get(identifier, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in record and (not isinstance(record[key], Tensor) or record[key].dtype != parameter.dtype
                                      or record[key].shape != parameter.shape):
                    raise ValueError("Howard HF原动量dtype/shape漂移，不允许load隐式修复")
        audit_task11_howard_hf_adamw(model, optimizer, stage=stage, expected_steps=16*len(rows))
    except (RuntimeError, KeyError, TypeError, IndexError) as error:
        raise ValueError("Howard HF原AdamW状态无法严格恢复全部56真实参数") from error
    if not rows:
        if (stage != STAGE1 or metadata["当前LF子网内容SHA256"] != identity.get("LF初始子网内容SHA256")
                or metadata["CPU初始化身份"].get("三网初始化状态内容SHA256") != _state_content_sha(states)):
            raise ValueError("Howard HF epoch0必须是本人LF strict迁移的新鲜三网初态，不是正式best")
    return selected


def _audit_stage_boundary(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    if (first.get("stage") != STAGE1 or second.get("stage") != STAGE2 or second.get("epoch") != 0
            or not _same(first["model_state"], second["model_state"])
            or not _same(first["random_state"], second["random_state"])
            or not _same(first["parameter_requires_grad"], second["parameter_requires_grad"])
            or not _same(first["optimizer_state"]["state"], second["optimizer_state"]["state"])):
        raise ValueError("Howard HF阶段边界必须继承本人真实最近模型、四RNG和全部56真实动量")
    expected = copy.deepcopy(first["optimizer_state"]["param_groups"])
    expected[0]["lr"] = .0001
    if not _same(expected, second["optimizer_state"]["param_groups"]):
        raise ValueError("Howard HF阶段边界只允许单组AdamW降低lr，不能冻LF/加组/重置动量")


def _audit_hf_view(view: Any, snapshot: Mapping[str, Any], *, seed: int, identity: Mapping[str, Any], best: bool) -> None:
    meta = snapshot["metadata"]
    selected = meta["选模重推"]
    expected_epoch = selected["全局最佳轮次"] if best else selected["全局实际轮次"]
    expected_score = selected["全局最佳分数"] if best else selected["最近合法分数"]
    if (not isinstance(view, dict) or type(view.get("schema_version")) is not int or view["schema_version"] != 1
            or view.get("method") != METHOD or type(view.get("seed")) is not int or view["seed"] != seed
            or type(view.get("epoch")) is not int or view["epoch"] != expected_epoch or expected_epoch <= 0
            or not _same(view.get("model_kwargs"), meta["模型构造参数"])
            or not _same(view.get("scales"), HF_BUDGET["scales"])
            or not _same(view.get("任11HowardHF事前来源"), dict(identity))
            or not _same(view.get("model_state"), snapshot["model_state"])
            or view.get("validation_selection_score_c") != expected_score
            or not _same(view.get("validation_modalities"), selected["最近合法分模态"])
            or view.get("HF训练许可") is not False or any(view.get(key) is not False for key in ISOLATION_FLAGS)):
        raise ValueError("Howard HF best/final必须与对应真实完整状态同权重、本人来源和合法验证身份")


def audit_task11_howard_hf_artifacts(
    objects: Mapping[str, Any], hashes: Mapping[str, str], *, seed: int,
    source_identity: Mapping[str, Any], finished: bool, formal: bool = True,
) -> dict[str, Any]:
    """Replay all immutable receipts and selections; do not score any labels."""
    try:
        rows = objects["training.jsonl"]
        if not isinstance(rows, list): raise ValueError("缺少连续训练日志")
        totals = audit_task11_howard_hf_log(rows, seed=seed)
        selected = replay_task11_howard_hf_selection(rows)
        source_record = objects["事前真实来源登记.json"]
        if (not _same(source_record.get("身份"), dict(source_identity))
                or source_record.get("LF本人best严格迁移") is not True
                or source_record.get("历史HF权重/AdamW/RNG读取") is not False
                or not _same(source_record.get("限制"), LIMITATIONS)
                or any(source_record.get(key) is not False for key in ISOLATION_FLAGS)
                or not isinstance(source_record.get("LF来源完整资格"), dict)
                or type(source_record["LF来源完整资格"].get("seed")) is not int
                or source_record["LF来源完整资格"]["seed"] != seed
                or canonical_json_sha256(source_record["LF来源完整资格"]) != source_identity.get("LF完整资格内容SHA256")
                or source_record["LF来源完整资格"].get("最佳LF检查点SHA256") != source_identity.get("LF最佳检查点SHA256")
                or source_record["LF来源完整资格"].get("最佳轮次") != source_identity.get("LF最佳轮次")
                or not _number(source_record["LF来源完整资格"].get("真实累计成本秒"), positive=True)):
            raise ValueError("Howard HF事前真实来源快照需本人Howard资格、严格LF迁移和固定限制")
        for key, name in zip(IDENTITY_FIELDS, ("登记.yaml", "源码.tar.gz", "LF目录.json", "HF开发目录.json")):
            if hashes.get("事前来源快照/"+name) != source_identity.get(key):
                raise ValueError("Howard HF四份事前快照必须逐字节等于ROOT活动四SHA，不能重写快照与收据")
        if selected["已完成"] is not finished or not rows:
            raise ValueError("Howard HF暂停/已完成状态不能混用或冒充空训练")
        recent, initial = objects["阶段_最近.pt"], objects["阶段_初始.pt"]
        audit_task11_howard_hf_state(initial, [], seed=seed, identity=source_identity, formal=formal)
        audit_task11_howard_hf_state(recent, rows, seed=seed, identity=source_identity, formal=formal)
        if recent["metadata"]["日志SHA256"] != hashes["training.jsonl"]:
            raise ValueError("Howard HF完整最近状态与真实原日志SHA不一致")
        if selected["阶段1截止轮次"] is not None:
            deadline = selected["阶段1截止轮次"]
            first = objects["阶段1_训练末.pt"]
            audit_task11_howard_hf_state(first, rows[:deadline], seed=seed, identity=source_identity, formal=formal)
            if selected["阶段"] == STAGE2:
                second = objects["阶段2_初始.pt"]
                audit_task11_howard_hf_state(second, rows[:deadline], seed=seed, identity=source_identity, formal=formal)
                _audit_stage_boundary(first, second)
        for label, stage in (("阶段1", STAGE1), ("阶段2", STAGE2)):
            events = selected[label+"最佳历史轮次"]
            expected_names = {f"{label}_最佳历史_{epoch:04d}.pt" for epoch in events}
            actual_names = {name for name in objects if re.fullmatch(label+r"_最佳历史_\d{4}\.pt", name)}
            if expected_names != actual_names:
                raise ValueError("Howard HF阶段最佳不可变历史缺失/额外或不由原日志重推")
            for epoch in events:
                saved = objects[f"{label}_最佳历史_{epoch:04d}.pt"]
                audit_task11_howard_hf_state(saved, rows[:epoch], seed=seed, identity=source_identity, formal=formal)
                if saved["stage"] != stage or saved["metadata"]["选模重推"]["阶段最佳轮次"] != saved["epoch"]:
                    raise ValueError("Howard HF阶段最佳历史不是本人实际合法改进轮")
            if events and not _same(objects[label+"_最佳.pt"], objects[f"{label}_最佳历史_{events[-1]:04d}.pt"]):
                raise ValueError("Howard HF阶段best与其不可变真实历史末件不相同")
        global_events = selected["全局最佳历史轮次"]
        for pattern, suffix in ((r"全局最佳历史_\d{4}\.pt", ""), (r"全局最佳模型历史_\d{4}\.pt", "模型")):
            expected_names = {f"全局最佳{suffix}历史_{epoch:04d}.pt" for epoch in global_events}
            if {name for name in objects if re.fullmatch(pattern, name)} != expected_names:
                raise ValueError("Howard HF全局best不可变历史必须跨两阶段且不能使用epoch0")
        for epoch in global_events:
            snapshot = objects[f"全局最佳历史_{epoch:04d}.pt"]
            audit_task11_howard_hf_state(snapshot, rows[:epoch], seed=seed, identity=source_identity, formal=formal)
            if snapshot["metadata"]["选模重推"]["全局最佳轮次"] != epoch:
                raise ValueError("Howard HF全局best历史不是实际日志合法首次改进")
            _audit_hf_view(objects[f"全局最佳模型历史_{epoch:04d}.pt"], snapshot, seed=seed, identity=source_identity, best=True)
        if global_events:
            snapshot = objects["阶段_观测最佳.pt"]
            if not _same(snapshot, objects[f"全局最佳历史_{global_events[-1]:04d}.pt"]):
                raise ValueError("Howard HF当前best完整状态不等于全局最佳不可变末件")
            _audit_hf_view(objects["best.pt"], snapshot, seed=seed, identity=source_identity, best=True)
        elif any(name in objects for name in ("best.pt", "阶段_观测最佳.pt")):
            raise ValueError("Howard HF不足10轮不能把初态冒作正式best")
        receipt_names = sorted(name for name in objects if name.startswith("HF会话收据_"))
        if not receipt_names or receipt_names != [f"HF会话收据_{n:04d}.json" for n in range(1, len(receipt_names)+1)]:
            raise ValueError("Howard HF会话收据只能唯一普通0001起连续，不接受重复别名")
        previous_epoch, previous_sha, train_seconds, setup_seconds, export_seconds, total_seconds, peak = 0, None, 0., 0., 0., 0., 0
        static_prefixes = ("config_snapshot/", "事前来源快照/", "事前真实来源登记.json", "阶段_初始.pt")
        for number, name in enumerate(receipt_names, 1):
            receipt = objects[name]
            end = receipt.get("累计实际轮次")
            done = finished and number == len(receipt_names)
            prefix_rows = rows[:end] if type(end) is int else []
            prefix_selection = replay_task11_howard_hf_selection(prefix_rows)
            if (type(end) is not int or not previous_epoch < end <= len(rows)
                    or type(receipt.get("起始已提交轮次")) is not int or receipt["起始已提交轮次"] != previous_epoch
                    or type(receipt.get("本会话实际轮次")) is not int or receipt["本会话实际轮次"] != end-previous_epoch
                    or not 1 <= end-previous_epoch <= 200 or not _same(receipt.get("身份"), dict(source_identity))
                    or receipt.get("状态") != ("已完成Howard三网HF正式训练" if done else "已暂停且完整Howard HF阶段提交")
                    or receipt.get("前驱收据SHA256") != previous_sha or prefix_selection["已完成"] is not done
                    or not _same(receipt.get("消费累计"), audit_task11_howard_hf_log(prefix_rows, seed=seed))
                    or not _same(receipt.get("选模重推"), prefix_selection)
                    or any(receipt.get(key) is not False for key in ISOLATION_FLAGS)
                    or not _number(receipt.get("训练墙钟秒"), positive=True)
                    or not _number(receipt.get("加载构建恢复墙钟秒"), positive=True)
                    or not _number(receipt.get("导出核源墙钟秒"), positive=True)
                    or not _number(receipt.get("本会话总墙钟秒"), positive=True)
                    or not math.isclose(receipt["本会话总墙钟秒"], sum(receipt[k] for k in
                        ("训练墙钟秒", "加载构建恢复墙钟秒", "导出核源墙钟秒")), rel_tol=1e-12, abs_tol=1e-7)
                    or type(receipt.get("峰值真实CUDA显存字节")) is not int
                    or (receipt["峰值真实CUDA显存字节"] <= 0 if formal else receipt["峰值真实CUDA显存字节"] != 0)):
                raise ValueError("Howard HF逐会话真实轮次/状态/来源/成本分项/峰值/消费算术不闭合")
            history = receipt.get("提交历史SHA256")
            expected_history = {f"最近提交历史_{number:04d}.pt", f"已提交日志_{number:04d}.jsonl"}
            if prefix_selection["全局最佳轮次"]:
                expected_history |= {f"观测最佳提交历史_{number:04d}.pt", f"最佳视图提交历史_{number:04d}.pt"}
            originals = receipt.get("原件SHA256")
            if (not isinstance(history, dict) or set(history) != expected_history or not isinstance(originals, dict)
                    or any(not _sha(digest) or hashes.get(relative) != digest for relative, digest in history.items())
                    or not {"阶段_最近.pt", "阶段_初始.pt", "training.jsonl", "事前真实来源登记.json"} <= originals.keys()):
                raise ValueError("Howard HF提交历史与原件SHA必须全部可追溯且不可变")
            saved = objects[f"最近提交历史_{number:04d}.pt"]
            audit_task11_howard_hf_state(saved, prefix_rows, seed=seed, identity=source_identity, formal=formal)
            if (not _same(objects[f"已提交日志_{number:04d}.jsonl"], prefix_rows)
                    or originals["阶段_最近.pt"] != history[f"最近提交历史_{number:04d}.pt"]
                    or originals["training.jsonl"] != history[f"已提交日志_{number:04d}.jsonl"]
                    or saved["metadata"]["日志SHA256"] != originals["training.jsonl"]):
                raise ValueError("Howard HF提交日志前缀/真实最近状态/原件SHA历史不闭合")
            for relative, digest in originals.items():
                if not _sha(digest): raise ValueError("Howard HF原件SHA格式不合法")
                if number == len(receipt_names) or relative.startswith(static_prefixes):
                    if hashes.get(relative) != digest:
                        raise ValueError("Howard HF当前原件或静态初态/事前快照与收据SHA漂移")
            if prefix_selection["全局最佳轮次"]:
                best = objects[f"观测最佳提交历史_{number:04d}.pt"]
                event = prefix_selection["全局最佳轮次"]
                if (not _same(best, objects[f"全局最佳历史_{event:04d}.pt"])
                        or originals.get("阶段_观测最佳.pt") != history[f"观测最佳提交历史_{number:04d}.pt"]
                        or originals.get("best.pt") != history[f"最佳视图提交历史_{number:04d}.pt"]):
                    raise ValueError("Howard HF逐收据best必须对应原日志本人全局最佳事件")
                _audit_hf_view(objects[f"最佳视图提交历史_{number:04d}.pt"], best, seed=seed, identity=source_identity, best=True)
            expected_resume = None if number == 1 else objects[receipt_names[number-2]]["原件SHA256"]["阶段_最近.pt"]
            if (receipt.get("恢复源before_SHA256") != expected_resume or receipt.get("恢复源after_SHA256") != expected_resume
                    or receipt.get("恢复原件") != (None if number == 1 else "阶段_最近.pt")):
                raise ValueError("Howard HF恢复源必须本人原最近提交且before/after SHA不改、不借初始化")
            previous_epoch, previous_sha = end, hashes[name]
            train_seconds += receipt["训练墙钟秒"]; setup_seconds += receipt["加载构建恢复墙钟秒"]
            export_seconds += receipt["导出核源墙钟秒"]; total_seconds += receipt["本会话总墙钟秒"]
            peak = max(peak, receipt["峰值真实CUDA显存字节"])
        if previous_epoch != len(rows): raise ValueError("Howard HF日志超过最近真实已提交收据")
        if finished:
            if not _same(objects["阶段2_训练末.pt"], recent) or not _same(objects["阶段_训练末.pt"], recent):
                raise ValueError("Howard HF真实final末件必须包含最近全56 optimizer/RNG/query，而不是best复制")
            _audit_hf_view(objects["final.pt"], recent, seed=seed, identity=source_identity, best=False)
            metrics = objects["metrics.json"]
            if (metrics.get("method") != METHOD or type(metrics.get("seed")) is not int or metrics["seed"] != seed
                    or metrics.get("status") != "completed_current_protocol_howard_hf"
                    or not _same(metrics.get("configuration"), HF_BUDGET)
                    or not _same(metrics.get("source_identity"), dict(source_identity))
                    or not _same(metrics.get("selection"), selected) or not _same(metrics.get("consumption"), totals)
                    or metrics.get("test") is not None or any(metrics.get(key) is not False for key in ISOLATION_FLAGS)
                    or metrics.get("peak_gpu_memory_bytes") != peak
                    or metrics.get("lf_training_seconds") != source_record["LF来源完整资格"]["真实累计成本秒"]
                    or not _same(metrics.get("限制"), LIMITATIONS)
                    or any(not _number(metrics.get(key)) or not math.isclose(metrics[key], expected, rel_tol=1e-12, abs_tol=1e-7)
                        for key, expected in (("training_seconds", train_seconds), ("setup_seconds", setup_seconds),
                                              ("export_seconds", export_seconds), ("total_session_seconds", total_seconds)))):
                raise ValueError("Howard HF终态metrics必须由真实多会话训练/加载/导出成本和best/final身份重推")
        elif any(name in objects for name in ("metrics.json", "final.pt", "阶段_训练末.pt", "阶段2_训练末.pt")):
            raise ValueError("Howard HF暂停不得存在伪已完成metrics/final或预算补跑")
        return {"seed": seed, **selected, "消费累计": totals, "真实HF训练累计成本秒": train_seconds,
            "真实加载构建恢复累计秒": setup_seconds, "真实导出核源累计秒": export_seconds,
            "真实HF会话累计总秒": total_seconds, "峰值CUDA显存字节": peak,
            "会话数": len(receipt_names), "HF训练许可": False, "原文精确三网联合训练复现": False}
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError("Howard HF本人完整工件/逐会话历史审计缺失或类型不合法") from error


def _load_artifacts(directory: Path, root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    objects, hashes = {}, {}
    for path in sorted(directory.rglob("*")):
        _path(path, root, file=False)
        if path.is_symlink(): raise ValueError("Howard HF真实工件不得包含链接")
        if not path.is_file(): continue
        name = path.relative_to(directory).as_posix()
        if name.endswith(".tmp"): raise ValueError("Howard HF存在未提交临时原件，不能旁路续跑或覆盖")
        hashes[name] = sha256_file(path)
        try:
            if path.suffix == ".pt": objects[name] = torch.load(path, map_location="cpu", weights_only=False)
            elif path.suffix == ".jsonl": objects[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            elif path.suffix == ".json": objects[name] = json.loads(path.read_text(encoding="utf-8"))
            else: objects[name] = path.read_bytes()
        except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as error:
            raise ValueError("Howard HF普通真实原件无法CPU读取，不能跳过缺证据工件") from error
    return objects, hashes


def verify_task11_howard_hf_resume(
    directory: str | Path, *, seed: int, identity: Mapping[str, Any], project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    directory = _path(directory, root, file=False)
    if directory != root / RUN_DIRECTORY / f"正式Howard_HF_seed{seed}" or not directory.is_dir():
        raise ValueError("Howard HF续跑仅本人同seed固定目录最近完整提交")
    objects, hashes = _load_artifacts(directory, root)
    audited = audit_task11_howard_hf_artifacts(objects, hashes, seed=seed, source_identity=identity, finished=False)
    number = audited["会话数"]+1
    for name in (f"HF会话收据_{number:04d}.json", f"最近提交历史_{number:04d}.pt", f"已提交日志_{number:04d}.jsonl",
                 f"观测最佳提交历史_{number:04d}.pt", f"最佳视图提交历史_{number:04d}.pt"):
        if name in objects: raise ValueError("Howard HF中断/未提交原件必须保留，不能覆盖旧断点补预算")
    return objects["阶段_最近.pt"]


def qualify_task11_howard_hf_source(**arguments: Any) -> dict[str, Any]:
    if arguments.get("resume_checkpoint") is not None:
        raise ValueError("Howard HF只读已完成资格不能混用续跑")
    checked = preflight_task11_howard_hf(**arguments, qualified_source_only=True)
    objects, hashes = _load_artifacts(Path(checked["输出"]), Path(checked["项目根"]))
    result = audit_task11_howard_hf_artifacts(objects, hashes, seed=checked["seed"],
        source_identity=checked["身份"], finished=True)
    _source_unchanged(checked)
    return {**result, "状态": "CPU Howard三网HF本人真实终态与当前来源审核PASS",
        "最佳HF检查点SHA256": hashes["best.pt"], "真实末HF检查点SHA256": hashes["final.pt"],
        "真实工件SHA256": hashes, "事前来源": checked["身份"],
        "限制": LIMITATIONS, "HF训练许可": False}


@contextmanager
def _run_claim(directory: Path):
    root = PROJECT_ROOT
    parent = _path(directory.parent, root, file=False)
    if not parent.is_dir():
        raise ValueError("Howard HF正式runroot必须已存在；不能创建替身目录")
    claim = _path(parent / (directory.name+".claim"), root, file=False)
    with claim.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Howard HF本seed已被另一个真实会话占用") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _write_new_json(path: Path, payload: Mapping[str, Any], root: Path) -> None:
    _path(path, root, file=False)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, indent=2)+"\n")


def _exclusive_copy(source: Path, target: Path, root: Path) -> None:
    _path(source, root)
    _path(target, root, file=False)
    with source.open("rb") as reader, target.open("xb") as writer:
        while True:
            block = reader.read(1024*1024)
            if not block: break
            writer.write(block)
    if sha256_file(source) != sha256_file(target):
        raise ValueError("Howard HF不可变普通提交副本必须与本人原件逐字节同SHA")


def _save_stage(path: Path, model: Task11HowardComposite, optimizer: torch.optim.AdamW,
                *, stage: str, metadata: Mapping[str, Any], root: Path, new: bool = False) -> None:
    _path(path, root, file=False)
    if (new and path.exists()) or path.with_name(path.name+".tmp").exists():
        raise ValueError("Howard HF旧不可变历史或中断临时断点必须保留，不能覆盖")
    selected = metadata["选模重推"]
    local = selected["阶段1实际轮次"] if stage == STAGE1 else selected["阶段2实际轮次"]
    save_training_state(path, model, optimizer, stage=stage, epoch=local, budget=STAGE_BUDGET, metadata=metadata)


def _save_hf_view(path: Path, architecture: Mapping[str, Any], model: Task11HowardComposite,
                  metadata: Mapping[str, Any], *, best: bool, root: Path) -> None:
    _path(path, root, file=False)
    temporary = _path(path.with_name(path.name+".tmp"), root, file=False)
    if temporary.exists(): raise ValueError("Howard HF中断model临时原件不可覆盖")
    selected = metadata["选模重推"]
    payload = {**copy.deepcopy(dict(architecture)), "model_state": model.state_dict(),
        "epoch": selected["全局最佳轮次"] if best else selected["全局实际轮次"],
        "validation_selection_score_c": selected["全局最佳分数"] if best else selected["最近合法分数"],
        "validation_modalities": copy.deepcopy(selected["最近合法分模态"]),
        "任11HowardHF事前来源": copy.deepcopy(metadata["身份"]), "HF训练许可": False,
        **dict.fromkeys(ISOLATION_FLAGS, False)}
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _artifact_hashes(directory: Path, root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(directory.rglob("*")):
        _path(path, root, file=False)
        if not path.is_file(): continue
        name = path.relative_to(directory).as_posix()
        if name.endswith(".tmp"): raise ValueError("Howard HF存在中断临时原件，不能宣称提交")
        if not name.startswith("HF会话收据_"): result[name] = sha256_file(path)
    return result


def _run_session(prereg: dict[str, Any], session_limit: int, arguments: dict[str, Any],
                 session_start: float) -> dict[str, Any]:
    root, directory = Path(prereg["项目根"]), Path(prereg["输出"])
    identity, seed = prereg["身份"], prereg["seed"]
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    log = directory / "training.jsonl"
    resume_sha = previous_sha = None
    rows: list[dict[str, Any]] = []
    stage, number = STAGE1, 1
    if "续跑状态" in prereg:
        saved = prereg["续跑状态"]
        stage = saved["stage"]
        initialization = saved["metadata"]["CPU初始化身份"]
        model = _fresh_cpu_model().to(device)
        # Rebind the real optimizer only after CUDA movement. No CPU references survive.
        optimizer = torch.optim.AdamW(model.parameters(), **copy.deepcopy(ADAMW_CONTRACT))
        resume_sha = sha256_file(_path(directory / "阶段_最近.pt", root))
        load_training_state(directory / "阶段_最近.pt", model, optimizer)
        if sha256_file(directory / "阶段_最近.pt") != resume_sha:
            raise ValueError("Howard HF恢复前后本人原断点SHA漂移")
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        receipts = sorted(directory.glob("HF会话收据_[0-9][0-9][0-9][0-9].json"))
        number, previous_sha = len(receipts)+1, sha256_file(receipts[-1])
    else:
        set_seed(seed)
        view = _load_view(Path(prereg["LF最佳原件"]))
        model, validation_only_optimizer, initialization = build_task11_howard_hf_initialization(
            view, seed=seed, query_points=prereg["查询点"], query_metadata=prereg["查询元数据"])
        del validation_only_optimizer
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), **copy.deepcopy(ADAMW_CONTRACT))
    audit_task11_howard_hf_adamw(model, optimizer, stage=stage, expected_steps=16*len(rows))
    splits = build_power_splits()
    assert_no_hf_leakage({"Top": splits.hf_train, "HotCold": splits.hf_train},
                         splits.hf_validation | splits.hf_test | splits.external_sensor_test)
    # No temperature decoder runs before the active gate and full CPU Howard qualification.
    train = _ir_dataset("train", splits.hf_train)
    val = _ir_dataset("validation", splits.hf_validation)
    sensor = _sensor_tensors(device, "train", splits.hf_train)
    val_sensor = _sensor_tensors(device, "validation", splits.hf_validation)
    replay = load_sampled_points(sorted(splits.simulation_train), 512, 7_070_000+seed, sampling_mode="material_time")
    if (len(train) != 29593 or len(val) != 7272 or len(sensor[0]) != 2985 or len(val_sensor[0]) != 752
            or len(replay) != 30720 or {round(float(p), 4) for p in replay.tensors[0][:, 3]} != splits.simulation_train):
        raise ValueError("Howard HF真实TRAIN/VAL数量与合法60 LF回放池身份不等于冻结合同")
    physics = PhysicsLossComputer(load_materials(), load_resolved_boundary_conditions(), PhysicsLossWeights())
    val_loader = DataLoader(val, batch_size=2048)
    architecture = _architecture(seed, model)
    if "续跑状态" not in prereg:
        directory.mkdir(parents=True, exist_ok=False)
        with log.open("x", encoding="utf-8"):
            pass
        write_config_snapshot(directory)
        snapshot = directory / "事前来源快照"
        snapshot.mkdir()
        for key, name in (("登记原件", "登记.yaml"), ("源码归档原件", "源码.tar.gz"),
                          ("LF目录原件", "LF目录.json"), ("HF数据目录原件", "HF开发目录.json")):
            _exclusive_copy(Path(prereg[key]), snapshot / name, root)
        _write_new_json(directory / "事前真实来源登记.json", {"身份": identity,
            "LF来源完整资格": prereg["LF来源资格"], "历史HF权重/AdamW/RNG读取": False,
            "LF本人best严格迁移": True, "限制": LIMITATIONS, **dict.fromkeys(ISOLATION_FLAGS, False)}, root)
        meta = _state_metadata(identity, model, rows, stage, log_sha=sha256_file(log), initialization=initialization)
        _save_stage(directory / "阶段_初始.pt", model, optimizer, stage=stage, metadata=meta, root=root, new=True)
        _save_stage(directory / "阶段_最近.pt", model, optimizer, stage=stage, metadata=meta, root=root, new=True)
    setup_seconds = time.perf_counter()-session_start
    training_seconds, start = 0., len(rows)
    selected = replay_task11_howard_hf_selection(rows)
    for _ in range(session_limit):
        if selected["已完成"]: raise ValueError("Howard HF已完成终态不能追加预算")
        if stage == STAGE1 and selected["阶段1截止轮次"] is not None:
            transition_task11_howard_hf_stage2(model, optimizer)
            stage = STAGE2
            meta = _state_metadata(identity, model, rows, stage, log_sha=sha256_file(log), initialization=initialization)
            _save_stage(directory / "阶段2_初始.pt", model, optimizer, stage=stage, metadata=meta, root=root, new=True)
        clock = time.perf_counter()
        global_epoch = len(rows)+1
        local = selected["阶段1实际轮次"]+1 if stage == STAGE1 else selected["阶段2实际轮次"]+1
        actual = howard_hf_epoch(model, optimizer, train, sensor, physics, replay, seed=seed, global_epoch=global_epoch)
        score = validation = None
        if local % 10 == 0:
            score, validation = _validation_selection(model, val_loader, val_sensor, device, HF_BUDGET["selection_weights"])
        training_seconds += time.perf_counter()-clock
        row = make_task11_howard_hf_log_row(seed=seed, global_epoch=global_epoch, stage=stage,
            local_epoch=local, actual=actual, score=score, validation=validation)
        rows.append(row)
        selected = replay_task11_howard_hf_selection(rows)
        audit_task11_howard_hf_adamw(model, optimizer, stage=stage, expected_steps=16*len(rows))
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False)+"\n")
            handle.flush()
            os.fsync(handle.fileno())
        meta = _state_metadata(identity, model, rows, stage, log_sha=sha256_file(log), initialization=initialization)
        _save_stage(directory / "阶段_最近.pt", model, optimizer, stage=stage, metadata=meta, root=root)
        label = "阶段1" if stage == STAGE1 else "阶段2"
        if selected[label+"最佳历史轮次"] and selected[label+"最佳历史轮次"][-1] == global_epoch:
            _save_stage(directory / (label+"_最佳.pt"), model, optimizer, stage=stage, metadata=meta, root=root)
            _exclusive_copy(directory / (label+"_最佳.pt"), directory / f"{label}_最佳历史_{global_epoch:04d}.pt", root)
        if selected["全局最佳轮次"] == global_epoch:
            _save_stage(directory / "阶段_观测最佳.pt", model, optimizer, stage=stage, metadata=meta, root=root)
            _save_hf_view(directory / "best.pt", architecture, model, meta, best=True, root=root)
            _exclusive_copy(directory / "阶段_观测最佳.pt", directory / f"全局最佳历史_{global_epoch:04d}.pt", root)
            _exclusive_copy(directory / "best.pt", directory / f"全局最佳模型历史_{global_epoch:04d}.pt", root)
        if stage == STAGE1 and selected["阶段1截止轮次"] == local:
            _save_stage(directory / "阶段1_训练末.pt", model, optimizer, stage=stage, metadata=meta, root=root, new=True)
        if selected["已完成"]:
            for name in ("阶段2_训练末.pt", "阶段_训练末.pt"):
                _save_stage(directory / name, model, optimizer, stage=stage, metadata=meta, root=root, new=True)
            _save_hf_view(directory / "final.pt", architecture, model, meta, best=False, root=root)
            break
    history = {}
    for original, name in (("阶段_最近.pt", f"最近提交历史_{number:04d}.pt"),
                           ("training.jsonl", f"已提交日志_{number:04d}.jsonl"),
                           *(([("阶段_观测最佳.pt", f"观测最佳提交历史_{number:04d}.pt"),
                                ("best.pt", f"最佳视图提交历史_{number:04d}.pt")]) if selected["全局最佳轮次"] else [])):
        _exclusive_copy(directory / original, directory / name, root)
        history[name] = sha256_file(directory / name)
    # Full own LF and structural HF source qualifications are rechecked before submission.
    checked_arguments = {**arguments, "resume_checkpoint": None, "qualified_source_only": True}
    checked = preflight_task11_howard_hf(**checked_arguments)
    _source_unchanged(checked)
    total_seconds = time.perf_counter()-session_start
    export_seconds = total_seconds-setup_seconds-training_seconds
    receipt = {"身份": identity, "起始已提交轮次": start, "本会话实际轮次": len(rows)-start,
        "累计实际轮次": len(rows), "选模重推": selected, "消费累计": audit_task11_howard_hf_log(rows, seed=seed),
        "训练墙钟秒": training_seconds, "加载构建恢复墙钟秒": setup_seconds,
        "导出核源墙钟秒": export_seconds, "本会话总墙钟秒": total_seconds,
        "成本口径": "真实墙钟从首轮来源预检至最终来源核验；含运行中日志/状态/历史导出，不含最终metrics/receipt JSON序列化",
        "峰值真实CUDA显存字节": torch.cuda.max_memory_allocated(device), "前驱收据SHA256": previous_sha,
        "恢复原件": None if resume_sha is None else "阶段_最近.pt",
        "恢复源before_SHA256": resume_sha, "恢复源after_SHA256": resume_sha,
        "提交历史SHA256": history, **dict.fromkeys(ISOLATION_FLAGS, False),
        "状态": "已完成Howard三网HF正式训练" if selected["已完成"] else "已暂停且完整Howard HF阶段提交"}
    if selected["已完成"]:
        previous = [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("HF会话收据_*.json")]
        metrics = {"method": METHOD, "seed": seed, "status": "completed_current_protocol_howard_hf",
            "configuration": HF_BUDGET, "source_identity": identity, "selection": selected,
            "consumption": receipt["消费累计"], "test": None, **dict.fromkeys(ISOLATION_FLAGS, False),
            "lf_training_seconds": prereg["LF来源资格"]["真实累计成本秒"],
            "training_seconds": sum(row["训练墙钟秒"] for row in previous)+training_seconds,
            "setup_seconds": sum(row["加载构建恢复墙钟秒"] for row in previous)+setup_seconds,
            "export_seconds": sum(row["导出核源墙钟秒"] for row in previous)+export_seconds,
            "total_session_seconds": sum(row["本会话总墙钟秒"] for row in previous)+total_seconds,
            "peak_gpu_memory_bytes": max([receipt["峰值真实CUDA显存字节"], *(row["峰值真实CUDA显存字节"] for row in previous)]),
            "成本口径": receipt["成本口径"], "限制": LIMITATIONS}
        _write_new_json(directory / "metrics.json", metrics, root)
    receipt["原件SHA256"] = _artifact_hashes(directory, root)
    _write_new_json(directory / f"HF会话收据_{number:04d}.json", receipt, root)
    objects, hashes = _load_artifacts(directory, root)
    audit_task11_howard_hf_artifacts(objects, hashes, seed=seed, source_identity=identity, finished=selected["已完成"])
    return receipt
