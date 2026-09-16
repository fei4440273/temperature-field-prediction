"""Task-09 formal subset training, independent from locked Task-07 code."""

from __future__ import annotations

import json
import hashlib
import io
import math
from pathlib import Path
import re
import shutil
import time
from collections import Counter
from typing import Any, Iterator, Mapping

import numpy as np
import polars as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.processed import processed_ir_path
from sic_cu.data.splits import assert_no_hf_leakage, build_power_splits
from sic_cu.losses import PhysicsLossComputer, PhysicsLossWeights
from sic_cu.physics import sample_collocation
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.train.common import (
    CONFIG_FILES, load_training_state, physics_optimizer_step,
    save_training_state, write_config_snapshot,
)
from sic_cu.train.multifidelity import (
    _ir_dataset, _macro_sensor_training_losses, _sensor_tensors,
)
from sic_cu.train.task04_joint import (
    PROJECTION_NAMES, _hf_modalities, _lf_material_validation,
    _validation_selection, task04_lf_keep_guardrail,
)
from sic_cu.train.task07_formal import (
    _rng_restore, _score_guardrails, _simulation_replay,
)
from sic_cu.train.task07_source import (
    MANIFEST_SHA256, Task07Source, _lf_tensor_sha256, fork_task07_initialization,
)
from sic_cu.train.task09_subset_gate import (
    REGISTRATION_SHA256, audit_task09_observations, load_task09_registry,
    load_task09_sources, task09_budget, validate_task09_budget_contract,
)


TASK09_FORMAL_ROOT = PROJECT_ROOT / "研究记录/任务09_高保真数据效率/正式F3子集训练"
CORRECTION_STAGE = "task09_f3_correction"
JOINT_STAGE = "task09_f3_restricted_joint"
STAGE_BUDGET = {"HF校正上限轮次": 1500, "受限联合上限轮次": 500}
TASK09_REGISTRY = PROJECT_ROOT / "研究记录/任务09_高保真数据效率/正式HF数据效率F3先导前登记.yaml"
TASK09_FORMAL_CLI = PROJECT_ROOT / "scripts/35_run_task09_subset_formal.py"
TASK09_FORMAL_TRAINER = PROJECT_ROOT / "src/sic_cu/train/task09_subset_formal.py"
TASK09_ENERGY_AUDITOR = PROJECT_ROOT / "src/sic_cu/eval/task09_energy.py"
TASK09_ENERGY_CLI = PROJECT_ROOT / "scripts/38_audit_task09_energy.py"
TASK09_REGISTRATION_CLI = PROJECT_ROOT / "scripts/39_register_task09_subset_formal.py"


def balanced_task09_lf_groups(hf_batches: int) -> list[int]:
    """Distribute precisely 60 LF mini-batches among actual subset HF steps."""
    if not isinstance(hf_batches, int) or not 3 <= hf_batches <= 12:
        raise ValueError("任09子集正式HF更新批次只许3至12，不复制12工况15批旧循环")
    quotient, remainder = divmod(60, hf_batches)
    return [quotient + (index < remainder) for index in range(hf_batches)]


def task09_lf_microbatch_scale(group_size: int) -> float:
    """One microbatch contributes 1/4, preserving 15x mean of all LF60 batches."""
    if not isinstance(group_size, int) or not 1 <= group_size <= 20:
        raise ValueError("任09LF公平组仅接受真实1至20个2048点小批")
    return 0.25


def iter_task09_lf_groups(
    loader: DataLoader, hf_batches: int,
) -> Iterator[list[tuple[Tensor, Tensor]]]:
    """Fail before yielding if LF data is not 60 complete 2048-point batches."""
    groups = balanced_task09_lf_groups(hf_batches)
    if loader.batch_size != 2048 or len(loader) != 60 or len(loader.dataset) != 60 * 2048:
        raise ValueError("任09联合每轮须真实LF60个2048点完整batch，不允许少一行")
    iterator = iter(loader)
    exposed = 0
    for group_size in groups:
        blocks = []
        for _ in range(group_size):
            item = next(iterator)
            if (len(item) != 2 or len(item[0]) != len(item[1])
                    or len(item[0]) != 2048 or item[0].shape[-1] != 5):
                raise ValueError("任09LF逐批须真实5输入坐标、标签和2048点")
            blocks.append(item)
            exposed += len(item[0])
        yield blocks
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("任09LF60回放后有额外未登记批次")
    if exposed != 60 * 2048:
        raise ValueError("任09LF60×2048联合实际消费点数不完整")


def require_task09_cuda(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("任09正式观测候选必须真实CUDA/GPU及CUDA随机源，CPU只可来源RED/GREEN")
    return device


def validate_task09_run_identity(name: str, size: int, seed: int) -> tuple[str, int, int]:
    if seed not in range(5):
        raise ValueError("任09seed只许五个真实配对种子0到4")
    arms = load_task09_registry()
    if name not in arms or size not in (3, 6, 9):
        raise ValueError("任09正式新子集只许原登记三序列的3/6/9功率")
    return name, size, seed


def validate_task09_output(
    output_directory: str | Path, name: str, size: int, seed: int,
) -> Path:
    validate_task09_run_identity(name, size, seed)
    path = Path(output_directory).resolve()
    parent = TASK09_FORMAL_ROOT.resolve()
    required = rf"正式F3_{name}_HF{size}_seed{seed}_\d{{8}}T\d{{6}}\+0800"
    if path.parent != parent or re.fullmatch(required, path.name) is None:
        raise ValueError("任09正式训练目录必须是本任务专属canonical三序列/HF子集/本seed新目录")
    return path


def assert_task09_epoch_budget(
    actual: Mapping[str, Any], sample: Mapping[str, Any], *, joint: bool,
) -> None:
    """Reject a seemingly completed epoch missing any HF, LF or physics step."""
    hf = int(sample["ir_batches_2048"])
    expected_group = balanced_task09_lf_groups(hf) if joint else []
    sensors = sum(int(value) for value in sample["sensor_rows"].values())
    expected = {
        "HF训练观测点": int(sample["ir_rows"]),
        "HF训练传感器点": hf * sensors,
        "HF观测优化步": hf,
        "物理优化步": 1,
        "物理配点": 256,
        "LF真实回放训练点": 60 * 2048 if joint else 0,
        "LF联合回放batch": 60 if joint else 0,
        "LF60每功率真2048已核验": joint,
        "LF每HF步真实分组": expected_group,
    }
    bad = {key: (actual.get(key), value) for key, value in expected.items()
           if actual.get(key) != value}
    if (joint and (
        actual.get("LF_Cu真实回放点", 0) <= 0
        or actual.get("LF_SiC真实回放点", 0) <= 0
        or actual["LF_Cu真实回放点"] + actual["LF_SiC真实回放点"] != 60 * 2048
    )) or (not joint and (
        actual.get("LF_Cu真实回放点") != 0
        or actual.get("LF_SiC真实回放点") != 0
    )):
        bad["LF两材料"] = "联合须完整双材料；校正不得偷消费LF"
    if bad:
        raise ValueError(f"任09HF、LF60×2048及独立256名义物理优化步真实预算不完整：{bad}")


def check_task09_model(model: torch.nn.Module, source: Task07Source, stage: str) -> None:
    """Correction freezes all original LF; joint permits only four last projections."""
    named = dict(model.named_parameters())
    movable = {name for name, parameter in named.items() if parameter.requires_grad}
    expected = {name for name in named if name.startswith("correction.")}
    if stage == JOINT_STAGE:
        expected |= PROJECTION_NAMES
    if (stage not in (CORRECTION_STAGE, JOINT_STAGE) or movable != expected
            or tuple(model.state_dict()["correction.0.weight"].shape) != (128, 6)
            or any(name.startswith("response_features.") for name in model.state_dict())):
        raise ValueError("任09新HF六输入校正器和LF冻结/联合四末投影不允许漂移")
    live = model.low_fidelity_model.state_dict()
    for name, original in source.lf_state.items():
        if stage == JOINT_STAGE and f"low_fidelity_model.{name}" in PROJECTION_NAMES:
            continue
        if not torch.equal(original, live[name].detach().cpu()):
            raise ValueError("任09同seed LF冻结全张量或联合其余LF权重/buffer已被改变")
    if (stage == CORRECTION_STAGE
            and _lf_tensor_sha256(live) != source.lf_tensor_sha256):
        raise ValueError("任09HF校正阶段必须真实完整保持同seed原LF张量")


def _task09_genuine_observation_hashes(
    arms: Mapping[str, Mapping[int, tuple[float, ...]]],
    metadata: Mapping[str, Any],
) -> tuple[dict[str, dict[int, dict[str, str]]], dict[str, str]]:
    """Hash genuine permitted rows, separately for IR, Hot and Cold."""
    ir_source, ring_source = processed_ir_path("train"), PROJECT_ROOT / (
        "data/processed/sensor_ring_raw.parquet"
    )
    # Input byte SHAs are computed before opening train/validation label columns.
    whole_ir, whole_ring = sha256_file(ir_source), sha256_file(ring_source)
    ir, ring = pl.read_parquet(ir_source), pl.read_parquet(ring_source)
    if (sha256_file(ir_source) != whole_ir or sha256_file(ring_source) != whole_ring
            or set(ir["split"].unique().to_list()) != {"train", "validation"}
            or set(ring["split"].unique().to_list()) != {"train", "validation"}):
        raise ValueError("任09真实HF原IR/Hot/Cold标签源只可train/validation原件且读前读后SHA不变")

    def fingerprint(frame: pl.DataFrame, split: str, powers: list[float],
                    sensor_type: str | None, expected_rows: int) -> str:
        keys = [int(round(power * 10000)) for power in powers]
        selected = frame.filter(pl.col("split") == split)
        if sensor_type is not None:
            selected = selected.filter(pl.col("sensor_type") == sensor_type)
        selected = selected.filter((pl.col("power_w") * 10000).round(0)
                                   .cast(pl.Int64).is_in(keys))
        if selected.height != expected_rows or not expected_rows:
            raise ValueError("任09HF IR/Hot/Cold逐真标签原行数不符元数据前登记")
        raw = io.BytesIO()
        selected.write_parquet(raw, compression="uncompressed")
        return hashlib.sha256(raw.getvalue()).hexdigest()

    def per_modalities(split: str, powers: list[float], counts: Mapping[str, Any]) -> dict[str, str]:
        return {
            "IR": fingerprint(ir, split, powers, None, counts["ir_rows"]),
            "Hot": fingerprint(ring, split, powers, "hot", counts["sensor_rows"]["Hot"]),
            "Cold": fingerprint(ring, split, powers, "cold", counts["sensor_rows"]["Cold"]),
        }

    actual = {
        name: {size: per_modalities("train", list(powers), metadata["arms"][name][size])
               for size, powers in levels.items() if size in (3, 6, 9)}
        for name, levels in arms.items()
    }
    validation = per_modalities("validation", metadata["validation"]["powers_w"],
                                metadata["validation"])
    if sha256_file(ir_source) != whole_ir or sha256_file(ring_source) != whole_ring:
        raise ValueError("任09IR/Hot/Cold原标签SHA前后发生变化，正式登记失效")
    return actual, validation


def expected_task09_registration() -> dict[str, Any]:
    """Build deterministic pre-run certificate from frozen source and metadata only."""
    from sic_cu.eval.task09_energy import TASK07_BEST_ENERGY

    validate_task09_budget_contract()
    arms = load_task09_registry()
    metadata = audit_task09_observations(arms)
    sources = load_task09_sources()
    splits = build_power_splits()
    real_sources, real_validation = _task09_genuine_observation_hashes(arms, metadata)
    frozen_energy = {
        seed: sha256_file(PROJECT_ROOT / "研究记录/任务07_正式五种子重训"
                          / folder / "审计工件SHA256.json")
        for seed, (folder, _) in sorted(TASK07_BEST_ENERGY.items())
    }
    if any(frozen_energy[seed] != bound_sha for seed, (_, bound_sha)
           in TASK07_BEST_ENERGY.items()):
        raise ValueError("任09旧十二HF乙五seed能源原工件SHA漂移，禁止重新登记旧原数")
    return {
        "schema_version": 1,
        "任务": "任09独立F3子集真实CUDA训练",
        "原覆盖序列前登记SHA256": REGISTRATION_SHA256,
        "V4_B0来源清单SHA256": MANIFEST_SHA256,
        "任07固定十二HF事前配置SHA256": sha256_file(
            PROJECT_ROOT / "研究记录/任务07_正式五种子重训/有效运行配置_实现修复后.yaml",
        ),
        "split原件SHA256": sha256_file(PROJECT_ROOT / "configs/splits.yaml"),
        "HF仅train_validation来源原件SHA256": {
            "experiment_ir_radial.parquet": sha256_file(processed_ir_path("train")),
            "sensor_ring_raw.parquet": sha256_file(
                PROJECT_ROOT / "data/processed/sensor_ring_raw.parquet",
            ),
        },
        "七份运营配置SHA256": {
            str(name): sha256_file(PROJECT_ROOT / name) for name in CONFIG_FILES
        },
        "五seed原LF配对源": {seed: {
            "原LF检查点SHA256": src.lf_checkpoint_sha256,
            "原LF整张量SHA256": src.lf_tensor_sha256,
            "历史HF架构只读SHA256": src.hf_checkpoint_sha256,
            "来源元数据SHA256": src.source_metadata_sha256,
        } for seed, src in sorted(sources.items())},
        "三序列HF功率": {
            name: {size: list(powers) for size, powers in levels.items()
                   if size in (3, 6, 9)}
            for name, levels in arms.items()
        },
        "三序列逐臂每轮数据与LF预算": {
            name: {size: {
                "IR真训练行": sample["ir_rows"],
                "IR真实HF批2048": sample["ir_batches_2048"],
                "Hot真训练行": sample["sensor_rows"]["Hot"],
                "Cold真训练行": sample["sensor_rows"]["Cold"],
                "LF60按HF实批公平组": task09_budget(sample)[
                    "LF60按实际HF批数公平分组_仅预算预测"
                ],
            } for size, sample in levels.items() if size in (3, 6, 9)}
            for name, levels in metadata["arms"].items()
        },
        "每臂HF真实标签原数SHA256": real_sources,
        "HF三功率完整合法验证真实标签原数SHA256": real_validation,
        "HF合法验证不缩减功率": sorted(splits.hf_validation),
        "HF固定测试功率仅身份不读温度": sorted(splits.hf_test),
        "LF真实训练功率数": len(splits.simulation_train),
        "LF每训练功率真回放点": 2048,
        "LF每个2048小批同旧07有效权重": 0.25,
        "LF全60批一轮名义损失系数": 15.0,
        "LF真实联合每HF步分组权重": {
            "HF3真实批3分组": 5.0, "HF3真实批4分组": 3.75,
            "HF不均匀批7分组9或8": [2.25, 2.0],
            "原十二HF批15分组4": 1.0,
        },
        "子集旧12HF训练轨迹严格相同": False,
        "校正每轮名义全项物理配点": 256,
        "联合每轮名义全项物理配点": 256,
        "每轮HF数据优化步": "实际HF观测batch数N，不强行15批",
        "每轮独立物理优化步": 1,
        "联合仅LF四末投影": sorted(PROJECTION_NAMES),
        "校正上限轮次": 1500,
        "联合上限轮次": 500,
        "同HF合法选分与早停": {"宏平均版本": "macro_v1",
                           "验证间隔轮次": 10, "耐心": 200, "最小改善": 0.0001},
        "先导次序": ["primary/HF3/seed0", "primary/HF6/seed0",
                  "primary/HF9/seed0", "left_center/HF3/seed0",
                  "right_center/HF3/seed0"],
        "首批逐个真实先导目标个数": 5,
        "主序列三档五seed候选数": 15,
        "五seed正式主曲线": list(range(5)),
        "额外三档序列运行数量以真实完成为准": True,
        "十二HF结果只复用已冻任07乙合法观测": True,
        "独立能源原件条件": {"六功率五时刻": 30,
                            "双阶双原数": [16, 64],
                            "原瓦数与散度原行各": 60},
        "任04独立原能源配置SHA256": sha256_file(
            PROJECT_ROOT / "研究记录/任务04_联合微调/有效运行配置.yaml",
        ),
        "任06独立原能源配置SHA256": sha256_file(
            PROJECT_ROOT / "研究记录/任务06_时间响应特征/HF三臂先导有效配置.yaml",
        ),
        "任07乙五seed合法观测S摘要SHA256": sha256_file(
            PROJECT_ROOT / (
                "研究记录/任务07_正式五种子重训/"
                "正式五种子HF合法验证逐窗明细_20260916T022125+0800/五种子观测验证摘要.json"
            ),
        ),
        "任07乙五seed观测最佳原能源清单SHA256": frozen_energy,
        "F1纯HF-only物理来源未独立关闭不得在此入口训练": True,
        "旧固定测试温度读取": False,
        "全三序列三新增HF档五seed候选数": 45,
        "全覆盖上界总轮次非承诺": 90000,
        "正式源代码原件SHA256": {
            "trainer": sha256_file(TASK09_FORMAL_TRAINER),
            "CLI": sha256_file(TASK09_FORMAL_CLI),
            "source_gate": sha256_file(
                PROJECT_ROOT / "src/sic_cu/train/task09_subset_gate.py",
            ),
            "source_cli": sha256_file(
                PROJECT_ROOT / "scripts/34_check_task09_subset_gate.py",
            ),
            "frozen_source_adapter": sha256_file(
                PROJECT_ROOT / "src/sic_cu/train/task07_source.py",
            ),
            "frozen_task07_physics": sha256_file(
                PROJECT_ROOT / "src/sic_cu/train/task07_formal.py",
            ),
            "frozen_task04_joint": sha256_file(
                PROJECT_ROOT / "src/sic_cu/train/task04_joint.py",
            ),
            "energy_auditor": sha256_file(TASK09_ENERGY_AUDITOR),
            "energy_cli": sha256_file(TASK09_ENERGY_CLI),
            "registration_generator": sha256_file(TASK09_REGISTRATION_CLI),
            "original_energy": sha256_file(
                PROJECT_ROOT / "src/sic_cu/eval/energy_v5.py",
            ),
            "original_energy_writer": sha256_file(
                PROJECT_ROOT / "scripts/21_audit_task04_energy.py",
            ),
            "original_energy_verifier": sha256_file(
                PROJECT_ROOT / "scripts/26_audit_task07_states.py",
            ),
        },
    }


def _task09_precheck_registry_bytes(path: str | Path | None, sha256: str | None) -> str:
    if path is None or sha256 is None or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("任09正式CUDA须事前登记路径和实际原件SHA256")
    candidate = Path(path).resolve()
    if candidate != TASK09_REGISTRY or not candidate.is_file():
        raise ValueError("任09正式事前登记仅允许项目内独立新原件，不借任07或旧子集登记")
    if sha256_file(candidate) != sha256:
        raise ValueError("任09正式事前登记原件SHA与命令行锁定值不一致")
    return sha256


def require_task09_registration(path: str | Path | None, sha256: str | None) -> str:
    sha256 = _task09_precheck_registry_bytes(path, sha256)
    candidate = TASK09_REGISTRY
    if load_yaml(candidate) != expected_task09_registration():
        raise ValueError("任09正式事前登记与实际三序列、源LF、模型入口SHA或预算不一致")
    return sha256


def task09_empty_consumption() -> dict[str, int]:
    return {
        "HF训练观测点": 0, "HF训练传感器点": 0,
        "LF真实回放训练点": 0, "LF联合回放batch": 0,
        "物理配点": 0, "HF观测优化步": 0, "物理优化步": 0,
    }


def task09_expected_consumption(
    sample: Mapping[str, Any], correction_epochs: int, joint_epochs: int,
) -> dict[str, int]:
    hf = int(sample["ir_batches_2048"])
    total = correction_epochs + joint_epochs
    if not 0 <= correction_epochs <= 1500 or not 0 <= joint_epochs <= 500:
        raise ValueError("任09实际校正/联合轮数超原预算1500+500")
    return {
        "HF训练观测点": int(sample["ir_rows"]) * total,
        "HF训练传感器点": hf * sum(sample["sensor_rows"].values()) * total,
        "LF真实回放训练点": 60 * 2048 * joint_epochs,
        "LF联合回放batch": 60 * joint_epochs,
        "物理配点": 256 * total,
        "HF观测优化步": hf * total,
        "物理优化步": total,
    }


def assert_task09_adamw_steps(
    optimizer: torch.optim.AdamW, named_parameters: Mapping[str, Tensor],
    *, hf_batches: int, correction_epochs: int, joint_epochs: int,
) -> dict[str, int]:
    """Inspect actual AdamW tensors; one physical step follows every HF data batch."""
    total = correction_epochs + joint_epochs
    if hf_batches not in range(3, 13) or total < 0:
        raise ValueError("任09AdamW动态真实HF批次或轮次无效")
    correction_expected = (hf_batches + 1) * total
    projection_expected = (hf_batches + 1) * joint_epochs
    checked = {}
    for name, parameter in named_parameters.items():
        if name.startswith("correction."):
            expected = correction_expected
        elif name in PROJECTION_NAMES or name.startswith("low_fidelity_model."):
            expected = projection_expected
        else:
            raise ValueError("任09AdamW不准优化来源之外的其它HF/LF参数")
        state = optimizer.state.get(parameter, {})
        if not state:
            actual = 0
        else:
            step = state.get("step")
            if step is None or not math.isfinite(float(step)):
                raise ValueError(f"任09{name}缺真实AdamW可核step张量")
            actual = int(step)
        if actual != expected:
            raise ValueError(f"任09{name}真实AdamW step={actual}，期望{expected}，不可冒已训练轮次")
        checked[name] = actual
    if not checked or not any(name.startswith("correction.") for name in checked):
        raise ValueError("任09正式HF新校正器AdamW参数和真实step缺失")
    return checked


def task09_phase_metadata(
    source: Task07Source, sample: Mapping[str, Any], *,
    stage: str, correction_epochs: int, joint_epochs: int,
    registry_sha256: str, consumption: Mapping[str, int],
    correction_completed: int | None = None,
    initial_score: float | None = None, best_score: float | None = None,
    best_epoch: int = 0, best_stage: str | None = None,
    physical_score: float | None = None, physical_epoch: int = 0,
    physical_stage: str | None = None,
    phase_best_score: float | None = None, phase_best_epoch: int = 0,
    lf_reference: Mapping[str, Any] | None = None,
    lf_keep: Mapping[str, Any] | None = None,
    correction_terminal_sha256: str | None = None,
    current_lf_sha256: str | None = None,
) -> dict[str, Any]:
    total = correction_epochs + joint_epochs
    if (stage not in (CORRECTION_STAGE, JOINT_STAGE)
            or consumption != task09_expected_consumption(sample, correction_epochs, joint_epochs)
            or source.seed not in range(5)
            or not re.fullmatch(r"[0-9a-f]{64}", registry_sha256)):
        raise ValueError("任09完整阶段须同seed、真实消费与正式事前SHA原件")
    return {
        "任09训练种子": source.seed, "任务臂": "F3",
        "所属功率序列": sample.get("arm", "三序列由运行入口绑定"),
        "当前真实HF子集功率": list(sample["powers_w"]),
        "原HF合法验证功率": list(source.hf_validation_powers_w),
        "原LF全部训练功率数": len(source.lf_train_powers_w),
        "原LF合法验证功率数": len(source.lf_validation_powers_w),
        "V4_B0来源清单SHA256": MANIFEST_SHA256,
        "原覆盖序列SHA256": REGISTRATION_SHA256,
        "正式F3事前登记SHA256": registry_sha256,
        "本seed原LF检查点SHA256": source.lf_checkpoint_sha256,
        "本seed历史HF架构只读SHA256": source.hf_checkpoint_sha256,
        "本seed来源身份SHA256": source.source_metadata_sha256,
        "实际本seed原LF整张量SHA256": source.lf_tensor_sha256,
        "当前LF整张量SHA256": current_lf_sha256 or source.lf_tensor_sha256,
        "当前阶段": stage,
        "校正实际轮次": correction_epochs,
        "联合实际轮次": joint_epochs,
        "完整全局实际轮次": total,
        "校正实际截止轮次": correction_completed,
        "已提交真实校正末态SHA256": correction_terminal_sha256,
        "初始合法HF选分_摄氏度": initial_score,
        "合法HF观测最佳选分_摄氏度": best_score,
        "观测最佳全局轮次": best_epoch,
        "观测最佳阶段": best_stage or CORRECTION_STAGE,
        "独立名义全项物理最佳损失": physical_score,
        "物理最佳全局轮次": physical_epoch,
        "物理最佳阶段": physical_stage or CORRECTION_STAGE,
        "本阶段合法最佳选分": phase_best_score,
        "本阶段早停最佳轮次": phase_best_epoch,
        "LF初始十验证功率两材料节点及体积": dict(lf_reference or {}),
        "LF两材料真实节点与体积5%资格": dict(lf_keep or {}),
        "累计实际消耗": dict(consumption),
        "真实HF校正器AdamW期望step": (sample["ir_batches_2048"] + 1) * total,
        "真实LF四末投影AdamW期望step": (sample["ir_batches_2048"] + 1) * joint_epochs,
        "阶段预算上限": dict(STAGE_BUDGET),
        "仅CPU来源报告不是正式成绩": True,
        "本阶段不是CPU正式成绩": True,
        "旧固定测试温度读取": False,
    }


def check_task09_resume_payload(
    payload: Mapping[str, Any], source: Task07Source,
    sample: Mapping[str, Any], *, registry_sha256: str,
) -> None:
    """Reject model-only historical best and altered source before any mutable restore."""
    necessary = {"training_state_schema_version", "stage", "epoch", "model_state",
                 "optimizer_state", "parameter_requires_grad", "random_state",
                 "budget", "metadata"}
    if not necessary <= payload.keys():
        raise ValueError("任09续跑须模型/AdamW/四RNG完整状态；旧best视图绝不续训")
    meta = payload["metadata"]
    stage = payload["stage"]
    corr, joint = meta["校正实际轮次"], meta["联合实际轮次"]
    state = payload["random_state"]
    if (payload["training_state_schema_version"] != 1
            or stage not in (CORRECTION_STAGE, JOINT_STAGE)
            or payload["epoch"] != (corr if stage == CORRECTION_STAGE else joint)
            or meta["当前阶段"] != stage
            or meta["完整全局实际轮次"] != corr + joint
            or meta["任09训练种子"] != source.seed
            or meta["正式F3事前登记SHA256"] != registry_sha256
            or meta["V4_B0来源清单SHA256"] != MANIFEST_SHA256
            or meta["本seed原LF检查点SHA256"] != source.lf_checkpoint_sha256
            or meta["实际本seed原LF整张量SHA256"] != source.lf_tensor_sha256
            or meta["本seed来源身份SHA256"] != source.source_metadata_sha256
            or meta["当前真实HF子集功率"] != list(sample["powers_w"])
            or meta["原HF合法验证功率"] != list(source.hf_validation_powers_w)
            or meta["累计实际消耗"] != task09_expected_consumption(sample, corr, joint)
            or payload["budget"] != STAGE_BUDGET
            or set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}
            or not isinstance(state["python"], tuple)
            or not isinstance(state["numpy"], tuple)
            or not isinstance(state["torch_cpu"], torch.Tensor)
            or state["torch_cpu"].dtype != torch.uint8
            or state["torch_cpu"].device.type != "cpu"
            or not isinstance(state["torch_cuda"], (tuple, list))
            or len(state["torch_cuda"]) != 1
            or any(not isinstance(cuda, torch.Tensor)
                   or cuda.dtype != torch.uint8 or cuda.device.type != "cpu"
                   or tuple(cuda.shape) != (16,)
                   for cuda in state["torch_cuda"])):
        raise ValueError("任09完整状态来源SHA、子集、训练预算、真实AdamW/四RNG或CUDA身份不合格")
    all_params = payload["parameter_requires_grad"]
    movable = {name for name, enabled in all_params.items() if enabled}
    expected = {name for name in all_params if name.startswith("correction.")}
    if stage == JOINT_STAGE:
        expected |= PROJECTION_NAMES
    if movable != expected:
        raise ValueError("任09续跑LF仅四末投影，历史HF/全部LF可训练状态禁止入场")
    opt = payload["optimizer_state"]
    if not isinstance(opt, Mapping) or "state" not in opt or "param_groups" not in opt:
        raise ValueError("任09续跑不得缺真实已步进HF校正器AdamW状态")
    groups = opt["param_groups"]
    if len(groups) != (2 if stage == JOINT_STAGE else 1):
        raise ValueError("任09联合AdamW仅新增LF四末投影组")
    corrections = [name for name in all_params if name.startswith("correction.")]
    projections = [name for name in all_params if name in PROJECTION_NAMES]
    if (len(groups[0]["params"]) != len(corrections)
            or (stage == JOINT_STAGE and len(groups[1]["params"]) != len(projections))
            or len(projections) != len(PROJECTION_NAMES)):
        raise ValueError("任09续跑真实AdamW参数组不得代替新HF六输入校正和LF四末投影")
    steps = [(groups[0]["params"], (sample["ir_batches_2048"] + 1) * (corr + joint))]
    if stage == JOINT_STAGE:
        steps.append((groups[1]["params"], (sample["ir_batches_2048"] + 1) * joint))
    parameter_ids = [index for parameters, _ in steps for index in parameters]
    if len(parameter_ids) != len(set(parameter_ids)) or set(opt["state"]) - set(parameter_ids):
        raise ValueError("任09续跑AdamW混入重复或模型外未知参数")
    for parameters, expected_steps in steps:
        for index in parameters:
            state_step = opt["state"].get(index, {}).get("step", 0)
            actual_step = float(state_step)
            if (not math.isfinite(actual_step) or not actual_step.is_integer()
                    or int(actual_step) != expected_steps):
                raise ValueError("任09每一张量真实AdamW step须与HF真批加物理一步和阶段轮次一致")
    live = payload["model_state"]
    prefix = "low_fidelity_model."
    actual_lf = {name[len(prefix):]: value for name, value in live.items()
                 if name.startswith(prefix)}
    if (set(actual_lf) != set(source.lf_state)
            or _lf_tensor_sha256(actual_lf) != meta["当前LF整张量SHA256"]
            or tuple(live["correction.0.weight"].shape) != (128, 6)
            or any(name.startswith("response_features.") for name in live)):
        raise ValueError("任09续跑model_state LF实际整张量SHA或新HF旧六列来源不合格")
    changed = {prefix + name for name, original in source.lf_state.items()
               if not torch.equal(original, actual_lf[name].detach().cpu())}
    if changed - (PROJECTION_NAMES if stage == JOINT_STAGE else set()):
        raise ValueError("任09续跑LF原来源实际张量改变；仅真实联合四末投影可不同")


def switch_task09_restricted_joint(
    model: torch.nn.Module, optimizer: torch.optim.AdamW,
    source: Task07Source, *, loading_committed_state: bool = False,
) -> None:
    if len(optimizer.param_groups) != 1:
        raise ValueError("任09校正到受限联合只许同一个真实HF AdamW一次切换参数组")
    check_task09_model(model, source, CORRECTION_STAGE)
    if not loading_committed_state and not optimizer.state:
        raise ValueError("任09联合不得从空HF AdamW替代真实校正训练动量")
    optimizer.param_groups[0]["lr"] = 0.0001
    model.freeze_low_fidelity(True)
    named = dict(model.named_parameters())
    for name in PROJECTION_NAMES:
        named[name].requires_grad_(True)
    optimizer.add_param_group({
        "params": [parameter for name, parameter in named.items()
                   if name in PROJECTION_NAMES],
        "lr": 0.00001,
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
    })
    check_task09_model(model, source, JOINT_STAGE)


def task09_model_view(
    model: torch.nn.Module, source: Task07Source, sample: Mapping[str, Any],
    *, registry_sha256: str, score: float, correction_epoch: int,
    joint_epoch: int, validation: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Only live new-E0 tensors; this view is never a resumable training state."""
    stage = JOINT_STAGE if joint_epoch else CORRECTION_STAGE
    check_task09_model(model, source, stage)
    if not math.isfinite(score) or not re.fullmatch(r"[0-9a-f]{64}", registry_sha256):
        raise ValueError("任09新最佳模型视图需合法HF选分和事前源SHA")
    return {
        "method": "multifidelity_correction", "seed": source.seed,
        "epoch": correction_epoch + joint_epoch,
        "low_fidelity_method": "deeponet_pinn",
        "low_fidelity_model_kwargs": dict(source.lf_model_kwargs),
        "correction_model_kwargs": dict(source.correction_model_kwargs),
        "scales": dict(source.scales),
        "model_state": {name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()},
        "hf_train_powers_w": list(sample["powers_w"]),
        "hf_validation_powers_w": list(source.hf_validation_powers_w),
        "sensors_used": True,
        "validation_selection_score_c": float(score),
        "validation_sensor": dict(validation or {}),
        "任09真实新HF观测来源": {
            "本seed原LF检查点SHA256": source.lf_checkpoint_sha256,
            "本seed原LF整张量SHA256": source.lf_tensor_sha256,
            "历史HF仅架构只读SHA256": source.hf_checkpoint_sha256,
            "正式F3事前登记SHA256": registry_sha256,
            "HF功率序列": list(sample["powers_w"]),
            "校正实际轮次": correction_epoch, "联合实际轮次": joint_epoch,
            "旧固定测试温度读取": False,
        },
        "任09可作训练续跑输入": False,
        "原12HF最佳校正器张量加载": False,
    }


def check_task09_best_view(
    view: Mapping[str, Any], full: Mapping[str, Any],
    source: Task07Source, sample: Mapping[str, Any], *, registry_sha256: str,
) -> None:
    """No historical HF best view may be substituted for this run's own best."""
    meta, identity = full["metadata"], view.get("任09真实新HF观测来源", {})
    model_state, recorded = full["model_state"], view.get("model_state", {})
    score = view.get("validation_selection_score_c")
    if (view.get("method") != "multifidelity_correction"
            or view.get("任09可作训练续跑输入") is not False
            or view.get("原12HF最佳校正器张量加载") is not False
            or view.get("seed") != source.seed
            or view.get("hf_train_powers_w") != list(sample["powers_w"])
            or view.get("hf_validation_powers_w") != list(source.hf_validation_powers_w)
            or view.get("epoch") != meta["观测最佳全局轮次"]
            or identity.get("正式F3事前登记SHA256") != registry_sha256
            or identity.get("本seed原LF整张量SHA256") != source.lf_tensor_sha256
            or identity.get("本seed原LF检查点SHA256") != source.lf_checkpoint_sha256
            or identity.get("HF功率序列") != list(sample["powers_w"])
            or identity.get("旧固定测试温度读取") is not False
            or type(score) not in (float, int)
            or not math.isfinite(score)
            or not math.isclose(score, meta["合法HF观测最佳选分_摄氏度"],
                                rel_tol=0.0, abs_tol=1e-4)
            or set(recorded) != set(model_state)
            or any(not isinstance(recorded[name], torch.Tensor)
                   or not torch.equal(recorded[name], value)
                   for name, value in model_state.items())):
        raise ValueError("任09best视图必须逐HF/LF张量、合法选分与本seed观测最佳完整状态一致")


def _task09_archive_source(output: Path) -> dict[str, str]:
    """Original bytes and SHAs are archived before a training-label read."""
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    inputs = {
        "task09_subset_formal.py": TASK09_FORMAL_TRAINER,
        "35_run_task09_subset_formal.py": TASK09_FORMAL_CLI,
        "task09_energy.py": TASK09_ENERGY_AUDITOR,
        "38_audit_task09_energy.py": TASK09_ENERGY_CLI,
        "39_register_task09_subset_formal.py": TASK09_REGISTRATION_CLI,
        "energy_v5.py": PROJECT_ROOT / "src/sic_cu/eval/energy_v5.py",
        "21_audit_task04_energy.py": PROJECT_ROOT / "scripts/21_audit_task04_energy.py",
        "26_audit_task07_states.py": PROJECT_ROOT / "scripts/26_audit_task07_states.py",
        "task09_subset_gate.py": PROJECT_ROOT / "src/sic_cu/train/task09_subset_gate.py",
        "task07_source.py": PROJECT_ROOT / "src/sic_cu/train/task07_source.py",
        "task07_formal.py": PROJECT_ROOT / "src/sic_cu/train/task07_formal.py",
        "task04_joint.py": PROJECT_ROOT / "src/sic_cu/train/task04_joint.py",
        "experiment_ir_radial.parquet": processed_ir_path("train"),
        "sensor_ring_raw.parquet": PROJECT_ROOT / "data/processed/sensor_ring_raw.parquet",
        "任09原额外序列登记.yaml": PROJECT_ROOT / (
            "研究记录/任务09_高保真数据效率/额外功率覆盖序列前登记.yaml"
        ),
        "任09正式事前登记.yaml": TASK09_REGISTRY,
        "V4_B0清单.yaml": PROJECT_ROOT / "reports/development_v4/baseline_manifest.yaml",
    }
    inputs.update({Path(name).name: PROJECT_ROOT / name for name in CONFIG_FILES})
    hashes = {}
    for name, source in inputs.items():
        destination = snapshot / name
        original = sha256_file(source)
        shutil.copyfile(source, destination)
        if original != sha256_file(destination):
            raise ValueError("任09正式源码和物理配置事前快照原件SHA不一致，不读训练标签")
        hashes[name] = original
    (output / "源码与登记事前快照SHA256.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return hashes


def _task09_state(
    output: Path, filename: str, model: torch.nn.Module,
    optimizer: torch.optim.AdamW, source: Task07Source,
    sample: Mapping[str, Any], metadata: Mapping[str, Any],
) -> Path:
    stage = metadata["当前阶段"]
    check_task09_model(model, source, stage)
    if (model.correction[0].weight.device.type != "cuda"
            or not torch.cuda.is_available()
            or not torch.cuda.get_rng_state_all()
            or metadata["当前LF整张量SHA256"] !=
            _lf_tensor_sha256(model.low_fidelity_model.state_dict())):
        raise ValueError("任09正式状态仅可从真实CUDA模型/四随机源及当前LF整张量提交")
    assert_task09_adamw_steps(
        optimizer, dict(model.named_parameters()),
        hf_batches=sample["ir_batches_2048"],
        correction_epochs=metadata["校正实际轮次"],
        joint_epochs=metadata["联合实际轮次"],
    )
    epoch = (metadata["校正实际轮次"] if stage == CORRECTION_STAGE else
             metadata["联合实际轮次"])
    return save_training_state(
        output / filename, model, optimizer, stage=stage, epoch=epoch,
        budget=STAGE_BUDGET, metadata=metadata,
    )


def _task09_view(
    output: Path, model: torch.nn.Module, source: Task07Source,
    sample: Mapping[str, Any], *, registry_sha256: str,
    score: float, correction_epoch: int, joint_epoch: int,
    validation: Mapping[str, float] | None,
) -> None:
    view = task09_model_view(
        model, source, sample, registry_sha256=registry_sha256,
        score=score, correction_epoch=correction_epoch, joint_epoch=joint_epoch,
        validation=validation,
    )
    path = output / "best.pt"
    temp = path.with_name(path.name + ".tmp")
    torch.save(view, temp)
    temp.replace(path)


def validate_task09_logged_history(
    rows: list[Mapping[str, Any]], meta: Mapping[str, Any],
    source: Task07Source, sample: Mapping[str, Any],
) -> None:
    """Cross-check real per-epoch receipts against *ordered* phase boundaries."""
    global_epoch = meta["完整全局实际轮次"]
    if len(rows) != global_epoch:
        raise ValueError("任09训练日志缺行或有未经提交轮次，不能借旧最近阶段续跑")
    correction_epochs = meta["校正实际轮次"]
    cumulative = task09_empty_consumption()
    for index, row in enumerate(rows, 1):
        joint = index > correction_epochs
        stage = JOINT_STAGE if joint else CORRECTION_STAGE
        local_epoch = index - correction_epochs if joint else index
        if (row["epoch"] != index or row["运行种子"] != source.seed
                or row["真实HF子集功率"] != list(sample["powers_w"])
                or row["训练阶段"] != stage
                or row["阶段实际轮次"] != local_epoch):
            raise ValueError("任09训练历史校正与联合阶段轮次顺序、跨seed/子集不合格")
        assert_task09_epoch_budget(row, sample, joint=joint)
        for key in cumulative:
            cumulative[key] += row[key]
        if cumulative != row["累计实际消耗"]:
            raise ValueError("任09每轮HF/LF/物理累计日志与真预算不一致")
    if cumulative != meta["累计实际消耗"]:
        raise ValueError("任09源阶段真消费与提交最近完整状态不一致")


def _task09_checkpoint_preflight(
    output: Path, payload: Mapping[str, Any], source: Task07Source,
    sample: Mapping[str, Any], *, registry_sha256: str,
) -> None:
    check_task09_resume_payload(payload, source, sample,
                                registry_sha256=registry_sha256)
    archive = output / "源码与登记事前快照SHA256.json"
    records = output / "training.jsonl"
    if not archive.is_file() or not records.is_file():
        raise ValueError("任09旧会话缺事前源码快照或真实训练日志，不可续跑")
    expected_sha = json.loads(archive.read_text(encoding="utf-8"))
    if any(sha256_file(output / "source_snapshot" / name) != sha
           for name, sha in expected_sha.items()):
        raise ValueError("任09旧会话登记、源码与配置原件归档SHA被改变")
    registration = expected_task09_registration()
    bound = {
        "任09正式事前登记.yaml": registry_sha256,
        "V4_B0清单.yaml": MANIFEST_SHA256,
        "任09原额外序列登记.yaml": REGISTRATION_SHA256,
        **registration["HF仅train_validation来源原件SHA256"],
        **{Path(key).name: value for key, value in
           registration["七份运营配置SHA256"].items()},
        **{
            "task09_subset_formal.py": registration["正式源代码原件SHA256"]["trainer"],
            "35_run_task09_subset_formal.py": registration["正式源代码原件SHA256"]["CLI"],
            "task09_energy.py": registration["正式源代码原件SHA256"]["energy_auditor"],
            "38_audit_task09_energy.py": registration["正式源代码原件SHA256"]["energy_cli"],
            "39_register_task09_subset_formal.py": registration[
                "正式源代码原件SHA256"]["registration_generator"
            ],
            "task09_subset_gate.py": registration["正式源代码原件SHA256"]["source_gate"],
            "task07_source.py": registration["正式源代码原件SHA256"]["frozen_source_adapter"],
            "task07_formal.py": registration["正式源代码原件SHA256"]["frozen_task07_physics"],
            "task04_joint.py": registration["正式源代码原件SHA256"]["frozen_task04_joint"],
            "energy_v5.py": registration["正式源代码原件SHA256"]["original_energy"],
            "21_audit_task04_energy.py": registration[
                "正式源代码原件SHA256"]["original_energy_writer"
            ],
            "26_audit_task07_states.py": registration[
                "正式源代码原件SHA256"]["original_energy_verifier"
            ],
        },
    }
    if any(expected_sha.get(name) != sha for name, sha in bound.items()):
        raise ValueError("任09事前归档必须保持原train/validation、LF源、七配置和独审源码实际SHA")
    rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    meta = payload["metadata"]
    validate_task09_logged_history(rows, meta, source, sample)
    for kind, epoch_key in (("观测", "观测最佳全局轮次"),
                            ("物理", "物理最佳全局轮次")):
        filename = f"阶段_{kind}最佳.pt"
        checkpoint = output / filename
        if not checkpoint.is_file():
            raise ValueError("任09完整两轨最佳状态原件缺失，不能凭旧best视图续跑")
        best = torch.load(checkpoint, map_location="cpu", weights_only=False)
        check_task09_resume_payload(best, source, sample,
                                    registry_sha256=registry_sha256)
        if best["metadata"]["完整全局实际轮次"] != meta[epoch_key]:
            raise ValueError("任09观测/物理最佳完整状态真实轮次与最近登记不一致")
        committed = meta.get(f"已提交旧{filename}SHA256")
        if committed and sha256_file(checkpoint) != committed:
            raise ValueError("任09观测/物理历史best原件SHA与完整阶段承诺不一致")
        if kind == "观测":
            view_path = output / "best.pt"
            if not view_path.is_file():
                raise ValueError("任09观测最佳新HF模型视图真实原件不可缺席")
            view = torch.load(view_path, map_location="cpu", weights_only=False)
            check_task09_best_view(view, best, source, sample,
                                   registry_sha256=registry_sha256)


def _task09_epoch_commit(
    output: Path, model: torch.nn.Module, optimizer: torch.optim.AdamW,
    source: Task07Source, sample: Mapping[str, Any], metadata: dict[str, Any],
    *, global_epoch: int, observed_improved: bool,
    physical_improved: bool, score: float | None,
    validation: Mapping[str, float] | None, registry_sha256: str,
) -> None:
    for kind, best_epoch in (("观测", metadata["观测最佳全局轮次"]),
                             ("物理", metadata["物理最佳全局轮次"])):
        if best_epoch < global_epoch:
            filename = f"阶段_{kind}最佳.pt"
            existing = output / filename
            if not existing.is_file():
                raise ValueError("任09历史最佳完整状态失踪，不能覆盖本轮最近阶段")
            metadata[f"已提交旧{filename}SHA256"] = sha256_file(existing)
    _task09_state(output, "阶段_最近.pt", model, optimizer, source, sample, metadata)
    if observed_improved:
        _task09_state(output, "阶段_观测最佳.pt", model, optimizer, source, sample, metadata)
        assert score is not None
        _task09_view(output, model, source, sample,
                     registry_sha256=registry_sha256, score=score,
                     correction_epoch=metadata["校正实际轮次"],
                     joint_epoch=metadata["联合实际轮次"], validation=validation)
    if physical_improved:
        _task09_state(output, "阶段_物理最佳.pt", model, optimizer, source, sample, metadata)


def run_task09_formal(
    *, name: str, size: int, seed: int, output_directory: str | Path,
    device_name: str = "cuda",
    registry_path: str | Path | None = None,
    registry_sha256: str | None = None,
    session_epoch_limit: int | None = None,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """CUDA candidate: one canonical seed/subset, true 1500+500 budget and full state."""
    device = require_task09_cuda(device_name)
    output = validate_task09_output(output_directory, name, size, seed)
    if session_epoch_limit is None or not 1 <= session_epoch_limit <= 2000:
        raise ValueError("任09正式每次CUDA会话须明示本次真实增加1至2000轮上限")
    started = time.perf_counter()
    registration = _task09_precheck_registry_bytes(registry_path, registry_sha256)
    sources = load_task09_sources()
    source = sources[seed]
    registered = load_task09_registry()
    subset = registered[name][size]
    expected_meta = audit_task09_observations(registered)["arms"][name][size]
    sample = {**expected_meta, "arm": name}
    splits = build_power_splits()
    assert_no_hf_leakage({"Top": subset, "Hot": subset, "Cold": subset},
                         splits.hf_validation | splits.hf_test | splits.external_sensor_test)
    if (tuple(source.hf_validation_powers_w) != tuple(sorted(splits.hf_validation))
            or tuple(source.lf_train_powers_w) != tuple(sorted(splits.simulation_train))
            or tuple(source.lf_validation_powers_w) != tuple(sorted(splits.simulation_validation))):
        raise ValueError("任09当前HF合法验证3/LF60+10与本seed原源身份不同")
    if resume_checkpoint is None and output.exists():
        raise FileExistsError("任09新正式会话不得覆盖已有子集seed原件")
    if resume_checkpoint is not None and not output.is_dir():
        raise ValueError("任09续跑只允许已有同seed同子集canonical新运行目录")
    if resume_checkpoint is None:
        output.mkdir(parents=True, exist_ok=False)
        _task09_archive_source(output)
        write_config_snapshot(output)
        (output / "training.jsonl").write_text("", encoding="utf-8")
    # Genuine allowed train/validation label fingerprints follow the copied
    # input bytes and code; fixed TEST files are never opened.
    registration = require_task09_registration(registry_path, registry_sha256)
    previous = None
    if resume_checkpoint is not None:
        checkpoint = Path(resume_checkpoint).resolve()
        if (checkpoint.parent != output
                or checkpoint.name not in ("阶段_最近.pt", "阶段_初始.pt")):
            raise ValueError("任09仅同seed新任务目录最近/初始完整状态可续跑，旧HF best不得冒名")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"任09续跑完整阶段源原件缺失：{checkpoint}")
        previous = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _task09_checkpoint_preflight(output, previous, source, sample,
                                     registry_sha256=registration)
    initial = fork_task07_initialization(sources, seed, "E0", device)
    if (not initial.random_state["torch_cuda"]
            or initial.optimizer.state_dict()["state"]
            or initial.model.correction[0].in_features != 6):
        raise ValueError("任09新HF必须真实CUDA、独立六输入随机态和空AdamW")
    model, optimizer = initial.model, initial.optimizer
    check_task09_model(model, source, CORRECTION_STAGE)
    config = load_yaml("configs/training.yaml")
    weights = config["loss_weights"]
    selection = config["multifidelity_selection_weights"]
    materials = load_materials()
    boundaries = load_resolved_boundary_conditions()
    physics = PhysicsLossComputer(
        materials, boundaries,
        PhysicsLossWeights(
            pde=float(weights["pde"]), boundary=float(weights["boundary"]),
            initial=float(weights["initial"]), interface=float(weights["interface"]),
        ),
    )
    train_data = _ir_dataset("train", subset)
    validation_data = _ir_dataset("validation", source.hf_validation_powers_w)
    validation_loader = DataLoader(validation_data, batch_size=2048, shuffle=False)
    sensor = _sensor_tensors(device, "train", subset)
    validation_sensor = _sensor_tensors(device, "validation",
                                        source.hf_validation_powers_w)
    if (len(train_data) != sample["ir_rows"] or len(sensor[0]) !=
            sum(sample["sensor_rows"].values()) or len(validation_data) != 7272
            or len(validation_sensor[0]) != 752):
        raise ValueError("任09实际HF Top/Hot/Cold合法标签与事前逐臂元数据或完整验证不符")
    consumption = task09_empty_consumption()
    stage = CORRECTION_STAGE
    correction_epochs = joint_epochs = 0
    correction_completed = None
    correction_terminal_sha = None
    best_epoch = physical_epoch = phase_best_epoch = 0
    best_stage = physical_stage = CORRECTION_STAGE
    lf_keep: dict[str, Any] | None = None
    replay = None
    last_row: dict[str, Any] = {}
    if resume_checkpoint is None:
        initial_score, initial_validation = _validation_selection(
            model, validation_loader, validation_sensor, device, selection,
        )
        initial_physical, _ = _score_guardrails(
            model, physics, materials, boundaries, device,
        )
        lf_reference = _lf_material_validation(
            model, list(source.lf_validation_powers_w), device,
        )
        if not all(math.isfinite(value) for value in (initial_score, initial_physical)):
            raise ValueError("任09合法HF/名义全项物理初始分数非有限，不提交正式原件")
        best_score = phase_best_score = initial_score
        physical_score = initial_physical
        _rng_restore(initial.random_state)
        meta = task09_phase_metadata(
            source, sample, stage=stage, correction_epochs=0, joint_epochs=0,
            registry_sha256=registration, consumption=consumption,
            initial_score=initial_score, best_score=best_score,
            physical_score=physical_score, phase_best_score=phase_best_score,
            lf_reference=lf_reference,
        )
        for filename in ("阶段_初始.pt", "阶段_最近.pt", "阶段_观测最佳.pt",
                         "阶段_物理最佳.pt"):
            _task09_state(output, filename, model, optimizer, source, sample, meta)
        _task09_view(output, model, source, sample, registry_sha256=registration,
                     score=initial_score, correction_epoch=0, joint_epoch=0,
                     validation=initial_validation)
    else:
        assert previous is not None
        stage = previous["stage"]
        if stage == JOINT_STAGE:
            switch_task09_restricted_joint(model, optimizer, source,
                                           loading_committed_state=True)
        load_training_state(checkpoint, model, optimizer)
        check_task09_model(model, source, stage)
        meta = previous["metadata"]
        correction_epochs, joint_epochs = meta["校正实际轮次"], meta["联合实际轮次"]
        correction_completed = meta["校正实际截止轮次"]
        correction_terminal_sha = meta["已提交真实校正末态SHA256"]
        if stage == JOINT_STAGE:
            terminal = output / "阶段_校正末.pt"
            if (correction_completed != correction_epochs
                    or not terminal.is_file()
                    or sha256_file(terminal) != correction_terminal_sha):
                raise ValueError("任09联合不可脱离真实同seed校正截止原件及其SHA")
            replay = _simulation_replay(source)
        elif correction_completed is not None:
            terminal = output / "阶段_校正末.pt"
            if not terminal.is_file() or sha256_file(terminal) != correction_terminal_sha:
                raise ValueError("任09已截止校正原件SHA失配，联合不得凭旧最近状态伪接")
        consumption = dict(meta["累计实际消耗"])
        initial_score = float(meta["初始合法HF选分_摄氏度"])
        best_score = float(meta["合法HF观测最佳选分_摄氏度"])
        best_epoch = int(meta["观测最佳全局轮次"])
        best_stage = meta["观测最佳阶段"]
        physical_score = float(meta["独立名义全项物理最佳损失"])
        physical_epoch = int(meta["物理最佳全局轮次"])
        physical_stage = meta["物理最佳阶段"]
        phase_best_score = float(meta["本阶段合法最佳选分"])
        phase_best_epoch = int(meta["本阶段早停最佳轮次"])
        lf_reference = meta["LF初始十验证功率两材料节点及体积"]
        lf_keep = meta["LF两材料真实节点与体积5%资格"] or None
        assert_task09_adamw_steps(
            optimizer, dict(model.named_parameters()),
            hf_batches=sample["ir_batches_2048"],
            correction_epochs=correction_epochs, joint_epochs=joint_epochs,
        )
        _rng_restore(previous["random_state"])
    total_start = correction_epochs + joint_epochs
    if total_start >= 2000 or joint_epochs >= 500:
        raise ValueError("任09该子集真实原阶段已达到1500+500上限，不再追加隐形轮次")
    session_end = min(2000, total_start + session_epoch_limit)
    if stage == CORRECTION_STAGE and correction_completed is None:
        correction_end = min(1500, session_end)
        for epoch in range(correction_epochs + 1, correction_end + 1):
            correction_epochs = epoch
            global_epoch = correction_epochs
            actual = task09_data_epoch(
                model, optimizer, train_data, sensor, device,
                seed=seed, global_epoch=global_epoch, simulation_data=None,
                weights=weights,
            )
            components = physics_optimizer_step(
                model, optimizer, physics,
                sample_collocation(
                    256, device, seed=7_090_000 + seed * 100_000 + global_epoch,
                ),
            )
            actual.update({"物理配点": 256, "物理优化步": 1,
                           "名义物理训练分项": {key: float(value.detach())
                                           for key, value in components.items()}})
            assert_task09_epoch_budget(actual, sample, joint=False)
            check_task09_model(model, source, CORRECTION_STAGE)
            for key in consumption:
                consumption[key] += actual[key]
            if consumption != task09_expected_consumption(sample, epoch, 0):
                raise ValueError("任09校正真IR批数/物理256/AdamW轮次未与累计预算一致")
            due = epoch % 10 == 0
            score = validation = modalities = guardrails = None
            observed_improved = physical_improved = False
            if due:
                score, validation = _validation_selection(
                    model, validation_loader, validation_sensor, device, selection,
                )
                modalities = _hf_modalities(model, device,
                                            list(source.hf_validation_powers_w))
                physical_now, guardrails = _score_guardrails(
                    model, physics, materials, boundaries, device,
                )
                observed_improved = score < best_score - 0.0001
                physical_improved = physical_now < physical_score
                if observed_improved:
                    best_score, best_epoch, best_stage = score, global_epoch, CORRECTION_STAGE
                    phase_best_score, phase_best_epoch = score, epoch
                if physical_improved:
                    physical_score, physical_epoch, physical_stage = (
                        physical_now, global_epoch, CORRECTION_STAGE,
                    )
            stop = due and (epoch == 1500 or epoch - phase_best_epoch >= 200)
            if stop:
                correction_completed = epoch
            current_lf_sha = _lf_tensor_sha256(model.low_fidelity_model.state_dict())
            last_row = {
                "epoch": global_epoch, "运行种子": seed, "运行臂": "F3",
                "真实HF子集功率": list(subset), "训练阶段": CORRECTION_STAGE,
                "阶段实际轮次": epoch,
                "HF合法验证选分_摄氏度": score,
                "HF合法验证分模态_摄氏度": modalities,
                "LF合法验证初态逐材料节点与体积": lf_reference if due else None,
                "当前本seed LF整张量SHA256": current_lf_sha,
                "独立名义物理损失": physical_now if due else None,
                "累计实际消耗": dict(consumption), **actual,
            }
            if guardrails is not None:
                last_row.update(guardrails)
            with (output / "training.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(last_row, ensure_ascii=False) + "\n")
            meta = task09_phase_metadata(
                source, sample, stage=CORRECTION_STAGE,
                correction_epochs=epoch, joint_epochs=0,
                registry_sha256=registration, consumption=consumption,
                correction_completed=correction_completed,
                initial_score=initial_score, best_score=best_score,
                best_epoch=best_epoch, best_stage=best_stage,
                physical_score=physical_score, physical_epoch=physical_epoch,
                physical_stage=physical_stage, phase_best_score=phase_best_score,
                phase_best_epoch=phase_best_epoch, lf_reference=lf_reference,
                current_lf_sha256=current_lf_sha,
            )
            _task09_epoch_commit(
                output, model, optimizer, source, sample, meta,
                global_epoch=global_epoch, observed_improved=observed_improved,
                physical_improved=physical_improved, score=score,
                validation=validation, registry_sha256=registration,
            )
            if stop:
                terminal = _task09_state(output, "阶段_校正末.pt", model, optimizer,
                                         source, sample, meta)
                correction_terminal_sha = sha256_file(terminal)
                break
    if (stage == CORRECTION_STAGE and correction_completed is not None
            and correction_epochs < session_end):
        if correction_terminal_sha is None:
            raise ValueError("任09校正截止真实完整原件SHA缺失，不进入受限联合")
        replay = _simulation_replay(source)
        phase_best_score, _ = _validation_selection(
            model, validation_loader, validation_sensor, device, selection,
        )
        switch_task09_restricted_joint(model, optimizer, source)
        stage = JOINT_STAGE
        joint_epochs = phase_best_epoch = 0
        lf_keep = None
        meta = task09_phase_metadata(
            source, sample, stage=JOINT_STAGE,
            correction_epochs=correction_epochs, joint_epochs=0,
            registry_sha256=registration, consumption=consumption,
            correction_completed=correction_completed,
            correction_terminal_sha256=correction_terminal_sha,
            initial_score=initial_score, best_score=best_score,
            best_epoch=best_epoch, best_stage=best_stage,
            physical_score=physical_score, physical_epoch=physical_epoch,
            physical_stage=physical_stage, phase_best_score=phase_best_score,
            lf_reference=lf_reference,
        )
        _task09_state(output, "阶段_联合初始.pt", model, optimizer, source, sample, meta)
        _task09_state(output, "阶段_最近.pt", model, optimizer, source, sample, meta)
    if stage == JOINT_STAGE and correction_epochs + joint_epochs < session_end:
        if replay is None:
            raise ValueError("任09联合必须从本seed原LF60真标签重新采样完整回放")
        joint_end = min(500, joint_epochs + session_end - correction_epochs - joint_epochs)
        for epoch in range(joint_epochs + 1, joint_end + 1):
            joint_epochs = epoch
            global_epoch = correction_epochs + joint_epochs
            actual = task09_data_epoch(
                model, optimizer, train_data, sensor, device,
                seed=seed, global_epoch=global_epoch, simulation_data=replay,
                expected_lf_powers=source.lf_train_powers_w, weights=weights,
            )
            components = physics_optimizer_step(
                model, optimizer, physics,
                sample_collocation(
                    256, device, seed=7_090_000 + seed * 100_000 + global_epoch,
                ),
            )
            actual.update({"物理配点": 256, "物理优化步": 1,
                           "名义物理训练分项": {key: float(value.detach())
                                           for key, value in components.items()}})
            assert_task09_epoch_budget(actual, sample, joint=True)
            check_task09_model(model, source, JOINT_STAGE)
            for key in consumption:
                consumption[key] += actual[key]
            if consumption != task09_expected_consumption(sample, correction_epochs, epoch):
                raise ValueError("任09联合须N个HF步+物理一步+真LF60×2048四投影逐轮累计")
            due = epoch % 10 == 0
            score = validation = modalities = guardrails = lf_values = None
            observed_improved = physical_improved = False
            if due:
                score, validation = _validation_selection(
                    model, validation_loader, validation_sensor, device, selection,
                )
                modalities = _hf_modalities(model, device,
                                            list(source.hf_validation_powers_w))
                lf_values = _lf_material_validation(
                    model, list(source.lf_validation_powers_w), device,
                )
                lf_keep = task04_lf_keep_guardrail(lf_reference, lf_values)
                physical_now, guardrails = _score_guardrails(
                    model, physics, materials, boundaries, device,
                )
                if lf_keep["LF两材料节点与真实体积均守住5%护栏"]:
                    if score < phase_best_score - 0.0001:
                        phase_best_score, phase_best_epoch = score, epoch
                    observed_improved = score < best_score - 0.0001
                    physical_improved = physical_now < physical_score
                    if observed_improved:
                        best_score, best_epoch, best_stage = score, global_epoch, JOINT_STAGE
                    if physical_improved:
                        physical_score, physical_epoch, physical_stage = (
                            physical_now, global_epoch, JOINT_STAGE,
                        )
            stop = due and (epoch == 500 or epoch - phase_best_epoch >= 200)
            current_lf_sha = _lf_tensor_sha256(model.low_fidelity_model.state_dict())
            last_row = {
                "epoch": global_epoch, "运行种子": seed, "运行臂": "F3",
                "真实HF子集功率": list(subset), "训练阶段": JOINT_STAGE,
                "阶段实际轮次": epoch,
                "HF合法验证选分_摄氏度": score,
                "HF合法验证分模态_摄氏度": modalities,
                "LF合法验证逐材料节点与体积": lf_values,
                "LF两材料真实节点与体积5%资格": lf_keep,
                "当前本seed LF整张量SHA256": current_lf_sha,
                "独立名义物理损失": physical_now if due else None,
                "累计实际消耗": dict(consumption), **actual,
            }
            if guardrails is not None:
                last_row.update(guardrails)
            with (output / "training.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(last_row, ensure_ascii=False) + "\n")
            meta = task09_phase_metadata(
                source, sample, stage=JOINT_STAGE,
                correction_epochs=correction_epochs, joint_epochs=epoch,
                registry_sha256=registration, consumption=consumption,
                correction_completed=correction_completed,
                correction_terminal_sha256=correction_terminal_sha,
                initial_score=initial_score, best_score=best_score,
                best_epoch=best_epoch, best_stage=best_stage,
                physical_score=physical_score, physical_epoch=physical_epoch,
                physical_stage=physical_stage,
                phase_best_score=phase_best_score,
                phase_best_epoch=phase_best_epoch, lf_reference=lf_reference,
                lf_keep=lf_keep, current_lf_sha256=current_lf_sha,
            )
            _task09_epoch_commit(
                output, model, optimizer, source, sample, meta,
                global_epoch=global_epoch, observed_improved=observed_improved,
                physical_improved=physical_improved, score=score,
                validation=validation, registry_sha256=registration,
            )
            if stop:
                for filename in ("阶段_联合末.pt", "阶段_训练末.pt"):
                    _task09_state(output, filename, model, optimizer,
                                  source, sample, meta)
                break
    finished = (output / "阶段_训练末.pt").is_file()
    state = ("受限联合真实截止；待独立CUDA合法HF/LF及16/64阶能源审核"
             if finished else "正式阶段完整会话暂停；仅可按本seed最近状态真实接续")
    if lf_keep is not None and not lf_keep["LF两材料节点与真实体积均守住5%护栏"]:
        state += "；最近联合LF四口径未守住5%护栏，不得采用该状态"
    report = {
        "状态": state, "资格": "真实CUDA候选；非精度改善或内部HF可信证明",
        "运行种子": seed, "序列": name, "HF子集功率数": size,
        "真实HF子集功率": list(subset),
        "正式登记SHA256": registration,
        "校正实际轮次": correction_epochs, "受限联合实际轮次": joint_epochs,
        "本seed原LF整张量SHA256": source.lf_tensor_sha256,
        "观测最佳合法HF选分_摄氏度": best_score,
        "观测最佳真实全局轮次": best_epoch,
        "物理最佳独立全项损失": physical_score,
        "物理最佳真实全局轮次": physical_epoch,
        "LF四口径最新5%保持": lf_keep,
        "逐真实轮次累计训练预算": dict(consumption),
        "最新实际轮次证据": last_row,
        "本CUDA会话耗时秒": time.perf_counter() - started,
        "观测误差正式曲线可登记": False,
        "16/64阶原瓦数独审已完成": False,
        "旧固定测试温度读取": False,
    }
    temp = output / "任09F3阶段报告.json.tmp"
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    temp.replace(output / "任09F3阶段报告.json")
    return report


def task09_data_epoch(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    train_data: Any, sensor: tuple[Tensor, ...], device: torch.device,
    *, seed: int, global_epoch: int, simulation_data: Any | None,
    expected_lf_powers: tuple[float, ...] | None = None,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Each actual HF batch computes one HF objective; LF mini-batches only accumulate.

    A fixed 1/4 factor for each of 60 LF mini-batches preserves the locked
    full-epoch nominal 15x LF60 mean even for uneven 9/8 or 6/5 groups.
    """
    losses = weights or {"ir": 1.0, "sensor_absolute": 5.0,
                         "sensor_delta": 1.0, "low_fidelity": 1.0}
    if (seed not in range(5) or global_epoch <= 0
            or not all(float(losses[key]) >= 0 for key in (
                "ir", "sensor_absolute", "sensor_delta", "low_fidelity"
            ))):
        raise ValueError("任09优化步骤须本seed完整真实轮次与共同非负观测损失权重")
    hf_loader = DataLoader(
        train_data, batch_size=2048, shuffle=True,
        generator=torch.Generator().manual_seed(7_070_000 + seed * 100_000 + global_epoch),
    )
    hf_batches = len(hf_loader)
    group_sizes = balanced_task09_lf_groups(hf_batches)
    sensor_x, sensor_target, sensor_delta, sensor_baseline = sensor
    if (not len(sensor_x) or not len(sensor_x) == len(sensor_target)
            == len(sensor_delta) == len(sensor_baseline)):
        raise ValueError("任09顶部真IR主批必须同步完整Hot/Cold环温训练观测")
    replay_groups = None
    expected = frozenset(round(float(power), 4) for power in (expected_lf_powers or ()))
    if simulation_data is not None:
        if len(expected) != 60:
            raise ValueError("任09联合回放必须事先登记原LF60个训练功率")
        replay_x = simulation_data.tensors[0]
        powers = Counter(round(float(power), 4) for power in replay_x[:, 3].tolist())
        if (set(powers) != expected or set(powers.values()) != {2048}
                or not (replay_x[:, 4] < 0.5).any()
                or not (replay_x[:, 4] >= 0.5).any()):
            raise ValueError("任09LF60训练功率必须逐功率真2048配点且覆盖Cu和SiC")
        lf_loader = DataLoader(
            simulation_data, batch_size=2048, shuffle=True,
            generator=torch.Generator().manual_seed(
                7_080_000 + seed * 100_000 + global_epoch,
            ),
        )
        replay_groups = iter_task09_lf_groups(lf_loader, hf_batches)
    model.train()
    exposed = replayed = copper = silicon_carbide = 0
    observed_lf = Counter()
    hf_steps = lf_batches = 0
    ir_losses, sensor_losses = [], []
    lf_loss_sum = 0.0
    for x, y, weight in hf_loader:
        x, y, weight = (value.to(device) for value in (x, y, weight))
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, fidelity="high")
        ir_loss = (
            weight * ((prediction - y) / model.scales.temperature_scale_k).square()
        ).sum() / weight.sum()
        ring_prediction = model(sensor_x, fidelity="high")
        absolute, delta = _macro_sensor_training_losses(
            ring_prediction, sensor_target, sensor_delta, sensor_baseline,
            sensor_x, model.scales.temperature_scale_k,
        )
        sensor_loss = (float(losses["sensor_absolute"]) * absolute
                       + float(losses["sensor_delta"]) * delta)
        objective = float(losses["ir"]) * ir_loss + sensor_loss
        if not torch.isfinite(objective):
            raise ValueError("任09HF训练目标非有限；不得提交旧来源或虚假训练结果")
        objective.backward()  # HF data term is evaluated exactly once per HF main batch.
        group = next(replay_groups) if replay_groups is not None else ()
        for lf_x_cpu, lf_y_cpu in group:
            lf_power = Counter(round(float(power), 4) for power in
                               lf_x_cpu[:, 3].tolist())
            observed_lf.update(lf_power)
            copper += int((lf_x_cpu[:, 4] < 0.5).sum())
            silicon_carbide += int((lf_x_cpu[:, 4] >= 0.5).sum())
            lf_x = lf_x_cpu.to(device)
            lf_y = lf_y_cpu.to(device)
            lf_prediction = model(lf_x, fidelity="low")
            replay_loss = (
                (lf_prediction - lf_y) / model.scales.temperature_scale_k
            ).square().mean()
            if not torch.isfinite(replay_loss):
                raise ValueError("任09LF真实回放损失非有限，不再推进HF优化器")
            (float(losses["low_fidelity"]) *
             task09_lf_microbatch_scale(len(group)) * replay_loss).backward()
            lf_loss_sum += float(replay_loss.detach())
            lf_batches += 1
            replayed += len(lf_x_cpu)
        optimizer.step()
        hf_steps += 1
        exposed += len(x)
        ir_losses.append(float(ir_loss.detach()))
        sensor_losses.append(float(sensor_loss.detach()))
    if (exposed != len(train_data) or hf_steps != hf_batches
            or (simulation_data is not None and (
                lf_batches != 60 or replayed != 60 * 2048
                or observed_lf != Counter({power: 2048 for power in expected})
                or not copper or not silicon_carbide
            ))):
        raise ValueError("任09真实HF主batch或LF60×2048训练消费预算不完整")
    return {
        "HF训练观测点": exposed, "HF训练传感器点": hf_steps * len(sensor_x),
        "HF观测优化步": hf_steps,
        "HF顶部训练损失": sum(ir_losses) / hf_steps,
        "HF环温训练损失": sum(sensor_losses) / hf_steps,
        "LF真实回放训练点": replayed, "LF联合回放batch": lf_batches,
        "LF_Cu真实回放点": copper, "LF_SiC真实回放点": silicon_carbide,
        "LF仅低保真训练损失": lf_loss_sum / lf_batches if lf_batches else None,
        "LF每HF步真实分组": group_sizes if simulation_data is not None else [],
        "LF60每功率真2048已核验": simulation_data is not None,
        "每LF2048小批损失名义系数": 0.25 if simulation_data is not None else None,
    }
