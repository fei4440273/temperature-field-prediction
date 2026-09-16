"""Fresh MLP HF training with byte-locked sources and complete session receipts."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import tarfile
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl
import torch
import yaml
from torch.utils.data import DataLoader

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import assert_no_hf_leakage, build_power_splits
from sic_cu.eval.protocol_checks import (
    checkpoint_provenance, current_protocol_fingerprints,
    validate_hf_checkpoint_provenance, validate_lf_checkpoint_provenance,
)
from sic_cu.losses import PhysicsLossComputer, PhysicsLossWeights
from sic_cu.models import AdditiveCorrectionModel, ModelScales
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.train.common import load_training_state, save_training_state, write_config_snapshot
from sic_cu.train.multifidelity import _ir_dataset, _sensor_tensors
from sic_cu.train.simulation import build_model, load_sampled_points, set_seed
from sic_cu.train.task04_joint import _validation_selection
from sic_cu.train.task07_formal import _hf_epoch
from sic_cu.train.task11_mlp_lf_formal import (
    SOURCE_MEMBERS as LF_SOURCE_MEMBERS, qualify_task11_mlp_lf_source,
)


ROOT_LEDGER = "多保真DeepONet预测精度优化总计划与执行台账.md"
ROOT_TOKEN = "TASK11_NEW_MLP_HF_GATE:v1"
RUN_DIRECTORY = "研究记录/任务11_外部对照/正式新MLP公平训练"
CORRECTION_STAGE = "task11_mlp_hf_correction"
JOINT_STAGE = "task11_mlp_hf_full_joint"
STAGE_BUDGET = {"HF校正上限轮次": 1500, "HF联合上限轮次": 500}
LF_KWARGS = {"width": 128, "depth": 5, "activation": "tanh", "include_material": True}
CORRECTION_KWARGS = {
    "width": 128, "depth": 4, "activation": "tanh", "include_material": True,
    "hard_initial_temperature_k": 295.15, "initial_ramp_time_s": 0.05,
    "correction_calibration_range_w": [55.0, 800.0],
    "correction_support_range_w": [0.0, 800.0],
    "correction_extrapolation_exponent": 2.0, "correction_power_scaling": "none",
    "correction_power_reference_w": 400.0, "correction_direct_power_input": True,
    "silicon_carbide_height_m": 0.012, "surface_guide_output": "residual",
}
HF_BUDGET = {
    "seeds": [0, 1, 2, 3, 4], "method": "mlp_pinn_additive",
    "correction_epochs": 1500, "joint_epochs": 500,
    "hf_train_powers": 12, "hf_validation_powers": 3,
    "hf_ir_train_points": 29593, "hf_ir_validation_points": 7272,
    "hf_sensor_train_points": 2985, "hf_sensor_validation_points": 752,
    "hf_batch_size": 2048, "hf_batches_per_epoch": 15,
    "sensor_repetitions_per_epoch": 15, "physics_collocation": 256,
    "physics_steps_per_epoch": 1, "joint_lf_train_powers": 60,
    "joint_lf_samples_per_power": 2048, "joint_lf_batches_per_epoch": 60,
    "joint_lf_update": "all_mlp_parameters", "correction_lf_frozen": True,
    "correction_learning_rate": 0.001, "joint_learning_rate": 0.0001,
    "weight_decay": 1e-6, "patience": 200, "min_delta": 0.0001,
    "validation_interval": 10, "selection_metric": "macro_v1",
    "selection_weights": {"ir_rmse": 1.0, "sensor_absolute_rmse": 0.2,
                          "sensor_delta_rmse": 1.0},
    "loss_weights": {"low_fidelity": 1.0, "ir": 1.0, "sensor_absolute": 5.0,
                     "sensor_delta": 1.0, "pde": 1.0, "boundary": 1.0,
                     "initial": 1.0, "interface": 1.0},
    "lf_model_kwargs": LF_KWARGS, "correction_model_kwargs": CORRECTION_KWARGS,
    "scales": asdict(ModelScales()), "session_epoch_limit": 200,
    "device": "cuda", "world_size": 1,
}
ENERGY_BUDGET = {
    "功率_瓦": [55, 115.2, 364.3, 403, 630.5, 729],
    "时刻_秒": [1, 10, 50, 100, 200], "求积阶数": [16, 64],
    "功率时刻点数": 30, "模型状态": ["best", "final"],
    "计算dtype": "float64", "名义吸收归一均值筛查": 0.05,
    "名义吸收归一95分位筛查": 0.10, "不用于训练选模": True,
    "工程安全阈值": False,
}
LF_REGISTRY = "研究记录/任务11_外部对照/任11_新MLP五种子真LF训练事前登记_20260916.yaml"
LF_REGISTRY_SHA = "510ca066f221ae79d2015c1c0ccec60a8235dafca82432fc84c1026f13202c19"
LF_TAR = "研究记录/任务11_外部对照/任11_新MLP五种子真LF源码事前冻结_20260916.tar.gz"
LF_TAR_SHA = "c70c82667fa8840f3a72078cc4be68dba9e47fc084de8cd06e6dca923e60c139"
LF_DATA = "研究记录/任务11_外部对照/任11_新MLP_LF真实模拟70源目录_事前20260916.json"
LF_DATA_SHA = "96a346c4eaa3fac63326a84d7e6ab8a1cee94d8b46083f5c41907db70be2c485"
SOURCE_MEMBERS = tuple(sorted(set(LF_SOURCE_MEMBERS) | {
    "src/sic_cu/__init__.py", "src/sic_cu/compat.py",
    "src/sic_cu/data/__init__.py", "src/sic_cu/data/processed.py", "src/sic_cu/data/sensors.py",
    "src/sic_cu/eval/__init__.py", "src/sic_cu/eval/metrics.py", "src/sic_cu/eval/energy_v5.py",
    "src/sic_cu/eval/validation.py", "src/sic_cu/eval/residual_interpolation.py",
    "src/sic_cu/models/deeponet_pinn.py", "src/sic_cu/models/gno_pinn.py",
    "src/sic_cu/models/lstm_pinn.py", "src/sic_cu/models/multifidelity.py",
    "src/sic_cu/models/interpolation.py", "src/sic_cu/prediction.py",
    "src/sic_cu/models/pod_pinn.py", "src/sic_cu/models/prc.py",
    "src/sic_cu/models/surface_residual.py", "src/sic_cu/models/residual_interpolation.py",
    "src/sic_cu/physics/__init__.py", "src/sic_cu/physics/geometry.py",
    "src/sic_cu/train/__init__.py", "src/sic_cu/train/multifidelity.py",
    "src/sic_cu/train/surface_residual.py",
    "src/sic_cu/train/task04_joint.py", "src/sic_cu/train/task07_formal.py",
    "src/sic_cu/train/task07_source.py", "src/sic_cu/train/task11_mlp_hf_formal.py",
    "scripts/51_run_task11_mlp_hf_formal.py", "tests/test_task11_mlp_hf_formal.py",
    "src/sic_cu/eval/task11_mlp_hf_energy.py", "scripts/52_audit_task11_mlp_hf_energy.py",
    "tests/test_task11_mlp_hf_energy.py",
}))


def _path(value: str | Path, root: Path = PROJECT_ROOT, *, file: bool = True) -> Path:
    supplied = Path(value)
    candidate = supplied if supplied.is_absolute() else root / supplied
    resolved = candidate.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError("任11文件/输出只能在项目内，不得越界")
    for parent in (candidate, *candidate.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("任11来源或输出不能经过符号链接")
    if file and not resolved.is_file():
        raise ValueError("任11来源必须是项目内真实普通文件")
    return resolved


def _tensor_sha(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        for text in (name, str(tensor.dtype), str(tuple(tensor.shape))):
            raw = text.encode("ascii")
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        raw = tensor.numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _lf_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "low_fidelity_model."
    return {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}


def collect_task11_mlp_lf_catalog() -> dict[str, Any]:
    rows = []
    for seed in range(5):
        directory = PROJECT_ROOT / RUN_DIRECTORY / f"正式MLP_LF_seed{seed}"
        qualification = qualify_task11_mlp_lf_source(
            registry=LF_REGISTRY, registry_sha=LF_REGISTRY_SHA,
            source_tar=LF_TAR, source_tar_sha=LF_TAR_SHA,
            catalog=LF_DATA, catalog_sha=LF_DATA_SHA, output=directory, seed=seed,
        )
        best = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        if (best.get("method") != "mlp_pinn" or best.get("model_kwargs") != LF_KWARGS
                or best.get("scales") != HF_BUDGET["scales"]):
            raise ValueError("任11新LF方法、宽深和尺度不等于冻结MLP结构")
        originals = {}
        for source in sorted(directory.rglob("*")):
            if source.is_symlink():
                raise ValueError("任11新LF原件目录不得包含符号链接")
            if source.is_file():
                originals[str(source.relative_to(PROJECT_ROOT))] = sha256_file(source)
        receipts = sorted(directory.glob("LF会话收据_[0-9][0-9][0-9][0-9].json"))
        rows.append({
            "seed": seed, "目录": str(directory.relative_to(PROJECT_ROOT)),
            "最佳检查点": str((directory / "best.pt").relative_to(PROJECT_ROOT)),
            "最佳检查点SHA256": qualification["当前新MLP_LF最佳检查点SHA256"],
            "LF初始张量SHA256": _tensor_sha(best["model_state"]),
            "实际轮次": qualification["当前新MLP_LF累计轮次"],
            "最佳LF验证轮次": best["epoch"],
            "最佳LF验证RMSE_摄氏度": best["validation_rmse_c"],
            "LF真实多会话累计成本秒": qualification["LF真实多会话累计成本秒"],
            "CUDA峰值显存字节": max(json.loads(p.read_text(encoding="utf-8"))[
                "峰值真实CUDA显存字节"] for p in receipts),
            "真实原件SHA256": originals,
            "会话收据": [{"原件": str(p.relative_to(PROJECT_ROOT)), "SHA256": sha256_file(p)}
                          for p in receipts],
        })
    if len({row["LF初始张量SHA256"] for row in rows}) != 5:
        raise ValueError("任11五seed新LF张量不能复制或重贴种子")
    return {"schema_version": 1, "阶段": "task11_completed_new_mlp_lf",
            "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
            "LF事前来源": {"YAML原件": LF_REGISTRY, "YAML_SHA256": LF_REGISTRY_SHA,
                          "TAR原件": LF_TAR, "TAR_SHA256": LF_TAR_SHA,
                          "CATALOG原件": LF_DATA, "CATALOG_SHA256": LF_DATA_SHA},
            "五seed新LF": rows}


def collect_task11_mlp_hf_data_catalog() -> dict[str, Any]:
    splits = build_power_splits()
    counts = {}
    originals = []
    sources = (("HF顶部开发", "data/processed/experiment_ir_radial.parquet", False),
               ("HF两环开发", "data/processed/sensor_ring_raw.parquet", True))
    for purpose, relative, sensor in sources:
        path = _path(relative)
        schema = pl.read_parquet_schema(path)
        required = {"power_w", "split", "sensor_type"} if sensor else {"power_w", "split"}
        if not required.issubset(schema):
            raise ValueError("任11HF原件缺少固定功率/划分身份结构列")
        # Only structural columns are decoded; neither temperature nor TEST is opened.
        structure = pl.scan_parquet(path).select(sorted(required)).collect()
        if set(structure["split"].to_list()) != {"train", "validation"}:
            raise ValueError("任11HF开发原件不能包含TEST或非开发折")
        for role, expected in (("train", splits.hf_train), ("validation", splits.hf_validation)):
            part = structure.filter(pl.col("split") == role)
            if {round(float(p), 4) for p in part["power_w"]} != expected:
                raise ValueError("任11HF来源功率折不等于固定12训练/3验证")
            counts[("环温" if sensor else "Top") + ("训练" if role == "train" else "验证")] = part.height
            if sensor and any(set(part.filter(pl.col("power_w").round(4) == power)[
                    "sensor_type"].to_list()) != {"hot", "cold"} for power in expected):
                raise ValueError("任11HF两环必须每个开发功率同时有Hot和Cold")
        originals.append({"用途": purpose, "原件": relative, "文件SHA256": sha256_file(path),
                          "真实行数": structure.height, "原列schema": dict(schema)})
    # Polars dtypes are serialized literally, not as executable objects.
    for row in originals:
        row["原列schema"] = {key: str(value) for key, value in row["原列schema"].items()}
    if counts != {"Top训练": 29593, "Top验证": 7272, "环温训练": 2985, "环温验证": 752}:
        raise ValueError("任11HF真实顶部/环温数量不等于已登记29593/7272/2985/752")
    return {"schema_version": 1, "阶段": "task11_hf_development_12_3",
            "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
            "HF训练功率_瓦": sorted(splits.hf_train),
            "HF合法验证功率_瓦": sorted(splits.hf_validation),
            "HF计数": counts, "HF源原件": originals,
            "协议指纹": current_protocol_fingerprints()}


def _root_active(ledger: Path, hashes: Mapping[str, str]) -> bool:
    expected = (f"{ROOT_TOKEN}; status=active; YAML_SHA256={hashes['YAML_SHA256']}; "
                f"TAR_SHA256={hashes['TAR_SHA256']}; LF_CATALOG_SHA256={hashes['LF_CATALOG_SHA256']}; "
                f"HF_DATA_CATALOG_SHA256={hashes['HF_DATA_CATALOG_SHA256']}")
    rows = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) >= 4 and re.fullmatch(r"录-\d{4}", fields[1]) and fields[2].startswith(ROOT_TOKEN):
            rows.append(fields[2])
    return len(rows) == 1 and (rows[0] == expected or rows[0].startswith(expected + "; "))


def preflight_task11_mlp_hf(
    *, registry: str | Path, registry_sha: str, source_tar: str | Path, source_tar_sha: str,
    lf_catalog: str | Path, lf_catalog_sha: str, hf_data_catalog: str | Path, hf_data_catalog_sha: str,
    output: str | Path, seed: int, resume_checkpoint: str | Path | None = None,
    qualified_source_only: bool = False, project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    files = {"登记原件": _path(registry, root), "源码归档原件": _path(source_tar, root),
             "LF目录原件": _path(lf_catalog, root), "HF数据目录原件": _path(hf_data_catalog, root)}
    hashes = {"YAML_SHA256": registry_sha, "TAR_SHA256": source_tar_sha,
              "LF_CATALOG_SHA256": lf_catalog_sha, "HF_DATA_CATALOG_SHA256": hf_data_catalog_sha}
    for name, digest in zip(files, hashes.values()):
        if not re.fullmatch(r"[0-9a-f]{64}", digest or "") or sha256_file(files[name]) != digest:
            raise ValueError("任11HF四份事前原件SHA缺失或漂移")
    budget = yaml.safe_load(files["登记原件"].read_text(encoding="utf-8"))
    if (not isinstance(budget, dict) or budget.get("schema_version") != 1
            or budget.get("阶段") != "fresh_mlp_high_fidelity" or budget.get("ROOT门禁标签") != ROOT_TOKEN
            or budget.get("旧固定TEST温度读取") is not False
            or budget.get("模拟测试功率温度读取") is not False or budget.get("LF全参联合更新") is not True
            or budget.get("正式预算") != HF_BUDGET or budget.get("独立能源审核") != ENERGY_BUDGET
            or budget.get("LF五seed目录SHA256") != lf_catalog_sha
            or budget.get("HF开发数据目录SHA256") != hf_data_catalog_sha
            or set(budget.get("源码普通成员SHA256", {})) != set(SOURCE_MEMBERS)):
        raise ValueError("任11HF只允许事前冻结的新MLP五seed完整校正/全LF联合合同")
    if not _root_active(_path(ROOT_LEDGER, root), hashes):
        raise ValueError("任11HF主ROOT缺同一唯一编号活动行四SHA")
    if type(seed) is not int or seed not in range(5) or os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("任11HF仅许登记seed0..4单卡单进程")
    destination = _path(output, root, file=False)
    if destination != root / RUN_DIRECTORY / f"正式MLP_HF_seed{seed}":
        raise ValueError("任11HF输出只能是同seed固定正式目录")
    if qualified_source_only:
        if resume_checkpoint is not None or not destination.is_dir():
            raise ValueError("任11HF只读终态资格需要已存在同seed目录")
    elif resume_checkpoint is None:
        if destination.exists():
            raise ValueError("任11HF新训练禁止覆盖已有同seed目录")
    elif _path(resume_checkpoint, root) != destination / "阶段_最近.pt":
        raise ValueError("任11HF续跑只接受同seed自身阶段_最近.pt")
    source_hashes = budget["源码普通成员SHA256"]
    try:
        with tarfile.open(files["源码归档原件"], "r:gz") as archive:
            members = archive.getmembers()
            if (len(members) != len(SOURCE_MEMBERS) or any(not m.isfile() for m in members)
                    or {m.name for m in members} != set(SOURCE_MEMBERS)):
                raise ValueError("任11HF源码tar必须逐普通成员完整封存")
            for member in members:
                stream = archive.extractfile(member)
                expected = source_hashes[member.name]
                if (stream is None or hashlib.sha256(stream.read()).hexdigest() != expected
                        or sha256_file(_path(member.name, root)) != expected):
                    raise ValueError("任11HF源码归档/声明/现场SHA漂移")
    except (OSError, tarfile.TarError) as error:
        raise ValueError("任11HF事前源码归档无法核验") from error
    lf = json.loads(files["LF目录原件"].read_text(encoding="utf-8"))
    hf = json.loads(files["HF数据目录原件"].read_text(encoding="utf-8"))
    if lf != collect_task11_mlp_lf_catalog() or hf != collect_task11_mlp_hf_data_catalog():
        raise ValueError("任11HF五新LF原件或开发数据/协议目录现场漂移")
    row = lf["五seed新LF"][seed]
    identity = {**hashes, "seed": seed, "LF最佳检查点SHA256": row["最佳检查点SHA256"],
                "LF初始张量SHA256": row["LF初始张量SHA256"]}
    result = {"状态": "CPU前置门禁PASS；未启动GPU或创建HF运行目录", **hashes,
              **{name: str(path) for name, path in files.items()},
              "输出": str(destination), "seed": seed, "身份": identity,
              "LF来源": row, "预算": budget, "项目根": str(root)}
    if resume_checkpoint is not None:
        result["续跑状态"] = verify_task11_mlp_hf_resume(
            destination, seed=seed, identity=identity, lf_row=row)
    return result


def build_task11_mlp_hf_initialization(
    lf_row: Mapping[str, Any], seed: int, device: torch.device = torch.device("cpu"),
) -> tuple[AdditiveCorrectionModel, torch.optim.AdamW, dict[str, Any]]:
    if lf_row.get("seed") != seed or seed not in range(5):
        raise ValueError("任11HF初始化只接受自己的新LF种子")
    path = _path(lf_row["最佳检查点"])
    if sha256_file(path) != lf_row["最佳检查点SHA256"]:
        raise ValueError("任11HF新LF起点SHA漂移")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_lf_checkpoint_provenance(payload)
    if (payload.get("seed") != seed or payload.get("method") != "mlp_pinn"
            or payload.get("model_kwargs") != LF_KWARGS or payload.get("scales") != HF_BUDGET["scales"]
            or _tensor_sha(payload["model_state"]) != lf_row["LF初始张量SHA256"]):
        raise ValueError("任11HF初始LF真种子/结构/张量与冻结目录不一致")
    scales = ModelScales(**payload["scales"])
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        low = build_model("mlp_pinn", scales, **LF_KWARGS)
        low.load_state_dict(payload["model_state"], strict=True)
        torch.default_generator.manual_seed(seed)
        model = AdditiveCorrectionModel(low, scales, freeze_low_fidelity=True,
                                        **copy.deepcopy(CORRECTION_KWARGS)).to(device)
    optimizer = torch.optim.AdamW(model.correction.parameters(), lr=0.001, weight_decay=1e-6)
    splits = build_power_splits()
    architecture = {"schema_version": 1, "method": "multifidelity_correction", "seed": seed,
                    "low_fidelity_method": "mlp_pinn", "scales": asdict(scales),
                    "low_fidelity_model_kwargs": copy.deepcopy(LF_KWARGS),
                    "correction_model_kwargs": copy.deepcopy(CORRECTION_KWARGS),
                    "hf_train_powers_w": sorted(splits.hf_train),
                    "hf_validation_powers_w": sorted(splits.hf_validation),
                    "sensors_used": True, "surface_residual_guide_spec": None,
                    "resolved_physics": asdict(load_resolved_boundary_conditions()),
                    "provenance": checkpoint_provenance(role="high_fidelity_multifidelity",
                        train_powers_w=splits.hf_train, validation_powers_w=splits.hf_validation),
                    "material_passport": {"simulation_data_used": True, "experiment_data_used": True,
                        "sensor_data_used": True, "physics_loss_used": True,
                        "internal_experiment_truth": "not available"}}
    return model, optimizer, architecture


def activate_task11_mlp_joint(model: AdditiveCorrectionModel, optimizer: torch.optim.AdamW) -> None:
    if len(optimizer.param_groups) != 1:
        raise ValueError("任11HF联合转换只能从完整单组校正AdamW一次进入")
    model.freeze_low_fidelity(False)
    optimizer.param_groups[0]["lr"] = 0.0001
    optimizer.add_param_group({"params": list(model.low_fidelity_model.parameters()),
                               "lr": 0.0001, "weight_decay": 1e-6})


def transition_task11_mlp_hf_joint(
    model: AdditiveCorrectionModel, optimizer: torch.optim.AdamW, metadata: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    correction = metadata.get("HF校正实际轮次")
    score = metadata.get("最近合法选分_摄氏度")
    validation = metadata.get("最近合法分模态")
    if (type(correction) is not int or not 200 <= correction <= 1500
            or correction % 10 != 0 or metadata.get("HF校正实际截止轮次") != correction
            or not isinstance(score, (int, float)) or not math.isfinite(score)
            or not isinstance(validation, dict) or not validation
            or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in validation.values())):
        raise ValueError("任11HF联合转换须继承真实校正截止的合法选分及分模态")
    activate_task11_mlp_joint(model, optimizer)
    return float(score), copy.deepcopy(validation)


def task11_phase_finished(epoch: int, best_epoch: int, limit: int) -> bool:
    return epoch == limit or (epoch % 10 == 0 and epoch - best_epoch >= 200)


def audit_task11_hf_log(rows: list[dict[str, Any]], seed: int,
                       correction_epoch: int, joint_epoch: int) -> dict[str, int]:
    totals = {"HF观测点": 0, "HF传感器点": 0, "HF观测优化步": 0,
              "物理优化步": 0, "物理配点": 0, "LF回放点": 0, "LF回放小批": 0}
    if (not 0 <= correction_epoch <= 1500 or not 0 <= joint_epoch <= 500
            or len(rows) != correction_epoch + joint_epoch):
        raise ValueError("任11HF日志长度/真实阶段轮次不闭合")
    for number, row in enumerate(rows, 1):
        joint = number > correction_epoch
        phase_epoch = number - correction_epoch if joint else number
        expected = {"HF观测点": 29593, "HF传感器点": 2985 * 15,
                    "HF观测优化步": 15, "物理优化步": 1, "物理配点": 256,
                    "LF回放点": 60 * 2048 if joint else 0, "LF回放小批": 60 if joint else 0}
        score = row.get("合法验证选分_摄氏度")
        if (row.get("epoch") != number or row.get("seed") != seed
                or row.get("阶段") != (JOINT_STAGE if joint else CORRECTION_STAGE)
                or row.get("阶段轮次") != phase_epoch
                or row.get("旧固定TEST温度读取") is not False
                or row.get("模拟测试功率温度读取") is not False
                or any(row.get(key) != value for key, value in expected.items())
                or (score is not None and (not isinstance(score, (int, float)) or not math.isfinite(score)))
                or ((phase_epoch % 10 == 0) != (score is not None))):
            raise ValueError("任11HF逐轮日志/真实标签/传感器/LF/物理消费不闭合")
        for key, value in expected.items():
            totals[key] += value
    return totals


def replay_task11_mlp_hf_selection(
    rows: list[dict[str, Any]], initial_score: float, correction_epoch: int, joint_epoch: int,
    *, stage: str, initial_validation: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if (type(correction_epoch) is not int or not 0 <= correction_epoch <= 1500
            or type(joint_epoch) is not int or not 0 <= joint_epoch <= 500
            or len(rows) != correction_epoch + joint_epoch
            or stage not in (CORRECTION_STAGE, JOINT_STAGE)
            or (stage == CORRECTION_STAGE and joint_epoch != 0)
            or not isinstance(initial_score, (int, float)) or not math.isfinite(initial_score)):
        raise ValueError("任11HF选分历史的初始分数/阶段/实际轮次不闭合")
    best = phase_best = recent = float(initial_score)
    best_epoch = best_local = phase_best_epoch = 0
    best_stage = CORRECTION_STAGE
    deadline = None
    validation = copy.deepcopy(initial_validation)
    phases = [(CORRECTION_STAGE, rows[:correction_epoch], 1500, 0)]
    if stage == JOINT_STAGE:
        phases.append((JOINT_STAGE, rows[correction_epoch:], 500, correction_epoch))
    for current_stage, phase_rows, limit, offset in phases:
        if current_stage == JOINT_STAGE:
            if deadline != correction_epoch:
                raise ValueError("任11HF联合日志没有本人首次合法校正截止")
            phase_best, phase_best_epoch = recent, 0
        for local, row in enumerate(phase_rows, 1):
            score = row.get("合法验证选分_摄氏度")
            if ((local % 10 == 0) != (score is not None)
                    or (score is not None and (not isinstance(score, (int, float))
                                              or not math.isfinite(score)))):
                raise ValueError("任11HF耐心重推须每10轮且仅每10轮有有限合法选分")
            if score is not None:
                recent = float(score)
                if initial_validation is not None:
                    validation = row.get("合法验证分模态")
                    if (not isinstance(validation, dict) or not validation
                            or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                                   for value in validation.values())):
                        raise ValueError("任11HF合法分模态缺少连续验证证据")
                if score < phase_best - 0.0001:
                    phase_best, phase_best_epoch = float(score), local
                if score < best - 0.0001:
                    best, best_epoch, best_stage, best_local = float(score), offset + local, current_stage, local
            if task11_phase_finished(local, phase_best_epoch, limit):
                if local != len(phase_rows):
                    raise ValueError("任11HF训练日志越过首次耐心早停或预算截止")
                if current_stage == CORRECTION_STAGE:
                    deadline = local
    return {"观测最佳全局轮次": best_epoch, "观测最佳选分_摄氏度": best,
            "观测最佳阶段": best_stage, "观测最佳阶段轮次": best_local,
            "本阶段最佳轮次": phase_best_epoch, "本阶段最佳选分": phase_best,
            "HF校正实际截止轮次": deadline, "最近合法选分_摄氏度": recent,
            "最近合法分模态": copy.deepcopy(validation)}


def _verify_full_state(saved: dict[str, Any], seed: int, identity: Mapping[str, Any]) -> None:
    meta = saved.get("metadata", {})
    correction, joint = meta.get("HF校正实际轮次", -1), meta.get("HF联合实际轮次", -1)
    stage = saved.get("stage")
    states = saved.get("model_state", {})
    named = saved.get("parameter_requires_grad", {})
    groups = saved.get("optimizer_state", {}).get("param_groups", [])
    movable = {name for name, enabled in named.items() if enabled}
    correction_names = [name for name in named if name.startswith("correction.")]
    lf_names = [name for name in named if name.startswith("low_fidelity_model.")]
    if (saved.get("training_state_schema_version") != 1 or saved.get("budget") != STAGE_BUDGET
            or stage not in (CORRECTION_STAGE, JOINT_STAGE)
            or type(correction) is not int or not 0 <= correction <= 1500
            or type(joint) is not int or not 0 <= joint <= 500
            or (stage == CORRECTION_STAGE and joint != 0)
            or saved.get("epoch") != (joint if stage == JOINT_STAGE else correction)
            or meta.get("全局实际轮次") != correction + joint or meta.get("seed") != seed
            or meta.get("身份") != dict(identity)
            or meta.get("旧固定TEST温度读取") is not False
            or meta.get("模拟测试功率温度读取") is not False
            or tuple(states.get("correction.0.weight", torch.empty(0)).shape) != (128, 6)
            or any(name.startswith(("response_features.", "surface_residual_guide.")) for name in states)
            or not correction_names or not lf_names
            or movable != set(correction_names + (lf_names if stage == JOINT_STAGE else []))
            or len(groups) != (2 if stage == JOINT_STAGE else 1)
            or meta.get("当前LF张量SHA256") != _tensor_sha(_lf_state(states))):
        raise ValueError("任11HF完整断点种子/结构/阶段/参数名单/来源不一致")
    if stage == CORRECTION_STAGE and _tensor_sha(_lf_state(states)) != identity.get("LF初始张量SHA256"):
        raise ValueError("任11HF校正阶段不能改变本seed源LF张量")
    deadline = meta.get("HF校正实际截止轮次")
    if ((stage == JOINT_STAGE and (correction < 200 or deadline != correction))
            or (stage == CORRECTION_STAGE and deadline is not None and deadline != correction)):
        raise ValueError("任11HF联合必须继承本seed真实校正截止轮次")
    local_epoch = joint if stage == JOINT_STAGE else correction
    best_global = meta.get("观测最佳全局轮次")
    best_local = meta.get("观测最佳阶段轮次")
    best_stage = meta.get("观测最佳阶段")
    phase_epoch = meta.get("本阶段最佳轮次")
    if (type(best_global) is not int or not 0 <= best_global <= correction + joint
            or type(best_local) is not int or best_local < 0
            or best_stage not in (CORRECTION_STAGE, JOINT_STAGE)
            or best_global != (correction + best_local if best_stage == JOINT_STAGE else best_local)
            or (best_stage == CORRECTION_STAGE and best_local > correction)
            or (best_stage == JOINT_STAGE and best_local > joint)
            or type(phase_epoch) is not int or not 0 <= phase_epoch <= local_epoch
            or phase_epoch % 10 != 0
            or any(not isinstance(meta.get(key), (int, float)) or not math.isfinite(meta[key])
                   for key in ("本阶段最佳选分", "观测最佳选分_摄氏度", "最近合法选分_摄氏度"))):
        raise ValueError("任11HF观测/阶段最佳选分与真实已提交轮次不一致")
    random = saved.get("random_state", {})
    if (set(random) != {"python", "numpy", "torch_cpu", "torch_cuda"}
            or any(random[key] is None for key in ("python", "numpy"))
            or not isinstance(random["torch_cpu"], torch.Tensor)
            or not isinstance(random["torch_cuda"], list) or len(random["torch_cuda"]) != 1
            or any(not isinstance(item, torch.Tensor) or item.dtype != torch.uint8
                   or item.ndim != 1 or item.numel() == 0
                   for item in [random["torch_cpu"], *random["torch_cuda"]])):
        raise ValueError("任11HF完整状态须真实保留四RNG及单卡CUDA随机字节")
    momentum = saved["optimizer_state"]["state"]
    group_names = [correction_names] + ([lf_names] if stage == JOINT_STAGE else [])
    expected_ids = []
    for index, (group, names) in enumerate(zip(groups, group_names)):
        rate = 0.001 if stage == CORRECTION_STAGE else 0.0001
        if len(group.get("params", [])) != len(names) or group.get("lr") != rate or group.get("weight_decay") != 1e-6:
            raise ValueError("任11HF校正/全LF联合AdamW组和学习率不一致")
        required = 16 * (correction + joint if index == 0 else joint)
        for identifier, name in zip(group["params"], names):
            expected_ids.append(identifier)
            record = momentum.get(identifier, {})
            if required == 0:
                if identifier in momentum:
                    raise ValueError("任11HF空初态或未更新LF不能带历史动量")
            elif (not {"step", "exp_avg", "exp_avg_sq"} <= record.keys()
                  or float(record["step"]) != required
                  or any(not isinstance(record[key], torch.Tensor)
                         or record[key].shape != states[name].shape
                         or not torch.isfinite(record[key]).all()
                         for key in ("exp_avg", "exp_avg_sq"))):
                raise ValueError("任11HF真实AdamW步数/动量张量与本人轮次不闭合")
    if set(momentum) - set(expected_ids) or any(not torch.isfinite(value).all() for value in states.values()):
        raise ValueError("任11HF状态含额外优化器来源或非有限权重")


def _receipts(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("HF会话收据_[0-9][0-9][0-9][0-9].json"))
    if not paths or any(path.name != f"HF会话收据_{number:04d}.json" for number, path in enumerate(paths, 1)):
        raise ValueError("任11HF完整会话收据必须从0001连续提交")
    return paths


def verify_task11_mlp_hf_resume(directory: Path, *, seed: int,
                              identity: Mapping[str, Any],
                              lf_row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    paths = _receipts(directory)
    latest = json.loads(paths[-1].read_text(encoding="utf-8"))
    if latest.get("状态") != "已暂停且完整HF阶段提交" or latest.get("身份") != dict(identity):
        raise ValueError("任11HF最近收据不是同seed合法暂停状态")
    for relative, digest in latest.get("原件SHA256", {}).items():
        if sha256_file(_path(directory / relative)) != digest:
            raise ValueError("任11HF最近完整断点/日志/原件SHA与收据不符")
    if not {"阶段_最近.pt", "training.jsonl", "best.pt", "阶段_观测最佳.pt"} <= set(latest.get("原件SHA256", {})):
        raise ValueError("任11HF最近提交缺完整状态/日志/最佳证据")
    saved = _load_full_state(directory / "阶段_最近.pt")
    _verify_full_state(saved, seed, identity)
    verify_task11_mlp_hf_phase_lineage(directory, saved, seed=seed, identity=identity)
    meta = saved["metadata"]
    rows = [json.loads(line) for line in (directory / "training.jsonl").read_text(encoding="utf-8").splitlines()]
    totals = audit_task11_hf_log(rows, seed, meta["HF校正实际轮次"], meta["HF联合实际轮次"])
    if (latest.get("累计实际轮次") != len(rows) or meta.get("消费累计") != totals
            or meta.get("日志SHA256") != sha256_file(directory / "training.jsonl")):
        raise ValueError("任11HF最近完整状态与本人连续日志/消费不一致")
    _audit_receipt_chain(directory, paths, identity, finished=False)
    _audit_selection_snapshots(directory, rows, seed=seed, identity=identity)
    if lf_row is not None:
        verify_task11_mlp_hf_initial_state(_load_full_state(directory / "阶段_初始.pt"), lf_row, seed)
    return saved


def _same_tree(first: Any, second: Any) -> bool:
    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        return (isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor)
                and first.dtype == second.dtype and first.shape == second.shape and torch.equal(first, second))
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return (isinstance(first, np.ndarray) and isinstance(second, np.ndarray)
                and first.dtype == second.dtype and first.shape == second.shape and np.array_equal(first, second))
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(_same_tree(value, second[key]) for key, value in first.items())
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return type(first) is type(second) and len(first) == len(second) and all(
            _same_tree(a, b) for a, b in zip(first, second))
    return type(first) is type(second) and first == second


def _load_full_state(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("任11HF校正/联合完整末状态原件缺失")
    try:
        saved = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError("任11HF完整阶段原件无法核验") from error
    if not isinstance(saved, dict):
        raise ValueError("任11HF阶段原件不是完整状态字典")
    return saved


def _audit_selection_snapshots(
    directory: Path, rows: list[dict[str, Any]], *, seed: int, identity: Mapping[str, Any],
    snapshots: Mapping[str, dict[str, Any]] | None = None,
) -> None:
    initial = _load_full_state(directory / "阶段_初始.pt")
    _verify_full_state(initial, seed, identity)
    if initial["stage"] != CORRECTION_STAGE or initial["epoch"] != 0:
        raise ValueError("任11HF选分重推必须从本人真实零轮初始状态开始")
    initial_meta = initial["metadata"]
    if snapshots is None:
        names = {"阶段_初始.pt", "阶段_观测最佳.pt", "阶段_最近.pt", "阶段_校正末.pt",
                 "阶段_联合初始.pt", "阶段_联合末.pt", "阶段_训练末.pt"}
        states = {name: _load_full_state(directory / name) for name in names if (directory / name).exists()}
    else:
        states = dict(snapshots)
    for path in directory.glob("*提交历史_[0-9][0-9][0-9][0-9].pt"):
        if path.name.startswith(("最近", "观测最佳")):
            states[path.name] = _load_full_state(path)
    raw_lines = (directory / "training.jsonl").read_bytes().splitlines(keepends=True)
    for saved in states.values():
        _verify_full_state(saved, seed, identity)
        meta = saved["metadata"]
        correction, joint = meta["HF校正实际轮次"], meta["HF联合实际轮次"]
        count = correction + joint
        prefix_rows = rows[:count]
        totals = audit_task11_hf_log(prefix_rows, seed, correction, joint)
        expected = replay_task11_mlp_hf_selection(
            prefix_rows, initial_meta["最近合法选分_摄氏度"], correction, joint,
            stage=saved["stage"], initial_validation=initial_meta["最近合法分模态"])
        if (any(not _same_tree(meta.get(key), value) for key, value in expected.items())
                or meta.get("消费累计") != totals or count > len(raw_lines)
                or meta.get("日志SHA256") != hashlib.sha256(b"".join(raw_lines[:count])).hexdigest()):
            raise ValueError("任11HF完整快照最佳/耐心/截止/消费与连续日志前缀重推不一致")


def verify_task11_mlp_hf_terminal_states(snapshots: Mapping[str, dict[str, Any]]) -> None:
    names = ("阶段_最近.pt", "阶段_联合末.pt", "阶段_训练末.pt")
    if any(name not in snapshots for name in names) or any(
            not _same_tree(snapshots[names[0]], snapshots[name]) for name in names[1:]):
        raise ValueError("任11HF最近/联合终态/训练终态须为同一完整状态，含AdamW及四RNG")


def verify_task11_mlp_hf_phase_lineage(directory: Path, recent: dict[str, Any], *,
                                     seed: int, identity: Mapping[str, Any]) -> None:
    meta = recent["metadata"]
    if meta["HF校正实际截止轮次"] is None:
        return
    terminal = _load_full_state(directory / "阶段_校正末.pt")
    _verify_full_state(terminal, seed, identity)
    end_meta = terminal["metadata"]
    correction = meta["HF校正实际轮次"]
    if (terminal["stage"] != CORRECTION_STAGE or terminal["epoch"] != correction
            or end_meta["HF校正实际截止轮次"] != correction
            or not task11_phase_finished(correction, end_meta["本阶段最佳轮次"], 1500)):
        raise ValueError("任11HF联合必须继承本人合法校正截止原件")
    if recent["stage"] == CORRECTION_STAGE:
        if not _same_tree(terminal, recent):
            raise ValueError("任11HF校正末与最近状态不是同一真实完整截止状态")
        return
    joint_initial = _load_full_state(directory / "阶段_联合初始.pt")
    _verify_full_state(joint_initial, seed, identity)
    expected_meta = copy.deepcopy(end_meta)
    expected_meta["本阶段最佳选分"] = end_meta["最近合法选分_摄氏度"]
    expected_meta["本阶段最佳轮次"] = 0
    expected_group = copy.deepcopy(terminal["optimizer_state"]["param_groups"][0])
    expected_group["lr"] = 0.0001
    if (joint_initial["stage"] != JOINT_STAGE or joint_initial["epoch"] != 0
            or not _same_tree(joint_initial["metadata"], expected_meta)
            or not _same_tree(joint_initial["model_state"], terminal["model_state"])
            or not _same_tree(joint_initial["optimizer_state"]["state"], terminal["optimizer_state"]["state"])
            or not _same_tree(joint_initial["optimizer_state"]["param_groups"][0], expected_group)
            or not _same_tree(joint_initial["random_state"], terminal["random_state"])):
        raise ValueError("任11HF联合初始须完整继承真实校正末选分/本人AdamW动量及四RNG")


def verify_task11_mlp_hf_initial_state(initial: dict[str, Any], lf_row: Mapping[str, Any], seed: int) -> None:
    fresh, _, _ = build_task11_mlp_hf_initialization(lf_row, seed)
    if (initial.get("stage") != CORRECTION_STAGE or initial.get("epoch") != 0
            or initial.get("metadata", {}).get("全局实际轮次") != 0
            or not _same_tree(initial.get("model_state", {}), fresh.state_dict())
            or initial.get("optimizer_state", {}).get("state")):
        raise ValueError("任11HF初态必须逐张量对应本seed空网络和源LF的新初始化")


def _artifact_hashes(directory: Path) -> dict[str, str]:
    fixed = {"training.jsonl", "best.pt", "final.pt", "metrics.json", "事前真实来源登记.json",
             "阶段_初始.pt", "阶段_观测最佳.pt", "阶段_最近.pt", "阶段_校正末.pt",
             "阶段_联合初始.pt", "阶段_联合末.pt", "阶段_训练末.pt"}
    result = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        registered = (relative.parts[0] in {"config_snapshot", "事前来源快照"}
                      or (len(relative.parts) == 1 and (path.name in fixed or re.fullmatch(
                          r"(?:最近|观测最佳|最佳视图)提交历史_\d{4}\.pt", path.name))))
        if registered:
            if path.is_symlink():
                raise ValueError("任11HF训练身份原件不能是符号链接")
            if path.is_file():
                result[str(relative)] = sha256_file(path)
    return result


def _audit_receipt_chain(directory: Path, paths: list[Path], identity: Mapping[str, Any],
                         *, finished: bool) -> tuple[int, float, int]:
    last = 0
    seconds = 0.0
    peak = 0
    previous_sha = None
    raw_lines = (directory / "training.jsonl").read_bytes().splitlines(keepends=True)
    for number, path in enumerate(paths, 1):
        row = json.loads(path.read_text(encoding="utf-8"))
        duration = row.get("本会话实际轮次")
        wall = row.get("本会话真实墙钟秒")
        if (row.get("身份") != dict(identity) or row.get("前驱收据SHA256") != previous_sha
                or row.get("起始已提交轮次") != last or type(duration) is not int or not 1 <= duration <= 200
                or row.get("累计实际轮次") != last + duration or last + duration > 2000
                or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0
                or type(row.get("峰值真实CUDA显存字节")) is not int or row["峰值真实CUDA显存字节"] <= 0
                or row.get("旧固定TEST温度读取") is not False
                or row.get("模拟测试功率温度读取") is not False):
            raise ValueError("任11HF真实会话收据的身份/成本/前驱/轮次不闭合")
        expected_status = "已完成新MLP HF正式训练" if finished and number == len(paths) else "已暂停且完整HF阶段提交"
        if row.get("状态") != expected_status:
            raise ValueError("任11HF不能将暂停阶段冒充正式完训")
        end = last + duration
        prefix = b"".join(raw_lines[:end])
        if (len(raw_lines) < end or hashlib.sha256(prefix).hexdigest()
                != row.get("原件SHA256", {}).get("training.jsonl")):
            raise ValueError("任11HF历史收据必须对应连续日志的原提交字节前缀")
        if number < len(paths):
            for name, original in ((f"最近提交历史_{number:04d}.pt", "阶段_最近.pt"),
                                   (f"观测最佳提交历史_{number:04d}.pt", "阶段_观测最佳.pt"),
                                   (f"最佳视图提交历史_{number:04d}.pt", "best.pt")):
                if sha256_file(_path(directory / name)) != row["原件SHA256"][original]:
                    raise ValueError("任11HF历史最近/观测最佳完整原件未按原SHA保存")
        last += duration
        seconds += float(wall)
        peak = max(peak, row.get("峰值真实CUDA显存字节", 0))
        previous_sha = sha256_file(path)
    return last, seconds, peak


@contextmanager
def _run_claim(directory: Path):
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock = directory.parent / ("." + directory.name + ".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("任11HF同seed正式事务已有进程持有，禁止并发") from error
        yield
    finally:
        os.close(descriptor)


def _save_view(path: Path, architecture: Mapping[str, Any], model: AdditiveCorrectionModel,
               metadata: Mapping[str, Any], score: float, validation: Mapping[str, float]) -> None:
    view = copy.deepcopy(dict(architecture))
    view.update(model_state={name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
                epoch=metadata["全局实际轮次"], validation_selection_score_c=score,
                validation_rmse_c=validation["顶部"], validation_sensor=dict(validation),
                任11新MLP_HF来源=copy.deepcopy(dict(metadata)),
                续跑资格="仅模型视图；不能恢复本轮AdamW或四类随机源")
    temporary = path.with_name(path.name + ".tmp")
    torch.save(view, temporary)
    os.replace(temporary, path)


def _state_metadata(identity: dict[str, Any], model: AdditiveCorrectionModel, *,
                    correction: int, joint: int, correction_done: int | None,
                    phase_best: float, phase_best_epoch: int, best: float, best_epoch: int,
                    best_stage: str, best_stage_epoch: int, score: float,
                    validation: Mapping[str, float], totals: dict[str, int], log: Path) -> dict[str, Any]:
    return {"身份": identity, "seed": identity["seed"], "HF校正实际轮次": correction,
            "HF联合实际轮次": joint, "HF校正实际截止轮次": correction_done,
            "全局实际轮次": correction + joint, "本阶段最佳选分": phase_best,
            "本阶段最佳轮次": phase_best_epoch, "观测最佳选分_摄氏度": best,
            "观测最佳全局轮次": best_epoch, "观测最佳阶段": best_stage,
            "观测最佳阶段轮次": best_stage_epoch, "最近合法选分_摄氏度": score,
            "最近合法分模态": dict(validation), "消费累计": totals,
            "当前LF张量SHA256": _tensor_sha(model.low_fidelity_model.state_dict()),
            "日志SHA256": sha256_file(log), "旧固定TEST温度读取": False,
            "模拟测试功率温度读取": False}


def run_task11_mlp_hf_formal(
    *, registry: str | Path, registry_sha: str, source_tar: str | Path, source_tar_sha: str,
    lf_catalog: str | Path, lf_catalog_sha: str, hf_data_catalog: str | Path, hf_data_catalog_sha: str,
    output: str | Path, seed: int, session_epoch_limit: int = 200, device_name: str = "cuda",
    resume_checkpoint: str | Path | None = None, project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    arguments = dict(registry=registry, registry_sha=registry_sha, source_tar=source_tar,
                     source_tar_sha=source_tar_sha, lf_catalog=lf_catalog, lf_catalog_sha=lf_catalog_sha,
                     hf_data_catalog=hf_data_catalog, hf_data_catalog_sha=hf_data_catalog_sha,
                     output=output, seed=seed, resume_checkpoint=resume_checkpoint, project_root=project_root)
    prereg = preflight_task11_mlp_hf(**arguments)
    if device_name != "cuda" or type(session_epoch_limit) is not int or not 1 <= session_epoch_limit <= 200:
        raise ValueError("任11HF正式只许真CUDA及每会话1至200轮")
    if Path(project_root).resolve() != PROJECT_ROOT:
        raise ValueError("任11HF隔离目录只供CPU预检，不能训练")
    directory = Path(prereg["输出"])
    with _run_claim(directory):
        prereg = preflight_task11_mlp_hf(**arguments)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("任11HF只许单张可见真GPU，不准CPU回退或多卡")
        return _run_session(prereg, session_epoch_limit, arguments)


def _run_session(prereg: dict[str, Any], session_limit: int,
                  preflight_arguments: dict[str, Any]) -> dict[str, Any]:
    directory = Path(prereg["输出"])
    identity, seed = prereg["身份"], prereg["seed"]
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    set_seed(seed)
    model, optimizer, architecture = build_task11_mlp_hf_initialization(prereg["LF来源"], seed, device)
    splits = build_power_splits()
    assert_no_hf_leakage({"Top": splits.hf_train, "HotCold": splits.hf_train},
                         splits.hf_validation | splits.hf_test | splits.external_sensor_test)
    # The complete ROOT/source/data gates above precede all temperature decoding.
    train = _ir_dataset("train", splits.hf_train)
    validation_data = _ir_dataset("validation", splits.hf_validation)
    sensor = _sensor_tensors(device, "train", splits.hf_train)
    validation_sensor = _sensor_tensors(device, "validation", splits.hf_validation)
    if (len(train) != 29593 or len(validation_data) != 7272
            or len(sensor[0]) != 2985 or len(validation_sensor[0]) != 752):
        raise ValueError("任11HF真实温度加载数量与冻结开发目录不一致")
    physics = PhysicsLossComputer(load_materials(), load_resolved_boundary_conditions(),
                                 PhysicsLossWeights(), temperature_scale_k=model.scales.temperature_scale_k)
    validation_loader = DataLoader(validation_data, batch_size=2048)
    log = directory / "training.jsonl"
    stage, correction, joint, correction_done = CORRECTION_STAGE, 0, 0, None
    best_stage, best_stage_epoch, best_epoch, phase_best_epoch = CORRECTION_STAGE, 0, 0, 0
    score, validation = _validation_selection(model, validation_loader, validation_sensor,
                                             device, HF_BUDGET["selection_weights"])
    if not math.isfinite(score):
        raise ValueError("任11HF初始合法选分非有限")
    best = phase_best = score
    start, number, previous_sha = 0, 1, None
    rows = []
    if "续跑状态" not in prereg:
        directory.mkdir(parents=True, exist_ok=False)
        log.write_text("", encoding="utf-8")
        write_config_snapshot(directory)
        snapshot = directory / "事前来源快照"
        snapshot.mkdir()
        for key, name in (("登记原件", "登记.yaml"), ("源码归档原件", "源码.tar.gz"),
                          ("LF目录原件", "LF目录.json"), ("HF数据目录原件", "HF开发目录.json")):
            shutil.copyfile(prereg[key], snapshot / name)
        (directory / "事前真实来源登记.json").write_text(json.dumps({"身份": identity,
            "LF来源": prereg["LF来源"], "历史HF权重/AdamW/RNG读取": False,
            "旧固定TEST温度读取": False, "模拟测试功率温度读取": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        totals = audit_task11_hf_log([], seed, 0, 0)
        meta = _state_metadata(identity, model, correction=0, joint=0, correction_done=None,
            phase_best=phase_best, phase_best_epoch=0, best=best, best_epoch=0,
            best_stage=best_stage, best_stage_epoch=0, score=score, validation=validation, totals=totals, log=log)
        for name in ("阶段_初始.pt", "阶段_观测最佳.pt", "阶段_最近.pt"):
            save_training_state(directory / name, model, optimizer, stage=stage, epoch=0,
                                budget=STAGE_BUDGET, metadata=meta)
        _save_view(directory / "best.pt", architecture, model, meta, best, validation)
    else:
        saved = prereg["续跑状态"]
        if saved["stage"] == JOINT_STAGE:
            activate_task11_mlp_joint(model, optimizer)
        load_training_state(directory / "阶段_最近.pt", model, optimizer)
        stage = saved["stage"]
        meta = saved["metadata"]
        correction, joint = meta["HF校正实际轮次"], meta["HF联合实际轮次"]
        correction_done = meta["HF校正实际截止轮次"]
        best, best_epoch = meta["观测最佳选分_摄氏度"], meta["观测最佳全局轮次"]
        best_stage, best_stage_epoch = meta["观测最佳阶段"], meta["观测最佳阶段轮次"]
        phase_best, phase_best_epoch = meta["本阶段最佳选分"], meta["本阶段最佳轮次"]
        score, validation = meta["最近合法选分_摄氏度"], meta["最近合法分模态"]
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        totals = audit_task11_hf_log(rows, seed, correction, joint)
        paths = _receipts(directory)
        number = len(paths) + 1
        previous_sha = sha256_file(paths[-1])
        start = correction + joint
        for original, target in (("阶段_最近.pt", f"最近提交历史_{number - 1:04d}.pt"),
                                 ("阶段_观测最佳.pt", f"观测最佳提交历史_{number - 1:04d}.pt"),
                                 ("best.pt", f"最佳视图提交历史_{number - 1:04d}.pt")):
            if (directory / target).exists():
                raise ValueError("任11HF历史提交副本禁止覆盖")
            shutil.copyfile(directory / original, directory / target)
    clock = time.perf_counter()
    replay = None
    completed = False
    for _ in range(session_limit):
        if stage == CORRECTION_STAGE and correction_done is not None:
            if not (directory / "阶段_校正末.pt").is_file():
                raise ValueError("任11HF未封真实校正末不能转入联合")
            phase_best, validation = transition_task11_mlp_hf_joint(model, optimizer, meta)
            stage = JOINT_STAGE
            phase_best_epoch = 0
            score = phase_best
            meta = _state_metadata(identity, model, correction=correction, joint=0,
                correction_done=correction_done, phase_best=phase_best, phase_best_epoch=0,
                best=best, best_epoch=best_epoch, best_stage=best_stage, best_stage_epoch=best_stage_epoch,
                score=score, validation=validation, totals=totals, log=log)
            save_training_state(directory / "阶段_联合初始.pt", model, optimizer, stage=stage,
                                epoch=0, budget=STAGE_BUDGET, metadata=meta)
        if stage == JOINT_STAGE and replay is None:
            replay = load_sampled_points(sorted(splits.simulation_train), 2048, 7_070_000 + seed,
                                         sampling_mode="material_time")
            if len(replay) != 60 * 2048:
                raise ValueError("任11HF联合必须60真实LF训练源各2048点")
        global_epoch = correction + joint + 1
        actual = _hf_epoch(model, optimizer, train, sensor, physics, HF_BUDGET["loss_weights"],
                           device, seed, global_epoch, replay if stage == JOINT_STAGE else None)
        if stage == CORRECTION_STAGE:
            correction += 1
            if _tensor_sha(model.low_fidelity_model.state_dict()) != identity["LF初始张量SHA256"]:
                raise ValueError("任11HF校正过程中本seed低保真权重发生改变")
        else:
            joint += 1
        phase_epoch = joint if stage == JOINT_STAGE else correction
        due = phase_epoch % 10 == 0
        improved = False
        if due:
            score, validation = _validation_selection(model, validation_loader, validation_sensor,
                                                     device, HF_BUDGET["selection_weights"])
            if not math.isfinite(score) or not all(math.isfinite(float(v)) for v in validation.values()):
                raise ValueError("任11HF合法宏选分/分模态必须有限")
            if score < phase_best - 0.0001:
                phase_best, phase_best_epoch = score, phase_epoch
            if score < best - 0.0001:
                best, best_epoch, best_stage, best_stage_epoch = score, global_epoch, stage, phase_epoch
                improved = True
        boundary = task11_phase_finished(phase_epoch, phase_best_epoch, 500 if stage == JOINT_STAGE else 1500)
        if boundary and stage == CORRECTION_STAGE:
            correction_done = correction
        row = {"epoch": global_epoch, "seed": seed, "阶段": stage, "阶段轮次": phase_epoch,
               "HF观测点": actual["HF训练观测点"], "HF传感器点": actual["HF训练传感器点"],
               "HF观测优化步": 15, "物理优化步": 1, "物理配点": 256,
               "LF回放点": actual["LF真实回放训练点"], "LF回放小批": actual["LF联合回放batch"],
               "合法验证选分_摄氏度": score if due else None,
               "合法验证分模态": validation if due else None,
               "名义物理训练分项": actual["名义物理训练分项"],
               "旧固定TEST温度读取": False, "模拟测试功率温度读取": False}
        rows.append(row)
        totals = audit_task11_hf_log(rows, seed, correction, joint)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        meta = _state_metadata(identity, model, correction=correction, joint=joint,
            correction_done=correction_done, phase_best=phase_best, phase_best_epoch=phase_best_epoch,
            best=best, best_epoch=best_epoch, best_stage=best_stage, best_stage_epoch=best_stage_epoch,
            score=score, validation=validation, totals=totals, log=log)
        save_training_state(directory / "阶段_最近.pt", model, optimizer, stage=stage,
                            epoch=phase_epoch, budget=STAGE_BUDGET, metadata=meta)
        if improved:
            save_training_state(directory / "阶段_观测最佳.pt", model, optimizer, stage=stage,
                                epoch=phase_epoch, budget=STAGE_BUDGET, metadata=meta)
            _save_view(directory / "best.pt", architecture, model, meta, best, validation)
        if boundary:
            if stage == CORRECTION_STAGE:
                save_training_state(directory / "阶段_校正末.pt", model, optimizer, stage=stage,
                                    epoch=phase_epoch, budget=STAGE_BUDGET, metadata=meta)
            else:
                for name in ("阶段_联合末.pt", "阶段_训练末.pt"):
                    save_training_state(directory / name, model, optimizer, stage=stage,
                                        epoch=phase_epoch, budget=STAGE_BUDGET, metadata=meta)
                _save_view(directory / "final.pt", architecture, model, meta, score, validation)
                completed = True
                break
    seconds = time.perf_counter() - clock
    # Recheck every frozen byte before claiming the session is submitted.
    checked = dict(preflight_arguments, resume_checkpoint=None, qualified_source_only=True)
    preflight_task11_mlp_hf(**checked)
    receipt = {"身份": identity, "起始已提交轮次": start, "本会话实际轮次": correction + joint - start,
               "累计实际轮次": correction + joint, "HF校正实际轮次": correction, "HF联合实际轮次": joint,
               "消费累计": totals, "本会话真实墙钟秒": seconds,
               "峰值真实CUDA显存字节": torch.cuda.max_memory_allocated(device),
               "前驱收据SHA256": previous_sha, "旧固定TEST温度读取": False,
               "模拟测试功率温度读取": False,
               "状态": "已完成新MLP HF正式训练" if completed else "已暂停且完整HF阶段提交"}
    if completed:
        costs = [json.loads(path.read_text(encoding="utf-8"))["本会话真实墙钟秒"]
                 for path in directory.glob("HF会话收据_*.json")]
        metrics = {"method": "mlp_pinn_additive", "seed": seed, "status": "completed_current_protocol_hf",
                   "configuration": HF_BUDGET, "source_identity": identity,
                   "correction_epochs_completed": correction, "joint_epochs_completed": joint,
                   "best_epoch": best_epoch, "best_validation_selection_score_c": best,
                   "training_seconds": sum(costs) + seconds,
                   "training_seconds_contract": "本人所有真实HF会话之和；LF成本另列",
                   "lf_training_seconds": prereg["LF来源"]["LF真实多会话累计成本秒"],
                   "stopping_reason": "joint_budget" if joint == 500 else "joint_validation_patience",
                   "consumption": totals, "test": None, "旧固定TEST温度读取": False,
                   "模拟测试功率温度读取": False, "energy_used_for_selection": False}
        (directory / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt["原件SHA256"] = _artifact_hashes(directory)
    destination = directory / f"HF会话收据_{number:04d}.json"
    if destination.exists():
        raise ValueError("任11HF真实会话收据禁止覆盖")
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def verify_task11_mlp_hf_metrics_identity(
    metrics: Mapping[str, Any], *, seed: int, lf_seconds: float,
) -> None:
    recorded = metrics.get("lf_training_seconds")
    if (metrics.get("method") != "mlp_pinn_additive" or type(metrics.get("seed")) is not int
            or metrics["seed"] != seed or metrics.get("energy_used_for_selection") is not False
            or type(recorded) not in (int, float) or not math.isfinite(recorded)
            or not math.isclose(recorded, lf_seconds, abs_tol=1e-7, rel_tol=1e-12)):
        raise ValueError("任11HF指标method/种子/本人LF真实成本/能源选模隔离不一致")


def verify_task11_mlp_hf_view_statistics(
    view: Mapping[str, Any], snapshot: Mapping[str, Any],
) -> None:
    meta = snapshot.get("metadata", {})
    validation = meta.get("最近合法分模态")
    if (view.get("method") != "multifidelity_correction" or not isinstance(validation, dict)
            or "顶部" not in validation
            or not _same_tree(view.get("validation_selection_score_c"), meta.get("最近合法选分_摄氏度"))
            or not _same_tree(view.get("validation_sensor"), validation)
            or not _same_tree(view.get("validation_rmse_c"), validation["顶部"])):
        raise ValueError("任11HF模型视图method/选分/分模态/顶部统计与完整快照不一致")


def qualify_task11_mlp_hf_source(**arguments: Any) -> dict[str, Any]:
    if arguments.get("resume_checkpoint") is not None:
        raise ValueError("任11HF已完成审计不能同时恢复训练")
    prereg = preflight_task11_mlp_hf(**arguments, qualified_source_only=True)
    directory, identity = Path(prereg["输出"]), prereg["身份"]
    paths = _receipts(directory)
    total_epochs, seconds, peak = _audit_receipt_chain(directory, paths, identity, finished=True)
    latest = json.loads(paths[-1].read_text(encoding="utf-8"))
    for relative, digest in latest["原件SHA256"].items():
        if sha256_file(_path(directory / relative)) != digest:
            raise ValueError("任11HF完成原件SHA与末会话收据不一致")
    required = {"阶段_初始.pt", "阶段_观测最佳.pt", "阶段_最近.pt", "阶段_校正末.pt",
                "阶段_联合初始.pt", "阶段_联合末.pt", "阶段_训练末.pt", "best.pt", "final.pt",
                "training.jsonl", "metrics.json", "事前真实来源登记.json"}
    if not required <= set(latest["原件SHA256"]):
        raise ValueError("任11HF正式完训缺阶段完整状态/最佳/真末/日志/成本")
    snapshots = {name: _load_full_state(directory / name)
                 for name in required if name.startswith("阶段_")}
    for saved in snapshots.values():
        _verify_full_state(saved, prereg["seed"], identity)
    verify_task11_mlp_hf_initial_state(snapshots["阶段_初始.pt"], prereg["LF来源"], prereg["seed"])
    recent, final = snapshots["阶段_最近.pt"], snapshots["阶段_训练末.pt"]
    verify_task11_mlp_hf_terminal_states(snapshots)
    meta = final["metadata"]
    verify_task11_mlp_hf_phase_lineage(directory, final, seed=prereg["seed"], identity=identity)
    correction, joint = meta["HF校正实际轮次"], meta["HF联合实际轮次"]
    if (recent["stage"] != JOINT_STAGE or recent["epoch"] != joint
            or correction + joint != total_epochs
            or not task11_phase_finished(joint, meta["本阶段最佳轮次"], 500)
            or snapshots["阶段_校正末.pt"]["epoch"] != correction
            or not task11_phase_finished(correction, snapshots["阶段_校正末.pt"]["metadata"]["本阶段最佳轮次"], 1500)):
        raise ValueError("任11HF校正/联合真实截止与最近/真末状态不闭合")
    rows = [json.loads(line) for line in (directory / "training.jsonl").read_text(encoding="utf-8").splitlines()]
    totals = audit_task11_hf_log(rows, prereg["seed"], correction, joint)
    _audit_selection_snapshots(directory, rows, seed=prereg["seed"], identity=identity, snapshots=snapshots)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    verify_task11_mlp_hf_metrics_identity(
        metrics, seed=prereg["seed"], lf_seconds=prereg["LF来源"]["LF真实多会话累计成本秒"])
    if (metrics.get("configuration") != HF_BUDGET or metrics.get("source_identity") != identity
            or metrics.get("status") != "completed_current_protocol_hf" or metrics.get("test") is not None
            or metrics.get("correction_epochs_completed") != correction
            or metrics.get("joint_epochs_completed") != joint or metrics.get("consumption") != totals
            or metrics.get("旧固定TEST温度读取") is not False
            or metrics.get("模拟测试功率温度读取") is not False
            or not math.isclose(metrics.get("training_seconds", math.nan), seconds, abs_tol=1e-7, rel_tol=1e-12)):
        raise ValueError("任11HF完训成本/预算/真实消费/TEST隔离不闭合")
    for name, snapshot in (("best.pt", snapshots["阶段_观测最佳.pt"]), ("final.pt", final)):
        view = torch.load(directory / name, map_location="cpu", weights_only=True)
        validate_hf_checkpoint_provenance(view)
        verify_task11_mlp_hf_view_statistics(view, snapshot)
        if (view.get("seed") != prereg["seed"] or view.get("low_fidelity_method") != "mlp_pinn"
                or view.get("low_fidelity_model_kwargs") != LF_KWARGS or view.get("scales") != HF_BUDGET["scales"]
                or view.get("correction_model_kwargs") != CORRECTION_KWARGS
                or view.get("任11新MLP_HF来源") != snapshot["metadata"]
                or view.get("epoch") != snapshot["metadata"]["全局实际轮次"]
                or set(view.get("model_state", {})) != set(snapshot["model_state"])
                or any(not torch.equal(value, snapshot["model_state"][key]) for key, value in view["model_state"].items())):
            raise ValueError("任11HF观测best/final视图与完整状态/结构/来源不一致")
    best_meta = snapshots["阶段_观测最佳.pt"]["metadata"]
    if (best_meta["全局实际轮次"] != meta["观测最佳全局轮次"]
            or best_meta["观测最佳选分_摄氏度"] != metrics.get("best_validation_selection_score_c")
            or meta["观测最佳全局轮次"] != metrics.get("best_epoch")):
        raise ValueError("任11HF观测最佳轮次和合法选分缺真实对应")
    originals = _artifact_hashes(directory)
    originals.update({str(path.relative_to(directory)): sha256_file(path) for path in paths})
    return {"状态": "CPU完整HF来源资格PASS；独立能源仍须另行审核", "seed": prereg["seed"],
            "登记原件": prereg["登记原件"], "登记SHA256": prereg["YAML_SHA256"],
            "源码归档原件": prereg["源码归档原件"], "源码归档SHA256": prereg["TAR_SHA256"],
            "LF目录原件": prereg["LF目录原件"], "LF目录SHA256": prereg["LF_CATALOG_SHA256"],
            "HF数据目录原件": prereg["HF数据目录原件"], "HF数据目录SHA256": prereg["HF_DATA_CATALOG_SHA256"],
            "本seed新LF最佳检查点原件": str(PROJECT_ROOT / prereg["LF来源"]["最佳检查点"]),
            "本seed新LF最佳检查点SHA256": identity["LF最佳检查点SHA256"],
            "本seedLF初始张量SHA256": identity["LF初始张量SHA256"],
            "HF校正实际轮次": correction, "HF联合实际轮次": joint,
            "观测最佳全局轮次": meta["观测最佳全局轮次"],
            "合法HF观测最佳选分_摄氏度": meta["观测最佳选分_摄氏度"],
            "本人HF累计真实成本秒": seconds, "本人LF累计真实成本秒": metrics["lf_training_seconds"],
            "全部会话CUDA峰值显存字节": peak, "完整原件SHA256": originals,
            "config_snapshot目录": str(directory / "config_snapshot"),
            "目录": str(directory), "旧固定TEST温度读取": False, "模拟测试功率温度读取": False}


def require_task11_mlp_hf_downstream_qualification(*, purpose: str, **arguments: Any) -> dict[str, Any]:
    if purpose not in {"best", "final"}:
        raise ValueError("任11HF独立下游只许观测best或真实final")
    audited = qualify_task11_mlp_hf_source(**arguments)
    model = Path(audited["目录"]) / f"{purpose}.pt"
    return {**audited, "模型原件": str(model), "模型SHA256": sha256_file(model),
            "选择状态": purpose, "身份审计": audited}
