"""Independent Howard LF teacher under the existing finite Task-11 LF budget.

This is a project-specific LF pretraining adaptation, not the paper's exact
three-network simultaneous training procedure. It creates no HF subnet or
dummy HF query point and grants no subsequent HF training permission.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tarfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch import Tensor, nn

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import build_power_splits
from sic_cu.eval.protocol_checks import (
    checkpoint_provenance, current_protocol_fingerprints,
    validate_lf_checkpoint_provenance,
)
from sic_cu.models.common import CoordinateScaler, ModelScales, parameter_count
from sic_cu.models.task11_howard_composite import _ModifiedDeepONet
from sic_cu.physics.collocation import sample_collocation
from sic_cu.physics.resolution import resolve_physics_state
from sic_cu.train.common import (
    CONFIG_FILES, load_training_state, physics_optimizer_step,
    save_training_state, write_config_snapshot,
)
from sic_cu.train.simulation import (
    _physics_ready, _rank_indices, _validation_sums, load_sampled_points, set_seed,
)
from sic_cu.train.task11_mlp_lf_formal import (
    LF_BUDGET as MLP_LF_BUDGET,
    SOURCE_MEMBERS as MLP_LF_SOURCE_MEMBERS,
    collect_task11_mlp_lf_sources, require_finite_lf_metrics,
)


ROOT_LEDGER = "多保真DeepONet预测精度优化总计划与执行台账.md"
ROOT_TOKEN = "TASK11_HOWARD_LF_GATE:v1"
REGISTRY = "研究记录/任务11_外部对照/Howard适配独立LF预算前登记.yaml"
RUN_DIRECTORY = "研究记录/任务11_外部对照/正式Howard适配公平训练"
METHOD = "task11_howard_lf_teacher"
TRAINING_STAGE = "howard_low_fidelity"
LF_MODEL_KWARGS = {"width": 128, "depth": 4, "latent_dim": 128,
                   "final_activation": False}
LF_PARAMETER_COUNT = 100864
MIN_DELTA_C = 0.0001
ADAMW_CONTRACT = {"lr": 0.001, "weight_decay": 1e-6, "betas": [0.9, 0.999],
                  "eps": 1e-8, "amsgrad": False, "maximize": False,
                  "capturable": False, "differentiable": False,
                  "foreach": None, "fused": None}
LF_BUDGET = {**MLP_LF_BUDGET, "method": METHOD,
             "model_kwargs": dict(LF_MODEL_KWARGS)}
SOURCE_MEMBERS = tuple(dict.fromkeys((
    "src/sic_cu/train/task11_howard_lf_formal.py",
    "scripts/54_run_task11_howard_lf_formal.py",
    "tests/test_task11_howard_lf_formal.py",
    "src/sic_cu/models/task11_howard_composite.py",
    MLP_LF_SOURCE_MEMBERS[0], *MLP_LF_SOURCE_MEMBERS[3:],
)))
ISOLATION_FLAGS = ("旧固定TEST温度读取", "模拟测试功率温度读取", "HF训练许可")
IDENTITY_FIELDS = ("YAML_SHA256", "TAR_SHA256", "CATALOG_SHA256")
SNAPSHOT_NAMES = ("config_snapshot/sha256.json", "config_snapshot/resolved_physics.yaml",
                  "config_snapshot/" + ROOT_LEDGER,
                  *("config_snapshot/" + Path(name).name for name in CONFIG_FILES))


class Task11HowardLFTeacher(nn.Module):
    """Only equations (2)-(6)'s fixed LF modified DeepONet, in Kelvin."""

    def __init__(self, *, scales: ModelScales | Mapping[str, float] = ModelScales(),
                 width: int = 128, depth: int = 4, latent_dim: int = 128,
                 final_activation: bool = False) -> None:
        super().__init__()
        if not isinstance(scales, ModelScales):
            try:
                scales = ModelScales(**dict(scales))
            except (TypeError, ValueError) as error:
                raise ValueError("Howard LF只能使用固定ModelScales尺度") from error
        if (scales != ModelScales() or type(width) is not int or width != 128
                or type(depth) is not int or depth != 4
                or type(latent_dim) is not int or latent_dim != 128
                or final_activation is not False):
            raise ValueError("Howard LF构造参数/尺度固定，禁止临时改配置或初始化HF")
        self.scales = scales
        self.scaler = CoordinateScaler(scales)
        self.network = _ModifiedDeepONet(1, 4, width, depth, latent_dim,
                                         final_activation=False)

    def forward(self, coordinates: Tensor, fidelity: str = "low") -> Tensor:
        if fidelity != "low":
            raise ValueError("Howard LF teacher仅接受low，不包含或许可HF网络")
        if (not isinstance(coordinates, Tensor) or coordinates.ndim != 2
                or coordinates.shape[1] != 5 or not coordinates.is_floating_point()
                or not torch.isfinite(coordinates).all()
                or not ((coordinates[:, 4] == 0) | (coordinates[:, 4] == 1)).all()):
            raise ValueError("Howard LF输入须为有限五列[r,z,t,power,material]，材料0或1")
        scaled = self.scaler(coordinates)
        output = self.network(scaled[:, 3:4], scaled[:, [0, 1, 2, 4]])
        return self.scales.temperature_offset_k + self.scales.temperature_scale_k * output

    @property
    def model_kwargs(self) -> dict[str, Any]:
        return dict(LF_MODEL_KWARGS)

    def parameter_count(self) -> int:
        return parameter_count(self)


def _project_path(value: str | Path, root: Path) -> Path:
    candidate = Path(value)
    lexical = candidate if candidate.is_absolute() else root / candidate
    if ".." in lexical.parts:
        raise ValueError("Howard LF文件禁止父路径跳转或项目越界")
    absolute = lexical.resolve()
    if absolute == root or root not in absolute.parents:
        raise ValueError("Howard LF登记与训练工件仅允许项目内具体路径")
    for item in (lexical, *lexical.parents):
        if item.is_symlink():
            raise ValueError("Howard LF文件及父目录禁止符号链接")
        if item == root:
            break
    return absolute


def _locked_project_file(value: str | Path, root: Path) -> Path:
    absolute = _project_path(value, root)
    if not absolute.is_file():
        raise ValueError("Howard LF登记与冻结文件必须为项目内真实普通文件")
    return absolute


def _root_active(ledger: Path, registry_sha: str, tar_sha: str, catalog_sha: str) -> bool:
    expected = (f"{ROOT_TOKEN}; status=active; YAML_SHA256={registry_sha}; "
                f"TAR_SHA256={tar_sha}; CATALOG_SHA256={catalog_sha}")
    rows, record_numbers = [], []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        fields = [value.strip() for value in line.split("|")]
        if len(fields) > 3 and re.fullmatch(r"录-\d{4}", fields[1]):
            number = int(fields[1][2:])
            record_numbers.append(number)
            if fields[2].startswith(ROOT_TOKEN):
                rows.append((number, fields[2]))
    return (len(rows) == 1 and rows[0][0] > 101 and rows[0][1] == expected
            and record_numbers.count(rows[0][0]) == 1)


def _validate_registry_contract(budget: Any, *, source_tar_sha: str,
                                catalog_sha: str) -> dict[str, Any]:
    if (not isinstance(budget, dict) or budget.get("schema_version") != 1
            or budget.get("阶段") != "fresh_howard_adapted_low_fidelity"
            or budget.get("ROOT门禁标签") != ROOT_TOKEN
            or any(budget.get(name) is not False for name in ISOLATION_FLAGS)
            or budget.get("新HF训练许可") is not False
            or budget.get("原文精确三网联合训练复现") is not False
            or budget.get("源码冻结tarSHA256") != source_tar_sha
            or budget.get("真实模拟70源目录SHA256") != catalog_sha
            or not _same(budget.get("正式预算"), LF_BUDGET)
            or not _same(budget.get("LF模型构造参数"), LF_MODEL_KWARGS)
            or budget.get("LF参数量") != LF_PARAMETER_COUNT
            or budget.get("模型尺度") != asdict(ModelScales())
            or budget.get("最小改善_摄氏度") != MIN_DELTA_C
            or budget.get("AdamW固定参数") != ADAMW_CONTRACT
            or not isinstance(budget.get("源码普通成员SHA256"), dict)
            or set(budget["源码普通成员SHA256"]) != set(SOURCE_MEMBERS)):
        raise ValueError("Howard LF仅允许固定五seed真实LF预算适配，不能借MLP门禁或许可HF")
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in budget["源码普通成员SHA256"].values()):
        raise ValueError("Howard LF冻结源码成员SHA256必须全部真实完整")
    return budget


def assert_task11_howard_lf_source_unchanged(prereg: Mapping[str, Any]) -> None:
    root = Path(prereg.get("项目根", PROJECT_ROOT)).resolve()
    path = _locked_project_file(prereg["真实源目录原件"], root)
    if (sha256_file(path) != prereg["CATALOG_SHA256"]
            or json.loads(path.read_text(encoding="utf-8")) != collect_task11_mlp_lf_sources()):
        raise ValueError("Howard LF实际70源在前置验收后发生数据来源SHA漂移")


def _same(first: Any, second: Any) -> bool:
    if isinstance(first, bool) or isinstance(second, bool):
        return type(first) is bool and type(second) is bool and first is second
    if isinstance(first, Tensor) or isinstance(second, Tensor):
        return (isinstance(first, Tensor) and isinstance(second, Tensor)
                and first.dtype == second.dtype and torch.equal(first.cpu(), second.cpu()))
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return (isinstance(first, np.ndarray) and isinstance(second, np.ndarray)
                and first.dtype == second.dtype and np.array_equal(first, second))
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(_same(value, second[name]) for name, value in first.items())
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(_same(a, b) for a, b in zip(first, second))
    return type(first) is type(second) and first == second


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def _history(rows: list[dict[str, Any]], seed: int) -> tuple[int, float | None, int]:
    best_epoch, best_score, patience = 0, math.inf, 0
    for epoch, row in enumerate(rows, 1):
        if (not isinstance(row, dict) or row.get("epoch") != epoch or row.get("seed") != seed
                or row.get("训练阶段") != "howard_adapted_low_fidelity"
                or row.get("LF训练功率数") != 60 or row.get("LF合法验证功率数") != 10
                or row.get("训练样本暴露") != 60 * 8192
                or row.get("观测优化步") != 60 or row.get("物理优化步") != 1
                or row.get("物理配点") != 256 or row.get("累计训练点") != 60 * 8192 * epoch
                or row.get("累计优化步") != 61 * epoch or row.get("learning_rate") != 0.001
                or any(row.get(name) is not False for name in ISOLATION_FLAGS)
                or any(not _finite_number(row.get(name)) for name in
                       ("train_rmse_c", "validation_rmse_c", "validation_mae_c"))
                or not isinstance(row.get("LF原物理损失"), dict)
                or not row["LF原物理损失"]
                or any(not _finite_number(value) for value in row["LF原物理损失"].values())):
            raise ValueError("Howard LF逐轮60×8192、验证、名义物理、优化步或TEST隔离不闭合")
        score = row["validation_rmse_c"]
        if score < best_score - MIN_DELTA_C:
            best_epoch, best_score, patience = epoch, score, 0
        else:
            patience += 1
        if patience >= 200 and epoch != len(rows):
            raise ValueError("Howard LF历史已达到耐心停止，不得隐藏后续额外训练")
    return best_epoch, (best_score if rows else None), patience


def _audit_rng(state: Any) -> None:
    if not isinstance(state, dict) or state.keys() != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Howard LF完整阶段必须包含四类真实RNG状态")
    try:
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        cpu = state["torch_cpu"]
        if not isinstance(cpu, Tensor) or cpu.dtype != torch.uint8 or cpu.ndim != 1:
            raise ValueError("CPU RNG格式错误")
        torch.Generator(device="cpu").set_state(cpu.cpu())
        cuda = state["torch_cuda"]
        if (not isinstance(cuda, (list, tuple)) or len(cuda) != 1
                or not isinstance(cuda[0], Tensor) or cuda[0].dtype != torch.uint8
                or cuda[0].ndim != 1 or cuda[0].numel() != 16):
            raise ValueError("CUDA RNG不得为空或多卡")
    except (ValueError, TypeError, RuntimeError, KeyError, IndexError) as error:
        raise ValueError("Howard LF四类RNG状态必须可恢复；CPU诊断不能冒充CUDA正式状态") from error


def _audit_stage(stage: Any, rows: list[dict[str, Any]], seed: int,
                 identity: Mapping[str, str], *, log_sha: str) -> None:
    epoch = len(rows)
    best_epoch, score, patience = _history(rows, seed)
    if not isinstance(stage, dict):
        raise ValueError("Howard LF完整阶段必须为真实状态字典")
    metadata = stage.get("metadata", {})
    if (stage.get("training_state_schema_version") != 1 or stage.get("stage") != TRAINING_STAGE
            or stage.get("epoch") != epoch or stage.get("budget") != {"LF轮次": 2000}
            or stage.get("scheduler_state") is not None or stage.get("sampler_epochs") != {}
            or not isinstance(metadata, dict) or metadata.get("seed") != seed
            or metadata.get("LF方法") != METHOD
            or not _same(metadata.get("LF模型构造参数"), LF_MODEL_KWARGS)
            or metadata.get("LF参数量") != LF_PARAMETER_COUNT
            or metadata.get("模型尺度") != asdict(ModelScales())
            or metadata.get("当前轮次") != epoch or metadata.get("累计训练点") != 60 * 8192 * epoch
            or metadata.get("累计优化步") != 61 * epoch
            or metadata.get("最佳LF验证轮次") != best_epoch
            or metadata.get("最佳LF验证RMSE_摄氏度") != score
            or metadata.get("无改善轮次") != patience or metadata.get("日志SHA256") != log_sha
            or metadata.get("旧LF/HF权重读取") is not False
            or any(metadata.get(name) is not False for name in ISOLATION_FLAGS)
            or any(metadata.get(name) != value for name, value in identity.items())):
        raise ValueError("Howard LF阶段seed/构造/来源/真实步数/历史最佳或耐心计数伪装")
    _audit_rng(stage.get("random_state"))
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        teacher = Task11HowardLFTeacher()
    named = dict(teacher.named_parameters())
    weights = stage.get("model_state", {})
    if (not isinstance(weights, dict) or weights.keys() != teacher.state_dict().keys()
            or any(not isinstance(value, Tensor) or value.shape != named[name].shape
                   or value.dtype != named[name].dtype or not torch.isfinite(value).all()
                   for name, value in weights.items())
            or not _same(stage.get("parameter_requires_grad"), {name: True for name in named})):
        raise ValueError("Howard LF完整网络张量/参数名/尺寸/可训练标志与固定teacher不符")
    if not epoch and not _same(weights, teacher.state_dict()):
        raise ValueError("Howard LF初始权重必须逐字等于注册seed的CPU fork_rng空网络重建")
    optimizer = stage.get("optimizer_state")
    groups = optimizer.get("param_groups", []) if isinstance(optimizer, dict) else []
    if (len(groups) != 1 or not _same(groups[0].get("params"), list(range(len(named))))
            or any(not _same(groups[0].get(name), value) for name, value in ADAMW_CONTRACT.items())):
        raise ValueError("Howard LF完整AdamW参数映射或固定算法超参数不符")
    states = optimizer.get("state")
    if not isinstance(states, dict) or set(states) != (set(range(len(named))) if epoch else set()):
        raise ValueError("Howard LF初始必须为空AdamW，训练后每个LF参数必须有真实动量")
    for index, (name, parameter) in enumerate(named.items()):
        if not epoch:
            continue
        item = states[index]
        if (not isinstance(item, dict) or set(item) != {"step", "exp_avg", "exp_avg_sq"}
                or not isinstance(item["step"], Tensor) or item["step"].numel() != 1
                or not torch.isfinite(item["step"]).all() or float(item["step"]) != 61 * epoch
                or any(not isinstance(item[key], Tensor) or item[key].shape != parameter.shape
                       or item[key].dtype != weights[name].dtype or not torch.isfinite(item[key]).all()
                       for key in ("exp_avg", "exp_avg_sq"))
                or (item["exp_avg_sq"] < 0).any()):
            raise ValueError("Howard LF真实AdamW每参数步数/动量/平方动量与61步每轮不符")


def _audit_view(view: Any, selected: dict[str, Any], rows: list[dict[str, Any]], seed: int,
                identity: Mapping[str, str]) -> None:
    epoch, score, _ = _history(rows, seed)
    if (not isinstance(view, dict) or view.get("schema_version") != 1
            or view.get("method") != METHOD or view.get("seed") != seed
            or view.get("epoch") != epoch or view.get("validation_rmse_c") != score
            or not _same(view.get("model_kwargs"), LF_MODEL_KWARGS)
            or view.get("scales") != asdict(ModelScales()) or view.get("LF参数量") != LF_PARAMETER_COUNT
            or view.get("任11Howard事前来源") != dict(identity)
            or any(view.get(name) is not False for name in ISOLATION_FLAGS)
            or not _same(view.get("model_state"), selected["model_state"])
            or not _same(view.get("lf_subnet_state"), {
                name.removeprefix("network."): value for name, value in selected["model_state"].items()})):
        raise ValueError("Howard LF best视图必须等于真实完整最佳状态且可严格迁移LF子网")


def audit_task11_howard_lf_artifacts(
    objects: Mapping[str, Any], hashes: Mapping[str, str], *, seed: int,
    source_identity: Mapping[str, str], finished: bool,
) -> dict[str, Any]:
    """Recompute a paused/finished state chain from already CPU-loaded artifacts."""
    try:
        rows = objects["training.jsonl"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 2000:
            raise ValueError("Howard LF缺真实连续训练日志")
        best_epoch, score, patience = _history(rows, seed)
        receipts = sorted(name for name in objects if re.fullmatch(r"LF会话收据_\d{4}\.json", name))
        if not receipts:
            raise ValueError("Howard LF缺真实连续分段提交收据")
        if set(source_identity) != set(IDENTITY_FIELDS):
            raise ValueError("Howard LF三SHA身份必须独立且完整")
        for name in hashes:
            if not re.fullmatch(r"[0-9a-f]{64}", hashes[name]):
                raise ValueError("Howard LF真实工件SHA格式不完整")
        _audit_stage(objects["阶段_初始.pt"], [], seed, source_identity,
                     log_sha=hashlib.sha256(b"").hexdigest())
        best_rows = rows[:best_epoch]
        selected = objects["阶段_LF观测最佳.pt"]
        best_prefix = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in best_rows).encode()
        _audit_stage(selected, best_rows, seed, source_identity,
                     log_sha=hashlib.sha256(best_prefix).hexdigest())
        _audit_stage(objects["阶段_最近.pt"], rows, seed, source_identity,
                     log_sha=hashes["training.jsonl"])
        _audit_view(objects["best.pt"], selected, rows, seed, source_identity)
        seconds, setup_seconds, export_seconds, total_seconds = 0.0, 0.0, 0.0, 0.0
        peak, last_epoch = 0, 0
        snapshot_hashes = {name: hashes[name] for name in SNAPSHOT_NAMES}
        if any(name not in objects for name in SNAPSHOT_NAMES):
            raise ValueError("Howard LF缺实际配置副本与名义物理快照")
        for number, name in enumerate(receipts, 1):
            if name != f"LF会话收据_{number:04d}.json":
                raise ValueError("Howard LF提交收据必须连续编号且不可覆盖")
            receipt = objects[name]
            end = receipt.get("累计实际轮次")
            duration = receipt.get("本会话实际轮次")
            wall = receipt.get("本会话真实墙钟秒")
            setup = receipt.get("本会话加载构建与恢复墙钟秒")
            exported = receipt.get("本会话导出与源核验墙钟秒")
            total = receipt.get("本会话加载训练与导出总墙钟秒")
            memory = receipt.get("峰值真实CUDA显存字节")
            final_receipt = number == len(receipts)
            expected_status = ("已完成Howard适配LF正式训练" if finished and final_receipt
                               else "已暂停且完整阶段提交")
            if (receipt.get("seed") != seed or receipt.get("LF方法") != METHOD
                    or any(receipt.get(key) != value for key, value in source_identity.items())
                    or any(receipt.get(key) is not False for key in ISOLATION_FLAGS)
                    or receipt.get("状态") != expected_status
                    or receipt.get("起始已提交轮次") != last_epoch
                    or type(duration) is not int or not 1 <= duration <= 200
                    or type(end) is not int or end != last_epoch + duration or end > len(rows)
                    or receipt.get("正式预算上限轮次") != 2000
                    or receipt.get("LF累计真实训练点") != 60 * 8192 * end
                    or receipt.get("LF累计真实优化步") != 61 * end
                    or not _finite_number(wall, positive=True)
                    or not _finite_number(setup) or not _finite_number(exported)
                    or not _finite_number(total, positive=True)
                    or not math.isclose(total, wall + setup + exported, rel_tol=1e-12, abs_tol=1e-7)
                    or receipt.get("配置快照SHA256") != snapshot_hashes
                    or type(memory) is not int or memory <= 0):
                raise ValueError("Howard LF全部分段身份/来源/1—200预算/真实成本不闭合")
            committed = {"日志SHA256": f"已提交日志_{number:04d}.jsonl",
                         "最近阶段SHA256": f"最近提交历史_{number:04d}.pt",
                         "观测最佳阶段SHA256": f"观测最佳提交历史_{number:04d}.pt",
                         "观测最佳模型SHA256": f"观测最佳模型历史_{number:04d}.pt",
                         "初始阶段SHA256": "阶段_初始.pt"}
            if any(receipt.get(key) != hashes[filename] for key, filename in committed.items()):
                raise ValueError("Howard LF不可覆盖历史完整状态/最佳/日志前缀原SHA不符")
            prefix = objects[committed["日志SHA256"]]
            if not _same(prefix, rows[:end]):
                raise ValueError("Howard LF历史提交日志不是当前同seed日志的真实前缀")
            previous_best, previous_score, previous_patience = _history(prefix, seed)
            if (receipt.get("观测最佳LF轮次") != previous_best
                    or receipt.get("观测最佳LF验证RMSE_摄氏度") != previous_score
                    or receipt.get("无改善轮次") != previous_patience
                    or (not (finished and final_receipt) and (previous_patience >= 200 or end >= 2000))):
                raise ValueError("Howard LF分段真实历史最佳、耐心或完成身份不符")
            _audit_stage(objects[committed["最近阶段SHA256"]], prefix, seed, source_identity,
                         log_sha=hashes[committed["日志SHA256"]])
            historical_best = objects[committed["观测最佳阶段SHA256"]]
            historical_rows = prefix[:previous_best]
            encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in historical_rows).encode()
            _audit_stage(historical_best, historical_rows, seed, source_identity,
                         log_sha=hashlib.sha256(encoded).hexdigest())
            _audit_view(objects[committed["观测最佳模型SHA256"]], historical_best,
                        prefix, seed, source_identity)
            if final_receipt:
                for key, current in (("日志SHA256", "training.jsonl"),
                                     ("最近阶段SHA256", "阶段_最近.pt"),
                                     ("观测最佳阶段SHA256", "阶段_LF观测最佳.pt"),
                                     ("观测最佳模型SHA256", "best.pt")):
                    if receipt[key] != hashes[current]:
                        raise ValueError("Howard LF当前状态必须等于本人最近已提交不可变工件")
            if finished and final_receipt:
                if receipt.get("真实训练末阶段SHA256") != hashes["阶段_LF训练末.pt"]:
                    raise ValueError("Howard LF真实末状态SHA没有在真实末收据提交")
            elif receipt.get("真实训练末阶段SHA256") is not None:
                raise ValueError("Howard LF暂停或历史会话不能伪造真末状态")
            seconds += wall
            setup_seconds += setup
            export_seconds += exported
            total_seconds += total
            peak = max(peak, memory)
            last_epoch = end
        if last_epoch != len(rows):
            raise ValueError("Howard LF本人连续收据未覆盖全部真实日志，不得裁剪未提交行")
        source = objects["事前真实来源登记.json"]
        if (source.get("LF方法") != METHOD or source.get("旧LF/HF权重读取") is not False
                or any(source.get(name) is not False for name in ISOLATION_FLAGS)
                or any(source.get(name) != value for name, value in source_identity.items())):
            raise ValueError("Howard LF必须从空网络开始，不借旧LF/HF或TEST来源")
        if finished:
            if not (patience == 200 or last_epoch == 2000):
                raise ValueError("Howard LF仅真实耐心200或预算2000截止才可完成")
            final = objects["阶段_LF训练末.pt"]
            _audit_stage(final, rows, seed, source_identity, log_sha=hashes["training.jsonl"])
            reason = "验证耐心提前停止" if patience == 200 else "预算截止"
            if final["metadata"].get("结束原因") != reason:
                raise ValueError("Howard LF真末停止原因须由完整历史重新计算")
            for key in ("model_state", "optimizer_state", "random_state", "parameter_requires_grad",
                        "scheduler_state", "sampler_epochs"):
                if not _same(final[key], objects["阶段_最近.pt"][key]):
                    raise ValueError("Howard LF真实末与最近完整网络/AdamW/RNG状态不一致")
            metrics = objects["metrics.json"]
            if (metrics.get("method") != METHOD or metrics.get("seed") != seed
                    or metrics.get("epochs_completed") != last_epoch
                    or metrics.get("best_epoch") != best_epoch
                    or metrics.get("best_validation_rmse_c") != score
                    or metrics.get("stopping_reason") != ("validation_patience" if patience == 200
                                                          else "planned_budget_completed")
                    or not _same(metrics.get("configuration"), LF_BUDGET)
                    or metrics.get("parameter_count") != LF_PARAMETER_COUNT
                    or metrics.get("LF参数量") != LF_PARAMETER_COUNT
                    or metrics.get("source_identity") != dict(source_identity)
                    or metrics.get("status") != "completed_current_protocol_lf_only"
                    or metrics.get("test") is not None
                    or any(metrics.get(name) is not False for name in ISOLATION_FLAGS)
                    or not _finite_number(metrics.get("training_seconds"), positive=True)
                    or not math.isclose(metrics["training_seconds"], seconds, rel_tol=1e-12, abs_tol=1e-7)
                    or any(not _finite_number(metrics.get(key))
                           or not math.isclose(metrics[key], expected, rel_tol=1e-12, abs_tol=1e-7)
                           for key, expected in (("setup_seconds", setup_seconds),
                                                 ("export_seconds", export_seconds),
                                                 ("total_session_seconds", total_seconds)))
                    or metrics.get("peak_gpu_memory_bytes") != peak):
                raise ValueError("Howard LF初始/最佳/真末/真实分段成本/固定预算与metrics不闭合")
        elif patience >= 200 or last_epoch >= 2000 or "阶段_LF训练末.pt" in objects or "metrics.json" in objects:
            raise ValueError("Howard LF已完成状态不能伪装暂停继续更新")
        return {"seed": seed, "累计轮次": last_epoch, "最佳轮次": best_epoch,
                "最佳LF验证RMSE_摄氏度": score, "无改善轮次": patience,
                "累计真实LF训练点": 60 * 8192 * last_epoch,
                "累计真实LF优化步": 61 * last_epoch,
                "真实累计成本秒": seconds, "峰值CUDA显存字节": peak,
                "真实累计加载构建与恢复墙钟秒": setup_seconds,
                "真实累计导出与源核验墙钟秒": export_seconds,
                "真实累计加载训练与导出墙钟秒": total_seconds,
                "LF参数量": LF_PARAMETER_COUNT, "LF模型构造参数": dict(LF_MODEL_KWARGS),
                "HF训练许可": False, "旧固定TEST温度读取": False,
                "模拟测试功率温度读取": False}
    except (KeyError, TypeError, AttributeError, RuntimeError, IndexError) as error:
        raise ValueError("Howard LF真实完整工件缺失、非法或不能闭合CPU状态链") from error


def _load_artifacts(directory: Path, root: Path, *, finished: bool) -> tuple[dict[str, Any], dict[str, str]]:
    receipts = sorted(directory.glob("LF会话收据_[0-9][0-9][0-9][0-9].json"))
    names = ["阶段_初始.pt", "阶段_LF观测最佳.pt", "阶段_最近.pt", "best.pt",
             "training.jsonl", "事前真实来源登记.json", *SNAPSHOT_NAMES]
    if finished:
        names += ["阶段_LF训练末.pt", "metrics.json"]
    elif (directory / "阶段_LF训练末.pt").exists() or (directory / "metrics.json").exists():
        raise ValueError("Howard LF续跑拒绝已出现真实末状态/完成metrics的目录")
    if not receipts:
        raise ValueError("Howard LF缺真实连续分段完成或暂停收据")
    for number, receipt in enumerate(receipts, 1):
        if receipt.name != f"LF会话收据_{number:04d}.json":
            raise ValueError("Howard LF提交收据编号不连续")
        names += [receipt.name, f"已提交日志_{number:04d}.jsonl",
                  f"最近提交历史_{number:04d}.pt", f"观测最佳提交历史_{number:04d}.pt",
                  f"观测最佳模型历史_{number:04d}.pt"]
    objects, hashes = {}, {}
    for name in names:
        path = _locked_project_file(directory / name, root)
        hashes[name] = sha256_file(path)
        try:
            if name.endswith(".pt"):
                objects[name] = torch.load(path, map_location="cpu", weights_only=False)
            elif name.endswith(".jsonl"):
                objects[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            elif name.endswith(".yaml"):
                objects[name] = yaml.safe_load(path.read_text(encoding="utf-8"))
            elif name.endswith(".md"):
                objects[name] = path.read_text(encoding="utf-8")
            else:
                objects[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, RuntimeError, EOFError) as error:
            raise ValueError("Howard LF完整本人状态/日志/收据不能CPU读取") from error
    configs = objects["config_snapshot/sha256.json"]
    if (not isinstance(configs, dict)
            or any(configs.get(name) != sha256_file(_locked_project_file(name, root))
                   or configs.get(name) != hashes["config_snapshot/" + Path(name).name]
                   for name in CONFIG_FILES)
            or configs.get(ROOT_LEDGER) != hashes["config_snapshot/" + ROOT_LEDGER]):
        raise ValueError("Howard LF实际物理/划分配置副本、快照清单或当前普通原件SHA漂移")
    expected_physics = resolve_physics_state()
    encoded_physics = yaml.safe_dump(expected_physics, sort_keys=False, allow_unicode=True).encode()
    if (objects["config_snapshot/resolved_physics.yaml"] != expected_physics
            or hashes["config_snapshot/resolved_physics.yaml"] != hashlib.sha256(encoded_physics).hexdigest()):
        raise ValueError("Howard LF实际名义物理快照副本与当前固定解析状态不符")
    best_identity = objects["best.pt"].get("任11Howard事前来源", {})
    if (not isinstance(best_identity, dict) or set(best_identity) != set(IDENTITY_FIELDS)
            or not _root_active(directory / "config_snapshot" / ROOT_LEDGER,
                                best_identity["YAML_SHA256"], best_identity["TAR_SHA256"],
                                best_identity["CATALOG_SHA256"])):
        raise ValueError("Howard LF事前主ROOT实际副本缺注册seed三SHA和唯一新活动编号")
    validate_lf_checkpoint_provenance(objects["best.pt"])
    for name in objects:
        if re.fullmatch(r"观测最佳模型历史_\d{4}\.pt", name):
            validate_lf_checkpoint_provenance(objects[name])
    return objects, hashes


def preflight_task11_howard_lf(
    *, registry: str | Path, registry_sha: str, source_tar: str | Path,
    source_tar_sha: str, catalog: str | Path, catalog_sha: str,
    output: str | Path, seed: int, ledger: str | Path = ROOT_LEDGER,
    project_root: str | Path = PROJECT_ROOT,
    resume_checkpoint: str | Path | None = None, qualified_source_only: bool = False,
) -> dict[str, Any]:
    """Validate the independent gate before CUDA, temperatures or any output write."""
    root = Path(project_root).resolve()
    registry_file = _locked_project_file(registry, root)
    archive_file = _locked_project_file(source_tar, root)
    catalog_file = _locked_project_file(catalog, root)
    ledger_file = _locked_project_file(ledger, root)
    if registry_file != root / REGISTRY or ledger_file != root / ROOT_LEDGER:
        raise ValueError("Howard LF仅认可固定任11登记位置和主ROOT，不接受替身台账")
    if (any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (registry_sha, source_tar_sha, catalog_sha))
            or sha256_file(registry_file) != registry_sha or sha256_file(archive_file) != source_tar_sha
            or sha256_file(catalog_file) != catalog_sha):
        raise ValueError("Howard LF事前YAML/源码tar/真实70catalog三SHA缺失或现场漂移")
    budget = _validate_registry_contract(yaml.safe_load(registry_file.read_text(encoding="utf-8")),
                                         source_tar_sha=source_tar_sha, catalog_sha=catalog_sha)
    if not _root_active(ledger_file, registry_sha, source_tar_sha, catalog_sha):
        raise ValueError("Howard LF主ROOT必须唯一独立活动且录号>101，三SHA不能借MLP门禁")
    if (type(seed) is not int or seed not in range(5) or os.environ.get("WORLD_SIZE", "1") != "1"
            or os.environ.get("RANK", "0") != "0" or os.environ.get("LOCAL_RANK", "0") != "0"):
        raise ValueError("Howard LF只许可五seed中的本seed、单卡单进程固定预算")
    destination = _project_path(output, root)
    if destination != root / RUN_DIRECTORY / f"正式Howard_LF_seed{seed}":
        raise ValueError("Howard LF输出只许本人同seed固定新目录")
    if qualified_source_only:
        if resume_checkpoint is not None or not destination.is_dir():
            raise ValueError("Howard LF只读完成审核需实际存在本人同seed目录")
    elif resume_checkpoint is None:
        if destination.exists():
            raise ValueError("Howard LF新会话不能覆盖已有正式seed目录")
    else:
        checkpoint = _locked_project_file(resume_checkpoint, root)
        if not destination.is_dir() or checkpoint != destination / "阶段_最近.pt":
            raise ValueError("Howard LF续跑仅接受本人同seed最近完整阶段")
    try:
        with tarfile.open(archive_file, "r:gz") as archive:
            members = archive.getmembers()
            if (len(members) != len(SOURCE_MEMBERS)
                    or any(not member.isfile() for member in members)
                    or {member.name for member in members} != set(SOURCE_MEMBERS)):
                raise ValueError("Howard LF源码tar只能含登记的全部普通唯一成员")
            for member in members:
                source = archive.extractfile(member)
                expected = budget["源码普通成员SHA256"][member.name]
                if (source is None or hashlib.sha256(source.read()).hexdigest() != expected
                        or sha256_file(_locked_project_file(member.name, root)) != expected):
                    raise ValueError("Howard LF源码tar/登记/当前普通源码原件SHA漂移")
    except (tarfile.TarError, OSError) as error:
        raise ValueError("Howard LF完整普通源码tar无法核验") from error
    recorded = json.loads(catalog_file.read_text(encoding="utf-8"))
    if recorded != collect_task11_mlp_lf_sources():
        raise ValueError("Howard LF真实70源与共同60/10有限LF预算的catalog SHA漂移")
    identity = dict(zip(IDENTITY_FIELDS, (registry_sha, source_tar_sha, catalog_sha)))
    if resume_checkpoint is not None:
        objects, hashes = _load_artifacts(destination, root, finished=False)
        audit_task11_howard_lf_artifacts(objects, hashes, seed=seed, source_identity=identity, finished=False)
        next_number = len([name for name in objects if re.fullmatch(r"LF会话收据_\d{4}\.json", name)]) + 1
        for name in (f"LF会话收据_{next_number:04d}.json", f"最近提交历史_{next_number:04d}.pt",
                     f"观测最佳提交历史_{next_number:04d}.pt", f"观测最佳模型历史_{next_number:04d}.pt",
                     f"已提交日志_{next_number:04d}.jsonl"):
            if _project_path(destination / name, root).exists():
                raise ValueError("Howard LF未提交旧侧件不得被新会话覆盖")
    return {"状态": "CPU前置门禁PASS；未探测CUDA或创建目录", "seed": seed,
            **identity, "预算": budget["正式预算"], "项目根": str(root),
            "登记原件": str(registry_file), "源码tar原件": str(archive_file),
            "ROOT台账原件": str(ledger_file), "真实源目录原件": str(catalog_file),
            "输出": str(destination), "HF训练许可": False,
            "原文精确三网联合训练复现": False}


def qualify_task11_howard_lf_source(
    *, registry: str | Path, registry_sha: str, source_tar: str | Path,
    source_tar_sha: str, catalog: str | Path, catalog_sha: str, output: str | Path,
    seed: int, ledger: str | Path = ROOT_LEDGER, project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Only CPU state/source qualification, never HF permission or label scoring."""
    identity = preflight_task11_howard_lf(
        registry=registry, registry_sha=registry_sha, source_tar=source_tar,
        source_tar_sha=source_tar_sha, catalog=catalog, catalog_sha=catalog_sha,
        output=output, seed=seed, ledger=ledger, project_root=project_root, qualified_source_only=True)
    objects, hashes = _load_artifacts(Path(identity["输出"]), Path(identity["项目根"]), finished=True)
    source_identity = {name: identity[name] for name in IDENTITY_FIELDS}
    audited = audit_task11_howard_lf_artifacts(objects, hashes, seed=seed,
                                              source_identity=source_identity, finished=True)
    assert_task11_howard_lf_source_unchanged(identity)
    return {**audited, "状态": "CPU Howard适配LF真实终态与当前来源审核PASS",
            "最佳LF检查点SHA256": hashes["best.pt"], "真实工件SHA256": hashes,
            "事前来源": source_identity, "原文精确三网联合训练复现": False,
            "后续HF": "必须另行锁定三网同时全参数更新合同；本审核不授权HF"}


def _guard_replace(destination: Path) -> None:
    _project_path(destination, PROJECT_ROOT)
    temporary = _project_path(destination.with_name(destination.name + ".tmp"), PROJECT_ROOT)
    if temporary.exists():
        raise ValueError("Howard LF旧临时状态不得被覆盖或用作旁路")


def _save_stage(destination: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                *, epoch: int, metadata: dict[str, Any]) -> None:
    _guard_replace(destination)
    save_training_state(destination, model, optimizer, stage=TRAINING_STAGE,
                        epoch=epoch, budget={"LF轮次": 2000}, metadata=metadata)


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    _project_path(path, PROJECT_ROOT)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _exclusive_copy(source: Path, destination: Path) -> None:
    source = _locked_project_file(source, PROJECT_ROOT)
    _project_path(destination, PROJECT_ROOT)
    with source.open("rb") as reader, destination.open("xb") as writer:
        while block := reader.read(1024 * 1024):
            writer.write(block)
    if sha256_file(source) != sha256_file(destination):
        raise ValueError("Howard LF本人不可覆盖提交副本SHA不等于实际原件")


def run_task11_howard_lf_formal(
    *, registry: str | Path, registry_sha: str, source_tar: str | Path,
    source_tar_sha: str, catalog: str | Path, catalog_sha: str, output: str | Path,
    seed: int, device_name: str = "cuda", session_epoch_limit: int = 200,
    resume_checkpoint: str | Path | None = None, ledger: str | Path = ROOT_LEDGER,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """One legal genuine CUDA LF session, never an HF or CPU training fallback."""
    prereg = preflight_task11_howard_lf(
        registry=registry, registry_sha=registry_sha, source_tar=source_tar,
        source_tar_sha=source_tar_sha, catalog=catalog, catalog_sha=catalog_sha,
        output=output, seed=seed, ledger=ledger, project_root=project_root,
        resume_checkpoint=resume_checkpoint)
    if (device_name != "cuda" or type(session_epoch_limit) is not int
            or not 1 <= session_epoch_limit <= 200):
        raise ValueError("Howard LF仅真单CUDA与每会话1—200轮，不许可改预算或CPU回退")
    if Path(project_root).resolve() != PROJECT_ROOT:
        raise ValueError("Howard LF真实训练只作用于实际项目，合成夹具仅能CPU验收")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Howard LF正式训练禁止静默CPU回退或多GPU")
    return _run_task11_howard_lf_session(prereg, seed=seed,
                                        session_epoch_limit=session_epoch_limit,
                                        resume_checkpoint=resume_checkpoint)


def _run_task11_howard_lf_session(
    prereg: dict[str, Any], *, seed: int, session_epoch_limit: int,
    resume_checkpoint: str | Path | None,
) -> dict[str, Any]:
    # The private backend also rechecks its own gate before touching CUDA.
    prereg = preflight_task11_howard_lf(
        registry=prereg["登记原件"], registry_sha=prereg["YAML_SHA256"],
        source_tar=prereg["源码tar原件"], source_tar_sha=prereg["TAR_SHA256"],
        catalog=prereg["真实源目录原件"], catalog_sha=prereg["CATALOG_SHA256"],
        output=prereg["输出"], seed=seed, ledger=prereg["ROOT台账原件"],
        project_root=prereg["项目根"], resume_checkpoint=resume_checkpoint)
    if (Path(prereg["项目根"]) != PROJECT_ROOT or type(session_epoch_limit) is not int
            or not 1 <= session_epoch_limit <= 200):
        raise ValueError("Howard LF私有后端也只接受自身活动门禁与实际单卡预算")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Howard LF私有后端禁止静默CPU回退或多GPU")
    output = Path(prereg["输出"])
    source_identity = {name: prereg[name] for name in IDENTITY_FIELDS}
    clock = time.perf_counter()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    set_seed(seed)
    splits = build_power_splits()
    train = load_sampled_points(sorted(splits.simulation_train), 8192, seed,
                                sampling_mode="material_time")
    validation = load_sampled_points(sorted(splits.simulation_validation), 8192, 10000,
                                     sampling_mode="material_time")
    if len(train) != 60 * 8192 or len(validation) != 10 * 8192:
        raise ValueError("Howard LF真实标签加载后必须恰为60/10功率各8192点")
    assert_task11_howard_lf_source_unchanged(prereg)
    coordinates, target = (tensor.to(device) for tensor in train.tensors)
    validation_x, validation_y = (tensor.to(device) for tensor in validation.tensors)
    model = Task11HowardLFTeacher().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)
    physics = _physics_ready()
    fingerprints = current_protocol_fingerprints()
    log_path = output / "training.jsonl"
    best_file = output / "best.pt"
    start_epoch = best_epoch = no_improvement = exposure = steps = 0
    best_rmse = math.inf
    session_index = 1
    initial_metadata = {**source_identity, "seed": seed, "LF方法": METHOD,
                        "LF模型构造参数": dict(LF_MODEL_KWARGS), "LF参数量": LF_PARAMETER_COUNT,
                        "模型尺度": asdict(model.scales),
                        "旧LF/HF权重读取": False, **{name: False for name in ISOLATION_FLAGS}}

    def metadata(epoch: int, *, reason: str | None = None) -> dict[str, Any]:
        result = {**initial_metadata, "当前轮次": epoch, "累计训练点": exposure,
                  "累计优化步": steps, "最佳LF验证RMSE_摄氏度": best_rmse if epoch else None,
                  "最佳LF验证轮次": best_epoch, "无改善轮次": no_improvement,
                  "日志SHA256": sha256_file(log_path)}
        if reason is not None:
            result["结束原因"] = reason
        return result

    if resume_checkpoint is None:
        output.mkdir(parents=True, exist_ok=False)
        with log_path.open("x", encoding="utf-8") as handle:
            handle.flush()
        write_config_snapshot(output)
        _write_new_json(output / "事前真实来源登记.json", {
            **initial_metadata, "来源性质": "Howard仅LF双编码网络从空网络初始化的统一预算适配",
            "原文精确三网联合训练复现": False})
        _save_stage(output / "阶段_初始.pt", model, optimizer, epoch=0, metadata=metadata(0))
    else:
        objects, hashes = _load_artifacts(output, PROJECT_ROOT, finished=False)
        audited = audit_task11_howard_lf_artifacts(objects, hashes, seed=seed,
                                                  source_identity=source_identity, finished=False)
        session_index = len([name for name in objects if re.fullmatch(r"LF会话收据_\d{4}\.json", name)]) + 1
        saved = load_training_state(output / "阶段_最近.pt", model, optimizer)
        start_epoch = audited["累计轮次"]
        if saved["epoch"] != start_epoch:
            raise ValueError("Howard LF恢复真实完整阶段与已验本人提交轮次不符")
        best_epoch = audited["最佳轮次"]
        best_rmse = audited["最佳LF验证RMSE_摄氏度"]
        no_improvement = audited["无改善轮次"]
        exposure = audited["累计真实LF训练点"]
        steps = audited["累计真实LF优化步"]
    end_epoch = min(2000, start_epoch + session_epoch_limit)
    loop_clock = time.perf_counter()
    finished_early = False
    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()
        indices = _rank_indices(len(coordinates), device, 0, 1, shuffle=True,
                                seed=seed * 1000000 + epoch)
        error_sse = torch.zeros(2, device=device, dtype=torch.float64)
        for offset in range(0, len(indices), 8192):
            chosen = indices[offset:offset + 8192]
            x, y = coordinates.index_select(0, chosen), target.index_select(0, chosen)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = ((prediction - y) / model.scales.temperature_scale_k).pow(2).mean()
            loss.backward()
            optimizer.step()
            error_sse[0] += (prediction.detach().double() - y.double()).pow(2).sum()
            error_sse[1] += y.numel()
        collocation = sample_collocation(256, device, seed=seed * 1000000 + epoch)
        nominal = physics_optimizer_step(model, optimizer, physics, collocation, 1.0)
        valid = _validation_sums(model, validation_x, validation_y, 8192, device, 0, 1)
        train_rmse = float(torch.sqrt(error_sse[0] / error_sse[1]))
        validation_rmse = float(torch.sqrt(valid[0] / valid[2]))
        validation_mae = float(valid[1] / valid[2])
        require_finite_lf_metrics(train_rmse, validation_rmse, validation_mae)
        exposure += len(indices)
        steps += 61
        improved = validation_rmse < best_rmse - MIN_DELTA_C
        if improved:
            best_rmse, best_epoch, no_improvement = validation_rmse, epoch, 0
        else:
            no_improvement += 1
        record = {"epoch": epoch, "seed": seed, "训练阶段": "howard_adapted_low_fidelity",
                  "LF训练功率数": 60, "LF合法验证功率数": 10,
                  "训练样本暴露": len(indices), "观测优化步": 60,
                  "物理优化步": 1, "物理配点": 256,
                  "累计训练点": exposure, "累计优化步": steps,
                  "train_rmse_c": train_rmse, "validation_rmse_c": validation_rmse,
                  "validation_mae_c": validation_mae, "learning_rate": optimizer.param_groups[0]["lr"],
                  "LF原物理损失": {name: float(value.detach()) for name, value in nominal.items()},
                  **{name: False for name in ISOLATION_FLAGS}}
        _history([record] if epoch == 1 else [
            *[json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()], record], seed)
        _locked_project_file(log_path, PROJECT_ROOT)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if improved:
            payload = {"schema_version": 1, "method": METHOD, "seed": seed, "epoch": epoch,
                       "model_state": model.state_dict(), "lf_subnet_state": model.network.state_dict(),
                       "model_kwargs": dict(LF_MODEL_KWARGS), "scales": asdict(model.scales),
                       "LF参数量": LF_PARAMETER_COUNT, "validation_rmse_c": validation_rmse,
                       "train_powers_w": sorted(splits.simulation_train),
                       "validation_powers_w": sorted(splits.simulation_validation),
                       "provenance": checkpoint_provenance(
                           role="low_fidelity_simulation", train_powers_w=splits.simulation_train,
                           validation_powers_w=splits.simulation_validation, fingerprints=fingerprints),
                       "material_passport": {"simulation_data_used": True, "experiment_data_used": False,
                                             "sensor_data_used": False, "physics_loss_used": True,
                                             "internal_experiment_truth": "not available"},
                       "任11Howard事前来源": source_identity,
                       "原文精确三网联合训练复现": False,
                       **{name: False for name in ISOLATION_FLAGS}}
            _guard_replace(best_file)
            temporary = best_file.with_name(best_file.name + ".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, best_file)
            _save_stage(output / "阶段_LF观测最佳.pt", model, optimizer,
                        epoch=epoch, metadata=metadata(epoch))
        _save_stage(output / "阶段_最近.pt", model, optimizer, epoch=epoch, metadata=metadata(epoch))
        if no_improvement >= 200:
            finished_early = True
            break
    completed = finished_early or epoch == 2000
    if completed:
        _save_stage(output / "阶段_LF训练末.pt", model, optimizer, epoch=epoch,
                    metadata=metadata(epoch, reason="验证耐心提前停止" if finished_early else "预算截止"))
    loop_finished = time.perf_counter()
    assert_task11_howard_lf_source_unchanged(prereg)
    copied = {"日志SHA256": (log_path, output / f"已提交日志_{session_index:04d}.jsonl"),
              "最近阶段SHA256": (output / "阶段_最近.pt", output / f"最近提交历史_{session_index:04d}.pt"),
              "观测最佳阶段SHA256": (output / "阶段_LF观测最佳.pt", output / f"观测最佳提交历史_{session_index:04d}.pt"),
              "观测最佳模型SHA256": (best_file, output / f"观测最佳模型历史_{session_index:04d}.pt")}
    for source, destination in copied.values():
        _exclusive_copy(source, destination)
    export_finished = time.perf_counter()
    seconds = loop_finished - loop_clock
    setup_seconds = loop_clock - clock
    export_seconds = export_finished - loop_finished
    total_seconds = export_finished - clock
    peak = torch.cuda.max_memory_allocated(device)
    receipt = {**source_identity, "seed": seed, "LF方法": METHOD,
               "起始已提交轮次": start_epoch, "本会话实际轮次": epoch - start_epoch,
               "累计实际轮次": epoch, "正式预算上限轮次": 2000,
               "LF累计真实训练点": exposure, "LF累计真实优化步": steps,
               "观测最佳LF验证RMSE_摄氏度": best_rmse, "观测最佳LF轮次": best_epoch,
               "无改善轮次": no_improvement, "本会话真实墙钟秒": seconds,
               "本会话加载构建与恢复墙钟秒": setup_seconds,
               "本会话导出与源核验墙钟秒": export_seconds,
               "本会话加载训练与导出总墙钟秒": total_seconds,
               "成本口径": "与新MLP同口径的真实epoch训练/逐轮验证/完整阶段保存之和；加载恢复及提交导出另列",
               "总墙钟边界": "本会话CUDA设备设置前至不可变状态/日志副本导出；不含收据JSON及收据后只读资格审计",
               "峰值真实CUDA显存字节": peak,
               **{name: False for name in ISOLATION_FLAGS},
               "状态": "已完成Howard适配LF正式训练" if completed else "已暂停且完整阶段提交",
               **{name: sha256_file(destination) for name, (_, destination) in copied.items()},
               "初始阶段SHA256": sha256_file(output / "阶段_初始.pt"),
               "配置快照SHA256": {name: sha256_file(_locked_project_file(output / name, PROJECT_ROOT))
                                   for name in SNAPSHOT_NAMES},
               "真实训练末阶段SHA256": sha256_file(output / "阶段_LF训练末.pt") if completed else None}
    _write_new_json(output / f"LF会话收据_{session_index:04d}.json", receipt)
    if completed:
        receipts = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(
            output.glob("LF会话收据_[0-9][0-9][0-9][0-9].json"))]
        overview = {"method": METHOD, "seed": seed, "epochs_completed": epoch,
                    "stopping_reason": "validation_patience" if finished_early else "planned_budget_completed",
                    "best_epoch": best_epoch, "best_validation_rmse_c": best_rmse,
                    "training_seconds": sum(item["本会话真实墙钟秒"] for item in receipts),
                    "training_seconds_contract": "与新MLP同口径：所有本人真实epoch训练/验证/阶段保存之和，加载及导出不混入",
                    "setup_seconds": sum(item["本会话加载构建与恢复墙钟秒"] for item in receipts),
                    "export_seconds": sum(item["本会话导出与源核验墙钟秒"] for item in receipts),
                    "total_session_seconds": sum(item["本会话加载训练与导出总墙钟秒"] for item in receipts),
                    "parameter_count": model.parameter_count(), "LF参数量": model.parameter_count(),
                    "peak_gpu_memory_bytes": max(item["峰值真实CUDA显存字节"] for item in receipts),
                    "status": "completed_current_protocol_lf_only", "configuration": LF_BUDGET,
                    "source_identity": source_identity,
                    "test": None, "test_status": "sealed_until_frozen_release",
                    "原文精确三网联合训练复现": False, **{name: False for name in ISOLATION_FLAGS}}
        _write_new_json(output / "metrics.json", overview)
    objects, hashes = _load_artifacts(output, PROJECT_ROOT, finished=completed)
    audit_task11_howard_lf_artifacts(objects, hashes, seed=seed,
                                    source_identity=source_identity, finished=completed)
    return receipt
