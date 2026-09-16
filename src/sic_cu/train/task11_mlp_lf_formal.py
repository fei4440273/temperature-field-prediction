"""Fresh Task-11 MLP LF training with preregistered simulation sources."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
import tarfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl
import torch
import yaml

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import build_power_splits
from sic_cu.eval.protocol_checks import (
    checkpoint_provenance, current_protocol_fingerprints,
    validate_lf_checkpoint_provenance,
)
from sic_cu.models.common import ModelScales, parameter_count
from sic_cu.physics.collocation import sample_collocation
from sic_cu.train.common import (
    CONFIG_FILES, load_training_state, physics_optimizer_step,
    save_training_state, write_config_snapshot,
)
from sic_cu.train.simulation import (
    _physics_ready, _rank_indices, _validation_sums, build_model,
    load_sampled_points, set_seed,
)


LF_COLUMNS = frozenset(("r_m", "z_m", "time_s", "power_w", "material_id", "temperature_k"))
SIMULATION_MANIFEST = PROJECT_ROOT / "data/processed/manifest.json"
ROOT_LEDGER = "多保真DeepONet预测精度优化总计划与执行台账.md"
ROOT_TOKEN = "TASK11_NEW_MLP_LF_GATE:v1"
RUN_DIRECTORY = "研究记录/任务11_外部对照/正式新MLP公平训练"
SOURCE_MEMBERS = (
    "src/sic_cu/train/task11_mlp_lf_formal.py",
    "scripts/49_run_task11_mlp_lf_formal.py",
    "tests/test_task11_mlp_lf_formal.py",
    "src/sic_cu/train/simulation.py",
    "src/sic_cu/train/common.py",
    "src/sic_cu/data/fields.py",
    "src/sic_cu/data/common.py",
    "src/sic_cu/data/balanced_sampler.py",
    "src/sic_cu/data/splits.py",
    "src/sic_cu/eval/protocol_checks.py",
    "src/sic_cu/models/mlp_pinn.py",
    "src/sic_cu/models/__init__.py",
    "src/sic_cu/models/common.py",
    "src/sic_cu/losses/physics.py",
    "src/sic_cu/losses/__init__.py",
    "src/sic_cu/physics/materials.py",
    "src/sic_cu/physics/boundary.py",
    "src/sic_cu/physics/configuration.py",
    "src/sic_cu/physics/heat_equation.py",
    "src/sic_cu/physics/interface.py",
    "src/sic_cu/physics/trainable_parameters.py",
    "src/sic_cu/physics/resolution.py",
    "src/sic_cu/physics/collocation.py",
    "src/sic_cu/config.py",
    "configs/splits.yaml",
    "configs/training.yaml",
    "configs/geometry.yaml",
    "configs/materials.yaml",
    "configs/boundary_conditions.yaml",
    "configs/data_metadata.yaml",
    "configs/release_manifest.yaml",
    "data/processed/manifest.json",
    "reports/current_protocol/data_inventory.json",
)
LF_BUDGET = {
    "seeds": [0, 1, 2, 3, 4], "method": "mlp_pinn",
    "epochs": 2000, "samples_per_power": 8192,
    "validation_samples_per_power": 8192, "batch_size": 8192,
    "learning_rate": 0.001, "patience": 200,
    "physics_collocation": 256, "physics_weight": 1.0,
    "lf_physics_mode": "original", "sampling_mode": "material_time",
    "device": "cuda", "world_size": 1,
}


def collect_task11_mlp_lf_sources() -> dict[str, Any]:
    """Lock only real LF training/validation files; test temperatures stay sealed."""
    manifest = json.loads(SIMULATION_MANIFEST.read_text(encoding="utf-8"))
    records = manifest.get("simulation", [])
    splits = build_power_splits()
    expected = {"train": splits.simulation_train,
                "validation": splits.simulation_validation,
                "test": splits.simulation_test}
    actual = {name: set() for name in expected}
    output: dict[str, Any] = {
        "schema_version": 1,
        "用途": "任11新MLP五seed低保真来源；仅70份实际训练与合法验证仿真温度",
        "模拟清单原文件SHA256": sha256_file(SIMULATION_MANIFEST),
        "处理来源嵌入清单SHA256": manifest.get("processed_manifest_sha256"),
        "模拟测试功率温度读取": False,
        "LF训练模拟原件": [],
        "LF合法验证模拟原件": [],
    }
    for record in records:
        role = record.get("split")
        power = float(record.get("power_w"))
        if role not in expected or power not in expected[role] or power in actual[role]:
            raise ValueError("任11 LF仿真清单功率折缺失、重复或与既定60/10/10不符")
        actual[role].add(power)
        if role == "test":
            continue
        relative = record.get("path")
        if (not isinstance(relative, str)
                or relative != f"data/processed/simulation/{power:g}W.parquet"):
            raise ValueError("任11 LF真实训练/验证仿真路径与功率身份不符")
        path = (PROJECT_ROOT / relative).resolve()
        if (PROJECT_ROOT not in path.parents or path.is_symlink() or not path.is_file()
                or not LF_COLUMNS.issubset(set(pl.read_parquet_schema(path)))):
            raise ValueError("任11 LF训练/验证原场缺失、符号链接或未提供真实六列")
        rows = int(pl.scan_parquet(path).select(pl.len()).collect().item())
        if rows != record.get("rows") or rows < 8192:
            raise ValueError("任11 LF现场仿真原场行数与原处理清单或采样预算不符")
        locked = {"功率_瓦": power, "原件": relative,
                  "真实行数": rows, "文件SHA256": sha256_file(path)}
        output["LF训练模拟原件" if role == "train"
               else "LF合法验证模拟原件"].append(locked)
    if actual != expected:
        raise ValueError("任11 LF仿真必须有60训练、10合法验证、10封存模拟测试身份")
    for name in ("LF训练模拟原件", "LF合法验证模拟原件"):
        output[name].sort(key=lambda row: row["功率_瓦"])
    return output


def assert_task11_lf_source_unchanged(prereg: dict[str, Any]) -> None:
    catalog_file = Path(prereg["真实源目录原件"])
    if (sha256_file(catalog_file) != prereg["CATALOG_SHA256"]
            or json.loads(catalog_file.read_text(encoding="utf-8"))
            != collect_task11_mlp_lf_sources()):
        raise ValueError("任11 LF真实70源在前置验收后、训练开始前发生漂移")


def require_finite_lf_metrics(train_rmse: float, validation_rmse: float,
                              validation_mae: float) -> None:
    if not all(math.isfinite(value) for value in
               (train_rmse, validation_rmse, validation_mae)):
        raise ValueError("任11 LF训练/合法验证误差必须为有限真实数值")


def _locked_project_file(value: str | Path, root: Path) -> Path:
    candidate = Path(value)
    absolute = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if (absolute == root or root not in absolute.parents
            or candidate.is_symlink() or not absolute.is_file()):
        raise ValueError("任11新MLP登记与冻结文件只能是项目内真实普通文件")
    return absolute


def _root_active(ledger: Path, registry_sha: str, tar_sha: str, catalog_sha: str) -> bool:
    prefix = (f"{ROOT_TOKEN}; status=active; YAML_SHA256={registry_sha}; "
              f"TAR_SHA256={tar_sha}; CATALOG_SHA256={catalog_sha}")
    rows = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        fields = [value.strip() for value in line.split("|")]
        if len(fields) < 4 or not re.fullmatch(r"录-\d{4}", fields[1]):
            continue
        if fields[2].startswith(ROOT_TOKEN):
            rows.append(fields[2])
    return len(rows) == 1 and (rows[0] == prefix or rows[0].startswith(prefix + "; "))


def preflight_task11_mlp_lf(
    *, registry: str | Path, registry_sha: str,
    source_tar: str | Path, source_tar_sha: str,
    catalog: str | Path, catalog_sha: str,
    output: str | Path, seed: int,
    ledger: str | Path = ROOT_LEDGER,
    project_root: str | Path = PROJECT_ROOT,
    resume_checkpoint: str | Path | None = None,
    qualified_source_only: bool = False,
) -> dict[str, Any]:
    """Verify ROOT, source bytes and 70 LF labels before any GPU or output action."""
    root = Path(project_root).resolve()
    registry_file = _locked_project_file(registry, root)
    archive_file = _locked_project_file(source_tar, root)
    catalog_file = _locked_project_file(catalog, root)
    root_ledger = _locked_project_file(ledger, root)
    if root_ledger != root / ROOT_LEDGER:
        raise ValueError("任11新MLP正式身份只认可核主ROOT台账")
    if (any(not re.fullmatch(r"[0-9a-f]{64}", digest or "") for digest in
            (registry_sha, source_tar_sha, catalog_sha))
            or sha256_file(registry_file) != registry_sha
            or sha256_file(archive_file) != source_tar_sha
            or sha256_file(catalog_file) != catalog_sha):
        raise ValueError("任11新MLP前登记YAML/源码tar/模拟70目录三SHA缺失或漂移")
    budget = yaml.safe_load(registry_file.read_text(encoding="utf-8"))
    if (not isinstance(budget, dict) or budget.get("schema_version") != 1
            or budget.get("阶段") != "fresh_mlp_low_fidelity"
            or budget.get("ROOT门禁标签") != ROOT_TOKEN
            or budget.get("旧固定TEST温度读取") is not False
            or budget.get("模拟测试功率温度读取") is not False
            or budget.get("新HF训练许可") is not False
            or budget.get("真实模拟70源目录SHA256") != catalog_sha
            or budget.get("正式预算") != LF_BUDGET
            or set(budget.get("源码普通成员SHA256", {})) != set(SOURCE_MEMBERS)):
        raise ValueError("任11新MLP仅允许锁定的五seed新LF训练合同，不授权旧LF或HF")
    if not _root_active(root_ledger, registry_sha, source_tar_sha, catalog_sha):
        raise ValueError("任11新MLP主ROOT未在编号行第二列事前激活三个SHA与唯一门禁")
    source_hashes = budget["源码普通成员SHA256"]
    try:
        with tarfile.open(archive_file, "r:gz") as archive:
            members = archive.getmembers()
            if (len(members) != len(SOURCE_MEMBERS)
                    or any(not member.isfile() for member in members)
                    or {member.name for member in members} != set(SOURCE_MEMBERS)):
                raise ValueError("任11新MLP源码归档成员不等于事前登记的真实普通原件")
            for member in members:
                source = archive.extractfile(member)
                current = _locked_project_file(member.name, root)
                expected = source_hashes[member.name]
                if (source is None or not isinstance(expected, str)
                        or hashlib.sha256(source.read()).hexdigest() != expected
                        or sha256_file(current) != expected):
                    raise ValueError("任11新MLP源码归档/预算/现场SHA漂移")
    except (tarfile.TarError, OSError) as error:
        raise ValueError("任11新MLP源码tar不能作为事前普通原件冻结") from error
    recorded = json.loads(catalog_file.read_text(encoding="utf-8"))
    if recorded != collect_task11_mlp_lf_sources():
        raise ValueError("任11真实70份LF仿真原件身份或处理清单SHA与冻结目录漂移")
    if (type(seed) is not int or seed not in range(5)
            or os.environ.get("WORLD_SIZE", "1") != "1"):
        raise ValueError("任11新MLP只许登记五seed单卡正式预算")
    destination = Path(output)
    destination = (destination if destination.is_absolute() else root / destination).resolve()
    expected = root / RUN_DIRECTORY / f"正式MLP_LF_seed{seed}"
    if destination != expected:
        raise ValueError("任11新MLP输出只能是同seed项目内正式目录")
    if qualified_source_only:
        if resume_checkpoint is not None or not destination.is_dir():
            raise ValueError("任11新MLP LF只读终态资格需已有实际完成同seed目录")
    elif resume_checkpoint is None:
        if destination.exists():
            raise ValueError("任11新MLP新训练不能覆盖已有同seed运行目录")
    else:
        supplied = Path(resume_checkpoint)
        checkpoint = (supplied if supplied.is_absolute() else root / supplied).resolve()
        if (not destination.is_dir() or supplied.is_symlink()
                or checkpoint != destination / "阶段_最近.pt"
                or not checkpoint.is_file()):
            raise ValueError("任11新MLP续跑只接受同seed自身最近完整checkpoint")
        receipts = sorted(destination.glob("LF会话收据_[0-9][0-9][0-9][0-9].json"))
        if not receipts:
            raise ValueError("任11新MLP续跑缺真实前会话提交SHA收据")
        latest = json.loads(receipts[-1].read_text(encoding="utf-8"))
        log_file = destination / "training.jsonl"
        if (latest.get("seed") != seed
                or latest.get("状态") != "已暂停且完整阶段提交"
                or latest.get("YAML_SHA256") != registry_sha
                or latest.get("TAR_SHA256") != source_tar_sha
                or latest.get("CATALOG_SHA256") != catalog_sha
                or not log_file.is_file()
                or latest.get("最近阶段SHA256") != sha256_file(checkpoint)
                or latest.get("日志SHA256") != sha256_file(log_file)):
            raise ValueError("任11新MLP续跑最近完整阶段/日志/冻结来源SHA与真实收据不符")
        # Complete optimizer/RNG stages contain NumPy objects and require the
        # repository's full-state loader after their recorded bytes are checked.
        try:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            rows = [json.loads(line) for line in
                    log_file.read_text(encoding="utf-8").splitlines()]
        except (OSError, RuntimeError, ValueError, TypeError, KeyError,
                json.JSONDecodeError) as error:
            raise ValueError("任11新MLP续跑本人完整断点/连续日志无法CPU核验") from error
        metadata = saved.get("metadata", {}) if isinstance(saved, dict) else {}
        epochs = latest.get("累计实际轮次")
        if (not isinstance(saved, dict)
                or saved.get("training_state_schema_version") != 1
                or saved.get("stage") != "low_fidelity"
                or saved.get("budget") != {"LF轮次": 2000}
                or type(epochs) is not int or not 0 < epochs < 2000
                or saved.get("epoch") != epochs
                or metadata.get("当前轮次") != epochs
                or metadata.get("seed") != seed
                or metadata.get("LF方法") != "mlp_pinn"
                or any(metadata.get(name) != sha for name, sha in (
                    ("YAML_SHA256", registry_sha),
                    ("TAR_SHA256", source_tar_sha),
                    ("CATALOG_SHA256", catalog_sha)))
                or metadata.get("日志SHA256") != latest.get("日志SHA256")
                or metadata.get("旧固定TEST温度读取") is not False
                or metadata.get("模拟测试功率温度读取") is not False
                or not isinstance(saved.get("model_state"), dict)
                or not isinstance(saved.get("optimizer_state"), dict)
                or set(saved.get("random_state", {}))
                   != {"python", "numpy", "torch_cpu", "torch_cuda"}
                or len(rows) != epochs
                or any(row.get("epoch") != index
                       or row.get("seed") != seed
                       or row.get("累计训练点") != 60 * 8192 * index
                       or row.get("累计优化步") != 61 * index
                       or row.get("旧固定TEST温度读取") is not False
                       or row.get("模拟测试功率温度读取") is not False
                       for index, row in enumerate(rows, 1))):
            raise ValueError("任11新MLP续跑断点内部seed/来源/预算/状态与本人收据不符")
    return {"状态": "CPU前置门禁PASS；未启动GPU或创建运行目录",
            "seed": seed, "预算": budget["正式预算"],
            "YAML_SHA256": registry_sha, "TAR_SHA256": source_tar_sha,
            "CATALOG_SHA256": catalog_sha,
            "真实源目录原件": str(catalog_file),
            "输出": str(destination)}


def qualify_task11_mlp_lf_source(
    *, registry: str | Path, registry_sha: str,
    source_tar: str | Path, source_tar_sha: str,
    catalog: str | Path, catalog_sha: str,
    output: str | Path, seed: int,
    ledger: str | Path = ROOT_LEDGER,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Audit a real completed LF seed without opening HF or simulated TEST labels."""
    identity = preflight_task11_mlp_lf(
        registry=registry, registry_sha=registry_sha,
        source_tar=source_tar, source_tar_sha=source_tar_sha,
        catalog=catalog, catalog_sha=catalog_sha,
        output=output, seed=seed, ledger=ledger,
        project_root=project_root, qualified_source_only=True,
    )
    directory = Path(identity["输出"])
    names = ("阶段_初始.pt", "阶段_LF观测最佳.pt", "阶段_最近.pt",
             "阶段_LF训练末.pt", "best.pt", "training.jsonl",
             "事前真实来源登记.json", "metrics.json")
    if any(not (directory / name).is_file() for name in names):
        raise ValueError("任11新MLP LF缺完整初始/最佳/最近/真实末、日志或完成收据")
    receipts = sorted(directory.glob("LF会话收据_[0-9][0-9][0-9][0-9].json"))
    if not receipts:
        raise ValueError("任11新MLP LF缺真实分段完成收据，不得进入后续HF")
    expected_sources = {name: identity[name] for name in
                        ("YAML_SHA256", "TAR_SHA256", "CATALOG_SHA256")}
    logs = directory / "training.jsonl"
    last_epoch = 0
    seconds = 0.0
    for number, receipt_file in enumerate(receipts, 1):
        if receipt_file.name != f"LF会话收据_{number:04d}.json":
            raise ValueError("任11新MLP LF提交收据必须连续编号，不许补假来源")
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        end = receipt.get("累计实际轮次")
        duration = receipt.get("本会话实际轮次")
        wall = receipt.get("本会话真实墙钟秒")
        if (receipt.get("seed") != seed
                or any(receipt.get(name) != sha
                       for name, sha in expected_sources.items())
                or receipt.get("旧固定TEST温度读取") is not False
                or receipt.get("模拟测试功率温度读取") is not False
                or receipt.get("起始已提交轮次") != last_epoch
                or type(duration) is not int or not 1 <= duration <= 200
                or end != last_epoch + duration or end > 2000
                or receipt.get("正式预算上限轮次") != 2000
                or receipt.get("LF累计真实训练点") != 60 * 8192 * end
                or receipt.get("LF累计真实优化步") != 61 * end
                or not isinstance(wall, (int, float))
                or not math.isfinite(wall) or wall <= 0):
            raise ValueError("任11新MLP LF全部分段收据身份/来源/单段预算/墙钟不闭合")
        if number != len(receipts):
            past = directory / f"最近提交历史_{number:04d}.pt"
            if (receipt.get("状态") != "已暂停且完整阶段提交"
                    or not past.is_file()
                    or sha256_file(past) != receipt.get("最近阶段SHA256")):
                raise ValueError("任11新MLP LF历史分段最近阶段原SHA未保留")
        else:
            if (receipt.get("状态") != "已完成新MLP LF正式训练"
                    or receipt.get("日志SHA256") != sha256_file(logs)
                    or receipt.get("最近阶段SHA256") != sha256_file(
                        directory / "阶段_最近.pt")
                    or receipt.get("观测最佳模型SHA256") != sha256_file(
                        directory / "best.pt")
                    or receipt.get("真实训练末阶段SHA256") != sha256_file(
                        directory / "阶段_LF训练末.pt")):
                raise ValueError("任11新MLP LF真末段收据/最近状态/日志SHA缺失或篡改")
        seconds += float(wall)
        last_epoch = end
    rows = [json.loads(line) for line in logs.read_text(encoding="utf-8").splitlines()]
    if (len(rows) != last_epoch or last_epoch < 201 or
            any(row.get("epoch") != index
                or row.get("seed") != seed
                or row.get("训练阶段") != "new_mlp_low_fidelity"
                or row.get("LF训练功率数") != 60
                or row.get("LF合法验证功率数") != 10
                or row.get("训练样本暴露") != 60 * 8192
                or row.get("观测优化步") != 60
                or row.get("物理优化步") != 1
                or row.get("物理配点") != 256
                or row.get("累计训练点") != 60 * 8192 * index
                or row.get("累计优化步") != 61 * index
                or row.get("旧固定TEST温度读取") is not False
                or row.get("模拟测试功率温度读取") is not False
                or not isinstance(row.get("validation_rmse_c"), (int, float))
                or not math.isfinite(row["validation_rmse_c"])
                for index, row in enumerate(rows, 1))):
        raise ValueError("任11新MLP LF逐轮60真实源、合法验证、物理点与旧TEST隔离不闭合")
    source = json.loads((directory / "事前真实来源登记.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if (any(source.get(name) != sha for name, sha in expected_sources.items())
            or source.get("旧MLP LF/HF权重读取") is not False
            or source.get("HF训练许可") is not False
            or metrics.get("seed") != seed or metrics.get("method") != "mlp_pinn"
            or metrics.get("epochs_completed") != last_epoch
            or metrics.get("configuration") != LF_BUDGET
            or metrics.get("status") != "completed_current_protocol_lf_only"
            or metrics.get("test") is not None
            or metrics.get("模拟测试功率温度读取") is not False
            or metrics.get("HF训练许可") is not False
            or not math.isclose(metrics.get("training_seconds", math.nan),
                                seconds, rel_tol=1e-12, abs_tol=1e-7)):
        raise ValueError("任11新MLP LF完训成本只准多会话真实墙钟和封存TEST来源")
    snapshots = {
        name: torch.load(directory / name, map_location="cpu", weights_only=False)
        for name in ("阶段_初始.pt", "阶段_LF观测最佳.pt",
                     "阶段_最近.pt", "阶段_LF训练末.pt")
    }
    initial = snapshots["阶段_初始.pt"]
    selected = snapshots["阶段_LF观测最佳.pt"]
    recent = snapshots["阶段_最近.pt"]
    final = snapshots["阶段_LF训练末.pt"]
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    validate_lf_checkpoint_provenance(checkpoint)
    best_epoch = checkpoint.get("epoch")
    model_state = checkpoint.get("model_state", {})
    if (type(best_epoch) is not int
            or best_epoch != metrics.get("best_epoch")
            or not 0 < best_epoch <= last_epoch
            or checkpoint.get("method") != "mlp_pinn"
            or checkpoint.get("seed") != seed
            or checkpoint.get("任11新MLP事前来源") != expected_sources
            or checkpoint.get("provenance", {}).get("test_labels_consumed") is not False
            or not math.isclose(checkpoint.get("validation_rmse_c", math.nan),
                                rows[best_epoch - 1]["validation_rmse_c"],
                                abs_tol=1e-7, rel_tol=1e-12)):
        raise ValueError("任11新MLP最佳视图缺同seed当前物理身份/合法LF观测轮次")
    for filename, stage, epoch in (
        ("阶段_初始.pt", initial, 0),
        ("阶段_LF观测最佳.pt", selected, best_epoch),
        ("阶段_最近.pt", recent, last_epoch),
        ("阶段_LF训练末.pt", final, last_epoch),
    ):
        metadata = stage.get("metadata", {})
        if (stage.get("training_state_schema_version") != 1
                or stage.get("stage") != "low_fidelity"
                or stage.get("epoch") != epoch
                or stage.get("budget") != {"LF轮次": 2000}
                or stage.get("random_state", {}).keys()
                   != {"python", "numpy", "torch_cpu", "torch_cuda"}
                or not isinstance(stage.get("optimizer_state"), dict)
                or metadata.get("seed") != seed
                or metadata.get("LF方法") != "mlp_pinn"
                or any(metadata.get(name) != sha
                       for name, sha in expected_sources.items())
                or metadata.get("旧固定TEST温度读取") is not False
                or metadata.get("模拟测试功率温度读取") is not False):
            raise ValueError(f"任11新MLP {filename}优化器/RNG/来源与真实身份不符")
    if (set(model_state) != set(selected["model_state"])
            or any(not torch.equal(value, selected["model_state"][name])
                   for name, value in model_state.items())
            or set(final["model_state"]) != set(recent["model_state"])
            or any(not torch.equal(value, recent["model_state"][name])
                   for name, value in final["model_state"].items())
            or set(initial.get("optimizer_state", {}).get("state", {}))
            or metrics.get("best_validation_rmse_c") != checkpoint["validation_rmse_c"]):
        raise ValueError("任11新MLP LF空初始AdamW/最佳完整状态/真实终态张量不一致")
    reconstructed = build_model("mlp_pinn", ModelScales(**checkpoint["scales"]),
                                **checkpoint["model_kwargs"])
    reconstructed.load_state_dict(model_state, strict=True)
    actual = {name: sha256_file(directory / name) for name in names}
    return {"seed": seed, "HF资格": "仅真实LF身份可另行登记；新HF五seed后锁未成立",
            "独立能源": "只读LF来源资格，HF能源须另行锁后双阶审计",
            "旧固定TEST温度读取": False,
            "模拟测试功率温度读取": False,
            "当前新MLP_LF最佳检查点SHA256": actual["best.pt"],
            "当前新MLP_LF累计轮次": last_epoch,
            "LF真实多会话累计成本秒": seconds,
            "事前来源": expected_sources,
            "真实原件SHA256": actual}


def run_task11_mlp_lf_formal(
    *, registry: str | Path, registry_sha: str,
    source_tar: str | Path, source_tar_sha: str,
    catalog: str | Path, catalog_sha: str,
    output: str | Path, seed: int, device_name: str = "cuda",
    session_epoch_limit: int = 200,
    resume_checkpoint: str | Path | None = None,
    ledger: str | Path = ROOT_LEDGER,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run one genuine LF session; paused models never qualify as HF sources."""
    prereg = preflight_task11_mlp_lf(
        registry=registry, registry_sha=registry_sha,
        source_tar=source_tar, source_tar_sha=source_tar_sha,
        catalog=catalog, catalog_sha=catalog_sha,
        output=output, seed=seed, ledger=ledger,
        project_root=project_root, resume_checkpoint=resume_checkpoint,
    )
    if device_name != "cuda" or not 1 <= session_epoch_limit <= 200:
        raise ValueError("任11新MLP正式训练仅单卡真CUDA与每次1—200真实轮次")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("任11新MLP LF正式训练不得静默回退CPU或并行多GPU")
    if Path(project_root).resolve() != PROJECT_ROOT:
        raise ValueError("任11新MLP训练后端只能作用于实际项目，隔离夹具仅供CPU预检")
    return _run_task11_lf_session(
        prereg, seed=seed, session_epoch_limit=session_epoch_limit,
        resume_checkpoint=resume_checkpoint,
    )


def _run_task11_lf_session(
    prereg: dict[str, Any], *, seed: int, session_epoch_limit: int,
    resume_checkpoint: str | Path | None,
) -> dict[str, Any]:
    output = Path(prereg["输出"])
    source_identity = {name: prereg[name] for name in
                       ("YAML_SHA256", "TAR_SHA256", "CATALOG_SHA256")}
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    set_seed(seed)
    splits = build_power_splits()
    train = load_sampled_points(sorted(splits.simulation_train), 8192, seed,
                                sampling_mode="material_time")
    validation = load_sampled_points(sorted(splits.simulation_validation),
                                     8192, 10_000, sampling_mode="material_time")
    if (len(train) != 60 * 8192 or len(validation) != 10 * 8192
            or {row["功率_瓦"] for row in collect_task11_mlp_lf_sources()[
                "LF训练模拟原件"]} != splits.simulation_train):
        raise ValueError("任11新MLP LF原标签真实加载后实际训练/验证来源与三SHA失配")
    assert_task11_lf_source_unchanged(prereg)
    coordinates, target = (tensor.to(device) for tensor in train.tensors)
    validation_x, validation_y = (tensor.to(device) for tensor in validation.tensors)
    scales = ModelScales()
    model_kwargs = {"width": 128, "depth": 5, "activation": "tanh",
                    "include_material": True}
    model = build_model("mlp_pinn", scales, **model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)
    physics = _physics_ready()
    fingerprints = current_protocol_fingerprints()
    initial_metadata = {**source_identity, "seed": seed, "LF方法": "mlp_pinn",
                        "旧固定TEST温度读取": False,
                        "模拟测试功率温度读取": False,
                        "累计训练点": 0, "累计优化步": 0,
                        "最佳LF验证RMSE_摄氏度": None,
                        "最佳LF验证轮次": 0, "无改善轮次": 0}
    log_path = output / "training.jsonl"
    best_file = output / "best.pt"
    start_epoch = best_epoch = no_improvement = exposure = steps = 0
    best_rmse = math.inf
    session_index = 1
    if resume_checkpoint is None:
        output.mkdir(parents=True, exist_ok=False)
        log_path.write_text("", encoding="utf-8")
        write_config_snapshot(output)
        (output / "事前真实来源登记.json").write_text(
            json.dumps({**source_identity, "来源性质": "新MLP LF从空网络新初始化",
                        "旧MLP LF/HF权重读取": False,
                        "HF训练许可": False}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        save_training_state(
            output / "阶段_初始.pt", model, optimizer,
            stage="low_fidelity", epoch=0, budget={"LF轮次": 2000},
            metadata=initial_metadata,
        )
    else:
        checkpoint = output / "阶段_最近.pt"
        receipts = sorted(output.glob("LF会话收据_[0-9][0-9][0-9][0-9].json"))
        session_index = len(receipts) + 1
        configs = json.loads((output / "config_snapshot/sha256.json").read_text(encoding="utf-8"))
        if any(configs.get(name) != sha256_file(PROJECT_ROOT / name)
               for name in CONFIG_FILES):
            raise ValueError("任11新MLP断点时原物理/划分配置发生漂移")
        state = load_training_state(checkpoint, model, optimizer)
        meta = state.get("metadata", {})
        if (state.get("stage") != "low_fidelity"
                or state.get("budget") != {"LF轮次": 2000}
                or state.get("epoch") != meta.get("当前轮次")
                or any(meta.get(name) != identity
                       for name, identity in source_identity.items())
                or meta.get("seed") != seed or meta.get("LF方法") != "mlp_pinn"
                or meta.get("旧固定TEST温度读取") is not False
                or meta.get("模拟测试功率温度读取") is not False
                or meta.get("日志SHA256") != sha256_file(log_path)):
            raise ValueError("任11新MLP接续状态不是本人完整LF事务或旧TEST来源")
        start_epoch = int(state["epoch"])
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != start_epoch or any(row.get("epoch") != epoch
                                           for epoch, row in enumerate(rows, 1)):
            raise ValueError("任11新MLP LF日志与最近完整状态不连续，禁止裁剪未提交行")
        best_epoch = meta["最佳LF验证轮次"]
        best_rmse = float(meta["最佳LF验证RMSE_摄氏度"])
        no_improvement = meta["无改善轮次"]
        exposure = meta["累计训练点"]
        steps = meta["累计优化步"]
        if (not 0 < start_epoch < 2000 or not 0 < best_epoch <= start_epoch
                or not math.isfinite(best_rmse) or no_improvement >= 200
                or exposure != 60 * 8192 * start_epoch
                or steps != 61 * start_epoch
                or rows[-1].get("累计训练点") != exposure
                or rows[-1].get("累计优化步") != steps):
            raise ValueError("任11新MLP LF续跑真实60×8192/61优化步与已提交轮次失配")
        frozen_copy = output / f"最近提交历史_{session_index - 1:04d}.pt"
        if frozen_copy.exists():
            raise ValueError("任11新MLP先前最近断点快照不得被覆盖")
        shutil.copyfile(checkpoint, frozen_copy)
        if sha256_file(frozen_copy) != sha256_file(checkpoint):
            raise ValueError("任11新MLP历史最近完整断点复制SHA不同")
    if start_epoch >= 2000:
        raise ValueError("任11新MLP LF已完成不得复用本目录再训练")
    clock = time.perf_counter()
    end_epoch = min(2000, start_epoch + session_epoch_limit)
    finished_early = False
    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()
        indices = _rank_indices(len(coordinates), device, 0, 1,
                                shuffle=True, seed=seed * 1_000_000 + epoch)
        error_sse = torch.zeros(2, device=device, dtype=torch.float64)
        for offset in range(0, len(indices), 8192):
            chosen = indices[offset:offset + 8192]
            x = coordinates.index_select(0, chosen)
            y = target.index_select(0, chosen)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = ((prediction - y) / scales.temperature_scale_k).pow(2).mean()
            loss.backward()
            optimizer.step()
            error_sse[0] += (prediction.detach().double() - y.double()).pow(2).sum()
            error_sse[1] += y.numel()
        collocation = sample_collocation(256, device, seed=seed * 1_000_000 + epoch)
        nominal = physics_optimizer_step(model, optimizer, physics, collocation, 1.0)
        valid = _validation_sums(model, validation_x, validation_y, 8192, device, 0, 1)
        train_rmse = float(torch.sqrt(error_sse[0] / error_sse[1]))
        validation_rmse = float(torch.sqrt(valid[0] / valid[2]))
        validation_mae = float(valid[1] / valid[2])
        require_finite_lf_metrics(train_rmse, validation_rmse, validation_mae)
        exposure += len(indices)
        steps += 61
        improved = validation_rmse < best_rmse - 0.0001
        if improved:
            best_rmse, best_epoch, no_improvement = validation_rmse, epoch, 0
            payload = {
                "schema_version": 1, "method": "mlp_pinn", "seed": seed,
                "epoch": epoch, "model_state": model.state_dict(),
                "model_kwargs": model_kwargs, "scales": asdict(scales),
                "validation_rmse_c": validation_rmse,
                "train_powers_w": sorted(splits.simulation_train),
                "validation_powers_w": sorted(splits.simulation_validation),
                "provenance": checkpoint_provenance(
                    role="low_fidelity_simulation",
                    train_powers_w=splits.simulation_train,
                    validation_powers_w=splits.simulation_validation,
                    fingerprints=fingerprints,
                ),
                "material_passport": {
                    "simulation_data_used": True, "experiment_data_used": False,
                    "sensor_data_used": False, "physics_loss_used": True,
                    "internal_experiment_truth": "not available",
                },
                "任11新MLP事前来源": source_identity,
            }
            temporary = best_file.with_name("best.pt.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, best_file)
            save_training_state(
                output / "阶段_LF观测最佳.pt", model, optimizer,
                stage="low_fidelity", epoch=epoch, budget={"LF轮次": 2000},
                metadata={**initial_metadata,
                          "当前轮次": epoch, "累计训练点": exposure,
                          "累计优化步": steps,
                          "最佳LF验证RMSE_摄氏度": best_rmse,
                          "最佳LF验证轮次": best_epoch},
            )
        else:
            no_improvement += 1
        record = {
            "epoch": epoch, "LF训练功率数": 60, "LF合法验证功率数": 10,
            "训练阶段": "new_mlp_low_fidelity", "seed": seed,
            "训练样本暴露": len(indices), "观测优化步": 60,
            "物理优化步": 1, "物理配点": 256,
            "累计训练点": exposure, "累计优化步": steps,
            "train_rmse_c": float(torch.sqrt(error_sse[0] / error_sse[1])),
            "validation_rmse_c": validation_rmse,
            "validation_mae_c": validation_mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "LF原物理损失": {name: float(value.detach())
                             for name, value in nominal.items()},
            "旧固定TEST温度读取": False,
            "模拟测试功率温度读取": False,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        save_training_state(
            output / "阶段_最近.pt", model, optimizer,
            stage="low_fidelity", epoch=epoch, budget={"LF轮次": 2000},
            metadata={**initial_metadata, "当前轮次": epoch,
                      "累计训练点": exposure, "累计优化步": steps,
                      "最佳LF验证RMSE_摄氏度": best_rmse,
                      "最佳LF验证轮次": best_epoch,
                      "无改善轮次": no_improvement,
                      "日志SHA256": sha256_file(log_path)},
        )
        if no_improvement >= 200:
            finished_early = True
            break
    completed = finished_early or epoch == 2000
    if completed:
        save_training_state(
            output / "阶段_LF训练末.pt", model, optimizer,
            stage="low_fidelity", epoch=epoch, budget={"LF轮次": 2000},
            metadata={**initial_metadata, "当前轮次": epoch,
                      "累计训练点": exposure, "累计优化步": steps,
                      "最佳LF验证RMSE_摄氏度": best_rmse,
                      "最佳LF验证轮次": best_epoch,
                      "无改善轮次": no_improvement,
                      "日志SHA256": sha256_file(log_path),
                      "结束原因": "验证耐心提前停止" if finished_early else "预算截止"},
        )
    seconds = time.perf_counter() - clock
    receipt = {**source_identity, "seed": seed,
               "起始已提交轮次": start_epoch, "本会话实际轮次": epoch - start_epoch,
               "累计实际轮次": epoch, "正式预算上限轮次": 2000,
               "LF累计真实训练点": exposure, "LF累计真实优化步": steps,
               "观测最佳LF验证RMSE_摄氏度": best_rmse,
               "观测最佳LF轮次": best_epoch,
               "本会话真实墙钟秒": seconds,
               "峰值真实CUDA显存字节": torch.cuda.max_memory_allocated(device),
               "旧固定TEST温度读取": False,
               "模拟测试功率温度读取": False,
               "状态": "已完成新MLP LF正式训练" if completed
                       else "已暂停且完整阶段提交",
               "日志SHA256": sha256_file(log_path),
               "最近阶段SHA256": sha256_file(output / "阶段_最近.pt"),
               "观测最佳模型SHA256": sha256_file(best_file),
               "真实训练末阶段SHA256": (sha256_file(output / "阶段_LF训练末.pt")
                                        if completed else None)}
    receipt_file = output / f"LF会话收据_{session_index:04d}.json"
    if receipt_file.exists():
        raise ValueError("任11新MLP LF会话收据禁止覆盖")
    receipt_file.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    if completed:
        checkpoint = torch.load(best_file, map_location="cpu", weights_only=True)
        validate_lf_checkpoint_provenance(checkpoint)
        if (checkpoint["seed"] != seed or checkpoint["method"] != "mlp_pinn"
                or checkpoint.get("任11新MLP事前来源") != source_identity):
            raise ValueError("任11新MLP HF后续只可登记同seed当前来源最佳LF权重")
        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        if (len(records) != epoch or
                any(row.get("epoch") != index or row.get("累计训练点") != 60 * 8192 * index
                    or row.get("累计优化步") != 61 * index
                    for index, row in enumerate(records, 1))):
            raise ValueError("任11新MLP HF后续资格缺逐轮60功率真LF点与优化步闭合")
        cost = [json.loads(path.read_text(encoding="utf-8"))["本会话真实墙钟秒"]
                for path in sorted(output.glob("LF会话收据_[0-9][0-9][0-9][0-9].json"))]
        overview = {"method": "mlp_pinn", "seed": seed,
                    "epochs_completed": epoch,
                    "stopping_reason": "validation_patience" if finished_early
                                       else "planned_budget_completed",
                    "best_epoch": best_epoch,
                    "best_validation_rmse_c": best_rmse,
                    "training_seconds": sum(cost),
                    "training_seconds_contract": "所有本人真实分段会话之和；非仅末会话",
                    "parameter_count": parameter_count(model),
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
                    "status": "completed_current_protocol_lf_only",
                    "configuration": LF_BUDGET,
                    "source_identity": source_identity,
                    "test": None, "test_status": "sealed_until_frozen_release",
                    "模拟测试功率温度读取": False,
                    "HF训练许可": False}
        (output / "metrics.json").write_text(
            json.dumps(overview, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return receipt
