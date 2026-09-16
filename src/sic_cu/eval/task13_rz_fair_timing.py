"""Fixed-RZ checkpoint-only inference timing without HF label access."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl
import torch
import yaml

from sic_cu.config import PROJECT_ROOT


BENCHMARK_POWER_W = 309.0
GRID_COLUMNS = ("time_s", "r_m", "z_m", "material_id", "node_label")
GRID_PATH = PROJECT_ROOT / "data/processed/simulation/10W.parquet"
OLD_RELEASES = {
    "旧B0发布清单SHA256": PROJECT_ROOT /
        "reports/releases/user_authorized_test_20260909_v1_deeponet/release_manifest.yaml",
    "旧MLP发布清单SHA256": PROJECT_ROOT /
        "reports/releases/user_authorized_test_20260909_v1_mlp/release_manifest.yaml",
}
ARMS = ("B0", "MLP", "E0", "F2")
RZ_NODES = 1159
RZ_TIMES = np.arange(101, dtype=np.float32) * 2.0
RUN_ROOT = PROJECT_ROOT / "reports/runs/sic-cu-v2-gpu1-20260908T072729Z-r2"
E0_ROOT = PROJECT_ROOT / "研究记录/任务07_正式五种子重训"
F2_ROOT = PROJECT_ROOT / "研究记录/任务08_贡献消融"
E0_DIRS = (
    "修复后正式_E0_种子0_20260916T011619+0800",
    "修复后正式_E0_种子1_20260916T012642+0800",
    "修复后正式_E0_种子2_20260916T012642+0800",
    "修复后正式_E0_种子3_20260916T012933+0800",
    "修复后正式_E0_种子4_20260916T012933+0800",
)
F2_DIRS = (
    "F2_事务修订正式_种子0_20260916T040215+0800",
    "F2_事务修订正式_种子1_20260916T042633+0800",
    "F2_事务修订正式_种子2_20260916T043607+0800",
    "F2_事务修订正式_种子3_20260916T045125+0800",
    "F2_事务修订正式_种子4_20260916T050138+0800",
)
SOURCE_MEMBERS = (
    "configs/geometry.yaml", "configs/splits.yaml",
    "src/sic_cu/data/fields.py", "src/sic_cu/models/mlp_pinn.py",
    "src/sic_cu/models/deeponet_pinn.py", "src/sic_cu/models/multifidelity.py",
    "src/sic_cu/prediction.py", "src/sic_cu/visualization/export.py",
    "src/sic_cu/eval/task13_rz_fair_timing.py",
    "scripts/40_run_task13_rz_fair_timing.py", "tests/test_task13_rz_fair_timing.py",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registered_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_benchmark_power(power_w: float) -> float:
    if not math.isfinite(float(power_w)) or float(power_w) != BENCHMARK_POWER_W:
        raise ValueError("任13推理计时只许已登记HF训练功率309 W，不读旧测试功率或HF标签")
    return BENCHMARK_POWER_W


@dataclass(frozen=True)
class RZTimingGrid:
    times_s: np.ndarray
    coordinates_rz_m: np.ndarray
    material_ids: np.ndarray
    node_labels: np.ndarray


def _check_rz_coordinates(coordinates: np.ndarray, materials: np.ndarray) -> None:
    if (coordinates.ndim != 2 or coordinates.shape[-1] != 2
            or materials.shape != (len(coordinates),) or not np.isfinite(coordinates).all()
            or not set(np.unique(materials)).issubset({0, 1})):
        raise ValueError("任13 RZ域网格形状、材料标签或有限性不合法")
    radius, z = coordinates[:, 0], coordinates[:, 1]
    if (np.any(radius < -1e-8) or np.any(z < -0.01750001) or np.any(z > 1e-8)
            or np.any((materials == 0) & (radius > 0.05834001))
            or np.any((materials == 1) & (radius > 0.02500001))
            or np.any((materials == 1) & (z < -0.01200001))):
        raise ValueError("任13 Cu/SiC各自真实RZ域外半径/深度点不可推理")


def load_registered_grid(source: str | Path, expected_sha256: str) -> RZTimingGrid:
    path = _registered_path(source)
    if not path.is_file() or _sha(path) != expected_sha256:
        raise ValueError("任13仅注册LF10W同网格五列数据源SHA不匹配")
    frame = pl.read_parquet(path, columns=list(GRID_COLUMNS)).sort(
        "time_s", "material_id", "node_label",
    )
    if frame.height != len(RZ_TIMES) * RZ_NODES:
        raise ValueError("任13时间和每时1159节点RZ网格尺寸改变")
    time_values = frame["time_s"].to_numpy().reshape(101, RZ_NODES)
    if not np.array_equal(time_values, np.broadcast_to(RZ_TIMES[:, None], time_values.shape)):
        raise ValueError("任13所有101个固定0..200s时刻、每时同网格必须一致")
    material = frame["material_id"].to_numpy().reshape(101, RZ_NODES)
    labels = frame["node_label"].to_numpy().reshape(101, RZ_NODES)
    coordinates = frame.select("r_m", "z_m").to_numpy().reshape(101, RZ_NODES, 2)
    if (np.any(material != material[0]) or np.any(labels != labels[0])
            or np.any(coordinates != coordinates[0])
            or np.count_nonzero(material[0] == 0) != 821
            or np.count_nonzero(material[0] == 1) != 338):
        raise ValueError("任13 Cu821/SiC338节点/材料/坐标必须在全时网格逐位一致")
    for material_id, count in ((0, 821), (1, 338)):
        if len(np.unique(labels[0, material[0] == material_id])) != count:
            raise ValueError("任13 RZ网格同材料节点标签不能重复")
    _check_rz_coordinates(coordinates[0], material[0])
    return RZTimingGrid(RZ_TIMES.copy(), coordinates[0].astype(np.float32),
                        material[0].astype(np.int64), labels[0].astype(np.int64))


def build_fixed_queries(grid: RZTimingGrid, power_w: float) -> np.ndarray:
    power = validate_benchmark_power(power_w)
    if (grid.coordinates_rz_m.shape != (RZ_NODES, 2)
            or grid.material_ids.shape != (RZ_NODES,)
            or not np.array_equal(grid.times_s, RZ_TIMES)):
        raise ValueError("任13预登记RZ节点/时窗网格不一致")
    _check_rz_coordinates(grid.coordinates_rz_m, grid.material_ids)
    coordinates = np.tile(grid.coordinates_rz_m, (101, 1))
    times = np.repeat(grid.times_s, RZ_NODES).reshape(-1, 1)
    powers = np.full((len(times), 1), power, dtype=np.float32)
    materials = np.tile(grid.material_ids, 101).reshape(-1, 1)
    return np.column_stack((coordinates, times, powers, materials)).astype(np.float32)


def _check_query(query: np.ndarray) -> None:
    if (query.ndim != 2 or query.shape[1] != 5 or not len(query)
            or not np.isfinite(query).all()
            or np.any(query[:, 3] != BENCHMARK_POWER_W)
            or np.any(query[:, 2] < 0) or np.any(query[:, 2] > 200)
            or not set(np.unique(query[:, 4])).issubset({0.0, 1.0})):
        raise ValueError("任13查询功率/时刻/材料/坐标须在预登记RZ域内")
    _check_rz_coordinates(query[:, :2], query[:, 4])


def run_model_forward(
    model: torch.nn.Module | None, query: np.ndarray, device: torch.device,
    *, batch_size: int,
) -> np.ndarray:
    if model is None or not isinstance(model, torch.nn.Module):
        raise ValueError("任13FEM静默回退禁止：必须有真实checkpoint模型")
    _check_query(np.asarray(query))
    if batch_size <= 0:
        raise ValueError("任13前向固定正整数batch大小")
    model = model.to(device).eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for first in range(0, len(query), batch_size):
            batch = torch.as_tensor(query[first:first + batch_size], dtype=torch.float32,
                                    device=device)
            outputs = model(batch)
            if outputs.shape != (len(batch), 1):
                raise ValueError("任13模型逐点输出必须仅一列温度K，不许静默FEM回退")
            chunks.append(outputs.detach())
    result = torch.cat(chunks).cpu().numpy().reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError("任13推理RZ温度须有限")
    return result


def check_registered_candidate(
    row: Mapping[str, Any], checkpoint: str | Path, receipt: str | Path,
) -> None:
    if (row.get("arm") not in ARMS or row.get("seed") not in range(5)
            or _registered_path(row.get("checkpoint", "")) != _registered_path(checkpoint)
            or _registered_path(row.get("receipt", "")) != _registered_path(receipt)):
        raise ValueError("任13来源方法须为同一RZ B0/MLP/E0/F2，拒绝一维板和FEM")
    for name, path in (("checkpoint_sha256", checkpoint), ("receipt_sha256", receipt)):
        artifact = _registered_path(path)
        if not artifact.is_file() or row.get(name) != _sha(artifact):
            raise ValueError("任13已登记模型来源/训练成本原件SHA变动或缺失")


def preflight_timing(
    registry: str | Path, registry_sha256: str, source_tar: str | Path,
    source_tar_sha256: str, ledger: str | Path, output: str | Path, *,
    require_cuda: bool,
) -> dict[str, Any]:
    registry_path, tar_path = _registered_path(registry), _registered_path(source_tar)
    if (not registry_path.is_file() or _sha(registry_path) != registry_sha256
            or not tar_path.is_file() or _sha(tar_path) != source_tar_sha256):
        raise ValueError("任13推理计时预算/来源与源码tar须先事前登记SHA")
    ledger_path = _registered_path(ledger)
    if not ledger_path.is_file() or not any(
        re.search(r"\|\s*录-\d+\s*\|", line)
        and registry_sha256 in line and source_tar_sha256 in line
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ):
        raise ValueError("任13同一主台账行须在真实计时前登记预算与源码两个SHA")
    output_path = _registered_path(output)
    if (output_path.exists() or output_path == PROJECT_ROOT
            or PROJECT_ROOT not in output_path.parents):
        raise ValueError("任13计时输出只许全新项目内目录，不覆盖既有原件")
    with registry_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if (not isinstance(cfg, dict) or cfg.get("schema_version") != 1
            or cfg.get("实验对象") != "RZ原装置历史与新同网格推理成本"
            or cfg.get("注册功率_W") != BENCHMARK_POWER_W
            or cfg.get("时间帧") != 101 or cfg.get("每帧节点") != RZ_NODES
            or cfg.get("读网格五列") != list(GRID_COLUMNS)
            or _registered_path(cfg.get("RZ网格原件", "")) != GRID_PATH.resolve()
            or cfg.get("RZ网格SHA256") != _sha(GRID_PATH)):
        raise ValueError("任13预算须锁真实同一RZ网格、合法309W、五列和0..200s")
    validate_registered_timing_protocol(cfg)
    for field, source in OLD_RELEASES.items():
        if (not source.is_file() or cfg.get(field) != _sha(source)
                or yaml.safe_load(source.read_text(encoding="utf-8")).get("status") != "frozen"):
            raise ValueError("任13原历史B0/MLP冻结发布SHA或状态不可用")
    check_source_tar_members(tar_path, registry_path)
    lf_rows = registered_lf_rows(cfg)
    rows = registered_candidate_rows(cfg, lf_rows)
    for lf_row in lf_rows:
        for name, path in (("checkpoint_sha256", lf_row["checkpoint"]),
                           ("receipt_sha256", lf_row["receipt"])):
            artifact = _registered_path(path)
            if not artifact.is_file() or lf_row[name] != _sha(artifact):
                raise ValueError("任13可复用LF预训原件来源或真实成本收据SHA不符")
        check_lf_receipt(lf_row, json.loads(Path(lf_row["receipt"]).read_text(encoding="utf-8")))
    for row in rows:
        check_registered_origin(row)
        check_registered_candidate(row, row["checkpoint"], row["receipt"])
        check_training_receipt(row, json.loads(Path(row["receipt"]).read_text(encoding="utf-8")))
    if require_cuda and (not torch.cuda.is_available()
                         or torch.cuda.device_count() < 1):
        raise ValueError("任13正式GPU计时必须是真CUDA；CPU仅合成夹具诊断")
    cfg["登记模型来源"] = rows
    cfg["登记LF来源"] = lf_rows
    return cfg


def check_checkpoint_payload(row: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    from sic_cu.data.splits import build_power_splits

    arm, seed = row.get("arm"), row.get("seed")
    low = "mlp_pinn" if arm == "MLP" else "deeponet_pinn"
    if (arm not in ARMS or seed not in range(5)
            or payload.get("method") != "multifidelity_correction"
            or payload.get("low_fidelity_method") != low
            or payload.get("seed") != seed):
        raise ValueError("任13注册来源模型方法、LF架构或seed身份不合RZ四臂")
    split = build_power_splits()
    for field, registered in (
        ("hf_train_powers_w", split.hf_train),
        ("hf_validation_powers_w", split.hf_validation),
        ("hf_test_powers_w", split.hf_test),
    ):
        if set(map(float, payload.get(field, ()))) != registered:
            raise ValueError("任13模型真实HF训练/验证/旧测试功率折不可混用")
    if payload.get("provenance", {}).get("test_labels_consumed") is not False:
        raise ValueError("任13模型旧TEST标签消费不合法")
    lf_sha = (payload.get("lf_checkpoint_sha256")
              or payload.get("provenance", {}).get("lf_checkpoint_sha256"))
    if lf_sha != row.get("lf_checkpoint_sha256"):
        raise ValueError("任13模型内嵌LF来源检查点SHA与同seed可复用LF不一致")
    if arm in ("E0", "F2"):
        key = ("任07新HF模型视图来源" if arm == "E0"
               else "任08F2模型视图来源")
        lineage = payload.get(key, {})
        if (lineage.get("训练种子") != seed or lineage.get("运行臂") != arm
                or lineage.get("旧test_Data温度标签读取") is not False
                or lineage.get("本seed真实LF检查点SHA256") != lf_sha
                or lineage.get("正式预登记配置SHA256")
                   != row.get("正式预登记配置SHA256")):
            raise ValueError("任13 E0/F2真实模型视图来源/身份与事前登记不同")


def check_training_receipt(row: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    arm, seed = row.get("arm"), row.get("seed")
    if arm in ("B0", "MLP"):
        seconds, epochs = receipt.get("training_seconds"), receipt.get("epochs_completed")
        if (receipt.get("seed") != seed or epochs != 1700
                or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("任13旧历史训练成本必须真实1700轮完整单次会话和正墙钟")
        return {"耗时秒": float(seconds), "实际轮次": int(epochs),
                "口径": "历史单次完整训练会话含评价", "可复用LF预训耗时秒": None}
    seconds = receipt.get("本会话耗时秒")
    correction, joint = receipt.get("校正实际轮次"), receipt.get("联合实际轮次")
    if (arm not in ("E0", "F2") or receipt.get("运行种子") != seed
            or receipt.get("运行臂") != arm
            or receipt.get("真实最后状态") != "阶段_训练末.pt"
            or receipt.get("旧test_Data温度标签读取") is not False
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds <= 0
            or not isinstance(correction, int) or not isinstance(joint, int)
            or correction <= 0 or joint < 0):
        raise ValueError("任13 E0/F2真实训练成本来源不得用诊断或缺失最后状态/墙钟")
    return {"耗时秒": float(seconds), "实际轮次": correction + joint,
            "口径": "仅当前真实会话，非全部断点累计", "可复用LF预训耗时秒": None}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_forward_parts(
    model: torch.nn.Module, query: np.ndarray, device: torch.device, *,
    batch_size: int, warmups: int, repeats: int,
) -> dict[str, Any]:
    if (not isinstance(model, torch.nn.Module) or batch_size <= 0 or warmups < 0
            or repeats <= 0):
        raise ValueError("任13仅真实checkpoint模型，固定batch/预热/有效重复计时")
    _check_query(np.asarray(query))
    transfer_clock = time.perf_counter()
    gpu_query = torch.as_tensor(np.ascontiguousarray(query), dtype=torch.float32,
                                device=device)
    _sync(device)
    transfer_seconds = time.perf_counter() - transfer_clock
    model.eval()
    timings: list[float] = []
    result: torch.Tensor | None = None
    with torch.no_grad():
        for repetition in range(warmups + repeats):
            _sync(device)
            start = time.perf_counter()
            output = [model(gpu_query[offset:offset + batch_size])
                      for offset in range(0, len(gpu_query), batch_size)]
            _sync(device)
            seconds = time.perf_counter() - start
            if any(chunk.shape != (min(batch_size, len(query) - offset), 1)
                   for offset, chunk in zip(range(0, len(query), batch_size), output)):
                raise ValueError("任13纯前向模型温度每点只能一列，不允许FEM静默回退")
            if repetition >= warmups:
                timings.append(seconds)
            if repetition == warmups + repeats - 1:
                result = torch.cat(output, dim=0)
    if result is None:
        raise ValueError("任13前向未输出任何节点")
    readback_clock = time.perf_counter()
    temperatures = result.cpu().numpy().reshape(-1)
    readback_seconds = time.perf_counter() - readback_clock
    if not np.isfinite(temperatures).all():
        raise ValueError("任13模型输出非有限温度，不许回退FEM")
    return {"纯模型同步前向秒": timings, "查询设备传输秒": transfer_seconds,
            "输出回读秒": readback_seconds, "temperature_k": temperatures}


def export_identical_rz_format(
    grid: RZTimingGrid, temperature_k: np.ndarray, destination: str | Path, *,
    checkpoint_sha: str,
) -> dict[str, Any]:
    from sic_cu.prediction import Prediction, PredictionMetadata, stable_time_from_maximum
    from sic_cu.visualization.export import export_prediction

    output = _registered_path(destination)
    if output.exists():
        raise FileExistsError("任13导出目录已有工件，不许覆盖旧原件")
    if output == PROJECT_ROOT or PROJECT_ROOT not in output.parents:
        raise ValueError("任13导出仅限全新项目内文件")
    field = np.asarray(temperature_k, dtype=np.float32)
    if (field.shape != (101, RZ_NODES) or not np.isfinite(field).all()
            or grid.coordinates_rz_m.shape != (RZ_NODES, 2)
            or grid.material_ids.shape != (RZ_NODES,)
            or not np.array_equal(grid.times_s, RZ_TIMES)
            or len(checkpoint_sha) != 64):
        raise ValueError("任13完整RZ场/型号SHA必须事前一致且有限")
    _check_rz_coordinates(grid.coordinates_rz_m, grid.material_ids)
    maximum = field.max(axis=1)
    metadata = PredictionMetadata(
        source=f"RZ_checkpoint_SHA256:{checkpoint_sha}",
        support_domain="registered_RZ_309W_HF_train_query",
        ir_coverage="inside_ir_power_range",
        lf_power_coverage="inside_lf_training_range",
        hf_power_coverage="inside_hf_training_range",
        time_support="inside_declared_0_200_s_model_domain",
        deterministic=True,
        warnings=("HF experimental internal ground truth unavailable; nominal energy audit failed.",
                  "Axisymmetric rotation is not azimuthal asymmetry or moving boundary."),
    )
    prediction = Prediction(
        power_w=BENCHMARK_POWER_W, times_s=grid.times_s,
        coordinates_rz_m=grid.coordinates_rz_m, material_ids=grid.material_ids,
        mean_temperature_k=field, q05_temperature_k=field.copy(),
        q95_temperature_k=field.copy(), max_temperature_k=maximum,
        stable_time_s=stable_time_from_maximum(grid.times_s, maximum), metadata=metadata,
    )
    return export_prediction(prediction, output, theta_resolution=12,
                             vtk_times_s=(0.0, 50.0, 100.0, 150.0, 200.0))


def check_lf_receipt(row: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    seconds = receipt.get("training_seconds")
    epochs = receipt.get("epochs_completed")
    if (row.get("lf_method") not in ("deeponet_pinn", "mlp_pinn")
            or receipt.get("method") != row.get("lf_method")
            or receipt.get("seed") != row.get("seed")
            or not isinstance(epochs, int) or epochs <= 0
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds <= 0
            or receipt.get("material_passport", {}).get("experiment_data_used") is not False):
        raise ValueError("任13 LF预训来源必须同seed/同方法、仅LF仿真且真实正训练成本")
    return {"可复用LF预训耗时秒": float(seconds), "LF预训实际轮次": epochs,
            "原LF_FEM仿真生成耗时秒": None,
            "口径": "历史LF网络预训练单次含评价；原FEM仿真原始成本未核"}


def check_checkpoint_receipt_consistency(
    row: Mapping[str, Any], checkpoint: Mapping[str, Any], receipt: Mapping[str, Any],
) -> None:
    arm = row.get("arm")
    if arm in ("B0", "MLP"):
        epoch = receipt.get("best_epoch")
        score = receipt.get("best_validation_selection_score_k")
        actual = checkpoint.get("validation_selection_score_k")
    elif arm in ("E0", "F2"):
        epoch = receipt.get("观测最佳全局轮次")
        score = receipt.get("观测最佳HF合法选分_摄氏度")
        actual = checkpoint.get("validation_selection_score_c")
    else:
        raise ValueError("任13最佳模型来源方法只能RZ注册四臂")
    if (checkpoint.get("seed") != row.get("seed")
            or checkpoint.get("epoch") != epoch or not isinstance(epoch, int)):
        raise ValueError("任13最佳模型来源实际轮次/seed与成本收据不符")
    if (not isinstance(score, (int, float)) or not isinstance(actual, (int, float))
            or not math.isfinite(score) or not math.isfinite(actual)
            or not math.isclose(float(score), float(actual), abs_tol=1e-4, rel_tol=0)):
        raise ValueError("任13最佳模型合法验证选分与已登记来源收据不同")


def check_registered_origin(row: Mapping[str, Any]) -> None:
    arm, seed = row.get("arm"), row.get("seed")
    if arm not in ARMS or seed not in range(5):
        raise ValueError("任13来源方法/seed非登记四臂五seed")
    if arm == "B0":
        root = RUN_ROOT / f"deeponet_pinn_mf_seed{seed}"
    elif arm == "MLP":
        root = RUN_ROOT / f"mlp_pinn_mf_seed{seed}"
    elif arm == "E0":
        root = E0_ROOT / E0_DIRS[seed]
    else:
        root = F2_ROOT / F2_DIRS[seed]
    receipt = "metrics.json" if arm in ("B0", "MLP") else "阶段报告.json"
    if (_registered_path(row.get("checkpoint", "")) != (root / "best.pt").resolve()
            or _registered_path(row.get("receipt", "")) != (root / receipt).resolve()):
        raise ValueError("任13模型来源必须原已封旧发布或已封E0/F2目录及其真实收据")


def check_source_tar_members(source_tar: str | Path, registry: str | Path) -> None:
    registry_path = _registered_path(registry)
    if PROJECT_ROOT not in registry_path.parents:
        raise ValueError("任13来源预算事前登记只能在项目内")
    members = set(SOURCE_MEMBERS) | {str(registry_path.relative_to(PROJECT_ROOT))}
    try:
        with tarfile.open(_registered_path(source_tar), mode="r:gz") as archive:
            saved = archive.getmembers()
            if (len(saved) != len(members) or {member.name for member in saved} != members
                    or any(not member.isfile() or Path(member.name).is_absolute()
                           or ".." in Path(member.name).parts for member in saved)):
                raise ValueError("任13源码tar成员不完整、非正规文件或项目外路径")
            for member in saved:
                source = archive.extractfile(member)
                assert source is not None
                if hashlib.sha256(source.read()).hexdigest() != _sha(PROJECT_ROOT / member.name):
                    raise ValueError("任13源码tar成员快照与执行现源码/预算字节不同")
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise ValueError("任13源码tar成员读取失败或不是有效归档") from exc


def _five_sha_rows(values: Any, description: str) -> list[dict[str, str]]:
    if (not isinstance(values, list) or len(values) != 5
            or any(not isinstance(row, dict)
                   or set(row) != {"checkpoint_sha256", "receipt_sha256"}
                   or any(not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)
                          for sha in row.values()) for row in values)):
        raise ValueError(f"任13{description}五seed检查点与真实成本登记必须各有两SHA")
    return values


def registered_lf_rows(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    index = cfg.get("LF来源")
    if not isinstance(index, dict) or set(index) != {"deeponet_pinn", "mlp_pinn"}:
        raise ValueError("任13可复用LF来源只许DeepONet/MLP独立各五seed")
    rows = []
    for method, values in index.items():
        for seed, hashes in enumerate(_five_sha_rows(values, f"{method}LF来源")):
            root = RUN_ROOT / f"{method}_lf_seed{seed}"
            rows.append({"lf_method": method, "seed": seed,
                         "checkpoint": str(root / "best.pt"),
                         "receipt": str(root / "metrics.json"), **hashes})
    return rows


def registered_candidate_rows(
    cfg: Mapping[str, Any], lf_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    index = cfg.get("模型")
    if not isinstance(index, dict) or set(index) != set(ARMS):
        raise ValueError("任13模型来源四方法各五seed且20身份不能缺席")
    paired = {(lf["lf_method"], lf["seed"]): lf for lf in lf_rows}
    if len(paired) != 10:
        raise ValueError("任13LF五seed可复用来源不可用其他seed偷换")
    if any(not isinstance(cfg.get(key), str)
           or not re.fullmatch(r"[0-9a-f]{64}", cfg[key])
           for key in ("正式E0预登记配置SHA256", "正式F2预登记配置SHA256")):
        raise ValueError("任13 E0/F2原已封正式训练预登记配置SHA不可缺失")
    rows = []
    for arm in ARMS:
        for seed, hashes in enumerate(_five_sha_rows(index[arm], f"{arm}模型来源")):
            if arm == "B0":
                root = RUN_ROOT / f"deeponet_pinn_mf_seed{seed}"
            elif arm == "MLP":
                root = RUN_ROOT / f"mlp_pinn_mf_seed{seed}"
            elif arm == "E0":
                root = E0_ROOT / E0_DIRS[seed]
            else:
                root = F2_ROOT / F2_DIRS[seed]
            lf_method = "mlp_pinn" if arm == "MLP" else "deeponet_pinn"
            rows.append({"arm": arm, "seed": seed,
                         "checkpoint": str(root / "best.pt"),
                         "receipt": str(root / ("metrics.json" if arm in ("B0", "MLP")
                                               else "阶段报告.json")),
                         "lf_checkpoint_sha256": paired[lf_method, seed]["checkpoint_sha256"],
                         "正式预登记配置SHA256":
                             cfg["正式E0预登记配置SHA256"] if arm == "E0" else
                             cfg["正式F2预登记配置SHA256"] if arm == "F2" else None,
                         **hashes})
    return rows


class HighOnlyRZModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        if not isinstance(model, torch.nn.Module):
            raise ValueError("任13必须有真实高保真checkpoint模型，FEM不能回退")
        self.model = model

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.model(values, fidelity="high")


def benchmark_registered_models(
    cfg: Mapping[str, Any], output: str | Path, *, device: torch.device,
) -> dict[str, Any]:
    from sic_cu.models.multifidelity import AdditiveCorrectionModel
    from sic_cu.prediction import Predictor

    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("任13正式20模型比较须真CUDA同步；CPU仅合成夹具")
    rows, lf_rows = cfg["登记模型来源"], cfg["登记LF来源"]
    if (len(rows) != 20 or len(lf_rows) != 10
            or {(item["arm"], item["seed"]) for item in rows}
               != {(arm, seed) for arm in ARMS for seed in range(5)}):
        raise ValueError("任13正式CUDA不得缺四臂五seed或十份真实LF来源")
    destination = _registered_path(output)
    if (destination.exists() or PROJECT_ROOT not in destination.parents
            or destination == PROJECT_ROOT):
        raise ValueError("任13正式输出只能全新项目内目录")
    first = time.perf_counter()
    for row in rows:
        check_registered_origin(row)
        check_registered_candidate(row, row["checkpoint"], row["receipt"])
        payload = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
        receipt = json.loads(Path(row["receipt"]).read_text(encoding="utf-8"))
        check_checkpoint_payload(row, payload)
        check_checkpoint_receipt_consistency(row, payload, receipt)
    identity_seconds = time.perf_counter() - first
    lf_by_identity = {(source["lf_method"], source["seed"]): source for source in lf_rows}
    for lf in lf_rows:
        for name, path in (("checkpoint_sha256", lf["checkpoint"]),
                           ("receipt_sha256", lf["receipt"])):
            artifact = _registered_path(path)
            if not artifact.is_file() or lf[name] != _sha(artifact):
                raise ValueError("任13 LF来源被修改，禁止开始正式GPU计时")
    destination.mkdir(parents=True)
    report: dict[str, Any] = {
        "对象": "RZ原装置四臂20模型同网格成本；不含任10一维人为板或FEM", "设备": str(device),
        "CUDA卡名称": torch.cuda.get_device_name(device), "旧TEST温度读取": False,
        "同功率HF实验温度读取": False, "20模型最佳身份预检秒": identity_seconds,
        "RZ源网格五列": list(GRID_COLUMNS), "功率_W": BENCHMARK_POWER_W,
        "时间帧": 101, "每帧节点": RZ_NODES, "模型": [],
    }
    for row in rows:
        model_root = destination / f"{row['arm']}_seed{row['seed']}"
        read_clock = time.perf_counter()
        grid = load_registered_grid(GRID_PATH, cfg["RZ网格SHA256"])
        mesh_read_seconds = time.perf_counter() - read_clock
        checkpoint_clock = time.perf_counter()
        predictor = Predictor(row["checkpoint"], device=device)
        _sync(device)
        checkpoint_seconds = time.perf_counter() - checkpoint_clock
        if (predictor.method != "multifidelity_correction"
                or not isinstance(predictor.model, AdditiveCorrectionModel)
                or not predictor.supports_point_queries):
            raise ValueError("任13 registered RZ checkpoint不是高保真校正器，不准静默FEM/LF回退")
        coordinate_clock = time.perf_counter()
        query = build_fixed_queries(grid, BENCHMARK_POWER_W)
        query_seconds = time.perf_counter() - coordinate_clock
        measured = measure_forward_parts(
            HighOnlyRZModel(predictor.model), query, device,
            batch_size=cfg["batch_size"], warmups=cfg["预热重复"],
            repeats=cfg["有效重复"],
        )
        model_result = measured.pop("temperature_k").reshape(101, RZ_NODES)
        export_clock = time.perf_counter()
        exported = export_identical_rz_format(grid, model_result, model_root,
                                              checkpoint_sha=row["checkpoint_sha256"])
        export_seconds = time.perf_counter() - export_clock
        lf_method = "mlp_pinn" if row["arm"] == "MLP" else "deeponet_pinn"
        lf = lf_by_identity[lf_method, row["seed"]]
        lf_receipt = json.loads(Path(lf["receipt"]).read_text(encoding="utf-8"))
        hf_receipt = json.loads(Path(row["receipt"]).read_text(encoding="utf-8"))
        costs = check_training_receipt(row, hf_receipt)
        lf_costs = check_lf_receipt(lf, lf_receipt)
        item = {
            "方法": row["arm"], "seed": row["seed"], "模型SHA256": row["checkpoint_sha256"],
            "HF训练成本原件SHA256": row["receipt_sha256"],
            "LF预训模型SHA256": lf["checkpoint_sha256"],
            "LF预训成本原件SHA256": lf["receipt_sha256"],
            "网格五列读取秒": mesh_read_seconds, "模型checkpoint读取与GPU重载秒": checkpoint_seconds,
            "查询五列构造秒": query_seconds, "查询设备传输秒": measured["查询设备传输秒"],
            "完整场同步纯模型前向各次秒": measured["纯模型同步前向秒"],
            "完整场同步纯模型前向中位秒": float(np.median(measured["纯模型同步前向秒"])),
            "输出回读秒": measured["输出回读秒"], "统一过程场导出秒": export_seconds,
            "HF离线训练成本": costs, "LF离线可复用预训成本": lf_costs,
            "原LF_FEM仿真生成成本": "缺真实原件，不得冒称总成本已闭合",
            "归档文件数": len([path for path in model_root.rglob("*") if path.is_file()]),
            "输出SHA256": {str(path.relative_to(model_root)): _sha(path)
                          for path in sorted(model_root.rglob("*")) if path.is_file()},
            "唯一模型来源非FEM": True, "旧TEST温度读取": False,
            "同功率HF实验温度读取": False,
        }
        (model_root / "计时分项与来源.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        report["模型"].append(item)
    (destination / "四臂五种子合法推理分项与成本.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return report


def validate_registered_timing_protocol(cfg: Mapping[str, Any]) -> None:
    expected = {
        "注册功率_W": BENCHMARK_POWER_W, "起止时刻_s": [0.0, 200.0],
        "时间步_s": 2.0, "batch_size": 2048,
        "预热重复": 2, "有效重复": 5, "旋转theta分辨率": 12,
        "vtk时刻_s": [0.0, 50.0, 100.0, 150.0, 200.0],
        "时间帧": 101, "每帧节点": RZ_NODES,
        "读网格五列": list(GRID_COLUMNS),
    }
    if any(cfg.get(field) != value for field, value in expected.items()):
        raise ValueError("任13固定功率/时窗、batch、预热、有效计时或导出格式预算变动")
