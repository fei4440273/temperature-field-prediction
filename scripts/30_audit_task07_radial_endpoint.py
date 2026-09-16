#!/usr/bin/env python
"""V4与任07合法HF顶部径向25毫米端点的版本化独立审计。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tarfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sic_cu.config import load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.processed import load_processed_ir_observations
from sic_cu.eval.development_v4 import (
    _observation_arrays, _empty_metrics, _predict, _top_coordinates, error_statistics,
    load_existing_baseline,
)
from sic_cu.train.task07_source import MANIFEST_PATH, MANIFEST_SHA256, validate_task07_sources


WINDOW_NAMES = ("center_0_8_mm", "middle_8_17_mm", "outer_17_25_mm")
WINDOW_BOUNDS = ((0.0, 8.0, True), (8.0, 17.0, False), (17.0, 25.0, False))
CONFIG_PATH = ROOT / "configs/optimization_v4.yaml"
CONFIG_SHA256 = "59c1e7f569ba846effc5b3fd7f1776d55f4c2341b25f012809e60b4c6b913334"
REGISTRY_PATH = ROOT / "研究记录/任务07_正式五种子重训/有效运行配置_实现修复后.yaml"
REGISTRY_SHA256 = "f92abf2cc2f97aa6bddcd16768707ee581470bbe0f9d3eb191598dbf7bbc0dba"
OBSERVATIONS_PATH = (ROOT / "研究记录/任务07_正式五种子重训/"
                     "正式五种子HF合法验证逐窗明细_20260916T022125+0800")
OBSERVATIONS_MANIFEST_SHA256 = "53356a08026a7837642aea1d3ba4e4f523a8888b88052115ea330562eec381ac"
INFERENCE_PATH = (ROOT / "研究记录/任务07_正式五种子重训/"
                  "正式五种子全场推理回归_单ULP修订_20260916T021940+0800")
INFERENCE_MANIFEST_SHA256 = "92e71b03b1b0d1c7d129d077426fd3905b477450584e9fccb5744916ef87e111"
POWER_COUNTS = {115.2: (1919, 19), 403.0: (2424, 24), 630.5: (2929, 29)}
RADIAL_LABELS = {
    "顶部中心0—8毫米": "center_0_8_mm",
    "顶部中圈8—17毫米": "middle_8_17_mm",
    "顶部外圈17—25毫米": "outer_17_25_mm",
}
SOURCE_FILES = (
    "scripts/30_audit_task07_radial_endpoint.py",
    "tests/test_task07_radial_endpoint_audit.py",
    "src/sic_cu/eval/development_v4.py",
    "src/sic_cu/data/processed.py",
    "src/sic_cu/train/task07_source.py",
    "src/sic_cu/train/task07_formal.py",
    "src/sic_cu/models/multifidelity.py",
    "src/sic_cu/prediction.py",
    "scripts/27_verify_task07_inference.py",
    "configs/optimization_v4.yaml",
    "configs/splits.yaml",
    "reports/development_v4/baseline_manifest.yaml",
    "研究记录/任务07_正式五种子重训/有效运行配置_实现修复后.yaml",
)


def radial_masks(
    radii_m: np.ndarray, windows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    radii = np.asarray(radii_m)
    if radii.ndim != 1 or radii.dtype != np.float32 or len(radii) == 0:
        raise ValueError("径向审计只能使用非空原始米制float32一维半径")
    if len(windows) != 3 or any((item.get("name"), item.get("lower"), item.get("upper"),
                                    item.get("lower_closed")) != (name, *bounds)
                                   for item, name, bounds in zip(windows, WINDOW_NAMES, WINDOW_BOUNDS)):
        raise ValueError("径向审计不可改变V4原三窗口的数学边界和开闭端点")
    represented_25m = np.float32(25.0 / 1000.0)
    if not np.isfinite(radii).all() or (radii < 0).any() or (radii > represented_25m).any():
        raise ValueError("合法验证Top半径须全部处于原[0,25]毫米定义域")
    old_mm = radii.astype(np.float64) * 1000.0
    old_masks = []
    native_masks = []
    for window in windows:
        lo_mm, hi_mm = float(window["lower"]), float(window["upper"])
        lo_m = np.float32(lo_mm / 1000.0)
        hi_m = np.float32(hi_mm / 1000.0)
        old_lower = old_mm >= lo_mm if window["lower_closed"] else old_mm > lo_mm
        native_lower = radii >= lo_m if window["lower_closed"] else radii > lo_m
        old_masks.append(old_lower & (old_mm <= hi_mm))
        native_masks.append(native_lower & (radii <= hi_m))
    old = tuple(old_masks)
    fixed = tuple(native_masks)
    if (any(np.count_nonzero(fixed[index] ^ old[index]) for index in (0, 1))
            or not np.array_equal(fixed[2] ^ old[2], radii == represented_25m)
            or not np.all(sum(mask.astype(np.int8) for mask in fixed) == 1)):
        raise ValueError("新旧径向窗的唯一合法差异是原生float32表示的25毫米闭端点")
    return old, fixed


def paired_radial_records(
    frame: pl.DataFrame, prediction: np.ndarray,
    windows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]], *,
    seed: int, power_w: float, arm: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if seed not in range(5) or arm not in ("B0", "任07E0") or power_w not in (115.2, 403.0, 630.5):
        raise ValueError("径向验证仅允许B0/任07E0五种子和原三份合法HF验证功率")
    predictions = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if frame.height != len(predictions) or frame.is_empty() or not np.isfinite(predictions).all():
        raise ValueError("顶部真实观测和对应HF模型逐点预测必须同序非空且有限")
    old_masks, fixed_masks = radial_masks(frame["r_m"].to_numpy(), windows)
    arrays = _observation_arrays(frame, predictions, "Top")
    result: list[list[dict[str, Any]]] = [[], []]
    for index, masks in enumerate((old_masks, fixed_masks)):
        for window, selected in zip(windows, masks):
            metrics = (
                error_statistics(arrays["target"][selected], arrays["prediction"][selected],
                                 arrays["weights"][selected])
                if selected.any() else _empty_metrics("no_top_observations_in_preregistered_radial_window")
            )
            result[index].append({
                "arm": arm, "seed": seed, "split": "validation", "power_w": power_w,
                "radial_window": window["name"], "lower_mm": window["lower"],
                "upper_mm": window["upper"], "lower_closed": window["lower_closed"],
                "upper_closed": True, "sample_count": int(selected.sum()), **metrics,
            })
    return result[0], result[1]


def _source_identity() -> dict[str, Any]:
    if sha256_file(MANIFEST_PATH) != MANIFEST_SHA256 or sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("V4历史来源清单或原径向配置SHA漂移，不能登记新版审计")
    if sha256_file(REGISTRY_PATH) != REGISTRY_SHA256:
        raise ValueError("任07实现修复后正式预登记原件SHA漂移")
    manifest = load_yaml(MANIFEST_PATH)
    config = load_yaml(CONFIG_PATH)
    if (manifest.get("test_labels_consumed_this_round") is not False
            or config.get("allow_test_labels") is not False
            or config["diagnostics"].get("evaluation_splits") != ["train", "validation"]
            or config["diagnostics"].get("radial_windows_mm") != [
                {"name": name, "lower": lower, "upper": upper, "lower_closed": closed}
                for name, (lower, upper, closed) in zip(WINDOW_NAMES, WINDOW_BOUNDS)]):
        raise ValueError("V4原径向窗口或合法开发验证集合与冻结登记不符")
    sources = validate_task07_sources()
    if set(sources) != set(range(5)) or any(source.hf_validation_powers_w !=
                                            (115.2, 403.0, 630.5) for source in sources.values()):
        raise ValueError("B0五个配对来源模型或HF合法验证功率不完整")
    old_radius = ROOT / "reports/development_v4/diagnostics/validation_by_radius.csv"
    archived_old_sha = manifest["diagnostic_file_sha256s"]["diagnostics/validation_by_radius.csv"]
    original_obs_manifest = OBSERVATIONS_PATH / "工件SHA256.json"
    if (sha256_file(old_radius) != archived_old_sha
            or sha256_file(original_obs_manifest) != OBSERVATIONS_MANIFEST_SHA256):
        raise ValueError("V4历史径向CSV或任07原合法观测SHA清单漂移")
    obs_hashes = json.loads(original_obs_manifest.read_text(encoding="utf-8"))
    if (set(obs_hashes) != {"五种子观测验证摘要.json", "合法验证逐功率逐模态时间窗.csv",
                           "原始合法指标与中文列映射.json", "合法验证逐功率逐模态.csv",
                           "合法验证顶部逐功率径向窗.csv"}
            or any(sha256_file(OBSERVATIONS_PATH / name) != digest
                   for name, digest in obs_hashes.items())):
        raise ValueError("任07正式原合法HF观察工件不完整或SHA漂移")
    known_best = _locked_formal_best_identity()
    return {
        "V4历史来源清单SHA256": MANIFEST_SHA256, "V4原配置SHA256": CONFIG_SHA256,
        "任07事前正式登记SHA256": REGISTRY_SHA256,
        "旧test_Data温度标签读取": False,
        "B0五份原最佳模型SHA256": {str(seed): sources[seed].hf_checkpoint_sha256
                                   for seed in range(5)},
        "原合法顶部IR数据SHA256": sha256_file(ROOT / "data/processed/experiment_ir_radial.parquet"),
        "原处理manifest_SHA256": sha256_file(ROOT / "data/processed/manifest.json"),
        "V4原径向合法指标CSV_SHA256": archived_old_sha,
        "任07原合法观测工件清单SHA256": OBSERVATIONS_MANIFEST_SHA256,
        "任07原径向合法指标CSV_SHA256": obs_hashes["合法验证顶部逐功率径向窗.csv"],
        "任07原推理工件清单SHA256": INFERENCE_MANIFEST_SHA256,
        "任07原五份观测最佳模型SHA256": {str(seed): row["best.pt"]
                                         for seed, row in known_best.items()},
    }


def register_source_snapshot(directory: str | Path) -> dict[str, Any]:
    output = Path(directory)
    if output.exists():
        raise FileExistsError("事前源码快照登记目录已存在，不可覆盖原件")
    identity = _source_identity()
    files = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    output.mkdir(parents=True)
    archive = output / "事前独立径向审计源码快照.tar.gz"
    with tarfile.open(archive, mode="w:gz") as tar:
        for name in SOURCE_FILES:
            tar.add(ROOT / name, arcname=name, recursive=False)
    payload = identity | {"源码逐文件SHA256": files,
                          "事前源码快照SHA256": sha256_file(archive),
                          "事前源码快照文件名": archive.name}
    manifest = output / "事前独立径向审计登记SHA256.json"
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                   allow_nan=False) + "\n", encoding="utf-8")
    return {"source_manifest": str(manifest), "source_manifest_sha256": sha256_file(manifest),
            "source_archive": str(archive), "source_archive_sha256": sha256_file(archive)}


def verify_source_snapshot(manifest_path: str | Path, manifest_sha256: str) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.is_file() or sha256_file(path) != manifest_sha256:
        raise ValueError("事前独立径向审计登记SHA缺失或原件漂移")
    data = json.loads(path.read_text(encoding="utf-8"))
    current = _source_identity()
    if any(data.get(name) != value for name, value in current.items()):
        raise ValueError("V4/任07五份历史来源或顶部合法数据与事前登记SHA漂移")
    if data.get("源码逐文件SHA256") != {name: sha256_file(ROOT / name) for name in SOURCE_FILES}:
        raise ValueError("审计脚本/测试/V4原源码或任07配对实现源码SHA事后漂移")
    archive = path.parent / "事前独立径向审计源码快照.tar.gz"
    if (data.get("事前源码快照文件名") != archive.name or not archive.is_file()
            or data.get("事前源码快照SHA256") != sha256_file(archive)):
        raise ValueError("事前源码快照tar原件SHA漂移或缺失")
    with tarfile.open(archive, mode="r:gz") as tar:
        if set(tar.getnames()) != set(SOURCE_FILES) or any(
            hashlib.sha256(tar.extractfile(name).read()).hexdigest() != data[
                "源码逐文件SHA256"][name] for name in SOURCE_FILES
        ):
            raise ValueError("源码快照tar内源文件无法与事前逐文件SHA交叉核对")
    return data


def _locked_reference_metrics() -> dict[str, list[dict[str, Any]]]:
    if sha256_file(MANIFEST_PATH) != MANIFEST_SHA256 or sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("V4冻结原窗口配置或五种子清单SHA漂移")
    manifest = load_yaml(MANIFEST_PATH)
    old = ROOT / "reports/development_v4/diagnostics/validation_by_radius.csv"
    obs_manifest = OBSERVATIONS_PATH / "工件SHA256.json"
    new = OBSERVATIONS_PATH / "合法验证顶部逐功率径向窗.csv"
    if (sha256_file(old) != manifest["diagnostic_file_sha256s"][
            "diagnostics/validation_by_radius.csv"]
            or sha256_file(obs_manifest) != OBSERVATIONS_MANIFEST_SHA256
            or sha256_file(new) != json.loads(obs_manifest.read_text(encoding="utf-8"))[
                "合法验证顶部逐功率径向窗.csv"]):
        raise ValueError("V4/任07原始径向45条指标CSV或独立SHA来源漂移")
    baseline = [row for row in pl.read_csv(old).to_dicts() if row["split"] == "validation"]
    recent: list[dict[str, Any]] = []
    for row in pl.read_csv(new).to_dicts():
        if row["数据划分"] != "高保真合法验证" or row["模态"] != "顶部红外":
            raise ValueError("任07原径向工件含非HF合法验证Top样本")
        name = RADIAL_LABELS.get(row["径向窗"])
        if name is None:
            raise ValueError("任07原径向窗名称不属于固定V4窗口")
        recent.append({
            "seed": row["种子"], "split": "validation", "power_w": row["功率_W"],
            "radial_window": name, "lower_mm": row["径向起毫米"],
            "upper_mm": row["径向止毫米"], "lower_closed": row["径向起含端点"],
            "upper_closed": row["径向止含端点"], "sample_count": row["实际观测点数"],
            "rmse_c": row["RMSE_摄氏度"], "mae_c": row["MAE_摄氏度"],
            "signed_bias_c": row["有符号偏差_摄氏度"],
            "p95_abs_error_c": row["绝对误差95分位_摄氏度"],
            "max_abs_error_c": row["绝对误差最大值_摄氏度"],
        })
    for arm, rows in (("B0", baseline), ("任07E0", recent)):
        if (len(rows) != 45 or len({(row["seed"], row["power_w"], row["radial_window"])
                                    for row in rows}) != 45
                or {row["seed"] for row in rows} != set(range(5))
                or {row["power_w"] for row in rows} != set(POWER_COUNTS)
                or {row["radial_window"] for row in rows} != set(WINDOW_NAMES)
                or any(row["split"] != "validation" or row["upper_closed"] is not True
                       or (row["lower_mm"], row["upper_mm"], row["lower_closed"]) !=
                       WINDOW_BOUNDS[WINDOW_NAMES.index(row["radial_window"])]
                       for row in rows)):
            raise ValueError(f"{arm}历史原径向CSV缺五种子45条合法上界原指标")
    return {"B0": baseline, "任07E0": recent}


def _locked_formal_best_identity() -> dict[int, dict[str, str]]:
    manifest = INFERENCE_PATH / "审计工件SHA256.json"
    if not manifest.is_file() or sha256_file(manifest) != INFERENCE_MANIFEST_SHA256:
        raise ValueError("任07五seed正式推理清单原件SHA漂移或缺失")
    listed = json.loads(manifest.read_text(encoding="utf-8"))
    if (len(listed) != 71 or "运行摘要.json" not in listed
            or len({name.split("/")[0] for name in listed if "/" in name}) != 5
            or {file.relative_to(INFERENCE_PATH).as_posix() for file in INFERENCE_PATH.rglob("*")
                if file.is_file()} - {manifest.name} != set(listed)
            or any(sha256_file(INFERENCE_PATH / name) != digest
                   for name, digest in listed.items())):
        raise ValueError("任07推理五子+顶层71个工件清单或实际文件SHA缩水/漂移")
    summary = json.loads((INFERENCE_PATH / "运行摘要.json").read_text(encoding="utf-8"))
    entries = summary.get("五种子独立重载合法HF验证与完整过程", {})
    if (summary.get("正式预登记配置SHA256") != REGISTRY_SHA256
            or summary.get("设备") != "cuda" or set(entries) != set(map(str, range(5)))):
        raise ValueError("任07原推理报告须来自已登记真CUDA完成五种子最佳视图")
    result: dict[int, dict[str, str]] = {}
    run_root = ROOT / "研究记录/任务07_正式五种子重训"
    for seed in range(5):
        matches = list(run_root.glob(f"修复后正式_E0_种子{seed}_*"))
        if len(matches) != 1 or not matches[0].is_dir():
            raise ValueError(f"任07推理原审的seed{seed}正式训练目录须唯一且仍存在")
        locked = entries[str(seed)].get("输入阶段工件SHA256", {})
        if (set(locked) != {"best.pt", "阶段报告.json", "阶段_最近.pt", "阶段_观测最佳.pt",
                           "阶段_物理最佳.pt", "阶段_校正末.pt", "阶段_联合末.pt",
                           "阶段_训练末.pt", "training.jsonl"}
                or entries[str(seed)].get("源最佳模型SHA256") != locked["best.pt"]
                or locked["best.pt"] == locked["阶段_训练末.pt"]
                or any(sha256_file(matches[0] / name) != digest
                       for name, digest in locked.items())):
            raise ValueError(f"任07seed{seed}推理观测最佳视图/完整阶段/末态SHA身份失配")
        result[seed] = locked
    return result


def _locked_validation_top_frame() -> pl.DataFrame:
    frame = load_processed_ir_observations("validation").sort("power_w", "time_s", "r_m")
    if (sha256_file(ROOT / "data/processed/experiment_ir_radial.parquet") !=
            _source_identity()["原合法顶部IR数据SHA256"]
            or frame.height != sum(count for count, _ in POWER_COUNTS.values())
            or frame["r_m"].dtype != pl.Float32
            or {round(float(value), 4) for value in frame["power_w"].unique().to_list()} != set(POWER_COUNTS)
            or set(frame["split"].unique().to_list()) != {"validation"}):
        raise ValueError("合法顶部IR只准原三HF验证功率7272点且半径float32，禁止测试温度")
    for power_w, (count, frames) in POWER_COUNTS.items():
        group = frame.filter(pl.col("power_w") == power_w)
        if group.height != count or group["time_s"].n_unique() != frames:
            raise ValueError("合法顶部IR原三个验证功率采样点或真实时间帧缩水")
    return frame


def _compare_legacy_original(actual: Mapping[str, Any], archived: Mapping[str, Any]) -> float:
    keys = ("seed", "split", "power_w", "radial_window", "lower_mm", "upper_mm",
            "lower_closed", "upper_closed", "sample_count")
    if any(actual.get(key) != archived.get(key) for key in keys):
        raise ValueError("B0/任07原指标逐seed功率径向窗口或旧实际观测点数与锁定CSV不符")
    max_delta = 0.0
    for key in ("rmse_c", "mae_c", "signed_bias_c", "p95_abs_error_c", "max_abs_error_c"):
        a, b = actual.get(key), archived.get(key)
        if (type(a) not in (int, float) or type(b) not in (int, float)
                or not math.isfinite(a) or not math.isfinite(b)):
            raise ValueError("B0/任07历史原指标须是真实有限合法HF摄氏度误差")
        max_delta = max(max_delta, abs(float(a) - float(b)))
    if max_delta > 1e-4:
        raise ValueError("B0/任07重载原模型独立重算旧窗RMSE/偏差与历史原指标不符")
    return max_delta


def _check_power_partition(
    power_w: float, original: list[Mapping[str, Any]], fixed: list[Mapping[str, Any]],
    real_frame_count: int,
) -> None:
    if power_w not in POWER_COUNTS or len(original) != 3 or len(fixed) != 3:
        raise ValueError("25毫米版本化端点审计须原三合法HF功率/三径向窗")
    total, expected_frames = POWER_COUNTS[power_w]
    if (real_frame_count != expected_frames
            or sum(int(row["sample_count"]) for row in original) != total - real_frame_count
            or sum(int(row["sample_count"]) for row in fixed) != total
            or any(original[idx]["sample_count"] != fixed[idx]["sample_count"] for idx in (0, 1))
            or int(fixed[2]["sample_count"]) - int(original[2]["sample_count"]) != real_frame_count):
        raise ValueError("新窗须仅回填每真实Top帧一个25毫米外圈端点，三窗恰好覆盖所有合法观测")


def _inference_gate():
    script = ROOT / "scripts/27_verify_task07_inference.py"
    spec = importlib.util.spec_from_file_location("task07_radial_official_inference_gate", script)
    if spec is None or spec.loader is None:
        raise ImportError("任07正式独立五seed best/末状态原件门禁入口缺失")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _localized_radius(row: Mapping[str, Any]) -> dict[str, Any]:
    inverse = {window: label for label, window in RADIAL_LABELS.items()}
    return {
        "模型组": "V4历史B0" if row["arm"] == "B0" else "任07修复后E0",
        "种子": row["seed"], "数据划分": "高保真合法验证Top",
        "功率_W": row["power_w"], "径向窗": inverse[row["radial_window"]],
        "径向起毫米": row["lower_mm"], "径向止毫米": row["upper_mm"],
        "径向起含端点": row["lower_closed"], "径向止含端点": row["upper_closed"],
        "实际观测点数": row["sample_count"], "RMSE_摄氏度": row["rmse_c"],
        "MAE_摄氏度": row["mae_c"], "有符号偏差_摄氏度": row["signed_bias_c"],
        "绝对误差95分位_摄氏度": row["p95_abs_error_c"],
        "绝对误差最大值_摄氏度": row["max_abs_error_c"],
    }


def _write_report(
    directory: str | Path, *, source_manifest: str | Path, source_manifest_sha256: str,
    snapshot: Mapping[str, Any], originals: Mapping[str, list[dict[str, Any]]],
    corrected: Mapping[str, list[dict[str, Any]]], lineage: list[dict[str, Any]],
    worst_legacy_delta: float, formal: bool,
) -> dict[str, Any]:
    destination = Path(directory)
    if destination.exists():
        raise FileExistsError("独立径向审计不得覆盖既有目录或报告原件")
    if (set(originals) != {"B0", "任07E0"} or set(corrected) != set(originals)
            or any(len(originals[arm]) != 45 or len(corrected[arm]) != 45 for arm in originals)
            or {item.get("种子") for item in lineage} != set(range(5))
            or len(lineage) != 5 or not math.isfinite(worst_legacy_delta)
            or worst_legacy_delta > 1e-4):
        raise ValueError("十模型B0/任07原与修各45行、五个真seed或历史原指标重算资格缩水")
    for arm in ("B0", "任07E0"):
        old = {(row["seed"], row["power_w"], row["radial_window"]): row
               for row in originals[arm]}
        new = {(row["seed"], row["power_w"], row["radial_window"]): row
               for row in corrected[arm]}
        if (len(old) != 45 or set(old) != set(new) or any(
            row["arm"] != arm or row["split"] != "validation"
            for row in originals[arm] + corrected[arm]
        )):
            raise ValueError("十模型逐seed/功率/径向窗的原修90行身份重复或非合法验证")
        for seed in range(5):
            for power_w, (_, frames) in POWER_COUNTS.items():
                ids = [(seed, power_w, name) for name in WINDOW_NAMES]
                _check_power_partition(power_w, [old[key] for key in ids],
                                       [new[key] for key in ids], frames)
    status = ("真实CUDA完成十个锁定观测最佳模型同数据旧表示与25毫米端点修订双规约重评"
              if formal else "CPU格式诊断；不可充当正式十模型GPU径向审计")
    localized: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    comparison: list[dict[str, Any]] = []
    for arm in ("B0", "任07E0"):
        source = [_localized_radius(row) for row in originals[arm]]
        repair = [_localized_radius(row) for row in corrected[arm]]
        localized[arm] = (source, repair)
        for a, b in zip(source, repair):
            identity = ("模型组", "种子", "数据划分", "功率_W", "径向窗",
                        "径向起毫米", "径向止毫米", "径向起含端点", "径向止含端点")
            if any(a[key] != b[key] for key in identity):
                raise ValueError("同表旧表示与新表原生米制float32模型组/窗口身份不一致")
            comparison.append({**{key: a[key] for key in identity},
                               "旧表示实际观测点数": a["实际观测点数"],
                               "闭端点修订实际观测点数": b["实际观测点数"],
                               **{f"旧表示{key}": a[key] for key in (
                                   "RMSE_摄氏度", "MAE_摄氏度", "有符号偏差_摄氏度",
                                   "绝对误差95分位_摄氏度", "绝对误差最大值_摄氏度")},
                               **{f"闭端点修订{key}": b[key] for key in (
                                   "RMSE_摄氏度", "MAE_摄氏度", "有符号偏差_摄氏度",
                                   "绝对误差95分位_摄氏度", "绝对误差最大值_摄氏度")}})
    if len(comparison) != 90:
        raise ValueError("十模型原修同表须具全部90行真实对应指标")
    full_group = 5 * sum(count for count, _ in POWER_COUNTS.values())
    omitted_group = 5 * sum(frames for _, frames in POWER_COUNTS.values())
    summary = {
        "报告资格": status, "审计来源登记原件": str(source_manifest),
        "事前源码登记文件SHA256": source_manifest_sha256,
        "事前源码快照tar_SHA256": snapshot["事前源码快照SHA256"],
        "V4原五种子来源清单SHA256": snapshot["V4历史来源清单SHA256"],
        "V4原径向配置SHA256": snapshot["V4原配置SHA256"],
        "任07事前正式训练登记SHA256": snapshot["任07事前正式登记SHA256"],
        "V4_B0原指标CSV_SHA256": snapshot["V4原径向合法指标CSV_SHA256"],
        "任07E0原指标CSV_SHA256": snapshot["任07原径向合法指标CSV_SHA256"],
        "原合法顶部IR数据SHA256": snapshot["原合法顶部IR数据SHA256"],
        "五种子真实模型来源": lineage, "旧指标独立重算最大温差_摄氏度": worst_legacy_delta,
        "模型组各五种子原表示总点数": full_group - omitted_group,
        "模型组各五种子闭端点修订总点数": full_group,
        "模型组各五种子恢复25毫米点数": omitted_group,
        "十模型总恢复25毫米点数": 2 * omitted_group,
        "原与修两套每模型组径向指标行数": 45,
        "同表原修两模型组对应行数": 90,
        "合法HF选择macro_v1和能源重算": False,
        "旧test_Data温度标签读取": False,
        "限制": "只纠正顶部径向窗25毫米float32表示舍入误排；不改主选分、能源、旧V4或任07冻结源码/原件，不允许由历史与新训练预算不同的数值差推断因果改善。",
    }
    lines = [
        "# 顶部合法HF验证径向25毫米端点版本化独立审计", "",
        f"状态：{status}。", "",
        f"事前源码快照SHA256：`{summary['事前源码快照tar_SHA256']}`；登记文件SHA256：`{source_manifest_sha256}`。",
        f"历史V4原配置SHA256：`{summary['V4原径向配置SHA256']}`；原来源清单SHA256：`{summary['V4原五种子来源清单SHA256']}`。",
        f"任07事前正式登记SHA256：`{summary['任07事前正式训练登记SHA256']}`。",
        "", "## 唯一修订口径", "",
        "原数学窗口保持[0,8]、(8,17]、(17,25]毫米；原V4将原生float32米制半径转float64再乘1000，",
        "存为25毫米的端点成为25.00000037252903毫米而被旧`<=25`判据遗漏。新版只在原生米制float32表示内比较原窗口的开闭端点，",
        "每功率每真实帧只回填外圈一个25毫米端点，中心和中圈指标逐点不变。", "",
        "| 合法HF验证功率（瓦） | 每种子Top全点 | 每种子旧三窗点 | 每种子仅外圈回填端点 |",
        "| ---: | ---: | ---: | ---: |",
        *(f"| {power_w:g} | {count} | {count-frames} | {frames} |"
          for power_w, (count, frames) in POWER_COUNTS.items()), "",
        f"每模型组五种子回填{omitted_group}点，两模型组十份最佳模型合计回填{2*omitted_group}点；四份45行原/修指标及90行同表配对CSV逐文件SHA另存。",
        "", "## 模型与结论边界", "",
        "逐seed历史B0最佳模型与任07新E0观测最佳模型源SHA不同但各自经原LF/HF配对、真实best/末状态交叉；",
        "任07物理最佳另有独立身份，不混作本轮观测最佳。", "",
        "主合法HF宏选择分`macro_v1`、能源、训练模型参数与旧历史CSV均未由本轮径向展示改动；"
        "历史B0与任07初始化及预算不逐位一致，不得由径向误差差别声称训练机制带来增益。",
        "旧test_Data温度标签读取：否；真实HF内部完整场/原FEM热预算未由顶部观测径向窗审核。", "",
    ]
    destination.mkdir(parents=True)
    names = {
        ("B0", "old"): "V4_B0旧表示原指标45.csv",
        ("B0", "new"): "V4_B0闭端点修订指标45.csv",
        ("任07E0", "old"): "任07_E0旧表示原指标45.csv",
        ("任07E0", "new"): "任07_E0闭端点修订指标45.csv",
    }
    for (arm, version), name in names.items():
        pl.DataFrame(localized[arm][0 if version == "old" else 1]).write_csv(
            destination / name, null_value="",
        )
    pl.DataFrame(comparison).write_csv(destination / "十模型原修径向逐窗90行对照.csv", null_value="")
    (destination / "十模型端点修订摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    (destination / "版本化顶部径向端点中文审计报告.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )
    digests = {item.name: sha256_file(item) for item in destination.iterdir() if item.is_file()}
    (destination / "工件SHA256.json").write_text(
        json.dumps(digests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return summary | {"输出": str(destination), "工件SHA256": digests}


def audit_radial_endpoint(
    seed_directories: Mapping[int, str | Path], output_directory: str | Path, *,
    source_manifest: str | Path, source_manifest_sha256: str,
) -> dict[str, Any]:
    if set(seed_directories) != set(range(5)):
        raise ValueError("独立径向审计须完整提供正式五种子0–4，且不得伪造其他种子")
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("版本化审计输出目录已存在，不可覆盖原件")
    snapshot = verify_source_snapshot(source_manifest, source_manifest_sha256)
    roots = {seed: (Path(folder) if Path(folder).is_absolute() else ROOT / folder)
             for seed, folder in seed_directories.items()}
    if (any(not root.is_dir() or not root.name.startswith(f"修复后正式_E0_种子{seed}_")
            for seed, root in roots.items())
            or len({root.resolve() for root in roots.values()}) != 5):
        raise FileNotFoundError("任07种子0–4的实现修复后正式目录不存在或缺五份真正不同的原件")
    destination = output if output.is_absolute() else ROOT / output
    destination = destination.resolve()
    if (not destination.is_relative_to(ROOT.resolve())
            or any(destination == root.resolve() or destination.is_relative_to(root.resolve())
                   for root in roots.values())
            or not Path(source_manifest).resolve().is_relative_to(ROOT.resolve())):
        raise ValueError("正式径向审计和事前源码登记须是独立项目新目录，不可写入训练原件或临时外部空间")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise ValueError("正式十份径向模型公平重评必须为真实CUDA，不接受CPU诊断代替")
    official = _inference_gate()._preflight_five_seeds(
        roots, registry_path=REGISTRY_PATH, registry_sha256=REGISTRY_SHA256, device=device,
    )
    sources = validate_task07_sources()
    reference = _locked_reference_metrics()
    top = _locked_validation_top_frame()
    previous_best = _locked_formal_best_identity()
    windows = load_yaml(CONFIG_PATH)["diagnostics"]["radial_windows_mm"]
    all_old: dict[str, list[dict[str, Any]]] = {"B0": [], "任07E0": []}
    all_new: dict[str, list[dict[str, Any]]] = {"B0": [], "任07E0": []}
    lineage: list[dict[str, Any]] = []
    worst_legacy_delta = 0.0
    for seed in range(5):
        source = sources[seed]
        _, baseline, _, _, lf_path, hf_path = load_existing_baseline(
            source.hf_checkpoint_path.parent.parent, seed, device,
        )
        formal = official[seed]
        new_view = formal["root"] / "best.pt"
        if (sha256_file(lf_path) != source.lf_checkpoint_sha256
                or sha256_file(hf_path) != source.hf_checkpoint_sha256
                or not new_view.is_file()
                or formal["source"].hf_checkpoint_sha256 != source.hf_checkpoint_sha256
                or sha256_file(new_view) != previous_best[seed]["best.pt"]
                or sha256_file(formal["root"] / "阶段_观测最佳.pt") != previous_best[
                    seed]["阶段_观测最佳.pt"]):
            raise ValueError(f"seed{seed} B0原LF配对HF或任07正式新视图SHA来源失配")
        new_model = formal["predictor"].model
        lineage.append({
            "种子": seed, "V4_B0配对LF检查点SHA256": source.lf_checkpoint_sha256,
            "V4_B0观测最佳模型SHA256": source.hf_checkpoint_sha256,
            "任07修复版观测最佳模型SHA256": sha256_file(new_view),
            "任07独立合法macro_v1选分_摄氏度": float(formal["score"]),
            "任07最佳模型阶段原件SHA256": sha256_file(formal["root"] / "阶段_观测最佳.pt"),
            "任07真实末阶段原件SHA256": sha256_file(formal["root"] / "阶段_训练末.pt"),
        })
        for arm, model in (("B0", baseline), ("任07E0", new_model)):
            prediction = _predict(model, _top_coordinates(top), device, 65536)
            if len(prediction) != top.height or not np.isfinite(prediction).all():
                raise ValueError(f"seed{seed} {arm} 真CUDA顶部HF预测点数缩水或输出非有限值")
            for power_w, (expected_count, expected_frames) in POWER_COUNTS.items():
                mask = np.isclose(top["power_w"].to_numpy().astype(np.float64), power_w,
                                  atol=1e-4, rtol=0)
                group = top.filter(pl.col("power_w") == power_w)
                if int(mask.sum()) != group.height or group.height != expected_count:
                    raise ValueError(f"seed{seed} {arm} 真CUDA验证功率{power_w}W顶部观测和预测顺序失配")
                old, new = paired_radial_records(group, prediction[mask], windows,
                                                  seed=seed, power_w=power_w, arm=arm)
                _check_power_partition(power_w, old, new, group["time_s"].n_unique())
                archive = {row["radial_window"]: row for row in reference[arm]
                           if row["seed"] == seed and row["power_w"] == power_w}
                for row in old:
                    worst_legacy_delta = max(worst_legacy_delta,
                                             _compare_legacy_original(row, archive[row["radial_window"]]))
                all_old[arm].extend(old)
                all_new[arm].extend(new)
    if (any(len(all_old[arm]) != 45 or len(all_new[arm]) != 45 for arm in all_old)
            or len(lineage) != 5):
        raise ValueError("十个B0/任07最佳模型45+45同规约原修重评未完成")
    return _write_report(
        destination, source_manifest=source_manifest, source_manifest_sha256=source_manifest_sha256,
        snapshot=snapshot, originals=all_old, corrected=all_new, lineage=lineage,
        worst_legacy_delta=worst_legacy_delta, formal=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="合法HF验证Top五种子B0与任07原/修25毫米径向端点独立CUDA审计")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register-source", help="仅登记不可覆盖的事前源码快照项目目录，不运行GPU")
    mode.add_argument("--audit-output", help="严格真CUDA完成十模型审计的全新项目输出目录")
    for seed in range(5):
        parser.add_argument(f"--seed{seed}")
    parser.add_argument("--source-manifest")
    parser.add_argument("--source-manifest-sha256")
    args = parser.parse_args()
    if args.register_source:
        result = register_source_snapshot(args.register_source)
    else:
        if not args.source_manifest or not args.source_manifest_sha256:
            parser.error("正式审计必须提供事前源码登记文件及逐字节SHA")
        result = audit_radial_endpoint(
            {seed: getattr(args, f"seed{seed}") for seed in range(5)}, args.audit_output,
            source_manifest=args.source_manifest,
            source_manifest_sha256=args.source_manifest_sha256,
        )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
