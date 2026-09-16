"""Independent NumPy audit of the frozen real validation observations.

No observer imports, Torch imports, model deserialization, training or test labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import yaml


ROOT = Path("/home/phl/lyf/Temperature Field Prediction")
BASE = ROOT / "研究记录/任务11_外部对照"
HERE = BASE / "新MLP真实观察独立复核_20260916"
OBS = BASE / "正式新MLP三模态合法观察_20260916T183039+0800"
OLD = BASE / "正式新MLP五种子HF与双原能源汇总_20260916T165422+0800/机器汇总.json"
ARCHIVE = BASE / "任11_新MLP真实观察执行期ROOT0109单原件字节封存_20260916T184921+0800.tar.gz"
REG = BASE / "新MLP合法三模态观察导出前登记.yaml"
TAR = BASE / "新MLP合法三模态观察源码事前冻结.tar.gz"
REPORT = BASE / "任11_新MLP真实三模态观察中文误差与验收_20260916T190502+0800.md"
ROOT_PROOF = BASE / "任11_新MLP真实三模态观察根四轮数值与447原件验收_20260916T190502+0800.json"
ROOT_HASH = "87470d66f2a9a053c73931f2625d7b8776f0d7213bcee08dc5caeed24a639f0b"
ARCHIVE_HASH = "b84b20ce4d537f792f4266dd0b04cdf8398330df3ec51dfd5cf7de327f4a8d3a"
OLD_HASH = "ee19571311abf84a3682d4548d96a0fa13c98917940e1a64298d6eb65b561068"
POWERS = (115.2, 403.0, 630.5)
STATES = ("best", "final")
MODALITIES = ("Top", "Hot", "Cold")
TIME = (("time_0_30_s", 0.0, 30.0, True),
        ("time_30_100_s", 30.0, 100.0, False),
        ("time_100_200_s", 100.0, 200.0, False))
RADIAL = (("center_0_8_mm", 0.0, 8.0, True),
          ("middle_8_17_mm", 8.0, 17.0, False),
          ("outer_17_25_mm", 17.0, 25.0, False))
METRICS = {
    "rmse_c": "绝对RMSE_摄氏度", "mae_c": "绝对MAE_摄氏度",
    "signed_bias_c": "有符号偏差_摄氏度", "p95_abs_error_c": "绝对误差经验p95_摄氏度",
    "max_abs_error_c": "最大绝对误差_摄氏度", "delta_rmse_c": "首实测差分RMSE_摄氏度",
    "delta_mae_c": "首实测差分MAE_摄氏度", "delta_signed_bias_c": "首实测差分有符号偏差_摄氏度",
    "delta_p95_abs_error_c": "首实测差分绝对误差经验p95_摄氏度",
    "delta_max_abs_error_c": "首实测差分最大绝对误差_摄氏度", "peak_error_c": "有符号峰值误差_摄氏度",
}
FIELDS = {
    "seed": "种子", "state": "模型状态原标识", "split": "划分原标识",
    "power_w": "功率_瓦", "modality": "模态原标识", "sample_count": "实测点数",
    "time_count": "实际时刻数", "window": "时间窗原标识", "radial_window": "径向窗原标识",
    "lower_s": "下界_秒", "upper_s": "上界_秒", "lower_mm": "下界_毫米",
    "upper_mm": "上界_毫米", "lower_closed": "下界闭端点", "upper_closed": "上界闭端点",
    "empty_reason": "空窗原因", **METRICS,
}
POINT = {
    "seed": "种子", "state": "模型状态原标识", "split": "划分原标识",
    "power_w": "功率_瓦", "modality": "模态原标识", "r_m": "半径_米",
    "z_m": "轴向坐标_米", "time_s": "实测时刻_秒", "material_id": "材料标识",
    "target_k": "实测温度_K", "prediction_k": "预测温度_K", "weight": "原统计权重",
    "reference_time_s": "本人曲线首实测时刻_秒", "reference_target_k": "本人首实测温度_K",
    "reference_prediction_k": "本人首实测预测温度_K", "target_delta_k": "首实测温升_K",
    "prediction_delta_k": "首实测预测温升_K", "error_c": "有符号温度误差_摄氏度",
    "delta_error_c": "有符号首实测差分误差_摄氏度", "source_row": "同序模态源行号",
}
VALUES = {"Top": "顶部", "Hot": "热环", "Cold": "冷环", "best": "观测最佳", "final": "训练末",
          "time_0_30_s": "[0,30]秒", "time_30_100_s": "(30,100]秒", "time_100_200_s": "(100,200]秒",
          "center_0_8_mm": "中心[0,8]毫米", "middle_8_17_mm": "中圈(8,17]毫米", "outer_17_25_mm": "外圈(17,25]毫米"}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def ordinary(path):
    path = Path(path)
    check(path.is_absolute() and path.is_relative_to(ROOT) and path != ROOT,
          f"来源或输出越出项目: {path}")
    for part in (path, *path.parents):
        if part == ROOT:
            break
        check(not part.is_symlink(), f"非普通来源路径: {path}")
    check(path.is_file(), f"来源文件缺席: {path}")
    return path


def digest(path):
    h = hashlib.sha256()
    with ordinary(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        check(key not in result, f"JSON重复字段: {key}")
        result[key] = value
    return result


def finite(value):
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)
    elif isinstance(value, float):
        check(math.isfinite(value), "JSON包含非有限值")


def read_json(path):
    value = json.loads(ordinary(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    finite(value)
    return value


def same(actual, expected, label, atol=1e-10):
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and set(actual) == set(expected), f"字段集合不一致: {label}")
        for key in expected:
            same(actual[key], expected[key], f"{label}.{key}", atol)
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), f"列表长度不一致: {label}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            same(left, right, f"{label}[{index}]", atol)
    elif isinstance(expected, float):
        check(type(actual) in (int, float) and math.isfinite(actual)
              and math.isclose(actual, expected, abs_tol=atol, rel_tol=0),
              f"数值不一致: {label}: {actual!r} != {expected!r}")
    else:
        check(type(actual) is type(expected) and actual == expected,
              f"值或类型不一致: {label}: {actual!r} != {expected!r}")


def localized(row, mapping):
    result = {mapping[key]: value for key, value in row.items()}
    for key in ("modality", "state", "window", "radial_window"):
        if key in row:
            result[mapping[key].replace("原标识", "中文名称")] = VALUES[row[key]]
    return result


def native_power(frame, power):
    dtype = frame["power_w"].to_numpy().dtype
    check(dtype in (np.dtype("float32"), np.dtype("float64")), "功率原dtype错误")
    return frame["power_w"].to_numpy() == np.asarray(power, dtype=dtype)


def load_originals():
    top = pl.read_parquet(ordinary(ROOT / "data/processed/experiment_ir_radial.parquet"))
    top = top.filter(pl.col("split") == "validation")
    raw = pl.read_parquet(ordinary(ROOT / "data/processed/sensor_ring_raw.parquet"))
    raw = raw.filter(pl.col("split") == "validation")
    md = yaml.safe_load(ordinary(ROOT / "configs/data_metadata.yaml").read_text(encoding="utf-8"))["sensors"]
    check(all(md[key] == "verified" for key in ("coordinate_unit_status", "value_unit_status", "time_unit_status"))
          and md["synchronized_start"] is True and md["time_column_interpretation"] == "elapsed_time_t_equals_index",
          "两环单位或同步合同未核实")
    scale_r = {"m": 1.0, "mm": 1e-3}[md["coordinate_unit"]]
    scale_t = {"s": 1.0, "ms": 1e-3}[md["time_unit"]]
    offset = {"degC": 273.15, "K": 0.0}[md["value_unit"]]
    sensors = {}
    for modality, sensor_type in (("Hot", "hot"), ("Cold", "cold")):
        frame = raw.filter(pl.col("sensor_type") == sensor_type)
        sensors[modality] = {
            "power": frame["power_w"].to_numpy(),
            "time": (frame["time_raw"].to_numpy().astype(np.float64) * scale_t).astype(np.float32),
            "radius": (frame["radius_raw"].to_numpy().astype(np.float64) * scale_r).astype(np.float32),
            "target": (frame["value_mean_raw"].to_numpy().astype(np.float64) + offset).astype(np.float32),
            "source_delta": frame["delta_value_raw"].to_numpy().astype(np.float32),
            "weight": np.ones(frame.height, dtype=np.float64),
        }
    check(top.height == 7272 and raw.height == 752, "合法validation点数不符")
    check(top["r_m"].dtype == pl.Float32, "Top原半径不是float32")
    data = {"Top": {"power": top["power_w"].to_numpy(), "time": top["time_s"].to_numpy(),
                    "radius": top["r_m"].to_numpy(), "target": top["temperature_mean_k"].to_numpy(),
                    "weight": top["frame_weight"].to_numpy()}, **sensors}
    for modality, arrays in data.items():
        check(len(arrays["time"]) == (7272 if modality == "Top" else 376), f"{modality}源点数")
        check(arrays["radius"].dtype == np.float32, f"{modality}原半径dtype")
        check(all(np.isfinite(array).all() for array in arrays.values()), f"{modality}源数据非有限")
        check(set(arrays["power"]) == set(np.asarray(POWERS, dtype=arrays["power"].dtype)), "混入不合法功率")
        check(np.all((arrays["time"] >= 0) & (arrays["time"] <= 200)), "时域超界")
        tuples = list(zip(arrays["power"].tolist(), arrays["time"].tolist(), arrays["radius"].tolist()))
        check(len(set(tuples)) == len(tuples) and tuples == sorted(tuples), "源点重复或规范同序被改动")
    for power, n, nt in ((115.2, 1919, 19), (403.0, 2424, 24), (630.5, 2929, 29)):
        selected = native_power(top, power)
        check(int(selected.sum()) == n and len(np.unique(data["Top"]["time"][selected])) == nt,
              "Top功率帧数与冻结来源不符")
        check(len(np.unique(data["Top"]["radius"][selected])) == 101, "Top半径不是101")
        check(all(count == 101 for count in Counter(data["Top"]["time"][selected]).values()), "Top每帧不完整")
    return data


def metric_values(target, prediction, target_delta, prediction_delta, weight, mask):
    if not mask.any():
        return {name: None for name in METRICS}
    selected_weight = weight[mask].astype(np.float64)
    selected_weight = selected_weight / np.sum(selected_weight)
    result = {}
    for prefix, error in (("", prediction[mask] - target[mask]),
                          ("delta_", prediction_delta[mask] - target_delta[mask])):
        result[prefix + "rmse_c"] = float(np.sqrt(np.sum(selected_weight * error ** 2)))
        result[prefix + "mae_c"] = float(np.sum(selected_weight * np.abs(error)))
        result[prefix + "signed_bias_c"] = float(np.sum(selected_weight * error))
        result[prefix + "p95_abs_error_c"] = float(np.quantile(np.abs(error), 0.95, method="linear"))
        result[prefix + "max_abs_error_c"] = float(np.max(np.abs(error)))
    result["peak_error_c"] = float(np.max(prediction[mask]) - np.max(target[mask]))
    return result


def seed_stats(values, reason=None):
    if all(value is None for value in values):
        return {"均值": None, "样本标准差": None, "最小值": None, "最大值": None,
                "逐种子": values, "样本数": 0, "ddof": 1, "原因": reason}
    check(len(values) == 5 and all(value is not None for value in values), "五seed缺席或NULL混合")
    array = np.asarray(values, dtype=np.float64)
    return {"样本数": 5, "ddof": 1, "均值": float(np.mean(array)), "样本标准差": float(np.std(array, ddof=1)),
            "最小值": float(np.min(array)), "最大值": float(np.max(array)), "逐种子": array.tolist()}


def scan_stats(value, counts):
    if isinstance(value, dict):
        if "逐种子" in value:
            counts["全部统计对象"] += 1
            if value["样本数"] == 0:
                counts["NULL统计对象"] += 1
                same(value, seed_stats(value["逐种子"], value["原因"]), "独立统计对象")
            else:
                counts["非空统计对象"] += 1
                same(value, seed_stats(value["逐种子"]), "独立统计对象")
        else:
            for item in value.values():
                scan_stats(item, counts)
    elif isinstance(value, list):
        for item in value:
            scan_stats(item, counts)


def group_statistics(results, kind):
    groups = defaultdict(list)
    for result in results:
        for row in result[kind]:
            parts = [str(row["power_w"]), row["modality"]]
            if kind in ("time", "radial"):
                parts.append(row["window" if kind == "time" else "radial_window"])
            groups["|".join(parts)].append(row)
    summary = {}
    for key, rows in groups.items():
        check([row["seed"] for row in rows] == list(range(5)), "各组seed必须0..4有序")
        record = {"分组中文名称": "|".join(VALUES.get(part, part) for part in key.split("|")),
                  "逐seed实测点数": [row["sample_count"] for row in rows]}
        for raw, zh in METRICS.items():
            record[zh] = seed_stats([row[raw] for row in rows], rows[0]["empty_reason"])
        summary[key] = record
    return summary


def csv_rows(path, expected):
    with ordinary(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        check(reader.fieldnames == list(expected[0]), f"CSV列顺序不一致: {path.name}")
        rows = list(reader)
    check(len(rows) == len(expected), f"CSV行数不一致: {path.name}")
    for index, (actual, row) in enumerate(zip(rows, expected)):
        check(set(actual) == set(row), "CSV字段缺席")
        for key, value in row.items():
            text = actual[key]
            if value is None:
                check(text == "", f"CSV缺失值不是空字段: {index}.{key}")
            elif isinstance(value, bool):
                check(text == str(value), f"CSV布尔值错误: {index}.{key}")
            elif isinstance(value, (int, float)):
                check(math.isclose(float(text), float(value), abs_tol=1e-10, rel_tol=0), f"CSV数值错误: {index}.{key}")
            else:
                check(text == value, f"CSV字符串错误: {index}.{key}")
    return len(rows), sum(len(row) for row in rows)


def save_new(path, value):
    check(path.parent == HERE and not path.exists(), f"不得覆写任何已有工件: {path}")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def audit_actual():
    started = datetime.now().astimezone().isoformat()
    machine = read_json(OBS / "机器汇总.json")
    original = read_json(OBS / "原始指标与中文映射.json")
    expected_hashes = dict(machine["读取原件执行前后SHA256"])
    check(len(expected_hashes) == 437, "不是全部437冻结原来源")
    check(set(path.name for path in OBS.iterdir()) == {
        "机器汇总.json", "真实逐点预测.jsonl", "原始指标与中文映射.json", "中文验收.md", "工件SHA256.json",
        "逐功率三模态.csv", "逐功率三模态时间窗.csv", "顶部逐功率原生径向窗.csv"}, "输出8工件不完整或有额外文件")
    manifest = read_json(OBS / "工件SHA256.json")
    check(len(manifest) == 7 and set(manifest) == {path.name for path in OBS.iterdir()} - {"工件SHA256.json"}, "工件manifest不精确")
    for name, value in manifest.items():
        expected_hashes[str(OBS / name)] = value
    expected_hashes[str(OBS / "工件SHA256.json")] = digest(OBS / "工件SHA256.json")
    expected_hashes[str(OLD)] = OLD_HASH
    expected_hashes[str(ARCHIVE)] = ARCHIVE_HASH
    check(len(expected_hashes) == 447, "447文件范围去重计数不符")
    before = {path: digest(path) for path in sorted(expected_hashes)}
    same(before, expected_hashes, "447文件执行前SHA", atol=0)
    source_digest = hashlib.sha256(json.dumps(before, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    print(f"执行前SHA: 447/447一致; 范围规范JSON_SHA256={source_digest}", flush=True)
    with tarfile.open(ordinary(ARCHIVE), "r:gz") as archive:
        members = archive.getmembers()
        check(len(members) == 1 and members[0].isfile() and members[0].name == "多保真DeepONet预测精度优化总计划与执行台账.md",
              "ROOT执行期归档不是普通唯一1成员")
        check(hashlib.sha256(archive.extractfile(members[0]).read()).hexdigest() == ROOT_HASH, "ROOTarchive内部SHA不符")
    registry = yaml.safe_load(ordinary(REG).read_text(encoding="utf-8"))
    same(machine["观察合同"], registry["观察合同"], "观察合同", atol=0)
    pins = registry["源码普通成员SHA256"]
    check(len(pins) == 72, "观察源码不是72成员")
    with tarfile.open(ordinary(TAR), "r:gz") as archive:
        members = archive.getmembers()
        check(len(members) == 72 and len({member.name for member in members}) == 72
              and all(member.isfile() for member in members) and {member.name for member in members} == set(pins),
              "观察冻结源码不是普通唯一完整72成员")
        for member in members:
            check(hashlib.sha256(archive.extractfile(member).read()).hexdigest() == pins[member.name], "观察tar内部源码SHA漂移")
            check(before[str(ROOT / member.name)] == pins[member.name], "观察现场源码SHA漂移")
    check(pins["src/sic_cu/eval/task11_mlp_observations.py"] == "019771b69bded17fb5e2baf3258706134f820b9f6ba35752b9824d0ad2249edc", "观察器冻结SHA不符")
    qualification_counts = []
    check([q["种子"] for q in machine["五seed本人完整来源资格"]] == list(range(5)), "五seed完整资格错序")
    for q in machine["五seed本人完整来源资格"]:
        for path, sha in q["本人原件SHA256"].items():
            source = BASE / f"正式新MLP公平训练/正式MLP_HF_seed{q['种子']}" / path
            check(before[str(source)] == sha, "本人完整资格原件未绑定437源")
        lf_path = str(BASE / f"正式新MLP公平训练/正式MLP_LF_seed{q['种子']}/best.pt")
        check(before[lf_path] == q["本人LF起点SHA256"], "本人LF起点不符")
        qualification_counts.append(len(q["本人原件SHA256"]))
    check(qualification_counts == [39, 55, 39, 47, 55], "本人完整来源资格39/55/39/47/55不符")
    check([(row["种子"], row["模型状态"]) for row in registry["模型十状态"]]
          == [(seed, state) for seed in range(5) for state in STATES], "十模型登记错序")
    for row in registry["模型十状态"]:
        check(before[str(ROOT / row["模型原件"])] == row["模型SHA256"], "十模型最佳或末态身份不符")
    data = load_originals()
    points = defaultdict(list)
    point_count = 0
    expected_point_fields = set(POINT.values()) | {"模态中文名称", "模型状态中文名称"}
    with ordinary(OBS / "真实逐点预测.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            p = json.loads(line, object_pairs_hook=unique)
            finite(p)
            check(set(p) == expected_point_fields, "逐点中文字段集合不完整")
            raw = {key: p[zh] for key, zh in POINT.items()}
            check(type(raw["seed"]) is int and raw["seed"] in range(5) and raw["state"] in STATES
                  and raw["modality"] in MODALITIES and raw["power_w"] in POWERS and raw["split"] == "validation", "逐点身份或划分不合法")
            same(p, localized(raw, POINT), "逐点全部中文映射", atol=0)
            points[(raw["seed"], raw["state"], raw["modality"])].append(raw)
            point_count += 1
    check(point_count == 80240 and len(points) == 30, "真实80240点或30身份组不完整")
    results = []
    maximum_point_gap = 0.0
    point_numeric_fields = 0
    endpoint_count = 0
    temporal_boundaries = Counter()
    for seed in range(5):
        for state in STATES:
            result = {"seed": seed, "state": state, "power": [], "time": [], "radial": []}
            for modality in MODALITIES:
                rows = points[(seed, state, modality)]
                arrays = data[modality]
                check([row["source_row"] for row in rows] == list(range(len(arrays["time"]))), "同序模态source-row不是全双射")
                check(all(type(row["source_row"]) is int for row in rows), "source-row必须整数")
                prediction_all = np.asarray([row["prediction_k"] for row in rows], dtype=np.float64)
                check(np.array_equal(prediction_all, prediction_all.astype(np.float32).astype(np.float64)), "预测不是原float32表示")
                for power in POWERS:
                    mask = arrays["power"] == np.asarray(power, dtype=arrays["power"].dtype)
                    indices = np.flatnonzero(mask)
                    target = arrays["target"][mask].astype(np.float64)
                    prediction = prediction_all[mask]
                    times = arrays["time"][mask].astype(np.float64)
                    radius = arrays["radius"][mask]
                    weight = arrays["weight"][mask].astype(np.float64)
                    first = np.empty(len(indices), dtype=np.int64)
                    curve = radius if modality == "Top" else np.zeros(len(indices), dtype=np.int8)
                    for value in np.unique(curve):
                        locations = np.flatnonzero(curve == value)
                        first[locations] = locations[np.argmin(times[locations])]
                    target_delta = target - target[first]
                    prediction_delta = prediction - prediction[first]
                    if modality != "Top":
                        check(np.max(np.abs(target_delta - arrays["source_delta"][mask].astype(np.float64))) <= 1e-4,
                              "原两环差分不对应本人首实测")
                    for local_index, source_index in enumerate(indices):
                        expected = {"seed": seed, "state": state, "split": "validation", "power_w": power,
                                    "modality": modality, "r_m": float(radius[local_index]), "z_m": 0.0 if modality == "Top" else -0.0175,
                                    "time_s": float(times[local_index]), "material_id": 1 if modality == "Top" else 0,
                                    "target_k": float(target[local_index]), "prediction_k": float(prediction[local_index]),
                                    "weight": float(weight[local_index]), "reference_time_s": float(times[first[local_index]]),
                                    "reference_target_k": float(target[first[local_index]]), "reference_prediction_k": float(prediction[first[local_index]]),
                                    "target_delta_k": float(target_delta[local_index]), "prediction_delta_k": float(prediction_delta[local_index]),
                                    "error_c": float(prediction[local_index] - target[local_index]),
                                    "delta_error_c": float(prediction_delta[local_index] - target_delta[local_index]), "source_row": int(source_index)}
                        actual = rows[source_index]
                        same(actual, expected, "R1实际源逐点全部21字段", atol=0)
                        for key, value in expected.items():
                            if isinstance(value, float):
                                point_numeric_fields += 1
                                maximum_point_gap = max(maximum_point_gap, abs(actual[key] - value))
                    identity = {"seed": seed, "state": state, "split": "validation", "power_w": power, "modality": modality}
                    full = np.ones(len(indices), dtype=bool)
                    result["power"].append({**identity, "sample_count": len(indices), "time_count": len(np.unique(times)),
                                            **metric_values(target, prediction, target_delta, prediction_delta, weight, full), "empty_reason": None})
                    time_masks = []
                    for name, lower, upper, lower_closed in TIME:
                        selected = ((times >= lower) if lower_closed else (times > lower)) & (times <= upper)
                        time_masks.append(selected)
                        result["time"].append({**identity, "window": name, "lower_s": lower, "upper_s": upper,
                                               "lower_closed": lower_closed, "upper_closed": True, "sample_count": int(selected.sum()),
                                               **metric_values(target, prediction, target_delta, prediction_delta, weight, selected),
                                               "empty_reason": None if selected.any() else "预登记时间窗内无观测，不外推真值"})
                    check(np.all(np.sum(time_masks, axis=0) == 1), "时间三窗重叠或漏点")
                    for boundary in (0.0, 30.0, 100.0, 200.0):
                        temporal_boundaries[f"{modality}:{boundary}"] += int((times == boundary).sum())
                    if modality == "Top":
                        radial_masks = []
                        old_masks = []
                        for name, lower, upper, lower_closed in RADIAL:
                            native_lower, native_upper = np.float32(lower / 1000.0), np.float32(upper / 1000.0)
                            selected = ((radius >= native_lower) if lower_closed else (radius > native_lower)) & (radius <= native_upper)
                            radial_masks.append(selected)
                            old_mm = radius.astype(np.float64) * 1000.0
                            old_masks.append(((old_mm >= lower) if lower_closed else (old_mm > lower)) & (old_mm <= upper))
                            result["radial"].append({**identity, "radial_window": name, "lower_mm": lower, "upper_mm": upper,
                                                     "lower_closed": lower_closed, "upper_closed": True, "sample_count": int(selected.sum()),
                                                     **metric_values(target, prediction, target_delta, prediction_delta, weight, selected),
                                                     "empty_reason": None if selected.any() else "预登记径向窗内无顶部观测，不外推真值"})
                        check(np.all(np.sum(radial_masks, axis=0) == 1), "原生径向三窗漏点或重叠")
                        endpoints = radius == np.float32(0.025)
                        check(all(np.array_equal(radial_masks[index], old_masks[index]) for index in (0, 1))
                              and np.array_equal(radial_masks[2] ^ old_masks[2], endpoints), "径向唯一差异不只是25毫米闭端点")
                        check(np.all(radial_masks[2][endpoints]), "25毫米闭端点漏点")
                        endpoint_count += int(endpoints.sum())
            top_rows = [row for row in result["power"] if row["modality"] == "Top"]
            sensor_rows = [row for row in result["power"] if row["modality"] != "Top"]
            modal = {"顶部": float(np.mean([row["rmse_c"] for row in top_rows])),
                     "absolute_rmse_c": float(np.mean([row["rmse_c"] for row in sensor_rows])),
                     "absolute_mae_c": float(np.mean([row["mae_c"] for row in sensor_rows])),
                     "delta_rmse_c": float(np.mean([row["delta_rmse_c"] for row in sensor_rows])),
                     "delta_mae_c": float(np.mean([row["delta_mae_c"] for row in sensor_rows]))}
            result["macro"] = {"合法HF原macro_v1_摄氏度": (modal["顶部"] + 0.2 * modal["absolute_rmse_c"] + modal["delta_rmse_c"]) / 2.2, **modal}
            results.append(result)
    print(f"R1 PASS: 实际源80240点; Top7272/Hot376/Cold376 x10全source-row双射; 数值字段{point_numeric_fields}; 最大差={maximum_point_gap}", flush=True)
    check([(row["种子"], row["模型状态"]) for row in original["十状态原指标"]]
          == [(seed, state) for seed in range(5) for state in STATES], "十状态原指标seed/state错序")
    rows_checked = 0
    field_count = 0
    empty_rows = 0
    max_metric_gap = 0.0
    for actual, result in zip(original["十状态原指标"], results):
        for raw_kind, chinese_kind in (("power", "逐功率"), ("time", "时间窗"), ("radial", "原生径向窗")):
            expected = [localized(row, FIELDS) for row in result[raw_kind]]
            same(actual[chinese_kind], expected, f"R2全部{chinese_kind}记录")
            rows_checked += len(expected)
            field_count += len(expected) * len(METRICS)
            empty_rows += sum(row["实测点数"] == 0 for row in expected)
            for left, right in zip(actual[chinese_kind], expected):
                for zh in METRICS.values():
                    if right[zh] is not None:
                        max_metric_gap = max(max_metric_gap, abs(left[zh] - right[zh]))
        same(actual["原macro核验"], result["macro"], "十状态原macro独立重算")
    check(rows_checked == 450 and field_count == 4950 and empty_rows == 30 and endpoint_count == 720,
          "R2全量450x11/30空窗/720闭端点不符")
    print(f"R2 PASS: 450指标记录 x11={field_count}字段; 30空窗全部11字段NULL; 720个25mm闭端点; 最大指标差={max_metric_gap:.12g}", flush=True)
    stats = {}
    for state in STATES:
        selected = [result for result in results if result["state"] == state]
        stats[state] = {"模型状态中文名称": VALUES[state],
                        "逐功率": group_statistics(selected, "power"), "时间窗": group_statistics(selected, "time"),
                        "原生径向窗": group_statistics(selected, "radial"),
                        "本人原macro_v1_摄氏度": seed_stats([row["macro"]["合法HF原macro_v1_摄氏度"] for row in selected]),
                        "分模态宏平均": {modality: {
                            zh: seed_stats([float(np.mean([row[raw] for row in result["power"] if row["modality"] == modality])) for result in selected])
                            for raw, zh in METRICS.items()} for modality in MODALITIES}}
    same(machine["模型双状态五种子统计"], stats, "R3全部双状态五seed统计")
    stat_count = Counter()
    scan_stats(machine["模型双状态五种子统计"], stat_count)
    check(stat_count["全部统计对象"] == 1058 and stat_count["NULL统计对象"] == 66, "不是全1058统计对象或66NULL对象")
    old = read_json(OLD)
    check([row["运行种子"] for row in old["逐种子原统计"]] == list(range(5)), "冻结旧聚合五seed错序")
    old_comparison = []
    max_old_gap = 0.0
    old_fields = {"顶部": "顶部RMSE_摄氏度", "absolute_rmse_c": "两环合并绝对RMSE_摄氏度",
                  "absolute_mae_c": "两环合并绝对MAE_摄氏度", "delta_rmse_c": "两环合并首时刻差分RMSE_摄氏度",
                  "delta_mae_c": "两环合并首时刻差分MAE_摄氏度"}
    for result in results:
        original_state = old["逐种子原统计"][result["seed"]][VALUES[result["state"]]]
        gaps = {"macro": abs(result["macro"]["合法HF原macro_v1_摄氏度"] - original_state["HF合法macro_v1_摄氏度"])}
        for raw, zh in old_fields.items():
            gaps[raw] = abs(result["macro"][raw] - original_state["HF合法分模态"][zh])
        check(max(gaps.values()) <= 1e-4, "旧冻结10macro/分模态不符1e-4容差")
        max_old_gap = max(max_old_gap, max(gaps.values()))
        old_comparison.append({"种子": result["seed"], "模型状态": result["state"], "逐字段绝对差_摄氏度": gaps})
    print(f"R3 PASS: 全1058五seed统计对象(992非空/66NULL), 原五值有序0..4; mean/std(ddof1)/min/max; 旧10macro及50分模态最大差={max_old_gap:.12g} C <=1e-4", flush=True)
    same(machine["中文列映射"], {"逐点": POINT, "记录": FIELDS}, "机器汇总所有中文字段映射", atol=0)
    same(original["中文列映射"], FIELDS, "原始指标所有中文字段映射", atol=0)
    csv_count = {}
    csv_field_total = 0
    for name, raw_kind in (("逐功率三模态.csv", "power"), ("逐功率三模态时间窗.csv", "time"), ("顶部逐功率原生径向窗.csv", "radial")):
        expected = [localized(row, FIELDS) for result in results for row in result[raw_kind]]
        count, fields = csv_rows(OBS / name, expected)
        csv_count[name] = count
        csv_field_total += fields
    check(sum(csv_count.values()) == 450, "CSV没有覆盖450记录")
    check(machine["状态"] == "新MLP五seed双状态合法实测三模态只读观察" and machine["总点数"] == 80240
          and machine["验收计数"] == {"逐功率": 90, "时间窗": 270, "径向窗": 90}, "真实来源或计数标签错误")
    check(all(value is False for value in machine["限制"].values()), "真实模式限制不是全部False")
    acceptance = ordinary(OBS / "中文验收.md").read_text(encoding="utf-8")
    for state in STATES:
        st = stats[state]["本人原macro_v1_摄氏度"]
        lines = [line for line in acceptance.splitlines() if line.startswith(VALUES[state] + "原macro_v1：")]
        check(len(lines) == 1, "中文验收本人macro行缺席或重复")
        mean_part, values_part = lines[0].split("；原五值=")
        raw_values, range_part = values_part.split("；范围=")
        check(mean_part == f"{VALUES[state]}原macro_v1：{st['均值']:.9f}±{st['样本标准差']:.9f}摄氏度",
              "中文验收macro均值或样本标准差不符")
        same(json.loads(raw_values), st["逐种子"], "中文验收五seed原五值")
        check(range_part == f"[{st['最小值']:.9f},{st['最大值']:.9f}]。", "中文验收macro范围不符")
    after = {path: digest(path) for path in sorted(expected_hashes)}
    same(after, before, "R4全部447文件执行后SHA", atol=0)
    after_digest = hashlib.sha256(json.dumps(after, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    compact_digest = hashlib.sha256(json.dumps(before, ensure_ascii=False, sort_keys=True,
                                              separators=(",", ":")).encode("utf-8")).hexdigest()
    print(f"R4 PASS: CSV450行/{csv_field_total}单元格与JSON全部字段/中文映射一致; 冻结447文件before/after无漂移; ROOTarchive唯一普通1成员={ROOT_HASH}", flush=True)
    print(f"执行后SHA: 447/447一致; 范围规范JSON_SHA256={after_digest}", flush=True)
    evidence = {
        "状态": "真实合法validation全量独立NumPy四轮PASS，非合成或观察器自证", "开始": started,
        "结束": datetime.now().astimezone().isoformat(), "Python": sys.executable, "NumPy": np.__version__,
        "禁用行为": {"观察器metric helper调用": False, "Torch导入": False, "真实PT反序列化": False,
                     "训练": False, "GPU使用": False, "旧固定TEST温度读取": False, "模拟LF_TEST温度读取": False},
        "R1": {"点数": point_count, "身份组": len(points), "每状态各模态点数": {"Top": 7272, "Hot": 376, "Cold": 376},
               "全源行双射": True, "完整逐点字段数": len(POINT), "比较浮点字段总数": point_numeric_fields, "逐点最大差": maximum_point_gap,
               "本人完整资格来源文件计数": qualification_counts},
        "R2": {"指标记录": rows_checked, "每记录原指标": len(METRICS), "原指标字段总数": field_count,
               "空窗记录": empty_rows, "空窗NULL字段": empty_rows * len(METRICS), "径向25毫米闭端点": endpoint_count,
               "最大原指标差": max_metric_gap, "时间边界点数_十状态": dict(temporal_boundaries),
               "口径": "每范围权重归一；经验p95未加权np.quantile(.95,linear)；bias=pred-target；peak=max(pred)-max(target)；三时间窗无漏点无重叠；原生float32三径向窗"},
        "R3": {**dict(stat_count), "seed顺序": list(range(5)), "样本标准差ddof": 1,
               "旧摘要最大60字段绝对差_摄氏度": max_old_gap, "容差_摄氏度": 1e-4, "旧摘要逐状态比较": old_comparison},
        "R4": {"CSV逐文件行数": csv_count, "CSV单元格数": csv_field_total, "全部绑定文件数": len(before),
               "冻结原来源": 437, "输出工件": 8, "旧摘要": 1, "ROOTarchive": 1, "源码tar普通unique成员": 72,
               "ROOTarchive普通unique成员": 1, "ROOTarchive成员SHA256": ROOT_HASH, "before规范JSON_SHA256": source_digest,
               "after规范JSON_SHA256": after_digest, "紧凑规范JSON_SHA256": compact_digest,
               "全部文件执行前后SHA256": {
                   path: {"预期": expected_hashes[path], "before": before[path], "after": after[path]} for path in sorted(before)}},
        "独立重算五seed统计": stats,
    }
    save_new(HERE / "真实全量四轮机器证据.json", evidence)
    print("终态 PASS; 独立真实全量四轮机器证据.json已新增; exit=0", flush=True)


def audit_report(final=False):
    evidence = read_json(HERE / "真实全量四轮机器证据.json")
    before = digest(REPORT)
    text = ordinary(REPORT).read_text(encoding="utf-8")
    print(f"中文报告执行前SHA256={before}; 字符={len(text)}; 行={len(text.splitlines())}", flush=True)
    sections = []
    current = []
    heading = ""
    for number, line in enumerate(text.splitlines(), 1):
        if line.startswith("#"):
            heading = line
        if line.startswith("|"):
            current.append((number, heading, [cell.strip() for cell in line.strip().strip("|").split("|")]))
        elif current:
            sections.append(current)
            current = []
    if current:
        sections.append(current)
    check(len(sections) == 7, "报告不是完整7张表")
    by_heading = {rows[0][1]: rows for rows in sections}
    stats = evidence["独立重算五seed统计"]
    table_evidence = []

    def fmt(stat):
        return "缺失（n=0）" if stat["样本数"] == 0 else f"{stat['均值']:.6f} ± {stat['样本标准差']:.6f}"

    def compare_table(heading, header, expected):
        actual = by_heading[heading]
        check(actual[0][2] == header, f"表头不符: {heading}")
        check(len(actual) == len(expected) + 2, f"表格数据行数不符: {heading}")
        for (line_number, _, row), desired in zip(actual[2:], expected):
            same(row, desired, f"报告第{line_number}行全部表值", atol=0)
        table_evidence.append({"标题": heading, "表头": header, "数据行数": len(expected),
                               "逐行原值与独立重算全部一致": True, "行号": [row[0] for row in actual[2:]]})

    abs_keys = ("rmse_c", "mae_c", "signed_bias_c", "p95_abs_error_c", "max_abs_error_c", "peak_error_c")
    delta_keys = ("delta_rmse_c", "delta_mae_c", "delta_signed_bias_c", "delta_p95_abs_error_c", "delta_max_abs_error_c")
    modality_labels = {"Top": "顶部", "Hot": "Hot环", "Cold": "Cold环"}
    expected_abs, expected_delta = [], []
    for state in STATES:
        for modality in MODALITIES:
            record = stats[state]["分模态宏平均"][modality]
            identity = [VALUES[state], modality_labels[modality]]
            expected_abs.append(identity + [fmt(record[METRICS[key]]) for key in abs_keys])
            expected_delta.append(identity + [fmt(record[METRICS[key]]) for key in delta_keys])
    compare_table("## 四、三模态绝对误差宏平均",
                  ["状态", "模态", "RMSE", "MAE", "有符号偏差", "绝对误差p95", "最大绝对误差", "有符号峰值误差"], expected_abs)
    compare_table("## 五、首实测差分误差宏平均",
                  ["状态", "模态", "RMSE", "MAE", "有符号偏差", "绝对误差p95", "最大绝对误差"], expected_delta)
    macro_table = by_heading["## 六、冻结原macro的只读数值回归"]
    check(macro_table[0][2] == ["状态", "原点重算macro：均值 ± 样本标准差", "五种子原点重算值（0至4）"]
          and len(macro_table) == 4, "macro表头或2行不符")
    for state, (line_number, _, row) in zip(STATES, macro_table[2:]):
        st = stats[state]["本人原macro_v1_摄氏度"]
        check(row[:2] == [VALUES[state], fmt(st)], f"macro第{line_number}行均值/STD不符")
        same([float(value.strip()) for value in row[2].split(",")], st["逐种子"],
             f"macro第{line_number}行全部原五值", atol=5e-14)
    table_evidence.append({"标题": "## 六、冻结原macro的只读数值回归", "数据行数": 2,
                           "逐行原值与独立重算全部一致": True, "行号": [row[0] for row in macro_table[2:]]})
    group_keys = ("rmse_c", "delta_rmse_c", "p95_abs_error_c", "peak_error_c")
    for heading, kind in (("## 七、逐功率三模态", "逐功率"), ("## 八、逐功率三模态时间窗", "时间窗"),
                          ("## 九、顶部原生径向窗", "原生径向窗")):
        expected = []
        for state in STATES:
            for record in stats[state][kind].values():
                counts = record["逐seed实测点数"]
                check(len(set(counts)) == 1, "报告固定每seed点数不一致")
                expected.append([VALUES[state], record["分组中文名称"].replace("|", " / "), str(counts[0])]
                                + [fmt(record[METRICS[key]]) for key in group_keys])
        compare_table(heading, ["状态", "原分组", "每seed真实点数", "绝对RMSE", "首实测差分RMSE", "绝对误差p95", "有符号峰值误差"], expected)
    sha_table = by_heading["## 十一、八输出SHA复现索引"]
    expected_sha = [[name, str((OBS / name).stat().st_size), digest(OBS / name)] for name in (
        "中文验收.md", "原始指标与中文映射.json", "机器汇总.json", "真实逐点预测.jsonl", "逐功率三模态.csv",
        "逐功率三模态时间窗.csv", "顶部逐功率原生径向窗.csv", "工件SHA256.json")]
    compare_table("## 十一、八输出SHA复现索引", ["原八工件", "字节", "SHA256"], expected_sha)
    print("中文报告7张表 PASS: 102误差行+2macro行+8SHA索引，全部值与独立重算一致", flush=True)
    snapshot = {path: record["before"] for path, record in evidence["R4"]["全部文件执行前后SHA256"].items()}
    ascii_digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    check(f"447件完整SHA目录规范化SHA256：{ascii_digest}" in text, "报告447规范化SHA不符默认ensure_ascii=True+compact协议")
    check(ascii_digest == "549e0ae1cd9064089647acc8e66d020c58cd23641574a835ebc27e2efa7900a0", "447范围ASCII规范摘要不符")
    check(f"ROOT0109整件SHA256：{ROOT_HASH}" in text and f"ROOT单原件tar SHA256：{ARCHIVE_HASH}" in text,
          "报告ROOT归档身份不符")
    old_before = digest(OLD)
    check(old_before == OLD_HASH, "旧HF10macro摘要漂移")
    old = read_json(OLD)
    old_best, old_final = old["观测最佳"]["HF合法macro_v1_摄氏度"], old["训练末"]["HF合法macro_v1_摄氏度"]
    check(f"best S={old_best['均值']} ± {old_best['样本标准差']}℃" in text
          and f"final S={old_final['均值']} ± {old_final['样本标准差']}℃" in text, "报告冻结旧HF均值或STD被改变")
    max_gap = evidence["R3"]["旧摘要最大60字段绝对差_摄氏度"]
    gap_tokens = re.findall(r"最大差为([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)℃", text)
    check(len(gap_tokens) == 1 and math.isclose(float(gap_tokens[0]), max_gap, abs_tol=1e-15, rel_tol=0),
          "报告旧10macro最大差不符")
    root_proof_before = digest(ROOT_PROOF)
    recorded = read_json(ROOT_PROOF)
    cli = recorded["正式CLI57实际证据"]
    receipt = cli["封闭回执"]
    check(cli["工具实际退出码"] == 0 and receipt["真实逐点数"] == 80240
          and receipt["CUDA初始化"] is True and receipt["可见CUDA卡数"] == 1,
          "报告所链CLI57回执状态不符")
    check(receipt["开始时间"] in text and receipt["结束时间"].split("T")[1] in text
          and str(receipt["外层总导出与核验墙钟秒"]) in text and "RTX 4090 D" in text,
          "报告真实CLI57记录的起止/墙钟/GPU名称不符")
    energy_records = []
    for row in old["十份独立能源原件"]:
        path = Path(row["能源原目录"]) / "汇总指标.json"
        energy_before = digest(path)
        check(energy_before == row["六工件SHA256"]["汇总指标.json"], "冻结原能源摘要SHA漂移")
        summary = read_json(path)
        screen = summary["名义吸收归一筛查"]
        check(screen["均值阈值"] == 0.05 and screen["95分位阈值"] == 0.1
              and summary["工程安全阈值已建立"] is False and summary["结果不用于训练选模"] is True,
              "能源名义/工程或选模边界不符")
        energy_records.append({"种子": row["运行种子"], "状态": row["状态原标识"], "摘要路径": str(path),
                               "名义均值通过": screen["均值通过"], "名义p95通过": screen["95分位通过"],
                               "beforeSHA256": energy_before, "afterSHA256": digest(path)})
        check(energy_records[-1]["beforeSHA256"] == energy_records[-1]["afterSHA256"], "能源摘要核验期漂移")
    check(len(energy_records) == 10 and sum(row["名义均值通过"] and row["名义p95通过"] for row in energy_records) == 0,
          "报告原名义能源通过0/10不符")
    check("仍通过0/10" in text and "不是新增独立测温" in text and "任12真实新增测温为0" in text
          and "不证明体内温度真值或工程安全" in text and "替换旧B0" in text
          and "任11整体仍未完成" in text and "总计划不标100%完成" in text, "报告受限边界或未完成说明缺席")
    failed_dimension = stats["best"]["分模态宏平均"]["Top"][METRICS["delta_rmse_c"]]
    check(f"首实测差分RMSE仍为{fmt(failed_dimension)}℃" in text and "该失败维度必须保留" in text,
          "报告Top差分失败维度被隐藏")
    check("ddof=1" in text and "未加权经验分位数" in text and "不是所有点混池" in text
          and "各功率/曲线首次真实采样" in text
          and "共30条原时间指标" in text and "6个组" in text and "全部720个25毫米闭端点实际保留" in text,
          "报告统计/空窗/闭端点限定不符")
    root_proof_after = digest(ROOT_PROOF)
    check(root_proof_after == root_proof_before and digest(OLD) == old_before, "报告相关冻结声明依据核验期间漂移")
    after = digest(REPORT)
    check(after == before, "报告核验期间SHA漂移")
    compact = {"状态": "中文报告完整表值与独立原源重算一致，工程和科学限制保留",
               "报告路径": str(REPORT), "beforeSHA256": before, "afterSHA256": after,
               "表格数": len(sections), "表数据行": sum(len(table) - 2 for table in sections),
               "误差主表数据行": 102, "macro数据行": 2, "八输出SHA索引行": 8, "逐表证据": table_evidence,
               "447默认ASCII紧凑规范SHA256": ascii_digest, "原能源名义通过": "0/10",
               "十原能源冻结摘要只读核对": energy_records, "重算能源": False,
               "所链CLI57日志依据SHA256": {"before": root_proof_before, "after": root_proof_after},
               "独立实际四轮依据": str(HERE / "真实全量四轮机器证据.json"),
               "GPU运行证明边界": "本独审没有GPU调用，CLI57时间/硬件只核对根所链实跑回执；独立证据是源点/指标/统计/字节全量重算",
               "不作新增测温/内部真值/工程安全/B0替换或任务整体完成主张": True}
    save_new(HERE / ("最终中文报告完整表值机器证据.json" if final else "中文报告完整表值机器证据.json"), compact)
    print(f"中文报告执行后SHA256={after}; 表格=7; 全部表数据行=112(误差102/macro2/SHA8); 全部数值/STD/原五值/6空组/720端点/0of10名义能源/受限边界PASS", flush=True)
    print("报告全量独立核对 PASS; exit=0", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("actual", "report", "report-final"), required=True)
    args = parser.parse_args()
    check(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU未显式禁用")
    check(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "字节码未禁用")
    env_root = BASE / "HowardLF正式执行环境_20260916T172037+0800"
    for env, directory in (("TMPDIR", "临时文件"), ("TMP", "临时文件"), ("TEMP", "临时文件"),
                           ("CUDA_CACHE_PATH", "GPU缓存"), ("XDG_CACHE_HOME", "应用缓存"), ("MPLCONFIGDIR", "绘图缓存"),
                           ("TORCH_EXTENSIONS_DIR", "Torch扩展缓存"), ("TORCHINDUCTOR_CACHE_DIR", "Inductor缓存")):
        check(os.environ.get(env) == str(env_root / directory), f"{env}缓存不是项目内指定路径")
    check(os.environ.get("OMP_NUM_THREADS") == "1" and os.environ.get("MKL_NUM_THREADS") == "1", "CPU线程数未固定")
    audit_actual() if args.phase == "actual" else audit_report(final=args.phase == "report-final")


if __name__ == "__main__":
    main()
