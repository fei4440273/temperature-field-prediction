"""Howard 本人五种子双状态合法三模态只读观察；独立六来源门禁。"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import re
import statistics
import tarfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
import torch
from torch import Tensor

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.processed import load_processed_ir_observations
from sic_cu.data.sensors import load_canonical_sensor_observations
from sic_cu.eval import task11_mlp_observations as numerical
from sic_cu.eval.development_v4 import _delta_from_first
from sic_cu.models.task11_howard_composite import Task11HowardComposite
from sic_cu.train import task11_howard_hf_formal as hf


ROOT_TOKEN = "TASK11_HOWARD_OBSERVATIONS_GATE:v1"
ROOT_IDENTITY_FIELDS = ("OBS_YAML_SHA256", "OBS_TAR_SHA256", *hf.IDENTITY_FIELDS)
OUTPUT_BASE = "研究记录/任务11_外部对照"
OBS_REGISTRY_PATH = OUTPUT_BASE + "/Howard本人合法三模态观察导出前登记.yaml"
OBS_TAR_PATH = OUTPUT_BASE + "/Howard本人合法三模态观察源码事前冻结.tar.gz"
HF_QUALIFIED_STATUS = "CPU Howard三网HF本人真实终态与当前来源审核PASS"
FIXED_POWERS = numerical.FIXED_POWERS
MODALITIES = numerical.MODALITIES
STATES = numerical.STATES
TOP_COUNTS = numerical.TOP_COUNTS
TIME_WINDOWS = copy.deepcopy(numerical.TIME_WINDOWS)
RADIAL_WINDOWS = copy.deepcopy(numerical.RADIAL_WINDOWS)
METRIC_ZH = copy.deepcopy(numerical.METRIC_ZH)
FIELD_ZH = copy.deepcopy(numerical.FIELD_ZH)
POINT_ZH = copy.deepcopy(numerical.POINT_ZH)
VALUE_ZH = copy.deepcopy(numerical.VALUE_ZH)
SOURCE_MEMBERS = tuple(sorted(set(hf.SOURCE_MEMBERS) | set(numerical.SOURCE_MEMBERS) | {
    "src/sic_cu/eval/task11_howard_observations.py", "scripts/58_export_task11_howard_observations.py",
    "tests/test_task11_howard_observations.py",
}))
OBSERVATION_CONTRACT = copy.deepcopy(numerical.OBSERVATION_CONTRACT)
OBSERVATION_CONTRACT.update({
    "模型方法": "Howard最近文献数学三网适配；非原文精确联合训练复现",
    "参数张量数": 56, "查询缓冲区": "固定PH/QH各24乘4；保存与运行均Float64，不随权重.float()",
    "径向端点限制": "仅本人Top原生米制Float32；[0,8]、(8,17]、(17,25]毫米，25毫米闭区不丢点",
    "Top权重": "本人原帧逆方差可靠性frame_weight逐点原值保留；每帧既有归一，统计范围仅原权重和归一；功率等权，不重算几何面积",
    "真实CUDA前向": "全部十状态本人模型及输入输出均cuda:0；禁止CPU伪标签或CPU替代",
    "科学合格主张": False, "工程安全合格主张": False, "采用B0部署": False,
    "新增测量": False, "任务目标完成": False,
})


def source_files() -> tuple[str, ...]:
    return SOURCE_MEMBERS


def _path(value: str | Path, root: Path, *, file: bool = True) -> Path:
    if ".." in Path(value).parts:
        raise ValueError("Howard观察来源与输出禁止父路径跳转")
    return hf._path(value, root, file=file)


def _hash(path: Path, digest: str) -> str:
    if not hf._sha(digest) or sha256_file(path) != digest:
        raise ValueError(f"Howard观察来源/模型/源码SHA256缺失或漂移：{path.name}")
    return digest


def _identity(identity: Mapping[str, Any]) -> dict[str, str]:
    names = ("registry", "source_tar", "lf_catalog", "hf_data_catalog")
    if (not isinstance(identity, dict) or set(identity) != {k for n in names for k in (n, n + "_sha")}
            or any(type(identity[n]) is not str or identity[n] != expected for n, expected in zip(
                names, (hf.REGISTRY, hf.SOURCE_TAR, hf.LF_CATALOG, hf.HF_DATA_CATALOG)))
            or any(not hf._sha(identity[n + "_sha"]) for n in names)):
        raise ValueError("Howard观察只接受本人HF四独立固定来源，不借MLP或能源许可")
    return copy.deepcopy(identity)


def root_gate_hashes(arguments: Mapping[str, Any]) -> dict[str, str]:
    return dict(zip(ROOT_IDENTITY_FIELDS, (arguments["observation_registry_sha"],
        arguments["observation_source_tar_sha"], arguments["registry_sha"], arguments["source_tar_sha"],
        arguments["lf_catalog_sha"], arguments["hf_data_catalog_sha"])))


def _catalog_schema(rows: Any) -> None:
    expected = [(seed, state) for seed in range(5) for state in STATES]
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("Howard观察必须冻结本人五seed各best/final十个状态")
    for identity, row in zip(expected, rows):
        if (not isinstance(row, dict) or set(row) != {"种子", "模型状态", "模型原件", "模型SHA256"}
                or type(row["种子"]) is not int or (row["种子"], row["模型状态"]) != identity
                or type(row["模型原件"]) is not str or row["模型原件"] !=
                    f"{hf.RUN_DIRECTORY}/正式Howard_HF_seed{identity[0]}/{identity[1]}.pt"
                or not hf._sha(row["模型SHA256"])):
            raise ValueError("Howard观察十模型必须本人同seed规范best/final唯一有序身份")


def registration_contract(*, source_hashes: Mapping[str, str], source_tar_sha: str,
    hf_source_identity: Mapping[str, Any], model_catalog: Any, registered_at: str,
    root_before_sha: str) -> dict[str, Any]:
    """纯合同构造器；不写登记/tar/ROOT、不读取模型或资格、不授予运行许可。"""
    identity = _identity(hf_source_identity)
    _catalog_schema(model_catalog)
    if (not isinstance(source_hashes, dict) or set(source_hashes) != set(SOURCE_MEMBERS)
            or any(not hf._sha(v) for v in source_hashes.values()) or not hf._sha(source_tar_sha)
            or not hf._sha(root_before_sha) or type(registered_at) is not str or not registered_at.strip()):
        raise ValueError("Howard观察前登记必须包含实际完整普通源码及事前ROOT身份")
    return {"schema_version": 1, "登记时间": registered_at, "登记前ROOT_SHA256": root_before_sha,
        "阶段": "Howard本人合法验证三模态只读观察", "ROOT门禁标签": ROOT_TOKEN,
        "HowardHF四来源": identity, "模型十状态": copy.deepcopy(model_catalog),
        "观察合同": copy.deepcopy(OBSERVATION_CONTRACT), "源码冻结tar原件": OBS_TAR_PATH,
        "源码冻结tarSHA256": source_tar_sha, "源码普通成员SHA256": copy.deepcopy(source_hashes),
        "真实资格API": "sic_cu.train.task11_howard_hf_formal.qualify_task11_howard_hf_source",
        "HF训练许可": False, "重新训练或优化": False, "重新选择best": False,
        "旧固定TEST温度读取": False, "模拟测试功率温度读取": False, "科学或工程安全资格": False}


def _plain_root_rows(text: str) -> list[tuple[int, str]]:
    rows = []
    fence = None
    html_block, html_end = False, None
    html = hf._RootHTMLGuard()
    for line in text.splitlines():
        if html_block:
            html.feed(line + "\n")
            if html_end is None:
                if not line.strip() and not html.hidden:
                    html_block = False
            elif re.search(html_end, line, re.I):
                html_block = html.hidden
                if html_block:
                    html_end = html.continuation_end()
            continue
        marker = re.match(r" {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence is not None:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= fence[1] and not marker[2].strip():
                fence = None
            continue
        if marker and (marker[1][0] != "`" or "`" not in marker[2]):
            fence = (marker[1][0], len(marker[1]))
            continue
        raw = re.match(r" {0,3}<(script|pre|style|textarea)(?:\s|>|$)", line, re.I)
        if "<!--" in line or re.match(r" {0,3}<", line):
            html.feed(line + "\n")
            if raw:
                html_end = rf"</{raw[1]}\s*>"
            elif re.match(r" {0,3}<!--", line):
                html_end = r"-->"
            elif re.match(r" {0,3}<\?", line):
                html_end = r"\?>"
            elif re.match(r" {0,3}<!\[CDATA\[", line):
                html_end = r"\]\]>"
            elif re.match(r" {0,3}<![A-Z]", line):
                html_end = r">"
            elif re.match(r" {0,3}<", line):
                html_end = None
            else:
                html_end = r"-->"
            html_block = True
            if html_end is not None and re.search(html_end, line, re.I):
                html_block = html.hidden
                if html_block:
                    html_end = html.continuation_end()
            continue
        if re.fullmatch(r"\|.*\|[ \t]*", line) is None:
            continue
        fields = [cell.strip() for cell in line.split("|")]
        if len(fields) >= 4 and re.fullmatch(r"录-\d{4}", fields[1]):
            rows.append((int(fields[1][2:]), fields[2]))
    return rows


def root_active(ledger: Path, hashes: Mapping[str, str]) -> bool:
    if set(hashes) != set(ROOT_IDENTITY_FIELDS) or any(not hf._sha(v) for v in hashes.values()):
        return False
    expected = f"{ROOT_TOKEN}; status=active; " + "; ".join(f"{k}={hashes[k]}" for k in ROOT_IDENTITY_FIELDS)
    text = ledger.read_text(encoding="utf-8")
    if text.count(ROOT_TOKEN) != 1:
        return False
    try:
        records = _plain_root_rows(text)
    except (AssertionError, ValueError):
        return False
    own = [(n, cell) for n, cell in records if cell.startswith(ROOT_TOKEN)]
    return (len(own) == 1 and own[0][0] > 107 and own[0][1] == expected
            and sum(n == own[0][0] for n, _ in records) == 1)


def _new_output(value: str | Path, root: Path) -> Path:
    path = _path(value, root, file=False)
    if path.parent != root / OUTPUT_BASE or path.name in {
        "正式新MLP公平训练", "正式HowardLF公平训练", "正式Howard公平训练"}:
        raise ValueError("Howard观察只能新建外部对照直属目录，不进入正式训练树")
    if path.exists():
        raise FileExistsError("Howard观察已有输出不可覆盖或复用")
    return path


def _audit_tar(path: Path, hashes: Mapping[str, str], root: Path) -> None:
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            if (len(members) != len(SOURCE_MEMBERS) or len({m.name for m in members}) != len(members)
                    or {m.name for m in members} != set(SOURCE_MEMBERS) or any(not m.isfile() for m in members)):
                raise ValueError("Howard观察源码tar必须完整普通成员，无link/重复/额外/缺失")
            for member in members:
                stream = archive.extractfile(member)
                if stream is None or hashlib.sha256(stream.read()).hexdigest() != hashes[member.name]:
                    raise ValueError("Howard观察tar普通成员与登记SHA漂移")
                _hash(_path(member.name, root), hashes[member.name])
    except (OSError, tarfile.TarError) as error:
        raise ValueError("Howard观察普通源码tar无法核验") from error


def preflight_task11_howard_observations(*, observation_registry: str | Path, observation_registry_sha: str,
    observation_source_tar: str | Path, observation_source_tar_sha: str, registry: str | Path, registry_sha: str,
    source_tar: str | Path, source_tar_sha: str, lf_catalog: str | Path, lf_catalog_sha: str,
    hf_data_catalog: str | Path, hf_data_catalog_sha: str, output: str | Path,
    project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """独立静态CPU门禁；仅哈希模型字节，不反序列化PT、不打开温度、不探测CUDA。"""
    root = Path(project_root).resolve()
    if root != PROJECT_ROOT.resolve() and not root.is_relative_to(PROJECT_ROOT.resolve() / OUTPUT_BASE):
        raise ValueError("Howard观察替代根仅准本项目外部对照内隔离CPU合成夹具")
    values = {"registry": registry, "source_tar": source_tar, "lf_catalog": lf_catalog,
              "hf_data_catalog": hf_data_catalog}
    digests = {"registry_sha": registry_sha, "source_tar_sha": source_tar_sha,
               "lf_catalog_sha": lf_catalog_sha, "hf_data_catalog_sha": hf_data_catalog_sha}
    identity = _identity({**{k: _path(v, root).relative_to(root).as_posix() for k, v in values.items()}, **digests})
    hashes = root_gate_hashes({"observation_registry_sha": observation_registry_sha,
        "observation_source_tar_sha": observation_source_tar_sha, **identity})
    files = [_path(observation_registry, root), _path(observation_source_tar, root)] + [
        _path(identity[k], root) for k in values]
    if files[:2] != [root / OBS_REGISTRY_PATH, root / OBS_TAR_PATH]:
        raise ValueError("Howard观察必须独立固定新YAML/tar原件，不继承其他活动许可")
    pins = {str(p): _hash(p, digest) for p, digest in zip(files, hashes.values())}
    ledger = _path(hf.ROOT_LEDGER, root)
    ledger_sha = sha256_file(ledger)
    if not root_active(ledger, hashes):
        raise ValueError("Howard观察需唯一普通ROOT录号>107同单格六独立SHA活动门禁")
    recorded = hf._load_yaml(files[0])
    if not isinstance(recorded, dict):
        raise ValueError("Howard观察登记缺严格完整schema")
    expected = registration_contract(source_hashes=recorded.get("源码普通成员SHA256"),
        source_tar_sha=observation_source_tar_sha, hf_source_identity=identity,
        model_catalog=recorded.get("模型十状态"), registered_at=recorded.get("登记时间"),
        root_before_sha=recorded.get("登记前ROOT_SHA256"))
    if not hf._same(recorded, expected):
        raise ValueError("Howard观察冻结字段/schema/float/bool/计数/来源/观察合同不一致")
    _audit_tar(files[1], recorded["源码普通成员SHA256"], root)
    for name, digest in recorded["源码普通成员SHA256"].items():
        pins[str(_path(name, root))] = digest
    _hash(Path(__file__), recorded["源码普通成员SHA256"]["src/sic_cu/eval/task11_howard_observations.py"])
    models = {}
    for row in recorded["模型十状态"]:
        path = _path(row["模型原件"], root)
        pins[str(path)] = _hash(path, row["模型SHA256"])
        models[(row["种子"], row["模型状态"])] = {"模型原件": str(path), "模型SHA256": row["模型SHA256"]}
    if any(os.environ.get(k, default) != default for k, default in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0"))):
        raise ValueError("Howard观察只能单卡单进程，无分布式fallback")
    destination = _new_output(output, root)
    pins[str(ledger)] = _hash(ledger, ledger_sha)
    record_number = next(n for n, cell in _plain_root_rows(ledger.read_text(encoding="utf-8")) if cell.startswith(ROOT_TOKEN))
    result = {"状态": "Howard观察独立CPU静态前核验；未读PT/温度或运行CUDA", "观察许可": False,
        "项目根": str(root), "输出": str(destination), "登记": recorded, "HowardHF四来源": identity,
        "ROOT六SHA": hashes, "ROOT记录号": f"录-{record_number:04d}", "ROOT原件": str(ledger),
        "ROOT整件SHA256": ledger_sha, "ROOT唯一活动行": f"{ROOT_TOKEN}; status=active; " +
            "; ".join(f"{k}={hashes[k]}" for k in ROOT_IDENTITY_FIELDS),
        "源码普通成员数": len(SOURCE_MEMBERS), "模型十身份": models, "读取原件SHA256": pins}
    verify_task11_howard_observation_runtime(result, {})
    return result


def verify_task11_howard_observation_runtime(checked: Mapping[str, Any], additional_pins: Mapping[str, str]) -> None:
    root = Path(checked["项目根"])
    original = checked["读取原件SHA256"]
    if any(p in original and original[p] != d for p, d in additional_pins.items()):
        raise ValueError("Howard观察末SHA不能替代执行前原件固定SHA")
    for path, digest in {**original, **additional_pins}.items():
        _hash(_path(path, root), digest)
    ledger = _path(hf.ROOT_LEDGER, root)
    if not root_active(ledger, checked["ROOT六SHA"]) or sha256_file(ledger) != checked["ROOT整件SHA256"]:
        raise ValueError("Howard观察前后ROOT整件或本人六SHA活动单格漂移")
    _hash(Path(__file__), checked["登记"]["源码普通成员SHA256"]["src/sic_cu/eval/task11_howard_observations.py"])


def native_radial_masks(radii_m: np.ndarray) -> tuple[np.ndarray, ...]:
    radii = np.asarray(radii_m)
    if (radii.ndim != 1 or radii.dtype != np.float32 or not len(radii) or not np.isfinite(radii).all()
            or (radii < 0).any() or (radii > np.float32(.025)).any()):
        raise ValueError("Howard径向窗只能使用本人[0,25]毫米原生Float32米制半径")
    masks = tuple((radii >= np.float32(w["lower"] / 1000) if w["lower_closed"] else
                   radii > np.float32(w["lower"] / 1000)) & (radii <= np.float32(w["upper"] / 1000)) for w in RADIAL_WINDOWS)
    if not np.all(sum(mask.astype(np.int8) for mask in masks) == 1):
        raise ValueError("Howard原生径向窗存在漏点或双计")
    return masks


def _validate_frames(top: pl.DataFrame, sensors: pl.DataFrame) -> None:
    required = {"power_w", "time_s", "r_m", "split"}
    if (not (required | {"temperature_mean_k", "frame_weight"}).issubset(top.columns)
            or not (required | {"temperature_k", "delta_temperature_k", "sensor_type"}).issubset(sensors.columns)
            or top.height != 7272 or sensors.height != 752
            or top["r_m"].dtype != pl.Float32 or sensors["r_m"].dtype != pl.Float32):
        raise ValueError("Howard合法观察须原schema的7272Top及752两环原生Float32点")
    for frame in (top, sensors):
        if frame["split"].null_count() or set(frame["split"].to_list()) != {"validation"}:
            raise ValueError("Howard观察仅准合法三功率validation，不读取旧TEST或TRAIN")
        numerical._frame_power_masks(frame)
        for column in ("time_s", "r_m"):
            value = numerical._vector(frame[column].to_numpy(), "原生" + column, frame.height)
            if (value < 0).any() or (column == "time_s" and (value > 200).any()):
                raise ValueError("Howard观察原生时间/半径超出合法定义域")
    if sensors["sensor_type"].null_count() or set(sensors["sensor_type"].to_list()) != {"hot", "cold"}:
        raise ValueError("Howard两环须本人小写hot/cold真实类型")
    if top.select("power_w", "time_s", "r_m").is_duplicated().any() or sensors.select(
            "power_w", "sensor_type", "time_s").is_duplicated().any():
        raise ValueError("Howard逐点曲线重复或时序不唯一")
    order = top.select("power_w", "time_s", "r_m")
    if not order.equals(order.sort("power_w", "time_s", "r_m")):
        raise ValueError("HowardTop必须本人功率/时间/原生半径规范同序")
    for kind in ("hot", "cold"):
        order = sensors.filter(pl.col("sensor_type") == kind).select("power_w", "time_s")
        if not order.equals(order.sort("power_w", "time_s")):
            raise ValueError("Howard每环必须本人功率/时间规范同序")
        if order.height != 376:
            raise ValueError("HowardHot/Cold分别须376真实点")
    for frame, columns in ((top, ("temperature_mean_k", "frame_weight")),
                           (sensors, ("temperature_k", "delta_temperature_k"))):
        for column in columns:
            numerical._vector(frame[column].to_numpy(), "原生" + column, frame.height)
    weights = top["frame_weight"].to_numpy()
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("HowardTop本人原帧权重须非负且和严格为正")
    masks = numerical._frame_power_masks(top)
    for power in FIXED_POWERS:
        frame = top[np.flatnonzero(masks[power]).tolist()]
        if ((frame.height, frame["time_s"].n_unique()) != TOP_COUNTS[power]
                or frame["r_m"].n_unique() != 101 or any(n != 101 for n in frame.group_by("time_s").len()["len"])):
            raise ValueError("HowardTop逐功率须冻结19/24/29完整101半径帧")
        native_radial_masks(frame["r_m"].to_numpy())
        if int((frame["r_m"].to_numpy() == np.float32(.025)).sum()) != TOP_COUNTS[power][1]:
            raise ValueError("HowardTop每个原生完整帧须含本人25毫米闭端点")
        if (frame.group_by("time_s").agg(pl.col("frame_weight").sum())["frame_weight"] <= 0).any():
            raise ValueError("Howard每个Top帧的本人原权重和须为正")
        totals = frame.group_by("time_s").agg(pl.col("frame_weight").sum())["frame_weight"].to_numpy()
        if not np.allclose(totals, 1., rtol=0, atol=1e-6):
            raise ValueError("Howard每帧原逆方差可靠性权重须为既有归一结果，保留Float32舍入容差")
    sensor_masks = numerical._frame_power_masks(sensors)
    for power in FIXED_POWERS:
        for kind in ("hot", "cold"):
            curve = sensors[np.flatnonzero(sensor_masks[power]).tolist()].filter(pl.col("sensor_type") == kind)
            if curve.height != {115.2: 100, 403.: 126, 630.5: 150}[power] or curve["r_m"].n_unique() != 1:
                raise ValueError("Howard每功率每环须本人单半径完整原生曲线")


def evaluate_task11_howard_observation_arrays(top: pl.DataFrame, sensors: pl.DataFrame,
    top_prediction: np.ndarray, sensor_prediction: np.ndarray, *, seed: int, state: str,
    bottom_z_m: float) -> dict[str, Any]:
    if type(seed) is not int or seed not in range(5) or state not in STATES or type(bottom_z_m) not in (float, int) or bottom_z_m != -.0175:
        raise ValueError("Howard观察仅本人五seed双状态与冻结两环底面-0.0175米")
    _validate_frames(top, sensors)
    top_prediction = numerical._vector(top_prediction, "HowardTop预测", top.height)
    sensor_prediction = numerical._vector(sensor_prediction, "Howard两环预测", sensors.height)
    result = {"seed": seed, "state": state, "points": [], "power": [], "time": [], "radial": []}
    frames = [("Top", top, top_prediction)]
    for modality, kind in (("Hot", "hot"), ("Cold", "cold")):
        mask = sensors["sensor_type"].to_numpy() == kind
        frames.append((modality, sensors.filter(pl.col("sensor_type") == kind), sensor_prediction[mask]))
    for modality, frame, predictions in frames:
        for power, mask in numerical._frame_power_masks(frame).items():
            indices = np.flatnonzero(mask)
            group = frame[indices.tolist()]
            target = numerical._vector(group["temperature_mean_k" if modality == "Top" else "temperature_k"].to_numpy(), "本人温度", group.height)
            prediction = predictions[indices]
            times = numerical._vector(group["time_s"].to_numpy(), "本人时刻", group.height)
            radii = group["r_m"].to_numpy()
            weights = numerical._vector(group["frame_weight"].to_numpy(), "本人帧权重", group.height) if modality == "Top" else np.ones(group.height)
            keys = radii if modality == "Top" else np.zeros(group.height, dtype=np.int8)
            target_delta, prediction_delta = _delta_from_first(target, prediction, times, keys)
            first_indices = np.empty(group.height, dtype=np.int64)
            for key in np.unique(keys):
                selected = np.flatnonzero(keys == key)
                first_indices[selected] = selected[np.argmin(times[selected])]
            if modality != "Top" and not np.allclose(target_delta, group["delta_temperature_k"].to_numpy(), atol=1e-4, rtol=0):
                raise ValueError("Howard两环差分必须本人首真实时刻，不能虚造t=0")
            arrays = {"target": target, "prediction": prediction, "times": times, "weights": weights,
                      "target_delta": target_delta, "prediction_delta": prediction_delta}
            identity = {"seed": seed, "state": state, "split": "validation", "power_w": power, "modality": modality}
            result["power"].append({**identity, "sample_count": group.height, "time_count": len(np.unique(times)),
                **numerical._metrics(arrays, np.ones(group.height, dtype=bool), "")})
            temporal_masks = []
            for window in TIME_WINDOWS:
                selected = (times >= window["lower"] if window["lower_closed"] else times > window["lower"]) & (times <= window["upper"])
                temporal_masks.append(selected)
                result["time"].append({**identity, "window": window["name"], "lower_s": window["lower"], "upper_s": window["upper"],
                    "lower_closed": window["lower_closed"], "upper_closed": True, "sample_count": int(selected.sum()),
                    **numerical._metrics(arrays, selected, "预登记时间窗内无观测，不外推真值")})
            if not np.all(sum(mask.astype(np.int8) for mask in temporal_masks) == 1):
                raise ValueError("Howard时间窗存在漏点或双计")
            if modality == "Top":
                for window, selected in zip(RADIAL_WINDOWS, native_radial_masks(radii)):
                    result["radial"].append({**identity, "radial_window": window["name"], "lower_mm": window["lower"], "upper_mm": window["upper"],
                        "lower_closed": window["lower_closed"], "upper_closed": True, "sample_count": int(selected.sum()),
                        **numerical._metrics(arrays, selected, "预登记径向窗内无顶部观测，不外推真值")})
            for index, first in enumerate(first_indices):
                result["points"].append({**identity, "r_m": float(radii[index]), "z_m": 0. if modality == "Top" else float(bottom_z_m),
                    "time_s": float(times[index]), "material_id": 1 if modality == "Top" else 0,
                    "target_k": float(target[index]), "prediction_k": float(prediction[index]), "weight": float(weights[index]),
                    "reference_time_s": float(times[first]), "reference_target_k": float(target[first]),
                    "reference_prediction_k": float(prediction[first]), "target_delta_k": float(target_delta[index]),
                    "prediction_delta_k": float(prediction_delta[index]), "error_c": float(prediction[index] - target[index]),
                    "delta_error_c": float(prediction_delta[index] - target_delta[index]), "source_row": int(indices[index])})
    numerical._finite_tree(result)
    if [len(result[k]) for k in ("points", "power", "time", "radial")] != [8024, 9, 27, 9]:
        raise ValueError("Howard逐点/功率/时间/径向覆盖计数不完整")
    return result


def reconstruct_task11_observation_macro(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return numerical.reconstruct_task11_observation_macro(records)


def five_seed_signed_statistics(values: Sequence[Any]) -> dict[str, Any]:
    return numerical.five_seed_signed_statistics(values)


def _group_statistics(results: Sequence[Mapping[str, Any]], kind: str) -> dict[str, Any]:
    return {v["分组中文名称"]: v for v in numerical._group_statistics(results, kind).values()}


def verify_task11_howard_observation_macro(result: Mapping[str, Any], view: Mapping[str, Any]) -> dict[str, float]:
    value = reconstruct_task11_observation_macro(result["power"])
    expected = view.get("validation_selection_score_c")
    modal = view.get("validation_modalities")
    keys = set(value) - {"合法HF原macro_v1_摄氏度"}
    if (type(expected) not in (int, float) or not math.isfinite(expected) or not isinstance(modal, dict)
            or set(modal) != keys or not math.isclose(value["合法HF原macro_v1_摄氏度"], expected, rel_tol=0, abs_tol=1e-4)
            or any(type(modal[k]) not in (int, float) or not math.isfinite(modal[k]) or
                not math.isclose(value[k], modal[k], rel_tol=0, abs_tol=1e-4) for k in keys)):
        raise ValueError("Howard本人视图选分/Top/两环/差分macro重算超过1e-4摄氏度或模态字段不符")
    return value


def task11_howard_observation_model_from_view(view: Mapping[str, Any], *, seed: int) -> Task11HowardComposite:
    try:
        identity, state = view["任11HowardHF事前来源"], view["model_state"]
        if (type(seed) is not int or seed not in range(5) or type(view.get("seed")) is not int or view["seed"] != seed
                or type(view.get("schema_version")) is not int or view["schema_version"] != 1
                or view.get("method") != hf.METHOD or type(view.get("epoch")) is not int or view["epoch"] <= 0
                or not isinstance(identity, dict) or type(identity.get("seed")) is not int or identity["seed"] != seed
                or identity.get("method") != hf.METHOD or view.get("HF训练许可") is not False
                or any(view.get(n) is not False for n in hf.ISOLATION_FLAGS)
                or not isinstance(state, dict) or not hf._same(view.get("scales"), hf.HF_BUDGET["scales"])):
            raise ValueError("Howard观察模型必须本人真实HF方法/seed/正轮次/隔离身份")
        hf._validate_queries(state["linear_query_points"], identity["固定PHQH查询"])
        hf._validate_queries(state["nonlinear_query_points"], identity["固定PHQH查询"])
        model = hf._fresh_cpu_model()
        reference = model.state_dict()
        if (not hf._same(view.get("model_kwargs"), model.model_kwargs) or state.keys() != reference.keys()
                or any(not isinstance(v, Tensor) or v.device.type != "cpu" or v.layout != torch.strided
                    or v.dtype != reference[n].dtype or v.shape != reference[n].shape or
                    not torch.isfinite(v).all() for n, v in state.items())):
            raise ValueError("Howard观察须全部56真实Float32参数及两个原Float64查询，无load隐式修复")
        model.load_state_dict(state, strict=True)
        hf._validate_model(model)
        return model.eval()
    except (KeyError, TypeError, AttributeError, RuntimeError) as error:
        raise ValueError("Howard本人HF视图metadata/state全56参数与PH/QH严格恢复失败") from error


def predict_task11_howard_observations(model: Task11HowardComposite, coordinates: np.ndarray,
    device: torch.device, batch_size: int) -> np.ndarray:
    hf._validate_model(model)
    if any(p.device != device for p in model.parameters()) and device.type != "cpu":
        raise ValueError("Howard观察模型与输入设备不一致")
    return numerical.predict_task11_observations(model, coordinates, device, batch_size)


def _pin_run(originals: Any, run: Path, root: Path, pins: dict[str, str]) -> None:
    if not isinstance(originals, dict) or not originals:
        raise ValueError("Howard观察资格缺真实完整工件SHA")
    for relative, digest in originals.items():
        if type(relative) is not str or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Howard观察真实工件必须本人目录内普通相对路径")
        path = _path(run / relative, root)
        if not path.is_relative_to(run):
            raise ValueError("Howard观察不能借用跨模型目录工件")
        pins[str(path)] = _hash(path, digest)


def qualify_task11_howard_observation_five(checked: Mapping[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[str, str]]:
    root = Path(checked["项目根"])
    verify_task11_howard_observation_runtime(checked, {})
    qualified, pins = {}, {}
    for seed in range(5):
        run = root / hf.RUN_DIRECTORY / f"正式Howard_HF_seed{seed}"
        q = hf.qualify_task11_howard_hf_source(**checked["HowardHF四来源"], output=run, seed=seed, project_root=root)
        identity = q.get("事前来源") if isinstance(q, dict) else None
        if (not isinstance(q, dict) or q.get("状态") != HF_QUALIFIED_STATUS or type(q.get("seed")) is not int or q["seed"] != seed
                or q.get("已完成") is not True or q.get("HF训练许可") is not False
                or not isinstance(identity, dict) or identity.get("method") != hf.METHOD
                or type(identity.get("seed")) is not int or identity["seed"] != seed
                or any(identity.get(k) != checked["ROOT六SHA"][k] for k in hf.IDENTITY_FIELDS)
                or not hf._same(q.get("限制"), hf.LIMITATIONS)):
            raise ValueError("Howard观察五seed必须全部为本人真实终态与当前来源完整CPU资格PASS")
        originals = q.get("真实工件SHA256")
        required = {"best.pt", "final.pt", "metrics.json", "training.jsonl", "阶段_最近.pt", "事前真实来源登记.json"}
        if not isinstance(originals, dict) or not required.issubset(originals):
            raise ValueError("Howard观察资格缺完整本人模型/日志/终态/来源收据")
        _pin_run(originals, run, root, pins)
        for state, key in (("best", "最佳HF检查点SHA256"), ("final", "真实末HF检查点SHA256")):
            if q.get(key) != originals[state + ".pt"] or q[key] != checked["模型十身份"][(seed, state)]["模型SHA256"]:
                raise ValueError("Howard观察登记十视图与本人完整HF资格SHA不一致")
        qualified[seed] = q
    catalog = numerical._read_json(_path(checked["HowardHF四来源"]["lf_catalog"], root))
    rows = catalog.get("五LF真实完整资格") if isinstance(catalog, dict) else None
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("Howard观察LF目录必须本人五LF完整资格，不能借MLP目录字段")
    for seed, q in enumerate(rows):
        if not isinstance(q, dict) or type(q.get("seed")) is not int or q["seed"] != seed:
            raise ValueError("Howard观察真实五LF来源种子错序")
        _pin_run(q.get("真实工件SHA256"), root / hf.LF_RUN_DIRECTORY / f"正式Howard_LF_seed{seed}", root, pins)
    for name, digest in hf.LF_SOURCE_IDENTITY.items():
        if not name.endswith("_sha"):
            pins[str(_path(digest, root))] = _hash(_path(digest, root), hf.LF_SOURCE_IDENTITY[name + "_sha"])
    hf_catalog = numerical._read_json(_path(checked["HowardHF四来源"]["hf_data_catalog"], root))
    rows = hf_catalog.get("HF源原件") if isinstance(hf_catalog, dict) else None
    if (not isinstance(rows, list) or hf_catalog.get("HF合法验证功率_瓦") != list(FIXED_POWERS)
            or hf_catalog.get("旧固定TEST温度读取") is not False or hf_catalog.get("模拟测试功率温度读取") is not False
            or len(rows) != 2 or {r.get("原件") for r in rows if isinstance(r, dict)} != {
                "data/processed/experiment_ir_radial.parquet", "data/processed/sensor_ring_raw.parquet"}):
        raise ValueError("Howard观察HF结构来源只能本人Top/两环合法VAL三功率")
    for row in rows:
        path = _path(row["原件"], root)
        pins[str(path)] = _hash(path, row["文件SHA256"])
    verify_task11_howard_observation_runtime(checked, pins)
    return qualified, pins


def require_task11_howard_observation_cuda(device_name: str) -> torch.device:
    if device_name != "cuda" or any(os.environ.get(k, d) != d for k, d in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0"))):
        raise ValueError("Howard正式观察只准单卡CUDA，不准CPU/多卡fallback")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise ValueError("Howard正式观察必须真实使用唯一可见CUDA卡cuda:0")
    return torch.device("cuda:0")


def _cuda_prediction(model: Task11HowardComposite, coordinates: np.ndarray, device: torch.device) -> tuple[np.ndarray, int]:
    hf._validate_model(model)
    values = np.asarray(coordinates)
    if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != 5 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Howard真实CUDA前向须原生有限Float32五列输入")
    if device != torch.device("cuda:0") or any(p.device != device for p in model.parameters()) or any(b.device != device for b in model.buffers()):
        raise ValueError("Howard真实前向模型56参数与两个查询必须确实在cuda:0")
    chunks = []
    with torch.no_grad():
        for offset in range(0, len(values), 8192):
            inputs = torch.from_numpy(values[offset:offset + 8192]).to(device)
            outputs = model(inputs)
            if inputs.device != device or outputs.device != device or outputs.dtype != torch.float32 or not torch.isfinite(outputs).all():
                raise ValueError("Howard预测必须真实CUDA输入和真实CUDA有限Float32输出，不能CPU伪替换")
            chunks.append(outputs.detach().cpu().numpy())
    torch.cuda.synchronize(device)
    return numerical._vector(np.concatenate(chunks).reshape(-1), "Howard真实CUDA预测", len(values)), len(chunks)


def verify_task11_howard_cuda_receipts(receipts: Any) -> None:
    expected = [(seed, state) for seed in range(5) for state in STATES]
    if not isinstance(receipts, list) or len(receipts) != 10:
        raise ValueError("Howard正式观察须十状态全部真实CUDA前向收据")
    fields = {"种子", "模型状态", "设备", "输入设备", "输出设备", "真实CUDA前向", "Top点数", "热环点数", "冷环点数", "实际前向批次数"}
    for row, identity in zip(receipts, expected):
        if (not isinstance(row, dict) or set(row) != fields or type(row["种子"]) is not int
                or (row["种子"], row["模型状态"]) != identity or row["真实CUDA前向"] is not True
                or any(row[n] != "cuda:0" for n in ("设备", "输入设备", "输出设备"))
                or any(type(row[n]) is not int or row[n] != v for n, v in (("Top点数", 7272), ("热环点数", 376), ("冷环点数", 376)))
                or type(row["实际前向批次数"]) is not int or row["实际前向批次数"] != 2):
            raise ValueError("Howard十状态CUDA收据必须有本人全点真实前向；CPU伪CUDA或缺席不合格")


def _validate_results(results: Sequence[Mapping[str, Any]]) -> None:
    if len(results) != 10 or [(r["seed"], r["state"]) for r in results] != [(s, v) for s in range(5) for v in STATES]:
        raise ValueError("Howard观察写出须本人五seed双状态十份有序完整结果")
    for result in results:
        if [len(result[k]) for k in ("points", "power", "time", "radial")] != [8024, 9, 27, 9]:
            raise ValueError("Howard观察逐点与分组计数缺失或双计")
        numerical._finite_tree(result)
        for kind in ("points", "power", "time", "radial"):
            if any(r["seed"] != result["seed"] or r["state"] != result["state"] or r["split"] != "validation" for r in result[kind]):
                raise ValueError("Howard观察结果与本人seed/state/validation不一致")
        for point in result["points"]:
            if set(point) != set(POINT_ZH) or any(type(point[k]) not in (int, float) or not math.isfinite(point[k]) for k in set(POINT_ZH) - {"state", "split", "modality"}):
                raise ValueError("Howard逐点schema/有限实数/中文映射不完整")
            if type(point["source_row"]) is not int or point["source_row"] < 0 or point["weight"] < 0:
                raise ValueError("Howard逐点源行与权重须非负本人身份")
        for modality in MODALITIES:
            points = [p for p in result["points"] if p["modality"] == modality]
            if [p["source_row"] for p in points] != list(range(len(points))):
                raise ValueError("Howard逐点源行须本人规范同序完整双射")
        top_points = [p for p in result["points"] if p["modality"] == "Top"]
        sensor_points = [p for p in result["points"] if p["modality"] in ("Hot", "Cold")]
        top = pl.DataFrame({"power_w": [p["power_w"] for p in top_points], "time_s": [p["time_s"] for p in top_points],
            "r_m": pl.Series([p["r_m"] for p in top_points], dtype=pl.Float32), "temperature_mean_k": [p["target_k"] for p in top_points],
            "frame_weight": [p["weight"] for p in top_points], "split": [p["split"] for p in top_points]})
        sensors = pl.DataFrame({"power_w": [p["power_w"] for p in sensor_points], "time_s": [p["time_s"] for p in sensor_points],
            "r_m": pl.Series([p["r_m"] for p in sensor_points], dtype=pl.Float32), "sensor_type": ["hot" if p["modality"] == "Hot" else "cold" for p in sensor_points],
            "temperature_k": [p["target_k"] for p in sensor_points], "delta_temperature_k": [p["target_delta_k"] for p in sensor_points],
            "split": [p["split"] for p in sensor_points]})
        regenerated = evaluate_task11_howard_observation_arrays(top, sensors, np.asarray([p["prediction_k"] for p in top_points]),
            np.asarray([p["prediction_k"] for p in sensor_points]), seed=result["seed"], state=result["state"], bottom_z_m=-.0175)
        for kind in ("points", "power", "time", "radial"):
            if not numerical._same_values(result[kind], regenerated[kind]):
                raise ValueError("Howard逐点重算与本人参考时刻/坐标/材料/差分/功率/时间/径向指标不一致")
        if not numerical._same_values(result.get("macro_reproduction"), reconstruct_task11_observation_macro(regenerated["power"])):
            raise ValueError("Howard逐点重算本人原macro不一致")


def _macro_zh(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"合法HF原macro_v1_摄氏度": "合法HF原macro_v1_摄氏度", "顶部": "顶部",
        "absolute_rmse_c": "两环绝对RMSE_摄氏度", "absolute_mae_c": "两环绝对MAE_摄氏度",
        "delta_rmse_c": "两环首实测差分RMSE_摄氏度", "delta_mae_c": "两环首实测差分MAE_摄氏度"}
    return {fields[k]: v for k, v in value.items()}


def write_task11_howard_observation_evidence(checked: Mapping[str, Any], results: Sequence[Mapping[str, Any]],
    qualifications_before: Mapping[int, Mapping[str, Any]], qualifications_after: Mapping[int, Mapping[str, Any]],
    *, additional_pins: Mapping[str, str] | None = None, synthetic_cpu: bool = False) -> dict[str, Any]:
    """公开写出只准显式隔离CPU夹具；真实写出仅由正式导出入口调用。"""
    if synthetic_cpu is not True:
        raise ValueError("Howard公开观察写出仅准显式CPU合成夹具，正式写出须本人CUDA导出入口")
    return _write_evidence(checked, results, qualifications_before, qualifications_after,
        additional_pins=dict(additional_pins or {}), synthetic_cpu=True, cuda_receipts=[])


def _write_evidence(checked: Mapping[str, Any], results: Sequence[Mapping[str, Any]],
    before: Mapping[int, Mapping[str, Any]], after: Mapping[int, Mapping[str, Any]], *,
    additional_pins: Mapping[str, str], synthetic_cpu: bool, cuda_receipts: list[dict[str, Any]]) -> dict[str, Any]:
    pins = dict(additional_pins)
    verify_task11_howard_observation_runtime(checked, pins)
    root = Path(checked["项目根"])
    if type(synthetic_cpu) is not bool or before != after:
        raise ValueError("Howard观察前后本人完整资格不相等或CPU标记非布尔")
    if synthetic_cpu:
        if before or after or cuda_receipts or root == PROJECT_ROOT.resolve() or not root.is_relative_to(PROJECT_ROOT.resolve() / OUTPUT_BASE):
            raise ValueError("HowardCPU合成写出只准隔离根，不得伪装真实资格或CUDA")
    else:
        if root != PROJECT_ROOT.resolve() or set(before) != set(range(5)):
            raise ValueError("Howard真实观察只准当前正式项目全部五seed资格")
        verify_task11_howard_cuda_receipts(cuda_receipts)
        require_task11_howard_observation_cuda("cuda")
        current, current_pins = qualify_task11_howard_observation_five(checked)
        if current != before or current != after or any(p in pins and pins[p] != d for p, d in current_pins.items()):
            raise ValueError("Howard真实观察写前fresh五seed资格或原件SHA与预测前后不相等")
        pins = {**current_pins, **pins}
    _validate_results(results)
    statistics_by_state = {}
    for state in STATES:
        selected = [r for r in results if r["state"] == state]
        statistics_by_state[VALUE_ZH[state]] = {"模型状态原标识": state,
            "逐功率": _group_statistics(selected, "power"), "时间窗": _group_statistics(selected, "time"),
            "原生径向窗": _group_statistics(selected, "radial"),
            "本人原macro_v1_摄氏度": five_seed_signed_statistics([r["macro_reproduction"]["合法HF原macro_v1_摄氏度"] for r in selected]),
            "分模态宏平均": {VALUE_ZH[m]: {zh: five_seed_signed_statistics([
                statistics.fmean(row[raw] for row in r["power"] if row["modality"] == m) for r in selected])
                for raw, zh in METRIC_ZH.items()} for m in MODALITIES}}
    summary = {"状态": "HowardCPU合成回归，非真实模型或实测验收" if synthetic_cpu else "Howard本人五seed双状态合法实测三模态只读观察",
        "方法": OBSERVATION_CONTRACT["模型方法"], "总点数": 80240, "验收计数": {"逐功率": 90, "时间窗": 270, "径向窗": 90},
        "观察合同": OBSERVATION_CONTRACT, "模型双状态五种子统计": statistics_by_state,
        "统计限定": {"独立试验单位": "五个本人模型种子，点不充当独立试验", "样本标准差自由度扣减": 1,
            "功率宏平均": "三个固定验证功率等权", "绝对经验p95": "各seed各功率各模态或窗口原始逐点绝对误差95分位，无权重，无合并五seed",
            "差分基准": "每条原生曲线首真实时刻，不虚造t=0", "有符号指标": "偏差及峰差均保留负号", "空窗": "全部指标null并给中文原因"},
        "限制": {"科学合格主张": False, "工程安全合格主张": False, "采用B0部署": False, "新增测量": False,
            "任务目标完成": False, "内部温度真值已证明": False, "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
            "训练或优化": False, "重新选模": False, "能源重新统计": False, "训练成本重新统计": False,
            "原文精确三网联合训练复现": False, "CPU合成回归": synthetic_cpu},
        "来源门禁": {k: checked[k] for k in ("ROOT记录号", "ROOT原件", "ROOT整件SHA256", "ROOT唯一活动行", "ROOT六SHA")},
        "五seed本人完整来源资格": [{"种子": seed, "本人真实工件SHA256": q["真实工件SHA256"],
            "本人LF起点SHA256": q["事前来源"]["LF最佳检查点SHA256"]} for seed, q in sorted(before.items())],
        "十状态真实CUDA前向收据": cuda_receipts, "读取原件执行前后SHA256": {**checked["读取原件SHA256"], **pins}}
    numerical._finite_tree(summary)
    content = {"机器汇总.json": json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "真实逐点预测.jsonl": "".join(json.dumps(numerical._localize(p, POINT_ZH), ensure_ascii=False, allow_nan=False) + "\n" for r in results for p in r["points"]),
        "原始指标与中文映射.json": json.dumps({"中文列映射": {"逐点": POINT_ZH, "记录": FIELD_ZH}, "十状态原指标": [
            {"种子": r["seed"], "模型状态中文名称": VALUE_ZH[r["state"]], "本人原macro核验": _macro_zh(r["macro_reproduction"]),
             "逐功率": [numerical._localize(v, FIELD_ZH) for v in r["power"]], "时间窗": [numerical._localize(v, FIELD_ZH) for v in r["time"]],
             "原生径向窗": [numerical._localize(v, FIELD_ZH) for v in r["radial"]]} for r in results]}, ensure_ascii=False, indent=2, allow_nan=False) + "\n"}
    lines = ["# Howard本人三模态观察验收", "", summary["状态"], "", summary["方法"], "",
        "仅固定合法VAL功率115.2/403/630.5瓦；五种子各观测最佳和训练末，十状态80240点。",
        "每状态顶部7272、热环376、冷环376点；逐功率90、时间窗270、原生径向窗90条。",
        "顶部采用本人原帧逆方差可靠性权重，每帧既有归一结果原值保留，统计范围仅原权重和归一；两环各自等点，三个功率等权。",
        "不重算几何面积，不在径向窗内重新逐帧配权，不修改已有macro与评价协议。",
        "时间窗[0,30]、(30,100]、(100,200]秒；顶部原生径向窗[0,8]、(8,17]、(17,25]毫米。",
        "25毫米原生Float32闭端点十状态共720点；8和17毫米按原生表示正确归属，不遗漏不双计。",
        "差分以每条本人原生曲线首真实实测时刻为基准；不虚造零时刻。",
        "p95为各seed各功率各模态或窗口未加权逐点经验95分位；峰差为max预测减max实测，保留符号。",
        "五种子保留原五值，以ddof=1计算样本标准差；空桶为null并说明原因，不补零不外推。", ""]
    for state, value in statistics_by_state.items():
        stat = value["本人原macro_v1_摄氏度"]
        lines.append(f"{state}原macro_v1均值={stat['均值']:.9f}摄氏度；样本标准差={stat['样本标准差']:.9f}；原五值={stat['逐种子']}。")
    lines += ["", "原选分和Top/两环/差分指标按本人冻结视图重算，容差0.0001摄氏度，不改变检查点选择。",
        "不训练、不重新选模、不重新统计能源或成本、不读取旧固定TEST与模拟TEST温度、不新增测量、不采用B0部署。",
        "不证明内部温度真值，不给出科学或工程安全合格主张，不代表总体任务目标完成。",
        "本目录仅CPU合成回归，没有真实HF资格或CUDA许可，不代表正式十状态观察已运行。" if synthetic_cpu else
        "正式模式全部五seed预测前、预测后及写出前完整资格与所有原件SHA均鲜验相同，十状态确实运行cuda:0。", ""]
    content["中文验收.md"] = "\n".join(lines)
    verify_task11_howard_observation_runtime(checked, pins)
    destination = _new_output(checked["输出"], root)
    destination.mkdir()
    for name, text in content.items():
        (destination / name).write_text(text, encoding="utf-8")
    for name, kind in (("逐功率三模态.csv", "power"), ("逐功率三模态时间窗.csv", "time"), ("顶部逐功率原生径向窗.csv", "radial")):
        rows = [numerical._localize(row, FIELD_ZH) for r in results for row in r[kind]]
        with (destination / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    verify_task11_howard_observation_runtime(checked, pins)
    manifest = {p.name: sha256_file(p) for p in sorted(destination.iterdir())}
    (destination / "工件SHA256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    verify_task11_howard_observation_runtime(checked, pins)
    return summary


def export_task11_howard_observations(*, device_name: str = "cuda", **arguments: Any) -> dict[str, Any]:
    checked = preflight_task11_howard_observations(**arguments)
    root = Path(checked["项目根"])
    if root != PROJECT_ROOT.resolve() or device_name != "cuda":
        raise ValueError("Howard正式观察仅当前正式项目CUDA入口，没有CPU夹具绕行")
    before, pins = qualify_task11_howard_observation_five(checked)
    device = require_task11_howard_observation_cuda(device_name)
    verify_task11_howard_observation_runtime(checked, pins)
    top = load_processed_ir_observations("validation").sort("power_w", "time_s", "r_m")
    metadata = _path("configs/data_metadata.yaml", root)
    _hash(metadata, checked["登记"]["源码普通成员SHA256"]["configs/data_metadata.yaml"])
    sensors = load_canonical_sensor_observations(metadata_path=str(metadata),
        processed_path=str(root / "data/processed/sensor_ring_raw.parquet"), split="validation").sort("power_w", "sensor_type", "time_s")
    _validate_frames(top, sensors)
    results, receipts = [], []
    for seed in range(5):
        geometry = _path(root / hf.RUN_DIRECTORY / f"正式Howard_HF_seed{seed}/config_snapshot/geometry.yaml", root)
        bottom = float(load_yaml(geometry)["embedding"]["copper_bottom_z_m"])
        coordinates_top = np.column_stack((top["r_m"].to_numpy(), np.zeros(top.height), top["time_s"].to_numpy(),
            top["power_w"].to_numpy(), np.ones(top.height))).astype(np.float32)
        coordinates_sensor = np.column_stack((sensors["r_m"].to_numpy(), np.full(sensors.height, bottom), sensors["time_s"].to_numpy(),
            sensors["power_w"].to_numpy(), np.zeros(sensors.height))).astype(np.float32)
        for state in STATES:
            verify_task11_howard_observation_runtime(checked, pins)
            row = checked["模型十身份"][(seed, state)]
            path = _path(row["模型原件"], root)
            _hash(path, row["模型SHA256"])
            view = torch.load(path, map_location="cpu", weights_only=True)
            _hash(path, row["模型SHA256"])
            if not isinstance(view, dict) or not hf._same(view.get("任11HowardHF事前来源"), before[seed]["事前来源"]):
                raise ValueError("Howard本人视图与真实完整HF资格来源不相等")
            model = task11_howard_observation_model_from_view(view, seed=seed).to(device).eval()
            pt, nt = _cuda_prediction(model, coordinates_top, device)
            ps, ns = _cuda_prediction(model, coordinates_sensor, device)
            result = evaluate_task11_howard_observation_arrays(top, sensors, pt, ps, seed=seed, state=state, bottom_z_m=bottom)
            result["macro_reproduction"] = verify_task11_howard_observation_macro(result, view)
            receipts.append({"种子": seed, "模型状态": state, "设备": str(device), "输入设备": str(device), "输出设备": str(device),
                "真实CUDA前向": True, "Top点数": len(pt), "热环点数": 376, "冷环点数": 376, "实际前向批次数": nt + ns})
            verify_task11_howard_observation_runtime(checked, pins)
            results.append(result)
            del model, view
    after, after_pins = qualify_task11_howard_observation_five(checked)
    if after != before or any(p in pins and pins[p] != d for p, d in after_pins.items()):
        raise ValueError("Howard预测前后全部五seed完整资格与本人原件SHA不相等")
    verify_task11_howard_cuda_receipts(receipts)
    return _write_evidence(checked, results, before, after, additional_pins=pins,
        synthetic_cpu=False, cuda_receipts=receipts)
