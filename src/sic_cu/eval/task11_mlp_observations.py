"""Byte-locked, validation-only post-hoc observations of the new Task-11 MLP."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import tarfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
import torch
import yaml
from torch import nn

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.processed import load_processed_ir_observations
from sic_cu.data.sensors import load_canonical_sensor_observations
from sic_cu.eval.development_v4 import _delta_from_first, _predict, error_statistics
from sic_cu.eval.task11_mlp_hf_energy import task11_energy_model_from_view
from sic_cu.train.task11_mlp_hf_formal import (
    HF_BUDGET, ROOT_LEDGER, RUN_DIRECTORY,
    SOURCE_MEMBERS as HF_SOURCE_MEMBERS, qualify_task11_mlp_hf_source,
)


ROOT_TOKEN = "TASK11_NEW_MLP_OBSERVATIONS_GATE:v1"
OUTPUT_BASE = "研究记录/任务11_外部对照"
OBS_REGISTRY_PATH = OUTPUT_BASE + "/新MLP合法三模态观察导出前登记.yaml"
OBS_TAR_PATH = OUTPUT_BASE + "/新MLP合法三模态观察源码事前冻结.tar.gz"
FIXED_POWERS = (115.2, 403.0, 630.5)
MODALITIES = ("Top", "Hot", "Cold")
STATES = ("best", "final")
TOP_COUNTS = {115.2: (1919, 19), 403.0: (2424, 24), 630.5: (2929, 29)}
TIME_WINDOWS = (
    {"name": "time_0_30_s", "lower": 0.0, "upper": 30.0, "lower_closed": True},
    {"name": "time_30_100_s", "lower": 30.0, "upper": 100.0, "lower_closed": False},
    {"name": "time_100_200_s", "lower": 100.0, "upper": 200.0, "lower_closed": False},
)
RADIAL_WINDOWS = (
    {"name": "center_0_8_mm", "lower": 0.0, "upper": 8.0, "lower_closed": True},
    {"name": "middle_8_17_mm", "lower": 8.0, "upper": 17.0, "lower_closed": False},
    {"name": "outer_17_25_mm", "lower": 17.0, "upper": 25.0, "lower_closed": False},
)
HF_SOURCE_IDENTITY = {
    "registry": {
        "原件": OUTPUT_BASE + "/任11_新MLP五种子HF校正全LF联合事前登记_20260916T154637+0800.yaml",
        "SHA256": "3710cadaeccb8dcd0c6a584d3a0da9eb1866594bebc3df417c83772a7e3fffe7"},
    "source_tar": {
        "原件": OUTPUT_BASE + "/任11_新MLP五种子HF校正全LF联合67源事前冻结_20260916T154637+0800.tar.gz",
        "SHA256": "e5dc2f64ee72b4da56266c0f22c3754af08bc7d77912d77ee375a35ef2d26952"},
    "lf_catalog": {
        "原件": OUTPUT_BASE + "/任11_新MLP五LF完训身份清单_20260916T153147.json",
        "SHA256": "37af1efbf779b6782983417805fae0e92be20b0b80835e087caf63b177869956"},
    "hf_data_catalog": {
        "原件": OUTPUT_BASE + "/任11_新MLP_HF12_3开发结构来源清单_20260916T153147.json",
        "SHA256": "7665603abb31fa225c5ef36394ddf7c3b97461340b8b3307863766cbae252e18"},
}
SOURCE_MEMBERS = tuple(sorted(set(HF_SOURCE_MEMBERS) | {
    "src/sic_cu/eval/development_v4.py", "scripts/30_audit_task07_radial_endpoint.py",
    "src/sic_cu/eval/task11_mlp_observations.py", "scripts/57_export_task11_mlp_observations.py",
    "tests/test_task11_mlp_observations.py",
}))
OBSERVATION_CONTRACT = {
    "种子": [0, 1, 2, 3, 4], "模型状态": ["best", "final"],
    "划分": "validation", "固定合法功率_瓦": list(FIXED_POWERS),
    "模态": list(MODALITIES), "每状态Top点数": 7272, "每状态两环点数": 752,
    "每状态总点数": 8024, "十状态总点数": 80240,
    "每状态逐功率记录": 9, "每状态时间窗记录": 27, "每状态径向窗记录": 9,
    "时间窗_秒": list(TIME_WINDOWS), "原生径向窗_毫米": list(RADIAL_WINDOWS),
    "预测dtype": "float32", "Top权重": "原实测frame_weight；每统计范围归一",
    "两环权重": "Hot和Cold分别等点；分模态和功率宏平均",
    "差分基准": "各功率各Top半径本人首实测；各功率各Hot或Cold本人首实测",
    "绝对p95": "各功率各模态或各窗未加权逐点绝对误差经验95分位；再按五seed统计",
    "差分p95": "相同实测曲线首时刻差分误差的未加权经验95分位；再按五seed统计",
    "峰值误差": "同一统计范围max(预测温度)-max(实测温度)，保留符号",
    "径向端点限制": "仅同源Top支持25毫米闭端点原生float32修复；不声称任意8或17毫米采样适用",
    "空窗": "全部指标null并给中文原因，不补零，不外推",
    "五seed统计": "均值和样本标准差ddof=1，保留原五值、最小与最大；点不充当独立试验",
    "原macro核验容差_摄氏度": 0.0001, "固定选分权重": HF_BUDGET["selection_weights"],
    "只读完训模型": True, "原件前后SHA一致": True,
    "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
    "训练或优化": False, "重新选模": False, "能源重新统计": False,
    "训练成本重新统计": False, "工程安全阈值已建立": False,
    "证明内部温度真值": False, "修改B0部署": False,
    "输出父目录": OUTPUT_BASE, "设备": "单卡CUDA；禁止CPU正式导出或多卡fallback",
}

METRIC_ZH = {
    "rmse_c": "绝对RMSE_摄氏度", "mae_c": "绝对MAE_摄氏度",
    "signed_bias_c": "有符号偏差_摄氏度", "p95_abs_error_c": "绝对误差经验p95_摄氏度",
    "max_abs_error_c": "最大绝对误差_摄氏度", "delta_rmse_c": "首实测差分RMSE_摄氏度",
    "delta_mae_c": "首实测差分MAE_摄氏度", "delta_signed_bias_c": "首实测差分有符号偏差_摄氏度",
    "delta_p95_abs_error_c": "首实测差分绝对误差经验p95_摄氏度",
    "delta_max_abs_error_c": "首实测差分最大绝对误差_摄氏度",
    "peak_error_c": "有符号峰值误差_摄氏度",
}
FIELD_ZH = {
    "seed": "种子", "state": "模型状态原标识", "split": "划分原标识",
    "power_w": "功率_瓦", "modality": "模态原标识", "sample_count": "实测点数",
    "time_count": "实际时刻数", "window": "时间窗原标识", "radial_window": "径向窗原标识",
    "lower_s": "下界_秒", "upper_s": "上界_秒", "lower_mm": "下界_毫米",
    "upper_mm": "上界_毫米", "lower_closed": "下界闭端点", "upper_closed": "上界闭端点",
    "empty_reason": "空窗原因", **METRIC_ZH,
}
POINT_ZH = {
    "seed": "种子", "state": "模型状态原标识", "split": "划分原标识",
    "power_w": "功率_瓦", "modality": "模态原标识", "r_m": "半径_米",
    "z_m": "轴向坐标_米", "time_s": "实测时刻_秒", "material_id": "材料标识",
    "target_k": "实测温度_K", "prediction_k": "预测温度_K", "weight": "原统计权重",
    "reference_time_s": "本人曲线首实测时刻_秒", "reference_target_k": "本人首实测温度_K",
    "reference_prediction_k": "本人首实测预测温度_K", "target_delta_k": "首实测温升_K",
    "prediction_delta_k": "首实测预测温升_K", "error_c": "有符号温度误差_摄氏度",
    "delta_error_c": "有符号首实测差分误差_摄氏度", "source_row": "同序模态源行号",
}
VALUE_ZH = {
    "Top": "顶部", "Hot": "热环", "Cold": "冷环", "best": "观测最佳", "final": "训练末",
    "time_0_30_s": "[0,30]秒", "time_30_100_s": "(30,100]秒",
    "time_100_200_s": "(100,200]秒", "center_0_8_mm": "中心[0,8]毫米",
    "middle_8_17_mm": "中圈(8,17]毫米", "outer_17_25_mm": "外圈(17,25]毫米",
}


def _project_path(value: str | Path, root: Path, *, file: bool = True) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("任11观察来源与输出仅允许项目内路径")
    for path in (candidate, *candidate.parents):
        if path == root:
            break
        if path.is_symlink():
            raise ValueError("任11观察来源与输出不得经过符号链接")
    if file and not resolved.is_file():
        raise ValueError("任11观察来源必须是项目内普通文件")
    return resolved


def _digest(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("任11观察SHA256缺失或格式错误")
    return value


def _check_hash(path: Path, digest: str) -> str:
    if sha256_file(path) != _digest(digest):
        raise ValueError(f"任11观察原件或源码SHA漂移：{path.name}")
    return digest


def _finite_tree(value: Any) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_tree(item)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError("任11观察输出不得含NaN或无穷数值")


def _read_json(path: Path) -> Any:
    def reject_constant(value):
        raise ValueError(f"任11观察来源JSON含非有限数值：{value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("任11观察来源JSON有重复字段")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant,
                       object_pairs_hook=unique_keys)
    _finite_tree(value)
    return value


def _read_registry(path: Path) -> Any:
    class UniqueLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            self.flatten_mapping(node)
            result = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if key in result:
                    raise ValueError("任11观察YAML不能含重复字段")
                result[key] = self.construct_object(value_node, deep=deep)
            return result

    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
        _finite_tree(value)
        return value
    except (yaml.YAMLError, TypeError) as error:
        raise ValueError("任11观察YAML结构无法唯一解析") from error


def _strict_contract(actual: Any, expected: Any) -> bool:
    try:
        return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def _new_output(value: str | Path, root: Path) -> Path:
    destination = _project_path(value, root, file=False)
    if destination.parent != root / OUTPUT_BASE or destination.name in {
            "正式新MLP公平训练", "正式HowardLF公平训练", "正式Howard公平训练"}:
        raise ValueError("任11观察输出仅准外部对照全新直属目录，不得进入正式HF/LF树")
    if destination.exists():
        raise FileExistsError("任11观察已有输出目录不可覆盖或复用")
    return destination


def validate_task11_observation_model_catalog(rows: Any, root: Path) -> dict[tuple[int, str], dict[str, str]]:
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("任11观察须独立登记五seed各best/final十模型SHA")
    result = {}
    identities = [(seed, state) for seed in range(5) for state in STATES]
    for expected, row in zip(identities, rows):
        if (not isinstance(row, dict) or set(row) != {"种子", "模型状态", "模型原件", "模型SHA256"}
                or type(row["种子"]) is not int or (row["种子"], row["模型状态"]) != expected):
            raise ValueError("任11观察十状态须按seed0..4各best/final真实身份，无重复或缺席")
        seed, state = expected
        path = _project_path(row["模型原件"], root)
        if path != root / RUN_DIRECTORY / f"正式MLP_HF_seed{seed}" / f"{state}.pt":
            raise ValueError("任11观察模型只能为本人规范新MLP观测best或真final")
        result[expected] = {"模型原件": str(path), "模型SHA256": _check_hash(path, row["模型SHA256"])}
    return result


def _root_gate(root: Path, obs_yaml_sha: str, obs_tar_sha: str, hf_yaml_sha: str) -> dict[str, str]:
    ledger = _project_path(ROOT_LEDGER, root)
    whole_sha = sha256_file(ledger)
    expected = (f"{ROOT_TOKEN}; status=active; OBS_YAML_SHA256={_digest(obs_yaml_sha)}; "
                f"OBS_TAR_SHA256={_digest(obs_tar_sha)}; HF_YAML_SHA256={_digest(hf_yaml_sha)}")
    numbered, matches = [], []
    text = ledger.read_text(encoding="utf-8")
    if text.count(ROOT_TOKEN) != 1:
        raise ValueError("任11观察ROOT全文件须唯一新门禁，不能藏入缺号或其他单元格")
    fence: tuple[str, int] | None = None
    html_comment = False
    for line in text.splitlines():
        marker = re.fullmatch(r" {0,3}(`{3,}|~{3,})(.*)", line)
        if fence is not None:
            if (marker is not None and marker[1][0] == fence[0] and len(marker[1]) >= fence[1]
                    and not marker[2].strip()):
                fence = None
            continue
        # Table-like lines inside Markdown code examples are not ledger records.
        if not html_comment and marker is not None and (marker[1][0] == "~" or "`" not in marker[2]):
            fence = (marker[1][0], len(marker[1]))
            continue
        position, commented = 0, html_comment
        while True:
            delimiter = "-->" if html_comment else "<!--"
            index = line.find(delimiter, position)
            if index < 0:
                break
            commented = True
            html_comment = not html_comment
            position = index + len(delimiter)
        if commented:
            continue
        if re.fullmatch(r" {0,3}\|.*\|[ \t]*", line) is None:
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) >= 4 and re.fullmatch(r"录-\d{4}", fields[1]):
            numbered.append(fields[1])
            if ROOT_TOKEN in fields[2]:
                matches.append((fields[1], fields[2]))
    if (len(matches) != 1 or matches[0][1] != expected or int(matches[0][0][2:]) <= 105
            or numbered.count(matches[0][0]) != 1):
        raise ValueError("任11观察ROOT缺唯一全单格录号>105三SHA活动登记；禁止模型或标签读取")
    _check_hash(ledger, whole_sha)
    return {"ROOT记录号": matches[0][0], "ROOT原件": str(ledger), "ROOT整件SHA256": whole_sha,
            "ROOT唯一活动行": expected}


def preflight_task11_mlp_observations(
    *, observation_registry: str | Path, observation_registry_sha: str,
    observation_source_tar: str | Path, observation_source_tar_sha: str,
    output: str | Path, project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if root != PROJECT_ROOT.resolve() and not root.is_relative_to(PROJECT_ROOT.resolve() / OUTPUT_BASE):
        raise ValueError("任11观察替代根仅准项目内外部对照CPU合成夹具，禁止外部源根")
    registry = _project_path(observation_registry, root)
    archive_path = _project_path(observation_source_tar, root)
    if registry != root / OBS_REGISTRY_PATH or archive_path != root / OBS_TAR_PATH:
        raise ValueError("任11观察只准全新固定事前登记与观察源码归档，不继承旧门禁")
    _check_hash(registry, observation_registry_sha)
    _check_hash(archive_path, observation_source_tar_sha)
    gate = _root_gate(root, observation_registry_sha, observation_source_tar_sha,
                      HF_SOURCE_IDENTITY["registry"]["SHA256"])
    budget = _read_registry(registry)
    expected_fields = {"schema_version", "阶段", "ROOT门禁标签", "观察合同", "原HF四身份",
                       "模型十状态", "源码普通成员SHA256", "源码冻结tar原件", "源码冻结tarSHA256"}
    if (not isinstance(budget, dict) or set(budget) != expected_fields or type(budget["schema_version"]) is not int
            or budget["schema_version"] != 1
            or budget["阶段"] != "task11_new_mlp_validation_observations"
            or budget["ROOT门禁标签"] != ROOT_TOKEN or not _strict_contract(budget["观察合同"], OBSERVATION_CONTRACT)
            or budget["原HF四身份"] != HF_SOURCE_IDENTITY or budget["源码冻结tar原件"] != OBS_TAR_PATH
            or budget["源码冻结tarSHA256"] != observation_source_tar_sha
            or not isinstance(budget["源码普通成员SHA256"], dict)
            or set(budget["源码普通成员SHA256"]) != set(SOURCE_MEMBERS)):
        raise ValueError("任11观察登记不等于新MLP五seed十状态合法三模态固定只读合同")
    destination = _new_output(output, root)
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("任11观察只准单进程；不准多卡fallback")
    pins = {str(registry): observation_registry_sha, str(archive_path): observation_source_tar_sha,
            gate["ROOT原件"]: gate["ROOT整件SHA256"]}
    hf_arguments = {}
    for key, row in HF_SOURCE_IDENTITY.items():
        path = _project_path(row["原件"], root)
        pins[str(path)] = _check_hash(path, row["SHA256"])
        hf_arguments[key], hf_arguments[key + "_sha"] = str(path), row["SHA256"]
    source_hashes = budget["源码普通成员SHA256"]
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if (len(members) != len(SOURCE_MEMBERS) or {item.name for item in members} != set(SOURCE_MEMBERS)
                    or any(not item.isfile() for item in members)):
                raise ValueError("任11观察源码归档必须逐普通成员完整去重，无链接或额外工件")
            for member in members:
                path = _project_path(member.name, root)
                digest = _digest(source_hashes[member.name])
                stream = archive.extractfile(member)
                if stream is None or hashlib.sha256(stream.read()).hexdigest() != digest:
                    raise ValueError("任11观察源码归档与逐成员登记SHA漂移")
                pins[str(path)] = _check_hash(path, digest)
    except (OSError, tarfile.TarError) as error:
        raise ValueError("任11观察事前源码归档无法核验") from error
    # The running implementation must also be the registered bytes, not merely a copied tree.
    _check_hash(Path(__file__), source_hashes["src/sic_cu/eval/task11_mlp_observations.py"])
    models = validate_task11_observation_model_catalog(budget["模型十状态"], root)
    for model in models.values():
        pins[model["模型原件"]] = model["模型SHA256"]
    _check_hash(registry, observation_registry_sha)
    return {"状态": "CPU身份与源码前置核验；未读取模型温度或运行CUDA", **gate,
            "项目根": str(root), "输出": str(destination), "登记原件": str(registry),
            "OBS_YAML_SHA256": observation_registry_sha, "OBS_TAR_SHA256": observation_source_tar_sha,
            "HF_YAML_SHA256": HF_SOURCE_IDENTITY["registry"]["SHA256"],
            "观察预算": budget, "HF资格参数": hf_arguments, "模型十身份": models,
            "读取原件SHA256": pins}


def verify_task11_observation_runtime(prereg: Mapping[str, Any], additional_pins: Mapping[str, str]) -> None:
    root = Path(prereg["项目根"])
    if any(path in prereg["读取原件SHA256"] and prereg["读取原件SHA256"][path] != digest
           for path, digest in additional_pins.items()):
        raise ValueError("任11观察执行末哈希不能替代执行前冻结原件SHA")
    for path, digest in {**prereg["读取原件SHA256"], **additional_pins}.items():
        _check_hash(_project_path(path, root), digest)
    current = _root_gate(root, prereg["OBS_YAML_SHA256"], prereg["OBS_TAR_SHA256"],
                         prereg["HF_YAML_SHA256"])
    if any(current[key] != prereg[key] for key in current):
        raise ValueError("任11观察执行前后ROOT整件或唯一活动行发生漂移")
    _check_hash(Path(__file__), prereg["观察预算"]["源码普通成员SHA256"][
        "src/sic_cu/eval/task11_mlp_observations.py"])


def _vector(value: Any, name: str, size: int) -> np.ndarray:
    result = np.asarray(value)
    if (result.ndim != 1 or result.shape != (size,)
            or not (np.issubdtype(result.dtype, np.integer) or np.issubdtype(result.dtype, np.floating))):
        raise ValueError(f"任11观察{name}必须与实测逐点同序一维向量")
    result = result.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"任11观察{name}含非有限值")
    return result


@lru_cache(maxsize=1)
def _radial_helper():
    spec = importlib.util.spec_from_file_location(
        "sic_cu_task11_frozen_native_radial", PROJECT_ROOT / "scripts/30_audit_task07_radial_endpoint.py")
    if spec is None or spec.loader is None:
        raise ValueError("任11观察无法加载已声明冻结的原生径向纯数组接口")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.radial_masks


def native_radial_masks(radii_m: np.ndarray) -> tuple[np.ndarray, ...]:
    return _radial_helper()(radii_m, RADIAL_WINDOWS)[1]


def _frame_power_masks(frame: pl.DataFrame) -> dict[float, np.ndarray]:
    dtype = frame["power_w"].dtype
    if dtype not in (pl.Float32, pl.Float64):
        raise ValueError("任11观察功率列须为原生float32或可追溯重算float64")
    physical = _vector(frame["power_w"].to_numpy(), "原生功率", frame.height)
    represented = np.asarray(FIXED_POWERS, dtype=np.float32 if dtype == pl.Float32 else np.float64).astype(np.float64)
    if set(physical) != set(represented):
        raise ValueError("任11观察只准固定三功率的精确原生表示，不许邻近功率混入")
    return {power: physical == value for power, value in zip(FIXED_POWERS, represented)}


def _validate_frames(top: pl.DataFrame, sensors: pl.DataFrame) -> None:
    top_required = {"power_w", "time_s", "r_m", "temperature_mean_k", "frame_weight", "split"}
    sensor_required = {"power_w", "time_s", "r_m", "temperature_k", "delta_temperature_k", "sensor_type", "split"}
    if (not top_required.issubset(top.columns) or not sensor_required.issubset(sensors.columns)
            or top.height != 7272 or sensors.height != 752 or top["r_m"].dtype != pl.Float32
            or sensors["r_m"].dtype != pl.Float32):
        raise ValueError("任11观察原列schema或固定7272Top/752两环点数不一致")
    for frame in (top, sensors):
        if frame["split"].null_count() or set(frame["split"].to_list()) != {"validation"}:
            raise ValueError("任11观察仅准合法validation；257训练及TEST不得混入")
        for column in ("power_w", "time_s", "r_m"):
            values = _vector(frame[column].to_numpy(), column, frame.height)
            if column != "power_w" and (values < 0).any():
                raise ValueError("任11观察半径和时间不得为负")
        _frame_power_masks(frame)
        if (frame["time_s"] > 200).any():
            raise ValueError("任11观察固定合法三功率或0至200秒定义域不一致")
    if sensors["sensor_type"].null_count() or set(sensors["sensor_type"].to_list()) != {"hot", "cold"}:
        raise ValueError("任11观察必须分拆真实小写hot/cold类型，不能冒用预测掩码")
    if top.select("power_w", "time_s", "r_m").is_duplicated().any() or sensors.select(
            "power_w", "sensor_type", "time_s").is_duplicated().any():
        raise ValueError("任11观察真实观测点重复或同一曲线时序不唯一")
    top_order = top.select("power_w", "time_s", "r_m")
    if not top_order.equals(top_order.sort("power_w", "time_s", "r_m")):
        raise ValueError("任11观察Top源行须为本人功率/时刻/半径规范同序")
    for sensor_type in ("hot", "cold"):
        sensor_order = sensors.filter(pl.col("sensor_type") == sensor_type).select("power_w", "time_s")
        if not sensor_order.equals(sensor_order.sort("power_w", "time_s")):
            raise ValueError("任11观察每模态两环源行须为本人功率/时刻规范同序")
    _vector(top["temperature_mean_k"].to_numpy(), "Top实测温度", top.height)
    weights = _vector(top["frame_weight"].to_numpy(), "Top原帧权重", top.height)
    _vector(sensors["temperature_k"].to_numpy(), "两环实测温度", sensors.height)
    _vector(sensors["delta_temperature_k"].to_numpy(), "两环原差分温度", sensors.height)
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("任11观察权重须有限非负且和严格为正")
    top_masks, sensor_masks = _frame_power_masks(top), _frame_power_masks(sensors)
    for power in FIXED_POWERS:
        frame = top[np.flatnonzero(top_masks[power]).tolist()]
        if (frame.height, frame["time_s"].n_unique()) != TOP_COUNTS[power] or frame["r_m"].n_unique() != 101:
            raise ValueError("任11观察Top逐功率点数或实际帧数不等于冻结来源")
        if not all(count == 101 for count in frame.group_by("time_s").len()["len"].to_list()):
            raise ValueError("任11观察Top每帧须完整101原生半径")
        native_radial_masks(frame["r_m"].to_numpy())
        for sensor_type in ("hot", "cold"):
            curve = sensors[np.flatnonzero(sensor_masks[power]).tolist()].filter(pl.col("sensor_type") == sensor_type)
            if curve.is_empty() or curve["r_m"].n_unique() != 1:
                raise ValueError("任11观察每功率Hot/Cold须本人完整单半径真实曲线")


def _metrics(arrays: Mapping[str, np.ndarray], selected: np.ndarray, reason: str) -> dict[str, Any]:
    if not selected.any():
        return {**{name: None for name in METRIC_ZH}, "empty_reason": reason}
    absolute = error_statistics(arrays["target"][selected], arrays["prediction"][selected], arrays["weights"][selected])
    delta = error_statistics(arrays["target_delta"][selected], arrays["prediction_delta"][selected], arrays["weights"][selected])
    return {**absolute, **{"delta_" + key: value for key, value in delta.items()},
            "peak_error_c": float(np.max(arrays["prediction"][selected]) - np.max(arrays["target"][selected])),
            "empty_reason": None}


def evaluate_task11_observation_arrays(
    top: pl.DataFrame, sensors: pl.DataFrame, top_prediction: np.ndarray, sensor_prediction: np.ndarray,
    *, seed: int, state: str, bottom_z_m: float,
) -> dict[str, Any]:
    if type(seed) is not int or seed not in range(5) or state not in STATES:
        raise ValueError("任11观察仅准本人五seed的观测best和真实final")
    if type(bottom_z_m) not in (float, int) or not math.isfinite(bottom_z_m) or bottom_z_m != -0.0175:
        raise ValueError("任11观察两环必须使用本人冻结底面坐标-0.0175米")
    _validate_frames(top, sensors)
    top_prediction = _vector(top_prediction, "Top预测", top.height)
    sensor_prediction = _vector(sensor_prediction, "两环预测", sensors.height)
    result: dict[str, Any] = {"seed": seed, "state": state, "points": [], "power": [], "time": [], "radial": []}
    frames = [("Top", top, top_prediction)]
    for modality, sensor_type in (("Hot", "hot"), ("Cold", "cold")):
        mask = sensors["sensor_type"].to_numpy() == sensor_type
        frames.append((modality, sensors.filter(pl.col("sensor_type") == sensor_type), sensor_prediction[mask]))
    for modality, frame, predictions in frames:
        power_masks = _frame_power_masks(frame)
        for power in FIXED_POWERS:
            indices = np.flatnonzero(power_masks[power])
            group = frame[indices.tolist()]
            target = _vector(group["temperature_mean_k" if modality == "Top" else "temperature_k"].to_numpy(),
                             "实测温度", group.height)
            prediction = predictions[indices]
            times = _vector(group["time_s"].to_numpy(), "实际时刻", group.height)
            radii = group["r_m"].to_numpy()
            weights = (_vector(group["frame_weight"].to_numpy(), "原帧权重", group.height)
                       if modality == "Top" else np.ones(group.height))
            keys = radii if modality == "Top" else np.zeros(group.height, dtype=np.int8)
            target_delta, prediction_delta = _delta_from_first(target, prediction, times, keys)
            reference_indices = np.empty(group.height, dtype=np.int64)
            for key in np.unique(keys):
                selected = np.flatnonzero(keys == key)
                reference_indices[selected] = selected[np.argmin(times[selected])]
            if modality != "Top" and not np.allclose(target_delta, group["delta_temperature_k"].to_numpy(),
                                                       atol=1e-4, rtol=0):
                raise ValueError("任11观察两环原差分列不对应本人首真实实测时刻")
            arrays = {"target": target, "prediction": prediction, "times": times, "weights": weights,
                      "target_delta": target_delta, "prediction_delta": prediction_delta}
            identity = {"seed": seed, "state": state, "split": "validation", "power_w": power, "modality": modality}
            result["power"].append({**identity, "sample_count": group.height,
                                    "time_count": len(np.unique(times)),
                                    **_metrics(arrays, np.ones(group.height, dtype=bool), "")})
            temporal_masks = []
            for window in TIME_WINDOWS:
                lower = times >= window["lower"] if window["lower_closed"] else times > window["lower"]
                selected = lower & (times <= window["upper"])
                temporal_masks.append(selected)
                result["time"].append({**identity, "window": window["name"], "lower_s": window["lower"],
                                       "upper_s": window["upper"], "lower_closed": window["lower_closed"],
                                       "upper_closed": True, "sample_count": int(selected.sum()),
                                       **_metrics(arrays, selected, "预登记时间窗内无观测，不外推真值")})
            if not np.all(sum(mask.astype(np.int8) for mask in temporal_masks) == 1):
                raise ValueError("任11观察时间窗存在遗漏或双计")
            if modality == "Top":
                for radial, selected in zip(RADIAL_WINDOWS, native_radial_masks(radii)):
                    result["radial"].append({**identity, "radial_window": radial["name"], "lower_mm": radial["lower"],
                                             "upper_mm": radial["upper"], "lower_closed": radial["lower_closed"],
                                             "upper_closed": True, "sample_count": int(selected.sum()),
                                             **_metrics(arrays, selected, "预登记径向窗内无顶部观测，不外推真值")})
            for index, first in enumerate(reference_indices):
                result["points"].append({**identity, "r_m": float(radii[index]),
                                          "z_m": 0.0 if modality == "Top" else float(bottom_z_m),
                                          "time_s": float(times[index]), "material_id": 1 if modality == "Top" else 0,
                                          "target_k": float(target[index]), "prediction_k": float(prediction[index]),
                                          "weight": float(weights[index]), "reference_time_s": float(times[first]),
                                          "reference_target_k": float(target[first]),
                                          "reference_prediction_k": float(prediction[first]),
                                          "target_delta_k": float(target_delta[index]),
                                          "prediction_delta_k": float(prediction_delta[index]),
                                          "error_c": float(prediction[index] - target[index]),
                                          "delta_error_c": float(prediction_delta[index] - target_delta[index]),
                                          "source_row": int(indices[index])})
    _finite_tree(result)
    if [len(result[key]) for key in ("points", "power", "time", "radial")] != [8024, 9, 27, 9]:
        raise ValueError("任11观察逐点/逐功率/时间/径向覆盖计数不完整")
    return result


def task11_observation_model_from_view(view: Mapping[str, Any], *, seed: int) -> nn.Module:
    if type(seed) is not int or seed not in range(5) or type(view.get("seed")) is not int or view["seed"] != seed:
        raise ValueError("任11观察模型视图不属于本人真实seed")
    state = view.get("model_state")
    if (not isinstance(state, dict) or not state or any(not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32 or not torch.isfinite(value).all().item() for value in state.values())):
        raise ValueError("任11观察原MLP视图须全部为有限float32张量")
    first = state.get("correction.0.weight")
    if first is None or tuple(first.shape) != (128, 6) or view.get("surface_residual_guide_spec") is not None:
        raise ValueError("任11观察仅准冻结六输入新MLP校正器，禁止旧硬引导身份")
    try:
        model = task11_energy_model_from_view(view).float().eval()
    except RuntimeError as error:
        raise ValueError("任11观察原视图参数键或尺寸不符合严格MLP结构") from error
    if model.correction[0].in_features != 6:
        raise ValueError("任11观察校正器只能为冻结六输入结构")
    return model


def predict_task11_observations(model: nn.Module, coordinates: np.ndarray,
                                device: torch.device, batch_size: int) -> np.ndarray:
    values = np.asarray(coordinates)
    if (values.ndim != 2 or values.shape[1] != 5 or values.dtype != np.float32 or len(values) == 0
            or not np.isfinite(values).all() or type(batch_size) is not int or batch_size <= 0):
        raise ValueError("任11观察坐标须非空有限float32五列，批量须为正整数")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("任11观察不得复用能源double副本")
    model.eval()
    return _vector(_predict(model, values, device, batch_size), "模型逐点预测", len(values))


def reconstruct_task11_observation_macro(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    expected = {(power, modality) for power in FIXED_POWERS for modality in MODALITIES}
    if len(records) != 9 or {(row["power_w"], row["modality"]) for row in records} != expected:
        raise ValueError("任11原macro复核须本人全部九逐功率三模态记录")
    top = [float(row["rmse_c"]) for row in records if row["modality"] == "Top"]
    sensors = [row for row in records if row["modality"] != "Top"]
    modal = {"顶部": statistics.fmean(top),
             "absolute_rmse_c": statistics.fmean(float(row["rmse_c"]) for row in sensors),
             "absolute_mae_c": statistics.fmean(float(row["mae_c"]) for row in sensors),
             "delta_rmse_c": statistics.fmean(float(row["delta_rmse_c"]) for row in sensors),
             "delta_mae_c": statistics.fmean(float(row["delta_mae_c"]) for row in sensors)}
    score = (modal["顶部"] + 0.2 * modal["absolute_rmse_c"] + modal["delta_rmse_c"]) / 2.2
    result = {"合法HF原macro_v1_摄氏度": score, **modal}
    _finite_tree(result)
    return result


def _verify_macro(result: Mapping[str, Any], view: Mapping[str, Any]) -> dict[str, float]:
    reconstructed = reconstruct_task11_observation_macro(result["power"])
    expected = view.get("validation_selection_score_c")
    original_modalities = view.get("validation_sensor")
    if (type(expected) not in (int, float) or not math.isfinite(expected) or not isinstance(original_modalities, dict)
            or set(original_modalities) != {"顶部", "absolute_rmse_c", "absolute_mae_c", "delta_rmse_c", "delta_mae_c"}
            or not math.isclose(reconstructed["合法HF原macro_v1_摄氏度"], expected, rel_tol=0, abs_tol=1e-4)):
        raise ValueError("任11观察本人原macro与冻结best/final视图超过1e-4摄氏度")
    for key, value in original_modalities.items():
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not math.isclose(reconstructed[key], value, rel_tol=0, abs_tol=1e-4)):
            raise ValueError("任11观察本人分模态原macro与冻结视图不一致")
    return reconstructed


def five_seed_signed_statistics(values: Sequence[Any]) -> dict[str, Any]:
    if len(values) != 5 or any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("任11观察统计只准五个不同seed的有限原有符号数值")
    numeric = [float(value) for value in values]
    return {"样本数": 5, "ddof": 1, "均值": statistics.fmean(numeric),
            "样本标准差": statistics.stdev(numeric), "最小值": min(numeric),
            "最大值": max(numeric), "逐种子": numeric}


def _localize(row: Mapping[str, Any], fields: Mapping[str, str]) -> dict[str, Any]:
    if set(row) - set(fields):
        raise ValueError("任11观察中文导出列映射不完整")
    result = {fields[key]: value for key, value in row.items()}
    for key in ("modality", "state", "window", "radial_window"):
        if key in row:
            result[fields[key].replace("原标识", "中文名称")] = VALUE_ZH[row[key]]
    return result


def _group_statistics(results: Sequence[Mapping[str, Any]], kind: str) -> dict[str, Any]:
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        for row in result[kind]:
            parts = [str(row["power_w"]), row["modality"]]
            if kind == "time":
                parts.append(row["window"])
            elif kind == "radial":
                parts.append(row["radial_window"])
            by_group.setdefault("|".join(parts), []).append(row)
    summary = {}
    for key, rows in by_group.items():
        if len(rows) != 5 or [row["seed"] for row in rows] != list(range(5)):
            raise ValueError("任11观察各分组须五个本人seed无缺席无双计")
        record: dict[str, Any] = {"分组中文名称": "|".join(VALUE_ZH.get(part, part) for part in key.split("|")),
                                  "逐seed实测点数": [row["sample_count"] for row in rows]}
        for raw, chinese in METRIC_ZH.items():
            values = [row[raw] for row in rows]
            if all(value is None for value in values):
                record[chinese] = {"均值": None, "样本标准差": None, "最小值": None,
                                   "最大值": None, "逐种子": values, "样本数": 0, "ddof": 1,
                                   "原因": rows[0]["empty_reason"]}
            elif any(value is None for value in values):
                raise ValueError("任11观察同源固定窗口不能混有缺席seed")
            else:
                record[chinese] = five_seed_signed_statistics(values)
        summary[key] = record
    return summary


def _qualify_five(prereg: Mapping[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[str, str]]:
    root = Path(prereg["项目根"])
    qualified, pins = {}, {}
    path_fields = {"registry": ("登记原件", "登记SHA256"), "source_tar": ("源码归档原件", "源码归档SHA256"),
                   "lf_catalog": ("LF目录原件", "LF目录SHA256"), "hf_data_catalog": ("HF数据目录原件", "HF数据目录SHA256")}
    for seed in range(5):
        run = root / RUN_DIRECTORY / f"正式MLP_HF_seed{seed}"
        q = qualify_task11_mlp_hf_source(**prereg["HF资格参数"], output=run, seed=seed, project_root=root)
        if (q.get("状态") != "CPU完整HF来源资格PASS；独立能源仍须另行审核" or type(q.get("seed")) is not int
                or q["seed"] != seed or _project_path(q.get("目录", ""), root, file=False) != run
                or q.get("旧固定TEST温度读取") is not False or q.get("模拟测试功率温度读取") is not False):
            raise ValueError("任11观察五seed必须全部为本人真实完训完整HF CPU资格")
        for key, (path_field, sha_field) in path_fields.items():
            if (_project_path(q.get(path_field, ""), root) != Path(prereg["HF资格参数"][key])
                    or q.get(sha_field) != prereg["HF资格参数"][key + "_sha"]):
                raise ValueError("任11观察完整HF资格与固定原四身份不一致")
        originals = q.get("完整原件SHA256")
        if not isinstance(originals, dict) or not {"best.pt", "final.pt", "metrics.json"}.issubset(originals):
            raise ValueError("任11观察完整HF资格缺原模型与本人指标SHA")
        for relative, digest in originals.items():
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise ValueError("任11观察HF原件须本人目录内相对路径")
            path = _project_path(run / relative, root)
            if not path.is_relative_to(run):
                raise ValueError("任11观察不得借用其他HF目录原件")
            pins[str(path)] = _check_hash(path, digest)
        for state in STATES:
            model = prereg["模型十身份"][(seed, state)]
            if originals[state + ".pt"] != model["模型SHA256"]:
                raise ValueError("任11观察登记十模型SHA与原HF资格本人视图不一致")
        low_path = _project_path(q["本seed新LF最佳检查点原件"], root)
        pins[str(low_path)] = _check_hash(low_path, q["本seed新LF最佳检查点SHA256"])
        qualified[seed] = q
    lf_catalog = _read_json(Path(prereg["HF资格参数"]["lf_catalog"]))
    hf_catalog = _read_json(Path(prereg["HF资格参数"]["hf_data_catalog"]))
    rows = lf_catalog.get("五seed新LF") if isinstance(lf_catalog, dict) else None
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("任11观察原LF来源目录缺本人五seed完整来源")
    for seed, row in enumerate(rows):
        if type(row.get("seed")) is not int or row["seed"] != seed:
            raise ValueError("任11观察原LF来源目录种子错序")
        for relative, digest in row["真实原件SHA256"].items():
            path = _project_path(relative, root)
            pins[str(path)] = _check_hash(path, digest)
    if (not isinstance(hf_catalog, dict) or hf_catalog.get("HF合法验证功率_瓦") != list(FIXED_POWERS)
            or hf_catalog.get("旧固定TEST温度读取") is not False
            or hf_catalog.get("模拟测试功率温度读取") is not False
            or not isinstance(hf_catalog.get("HF源原件"), list)):
        raise ValueError("任11观察HF结构目录与固定三合法验证功率不一致")
    if {row["原件"] for row in hf_catalog["HF源原件"]} != {
            "data/processed/experiment_ir_radial.parquet", "data/processed/sensor_ring_raw.parquet"}:
        raise ValueError("任11观察HF结构目录只能是原开发Top与两环，不允许TEST")
    for row in hf_catalog["HF源原件"]:
        path = _project_path(row["原件"], root)
        pins[str(path)] = _check_hash(path, row["文件SHA256"])
    verify_task11_observation_runtime(prereg, pins)
    return qualified, pins


def _require_cuda(device_name: str) -> torch.device:
    if device_name != "cuda" or os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("任11正式观察只准单进程单卡CUDA，无CPU或多卡fallback")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("任11正式观察须恰好一张可见CUDA卡")
    return torch.device("cuda")


def _validate_results(results: Sequence[Mapping[str, Any]]) -> None:
    if len(results) != 10 or [(row["seed"], row["state"]) for row in results] != [
            (seed, state) for seed in range(5) for state in STATES]:
        raise ValueError("任11观察导出须本人五seed双状态十份同序完整结果")
    for result in results:
        if [len(result[key]) for key in ("points", "power", "time", "radial")] != [8024, 9, 27, 9]:
            raise ValueError("任11观察导出计数缺席或双计")
        if any(row["seed"] != result["seed"] or row["state"] != result["state"] or row["split"] != "validation"
               for kind in ("points", "power", "time", "radial") for row in result[kind]):
            raise ValueError("任11观察结果与本人seed/state/validation身份不一致")
        _finite_tree(result)
        if any(set(point) != set(POINT_ZH) for point in result["points"]):
            raise ValueError("任11观察逐点可追溯列schema不完整或含额外字段")
        for point in result["points"]:
            for name in set(POINT_ZH) - {"state", "split", "modality"}:
                if type(point[name]) not in (int, float) or not math.isfinite(point[name]):
                    raise ValueError("任11观察逐点数值列须有限原实数，不得混入布尔或对象")
            if type(point["source_row"]) is not int or point["source_row"] < 0 or point["weight"] < 0:
                raise ValueError("任11观察逐点源行号与权重须本人非负身份")
        top_points = [point for point in result["points"] if point["modality"] == "Top"]
        sensor_points = [point for point in result["points"] if point["modality"] in ("Hot", "Cold")]
        for modality in MODALITIES:
            modality_points = [point for point in result["points"] if point["modality"] == modality]
            if [point["source_row"] for point in modality_points] != list(range(len(modality_points))):
                raise ValueError("任11观察源行号须与本人每模态规范源行同序、范围内且完整双射")
        top = pl.DataFrame({
            "power_w": [point["power_w"] for point in top_points],
            "time_s": [point["time_s"] for point in top_points],
            "r_m": pl.Series([point["r_m"] for point in top_points], dtype=pl.Float32),
            "temperature_mean_k": pl.Series([point["target_k"] for point in top_points], dtype=pl.Float64),
            "frame_weight": pl.Series([point["weight"] for point in top_points], dtype=pl.Float64),
            "split": [point["split"] for point in top_points],
        })
        sensors = pl.DataFrame({
            "power_w": [point["power_w"] for point in sensor_points],
            "time_s": [point["time_s"] for point in sensor_points],
            "r_m": pl.Series([point["r_m"] for point in sensor_points], dtype=pl.Float32),
            "sensor_type": ["hot" if point["modality"] == "Hot" else "cold" for point in sensor_points],
            "temperature_k": pl.Series([point["target_k"] for point in sensor_points], dtype=pl.Float64),
            "delta_temperature_k": pl.Series([point["target_delta_k"] for point in sensor_points], dtype=pl.Float64),
            "split": [point["split"] for point in sensor_points],
        })
        regenerated = evaluate_task11_observation_arrays(
            top, sensors, np.asarray([point["prediction_k"] for point in top_points]),
            np.asarray([point["prediction_k"] for point in sensor_points]),
            seed=result["seed"], state=result["state"], bottom_z_m=-0.0175)
        for kind in ("power", "time", "radial"):
            if not _same_values(result[kind], regenerated[kind]):
                raise ValueError("任11观察逐点重算与逐功率/时间/原生径向指标不一致")
        for actual, expected in zip(result["points"], regenerated["points"]):
            if not _same_values(actual, expected):
                raise ValueError("任11观察逐点首实际时刻/材料/坐标/权重/差分与本人曲线不一致")
        if not _same_values(result.get("macro_reproduction"), reconstruct_task11_observation_macro(regenerated["power"])):
            raise ValueError("任11观察逐点重算本人原macro不一致")


def _same_values(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and set(actual) == set(expected) and all(
            _same_values(actual[key], value) for key, value in expected.items())
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
            _same_values(value, reference) for value, reference in zip(actual, expected))
    if type(expected) is float:
        return type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10)
    return type(actual) is type(expected) and actual == expected


def write_task11_observation_evidence(
    prereg: Mapping[str, Any], results: Sequence[Mapping[str, Any]],
    qualifications_before: Mapping[int, Mapping[str, Any]], qualifications_after: Mapping[int, Mapping[str, Any]],
    *, additional_pins: Mapping[str, str] | None = None, synthetic_cpu: bool = False,
) -> dict[str, Any]:
    """Public fixture serializer; formal evidence is written only by the exporter."""
    if synthetic_cpu is not True:
        raise ValueError("任11公开观察写出仅准显式CPU合成夹具；真实观察必须经正式CUDA导出入口")
    return _write_task11_observation_evidence(
        prereg, results, qualifications_before, qualifications_after,
        additional_pins=additional_pins, synthetic_cpu=True)


def _write_task11_observation_evidence(
    prereg: Mapping[str, Any], results: Sequence[Mapping[str, Any]],
    qualifications_before: Mapping[int, Mapping[str, Any]], qualifications_after: Mapping[int, Mapping[str, Any]],
    *, additional_pins: Mapping[str, str] | None = None, synthetic_cpu: bool = False,
) -> dict[str, Any]:
    pins = dict(additional_pins or {})
    verify_task11_observation_runtime(prereg, pins)
    if type(synthetic_cpu) is not bool:
        raise ValueError("任11观察CPU合成标记须显式布尔")
    root = Path(prereg["项目根"]).resolve()
    if synthetic_cpu and (root == PROJECT_ROOT.resolve()
                          or not root.is_relative_to(PROJECT_ROOT.resolve() / OUTPUT_BASE)):
        raise ValueError("CPU合成导出仅准项目内隔离夹具，不能写入正式项目或外部路径")
    if not synthetic_cpu and root != PROJECT_ROOT.resolve():
        raise ValueError("任11真实观察导出只准当前唯一正式项目根，不接受合成夹具根")
    _validate_results(results)
    if qualifications_before != qualifications_after or (not synthetic_cpu and set(qualifications_before) != set(range(5))):
        raise ValueError("任11观察前后本人五seed完整CPU资格或原件清单不相等")
    if synthetic_cpu and (qualifications_before or qualifications_after):
        raise ValueError("CPU合成导出不能伪装为真实模型完整资格")
    if not synthetic_cpu:
        current, current_pins = _qualify_five(prereg)
        if current != qualifications_before or current != qualifications_after:
            raise ValueError("任11真实观察写出前fresh五seed完整资格与预测前后原资格不相等")
        if any(path in pins and pins[path] != digest for path, digest in current_pins.items()):
            raise ValueError("任11真实观察写出末哈希不能替代预测前原件SHA")
        pins = {**current_pins, **pins}
        _require_cuda("cuda")
    destination = _new_output(prereg["输出"], root)
    statistics_by_state = {}
    for state in STATES:
        selected = [result for result in results if result["state"] == state]
        statistics_by_state[state] = {
            "模型状态中文名称": VALUE_ZH[state],
            "逐功率": _group_statistics(selected, "power"),
            "时间窗": _group_statistics(selected, "time"),
            "原生径向窗": _group_statistics(selected, "radial"),
            "本人原macro_v1_摄氏度": five_seed_signed_statistics([
                result["macro_reproduction"]["合法HF原macro_v1_摄氏度"] for result in selected]),
            "分模态宏平均": {
                modality: {chinese: five_seed_signed_statistics([
                    statistics.fmean(row[raw] for row in result["power"] if row["modality"] == modality)
                    for result in selected]) for raw, chinese in METRIC_ZH.items()}
                for modality in MODALITIES},
        }
    summary = {
        "状态": "CPU合成回归，非真实模型或实测验收" if synthetic_cpu else "新MLP五seed双状态合法实测三模态只读观察",
        "总点数": sum(len(result["points"]) for result in results),
        "验收计数": {"逐功率": 90, "时间窗": 270, "径向窗": 90},
        "观察合同": OBSERVATION_CONTRACT, "模型双状态五种子统计": statistics_by_state,
        "统计限定": {"ddof": 1, "独立试验单位": "五个本人模型seed；实测点不冒充独立试验",
                       "p95顺序": "先各seed各功率各模态或各窗计算未加权经验p95，再对五seed统计",
                       "模态宏平均": "三个固定验证功率等权；不是合并全部逐点p95",
                       "差分时间基准": "本人曲线首真实观测，不虚造t=0",
                       "有符号指标": "偏差和峰值均保留符号，不取绝对值", "空窗处理": "null与中文原因"},
        "限制": {"工程安全合格主张": False, "内部温度真值已证明": False,
                  "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
                  "训练或优化": False, "重新选模": False, "能源重新统计": False,
                  "训练成本重新统计": False, "修改B0部署": False,
                  "CPU合成回归": synthetic_cpu},
        "来源门禁": {key: prereg[key] for key in ("ROOT记录号", "ROOT原件", "ROOT整件SHA256",
                                                  "ROOT唯一活动行", "OBS_YAML_SHA256", "OBS_TAR_SHA256", "HF_YAML_SHA256")},
        "五seed本人完整来源资格": [{"种子": seed, "本人原件SHA256": q["完整原件SHA256"],
                                    "本人LF起点SHA256": q["本seed新LF最佳检查点SHA256"]}
                                   for seed, q in sorted(qualifications_before.items())],
        "读取原件执行前后SHA256": {**prereg["读取原件SHA256"], **pins},
        "中文列映射": {"逐点": POINT_ZH, "记录": FIELD_ZH},
    }
    _finite_tree(summary)
    content: dict[str, str] = {
        "机器汇总.json": json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "真实逐点预测.jsonl": "".join(json.dumps(_localize(point, POINT_ZH), ensure_ascii=False, allow_nan=False) + "\n"
                                     for result in results for point in result["points"]),
        "原始指标与中文映射.json": json.dumps({"中文列映射": FIELD_ZH,
            "十状态原指标": [{"种子": result["seed"], "模型状态": result["state"],
                               "原macro核验": result["macro_reproduction"],
                               "逐功率": [_localize(row, FIELD_ZH) for row in result["power"]],
                               "时间窗": [_localize(row, FIELD_ZH) for row in result["time"]],
                               "原生径向窗": [_localize(row, FIELD_ZH) for row in result["radial"]]}
                              for result in results]}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    }
    lines = ["# 任11新MLP三模态观察验收", "", summary["状态"], "",
             "固定validation三功率，五seed各观测最佳与训练末；十状态共80240点。",
             "逐功率90条、时间窗270条、原生顶部径向窗90条；时间与径向均无漏点无双计。",
             "Top使用本人原帧权重，Hot和Cold各自等点；每条曲线使用本人首真实实测，不虚造t=0。",
             "p95先各功率各模态各窗各seed算未加权经验分位，再按五seed统计；偏差和峰值保留符号。",
             "以下均值±样本标准差仅描述五个seed，ddof=1；不是工程容差或独立实测置信区间。", ""]
    for state in STATES:
        stat = statistics_by_state[state]["本人原macro_v1_摄氏度"]
        lines.append(f"{VALUE_ZH[state]}原macro_v1：{stat['均值']:.9f}±{stat['样本标准差']:.9f}摄氏度；"
                     f"原五值={stat['逐种子']}；范围=[{stat['最小值']:.9f},{stat['最大值']:.9f}]。")
    lines.extend(["", "空窗全部指标为null并保留中文原因，不补0，不外推。",
                  "原生径向修复仅支持已冻结同源Top的25毫米闭端点；不声称任意8或17毫米采样适用。",
                  "仅合法实测观测，不证明内部温度真值；未设工程安全阈值，不作安全合格判定。",
                  "不训练、不优化、不重新选模、不重新统计能源或训练成本，不修改B0部署。",
                  "旧固定TEST与模拟TEST温度未读取；本观察不扩大开发折。",
                  "真实模式写入前完整五seed CPU资格与所有读原件SHA前后相等。" if not synthetic_cpu
                  else "本目录仅CPU合成回归，未运行真实完整资格或真实CUDA，不代表正式观察完成。", ""])
    content["中文验收.md"] = "\n".join(lines)
    verify_task11_observation_runtime(prereg, pins)
    destination = _new_output(destination, root)
    destination.mkdir()
    for name, text in content.items():
        (destination / name).write_text(text, encoding="utf-8")
    for name, kind in (("逐功率三模态.csv", "power"), ("逐功率三模态时间窗.csv", "time"),
                       ("顶部逐功率原生径向窗.csv", "radial")):
        rows = [_localize(row, FIELD_ZH) for result in results for row in result[kind]]
        with (destination / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = {path.name: sha256_file(path) for path in sorted(destination.iterdir())}
    (destination / "工件SHA256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8")
    return summary


def export_task11_mlp_observations(
    *, observation_registry: str | Path, observation_registry_sha: str,
    observation_source_tar: str | Path, observation_source_tar_sha: str,
    output: str | Path, device_name: str = "cuda", project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    prereg = preflight_task11_mlp_observations(
        observation_registry=observation_registry, observation_registry_sha=observation_registry_sha,
        observation_source_tar=observation_source_tar, observation_source_tar_sha=observation_source_tar_sha,
        output=output, project_root=project_root)
    if Path(prereg["项目根"]) != PROJECT_ROOT.resolve() or device_name != "cuda":
        raise ValueError("任11真实观察仅准当前正式项目根的CUDA模式；无CPU或夹具绕行入口")
    qualified, pins = _qualify_five(prereg)
    device = _require_cuda(device_name)
    root = Path(prereg["项目根"])
    # Temperature columns are opened only after the new static gate and all five complete CPU qualifications.
    top = load_processed_ir_observations("validation").sort("power_w", "time_s", "r_m")
    metadata = _project_path("configs/data_metadata.yaml", root)
    pins[str(metadata)] = sha256_file(metadata)
    sensors = load_canonical_sensor_observations(
        metadata_path=str(metadata), processed_path=str(root / "data/processed/sensor_ring_raw.parquet"),
        split="validation").sort("power_w", "sensor_type", "time_s")
    _validate_frames(top, sensors)
    results = []
    for seed in range(5):
        geometry_path = _project_path(Path(qualified[seed]["config_snapshot目录"]) / "geometry.yaml", root)
        pins[str(geometry_path)] = sha256_file(geometry_path)
        bottom = float(load_yaml(geometry_path)["embedding"]["copper_bottom_z_m"])
        top_coordinates = np.column_stack((top["r_m"].to_numpy(), np.zeros(top.height), top["time_s"].to_numpy(),
                                            top["power_w"].to_numpy(), np.ones(top.height))).astype(np.float32)
        sensor_coordinates = np.column_stack((sensors["r_m"].to_numpy(), np.full(sensors.height, bottom),
                                               sensors["time_s"].to_numpy(), sensors["power_w"].to_numpy(),
                                               np.zeros(sensors.height))).astype(np.float32)
        for state in STATES:
            model_row = prereg["模型十身份"][(seed, state)]
            path = _project_path(model_row["模型原件"], root)
            _check_hash(path, model_row["模型SHA256"])
            view = torch.load(path, map_location="cpu", weights_only=True)
            model = task11_observation_model_from_view(view, seed=seed).to(device).eval()
            top_prediction = predict_task11_observations(model, top_coordinates, device, 8192)
            sensor_prediction = predict_task11_observations(model, sensor_coordinates, device, 8192)
            result = evaluate_task11_observation_arrays(top, sensors, top_prediction, sensor_prediction,
                                                        seed=seed, state=state, bottom_z_m=bottom)
            result["macro_reproduction"] = _verify_macro(result, view)
            _check_hash(path, model_row["模型SHA256"])
            results.append(result)
            del model, view
    after, after_pins = _qualify_five(prereg)
    if after != qualified or any(after_pins.get(path) != digest for path, digest in pins.items()
                                  if path in after_pins):
        raise ValueError("任11观察预测前后完整CPU资格或本人原件SHA不相等")
    return _write_task11_observation_evidence(prereg, results, qualified, after, additional_pins=pins)
