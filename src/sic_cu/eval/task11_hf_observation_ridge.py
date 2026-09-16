"""HF-observation-only Ridge: deterministic, location-limited engineering baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.data.processed import load_processed_ir_observations, processed_ir_path
from sic_cu.data.sensors import load_canonical_sensor_observations
from sic_cu.data.splits import build_power_splits
from sic_cu.eval.development_v4 import _observation_arrays, _power_record
from sic_cu.eval.metrics import macro_v1_selection_score


REGISTRATION_FILE = "仅HF观测Ridge事前来源与固定系数.json"
MODEL_FILE = "仅HF训练_已锁Ridge系数.json"
DEGREE = 2
ALPHA = 0.01
MODULE = PROJECT_ROOT / "src/sic_cu/eval/task11_hf_observation_ridge.py"
CLI = PROJECT_ROOT / "scripts/37_run_task11_hf_only_observation_ridge.py"
TEST = PROJECT_ROOT / "tests/test_task11_hf_observation_ridge.py"


def project_output(path: str | Path) -> Path:
    path = Path(path)
    resolved = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if PROJECT_ROOT.resolve() not in resolved.parents:
        raise ValueError("任11仅HF观测回归的登记/训练/验证只能写项目内严格子目录")
    return resolved


def _validate_frames(top: pl.DataFrame, rings: pl.DataFrame, split: str) -> None:
    top_columns = {"split", "power_w", "r_m", "time_s", "temperature_mean_k", "frame_weight"}
    ring_columns = {"split", "power_w", "r_m", "time_s", "temperature_k", "sensor_type"}
    expected = build_power_splits().hf_train if split == "train" else build_power_splits().hf_validation
    if (not top_columns <= set(top.columns) or not ring_columns <= set(rings.columns)
        or top.is_empty() or rings.is_empty()
        or set(top["split"]) != {split} or set(rings["split"]) != {split}
        or {round(float(x), 4) for x in top["power_w"].unique()} != expected):
        raise ValueError(f"任11仅HF观测须完整{split}顶部功率/训练或验证同折，不能筛误差功率")
    for kind in ("hot", "cold"):
        selected = rings.filter(pl.col("sensor_type") == kind)
        if selected.is_empty() or {round(float(x), 4) for x in selected["power_w"].unique()} != expected:
            raise ValueError(f"任11仅HF观测{split}缺{kind}完整Hot/Cold环温功率")
    if set(rings["sensor_type"]) != {"hot", "cold"}:
        raise ValueError("任11HF观测只能消费合法Hot/Cold两环，不准伪模态")


def validate_training_observations(top: pl.DataFrame, rings: pl.DataFrame) -> None:
    _validate_frames(top, rings, "train")


def _features(frame: pl.DataFrame, modality: str) -> np.ndarray:
    if modality not in ("Top", "Hot", "Cold"):
        raise ValueError("任11仅HF观测最多Top/Hot/Cold三个观察位置")
    coordinates = frame.select("r_m", "time_s", "power_w").to_numpy().astype(np.float64)
    return np.column_stack((
        coordinates[:, 0] / 0.05834, coordinates[:, 1] / 200.0,
        coordinates[:, 2] / 800.0,
        np.full(frame.height, float(modality == "Hot")),
        np.full(frame.height, float(modality == "Cold")),
    ))


def _observations(top: pl.DataFrame, rings: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs, targets, weights = [], [], []
    powers = sorted({round(float(x), 4) for x in top["power_w"].unique()})
    for modality, frame, column in (
        ("Top", top, "temperature_mean_k"),
        ("Hot", rings.filter(pl.col("sensor_type") == "hot"), "temperature_k"),
        ("Cold", rings.filter(pl.col("sensor_type") == "cold"), "temperature_k"),
    ):
        for power in powers:
            group = frame.filter((pl.col("power_w") - power).abs() < 1e-3)
            if group.is_empty():
                raise ValueError("任11 HF-only观察须三模态每训练功率都有原真实行")
            source_weight = (group["frame_weight"].to_numpy().astype(np.float64)
                             if modality == "Top" else np.ones(group.height))
            if not np.isfinite(source_weight).all() or source_weight.sum() <= 0:
                raise ValueError("HF训练真实顶部帧/两环权重非有效")
            inputs.append(_features(group, modality))
            targets.append(group[column].to_numpy().astype(np.float64) - 295.15)
            weights.append(source_weight / source_weight.sum())
    return np.concatenate(inputs), np.concatenate(targets), np.concatenate(weights)


def fit_hf_observation_ridge(
    top: pl.DataFrame, rings: pl.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_training_observations(top, rings)
    x, y, w = _observations(top, rings)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or len(w) != len(x):
        raise ValueError("任11HF-only训练特征与真实温度须有限完整")
    transform = PolynomialFeatures(degree=DEGREE, include_bias=False)
    features = transform.fit_transform(x)
    model = Ridge(alpha=ALPHA, fit_intercept=True, solver="cholesky")
    model.fit(features, y, sample_weight=w)
    state = {
        "基线资格": "仅HF训练观测的确定性Ridge；无PINN物理或内部完整场",
        "degree": DEGREE, "alpha": ALPHA,
        "sklearn版本": sklearn.__version__, "特征维数": int(features.shape[1]),
        "coef": model.coef_.tolist(), "intercept": float(model.intercept_),
        "实际HF训练功率": sorted(build_power_splits().hf_train),
        "LF训练仿真标签消费": 0,
        "旧test_Data温度标签读取": False,
    }
    return state, {
        "顶部HF训练功率数": 12, "顶部HF训练行": top.height,
        "两环HF训练行": rings.height, "LF训练仿真标签消费": 0,
        "物理PDE/边界/界面训练梯度": 0,
        "方法": "固定二次多项式Ridge、alpha=0.01；按功率/模态等总权重，不用合法验证挑超参数",
    }


def _predict(state: Mapping[str, Any], group: pl.DataFrame, modality: str) -> np.ndarray:
    if (state.get("degree") != DEGREE or state.get("alpha") != ALPHA or
        state.get("sklearn版本") != sklearn.__version__ or
        state.get("LF训练仿真标签消费") != 0):
        raise ValueError("任11仅HF训练回归模型固定配置或版本来源漂移")
    phi = PolynomialFeatures(degree=DEGREE, include_bias=False).fit_transform(
        _features(group, modality))
    coef = np.asarray(state.get("coef"), dtype=np.float64)
    if phi.shape[1] != state.get("特征维数") or coef.shape != (phi.shape[1],):
        raise ValueError("任11已锁仅HF回归特征数/系数不一致")
    predictions = 295.15 + phi @ coef + float(state["intercept"])
    if not np.isfinite(predictions).all():
        raise ValueError("任11仅HF观测真预测非有限")
    return predictions


def evaluate_hf_observation_ridge(
    state: Mapping[str, Any], top: pl.DataFrame, rings: pl.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_frames(top, rings, "validation")
    by_power = []
    for modality, frame in (
        ("Top", top), ("Hot", rings.filter(pl.col("sensor_type") == "hot")),
        ("Cold", rings.filter(pl.col("sensor_type") == "cold")),
    ):
        for power in sorted(build_power_splits().hf_validation):
            group = frame.filter((pl.col("power_w") - power).abs() < 1e-3).sort("time_s", "r_m")
            prediction = _predict(state, group, modality)
            record = _power_record(seed=-1, split="validation", power_w=power,
                                   modality=modality,
                                   arrays=_observation_arrays(group, prediction, modality))
            record.pop("seed")
            record["对照性质"] = "只拟合HF训练Top/Hot/Cold位置；不具HF内部场/物理训练合同"
            by_power.append(record)
    top_scores = [row["rmse_c"] for row in by_power if row["modality"] == "Top"]
    absolute = [row["rmse_c"] for row in by_power if row["modality"] in ("Hot", "Cold")]
    deltas = [row["delta_rmse_c"] for row in by_power if row["modality"] in ("Hot", "Cold")]
    summary = {
        "合法HF三模态宏选分_摄氏度": macro_v1_selection_score(
            float(np.mean(top_scores)), float(np.mean(absolute)), float(np.mean(deltas))),
        "逐模态合法HF验证RMSE均值_摄氏度": {
            modality: float(np.mean([row["rmse_c"] for row in by_power
                                     if row["modality"] == modality]))
            for modality in ("Top", "Hot", "Cold")},
        "顶部验证点": sum(row["sample_count"] for row in by_power if row["modality"] == "Top"),
        "两环验证点": sum(row["sample_count"] for row in by_power if row["modality"] != "Top"),
        "验证逐功率逐模态行数": 9, "完整HF内部预测资格": False,
        "F1纯HF物理PINN资格": False, "旧test_Data温度标签读取": False,
    }
    return by_power, summary


def _source_identity() -> dict[str, Any]:
    return {
        "固定方法": "HF训练观测位置二次多项式Ridge；不读LF，不声称完整体场",
        "degree": DEGREE, "alpha": ALPHA, "sklearn版本": sklearn.__version__,
        "HF训练顶部IR字节SHA256": sha256_file(processed_ir_path("train")),
        "HF训练两环字节SHA256": sha256_file(PROJECT_ROOT / "data/processed/sensor_ring_raw.parquet"),
        "固定功率划分SHA256": sha256_file(PROJECT_ROOT / "configs/splits.yaml"),
        "两环单位/时间语义SHA256": sha256_file(PROJECT_ROOT / "configs/data_metadata.yaml"),
        "固定程序与测试逐字SHA256": {
            "训练和合法评价": sha256_file(MODULE), "分段CLI": sha256_file(CLI),
            "拒绝泄漏TDD": sha256_file(TEST),
            "正规HotCold读取": sha256_file(PROJECT_ROOT / "src/sic_cu/data/sensors.py"),
            "Top及指标定义": sha256_file(PROJECT_ROOT / "src/sic_cu/eval/development_v4.py"),
        },
        "旧test_Data温度标签读取": False,
    }


def register_hf_ridge_sources(directory: str | Path) -> dict[str, Any]:
    directory = project_output(directory)
    if directory.exists():
        raise FileExistsError("任11仅HF-only来源登记已存在，绝不覆盖")
    identity = _source_identity()
    directory.mkdir(parents=True)
    file = directory / REGISTRATION_FILE
    file.write_text(json.dumps(identity, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return {"来源文件": str(file), "来源文件SHA256": sha256_file(file)}


def _registered(path: str | Path, digest: str) -> None:
    path = project_output(path)
    if not path.is_file() or path.name != REGISTRATION_FILE or sha256_file(path) != digest:
        raise ValueError("任11先封仅HF训练来源字节SHA再访问温度")
    if json.loads(path.read_text(encoding="utf-8")) != _source_identity():
        raise ValueError("任11登记的HF温度、两环单位、源码/固定配置字节出现漂移")


def train_registered_hf_ridge(
    directory: str | Path, registration_file: str | Path, registration_sha256: str,
) -> dict[str, Any]:
    directory = project_output(directory)
    if directory.exists():
        raise FileExistsError("任11仅HF观测模型训练目录已有用户原件，不能覆盖")
    _registered(registration_file, registration_sha256)
    top = load_processed_ir_observations("train")
    rings = load_canonical_sensor_observations(split="train")
    if top.height != 29593 or rings.height != 2985:
        raise ValueError("任11HF训练Top29593/两环2985个真实点不完整")
    state, receipt = fit_hf_observation_ridge(top, rings)
    state["事前来源SHA256"] = registration_sha256
    directory.mkdir(parents=True)
    model = directory / MODEL_FILE
    model.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                     encoding="utf-8")
    receipt["模型锁定SHA256"] = sha256_file(model)
    receipt["事前来源SHA256"] = registration_sha256
    (directory / "真HF训练预算与来源.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt | {"已封模型": str(model)}


def evaluate_registered_hf_ridge(
    directory: str | Path, registration_file: str | Path, registration_sha256: str,
    model_file: str | Path, model_sha256: str,
) -> dict[str, Any]:
    directory = project_output(directory)
    model_file = project_output(model_file)
    if directory.exists() or directory == model_file.parent or model_file.parent in directory.parents:
        raise FileExistsError("任11验证不能覆盖或嵌入同一HF模型原件目录")
    _registered(registration_file, registration_sha256)
    if (not model_file.is_file() or model_file.name != MODEL_FILE
        or sha256_file(model_file) != model_sha256):
        raise ValueError("任11须先锁真HF训练完整Ridge系数字节SHA才可读合法验证温度")
    state = json.loads(model_file.read_text(encoding="utf-8"))
    if state.get("事前来源SHA256") != registration_sha256:
        raise ValueError("任11回归模型并非此份原HF训练来源")
    top = load_processed_ir_observations("validation")
    rings = load_canonical_sensor_observations(split="validation")
    if top.height != 7272 or rings.height != 752:
        raise ValueError("任11HF合法验证Top7272/两环752真实行不完整")
    records, summary = evaluate_hf_observation_ridge(state, top, rings)
    directory.mkdir(parents=True)
    pl.DataFrame(records).write_csv(directory / "逐功率三模态合法HF验证.csv")
    summary["模型训练预算"] = "仅观察位置的解析回归，无HF PDE或LF仿真成本；与DeepONet不同"
    summary["事前原件SHA256"] = registration_sha256
    summary["冻结模型SHA256"] = model_sha256
    (directory / "独立有限工程对照与缺项.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary | {"原件SHA256": {file.name: sha256_file(file)
                             for file in directory.iterdir() if file.is_file()}}
