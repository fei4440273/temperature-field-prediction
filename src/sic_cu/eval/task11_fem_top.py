"""LF-trained FEM power interpolation: limited, deterministic HF Top comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.data.processed import load_processed_ir_observations, processed_ir_path
from sic_cu.data.splits import build_power_splits
from sic_cu.eval.development_v4 import _observation_arrays, _power_record
from sic_cu.train.surface_residual import _lf_surface


REGISTRATION_FILE = "只读LF_FEM事前来源SHA256.json"
IMPLEMENTATION = PROJECT_ROOT / "src/sic_cu/eval/task11_fem_top.py"
CLI = PROJECT_ROOT / "scripts/36_audit_task11_fem_top.py"
TESTS = PROJECT_ROOT / "tests/test_task11_fem_top_validation.py"


def project_output(path: str | Path) -> Path:
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()
    if PROJECT_ROOT.resolve() not in resolved.parents:
        raise ValueError("任11所有新登记、结果与临时工件只能写项目内严格子目录")
    return resolved


def require_lf_train_only(powers: Iterable[float]) -> tuple[float, ...]:
    actual = tuple(round(float(value), 4) for value in powers)
    expected = tuple(sorted(build_power_splits().simulation_train))
    if actual != expected:
        raise ValueError("任11 FEM功率插值支点只能是原LF60训练功率，不许验证/测试或缺支点")
    return actual


def _source_identity() -> dict[str, object]:
    supports = require_lf_train_only(sorted(build_power_splits().simulation_train))
    sources = {
        f"{power:g}W.parquet": sha256_file(
            PROJECT_ROOT / "data/processed/simulation" / f"{power:g}W.parquet")
        for power in supports
    }
    return {
        "登记性质": "仅LF60训练支点；合法HF验证Top有限对照，不参与三模态综合排名",
        "训练LF支点功率_W": list(supports),
        "处理LF训练场逐原件SHA256": sources,
        "合法HF开发IR原件SHA256": sha256_file(processed_ir_path("validation")),
        "冻结划分SHA256": sha256_file(PROJECT_ROOT / "configs/splits.yaml"),
        "程序与TDD事前SHA256": {
            "FEM训练支点插值核心": sha256_file(PROJECT_ROOT / "src/sic_cu/models/interpolation.py"),
            "LF原件场读取": sha256_file(PROJECT_ROOT / "src/sic_cu/data/fields.py"),
            "FEM观测位置提取": sha256_file(PROJECT_ROOT / "src/sic_cu/train/surface_residual.py"),
            "限界评价": sha256_file(IMPLEMENTATION),
            "CLI": sha256_file(CLI), "TDD": sha256_file(TESTS),
        },
        "禁止旧test_Data温度标签读取": True,
    }


def register_lf_fem_sources(output_directory: str | Path) -> dict[str, object]:
    output = project_output(output_directory)
    if output.exists():
        raise FileExistsError("任11新的事前来源登记目标已存在，绝不覆盖")
    output.mkdir(parents=True)
    identity = _source_identity()
    (output / REGISTRATION_FILE).write_text(
        json.dumps(identity, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"登记路径": str(output / REGISTRATION_FILE),
            "登记SHA256": sha256_file(output / REGISTRATION_FILE),
            "LF训练功率数": len(identity["训练LF支点功率_W"])}


def evaluate_lf_fem_top(frame: pl.DataFrame, lf_train_powers: Iterable[float]) -> list[dict[str, object]]:
    supports = require_lf_train_only(lf_train_powers)
    required = {"split", "power_w", "r_m", "time_s", "temperature_mean_k", "frame_weight"}
    if not required <= set(frame.columns) or frame.is_empty() or set(frame["split"]) != {"validation"}:
        raise ValueError("任11只准读取完整合法HF验证Top观察，不读旧测试标签")
    expected = set(build_power_splits().hf_validation)
    available = {round(float(value), 4) for value in frame["power_w"].unique()}
    if available != expected:
        raise ValueError("任11合法HF验证三个固定功率必须完整，不能挑误差好看的功率")
    rows = []
    for power in sorted(expected):
        group = frame.filter((pl.col("power_w") - power).abs() < 1e-3).sort("time_s", "r_m")
        if group.is_empty():
            raise ValueError("任11合法HF验证Top功率无真实观测点")
        prediction = _lf_surface(group, list(supports))
        if len(prediction) != group.height or not np.isfinite(prediction).all():
            raise ValueError("任11 LF FEM投影到合法HF顶部的真实预测非完整有限值")
        arrays = _observation_arrays(group, prediction, "Top")
        record = _power_record(seed=-1, split="validation", power_w=power,
                               modality="Top", arrays=arrays)
        record.pop("seed")
        record["对照性质"] = "无HF拟合、仅LF训练支点的确定性FEM顶部基线"
        rows.append(record)
    return rows


def run_lf_fem_top_validation(
    output_directory: str | Path, registration_file: str | Path,
    registration_sha256: str,
) -> dict[str, object]:
    output, registered = project_output(output_directory), project_output(registration_file)
    if output.exists() or output == registered.parent or registered.parent in output.parents:
        raise FileExistsError("任11不得覆盖或在来源登记目录中嵌套写评估结果")
    if not registered.is_file() or registered.name != REGISTRATION_FILE:
        raise ValueError("任11必须先封独立来源登记再访问合法HF温度")
    if len(registration_sha256) != 64 or sha256_file(registered) != registration_sha256:
        raise ValueError("任11实际字节事前登记SHA与请求不符")
    identity = json.loads(registered.read_text(encoding="utf-8"))
    if identity != _source_identity():
        raise ValueError("任11事前源码/HF处理IR或任一LF训练场温度原件已漂移")
    frame = load_processed_ir_observations("validation")
    rows = evaluate_lf_fem_top(frame, identity["训练LF支点功率_W"])
    output.mkdir(parents=True)
    pl.DataFrame(rows).write_csv(output / "合法HF验证_顶部逐功率LF_FEM插值.csv")
    payload: dict[str, object] = {
        "状态": "任11仅合法HF验证Top的无HF拟合LF FEM基线；不参与三模态宏选择分或内部HF场排名",
        "逐功率顶部RMSE_摄氏度": {f"{row['power_w']:g}": row["rmse_c"] for row in rows},
        "训练LF支点数": 60, "合法HF验证顶部功率数": len(rows),
        "合法HF验证顶部观察行数": sum(int(row["sample_count"]) for row in rows),
        "五种子学习效果": None, "HotCold两环LF_FEM验证": None,
        "独立HF内部实测真值": None,
        "事前LF_FEM来源SHA256": registration_sha256,
        "旧test_Data温度标签读取": False,
    }
    (output / "受限对照与缺项摘要.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    payload["输出工件SHA256"] = {file.name: sha256_file(file)
                             for file in output.iterdir() if file.is_file()}
    return payload
