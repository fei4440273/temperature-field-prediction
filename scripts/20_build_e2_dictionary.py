#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import yaml

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.data.fields import load_processed_field
from sic_cu.data.splits import build_power_splits
from sic_cu.data.time_dictionary import (
    build_curve_matrix, fit_time_dictionary, project_fixed_dictionary,
    select_spatial_nodes, validate_training_power_contract,
)


REGISTRATION = PROJECT_ROOT / "研究记录/任务06_时间响应特征/字典拟合前登记.yaml"
ROOT_OUTPUT = PROJECT_ROOT / "研究记录/任务06_时间响应特征"
INITIAL_GROUPS = ((2.0, 10.0, 50.0, 200.0), (1.0, 5.0, 25.0, 100.0))
# This file hash predates the first LF fit; a replacement requires new provenance and ledger approval.
REGISTERED_SHA256 = "22b9a7a762d449ab083d7cbaed447d5c668761d063a94581b2e24635ae961cb5"


def validate_registration(policy: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    source = policy["source"]
    nodes = policy["node_selection"]
    curves = policy["curve_weighting"]
    dictionary = policy["dictionary"]
    initial = tuple(float(x) for x in dictionary["tau_seconds_initial_group_1"])
    second = tuple(float(x) for x in dictionary["tau_seconds_initial_group_2"])
    if (initial, second) != INITIAL_GROUPS:
        raise ValueError("Two registered initial tau groups must not be changed after pre-fit registration")
    if (policy["schema_version"] != 1 or source["split_config"] != "configs/splits.yaml"
            or source["split_key"] != "simulation.training_powers_w"
            or source["trajectories"] != "data/processed/simulation/{power}W.parquet"
            or source["power_count"] != 60 or nodes["per_material"] != 32
            or dictionary["modes"] != 4 or dictionary["dtype"] != "float64"
            or float(curves["ridge_lambda"]) != 1e-6):
        raise ValueError("Registered LF training source, spatial count, float64 four-mode budget or ridge changed")
    return initial, second


def load_locked_registration(path: Path) -> tuple[dict, tuple[tuple[float, ...], tuple[float, ...]]]:
    if path != REGISTRATION:
        raise ValueError(f"Registration path differs from the registered pre-fit source: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    original_bytes = path.read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != REGISTERED_SHA256:
        raise ValueError("Registration SHA differs from the full pre-fit LF dictionary source")
    policy = yaml.safe_load(original_bytes.decode("utf-8"))
    return policy, validate_registration(policy)


def create_fresh_output(path: Path) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Existing dictionary output is never overwritten: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _source_fields(powers: list[float], split: str):
    fields = []
    sources = []
    for power in powers:
        path = PROJECT_ROOT / f"data/processed/simulation/{power:g}W.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata = pl.read_parquet(path, columns=["power_w", "split"])
        assigned = metadata["split"].unique().to_list()
        parquet_powers = metadata["power_w"].unique().to_list()
        if assigned != [split] or len(parquet_powers) != 1 or abs(parquet_powers[0] - power) > 1e-4:
            raise ValueError(f"Wrong LF source split or power in {path}: {assigned}, {parquet_powers}")
        fields.append(load_processed_field(power))
        sources.append({"功率_W": power, "相对路径": str(path.relative_to(PROJECT_ROOT)),
                        "SHA256": sha256_file(path), "parquet_split": split})
    return fields, sources


def _candidate_row(index: int, candidate) -> dict:
    return {
        "注册初值编号": index + 1,
        "注册初值_tau秒": list(candidate.initial_tau_seconds),
        "拟合_tau秒": [float(value) for value in candidate.tau_seconds],
        "LF训练归一化RMSE": candidate.training_normalized_rmse,
        "岭正则目标函数": candidate.regularized_objective,
        "优化器收敛": candidate.converged,
        "优化器消息": candidate.optimizer_message,
        "迭代数": candidate.iterations,
        "函数计算数": candidate.function_evaluations,
        "响应特征条件数": _finite(candidate.feature_condition_number),
        "带岭法方程条件数": _finite(candidate.ridge_condition_number),
    }


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _material_power_rows(matrix, tau, powers: list[float], node_count: int):
    diagnostics = project_fixed_dictionary(matrix.normalized_deltas, matrix.times_s, tau)
    residual = np.asarray(diagnostics["residual"]).reshape(len(powers), 2, node_count, 100)
    rows = []
    for p, power in enumerate(powers):
        for material, name in enumerate(("铜", "碳化硅")):
            rows.append({"功率_W": power, "材料": name, "节点数": node_count,
                         "归一化RMSE": float(np.sqrt(np.mean(residual[p, material] ** 2)))})
    return rows


def _save_group_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("功率_W", "材料", "节点数", "归一化RMSE"))
        writer.writeheader()
        writer.writerows(rows)


def _amplitude_diagnostic(normalized_amplitudes, scales, times, tau):
    values_k = normalized_amplitudes * scales[:, None]
    terminal = -np.expm1(-float(times[-1]) / tau)
    modal_terminal = values_k * terminal[None, :]
    cancellation = (np.sum(np.abs(modal_terminal), axis=1)
                    / (np.abs(np.sum(modal_terminal, axis=1)) + 1e-6))
    feature_terminal = terminal.tolist()
    return {
        "训练幅值最小_K": float(values_k.min()),
        "训练幅值最大_K": float(values_k.max()),
        "负幅值比例": float(np.mean(values_k < 0)),
        "幅值末时刻补偿比_中位数": float(np.median(cancellation)),
        "幅值末时刻补偿比_95分位": float(np.percentile(cancellation, 95)),
        "200秒响应特征": feature_terminal,
        "200秒超过0点95的饱和模式数": int(np.sum(terminal >= 0.95)),
        "诊断说明": "幅值允许正负且有补偿；病态或饱和时tau不解释为唯一真实热模态。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="任06：仅LF训练源的四模态时间字典与LF冻结验证诊断")
    parser.add_argument("--registration", default=str(REGISTRATION))
    parser.add_argument("--output", help="新的独占工件目录；已有目录拒绝覆盖")
    args = parser.parse_args()
    destination = Path(args.output) if args.output else (ROOT_OUTPUT / (
        "LF字典准备_" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S%z")
    ))
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    if destination.exists():
        raise FileExistsError(f"Existing dictionary output is never overwritten: {destination}")
    registered = Path(args.registration)
    if not registered.is_absolute():
        registered = PROJECT_ROOT / registered
    policy, initial = load_locked_registration(registered)
    splits = build_power_splits()
    train = validate_training_power_contract(splits.simulation_train)
    validation = sorted(splits.simulation_validation)
    if len(validation) != 10 or set(train) & set(validation):
        raise RuntimeError("LF training/validation split is not the registered 60/10 protocol")
    output = create_fresh_output(destination)
    print("任06字典实验：仅消费LF TRAIN 60功率、SiC/Cu各32节点，验证10功率仅做冻结tau线性表示诊断；不读取HF和LF测试标签。", flush=True)
    print(f"预登记两组初值={initial}；节点清单会在tau优化前独立落盘。", flush=True)
    training_fields, train_sources = _source_fields(train, "train")
    if training_fields[0].power_w != 10.0:
        raise RuntimeError("Spatial node source must be 10W LF TRAIN")
    selection = select_spatial_nodes(training_fields[0], per_material=32)
    matrix = build_curve_matrix(training_fields, selection)
    if (selection.excluded_copper_count != 19 or len(selection.nodes) != 64
            or matrix.normalized_deltas.shape != (3840, 100)
            or np.max(matrix.initial_abs_delta_k) != 0.0):
        raise RuntimeError("Registered LF outer boundary, node/time grid or t0 contract changed")
    provenance = {
        "独立冻结时间": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "预登记源": str(registered.relative_to(PROJECT_ROOT)),
        "预登记SHA256": sha256_file(registered),
        "LF分割SHA256": sha256_file(PROJECT_ROOT / "configs/splits.yaml"),
        "10W源SHA256": train_sources[0]["SHA256"],
        "60个LF训练parquet源与SHA256": train_sources,
        "Cu外圆定温零响应候选节点数_已排除": selection.excluded_copper_count,
        "Cu定温外圆半径_m": selection.copper_outer_radius_m,
        "两材料冻结节点": [asdict(node) for node in selection.nodes],
        "温度差与归一化": "各原始LF曲线减t0；按max(0.1,max|delta_K|)缩放，训练60功率×两材料×32条等权，t2..200等权。",
    }
    _write_json(output / "节点冻结清单.json", provenance)
    print(f"优化前节点冻结已写入；Cu外圆定温候选={selection.excluded_copper_count}，拟合曲线={len(matrix.power_w)}。", flush=True)
    try:
        fit = fit_time_dictionary(matrix.normalized_deltas, matrix.times_s,
                                  initial_groups=initial, ridge_lambda=1e-6)
        candidate_rows = [_candidate_row(index, result) for index, result in enumerate(fit.candidates)]
        if fit.selected_candidate_index is None:
            _write_json(output / "字典数值失败.json", {
                "结论": "两组LF TRAIN变投影均未收敛，E2不可称为已提取成功；保持固定E1。",
                "注册初值拟合": candidate_rows,
                "节点冻结清单SHA256": sha256_file(output / "节点冻结清单.json"),
            })
            raise RuntimeError("Neither registered LF TRAIN tau optimization converged")
        chosen = fit.candidates[fit.selected_candidate_index]
        selected_tau = chosen.tau_seconds
        training_rows = _material_power_rows(matrix, selected_tau, train, 32)
        _save_group_rows(output / "LF训练_各功率材料归一化误差.csv", training_rows)
        print(f"两组TRAIN初值结果={candidate_rows}；选用TRAIN目标最小且收敛的初值组={fit.selected_candidate_index+1}；现在才读取LF合法验证作冻结诊断。", flush=True)
        validation_fields, validation_sources = _source_fields(validation, "validation")
        val_matrix = build_curve_matrix(validation_fields, selection)
        if val_matrix.normalized_deltas.shape != (640, 100):
            raise ValueError("LF validation lacks exactly 10 powers × 2 materials × 32 nodes")
        validation_reports = []
        for index, candidate in enumerate(fit.candidates):
            projection = project_fixed_dictionary(val_matrix.normalized_deltas,
                                                  val_matrix.times_s, candidate.tau_seconds)
            validation_reports.append({
                "注册初值编号": index + 1,
                "冻结tau的LF验证归一化RMSE": projection["normalized_rmse"],
                "仅独立线性幅值的LF验证曲线数": projection["curve_count"],
                "冻结特征条件数": _finite(projection["feature_condition_number"]),
            })
        validation_rows = _material_power_rows(val_matrix, selected_tau, validation, 32)
        _save_group_rows(output / "LF验证_各功率材料冻结字典归一化误差.csv", validation_rows)
        _write_json(output / "LF验证来源清单.json", {
            "分割": "仅simulation.validation_powers_w；冻结共同tau后独立拟合每条LF验证幅值作表示诊断，不回传tau梯度",
            "10个LF验证parquet源与SHA256": validation_sources,
        })
        _write_json(output / "时间字典诊断.json", {
            "结论": "LF TRAIN时间字典已拟合；仅构成任06的E2输入候选准备，不构成HF E0/E1/E2增益或名义物理资格结论。",
            "选用TRAIN正则目标最小且收敛的注册初值组": fit.selected_candidate_index + 1,
            "冻结共同tau_秒": [float(x) for x in selected_tau],
            "注册初值拟合": candidate_rows,
            "LF合法验证仅线性幅值表示诊断": validation_reports,
            "LF训练曲线数": len(matrix.power_w),
            "LF验证曲线数": len(val_matrix.power_w),
            "TRAIN温度尺度下限_K": float(matrix.scales_k.min()),
            "TRAIN尺度小于0点10000001K的曲线数": int(np.sum(matrix.scales_k < 0.10000001)),
            "t0温升绝对最大_K": float(matrix.initial_abs_delta_k.max()),
            "TRAIN分组误差行数": len(training_rows),
            "VALIDATION分组误差行数": len(validation_rows),
            "响应饱和与幅值补偿": _amplitude_diagnostic(fit.training_amplitudes,
                                               matrix.scales_k, matrix.times_s, selected_tau),
            "预登记SHA256": sha256_file(registered),
            "节点冻结清单SHA256": sha256_file(output / "节点冻结清单.json"),
            "LF训练分组CSV_SHA256": sha256_file(output / "LF训练_各功率材料归一化误差.csv"),
            "LF验证分组CSV_SHA256": sha256_file(output / "LF验证_各功率材料冻结字典归一化误差.csv"),
        })
        print(json.dumps({"LF字典候选冻结tau_秒": [float(x) for x in selected_tau],
                          "报告": str(output.relative_to(PROJECT_ROOT))}, ensure_ascii=False), flush=True)
    except Exception as exc:
        failure = output / "LF字典执行异常.json"
        if not failure.exists():
            _write_json(failure, {"错误类型": type(exc).__name__, "错误描述": str(exc),
                                  "结论": "保留节点冻结来源和固定E1；不能宣称E2成功。"})
        raise


if __name__ == "__main__":
    main()
