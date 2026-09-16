#!/usr/bin/env python
"""任07五种子合法HF验证的逐功率/时间/径向中文指标导出。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping

import polars as pl
import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.eval.development_v4 import evaluate_hf_observations
from sic_cu.train.task07_formal import TASK07_REGISTRATION


MODALITIES = {"Top": "顶部红外", "Hot": "热端环温", "Cold": "冷端环温"}
WINDOWS = {
    "time_0_30_s": "初期0—30秒",
    "time_30_100_s": "中期30—100秒",
    "time_100_200_s": "后期100—200秒",
}
RADII = {
    "center_0_8_mm": "顶部中心0—8毫米",
    "middle_8_17_mm": "顶部中圈8—17毫米",
    "outer_17_25_mm": "顶部外圈17—25毫米",
}
EMPTY_REASONS = {
    "no_observations_in_preregistered_window": "预登记时间窗内无观测，不外推真值",
    "no_top_observations_in_preregistered_radial_window": "预登记径向窗内无顶部观测，不外推真值",
}
NUMERIC_FIELDS = {
    "sample_count": "实际观测点数", "time_count": "真实观测时刻数",
    "rmse_c": "RMSE_摄氏度", "mae_c": "MAE_摄氏度",
    "signed_bias_c": "有符号偏差_摄氏度",
    "p95_abs_error_c": "绝对误差95分位_摄氏度",
    "max_abs_error_c": "绝对误差最大值_摄氏度",
    "delta_rmse_c": "首观测差分RMSE_摄氏度",
    "delta_mae_c": "首观测差分MAE_摄氏度",
    "delta_signed_bias_c": "首观测差分有符号偏差_摄氏度",
    "delta_p95_abs_error_c": "首观测差分绝对误差95分位_摄氏度",
    "peak_error_c": "峰值偏差_摄氏度",
}


def localize_observation_record(record: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    if kind not in {"power", "time", "radius"} or record.get("split") != "validation":
        raise ValueError("任07只允许已有合法HF验证集合的逐功率/时间/径向观测明细")
    modality = "Top" if kind == "radius" else record.get("modality")
    if modality not in MODALITIES:
        raise ValueError("任07验证指标包含不属于顶部/热端/冷端的观测模态")
    row: dict[str, Any] = {
        "种子": int(record["seed"]), "数据划分": "高保真合法验证",
        "功率_W": float(record["power_w"]), "模态": MODALITIES[modality],
    }
    if kind == "time":
        window = record.get("window")
        if window not in WINDOWS:
            raise ValueError("任07时间窗不属于锁定V4窗口，不可事后改变")
        row.update({"时间窗": WINDOWS[window], "窗起秒": record["lower_s"],
                    "窗止秒": record["upper_s"],
                    "窗起含端点": record["lower_closed"],
                    "窗止含端点": record["upper_closed"]})
    if kind == "radius":
        radial = record.get("radial_window")
        if radial not in RADII:
            raise ValueError("任07径向窗不属于锁定V4窗口，不可事后改变")
        row.update({"径向窗": RADII[radial], "径向起毫米": record["lower_mm"],
                    "径向止毫米": record["upper_mm"],
                    "径向起含端点": record["lower_closed"],
                    "径向止含端点": record["upper_closed"]})
    for source, readable in NUMERIC_FIELDS.items():
        if source in record:
            row[readable] = record[source]
    if "empty_reason" in record:
        reason = record["empty_reason"]
        if reason not in EMPTY_REASONS:
            raise ValueError("任07空窗原因不属于预登记窗口，不能静默显示成绩")
        row["空窗原因"] = EMPTY_REASONS[reason]
    return row


def summarize_five_seed_scores(scores: Mapping[int, float]) -> dict[str, Any]:
    if set(scores) != set(range(5)) or any(not math.isfinite(float(score)) for score in scores.values()):
        raise ValueError("任07必须有五份0—4种子的真实有限合法macro_v1选分")
    values = [float(scores[seed]) for seed in range(5)]
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    worst = max(range(5), key=lambda seed: values[seed])
    return {
        "逐种子选分_摄氏度": {str(seed): values[seed] for seed in range(5)},
        "五种子均值_摄氏度": mean, "五种子样本标准差_摄氏度": std,
        "样本标准差分母": 4, "最坏种子": worst,
        "最坏选分_摄氏度": values[worst],
    }


def _entry() -> Any:
    script = PROJECT_ROOT / "scripts/27_verify_task07_inference.py"
    spec = importlib.util.spec_from_file_location("task07_inference_preflight_for_metrics", script)
    if spec is None or spec.loader is None:
        raise ImportError("任07完整五种子推理原件门禁不存在")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def export_task07_observations(
    seed_directories: Mapping[int, str | Path], output_directory: str | Path, *,
    registry_sha256: str, device_name: str = "cuda",
) -> dict[str, Any]:
    if set(seed_directories) != set(range(5)):
        raise ValueError("任07合法观测报告缺失五种子0到4，不得提前创建输出或读取温度")
    output = _path(output_directory)
    if output.exists() or any(_path(root).resolve() in output.resolve().parents
                              or _path(root).resolve() == output.resolve()
                              for root in seed_directories.values()):
        raise FileExistsError("任07观测明细输出不可覆盖或嵌在真实种子训练原件目录内")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("任07正式五种子观测报告只在真实CUDA及完整推理原件门禁后输出")
    registered = _entry()._preflight_five_seeds(
        seed_directories, registry_path=TASK07_REGISTRATION,
        registry_sha256=registry_sha256, device=device,
    )
    manifest = load_yaml("reports/development_v4/baseline_manifest.yaml")
    comparison = manifest["data_and_protocol_hashes"]["optimization_v4_config_sha256"]
    config_path = PROJECT_ROOT / "configs/optimization_v4.yaml"
    if sha256_file(config_path) != comparison:
        raise ValueError("任07时间窗/径向窗原V4登记SHA漂移，不得事后改变展示口径")
    config = load_yaml(config_path)
    if config.get("allow_test_labels") is not False or config["diagnostics"].get(
        "evaluation_splits") != ["train", "validation"]:
        raise ValueError("任07合法HF观察指标的V4窗口配置禁止读取旧固定测试温度")
    diagnostic = config["diagnostics"]
    bottom_z_m = float(load_yaml("configs/geometry.yaml")["embedding"]["copper_bottom_z_m"])
    raw: dict[str, list[dict[str, Any]]] = {"power": [], "time": [], "radius": []}
    new_scores: dict[int, float] = {}
    for seed in range(5):
        info = registered[seed]
        by_power, by_time, by_radius, _ = evaluate_hf_observations(
            info["predictor"].model, seed, "validation", device, diagnostic, bottom_z_m,
        )
        if (len(by_power) != 9 or len(by_time) != 27 or len(by_radius) != 9
                or {row["power_w"] for row in by_power} != set(info["source"].hf_validation_powers_w)):
            raise ValueError(f"任07 seed{seed} 独立HF验证三个功率三模态/预定时间径向窗不完整")
        raw["power"].extend(by_power)
        raw["time"].extend(by_time)
        raw["radius"].extend(by_radius)
        new_scores[seed] = float(info["score"])
    original = {int(row["seed"]): float(row["validation_reproduction"]["reproduced_macro_v1"])
                for row in manifest["checkpoints"]}
    summary: dict[str, Any] = {
        "状态": "修复版五种子完整状态门禁后的合法HF验证观测明细；不得据此称能源或内部真值合格",
        "本轮来源登记SHA256": registry_sha256,
        "V4历史原配置SHA256": comparison,
        "V4历史五种子来源清单SHA256": sha256_file(
            PROJECT_ROOT / "reports/development_v4/baseline_manifest.yaml"
        ),
        "本轮合法macro_v1": summarize_five_seed_scores(new_scores),
        "历史B0合法macro_v1": summarize_five_seed_scores(original),
        "逐种子相对历史B0数值改善率_百分比_不同预算不可直接归因": {
            str(seed): 100.0 * (original[seed] - new_scores[seed]) / original[seed]
            for seed in range(5)
        },
        "旧test_Data温度标签读取": False,
        "方法限制": "历史B0和本轮新HF随机初始化/早停/受限联合预算不逐位同起点；数值差不等于E0或联合机制增益；观测、能源、HF内部实测真值分别验收",
    }
    rows = {kind: [localize_observation_record(record, kind=kind) for record in records]
            for kind, records in raw.items()}
    output.mkdir(parents=True)
    filenames = {"power": "合法验证逐功率逐模态.csv",
                 "time": "合法验证逐功率逐模态时间窗.csv",
                 "radius": "合法验证顶部逐功率径向窗.csv"}
    for kind, filename in filenames.items():
        pl.DataFrame(rows[kind]).write_csv(output / filename, null_value="")
    (output / "原始合法指标与中文列映射.json").write_text(
        json.dumps({"原始指标": raw, "原始到中文数值列": NUMERIC_FIELDS,
                    "原始到中文模态": MODALITIES}, ensure_ascii=False,
                   indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    (output / "五种子观测验证摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    hashes = {file.name: sha256_file(file) for file in output.iterdir() if file.is_file()}
    (output / "工件SHA256.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return summary | {"输出": str(output), "已核工件SHA256": hashes,
                      "实际明细行数": {kind: len(rows[kind]) for kind in rows}}


def main() -> None:
    parser = argparse.ArgumentParser(description="任07五种子合法HF验证逐功率中文明细")
    for seed in range(5):
        parser.add_argument(f"--seed{seed}", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    result = export_task07_observations(
        {seed: getattr(args, f"seed{seed}") for seed in range(5)}, args.output,
        registry_sha256=args.registry_sha256, device_name=args.device,
    )
    print(json.dumps({"状态": result["状态"], "输出": result["输出"],
                      "实际明细行数": result["实际明细行数"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
