#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCENARIOS = ("low", "mid", "high")


def summarize(paths: list[Path], insensitivity_threshold_k: float) -> dict[str, Any]:
    if len(paths) != len(SCENARIOS):
        raise ValueError("Exactly low, mid, and high sensitivity runs are required")
    records = []
    seeds = set()
    for scenario, path in zip(SCENARIOS, paths, strict=True):
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        config = metrics["configuration"]
        if config.get("test_evaluation_enabled") is not False:
            raise ValueError(f"Sensitivity selection accessed test data: {path}")
        if config.get("physics_parameters_frozen") is not True:
            raise ValueError(f"Sensitivity parameters were not frozen: {path}")
        if metrics.get("test_ir") is not None or metrics.get("test_sensor") is not None:
            raise ValueError(f"Sensitivity run contains test results: {path}")
        seeds.add(int(metrics["seed"]))
        values = metrics["identified_physics_parameters"]
        records.append(
            {
                "scenario": scenario,
                "path": str(path),
                "seed": int(metrics["seed"]),
                "best_epoch": int(metrics["best_epoch"]),
                "validation_ir_rmse_k": float(metrics["best_validation_ir_rmse_k"]),
                "silicon_carbide_emissivity": float(
                    values["silicon_carbide_emissivity"]
                ),
                "copper_emissivity": float(values["copper_emissivity"]),
                "contact_resistance_m2_k_w": float(
                    values["contact_resistance_m2_k_w"]
                ),
            }
        )
    if len(seeds) != 1:
        raise ValueError("Sensitivity runs must use one common seed")
    validation_values = [record["validation_ir_rmse_k"] for record in records]
    spread = max(validation_values) - min(validation_values)
    insensitive = spread <= insensitivity_threshold_k
    selected = "mid" if insensitive else min(
        records, key=lambda record: record["validation_ir_rmse_k"]
    )["scenario"]
    return {
        "schema_version": 1,
        "status": "PHYSICS_SENSITIVITY_SELECTION_COMPLETE",
        "test_powers_accessed": False,
        "common_seed": next(iter(seeds)),
        "runs": records,
        "validation_rmse_spread_k": spread,
        "insensitivity_threshold_k": insensitivity_threshold_k,
        "emissivity_sensitivity_classification": (
            "INSENSITIVE_WITHIN_SCREEN" if insensitive else "VALIDATION_SENSITIVE"
        ),
        "selected_nominal_scenario": selected,
        "selection_rule": (
            "Use neutral mid scenario when validation spread is below threshold; otherwise "
            "select minimum validation RMSE. Never interpret this as material-property truth."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {run['scenario']} | {run['silicon_carbide_emissivity']:.3f} | "
        f"{run['copper_emissivity']:.3f} | "
        f"{run['contact_resistance_m2_k_w']:.8e} | "
        f"{run['best_epoch']} | {run['validation_ir_rmse_k']:.6f} |"
        for run in summary["runs"]
    )
    return f"""# 固定物理参数敏感性筛选

状态：`{summary['status']}`  
测试集：未访问  
共同随机种子：`{summary['common_seed']}`

| 场景 | SiC 辐射率 | Cu 辐射率 | Rc (m²·K/W) | 最佳 epoch | 验证 IR RMSE (K) |
|---|---:|---:|---:|---:|---:|
{rows}

三组验证 RMSE 的最大差为 `{summary['validation_rmse_spread_k']:.6f} K`，判定阈值为
`{summary['insensitivity_threshold_k']:.3f} K`，因此分类为
`{summary['emissivity_sensitivity_classification']}`。

名义模型锁定 `{summary['selected_nominal_scenario']}` 场景。该选择的含义是：在当前观测和
训练精度下，low/mid/high 辐射率对验证温度误差的影响不可区分，因此采用中性值用于后续
模型开发，同时保留两端场景报告敏感性。它不代表辐射率被实验测得或唯一辨识。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fixed physics sensitivity runs")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            Path("reports/runs/physics_sensitivity_seed30_low"),
            Path("reports/runs/physics_sensitivity_seed30_mid"),
            Path("reports/runs/physics_sensitivity_seed30_high"),
        ],
    )
    parser.add_argument("--insensitivity-threshold-k", type=float, default=0.1)
    parser.add_argument(
        "--output-json", type=Path, default=Path("reports/physics_sensitivity.json")
    )
    parser.add_argument(
        "--output-markdown", type=Path, default=Path("reports/physics_sensitivity.md")
    )
    args = parser.parse_args()
    result = summarize(args.paths, args.insensitivity_threshold_k)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.output_markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
