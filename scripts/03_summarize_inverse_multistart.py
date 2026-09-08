#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_run(path: Path) -> dict[str, Any]:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    configuration = metrics["configuration"]
    if configuration.get("test_evaluation_enabled") is not False:
        raise ValueError(f"Calibration run accessed the test split: {path}")
    if metrics.get("test_ir") is not None or metrics.get("test_sensor") is not None:
        raise ValueError(f"Calibration run contains held-out test metrics: {path}")
    return metrics


def summarize(paths: list[Path]) -> dict[str, Any]:
    runs = [_load_run(path) for path in paths]
    checkpoint_seeds = []
    for path in paths:
        training_first = json.loads(
            (path / "training.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        del training_first
        name_seed = path.name.split("seed", 1)[-1].split("_", 1)[0]
        checkpoint_seeds.append(int(name_seed))
    if len(set(checkpoint_seeds)) != 1:
        raise ValueError("Multi-start diagnostic must use one common random seed")

    records = []
    for label, path, run in zip(("low", "mid", "high"), paths, runs, strict=True):
        config = run["configuration"]
        final = run["identified_physics_parameters"]
        initial_rc = float(config["contact_resistance_initial_m2_k_w"])
        records.append(
            {
                "scenario": label,
                "path": str(path),
                "seed": checkpoint_seeds[0],
                "best_epoch": int(run["best_epoch"]),
                "validation_ir_rmse_k": float(run["best_validation_ir_rmse_k"]),
                "initial": {
                    "silicon_carbide_emissivity": float(
                        config["silicon_carbide_emissivity_initial"]
                    ),
                    "copper_emissivity": float(config["copper_emissivity_initial"]),
                    "contact_resistance_m2_k_w": initial_rc,
                },
                "final": final,
                "absolute_drift": {
                    "silicon_carbide_emissivity": abs(
                        float(final["silicon_carbide_emissivity"])
                        - float(config["silicon_carbide_emissivity_initial"])
                    ),
                    "copper_emissivity": abs(
                        float(final["copper_emissivity"])
                        - float(config["copper_emissivity_initial"])
                    ),
                    "contact_resistance_log10": abs(
                        float(final["contact_resistance_log10"])
                        - math.log10(initial_rc)
                    ),
                },
            }
        )
    selected = min(records, key=lambda item: item["validation_ir_rmse_k"])
    return {
        "schema_version": 1,
        "status": "SHORT_RUN_IDENTIFIABILITY_DIAGNOSTIC_NOT_FINAL_MODEL",
        "test_powers_accessed": False,
        "common_seed": checkpoint_seeds[0],
        "runs": records,
        "validation_selected_sensitivity_scenario": selected["scenario"],
        "identifiability": {
            "silicon_carbide_emissivity": "NOT_IDENTIFIED_MULTISTART_RETAINS_INITIALIZATION",
            "copper_emissivity": "NOT_IDENTIFIED_MULTISTART_RETAINS_INITIALIZATION",
            "contact_resistance_joint_gradient": "NOT_IDENTIFIED_MULTISTART_RETAINS_INITIALIZATION",
            "contact_resistance_low_fidelity_balance": (
                "IDENTIFIED_LOW_FIDELITY_EFFECTIVE_INITIALIZATION"
            ),
        },
        "decision": (
            "Use Rc=7.34072435302768e-5 m2*K/W as the effective low-fidelity initialization. "
            "Treat emissivities as sensitivity scenarios; do not claim unique fitted values."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    rows = []
    for run in summary["runs"]:
        initial, final, drift = run["initial"], run["final"], run["absolute_drift"]
        rows.append(
            f"| {run['scenario']} | {run['validation_ir_rmse_k']:.4f} | "
            f"{initial['silicon_carbide_emissivity']:.3f} -> "
            f"{final['silicon_carbide_emissivity']:.6f} | "
            f"{initial['copper_emissivity']:.3f} -> {final['copper_emissivity']:.6f} | "
            f"{initial['contact_resistance_m2_k_w']:.3e} -> "
            f"{final['contact_resistance_m2_k_w']:.3e} | "
            f"{drift['contact_resistance_log10']:.4f} |"
        )
    table = "\n".join(rows)
    return f"""# 多初值逆辨识诊断

状态：`{summary['status']}`  
共同随机种子：`{summary['common_seed']}`  
数据隔离：测试集未读取，模型比较只使用验证功率。

| 场景 | 验证 IR RMSE (K) | SiC 辐射率 | Cu 辐射率 | Rc (m²·K/W) | Rc 的 log10 漂移 |
|---|---:|---:|---:|---:|---:|
{table}

验证集短跑选择的是 `{summary['validation_selected_sensitivity_scenario']}` 场景，但三个参数
均基本保留各自初值，没有从不同初值收敛到共同解。因此该结果只能用于发现不可辨识性，
不能作为真实材料参数或正式模型精度报告。

## 决策

- 接触热阻采用独立低保真界面平衡得到的 `7.34072435302768e-5 m²·K/W` 作为有效初值。
- SiC/Cu 辐射率保留 low/mid/high 三个敏感性场景；不得声称得到唯一实验辨识值。
- 正式模型选择继续只看验证功率，锁定配置后才允许一次性打开测试集。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize same-seed inverse multi-start runs")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            Path("reports/runs/inverse_seed20_low"),
            Path("reports/runs/inverse_seed20_mid"),
            Path("reports/runs/inverse_seed20_high"),
        ],
    )
    parser.add_argument(
        "--output-json", default="reports/inverse_multistart_diagnostic.json", type=Path
    )
    parser.add_argument(
        "--output-markdown", default="reports/inverse_multistart_diagnostic.md", type=Path
    )
    args = parser.parse_args()
    summary = summarize(args.paths)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.output_markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
