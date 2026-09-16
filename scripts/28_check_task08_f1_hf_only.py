#!/usr/bin/env python
"""任08 F1独立HF来源CPU极小诊断入口；不得作正式PINN消融。"""

from __future__ import annotations

import argparse
import json

from sic_cu.train.task08_f1_hf_only import run_task08_f1_cpu_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description="任08 F1仅HF来源与模型一优化步CPU诊断")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    if not args.diagnostic_only:
        raise ValueError("任08 F1只有不可正式采用的CPU来源诊断；须指定--diagnostic-only")
    report = run_task08_f1_cpu_diagnostic(seed=args.seed, output_directory=args.output)
    print(json.dumps({key: report[key] for key in (
        "资格", "运行种子", "HF合法训练功率数", "HF合法验证功率数",
        "HF观测优化步", "物理损失优化步", "物理来源限制", "逐工件SHA256",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
