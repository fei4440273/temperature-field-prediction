#!/usr/bin/env python
"""任08 F2真正不求体内PDE的CPU一轮诊断；正式五种子未开放。"""

from __future__ import annotations

import argparse
import json

from sic_cu.train.task08_f2_no_pde import run_task08_f2_cpu_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description="任08 F2同seed五份来源与仅CPU一轮通路诊断")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--seed", type=int, required=True, choices=range(5))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    if not args.diagnostic_only:
        raise ValueError("任08 F2冻结前只有CPU诊断；须写--diagnostic-only，不可冒充正式候选")
    report = run_task08_f2_cpu_diagnostic(seed=args.seed, output_directory=args.output)
    print(json.dumps({name: report[name] for name in (
        "资格", "运行种子", "HF观测优化步", "物理优化步", "物理配点",
        "训练PDE残差", "冻结LF张量未变", "逐工件SHA256",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
