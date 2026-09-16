#!/usr/bin/env python
"""任09仅CPU来源、嵌套功率与每轮预算入场；不训练GPU模型。"""

from __future__ import annotations

import argparse
import json

from sic_cu.train.task09_subset_gate import run_task09_cpu_entry


def main() -> None:
    parser = argparse.ArgumentParser(description="任09子集来源与预算只读CPU入场")
    parser.add_argument("--cpu-source-only", action="store_true")
    parser.add_argument("--arm", required=True,
                        choices=("primary", "left_center", "right_center"))
    parser.add_argument("--size", type=int, required=True, choices=(3, 6, 9, 12))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.cpu_source_only:
        parser.error("必须明确--cpu-source-only；任09此入口不能执行正式GPU训练")
    report = run_task09_cpu_entry(args.output, name=args.arm, size=args.size)
    print(json.dumps({key: report[key] for key in (
        "资格", "选定臂", "选定HF功率数", "预登记原件SHA256",
        "HF合法验证固定功率数", "已执行HF观测优化步",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
