#!/usr/bin/env python
"""任09专属F3子集正式CUDA运行；事前登记缺失时拒绝入场。"""

from __future__ import annotations

import argparse
import json

from sic_cu.train.task09_subset_formal import run_task09_formal


def main() -> None:
    parser = argparse.ArgumentParser(description="任09原登记三序列独立F3真实子集训练")
    parser.add_argument("--arm", required=True,
                        choices=("primary", "left_center", "right_center"))
    parser.add_argument("--size", required=True, type=int, choices=(3, 6, 9))
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--output", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--session-epoch-limit", required=True, type=int)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    report = run_task09_formal(
        name=args.arm, size=args.size, seed=args.seed,
        output_directory=args.output, device_name=args.device,
        registry_path=args.registry, registry_sha256=args.registry_sha256,
        session_epoch_limit=args.session_epoch_limit,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
