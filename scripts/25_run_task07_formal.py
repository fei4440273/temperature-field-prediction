#!/usr/bin/env python
"""任-07五种子E0 HF校正与LF四末投影受限联合训练入口。"""

from __future__ import annotations

import argparse
import json

from sic_cu.train.task07_formal import TASK07_REGISTRATION, run_task07_formal


def main() -> None:
    parser = argparse.ArgumentParser(description="任-07配对V4 B0五种子E0同组1500校正＋500受限联合")
    parser.add_argument("--seed", type=int, required=True, choices=range(5))
    parser.add_argument("--arm", choices=("E0",), default="E0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--session-epoch-limit", type=int)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--diagnostic-joint-preview", action="store_true")
    parser.add_argument("--resume-training-checkpoint")
    parser.add_argument("--registry-sha256")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    options = {
        "seed": args.seed, "output_directory": args.output,
        "session_epoch_limit": args.session_epoch_limit,
        "diagnostic_only": args.diagnostic_only,
        "diagnostic_joint_preview": args.diagnostic_joint_preview,
        "resume_training_checkpoint": args.resume_training_checkpoint,
        "device_name": args.device,
    }
    if args.registry_sha256 is not None:
        options["formal_registry_path"] = TASK07_REGISTRATION
        options["formal_registry_sha256"] = args.registry_sha256
    report = run_task07_formal(**options)
    print(json.dumps({key: report[key] for key in (
        "状态", "运行种子", "运行臂", "运行资格", "正式预登记配置SHA256",
        "校正实际轮次", "联合实际轮次", "LF保持资格", "累计实际消耗",
        "真实最后状态",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
