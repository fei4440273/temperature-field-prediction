#!/usr/bin/env python
"""任08 F2五种子仅移除HF体内PDE的独立正式训练CLI。"""

from __future__ import annotations

import argparse
import json

from sic_cu.train.task08_f2_formal import TASK08_F2_REGISTRATION, run_task08_f2_formal


def main() -> None:
    parser = argparse.ArgumentParser(description="任08 F2同seed HF校正1500＋受限联合500")
    parser.add_argument("--seed", type=int, required=True, choices=range(5))
    parser.add_argument("--arm", choices=("F2",), default="F2")
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
        options["formal_registry_path"] = TASK08_F2_REGISTRATION
        options["formal_registry_sha256"] = args.registry_sha256
    report = run_task08_f2_formal(**options)
    print(json.dumps({key: report[key] for key in (
        "状态", "运行种子", "运行臂", "运行资格", "正式预登记配置SHA256",
        "校正实际轮次", "联合实际轮次", "观测最佳HF合法选分_摄氏度",
        "物理最佳独立全项损失", "训练PDE", "累计实际消耗", "真实最后状态",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
