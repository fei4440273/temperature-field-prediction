#!/usr/bin/env python
"""任-04冻结与有限解冻臂的同源AdamW/RNG继续训练入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sic_cu.config import PROJECT_ROOT
from sic_cu.train.task04_joint import run_task04_joint


def main() -> None:
    parser = argparse.ArgumentParser(description="任-04同校正末源的冻结/有限解冻200轮对照")
    parser.add_argument("--arm", required=True, choices=("冻结", "有限解冻"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-epochs", type=int, default=200)
    parser.add_argument("--session-epoch-limit", type=int)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--decision-path")
    parser.add_argument("--resume-training-checkpoint")
    args = parser.parse_args()
    options = {
        "arm": args.arm, "output_directory": args.output,
        "budget_epochs": args.budget_epochs,
        "session_epoch_limit": args.session_epoch_limit,
        "diagnostic_only": args.diagnostic_only,
        "device_name": args.device,
        "resume_training_checkpoint": args.resume_training_checkpoint,
    }
    if args.source_checkpoint is not None:
        options["source_checkpoint"] = args.source_checkpoint
    if args.decision_path is not None:
        options["decision_path"] = args.decision_path
    try:
        result = run_task04_joint(**options)
    except (Exception, KeyboardInterrupt) as exc:
        output = Path(args.output)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        if output.is_dir():
            latest = output / "阶段_最近.pt"
            report = {
                "状态": "任-04会话异常，不产生正式采用资格",
                "异常类型": type(exc).__name__, "异常原因": str(exc),
                "真实最后可续跑状态": str(latest) if latest.is_file() else "无连续末态",
                "接续要求": "核对原200轮预算、源SHA与本臂最近完整状态后再续跑，不使用历史最佳优化器",
            }
            temporary = output / "异常与接续.json.tmp"
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(output / "异常与接续.json")
        raise
    print(json.dumps({key: result[key] for key in (
        "状态", "运行臂", "本臂实际完成轮次", "原预登记预算", "源校正末SHA256",
        "真实最后状态", "累计实际消耗", "策略采用",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
