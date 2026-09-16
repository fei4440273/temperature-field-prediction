#!/usr/bin/env python
"""任-06 E0/E1/E2同源600轮HF校正先导训练入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sic_cu.config import PROJECT_ROOT
from sic_cu.train.task06_time_features import run_task06_time_features


def main() -> None:
    parser = argparse.ArgumentParser(description="任-06三臂同LF及同旧校正权重的新AdamW 600轮HF先导")
    parser.add_argument("--arm", required=True, choices=("E0", "E1", "E2"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-epochs", type=int, default=600)
    parser.add_argument("--session-epoch-limit", type=int)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--resume-training-checkpoint")
    args = parser.parse_args()
    try:
        result = run_task06_time_features(
            arm=args.arm, output_directory=args.output,
            budget_epochs=args.budget_epochs,
            session_epoch_limit=args.session_epoch_limit,
            diagnostic_only=args.diagnostic_only,
            device_name=args.device,
            resume_training_checkpoint=args.resume_training_checkpoint,
        )
    except (Exception, KeyboardInterrupt) as exc:
        output = Path(args.output)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        if output.is_dir():
            recent = output / "阶段_最近.pt"
            report = {
                "状态": "任-06训练会话异常，不产生本会话正式采用结论",
                "异常类型": type(exc).__name__, "异常原因": str(exc),
                "本臂可验证最近完整状态": str(recent) if recent.is_file() else "无已提交阶段_最近.pt",
                "接续要求": "须核任04第120轮真实起点、三臂原600轮预算、tau真buffer及历史最佳完整状态；不得从best.pt虚构AdamW续跑",
            }
            temporary = output / "异常与接续.json.tmp"
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(output / "异常与接续.json")
        raise
    print(json.dumps({key: result[key] for key in (
        "状态", "运行臂", "运行资格", "本臂实际完成轮次", "原预登记预算",
        "任06预登记配置SHA256", "LF共同真实张量SHA256", "真实最后状态", "累计实际消耗",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
