#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.train.multifidelity import train_multifidelity, write_resume_status


def main() -> None:
    parser = argparse.ArgumentParser(description="任-03同初始化HF校正器的原LF/新LF传递对照")
    parser.add_argument("--arm", choices=("old", "new"), required=True)
    parser.add_argument("--new-lf-source", choices=("space", "control"), default="space")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--session-epoch-limit", type=int)
    parser.add_argument("--resume-training-checkpoint")
    args = parser.parse_args()
    locked = load_yaml("研究记录/任务03_低保真精度修复/有效运行配置.yaml")
    if args.epochs < 1 or args.epochs > locked["HF每臂逻辑轮次上限"]:
        raise ValueError("任-03 HF校正轮次不得超预登记上限")
    if args.epochs != locked["HF每臂逻辑轮次上限"] and not args.diagnostic_only:
        raise ValueError("任-03正式候选必须300轮；短门禁须标注--diagnostic-only且不得用于采用判断")
    registered_candidates = {
        "space": "研究记录/任务03_低保真精度修复/任务03_LF空间_种子0_20260915T184423+0800/best.pt",
        "control": "研究记录/任务03_低保真精度修复/任务03_LF控制_种子0_20260915T184117+0800/best.pt",
    }
    low_fidelity = (
        locked["旧LF起点"] if args.arm == "old"
        else registered_candidates[args.new_lf_source]
    )
    try:
        result = train_multifidelity(
            low_fidelity_checkpoint=low_fidelity,
            locked_historical_lf_checkpoint=(
                None if args.arm == "old" else locked["旧LF起点"]
            ),
            start_checkpoint=locked["HF配对B0起点"],
            output_directory=args.output, seed=locked["种子"],
            correction_epochs=args.epochs, joint_epochs=0,
            batch_size=locked["HF每卡batch"],
            physics_collocation=locked["HF每项物理整包配点"],
            width=128, depth=4,
            sensor_absolute_weight=locked["HF传感器绝对与温升权重"][0],
            sensor_delta_weight=locked["HF传感器绝对与温升权重"][1],
            physics_schedule="separate",
            continuation_learning_rate=locked["HF学习率"],
            validation_interval=10, patience=args.epochs + 1,
            resume_training_checkpoint=args.resume_training_checkpoint,
            session_epoch_limit=args.session_epoch_limit,
            evaluate_test=False,
        )
    except (Exception, KeyboardInterrupt) as exc:
        output = PROJECT_ROOT / args.output
        last = output / "阶段_最近.pt"
        write_resume_status(
            PROJECT_ROOT / "研究记录/当前接续状态.md",
            task="任-03", run=args.output,
            last_checkpoint=str(last) if last.exists() else "无可恢复状态",
            reason=f"HF收益传递异常：{type(exc).__name__}: {exc}",
            next_action="原预算、源文件与原目录核对后，只从对应完整状态接续；不覆盖旧B0或门禁产物",
        )
        raise
    fields = (
        "status", "best_epoch", "best_validation_selection_score_c",
        "best_validation_ir_rmse_c", "epochs_completed", "training_seconds",
        "lf_checkpoint_sha256", "data_consumption_rank0",
    )
    summary = {name: result.get(name) for name in fields}
    if args.diagnostic_only:
        summary["资格"] = "短诊断；不参与任-03同预算采用判断"
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
