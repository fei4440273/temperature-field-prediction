#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from sic_cu.config import load_yaml
from sic_cu.train.simulation import train_simulation_model


def main() -> None:
    parser = argparse.ArgumentParser(description="任-03锁定旧LF后的仿真监督接续训练")
    parser.add_argument("--arm", choices=("control", "space"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--samples-per-power", type=int, default=8192)
    parser.add_argument("--validation-samples-per-power", type=int, default=8192)
    parser.add_argument("--session-epoch-limit", type=int)
    parser.add_argument("--resume-training-checkpoint")
    args = parser.parse_args()
    locked = load_yaml("研究记录/任务03_低保真精度修复/有效运行配置.yaml")
    if args.epochs < 1 or args.epochs > locked["LF每臂逻辑轮次上限"]:
        raise ValueError("任-03 LF预算不得超过预登记的300轮上限")
    result = train_simulation_model(
        method="deeponet_pinn", seed=locked["种子"],
        output_directory=args.output, epochs=args.epochs,
        samples_per_power=args.samples_per_power,
        validation_samples_per_power=args.validation_samples_per_power,
        batch_size=locked["LF每卡batch"],
        learning_rate=locked["LF学习率"], patience=args.epochs + 1,
        initial_checkpoint=locked["旧LF起点"],
        sampling_mode=("material_time" if args.arm == "control" else "material_time_space"),
        lf_physics_mode="none", session_epoch_limit=args.session_epoch_limit,
        resume_training_checkpoint=args.resume_training_checkpoint,
    )
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
