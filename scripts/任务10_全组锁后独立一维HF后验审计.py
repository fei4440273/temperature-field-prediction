"""Formal one-dimensional F1/F2/F3 posthoc; never open HF before two ledger gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.eval.task10_plate_posthoc import run_complete_plate_posthoc


ROOT = PROJECT_ROOT / "研究记录/任务10_独立双层场基准"
REGISTRATION = ROOT / "正式人为多热流入场前登记.yaml"
SOURCE = ROOT / "九热流受限数值源_台账确认后_20260916T032402+0800"
METHOD_BUDGET = ROOT / "正式同板F1_F2_F3重训方法预算前登记_v2.yaml"
POSTHOC_BUDGET = ROOT / "正式同板全组后验指标及源码前登记.yaml"
POSTHOC_TAR = ROOT / "正式同板后验十源事前冻结.tar.gz"
LEDGER = PROJECT_ROOT / "多保真DeepONet预测精度优化总计划与执行台账.md"
METHOD_BUDGET_SHA256 = "cf05d16a3cd754e7f75321418ec6186e42673e33f8774492f3eea985105fba94"


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description="任务10合法全25锁后独立同板完整HF后验")
    parser.add_argument("--全组身份文件", required=True, type=Path)
    parser.add_argument("--全组SHA256", required=True)
    parser.add_argument("--二级预算SHA256", required=True)
    parser.add_argument("--二级源码tarSHA256", required=True)
    parser.add_argument("--输出目录", required=True, type=Path)
    args = parser.parse_args(argv)
    if sha256_file(METHOD_BUDGET) != METHOD_BUDGET_SHA256:
        raise PermissionError("已事前总台账冻结的11源训练方法预算SHA漂移，完整HF拒绝首读")
    report = run_complete_plate_posthoc(
        output_directory=args.输出目录,
        group_path=args.全组身份文件,
        group_sha256=args.全组SHA256,
        registration_path=REGISTRATION,
        archive_root=SOURCE,
        method_budget_path=METHOD_BUDGET,
        method_budget_sha256=METHOD_BUDGET_SHA256,
        posthoc_budget_path=POSTHOC_BUDGET,
        posthoc_budget_sha256=args.二级预算SHA256,
        posthoc_tar_path=POSTHOC_TAR,
        posthoc_tar_sha256=args.二级源码tarSHA256,
        ledger_path=LEDGER,
        device=torch.device("cuda"),
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return report


if __name__ == "__main__":
    main()
