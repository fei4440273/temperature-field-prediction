"""Registered Task 10 v3 full-HF posthoc after the real 25-board group is locked."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.eval.task10_plate_posthoc_v3 import (
    METHOD_BUDGET_SHA, METHOD_TAR_SHA, ORIGINAL_POSTHOC_BUDGET_SHA,
    ORIGINAL_POSTHOC_TAR_SHA, REGISTRATION_SHA, SOURCE_INDEX_SHA,
    run_complete_plate_posthoc_v3,
)


ROOT = PROJECT_ROOT / "研究记录/任务10_独立双层场基准"
REGISTRATION = ROOT / "正式人为多热流入场前登记.yaml"
SOURCE = ROOT / "九热流受限数值源_台账确认后_20260916T032402+0800"
METHOD_BUDGET = ROOT / "正式同板F1_F2_F3重训方法预算前登记_v2.yaml"
METHOD_TAR = ROOT / "正式同板方法十一源源码冻结_v2_20260916T050000+0800.tar.gz"
OLD_POSTHOC_BUDGET = ROOT / "正式同板全组后验指标及源码前登记.yaml"
OLD_POSTHOC_TAR = ROOT / "正式同板后验十源事前冻结.tar.gz"
V3_BUDGET = ROOT / "正式同板全组后验真实LF配对v3前登记.yaml"
V3_TAR = ROOT / "正式同板后验真实LF配对v3七源事前冻结.tar.gz"
MAIN_LEDGER = PROJECT_ROOT / "多保真DeepONet预测精度优化总计划与执行台账.md"


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(
        description="任10全25合法探针和F2/F3真实LF逐张量已锁后正式数值板v3后验"
    )
    parser.add_argument("--全组身份文件", required=True, type=Path)
    parser.add_argument("--全组SHA256", required=True)
    parser.add_argument("--v3预算SHA256", required=True)
    parser.add_argument("--v3七源tarSHA256", required=True)
    parser.add_argument("--输出目录", required=True, type=Path)
    arguments = parser.parse_args(argv)
    immutable = (
        (REGISTRATION, REGISTRATION_SHA),
        (SOURCE / "探针与源场SHA清单.json", SOURCE_INDEX_SHA),
        (METHOD_BUDGET, METHOD_BUDGET_SHA),
        (METHOD_TAR, METHOD_TAR_SHA),
        (OLD_POSTHOC_BUDGET, ORIGINAL_POSTHOC_BUDGET_SHA),
        (OLD_POSTHOC_TAR, ORIGINAL_POSTHOC_TAR_SHA),
    )
    if any(sha256_file(path) != digest for path, digest in immutable):
        raise PermissionError("录0064/0066冻结方法、二级指标或九档数值原SHA漂移")
    report = run_complete_plate_posthoc_v3(
        output_directory=arguments.输出目录,
        group_path=arguments.全组身份文件,
        group_sha256=arguments.全组SHA256,
        registration_path=REGISTRATION,
        archive_root=SOURCE,
        method_budget_path=METHOD_BUDGET,
        method_budget_sha256=METHOD_BUDGET_SHA,
        original_posthoc_budget_path=OLD_POSTHOC_BUDGET,
        original_posthoc_budget_sha256=ORIGINAL_POSTHOC_BUDGET_SHA,
        original_posthoc_tar_path=OLD_POSTHOC_TAR,
        original_posthoc_tar_sha256=ORIGINAL_POSTHOC_TAR_SHA,
        v3_budget_path=V3_BUDGET,
        v3_budget_sha256=arguments.v3预算SHA256,
        v3_tar_path=V3_TAR,
        v3_tar_sha256=arguments.v3七源tarSHA256,
        ledger_path=MAIN_LEDGER,
        device=torch.device("cuda"),
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return report


if __name__ == "__main__":
    main()
