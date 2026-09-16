#!/usr/bin/env python
"""Sole Task 10 posthoc entry after the row-0074 real-CUDA qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.eval.task10_plate_cuda_gate import (
    METHOD_BUDGET_SHA256, METHOD_TAR_SHA256,
    ORIGINAL_POSTHOC_BUDGET_SHA256, ORIGINAL_POSTHOC_TAR_SHA256,
    V3_BUDGET_SHA256, V3_TAR_SHA256,
    verify_25_cuda_evidence, verify_cuda_source_freeze,
)
from sic_cu.eval.task10_plate_posthoc import verify_late_posthoc_contract
from sic_cu.eval.task10_plate_posthoc_v3 import (
    REGISTRATION_SHA, SOURCE_INDEX_SHA, run_complete_plate_posthoc_v3,
    verify_v3_source_freeze,
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
CUDA_BUDGET = ROOT / "正式同板真CUDA训练与后验资格v4前登记.yaml"
CUDA_TAR = ROOT / "正式同板真CUDA资格十二源事前冻结.tar.gz"
LEDGER = PROJECT_ROOT / "多保真DeepONet预测精度优化总计划与执行台账.md"


def _assert_old_frozen_bytes() -> None:
    immutable = {
        REGISTRATION: REGISTRATION_SHA,
        SOURCE / "探针与源场SHA清单.json": SOURCE_INDEX_SHA,
        METHOD_BUDGET: METHOD_BUDGET_SHA256,
        METHOD_TAR: METHOD_TAR_SHA256,
        OLD_POSTHOC_BUDGET: ORIGINAL_POSTHOC_BUDGET_SHA256,
        OLD_POSTHOC_TAR: ORIGINAL_POSTHOC_TAR_SHA256,
        V3_BUDGET: V3_BUDGET_SHA256,
        V3_TAR: V3_TAR_SHA256,
    }
    if any(not path.is_file() or sha256_file(path) != digest
           for path, digest in immutable.items()):
        raise PermissionError("录0064/0066/0070任一冻结原件SHA漂移")


def _group_identities(path: Path) -> list[dict]:
    resolved = path.resolve()
    if PROJECT_ROOT not in resolved.parents or not resolved.is_file():
        raise PermissionError("真CUDA全组身份文件必须是项目内原件")
    try:
        group = json.loads(resolved.read_text(encoding="utf-8"))
        identities = group["身份"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise PermissionError("真CUDA全组身份文件不可机读") from error
    if not isinstance(identities, list):
        raise PermissionError("真CUDA全组身份必须为列表")
    return identities


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="任10录0064/0066/0070及真CUDA25资格后的唯一完整HF后验v4入口"
    )
    parser.add_argument("--全组身份文件", required=True, type=Path)
    parser.add_argument("--全组SHA256", required=True)
    parser.add_argument("--CUDA资格预算SHA256", required=True)
    parser.add_argument("--CUDA资格源码tarSHA256", required=True)
    parser.add_argument("--输出目录", required=True, type=Path)
    arguments = parser.parse_args(argv)

    _assert_old_frozen_bytes()
    cuda_freeze = verify_cuda_source_freeze(
        cuda_budget_path=CUDA_BUDGET,
        cuda_budget_sha256=arguments.CUDA资格预算SHA256,
        cuda_tar_path=CUDA_TAR,
        cuda_tar_sha256=arguments.CUDA资格源码tarSHA256,
        ledger_path=LEDGER,
    )
    old_contract = verify_late_posthoc_contract(
        group_path=arguments.全组身份文件,
        group_sha256=arguments.全组SHA256,
        registration_path=REGISTRATION,
        archive_root=SOURCE,
        method_budget_path=METHOD_BUDGET,
        method_budget_sha256=METHOD_BUDGET_SHA256,
        posthoc_budget_path=OLD_POSTHOC_BUDGET,
        posthoc_budget_sha256=ORIGINAL_POSTHOC_BUDGET_SHA256,
        posthoc_tar_path=OLD_POSTHOC_TAR,
        posthoc_tar_sha256=ORIGINAL_POSTHOC_TAR_SHA256,
        ledger_path=LEDGER,
    )
    v3_contract = verify_v3_source_freeze(
        v3_budget_path=V3_BUDGET,
        v3_budget_sha256=V3_BUDGET_SHA256,
        v3_tar_path=V3_TAR,
        v3_tar_sha256=V3_TAR_SHA256,
        group_sha256=arguments.全组SHA256,
        method_budget_sha256=METHOD_BUDGET_SHA256,
        original_posthoc_budget_sha256=ORIGINAL_POSTHOC_BUDGET_SHA256,
        original_posthoc_tar_sha256=ORIGINAL_POSTHOC_TAR_SHA256,
        ledger_path=LEDGER,
    )
    cuda_group = verify_25_cuda_evidence(_group_identities(arguments.全组身份文件))

    report = run_complete_plate_posthoc_v3(
        output_directory=arguments.输出目录,
        group_path=arguments.全组身份文件,
        group_sha256=arguments.全组SHA256,
        registration_path=REGISTRATION,
        archive_root=SOURCE,
        method_budget_path=METHOD_BUDGET,
        method_budget_sha256=METHOD_BUDGET_SHA256,
        original_posthoc_budget_path=OLD_POSTHOC_BUDGET,
        original_posthoc_budget_sha256=ORIGINAL_POSTHOC_BUDGET_SHA256,
        original_posthoc_tar_path=OLD_POSTHOC_TAR,
        original_posthoc_tar_sha256=ORIGINAL_POSTHOC_TAR_SHA256,
        v3_budget_path=V3_BUDGET,
        v3_budget_sha256=V3_BUDGET_SHA256,
        v3_tar_path=V3_TAR,
        v3_tar_sha256=V3_TAR_SHA256,
        ledger_path=LEDGER,
        device=torch.device("cuda"),
    )
    qualification = {
        "阶段": "录0064/0066/0070及录0074真CUDA资格后正式后验",
        "录0074源码冻结": cuda_freeze,
        "录0066旧二级门禁": old_contract,
        "录0070真实LF二级门禁": v3_contract,
        "25身份真CUDA资格": cuda_group,
        "完整HF后验已由原v3执行": True,
        "失败训练批次恢复": False,
    }
    qualification_path = Path(report["唯一输出目录"]) / "真CUDA资格前置原件摘要.json"
    qualification_path.write_text(
        json.dumps(qualification, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result = {**report,
              "真CUDA资格前置原件摘要": str(qualification_path),
              "真CUDA资格前置原件摘要_SHA256": sha256_file(qualification_path)}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
