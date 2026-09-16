#!/usr/bin/env python
"""Registered CPU-only Task-11 origin and budget audit; it cannot train models."""

from __future__ import annotations

import argparse

from sic_cu.eval.task11_method_fairness_gate import (
    audit_registered_existing_sources,
    require_task11_registration,
    write_chinese_contract_report,
)


TASK11_REGISTRY = "研究记录/任务11_外部对照/任11_方法级公平预算与来源身份预检前登记.yaml"
TASK13_REGISTRY = "研究记录/任务13_发布与验收/任13_原装置四臂同RZ网格推理分项事前登记.yaml"
TASK13_REGISTRY_SHA = "b82b2c540ce12eeb4d8b8f8677c466211e1860fd301299c224770b8236602d42"
TASK13_SOURCE_TAR = "研究记录/任务13_发布与验收/任13_四臂同RZ网格推理十二源冻结_20260916T054001+0800.tar.gz"
TASK13_SOURCE_SHA = "75725882f8f865a388e0da6cd0d423c5b6c8c1b1db68bb46bf7759190a0752b4"
LEDGER = "多保真DeepONet预测精度优化总计划与执行台账.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="任11既有方法来源与异口径预算CPU预检")
    parser.add_argument("--registry", default=TASK11_REGISTRY)
    parser.add_argument("--registry-sha", required=True)
    parser.add_argument("--source-tar", required=True)
    parser.add_argument("--source-tar-sha", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    require_task11_registration(
        options.registry, options.registry_sha, options.source_tar,
        options.source_tar_sha, LEDGER,
    )
    summary = audit_registered_existing_sources(
        registry=TASK13_REGISTRY, registry_sha=TASK13_REGISTRY_SHA,
        source_tar=TASK13_SOURCE_TAR, source_tar_sha=TASK13_SOURCE_SHA,
        ledger=LEDGER, output=options.output,
    )
    written = write_chinese_contract_report(summary, options.output)
    print(f"任11现有来源CPU预检完成：HF {len(summary['已有HF来源身份'])}/20，"
          f"LF {len(summary['已有LF来源身份'])}/10；新方法模型0；{written}")


if __name__ == "__main__":
    main()
