#!/usr/bin/env python
"""Registered Task-13 fixed-RZ twenty-model CUDA timing; no HF labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sic_cu.eval.task13_rz_fair_timing import (
    benchmark_registered_models, preflight_timing,
)


REGISTRY = ("研究记录/任务13_发布与验收/"
            "任13_原装置四臂同RZ网格推理分项事前登记.yaml")
LEDGER = "多保真DeepONet预测精度优化总计划与执行台账.md"


def main():
    parser = argparse.ArgumentParser(description="任13四臂五seed同RZ网格推理独立成本")
    parser.add_argument("--registry", default=REGISTRY)
    parser.add_argument("--registry-sha", required=True)
    parser.add_argument("--source-tar", required=True)
    parser.add_argument("--source-tar-sha", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    options = parser.parse_args()
    cfg = preflight_timing(
        Path(options.registry), options.registry_sha, Path(options.source_tar),
        options.source_tar_sha, Path(LEDGER), Path(options.output), require_cuda=True,
    )
    cfg["批次来源"] = "正式CPU合成夹具测试之后由总台账前置登记批准的实际CUDA现场"
    report = benchmark_registered_models(cfg, Path(options.output),
                                         device=torch.device(options.device))
    print(f"任13 RZ四臂CUDA20模型分项归档{len(report['模型'])}/20")


if __name__ == "__main__":
    main()
