#!/usr/bin/env python
"""Generate the Task 10 all-25 CUDA identity catalog before ROOT registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sic_cu.eval.task10_plate_group_catalog import build_cuda_group_catalog


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="任10十五个真CUDA批次生成全25身份JSON；不读取完整HF且不写ROOT",
    )
    parser.add_argument(
        "--批次目录", action="append", required=True, type=Path,
        help="重复15次：5个F1批次及两LF来源各5个PAIR批次",
    )
    parser.add_argument("--输出", required=True, type=Path)
    parser.add_argument("--CUDA资格预算SHA256", required=True)
    parser.add_argument("--CUDA资格源码tarSHA256", required=True)
    arguments = parser.parse_args(argv)
    result = build_cuda_group_catalog(
        batch_directories=arguments.批次目录,
        output_path=arguments.输出,
        cuda_budget_sha256=arguments.CUDA资格预算SHA256,
        cuda_tar_sha256=arguments.CUDA资格源码tarSHA256,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
