#!/usr/bin/env python
"""一次生成任09主序列剩余14模型v3事前登记。"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from sic_cu.data.common import sha256_file
from sic_cu.train.task09_subset_formal_v3 import (
    TASK09_V3_REGISTRY,
    expected_task09_v3_registration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="任09剩余14模型v3来源与预算事前登记")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--dry-run", action="store_true")
    operation.add_argument("--output")
    options = parser.parse_args()
    if options.output is not None and Path(options.output).resolve() != TASK09_V3_REGISTRY.resolve():
        raise ValueError("任09V3登记只可写项目内专属新YAML，禁止改v1/v2冻结原件")
    encoded = yaml.safe_dump(
        expected_task09_v3_registration(), allow_unicode=True, sort_keys=True,
    )
    if options.dry_run:
        print(encoded, end="")
        return
    with TASK09_V3_REGISTRY.open("x", encoding="utf-8") as output:
        output.write(encoded)
    print(f"任09V3事前YAML原件SHA256 {sha256_file(TASK09_V3_REGISTRY)}")


if __name__ == "__main__":
    main()
