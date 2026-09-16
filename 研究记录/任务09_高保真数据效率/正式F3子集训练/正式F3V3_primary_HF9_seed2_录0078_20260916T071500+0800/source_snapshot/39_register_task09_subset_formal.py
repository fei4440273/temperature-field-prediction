#!/usr/bin/env python
"""任09源/预算结构化事前登记，只许项目内新原件一次生成。"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from sic_cu.data.common import sha256_file
from sic_cu.train.task09_subset_formal import TASK09_REGISTRY, expected_task09_registration


def main() -> None:
    parser = argparse.ArgumentParser(description="任09三序列F3来源和计算预算事前YAML登记")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--dry-run", action="store_true")
    operation.add_argument("--output")
    options = parser.parse_args()
    if options.output is not None and Path(options.output).resolve() != TASK09_REGISTRY.resolve():
        raise ValueError("任09独立事前原件仅可写项目专属新YAML，不触碰任07/旧任09文件")
    original = expected_task09_registration()
    encoded = yaml.safe_dump(original, allow_unicode=True, sort_keys=True)
    if options.dry_run:
        print(encoded, end="")
        return
    with TASK09_REGISTRY.open("x", encoding="utf-8") as output:
        output.write(encoded)
    print(f"任09正式事前YAML原件SHA256 {sha256_file(TASK09_REGISTRY)}")


if __name__ == "__main__":
    main()
