#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import polars as pl

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.eval.energy_v5 import chinese_energy_terms


def main() -> None:
    parser = argparse.ArgumentParser(description="从已审计原始能量CSV导出中文分项")
    parser.add_argument("--source", default="reports/development_v5/m2_energy_terms.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--order", type=int, default=64)
    args = parser.parse_args()
    source = Path(args.source)
    destination = Path(args.output)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.mkdir(parents=True, exist_ok=True)
    csv_file = destination / "指标明细.csv"
    summary_file = destination / "汇总指标.json"
    if csv_file.exists() or summary_file.exists():
        raise FileExistsError("Already exported energy terms cannot be overwritten")
    print("本次只转换既有V5能量分项和原相对分母，不执行新求积或训练。")
    mapped, aggregate = chinese_energy_terms(pl.read_csv(source), args.order)
    csv_tmp = csv_file.with_name(csv_file.name + ".tmp")
    json_tmp = summary_file.with_name(summary_file.name + ".tmp")
    mapped.write_csv(csv_tmp)
    json_tmp.write_text(
        json.dumps(
            {"来源文件": str(source.relative_to(PROJECT_ROOT)), "来源SHA256": sha256_file(source),
             "求积阶数": args.order, "原始行数": mapped.height, **aggregate},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(csv_tmp, csv_file)
    os.replace(json_tmp, summary_file)
    print(json.dumps(aggregate, ensure_ascii=False))


if __name__ == "__main__":
    main()
