#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys

from sic_cu.eval.development_v5 import export_v5_m2_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="只读导出 V5 M2-E/M2-L/M2-O 审计，不训练且不读取测试温度。"
    )
    parser.add_argument("--config", default="configs/diagnostic_v5.yaml")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    destination = export_v5_m2_audit(
        config_path=args.config,
        output_root=args.output_root,
        effective_cli=["python", "scripts/08_export_v5_m2_audit.py", *sys.argv[1:]],
    )
    print(destination)


if __name__ == "__main__":
    main()
