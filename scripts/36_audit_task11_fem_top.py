#!/usr/bin/env python
"""任11：先登记LF训练原件，再做仅顶部合法HF验证FEM插值。"""

from __future__ import annotations

import argparse
import json

from sic_cu.eval.task11_fem_top import register_lf_fem_sources, run_lf_fem_top_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="任11 LF60 FEM受限顶部对照；不访问旧测试")
    sub = parser.add_subparsers(dest="action", required=True)
    register = sub.add_parser("register")
    register.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--registration-file", required=True)
    evaluate.add_argument("--registration-sha256", required=True)
    args = parser.parse_args()
    report = (register_lf_fem_sources(args.output) if args.action == "register" else
              run_lf_fem_top_validation(args.output, args.registration_file,
                                        args.registration_sha256))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
