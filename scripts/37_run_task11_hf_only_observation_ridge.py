#!/usr/bin/env python
"""任11轻量HF观测工程基线：来源、HF训练模型、合法验证分开锁。"""

from __future__ import annotations

import argparse
import json

from sic_cu.eval.task11_hf_observation_ridge import (
    evaluate_registered_hf_ridge, register_hf_ridge_sources, train_registered_hf_ridge,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="仅HF训练的固定Ridge工程观测基线")
    sub = parser.add_subparsers(dest="stage", required=True)
    source = sub.add_parser("register")
    source.add_argument("--output", required=True)
    train = sub.add_parser("train")
    train.add_argument("--output", required=True)
    train.add_argument("--registration-file", required=True)
    train.add_argument("--registration-sha256", required=True)
    verify = sub.add_parser("validate")
    verify.add_argument("--output", required=True)
    verify.add_argument("--registration-file", required=True)
    verify.add_argument("--registration-sha256", required=True)
    verify.add_argument("--model-file", required=True)
    verify.add_argument("--model-sha256", required=True)
    args = parser.parse_args()
    report = (register_hf_ridge_sources(args.output) if args.stage == "register" else
              train_registered_hf_ridge(args.output, args.registration_file,
                                        args.registration_sha256) if args.stage == "train" else
              evaluate_registered_hf_ridge(
                  args.output, args.registration_file, args.registration_sha256,
                  args.model_file, args.model_sha256))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
