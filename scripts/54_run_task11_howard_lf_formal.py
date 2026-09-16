#!/usr/bin/env python
"""任11 Howard独立LF有限预算预检、正式单CUDA会话与只读终态审核。"""

from __future__ import annotations

import argparse
import json

from sic_cu.train.task11_howard_lf_formal import (
    REGISTRY, preflight_task11_howard_lf, qualify_task11_howard_lf_source,
    run_task11_howard_lf_formal,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Howard LF统一有限预算项目适配；非原文三网联合训练复现")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true", help="仅CPU核自身ROOT与源码/真实70源；不探测CUDA或创建输出")
    mode.add_argument("--audit-finished", action="store_true", help="只读CPU重推本人完整终态、历史best/耐心/AdamW/RNG/成本")
    parser.add_argument("--registry", default=REGISTRY, help="固定任11 Howard LF事前登记YAML")
    parser.add_argument("--registry-sha", required=True, help="独立事前YAML精确SHA256")
    parser.add_argument("--source-tar", required=True, help="全部普通源码事前冻结tar.gz")
    parser.add_argument("--source-tar-sha", required=True, help="全部普通源码tar精确SHA256")
    parser.add_argument("--catalog", required=True, help="共同真实60/10 LF源catalog原件，不读取模拟TEST温度")
    parser.add_argument("--catalog-sha", required=True, help="真实70源catalog精确SHA256")
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--output", required=True, help="本人seed固定正式Howard_LF_seed目录")
    parser.add_argument("--resume-checkpoint", help="仅本人seed阶段_最近.pt；不能借其他LF/HF初始化")
    parser.add_argument("--session-epoch-limit", type=int, default=200)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if args.audit_finished and args.resume_checkpoint is not None:
        parser.error("Howard LF CPU已完成审核不能混用续跑")
    identity = {"registry": args.registry, "registry_sha": args.registry_sha,
                "source_tar": args.source_tar, "source_tar_sha": args.source_tar_sha,
                "catalog": args.catalog, "catalog_sha": args.catalog_sha,
                "seed": args.seed, "output": args.output}
    if args.preflight_only:
        result = preflight_task11_howard_lf(**identity, resume_checkpoint=args.resume_checkpoint)
    elif args.audit_finished:
        result = qualify_task11_howard_lf_source(**identity)
    else:
        result = run_task11_howard_lf_formal(**identity, device_name=args.device,
                                            session_epoch_limit=args.session_epoch_limit,
                                            resume_checkpoint=args.resume_checkpoint)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
