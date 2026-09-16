#!/usr/bin/env python
"""任11五种子新MLP低保真源码/真源先锁与正式CUDA训练入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sic_cu.config import PROJECT_ROOT
from sic_cu.train.task11_mlp_lf_formal import (
    collect_task11_mlp_lf_sources, preflight_task11_mlp_lf,
    qualify_task11_mlp_lf_source, run_task11_mlp_lf_formal,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="任11新MLP低保真60/10独立真源五种子")
    parser.add_argument("--freeze-catalog", help="CPU只核真实60/10文件SHA，写全新项目内来源目录")
    parser.add_argument("--preflight-only", action="store_true",
                        help="仅CPU核三SHA、ROOT活动与真实70场；不探测CUDA或创建输出")
    parser.add_argument("--audit-finished", action="store_true",
                        help="仅CPU复核已完训同seed日志、完整状态、真成本和当前LF来源")
    parser.add_argument("--registry", help="事前新MLP LF预算YAML")
    parser.add_argument("--registry-sha", help="事前YAML的精确SHA256")
    parser.add_argument("--source-tar", help="事前全部普通源码原件冻结tar.gz")
    parser.add_argument("--source-tar-sha", help="全部冻结源码tar精确SHA256")
    parser.add_argument("--catalog", help="事前真实模拟训练/验证70原件SHA目录")
    parser.add_argument("--catalog-sha", help="真实模拟70原件目录精确SHA256")
    parser.add_argument("--seed", type=int, choices=range(5))
    parser.add_argument("--output", help="事前同seed固定项目内全新输出目录")
    parser.add_argument("--resume-checkpoint", help="只接受本seed正式目录中的阶段_最近.pt")
    parser.add_argument("--session-epoch-limit", type=int, default=200)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if args.freeze_catalog:
        if args.preflight_only or args.audit_finished:
            parser.error("CPU真源目录冻结不得混用只读审计或正式门禁")
        if any(value is not None for value in
               (args.registry, args.registry_sha, args.source_tar,
                args.source_tar_sha, args.catalog, args.catalog_sha,
                args.seed, args.output, args.resume_checkpoint)):
            parser.error("CPU真源目录冻结模式不能混用正式训练身份或输出")
        path = Path(args.freeze_catalog)
        destination = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
        if (PROJECT_ROOT not in destination.parents or destination.exists()
                or path.is_symlink()):
            parser.error("任11真源目录只能写入全新项目内文件")
        catalog = collect_task11_mlp_lf_sources()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(catalog, ensure_ascii=False, sort_keys=True,
                                          indent=2) + "\n", encoding="utf-8")
        checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
        print(json.dumps({"CPU真源事前目录": str(destination.relative_to(PROJECT_ROOT)),
                          "SHA256": checksum, "真实训练仿真功率数": 60,
                          "真实合法验证仿真功率数": 10,
                          "模拟测试功率温度读取": False}, ensure_ascii=False))
        return
    required = ("registry", "registry_sha", "source_tar", "source_tar_sha",
                "catalog", "catalog_sha", "seed", "output")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("正式CUDA运行缺事前身份：" + ", ".join(missing))
    if args.preflight_only and args.audit_finished:
        parser.error("任11 LF运行CPU前置与已完成资格审计不能混用")
    if args.preflight_only:
        result = preflight_task11_mlp_lf(
            registry=args.registry, registry_sha=args.registry_sha,
            source_tar=args.source_tar, source_tar_sha=args.source_tar_sha,
            catalog=args.catalog, catalog_sha=args.catalog_sha,
            output=args.output, seed=args.seed,
            resume_checkpoint=args.resume_checkpoint,
        )
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.audit_finished:
        if args.resume_checkpoint is not None:
            parser.error("任11 LF已完成CPU审计不能接续训练")
        audited = qualify_task11_mlp_lf_source(
            registry=args.registry, registry_sha=args.registry_sha,
            source_tar=args.source_tar, source_tar_sha=args.source_tar_sha,
            catalog=args.catalog, catalog_sha=args.catalog_sha,
            output=args.output, seed=args.seed,
        )
        print(json.dumps(audited, ensure_ascii=False))
        return
    receipt = run_task11_mlp_lf_formal(
        registry=args.registry, registry_sha=args.registry_sha,
        source_tar=args.source_tar, source_tar_sha=args.source_tar_sha,
        catalog=args.catalog, catalog_sha=args.catalog_sha,
        output=args.output, seed=args.seed,
        device_name=args.device, session_epoch_limit=args.session_epoch_limit,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
