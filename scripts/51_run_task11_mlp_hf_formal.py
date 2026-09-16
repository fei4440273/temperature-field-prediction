#!/usr/bin/env python
"""任11新MLP HF来源目录、CPU门禁和完整CUDA分段入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sic_cu.config import PROJECT_ROOT
from sic_cu.train.task11_mlp_hf_formal import (
    collect_task11_mlp_hf_data_catalog, collect_task11_mlp_lf_catalog,
    preflight_task11_mlp_hf, qualify_task11_mlp_hf_source, run_task11_mlp_hf_formal,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="任11新MLP五种子HF校正与全LF联合")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--collect-lf-catalog", help="仅CPU重核五新LF完训身份，写全新项目内目录文件")
    group.add_argument("--collect-hf-data-catalog", help="只读HF开发结构列/原SHA，不读温度或TEST")
    group.add_argument("--preflight-only", action="store_true", help="CPU核四SHA、ROOT、源码/源目录及自身恢复事务")
    group.add_argument("--audit-finished", action="store_true", help="CPU独立核完整HF完训状态、连续收据与真实成本")
    for name in ("registry", "registry-sha", "source-tar", "source-tar-sha",
                 "lf-catalog", "lf-catalog-sha", "hf-data-catalog", "hf-data-catalog-sha", "output"):
        parser.add_argument("--" + name)
    parser.add_argument("--seed", type=int, choices=range(5))
    parser.add_argument("--resume-checkpoint", help="仅接受自身正式目录阶段_最近.pt")
    parser.add_argument("--session-epoch-limit", type=int, default=200)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    collection = args.collect_lf_catalog or args.collect_hf_data_catalog
    names = ("registry", "registry_sha", "source_tar", "source_tar_sha", "lf_catalog", "lf_catalog_sha",
             "hf_data_catalog", "hf_data_catalog_sha", "output", "seed")
    if collection:
        if any(getattr(args, name) is not None for name in (*names, "resume_checkpoint")):
            parser.error("CPU目录采集不得混用正式身份/训练参数")
        supplied = Path(collection)
        destination = (supplied if supplied.is_absolute() else PROJECT_ROOT / supplied).resolve()
        if (PROJECT_ROOT not in destination.parents or destination.exists()
                or any(path.is_symlink() for path in (supplied, *supplied.parents))):
            parser.error("派生目录只能写入全新项目内普通文件，不可覆盖或软链")
        result = (collect_task11_mlp_lf_catalog() if args.collect_lf_catalog
                  else collect_task11_mlp_hf_data_catalog())
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        print(json.dumps({"CPU派生目录": str(destination.relative_to(PROJECT_ROOT)),
                          "SHA256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                          "旧固定TEST温度读取": False, "模拟测试功率温度读取": False}, ensure_ascii=False))
        return
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        parser.error("任11HF缺事前四SHA身份：" + ", ".join(missing))
    arguments = {name: getattr(args, name) for name in names}
    if args.preflight_only:
        result = preflight_task11_mlp_hf(**arguments, resume_checkpoint=args.resume_checkpoint)
        result.pop("续跑状态", None)
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.audit_finished:
        if args.resume_checkpoint is not None:
            parser.error("完训CPU审计不能接续训练")
        print(json.dumps(qualify_task11_mlp_hf_source(**arguments), ensure_ascii=False))
        return
    result = run_task11_mlp_hf_formal(**arguments, resume_checkpoint=args.resume_checkpoint,
                                    session_epoch_limit=args.session_epoch_limit, device_name=args.device)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
