#!/usr/bin/env python
"""Howard HF自身四SHA门禁：CPU预检、真单CUDA有限会话、CPU终态审计。"""

from __future__ import annotations

import argparse
import json

import torch

from sic_cu.train.task11_howard_hf_formal import (
    REGISTRY, SOURCE_TAR, LF_CATALOG, HF_DATA_CATALOG,
    preflight_task11_howard_hf, qualify_task11_howard_hf_source, run_task11_howard_hf_formal,
)


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return {"dtype": str(value.dtype), "device": str(value.device), "values": value.tolist()}
    raise TypeError("Howard HF CLI输出存在非JSON对象")


def main() -> None:
    parser = argparse.ArgumentParser(description="Howard适配三网从开始全56参数联合有限预算；非作者exact Adam复现")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true", help="仅CPU自身四SHA/五HowardLF资格/固定几何前检；不启动CUDA或读HF温度")
    mode.add_argument("--audit-finished", action="store_true", help="只读CPU重推本人全56 optimizer/RNG/best/final/历史和真实成本")
    parser.add_argument("--registry", default=REGISTRY)
    parser.add_argument("--registry-sha", required=True)
    parser.add_argument("--source-tar", default=SOURCE_TAR)
    parser.add_argument("--source-tar-sha", required=True)
    parser.add_argument("--lf-catalog", default=LF_CATALOG)
    parser.add_argument("--lf-catalog-sha", required=True)
    parser.add_argument("--hf-data-catalog", default=HF_DATA_CATALOG)
    parser.add_argument("--hf-data-catalog-sha", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-checkpoint", help="仅本人seed阶段_最近.pt；保留中断原件，禁止跨seed/重贴预算")
    parser.add_argument("--session-epoch-limit", type=int, default=200)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if args.audit_finished and args.resume_checkpoint is not None:
        parser.error("Howard HF只读终态审核不能混用恢复")
    identity = {"registry": args.registry, "registry_sha": args.registry_sha,
        "source_tar": args.source_tar, "source_tar_sha": args.source_tar_sha,
        "lf_catalog": args.lf_catalog, "lf_catalog_sha": args.lf_catalog_sha,
        "hf_data_catalog": args.hf_data_catalog, "hf_data_catalog_sha": args.hf_data_catalog_sha,
        "seed": args.seed, "output": args.output}
    if args.preflight_only:
        result = preflight_task11_howard_hf(**identity, resume_checkpoint=args.resume_checkpoint)
    elif args.audit_finished:
        result = qualify_task11_howard_hf_source(**identity)
    else:
        result = run_task11_howard_hf_formal(**identity, device_name=args.device,
            session_epoch_limit=args.session_epoch_limit, resume_checkpoint=args.resume_checkpoint)
    print(json.dumps(result, ensure_ascii=False, default=_json_value))


if __name__ == "__main__":
    main()
