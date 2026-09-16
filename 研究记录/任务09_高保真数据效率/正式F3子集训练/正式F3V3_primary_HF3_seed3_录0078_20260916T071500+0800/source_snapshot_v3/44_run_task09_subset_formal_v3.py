#!/usr/bin/env python
"""任09主序列剩余14模型通用正式v3真CUDA入口。"""

from __future__ import annotations

import argparse
import json

from sic_cu.train.task09_subset_formal_v3 import run_task09_formal_v3


def main() -> None:
    parser = argparse.ArgumentParser(description="任09 primary剩余14模型正式v3入口")
    parser.add_argument("--arm", required=True, choices=("primary",))
    parser.add_argument("--size", required=True, type=int, choices=(3, 6, 9))
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--output", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--original-registry", required=True)
    parser.add_argument("--original-registry-sha256", required=True)
    parser.add_argument("--root-ledger-entry", required=True, choices=("录0078",))
    parser.add_argument("--session-epoch-limit", required=True, type=int)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    options = parser.parse_args()
    report = run_task09_formal_v3(
        name=options.arm, size=options.size, seed=options.seed,
        output_directory=options.output,
        v3_registry_path=options.registry,
        v3_registry_sha256=options.registry_sha256,
        v3_source_archive=options.source_archive,
        v3_source_archive_sha256=options.source_archive_sha256,
        original_registry_path=options.original_registry,
        original_registry_sha256=options.original_registry_sha256,
        root_ledger_entry=options.root_ledger_entry,
        session_epoch_limit=options.session_epoch_limit,
        resume_checkpoint=options.resume_checkpoint,
        device_name=options.device,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
