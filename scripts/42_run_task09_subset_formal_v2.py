#!/usr/bin/env python
"""任09 P1修订恢复：CPU准备与真CUDA续跑分离。"""

from __future__ import annotations

import argparse
import json

from sic_cu.train.task09_subset_formal_v2 import (
    FAILED_RUN_DIRECTORY,
    run_task09_formal_v2,
    stage_task09_v2_recovery,
)


def _identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arm", required=True,
                        choices=("primary", "left_center", "right_center"))
    parser.add_argument("--size", required=True, type=int, choices=(3, 6, 9))
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--output", required=True)
    parser.add_argument("--repair-registry", required=True)
    parser.add_argument("--repair-registry-sha256", required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="任09 P1冻结LF AdamW门禁修订恢复")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="只读旧失败现场并新建corr500恢复目录")
    _identity(prepare)
    prepare.add_argument("--source", default=str(FAILED_RUN_DIRECTORY))

    run = commands.add_parser("run", help="录0075后从新目录真CUDA续跑")
    _identity(run)
    run.add_argument("--original-registry", required=True)
    run.add_argument("--original-registry-sha256", required=True)
    run.add_argument("--root-ledger-entry", required=True, choices=("录0075",))
    run.add_argument("--source-archive", required=True)
    run.add_argument("--source-archive-sha256", required=True)
    run.add_argument("--session-epoch-limit", required=True, type=int)
    run.add_argument("--resume-checkpoint", required=True)
    run.add_argument("--device", choices=("cuda",), default="cuda")
    options = parser.parse_args()

    if options.command == "prepare":
        result = stage_task09_v2_recovery(
            options.source, options.output, name=options.arm,
            size=options.size, seed=options.seed,
            v2_registry_path=options.repair_registry,
            v2_registry_sha256=options.repair_registry_sha256,
        )
    else:
        result = run_task09_formal_v2(
            name=options.arm, size=options.size, seed=options.seed,
            output_directory=options.output,
            v2_registry_path=options.repair_registry,
            v2_registry_sha256=options.repair_registry_sha256,
            original_registry_path=options.original_registry,
            original_registry_sha256=options.original_registry_sha256,
            root_ledger_entry=options.root_ledger_entry,
            session_epoch_limit=options.session_epoch_limit,
            resume_checkpoint=options.resume_checkpoint,
            device_name=options.device,
            v2_source_archive=options.source_archive,
            v2_source_archive_sha256=options.source_archive_sha256,
        )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
