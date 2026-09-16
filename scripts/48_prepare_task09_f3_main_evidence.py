"""Freeze a complete evidence catalog, without running formal aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from sic_cu.data.common import sha256_file
from sic_cu.eval.task09_f3_main_aggregate import (
    TASK09_ROOT, prepare_task09_main_evidence_from_prereg,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="任09 F3主线十五份完训后的逐SHA证据封存")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    expected = TASK09_ROOT / "正式F3主序列十五身份全原件证据清单.yaml"
    if destination != expected or destination.exists() or destination.is_symlink():
        raise ValueError("任09全身份证据清单只准事前已锁的唯一项目内新路径，不能覆盖")
    evidence = prepare_task09_main_evidence_from_prereg(
        registry_path=args.registry, registry_sha=args.registry_sha256,
        archive_path=args.source_archive, archive_sha=args.source_archive_sha256,
    )
    with destination.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(evidence, stream, allow_unicode=True, sort_keys=False)
    print(json.dumps({"清单路径": str(destination),
                      "清单原件SHA256": sha256_file(destination),
                      "固定身份数": 15, "正式观测聚合": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
