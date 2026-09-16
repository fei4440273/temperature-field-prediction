#!/usr/bin/env python
"""任09剩余14模型V3谱系强门禁后的正式CUDA能源入口。"""

from __future__ import annotations

import argparse
import json

from sic_cu.train.task09_subset_formal_v3 import audit_task09_v3_formal_energy


def main() -> None:
    parser = argparse.ArgumentParser(description="任09V3 primary剩余14模型best/final独立能源入口")
    parser.add_argument("--arm", required=True, choices=("primary",))
    parser.add_argument("--size", required=True, type=int, choices=(3, 6, 9))
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state", required=True, choices=("best", "final"))
    parser.add_argument("--v3-registry", required=True)
    parser.add_argument("--v3-registry-sha256", required=True)
    parser.add_argument("--v3-source-archive", required=True)
    parser.add_argument("--v3-source-archive-sha256", required=True)
    parser.add_argument("--original-registry", required=True)
    parser.add_argument("--original-registry-sha256", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    options = parser.parse_args()
    payload = audit_task09_v3_formal_energy(
        run=options.run, name=options.arm, size=options.size, seed=options.seed,
        state=options.state, output=options.output, device_name=options.device,
        v3_registry_path=options.v3_registry,
        v3_registry_sha256=options.v3_registry_sha256,
        v3_source_archive=options.v3_source_archive,
        v3_source_archive_sha256=options.v3_source_archive_sha256,
        original_registry_path=options.original_registry,
        original_registry_sha256=options.original_registry_sha256,
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
