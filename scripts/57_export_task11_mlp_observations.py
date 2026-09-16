#!/usr/bin/env python
"""任11新MLP五seed双状态合法实测三模态只读CUDA观察导出。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sic_cu.eval.task11_mlp_observations import export_task11_mlp_observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observation-registry", "observation-registry-sha", "observation-source-tar",
                 "observation-source-tar-sha", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    args = parser.parse_args()
    result = export_task11_mlp_observations(
        observation_registry=args.observation_registry, observation_registry_sha=args.observation_registry_sha,
        observation_source_tar=args.observation_source_tar, observation_source_tar_sha=args.observation_source_tar_sha,
        output=args.output, device_name=args.device)
    print(json.dumps({"状态": result["状态"], "总点数": result["总点数"], "验收计数": result["验收计数"]},
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
