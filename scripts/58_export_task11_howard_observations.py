#!/usr/bin/env python
"""Howard本人五种子双状态合法实测三模态只读CUDA观察导出。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sic_cu.eval.task11_howard_observations import export_task11_howard_observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observation-registry", "observation-registry-sha", "observation-source-tar",
                 "observation-source-tar-sha", "registry", "registry-sha", "source-tar", "source-tar-sha",
                 "lf-catalog", "lf-catalog-sha", "hf-data-catalog", "hf-data-catalog-sha", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    args = vars(parser.parse_args())
    args["device_name"] = args.pop("device")
    result = export_task11_howard_observations(**args)
    print(json.dumps({"状态": result["状态"], "方法": result["方法"], "总点数": result["总点数"],
                     "验收计数": result["验收计数"]}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
