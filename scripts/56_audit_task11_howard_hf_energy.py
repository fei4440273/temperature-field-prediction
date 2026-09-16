#!/usr/bin/env python
"""任11 Howard本人真实best/final独立六SHA能源门禁入口。"""

from __future__ import annotations

import argparse
import json

from sic_cu.eval.task11_howard_hf_energy import audit_task11_howard_hf_energy


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Howard HF本人30点16/64阶只读CUDA名义能源审核")
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    parser.add_argument("--state", choices=("best", "final"), required=True)
    for name in ("energy-registry", "energy-registry-sha", "energy-source-tar", "energy-source-tar-sha",
                 "registry", "registry-sha", "source-tar", "source-tar-sha", "lf-catalog", "lf-catalog-sha",
                 "hf-data-catalog", "hf-data-catalog-sha"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    arguments = vars(parser.parse_args(argv))
    arguments["device_name"] = arguments.pop("device")
    result = audit_task11_howard_hf_energy(**arguments)
    print(json.dumps({n: result[n] for n in ("运行种子", "审核状态", "输出目录", "名义吸收归一筛查",
        "吸收功率归一宏均值", "吸收功率归一95分位", "科学或工程安全资格")}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
