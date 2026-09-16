#!/usr/bin/env python
"""任11新MLP真实完训最佳/末态独立30点双阶能源入口。"""

from __future__ import annotations

import argparse
import json

from sic_cu.eval.task11_mlp_hf_energy import audit_task11_mlp_hf_energy


def main() -> None:
    parser = argparse.ArgumentParser(description="任11新MLP best/final原瓦数只读CUDA独审")
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--state", required=True, choices=("best", "final"))
    parser.add_argument("--registry", required=True)
    parser.add_argument("--registry-sha", required=True)
    parser.add_argument("--source-tar", required=True)
    parser.add_argument("--source-tar-sha", required=True)
    parser.add_argument("--lf-catalog", required=True)
    parser.add_argument("--lf-catalog-sha", required=True)
    parser.add_argument("--hf-data-catalog", required=True)
    parser.add_argument("--hf-data-catalog-sha", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    arguments = parser.parse_args()
    result = audit_task11_mlp_hf_energy(
        run=arguments.run, output=arguments.output, seed=arguments.seed,
        state=arguments.state, registry=arguments.registry,
        registry_sha=arguments.registry_sha, source_tar=arguments.source_tar,
        source_tar_sha=arguments.source_tar_sha, lf_catalog=arguments.lf_catalog,
        lf_catalog_sha=arguments.lf_catalog_sha,
        hf_data_catalog=arguments.hf_data_catalog,
        hf_data_catalog_sha=arguments.hf_data_catalog_sha, device_name=arguments.device,
    )
    print(json.dumps({key: result[key] for key in (
        "运行种子", "审核状态", "绝对平衡宏均值_瓦", "绝对平衡95分位_瓦",
        "吸收功率归一宏均值", "吸收功率归一95分位",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
