#!/usr/bin/env python
"""任09自己F3子集正式状态的只读CUDA原30点16/64阶独立能源审核。"""

from __future__ import annotations

import argparse
import json

from sic_cu.eval.task09_energy import audit_task09_formal_energy


def main() -> None:
    parser = argparse.ArgumentParser(description="任09专属F3观测最佳/训练末名义原能源独审")
    parser.add_argument("--arm", required=True,
                        choices=("primary", "left_center", "right_center"))
    parser.add_argument("--size", required=True, type=int, choices=(3, 6, 9))
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state", required=True, choices=("best", "final"))
    parser.add_argument("--registry", required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    options = parser.parse_args()
    payload = audit_task09_formal_energy(
        options.run, name=options.arm, size=options.size, seed=options.seed,
        state=options.state, output=options.output, device_name=options.device,
        registry_path=options.registry,
        registry_sha256=options.registry_sha256,
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
