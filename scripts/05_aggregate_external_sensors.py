#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from sic_cu.eval.aggregate_runs import aggregate_external_sensor_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate 5-seed external sensor tests")
    parser.add_argument(
        "--run-prefix",
        default="mf_pinn_nominal",
        help="Shared report/run prefix before _seedN",
    )
    parser.add_argument(
        "--output",
        default="reports/mf_pinn_nominal_external_sensor_5seed_summary.json",
    )
    args = parser.parse_args()
    inputs = [
        f"reports/{args.run_prefix}_seed{seed}_external_sensor.json"
        for seed in range(5)
    ]
    result = aggregate_external_sensor_runs(inputs, args.output)
    print(json.dumps(result["aggregate_across_seeds"], indent=2))


if __name__ == "__main__":
    main()
