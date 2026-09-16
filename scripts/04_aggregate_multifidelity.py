#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from sic_cu.eval.aggregate_runs import aggregate_multifidelity_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate locked 5-seed MF-PINN runs")
    parser.add_argument(
        "--run-prefix",
        default="mf_pinn_nominal",
        help="Shared report/run prefix before _seedN",
    )
    parser.add_argument(
        "--output", default="reports/mf_pinn_nominal_5seed_summary.json"
    )
    args = parser.parse_args()
    runs = [f"reports/runs/{args.run_prefix}_seed{seed}" for seed in range(5)]
    ir_evaluations = [
        f"reports/{args.run_prefix}_seed{seed}_ir_test.json" for seed in range(5)
    ]
    result = aggregate_multifidelity_runs(runs, ir_evaluations, args.output)
    print(json.dumps(result["aggregate_across_seeds"], indent=2))
    print(json.dumps(result["acceptance"], indent=2))


if __name__ == "__main__":
    main()
