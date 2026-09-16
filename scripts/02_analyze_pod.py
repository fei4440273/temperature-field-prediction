#!/usr/bin/env python
import argparse
import json

from sic_cu.eval.pod_analysis import run_pod_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit leakage-safe global/material POD bases")
    parser.add_argument("--output", default="data/cache/pod")
    parser.add_argument("--report", default="reports/pod_energy.json")
    parser.add_argument("--max-modes", type=int, default=50)
    parser.add_argument("--plot", default="reports/pod_energy.png")
    args = parser.parse_args()
    print(
        json.dumps(
            run_pod_analysis(args.output, args.report, args.max_modes, args.plot),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
