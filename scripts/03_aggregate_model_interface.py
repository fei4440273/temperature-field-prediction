#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from sic_cu.eval.aggregate_runs import aggregate_interface_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate interface metrics across seeds")
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = aggregate_interface_runs(args.results, args.output)
    print(json.dumps(result["aggregate_across_seeds"], indent=2))


if __name__ == "__main__":
    main()
