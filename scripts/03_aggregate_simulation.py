#!/usr/bin/env python
import argparse
import json

from sic_cu.eval.aggregate_runs import aggregate_simulation_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate independent simulation-model seeds")
    parser.add_argument("runs", nargs="+", help="Run directories containing metrics.json")
    parser.add_argument("--output", default="reports/simulation_model_5seed_summary.json")
    args = parser.parse_args()
    print(json.dumps(aggregate_simulation_runs(args.runs, args.output), indent=2))


if __name__ == "__main__":
    main()
