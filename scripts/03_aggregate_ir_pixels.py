#!/usr/bin/env python
import argparse
import json

from sic_cu.eval.aggregate_runs import aggregate_ir_pixel_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate independent 2D IR pixel evaluations")
    parser.add_argument("results", nargs="+", help="IR pixel evaluation JSON files")
    parser.add_argument(
        "--output", default="reports/ir_pixel_surface_residual_5seed_summary.json"
    )
    args = parser.parse_args()
    print(json.dumps(aggregate_ir_pixel_runs(args.results, args.output), indent=2))


if __name__ == "__main__":
    main()
