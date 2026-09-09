#!/usr/bin/env python
import argparse
import json

from sic_cu.eval.aggregate_runs import aggregate_surface_residual_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate independent surface-residual seeds")
    parser.add_argument(
        "runs",
        nargs="*",
        default=[f"reports/runs/surface_residual_seed{seed}" for seed in range(5)],
    )
    parser.add_argument("--output", default="reports/surface_residual_5seed_summary.json")
    parser.add_argument("--release-manifest", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            aggregate_surface_residual_runs(
                args.runs, args.output, args.release_manifest
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
