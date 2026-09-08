#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from sic_cu.train.multifidelity import evaluate_multifidelity_test_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen protocol-compatible MF checkpoint on test_Data"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output-json",
        default="reports/test_three_power_comparison.json",
    )
    parser.add_argument(
        "--output-csv",
        default="reports/test_three_power_comparison.csv",
    )
    parser.add_argument("--device")
    args = parser.parse_args()
    result = evaluate_multifidelity_test_data(
        args.checkpoint,
        output_json=args.output_json,
        output_csv=args.output_csv,
        device_name=args.device,
    )
    print(
        json.dumps(
            {"per_power": result["per_power"], "aggregate": result["aggregate"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
