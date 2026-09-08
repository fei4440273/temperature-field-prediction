from __future__ import annotations

import argparse
import json

from sic_cu.eval.residual_interpolation import evaluate_all_protocols


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate leakage-safe complete-power residual interpolation"
    )
    parser.add_argument(
        "--output", default="reports/residual_interpolation_cv.json"
    )
    args = parser.parse_args()
    result = evaluate_all_protocols(args.output)
    summary = {
        "fixed_selected_method": result["fixed_split"]["selected_method"],
        "fixed_test": result["fixed_split"]["test"]["aggregate"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
