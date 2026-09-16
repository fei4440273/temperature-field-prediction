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
    parser.add_argument("--release-manifest")
    args = parser.parse_args()
    result = evaluate_all_protocols(args.output, args.release_manifest)
    summary = {
        "fixed_selected_method": result["fixed_split"]["selected_method"],
        "fixed_validation": result["fixed_split"]["validation_method_comparison"][
            result["fixed_split"]["selected_method"]
        ]["aggregate"],
        "fixed_test": (
            None
            if result["fixed_split"]["test"] is None
            else result["fixed_split"]["test"]["aggregate"]
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
