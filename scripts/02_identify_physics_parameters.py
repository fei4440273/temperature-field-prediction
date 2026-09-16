#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from sic_cu.physics.parameter_identification import identify_physics_parameters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate emissivities and SiC-Cu contact resistance from training simulations"
    )
    parser.add_argument(
        "--output",
        default="reports/physics_parameter_identification.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="reports/physics_parameter_identification.md",
    )
    args = parser.parse_args()
    result = identify_physics_parameters(args.output, args.markdown_output)
    print(json.dumps(result["fits"], indent=2))


if __name__ == "__main__":
    main()
