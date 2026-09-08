#!/usr/bin/env python
import argparse
import json

from sic_cu.eval.external_sensor import evaluate_external_sensors
from sic_cu.physics.materials import PhysicsConfigurationError


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen external Hot/Cold test")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="reports/external_sensor_evaluation.json")
    parser.add_argument("--device")
    args = parser.parse_args()
    try:
        result = evaluate_external_sensors(args.checkpoint, args.output, args.device)
    except PhysicsConfigurationError as error:
        parser.exit(2, f"BLOCKED_UNVERIFIED_SENSOR_METADATA: {error}\n")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
