#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from sic_cu.train.simulation import evaluate_saved_simulation_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing simulation checkpoint without retraining"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--template-metrics")
    parser.add_argument("--training-seconds", type=float)
    parser.add_argument("--release-manifest", required=True)
    args = parser.parse_args()
    result = evaluate_saved_simulation_model(
        args.checkpoint,
        args.output,
        device_name=args.device,
        batch_size=args.batch_size,
        template_metrics_path=args.template_metrics,
        training_seconds=args.training_seconds,
        release_manifest_path=args.release_manifest,
    )
    print(json.dumps(result["test"]["aggregate"], indent=2))


if __name__ == "__main__":
    main()
