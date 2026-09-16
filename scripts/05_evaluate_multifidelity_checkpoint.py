from __future__ import annotations

import argparse
import json

from sic_cu.train.multifidelity import evaluate_multifidelity_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a selected MF checkpoint on its declared test powers"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--release-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = evaluate_multifidelity_checkpoint(
        args.checkpoint,
        args.release_manifest,
        output_path=args.output,
        device_name=args.device,
    )
    print(
        json.dumps(
            {"test_ir": result["test_ir"], "test_sensor": result["test_sensor"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
