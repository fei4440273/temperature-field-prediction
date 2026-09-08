#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from sic_cu.eval.interface_model import evaluate_pointwise_interface_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a pointwise checkpoint on locked simulation interface jumps"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = evaluate_pointwise_interface_checkpoint(
        args.checkpoint,
        args.output,
        device=args.device,
    )
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
