#!/usr/bin/env python
import argparse
import json

from sic_cu.eval.high_fidelity import evaluate_ir_surface


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen IR power split")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output", default="reports/ir_surface_evaluation.json")
    parser.add_argument("--device")
    args = parser.parse_args()
    result = evaluate_ir_surface(args.checkpoint, args.split, args.output, args.device)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
