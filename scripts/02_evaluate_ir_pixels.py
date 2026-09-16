#!/usr/bin/env python
import argparse
import json

from sic_cu.eval.ir_pixels import evaluate_ir_pixels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reconstructed raw 2D IR pixel fields")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--checkpoint")
    checkpoint_group.add_argument("--surface-checkpoint")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--release-manifest")
    parser.add_argument("--output", default="reports/ir_pixel_evaluation.json")
    parser.add_argument("--figures", default="reports/figures/ir_pixels")
    parser.add_argument("--device")
    args = parser.parse_args()
    result = evaluate_ir_pixels(
        checkpoint=args.checkpoint,
        surface_checkpoint=args.surface_checkpoint,
        split=args.split,
        output_path=args.output,
        figure_directory=args.figures,
        device=args.device,
        release_manifest_path=args.release_manifest,
    )
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
