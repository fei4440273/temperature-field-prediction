from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sic_cu.prediction import Predictor
from sic_cu.visualization import export_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict and export a SiC-Cu transient field")
    parser.add_argument("--power", type=float, required=True)
    parser.add_argument("--t-start", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--interpolation", choices=("linear", "cubic"), default="linear")
    parser.add_argument("--theta-resolution", type=int, default=72)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.dt <= 0 or args.t_end < args.t_start:
        parser.error("Require dt > 0 and t_end >= t_start")
    times = np.arange(args.t_start, args.t_end + args.dt * 0.5, args.dt)
    predictor = Predictor(checkpoint=args.checkpoint, interpolation_kind=args.interpolation)
    prediction = predictor.predict(
        args.power,
        times_s=times,
        theta_resolution=args.theta_resolution,
    )
    output = Path(args.output or f"predictions/{args.power:g}W")
    result = export_prediction(prediction, output, args.theta_resolution)
    print(json.dumps(result | {"warnings": prediction.metadata.warnings}, indent=2))
