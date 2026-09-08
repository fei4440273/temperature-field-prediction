#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add confirmed initial-condition and conservative extrapolation constraints"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/deployment/mf_pinn_nominal_seed1_constrained.pt"),
    )
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("method") != "multifidelity_correction":
        raise ValueError("Only a multi-fidelity correction checkpoint can be constrained")
    kwargs = dict(payload["correction_model_kwargs"])
    kwargs.update(
        {
            "hard_initial_temperature_k": 295.15,
            "initial_ramp_time_s": 0.05,
            "hard_minimum_temperature_k": 295.15,
            "minimum_temperature_beta_per_k": 10.0,
            "hard_cooling_radius_m": 0.05834,
            "hard_cooling_temperature_k": 295.15,
            "hard_cooling_tolerance_m": 1e-7,
            "correction_calibration_range_w": [115.2, 800.0],
            "correction_support_range_w": [0.0, 800.0],
            "correction_extrapolation_exponent": 2.0,
        }
    )
    payload["correction_model_kwargs"] = kwargs
    passport = dict(payload.get("material_passport", {}))
    passport.update(
        {
            "hard_initial_condition": "T(r,z,0,P)=295.15 K",
            "hard_minimum_temperature": "T(r,z,t,P)>=295.15 K",
            "hard_cooling_boundary": "T_Cu(r=0.05834 m,z,t,P)=295.15 K",
            "low_power_correction": (
                "quadratic shrinkage to low-fidelity prior from 115.2 W to 0 W"
            ),
            "deployment_status": "prototype_not_internal_field_validated",
            "source_checkpoint": str(args.checkpoint),
        }
    )
    payload["material_passport"] = passport
    payload["schema_version"] = max(int(payload.get("schema_version", 1)), 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
