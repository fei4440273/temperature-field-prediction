#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from sic_cu.data.splits import build_power_splits
from sic_cu.train.multifidelity import train_multifidelity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the selected MF-PINN with the fixed train/validation split"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lf-checkpoint", default="reports/runs/deeponet_data_seed0/best.pt"
    )
    parser.add_argument(
        "--output",
        default="reports/runs/mf_pinn_hard_surface_temperature_fixed_split_seed0",
    )
    args = parser.parse_args()
    splits = build_power_splits()
    result = train_multifidelity(
        low_fidelity_checkpoint=args.lf_checkpoint,
        output_directory=args.output,
        seed=args.seed,
        correction_epochs=300,
        joint_epochs=100,
        batch_size=4096,
        physics_collocation=128,
        width=64,
        depth=4,
        identify_physics_parameters=True,
        silicon_carbide_emissivity_initial=0.5,
        copper_emissivity_initial=0.5,
        contact_resistance_initial_m2_k_w=7.34072435302768e-5,
        evaluate_test=False,
        freeze_physics_parameters=True,
        hard_deployment_constraints=True,
        sensor_absolute_weight=5.0,
        sensor_delta_weight=1.0,
        hf_train_powers_w=splits.hf_train,
        hf_validation_powers_w=splits.hf_validation,
        hf_test_powers_w=splits.hf_test,
        correction_power_scaling="linear",
        correction_power_reference_w=400.0,
        correction_direct_power_input=False,
        hard_surface_residual_guide=True,
        checkpoint_selection="validation",
    )
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
