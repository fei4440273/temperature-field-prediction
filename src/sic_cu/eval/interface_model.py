from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.fields import load_processed_field
from sic_cu.data.splits import build_power_splits
from sic_cu.eval.interface_audit import _paired_interface_nodes
from sic_cu.eval.protocol_checks import (
    validate_lf_checkpoint_provenance,
    validate_release_checkpoint,
)
from sic_cu.models import ModelScales
from sic_cu.train.simulation import build_model, predict_field


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "mae_c": float(np.mean(np.abs(error))),
        "rmse_c": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_c": float(np.max(np.abs(error))),
        "target_mean_abs_jump_c": float(np.mean(np.abs(target))),
        "prediction_mean_abs_jump_c": float(np.mean(np.abs(prediction))),
    }


def evaluate_pointwise_interface_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path,
    powers: list[float] | None = None,
    device: str = "cpu",
    release_manifest_path: str | None = None,
) -> dict[str, Any]:
    checkpoint = PROJECT_ROOT / checkpoint_path
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    method = str(payload["method"])
    if method not in {
        "mlp",
        "mlp_pinn",
        "lstm",
        "lstm_pinn",
        "deeponet",
        "deeponet_pinn",
    }:
        raise ValueError(f"Unsupported pointwise checkpoint method: {method}")
    model = build_model(
        method,
        ModelScales(**payload["scales"]),
        **payload.get("model_kwargs", {}),
    ).to(device)
    model.load_state_dict(payload["model_state"])

    splits = build_power_splits()
    evaluation_powers = powers or sorted(splits.simulation_validation)
    reads_test = bool(
        {round(float(power), 4) for power in evaluation_powers}
        & {round(float(power), 4) for power in splits.simulation_test}
    )
    if reads_test:
        if release_manifest_path is None:
            raise RuntimeError("Simulation test evaluation requires a frozen release")
        validate_lf_checkpoint_provenance(payload)
        validate_release_checkpoint(release_manifest_path, checkpoint)
    per_power: list[dict[str, Any]] = []
    all_target: list[np.ndarray] = []
    all_prediction: list[np.ndarray] = []
    for power in evaluation_powers:
        field = load_processed_field(power)
        prediction, _ = predict_field(model, power, torch.device(device))
        pairs = _paired_interface_nodes(field.coordinates_rz_m, field.material_ids)
        sic_indices = np.asarray([pair[0] for pair in pairs])
        copper_indices = np.asarray([pair[1] for pair in pairs])
        target_jump = (
            field.temperature_k[:, sic_indices] - field.temperature_k[:, copper_indices]
        )
        prediction_jump = prediction[:, sic_indices] - prediction[:, copper_indices]
        all_target.append(target_jump)
        all_prediction.append(prediction_jump)
        per_power.append(
            {
                "power_w": float(power),
                "paired_nodes": len(pairs),
                "metrics": _metrics(target_jump, prediction_jump),
            }
        )

    target = np.concatenate([values.reshape(-1) for values in all_target])
    prediction = np.concatenate([values.reshape(-1) for values in all_prediction])
    result = {
        "schema_version": 1,
        "temperature_error_unit": "℃",
        "checkpoint": str(Path(checkpoint_path)),
        "method": method,
        "seed": int(payload["seed"]),
        "material_aware": bool(payload.get("model_kwargs", {}).get("include_material", False)),
        "evaluation_domain": (
            "frozen simulation test powers, same-coordinate material-side pairs"
            if reads_test
            else "simulation validation powers, same-coordinate material-side pairs"
        ),
        "evaluation_powers_w": [float(power) for power in evaluation_powers],
        "aggregate": _metrics(target, prediction),
        "per_power": per_power,
        "interpretation": (
            "This evaluates the signed SiC-minus-copper temperature jump encoded by duplicate "
            "interface-side nodes. It does not identify contact resistance or heat flux."
        ),
        "material_passport": payload.get("material_passport", {}),
    }
    destination = PROJECT_ROOT / output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
