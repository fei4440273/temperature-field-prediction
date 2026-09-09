#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.eval.ir_pixels import evaluate_ir_pixels
from sic_cu.eval.protocol_checks import (
    validate_hf_checkpoint_provenance,
    validate_release_checkpoint,
)
from sic_cu.train.multifidelity import evaluate_multifidelity_test_data


METRIC_NAMES = ("mean_error_c", "mae_c", "rmse_c", "max_abs_error_c")


def _statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    result = {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }
    if len(array) == 5:
        result["ci95_half_width"] = float(2.776 * result["std"] / np.sqrt(5.0))
    return result


def _aggregate_five_seeds(
    results: list[dict[str, Any]],
    pixel_results: list[dict[str, Any]],
) -> dict[str, Any]:
    powers = results[0]["powers_w"]
    if any(result["powers_w"] != powers for result in results[1:]):
        raise RuntimeError("Final-test power order differs between seeds")
    per_power = []
    for index, power in enumerate(powers):
        seed_records = [result["per_power"][index] for result in results]
        modalities = {
            modality: {
                metric: _statistics(
                    [record["modalities"][modality][metric] for record in seed_records]
                )
                for metric in METRIC_NAMES
            }
            for modality in ("top_surface", "hot", "cold")
        }
        combined = {
            metric: _statistics(
                [record["combined"][metric] for record in seed_records]
            )
            for metric in METRIC_NAMES
        }
        per_power.append(
            {"power_w": float(power), "modalities": modalities, "combined": combined}
        )
    aggregate_names = tuple(results[0]["aggregate"])
    aggregate = {
        name: _statistics([float(result["aggregate"][name]) for result in results])
        for name in aggregate_names
    }
    pixel_names = tuple(pixel_results[0]["aggregate"])
    pixel_aggregate = {
        name: _statistics(
            [float(result["aggregate"][name]["mean"]) for result in pixel_results]
        )
        for name in pixel_names
    }
    return {
        "schema_version": 2,
        "temperature_error_unit": "℃",
        "split": "test",
        "selection_completed_before_test_access": True,
        "seed_statistics_interpretation": (
            "Optimization repeatability only; not experimental-population uncertainty"
        ),
        "seeds": sorted(int(result["seed"]) for result in results),
        "powers_w": powers,
        "per_power": per_power,
        "aggregate_across_seeds": aggregate,
        "ir_pixels_across_seeds": pixel_aggregate,
        "per_seed": [
            {
                "seed": int(result["seed"]),
                "checkpoint": result["checkpoint"],
                "comparison": result,
                "ir_pixel_aggregate": pixels["aggregate"],
                "ir_pixel_report": pixels["output_path"],
            }
            for result, pixels in zip(results, pixel_results, strict=True)
        ],
    }


def _write_aggregate_csv(result: dict[str, Any], output_path: str | Path) -> None:
    rows = []
    for item in result["per_power"]:
        row: dict[str, float] = {"power_w": float(item["power_w"])}
        for modality, metrics in item["modalities"].items():
            for metric, statistics in metrics.items():
                row.update(
                    {
                        f"{modality}_{metric}_{name}": float(value)
                        for name, value in statistics.items()
                    }
                )
        for metric, statistics in item["combined"].items():
            row.update(
                {
                    f"combined_{metric}_{name}": float(value)
                    for name, value in statistics.items()
                }
            )
        rows.append(row)
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(destination)


def _preflight(checkpoints: list[str], release_manifest: str) -> list[int]:
    expected = sorted(int(seed) for seed in load_yaml("configs/training.yaml")["seeds"])
    seeds = []
    model_signatures = set()
    for checkpoint in checkpoints:
        validate_release_checkpoint(release_manifest, checkpoint)
        path = Path(checkpoint)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_hf_checkpoint_provenance(payload)
        if payload.get("sensors_used") is not True:
            raise RuntimeError("Final three-power comparison requires Hot/Cold supervision")
        seeds.append(int(payload["seed"]))
        construction = payload.get(
            "model_kwargs", payload.get("correction_model_kwargs", {})
        )
        model_signatures.add(
            (str(payload.get("method")), json.dumps(construction, sort_keys=True))
        )
    if sorted(seeds) != expected or len(seeds) != len(set(seeds)):
        raise RuntimeError(
            f"Final test requires exactly the frozen seeds {expected}; observed {sorted(seeds)}"
        )
    if len(model_signatures) != 1:
        raise RuntimeError("Final seed aggregate cannot mix model architectures or variants")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate five frozen protocol-compatible checkpoints on test_Data once"
    )
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--release-manifest", required=True)
    parser.add_argument(
        "--output-json",
        default="reports/current_protocol/final_test_summary.json",
    )
    parser.add_argument(
        "--output-csv",
        default="reports/current_protocol/final_test_per_power.csv",
    )
    parser.add_argument("--device")
    args = parser.parse_args()
    seeds = _preflight(args.checkpoint, args.release_manifest)

    output_json = Path(args.output_json)
    output_root = output_json.parent
    results = []
    pixel_results = []
    for checkpoint, seed in sorted(zip(args.checkpoint, seeds), key=lambda item: item[1]):
        seed_json = output_root / f"final_test_seed{seed}.json"
        seed_csv = output_root / f"final_test_seed{seed}.csv"
        comparison = evaluate_multifidelity_test_data(
            checkpoint,
            args.release_manifest,
            output_json=seed_json,
            output_csv=seed_csv,
            device_name=args.device,
        )
        comparison["seed"] = seed
        pixel_path = output_root / f"final_test_ir_pixels_seed{seed}.json"
        pixels = evaluate_ir_pixels(
            checkpoint=checkpoint,
            split="test",
            output_path=pixel_path,
            figure_directory=(
                f"reports/current_protocol/figures/final_test_ir_pixels_seed{seed}"
            ),
            device=args.device,
            release_manifest_path=args.release_manifest,
        )
        pixels["output_path"] = str(pixel_path)
        results.append(comparison)
        pixel_results.append(pixels)

    aggregate = _aggregate_five_seeds(results, pixel_results)
    destination = output_json if output_json.is_absolute() else PROJECT_ROOT / output_json
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    _write_aggregate_csv(aggregate, args.output_csv)
    print(
        json.dumps(
            {
                "seeds": aggregate["seeds"],
                "per_power": aggregate["per_power"],
                "aggregate_across_seeds": aggregate["aggregate_across_seeds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
