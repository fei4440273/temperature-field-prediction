from __future__ import annotations

import json
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.eval.protocol_checks import (
    current_protocol_fingerprints,
    dump_yaml,
    source_worktree_status,
    validate_hf_checkpoint_provenance,
    validate_lf_checkpoint_provenance,
)
from sic_cu.train.common import CONFIG_FILES


def _clean_tracked_worktree() -> bool:
    return not source_worktree_status()


def _environment_text() -> str:
    packages = ("numpy", "polars", "PyYAML", "scipy", "torch")
    lines = [
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
    ]
    for package in packages:
        try:
            lines.append(f"{package}={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{package}=NOT_INSTALLED")
    return "\n".join(lines) + "\n"


def freeze_release(
    release_id: str,
    checkpoint_paths: Iterable[str | Path],
    model_config_paths: Iterable[str | Path],
    command: str,
    *,
    output_root: str | Path = "reports/releases",
    require_all_seeds: bool = True,
    require_clean_worktree: bool = True,
) -> dict[str, Any]:
    if not release_id or any(character in release_id for character in "/\\"):
        raise ValueError("release_id must be one safe path component")
    if require_clean_worktree and not _clean_tracked_worktree():
        raise RuntimeError("Commit all code/config changes before freezing a release")
    checkpoints = [Path(path) for path in checkpoint_paths]
    configs = [Path(path) for path in model_config_paths]
    if not checkpoints or not configs:
        raise ValueError("At least one checkpoint and model config are required")
    checkpoints = [path if path.is_absolute() else PROJECT_ROOT / path for path in checkpoints]
    configs = [path if path.is_absolute() else PROJECT_ROOT / path for path in configs]
    if any(not path.is_file() for path in checkpoints + configs):
        missing = [str(path) for path in checkpoints + configs if not path.is_file()]
        raise FileNotFoundError(f"Release inputs are missing: {missing}")

    checkpoint_index = []
    seeds: set[int] = set()
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        role = payload.get("provenance", {}).get("role")
        if role == "low_fidelity_simulation":
            validate_lf_checkpoint_provenance(payload)
        elif role in {
            "high_fidelity_multifidelity",
            "prc_multifidelity",
            "surface_multifidelity",
        }:
            validate_hf_checkpoint_provenance(payload)
        else:
            raise RuntimeError(f"Checkpoint has no recognized protocol role: {path}")
        seed = int(payload["seed"])
        seeds.add(seed)
        checkpoint_index.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
                "method": str(payload["method"]),
                "seed": seed,
                "role": role,
                "validation_score_c": payload.get(
                    "validation_selection_score_c",
                    payload.get("validation_selection_score_k"),
                ),
            }
        )
    expected_seeds = set(int(value) for value in load_yaml("configs/training.yaml")["seeds"])
    if require_all_seeds and seeds != expected_seeds:
        raise RuntimeError(
            f"Frozen release seeds must be {sorted(expected_seeds)}, observed {sorted(seeds)}"
        )

    root = Path(output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    destination = root / release_id
    if destination.exists():
        raise FileExistsError(f"Release directory already exists: {destination}")
    snapshot = destination / "config_snapshot"
    snapshot.mkdir(parents=True)
    for relative in CONFIG_FILES:
        source = PROJECT_ROOT / relative
        shutil.copy2(source, snapshot / source.name)
    for source in configs:
        shutil.copy2(source, snapshot / source.name)

    fingerprints = current_protocol_fingerprints()
    manifest = {
        "schema_version": 1,
        "status": "frozen",
        **fingerprints,
        "frozen_model_configs": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in configs
        },
        "frozen_seeds": sorted(seeds),
        "frozen_checkpoint_sha256s": {
            item["path"]: item["sha256"] for item in checkpoint_index
        },
        "test_access_purpose": "final_evaluation_only",
        "historically_exposed_test": True,
    }
    dump_yaml(manifest, destination / "release_manifest.yaml")
    (destination / "checkpoint_index.json").write_text(
        json.dumps(checkpoint_index, indent=2), encoding="utf-8"
    )
    (destination / "environment.txt").write_text(_environment_text(), encoding="utf-8")
    (destination / "command.txt").write_text(command.strip() + "\n", encoding="utf-8")
    return manifest
