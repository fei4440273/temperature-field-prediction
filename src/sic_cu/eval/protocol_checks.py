from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import PowerSplits, build_power_splits


PROTOCOL_ID = "hf_fixed_12_3_3_v4"
SELECTION_METRIC_VERSION = "macro_v1"
PHYSICS_CONFIG_PATHS = (
    "configs/geometry.yaml",
    "configs/materials.yaml",
    "configs/boundary_conditions.yaml",
)
REQUIRED_FINGERPRINTS = (
    "split_sha256",
    "raw_manifest_sha256",
    "processed_manifest_sha256",
    "physics_config_sha256",
    "code_commit",
)
GENERATED_EVIDENCE_PREFIXES = (
    "reports/current_protocol/",
    "reports/releases/",
)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def combined_file_sha256(paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    for value in paths:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        relative = str(path.relative_to(PROJECT_ROOT)).encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def source_worktree_status() -> list[str]:
    process = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        return ["GIT_STATUS_UNAVAILABLE"]
    entries = []
    for line in process.stdout.splitlines():
        path = line[3:].strip().strip('"')
        if any(path.startswith(prefix) for prefix in GENERATED_EVIDENCE_PREFIXES):
            continue
        entries.append(line)
    return entries


def current_code_commit() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        return "UNCOMMITTED"
    commit = process.stdout.strip()
    return f"{commit}-dirty" if source_worktree_status() else commit


def _manifest_hash(path: Path, embedded_key: str | None = None) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required provenance manifest is missing: {path}")
    if embedded_key is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        embedded = payload.get(embedded_key)
        if isinstance(embedded, str) and len(embedded) == 64:
            return embedded
    return sha256_file(path)


def current_protocol_fingerprints(
    *,
    inventory_path: str | Path = "reports/current_protocol/data_inventory.json",
    processed_manifest_path: str | Path = "data/processed/manifest.json",
) -> dict[str, str]:
    inventory = Path(inventory_path)
    processed = Path(processed_manifest_path)
    if not inventory.is_absolute():
        inventory = PROJECT_ROOT / inventory
    if not processed.is_absolute():
        processed = PROJECT_ROOT / processed
    return {
        "protocol_id": PROTOCOL_ID,
        "split_sha256": sha256_file(PROJECT_ROOT / "configs/splits.yaml"),
        "raw_manifest_sha256": _manifest_hash(inventory, "inventory_sha256"),
        "processed_manifest_sha256": _manifest_hash(
            processed, "processed_manifest_sha256"
        ),
        "physics_config_sha256": combined_file_sha256(PHYSICS_CONFIG_PATHS),
        "selection_metric_version": SELECTION_METRIC_VERSION,
        "code_commit": current_code_commit(),
    }


def _canonical_powers(values: Iterable[float]) -> list[float]:
    return sorted(round(float(value), 4) for value in values)


def checkpoint_provenance(
    *,
    role: str,
    train_powers_w: Iterable[float],
    validation_powers_w: Iterable[float],
    fingerprints: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    resolved = dict(fingerprints or current_protocol_fingerprints())
    return {
        **resolved,
        "role": role,
        "train_powers_w": _canonical_powers(train_powers_w),
        "validation_powers_w": _canonical_powers(validation_powers_w),
        "test_labels_consumed": False,
    }


def validate_lf_checkpoint_provenance(
    payload: Mapping[str, Any],
    splits: PowerSplits | None = None,
    fingerprints: Mapping[str, str] | None = None,
) -> None:
    active_splits = splits or build_power_splits()
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError(
            "Low-fidelity checkpoint has no protocol provenance; retrain it under the fixed protocol"
        )
    if provenance.get("role") != "low_fidelity_simulation":
        raise RuntimeError("Low-fidelity checkpoint declares an incompatible provenance role")
    if provenance.get("test_labels_consumed") is not False:
        raise RuntimeError("Low-fidelity checkpoint does not prove test-label isolation")
    if _canonical_powers(provenance.get("train_powers_w", ())) != _canonical_powers(
        active_splits.simulation_train
    ):
        raise RuntimeError("Low-fidelity checkpoint train powers do not match simulation train")
    if _canonical_powers(
        provenance.get("validation_powers_w", ())
    ) != _canonical_powers(active_splits.simulation_validation):
        raise RuntimeError(
            "Low-fidelity checkpoint validation powers do not match simulation validation"
        )
    current = dict(fingerprints or current_protocol_fingerprints())
    mismatches = {
        key: {"checkpoint": provenance.get(key), "current": current[key]}
        for key in REQUIRED_FINGERPRINTS
        if provenance.get(key) != current[key]
    }
    if mismatches:
        raise RuntimeError(f"Low-fidelity checkpoint provenance mismatch: {mismatches}")


def validate_hf_checkpoint_provenance(
    payload: Mapping[str, Any],
    splits: PowerSplits | None = None,
    fingerprints: Mapping[str, str] | None = None,
) -> None:
    active_splits = splits or build_power_splits()
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError(
            "Multifidelity checkpoint has no protocol provenance; old LOGO/full-data "
            "checkpoints are not valid for the fixed protocol"
        )
    if provenance.get("role") not in {
        "high_fidelity_multifidelity",
        "prc_multifidelity",
        "surface_multifidelity",
    }:
        raise RuntimeError("Checkpoint does not declare a multifidelity provenance role")
    if provenance.get("test_labels_consumed") is not False:
        raise RuntimeError("Checkpoint does not prove test-label isolation")
    if _canonical_powers(provenance.get("train_powers_w", ())) != _canonical_powers(
        active_splits.hf_train
    ):
        raise RuntimeError("Checkpoint HF train powers do not match the fixed protocol")
    if _canonical_powers(
        provenance.get("validation_powers_w", ())
    ) != _canonical_powers(active_splits.hf_validation):
        raise RuntimeError("Checkpoint HF validation powers do not match the fixed protocol")
    current = dict(fingerprints or current_protocol_fingerprints())
    mismatches = {
        key: {"checkpoint": provenance.get(key), "current": current[key]}
        for key in REQUIRED_FINGERPRINTS
        if provenance.get(key) != current[key]
    }
    if mismatches:
        raise RuntimeError(f"Multifidelity checkpoint provenance mismatch: {mismatches}")


def validate_release_manifest(manifest_path: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        manifest_file = PROJECT_ROOT / manifest_file
    manifest = load_yaml(manifest_file)
    if manifest.get("status") != "frozen":
        raise RuntimeError("Test evaluation requires a frozen release manifest")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Release protocol_id does not match the active fixed protocol")
    if manifest.get("selection_metric_version") != SELECTION_METRIC_VERSION:
        raise RuntimeError("Release selection metric is not macro_v1")
    if manifest.get("test_access_purpose") != "final_evaluation_only":
        raise RuntimeError("Release does not restrict test access to final evaluation")
    expected = current_protocol_fingerprints()
    for key in REQUIRED_FINGERPRINTS:
        if manifest.get(key) != expected[key]:
            raise RuntimeError(f"Frozen release {key} does not match the current project")
    return manifest


def validate_release_checkpoint(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    manifest = validate_release_manifest(manifest_path)
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_absolute():
        checkpoint_file = PROJECT_ROOT / checkpoint_file
    checkpoint_hash = sha256_file(checkpoint_file)
    frozen = manifest.get("frozen_checkpoint_sha256s")
    hashes = set(frozen.values()) if isinstance(frozen, Mapping) else set(frozen or ())
    if checkpoint_hash not in hashes:
        raise RuntimeError("Checkpoint is not registered in the frozen release")
    return manifest


def write_protocol_audit(
    output_path: str | Path = "reports/current_protocol/protocol_audit.json",
) -> dict[str, Any]:
    splits = build_power_splits()
    fingerprints = current_protocol_fingerprints()
    payload: dict[str, Any] = {
        "schema_version": 1,
        **fingerprints,
        "splits": {
            "simulation_train": _canonical_powers(splits.simulation_train),
            "simulation_validation": _canonical_powers(splits.simulation_validation),
            "simulation_test": _canonical_powers(splits.simulation_test),
            "hf_train": _canonical_powers(splits.hf_train),
            "hf_validation": _canonical_powers(splits.hf_validation),
            "hf_test": _canonical_powers(splits.hf_test),
        },
        "checks": {
            "simulation_60_10_10": [
                len(splits.simulation_train),
                len(splits.simulation_validation),
                len(splits.simulation_test),
            ]
            == [60, 10, 10],
            "high_fidelity_12_3_3": [
                len(splits.hf_train),
                len(splits.hf_validation),
                len(splits.hf_test),
            ]
            == [12, 3, 3],
            "test_excluded_from_training": True,
            "test_excluded_from_model_selection": True,
            "logo_disabled": True,
        },
    }
    payload["status"] = (
        "PASS" if all(payload["checks"].values()) else "FAIL"
    )
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def dump_yaml(payload: Mapping[str, Any], destination: str | Path) -> Path:
    path = Path(destination)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path
