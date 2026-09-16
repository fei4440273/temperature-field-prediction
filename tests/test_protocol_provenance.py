from __future__ import annotations

import pytest

from sic_cu.data.splits import build_power_splits
from sic_cu.eval.protocol_checks import (
    checkpoint_provenance,
    validate_hf_checkpoint_provenance,
    validate_lf_checkpoint_provenance,
)


FINGERPRINTS = {
    "protocol_id": "hf_fixed_12_3_3_v4",
    "split_sha256": "split",
    "raw_manifest_sha256": "raw",
    "processed_manifest_sha256": "processed",
    "physics_config_sha256": "physics",
    "selection_metric_version": "macro_v1",
    "code_commit": "commit",
}


def test_lf_checkpoint_requires_exact_fixed_source() -> None:
    splits = build_power_splits()
    provenance = checkpoint_provenance(
        role="low_fidelity_simulation",
        train_powers_w=splits.simulation_train,
        validation_powers_w=splits.simulation_validation,
        fingerprints=FINGERPRINTS,
    )
    validate_lf_checkpoint_provenance(
        {"provenance": provenance}, splits, FINGERPRINTS
    )

    provenance["train_powers_w"] = provenance["train_powers_w"] + [90.0]
    with pytest.raises(RuntimeError, match="train powers"):
        validate_lf_checkpoint_provenance(
            {"provenance": provenance}, splits, FINGERPRINTS
        )


def test_old_checkpoint_without_provenance_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="no protocol provenance"):
        validate_lf_checkpoint_provenance({}, fingerprints=FINGERPRINTS)
    with pytest.raises(RuntimeError, match="no protocol provenance"):
        validate_hf_checkpoint_provenance({}, fingerprints=FINGERPRINTS)


def test_hf_provenance_rejects_test_as_training_power() -> None:
    splits = build_power_splits()
    provenance = checkpoint_provenance(
        role="high_fidelity_multifidelity",
        train_powers_w=splits.hf_train,
        validation_powers_w=splits.hf_validation,
        fingerprints=FINGERPRINTS,
    )
    provenance["train_powers_w"] = sorted(splits.hf_train | {169.0})
    with pytest.raises(RuntimeError, match="HF train powers"):
        validate_hf_checkpoint_provenance(
            {"provenance": provenance}, splits, FINGERPRINTS
        )


def test_hf_provenance_accepts_only_declared_nested_training_subset() -> None:
    splits = build_power_splits()
    subset = [55.0, 364.3, 729.0]
    provenance = checkpoint_provenance(
        role="high_fidelity_multifidelity",
        train_powers_w=subset,
        validation_powers_w=splits.hf_validation,
        fingerprints=FINGERPRINTS,
    ) | {
        "global_hf_train_powers_w": sorted(splits.hf_train),
        "hf_training_subset_w": subset,
    }
    validate_hf_checkpoint_provenance(
        {"provenance": provenance}, splits, FINGERPRINTS
    )

    provenance["hf_training_subset_w"] = subset + [115.2]
    with pytest.raises(RuntimeError, match="non-training powers"):
        validate_hf_checkpoint_provenance(
            {"provenance": provenance}, splits, FINGERPRINTS
        )
