"""Task-07 source verification and reproducible, unselected HF initialization.

This module does not train, select an arm, or restore historical HF optimizer/RNG.
"""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import build_power_splits
from sic_cu.eval.protocol_checks import (
    canonical_json_sha256,
    validate_hf_checkpoint_provenance,
    validate_lf_checkpoint_provenance,
)
from sic_cu.models import AdditiveCorrectionModel, DeepONetPINN, ModelScales
from sic_cu.train.simulation import build_model


MANIFEST_PATH = PROJECT_ROOT / "reports/development_v4/baseline_manifest.yaml"
MANIFEST_SHA256 = "423d287de76058d7124c9072e8d094bc5c960e48f4e7b99d7246feda2988c4de"
SEEDS = frozenset(range(5))
RUN_ROOT = PROJECT_ROOT / "reports/runs/sic-cu-v2-gpu1-20260908T072729Z-r2"


@dataclass(frozen=True)
class Task07Source:
    seed: int
    lf_checkpoint_path: Path
    lf_checkpoint_sha256: str
    lf_tensor_sha256: str
    hf_checkpoint_path: Path
    hf_checkpoint_sha256: str
    lf_state: Mapping[str, Tensor]
    lf_model_kwargs: Mapping[str, Any]
    correction_model_kwargs: Mapping[str, Any]
    scales: Mapping[str, float]
    lf_train_powers_w: tuple[float, ...]
    lf_validation_powers_w: tuple[float, ...]
    hf_train_powers_w: tuple[float, ...]
    hf_validation_powers_w: tuple[float, ...]
    source_metadata_sha256: str


@dataclass(frozen=True)
class Task07Initialization:
    seed: int
    arm: str
    model: AdditiveCorrectionModel
    optimizer: torch.optim.AdamW
    random_state: Mapping[str, Any]
    lf_checkpoint_sha256: str
    lf_tensor_sha256: str
    selected_for_training: bool = False


def _powers(values: Any) -> tuple[float, ...]:
    return tuple(sorted(round(float(value), 4) for value in values))


def _lf_tensor_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(
            f"low_fidelity_model.{name}|{value.dtype}|{tuple(value.shape)}|".encode("ascii")
        )
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _source_metadata_sha256(source: Task07Source) -> str:
    return canonical_json_sha256({
        "seed": source.seed,
        "lf_checkpoint_path": str(source.lf_checkpoint_path),
        "lf_checkpoint_sha256": source.lf_checkpoint_sha256,
        "hf_checkpoint_path": str(source.hf_checkpoint_path),
        "hf_checkpoint_sha256": source.hf_checkpoint_sha256,
        "lf_model_kwargs": source.lf_model_kwargs,
        "correction_model_kwargs": source.correction_model_kwargs,
        "scales": source.scales,
        "lf_train_powers_w": source.lf_train_powers_w,
        "lf_validation_powers_w": source.lf_validation_powers_w,
        "hf_train_powers_w": source.hf_train_powers_w,
        "hf_validation_powers_w": source.hf_validation_powers_w,
    })


def _registered_rows() -> tuple[dict[str, Any], ...]:
    if sha256_file(MANIFEST_PATH) != MANIFEST_SHA256:
        raise ValueError("Task-07 V4 manifest byte SHA changed; revise source explicitly")
    manifest = load_yaml(MANIFEST_PATH)
    splits = build_power_splits()
    historical = manifest["data_and_protocol_hashes"]["checkpoint_training_provenance"]
    if (
        manifest.get("status") != "locked_existing_artifacts_with_historical_source_caveat"
        or manifest.get("protocol_id") != "hf_fixed_12_3_3_v4"
        or sha256_file(PROJECT_ROOT / "configs/splits.yaml") != historical["split_sha256"]
        or len(splits.simulation_train) != 60
        or len(splits.simulation_validation) != 10
        or len(splits.simulation_test) != 10
        or len(splits.hf_train) != 12
        or len(splits.hf_validation) != 3
        or len(splits.hf_test) != 3
    ):
        raise ValueError("Task-07 current LF60/10/10 or HF12/3/3 split differs from V4")
    entries = manifest["checkpoints"]
    if len(entries) != 5 or {entry["seed"] for entry in entries} != SEEDS:
        raise ValueError("Task-07 requires five distinct V4 B0 seed sources")
    for entry in entries:
        seed = entry["seed"]
        lf_path = PROJECT_ROOT / entry["lf_checkpoint"]
        hf_path = PROJECT_ROOT / entry["hf_checkpoint"]
        if (
            lf_path != RUN_ROOT / f"deeponet_pinn_lf_seed{seed}/best.pt"
            or hf_path != RUN_ROOT / f"deeponet_pinn_mf_seed{seed}/best.pt"
            or entry.get("selected_stage") != "correction_lf_frozen"
            or entry.get("test_labels_consumed") is not False
            or sha256_file(lf_path) != entry["lf_checkpoint_sha256"]
            or sha256_file(hf_path) != entry["hf_checkpoint_sha256"]
        ):
            raise ValueError(f"Task-07 seed{seed} locked LF/HF path or SHA is invalid")
    return tuple(sorted(entries, key=lambda entry: entry["seed"]))


def validate_task07_sources() -> dict[int, Task07Source]:
    """Read only the five V4 checkpoints, never train or test-label files."""
    entries = _registered_rows()  # Reject changed source files before opening any checkpoint.
    historical = load_yaml(MANIFEST_PATH)["data_and_protocol_hashes"][
        "checkpoint_training_provenance"
    ]
    splits = build_power_splits()
    sources: dict[int, Task07Source] = {}
    for entry in entries:
        seed = entry["seed"]
        lf_path = PROJECT_ROOT / entry["lf_checkpoint"]
        hf_path = PROJECT_ROOT / entry["hf_checkpoint"]
        lf = torch.load(lf_path, map_location="cpu", weights_only=False)
        hf = torch.load(hf_path, map_location="cpu", weights_only=False)
        fingerprints = historical | {"code_commit": entry["checkpoint_code_commit"]}
        if lf.get("seed") != seed or hf.get("seed") != seed:
            raise ValueError(f"Task-07 seed{seed} checkpoint seed metadata differs")
        validate_lf_checkpoint_provenance(lf, splits=splits, fingerprints=fingerprints)
        validate_hf_checkpoint_provenance(hf, splits=splits, fingerprints=fingerprints)
        if (
            lf.get("method") != "deeponet_pinn"
            or hf.get("method") != "multifidelity_correction"
            or hf.get("low_fidelity_method") != "deeponet_pinn"
            or hf.get("low_fidelity_model_kwargs") != lf.get("model_kwargs")
            or hf.get("scales") != lf.get("scales")
            or hf.get("surface_residual_guide_spec") is not None
            or "response_tau_seconds" in hf.get("correction_model_kwargs", {})
            or _powers(lf["train_powers_w"]) != _powers(splits.simulation_train)
            or _powers(lf["validation_powers_w"]) != _powers(splits.simulation_validation)
            or _powers(hf["hf_train_powers_w"]) != _powers(splits.hf_train)
            or _powers(hf["hf_validation_powers_w"]) != _powers(splits.hf_validation)
        ):
            raise ValueError(f"Task-07 seed{seed} LF/HF architecture or legal split is invalid")
        lf_state = {name: value.detach().cpu().clone() for name, value in lf["model_state"].items()}
        embedded_lf = {name[len("low_fidelity_model."):]: value for name, value in
                       hf["model_state"].items() if name.startswith("low_fidelity_model.")}
        if set(embedded_lf) != set(lf_state) or any(
            not torch.equal(value, embedded_lf[name]) for name, value in lf_state.items()
        ):
            raise ValueError(f"Task-07 seed{seed} historical HF does not embed its paired frozen LF")
        old_first = hf["model_state"].get("correction.0.weight")
        if old_first is None or tuple(old_first.shape) != (128, 6):
            raise ValueError("Task-07 historical E0 correction must retain six old inputs")
        source = Task07Source(
            seed=seed,
            lf_checkpoint_path=lf_path,
            lf_checkpoint_sha256=entry["lf_checkpoint_sha256"],
            lf_tensor_sha256=_lf_tensor_sha256(lf_state),
            hf_checkpoint_path=hf_path,
            hf_checkpoint_sha256=entry["hf_checkpoint_sha256"],
            lf_state=lf_state,
            lf_model_kwargs=copy.deepcopy(lf["model_kwargs"]),
            correction_model_kwargs=copy.deepcopy(hf["correction_model_kwargs"]),
            scales=copy.deepcopy(lf["scales"]),
            lf_train_powers_w=_powers(lf["train_powers_w"]),
            lf_validation_powers_w=_powers(lf["validation_powers_w"]),
            hf_train_powers_w=_powers(hf["hf_train_powers_w"]),
            hf_validation_powers_w=_powers(hf["hf_validation_powers_w"]),
            source_metadata_sha256="",
        )
        sources[seed] = replace(source, source_metadata_sha256=_source_metadata_sha256(source))
    if len({source.lf_tensor_sha256 for source in sources.values()}) != 5:
        raise ValueError("Task-07 requires five genuinely distinct paired LF tensor states")
    return sources


def _tau(arm: str) -> tuple[float, ...] | None:
    if arm == "E0":
        return None
    from sic_cu.train.task06_time_features import (
        CONFIG, CONFIG_SHA, _registered_tau,
    )

    if sha256_file(CONFIG) != CONFIG_SHA:
        raise ValueError("Task-07 E1/E2 must use unmodified Task-06 response preregistration")
    return _registered_tau(load_yaml(CONFIG), arm)


def fork_task07_initialization(
    sources: Mapping[int, Task07Source], seed: int, arm: str,
    device: torch.device = torch.device("cpu"),
) -> Task07Initialization:
    """Prepare one seed/arm; all arms remain explicitly unelected and untrained."""
    if arm not in {"E0", "E1", "E2"}:
        raise ValueError("Task-07 preparation permits only E0/E1/E2; no arm selected")
    registered_tau = _tau(arm)
    entries = _registered_rows()
    if set(sources) != SEEDS or seed not in SEEDS:
        raise ValueError("Task-07 seed set must remain complete and explicit 0..4")
    by_seed = {entry["seed"]: entry for entry in entries}
    for actual_seed, source in sources.items():
        entry = by_seed[actual_seed]
        if (
            source.seed != actual_seed
            or source.lf_checkpoint_path != PROJECT_ROOT / entry["lf_checkpoint"]
            or source.lf_checkpoint_sha256 != entry["lf_checkpoint_sha256"]
            or source.hf_checkpoint_path != PROJECT_ROOT / entry["hf_checkpoint"]
            or source.hf_checkpoint_sha256 != entry["hf_checkpoint_sha256"]
            or _lf_tensor_sha256(source.lf_state) != source.lf_tensor_sha256
            or _source_metadata_sha256(source) != source.source_metadata_sha256
        ):
            raise ValueError(f"Task-07 seed{actual_seed} source cannot be relabeled or modified")
    if len({source.lf_tensor_sha256 for source in sources.values()}) != 5:
        raise ValueError("Task-07 source LF tensors cannot be duplicated across seeds")

    source = sources[seed]
    scales = ModelScales(**source.scales)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        low = build_model("deeponet_pinn", scales, **source.lf_model_kwargs)
        if not isinstance(low, DeepONetPINN):
            raise ValueError("Task-07 locked LF model must be DeepONetPINN")
        low.load_state_dict(source.lf_state, strict=True)
        # Reset after LF construction so HF initialization depends on the seed alone.
        torch.manual_seed(seed)
        kwargs = copy.deepcopy(source.correction_model_kwargs)
        baseline = AdditiveCorrectionModel(low, scales, freeze_low_fidelity=True, **kwargs)
        cpu_rng = torch.get_rng_state().clone()
        if registered_tau is None:
            model = baseline
        else:
            kwargs["response_tau_seconds"] = registered_tau
            model = AdditiveCorrectionModel(low, scales, freeze_low_fidelity=True, **kwargs)
            expanded = model.state_dict()
            for name, old_value in baseline.state_dict().items():
                if name == "correction.0.weight":
                    expanded[name][:, :6].copy_(old_value)
                    expanded[name][:, 6:].zero_()
                else:
                    expanded[name].copy_(old_value)
            model.load_state_dict(expanded, strict=True)
        if model.correction[0].in_features != (6 if arm == "E0" else 10):
            raise ValueError("Task-07 E0/E1/E2 first-layer dimension is wrong")

    model = model.to(device)
    model.freeze_low_fidelity(True)
    if any(parameter.requires_grad for parameter in model.low_fidelity_model.parameters()):
        raise ValueError("Task-07 all five paired LF models must remain fully frozen")
    training = load_yaml("configs/training.yaml")["optimizer"]
    if training["name"] != "adamw":
        raise ValueError("Task-07 correction initialization requires fresh AdamW")
    optimizer = torch.optim.AdamW(
        model.correction.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    if optimizer.state_dict()["state"]:
        raise ValueError("Task-07 cannot restore historical HF optimizer momentum")
    cuda_rng = None
    if device.type == "cuda":
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.cuda.manual_seed_all(seed)
            cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
    return Task07Initialization(
        seed=seed, arm=arm, model=model, optimizer=optimizer,
        random_state={
            "python": random.Random(seed).getstate(),
            "numpy": np.random.RandomState(seed).get_state(),
            "torch_cpu": cpu_rng,
            "torch_cuda": cuda_rng,
        },
        lf_checkpoint_sha256=source.lf_checkpoint_sha256,
        lf_tensor_sha256=source.lf_tensor_sha256,
    )
