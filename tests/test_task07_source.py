"""Task-07 five-seed source and fresh-correction initialization only."""

from __future__ import annotations

import copy
import importlib
from dataclasses import replace

import numpy as np
import pytest
import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.models.common import parameter_count


def _adapter():
    return importlib.import_module("sic_cu.train.task07_source")


def test_task07_source_adapter_is_available() -> None:
    try:
        _adapter()
    except ModuleNotFoundError:
        pytest.fail("Task-07 source adapter is not implemented")


@pytest.fixture(scope="module")
def sources():
    return _adapter().validate_task07_sources()


def _coordinates() -> torch.Tensor:
    return torch.tensor(
        [[0.005, -0.006, 0.0, 55.0, 1.0],
         [0.018, -0.011, 1.0, 364.3, 1.0],
         [0.040, -0.015, 20.0, 630.5, 0.0],
         [0.030, -0.009, 100.0, 729.0, 0.0]],
        dtype=torch.float64,
    )


def test_five_locked_lf_states_are_distinct_and_provenant(sources) -> None:
    manifest = load_yaml("reports/development_v4/baseline_manifest.yaml")
    assert set(sources) == set(range(5))
    assert len({item.lf_tensor_sha256 for item in sources.values()}) == 5
    for item in manifest["checkpoints"]:
        source = sources[item["seed"]]
        assert source.lf_checkpoint_path == PROJECT_ROOT / item["lf_checkpoint"]
        assert source.lf_checkpoint_sha256 == item["lf_checkpoint_sha256"]
        assert source.hf_checkpoint_path == PROJECT_ROOT / item["hf_checkpoint"]
        assert source.hf_checkpoint_sha256 == item["hf_checkpoint_sha256"]
        assert len(source.lf_train_powers_w) == 60
        assert len(source.lf_validation_powers_w) == 10
        assert len(source.hf_train_powers_w) == 12
        assert len(source.hf_validation_powers_w) == 3
        assert not set(source.hf_train_powers_w) & set(source.hf_validation_powers_w)


def test_stale_manifest_rejected_before_loading_checkpoints(monkeypatch) -> None:
    adapter = _adapter()
    original_sha = adapter.sha256_file

    def changed_sha(path):
        if str(path).endswith("baseline_manifest.yaml"):
            return "0" * 64
        return original_sha(path)

    monkeypatch.setattr(adapter, "sha256_file", changed_sha)
    monkeypatch.setattr(adapter.torch, "load", lambda *_a, **_kw: pytest.fail("loaded checkpoint"))
    with pytest.raises(ValueError, match="manifest|清单"):
        adapter.validate_task07_sources()


def test_stale_lf_sha_rejected_before_loading_checkpoints(monkeypatch) -> None:
    adapter = _adapter()
    original_sha = adapter.sha256_file

    def changed_sha(path):
        if str(path).endswith("deeponet_pinn_lf_seed1/best.pt"):
            return "0" * 64
        return original_sha(path)

    monkeypatch.setattr(adapter, "sha256_file", changed_sha)
    monkeypatch.setattr(adapter.torch, "load", lambda *_a, **_kw: pytest.fail("loaded checkpoint"))
    with pytest.raises(ValueError, match="SHA|sha|哈希"):
        adapter.validate_task07_sources()


def test_checkpoint_with_other_seed_metadata_is_refused(monkeypatch) -> None:
    adapter = _adapter()
    original_load = torch.load

    def forged_load(path, **kwargs):
        checkpoint = original_load(path, **kwargs)
        if str(path).endswith("deeponet_pinn_lf_seed1/best.pt"):
            checkpoint = dict(checkpoint, seed=0)
        return checkpoint

    monkeypatch.setattr(adapter.torch, "load", forged_load)
    with pytest.raises(ValueError, match="seed|种子"):
        adapter.validate_task07_sources()


def test_heldout_hf_validation_is_never_a_training_source(monkeypatch) -> None:
    adapter = _adapter()
    original_load = torch.load

    def forged_load(path, **kwargs):
        checkpoint = original_load(path, **kwargs)
        if str(path).endswith("deeponet_pinn_mf_seed1/best.pt"):
            checkpoint = copy.copy(checkpoint)
            checkpoint["provenance"] = dict(checkpoint["provenance"])
            checkpoint["provenance"]["train_powers_w"] = list(
                checkpoint["provenance"]["validation_powers_w"]
            )
        return checkpoint

    monkeypatch.setattr(adapter.torch, "load", forged_load)
    with pytest.raises((ValueError, RuntimeError), match="train|训练"):
        adapter.validate_task07_sources()


def test_e0_uses_exact_seed_lf_weights_with_fresh_correction_and_empty_adamw(sources) -> None:
    adapter = _adapter()
    correction_weights = []
    for seed in range(5):
        initial = adapter.fork_task07_initialization(sources, seed, "E0", torch.device("cpu"))
        model = initial.model
        source = sources[seed]
        assert initial.seed == seed and initial.arm == "E0"
        assert model.correction[0].in_features == 6
        assert "response_features.tau_seconds" not in model.state_dict()
        assert all(not p.requires_grad for p in model.low_fidelity_model.parameters())
        assert all(p.requires_grad for p in model.correction.parameters())
        assert initial.optimizer.state_dict()["state"] == {}
        assert set(initial.random_state) == {"python", "numpy", "torch_cpu", "torch_cuda"}
        assert initial.random_state["torch_cuda"] is None  # CPU-only test cannot claim CUDA state.
        assert torch.equal(
            initial.random_state["torch_cpu"],
            adapter.fork_task07_initialization(sources, seed, "E0", torch.device("cpu"))
            .random_state["torch_cpu"],
        )
        assert initial.random_state["python"] == adapter.fork_task07_initialization(
            sources, seed, "E0", torch.device("cpu")
        ).random_state["python"]
        assert np.array_equal(
            initial.random_state["numpy"][1],
            adapter.fork_task07_initialization(sources, seed, "E0", torch.device("cpu"))
            .random_state["numpy"][1],
        )
        for name, value in source.lf_state.items():
            assert torch.equal(value, model.low_fidelity_model.state_dict()[name])
        correction_weights.append(model.correction[0].weight.detach().clone())
    assert len({value.numpy().tobytes() for value in correction_weights}) == 5


@pytest.mark.parametrize("seed", range(5))
def test_unselected_e1_e2_start_with_same_six_weights_and_true_equal_output(sources, seed) -> None:
    adapter = _adapter()
    models = {arm: adapter.fork_task07_initialization(sources, seed, arm, torch.device("cpu"))
              for arm in ("E0", "E1", "E2")}
    baseline = models["E0"].model
    for arm in ("E1", "E2"):
        enhanced = models[arm].model
        assert enhanced.correction[0].in_features == 10
        assert parameter_count(enhanced) - parameter_count(baseline) == 512
        assert torch.equal(enhanced.correction[0].weight[:, :6], baseline.correction[0].weight)
        assert torch.count_nonzero(enhanced.correction[0].weight[:, 6:]) == 0
        assert "response_features.tau_seconds" in enhanced.state_dict()
        assert enhanced.response_features.tau_seconds.dtype == torch.float64
        assert all(torch.equal(value, enhanced.state_dict()[name]) for name, value
                   in baseline.state_dict().items() if name != "correction.0.weight")
        assert models[arm].optimizer.state_dict()["state"] == {}
        assert torch.equal(models[arm].random_state["torch_cpu"],
                           models["E0"].random_state["torch_cpu"])
        x = _coordinates()
        assert torch.equal(baseline.double()(x), enhanced.double()(x))
        assert torch.equal(baseline(x, fidelity="low"), enhanced(x, fidelity="low"))
        assert not models[arm].selected_for_training
    assert not models["E0"].selected_for_training
    assert not torch.equal(models["E1"].model.response_features.tau_seconds,
                           models["E2"].model.response_features.tau_seconds)


def test_mixed_sources_cannot_relabel_seed0_lf_as_seed1(sources) -> None:
    adapter = _adapter()
    mixed = dict(sources)
    mixed[1] = replace(sources[0], seed=1)
    with pytest.raises(ValueError, match="种子|seed|SHA|来源"):
        adapter.fork_task07_initialization(mixed, 1, "E0", torch.device("cpu"))


@pytest.mark.parametrize("field", ["scales", "correction_model_kwargs", "hf_train_powers_w"])
def test_checked_source_metadata_cannot_change_before_fresh_hf_initialization(sources, field) -> None:
    adapter = _adapter()
    changed = {
        "scales": dict(sources[0].scales, temperature_scale_k=100.0),
        "correction_model_kwargs": dict(sources[0].correction_model_kwargs, initial_ramp_time_s=1.0),
        "hf_train_powers_w": tuple(sources[0].hf_validation_powers_w),
    }
    forged = dict(sources)
    forged[0] = replace(sources[0], **{field: changed[field]})
    with pytest.raises(ValueError, match="source|来源|metadata|元数据"):
        adapter.fork_task07_initialization(forged, 0, "E0", torch.device("cpu"))


def test_unsupported_arm_refused_before_model_creation(sources) -> None:
    adapter = _adapter()
    with pytest.raises(ValueError, match="E0|E1|E2"):
        adapter.fork_task07_initialization(sources, 0, "E3", torch.device("cpu"))
