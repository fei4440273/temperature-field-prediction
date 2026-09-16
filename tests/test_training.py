from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from sic_cu.config import PROJECT_ROOT
from sic_cu.physics.materials import PhysicsConfigurationError
from sic_cu.data.balanced_sampler import (
    balanced_material_time_indices, balanced_material_time_space_indices,
)
from sic_cu.eval.interface_model import evaluate_pointwise_interface_checkpoint
from sic_cu.train.gno import train_gno_model
from sic_cu.train.multifidelity import _ir_dataset, _macro_sensor_training_losses
from sic_cu.train.simulation import (
    _rank_indices,
    build_model,
    evaluate_saved_simulation_model,
    load_sampled_points,
)
from sic_cu.models import ModelScales, PRCMultifidelityModel
from sic_cu.train import common
from sic_cu.train import simulation


def test_resolve_device_prefers_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert common.resolve_device() == torch.device("cuda")


def test_resolve_device_respects_explicit_device(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert common.resolve_device("cpu") == torch.device("cpu")


def test_evaluation_entry_points_default_to_automatic_device() -> None:
    simulation_default = inspect.signature(evaluate_saved_simulation_model).parameters[
        "device_name"
    ].default
    interface_default = inspect.signature(
        evaluate_pointwise_interface_checkpoint
    ).parameters["device"].default

    assert simulation_default is None
    assert interface_default is None


def test_rank_indices_cover_validation_without_duplicates() -> None:
    rank0 = _rank_indices(7, torch.device("cpu"), 0, 2, shuffle=False)
    rank1 = _rank_indices(7, torch.device("cpu"), 1, 2, shuffle=False)
    combined = torch.cat((rank0, rank1)).tolist()
    assert sorted(combined) == list(range(7))


def test_rank_indices_make_equal_deterministic_training_shards() -> None:
    rank0 = _rank_indices(
        7, torch.device("cpu"), 0, 2, shuffle=True, seed=4, equal_length=True
    )
    rank1 = _rank_indices(
        7, torch.device("cpu"), 1, 2, shuffle=True, seed=4, equal_length=True
    )
    repeated = _rank_indices(
        7, torch.device("cpu"), 0, 2, shuffle=True, seed=4, equal_length=True
    )
    assert len(rank0) == len(rank1) == 4
    assert torch.equal(rank0, repeated)
    assert set(torch.cat((rank0, rank1)).tolist()) == set(range(7))


def test_sampled_simulation_points_include_material_label() -> None:
    coordinates, temperatures = load_sampled_points([10.0], 128, seed=7).tensors
    assert coordinates.shape == (128, 5)
    assert temperatures.shape == (128, 1)
    assert set(coordinates[:, 4].unique().tolist()).issubset({0.0, 1.0})
    assert set(coordinates[:, 4].unique().tolist()) == {0.0, 1.0}


def test_balanced_sampler_covers_both_materials_and_time_ranges() -> None:
    materials = np.repeat([0, 1], 30)
    times = np.tile(np.repeat([10.0, 50.0, 150.0], 10), 2)
    indices = balanced_material_time_indices(
        materials, times, 24, np.random.default_rng(3)
    )
    assert set(materials[indices]) == {0, 1}
    assert set(np.digitize(times[indices], [30.0, 100.0])) == {0, 1, 2}


def test_spatial_sampling_covers_each_material_time_quadrant_without_changing_exposure() -> None:
    materials = np.repeat([0, 1], 3 * 16)
    times = np.tile(np.repeat([10.0, 50.0, 150.0], 16), 2)
    radial = np.tile(np.repeat([0.1, 0.9], 8), 6)
    depth = np.tile(np.tile(np.repeat([0.1, 0.9], 4), 2), 6)
    selected = balanced_material_time_space_indices(
        materials, times, radial, depth, 24, np.random.default_rng(4),
    )
    repeated = balanced_material_time_space_indices(
        materials, times, radial, depth, 24, np.random.default_rng(4),
    )
    assert np.array_equal(selected, repeated)
    assert len(selected) == len(set(selected.tolist())) == 24
    for material in (0, 1):
        for time_s in (10.0, 50.0, 150.0):
            mask = (materials[selected] == material) & (times[selected] == time_s)
            assert mask.sum() == 4
            quadrants = {(r > 0.5, z > 0.5) for r, z in zip(radial[selected][mask], depth[selected][mask])}
            assert len(quadrants) == 4


def test_real_lf_sampler_keeps_power_and_sample_budget_for_spatial_candidate() -> None:
    original_x, original_y = load_sampled_points(
        [10.0, 20.0], 128, seed=7, sampling_mode="material_time",
    ).tensors
    spatial_x, spatial_y = load_sampled_points(
        [10.0, 20.0], 128, seed=7, sampling_mode="material_time_space",
    ).tensors
    assert original_x.shape == spatial_x.shape == (256, 5)
    assert original_y.shape == spatial_y.shape == (256, 1)
    for power in (10.0, 20.0):
        for material in (0.0, 1.0):
            for lower, upper in ((0, 30), (30, 100), (100, 201)):
                def count(x):
                    return int(((x[:, 3] == power) & (x[:, 4] == material)
                                & (x[:, 2] >= lower) & (x[:, 2] < upper)).sum())
                assert count(original_x) == count(spatial_x)


def test_task03_lf_continuation_locks_the_real_pretrained_checkpoint() -> None:
    root = PROJECT_ROOT / "reports/runs/sic-cu-v2-gpu1-20260908T072729Z-r2"
    low = root / "deeponet_pinn_lf_seed0/best.pt"
    payload = simulation.validate_locked_lf_start(low, seed=0)
    assert payload["epoch"] == 178
    assert payload["method"] == "deeponet_pinn"
    with pytest.raises(ValueError, match="locked B0"):
        simulation.validate_locked_lf_start(low, seed=1)


def test_task03_lf_physics_mode_does_not_create_nominal_hf_zero_weight_updates() -> None:
    assert simulation.lf_physics_enabled("deeponet_pinn", "none") is False
    assert simulation.lf_physics_enabled("deeponet_pinn", "original") is True
    assert simulation.lf_physics_enabled("deeponet", "none") is False
    with pytest.raises(ValueError, match="LF physics"):
        simulation.lf_physics_enabled("deeponet_pinn", "unknown")


def test_locked_lf_best_epoch_keeps_full_optimizer_and_rng_state(tmp_path) -> None:
    old = PROJECT_ROOT / (
        "reports/runs/sic-cu-v2-gpu1-20260908T072729Z-r2/"
        "deeponet_pinn_lf_seed0/best.pt"
    )
    root = tmp_path / "locked_lf"
    simulation.train_simulation_model(
        "deeponet_pinn", 0, str(root), epochs=1,
        samples_per_power=4, validation_samples_per_power=4,
        batch_size=32, learning_rate=0.0001, patience=2,
        initial_checkpoint=str(old), lf_physics_mode="none",
    )
    best = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
    stage = torch.load(root / "阶段_LF观测最佳.pt", map_location="cpu", weights_only=False)
    assert stage["epoch"] == best["epoch"] == 1
    assert stage["optimizer_state"]["state"]
    assert stage["random_state"]["torch_cpu"] is not None
    assert all(torch.equal(stage["model_state"][name], value)
               for name, value in best["model_state"].items())
    assert best["lf_lineage"]["logical_budget_epochs"] == 1


def test_ir_training_weights_give_each_power_equal_total_weight() -> None:
    coordinates, _, weights = _ir_dataset(split="validation").tensors
    totals = []
    for power in torch.unique(coordinates[:, 3]):
        totals.append(weights[coordinates[:, 3] == power].sum())
    assert torch.allclose(torch.stack(totals), torch.ones(len(totals)), atol=1e-5)


def test_sensor_training_loss_is_equal_curve_macro() -> None:
    coordinates = torch.tensor(
        [
            [0.028, 0.0, 0.0, 100.0, 0.0],
            [0.028, 0.0, 1.0, 100.0, 0.0],
            [0.028, 0.0, 2.0, 100.0, 0.0],
            [0.0415, 0.0, 0.0, 100.0, 0.0],
        ]
    )
    target = torch.zeros((4, 1))
    prediction = torch.tensor([[0.0], [0.0], [0.0], [2.0]])
    delta = torch.zeros((4, 1))
    baseline = torch.tensor([0, 0, 0, 3])
    absolute, delta_loss = _macro_sensor_training_losses(
        prediction, target, delta, baseline, coordinates, 1.0
    )
    assert absolute.item() == pytest.approx(2.0)
    assert delta_loss.item() == pytest.approx(0.0)


def test_prc_lf_builder_rejects_irrelevant_kwargs_and_freezes_correction() -> None:
    model = build_model(
        "prc_lf", ModelScales(), modes=4, width=8, depth=2, include_material=True
    )
    assert isinstance(model, PRCMultifidelityModel)
    assert all(
        not parameter.requires_grad for parameter in model.high_fidelity_parameters()
    )


def test_gno_physics_training_is_fail_closed() -> None:
    with pytest.raises(PhysicsConfigurationError, match="graph differential/Jacobian"):
        train_gno_model("unused", epochs=1, use_physics=True)
