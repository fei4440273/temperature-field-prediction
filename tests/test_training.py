from __future__ import annotations

import numpy as np
import pytest
import torch

from sic_cu.physics.materials import PhysicsConfigurationError
from sic_cu.data.balanced_sampler import balanced_material_time_indices
from sic_cu.train.gno import train_gno_model
from sic_cu.train.multifidelity import _ir_dataset, _macro_sensor_training_losses
from sic_cu.train.simulation import (
    _rank_indices,
    build_model,
    load_sampled_points,
)
from sic_cu.models import ModelScales, PRCMultifidelityModel


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
