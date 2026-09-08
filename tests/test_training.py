from __future__ import annotations

import pytest
import torch

from sic_cu.physics.materials import PhysicsConfigurationError
from sic_cu.train.gno import train_gno_model
from sic_cu.train.simulation import _rank_indices, load_sampled_points


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


def test_gno_physics_training_is_fail_closed() -> None:
    with pytest.raises(PhysicsConfigurationError, match="graph differential/Jacobian"):
        train_gno_model("unused", epochs=1, use_physics=True)
