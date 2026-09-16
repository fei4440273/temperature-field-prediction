from __future__ import annotations

import copy

import pytest
import torch

from sic_cu.models import AdditiveCorrectionModel, MLPPINN
from sic_cu.models.common import parameter_count


E1_TAU = (2.0, 10.0, 50.0, 200.0)
E2_TAU = (
    0.8341738692346146, 2.456793356991631,
    11.306550331610655, 14.206694104729392,
)


def _base_model(width: int = 128) -> AdditiveCorrectionModel:
    torch.manual_seed(306)
    low = MLPPINN(width=8, depth=2, include_material=True)
    model = AdditiveCorrectionModel(low, width=width, depth=2)
    with torch.no_grad():
        model.correction[-1].weight.fill_(0.025)
        model.correction[-1].bias.fill_(0.02)
    return model


def _coordinates() -> torch.Tensor:
    return torch.tensor(
        [[0.005, -0.006, 0.0, 55.0, 1.0],
         [0.018, -0.011, 1.0, 364.3, 1.0],
         [0.040, -0.015, 20.0, 630.5, 0.0],
         [0.030, -0.009, 100.0, 729.0, 0.0]],
        dtype=torch.float64,
    )


def _copy_e0_weights_with_zero_extra(
    e0: AdditiveCorrectionModel, enhanced: AdditiveCorrectionModel,
) -> None:
    old = e0.state_dict()
    new = enhanced.state_dict()
    with torch.no_grad():
        for name, old_value in old.items():
            if name == "correction.0.weight":
                new[name][:, :old_value.shape[1]].copy_(old_value)
                new[name][:, old_value.shape[1]:].zero_()
            else:
                new[name].copy_(old_value)
    enhanced.load_state_dict(new, strict=True)


def test_e0_default_keeps_legacy_six_inputs_and_strict_old_reload() -> None:
    e0 = _base_model()
    assert e0.correction[0].in_features == 6
    assert "response_features.tau_seconds" not in e0.state_dict()
    restored = _base_model()
    restored.load_state_dict(copy.deepcopy(e0.state_dict()), strict=True)
    x = _coordinates()
    assert torch.equal(e0.double()(x), restored.double()(x))


@pytest.mark.parametrize("tau", [E1_TAU, E2_TAU], ids=["E1", "E2"])
def test_added_four_channels_zero_start_exactly_matches_e0_and_preserves_lf(tau) -> None:
    e0 = _base_model()
    enhanced = AdditiveCorrectionModel(
        copy.deepcopy(e0.low_fidelity_model), width=128, depth=2,
        response_tau_seconds=tau,
    )
    assert enhanced.correction[0].in_features == 10
    assert parameter_count(enhanced) - parameter_count(e0) == 512
    assert "response_features.tau_seconds" in dict(enhanced.named_buffers())
    assert torch.equal(
        enhanced.response_features.tau_seconds,
        torch.tensor(tau, dtype=torch.float64),
    )
    _copy_e0_weights_with_zero_extra(e0, enhanced)
    old_weight = e0.correction[0].weight.detach()
    new_weight = enhanced.correction[0].weight.detach()
    assert torch.equal(old_weight, new_weight[:, :6])
    assert torch.count_nonzero(new_weight[:, 6:]).item() == 0
    x = _coordinates()
    e0.double().eval()
    enhanced.double().eval()
    assert torch.equal(enhanced(x, fidelity="low"), e0(x, fidelity="low"))
    assert torch.equal(enhanced(x, fidelity="high"), e0(x, fidelity="high"))


@pytest.mark.parametrize("tau", [E1_TAU, E2_TAU], ids=["E1", "E2"])
def test_response_channels_keep_nonzero_first_and_second_time_gradients(tau) -> None:
    low = MLPPINN(width=8, depth=2, include_material=True)
    model = AdditiveCorrectionModel(
        low, width=4, depth=1, response_tau_seconds=tau,
    ).double()
    with torch.no_grad():
        model.correction[0].weight.zero_()
        model.correction[0].bias.zero_()
        model.correction[0].weight[0, 6] = 1.0
        model.correction[-1].weight.zero_()
        model.correction[-1].bias.zero_()
        model.correction[-1].weight[0, 0] = 0.25
    x = _coordinates()[1:2].clone().requires_grad_(True)
    y = model(x)
    first = torch.autograd.grad(y.sum(), x, create_graph=True)[0][:, 2]
    second = torch.autograd.grad(first.sum(), x)[0][:, 2]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert first.abs().min().item() > 1e-9
    assert second.abs().min().item() > 1e-9


def test_strict_state_reload_restores_original_tau_instead_of_constructor_tau() -> None:
    original = AdditiveCorrectionModel(
        MLPPINN(width=8, depth=2, include_material=True),
        width=4, depth=1, response_tau_seconds=E1_TAU,
    ).double()
    other = AdditiveCorrectionModel(
        MLPPINN(width=8, depth=2, include_material=True),
        width=4, depth=1, response_tau_seconds=E2_TAU,
    ).double()
    with torch.no_grad():
        original.correction[0].weight.zero_()
        original.correction[0].weight[0, 6] = 1.0
        original.correction[-1].weight.zero_()
        original.correction[-1].weight[0, 0] = 0.25
    saved = copy.deepcopy(original.state_dict())
    x = _coordinates()[1:3]
    other.load_state_dict(saved, strict=True)
    assert torch.equal(other.response_features.tau_seconds, original.response_features.tau_seconds)
    assert torch.equal(other(x), original(x))
