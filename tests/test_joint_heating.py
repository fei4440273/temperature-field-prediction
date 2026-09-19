"""Heating-response contracts; small CPU inputs are not experimental results."""
import copy
import math
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from joint_temperature_core import (
    Geometry, JointDeepONet, PhysicalSettings, grad, load_model,
    physics_loss, read_yaml, save_atomic, snapshot,
)


def physical_settings():
    return PhysicalSettings(
        295.15, 298.15, 295.15, 295.15, 295.15, .8, .02, 9., 4.3,
        .5, .5, True, 8900., 400., 401., 3170., 700., 120.,
        7.34072435302768e-5,
    )


def small_model(mode='monotone_heating', settings=None):
    torch.manual_seed(42)
    cfg = {
        'width': 8, 'latent_dim': 8, 'blocks': 1,
        'correction_width': 8, 'correction_depth': 2,
        'temperature_scale_k': 250., 'learn_contact': True,
    }
    if mode is not None:
        cfg['time_response'] = mode
    return JointDeepONet(Geometry(), settings or physical_settings(), cfg)


def signed_parameters(model):
    generator = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name != 'log_contact':
                parameter.uniform_(-.18, .18, generator=generator)


def trajectories(times=(130., 0., 5., 200., 100.)):
    locations = (
        (0., -.002, 55., 1.), (.01, 0., 400., 1.),
        (.0246, -.006, 634., 1.), (.035, -.015, 216.8, 0.),
        (.05834, -.0175, 729., 0.), (.026, -.014, 800., 0.),
    )
    return torch.tensor([
        [r, z, t, power, material]
        for r, z, power, material in locations for t in times
    ])


def assert_heating(model, fidelity):
    times = torch.tensor([130., 0., 5., 200., 100.])
    query = trajectories(times.tolist())
    values = model(query, fidelity).reshape(-1, len(times))
    ordered = values[:, torch.argsort(times)]
    assert torch.isfinite(values).all()
    assert bool((ordered[:, 1:] - ordered[:, :-1] >= -1e-7).all())
    assert bool((values >= ordered[:, :1] - 1e-7).all())
    individual = torch.cat([model(row[None], fidelity) for row in query])
    torch.testing.assert_close(values.flatten(), individual.flatten(), rtol=1e-6, atol=1e-4)


@pytest.fixture
def project_tmpdir():
    directory = Path(tempfile.mkdtemp(prefix='临时升温测试_', dir=ROOT / '研究记录'))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def checkpoint_state(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    return snapshot(
        model, optimizer, scheduler, 12, [],
        {'model': copy.deepcopy(model.model_config)}, {}, 2.5,
        np.random.default_rng(1),
    )


def test_default_configuration_uses_monotone_heating():
    cfg = read_yaml(ROOT / 'configs/联合训练8000轮.yaml')
    assert cfg['model'].get('time_response') == 'monotone_heating'


def test_monotone_coefficients_do_not_take_time_as_input():
    model = small_model()
    assert model.branch.input.in_features == 1
    assert model.trunk.input.in_features == 3
    linears = [layer for layer in model.correction if isinstance(layer, torch.nn.Linear)]
    assert linears[0].in_features == 5
    assert linears[-1].out_features == model.model_config['latent_dim'] + 1


@pytest.mark.parametrize('fidelity', ['low', 'high'])
def test_signed_parameters_keep_unordered_trajectories_heating(fidelity):
    model = small_model()
    signed_parameters(model)
    assert_heating(model, fidelity)


@pytest.mark.parametrize('fidelity', ['low', 'high'])
def test_default_zero_initial_offsets_equal_nominal_initial(fidelity):
    model = small_model()
    query = trajectories((0.,))
    expected = (model.settings.initial_simulation_k if fidelity == 'low'
                else model.settings.initial_experiment_k)
    assert torch.equal(model(query, fidelity), torch.full((len(query), 1), expected))


@pytest.mark.parametrize('fidelity', ['low', 'high'])
def test_time_derivatives_are_nonnegative_for_signed_parameters(fidelity):
    model = small_model().double()
    signed_parameters(model)
    query = trajectories((0., 5., 100., 130., 200., 500.)).double().requires_grad_()
    derivative = grad(model(query, fidelity), query)
    assert torch.isfinite(derivative).all()
    assert bool((derivative[:, 2] >= -1e-7).all())


@pytest.mark.parametrize('fidelity', ['low', 'high'])
def test_spatial_first_and_second_derivatives_remain_finite(fidelity):
    model = small_model().double()
    signed_parameters(model)
    query = trajectories((5., 100., 130.)).double().requires_grad_()
    first = grad(model(query, fidelity), query)
    radial_second = grad(first[:, 0:1], query)[:, 0]
    axial_second = grad(first[:, 1:2], query)[:, 1]
    assert torch.isfinite(first).all()
    assert torch.isfinite(radial_second).all()
    assert torch.isfinite(axial_second).all()


def test_heating_physics_residuals_and_parameter_gradients_are_finite():
    model = small_model().double()
    signed_parameters(model)
    parts = physics_loss(model, 4, torch.Generator().manual_seed(7))
    assert all(torch.isfinite(value) for value in parts.values())
    sum(parts.values()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_interface_loss_has_nonzero_contact_resistance_gradient():
    model = small_model().double()
    signed_parameters(model)
    parts = physics_loss(model, 4, torch.Generator().manual_seed(7))
    parts['材料界面'].backward()
    assert model.log_contact.grad is not None
    assert torch.isfinite(model.log_contact.grad)
    assert model.log_contact.grad.abs() > 0


def test_low_supervision_does_not_update_correction_or_contact():
    model = small_model()
    loss = ((model(trajectories((5., 100.)), 'low') - 330.) / 250.).square().mean()
    loss.backward()
    assert all(p.grad is None for p in model.correction.parameters())
    assert model.log_contact.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.branch_projection.parameters())


def test_high_supervision_updates_low_and_correction_networks_after_two_steps():
    model = small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=0.)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(trajectories((5., 100.)), 'high') - 330.) / 250.).square().mean()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.log_contact.grad is None
        if step == 1:
            for network in (model.branch, model.trunk, model.correction):
                assert any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in network.parameters())
        optimizer.step()
    for prefix in ('branch.', 'trunk.', 'correction.'):
        assert any(not torch.equal(before[name], p)
                   for name, p in model.named_parameters() if name.startswith(prefix))


@pytest.mark.parametrize('sign', [-1., 1.])
def test_high_temperature_can_be_below_or_above_low_without_cooling(sign):
    model = small_model()
    rank = model.model_config['latent_dim']
    last = [layer for layer in model.correction if isinstance(layer, torch.nn.Linear)][-1]
    assert last.out_features == rank + 1
    with torch.no_grad():
        last.bias[:rank].fill_(sign)
        last.bias[-1] = 0.
    query = trajectories((5., 100., 130., 200.))
    assert bool((sign * (model(query, 'high') - model(query, 'low')) > 0).all())
    assert_heating(model, 'low')
    assert_heating(model, 'high')


def test_high_initial_offset_can_fit_observed_25_point_289_c_without_cooling():
    model = small_model()
    last = [layer for layer in model.correction if isinstance(layer, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.bias[-1] = 3.28903 / 250.
    actual = model(trajectories((0.,)), 'high')
    expected = torch.full_like(actual, 273.15 + 25.28903)
    torch.testing.assert_close(actual, expected, rtol=0., atol=1e-4)
    assert_heating(model, 'high')


def test_unknown_time_response_is_rejected():
    with pytest.raises(ValueError):
        small_model('not_a_temperature_response')


@pytest.mark.parametrize('changes', [
    {'initial_experiment_k': 296.15},
    {'cooling_simulation_k': 294.15},
])
def test_heating_rejects_nominal_initial_above_ambient_or_cooling(changes):
    with pytest.raises(ValueError):
        small_model(settings=replace(physical_settings(), **changes))


def test_heating_checkpoint_has_new_schema_and_roundtrips(project_tmpdir):
    model = small_model()
    signed_parameters(model)
    state = checkpoint_state(model)
    assert state['schema'] == 'joint_deeponet_8000_v2_heating'
    path = project_tmpdir / 'heating.pt'
    save_atomic(state, path)
    loaded, payload = load_model(path)
    assert payload['model_config']['time_response'] == 'monotone_heating'
    for fidelity in ('low', 'high'):
        assert torch.equal(model(trajectories(), fidelity), loaded(trajectories(), fidelity))


@pytest.mark.parametrize('schema,mode', [
    ('joint_deeponet_8000_v1', 'monotone_heating'),
    ('joint_deeponet_8000_v2_heating', 'free'),
])
def test_checkpoint_schema_and_response_mode_mismatch_is_rejected(project_tmpdir, schema, mode):
    state = checkpoint_state(small_model())
    state['schema'] = schema
    state['model_config']['time_response'] = mode
    state['config']['model']['time_response'] = mode
    path = project_tmpdir / 'mismatched.pt'
    save_atomic(state, path)
    with pytest.raises(ValueError, match='版本|模式|检查点'):
        load_model(path)


def test_free_and_missing_mode_preserve_legacy_temperature_equations():
    model = small_model(None)
    signed_parameters(model)
    explicit_free = small_model('free')
    explicit_free.load_state_dict(model.state_dict())
    query = trajectories()
    scaled = model.scaled(query)
    branch = model.branch_projection(model.branch(scaled[:, 3:4]))
    trunk = model.trunk_projection(model.trunk(scaled[:, [0, 1, 2, 4]]))
    low = model.temperature_offset + model.temperature_scale * (
        (branch * trunk).sum(1, keepdim=True) / math.sqrt(branch.shape[1]) + model.low_bias
    )
    correction_input = torch.cat((scaled, (low - model.temperature_offset) / model.temperature_scale), 1)
    high = low + model.temperature_scale * model.correction(correction_input)
    for candidate in (model, explicit_free):
        assert torch.equal(candidate(query, 'low'), low)
        assert torch.equal(candidate(query, 'high'), high)


def test_legacy_v1_checkpoint_remains_loadable(project_tmpdir):
    model = small_model(None)
    signed_parameters(model)
    state = checkpoint_state(model)
    assert state['schema'] == 'joint_deeponet_8000_v1'
    path = project_tmpdir / 'legacy.pt'
    save_atomic(state, path)
    loaded, _ = load_model(path)
    assert torch.equal(model(trajectories()), loaded(trajectories()))


@pytest.mark.parametrize('mode,constant_heating,raises', [
    ('monotone_heating', False, True),
    ('monotone_heating', None, True),
    ('monotone_heating', True, False),
    ('free', False, False),
])
def test_project_heating_mode_requires_constant_laser(project_tmpdir, mode, constant_heating, raises):
    configs = project_tmpdir / 'configs'
    configs.mkdir()
    shutil.copy2(ROOT / 'configs/materials.yaml', configs / 'materials.yaml')
    boundary = read_yaml(ROOT / 'configs/boundary_conditions.yaml')
    if constant_heating is None:
        boundary['laser'].pop('constant_during_heating', None)
    else:
        boundary['laser']['constant_during_heating'] = constant_heating
    (configs / 'boundary_conditions.yaml').write_text(yaml.safe_dump(boundary), encoding='utf-8')
    cfg = read_yaml(ROOT / 'configs/联合训练8000轮.yaml')
    cfg['model']['time_response'] = mode
    if raises:
        with pytest.raises(ValueError):
            PhysicalSettings.from_project(project_tmpdir, cfg)
    else:
        assert isinstance(PhysicalSettings.from_project(project_tmpdir, cfg), PhysicalSettings)
