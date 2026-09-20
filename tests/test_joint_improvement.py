"""Regression contracts; synthetic inputs are not experimental accuracy results."""
import copy
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import joint_temperature_core as core
from test_joint_heating import physical_settings, trajectories, signed_parameters, assert_heating
from test_joint_8000 import small_joint_project, small_training_entrypoint, small_training_config


def smooth_model(degree=2, independent_initial=False):
    torch.manual_seed(42)
    config = {
        'time_response': 'monotone_heating', 'width': 8, 'latent_dim': 8, 'blocks': 1,
        'correction_width': 8, 'correction_depth': 2, 'temperature_scale_k': 250.,
        'learn_contact': True, 'correction_power_degree': degree,
    }
    if independent_initial:
        config['power_independent_initial'] = True
    return core.JointDeepONet(core.Geometry(), physical_settings(), config)


def test_power_correction_has_spatial_only_inputs_and_small_polynomial_degree():
    model = smooth_model()
    layers = [x for x in model.correction if isinstance(x, torch.nn.Linear)]
    assert layers[0].in_features == 3
    assert layers[-1].out_features == 3 * (8 + 1)


def test_raw_correction_is_quadratic_in_power_not_an_unrestricted_power_mlp():
    model = smooth_model().double()
    signed_parameters(model)
    x = torch.tensor([[.014, -.002, 130., p, 1.] for p in (400., 450., 500., 550.)], dtype=torch.float64)
    z, raw, _ = model.heating_features(x)
    corrections = model.heating_correction(z, raw)
    third_difference = corrections[3] - 3 * corrections[2] + 3 * corrections[1] - corrections[0]
    torch.testing.assert_close(third_difference, torch.zeros_like(third_difference), rtol=0., atol=1e-12)


def test_new_mode_keeps_preheating_temperature_independent_of_laser_power():
    model = smooth_model(independent_initial=True).double()
    signed_parameters(model)
    x = torch.tensor([[.028, -.0175, 0., p, 0.] for p in (55., 216.8, 729.)], dtype=torch.float64)
    initial = model(x)
    torch.testing.assert_close(initial, initial[:1].expand_as(initial), rtol=0., atol=1e-12)
    warming = x.clone()
    warming[:, 2] = 100.
    warmed = model(warming)
    assert (warmed.max() - warmed.min()).item() > 1e-6
    assert_heating(model.float(), 'high')


def test_independent_initial_is_only_valid_for_smooth_heating_correction():
    with pytest.raises(ValueError, match='初温|功率'):
        smooth_model(degree=None, independent_initial=True)


def test_training_configuration_enables_power_independent_initial():
    config = core.read_yaml(ROOT / 'configs/联合训练600轮_低保真初温平滑锚定.yaml')
    assert config['model']['power_independent_initial'] is True
    settings = core.PhysicalSettings.from_project(ROOT, config)
    model = core.JointDeepONet(core.Geometry.from_project(ROOT), settings, config['model'])
    x = torch.tensor([[.028, -.0175, 0., power, 0.] for power in (55., 115.2, 729.)])
    predicted = model(x)
    torch.testing.assert_close(predicted, predicted[:1].expand_as(predicted), rtol=0., atol=1e-6)


def test_current_development_configuration_runs_600_epochs_without_test_export():
    config = core.read_yaml(ROOT / 'configs/联合训练600轮_低保真初温平滑锚定.yaml')
    assert config['training']['epochs'] == 600
    assert config['visualization']['export_test_after_training'] is False
    assert config['training']['sensor_sampling'] == 'full'
    assert config['loss_weights']['热端末段趋势'] > 0
    assert config['loss_weights']['冷端末段趋势'] > 0


def test_removed_copper_sensor_sampling_option_is_rejected_before_training(
        small_joint_project, monkeypatch):
    import argparse
    import yaml

    config = small_training_config('all_training')
    config['training']['simulation_sensor_anchors'] = True
    path = small_joint_project / 'configs/unsupported_sampling.yaml'
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    out = small_joint_project / '研究记录/unsupported_sampling'
    trainer = small_training_entrypoint(monkeypatch)
    with pytest.raises(ValueError, match='不支持.*仿真铜测点采样'):
        trainer.train(argparse.Namespace(root=str(small_joint_project), config=str(path),
                                     device='cpu', output=str(out), resume=None))
    assert not out.exists()


def test_fixed_low_initial_training_changes_only_simulation_initial_response():
    config = core.read_yaml(ROOT / 'configs/联合训练600轮_低保真初温平滑锚定.yaml')
    assert config['model']['low_initial_floor_width_k'] == .5
    assert config['model']['high_response_center_max_s'] == 160.
    assert config['model']['power_independent_initial'] is True
    assert config['model']['correction_power_degree'] == 2
    assert config['training']['epochs'] == 600
    assert config['data']['low_fidelity_mode'] == 'all_training'
    assert config['visualization']['export_test_after_training'] is False


def test_current_sensor_losses_and_selection_match_saved_best_checkpoint():
    import json

    config = core.read_yaml(ROOT / 'configs/联合训练600轮_低保真初温平滑锚定.yaml')
    assert config['training']['sensor_sampling'] == 'full'
    assert config['training']['low_freeze_epochs'] == 300
    assert config['loss_weights']['热端绝对温度'] == config['loss_weights']['冷端绝对温度'] == 10.
    assert config['loss_weights']['热端末段趋势'] == config['loss_weights']['冷端末段趋势'] == 30.
    assert config['validation_selection']['weights'] == {'顶部': .5, '热端': .25, '冷端': .25}
    saved = ROOT / '研究记录/联合训练600轮_第09轮_低保真初温平滑锚定_20260920'
    _, state = core.load_model(saved / '验证最佳模型.pt')
    complete = json.loads((saved / '开发集完成记录.json').read_text(encoding='utf-8'))
    assert state['schema'] == 'joint_deeponet_8000_v6_fixed_low_initial'
    assert state['epoch'] == complete['验证最佳轮次'] == 575
    assert complete['完成轮数'] == 600
    assert complete['验证最佳检查点SHA256'] == core.sha256(saved / '验证最佳模型.pt')
    assert not complete['测试标签参与训练']
    assert not complete['测试标签参与选模']
    assert not complete['测试数据已读取']
    original = copy.deepcopy(state['config'])
    original['training']['initialize_low_from'] = config['training']['initialize_low_from']
    assert original == config


def test_low_fidelity_initialization_is_available_and_unchanged():
    config = core.read_yaml(ROOT / 'configs/联合训练600轮_低保真初温平滑锚定.yaml')
    source = (ROOT / config['training']['initialize_low_from']).resolve()
    assert ROOT in source.parents
    assert source.is_file()
    assert core.sha256(source) == 'ef2793367e335513189a72961337c84754d18d788fd326d4f52c4f4d14169a02'


def test_higher_power_correction_schema_remains_loadable_without_changing_default():
    config = core.read_yaml(ROOT / 'configs/联合训练600轮_低保真初温平滑锚定.yaml')
    assert config['model']['correction_power_degree'] == 2
    model = smooth_model(degree=3, independent_initial=True).double()
    signed_parameters(model)
    x = torch.tensor([[.028, -.0175, 0., p, 0.]
                      for p in (400., 450., 500., 550., 600.)], dtype=torch.float64)
    initial = model(x)
    torch.testing.assert_close(initial, initial[:1].expand_as(initial), rtol=0., atol=1e-12)
    z, raw, _ = model.heating_features(x)
    corrections = model.heating_correction(z, raw)
    third = corrections[3] - 3 * corrections[2] + 3 * corrections[1] - corrections[0]
    fourth = corrections[4] - 4 * corrections[3] + 6 * corrections[2] - 4 * corrections[1] + corrections[0]
    assert torch.max(third.abs()) > 1e-7
    torch.testing.assert_close(fourth, torch.zeros_like(fourth), rtol=0., atol=1e-12)
    assert_heating(model.float(), 'high')


def test_simulation_sampling_uses_single_random_stream_without_sensor_anchors():
    x = np.array([[.028, -.0175, t, p, 0.] for p in (100., 200.) for t in (0., 2., 4.)],
                 dtype=np.float32)
    table = core.Table(x, np.full(len(x), 295.15), np.ones(len(x)),
                       np.repeat([0, 2], 3)).validate()
    pool = core.Pool(table, torch.device('cpu'))
    rng = np.random.default_rng(42)
    picked = pool.sample(4, rng)
    expected_rng = np.random.default_rng(42)
    expected = np.concatenate([expected_rng.choice(group, 4, replace=True)
                               for group in pool.groups])
    np.testing.assert_array_equal(picked, expected)
    assert rng.bit_generator.state == expected_rng.bit_generator.state


@pytest.mark.parametrize('degree', [-1, 0, 4, 2.5, True])
def test_invalid_or_excessively_flexible_power_degree_is_rejected(degree):
    with pytest.raises(ValueError, match='功率'):
        smooth_model(degree)


@pytest.mark.parametrize('fidelity', ['low', 'high'])
def test_power_smoothing_preserves_time_heating_and_spatial_second_derivatives(fidelity):
    model = smooth_model().double()
    signed_parameters(model)
    assert_heating(model.float(), fidelity)
    model.double()
    x = trajectories((0., 5., 100., 200.)).double().requires_grad_()
    first = core.grad(model(x, fidelity), x)
    assert (first[:, 2] >= -1e-10).all()
    assert torch.isfinite(core.grad(first[:, 0:1], x)).all()


def checkpoint(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    return core.snapshot(model, optimizer, scheduler, 5, [], {}, {}, 1., np.random.default_rng(0))


def test_smooth_power_checkpoint_has_distinct_schema_and_roundtrips(tmp_path):
    model = smooth_model()
    signed_parameters(model)
    state = checkpoint(model)
    assert state['schema'] == 'joint_deeponet_8000_v3_smooth_power'
    path = tmp_path / 'smooth.pt'
    core.save_atomic(state, path)
    restored, _ = core.load_model(path)
    assert torch.equal(model(trajectories()), restored(trajectories()))


def test_independent_initial_checkpoint_roundtrips_without_reinterpreting_v3(tmp_path):
    model = smooth_model(independent_initial=True)
    signed_parameters(model)
    state = checkpoint(model)
    assert state['schema'] == 'joint_deeponet_8000_v4_power_independent_initial'
    path = tmp_path / 'independent.pt'
    core.save_atomic(state, path)
    restored, _ = core.load_model(path)
    assert torch.equal(model(trajectories()), restored(trajectories()))
    state['schema'] = 'joint_deeponet_8000_v3_smooth_power'
    core.save_atomic(state, path)
    with pytest.raises(ValueError, match='版本|模式|检查点'):
        core.load_model(path)


def test_high_observed_window_omits_only_unmeasured_late_high_basis():
    baseline = smooth_model(independent_initial=True)
    signed_parameters(baseline)
    config = copy.deepcopy(baseline.model_config)
    config['high_response_center_max_s'] = 160.
    capped = core.JointDeepONet(core.Geometry(), physical_settings(), config)
    capped.load_state_dict(baseline.state_dict())
    x = torch.tensor([[.028, -.0175, t, 729., 0.] for t in (0., 80., 160., 180., 200.)])
    torch.testing.assert_close(capped(x, 'low'), baseline(x, 'low'), rtol=0., atol=0.)
    torch.testing.assert_close(capped(x[:1], 'high'), baseline(x[:1], 'high'), rtol=0., atol=0.)
    assert torch.all(capped(x[1:], 'high') < baseline(x[1:], 'high'))
    baseline_late = baseline(x[-1:], 'high') - baseline(x[-2:-1], 'high')
    capped_late = capped(x[-1:], 'high') - capped(x[-2:-1], 'high')
    assert torch.all(0 <= capped_late) and torch.all(capped_late < baseline_late)
    assert_heating(capped, 'high')


@pytest.mark.parametrize('limit', [-1., 0., 200., float('nan'), True])
def test_high_observed_window_requires_a_finite_intermediate_time(limit):
    config = copy.deepcopy(smooth_model(independent_initial=True).model_config)
    config['high_response_center_max_s'] = limit
    with pytest.raises(ValueError, match='时间|中心|观测'):
        core.JointDeepONet(core.Geometry(), physical_settings(), config)


def test_high_observed_window_schema_roundtrips_without_reinterpreting_v4(tmp_path):
    old = smooth_model(independent_initial=True)
    signed_parameters(old)
    config = copy.deepcopy(old.model_config)
    config['high_response_center_max_s'] = 160.
    capped = core.JointDeepONet(core.Geometry(), physical_settings(), config)
    core.initialize_low_network(capped, _saved_checkpoint(old, tmp_path / 'old.pt'))
    torch.testing.assert_close(old(trajectories(), 'low'), capped(trajectories(), 'low'),
                               rtol=0., atol=0.)
    state = checkpoint(capped)
    assert state['schema'] == 'joint_deeponet_8000_v5_high_observed_window'
    path = tmp_path / 'capped.pt'
    core.save_atomic(state, path)
    restored, _ = core.load_model(path)
    torch.testing.assert_close(capped(trajectories()), restored(trajectories()), rtol=0., atol=0.)
    state['schema'] = 'joint_deeponet_8000_v4_power_independent_initial'
    core.save_atomic(state, path)
    with pytest.raises(ValueError, match='版本|模式|检查点'):
        core.load_model(path)


def test_low_initial_floor_matches_simulation_initial_and_preserves_high_predictions():
    config = copy.deepcopy(smooth_model(independent_initial=True).model_config)
    config['high_response_center_max_s'] = 160.
    baseline = core.JointDeepONet(core.Geometry(), physical_settings(), config).double()
    with torch.no_grad():
        baseline.low_bias.fill_(-4.25 / 250.)
    config['low_initial_floor_width_k'] = .5
    floored = core.JointDeepONet(core.Geometry(), physical_settings(), config).double()
    floored.load_state_dict(baseline.state_dict())
    query = trajectories((0., 2., 10., 100., 200.)).double()
    initial = query[query[:, 2] == 0.]
    torch.testing.assert_close(floored(initial, 'low'),
                               floored.temperature_offset.expand(len(initial), 1),
                               rtol=0., atol=0.)
    assert float(floored.temperature_offset) - 273.15 == pytest.approx(22., abs=1e-5)
    torch.testing.assert_close(floored(query, 'high'), baseline(query, 'high'), rtol=0., atol=0.)
    warming = floored(query, 'low').reshape(-1, 5)
    assert bool((warming[:, 1:] >= warming[:, :-1]).all())
    assert (baseline(initial, 'low') < 295.15).all()


def test_low_initial_floor_is_smooth_and_has_finite_physics_derivatives():
    config = copy.deepcopy(smooth_model(independent_initial=True).model_config)
    config.update(high_response_center_max_s=160., low_initial_floor_width_k=.5)
    model = core.JointDeepONet(core.Geometry(), physical_settings(), config).double()
    with torch.no_grad():
        model.low_bias.fill_(-4.25 / 250.)
    query = trajectories((0., 2., 100., 200.)).double().requires_grad_()
    first = core.grad(model(query, 'low'), query)
    assert torch.isfinite(first).all()
    assert bool((first[:, 2] >= -1e-10).all())
    assert torch.isfinite(core.grad(first[:, 0:1], query)).all()
    assert torch.isfinite(core.grad(first[:, 1:2], query)).all()


@pytest.mark.parametrize('width', [-1., 0., float('inf'), float('nan'), True, '0.5'])
def test_low_initial_floor_rejects_invalid_width(width):
    config = copy.deepcopy(smooth_model(independent_initial=True).model_config)
    config.update(high_response_center_max_s=160., low_initial_floor_width_k=width)
    with pytest.raises(ValueError, match='初温|低保真'):
        core.JointDeepONet(core.Geometry(), physical_settings(), config)


def test_low_initial_floor_requires_observed_high_time_window():
    config = copy.deepcopy(smooth_model(independent_initial=True).model_config)
    config['low_initial_floor_width_k'] = .5
    with pytest.raises(ValueError, match='初温|低保真'):
        core.JointDeepONet(core.Geometry(), physical_settings(), config)


def test_low_initial_floor_checkpoint_roundtrips_and_rejects_mislabeling(tmp_path):
    config = copy.deepcopy(smooth_model(independent_initial=True).model_config)
    config.update(high_response_center_max_s=160., low_initial_floor_width_k=.5)
    model = core.JointDeepONet(core.Geometry(), physical_settings(), config)
    with torch.no_grad():
        model.low_bias.fill_(-4.25 / 250.)
    state = checkpoint(model)
    assert state['schema'] == 'joint_deeponet_8000_v6_fixed_low_initial'
    path = tmp_path / 'fixed_low.pt'
    core.save_atomic(state, path)
    restored, _ = core.load_model(path)
    for fidelity in ('low', 'high'):
        torch.testing.assert_close(model(trajectories(), fidelity),
                                   restored(trajectories(), fidelity), rtol=0., atol=0.)
    state['schema'] = 'joint_deeponet_8000_v5_high_observed_window'
    core.save_atomic(state, path)
    with pytest.raises(ValueError, match='版本|模式|检查点'):
        core.load_model(path)
    state['schema'] = 'joint_deeponet_8000_v6_fixed_low_initial'
    state['model_config'].pop('low_initial_floor_width_k')
    core.save_atomic(state, path)
    with pytest.raises(ValueError, match='版本|模式|检查点'):
        core.load_model(path)


def _saved_checkpoint(model, path):
    core.save_atomic(checkpoint(model), path)
    return path


@pytest.mark.parametrize('fixed_low, expected_schema', [
    (False, 'v5_high_observed_window'), (True, 'v6_fixed_low_initial'),
])
def test_high_observed_window_training_records_actual_checkpoint_schema(
        small_joint_project, monkeypatch, fixed_low, expected_schema):
    import argparse
    import yaml
    trainer = small_training_entrypoint(monkeypatch)
    config = small_training_config('all_training')
    config['model'].update(correction_power_degree=2, power_independent_initial=True,
                           high_response_center_max_s=5.)
    if not fixed_low:
        config['model'].pop('low_initial_floor_width_k')
    path = small_joint_project / 'configs/high_observed_window.yaml'
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    out = small_joint_project / '研究记录/high_observed_window'
    trainer.train(argparse.Namespace(root=str(small_joint_project), config=str(path),
                                 device='cpu', output=str(out), resume=None))
    _, state = core.load_model(out / '验证最佳模型.pt')
    assert state['schema'] == 'joint_deeponet_8000_' + expected_schema
    record = (out / '实施记录.md').read_text(encoding='utf-8')
    assert '检查点' + expected_schema in record
    initial_description = (
        '低保真零秒固定仿真初温22℃；高保真初温仍可拟合，原初始与水冷软约束保留'
        if fixed_low else '初温仍可拟合，原初始与水冷软约束保留'
    )
    assert ('|启用持续升温响应|低保真与高保真输出均使用非负升温系数；'
            + initial_description + '|') in record
    assert not (small_joint_project / 'data/processed/test_sensor_ring_raw.parquet').exists()


def test_v2_schema_cannot_claim_smooth_power_correction(tmp_path):
    state = checkpoint(smooth_model())
    state['schema'] = 'joint_deeponet_8000_v2_heating'
    path = tmp_path / 'wrong.pt'
    core.save_atomic(state, path)
    with pytest.raises(ValueError, match='版本|模式|检查点'):
        core.load_model(path)


def test_low_initialization_copies_only_low_network_and_keeps_new_correction(tmp_path):
    cfg = copy.deepcopy(smooth_model().model_config)
    cfg.pop('correction_power_degree')
    old = core.JointDeepONet(core.Geometry(), physical_settings(), cfg)
    signed_parameters(old)
    path = tmp_path / 'old.pt'
    core.save_atomic(checkpoint(old), path)
    model = smooth_model()
    correction_before = {k: v.clone() for k, v in model.correction.state_dict().items()}
    core.initialize_low_network(model, path)
    assert torch.equal(old(trajectories(), 'low'), model(trajectories(), 'low'))
    assert all(torch.equal(v, model.correction.state_dict()[k]) for k, v in correction_before.items())
    assert model.log_contact.item() == pytest.approx(np.log(model.settings.contact_initial), abs=1e-6)


@pytest.mark.parametrize('target_change', [{'blocks': 2}, {'temperature_scale_k': 300.}])
def test_low_initialization_rejects_partial_architecture_or_scale_migration_without_mutation(tmp_path, target_change):
    source = smooth_model()
    path = tmp_path / 'source.pt'
    core.save_atomic(checkpoint(source), path)
    cfg = copy.deepcopy(source.model_config)
    cfg.update(target_change)
    target = core.JointDeepONet(core.Geometry(), physical_settings(), cfg)
    before = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(ValueError, match='初始化|尺寸|尺度'):
        core.initialize_low_network(target, path)
    assert all(torch.equal(value, target.state_dict()[name]) for name, value in before.items())


def test_low_initialization_rejects_changed_simulation_material_conditions(tmp_path):
    source = smooth_model()
    path = tmp_path / 'source.pt'
    core.save_atomic(checkpoint(source), path)
    target = core.JointDeepONet(core.Geometry(), replace(physical_settings(), conductivity_sic=130.), source.model_config)
    before = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(ValueError, match='初始化|仿真'):
        core.initialize_low_network(target, path)
    assert all(torch.equal(value, target.state_dict()[name]) for name, value in before.items())


def test_freezing_low_network_keeps_coordinate_derivatives_and_can_be_reversed():
    model = smooth_model().double()
    signed_parameters(model)
    core.set_low_trainable(model, False)
    x = trajectories((5., 100.)).double().requires_grad_()
    first = core.grad(model(x), x)
    assert torch.isfinite(core.grad(first[:, 1:2], x)).all()
    model(x).square().mean().backward()
    assert all(p.grad is None for p in core.low_parameters(model))
    assert any(p.grad is not None for p in model.correction.parameters())
    core.set_low_trainable(model, True)
    assert all(p.requires_grad for p in core.low_parameters(model))


def test_worst_curve_loss_does_not_hide_one_bad_power():
    errors = torch.tensor([[0.], [0.], [10.]])
    groups = torch.arange(3)
    weights = torch.ones_like(errors)
    legacy = core.grouped_mse(errors, weights, groups)
    guarded = core.grouped_mse(errors, weights, groups, worst_fraction=.2)
    assert guarded == pytest.approx(.8 * float(legacy) + .2 * 100.)


def test_sensor_absolute_offset_is_penalized_even_when_temperature_rise_is_correct():
    class Offset(torch.nn.Module):
        temperature_scale = torch.tensor(250.)
        def forward(self, x):
            return 300. + x[:, 2:3] + 7.
    x = np.array([[.028, -.0175, t, 100., 0.] for t in (1., 2., 3.)], dtype=np.float32)
    table = core.Table(x, 300. + x[:, 2], np.ones(3), np.zeros(3), np.zeros(3)).validate()
    absolute, rise = core.observation_loss(Offset(), core.Pool(table, 'cpu'), np.arange(3), True, worst_fraction=.2)
    assert absolute == pytest.approx((7. / 250.) ** 2)
    assert rise == 0.


def test_power_curvature_loss_detects_sharp_dips_and_has_finite_gradients():
    class PowerCurve(torch.nn.Module):
        geometry = core.Geometry()
        temperature_scale = torch.tensor(250.)
        def __init__(self, curved):
            super().__init__()
            self.amplitude = torch.nn.Parameter(torch.tensor(float(curved)))
        def forward(self, x, fidelity='high'):
            return 300. + .2 * x[:, 3:4] + self.amplitude * (x[:, 3:4] / 800.).square() * 200.
    flat = core.power_smoothness_loss(PowerCurve(0.), 8, torch.Generator().manual_seed(5), 40.)
    model = PowerCurve(1.)
    curved = core.power_smoothness_loss(model, 8, torch.Generator().manual_seed(5), 40.)
    assert flat.item() < 1e-12
    assert curved.item() > 0.
    curved.backward()
    assert torch.isfinite(model.amplitude.grad) and model.amplitude.grad > 0


def test_power_smoothness_penalizes_localized_two_watt_dip():
    class LocalizedDip(torch.nn.Module):
        geometry = core.Geometry()
        temperature_scale = torch.tensor(250.)
        def __init__(self, amplitude):
            super().__init__()
            self.amplitude = torch.nn.Parameter(torch.tensor(float(amplitude)))
        def forward(self, x, fidelity='high'):
            p = x[:, 3:4]
            return 600. + .2 * p - self.amplitude * torch.exp(-((p - 400.) / 2.).square())
    linear = core.power_smoothness_loss(LocalizedDip(0.), 1024, torch.Generator().manual_seed(7), 40.)
    model = LocalizedDip(200.)
    penalty = core.power_smoothness_loss(model, 1024, torch.Generator().manual_seed(7), 40.)
    assert linear.item() < 1e-12
    assert penalty.item() > 1e-4
    penalty.backward()
    assert torch.isfinite(model.amplitude.grad) and model.amplitude.grad > 0


def trainer_module():
    spec = importlib.util.spec_from_file_location('improvement_trainer', ROOT / 'scripts/联合训练8000轮.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


def test_diagnostic_training_limit_requires_validation_only_and_cannot_read_test_labels():
    trainer = trainer_module()
    assert trainer.training_limit(8000, 30, True) == 30
    with pytest.raises(ValueError, match='验证'):
        trainer.training_limit(8000, 30, False)
    with pytest.raises(ValueError):
        trainer.training_limit(8000, 8001, True)


def test_snapshot_records_configured_epoch_total():
    model = smooth_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    state = core.snapshot(model, optimizer, scheduler, 0, [], {'training': {'epochs': 2000}},
                          {}, 1., np.random.default_rng(0))
    assert state['planned_epochs'] == 2000


def test_formal_training_uses_configured_epoch_total(small_joint_project, monkeypatch, capsys):
    import argparse
    import yaml
    trainer = small_training_entrypoint(monkeypatch)
    cfg = small_training_config('all_training')
    cfg['training']['epochs'] = 2
    path = small_joint_project / 'configs/short_formal.yaml'
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    out = small_joint_project / '研究记录/联合训练2轮_单元测试'
    trainer.train(argparse.Namespace(root=str(small_joint_project), config=str(path),
                                     device='cpu', output=str(out), resume=None))
    output = capsys.readouterr().out
    assert '第2/2轮' in output
    assert '未读取测试' in output
    assert '图册：' not in output
    assert (out / '第2轮模型.pt').is_file()
    _, state = core.load_model(out / '第2轮模型.pt')
    assert state['epoch'] == state['planned_epochs'] == 2
    complete = __import__('json').loads((out / '开发集完成记录.json').read_text(encoding='utf-8'))
    assert complete['完成轮数'] == 2
    assert complete['测试数据已读取'] is False
    assert complete['固定图册已更新'] is False
    assert not (out / '测试出图锁定记录.json').exists()
    assert not (out / '结果总览.html').exists()


def test_improvement_diagnostic_freeze_resume_and_no_test_publication(small_joint_project, monkeypatch):
    import argparse
    import json
    import yaml
    trainer = small_training_entrypoint(monkeypatch)
    monkeypatch.setattr(trainer, 'TOTAL_EPOCHS', 3)
    cfg = small_training_config('all_training')
    cfg['model']['correction_power_degree'] = 2
    cfg['model']['power_independent_initial'] = True
    cfg['training'].update(epochs=3, low_freeze_epochs=1, low_learning_rate=1e-5,
                           sensor_sampling='full', validation_every=1)
    cfg['loss_weights'].update(热端绝对温度=10., 冷端绝对温度=10., 功率平滑=10.)
    cfg['validation_selection'] = {'weights': {'顶部': .5, '热端': .25, '冷端': .25},
                                   'worst_power_fraction': .2}
    path = small_joint_project / 'configs/improvement.yaml'
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    out = small_joint_project / '研究记录/diagnostic'
    args = argparse.Namespace(root=str(small_joint_project), config=str(path), device='cpu',
                              output=str(out), resume=None, max_epochs=1, validation_only=True)
    trainer.train(args)
    _, first = core.load_model(out / '最近模型.pt')
    assert first['schema'] == 'joint_deeponet_8000_v6_fixed_low_initial'
    assert all(torch.equal(first['model_state'][k], core.load_model(out / '验证最佳模型.pt')[1]['model_state'][k])
               for k in first['model_state'])
    best_metrics = json.loads((out / '验证最佳指标.json').read_text(encoding='utf-8'))
    assert best_metrics['验证最佳轮次'] == 1
    assert best_metrics['综合选择分数_℃'] == first['best_score']
    assert not first['history'][0]['低保真参数参与更新']
    assert first['history'][0]['传感器训练方式'] == 'full'
    assert all(key in first['history'][0] for key in ('热端绝对温度', '冷端绝对温度', '功率平滑'))
    args.resume = str(out / '最近模型.pt')
    args.max_epochs = 3
    legacy_config = copy.deepcopy(cfg)
    legacy_config['model']['power_independent_initial'] = False
    path.write_text(yaml.safe_dump(legacy_config, allow_unicode=True), encoding='utf-8')
    with pytest.raises(ValueError, match='恢复时配置'):
        trainer.train(args)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    trainer.train(args)
    _, resumed = core.load_model(out / '最近模型.pt')
    assert resumed['epoch'] == 3 and len(resumed['history']) == 3
    _, resumed_best = core.load_model(out / '验证最佳模型.pt')
    resumed_metrics = json.loads((out / '验证最佳指标.json').read_text(encoding='utf-8'))
    assert resumed_metrics['验证最佳轮次'] == resumed_best['epoch']
    assert resumed_metrics['综合选择分数_℃'] == pytest.approx(resumed_best['best_score'])
    assert resumed['history'][1]['低保真参数参与更新']
    assert any(not torch.equal(first['model_state'][k], resumed['model_state'][k])
               for k in first['model_state'] if k.startswith('branch.'))
    complete = json.loads((out / '验证训练完成记录.json').read_text(encoding='utf-8'))
    assert complete['实际完成轮数'] == 3 and not complete['测试数据已读取']
    assert not (out / '测试出图锁定记录.json').exists()
    assert not (out / '第8000轮模型.pt').exists()
    assert not (out / '结果总览.html').exists()
    args.resume = None
    args.output = str(small_joint_project / '研究记录/uninterrupted')
    trainer.train(args)
    _, uninterrupted = core.load_model(Path(args.output) / '最近模型.pt')
    assert all(torch.equal(v, uninterrupted['model_state'][k]) for k, v in resumed['model_state'].items())


def test_training_loss_plot_includes_both_absolute_sensor_losses_and_power_smoothness(tmp_path, monkeypatch):
    import joint_temperature_figures as figures
    plotted = {}
    def capture(path, x, series, *args, **kwargs):
        plotted[path.name] = [name for name, _ in series]
    monkeypatch.setattr(figures, 'save_curve', capture)
    history = [{'轮次': 1, '接触热阻_m2K_W': 7e-5, '总损失': .1,
                '热端绝对温度': .01, '冷端绝对温度': .01, '功率平滑': .001}]
    figures.training_plots(history, tmp_path, {})
    names = plotted['损失函数变化.png']
    assert all(name in names for name in ('热端绝对温度', '冷端绝对温度', '功率平滑'))
    assert '环温绝对值' not in names


def test_training_loss_plot_shows_early_rise_losses_when_present(tmp_path, monkeypatch):
    import joint_temperature_figures as figures
    plotted = {}
    monkeypatch.setattr(figures, 'save_curve',
                        lambda path, x, series, *args, **kwargs:
                        plotted.update({path.name: [name for name, _ in series]}))
    history = [{'轮次': 1, '接触热阻_m2K_W': 7e-5, '总损失': .1,
                '热端早期温升': .001, '冷端早期温升': .002,
                '热端末点绝对温度': .003}]
    figures.training_plots(history, tmp_path, {})
    assert all(name in plotted['损失函数变化.png'] for name in ('热端早期温升', '冷端早期温升'))
    assert '热端末点绝对温度' in plotted['损失函数变化.png']


def test_contact_resistance_plot_has_readable_scale_label(tmp_path, monkeypatch):
    import joint_temperature_figures as figures
    labels = {}
    monkeypatch.setattr(figures, 'save_curve',
                        lambda path, x, series, title, xlabel, ylabel, *args, **kwargs:
                        labels.update({path.name: ylabel}))
    figures.training_plots([{'轮次': 1, '总损失': .1,
                             '接触热阻_m2K_W': 7e-5}], tmp_path, {})
    label = labels['界面接触热阻变化.png']
    assert '10^-5' in label
    assert '\N{SUPERSCRIPT MINUS}' not in label
    assert '\N{SUPERSCRIPT FIVE}' not in label


def test_multi_series_loss_curve_places_legend_outside_the_data_axes(tmp_path, monkeypatch):
    import joint_temperature_figures as figures
    from matplotlib.axes import Axes
    figures.setup_font()
    actual_legend = Axes.legend
    seen = []
    def record_legend(axis, *args, **kwargs):
        seen.append(kwargs)
        return actual_legend(axis, *args, **kwargs)
    monkeypatch.setattr(Axes, 'legend', record_legend)
    names = [(f'第{i}项', np.ones(4) * (i + 1)) for i in range(15)]
    figures.save_curve(tmp_path / '复杂损失.png', range(4), names, '损失', '轮次', '损失', True)
    assert seen[-1]['bbox_to_anchor'][0] >= 1.
    assert (tmp_path / '复杂损失.png').is_file()
    figures.save_curve(tmp_path / '普通曲线.png', range(4), names[:2], '损失', '轮次', '损失', True)
    assert 'bbox_to_anchor' not in seen[-1]
