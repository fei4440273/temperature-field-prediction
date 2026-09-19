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


def smooth_model(degree=2):
    torch.manual_seed(42)
    return core.JointDeepONet(core.Geometry(), physical_settings(), {
        'time_response': 'monotone_heating', 'width': 8, 'latent_dim': 8, 'blocks': 1,
        'correction_width': 8, 'correction_depth': 2, 'temperature_scale_k': 250.,
        'learn_contact': True, 'correction_power_degree': degree,
    })


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


def test_improvement_diagnostic_freeze_resume_and_no_test_publication(small_joint_project, monkeypatch):
    import argparse
    import json
    import yaml
    trainer = small_training_entrypoint(monkeypatch)
    monkeypatch.setattr(trainer, 'TOTAL_EPOCHS', 3)
    cfg = small_training_config('all_training')
    cfg['model']['correction_power_degree'] = 2
    cfg['training'].update(epochs=3, low_freeze_epochs=1, low_learning_rate=1e-5,
                           sensor_sampling='full', validation_every=1)
    cfg['loss_weights'].pop('环温绝对值')
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
