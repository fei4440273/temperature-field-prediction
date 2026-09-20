"""Training-only sensor trend checks; synthetic data are not experimental results."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import joint_temperature_core as core
from test_joint_8000 import small_joint_project, small_training_config, small_training_entrypoint


class LinearSensor(torch.nn.Module):
    temperature_scale = torch.tensor(250.)

    def __init__(self, slope, offset=0.):
        super().__init__()
        self.slope = torch.nn.Parameter(torch.tensor(float(slope)))
        self.offset = offset

    def forward(self, x):
        return 298.15 + self.offset + self.slope * x[:, 2:3]


def sensor_table(duration=40):
    rows = []
    for group, (power, slope) in enumerate(((55., .005), (729., .01))):
        for second in range(duration + 1):
            rows.append((group, [.028, -.0175, second, power, 0.], 298.15 + slope * second))
    return core.Table(np.array([r[1] for r in rows]), np.array([r[2] for r in rows]),
                      np.ones(len(rows)), np.array([r[0] for r in rows])).validate()


def test_tail_slope_loss_matches_measured_trends_per_power_and_has_gradient():
    pool = core.sensor_tail_pool(sensor_table(), 'cpu', window_seconds=20.)
    model = LinearSensor(.05, offset=7.)
    loss = core.sensor_tail_slope_loss(model, pool, window_seconds=20., worst_fraction=.2)
    per_group = np.array([(.05 - .005) * 20 / 250, (.05 - .01) * 20 / 250]) ** 2
    assert float(loss) == pytest.approx(.8 * per_group.mean() + .2 * per_group.max(), rel=1e-4)
    loss.backward()
    assert model.slope.grad is not None and float(model.slope.grad) > 0
    assert float(core.sensor_tail_slope_loss(LinearSensor(.05, offset=-70.), pool)) == pytest.approx(
        float(core.sensor_tail_slope_loss(LinearSensor(.05), pool)), rel=1e-5)


def test_tail_pool_uses_only_each_groups_last_observed_twenty_seconds():
    table = sensor_table()
    table.y[table.x[:, 2] < 20] += 900.
    pool = core.sensor_tail_pool(table, 'cpu', window_seconds=20.)
    assert len(pool.table.x) == 42
    assert set(pool.table.x[:, 2]) == set(range(20, 41))
    assert set(pool.table.x[:, 3]) == {55., 729.}


def test_tail_pool_rejects_curves_shorter_than_the_declared_window():
    with pytest.raises(ValueError, match='末段|20'):
        core.sensor_tail_pool(sensor_table(duration=10), 'cpu', window_seconds=20.)


def test_tail_endpoint_loss_anchors_only_real_last_point_per_training_power():
    table = sensor_table()
    keep = ~((table.x[:, 3] == 729.) & (table.x[:, 2] == 40.))
    table = core.Table(table.x[keep], table.y[keep], table.weight[keep],
                       table.group[keep]).validate()
    table.y[table.x[:, 2] < 39.] += 900.
    pool = core.sensor_tail_pool(table, 'cpu', window_seconds=20.)
    model = LinearSensor(.05, offset=7.)
    loss = core.sensor_tail_endpoint_loss(model, pool, worst_fraction=.2)
    errors = np.array([7. + (.05 - .005) * 40., 7. + (.05 - .01) * 39.]) / 250.
    assert float(loss) == pytest.approx(.8 * np.mean(errors**2) + .2 * max(errors**2), rel=1e-4)
    assert float(loss) != pytest.approx(float(core.sensor_tail_endpoint_loss(
        LinearSensor(.05), pool, worst_fraction=.2)))
    loss.backward()
    assert model.slope.grad is not None and float(model.slope.grad) > 0


def test_early_rise_uses_only_observed_one_to_five_seconds_and_preserves_gradients():
    table = sensor_table()
    table.y[(table.x[:, 2] < 1) | (table.x[:, 2] > 5)] += 900.
    pool = core.sensor_early_pool(table, 'cpu', end_seconds=5.)
    assert len(pool.table.x) == 10
    assert set(pool.table.x[:, 2]) == set(range(1, 6))
    assert set(pool.table.x[:, 3]) == {55., 729.}
    model = LinearSensor(.05, offset=7.)
    loss = core.sensor_early_rise_loss(model, pool, worst_fraction=.2)
    per_group = np.array([(.05 - .005) / 250, (.05 - .01) / 250]) ** 2 * 6
    assert float(loss) == pytest.approx(.8 * per_group.mean() + .2 * per_group.max(), rel=5e-4)
    loss.backward()
    assert model.slope.grad is not None and float(model.slope.grad) > 0
    assert float(core.sensor_early_rise_loss(LinearSensor(.05, offset=-70.), pool)) == pytest.approx(
        float(core.sensor_early_rise_loss(LinearSensor(.05), pool)), rel=5e-4)


def test_early_rise_refuses_a_missing_observed_endpoint():
    table = sensor_table()
    missing = ~((table.x[:, 3] == 55.) & (table.x[:, 2] == 5.))
    table = core.Table(table.x[missing], table.y[missing], table.weight[missing],
                       table.group[missing]).validate()
    with pytest.raises(ValueError, match='早期|5'):
        core.sensor_early_pool(table, 'cpu', end_seconds=5.)
    with pytest.raises(ValueError, match='早期|1'):
        core.sensor_early_pool(sensor_table(), 'cpu', end_seconds=1.)


def test_training_logs_separate_early_sensor_losses_without_test_files(small_joint_project, monkeypatch):
    import yaml
    trainer = small_training_entrypoint(monkeypatch)
    cfg = small_training_config('all_training')
    cfg['training'].update(sensor_sampling='full', sensor_early_window_s=2.)
    cfg['loss_weights'].update(热端绝对温度=10., 冷端绝对温度=10.,
                               热端早期温升=10., 冷端早期温升=10.)
    path = small_joint_project / 'configs/early_rise.yaml'
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    out = small_joint_project / '研究记录/early_rise'
    trainer.train(argparse.Namespace(root=str(small_joint_project), config=str(path),
                                 device='cpu', output=str(out), resume=None))
    row = json.loads((out / '训练记录.jsonl').read_text(encoding='utf-8'))
    assert row['热端早期温升'] > 0 and row['冷端早期温升'] > 0
    assert json.loads((out / '开发集完成记录.json').read_text(encoding='utf-8'))['测试数据已读取'] is False
    assert not (small_joint_project / 'data/processed/test_sensor_ring_raw.parquet').exists()


def test_training_logs_actual_hot_endpoint_loss_without_test_files(small_joint_project, monkeypatch):
    import yaml
    trainer = small_training_entrypoint(monkeypatch)
    cfg = small_training_config('all_training')
    cfg['training'].update(sensor_sampling='full', sensor_tail_window_s=3.)
    cfg['loss_weights'].update(热端绝对温度=10., 冷端绝对温度=10.,
                               热端末段趋势=30., 冷端末段趋势=30.,
                               热端末点绝对温度=10.)
    path = small_joint_project / 'configs/hot_endpoint.yaml'
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    out = small_joint_project / '研究记录/hot_endpoint'
    trainer.train(argparse.Namespace(root=str(small_joint_project), config=str(path),
                                 device='cpu', output=str(out), resume=None))
    row = json.loads((out / '训练记录.jsonl').read_text(encoding='utf-8'))
    assert row['热端末点绝对温度'] > 0
    assert row['热端末段趋势'] > 0
    assert not (out / '测试出图锁定记录.json').exists()
    assert not (small_joint_project / 'data/processed/test_sensor_ring_raw.parquet').exists()
