"""Low-fidelity analysis must use only simulated training labels."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from joint_temperature_core import (
    Geometry, JointDeepONet, PhysicalSettings, fixed_splits, predict, read_yaml,
    save_atomic, sha256, snapshot,
)
from test_joint_8000 import small_joint_project, small_training_config


def analysis_module():
    path = ROOT / 'scripts/联合训练低保真训练集分析.py'
    assert path.is_file(), '尚未实现低保真训练集分析脚本'
    spec = importlib.util.spec_from_file_location('joint_low_training_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def completed_low_run(small_joint_project):
    root = small_joint_project
    config = small_training_config('all_training')
    config['training']['epochs'] = 2
    (root / 'configs/联合训练600轮_单元测试.yaml').write_text(
        yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    run = root / '研究记录/单元测试已完成低保真分析'
    run.mkdir(parents=True)
    (run / '实际配置.yaml').write_text(
        yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    (root / 'data/processed/manifest.json').write_text(
        '{"仅合成单元测试使用": true}', encoding='utf-8')
    geometry = Geometry.from_project(root)
    physical = PhysicalSettings.from_project(root, config)
    model = JointDeepONet(geometry, physical, config['model'])
    with torch.no_grad():
        model.low_bias.fill_(-.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    splits = fixed_splits(root, 'all_training')
    for epoch, name in ((1, '验证最佳模型.pt'), (2, '第2轮模型.pt')):
        state = snapshot(model, optimizer, scheduler, epoch, [], config, splits,
                         1., np.random.default_rng(1))
        save_atomic(state, run / name)
    (run / '开发集完成记录.json').write_text(json.dumps({
        '完成轮数': 2, '验证最佳轮次': 1,
        '验证最佳检查点SHA256': sha256(run / '验证最佳模型.pt'),
        '测试数据已读取': False,
    }, ensure_ascii=False), encoding='utf-8')
    times = np.arange(0., 201., 2., dtype=np.float32)
    for power in splits['low']['training']:
        rows = []
        for time_s in times:
            for material, radius, height in ((0, .035, -.015), (1, 0., 0.),
                                             (1, .001, 0.), (1, .01, -.005)):
                rows.append({'r_m': radius, 'z_m': height, 'time_s': float(time_s),
                             'power_w': power, 'material_id': material,
                             'temperature_k': 295.15 + float(time_s) * power / 1000.})
        pl.DataFrame(rows).write_parquet(
            root / 'data/processed/simulation' / f'{power:g}W.parquet')
    return root, run, model


def test_full_analysis_reports_actual_time_grid_and_central_two_second_rise(completed_low_run, monkeypatch):
    root, run, model = completed_low_run
    module = analysis_module()
    original_read = module.read_parquet
    touched = []

    def inspect_read(path):
        touched.append(Path(path))
        assert Path(path).parent == root / 'data/processed/simulation'
        return original_read(path)

    monkeypatch.setattr(module, 'read_parquet', inspect_read)
    output = module.run_analysis(run, root=root, batch_size=64)
    assert output == run / '低保真训练集分析'
    assert len(touched) == 80
    assert {path.stem for path in touched} == {f'{power}W' for power in range(10, 801, 10)}
    intervals = read_rows(output / '逐功率逐材料时间段误差.csv')
    samples = read_rows(output / '逐功率逐材料指定时刻误差.csv')
    global_intervals = read_rows(output / '全场时间段误差.csv')
    global_samples = read_rows(output / '全场指定时刻误差.csv')
    power_rows = read_rows(output / '逐功率全场误差.csv')
    top = read_rows(output / 'SiC顶面近中心2秒.csv')
    assert len(intervals) == 80 * 2 * 8
    assert len(samples) == 80 * 2 * 8
    assert len(global_intervals) == len(global_samples) == 8
    assert len(power_rows) == len(top) == 80
    assert '仿真2秒温度_℃' in top[0]
    assert '预测减仿真温升_℃' in top[0]
    assert not any('实测' in column for column in top[0])
    assert {float(row['时间_s']) for row in samples} == {0., 2., 4., 10., 20., 50., 100., 200.}
    assert {row['时间段_s'] for row in intervals} == {
        '0', '2', '4', '6-10', '12-20', '22-50', '52-100', '102-200',
    }
    assert {row['时间段_s'] for row in global_intervals} == {
        '0', '2', '4', '6-10', '12-20', '22-50', '52-100', '102-200',
    }
    assert {float(row['时间_s']) for row in global_samples} == {
        0., 2., 4., 10., 20., 50., 100., 200.,
    }
    assert all(int(row['测点数']) > 0 for row in global_intervals + global_samples)
    zero = next(row for row in global_samples if float(row['时间_s']) == 0.)
    assert int(zero['测点数']) == 80 * 4
    assert float(zero['均方根误差_℃']) == pytest.approx(0., abs=1e-4)
    assert float(zero['预测低于仿真初温比例_%']) == pytest.approx(0.)
    assert all(int(row['测点数']) > 0 for row in samples)
    assert all(int(row['中心测点数']) == 2 for row in top)
    item = next(row for row in top if float(row['功率_W']) == 10.)
    x = np.array([[radius, 0., t, 10., 1.]
                  for t in (0., 2.) for radius in (0., .001)], dtype=np.float32)
    values = predict(model, x, fidelity='low').reshape(2, 2).astype(np.float64).mean(1)
    measured_rise = float(np.float32(295.15 + .02) - np.float32(295.15))
    assert float(item['预测减仿真温升_℃']) == pytest.approx(
        (values[1] - values[0]) - measured_rise, abs=1e-5)
    report = (output / '分析报告.md').read_text(encoding='utf-8')
    assert '80个仿真训练功率' in report
    assert '最差功率' in report and '零秒' in report
    assert '全场分时间段误差' in report and '|102-200|' in report
    assert sha256(ROOT / 'scripts/联合训练低保真训练集分析.py') in report
    assert sha256(root / 'data/processed/manifest.json') in report
    assert '仿真训练集' in report and '测试标签未读取' in report
    assert '169.000W' not in report and '339.000W' not in report


def test_incomplete_or_wrong_split_is_rejected_without_output(completed_low_run):
    root, run, _ = completed_low_run
    module = analysis_module()
    final = run / '第2轮模型.pt'
    final.rename(run / '未完成模型.pt')
    with pytest.raises((ValueError, FileNotFoundError), match='完成|末轮'):
        module.run_analysis(run, root=root)
    assert not (run / '低保真训练集分析').exists()
    (run / '未完成模型.pt').rename(final)
    config = read_yaml(run / '实际配置.yaml')
    config['data']['low_fidelity_mode'] = 'legacy_split'
    (run / '实际配置.yaml').write_text(
        yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    with pytest.raises(ValueError, match='全量|80|划分'):
        module.run_analysis(run, root=root)
    assert not (run / '低保真训练集分析').exists()


def test_missing_simulation_time_is_rejected_without_partial_output(completed_low_run):
    root, run, _ = completed_low_run
    module = analysis_module()
    path = root / 'data/processed/simulation/800W.parquet'
    pl.read_parquet(path).filter(pl.col('time_s') != 4.).write_parquet(path)
    with pytest.raises(ValueError, match='4|时间'):
        module.run_analysis(run, root=root, batch_size=128)
    assert not (run / '低保真训练集分析').exists()


def test_run_outside_project_is_rejected_without_reading_data(completed_low_run, tmp_path):
    root, _, _ = completed_low_run
    module = analysis_module()
    with pytest.raises(ValueError, match='项目内|研究记录'):
        module.run_analysis(tmp_path / '外部项目', root=root)


def test_simulation_symlink_is_rejected_before_reading_observations(completed_low_run, monkeypatch):
    root, run, _ = completed_low_run
    module = analysis_module()
    path = root / 'data/processed/simulation/20W.parquet'
    path.unlink()
    path.symlink_to(root / 'data/processed/experiment_ir_radial.parquet')
    original = module.read_parquet

    def prohibit_read(target):
        if Path(target) == path:
            raise AssertionError('不能把实验温度当作仿真训练标签读取')
        return original(target)

    monkeypatch.setattr(module, 'read_parquet', prohibit_read)
    with pytest.raises(ValueError, match='符号链接|仿真文件'):
        module.run_analysis(run, root=root, batch_size=128)
    assert not (run / '低保真训练集分析').exists()


def test_symlinked_manifest_rejected_before_creating_report(completed_low_run):
    root, run, _ = completed_low_run
    module = analysis_module()
    manifest = root / 'data/processed/manifest.json'
    manifest.unlink()
    manifest.symlink_to(root / 'data/processed/experiment_ir_radial.parquet')
    with pytest.raises(ValueError, match='处理清单|符号链接'):
        module.run_analysis(run, root=root, batch_size=128)
    assert not (run / '低保真训练集分析').exists()


@pytest.mark.parametrize('digest', [None, 'f' * 64])
def test_missing_or_tampered_best_model_hash_is_rejected(completed_low_run, digest):
    root, run, _ = completed_low_run
    module = analysis_module()
    path = run / '开发集完成记录.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    if digest is None:
        del record['验证最佳检查点SHA256']
    else:
        record['验证最佳检查点SHA256'] = digest
    path.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
    with pytest.raises(ValueError, match='SHA256'):
        module.run_analysis(run, root=root)
    assert not (run / '低保真训练集分析').exists()


def test_missing_one_material_node_at_one_time_is_rejected(completed_low_run):
    root, run, _ = completed_low_run
    module = analysis_module()
    path = root / 'data/processed/simulation/800W.parquet'
    frame = pl.read_parquet(path)
    frame.filter(~((pl.col('time_s') == 4.) & (pl.col('material_id') == 1)
                   & (pl.col('r_m') == 0.))).write_parquet(path)
    with pytest.raises(ValueError, match='节点|时间'):
        module.run_analysis(run, root=root, batch_size=128)
    assert not (run / '低保真训练集分析').exists()


@pytest.mark.parametrize('relative', ('data', 'data/processed', 'data/processed/simulation'))
def test_symlinked_simulation_parent_is_rejected_before_reading(completed_low_run, monkeypatch,
                                                                 relative):
    root, run, _ = completed_low_run
    module = analysis_module()
    folder = root / relative
    destination = folder.with_name(folder.name + '_backup_for_test')
    folder.rename(destination)
    folder.symlink_to(destination, target_is_directory=True)

    def prohibit_read(_):
        raise AssertionError('不能经目录符号链接读取仿真文件')

    monkeypatch.setattr(module, 'read_parquet', prohibit_read)
    with pytest.raises(ValueError, match='符号链接|目录'):
        module.run_analysis(run, root=root, batch_size=128)
    assert not (run / '低保真训练集分析').exists()
