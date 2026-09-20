"""开发集分析只允许使用训练/验证观测，不触碰测试标签或正式图册。"""
from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/联合训练开发结果分析.py'
SOURCE = ROOT / '研究记录/联合训练600轮_第09轮_低保真初温平滑锚定_20260920'
sys.path.insert(0, str(ROOT / 'scripts'))


def analysis_module():
    assert SCRIPT.is_file(), '尚未实现训练/验证独立分析脚本'
    spec = importlib.util.spec_from_file_location('joint_development_analysis', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_late_trend_uses_observed_last_twenty_seconds_and_ols():
    module = analysis_module()
    time = np.array([0., 5., 12., 20., 30.])
    measured = np.array([20., 22., 25., 25., 25.]) + 273.15
    estimated = np.array([20., 22., 25., 25.8, 26.8]) + 273.15

    result = module.late_trend(time, measured, estimated)

    assert result['窗口起点_s'] == 10.
    assert result['窗口点数'] == 3
    assert result['实测趋势_℃每20秒'] == pytest.approx(0.)
    assert result['预测趋势_℃每20秒'] == pytest.approx(2.)
    assert result['趋势误差_℃每20秒'] == pytest.approx(2.)
    assert result['末时刻误差_℃'] == pytest.approx(1.8)
    assert result['末观测时间_s'] == 30.


@pytest.mark.parametrize('times,measured,predicted', [
    ([1.], [25.], [25.]),
    ([1., 30.], [25., 26.], [25., 27.]),
    ([1., 2.], [25., float('nan')], [25., 27.]),
])
def test_late_trend_rejects_ill_defined_slope(times, measured, predicted):
    with pytest.raises(ValueError):
        analysis_module().late_trend(times, measured, predicted)


def test_top_time_rows_use_frame_weights_and_reconstruct_power_metrics():
    module = analysis_module()
    from joint_temperature_core import Table, table_metrics

    coordinates = np.array([
        [0., 0., 0., 55., 1.], [1., 0., 0., 55., 1.],
        [0., 0., 1., 55., 1.], [1., 0., 1., 55., 1.],
        [0., 0., 0., 115.2, 1.],
    ])
    table = Table(coordinates, np.zeros(5), np.array([1., 3., 1., 1., 2.]),
                  np.array([0, 0, 0, 0, 1])).validate()
    prediction = np.array([2., -2., 4., 0., -3.])

    rows = module.top_time_rows(table, prediction, '训练')

    assert [row['功率_W'] for row in rows] == pytest.approx([55., 55., 115.2])
    assert [row['时间_s'] for row in rows] == pytest.approx([0., 1., 0.])
    assert [row['时间权重和'] for row in rows] == pytest.approx([4., 2., 2.])
    assert [row['均方根误差_℃'] for row in rows] == pytest.approx([2., np.sqrt(8.), 3.])
    assert [row['平均偏差_℃'] for row in rows] == pytest.approx([-1., 2., -3.])
    original = table_metrics(table, prediction)['逐功率'][0]
    per_time = rows[:2]
    weights = np.array([row['时间权重和'] for row in per_time])
    assert np.sqrt(np.average([row['均方根误差_℃'] ** 2 for row in per_time], weights=weights)) == pytest.approx(original['均方根误差_℃'])
    assert np.average([row['平均偏差_℃'] for row in per_time], weights=weights) == pytest.approx(original['平均偏差_℃'])


def test_report_rejects_outside_project_and_incomplete_run():
    module = analysis_module()
    with pytest.raises(ValueError, match='项目内'):
        module.run_analysis(Path('/'))
    with tempfile.TemporaryDirectory(dir=ROOT / '研究记录') as folder:
        run = Path(folder)
        (run / '实际配置.yaml').symlink_to(SOURCE / '实际配置.yaml')
        (run / '验证最佳模型.pt').symlink_to(SOURCE / '验证最佳模型.pt')
        with pytest.raises((ValueError, FileNotFoundError), match='完成|末轮|模型'):
            module.run_analysis(run)
        assert not (run / '开发集分析').exists()


def test_complete_run_exports_train_validation_curves_without_test_data(monkeypatch):
    module = analysis_module()
    import joint_temperature_core as core

    parquet_reads = []
    original_read = core.read_parquet

    def inspect_read(path):
        parquet_reads.append(Path(path).name)
        if Path(path).name.startswith('test_'):
            raise AssertionError('开发集分析不允许读取测试观测')
        return original_read(path)

    monkeypatch.setattr(core, 'read_parquet', inspect_read)
    with tempfile.TemporaryDirectory(dir=ROOT / '研究记录') as folder:
        run = Path(folder)
        for name in ('实际配置.yaml', '验证最佳模型.pt', '第600轮模型.pt'):
            (run / name).symlink_to(SOURCE / name)
        output = module.run_analysis(run)
        assert output == run / '开发集分析'
        assert (output / '分析报告.md').is_file()
        report = (output / '分析报告.md').read_text(encoding='utf-8')
        assert '仅训练集和验证集' in report
        assert '无实测外推' in report
        assert '顶部' in report and '20秒' in report
        assert '末段趋势误差绝对值平均' in report
        assert '顶部RMSE逐功率平均' in report
        assert '|训练|热端|' in report and '|验证|冷端|' in report
        assert '169.000W' not in report and '339.000W' not in report and '634.000W' not in report
        assert set(parquet_reads) == {'experiment_ir_radial.parquet', 'sensor_ring_raw.parquet'}
        assert (output / '训练/55.000W/热端温度曲线.png').is_file()
        assert (output / '训练/55.000W/热端误差.png').is_file()
        assert (output / '验证/115.200W/冷端温度曲线.csv').is_file()
        assert (output / '验证/115.200W/冷端误差.csv').is_file()
        assert len(list(output.rglob('*温度曲线.png'))) == 30
        assert len(list(output.rglob('*误差.png'))) == 30
        assert len(list(output.rglob('*温度曲线.csv'))) == 30
        assert len(list(output.rglob('*误差.csv'))) == 30
        assert (output / '顶部逐功率RMSE.csv').is_file()
        with (output / '训练/55.000W/热端误差.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 50
        assert {'时间_s', '预测减实测_℃'} <= set(rows[0])
        with (output / '顶部逐功率RMSE.csv').open(encoding='utf-8-sig', newline='') as handle:
            top = list(csv.DictReader(handle))
        assert len(top) == 15
        assert {row['数据划分'] for row in top} == {'训练', '验证'}
        assert len(list(output.rglob('顶部误差随时间.png'))) == 15
        assert len(list(output.rglob('顶部误差随时间.csv'))) == 15
        assert (output / '验证/115.200W/顶部误差随时间.png').is_file()
        with (output / '验证/115.200W/顶部误差随时间.csv').open(encoding='utf-8-sig', newline='') as handle:
            time_rows = list(csv.DictReader(handle))
        assert len(time_rows) > 1
        assert {'时间_s', '均方根误差_℃', '平均偏差_℃', '时间权重和'} <= set(time_rows[0])
        assert [float(row['时间_s']) for row in time_rows] == sorted(float(row['时间_s']) for row in time_rows)
        top_115 = next(row for row in top if row['数据划分'] == '验证' and
                       float(row['功率_W']) == pytest.approx(115.2))
        weights = [float(row['时间权重和']) for row in time_rows]
        assert np.sqrt(np.average([float(row['均方根误差_℃']) ** 2 for row in time_rows],
                                  weights=weights)) == pytest.approx(float(top_115['均方根误差_℃']))
        assert np.average([float(row['平均偏差_℃']) for row in time_rows],
                          weights=weights) == pytest.approx(float(top_115['平均偏差_℃']))
