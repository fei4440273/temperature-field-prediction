#!/usr/bin/env python3
"""对已完成的联合训练运行仅用训练/验证观测生成可审查的中文分析。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from joint_temperature_core import (
    KELVIN, Geometry, fixed_splits, load_model, load_observations,
    predict, read_yaml, sha256, table_metrics,
)
from joint_temperature_figures import save_comparison_curve, save_curve, setup_font


ROOT = Path(__file__).resolve().parents[1]


def late_trend(time_s, measured_k, predicted_k):
    """计算最后20秒实测窗口内的两条OLS直线斜率，单位为℃/20秒。"""
    t = np.asarray(time_s, dtype=np.float64).reshape(-1)
    measured = np.asarray(measured_k, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted_k, dtype=np.float64).reshape(-1)
    if (len(t) < 2 or len(t) != len(measured) or len(t) != len(predicted)
            or not all(np.isfinite(values).all() for values in (t, measured, predicted))
            or np.any(np.diff(t) <= 0)):
        raise ValueError('末段趋势需要至少两个按时间递增的有效温度观测。')
    start = float(t[-1] - 20.)
    mask = t >= start
    selected_t = t[mask]
    if len(selected_t) < 2:
        raise ValueError('末20秒内不足两个不同时间的实测点，不能计算趋势。')
    centered = selected_t - selected_t.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        raise ValueError('末段实测时间重复，不能计算趋势。')
    true_slope = float(np.dot(centered, measured[mask] - measured[mask].mean()) / denominator * 20.)
    predicted_slope = float(np.dot(centered, predicted[mask] - predicted[mask].mean()) / denominator * 20.)
    return {
        '窗口起点_s': start,
        '窗口点数': int(mask.sum()),
        '末观测时间_s': float(t[-1]),
        '实测趋势_℃每20秒': true_slope,
        '预测趋势_℃每20秒': predicted_slope,
        '趋势误差_℃每20秒': predicted_slope - true_slope,
        '末时刻误差_℃': float(predicted[-1] - measured[-1]),
    }


def top_time_rows(table, prediction, split):
    """按各实测时间的径向 frame_weight 汇总，与逐功率指标保持同一权重口径。"""
    error = np.asarray(prediction).reshape(-1) - table.y.reshape(-1)
    if len(error) != len(table.x) or not np.isfinite(error).all():
        raise ValueError('顶部预测长度不符或包含无效值。')
    rows = []
    for gid in np.unique(table.group):
        group = table.group == gid
        power = float(table.x[group][0, 3])
        for time in np.unique(table.x[group, 2]):
            mask = group & (table.x[:, 2] == time)
            weights = table.weight[mask].reshape(-1).astype(np.float64)
            total = float(weights.sum())
            if total <= 0:
                raise ValueError(f'{power:.3f} W 在 {time:g} 秒的顶面权重和为零。')
            values = error[mask].astype(np.float64)
            rows.append({'数据划分': split, '功率_W': power, '时间_s': float(time),
                         '时间权重和': total,
                         '均方根误差_℃': float(np.sqrt(np.sum(weights * values ** 2) / total)),
                         '平均偏差_℃': float(np.sum(weights * values) / total)})
    return rows


def _inside_project(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if root not in resolved.parents:
        raise ValueError('运行目录及所有输出必须位于本项目内。')
    return path


def _write_csv(path: Path, root: Path, columns: list[str], rows: list[dict]):
    _inside_project(path, root)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _completed_run(run: Path, root: Path):
    _inside_project(run, root)
    config_path = _inside_project(run / '实际配置.yaml', root)
    if not config_path.is_file():
        raise FileNotFoundError('缺少实际配置，无法核实运行是否完成。')
    config = read_yaml(config_path)
    planned = config.get('training', {}).get('epochs')
    if type(planned) is not int or planned < 1:
        raise ValueError('实际配置的计划训练轮数无效。')
    final_path = _inside_project(run / f'第{planned}轮模型.pt', root)
    best_path = _inside_project(run / '验证最佳模型.pt', root)
    if not final_path.is_file() or not best_path.is_file():
        raise FileNotFoundError('缺少末轮模型或验证最佳模型，不能分析未完成的运行。')
    splits = fixed_splits(root, config.get('data', {}).get('low_fidelity_mode', 'legacy_split'))
    _, final_state = load_model(final_path, 'cpu')
    model, best_state = load_model(best_path, 'cpu')
    for state in (final_state, best_state):
        if (state.get('planned_epochs') != planned or state.get('config') != config
                or state.get('splits') != splits):
            raise ValueError('检查点的轮数、实际配置或数据划分不一致，拒绝生成分析。')
    if final_state['epoch'] != planned or not 1 <= best_state['epoch'] <= planned:
        raise ValueError('末轮模型尚未完成计划轮数，或验证最佳轮次无效。')
    return config, splits, model, best_state['epoch'], best_path


def _sensor_rows(model, table, name, split, output: Path, root: Path):
    prediction = predict(model, table.x)
    results = []
    for gid in np.unique(table.group):
        ids = np.flatnonzero(table.group == gid)
        ids = ids[np.argsort(table.x[ids, 2])]
        t = table.x[ids, 2]
        measured = table.y[ids, 0]
        estimated = prediction[ids]
        power = float(table.x[ids[0], 3])
        folder = _inside_project(output / split / f'{power:.3f}W', root)
        folder.mkdir(parents=True, exist_ok=True)
        curve_path = _inside_project(folder / f'{name}温度曲线.png', root)
        error_path = _inside_project(folder / f'{name}误差.png', root)
        save_comparison_curve(curve_path, t, measured - KELVIN, estimated - KELVIN,
                              f'{split} {power:.3f} W {name}', '时间 / s')
        save_curve(error_path, t, [('预测减实测', estimated - measured)],
                   f'{split} {power:.3f} W {name}误差', '时间 / s', '误差 / ℃')
        _write_csv(folder / f'{name}温度曲线.csv', root,
                   ['时间_s', '实测温度_℃', '预测温度_℃'],
                   [{'时间_s': f'{time:.3f}', '实测温度_℃': f'{true - KELVIN:.4f}',
                     '预测温度_℃': f'{pred - KELVIN:.4f}'}
                    for time, true, pred in zip(t, measured, estimated)])
        _write_csv(folder / f'{name}误差.csv', root,
                   ['时间_s', '预测减实测_℃'],
                   [{'时间_s': f'{time:.3f}', '预测减实测_℃': f'{pred - true:.4f}'}
                    for time, true, pred in zip(t, measured, estimated)])
        results.append({'数据划分': split, '功率_W': power, '测点': name,
                        **late_trend(t, measured, estimated)})
    return results


def run_analysis(run: Path, root: Path = ROOT) -> Path:
    root = root.resolve()
    run = Path(run).resolve()
    config, splits, model, best_epoch, best_path = _completed_run(run, root)
    geometry = Geometry.from_project(root)
    # 明确不调用 load_observations(..., 'test', ...)，不读取测试指标或发布结果图册。
    tables = {part: load_observations(root, part, splits, geometry)
              for part in ('train', 'validation')}
    output = _inside_project(run / '开发集分析', root)
    output.mkdir(exist_ok=True)
    setup_font()
    trend_rows = []
    top_rows = []
    for part, label in (('train', '训练'), ('validation', '验证')):
        for name in ('热端', '冷端'):
            trend_rows.extend(_sensor_rows(model, tables[part][name], name, label, output, root))
        top = tables[part]['顶部']
        top_prediction = predict(model, top.x)
        for record in table_metrics(top, top_prediction)['逐功率']:
            top_rows.append({'数据划分': label, **record})
        top_times = top_time_rows(top, top_prediction, label)
        for gid in np.unique(top.group):
            power = float(top.x[top.group == gid][0, 3])
            rows = [row for row in top_times if row['功率_W'] == power]
            folder = _inside_project(output / label / f'{power:.3f}W', root)
            folder.mkdir(parents=True, exist_ok=True)
            _write_csv(folder / '顶部误差随时间.csv', root,
                       ['数据划分', '功率_W', '时间_s', '时间权重和',
                        '均方根误差_℃', '平均偏差_℃'], rows)
            save_curve(_inside_project(folder / '顶部误差随时间.png', root),
                       [row['时间_s'] for row in rows],
                       [('径向加权RMSE', [row['均方根误差_℃'] for row in rows]),
                        ('径向加权平均偏差', [row['平均偏差_℃'] for row in rows])],
                       f'{label} {power:.3f} W 顶部误差随时间', '时间 / s', '误差 / ℃')
    _write_csv(output / '顶部逐功率RMSE.csv', root,
               ['数据划分', '功率_W', '均方根误差_℃', '平均绝对误差_℃',
                '平均偏差_℃', '最大绝对误差_℃'], top_rows)
    _write_csv(output / '冷热端末20秒趋势.csv', root,
               ['数据划分', '功率_W', '测点', '窗口起点_s', '窗口点数', '末观测时间_s',
                '实测趋势_℃每20秒', '预测趋势_℃每20秒', '趋势误差_℃每20秒', '末时刻误差_℃'], trend_rows)

    lines = [
        '# 联合训练开发集结果分析', '',
        '范围：仅训练集和验证集；测试观测及测试结果未读取，未用于训练或选模；本脚本不更新固定图册。',
        f'分析检查点：验证最佳模型（第{best_epoch}轮）；SHA256：`{sha256(best_path)}`。',
        f'运行计划轮数：{config["training"]["epochs"]}；低保真/高保真划分：'
        f'{len(splits["low"]["training"])}/0/0 与 '
        f'{len(splits["high"]["training"])}/{len(splits["high"]["validation"])}/'
        f'{len(splits["high"]["test"])}（最后一组仅列划分数量，未读测试标签）。',
        '', '## 计算方法', '',
        '末20秒趋势：对每条曲线中时间不早于其最后一个实测时刻减20秒的所有实测点，'
        '分别对实测温度、同时间的预测温度做最小二乘直线拟合；斜率乘20，单位为℃/20秒。'
        '趋势误差＝预测趋势－实测趋势。末时刻误差＝最后实测时刻的预测温度－实测温度。'
        '温度差不依赖摄氏度与开尔文的零点。顶部RMSE按现有分功率加权评价计算。'
        '顶面逐时间RMSE和平均偏差在每个实测时间使用同一frame_weight对径向点加权，'
        'CSV记录每个时间的权重和；按这些权重和汇总可还原原有逐功率RMSE和平均偏差。'
        '汇总表对不同功率等权平均，不用验证数据更新模型。',
        '', '## 结果摘要', '',
        '|数据划分|测点|末段趋势误差绝对值平均 ℃/20秒|末点绝对误差平均 ℃|顶部RMSE逐功率平均 ℃|',
        '|---|---|---:|---:|---:|',
    ]
    for split in ('训练', '验证'):
        mean_top = float(np.mean([row['均方根误差_℃'] for row in top_rows
                                  if row['数据划分'] == split]))
        for name in ('热端', '冷端'):
            rows = [row for row in trend_rows if row['数据划分'] == split and row['测点'] == name]
            mean_trend = float(np.mean([abs(row['趋势误差_℃每20秒']) for row in rows]))
            mean_endpoint = float(np.mean([abs(row['末时刻误差_℃']) for row in rows]))
            lines.append(f'|{split}|{name}|{mean_trend:.3f}|{mean_endpoint:.3f}|{mean_top:.3f}|')
    lines.extend(['', '## 冷热端逐功率结果', '',
        '|数据划分|功率 W|测点|末观测 s|实测趋势 ℃/20秒|预测趋势 ℃/20秒|趋势误差 ℃/20秒|末时刻误差 ℃|',
        '|---|---:|---|---:|---:|---:|---:|---:|',
    ])
    for row in trend_rows:
        lines.append('|{数据划分}|{功率_W:.3f}|{测点}|{末观测时间_s:.0f}|{实测趋势_℃每20秒:+.3f}|'
                     '{预测趋势_℃每20秒:+.3f}|{趋势误差_℃每20秒:+.3f}|{末时刻误差_℃:+.3f}|'.format(**row))
    lines.extend(['', '## 顶部分功率误差', '',
                  '|数据划分|功率 W|RMSE ℃|平均偏差 ℃|', '|---|---:|---:|---:|'])
    for row in top_rows:
        lines.append(f'|{row["数据划分"]}|{row["功率_W"]:.3f}|{row["均方根误差_℃"]:.3f}|'
                     f'{row["平均偏差_℃"]:+.3f}|')
    lines.extend(['', '## 数据边界与查看位置', '',
                  '所有逐时刻数据和图像保存在 `训练/` 与 `验证/` 的各功率目录，'
                  '包括 `顶部误差随时间.csv` 和同名PNG；'
                  '总表为 `冷热端末20秒趋势.csv`、`顶部逐功率RMSE.csv`。',
                  '无实测外推：每个功率的最后观测时刻之后至200秒没有对应的高保真实测温度。'
                  '若单独查看该区间的模型输出，只能作为无实测外推，不能报告实测误差或宣称已验证稳定。',
                  '上述表格和图只到各功率的最后实测时刻，不将任何模型外推画成实验数据。', ''])
    _inside_project(output / '分析报告.md', root).write_text('\n'.join(lines), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description='分析已完成联合训练的训练/验证曲线，不读取测试集')
    parser.add_argument('--run', required=True, type=Path, help='项目内已完成轮数的运行目录')
    args = parser.parse_args()
    print(f'中文分析报告：{run_analysis(args.run) / "分析报告.md"}')


if __name__ == '__main__':
    main()
