#!/usr/bin/env python3
"""仅用80个仿真训练功率诊断已完成联合训练的低保真预测。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from joint_temperature_core import (
    KELVIN, Geometry, fixed_splits, load_model, power_keys, predict,
    read_parquet, read_yaml, sha256,
)


ROOT = Path(__file__).resolve().parents[1]
TIMES = (0, 2, 4, 10, 20, 50, 100, 200)
SEGMENTS = ((0, 0), (2, 2), (4, 4), (6, 10), (12, 20),
            (22, 50), (52, 100), (102, 200))
MATERIALS = ((0, '铜'), (1, 'SiC'))


def _within_records(path: Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    records = (root / '研究记录').resolve()
    if records not in resolved.parents or root not in records.parents:
        raise ValueError('运行目录及分析输出必须位于本项目内的研究记录目录。')
    return resolved


def _completed_run(run: Path, root: Path, device: str):
    run = _within_records(run, root)
    actual = _within_records(run / '实际配置.yaml', root)
    if not actual.is_file():
        raise FileNotFoundError('缺少实际配置，无法核实已完成运行。')
    config = read_yaml(actual)
    if config.get('data', {}).get('low_fidelity_mode') != 'all_training':
        raise ValueError('仅支持80个低保真功率全量训练的运行，不接受旧划分。')
    planned = config.get('training', {}).get('epochs')
    if type(planned) is not int or planned < 1:
        raise ValueError('实际配置中的计划轮数无效。')
    final_path = _within_records(run / f'第{planned}轮模型.pt', root)
    best_path = _within_records(run / '验证最佳模型.pt', root)
    record_path = _within_records(run / '开发集完成记录.json', root)
    if not all(path.is_file() for path in (final_path, best_path, record_path)):
        raise FileNotFoundError('缺少末轮模型、验证最佳模型或开发集完成记录，不能分析未完成运行。')
    completed = json.loads(record_path.read_text(encoding='utf-8'))
    if (completed.get('完成轮数') != planned or completed.get('测试数据已读取') is not False):
        raise ValueError('完成轮数或开发集测试数据状态不符。')
    splits = fixed_splits(root, 'all_training')
    if tuple(splits['low']['training']) != tuple(float(power) for power in range(10, 801, 10)):
        raise ValueError('低保真训练功率未覆盖规定的80个功率。')
    _, final = load_model(final_path, 'cpu')
    model, best = load_model(best_path, device)
    for state in (final, best):
        if (state.get('planned_epochs') != planned or state.get('config') != config
                or state.get('splits') != splits):
            raise ValueError('检查点轮数、训练配置或功率划分与运行来源不一致。')
    if (final['epoch'] != planned or not 1 <= best['epoch'] <= planned
            or completed.get('验证最佳轮次') != best['epoch']):
        raise ValueError('末轮模型未完成或最佳检查点与完成记录不符。')
    digest = sha256(best_path)
    if completed.get('验证最佳检查点SHA256') != digest:
        raise ValueError('完成记录中的验证最佳检查点SHA256不一致。')
    return run, model, splits, planned, best['epoch'], digest


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _stats(truth: np.ndarray, prediction: np.ndarray, initial_k: float):
    if not len(truth):
        raise ValueError('某个仿真时间段没有训练测点，拒绝汇报空结果。')
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return {
        '测点数': len(error),
        '均方根误差_℃': float(np.sqrt(np.mean(error ** 2))),
        '平均偏差_℃': float(np.mean(error)),
        '预测低于仿真初温比例_%': float(np.mean(prediction < initial_k) * 100.),
    }


def _validate_file(x: np.ndarray, truth: np.ndarray, power: float, initial_k: float,
                   expected_times: np.ndarray):
    if (x.ndim != 2 or x.shape[1] != 5 or truth.shape != (len(x),)
            or not len(x) or not np.isfinite(x).all() or not np.isfinite(truth).all()):
        raise ValueError(f'{power:g} W仿真训练数据为空或含无效数值。')
    if set(power_keys(x[:, 3])) != {int(round(power * 10000))}:
        raise ValueError(f'{power:g} W仿真文件功率内容不匹配。')
    if set(np.unique(x[:, 4])) != {0., 1.}:
        raise ValueError(f'{power:g} W仿真必须有铜和SiC两个材料。')
    for material, name in MATERIALS:
        material_times, counts = np.unique(x[x[:, 4] == material, 2], return_counts=True)
        if not np.array_equal(material_times, expected_times):
            raise ValueError(f'{power:g} W的{name}实际仿真时间网格不完整：须覆盖0、2、4至200秒。')
        if not np.all(counts == counts[0]):
            raise ValueError(f'{power:g} W的{name}各仿真时间点网格节点数不一致。')
    initial = truth[x[:, 2] == 0]
    if not np.allclose(initial, initial_k, atol=1e-3, rtol=0.):
        raise ValueError(f'{power:g} W仿真零秒标签与配置的仿真初温不符。')


def _top_center_two_seconds(x: np.ndarray, truth: np.ndarray, prediction: np.ndarray,
                            power: float):
    near_center = ((x[:, 4] == 1.) & (np.abs(x[:, 1]) <= 1e-7)
                   & (x[:, 0] >= 0.) & (x[:, 0] <= .001 + 1e-7))
    first = np.flatnonzero(near_center & (x[:, 2] == 0.))
    second = np.flatnonzero(near_center & (x[:, 2] == 2.))
    first = first[np.argsort(x[first, 0])]
    second = second[np.argsort(x[second, 0])]
    if (not len(first) or len(first) != len(second)
            or not np.allclose(x[first, :2], x[second, :2], rtol=0., atol=1e-7)):
        raise ValueError(f'{power:g} W的SiC顶面近中心0秒与2秒没有相同的实际测点。')
    sim0 = float(np.mean(truth[first].astype(np.float64)))
    sim2 = float(np.mean(truth[second].astype(np.float64)))
    pred0 = float(np.mean(prediction[first].astype(np.float64)))
    pred2 = float(np.mean(prediction[second].astype(np.float64)))
    return {
        '功率_W': power, '中心测点数': len(first),
        '仿真2秒温度_℃': sim2 - KELVIN, '预测2秒温度_℃': pred2 - KELVIN,
        '预测减仿真2秒温度_℃': pred2 - sim2,
        '仿真0到2秒温升_℃': sim2 - sim0,
        '预测0到2秒温升_℃': pred2 - pred0,
        '预测减仿真温升_℃': (pred2 - pred0) - (sim2 - sim0),
    }


def _global_rows(buckets: dict, labels: tuple, column: str):
    rows = []
    for label in labels:
        count, squared, signed, below = buckets[label]
        if not count:
            raise ValueError(f'{label}没有可用的仿真训练测点。')
        rows.append({column: label, '测点数': count,
                     '均方根误差_℃': float(np.sqrt(squared / count)),
                     '平均偏差_℃': signed / count,
                     '预测低于仿真初温比例_%': below / count * 100.})
    return rows


def run_analysis(run: Path, root: Path = ROOT, device: str = 'cpu',
                 batch_size: int = 8192) -> Path:
    """逐文件逐批推理；只对低保真训练文件计数，不加载实验观测。"""
    root = Path(root).resolve()
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('推理批大小必须为正整数。')
    run, model, splits, planned, best_epoch, best_hash = _completed_run(run, root, device)
    output = _within_records(run / '低保真训练集分析', root)
    if output.exists():
        raise FileExistsError('低保真训练集分析已存在，不覆盖已保存的结果。')
    for folder in (root / 'data', root / 'data/processed',
                   root / 'data/processed/simulation'):
        if folder.is_symlink() or root not in folder.resolve().parents:
            raise ValueError('仿真训练文件及上层目录不能是符号链接，且必须位于项目内。')
    geometry = Geometry.from_project(root)
    if geometry.time_max != 200.:
        raise ValueError('仿真时间上限不是本项目已核对的200秒。')
    expected_times = np.arange(0., 201., 2., dtype=np.float32)
    initial_k = float(model.temperature_offset.detach().cpu())
    manifest = root / 'data/processed/manifest.json'
    if manifest.is_symlink():
        raise ValueError('数据处理清单不能是符号链接。')
    manifest_hash = sha256(manifest) if manifest.is_file() else '处理清单不存在'
    intervals, samples, material_totals, power_totals, centers = [], [], [], [], []
    segment_labels = tuple(str(begin) if begin == end else f'{begin}-{end}'
                           for begin, end in SEGMENTS)
    global_segments = {label: [0, 0., 0., 0] for label in segment_labels}
    global_times = {time_s: [0, 0., 0., 0] for time_s in TIMES}
    accum = {label: [0, 0., 0] for _, label in MATERIALS}
    overall = [0, 0., 0]
    simulation_dir = (root / 'data/processed/simulation').resolve()
    if root not in simulation_dir.parents:
        raise ValueError('仿真训练文件目录必须位于本项目内。')
    for power in splits['low']['training']:
        path = root / 'data/processed/simulation' / f'{power:g}W.parquet'
        if path.is_symlink() or path.resolve().parent != simulation_dir:
            raise ValueError(f'{power:g} W仿真文件路径包含符号链接或不在训练仿真目录。')
        frame = read_parquet(path)
        x = frame.select('r_m', 'z_m', 'time_s', 'power_w', 'material_id').to_numpy().astype(np.float32)
        truth = frame['temperature_k'].to_numpy().astype(np.float32)
        _validate_file(x, truth, power, initial_k, expected_times)
        prediction = predict(model, x, batch_size=batch_size, fidelity='low')
        if not np.isfinite(prediction).all():
            raise ValueError(f'{power:g} W低保真预测含无效温度。')
        centers.append(_top_center_two_seconds(x, truth, prediction, power))
        power_totals.append({'功率_W': power, **_stats(truth, prediction, initial_k)})
        for material, label in MATERIALS:
            selected = x[:, 4] == material
            truth_part, pred_part = truth[selected], prediction[selected]
            material_totals.append({'功率_W': power, '材料': label,
                                    **_stats(truth_part, pred_part, initial_k)})
            errors = pred_part.astype(np.float64) - truth_part.astype(np.float64)
            accum[label][0] += len(errors)
            accum[label][1] += float(np.dot(errors, errors))
            accum[label][2] += int(np.count_nonzero(pred_part < initial_k))
            times = x[selected, 2]
            for begin, end in SEGMENTS:
                mask = (times >= begin) & (times <= end)
                segment = str(begin) if begin == end else f'{begin}-{end}'
                intervals.append({'功率_W': power, '材料': label, '时间段_s': segment,
                                  **_stats(truth_part[mask], pred_part[mask], initial_k)})
                selected_errors = errors[mask]
                bucket = global_segments[segment]
                bucket[0] += len(selected_errors)
                bucket[1] += float(np.dot(selected_errors, selected_errors))
                bucket[2] += float(selected_errors.sum())
                bucket[3] += int(np.count_nonzero(pred_part[mask] < initial_k))
            for time_s in TIMES:
                mask = times == time_s
                samples.append({'功率_W': power, '材料': label, '时间_s': time_s,
                                **_stats(truth_part[mask], pred_part[mask], initial_k)})
                selected_errors = errors[mask]
                bucket = global_times[time_s]
                bucket[0] += len(selected_errors)
                bucket[1] += float(np.dot(selected_errors, selected_errors))
                bucket[2] += float(selected_errors.sum())
                bucket[3] += int(np.count_nonzero(pred_part[mask] < initial_k))
        errors = prediction.astype(np.float64) - truth.astype(np.float64)
        overall[0] += len(errors)
        overall[1] += float(np.dot(errors, errors))
        overall[2] += int(np.count_nonzero(prediction < initial_k))

    output.mkdir()
    columns = ('测点数', '均方根误差_℃', '平均偏差_℃', '预测低于仿真初温比例_%')
    all_segments = _global_rows(global_segments, segment_labels, '时间段_s')
    all_times = _global_rows(global_times, TIMES, '时间_s')
    _write_csv(output / '全场时间段误差.csv', ('时间段_s', *columns), all_segments)
    _write_csv(output / '全场指定时刻误差.csv', ('时间_s', *columns), all_times)
    _write_csv(output / '逐功率逐材料时间段误差.csv', ('功率_W', '材料', '时间段_s', *columns), intervals)
    _write_csv(output / '逐功率逐材料指定时刻误差.csv', ('功率_W', '材料', '时间_s', *columns), samples)
    _write_csv(output / '逐功率逐材料全时段误差.csv', ('功率_W', '材料', *columns), material_totals)
    _write_csv(output / '逐功率全场误差.csv', ('功率_W', *columns), power_totals)
    _write_csv(output / 'SiC顶面近中心2秒.csv', (
        '功率_W', '中心测点数', '仿真2秒温度_℃', '预测2秒温度_℃',
        '预测减仿真2秒温度_℃', '仿真0到2秒温升_℃', '预测0到2秒温升_℃',
        '预测减仿真温升_℃',
    ), centers)

    worst = max(power_totals, key=lambda row: row['均方根误差_℃'])
    worst_center = max(centers, key=lambda row: abs(row['预测减仿真温升_℃']))
    lines = [
        '# 低保真仿真训练集结果分析', '',
        '范围：仅使用80个仿真训练功率的仿真温度标签，未读取高保真实验观测；测试标签未读取。'
        '本报告不评价实验冷热端误差，也不更新测试图册。',
        f'训练已完成{planned}轮；使用验证最佳模型第{best_epoch}轮，SHA256：`{best_hash}`。',
        f'分析脚本SHA256：`{sha256(Path(__file__))}`；'
        f'`data/processed/manifest.json` SHA256：`{manifest_hash}`。',
        '所有80个功率、两种材料均核对了实际离散时间：零秒至200秒，每2秒一个点，共101个时刻。'
        '仅在文件中实际存在的时间点计算误差，不插值、不外推。',
        'SiC顶面近中心取z=0且半径0至1毫米的对应网格点，分别平均其0秒和2秒温度，'
        f'再计算0至2秒温升；本模型的仿真初温取实际配置的{initial_k - KELVIN:.3f}℃，'
        '低于初温比例按逐点严格小于该温度统计。',
        '各RMSE先逐点平方平均再开方；总计按仿真网格点数加权，不把不同功率的RMSE直接平均。',
        '', '## 全场汇总', '',
        '|范围|仿真测点数|RMSE ℃|预测低于仿真初温比例 %|',
        '|---|---:|---:|---:|',
        f'|全部功率与材料|{overall[0]}|{np.sqrt(overall[1] / overall[0]):.4f}|'
        f'{overall[2] / overall[0] * 100:.3f}|',
    ]
    for _, label in MATERIALS:
        n, sum_square, below = accum[label]
        lines.append(f'|{label}|{n}|{np.sqrt(sum_square / n):.4f}|{below / n * 100:.3f}|')
    lines.extend([
        '', f'全场RMSE最差功率：{worst["功率_W"]:g} W，'
        f'{worst["均方根误差_℃"]:.4f} ℃，测点数{worst["测点数"]}。',
        f'近中心0至2秒温升绝对误差最大功率：{worst_center["功率_W"]:g} W，'
        f'{worst_center["预测减仿真温升_℃"]:+.4f} ℃（预测减仿真）。',
        '', '## 全场分时间段误差', '',
        '|时间段 s|仿真测点数|RMSE ℃|预测低于仿真初温比例 %|',
        '|---|---:|---:|---:|',
    ])
    for row in all_segments:
        lines.append(f'|{row["时间段_s"]}|{row["测点数"]}|{row["均方根误差_℃"]:.4f}|'
                     f'{row["预测低于仿真初温比例_%"]:.3f}|')
    lines.extend([
        '', '## 全场指定时刻误差', '',
        '|时刻 s|仿真测点数|RMSE ℃|预测低于仿真初温比例 %|',
        '|---:|---:|---:|---:|',
    ])
    for row in all_times:
        lines.append(f'|{row["时间_s"]}|{row["测点数"]}|{row["均方根误差_℃"]:.4f}|'
                     f'{row["预测低于仿真初温比例_%"]:.3f}|')
    lines.extend([
        '', '## 时间段定义', '',
        '|时间段 s|时间点数|', '|---|---:|',
    ])
    for begin, end in SEGMENTS:
        lines.append(f'|{begin if begin == end else f"{begin}-{end}"}|{(end - begin) // 2 + 1}|')
    lines.extend([
        '', '文件：`全场时间段误差.csv`与`全场指定时刻误差.csv`保存跨80个功率的测点加权汇总；'
        '`逐功率逐材料时间段误差.csv`保存互不重叠的全场时段；'
        '`逐功率逐材料指定时刻误差.csv`独立列出0、2、4、10、20、50、100、200秒的精确测点；'
        '`逐功率逐材料全时段误差.csv`、`逐功率全场误差.csv`及'
        '`SiC顶面近中心2秒.csv`保存完整逐功率核查数据。',
        '本结果只说明仿真数据的拟合，不能用200秒仿真标签证明实验传感器200秒已经稳定。', '',
    ])
    (output / '分析报告.md').write_text('\n'.join(lines), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description='已完成联合训练的80功率低保真训练集中文分析')
    parser.add_argument('--run', required=True, type=Path, help='项目内研究记录中的完整训练目录')
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'), help='推理设备，默认CPU')
    parser.add_argument('--batch-size', type=int, default=8192, help='单次推理点数，默认8192')
    args = parser.parse_args()
    output = run_analysis(args.run, device=args.device, batch_size=args.batch_size)
    print(f'低保真训练集中文报告：{output / "分析报告.md"}')


if __name__ == '__main__':
    main()
