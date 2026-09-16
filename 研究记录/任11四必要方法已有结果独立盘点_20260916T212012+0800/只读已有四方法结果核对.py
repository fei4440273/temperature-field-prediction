#!/usr/bin/env python3
"""仅重算现存指标与成本，不加载模型、原温度或训练器。"""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import stat
import statistics
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import yaml

ROOT = Path('/home/phl/lyf/Temperature Field Prediction')
BASE = ROOT / '研究记录/任务11_外部对照'
AGG = BASE / '正式新MLP五种子HF与双原能源汇总_20260916T165422+0800'
OBS = BASE / '正式新MLP三模态合法观察_20260916T183039+0800'
FEM = BASE / 'LF_FEM合法HF顶部真实验证_20260916T032116+0800'
RIDGE = BASE / 'HFonly固定Ridge合法HF三模态验证_20260916T033505+0800'
HELPER = Path('/home/phl/.codex/skills/data-analysis/scripts/stat_summary.py')
SEEDS = list(range(5))
POWERS = [115.2, 403.0, 630.5]
STATES = {'best': '观测最佳', 'final': '训练末'}


def stamp():
    return datetime.now().astimezone().isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pointer(parts):
    return '/' + '/'.join(str(x).replace('~', '~0').replace('/', '~1') for x in parts)


def unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('重复JSON键: ' + key)
        out[key] = value
    return out


class StrictYaml(yaml.SafeLoader):
    pass


def yaml_mapping(loader, node, deep=False):
    return unique_pairs([(loader.construct_object(k, deep=deep),
                          loader.construct_object(v, deep=deep)) for k, v in node.value])


StrictYaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, yaml_mapping)


class Audit:
    def __init__(self):
        self.sources = {}
        self.checks = []
        self.stats = []
        spec = importlib.util.spec_from_file_location('已有数字描述统计助手', HELPER)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        self.describe = helper.describe_column

    def read(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        if path.resolve() != path or ROOT not in path.parents:
            raise ValueError('拒绝项目外或链接原件: ' + str(path))
        if path.suffix not in {'.json', '.csv', '.md', '.yaml'}:
            raise ValueError('拒绝模型、温度、逐点JSONL或其他非指标格式: ' + str(path))
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('原件不是普通文件: ' + str(path))
        data = path.read_bytes()
        ident = {'路径': str(path), '字节': len(data), 'SHA256': digest(data),
                 '权限': oct(stat.S_IMODE(info.st_mode)), '修改时间_ns': info.st_mtime_ns}
        if str(path) in self.sources:
            self.check('复读原件身份一致', self.sources[str(path)] == ident, str(path))
        else:
            self.sources[str(path)] = ident
        return data

    def js(self, path):
        return json.loads(self.read(path).decode('utf-8-sig'), object_pairs_hook=unique_pairs)

    def ys(self, path):
        return yaml.load(self.read(path).decode('utf-8-sig'), Loader=StrictYaml)

    def rows(self, path):
        reader = csv.DictReader(io.StringIO(self.read(path).decode('utf-8-sig')))
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('重复CSV列: ' + str(path))
        return list(reader)

    def check(self, name, ok, source=None, detail=None):
        self.checks.append({'检查': name, '通过': bool(ok), '来源': source, '细节': detail})

    def close(self, name, actual, expected, source=None):
        self.check(name, math.isfinite(float(actual)) and math.isfinite(float(expected))
                   and math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-11),
                   source, {'重算': actual, '原值': expected, '差': float(actual) - float(expected)})

    def summary(self, values, path, parts, published=None, unit='摄氏度', published_source=None):
        self.check('原五值有限且非bool', len(values) == 5 and all(type(x) in (int, float)
                   and math.isfinite(x) for x in values), str(path) + '#' + pointer(parts), values)
        desc = self.describe(values)
        self.close('NumPy与标准库均值一致', desc['mean'], statistics.mean(values))
        self.close('NumPy与标准库样本标准差一致', desc['std'], statistics.stdev(values))
        result = {'原件': str(path), 'JSONpointer或CSV派生': pointer(parts), '单位': unit,
                  '种子顺序': SEEDS, '样本数': 5, 'ddof': 1, '原五值': values,
                  '均值': desc['mean'], '样本标准差': desc['std'],
                  '最小值': desc['min'], '最大值': desc['max']}
        if published is not None:
            pub_path, pub_parts = published_source or (path, parts)
            result['原五值'] = published['逐种子']
            result['重算五值'] = values
            result['原聚合精确字段'] = published
            result['原聚合原件及JSONpointer'] = str(pub_path) + '#' + pointer(pub_parts)
            for key, target in [('均值', 'mean'), ('样本标准差', 'std'),
                                ('最小值', 'min'), ('最大值', 'max')]:
                self.close('原聚合' + key + '一致', desc[target], published[key],
                           str(pub_path) + '#' + pointer(pub_parts + [key]))
            if '样本数' in published:
                self.check('原样本数五且ddof1', published['样本数'] == 5 and
                           type(published['样本数']) is int and published['ddof'] == 1)
            for seed, (actual, expected) in enumerate(zip(values, published['逐种子'])):
                self.close('本人原五值与重算一致_允许浮点舍入', actual, expected,
                           str(pub_path) + '#' + pointer(pub_parts + ['逐种子', seed]))
        self.stats.append(result)
        return result

    def leaves(self, obj, path, parts=None, unit='摄氏度'):
        parts = [] if parts is None else parts
        if isinstance(obj, dict) and '逐种子' in obj and '样本标准差' in obj:
            self.summary(obj['逐种子'], path, parts, obj, unit)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                self.leaves(value, path, parts + [key], unit)

    def reported_hash(self, path, expected):
        self.read(path)
        self.check('原记录SHA与实际普通原件SHA一致',
                   self.sources[str(path)]['SHA256'] == expected, str(path), expected)


def check_all(output):
    began = stamp()
    audit = Audit()
    reports = [BASE / 'LF_FEM仅顶部合法HF真实对照与方法排名限界.md',
               BASE / '仅HF训练固定Ridge真实合法三模态观测与物理限界.md',
               BASE / '任11_方法公平来源CPU预检_20260916T062100+0800/任11方法公平预算与来源限界报告.md',
               BASE / '任11_新MLP五种子十原能源正式汇总根复审中文验收_20260916T172037+0800.md',
               BASE / '任11_新MLP真实三模态观察中文误差与验收_20260916T190502+0800.md',
               BASE / 'Howard等复合DeepONet原文与代码来源入场限界.md',
               BASE / '任11_Howard五种子独立LF真实完训中文验收_20260916T181344+0800.md',
               BASE / '任11_Howard_HF_seed2第304轮数值失败中文诊断_20260916T205117+0800.md']
    for path in reports:
        audit.read(path)
    status_path = ROOT / '研究记录/总任务状态.json'
    status_bytes = status_path.read_bytes()
    task = json.loads(status_bytes, object_pairs_hook=unique_pairs)['任务']['任-11']
    aggregate_path = AGG / '机器汇总.json'
    aggregate = audit.js(aggregate_path)
    audit.read(AGG / '中文验收.md')
    audit.reported_hash(aggregate_path, task['新MLP正式机器汇总SHA256'])
    obs_path = OBS / '机器汇总.json'
    obs = audit.js(obs_path)
    obs_raw_path = OBS / '原始指标与中文映射.json'
    obs_raw = audit.js(obs_raw_path)['十状态原指标']
    obs_rows_path = OBS / '逐功率三模态.csv'
    obs_rows = audit.rows(obs_rows_path)
    obs_time = audit.rows(OBS / '逐功率三模态时间窗.csv')
    obs_radial = audit.rows(OBS / '顶部逐功率原生径向窗.csv')
    audit.js(OBS / '工件SHA256.json')
    audit.read(OBS / '中文验收.md')
    for name, ident in task['新MLP观察八输出实际字节与SHA256'].items():
        if name.endswith('.jsonl'):
            continue
        path = OBS / name
        audit.reported_hash(path, ident['SHA256'])
        audit.check('观察输出字节与原记录一致', audit.sources[str(path)]['字节'] == ident['字节'])
    audit.check('三模态CSV90唯一身份', len(obs_rows) == 90 and len({
        (r['种子'], r['模型状态原标识'], r['功率_瓦'], r['模态原标识']) for r in obs_rows}) == 90)
    audit.check('合法三功率九行每身份', all(
        sorted({float(r['功率_瓦']) for r in obs_rows if int(r['种子']) == seed and
                r['模型状态原标识'] == state}) == POWERS for seed in SEEDS for state in STATES))
    audit.check('90行仅VAL且TopHotCold', all(r['划分原标识'] == 'validation' and
        r['模态原标识'] in {'Top', 'Hot', 'Cold'} for r in obs_rows))
    audit.check('80240现存观察点而非本次新实测', sum(int(r['实测点数']) for r in obs_rows)
                == obs['总点数'] == 80240)
    audit.check('原时间径向窗计数一致', len(obs_time) == obs['验收计数']['时间窗'] == 270 and
                len(obs_radial) == obs['验收计数']['径向窗'] == 90)
    mlp = {'冻结口径': {}, '观察Float64口径': {}, '五本人训练身份成本': [], '十态能源': [],
           '观察原始十状态macro原值与指针': [], 'CSV重算十状态macro原值': []}
    for state, zh in STATES.items():
        published = aggregate[zh]
        audit.leaves(published, aggregate_path, [zh])
        frozen = {}
        fields = {'S': ('HF合法macro_v1_摄氏度',), 'Top': ('HF合法分模态', '顶部RMSE_摄氏度'),
                  'ABS': ('HF合法分模态', '两环合并绝对RMSE_摄氏度'),
                  'Delta': ('HF合法分模态', '两环合并首时刻差分RMSE_摄氏度')}
        for name, keys in fields.items():
            leaf = published
            for key in keys:
                leaf = leaf[key]
            values = []
            for row in aggregate['逐种子原统计']:
                value = row[zh]
                for key in keys:
                    value = value[key]
                values.append(value)
            frozen[name] = audit.summary(values, aggregate_path, [zh] + list(keys), leaf)
        mlp['冻结口径'][state] = frozen
        obs_state = obs['模型双状态五种子统计'][state]
        audit.leaves(obs_state['分模态宏平均'], obs_path,
                     ['模型双状态五种子统计', state, '分模态宏平均'])
        computed = {name: [] for name in ['S', 'Top', 'ABS', 'Delta']}
        for seed in SEEDS:
            per_modality = {}
            for modality in ['Top', 'Hot', 'Cold']:
                rows = [r for r in obs_rows if int(r['种子']) == seed and
                        r['模型状态原标识'] == state and r['模态原标识'] == modality]
                audit.check('每seed状态模态三功率等权', len(rows) == 3)
                per_modality[modality] = {}
                for metric, leaf in obs_state['分模态宏平均'][modality].items():
                    value = float(np.mean([float(r[metric]) for r in rows]))
                    audit.close('CSV按三功率宏平均与JSON原五值一致', value, leaf['逐种子'][seed],
                                str(obs_rows_path) + '#seed=' + str(seed) + '/' + state + '/' + modality + '/' + metric)
                    per_modality[modality][metric] = value
            top = per_modality['Top']['绝对RMSE_摄氏度']
            absolute = (per_modality['Hot']['绝对RMSE_摄氏度'] +
                        per_modality['Cold']['绝对RMSE_摄氏度']) / 2
            delta = (per_modality['Hot']['首实测差分RMSE_摄氏度'] +
                     per_modality['Cold']['首实测差分RMSE_摄氏度']) / 2
            score = (top + .2 * absolute + delta) / 2.2
            audit.close('观察三模态组合S一致', score,
                        obs_state['本人原macro_v1_摄氏度']['逐种子'][seed])
            raw = next(r for r in obs_raw if r['种子'] == seed and r['模型状态'] == state)['原macro核验']
            raw_index = next(i for i, r in enumerate(obs_raw) if r['种子'] == seed and r['模型状态'] == state)
            mlp['观察原始十状态macro原值与指针'].append({'种子': seed, '状态': state,
                '原件': str(obs_raw_path), 'JSONpointer': pointer(['十状态原指标', raw_index, '原macro核验']),
                '本人原macro完整精确字段': raw})
            mlp['CSV重算十状态macro原值'].append({'种子': seed, '状态': state,
                'Top': top, 'ABS': absolute, 'Delta': delta, 'S': score})
            for key, value in [('顶部', top), ('absolute_rmse_c', absolute),
                               ('delta_rmse_c', delta), ('合法HF原macro_v1_摄氏度', score)]:
                audit.close('观察原指标与CSV重算一致', value, raw[key])
            for name, value in [('S', score), ('Top', top), ('ABS', absolute), ('Delta', delta)]:
                original_key = {'S': '合法HF原macro_v1_摄氏度', 'Top': '顶部',
                                'ABS': 'absolute_rmse_c', 'Delta': 'delta_rmse_c'}[name]
                computed[name].append(raw[original_key])
                audit.check('Float64观察与冻结字段差在原1e-4合同内',
                            abs(value - frozen[name]['原五值'][seed]) <= 1e-4,
                            str(obs_rows_path), {'种子': seed, '状态': state, '项': name,
                                               '差': value - frozen[name]['原五值'][seed]})
        mlp['观察Float64口径'][state] = {name: audit.summary(values, obs_raw_path,
            ['十状态原指标', '按seed及状态取本人原macro核验', state, name],
            obs_state['本人原macro_v1_摄氏度'] if name == 'S' else None,
            published_source=(obs_path, ['模型双状态五种子统计', state, '本人原macro_v1_摄氏度'])
            if name == 'S' else None)
            for name, values in computed.items()}
    audit.leaves(aggregate['训练与成本统计'], aggregate_path, ['训练与成本统计'], '字段原单位')
    for identity in aggregate['完整来源资格原件']:
        seed = identity['seed']
        hf_dir = Path(identity['目录'])
        lf_dir = Path(identity['本seed新LF最佳检查点原件']).parent
        hf_metric = audit.js(hf_dir / 'metrics.json')
        lf_metric = audit.js(lf_dir / 'metrics.json')
        audit.check('本人LF/HF五种子身份对应', hf_metric['seed'] == lf_metric['seed'] == seed)
        audit.check('本人LF/HF实际终态完成', hf_metric['status'] == 'completed_current_protocol_hf' and
                    lf_metric['status'] == 'completed_current_protocol_lf_only')
        cost_row = {'种子': seed, 'HF目录': str(hf_dir), 'LF目录': str(lf_dir)}
        for label, directory, metric in [('HF', hf_dir, hf_metric), ('LF', lf_dir, lf_metric)]:
            receipts = sorted(directory.glob(label + '会话收据_*.json'))
            durations, peaks, epochs = [], [], 0
            prior_sha = None
            for index, path in enumerate(receipts, 1):
                receipt = audit.js(path)
                if label == 'HF':
                    audit.check('HF收据身份及链条一致', receipt['身份']['seed'] == seed and
                                receipt['前驱收据SHA256'] == prior_sha)
                    audit.reported_hash(path, identity['完整原件SHA256'][path.name])
                else:
                    audit.check('LF收据本人种子一致', receipt['seed'] == seed)
                audit.check('本人会话不重复连续累计轮次',
                            receipt['起始已提交轮次'] == epochs and receipt['本会话实际轮次'] > 0 and
                            receipt['累计实际轮次'] == epochs + receipt['本会话实际轮次'])
                epochs = receipt['累计实际轮次']
                durations.append(receipt['本会话真实墙钟秒'])
                peaks.append(receipt['峰值真实CUDA显存字节'])
                prior_sha = audit.sources[str(path)]['SHA256']
            elapsed = float(np.sum(durations))
            peak = max(peaks)
            audit.close(label + '完整会话之和与metrics本人循环成本一致', elapsed, metric['training_seconds'])
            audit.close(label + '完整会话之和与聚合身份成本一致', elapsed, identity['本人' + label + '累计真实成本秒'])
            expected_epochs = (metric['correction_epochs_completed'] + metric['joint_epochs_completed']
                               if label == 'HF' else metric['epochs_completed'])
            audit.check(label + '累计会话轮次与最终metrics一致', epochs == expected_epochs)
            if label == 'HF':
                audit.close('HF当前metrics与冻结SHA一致数值', metric['best_validation_selection_score_c'],
                            mlp['冻结口径']['best']['S']['原五值'][seed])
                audit.reported_hash(hf_dir / 'metrics.json', identity['完整原件SHA256']['metrics.json'])
                audit.check('HF峰值取会话max而非sum', peak == identity['全部会话CUDA峰值显存字节'])
            else:
                audit.check('LF峰值取会话max与metrics一致', peak == metric['peak_gpu_memory_bytes'])
            cost_row.update({label + '本人循环秒': elapsed, label + '实际轮次': epochs,
                             label + '会话数': len(receipts), label + '峰值显存字节': peak,
                             label + '会话循环秒原值': durations})
        cost_row['LF加HF本人循环秒一次计'] = cost_row['LF本人循环秒'] + cost_row['HF本人循环秒']
        cost_row['全链条预处理FEM训练导出成本秒'] = None
        mlp['五本人训练身份成本'].append(cost_row)
    for field, costs_key in [('HF训练成本_秒', 'HF本人循环秒'), ('LF训练成本_秒', 'LF本人循环秒')]:
        values = [r[costs_key] for r in mlp['五本人训练身份成本']]
        audit.summary(values, aggregate_path, ['训练与成本统计', field], aggregate['训练与成本统计'][field], '秒')
    mlp['五身份LF加HF循环成本'] = audit.summary(
        [r['LF加HF本人循环秒一次计'] for r in mlp['五本人训练身份成本']], aggregate_path,
        ['五种子累计总量', 'LF与HF训练成本合计_秒'], unit='秒')
    mlp['五身份LF加HF循环秒合计'] = sum(r['LF加HF本人循环秒一次计'] for r in mlp['五本人训练身份成本'])
    mlp['全部十训练身份显存峰值字节取max'] = max(
        r[label + '峰值显存字节'] for r in mlp['五本人训练身份成本'] for label in ['LF', 'HF'])
    audit.close('五身份成本与正式总量一致', mlp['五身份LF加HF循环秒合计'],
                aggregate['五种子累计总量']['LF与HF训练成本合计_秒'])
    for energy in aggregate['十份独立能源原件']:
        path = Path(energy['能源原目录']) / '汇总指标.json'
        result = audit.js(path)
        audit.reported_hash(path, energy['六工件SHA256']['汇总指标.json'])
        audit.check('能源原summary30点16与64阶', result['功率时刻审核行数'] == 30 and
                    result['求积阶数'] == [16, 64])
        flags = result['名义吸收归一筛查']
        audit.check('名义两阈值原失败未包装合格', flags['均值通过'] is False and
                    flags['95分位通过'] is False and result['工程安全阈值已建立'] is False)
        mlp['十态能源'].append({'种子': energy['运行种子'], '状态': energy['状态原标识'],
            '原件': str(path), 'JSONpointer': '/名义吸收归一筛查',
            '名义均值通过': flags['均值通过'], '名义p95通过': flags['95分位通过'],
            '绝对平衡宏均值_瓦': result['绝对平衡宏均值_瓦'],
            '绝对平衡95分位_瓦': result['绝对平衡95分位_瓦'],
            '原六工件SHA记录_仅summary已本次读取': energy['六工件SHA256']})
    audit.check('十能源身份不漏不重', len(mlp['十态能源']) == 10 and
                {(r['种子'], r['状态']) for r in mlp['十态能源']} == {(s, v) for s in SEEDS for v in STATES})
    for state, zh in STATES.items():
        values = [next(r['绝对平衡宏均值_瓦'] for r in mlp['十态能源']
                       if r['种子'] == seed and r['状态'] == state) for seed in SEEDS]
        audit.summary(values, aggregate_path, [zh, '独立原能源', '绝对平衡宏均值_瓦'],
                      aggregate[zh]['独立原能源']['绝对平衡宏均值_瓦'], '瓦')
    fem_path = FEM / '受限对照与缺项摘要.json'
    fem_json = audit.js(fem_path)
    fem_rows_path = FEM / '合法HF验证_顶部逐功率LF_FEM插值.csv'
    fem_rows = audit.rows(fem_rows_path)
    audit.check('FEM仅三个合法Top功率', len(fem_rows) == 3 and
                sorted(float(r['power_w']) for r in fem_rows) == POWERS and
                all(r['modality'] == 'Top' and r['split'] == 'validation' for r in fem_rows))
    for row in fem_rows:
        audit.close('FEM逐功率RMSE JSON/CSV一致', float(row['rmse_c']),
                    fem_json['逐功率顶部RMSE_摄氏度'][format(float(row['power_w']), 'g')])
    fem_out = {'训练LF支点数': fem_json['训练LF支点数'], '顶部逐功率RMSE_摄氏度':
               fem_json['逐功率顶部RMSE_摄氏度'], '顶部三功率等权宏RMSE_摄氏度':
               float(np.mean([float(r['rmse_c']) for r in fem_rows])), '已观察Top点':
               sum(int(r['sample_count']) for r in fem_rows), '完整S': None,
               'Hot': None, 'Cold': None, 'ABS': None, 'Delta': None,
               '原FEM生成及完整全链条成本秒': None, '种子标准差': None,
               '标准差语义': 'NA：确定性插值，三功率不是训练种子'}
    audit.check('FEM缺项保持null', fem_json['训练LF支点数'] == 60 and
                fem_json['HotCold两环LF_FEM验证'] is None and fem_json['五种子学习效果'] is None)
    ridge_path = RIDGE / '独立有限工程对照与缺项.json'
    ridge_json = audit.js(ridge_path)
    ridge_rows_path = RIDGE / '逐功率三模态合法HF验证.csv'
    ridge_rows = audit.rows(ridge_rows_path)
    prereg_path = BASE / 'HFonly固定Ridge事前输入登记_20260916T033255+0800/仅HF观测Ridge事前来源与固定系数.json'
    coefficients_path = BASE / 'HFonly固定Ridge真HF训练模型锁_20260916T033400+0800/仅HF训练_已锁Ridge系数.json'
    prereg, coefficients = audit.js(prereg_path), audit.js(coefficients_path)
    ridge_budget = audit.js(coefficients_path.parent / '真HF训练预算与来源.json')
    audit.check('Ridge固定degree2 alpha0.01与20features',
                prereg['degree'] == coefficients['degree'] == 2 and
                prereg['alpha'] == coefficients['alpha'] == .01 and coefficients['特征维数'] == 20)
    audit.reported_hash(prereg_path, coefficients['事前来源SHA256'])
    audit.reported_hash(coefficients_path, ridge_json['冻结模型SHA256'])
    audit.check('Ridge仅HF TRAIN12无LF无PDE训练', ridge_budget['顶部HF训练功率数'] == 12 and
                ridge_budget['顶部HF训练行'] == 29593 and ridge_budget['两环HF训练行'] == 2985 and
                ridge_budget['LF训练仿真标签消费'] == 0 and ridge_budget['物理PDE/边界/界面训练梯度'] == 0)
    ridge_macro = {}
    for modality in ['Top', 'Hot', 'Cold']:
        rows = [r for r in ridge_rows if r['modality'] == modality]
        audit.check('Ridge每模态三合法功率', len(rows) == 3 and
                    sorted(float(r['power_w']) for r in rows) == POWERS and
                    all(r['split'] == 'validation' for r in rows))
        ridge_macro[modality] = {key: float(np.mean([float(r[key]) for r in rows]))
                                for key in ['rmse_c', 'delta_rmse_c']}
        audit.close('Ridge逐模态RMSE等功率宏平均一致', ridge_macro[modality]['rmse_c'],
                    ridge_json['逐模态合法HF验证RMSE均值_摄氏度'][modality])
    ridge_absolute = (ridge_macro['Hot']['rmse_c'] + ridge_macro['Cold']['rmse_c']) / 2
    ridge_delta = (ridge_macro['Hot']['delta_rmse_c'] + ridge_macro['Cold']['delta_rmse_c']) / 2
    ridge_score = (ridge_macro['Top']['rmse_c'] + .2 * ridge_absolute + ridge_delta) / 2.2
    audit.close('Ridge macro_v1组合S与原JSON一致', ridge_score, ridge_json['合法HF三模态宏选分_摄氏度'])
    ridge_out = {'逐模态三功率等权宏平均_摄氏度': ridge_macro, 'ABS_摄氏度': ridge_absolute,
                 'Delta_摄氏度': ridge_delta, 'S_摄氏度': ridge_score, '原逐功率九行': ridge_rows,
                 '训练特征': 20, 'degree': 2, 'alpha': .01, '训练循环实际成本秒': None,
                 '种子标准差': None, '标准差语义': 'NA：固定解析回归，不是五种子训练',
                 'S公式': '(Top + 0.2*两环ABS + 两环Delta)/2.2；TopDelta不计入S'}
    fair_path = BASE / '任11_方法公平来源CPU预检_20260916T062100+0800/任11方法公平预算与来源机器摘要.json'
    fair = audit.js(fair_path)
    mlp_y_path = Path(aggregate['完整来源资格原件'][0]['登记原件'])
    mlp_y = audit.ys(mlp_y_path)
    howard_y_path = BASE / 'Howard适配三网HF联合预算前登记.yaml'
    howard_y = audit.ys(howard_y_path)
    hf_catalog_path = Path(aggregate['完整来源资格原件'][0]['HF数据目录原件'])
    audit.reported_hash(hf_catalog_path, howard_y['HF开发数据目录SHA256'])
    audit.check('MLP Howard本人共同HF12/3数据目录身份',
                mlp_y['HF开发数据目录SHA256'] == howard_y['HF开发数据目录SHA256'])
    audit.check('共同合法三VAL功率与未准测试', fair['协议']['HF合法验证功率_W'] == POWERS and
                howard_y['旧固定TEST温度读取'] is False and mlp_y['旧固定TEST温度读取'] is False)
    hy, my = howard_y['正式预算'], mlp_y['正式预算']
    for field in ['hf_ir_train_points', 'hf_ir_validation_points', 'hf_sensor_train_points',
                  'hf_sensor_validation_points', 'hf_batch_size', 'hf_batches_per_epoch',
                  'physics_collocation', 'physics_steps_per_epoch', 'patience', 'min_delta']:
        audit.check('共同法定折和限定预算字段' + field, hy[field] == my[field])
    audit.check('明确不授予架构单因素因果与作者精确复现',
                hy['strict_update_conditions_equal'] is False and
                howard_y['限制']['仅architecture因果归因'] is False and
                howard_y['原文精确三网联合训练复现'] is False)
    howard_lf_path = BASE / 'Howard适配五LF完训身份清单.json'
    howard_lf = audit.js(howard_lf_path)
    audit.reported_hash(howard_lf_path, howard_y['LF五seed目录SHA256'])
    audit.leaves(howard_lf['五seed描述统计'], howard_lf_path, ['五seed描述统计'], '字段原单位')
    for field in ['最佳LF验证RMSE_摄氏度', '真实累计成本秒']:
        audit.summary([r[field] for r in howard_lf['五LF真实完整资格']], howard_lf_path,
                      ['五seed描述统计', field], howard_lf['五seed描述统计'][field],
                      '秒' if '成本' in field else '摄氏度')
    failure_path = BASE / '任11_Howard_HF_seed2数值失败现场与原303断点CPU304真实复现_20260916T205117+0800.json'
    audit.js(failure_path)
    audit.check('Howard失败seed2不能作为五HF完成', 2 in task['HowardHF失败种子'] and
                task['HowardHF数值失败原结论']['五HF全完训'] is False and
                task['HowardHF数值失败原结论']['五HF最佳S均值'] is None and
                task['HowardHF数值失败原结论']['五HF最佳S样本标准差'] is None)
    attempts_path = BASE / '任11_Howard原五HF有界尝试终态与seed4失败现场_20260916T212312+0800.json'
    attempts = audit.js(attempts_path)
    audit.reported_hash(attempts_path, '214d29a2e88b79f1d285b64d62950b7b2d59235a59c7e675d2af82fbaa890d64')
    queue = attempts['后续唯一必要83493队列']
    prefix = 'HOWARD_HF_BOUNDED_ATTEMPTS_COMPLETE='
    complete_lines = [line[len(prefix):] for line in queue['原完整stdout及stderr工具合并输出'].splitlines()
                      if line.startswith(prefix)]
    audit.check('Howard原完整stdout中唯一尝试终态事件', len(complete_lines) == 1)
    complete = json.loads(complete_lines[0], object_pairs_hook=unique_pairs)
    successes = complete['实际成功种子本人完整资格及原末收据']
    failures = complete['未取得本人完整资格的实际失败种子']
    audit.check('Howard父exit0只表示五原尝试终态而非五资格',
                queue['真实末工具退出0']['exit_code'] == 0 and
                complete['实际已尝试种子数'] == 5 and complete['已本人完整资格种子数'] == 3 and
                complete['五HF全部完训验收'] is False and sorted(map(int, successes)) == [0, 1, 3] and
                sorted(map(int, failures)) == [2, 4])
    audit.check('原raw完整终态与存档摘要互核', queue['实际成功种子'] == [0, 1, 3] and
                queue['实际失败种子'] == [2, 4] and queue['实际尝试数'] == 5 and
                queue['实际成功数'] == 3)
    success_rows = []
    for seed, value in successes.items():
        qualification = value['本人完整资格']
        audit.check('原成功资格真实seed且已完成_只读存档非本次parser',
                    qualification['seed'] == int(seed) and qualification['已完成'] is True and
                    qualification['原文精确三网联合训练复现'] is False)
        success_rows.append({'种子': int(seed), 'S_best_摄氏度': qualification['全局最佳分数'],
                             'S_final_摄氏度': qualification['最近合法分数'],
                             '原值所在raw事件JSONpointer': '/实际成功种子本人完整资格及原末收据/' + seed + '/本人完整资格',
                             '不得统计成功子集冒五seed': True})
    audit.check('两失败CLI exit1完整保留', failures['2']['真实先前CLI退出码'] == 1 and
                failures['4']['真实CLI包装退出码'] == 1)
    audit.check('seed4失败真实stderr同有限AdamW门禁',
                '有限同dtype/shape动量' in failures['4']['真实完整stderr'])
    fourth = attempts['seed4只读真实现场']
    audit.check('seed4未知坏GPU动量数及失败成本保持NULL',
                fourth['坏GPU动量具体+inf参数与元素计数'] is None and
                fourth['额外失败段单独真实墙钟秒'] is None and
                fourth['正式final或最终metrics存在'] is False)
    howard_out = {'性质': howard_y['实施性质'], '作者exact复现': False,
                  '本人LF完整种子': [r['seed'] for r in howard_lf['五LF真实完整资格']],
                  'LF抽样监控RMSE': howard_lf['五seed描述统计']['最佳LF验证RMSE_摄氏度'],
                  'HF完成种子_总状态读取时': task['HowardHF真实完训种子'],
                  'HF失败种子_总状态读取时': task['HowardHF失败种子'],
                  '原五HF全部尝试真实终态原件': str(attempts_path),
                  '原队列完整stdout字符串JSONpointer': '/后续唯一必要83493队列/原完整stdout及stderr工具合并输出',
                  '原raw事件名称': prefix[:-1], '原五HF尝试结束时间': complete['结束'],
                  'HF实际完整成功种子': sorted(map(int, successes)),
                  'HF实际失败种子': sorted(map(int, failures)), '成功种子原分数_不汇总排名': success_rows,
                  'seed4失败现场有限数字': {'失败尝试轮次': fourth['实际失败尝试轮次'],
                       '坏GPU动量inf计数': None, '失败额外完整墙钟秒': None,
                       '没有CPU重跑210或更多训练': fourth['没有CPU重跑210或更多训练']},
                  'HF失败原结论_总状态': task['HowardHF数值失败原结论'],
                  '五HF最佳及末S均值标准差': None, '完整十态能源及OBS成绩': None,
                  '不能成功子集排名': True, '总状态效果结论原文': task['效果结论'],
                  '总状态锚JSONpointer': '/任务/任-11/HowardHF数值失败原结论'}
    after = {}
    for path, ident in audit.sources.items():
        data = Path(path).read_bytes()
        after[path] = digest(data)
        audit.check('静态引用普通原件SHA前后零漂移', after[path] == ident['SHA256'], path)
    source_map = {path: ident['SHA256'] for path, ident in sorted(audit.sources.items())}
    current_status = status_path.read_bytes()
    result = {'审核性质': 'CPU纯现存数字描述统计与来源指向，不是新模型/温度/能源测试',
              '开始时间': began, '结束时间': stamp(), 'PID': os.getpid(), '实际cwd': str(Path.cwd()),
              '环境': {k: os.environ.get(k) for k in ['PYTHONDONTWRITEBYTECODE', 'CUDA_VISIBLE_DEVICES',
                 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'TMPDIR', 'XDG_CACHE_HOME', 'MPLCONFIGDIR',
                 'TORCH_EXTENSIONS_DIR', 'TORCHINDUCTOR_CACHE_DIR']}, 'NumPy版本': np.__version__,
              '描述统计助手只调用': 'describe_column', '描述统计助手SHA256': digest(HELPER.read_bytes()),
              '四轮描述统计复核': ['原五值有限、NumPy与标准库mean/stdev及ddof1互核',
                 '仅合法VAL指标CSV、摄氏度/瓦/秒/字节、空缺与身份核对',
                 'seed为单位、功率等权、ABS/Delta逐seed合并、成本逐会话不重复',
                 '跨表JSON/CSV一致、SOURCE指向/静态SHA前后、方法资格与无超界结论'],
              '检查总数': len(audit.checks), '检查失败数': sum(not r['通过'] for r in audit.checks),
              '所有检查': audit.checks, '五种子描述统计': audit.stats, '新MLP': mlp,
              'FEM': fem_out, 'Ridge': ridge_out, 'Howard': howard_out,
              '引用原件': list(audit.sources.values()), '静态引用原件数': len(source_map),
              '原件SHA256映射': source_map, '原件SHA256映射canonicalJSON_SHA256':
              digest(json.dumps(source_map, ensure_ascii=False, sort_keys=True,
                                separators=(',', ':')).encode('utf-8')),
              '原件SHA256映射_复核后': after,
              '可合法变化的总状态': {'路径': str(status_path), '起始字节': len(status_bytes),
                   '起始SHA256': digest(status_bytes), '结束字节': len(current_status),
                   '结束SHA256': digest(current_status), '本次起始任11快照': task,
                   '说明': '根继续原Howard3/4及合法进度追加；不要求跨合法记录整ROOT或总状态恒定'},
              '限界': {'读取实际模型PT': False, '读取原温度或逐点预测JSONL': False,
                   '调用CUDA或正式模型来源parser': False, '新训练或新测量': False,
                   '任务12或13启动': False, 'B0替换': False, '全局公平排名': False,
                   '端到端FEM预处理成本已知': False, '工程资格': False, '总goal完成': False}}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'检查总数': result['检查总数'], '检查失败数': result['检查失败数'],
        '五种子统计组数': len(audit.stats), '静态原件数': len(source_map),
        '来源映射SHA256': result['原件SHA256映射canonicalJSON_SHA256'],
        'MLP冻结口径': mlp['冻结口径'], 'MLP观察Float64口径': mlp['观察Float64口径'],
        'MLP成本五原值': mlp['五身份LF加HF循环成本'],
        'MLP循环秒合计': mlp['五身份LF加HF循环秒合计'],
        'MLP最大显存字节': mlp['全部十训练身份显存峰值字节取max'],
        'FEM': fem_out, 'Ridge宏': {k: v for k, v in ridge_out.items() if k != '原逐功率九行'},
        'Howard完整HF种子原终态': howard_out['HF实际完整成功种子'],
        'Howard失败种子原终态': howard_out['HF实际失败种子'],
        '失败检查': [r for r in audit.checks if not r['通过']]},
        ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result['检查失败数'] == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--模式', choices=['核对', '封存'], required=True)
    parser.add_argument('--前缀', required=True)
    args = parser.parse_args()
    if Path.cwd() != ROOT or '/' in args.前缀 or '\\' in args.前缀:
        raise ValueError('仅项目cwd与本档案固定子文件名')
    directory = Path(__file__).resolve().parent
    output = directory / (args.前缀 + '_四方法完整机器证据.json')
    if args.模式 == '核对':
        return check_all(output)
    argv = [sys.executable, '-B', str(Path(__file__).resolve()), '--模式', '核对', '--前缀', args.前缀]
    began, monotonic = stamp(), time.perf_counter()
    process = subprocess.run(argv, cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True)
    receipt = {'开始时间': began, '结束时间': stamp(), 'PID': os.getpid(),
               '实际cwd': str(ROOT), '实际argv': argv, '实际子退出码': process.returncode,
               '完整stdout': process.stdout, '完整stderr': process.stderr,
               '仅盘点脚本墙钟秒_不是训练或推断成本': time.perf_counter() - monotonic,
               '盘点脚本SHA256': digest(Path(__file__).read_bytes()),
               '机器证据实际存在': output.is_file()}
    with (directory / (args.前缀 + '_实际命令完整终态.json')).open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(process.stdout, end='')
    print(process.stderr, end='', file=sys.stderr)
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
