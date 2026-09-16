#!/usr/bin/env python3
"""只读取既有中文报告和 JSON/CSV；不读取模型、温度档或启动实验。"""

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
import sys
from datetime import datetime, timezone, timedelta

import numpy as np


ROOT = Path('/home/phl/lyf/Temperature Field Prediction')
HERE = Path(__file__).resolve().parent
TZ = timezone(timedelta(hours=8))
PLAN = '多保真DeepONet预测精度优化总计划与执行台账.md'
STATUS = '研究记录/总任务状态.json'
REPORTS = [
    '研究记录/任务01_启动与可追溯性/验收报告.md',
    '研究记录/任务02_训练排程/验收报告.md',
    '研究记录/任务03_低保真精度修复/验收判定_预算门禁复核.json',
    '研究记录/任务04_联合微调/正式两臂与四状态验收报告.md',
    '研究记录/任务05_物理一致性/任务05_条件判定与有限退出报告.md',
    '研究记录/任务06_时间响应特征/HF三臂真实600轮与六状态能源验收.md',
    '研究记录/任务07_正式五种子重训/修复版五种子合法验证稳定性与双轨能源验收.md',
    '研究记录/任务08_贡献消融/F2_正式五种子三态能源与合法HF受限验收总报告.md',
    '研究记录/任务09_高保真数据效率/正式F3主序列汇总_录0096_20260916T122137+0800/F3主序列正式中文验收报告.md',
    '研究记录/任务10_独立双层场基准/任10_全25真CUDA九档完整场后验正式中文验收_20260916T1258.md',
]
T7 = '研究记录/任务07_正式五种子重训/正式五种子HF合法验证逐窗明细_20260916T022125+0800/五种子观测验证摘要.json'
T8 = '研究记录/任务08_贡献消融/F2_五种子合法HF逐窗与径向_20260916T051312+0800/五种子合法观测摘要.json'
T8E = '研究记录/任务08_贡献消融/F2_五种子三态独立能源_机器验收摘要.json'
T8C = '研究记录/任务08_贡献消融/F2_五种子三态独立能源_逐态原件与有量纲统计.csv'
T9 = '研究记录/任务09_高保真数据效率/正式F3主序列汇总_录0096_20260916T122137+0800'
T10 = '研究记录/任务10_独立双层场基准/正式九热流全25后验_录0098_20260916T124800+0800'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('JSON 重复键: ' + key)
        out[key] = value
    return out


class Audit:
    def __init__(self):
        self.original = {}
        self.content = {}
        self.checks = []
        self.statistics = {}
        self.claims = []

    def read(self, relative):
        if relative in self.content:
            return self.content[relative]
        path = ROOT / relative
        if path.resolve() != path or ROOT not in path.parents:
            raise ValueError('非项目内原生路径: ' + relative)
        if path.suffix.lower() not in {'.json', '.csv', '.md'}:
            raise ValueError('仅允许报告 JSON/CSV/Markdown: ' + relative)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('非普通原件: ' + relative)
        data = path.read_bytes()
        self.original[relative] = {
            '精确绝对路径': str(path), 'SHA256': sha(data), '字节数': len(data),
            '普通文件': True, '权限八进制': oct(stat.S_IMODE(info.st_mode)),
            '读取时原件mtime_ns': info.st_mtime_ns,
        }
        self.content[relative] = data
        return data

    def text(self, relative):
        return self.read(relative).decode('utf-8-sig')

    def js(self, relative):
        return json.loads(self.text(relative), object_pairs_hook=unique_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    def rows(self, relative):
        reader = csv.DictReader(io.StringIO(self.text(relative)))
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('CSV 表头为空或重复: ' + relative)
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError('CSV 列数不一致: ' + relative)
        return rows

    def check(self, name, actual, expected, location, tolerance=0.0):
        if isinstance(expected, float):
            passed = isinstance(actual, (int, float)) and not isinstance(actual, bool)
            passed = passed and math.isfinite(float(actual)) and abs(float(actual) - expected) <= tolerance
        else:
            passed = type(actual) is type(expected) and actual == expected
        self.checks.append({'核对项': name, '实际重算或原字段': actual, '原验收值或约束': expected,
                            '通过': bool(passed), '允许绝对误差': tolerance, '原件定位': location})

    def ref(self, relative, pointer=None, selection=None, unit=None, lines=None):
        self.read(relative)
        return {'精确路径': str(ROOT / relative), '相对路径': relative,
                '来源SHA256': self.original[relative]['SHA256'], 'JSON指针': pointer,
                'CSV筛选与聚合': selection, '单位': unit, '报告行号': lines}

    def five(self, key, values, source):
        if len(values) != 5 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
            raise ValueError('必须五个有限真实数值，不补缺值: ' + key)
        summary = describe_column(values)
        # 第二种标准库算法交叉核对，仅是既有五个值的描述统计。
        import statistics
        self.check(key + ' 两算法均值', summary['mean'], statistics.mean(values), source, 2e-12)
        self.check(key + ' 两算法样本标准差', summary['std'], statistics.stdev(values), source, 2e-12)
        self.statistics[key] = {'已有五种子原值': values, '已有种子数': 5,
                                '均值': summary['mean'], '样本标准差_ddof1': summary['std'],
                                '样本分母': 4, '原件定位': source,
                                '不是新实验次数': True, 'p值': None, '置信区间': None}
        return summary

    def table_rows(self, relative, header):
        lines = self.text(relative).splitlines()
        out = []
        in_table = False
        for line_number, line in enumerate(lines, 1):
            if not line.startswith('|'):
                if in_table:
                    break
                continue
            cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
            if cells == header:
                in_table = True
                continue
            if in_table and not all(set(cell) <= set('-: ') for cell in cells):
                if len(cells) != len(header):
                    raise ValueError('验收 Markdown 表列数错误')
                out.append((line_number, dict(zip(header, cells))))
        if not out:
            raise ValueError('验收表头未找到: ' + str(header))
        return out


skill = Path('/home/phl/.codex/skills/data-analysis/scripts/stat_summary.py')
spec = importlib.util.spec_from_file_location('existing_stats_helper', skill)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
describe_column = module.describe_column


def numeric(row, column):
    value = float(row[column])
    if not math.isfinite(value):
        raise ValueError('CSV 非有限数值: ' + column)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--机器输出', required=True)
    parser.add_argument('--结果表', required=True)
    args = parser.parse_args()
    for target in (args.机器输出, args.结果表):
        path = Path(target).resolve()
        if path.parent != HERE or path.exists():
            raise ValueError('只能创建本人目录中新的输出，不覆盖: ' + target)
    if Path.cwd() != ROOT or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('要求项目 cwd 与 CUDA_VISIBLE_DEVICES 空字符串')
    for key in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'CUDA_CACHE_PATH', 'MPLCONFIGDIR', 'TORCH_EXTENSIONS_DIR', 'TORCHINDUCTOR_CACHE_DIR'):
        value = Path(os.environ[key]).resolve()
        if ROOT not in value.parents or not value.is_dir():
            raise ValueError('临时/缓存路径必须已有且项目内: ' + key)
    audit = Audit()
    begin = datetime.now(TZ).isoformat()
    audit.text(PLAN)
    status = audit.js(STATUS)
    task_subset = {key: status['任务'][key] for key in ('任-%02d' % n for n in range(1, 11))}
    for n, report in enumerate(REPORTS, 1):
        audit.text(report)
        item = task_subset['任-%02d' % n]
        audit.check('任%02d 当前状态证据路径' % n, item['证据'], report,
                    audit.ref(STATUS, '/任务/任-%02d/证据' % n))
        if '证据SHA256' in item:
            audit.check('任%02d 主报告现SHA与登记' % n, audit.original[report]['SHA256'], item['证据SHA256'],
                        audit.ref(STATUS, '/任务/任-%02d/证据SHA256' % n))
    registered_results = [
        ('任-07', T7, '合法HF观测中文摘要SHA256'),
        ('任-08', T8, 'F2五seed合法观测摘要SHA256'),
        ('任-08', T8E, '机器验收摘要SHA256'),
        ('任-08', T8C, '十五态原瓦数与SHA索引SHA256'),
        ('任-09', T9 + '/F3主序列十五身份机器摘要.json', '正式聚合机器摘要SHA256'),
        ('任-09', T9 + '/F3主序列HF数量原点.csv', '正式聚合三原点CSV_SHA256'),
        ('任-09', T9 + '/F3主序列五seed逐模型.csv', '正式聚合逐模型CSV_SHA256'),
        ('任-10', T10 + '/九档25板模型实际双材料内部误差.csv', '九档完整场CSV_SHA256'),
        ('任-10', T10 + '/九档25板模型非观测窗.csv', '九档非观测窗CSV_SHA256'),
        ('任-10', T10 + '/九档25板模型七时刻名义工程能源.csv', '九档名义能源CSV_SHA256'),
        ('任-10', T10 + '/九热流后验资格和全组中文摘要.json', '后验资格摘要SHA256'),
    ]
    for task, path, field in registered_results:
        audit.read(path)
        audit.check('已有正式结果源SHA互锁' + task + field, audit.original[path]['SHA256'],
                    task_subset[task][field], audit.ref(STATUS, '/任务/' + task + '/' + field))

    t1 = '研究记录/任务01_启动与可追溯性/任务01_短跑_种子0_20260915T171921+0800/metrics.json'
    one = audit.js(t1)
    audit.check('任01 单种子启动轮次', one['epochs_completed'], 1, audit.ref(t1, '/epochs_completed'))
    audit.claims.append({'任务': '任01', '指标': '启动1轮S', '值': one['best_validation_selection_score_c'],
                         '样本标准差': None, '已有种子数': 1, '来源': audit.ref(t1, '/best_validation_selection_score_c', unit='摄氏度')})

    t2_paths = ['研究记录/任务02_训练排程/任务02_S0正式_种子0_20260915T174917+0800/metrics.json',
                '研究记录/任务02_训练排程/任务02_S1正式_种子0_20260915T175228+0800/metrics.json']
    t2_energy_paths = ['研究记录/任务02_训练排程/任务02_S0最佳_能量审核_20260915T175825+0800/汇总指标.json',
                       '研究记录/任务02_训练排程/任务02_S1最佳_能量审核_20260915T175857+0800/汇总指标.json']
    t2scores = []
    t2energy = []
    for path, ep in zip(t2_paths, (140, 280)):
        item = audit.js(path)
        audit.check('任02 最佳轮', item['best_epoch'], ep, audit.ref(path, '/best_epoch'))
        audit.check('任02 真300轮原声明', item['epochs_completed'], 300, audit.ref(path, '/epochs_completed'))
        t2scores.append(item['best_validation_selection_score_c'])
    for path in t2_energy_paths:
        item = audit.js(path)
        t2energy.append((item['绝对平衡宏均值_瓦'], item['绝对平衡95分位_瓦']))
    improvement2 = (t2scores[0] - t2scores[1]) / t2scores[0] * 100
    audit.check('任02 报告1.285%六位附近', round(improvement2, 3), 1.285, audit.ref(REPORTS[1], lines=[7]), 1e-12)
    improvement2_energy = (t2energy[0][0] - t2energy[1][0]) / t2energy[0][0] * 100
    worsening2_p95 = (t2energy[1][1] - t2energy[0][1]) / t2energy[0][1] * 100
    audit.check('任02 能源均值改善4.347%原报告', round(improvement2_energy, 3), 4.347, audit.ref(REPORTS[1], lines=[12]), 1e-12)
    audit.check('任02 能源p95恶化0.434%原报告', round(worsening2_p95, 3), 0.434, audit.ref(REPORTS[1], lines=[13]), 1e-12)
    audit.claims.append({'任务': '任02', '同预算S0与S1_摄氏度': t2scores, 'S1相对S0改善百分比': improvement2,
                         '同状态最佳能源均值_p95_瓦': t2energy, '样本标准差': None, '已有种子数': 1,
                         '来源': [audit.ref(p) for p in t2_paths + t2_energy_paths]})
    t2_windows_path = '研究记录/任务02_训练排程/任务02_观测窗口复核_20260915T180950+0800/验证窗口原始重算.json'
    windows2 = audit.js(t2_windows_path)
    for arm in ('S0', 'S1'):
        top = windows2[arm]['顶部验证']['per_power']
        empty = [x for x in top if abs(x['power_w'] - 115.2) < 0.001]
        audit.check('任02 ' + arm + '115.2W后期无顶部值', empty[0]['time_windows']['time_100_200_s'], None,
                    audit.ref(t2_windows_path, '/' + arm + '/顶部验证/per_power/0/time_windows/time_100_200_s'))

    t3 = audit.js(REPORTS[2])
    for path, expected_sha in t3['输入SHA256'].items():
        audit.read(path)
        audit.check('任03 原验收输入SHA', audit.original[path]['SHA256'], expected_sha, audit.ref(REPORTS[2], '/输入SHA256/' + path.replace('~', '~0').replace('/', '~1')))
    lf_csv = next(path for path in t3['输入SHA256'] if path.endswith('LF验证材料时间宏指标.csv'))
    lf_rows = audit.rows(lf_csv)
    for arm, materials in t3['LF完整场'].items():
        for material, weights in materials.items():
            for weight, expected in weights.items():
                label = '节点等权' if weight == 'node' else '轴对称有限元集总真实体积'
                selected = [r for r in lf_rows if r['运行臂'] == arm and r['材料'] == material and r['时间窗'] == '全时段' and r['权重口径'] == label]
                audit.check('任03 LF全时段唯一行', len(selected), 1, audit.ref(lf_csv, selection={'运行臂': arm, '材料': material, '时间窗': '全时段', '权重口径': label}))
                audit.check('任03 LF原CSV与验收', numeric(selected[0], '逐功率等权宏RMSE_摄氏度'), expected,
                            audit.ref(lf_csv, selection={'运行臂': arm, '材料': material, '时间窗': '全时段', '权重口径': label}, unit='摄氏度'), 1e-12)
    control3 = float(np.mean([t3['LF完整场']['原采样接续'][m]['volume'] for m in ('Cu', 'SiC')]))
    balanced3 = float(np.mean([t3['LF完整场']['空间平衡接续'][m]['volume'] for m in ('Cu', 'SiC')]))
    change3 = (control3 - balanced3) / control3 * 100
    guard3 = t3['LF空间机制预设体积护栏']
    for actual, key in [(control3, '原采样LF体积等权平均_摄氏度'), (balanced3, '空间LF体积等权平均_摄氏度'), (change3, '空间LF体积平均改善率_百分比')]:
        audit.check('任03 LF机制重算' + key, actual, guard3[key], audit.ref(REPORTS[2], '/LF空间机制预设体积护栏/' + key), 1e-12)
    audit.check('任03 空间机制未采纳', guard3['LF空间机制满足采用门槛'], False, audit.ref(REPORTS[2], '/LF空间机制预设体积护栏/LF空间机制满足采用门槛'))
    for arm in ('空间LF同源', '拟合LF同源'):
        source = t3['HF三臂']['旧LF同源']
        candidate = t3['HF三臂'][arm]
        result = t3['新LF-HF组合采用护栏'][arm]
        improvement = (source['选择分数_摄氏度'] - candidate['选择分数_摄氏度']) / source['选择分数_摄氏度'] * 100
        audit.check('任03 HF改善百分比重算', improvement, result['HF选分改善率_百分比'], audit.ref(REPORTS[2], '/新LF-HF组合采用护栏/' + arm + '/HF选分改善率_百分比'), 1e-12)
        audit.check('任03 HF不采用', result['HF组合值得采用'], False, audit.ref(REPORTS[2], '/新LF-HF组合采用护栏/' + arm + '/HF组合值得采用'))
        top_worsening = candidate['顶部热端冷端逐功率宏RMSE_摄氏度']['top'] - source['顶部热端冷端逐功率宏RMSE_摄氏度']['top']
        audit.check('任03 顶部实际超过护栏', top_worsening > result['顶部允许增加_摄氏度'], True, audit.ref(REPORTS[2], '/新LF-HF组合采用护栏/' + arm))
        audit.check('任03 工程p95确实恶化', candidate['绝对能量95分位_瓦'] > source['绝对能量95分位_瓦'], True, audit.ref(REPORTS[2], '/HF三臂/' + arm + '/绝对能量95分位_瓦'))
    audit.claims.append({'任务': '任03', 'LF控制与空间两材料volume均值_摄氏度': [control3, balanced3], '空间改善百分比': change3,
                         'LF拟合收益不能称HF采用': True, 'HF三臂原字段': t3['HF三臂'], '来源': audit.ref(REPORTS[2])})

    t4paths = ['研究记录/任务04_联合微调/任04_冻结LF_种子0_20260915T210001+0800/阶段报告.json',
               '研究记录/任务04_联合微调/任04_有限解冻_种子0_20260915T210714+0800/阶段报告.json']
    t4 = [audit.js(p) for p in t4paths]
    review4 = task_subset['任-04']['独立二审']
    audit.read(review4)
    audit.check('任04 已有正式独立二审SHA', audit.original[review4]['SHA256'], task_subset['任-04']['独立二审SHA256'], audit.ref(STATUS, '/任务/任-04/独立二审SHA256'))
    t4scores = [x['观测最佳验证选分_摄氏度'] for x in t4]
    t4final = [x['最后一轮门禁证据']['HF合法验证选分_摄氏度'] for x in t4]
    for path, item in zip(t4paths, t4):
        for key, expected in [('本臂实际完成轮次', 200), ('观测最佳任04轮次', 120), ('物理最佳任04轮次', 180)]:
            audit.check('任04 ' + key, item[key], expected, audit.ref(path, '/' + key))
        audit.check('任04 每臂物理配点', item['累计实际消耗']['物理配点'], 51200, audit.ref(path, '/累计实际消耗/物理配点'))
    t4energy_paths = [
        '研究记录/任务04_联合微调/独立能量_冻结最佳_20260915T215820+0800/汇总指标.json',
        '研究记录/任务04_联合微调/独立能量_冻结末态_20260915T221905+0800/汇总指标.json',
        '研究记录/任务04_联合微调/独立能量_联合最佳_20260915T222106+0800/汇总指标.json',
        '研究记录/任务04_联合微调/独立能量_联合末态_20260915T222214+0800/汇总指标.json',
    ]
    t4energy = [audit.js(p) for p in t4energy_paths]
    for path, item in zip(t4energy_paths, t4energy):
        audit.check('任04 独立能源30点原声明', item['功率时刻审核行数'], 30, audit.ref(path, '/功率时刻审核行数'))
        audit.check('任04 求积原阶数', item['求积阶数'], [16, 64], audit.ref(path, '/求积阶数'))
    audit.claims.append({'任务': '任04', '同预算两臂最佳S_摄氏度': t4scores, '同轮两臂末态S_摄氏度': t4final,
                         '联合相对冻结改善百分比': (t4scores[0] - t4scores[1]) / t4scores[0] * 100,
                         '四状态能源原字段': t4energy, '已有种子数': 1, '样本标准差': None,
                         '来源': [audit.ref(p) for p in t4paths + t4energy_paths]})
    audit.claims.append({'任务': '任05', '正式新增训练候选数': 0, '不是新能源测量': True,
                         '三项未触发原说明': audit.ref(REPORTS[4], lines=[13, 15, 16, 17, 23]),
                         '沿用任04联合最佳能源': audit.ref(t4energy_paths[2])})

    t6paths = ['研究记录/任务06_时间响应特征/HF先导_E0_正式_种子0_20260915T234337+0800/阶段报告.json',
               '研究记录/任务06_时间响应特征/HF先导_E1_正式_种子0_20260915T234850+0800/阶段报告.json',
               '研究记录/任务06_时间响应特征/HF先导_E2_正式_种子0_20260915T235325+0800/阶段报告.json']
    t6 = [audit.js(p) for p in t6paths]
    for path, item in zip(t6paths, t6):
        for key, expected in [('本臂实际完成轮次', 600), ('观测最佳任06轮次', 0), ('物理最佳任06轮次', 580)]:
            audit.check('任06 ' + key, item[key], expected, audit.ref(path, '/' + key))
        audit.check('任06 三臂最佳同源S', item['观测最佳HF合法选分_摄氏度'], t4scores[1], audit.ref(path, '/观测最佳HF合法选分_摄氏度'), 1e-12)
        audit.check('任06 物理配点', item['累计实际消耗']['物理配点'], 153600, audit.ref(path, '/累计实际消耗/物理配点'))
    t6energy = {}
    for arm in ('E0', 'E1', 'E2'):
        for state_label in ('最佳', '末态'):
            path = '研究记录/任务06_时间响应特征/独立能量_' + arm + '_' + state_label + '_20260916T001311+0800/汇总指标.json'
            item = audit.js(path)
            t6energy[arm + state_label] = {'均值_瓦': item['绝对平衡宏均值_瓦'], 'p95_瓦': item['绝对平衡95分位_瓦'], '来源': audit.ref(path)}
            if state_label == '最佳':
                audit.check('任06 最佳能源同任04联合最佳', item['绝对平衡宏均值_瓦'], t4energy[2]['绝对平衡宏均值_瓦'], audit.ref(path, '/绝对平衡宏均值_瓦'), 1e-9)
    audit.claims.append({'任务': '任06', '三臂最佳S_摄氏度': [x['观测最佳HF合法选分_摄氏度'] for x in t6],
                         '三臂第600轮S_摄氏度': [x['最后一轮门禁证据']['HF合法验证选分_摄氏度'] for x in t6],
                         '六状态既有能源': t6energy, '已有种子数': 1, '样本标准差': None,
                         '来源': [audit.ref(p) for p in t6paths]})

    seven = audit.js(T7)
    for name in ('本轮合法macro_v1', '历史B0合法macro_v1'):
        obj = seven[name]
        values = [obj['逐种子选分_摄氏度'][str(seed)] for seed in range(5)]
        source = audit.ref(T7, '/' + name + '/逐种子选分_摄氏度', unit='摄氏度')
        result = audit.five('任07_' + name, values, source)
        audit.check('任07 已有均值', result['mean'], obj['五种子均值_摄氏度'], audit.ref(T7, '/' + name + '/五种子均值_摄氏度'), 1e-12)
        audit.check('任07 已有样本std', result['std'], obj['五种子样本标准差_摄氏度'], audit.ref(T7, '/' + name + '/五种子样本标准差_摄氏度'), 1e-12)
        audit.check('任07 样本分母', obj['样本标准差分母'], 4, audit.ref(T7, '/' + name + '/样本标准差分母'))
    audit.check('任07 每seed历史B0更优', all(a < b for a, b in zip(audit.statistics['任07_历史B0合法macro_v1']['已有五种子原值'], audit.statistics['任07_本轮合法macro_v1']['已有五种子原值'])), True, audit.ref(T7))
    energy7_paths = {
        '观测最佳': [
            '独立能源_修版seed0_观测最佳_20260916T015810+0800',
            '独立能源_修版seed1_观测最佳_20260916T020040+0800',
            '独立能源_修版seed2_观测最佳_20260916T020040+0800',
            '独立能源_修版seed3_观测最佳_20260916T020040+0800',
            '独立能源_修版seed4_观测最佳_20260916T020244+0800',
        ],
        '真末': [
            '独立能源_修版seed0_真末_20260916T020040+0800',
            '独立能源_修版seed1_真末_20260916T020244+0800',
            '独立能源_修版seed2_真末_20260916T020244+0800',
            '独立能源_修版seed3_真末_20260916T020244+0800',
            '独立能源_修版seed4_真末_20260916T020600+0800',
        ],
    }
    energy7 = {}
    for state_label, dirs in energy7_paths.items():
        items = []
        refs = []
        for seed, directory in enumerate(dirs):
            base = '研究记录/任务07_正式五种子重训/' + directory
            path = base + '/汇总指标.json'
            item = audit.js(path)
            index = audit.js(base + '/审计工件SHA256.json')
            audit.check('任07 原能源汇总与原工件索引SHA', audit.original[path]['SHA256'], index['汇总指标.json'], audit.ref(base + '/审计工件SHA256.json', '/汇总指标.json'))
            audit.check('任07 原能源对应本人seed', item['运行种子'], seed, audit.ref(path, '/运行种子'))
            audit.check('任07 原能源30点', item['功率时刻审核行数'], 30, audit.ref(path, '/功率时刻审核行数'))
            audit.check('任07 原能源16/64阶', item['求积阶数'], [16, 64], audit.ref(path, '/求积阶数'))
            items.append(item)
            refs.append(audit.ref(path, '/绝对平衡宏均值_瓦', unit='瓦'))
        values = [item['绝对平衡宏均值_瓦'] for item in items]
        summary = audit.five('任07_' + state_label + '_能源', values, refs)
        energy7[state_label] = {'五seed原宏均值_瓦': values, '五seed原p95_瓦': [item['绝对平衡95分位_瓦'] for item in items],
                               '均值_瓦': summary['mean'], '样本std_瓦': summary['std'], '来源': refs}
        reported_mean = 947.256 if state_label == '观测最佳' else 925.389
        audit.check('任07 原报告能源五seed均值三位', round(summary['mean'], 3), reported_mean, audit.ref(REPORTS[6], lines=[27]), 1e-12)
    header7 = ['seed', '真校正＋真联合轮', '观测最佳全局轮／独立物理最佳轮', '合法最佳S／℃', '历史B0同seed S／℃', '最佳态名义绝对平衡均值／p95，W', '真末名义绝对平衡均值／p95，W']
    for line, row in audit.table_rows(REPORTS[6], header7):
        seed = int(row['seed'])
        audit.check('任07 原S表本人最佳逐seed', '%.6f' % audit.statistics['任07_本轮合法macro_v1']['已有五种子原值'][seed], row['合法最佳S／℃'], audit.ref(REPORTS[6], lines=[line]))
        audit.check('任07 原S表历史逐seed', '%.6f' % audit.statistics['任07_历史B0合法macro_v1']['已有五种子原值'][seed], row['历史B0同seed S／℃'], audit.ref(REPORTS[6], lines=[line]))
        for state_label, column in [('观测最佳', '最佳态名义绝对平衡均值／p95，W'), ('真末', '真末名义绝对平衡均值／p95，W')]:
            item = energy7[state_label]
            displayed = '%.3f／%.3f' % (item['五seed原宏均值_瓦'][seed], item['五seed原p95_瓦'][seed])
            audit.check('任07 原能源表逐格原五值', displayed, row[column], audit.ref(REPORTS[6], lines=[line]))
    for field in ('P2径向中文证据', 'P2径向项目内断言与双GPU说明'):
        path = task_subset['任-07'][field]
        audit.read(path)
        audit.check('任07 后续径向中文原件SHA', audit.original[path]['SHA256'], task_subset['任-07'][field + 'SHA256'], audit.ref(STATUS, '/任务/任-07/' + field + 'SHA256'))
    f1_path = '研究记录/任务08_贡献消融/F1_HF独立入场与物理来源限制.md'
    audit.read(f1_path)
    audit.check('任08 F1来源限制报告SHA', audit.original[f1_path]['SHA256'], task_subset['任-08']['F1独立来源中文证据SHA256'], audit.ref(STATUS, '/任务/任-08/F1独立来源中文证据SHA256'))

    eight = audit.js(T8)
    values8 = [eight['逐seed真实合法选择分_摄氏度'][str(seed)] for seed in range(5)]
    summary8 = audit.five('任08_F2合法S', values8, audit.ref(T8, '/逐seed真实合法选择分_摄氏度', unit='摄氏度'))
    audit.check('任08 已有S均值', summary8['mean'], eight['五种子宏平均选择分_摄氏度'], audit.ref(T8, '/五种子宏平均选择分_摄氏度'), 1e-12)
    audit.check('任08 已有S样本std', summary8['std'], eight['五种子宏选择分样本标准差_摄氏度'], audit.ref(T8, '/五种子宏选择分样本标准差_摄氏度'), 1e-12)
    energy8 = audit.js(T8E)
    rows8 = audit.rows(T8C)
    audit.check('任08 三状态五seed15行', len(rows8), 15, audit.ref(T8C))
    for state_label in ('best', 'physical', 'final'):
        selected = sorted([r for r in rows8 if r['state'] == state_label], key=lambda r: int(r['seed']))
        audit.check('任08 状态完整0..4', [int(r['seed']) for r in selected], list(range(5)), audit.ref(T8C, selection={'state': state_label}))
        values = [numeric(row, 'abs_balance_macro_w') for row in selected]
        summary = audit.five('任08_' + state_label + '_能源', values, audit.ref(T8C, selection={'state': state_label, '排序': 'seed0..4', '列': 'abs_balance_macro_w'}, unit='瓦'))
        old = energy8['五种子逐状态绝对工程平衡宏均值样本_瓦'][state_label]
        for actual, key in [(summary['mean'], '均值'), (summary['std'], '样本标准差_ddof1')]:
            audit.check('任08 既有状态能源' + key, actual, old[key], audit.ref(T8E, '/五种子逐状态绝对工程平衡宏均值样本_瓦/' + state_label + '/' + key), 1e-10)
        audit.check('任08 五seed原数组与CSV一致', values, old['逐种子'], audit.ref(T8E, '/五种子逐状态绝对工程平衡宏均值样本_瓦/' + state_label + '/逐种子'))
        p95mean = float(np.mean([numeric(r, 'abs_balance_p95_w') for r in selected]))
        audit.check('任08 是逐seed p95均值', p95mean, old['五个逐seed_p95的均值'], audit.ref(T8E, '/五种子逐状态绝对工程平衡宏均值样本_瓦/' + state_label + '/五个逐seed_p95的均值'), 1e-10)
    for key in ('F2名义工程能源合格断言', '严格字节级唯PDE因果收益可宣称', '内部HF实验真值可得'):
        audit.check('任08 资格不可推断' + key, energy8[key], False, audit.ref(T8E, '/' + key))
    audit.check('任08 缺PDE是null不是0', energy8['训练体内PDE'], None, audit.ref(T8E, '/训练体内PDE'))
    for key, expected in [('完整状态数', 15), ('总64阶有量纲工程指标行数', 450), ('总原始散度行数', 900), ('总原始能量行数', 900)]:
        audit.check('任08 最终原计数' + key, energy8[key], expected, audit.ref(T8E, '/' + key))
    audit.check('任08 能源摘要复用S均值', energy8['合法HF五种子选择分宏均值_摄氏度'], summary8['mean'], audit.ref(T8E, '/合法HF五种子选择分宏均值_摄氏度'), 1e-12)
    audit.check('任08 能源摘要复用S样本std', energy8['合法HF五种子选择分样本标准差_摄氏度'], summary8['std'], audit.ref(T8E, '/合法HF五种子选择分样本标准差_摄氏度'), 1e-12)

    nine_path = T9 + '/F3主序列十五身份机器摘要.json'
    nine = audit.js(nine_path)
    rows9_path = T9 + '/F3主序列五seed逐模型.csv'
    points9_path = T9 + '/F3主序列HF数量原点.csv'
    rows9 = audit.rows(rows9_path)
    points9 = audit.rows(points9_path)
    audit.check('任09 已有15行模型', len(rows9), 15, audit.ref(rows9_path))
    audit.check('任09 已有3原点', len(points9), 3, audit.ref(points9_path))
    for old in nine['主序列三个原点']:
        count = old['HF功率数']
        selected = sorted([r for r in rows9 if int(r['HF功率数']) == count], key=lambda r: int(r['seed']))
        audit.check('任09 HF档0..4齐', [int(r['seed']) for r in selected], list(range(5)), audit.ref(rows9_path, selection={'HF功率数': count}))
        point = next(r for r in points9 if int(r['HF功率数']) == count)
        for label, column, mean_key, std_key in [
            ('bestS', 'best合法S_摄氏度', '观测最佳S_均值_摄氏度', '观测最佳S_样本标准差_摄氏度'),
            ('finalS', 'final合法S_摄氏度', '训练末S_均值_摄氏度', '训练末S_样本标准差_摄氏度'),
            ('best能源', 'best原能源宏均值_瓦', 'best原能源宏均值_五seed均值_瓦', 'best原能源宏均值_五seed样本标准差_瓦'),
            ('final能源', 'final原能源宏均值_瓦', 'final原能源宏均值_五seed均值_瓦', 'final原能源宏均值_五seed样本标准差_瓦'),
        ]:
            summary = audit.five('任09_HF%d_%s' % (count, label), [numeric(r, column) for r in selected], audit.ref(rows9_path, selection={'HF功率数': count, '列': column, '排序': 'seed0..4'}, unit='摄氏度' if label.endswith('S') else '瓦'))
            for actual, key in [(summary['mean'], mean_key), (summary['std'], std_key)]:
                audit.check('任09 原JSON与逐模型重算' + key, actual, old[key], audit.ref(nine_path, '/主序列三个原点/' + str((3, 6, 9).index(count)) + '/' + key), 1e-10)
                if key in point:
                    audit.check('任09 原点CSV与逐模型重算' + key, actual, numeric(point, key), audit.ref(points9_path, selection={'HF功率数': count, '列': key}), 1e-10)
        for state_label in ('best', 'final'):
            flags = [r[state_label + '能源资格'] for r in selected]
            audit.check('任09 原资格均明确False', flags, ['False'] * 5, audit.ref(rows9_path, selection={'HF功率数': count, '列': state_label + '能源资格'}))
            audit.check('任09 能源0/5', old[state_label + '能源安全种子数'], 0, audit.ref(nine_path, '/主序列三个原点/' + str((3, 6, 9).index(count)) + '/' + state_label + '能源安全种子数'))
    for key in ('同一目标误差所需HF数量', '高保真数量节省比例'):
        audit.check('任09 不把三点最低当数量节省', nine[key], None, audit.ref(nine_path, '/' + key))
    audit.check('任09 HF12异轨参照', nine['HF12历史参照_不属新三档同轨迹']['与新十五模型同一HF训练轨迹'], False, audit.ref(nine_path, '/HF12历史参照_不属新三档同轨迹/与新十五模型同一HF训练轨迹'))

    ten_error_path = T10 + '/九档25板模型实际双材料内部误差.csv'
    ten_window_path = T10 + '/九档25板模型非观测窗.csv'
    ten_energy_path = T10 + '/九档25板模型七时刻名义工程能源.csv'
    ten_json_path = T10 + '/九热流后验资格和全组中文摘要.json'
    error10 = audit.rows(ten_error_path)
    window10 = audit.rows(ten_window_path)
    energy10 = audit.rows(ten_energy_path)
    ten = audit.js(ten_json_path)
    for path, rows, expected in [(ten_error_path, error10, 225), (ten_window_path, window10, 675), (ten_energy_path, energy10, 1575)]:
        audit.check('任10 原CSV行数', len(rows), expected, audit.ref(path))
    for key in ('工程装置HF内部实测', '旧固定TEST温度读取'):
        audit.check('任10 人为数值不是真实实验' + key, ten[key], False, audit.ref(ten_json_path, '/' + key))
    audit.check('任10 隐藏42/58未选择修订原声明', ten['隐藏42和58千不用于选择或修订'], True, audit.ref(ten_json_path, '/隐藏42和58千不用于选择或修订'))
    groups10 = sorted(set((r['方法'], r['LF来源']) for r in error10))
    audit.check('任10 五组比较', len(groups10), 5, audit.ref(ten_error_path))
    labels = {'': '无LF', 'lf_same_physics_coarse': '同物理粗网格', 'lf_contact_mismatch_coarse': '接触热阻失配粗网格'}
    field_header = ['方法', 'LF来源', '九档双材料RMSE（K）', '隐藏两档双材料RMSE（K）', '九档SiC RMSE（K）', '九档Cu RMSE（K）', '九档非探针内部RMSE（K）', '全部非观测窗RMSE（K）']
    field_table = audit.table_rows(REPORTS[9], field_header)
    energy_table = audit.table_rows(REPORTS[9], ['方法', 'LF来源', '每seed绝对余额均值（W）', '315点合并p95（W）', '315点最大值（W）'])
    ten_summary = {}
    for method, lf in groups10:
        group = method + '/' + (lf or '无LF')
        selected_error = [r for r in error10 if (r['方法'], r['LF来源']) == (method, lf)]
        flux = sorted(set(numeric(r, '人为已吸收一维热流_W_m2') for r in selected_error))
        hidden = sorted(set(numeric(r, '人为已吸收一维热流_W_m2') for r in selected_error if r['数值来源折'] == 'hidden_test'))
        audit.check('任10 每组9人为热流档', len(flux), 9, audit.ref(ten_error_path, selection={'方法': method, 'LF来源': lf}))
        audit.check('任10 隐藏热流固定42/58kW/m2', hidden, [42000.0, 58000.0], audit.ref(ten_error_path, selection={'方法': method, 'LF来源': lf, '数值来源折': 'hidden_test'}, unit='瓦/平方米'))
        item = {'方法': method, 'LF来源原字符串': lf, '九档热流_瓦每平方米': flux}
        for label, column, rows, path, only_hidden in [
            ('九档双材料RMSE（K）', '双材料厚度加权RMSE_K', error10, ten_error_path, False),
            ('隐藏两档双材料RMSE（K）', '双材料厚度加权RMSE_K', error10, ten_error_path, True),
            ('九档SiC RMSE（K）', 'SiC厚度加权RMSE_K', error10, ten_error_path, False),
            ('九档Cu RMSE（K）', 'Cu厚度加权RMSE_K', error10, ten_error_path, False),
            ('九档非探针内部RMSE（K）', '非探针附近内部RMSE_K', error10, ten_error_path, False),
            ('全部非观测窗RMSE（K）', '非探针内部厚度加权RMSE_K', window10, ten_window_path, False),
        ]:
            per_seed = []
            for seed in range(5):
                selected = [r for r in rows if (r['方法'], r['LF来源'], int(r['随机种子'])) == (method, lf, seed) and (not only_hidden or r['数值来源折'] == 'hidden_test')]
                expected_count = 2 if only_hidden else (27 if rows is window10 else 9)
                audit.check('任10 每seed宏原行数' + label, len(selected), expected_count, audit.ref(path, selection={'方法': method, 'LF来源': lf, '随机种子': seed, '隐藏限定': only_hidden}))
                per_seed.append(float(np.mean([numeric(r, column) for r in selected])))
            summary = audit.five('任10_' + group + '_' + label, per_seed, audit.ref(path, selection={'方法': method, 'LF来源': lf, '隐藏限定': only_hidden, '列': column, '宏': '每seed自身档/窗均值，再五seed均值和ddof1'}, unit='开尔文误差'))
            item[label] = {'五seed宏原值': per_seed, '均值': summary['mean'], '样本std': summary['std']}
            matching = [(line, row) for line, row in field_table if row['方法'] == method and row['LF来源'] == labels[lf]]
            audit.check('任10 验收表对应行唯一', len(matching), 1, audit.ref(REPORTS[9], lines=[x[0] for x in matching]))
            displayed = '%.6f±%.6f' % (summary['mean'], summary['std'])
            audit.check('任10 原场表逐格六位复核' + label, displayed, matching[0][1][label], audit.ref(REPORTS[9], lines=[matching[0][0]]))
        selected_energy = [r for r in energy10 if (r['方法'], r['LF来源']) == (method, lf)]
        absolute = [abs(numeric(r, '已知入流热预算差_W')) for r in selected_energy]
        audit.check('任10 315原点能源p95合并', len(absolute), 315, audit.ref(ten_energy_path, selection={'方法': method, 'LF来源': lf}))
        seed_energy = [float(np.mean([abs(numeric(r, '已知入流热预算差_W')) for r in selected_energy if int(r['随机种子']) == seed])) for seed in range(5)]
        summary = audit.five('任10_' + group + '_名义能源', seed_energy, audit.ref(ten_energy_path, selection={'方法': method, 'LF来源': lf, '列': '已知入流热预算差_W', '宏': '先逐原行abs，再每seed63点平均，再五seedddof1'}, unit='瓦'))
        item['名义能源'] = {'五seed宏原值': seed_energy, '均值': summary['mean'], '样本std': summary['std'], '315点合并p95': float(np.percentile(absolute, 95)), '315点最大值': max(absolute)}
        matching_energy = [(line, row) for line, row in energy_table if row['方法'] == method and row['LF来源'] == labels[lf]]
        audit.check('任10 原能源均值±std逐格', '%.6f±%.6f' % (summary['mean'], summary['std']), matching_energy[0][1]['每seed绝对余额均值（W）'], audit.ref(REPORTS[9], lines=[matching_energy[0][0]]))
        for key, value in [('315点合并p95（W）', item['名义能源']['315点合并p95']), ('315点最大值（W）', item['名义能源']['315点最大值'])]:
            audit.check('任10 原能源表逐格' + key, '%.6f' % value, matching_energy[0][1][key], audit.ref(REPORTS[9], lines=[matching_energy[0][0]]))
        ten_summary[group] = item
    comparisons10 = {}
    for lf in ('lf_same_physics_coarse', 'lf_contact_mismatch_coarse'):
        f2 = ten_summary['F2/' + lf]['隐藏两档双材料RMSE（K）']['五seed宏原值']
        f3 = ten_summary['F3/' + lf]['隐藏两档双材料RMSE（K）']['五seed宏原值']
        differences = [b - a for a, b in zip(f2, f3)]
        summary = audit.five('任10_' + lf + '_隐藏配对F3减F2', differences, audit.ref(ten_error_path, selection={'同一LF来源': lf, '同seed配对': True, '隐藏限定': True}, unit='开尔文误差'))
        comparisons10[lf] = {'五seed配对差': differences, 'F3减F2均值_K': summary['mean'], '样本std_K': summary['std'], 'F3较优种子数': sum(x < 0 for x in differences), 'F3相对F2降低百分比': (float(np.mean(f2)) - float(np.mean(f3))) / float(np.mean(f2)) * 100}
    audit.check('任10 同物理F3仅1/5优', comparisons10['lf_same_physics_coarse']['F3较优种子数'], 1, audit.ref(REPORTS[9], lines=[31]))
    audit.check('任10 失配F3仅4/5优', comparisons10['lf_contact_mismatch_coarse']['F3较优种子数'], 4, audit.ref(REPORTS[9], lines=[32]))
    audit.check('任10 同物理隐藏差1.64848%方向', -comparisons10['lf_same_physics_coarse']['F3相对F2降低百分比'], 1.6484839916908534, audit.ref(STATUS, '/任务/任-10/效果结论'), 1e-10)
    audit.check('任10 失配隐藏改善8.64125%方向', comparisons10['lf_contact_mismatch_coarse']['F3相对F2降低百分比'], 8.6412459179485, audit.ref(STATUS, '/任务/任-10/效果结论'), 1e-10)
    index_path = T10 + '/原始后验全部原件SHA256.json'
    for name, expected in audit.js(index_path).items():
        path = T10 + '/' + name
        audit.read(path)
        audit.check('任10 后验六普通原件索引', audit.original[path]['SHA256'], expected, audit.ref(index_path, '/' + name))

    stable_after = {}
    changed = []
    for path, before in list(audit.original.items()):
        data = (ROOT / path).read_bytes()
        stable_after[path] = sha(data)
        if path not in (PLAN, STATUS) and stable_after[path] != before['SHA256']:
            changed.append(path)
    status_after = json.loads((ROOT / STATUS).read_bytes(), object_pairs_hook=unique_pairs)
    subset_after = {key: status_after['任务'][key] for key in task_subset}
    audit.check('任01..10状态语义短窗口不变', sha(canonical(subset_after)), sha(canonical(task_subset)), audit.ref(STATUS, '/任务', selection={'仅取已有任务': list(task_subset)}))
    audit.check('所有静态引用原件前后SHA不变', changed, [], {'引用原件数': len(audit.original), '动态主计划与总状态独立记两次SHA': True})
    static_map = {path: metadata['SHA256'] for path, metadata in audit.original.items() if path not in (PLAN, STATUS)}
    failed = [x for x in audit.checks if not x['通过']]
    rows_out = [
        ('任01', '启动/单卡恢复已验收', '1轮S %.6f℃；无收益' % one['best_validation_selection_score_c'], '不采用短跑；多卡恢复未验收'),
        ('任02', '单种子排程验收／回退', 'S0/S1 %.6f/%.6f℃；S1改善%.3f%%' % (*t2scores, improvement2), 'p95恶化，保留S0；不是五seed策略验证'),
        ('任03', 'LF拟合证据／HF替换回退', 'LF空间体积平均恶化%.4f%%' % -change3, 'HF顶部/p95护栏失败；LF收益不是HF收益'),
        ('任04', '单种子联合方向受限保留', '冻结/联合best %.6f/%.6f℃；改善%.4f%%' % (*t4scores, (t4scores[0] - t4scores[1]) / t4scores[0] * 100), '最佳能源均值/p95更差，末态退化；不换B0'),
        ('任05', '顺序条件判定有限完成', '甲乙丙未触发，新增训练候选0个', '原FEM/分项定位/独立标定缺；不是物理修复'),
        ('任06', '三臂600轮及六能源验收', '三臂best均0轮S %.6f℃' % t4scores[1], 'E1/E2无收益，不采用；仅E0进入07'),
        ('任07', '五seed观测重复性受限冻结', 'S %.6f±%.6f℃；历史B0 %.6f±%.6f℃' % (audit.statistics['任07_本轮合法macro_v1']['均值'], audit.statistics['任07_本轮合法macro_v1']['样本标准差_ddof1'], audit.statistics['任07_历史B0合法macro_v1']['均值'], audit.statistics['任07_历史B0合法macro_v1']['样本标准差_ddof1']), '历史每seed更优；乙非物理合格；径向720点已版本化补齐'),
        ('任08', 'F2五seed/15能源受限完成', 'F2 S %.6f±%.6f℃；best能源 %.3f±%.3fW' % (summary8['mean'], summary8['std'], audit.statistics['任08_best_能源']['均值'], audit.statistics['任08_best_能源']['样本标准差_ddof1']), '原装置F1独立物理与严格F3贡献未齐；能源不合格'),
        ('任09', 'F3主序列数据效率受限完成', 'HF3/6/9 best %.6f/%.6f/%.6f℃' % tuple(audit.statistics['任09_HF%d_bestS' % count]['均值'] for count in (3, 6, 9)), 'HF9比HF6差0.123782℃；所需数量/节省null，无子集方差'),
        ('任10', '人为一维板25模型九档受限完成', '隐藏同物理F3较F2差1.6485%；失配改善8.6412%', '仅数值板；非原RZ内场/任12实测；无安全W阈值，F4缺'),
    ]
    evidence = {
        '本件类型': '任01至10既有结果纯只读独立描述统计核对',
        '执行开始_东八区': begin, '执行结束_东八区': datetime.now(TZ).isoformat(),
        '执行PID': os.getpid(), '父PID': os.getppid(), 'cwd': str(Path.cwd()),
        'Python': sys.executable, 'argv': sys.argv, 'numpy版本': np.__version__,
        '完整环境约束': {key: os.environ.get(key) for key in ('PYTHONDONTWRITEBYTECODE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD', 'TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'CUDA_CACHE_PATH', 'MPLCONFIGDIR', 'TORCH_EXTENSIONS_DIR', 'TORCHINDUCTOR_CACHE_DIR')},
        '技能复用': {'已有describe_column精确路径': str(skill), '技能统计脚本SHA256': sha(skill.read_bytes()), '仅描述统计不调用检验或推荐函数': True},
        '核对范围不是新科学实验': {'GPU或CUDA调用': False, '真实模型PT读取': False, '温度PARQ读取': False, '旧固定TEST标签读取': False, '新训练': False, '新模型推理或能源度量': False, '开始任12或13': False, '更换B0或发布': False, '任11完成声明': False, '总goal完成声明': False},
        '全部引用原件读取时身份': audit.original,
        '全部引用原件结束时SHA256': stable_after,
        '静态原件SHA映射规范JSON_SHA256': sha(canonical(static_map)),
        '静态原件数': len(static_map), '任01至10状态语义SHA256_前': sha(canonical(task_subset)),
        '任01至10状态语义SHA256_后': sha(canonical(subset_after)),
        '动态主计划总状态wholeSHA允许因任11合法进度变化': True,
        '十任务状态原始快照': task_subset, '十任务简洁结果表': [dict(zip(('任务', '有限完成口径', '已有真实结果', '采纳结论及剩余条件'), row)) for row in rows_out],
        '单种子和条件事实': audit.claims, '已有五种子独立重算': audit.statistics,
        '任07十能源原五值重算': energy7,
        '任10九档与隐藏窗能源重算': ten_summary, '任10同来源同seed配对描述': comparisons10,
        '四轮描述统计代码复核': [
            {'轮': 1, '核对': 'numpy均值/ddof1与statistics.mean/stdev独立算法交叉；不选择模型，不做事后p/CI', '范围': len(audit.statistics)},
            {'轮': 2, '核对': '中文键/BOM CSV/单位/隐藏折/空窗；任08逐seedp95均值和任10合并315点p95不同，不补缺值'},
            {'轮': 3, '核对': 'Task07/08/09现存原5与Task10身份级原5；单seed无样本std；字段逐原件比较并保留所有不利点'},
            {'轮': 4, '核对': '报告/总状态/原CSV一致性与有限验收；任07五seed与任08 F2完成不等于任02/03/04各策略五seed同起点因果对照已补；任07 F2仍无成绩属于历史条目；真F1/FEM/内部实验仍缺'},
        ],
        '所有逐项核对': audit.checks, '核对总数': len(audit.checks), '实际失败核对数': len(failed), '实际失败逐项': failed,
    }
    with open(args.机器输出, 'x', encoding='utf-8') as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    with open(args.结果表, 'x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('任务', '有限完成口径', '已有真实结果', '采纳结论及剩余条件'))
        writer.writerows(rows_out)
    print(json.dumps({'核对总数': len(audit.checks), '失败数': len(failed), '已有五种子描述统计组数': len(audit.statistics),
                      '静态引用普通原件数': len(static_map), '静态来源映射SHA256': evidence['静态原件SHA映射规范JSON_SHA256'],
                      '机器输出': args.机器输出, '结果表': args.结果表, '非真实CUDA或新科学实验': True}, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
