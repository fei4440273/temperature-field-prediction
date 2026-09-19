"""联合训练的验证选模评分，不读取数据或模型。"""
import math
from collections.abc import Mapping
from numbers import Real


_MODALITIES = ('顶部', '热端', '冷端')
_ABSOLUTE_RMSE = '均方根误差_℃'
_RISE_RMSE = '温升均方根误差_℃'


def _mapping(value, label):
    if not isinstance(value, Mapping):
        raise ValueError(f'{label}必须是字典。')
    return value


def _nonnegative_finite(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{label}必须是有限非负数。')
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f'{label}必须是有限非负数。') from None
    if not math.isfinite(result) or result < 0.:
        raise ValueError(f'{label}必须是有限非负数。')
    return result


def _summary_rmse(detail, name, metric):
    if name not in detail:
        raise ValueError(f'缺少{name}验证指标。')
    modality = _mapping(detail[name], f'{name}指标')
    summary = _mapping(modality.get('汇总'), f'{name}汇总指标')
    if metric not in summary:
        raise ValueError(f'缺少{name}的{metric}。')
    return _nonnegative_finite(summary[metric], f'{name}的{metric}')


def selection_score(detail, config=None):
    """返回验证评分：无配置沿用旧公式；新配置兼顾绝对温度均值与最差功率。

    新配置需提供三类 weights 和 [0, 1] 内的 worst_power_fraction。
    每类先混合汇总 RMSE 与逐功率最大 RMSE，再按权重求均值；不使用温升。
    """
    detail = _mapping(detail, '验证分项')
    absolute = {name: _summary_rmse(detail, name, _ABSOLUTE_RMSE) for name in _MODALITIES}
    if config is None:
        sensor_absolute = (absolute['热端'] + absolute['冷端']) / 2.
        sensor_rise = (_summary_rmse(detail, '热端', _RISE_RMSE)
                       + _summary_rmse(detail, '冷端', _RISE_RMSE)) / 2.
        score = (absolute['顶部'] + .2 * sensor_absolute + sensor_rise) / 2.2
    else:
        config = _mapping(config, '选模配置')
        if set(config) != {'weights', 'worst_power_fraction'}:
            raise ValueError('选模配置仅且必须包含 weights 和 worst_power_fraction。')
        weights = _mapping(config['weights'], '选模权重')
        if set(weights) != set(_MODALITIES):
            raise ValueError('选模权重仅且必须包含顶部、热端、冷端。')
        weights = {name: _nonnegative_finite(weights[name], f'{name}选模权重') for name in _MODALITIES}
        largest_weight = max(weights.values())
        if largest_weight == 0.:
            raise ValueError('选模权重总和必须大于零。')
        fraction = _nonnegative_finite(config['worst_power_fraction'], '最差功率混合比例')
        if fraction > 1.:
            raise ValueError('最差功率混合比例必须在 [0, 1] 内。')
        modality_scores = {}
        for name in _MODALITIES:
            rows = detail[name].get('逐功率')
            if not isinstance(rows, (list, tuple)) or not rows:
                raise ValueError(f'{name}缺少非空逐功率指标。')
            power_errors = []
            for row in rows:
                row = _mapping(row, f'{name}逐功率指标')
                if _ABSOLUTE_RMSE not in row:
                    raise ValueError(f'{name}逐功率指标缺少{_ABSOLUTE_RMSE}。')
                power_errors.append(_nonnegative_finite(row[_ABSOLUTE_RMSE], f'{name}逐功率 RMSE'))
            modality_scores[name] = (1. - fraction) * absolute[name] + fraction * max(power_errors)
        normalized = {name: weight / largest_weight for name, weight in weights.items()}
        total_weight = math.fsum(normalized.values())
        score = math.fsum((normalized[name] / total_weight) * modality_scores[name] for name in _MODALITIES)
    if not math.isfinite(score):
        raise ValueError('综合选模评分必须有限。')
    return float(score)
