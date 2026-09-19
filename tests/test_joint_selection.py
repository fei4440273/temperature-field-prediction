"""选模评分仅使用合成指标，不读取真实测试数据。"""
import copy
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def selection_score(detail, config=None):
    assert importlib.util.find_spec('joint_temperature_selection') is not None, '独立选模函数尚未实现'
    return importlib.import_module('joint_temperature_selection').selection_score(detail, config)


def detail(top=(1., 2., 3.), hot=(2., 4., 6.), cold=(3., 3., 3.), rise=(.5, 1.5)):
    result = {}
    for name, values in (('顶部', top), ('热端', hot), ('冷端', cold)):
        result[name] = {
            '汇总': {'均方根误差_℃': float(np.mean(values))},
            '逐功率': [{'功率_W': float(i + 1), '均方根误差_℃': value}
                      for i, value in enumerate(values)],
        }
    for name, value in zip(('热端', '冷端'), rise):
        result[name]['汇总']['温升均方根误差_℃'] = value
    return result


def config(fraction=.2, weights=None):
    return {'weights': weights if weights is not None else {'顶部': .5, '热端': .25, '冷端': .25},
            'worst_power_fraction': fraction}


@pytest.mark.parametrize('values', [(0., 0., 0., 0., 0.),
                                    (.1, .2, .3, .4, .5),
                                    (9.490083940328061, 4.105043676735448, 1.379471884684954,
                                     2.463486530065253, 1.3284450121432971)])
def test_none_config_preserves_legacy_formula_exactly(values):
    top, hot, cold, hot_rise, cold_rise = values
    metrics = detail((top,), (hot,), (cold,), (hot_rise, cold_rise))
    absolute = np.mean([hot, cold])
    rise = np.mean([hot_rise, cold_rise])
    expected = float((top + .2 * absolute + rise) / 2.2)
    score = selection_score(metrics)
    assert type(score) is float
    assert score == expected


def test_new_score_combines_macro_and_worst_power():
    assert selection_score(detail(), config()) == pytest.approx(2.95)


def test_constant_sensor_offset_is_penalized_without_temperature_rise_error():
    baseline = detail((0.,), (0.,), (0.,), (0., 0.))
    offset = detail((0.,), (5.,), (5.,), (0., 0.))
    assert selection_score(baseline, config()) == 0.
    assert selection_score(offset, config()) == 2.5
    assert selection_score(offset) == pytest.approx(.2 * 5. / 2.2)


@pytest.mark.parametrize('name, expected', [('顶部', 2.), ('热端', 1.), ('冷端', 1.)])
def test_each_absolute_temperature_modality_has_positive_weight(name, expected):
    metrics = detail((0.,), (0.,), (0.,))
    metrics[name]['汇总']['均方根误差_℃'] = 4.
    metrics[name]['逐功率'][0]['均方根误差_℃'] = 4.
    assert selection_score(metrics, config()) == expected


def test_worst_power_is_not_completely_diluted_by_macro_average():
    metrics = detail((0., 0., 30.), (0.,), (0.,))
    assert selection_score(metrics, config(0.)) == 5.
    assert selection_score(metrics, config(.2)) == 7.
    assert selection_score(metrics, config(1.)) == 15.


def test_weight_scale_does_not_change_weighted_mean():
    assert selection_score(detail(), config(weights={'顶部': 5., '热端': 2.5, '冷端': 2.5})) == pytest.approx(2.95)


def test_finite_large_weights_and_rmse_do_not_overflow_the_weighted_mean():
    metrics = detail((1e308,), (1e308,), (1e308,))
    settings = config(weights={'顶部': 1e308, '热端': 1e308, '冷端': 1e308})
    assert selection_score(metrics, settings) == pytest.approx(1e308)


def test_zero_modality_weights_are_allowed_with_positive_total():
    assert selection_score(detail(), config(weights={'顶部': 0., '热端': 2., '冷端': 0.})) == pytest.approx(4.4)


def test_new_score_does_not_require_rise_metrics():
    metrics = detail()
    for name in ('热端', '冷端'):
        del metrics[name]['汇总']['温升均方根误差_℃']
    assert selection_score(metrics, config()) == pytest.approx(2.95)


def test_legacy_score_does_not_require_per_power_metrics():
    metrics = detail()
    for value in metrics.values():
        del value['逐功率']
    assert selection_score(metrics) == float((2. + .2 * 3.5 + 1.) / 2.2)


def test_scoring_does_not_mutate_inputs():
    metrics, settings = detail(), config()
    before = copy.deepcopy((metrics, settings))
    selection_score(metrics, settings)
    assert (metrics, settings) == before


@pytest.mark.parametrize('settings', [
    {},
    {'weights': {'顶部': .5, '热端': .25, '冷端': .25}},
    {'worst_power_fraction': .2},
    {'weights': {'顶部': .5, '热端': .25, '冷端': .25}, 'worst_power_fraction': .2, 'typo': 1.},
    config(weights={'顶部': .5, '热端': .25}),
    config(weights={'顶部': .5, '热端': .25, '冷端': .25, '温升': 1.}),
    config(weights={'顶部': 0., '热端': 0., '冷端': 0.}),
    config(weights={'顶部': -.5, '热端': .25, '冷端': .25}),
    config(weights={'顶部': float('nan'), '热端': .25, '冷端': .25}),
    config(weights={'顶部': .5, '热端': float('inf'), '冷端': .25}),
    config(weights={'顶部': True, '热端': .25, '冷端': .25}),
    config(weights={'顶部': '.5', '热端': .25, '冷端': .25}),
    config(-.01),
    config(1.01),
    config(float('nan')),
    config(float('inf')),
    config(True),
    config('.2'),
    config(weights=[]),
    [],
])
def test_invalid_selection_config_is_rejected(settings):
    with pytest.raises(ValueError):
        selection_score(detail(), settings)


@pytest.mark.parametrize('location, value', [('summary', float('nan')),
                                          ('summary', float('inf')),
                                          ('summary', -1.),
                                          ('summary', True),
                                          ('power', float('nan')),
                                          ('power', float('inf')),
                                          ('power', -1.),
                                          ('power', '3.')])
def test_invalid_absolute_rmse_is_rejected(location, value):
    metrics = detail()
    target = metrics['热端']['汇总'] if location == 'summary' else metrics['热端']['逐功率'][0]
    target['均方根误差_℃'] = value
    with pytest.raises(ValueError):
        selection_score(metrics, config())


@pytest.mark.parametrize('missing', ['modality', 'summary', 'summary_rmse', 'powers', 'power_rmse'])
def test_missing_new_score_data_is_rejected(missing):
    metrics = detail()
    if missing == 'modality':
        del metrics['冷端']
    elif missing == 'summary':
        del metrics['冷端']['汇总']
    elif missing == 'summary_rmse':
        del metrics['冷端']['汇总']['均方根误差_℃']
    elif missing == 'powers':
        del metrics['冷端']['逐功率']
    else:
        del metrics['冷端']['逐功率'][0]['均方根误差_℃']
    with pytest.raises(ValueError):
        selection_score(metrics, config())


@pytest.mark.parametrize('powers', [[], None, {}, ['not a metric row']])
def test_missing_or_invalid_per_power_data_is_rejected(powers):
    metrics = detail()
    metrics['顶部']['逐功率'] = powers
    with pytest.raises(ValueError):
        selection_score(metrics, config())


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.])
def test_legacy_nonfinite_or_negative_rise_rmse_is_rejected(value):
    metrics = detail()
    metrics['冷端']['汇总']['温升均方根误差_℃'] = value
    with pytest.raises(ValueError):
        selection_score(metrics)


def test_legacy_missing_rise_data_is_rejected():
    metrics = detail()
    del metrics['冷端']['汇总']['温升均方根误差_℃']
    with pytest.raises(ValueError):
        selection_score(metrics)


@pytest.mark.parametrize('metrics', [None, [], {'顶部': None, '热端': None, '冷端': None}])
def test_invalid_detail_structure_is_rejected(metrics):
    with pytest.raises(ValueError):
        selection_score(metrics, config())
