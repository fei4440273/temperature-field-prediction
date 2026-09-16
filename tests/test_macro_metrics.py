from __future__ import annotations

import numpy as np
import json
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.eval.metrics import (
    configured_radial_mask,
    macro_metric_summary,
    macro_v1_selection_score,
    weighted_metrics,
)
from sic_cu.eval.guardrails import (
    task03_volume_guardrails, task03_hf_guardrails, validate_task03_hf_budget,
)
from sic_cu.train.multifidelity import _weighted_validation


class _PowerOffset(nn.Module):
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return coordinates[:, 3:4]


def test_macro_rmse_does_not_weight_a_long_condition_more() -> None:
    records = [
        {"rmse_c": 1.0, "mae_c": 1.0, "mean_error_c": 1.0},
        {"rmse_c": 9.0, "mae_c": 9.0, "mean_error_c": 9.0},
    ]
    result = macro_metric_summary(records)

    assert result["rmse_c"] == pytest.approx(5.0)
    assert result["equal_power_mse_rmse_c"] == pytest.approx(np.sqrt(41.0))


def test_weighted_metrics_are_temperature_offset_invariant() -> None:
    target = np.array([300.0, 310.0, 320.0])
    prediction = np.array([301.0, 308.0, 324.0])
    weights = np.array([0.2, 0.3, 0.5])

    kelvin = weighted_metrics(target, prediction, weights)
    celsius = weighted_metrics(target - 273.15, prediction - 273.15, weights)

    assert kelvin == pytest.approx(celsius)


def test_macro_validation_is_batch_size_invariant() -> None:
    power = torch.tensor([1.0] * 2 + [9.0] * 8)
    coordinates = torch.zeros((10, 5))
    coordinates[:, 3] = power
    target = torch.zeros((10, 1))
    weights = torch.ones((10, 1))
    dataset = TensorDataset(coordinates, target, weights)
    model = _PowerOffset()

    small = _weighted_validation(
        model, DataLoader(dataset, batch_size=3), torch.device("cpu")
    )
    large = _weighted_validation(
        model, DataLoader(dataset, batch_size=10), torch.device("cpu")
    )

    assert torch.equal(small, large)
    assert small[0].item() == pytest.approx(5.0)
    assert small[2].item() == pytest.approx(np.sqrt(41.0))


def test_macro_v1_selection_formula_is_frozen() -> None:
    assert macro_v1_selection_score(2.0, 5.0, 3.0) == pytest.approx(6.0 / 2.2)


def test_preregistered_radial_windows_partition_exact_eight_and_seventeen_mm() -> None:
    radii = np.array([0.0, 8.0, 8.001, 17.0, 17.001, 25.0, 25.001])
    configured = (
        {"lower": 0.0, "upper": 8.0, "lower_closed": True},
        {"lower": 8.0, "upper": 17.0, "lower_closed": False},
        {"lower": 17.0, "upper": 25.0, "lower_closed": False},
    )
    masks = [configured_radial_mask(radii, window) for window in configured]
    assert [np.flatnonzero(mask).tolist() for mask in masks] == [[0, 1], [2, 3], [4, 5]]
    assert np.sum(masks, axis=0).tolist() == [1, 1, 1, 1, 1, 1, 0]


def test_task03_lf_guardrail_rejects_better_sampled_score_when_volume_is_worse() -> None:
    control = {"Cu": {"node": 0.17409, "volume": 0.1214},
               "SiC": {"node": 0.11604, "volume": 0.10696}}
    spatial = {"Cu": {"node": 0.17094, "volume": 0.12346},
               "SiC": {"node": 0.11962, "volume": 0.11007}}
    result = task03_volume_guardrails(control, spatial)
    assert result["空间LF体积平均改善率_百分比"] < 0
    assert result["两材料体积改善至少5百分比"] is False
    assert result["各材料节点及体积恶化不超5百分比"] is True
    assert result["LF空间机制满足采用门槛"] is False


def test_task03_hf_guardrail_rejects_top_regression_despite_selection_improvement() -> None:
    result = task03_hf_guardrails(
        old_score=2.263391, new_score=2.248172,
        old_modalities={"top": 3.976143, "hot": 1.185243, "cold": 0.588335},
        new_modalities={"top": 4.315577, "hot": 0.985874, "cold": 0.422920},
        old_energy={"mean_w": 929.676929, "p95_w": 1884.784399},
        new_energy={"mean_w": 922.723459, "p95_w": 2331.474214},
    )
    assert result["选分方向改善"] is True
    assert result["顶部允许增加_摄氏度"] == pytest.approx(3.976143 * 0.05)
    assert result["单模态护栏全部满足"] is False
    assert result["工程能量改善至少50百分比"] is False
    assert result["工程能量95分位不恶化"] is False
    assert result["HF组合值得采用"] is False


def test_task03_hf_budget_validates_real_300_epoch_control_and_rejects_short_arm() -> None:
    root = PROJECT_ROOT / (
        "研究记录/任务03_低保真精度修复/"
        "任务03_HF旧LF同源_种子0_20260915T184956+0800"
    )
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    logs = [json.loads(line) for line in (root / "training.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()]
    locked = load_yaml("研究记录/任务03_低保真精度修复/有效运行配置.yaml")
    consumption = validate_task03_hf_budget(metrics, logs, locked)
    assert consumption["观测优化步"] == 4500
    assert consumption["物理优化步"] == 300
    assert consumption["物理配点数"] == 76800
    short = json.loads(json.dumps(metrics))
    short["configuration"]["correction_epochs"] = 100
    with pytest.raises(ValueError, match="HF.*budget|HF.*预算"):
        validate_task03_hf_budget(short, logs, locked)
    with pytest.raises(ValueError, match="HF.*budget|HF.*预算"):
        validate_task03_hf_budget(metrics, logs[:-1], locked)
