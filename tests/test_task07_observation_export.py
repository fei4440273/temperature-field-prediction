"""任07五种子合法HF验证观测中文导出，不读旧TEST温度。"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch


def _entry():
    script = Path(__file__).resolve().parents[1] / "scripts/28_export_task07_observations.py"
    assert script.is_file(), "任务07合法观测导出入口尚不存在"
    spec = importlib.util.spec_from_file_location("task07_observation_export", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_five_seeds_refuses_output_before_reading_temperature(tmp_path) -> None:
    entry = _entry()
    output = tmp_path / "缺五份必须拒绝"
    with pytest.raises(ValueError, match="五种子|0.*4|缺失"):
        entry.export_task07_observations(
            {0: tmp_path / "只有种子0"}, output,
            registry_sha256="f92abf2cc2f97aa6bddcd16768707ee581470bbe0f9d3eb191598dbf7bbc0dba",
        )
    assert not output.exists()


def test_real_v4_validation_record_localizes_without_modifying_original_rmse() -> None:
    entry = _entry()
    from sic_cu.config import load_yaml
    from sic_cu.eval.development_v4 import evaluate_hf_observations, load_existing_baseline
    from sic_cu.train.task07_source import validate_task07_sources

    source = validate_task07_sources()[0]
    _, model, _, _, _, _ = load_existing_baseline(
        source.hf_checkpoint_path.parent.parent, 0, torch.device("cpu"),
    )
    cfg = load_yaml("configs/optimization_v4.yaml")["diagnostics"]
    geometry = load_yaml("configs/geometry.yaml")
    power, windows, radii, _ = evaluate_hf_observations(
        model, 0, "validation", torch.device("cpu"), cfg,
        float(geometry["embedding"]["copper_bottom_z_m"]),
    )
    assert len(power) == 3 * 3
    assert {row["split"] for row in power + windows + radii} == {"validation"}
    assert {row["power_w"] for row in power} == {115.2, 403.0, 630.5}
    assert {row["modality"] for row in power} == {"Top", "Hot", "Cold"}
    original = power[0]
    localized = entry.localize_observation_record(original, kind="power")
    assert localized["种子"] == 0
    assert localized["数据划分"] == "高保真合法验证"
    assert localized["功率_W"] == original["power_w"]
    assert localized["模态"] in {"顶部红外", "热端环温", "冷端环温"}
    assert localized["RMSE_摄氏度"] == original["rmse_c"]
    assert localized["MAE_摄氏度"] == original["mae_c"]
    assert localized["绝对误差95分位_摄氏度"] == original["p95_abs_error_c"]
    assert localized["峰值偏差_摄氏度"] == original["peak_error_c"]
    assert "test" not in str(localized).lower()
    assert entry.localize_observation_record(windows[0], kind="time")["时间窗"]
    assert entry.localize_observation_record(radii[0], kind="radius")["径向窗"]


def test_five_seed_summary_uses_sample_standard_deviation_and_keeps_worst() -> None:
    entry = _entry()
    sample = {seed: float(seed + 1) for seed in range(5)}
    result = entry.summarize_five_seed_scores(sample)
    assert result["五种子均值_摄氏度"] == 3.0
    assert math.isclose(result["五种子样本标准差_摄氏度"], math.sqrt(2.5))
    assert result["最坏种子"] == 4
    assert result["最坏选分_摄氏度"] == 5.0
    assert result["逐种子选分_摄氏度"] == {str(seed): score for seed, score in sample.items()}
    assert np.isfinite(result["五种子样本标准差_摄氏度"])


def test_empty_window_exposes_null_and_chinese_reason_without_fake_label() -> None:
    entry = _entry()
    from sic_cu.eval.development_v4 import _empty_metrics

    raw = {
        "seed": 0, "split": "validation", "power_w": 115.2,
        "modality": "Top", "window": "time_0_30_s",
        "lower_s": 0.0, "upper_s": 30.0,
        "lower_closed": True, "upper_closed": True,
        "sample_count": 0,
        **_empty_metrics("no_observations_in_preregistered_window"),
    }
    localized = entry.localize_observation_record(raw, kind="time")
    assert localized["实际观测点数"] == 0
    assert localized["RMSE_摄氏度"] is None
    assert localized["空窗原因"] == "预登记时间窗内无观测，不外推真值"
