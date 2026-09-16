"""Task 11: independent HF-only observation Ridge is not F1 PINN."""

from __future__ import annotations

import importlib

import numpy as np
import polars as pl
import pytest

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import build_power_splits


def _entry():
    return importlib.import_module("sic_cu.eval.task11_hf_observation_ridge")


def _frames(split="train"):
    powers = sorted(build_power_splits().hf_train if split == "train"
                    else build_power_splits().hf_validation)
    top_rows, sensor_rows = [], []
    for power in powers:
        for second in (1.0, 30.0):
            top_rows.append({"split": split, "power_w": power, "r_m": 0.0,
                             "time_s": second, "temperature_mean_k": 295.15 + power / 500,
                             "frame_weight": 1.0})
            for name, radius in (("hot", 0.028), ("cold", 0.0415)):
                sensor_rows.append({"split": split, "power_w": power,
                                    "sensor_type": name, "r_m": radius,
                                    "time_s": second, "temperature_k": 295.15 + power / 650})
    return pl.DataFrame(top_rows), pl.DataFrame(sensor_rows)


def test_hf_only_fit_rejects_missing_hot_cold_or_development_temperature():
    entry = _entry()
    top, sensors = _frames()
    entry.validate_training_observations(top, sensors)
    no_hot = sensors.filter(pl.col("sensor_type") != "hot")
    with pytest.raises(ValueError, match="Hot|hot|热端"):
        entry.validate_training_observations(top, no_hot)
    validation, _ = _frames("validation")
    with pytest.raises(ValueError, match="训练|train|验证"):
        entry.validate_training_observations(validation, sensors)
    with pytest.raises(ValueError, match="训练|train|验证"):
        entry.validate_training_observations(top, _frames("validation")[1])


def test_hf_only_fixed_fit_has_no_lf_component_and_all_three_validation_modalities():
    entry = _entry()
    top, sensors = _frames()
    hf_regressor, train_stats = entry.fit_hf_observation_ridge(top, sensors)
    assert train_stats["顶部HF训练功率数"] == 12
    assert train_stats["顶部HF训练行"] == 24
    assert train_stats["两环HF训练行"] == 48
    assert train_stats["LF训练仿真标签消费"] == 0
    assert train_stats["物理PDE/边界/界面训练梯度"] == 0
    assert hf_regressor["degree"] == 2 and hf_regressor["alpha"] == 0.01
    validation, rings = _frames("validation")
    rows, summary = entry.evaluate_hf_observation_ridge(hf_regressor, validation, rings)
    assert len(rows) == 9
    assert {row["modality"] for row in rows} == {"Top", "Hot", "Cold"}
    assert {row["power_w"] for row in rows} == build_power_splits().hf_validation
    assert np.isfinite(summary["合法HF三模态宏选分_摄氏度"])
    with pytest.raises(ValueError, match="验证|validation"):
        entry.evaluate_hf_observation_ridge(hf_regressor, top, sensors)


def test_hf_only_new_output_must_remain_in_project_and_refuse_external_symlink(tmp_path):
    entry = _entry()
    assert entry.project_output(tmp_path / "HF独立工程观测") == (tmp_path / "HF独立工程观测").resolve()
    outside = PROJECT_ROOT.parent / f"不得项目外写任11Ridge_{tmp_path.name}"
    with pytest.raises(ValueError, match="项目内"):
        entry.project_output(outside)
    bridge = tmp_path / "软链"
    bridge.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="项目内"):
        entry.project_output(bridge / "新结果")
    assert not outside.exists()


def test_registry_or_model_mismatch_refuses_before_reading_any_temperature(tmp_path, monkeypatch):
    entry = _entry()
    registered = tmp_path / "Ridge训练温度登记"
    train = tmp_path / "绝不污染真HF训练"
    validation = tmp_path / "未锁模型不得看验证"
    identity = {"HF训练顶部IR字节SHA256": "一份可核来源", "alpha": 0.01}
    monkeypatch.setattr(entry, "_source_identity", lambda: dict(identity))
    accessed = []
    monkeypatch.setattr(entry, "load_processed_ir_observations",
                        lambda split: accessed.append(("IR", split)))
    monkeypatch.setattr(entry, "load_canonical_sensor_observations",
                        lambda *, split: accessed.append(("环温", split)))
    source = entry.register_hf_ridge_sources(registered)
    path = registered / entry.REGISTRATION_FILE
    assert source["来源文件SHA256"] == sha256_file(path) and not accessed
    with pytest.raises(ValueError, match="SHA|字节|来源"):
        entry.train_registered_hf_ridge(train, path, "0" * 64)
    assert not accessed and not train.exists()
    identity["HF训练顶部IR字节SHA256"] = "原件漂移"
    with pytest.raises(ValueError, match="漂移|来源"):
        entry.train_registered_hf_ridge(train, path, source["来源文件SHA256"])
    assert not accessed and not train.exists()
    identity["HF训练顶部IR字节SHA256"] = "一份可核来源"
    with pytest.raises(ValueError, match="锁|系数|模型"):
        entry.evaluate_registered_hf_ridge(
            validation, path, source["来源文件SHA256"],
            tmp_path / "假训练模型目录" / entry.MODEL_FILE, "0" * 64)
    assert not accessed and not validation.exists()
