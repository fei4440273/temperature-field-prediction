"""Task 11: LF-train-only FEM interpolation is a limited HF Top comparator."""

from __future__ import annotations

import importlib
import json

import numpy as np
import polars as pl
import pytest

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import build_power_splits


def _entry():
    return importlib.import_module("sic_cu.eval.task11_fem_top")


def _observations():
    powers = (115.2, 403.0, 630.5)
    return pl.DataFrame({
        "split": ["validation"] * 6,
        "power_w": np.repeat(np.asarray(powers, dtype=np.float32), 2),
        "time_s": np.tile(np.asarray([0.0, 30.0], dtype=np.float32), 3),
        "r_m": np.zeros(6, dtype=np.float32),
        "temperature_mean_k": np.asarray([295.15, 296.15] * 3, dtype=np.float32),
        "frame_weight": np.ones(6, dtype=np.float32),
    })


def test_fem_top_refuses_hf_validation_or_lf_validation_as_fem_support():
    entry = _entry()
    splits = build_power_splits()
    allowed = tuple(sorted(splits.simulation_train))
    assert len(allowed) == 60
    entry.require_lf_train_only(allowed)
    with pytest.raises(ValueError, match="LF|训练|支点"):
        entry.require_lf_train_only(allowed + (115.2,))
    with pytest.raises(ValueError, match="LF|训练|支点"):
        entry.require_lf_train_only(allowed + (90.0,))
    with pytest.raises(ValueError, match="LF|训练|支点"):
        entry.require_lf_train_only(allowed[:-1])


def test_limited_hf_top_metrics_keep_power_count_and_reject_non_validation(monkeypatch):
    entry = _entry()
    frame = _observations()
    used = []

    def fake_surface(group, supports):
        entry.require_lf_train_only(supports)
        used.append(round(float(group["power_w"][0]), 4))
        return group["temperature_mean_k"].to_numpy() - 1.0

    monkeypatch.setattr(entry, "_lf_surface", fake_surface)
    records = entry.evaluate_lf_fem_top(frame, tuple(sorted(build_power_splits().simulation_train)))
    assert len(records) == 3 and set(used) == {115.2, 403.0, 630.5}
    assert {r["modality"] for r in records} == {"Top"}
    assert all(r["sample_count"] == 2 and r["rmse_c"] == 1.0 for r in records)
    with pytest.raises(ValueError, match="验证|validation"):
        entry.evaluate_lf_fem_top(frame.with_columns(pl.lit("test").alias("split")),
                                  tuple(sorted(build_power_splits().simulation_train)))


def test_all_outputs_stay_inside_project_and_do_not_follow_external_symlink(tmp_path):
    entry = _entry()
    internal = tmp_path / "独立受限LF_FEM"
    assert entry.project_output(internal) == internal.resolve()
    outside = PROJECT_ROOT.parent / f"禁止项目外任11_{tmp_path.name}"
    with pytest.raises(ValueError, match="项目内"):
        entry.project_output(outside)
    link = tmp_path / "越界软链"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="项目内"):
        entry.project_output(link / "新结果")
    assert not outside.exists()


def test_source_registers_before_any_validation_and_rejects_stale_bytes(tmp_path, monkeypatch):
    entry = _entry()
    source_dir = tmp_path / "只读支点来源登记"
    output = tmp_path / "新顶部验证证据"
    identity = {"训练LF支点功率_W": list(sorted(build_power_splits().simulation_train)),
                "合法HF开发IR原件SHA256": "原件固定值"}
    monkeypatch.setattr(entry, "_source_identity", lambda: dict(identity))
    read_calls = []

    def read_validation(split):
        read_calls.append(split)
        return _observations()

    monkeypatch.setattr(entry, "load_processed_ir_observations", read_validation)
    monkeypatch.setattr(entry, "_lf_surface",
                        lambda group, supports: group["temperature_mean_k"].to_numpy())
    registered = entry.register_lf_fem_sources(source_dir)
    assert not read_calls and registered["LF训练功率数"] == 60
    filename = source_dir / entry.REGISTRATION_FILE
    assert registered["登记SHA256"] == sha256_file(filename)
    with pytest.raises(ValueError, match="SHA|登记"):
        entry.run_lf_fem_top_validation(output, filename, "0" * 64)
    assert not read_calls and not output.exists()
    identity["合法HF开发IR原件SHA256"] = "原件变更"
    with pytest.raises(ValueError, match="漂移|来源"):
        entry.run_lf_fem_top_validation(output, filename, registered["登记SHA256"])
    assert not read_calls and not output.exists()
    identity["合法HF开发IR原件SHA256"] = "原件固定值"
    result = entry.run_lf_fem_top_validation(output, filename, registered["登记SHA256"])
    assert read_calls == ["validation"] and result["HotCold两环LF_FEM验证"] is None
    assert result["五种子学习效果"] is None
    assert result["旧test_Data温度标签读取"] is False
    assert json.loads((output / "受限对照与缺项摘要.json").read_text(encoding="utf-8"))[
        "合法HF验证顶部功率数"] == 3
    with pytest.raises(FileExistsError):
        entry.run_lf_fem_top_validation(output, filename, registered["登记SHA256"])
