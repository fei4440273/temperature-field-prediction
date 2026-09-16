"""Task-09 subset source and budget gate; never run a formal trainer here."""

from __future__ import annotations

import copy
import importlib
import json
import math
import subprocess
import sys

import polars as pl
import pytest
import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import build_power_splits


def _gate():
    try:
        return importlib.import_module("sic_cu.train.task09_subset_gate")
    except ModuleNotFoundError:
        pytest.fail("Task-09 independent subset gate has not been implemented")


def test_registered_three_arms_are_exact_nested_subsets_with_unshrunk_validation():
    gate = _gate()
    arms = gate.load_task09_registry()
    splits = build_power_splits()
    assert set(arms) == {"primary", "left_center", "right_center"}
    for arm, sizes in arms.items():
        assert set(sizes) == {3, 6, 9, 12}
        assert [len(sizes[n]) for n in (3, 6, 9, 12)] == [3, 6, 9, 12]
        assert set(sizes[3]) < set(sizes[6]) < set(sizes[9]) < set(sizes[12])
        assert set(sizes[12]) == splits.hf_train
        assert set(sizes[3]) & (splits.hf_validation | splits.hf_test) == set()
        assert sizes[3][0] == 55.0 and sizes[3][-1] == 729.0
    assert arms["primary"][3] == (55.0, 364.3, 729.0)
    assert arms["left_center"][3] == (55.0, 309.0, 729.0)
    assert arms["right_center"][3] == (55.0, 430.0, 729.0)
    assert sha256_file(PROJECT_ROOT / gate.REGISTRATION_PATH) == gate.REGISTRATION_SHA256


def test_real_metadata_checks_exact_ir_rows_and_hot_cold_per_arm_without_test_labels():
    gate = _gate()
    observed = gate.audit_task09_observations(gate.load_task09_registry())
    expected = build_power_splits().hf_validation
    assert observed["validation"]["powers_w"] == sorted(expected)
    assert observed["validation"]["ir_rows"] > 0
    assert set(observed["validation"]["sensor_rows"]) == {"Hot", "Cold"}
    for arm in observed["arms"].values():
        for size in (3, 6, 9, 12):
            sample = arm[size]
            assert sample["ir_rows"] > 0
            assert set(sample["sensor_rows"]) == {"Hot", "Cold"}
            assert all(sample["sensor_rows"][name] > 0 for name in ("Hot", "Cold"))
            assert sample["ir_batches_2048"] == math.ceil(sample["ir_rows"] / 2048)
    assert observed["arms"]["primary"][3]["ir_rows"] == 6868
    assert observed["arms"]["primary"][6]["ir_rows"] == 15150
    assert observed["arms"]["primary"][9]["ir_rows"] == 22624
    assert observed["arms"]["primary"][12]["ir_rows"] == 29593


def test_metadata_rejects_missing_sensor_modality_or_shrunk_hf_validation():
    gate = _gate()
    arms = gate.load_task09_registry()
    ir, sensors = gate.read_task09_metadata()
    missing_hot = sensors.filter(~((pl.col("split") == "train") &
                                   (pl.col("power_w") == 55.0) &
                                   (pl.col("sensor_type") == "hot")))
    with pytest.raises(ValueError, match="hot|Hot|热端"):
        gate.audit_task09_observations(arms, ir, missing_hot)
    incomplete_validation = ir.filter(~((pl.col("split") == "validation") &
                                       (pl.col("power_w").is_between(115.19, 115.21))))
    with pytest.raises(ValueError, match="验证|validation"):
        gate.audit_task09_observations(arms, incomplete_validation, sensors)


def test_subset_budget_discloses_actual_batch_shortfall_and_full_lf60_forecast():
    gate = _gate()
    observed = gate.audit_task09_observations(gate.load_task09_registry())
    for arm in observed["arms"].values():
        for n, sample in arm.items():
            budget = gate.task09_budget(sample)
            assert budget["性质"] == "训练前来源预测，非实际训练消耗"
            assert budget["HF顶部IR真实训练行"] == sample["ir_rows"]
            assert budget["HF观测batch2048"] == sample["ir_batches_2048"]
            assert budget["HF训练传感器点_每批全量同步"] == sum(sample["sensor_rows"].values())
            assert budget["物理每轮配点"] == 256
            assert budget["LF固定训练功率数"] == 60
            assert budget["LF联合需真实回放点"] == 60 * 2048
            assert budget["LF四批按旧HF更新可消费批次"] == sample["ir_batches_2048"] * 4
            assert budget["LF旧四批方式缺口"] == 60 - sample["ir_batches_2048"] * 4
            assert budget["LF仅四末投影名称"] == sorted(gate.PROJECTION_NAMES)
            assert budget["LF旧四批方式可直接执行"] is (n == 12)


def test_training_budget_contract_rejects_changed_lf2048_or_macro_selection(monkeypatch):
    gate = _gate()
    gate.validate_task09_budget_contract()
    actual = load_yaml("configs/training.yaml")
    bad = copy.deepcopy(actual)
    bad["multifidelity"]["joint_simulation_samples_per_power"] = 1024
    monkeypatch.setattr(gate, "load_yaml", lambda path: bad if path == "configs/training.yaml"
                        else load_yaml(path))
    with pytest.raises(ValueError, match="2048|预算|LF"):
        gate.validate_task09_budget_contract()
    bad["multifidelity"]["joint_simulation_samples_per_power"] = 2048
    bad["selection_metric_version"] = "power_micro"
    with pytest.raises(ValueError, match="macro|合法选分"):
        gate.validate_task09_budget_contract()


@pytest.fixture(scope="module")
def paired_sources():
    return _gate().load_task09_sources()


def test_all_five_f3_starts_are_paired_new_hf_empty_adamw_and_original_lf(paired_sources):
    gate = _gate()
    assert set(paired_sources) == set(range(5))
    assert len({source.lf_tensor_sha256 for source in paired_sources.values()}) == 5
    starts = []
    for seed in range(5):
        start = gate.fork_task09_f3_start(paired_sources, seed, "primary", 3)
        lf = start.model.low_fidelity_model
        assert start.arm == "E0" and start.selected_for_training is False
        assert start.optimizer.state_dict()["state"] == {}
        assert all(not p.requires_grad for p in lf.parameters())
        assert all(p.requires_grad for p in start.model.correction.parameters())
        assert gate._lf_tensor_sha256(lf.state_dict()) == paired_sources[seed].lf_tensor_sha256
        assert start.random_state["torch_cuda"] is None
        starts.append(start.model.correction[0].weight.detach().clone())
    assert len({tensor.numpy().tobytes() for tensor in starts}) == 5
    first = gate.fork_task09_f3_start(paired_sources, 0, "primary", 3)
    second = gate.fork_task09_f3_start(paired_sources, 0, "left_center", 6)
    assert first.model is not second.model
    assert first.optimizer is not second.optimizer
    assert torch.equal(first.model.correction[0].weight, second.model.correction[0].weight)
    with pytest.raises(ValueError, match="序列|功率"):
        gate.fork_task09_f3_start(paired_sources, 0, "not_registered", 3)


def test_cpu_entry_archives_source_sha_before_reading_metadata_and_never_claims_scores(tmp_path, monkeypatch):
    gate = _gate()
    destination = tmp_path / "任09新来源报告"
    actual_read = gate.read_task09_metadata

    def verify_snapshot_before_read():
        assert (destination / "源码事前SHA256.json").is_file()
        assert (destination / "source_snapshot/task09_subset_gate.py").is_file()
        assert (destination / "source_snapshot/34_check_task09_subset_gate.py").is_file()
        return actual_read()

    monkeypatch.setattr(gate, "read_task09_metadata", verify_snapshot_before_read)
    report = gate.run_task09_cpu_entry(destination, name="primary", size=3)
    assert report["资格"] == "CPU来源入场；无正式训练、误差或精度曲线"
    assert report["已执行HF观测优化步"] == 0
    assert report["已执行LF联合回放点"] == 0
    assert report["F1纯HF-only正式物理入场"] is False
    assert report["HF合法验证固定功率数"] == 3
    assert set(report["五种子配对LF初始张量SHA256"]) == set(range(5))
    assert (destination / "只读入场与预算核查.json").is_file()
    assert (destination / "子集训练CPU来源入场.md").is_file()
    assert "旧TEST" not in (destination / "子集训练CPU来源入场.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        gate.run_task09_cpu_entry(destination, name="primary", size=3)


def test_cli_requires_explicit_cpu_gate_and_avoids_training_flags(tmp_path):
    cli = PROJECT_ROOT / "scripts/34_check_task09_subset_gate.py"
    assert cli.is_file(), "Task-09 gate CLI must exist before invoking it"
    destination = tmp_path / "任09CLI新工件"
    command = [sys.executable, str(cli), "--output", str(destination),
               "--arm", "right_center", "--size", "6"]
    refused = subprocess.run(command, capture_output=True, text=True, check=False)
    assert refused.returncode != 0
    assert not destination.exists()
    allowed = subprocess.run(command + ["--cpu-source-only"],
                             capture_output=True, text=True, check=False)
    assert allowed.returncode == 0, allowed.stderr
    summary = json.loads(allowed.stdout)
    assert summary["已执行HF观测优化步"] == 0
    assert summary["选定臂"] == "right_center" and summary["选定HF功率数"] == 6
    assert (destination / "子集训练CPU来源入场.md").is_file()
