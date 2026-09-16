"""CPU contracts for fresh Task-11 MLP HF training and immutable sessions."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import numpy as np
import torch
import yaml

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.splits import build_power_splits


def _formal():
    return importlib.import_module("sic_cu.train.task11_mlp_hf_formal")


def test_completed_lf_catalog_has_five_fresh_mlp_seeds() -> None:
    formal = _formal()
    catalog = formal.collect_task11_mlp_lf_catalog()
    assert catalog["阶段"] == "task11_completed_new_mlp_lf"
    assert [row["seed"] for row in catalog["五seed新LF"]] == list(range(5))
    assert catalog["旧固定TEST温度读取"] is False
    assert catalog["模拟测试功率温度读取"] is False
    assert all(len(row["最佳检查点SHA256"]) == 64
               and row["实际轮次"] >= 201
               and row["LF真实多会话累计成本秒"] > 0
               for row in catalog["五seed新LF"])


def test_hf_catalog_contains_only_structural_development_sources(monkeypatch) -> None:
    formal = _formal()

    def forbid_eager_labels(*args, **kwargs):
        raise AssertionError("目录采集不得解码温度或TEST")

    monkeypatch.setattr(formal.pl, "read_parquet", forbid_eager_labels)
    catalog = formal.collect_task11_mlp_hf_data_catalog()
    splits = build_power_splits()
    assert catalog["HF计数"] == {"Top训练": 29593, "Top验证": 7272,
                                    "环温训练": 2985, "环温验证": 752}
    assert catalog["HF训练功率_瓦"] == sorted(splits.hf_train)
    assert catalog["HF合法验证功率_瓦"] == sorted(splits.hf_validation)
    assert {row["原件"] for row in catalog["HF源原件"]} == {
        "data/processed/experiment_ir_radial.parquet",
        "data/processed/sensor_ring_raw.parquet",
    }
    assert catalog["旧固定TEST温度读取"] is False


def test_fresh_hf_has_six_inputs_empty_adamw_and_exact_own_lf() -> None:
    formal = _formal()
    row = formal.collect_task11_mlp_lf_catalog()["五seed新LF"][0]
    model, optimizer, architecture = formal.build_task11_mlp_hf_initialization(row, 0)
    original = torch.load(PROJECT_ROOT / row["最佳检查点"], map_location="cpu", weights_only=True)
    assert model.correction[0].in_features == 6
    assert architecture["low_fidelity_method"] == "mlp_pinn"
    assert architecture["correction_model_kwargs"] == {
        "width": 128, "depth": 4, "activation": "tanh", "include_material": True,
        "hard_initial_temperature_k": 295.15, "initial_ramp_time_s": 0.05,
        "correction_calibration_range_w": [55.0, 800.0],
        "correction_support_range_w": [0.0, 800.0],
        "correction_extrapolation_exponent": 2.0, "correction_power_scaling": "none",
        "correction_power_reference_w": 400.0, "correction_direct_power_input": True,
        "silicon_carbide_height_m": 0.012, "surface_guide_output": "residual",
    }
    assert optimizer.state_dict()["state"] == {}
    assert all(not parameter.requires_grad for parameter in model.low_fidelity_model.parameters())
    assert all(torch.equal(value, model.low_fidelity_model.state_dict()[name])
               for name, value in original["model_state"].items())
    second, _, _ = formal.build_task11_mlp_hf_initialization(row, 0)
    assert all(torch.equal(value, second.state_dict()[name])
               for name, value in model.state_dict().items())


def test_joint_optimizer_updates_all_lf_and_keeps_correction_momentum() -> None:
    formal = _formal()
    row = formal.collect_task11_mlp_lf_catalog()["五seed新LF"][0]
    model, optimizer, _ = formal.build_task11_mlp_hf_initialization(row, 0)
    optimizer.zero_grad()
    model(torch.tensor([[0.01, -0.005, 1.0, 100.0, 1.0]])).sum().backward()
    optimizer.step()
    old = {id(parameter): copy.deepcopy(optimizer.state[parameter])
           for parameter in model.correction.parameters()}
    formal.activate_task11_mlp_joint(model, optimizer)
    assert len(optimizer.param_groups) == 2
    assert {id(p) for p in optimizer.param_groups[1]["params"]} == {
        id(p) for p in model.low_fidelity_model.parameters()}
    assert all(p.requires_grad for p in model.parameters())
    assert all(group["lr"] == 0.0001 for group in optimizer.param_groups)
    assert all(torch.equal(optimizer.state[p]["exp_avg"], old[id(p)]["exp_avg"])
               for p in model.correction.parameters())
    assert all(p not in optimizer.state for p in model.low_fidelity_model.parameters())


def _frozen(tmp_path: Path):
    formal = _formal()
    sources = {}
    archive_path = tmp_path / "HF源码.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in formal.SOURCE_MEMBERS:
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if (PROJECT_ROOT / name).is_file():
                shutil.copyfile(PROJECT_ROOT / name, target)
            else:
                target.write_text("隔离门禁原件", encoding="utf-8")
            sources[name] = hashlib.sha256(target.read_bytes()).hexdigest()
            archive.add(target, arcname=name, recursive=False)
    lf_path = tmp_path / "LF目录.json"
    hf_path = tmp_path / "HF目录.json"
    lf_path.write_text(json.dumps(formal.collect_task11_mlp_lf_catalog(), ensure_ascii=False), encoding="utf-8")
    hf_path.write_text(json.dumps(formal.collect_task11_mlp_hf_data_catalog(), ensure_ascii=False), encoding="utf-8")
    checksum = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    registry = {"schema_version": 1, "阶段": "fresh_mlp_high_fidelity",
                "ROOT门禁标签": formal.ROOT_TOKEN,
                "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
                "LF全参联合更新": True, "正式预算": formal.HF_BUDGET,
                "独立能源审核": formal.ENERGY_BUDGET,
                "LF五seed目录SHA256": checksum(lf_path),
                "HF开发数据目录SHA256": checksum(hf_path),
                "源码普通成员SHA256": sources}
    registry_path = tmp_path / "HF登记.yaml"
    registry_path.write_text(yaml.safe_dump(registry, allow_unicode=True, sort_keys=False), encoding="utf-8")
    arguments = dict(registry=registry_path, registry_sha=checksum(registry_path),
                     source_tar=archive_path, source_tar_sha=checksum(archive_path),
                     lf_catalog=lf_path, lf_catalog_sha=checksum(lf_path),
                     hf_data_catalog=hf_path, hf_data_catalog_sha=checksum(hf_path),
                     seed=0, output=tmp_path / formal.RUN_DIRECTORY / "正式MLP_HF_seed0",
                     project_root=tmp_path)
    ledger = tmp_path / formal.ROOT_LEDGER
    ledger.write_text("", encoding="utf-8")
    return formal, arguments, ledger, registry


def _activate(formal, arguments, ledger, duplicate=False):
    second = (f"{formal.ROOT_TOKEN}; status=active; YAML_SHA256={arguments['registry_sha']}; "
              f"TAR_SHA256={arguments['source_tar_sha']}; LF_CATALOG_SHA256={arguments['lf_catalog_sha']}; "
              f"HF_DATA_CATALOG_SHA256={arguments['hf_data_catalog_sha']}")
    text = f"| 录-0101 | {second} | CPU隔离测试 |\n"
    ledger.write_text(text + (text if duplicate else ""), encoding="utf-8")


def test_missing_root_gate_rejects_before_cuda_output_or_labels(tmp_path, monkeypatch) -> None:
    formal, arguments, _, _ = _frozen(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("未锁身份不得探CUDA或读HF标签")

    monkeypatch.setattr(formal.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(formal, "_ir_dataset", forbidden)
    monkeypatch.setattr(formal, "_sensor_tensors", forbidden)
    with pytest.raises(ValueError, match="ROOT"):
        formal.run_task11_mlp_hf_formal(**arguments)
    assert not arguments["output"].exists()


def test_duplicate_root_gate_is_not_active(tmp_path) -> None:
    formal, arguments, ledger, _ = _frozen(tmp_path)
    _activate(formal, arguments, ledger, duplicate=True)
    with pytest.raises(ValueError, match="ROOT"):
        formal.preflight_task11_mlp_hf(**arguments)


def test_archived_source_drift_rejects_before_labels(tmp_path, monkeypatch) -> None:
    formal, arguments, ledger, registry = _frozen(tmp_path)
    _activate(formal, arguments, ledger)
    changed = tmp_path / next(iter(registry["源码普通成员SHA256"]))
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="源码|SHA"):
        formal.preflight_task11_mlp_hf(**arguments)
    assert not arguments["output"].exists()


def test_output_other_seed_rejects_before_source_catalog_loading(tmp_path, monkeypatch) -> None:
    formal, arguments, ledger, _ = _frozen(tmp_path)
    _activate(formal, arguments, ledger)
    arguments["output"] = tmp_path / formal.RUN_DIRECTORY / "正式MLP_HF_seed1"
    with pytest.raises(ValueError, match="同seed|输出"):
        formal.preflight_task11_mlp_hf(**arguments)


def test_early_stop_checks_only_validation_boundaries_and_budget() -> None:
    formal = _formal()
    assert not formal.task11_phase_finished(199, 0, 1500)
    assert formal.task11_phase_finished(200, 0, 1500)
    assert not formal.task11_phase_finished(209, 0, 1500)
    assert formal.task11_phase_finished(1500, 1490, 1500)
    assert formal.task11_phase_finished(500, 490, 500)


def test_full_log_requires_exact_sensor_lf_physics_consumption() -> None:
    formal = _formal()
    rows = [{"epoch": 1, "seed": 0, "阶段": formal.CORRECTION_STAGE,
             "阶段轮次": 1, "HF观测点": 29593, "HF传感器点": 2985 * 15,
             "HF观测优化步": 15, "物理优化步": 1, "物理配点": 256,
             "LF回放点": 0, "LF回放小批": 0,
             "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
             "合法验证选分_摄氏度": None}]
    assert formal.audit_task11_hf_log(rows, 0, 1, 0)["HF观测点"] == 29593
    wrong = copy.deepcopy(rows)
    wrong[0]["HF传感器点"] -= 1
    with pytest.raises(ValueError, match="消费|日志"):
        formal.audit_task11_hf_log(wrong, 0, 1, 0)


def test_pause_without_complete_own_state_cannot_resume(tmp_path) -> None:
    formal = _formal()
    output = tmp_path / "run"
    output.mkdir()
    (output / "阶段_最近.pt").write_text("模型视图不能恢复AdamW和四RNG", encoding="utf-8")
    with pytest.raises(ValueError, match="收据|完整|最近|断点"):
        formal.verify_task11_mlp_hf_resume(output, seed=0, identity={})


def test_formal_cli_has_no_old_lf_hf_or_test_override() -> None:
    _formal()
    process = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/51_run_task11_mlp_hf_formal.py"), "--help"],
                             cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert process.returncode == 0
    for required in ("--collect-lf-catalog", "--collect-hf-data-catalog", "--preflight-only",
                     "--audit-finished", "--lf-catalog-sha", "--hf-data-catalog-sha",
                     "--resume-checkpoint", "--session-epoch-limit"):
        assert required in process.stdout
    for forbidden in ("--evaluate-test", "--hf-test-powers", "--lf-checkpoint",
                      "--start-checkpoint", "--correction-epochs", "--hf-arm"):
        assert forbidden not in process.stdout


def test_source_members_cover_loaded_transitive_training_dependencies() -> None:
    formal = _formal()
    assert {"src/sic_cu/models/interpolation.py", "src/sic_cu/prediction.py",
            "src/sic_cu/train/surface_residual.py"} <= set(formal.SOURCE_MEMBERS)


def test_independent_energy_and_reports_do_not_change_training_identity(tmp_path) -> None:
    formal = _formal()
    (tmp_path / "training.jsonl").write_text("training", encoding="ascii")
    (tmp_path / "best.pt").write_bytes(b"view")
    before = formal._artifact_hashes(tmp_path)
    derived = tmp_path / "独立原能源_观测最佳_20260916T180000+0800"
    derived.mkdir()
    (derived / "原始能量.jsonl").write_text("audit", encoding="ascii")
    (tmp_path / "中文完训报告.md").write_text("派生报告", encoding="utf-8")
    assert formal._artifact_hashes(tmp_path) == before


def _initial_state(formal, joint=False):
    row = formal.collect_task11_mlp_lf_catalog()["五seed新LF"][0]
    model, optimizer, _ = formal.build_task11_mlp_hf_initialization(row, 0)
    if joint:
        formal.activate_task11_mlp_joint(model, optimizer)
    identity = {"seed": 0, "LF初始张量SHA256": row["LF初始张量SHA256"]}
    metadata = {"seed": 0, "身份": identity, "HF校正实际轮次": 0,
                "HF联合实际轮次": 0, "HF校正实际截止轮次": None,
                "全局实际轮次": 0, "当前LF张量SHA256": row["LF初始张量SHA256"],
                "本阶段最佳选分": 1.0, "本阶段最佳轮次": 0,
                "观测最佳选分_摄氏度": 1.0, "观测最佳全局轮次": 0,
                "观测最佳阶段": formal.CORRECTION_STAGE, "观测最佳阶段轮次": 0,
                "最近合法选分_摄氏度": 1.0, "旧固定TEST温度读取": False,
                "模拟测试功率温度读取": False}
    saved = {"training_state_schema_version": 1, "budget": formal.STAGE_BUDGET,
             "stage": formal.JOINT_STAGE if joint else formal.CORRECTION_STAGE,
             "epoch": 0, "metadata": metadata, "model_state": model.state_dict(),
             "optimizer_state": optimizer.state_dict(),
             "parameter_requires_grad": {name: p.requires_grad for name, p in model.named_parameters()},
             "random_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                              "torch_cpu": torch.get_rng_state(),
                              "torch_cuda": [torch.zeros(1, dtype=torch.uint8)]}}
    return saved, identity


def test_joint_state_without_real_correction_terminal_is_rejected() -> None:
    formal = _formal()
    saved, identity = _initial_state(formal, joint=True)
    with pytest.raises(ValueError, match="截止|校正|阶段"):
        formal._verify_full_state(saved, 0, identity)


def test_full_state_rejects_best_epoch_from_the_future() -> None:
    formal = _formal()
    saved, identity = _initial_state(formal)
    saved["metadata"]["观测最佳全局轮次"] = 10
    with pytest.raises(ValueError, match="最佳|轮次|选分"):
        formal._verify_full_state(saved, 0, identity)


def test_valid_cpu_preflight_does_not_probe_cuda_or_decode_labels(tmp_path, monkeypatch) -> None:
    formal, arguments, ledger, _ = _frozen(tmp_path)
    _activate(formal, arguments, ledger)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU门禁不得探CUDA或解温度")

    monkeypatch.setattr(formal.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(formal.pl, "read_parquet", forbidden)
    result = formal.preflight_task11_mlp_hf(**arguments)
    assert result["状态"].startswith("CPU前置门禁PASS")
    assert not arguments["output"].exists()


def test_historical_receipt_requires_exact_committed_log_prefix(tmp_path) -> None:
    formal = _formal()
    (tmp_path / "training.jsonl").write_bytes(b"changed-first\nsecond\n")
    originals = {"training.jsonl": hashlib.sha256(b"first\n").hexdigest()}
    for history, current in (("最近提交历史_0001.pt", "阶段_最近.pt"),
                             ("观测最佳提交历史_0001.pt", "阶段_观测最佳.pt"),
                             ("最佳视图提交历史_0001.pt", "best.pt")):
        (tmp_path / history).write_bytes(b"unit-history")
        originals[current] = hashlib.sha256(b"unit-history").hexdigest()
    row = {"身份": {}, "前驱收据SHA256": None, "起始已提交轮次": 0,
           "本会话实际轮次": 1, "累计实际轮次": 1, "本会话真实墙钟秒": 1.0,
           "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
           "状态": "已暂停且完整HF阶段提交", "峰值真实CUDA显存字节": 1,
           "原件SHA256": originals}
    first = tmp_path / "HF会话收据_0001.json"
    first.write_text(json.dumps(row), encoding="utf-8")
    second = tmp_path / "HF会话收据_0002.json"
    row = dict(row, 前驱收据SHA256=hashlib.sha256(first.read_bytes()).hexdigest(),
               起始已提交轮次=1, 累计实际轮次=2)
    second.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="日志|前缀|提交"):
        formal._audit_receipt_chain(tmp_path, [first, second], {}, finished=False)


def test_joint_phase_lineage_needs_locked_real_correction_terminal(tmp_path) -> None:
    formal = _formal()
    recent = {"stage": formal.JOINT_STAGE,
              "metadata": {"HF校正实际截止轮次": 200, "HF校正实际轮次": 200}}
    with pytest.raises(ValueError, match="校正|末|截止"):
        formal.verify_task11_mlp_hf_phase_lineage(tmp_path, recent, seed=0, identity={})


def test_initial_hf_tensor_must_be_own_seed_fresh_initialization() -> None:
    formal = _formal()
    saved, _ = _initial_state(formal)
    saved["model_state"]["correction.0.weight"] = saved["model_state"]["correction.0.weight"] + 1.0
    row = formal.collect_task11_mlp_lf_catalog()["五seed新LF"][0]
    with pytest.raises(ValueError, match="初态|初始|空网络|种子"):
        formal.verify_task11_mlp_hf_initial_state(saved, row, 0)


@pytest.mark.parametrize("field", ["torch_cpu", "torch_cuda"])
def test_complete_rng_rejects_nontensor_cpu_or_multiple_cuda_cards(field) -> None:
    formal = _formal()
    saved, identity = _initial_state(formal)
    saved["random_state"][field] = "invalid" if field == "torch_cpu" else [torch.zeros(1), torch.zeros(1)]
    with pytest.raises(ValueError, match="RNG|单卡|随机"):
        formal._verify_full_state(saved, 0, identity)


def test_rng_tree_compares_numpy_and_torch_bytes_exactly() -> None:
    formal = _formal()
    first = {"numpy": ("MT", np.array([1, 2], dtype=np.uint32)), "torch_cpu": torch.zeros(2, dtype=torch.uint8)}
    assert formal._same_tree(first, copy.deepcopy(first))
    wrong = copy.deepcopy(first)
    wrong["numpy"][1][0] += 1
    assert not formal._same_tree(first, wrong)


@pytest.mark.parametrize("field", ["optimizer_state", "random_state", "parameter_requires_grad", "metadata"])
def test_recent_joint_terminal_and_final_must_match_complete_state(field) -> None:
    formal = _formal()
    snapshots = {name: {"model_state": {"w": torch.ones(1)},
                        "optimizer_state": {"step": 1}, "random_state": {"cpu": torch.zeros(1)},
                        "parameter_requires_grad": {"w": True}, "metadata": {"epoch": 1},
                        "stage": formal.JOINT_STAGE, "epoch": 1}
                 for name in ("阶段_最近.pt", "阶段_联合末.pt", "阶段_训练末.pt")}
    snapshots["阶段_联合末.pt"][field] = {"changed": 1}
    with pytest.raises(ValueError, match="终态|最近|完整|一致"):
        formal.verify_task11_mlp_hf_terminal_states(snapshots)


def test_joint_transition_reuses_correction_score_without_validation_or_rng_change(monkeypatch) -> None:
    formal = _formal()
    row = formal.collect_task11_mlp_lf_catalog()["五seed新LF"][0]
    model, optimizer, _ = formal.build_task11_mlp_hf_initialization(row, 0)
    meta = {"最近合法选分_摄氏度": 2.0, "最近合法分模态": {"顶部": 3.0},
            "HF校正实际截止轮次": 200, "HF校正实际轮次": 200}
    before = {"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.get_rng_state()}

    def forbidden(*args, **kwargs):
        raise AssertionError("同张量阶段转换不能再次验证或消费随机源")

    monkeypatch.setattr(formal, "_validation_selection", forbidden)
    score, validation = formal.transition_task11_mlp_hf_joint(model, optimizer, meta)
    after = {"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.get_rng_state()}
    assert score == 2.0 and validation == {"顶部": 3.0}
    assert formal._same_tree(before, after)


def test_selection_history_cannot_run_beyond_first_patience_stop() -> None:
    formal = _formal()
    rows = [{"合法验证选分_摄氏度": 10.0 if i % 10 == 0 else None} for i in range(1, 211)]
    with pytest.raises(ValueError, match="耐心|截止|早停"):
        formal.replay_task11_mlp_hf_selection(rows, 10.0, 210, 0, stage=formal.CORRECTION_STAGE)


def test_selection_history_reconstructs_global_and_phase_best() -> None:
    formal = _formal()
    rows = [{"合法验证选分_摄氏度": (9.0 if i >= 100 else 10.0) if i % 10 == 0 else None}
            for i in range(1, 301)]
    rows += [{"合法验证选分_摄氏度": (8.0 if i >= 100 else 9.0) if i % 10 == 0 else None}
             for i in range(1, 201)]
    result = formal.replay_task11_mlp_hf_selection(rows, 10.0, 300, 200, stage=formal.JOINT_STAGE)
    assert result["观测最佳全局轮次"] == 400
    assert result["观测最佳选分_摄氏度"] == 8.0
    assert result["本阶段最佳轮次"] == 100
    assert result["HF校正实际截止轮次"] == 300


def test_cpu_fresh_initialization_never_seeds_cuda(monkeypatch) -> None:
    formal = _formal()
    row = formal.collect_task11_mlp_lf_catalog()["五seed新LF"][0]

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU初态核验不得改写CUDA随机源")

    monkeypatch.setattr(formal.torch.cuda, "manual_seed_all", forbidden)
    model, optimizer, _ = formal.build_task11_mlp_hf_initialization(row, 0)
    assert model.correction[0].in_features == 6 and not optimizer.state


@pytest.mark.parametrize("field", ["method", "seed", "lf_training_seconds", "energy_used_for_selection"])
def test_metrics_must_match_true_own_lf_cost_seed_and_no_energy_selection(field) -> None:
    formal = _formal()
    metrics = {"method": "mlp_pinn_additive", "seed": 0, "lf_training_seconds": 3.5,
               "energy_used_for_selection": False}
    formal.verify_task11_mlp_hf_metrics_identity(metrics, seed=0, lf_seconds=3.5)
    metrics[field] = {"method": "other", "seed": 1, "lf_training_seconds": 1.0,
                      "energy_used_for_selection": True}[field]
    with pytest.raises(ValueError, match="成本|种子|能源|指标|method"):
        formal.verify_task11_mlp_hf_metrics_identity(metrics, seed=0, lf_seconds=3.5)


@pytest.mark.parametrize("field", ["method", "validation_selection_score_c", "validation_sensor", "validation_rmse_c"])
def test_model_view_statistics_must_match_complete_snapshot(field) -> None:
    formal = _formal()
    snapshot = {"metadata": {"最近合法选分_摄氏度": 2.0, "最近合法分模态": {"顶部": 3.0}}}
    view = {"method": "multifidelity_correction", "validation_selection_score_c": 2.0,
            "validation_sensor": {"顶部": 3.0}, "validation_rmse_c": 3.0}
    formal.verify_task11_mlp_hf_view_statistics(view, snapshot)
    view[field] = {"method": "other", "validation_selection_score_c": 1.0,
                   "validation_sensor": {"顶部": 1.0}, "validation_rmse_c": 1.0}[field]
    with pytest.raises(ValueError, match="视图|选分|分模态|统计|method"):
        formal.verify_task11_mlp_hf_view_statistics(view, snapshot)
