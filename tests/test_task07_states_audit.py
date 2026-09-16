"""任-07五种子完整阶段的独立能源审核门禁。"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import torch
import yaml

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.train.task07_source import validate_task07_sources


AUDITOR_PATH = PROJECT_ROOT / "scripts/26_audit_task07_states.py"


def _auditor():
    assert AUDITOR_PATH.is_file(), "任-07独立五种子状态审核脚本尚未实现"
    spec = importlib.util.spec_from_file_location("task07_states_energy_audit", AUDITOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prior_registered_30_points_and_two_orders_are_the_only_audit_schedule() -> None:
    audit = _auditor()
    assert audit._fixed_schedule() == (
        [55.0, 115.2, 364.3, 403.0, 630.5, 729.0],
        [1.0, 10.0, 50.0, 100.0, 200.0],
        [16, 64],
    )


def test_short_cpu_diagnostic_and_missing_real_joint_terminal_cannot_create_audit_output(
    tmp_path: Path,
) -> None:
    audit = _auditor()
    short = tmp_path / "任07种子0_CPU短诊断"
    short.mkdir()
    (short / "阶段报告.json").write_text(json.dumps({
        "运行种子": 0, "运行资格": "短诊断；不参与任07正式五种子采用",
        "校正实际轮次": 2, "联合实际轮次": 1,
    }, ensure_ascii=False), encoding="utf-8")
    destination = tmp_path / "未通过的审核输出"
    with pytest.raises((ValueError, FileNotFoundError), match="正式|诊断|联合末|完整|末态"):
        audit.audit_completed_state(short, seed=0, state="best", output=destination,
                                    device_name="cpu")
    assert not destination.exists()


def test_bare_pinn_cli_help_without_pythonpath_or_ros_scripts_collision() -> None:
    assert AUDITOR_PATH.is_file(), "任-07独立五种子状态审核脚本尚未实现"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        ["/home/phl/anaconda3/envs/PINN/bin/python", str(AUDITOR_PATH), "--help"],
        cwd=PROJECT_ROOT, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--seed" in result.stdout and "--state" in result.stdout


def _four_hundred_rows(seed: int = 0) -> list[dict]:
    source_sha = "c01575e2b845c25097f14d5d2cddb0a0f2763d02c9d467af0d25cbc6790ef5f8"
    baseline = {
        "Cu": {"node": 3.3274873948763086, "volume": 2.812075453948512},
        "SiC": {"node": 9.356307395999307, "volume": 7.533927168970277},
    }
    good = {"LF两材料节点与真实体积均守住5%护栏": True}
    history = []
    for epoch in range(1, 401):
        correction = epoch <= 200
        local = epoch if correction else epoch - 200
        replay_batches = 0 if correction else 60
        due = local % 10 == 0
        history.append({
            "epoch": epoch, "全局实际轮次": epoch,
            "运行种子": seed, "运行臂": "E0",
            "训练阶段": "task07_correction" if correction else "task07_restricted_joint",
            "阶段实际轮次": local,
            "HF训练观测点": 29593, "HF训练传感器点": 44775,
            "HF观测优化步": 15, "物理优化步": 1, "物理配点": 256,
            "LF真实回放训练点": replay_batches * 2048,
            "LF联合回放batch": replay_batches,
            "LF_Cu真实回放点": 0 if correction else 60 * 1024,
            "LF_SiC真实回放点": 0 if correction else 60 * 1024,
            "LF当前真实张量SHA256": source_sha if correction else "f" * 64,
            "HF合法验证选分_摄氏度": 2.1 if due else None,
            "HF合法验证分模态_摄氏度": {
                "top": 3.0, "hot": 1.0, "cold": 0.5,
            } if due else None,
            "LF合法验证初态逐材料节点与体积RMSE_摄氏度": (
                baseline if due and correction else None
            ),
            "LF合法验证逐材料节点与体积RMSE_摄氏度": baseline if due and not correction else None,
            "LF逐材料节点和真实体积5%护栏": good if due and not correction else None,
            "独立局部物理损失": {
                "pde": 0.01, "initial": 0.0, "boundary": 0.02,
                "interface": 0.01, "physics_total": 0.05,
            } if due else None,
            "累计实际消耗": {
                "HF训练观测点": 29593 * epoch,
                "HF训练传感器点": 44775 * epoch,
                "LF真实回放训练点": max(epoch - 200, 0) * 60 * 2048,
                "物理配点": 256 * epoch,
                "HF观测优化步": 15 * epoch,
                "物理优化步": epoch,
                "LF联合回放batch": 60 * max(epoch - 200, 0),
            },
        })
    return history


@pytest.mark.parametrize("tamper", ["missing_epoch", "one_hf_point", "fake_physics_step",
                                      "joint_lf_short", "early_fake_validation"])
def test_full_history_rejects_missing_epochs_shrunk_samples_or_unregistered_scores(
    tamper: str,
) -> None:
    audit = _auditor()
    history = _four_hundred_rows()
    if tamper == "missing_epoch":
        del history[49]
    elif tamper == "one_hf_point":
        history[51]["HF训练观测点"] = 1
    elif tamper == "fake_physics_step":
        history[51]["物理优化步"] = 0
    elif tamper == "joint_lf_short":
        history[251]["LF真实回放训练点"] = 1
    else:
        history[51]["HF合法验证选分_摄氏度"] = 0.1
    with pytest.raises(ValueError, match="日志|轮次|HF|物理|LF|验证|预算"):
        audit._verify_history(history, seed=0, correction_epoch=200, joint_epoch=200,
                              source_lf_sha=history[0]["LF当前真实张量SHA256"],
                              terminal_lf_sha=history[-1]["LF当前真实张量SHA256"],
                              initial_score=2.0, initial_physical=0.04)


def test_fully_committed_two_stage_log_keeps_stage_qualified_initial_best() -> None:
    audit = _auditor()
    history = _four_hundred_rows()
    selected = audit._verify_history(
        history, seed=0, correction_epoch=200, joint_epoch=200,
        source_lf_sha=history[0]["LF当前真实张量SHA256"],
        terminal_lf_sha=history[-1]["LF当前真实张量SHA256"],
        initial_score=2.0, initial_physical=0.04,
    )
    assert selected["观测最佳全局轮次"] == 0
    assert selected["物理最佳全局轮次"] == 0
    assert selected["校正实际截止轮次"] == 200
    assert selected["联合实际截止轮次"] == 200
    assert selected["累计实际消耗"]["HF训练观测点"] == 29593 * 400
    assert selected["累计实际消耗"]["LF真实回放训练点"] == 60 * 2048 * 200


@pytest.fixture(scope="module")
def paired_sources():
    return validate_task07_sources()


def _fresh_initial_stage(paired_sources, seed: int = 0) -> dict:
    from sic_cu.train.task07_formal import CORRECTION_STAGE, STAGE_BUDGET, _metadata, fork_task07_e0

    source = paired_sources[seed]
    initial = fork_task07_e0(paired_sources, seed, torch.device("cpu"))
    model = initial.model
    baseline = {
        "Cu": {"node": 3.3274873948763086, "volume": 2.812075453948512},
        "SiC": {"node": 9.356307395999307, "volume": 7.533927168970277},
    }
    usage = {name: 0 for name in (
        "HF训练观测点", "HF训练传感器点", "LF真实回放训练点",
        "物理配点", "HF观测优化步", "物理优化步", "LF联合回放batch",
    )}
    meta = _metadata(
        source, model, CORRECTION_STAGE, "f" * 64, False,
        correction_epoch=0, joint_epoch=0, correction_completed=None,
        initial_score=2.0, best_score=2.0, best_global_epoch=0,
        best_stage=CORRECTION_STAGE, best_stage_epoch=0,
        physical_score=0.04, physical_global_epoch=0,
        physical_stage=CORRECTION_STAGE, physical_stage_epoch=0,
        phase_best_epoch=0, lf_reference=baseline, consumption=usage,
    )
    rng = initial.random_state.copy()
    rng["torch_cuda"] = [torch.zeros(16, dtype=torch.uint8)]
    return {
        "training_state_schema_version": 1, "stage": CORRECTION_STAGE,
        "epoch": 0, "budget": STAGE_BUDGET.copy(), "metadata": meta,
        "model_state": {name: tensor.cpu().clone()
                        for name, tensor in model.state_dict().items()},
        "parameter_requires_grad": {
            name: parameter.requires_grad for name, parameter in model.named_parameters()
        },
        "optimizer_state": initial.optimizer.state_dict(),
        "random_state": rng,
    }


def test_seed_zero_valid_all_zero_cuda_state_and_fresh_hf_weights_are_accepted(
    paired_sources,
) -> None:
    audit = _auditor()
    initial = _fresh_initial_stage(paired_sources)
    audit._verify_stage(initial, paired_sources, seed=0,
                        registry_sha="f" * 64, expected_stage="task07_correction",
                        correction_epoch=0, joint_epoch=0)


@pytest.mark.parametrize("tamper", ["historic_hf", "borrowed_momentum", "missing_cuda",
                                      "wrong_cuda_shape", "wrong_lf_source"])
def test_initial_complete_stage_rejects_hist_hf_cpu_rng_and_wrong_paired_lf(
    paired_sources, tamper: str,
) -> None:
    audit = _auditor()
    stage = _fresh_initial_stage(paired_sources)
    if tamper == "historic_hf":
        historical = torch.load(paired_sources[0].hf_checkpoint_path, map_location="cpu",
                                weights_only=False)
        stage["model_state"]["correction.0.weight"] = historical["model_state"][
            "correction.0.weight"
        ].clone()
    elif tamper == "borrowed_momentum":
        stage["optimizer_state"]["state"] = {0: {"step": torch.tensor(160)}}
    elif tamper == "missing_cuda":
        stage["random_state"]["torch_cuda"] = None
    elif tamper == "wrong_cuda_shape":
        stage["random_state"]["torch_cuda"] = [torch.zeros(17, dtype=torch.uint8)]
    else:
        stage["model_state"]["low_fidelity_model.branch_projection.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="来源|初态|AdamW|随机|CUDA|LF|HF"):
        audit._verify_stage(stage, paired_sources, seed=0,
                            registry_sha="f" * 64, expected_stage="task07_correction",
                            correction_epoch=0, joint_epoch=0)


def _trained_correction_stage(paired_sources, correction_epoch: int = 200) -> dict:
    state = _fresh_initial_stage(paired_sources)
    state["epoch"] = correction_epoch
    metadata = state["metadata"]
    metadata["校正实际轮次"] = correction_epoch
    metadata["校正实际截止轮次"] = correction_epoch
    metadata["全局累计实际轮次"] = correction_epoch
    metadata["累计实际消耗"] = {
        "HF训练观测点": 29593 * correction_epoch,
        "HF训练传感器点": 44775 * correction_epoch,
        "LF真实回放训练点": 0,
        "物理配点": 256 * correction_epoch,
        "HF观测优化步": 15 * correction_epoch,
        "物理优化步": correction_epoch,
        "LF联合回放batch": 0,
    }
    hf_names = [name for name in state["parameter_requires_grad"]
                if name.startswith("correction.")]
    group = state["optimizer_state"]["param_groups"][0]
    state["optimizer_state"]["state"] = {
        index: {
            "step": torch.tensor(16.0 * correction_epoch),
            "exp_avg": torch.zeros_like(state["model_state"][name]),
            "exp_avg_sq": torch.zeros_like(state["model_state"][name]),
        }
        for index, name in zip(group["params"], hf_names)
    }
    return state


def test_every_hf_adamw_parameter_requires_exactly_fifteen_plus_one_steps(
    paired_sources,
) -> None:
    audit = _auditor()
    trained = _trained_correction_stage(paired_sources)
    audit._verify_stage(trained, paired_sources, seed=0, registry_sha="f" * 64,
                        expected_stage="task07_correction", correction_epoch=200,
                        joint_epoch=0)
    for tamper in ("wiped", "short", "extra", "wrong_shape"):
        poisoned = _trained_correction_stage(paired_sources)
        first = poisoned["optimizer_state"]["param_groups"][0]["params"][0]
        if tamper == "wiped":
            poisoned["optimizer_state"]["state"].pop(first)
        elif tamper == "short":
            poisoned["optimizer_state"]["state"][first]["step"] = torch.tensor(3199.0)
        elif tamper == "extra":
            poisoned["optimizer_state"]["state"][first]["step"] = torch.tensor(3201.0)
        else:
            record = poisoned["optimizer_state"]["state"][first]
            record["exp_avg"] = record["exp_avg"].flatten()
        with pytest.raises(ValueError, match="AdamW|动量|步|真实"):
            audit._verify_stage(poisoned, paired_sources, seed=0, registry_sha="f" * 64,
                                expected_stage="task07_correction", correction_epoch=200,
                                joint_epoch=0)


def _trained_joint_stage(paired_sources, joint_epoch: int) -> dict:
    from sic_cu.train.task07_source import _lf_tensor_sha256

    state = _trained_correction_stage(paired_sources)
    state["stage"] = "task07_restricted_joint"
    state["epoch"] = joint_epoch
    meta = state["metadata"]
    meta["当前阶段"] = state["stage"]
    meta["校正实际截止轮次"] = 200
    meta["已提交正式校正末原件SHA256"] = "e" * 64
    meta["联合实际轮次"] = joint_epoch
    meta["全局累计实际轮次"] = 200 + joint_epoch
    meta["本阶段早停最佳轮次"] = 0
    meta["累计实际消耗"] = {
        "HF训练观测点": 29593 * (200 + joint_epoch),
        "HF训练传感器点": 44775 * (200 + joint_epoch),
        "LF真实回放训练点": 60 * 2048 * joint_epoch,
        "物理配点": 256 * (200 + joint_epoch),
        "HF观测优化步": 15 * (200 + joint_epoch),
        "物理优化步": 200 + joint_epoch,
        "LF联合回放batch": 60 * joint_epoch,
    }
    projections = [name for name in state["parameter_requires_grad"]
                   if name in (
                       "low_fidelity_model.branch_projection.weight",
                       "low_fidelity_model.branch_projection.bias",
                       "low_fidelity_model.trunk_projection.weight",
                       "low_fidelity_model.trunk_projection.bias",
                   )]
    source = paired_sources[0]
    for name in projections:
        state["parameter_requires_grad"][name] = True
        if joint_epoch:
            state["model_state"][name] += 1e-4
    lf = {name: state["model_state"][f"low_fidelity_model.{name}"]
          for name in source.lf_state}
    meta["当前真实LF张量SHA256"] = _lf_tensor_sha256(lf)
    first = state["optimizer_state"]["param_groups"][0]
    first["lr"] = 0.0001
    second = first.copy()
    second["lr"] = 0.00001
    second["params"] = list(range(len(first["params"]), len(first["params"]) + 4))
    state["optimizer_state"]["param_groups"].append(second)
    hf_moment = state["optimizer_state"]["state"]
    for record in hf_moment.values():
        record["step"] = torch.tensor(16.0 * (200 + joint_epoch))
    if joint_epoch:
        for index, name in zip(second["params"], projections):
            hf_moment[index] = {
                "step": torch.tensor(16.0 * joint_epoch),
                "exp_avg": torch.zeros_like(state["model_state"][name]),
                "exp_avg_sq": torch.zeros_like(state["model_state"][name]),
            }
    return state


def test_joint_initial_has_preserved_hf_momentum_but_no_untrained_lf_momentum(
    paired_sources,
) -> None:
    audit = _auditor()
    stage = _trained_joint_stage(paired_sources, 0)
    audit._verify_stage(stage, paired_sources, seed=0, registry_sha="f" * 64,
                        expected_stage="task07_restricted_joint", correction_epoch=200,
                        joint_epoch=0)


def test_joint_real_one_epoch_keeps_independent_hf_and_four_lf_optimizer_steps(
    paired_sources,
) -> None:
    audit = _auditor()
    stage = _trained_joint_stage(paired_sources, 1)
    audit._verify_stage(stage, paired_sources, seed=0, registry_sha="f" * 64,
                        expected_stage="task07_restricted_joint", correction_epoch=200,
                        joint_epoch=1)


@pytest.mark.parametrize("tamper", ["missing_lf_momentum", "lf_steal_step", "lf_other_changed",
                                      "fake_hf_group", "projection_not_changed"])
def test_joint_real_epoch_rejects_each_four_projection_and_other_lf_tamper(
    paired_sources, tamper: str,
) -> None:
    audit = _auditor()
    stage = _trained_joint_stage(paired_sources, 1)
    if tamper == "missing_lf_momentum":
        first_lf = stage["optimizer_state"]["param_groups"][1]["params"][0]
        stage["optimizer_state"]["state"].pop(first_lf)
    elif tamper == "lf_steal_step":
        first_lf = stage["optimizer_state"]["param_groups"][1]["params"][0]
        stage["optimizer_state"]["state"][first_lf]["step"] = torch.tensor(17.0)
    elif tamper == "lf_other_changed":
        other = next(name for name, value in stage["model_state"].items()
                     if name.startswith("low_fidelity_model.")
                     and "_projection." not in name and value.ndim == 2)
        stage["model_state"][other][0, 0] += 1
    elif tamper == "projection_not_changed":
        from sic_cu.train.task07_source import _lf_tensor_sha256

        projection = "low_fidelity_model.branch_projection.bias"
        stage["model_state"][projection] = paired_sources[0].lf_state[
            "branch_projection.bias"
        ].clone()
        stage["metadata"]["当前真实LF张量SHA256"] = _lf_tensor_sha256({
            name: stage["model_state"][f"low_fidelity_model.{name}"]
            for name in paired_sources[0].lf_state
        })
    else:
        stage["optimizer_state"]["param_groups"][0]["lr"] = 0.001
    with pytest.raises(ValueError, match="AdamW|动量|步|LF|来源"):
        audit._verify_stage(stage, paired_sources, seed=0, registry_sha="f" * 64,
                            expected_stage="task07_restricted_joint", correction_epoch=200,
                            joint_epoch=1)


def _registered_test_audit(audit, paired_sources, tmp_path, monkeypatch,
                           *, budget: int = 1500, ledger_record: str = "录-0036"):
    from sic_cu.train.task07_formal import _expected_registration

    config = _expected_registration(paired_sources)
    config["固定训练合同"]["HF校正预算上限"] = budget
    registry = tmp_path / "有效运行配置_实现修复后.yaml"
    registry.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    digest = sha256_file(registry)
    ledger = tmp_path / "预登记源追踪总台账.md"
    ledger.write_text(
        "## 九、已实施事项与改动记录表\n"
        f"| {ledger_record} | 任-07修复后事前预登记 | {registry.name} | SHA `{digest}` |\n"
        "## 十、指标改善明细表\n", encoding="utf-8",
    )
    monkeypatch.setattr(audit, "TASK07_REGISTRATION", registry, raising=False)
    monkeypatch.setattr(audit, "TASK07_LEDGER", ledger, raising=False)
    return registry, digest


def test_registration_uses_repaired_new_ledger_record_and_five_source_shas(
    paired_sources, tmp_path, monkeypatch,
) -> None:
    audit = _auditor()
    _, digest = _registered_test_audit(audit, paired_sources, tmp_path, monkeypatch)
    assert audit._verify_registry(paired_sources, digest)["sha256"] == digest


@pytest.mark.parametrize("tamper", ["budget_self_signed", "old_ledger_record", "wrong_seed_lf_sha"])
def test_self_signed_yaml_and_ledger_cannot_change_source_or_official_1500_budget(
    paired_sources, tmp_path, monkeypatch, tamper: str,
) -> None:
    audit = _auditor()
    registry, digest = _registered_test_audit(
        audit, paired_sources, tmp_path, monkeypatch,
        budget=1 if tamper == "budget_self_signed" else 1500,
        ledger_record="录-0033" if tamper == "old_ledger_record" else "录-0036",
    )
    if tamper == "wrong_seed_lf_sha":
        yaml_data = yaml.safe_load(registry.read_text(encoding="utf-8"))
        yaml_data["五种子配对来源"][0]["LF检查点SHA256"] = "0" * 64
        registry.write_text(yaml.safe_dump(yaml_data, allow_unicode=True), encoding="utf-8")
        digest = sha256_file(registry)
        ledger = audit.TASK07_LEDGER
        ledger.write_text(
            "## 九、已实施事项与改动记录表\n"
            f"| 录-0036 | 任-07修复后事前预登记 | {registry.name} | SHA `{digest}` |\n"
            "## 十、指标改善明细表\n", encoding="utf-8",
        )
    with pytest.raises(ValueError, match="预登记|录-0036|五种子|来源|SHA|预算"):
        audit._verify_registry(paired_sources, digest)


@pytest.mark.parametrize("tamper", ["self_signed_geometry", "resolved_physics", "copied_ledger"])
def test_legacy_source_snapshot_does_not_accept_self_signed_fake_run_config_or_physics(
    tmp_path, tamper: str,
) -> None:
    from sic_cu.train.common import write_config_snapshot

    audit = _auditor()
    run = tmp_path / "修复后正式_E0_种子0_来源伪装"
    run.mkdir()
    write_config_snapshot(run)
    snapshot = run / "config_snapshot"
    if tamper == "self_signed_geometry":
        copied = snapshot / "geometry.yaml"
        copied.write_bytes(copied.read_bytes() + b"\n# forged copy\n")
        digest = json.loads((snapshot / "sha256.json").read_text(encoding="utf-8"))
        digest["configs/geometry.yaml"] = sha256_file(copied)
        (snapshot / "sha256.json").write_text(json.dumps(digest), encoding="utf-8")
    elif tamper == "resolved_physics":
        copied = snapshot / "resolved_physics.yaml"
        loaded = yaml.safe_load(copied.read_text(encoding="utf-8"))
        loaded["values"]["silicon_carbide_emissivity"]["value"] = 0.99
        copied.write_text(yaml.safe_dump(loaded, allow_unicode=True), encoding="utf-8")
    else:
        copied = snapshot / "多保真DeepONet预测精度优化总计划与执行台账.md"
        copied.write_bytes(copied.read_bytes() + b"\nforged ledger after the run\n")
    with pytest.raises(ValueError, match="快照|SHA|物理|台账"):
        audit._verify_snapshot(run)


def test_real_seven_operation_config_and_copied_chinese_ledger_snapshot_pass(
    tmp_path,
) -> None:
    from sic_cu.train.common import write_config_snapshot

    audit = _auditor()
    run = tmp_path / "修复后正式_E0_种子0_完整快照"
    run.mkdir()
    write_config_snapshot(run)
    observed = audit._verify_snapshot(run)
    assert len(observed["七份运营配置当前与复制双SHA"]) == 7
    assert observed["总台账运行时副本SHA256"] == sha256_file(
        run / "config_snapshot/多保真DeepONet预测精度优化总计划与执行台账.md"
    )


def _synthetic_completed_run(tmp_path, paired_sources):
    from sic_cu.train.common import write_config_snapshot
    from sic_cu.train.task07_formal import _model_view, fork_task07_e0, TASK07_REGISTRATION

    run = tmp_path / "修复后正式_E0_种子0_合成阶段非训练"
    run.mkdir()
    write_config_snapshot(run)
    source = paired_sources[0]
    registry_sha = sha256_file(TASK07_REGISTRATION)
    fresh = fork_task07_e0(paired_sources, 0, torch.device("cpu"))
    initial = _fresh_initial_stage(paired_sources)
    initial["metadata"]["正式预登记配置SHA256"] = registry_sha
    torch.save(initial, run / "阶段_初始.pt")
    torch.save(initial, run / "阶段_观测最佳.pt")
    torch.save(initial, run / "阶段_物理最佳.pt")
    _model_view(
        run / "best.pt",
        torch.load(source.hf_checkpoint_path, map_location="cpu", weights_only=False),
        fresh.model, source, global_epoch=0, correction_epoch=0, joint_epoch=0,
        score=2.0, validation={
            "顶部": 3.0, "absolute_rmse_c": 1.0, "delta_rmse_c": 0.5,
        }, registry_sha=registry_sha, diagnostic_only=False,
    )
    correction = _trained_correction_stage(paired_sources)
    correction["metadata"]["正式预登记配置SHA256"] = registry_sha
    correction["metadata"]["已提交旧阶段_观测最佳.ptSHA256"] = sha256_file(
        run / "阶段_观测最佳.pt"
    )
    correction["metadata"]["已提交旧阶段_物理最佳.ptSHA256"] = sha256_file(
        run / "阶段_物理最佳.pt"
    )
    torch.save(correction, run / "阶段_校正末.pt")
    joint_initial = _trained_joint_stage(paired_sources, 0)
    joint_initial["metadata"]["正式预登记配置SHA256"] = registry_sha
    joint_initial["metadata"]["已提交正式校正末原件SHA256"] = sha256_file(
        run / "阶段_校正末.pt"
    )
    joint_initial["metadata"]["本阶段最低合格HF选分_摄氏度"] = 2.1
    torch.save(joint_initial, run / "阶段_联合初始.pt")
    final = _trained_joint_stage(paired_sources, 200)
    final_meta = final["metadata"]
    final_meta["正式预登记配置SHA256"] = registry_sha
    final_meta["已提交正式校正末原件SHA256"] = sha256_file(run / "阶段_校正末.pt")
    final_meta["本阶段最低合格HF选分_摄氏度"] = 2.1
    final_meta["LF逐材料节点和真实体积5%护栏"] = {
        "LF两材料节点与真实体积均守住5%护栏": True,
    }
    final_meta["已提交旧阶段_观测最佳.ptSHA256"] = sha256_file(
        run / "阶段_观测最佳.pt"
    )
    final_meta["已提交旧阶段_物理最佳.ptSHA256"] = sha256_file(
        run / "阶段_物理最佳.pt"
    )
    for filename in ("阶段_最近.pt", "阶段_联合末.pt", "阶段_训练末.pt"):
        torch.save(final, run / filename)
    history = _four_hundred_rows()
    for row in history:
        row["LF当前真实张量SHA256"] = (
            source.lf_tensor_sha256 if row["训练阶段"] == "task07_correction"
            else final_meta["当前真实LF张量SHA256"]
        )
    (run / "training.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in history),
        encoding="utf-8",
    )
    report = {
        "状态": "正式受限联合已达同组截止，仍须五种子能源审计",
        "运行资格": "五种子正式E0候选；须五seed完整归档与能源审核后决定采用",
        "运行种子": 0, "运行臂": "E0", "正式预登记配置SHA256": registry_sha,
        "本seed源LF检查点SHA256": source.lf_checkpoint_sha256,
        "本seed真实LF初始张量SHA256": source.lf_tensor_sha256,
        "本seed历史HF架构视图SHA256": source.hf_checkpoint_sha256,
        "校正实际轮次": 200, "联合实际轮次": 200,
        "原校正预算上限": 1500, "原受限联合预算上限": 500,
        "初始HF合法选分_摄氏度": 2.0,
        "观测最佳HF合法选分_摄氏度": 2.0,
        "观测最佳全局轮次": 0,
        "物理最佳独立损失": 0.04,
        "物理最佳全局轮次": 0,
        "LF合法验证初态Cu_SiC节点及真实体积RMSE_摄氏度": {
            "Cu": {"node": 3.3274873948763086, "volume": 2.812075453948512},
            "SiC": {"node": 9.356307395999307, "volume": 7.533927168970277},
        },
        "LF合法验证本轮Cu_SiC节点及真实体积RMSE_摄氏度": {
            "Cu": {"node": 3.3274873948763086, "volume": 2.812075453948512},
            "SiC": {"node": 9.356307395999307, "volume": 7.533927168970277},
        },
        "LF逐材料节点和真实体积5%护栏": {
            "LF两材料节点与真实体积均守住5%护栏": True,
        },
        "LF保持资格": "最近完成的合法LF四口径检查守住；仍须真实阶段完成和独立审核",
        "旧test_Data温度标签读取": False,
        "真实最后状态": "阶段_训练末.pt",
        "累计实际消耗": history[-1]["累计实际消耗"],
        "最近一轮真实入场证据": history[-1],
    }
    (run / "阶段报告.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return run


def test_qualified_static_completed_run_tracks_every_stage_original_and_view_without_energy(
    tmp_path, paired_sources, monkeypatch,
) -> None:
    audit = _auditor()
    run = _synthetic_completed_run(tmp_path, paired_sources)
    monkeypatch.setattr(audit, "TASK07_RUN_ROOT", tmp_path, raising=False)
    qualified = audit.validate_completed_run(run, seed=0)
    assert qualified["运行种子"] == 0
    assert qualified["校正实际轮次"] == qualified["联合实际轮次"] == 200
    assert qualified["状态文件"]["best"] == run / "best.pt"
    assert qualified["状态文件"]["final"] == run / "阶段_训练末.pt"
    assert qualified["阶段SHA256"]["观测最佳完整状态"] == sha256_file(
        run / "阶段_观测最佳.pt"
    )
    assert not (tmp_path / "伪造名义能源指标").exists()


def test_joint_preflight_may_consume_cpu_torch_rng_without_replacing_hf_momentum(
    tmp_path, paired_sources, monkeypatch,
) -> None:
    audit = _auditor()
    run = _synthetic_completed_run(tmp_path, paired_sources)
    monkeypatch.setattr(audit, "TASK07_RUN_ROOT", tmp_path, raising=False)
    joint = torch.load(run / "阶段_联合初始.pt", map_location="cpu", weights_only=False)
    joint["random_state"]["torch_cpu"] = joint["random_state"]["torch_cpu"].clone()
    joint["random_state"]["torch_cpu"][1] ^= 1
    torch.save(joint, run / "阶段_联合初始.pt")
    qualified = audit.validate_completed_run(run, seed=0)
    assert qualified["校正实际轮次"] == 200
    assert qualified["联合实际轮次"] == 200


def _synthetic_30_point_watts(audit):
    from sic_cu.eval.energy_v5 import chinese_energy_terms

    powers, times, orders = audit._fixed_schedule()
    energy, divergence = [], []
    for power in powers:
        for time_s in times:
            for order in orders:
                absorbed, storage = power / 2, power / 4
                loss = power / 8
                balance = storage + loss - absorbed
                denominator = max(abs(absorbed), abs(storage), abs(loss), 1e-12)
                v, j, d = balance + 1.0, 0.5, 0.5
                identity = {
                    "power_w": power, "time_s": time_s, "quadrature_order": order,
                }
                energy.append({
                    **identity, "absorbed_power_w": absorbed,
                    "storage_rate_w": storage, "cooling_heat_w": loss,
                    "convection_heat_w": 0.0, "radiation_heat_w": 0.0,
                    "balance_w": balance, "relative_balance_denominator_w": denominator,
                    "relative_balance": balance / denominator,
                    "boundary_gradient_mode": "inner_autodiff_at_R_minus_epsilon",
                    "outer_epsilon_m": 1e-6,
                })
                divergence.append({
                    **identity, "integrated_pde_residual_w": v,
                    "interface_two_sided_flux_w": j,
                    "boundary_flux_residual_w": d,
                    "explained_engineering_balance_w": v - j - d,
                    "engineering_balance_w": balance,
                    "engineering_explanation_gap_w": balance - (v - j - d),
                })
    mapped, summary = chinese_energy_terms(pl.DataFrame(energy), 64)
    frame = pl.DataFrame(divergence).filter(pl.col("quadrature_order") == 64).select(
        pl.col("power_w").alias("功率_瓦"), pl.col("time_s").alias("时刻_秒"),
        pl.col("integrated_pde_residual_w").alias("体积分残差V_瓦"),
        pl.col("interface_two_sided_flux_w").alias("界面双侧通量J_瓦"),
        pl.col("boundary_flux_residual_w").alias("边界失配D_瓦"),
        pl.col("engineering_balance_w").alias("原工程平衡_瓦"),
        pl.col("explained_engineering_balance_w").alias("解释工程平衡_瓦"),
        pl.col("engineering_explanation_gap_w").alias("工程解释剩余差_瓦"),
    )
    summary.update({
        "原定义相对平衡分母已保持": True,
        "工程平衡与原瓦数逐行一致": True,
        "最大相邻阶变化对吸收功率比": 0.0,
        "最大分解剩余差_瓦": 0.0,
    })
    return {"指标明细": mapped, "物理分解": frame, "汇总": summary,
            "原始能量": energy, "原始散度": divergence}


def test_formal_energy_raw_30x16_and_64_crosschecks_original_watts_and_vjd() -> None:
    audit = _auditor()
    result = audit._verify_energy_raw(_synthetic_30_point_watts(audit))
    assert result["16阶V-J-D散度积分剩余差最大_瓦"] == 0.0
    assert result["64阶V-J-D散度积分剩余差宏均值_瓦"] == 0.0


@pytest.mark.parametrize("tamper", ["point_shrink", "missing16", "fake16_vjd",
                                      "bad64_denominator", "fake64_watt_balance",
                                      "wrong30_merged_csv", "wrong_physics_csv",
                                      "wrong_original_sign"])
def test_energy_raw_refuses_bad_grid_lower_order_divergence_or_relative_denominator(
    tamper: str,
) -> None:
    audit = _auditor()
    raw = _synthetic_30_point_watts(audit)
    if tamper == "point_shrink":
        raw["原始能量"].pop()
    elif tamper == "missing16":
        raw["原始散度"].pop(0)
    elif tamper == "fake16_vjd":
        raw["原始散度"][0]["explained_engineering_balance_w"] += 10.0
    elif tamper == "bad64_denominator":
        raw["原始能量"][1]["relative_balance_denominator_w"] = 1.0
    elif tamper == "fake64_watt_balance":
        raw["原始能量"][1]["balance_w"] += 1.0
    elif tamper == "wrong30_merged_csv":
        raw["指标明细"] = raw["指标明细"].with_columns(
            (pl.col("原定义相对平衡分母_瓦") + 1).alias("原定义相对平衡分母_瓦")
        )
    elif tamper == "wrong_physics_csv":
        raw["物理分解"] = raw["物理分解"].with_columns(
            (pl.col("解释工程平衡_瓦") + 1).alias("解释工程平衡_瓦")
        )
    else:
        raw["原始能量"][1]["absorbed_power_w"] *= -1
    with pytest.raises(ValueError, match="30|双阶|原瓦|分母|散度|V-J-D|来源|一致"):
        audit._verify_energy_raw(raw)


def test_formal_cuda_physical_best_recompute_refuses_cpu_even_when_seed_same(tmp_path) -> None:
    audit = _auditor()
    with pytest.raises(ValueError, match="CUDA|CPU|GPU"):
        audit._recompute_physical_best({"运行目录": tmp_path}, None, torch.device("cpu"))


@pytest.mark.parametrize("tamper", [None, "shrink_ir", "shrink_sensor",
                                      "fake_selected_score", "fake_lf_volume"])
def test_selected_original_state_independently_checks_7272_hf_validation_and_lf10_materials(
    tmp_path, paired_sources, monkeypatch, tamper,
) -> None:
    audit = _auditor()
    run = _synthetic_completed_run(tmp_path, paired_sources)
    monkeypatch.setattr(audit, "TASK07_RUN_ROOT", tmp_path, raising=False)
    qualified = audit.validate_completed_run(run, seed=0)
    reference = qualified["运行报告"]["LF合法验证初态Cu_SiC节点及真实体积RMSE_摄氏度"]

    def fake_ir(split, powers_w):
        assert split == "validation" and list(powers_w) == [115.2, 403.0, 630.5]
        return torch.utils.data.TensorDataset(
            torch.empty(7271 if tamper == "shrink_ir" else 7272, 5)
        )

    def fake_sensor(device, split, powers_w):
        assert split == "validation" and list(powers_w) == [115.2, 403.0, 630.5]
        return (torch.empty(751 if tamper == "shrink_sensor" else 752, 5),)

    def fake_lf(model, powers, device):
        assert list(powers) == [90.0, 170.0, 250.0, 330.0, 410.0,
                                490.0, 570.0, 650.0, 730.0, 800.0]
        materials = {m: value.copy() for m, value in reference.items()}
        if tamper == "fake_lf_volume":
            materials["SiC"]["volume"] += 0.1
        return materials

    monkeypatch.setattr(audit, "_ir_dataset", fake_ir)
    monkeypatch.setattr(audit, "_sensor_tensors", fake_sensor)
    monkeypatch.setattr(audit, "_validation_selection", lambda *args: (
        1.0 if tamper == "fake_selected_score" else 2.0,
        {"顶部": 3.0},
    ))
    monkeypatch.setattr(audit, "_lf_material_validation", fake_lf)
    if tamper is None:
        evidence = audit._recompute_selected_hf_lf(qualified, None, torch.device("cpu"),
                                                    state="best")
        assert evidence["合法HF样本点数"] == 7272
        assert evidence["合法LF验证功率数"] == 10
    else:
        with pytest.raises(ValueError, match="HF|LF|样本|分|材料|真实体积|验证"):
            audit._recompute_selected_hf_lf(qualified, None, torch.device("cpu"),
                                            state="best")


def test_cpu_diagnostic_only_static_does_not_write_false_formal_energy(
    tmp_path, paired_sources, monkeypatch,
) -> None:
    audit = _auditor()
    run = _synthetic_completed_run(tmp_path, paired_sources)
    monkeypatch.setattr(audit, "TASK07_RUN_ROOT", tmp_path, raising=False)
    destination = tmp_path / "静态CPU诊断不能生成正式名义能源"
    monkeypatch.setattr(sys, "argv", ["audit", "--run", str(run), "--seed", "0",
                                          "--state", "best", "--output", str(destination),
                                          "--device", "cpu", "--diagnostic-only"])
    audit.main()
    assert not destination.exists()


def test_formal_five_seed_best_and_true_final_separate_original_outputs_no_overwrite(
    tmp_path, paired_sources, monkeypatch,
) -> None:
    audit = _auditor()
    run = _synthetic_completed_run(tmp_path, paired_sources)
    monkeypatch.setattr(audit, "TASK07_RUN_ROOT", tmp_path, raising=False)
    qualified = audit.validate_completed_run(run, seed=0)
    monkeypatch.setattr(audit, "validate_completed_run", lambda selected, seed: qualified)
    monkeypatch.setattr(audit.torch.cuda, "is_available", lambda: True)

    class StubModel:
        def requires_grad_(self, flag):
            return self

        def to(self, dtype=None):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(audit, "_load_selected_model", lambda *args, **kwargs: StubModel())
    monkeypatch.setattr(audit, "_recompute_selected_hf_lf", lambda *args, **kwargs: {
        "合法HF原macro_v1选分独立重算_摄氏度": 2.0,
        "合法HF样本点数": 7272, "合法HF传感器点数": 752,
        "合法LF验证功率数": 10,
        "LF逐材料节点和真实体积5%原护栏实际资格": True,
    })
    monkeypatch.setattr(audit, "_recompute_physical_best", lambda *args: {
        "pde": 0.01, "initial": 0, "boundary": 0.02,
        "interface": 0.01, "physics_total": 0.04,
    })
    monkeypatch.setattr(audit, "audit_schedule_energy", lambda *args, **kwargs: (
        _synthetic_30_point_watts(audit)
    ))
    monkeypatch.setattr(audit, "load_resolved_boundary_conditions", lambda *args: None)
    out_best, out_final = tmp_path / "seed0_best_energy", tmp_path / "seed0_final_energy"
    best = audit.audit_completed_state(run, seed=0, state="best", output=out_best,
                                       device_name="cuda")
    final = audit.audit_completed_state(run, seed=0, state="final", output=out_final,
                                        device_name="cuda")
    assert best["选择模型SHA256"] == sha256_file(run / "best.pt")
    assert final["选择模型SHA256"] == sha256_file(run / "阶段_训练末.pt")
    assert best["选择模型SHA256"] != final["选择模型SHA256"]
    assert best["审核状态"] == "best" and final["审核状态"] == "final"
    for output in (out_best, out_final):
        assert set(path.name for path in output.iterdir()) == {
            "指标明细.csv", "物理分解.csv", "原始能量.jsonl", "原始散度.jsonl",
            "汇总指标.json", "审计工件SHA256.json",
        }
        report = json.loads((output / "汇总指标.json").read_text(encoding="utf-8"))
        assert report["16阶V-J-D散度积分剩余差最大_瓦"] == 0.0
        assert report["64阶V-J-D散度积分剩余差最大_瓦"] == 0.0
        assert report["名义能量不证明原FEM热预算或内部温度真值"] is True
        manifest = json.loads((output / "审计工件SHA256.json").read_text())
        assert all(manifest[path.name] == sha256_file(path) for path in output.iterdir()
                   if path.name in manifest)
    with pytest.raises(FileExistsError, match="覆盖|已存在|覆盖|已有"):
        audit.audit_completed_state(run, seed=0, state="best", output=out_best,
                                    device_name="cuda")
    original_physical = audit._recompute_physical_best

    def replace_historical_joint_before_writer(*args):
        stage = run / "阶段_联合初始.pt"
        stage.write_bytes(stage.read_bytes() + b"tampered after validation")
        return original_physical(*args)

    monkeypatch.setattr(audit, "_recompute_physical_best", replace_historical_joint_before_writer)
    future = tmp_path / "测量中原件改变必须全拒绝"
    with pytest.raises(ValueError, match="阶段|原件|被改动|更改"):
        audit.audit_completed_state(run, seed=0, state="best", output=future,
                                    device_name="cuda")
    assert not future.exists()


@pytest.mark.parametrize("tamper", ["missing_joint_terminal", "wrong_checkpoint_registry",
                                      "bad_best_view", "replaced_final_adamw",
                                      "missing_history_epoch"])
def test_cross_stage_static_audit_rejects_alias_hash_view_and_real_budget_tamper(
    tmp_path, paired_sources, monkeypatch, tamper,
) -> None:
    audit = _auditor()
    run = _synthetic_completed_run(tmp_path, paired_sources)
    monkeypatch.setattr(audit, "TASK07_RUN_ROOT", tmp_path, raising=False)
    if tamper == "missing_joint_terminal":
        (run / "阶段_联合末.pt").unlink()
    elif tamper == "wrong_checkpoint_registry":
        stage = torch.load(run / "阶段_训练末.pt", map_location="cpu", weights_only=False)
        stage["metadata"]["正式预登记配置SHA256"] = "0" * 64
        torch.save(stage, run / "阶段_训练末.pt")
    elif tamper == "bad_best_view":
        view = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        view["model_state"]["correction.0.weight"][0, 0] += 0.1
        torch.save(view, run / "best.pt")
    elif tamper == "replaced_final_adamw":
        stage = torch.load(run / "阶段_训练末.pt", map_location="cpu", weights_only=False)
        stage["optimizer_state"]["state"] = {}
        torch.save(stage, run / "阶段_训练末.pt")
    else:
        rows = (run / "training.jsonl").read_text(encoding="utf-8").splitlines()
        rows.pop(100)
        (run / "training.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError), match="末|日志|SHA|阶段|AdamW|模型|来源"):
        audit.validate_completed_run(run, seed=0)
