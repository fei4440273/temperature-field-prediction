from __future__ import annotations

import copy
import json
import random
import runpy
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.splits import build_power_splits
from sic_cu.train import multifidelity, task04_joint


DECISION = PROJECT_ROOT / "研究记录/任务03_低保真精度修复/验收判定_预算门禁复核.json"
TASK04_CONFIG = PROJECT_ROOT / "研究记录/任务04_联合微调/有效运行配置.yaml"
SOURCE = PROJECT_ROOT / (
    "研究记录/任务03_低保真精度修复/"
    "任务03_HF旧LF同源_种子0_20260915T184956+0800/阶段_校正末.pt"
)


def test_source_rejects_model_only_best_and_unregistered_full_state() -> None:
    state, architecture = task04_joint.validate_task04_source(SOURCE, DECISION)
    assert state["stage"] == "correction" and state["epoch"] == 300
    assert architecture["method"] == "multifidelity_correction"
    with pytest.raises(ValueError, match="登记|校正末"):
        task04_joint.validate_task04_source(SOURCE.with_name("阶段_观测最佳.pt"), DECISION)
    with pytest.raises(ValueError, match="登记|校正末"):
        task04_joint.validate_task04_source(SOURCE.with_name("best.pt"), DECISION)


def test_registered_architecture_best_hash_is_locked_independently_of_real_terminal(monkeypatch) -> None:
    real_hash = task04_joint.sha256_file

    def forged_architecture_hash(path):
        if Path(path).name == "best.pt":
            return "任03源结构意外改变"
        return real_hash(path)

    monkeypatch.setattr(task04_joint, "sha256_file", forged_architecture_hash)
    with pytest.raises(ValueError, match="best|架构|最佳|登记"):
        task04_joint.validate_task04_source(SOURCE, DECISION)


def test_pre_registered_task04_contract_rejects_any_projection_or_lr_drift() -> None:
    registered = task04_joint.load_yaml(str(TASK04_CONFIG))
    origin = json.loads(DECISION.read_text(encoding="utf-8"))
    task04_joint.validate_task04_registration(registered, origin, build_power_splits())
    for changed in (
        {"LF可解冻模块": ["branch", "trunk"]},
        {"校正器学习率": 0.0002},
        {"物理整包配点_每轮": 128},
        {"HF传感器绝对与温升损失权重": [1.0, 1.0]},
        {"先导后续轮次_每臂": 201},
        {"HF合法验证功率_瓦": [169.0, 339.0, 634.0]},
    ):
        with pytest.raises(ValueError, match="预登记|运行配置"):
            task04_joint.validate_task04_registration({**registered, **changed}, origin, build_power_splits())


def test_source_requires_real_hf_optimizer_and_four_random_sources(tmp_path) -> None:
    payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    payload["optimizer_state"] = copy.deepcopy(payload["optimizer_state"])
    payload["optimizer_state"]["state"] = {}
    corrupt = tmp_path / "阶段_校正末.pt"
    torch.save(payload, corrupt)
    with pytest.raises(ValueError, match="优化器"):
        task04_joint.validate_task04_source(corrupt, DECISION, require_registration=False)
    payload["optimizer_state"] = torch.load(SOURCE, map_location="cpu", weights_only=False)["optimizer_state"]
    payload["random_state"] = dict(payload["random_state"])
    payload["random_state"].pop("numpy")
    torch.save(payload, corrupt)
    with pytest.raises(ValueError, match="随机"):
        task04_joint.validate_task04_source(corrupt, DECISION, require_registration=False)


@pytest.mark.parametrize("name", ("geometry", "materials", "boundary_conditions", "splits"))
def test_source_rejects_semantically_changed_operational_yaml_without_locking_ledger_text(
    monkeypatch, name,
) -> None:
    original = task04_joint.load_yaml

    def fake_yaml(path):
        content = original(path)
        if path == f"configs/{name}.yaml":
            return {**content, "额外禁用结构变更": 1}
        return content

    monkeypatch.setattr(task04_joint, "load_yaml", fake_yaml)
    with pytest.raises(ValueError, match="几何|材料|物理|划分|快照"):
        task04_joint.validate_task04_source(SOURCE, DECISION)


def test_fork_preserves_identical_model_hf_momentum_and_rng_but_only_projection_unfreezes() -> None:
    state, architecture = task04_joint.validate_task04_source(SOURCE, DECISION)
    frozen, frozen_optimizer = task04_joint.fork_task04_model(state, architecture, "冻结", torch.device("cpu"))
    frozen_random = (torch.rand(2), np.random.random(2), random.random())
    joint, joint_optimizer = task04_joint.fork_task04_model(state, architecture, "有限解冻", torch.device("cpu"))
    joint_random = (torch.rand(2), np.random.random(2), random.random())
    assert torch.equal(frozen_random[0], joint_random[0])
    assert np.array_equal(frozen_random[1], joint_random[1])
    assert frozen_random[2] == joint_random[2]
    assert all(torch.equal(a, joint.state_dict()[name]) for name, a in frozen.state_dict().items())
    frozen_trainable = {name for name, param in frozen.named_parameters() if param.requires_grad}
    joint_trainable = {name for name, param in joint.named_parameters() if param.requires_grad}
    projection = {
        "low_fidelity_model.branch_projection.weight", "low_fidelity_model.branch_projection.bias",
        "low_fidelity_model.trunk_projection.weight", "low_fidelity_model.trunk_projection.bias",
    }
    assert frozen_trainable == {name for name, _ in frozen.named_parameters() if name.startswith("correction.")}
    assert joint_trainable == frozen_trainable | projection
    assert "low_fidelity_model.bias" not in joint_trainable
    assert len(frozen_optimizer.param_groups) == 1
    assert len(joint_optimizer.param_groups) == 2
    assert [group["lr"] for group in joint_optimizer.param_groups] == [1e-4, 1e-5]
    for (_, a), (_, b) in zip(frozen.correction.named_parameters(), joint.correction.named_parameters()):
        left, right = frozen_optimizer.state[a], joint_optimizer.state[b]
        assert left.keys() == right.keys() == {"step", "exp_avg", "exp_avg_sq"}
        assert all(torch.equal(left[key], right[key]) for key in left)
        assert left["step"].item() > 0


def test_optimizer_momentum_matching_follows_parameter_ids_not_state_dict_order(tmp_path) -> None:
    source, architecture = task04_joint.validate_task04_source(SOURCE, DECISION)
    rearranged = copy.deepcopy(source)
    original_state = rearranged["optimizer_state"]["state"]
    rearranged["optimizer_state"]["state"] = dict(reversed(list(original_state.items())))
    path = tmp_path / "阶段_校正末.pt"
    torch.save(rearranged, path)
    shutil.copy(SOURCE.parent / "best.pt", tmp_path / "best.pt")
    shutil.copytree(SOURCE.parent / "config_snapshot", tmp_path / "config_snapshot")
    checked, _ = task04_joint.validate_task04_source(path, DECISION, require_registration=False)
    model, optimizer = task04_joint.fork_task04_model(
        checked, architecture, "冻结", torch.device("cpu"), source_path=path,
    )
    for parameter, original_id in zip(optimizer.param_groups[0]["params"],
                                      source["optimizer_state"]["param_groups"][0]["params"]):
        assert torch.equal(optimizer.state[parameter]["exp_avg"],
                           source["optimizer_state"]["state"][original_id]["exp_avg"])


def test_protocol_uses_only_twelve_hf_train_three_validation_and_sixty_lf_train() -> None:
    splits = build_power_splits()
    contract = task04_joint.validate_task04_splits(splits)
    assert contract["HF训练功率"] == sorted(splits.hf_train)
    assert contract["HF合法验证功率"] == sorted(splits.hf_validation)
    assert contract["LF仿真回放功率"] == sorted(splits.simulation_train)
    assert len(contract["LF仿真回放功率"]) == 60
    bad = copy.copy(splits)
    object.__setattr__(bad, "simulation_train", bad.simulation_train | bad.simulation_test)
    with pytest.raises(RuntimeError, match="仿真|训练"):
        task04_joint.validate_task04_splits(bad)


def test_lf_keep_guardrail_checks_both_materials_and_real_volume_rmse_independently() -> None:
    baseline = {"Cu": {"node": 1.0, "volume": 2.0},
                "SiC": {"node": 3.0, "volume": 4.0}}
    kept = {"Cu": {"node": 1.04, "volume": 2.08},
            "SiC": {"node": 3.12, "volume": 4.16}}
    accepted = task04_joint.task04_lf_keep_guardrail(baseline, kept)
    assert accepted["LF两材料节点与真实体积均守住5%护栏"] is True
    worsened = copy.deepcopy(kept)
    worsened["Cu"]["volume"] = 2.12
    refused = task04_joint.task04_lf_keep_guardrail(baseline, worsened)
    assert refused["LF两材料节点与真实体积均守住5%护栏"] is False
    assert refused["逐材料逐口径恶化率_百分比"]["Cu"]["volume"] > 5.0
    with pytest.raises(ValueError, match="Cu|SiC|材料"):
        task04_joint.task04_lf_keep_guardrail({"Cu": baseline["Cu"]}, kept)


def test_lf_replay_never_supervises_high_fidelity_or_changes_frozen_parameters() -> None:
    state, architecture = task04_joint.validate_task04_source(SOURCE, DECISION)
    model, optimizer = task04_joint.fork_task04_model(state, architecture, "冻结", torch.device("cpu"))
    coordinates = torch.tensor([[0.01, -0.01, 20.0, 100.0, 1.0], [0.04, -0.015, 80.0, 220.0, 0.0]])
    truth = model(coordinates, fidelity="low").detach() + 2.0
    before = {name: value.clone() for name, value in model.low_fidelity_model.state_dict().items()}
    loss = task04_joint.low_replay_loss(model, coordinates, truth)
    assert loss.item() > 0
    assert loss.requires_grad is False
    optimizer.zero_grad(set_to_none=True)
    assert all(parameter.grad is None for parameter in model.correction.parameters())
    ((model(coordinates) - truth) / 250.0).square().mean().backward()
    optimizer.step()
    assert all(torch.equal(value, model.low_fidelity_model.state_dict()[name]) for name, value in before.items())
    assert any(parameter.grad is not None for parameter in model.correction.parameters())


def test_continuation_rejects_historical_best_as_optimizer_source(tmp_path) -> None:
    for name in ("阶段_观测最佳.pt", "阶段_物理最佳.pt", "best.pt", "阶段_联合末.pt"):
        with pytest.raises(ValueError, match="最近|初始"):
            task04_joint.validate_task04_resume_name(tmp_path / name)
    assert task04_joint.validate_task04_resume_name(tmp_path / "阶段_最近.pt") is None
    assert task04_joint.validate_task04_resume_name(tmp_path / "阶段_初始.pt") is None


def test_joint_lf_replay_updates_only_projection_and_not_hf_correction() -> None:
    state, architecture = task04_joint.validate_task04_source(SOURCE, DECISION)
    model, optimizer = task04_joint.fork_task04_model(state, architecture, "有限解冻", torch.device("cpu"))
    coordinates = torch.tensor([[0.01, -0.01, 20.0, 100.0, 1.0], [0.04, -0.015, 80.0, 220.0, 0.0]])
    truth = model(coordinates, fidelity="low").detach() + 20.0
    before = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer.zero_grad(set_to_none=True)
    loss = task04_joint.low_replay_loss(model, coordinates, truth)
    loss.backward()
    assert all(parameter.grad is None for parameter in model.correction.parameters())
    optimizer.step()
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items()
               if name.startswith("low_fidelity_model.") and "projection." in name)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items()
               if name.startswith("low_fidelity_model.") and "projection." not in name)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items()
               if name.startswith("correction."))


def test_metadata_lists_all_frozen_lf_names_and_explains_new_shuffle_order() -> None:
    source, _ = task04_joint.validate_task04_source(SOURCE, DECISION)
    origin = json.loads(DECISION.read_text(encoding="utf-8"))["任04共同阶段起点"]
    kwargs = dict(initial_score=1.0, best=1.0, best_epoch=0, physical=2.0,
                  physical_epoch=0, cumulative={}, source_hf_step=4800.0,
                  source_parameter_names=source["parameter_requires_grad"],
                  config_sha=task04_joint.sha256_file(TASK04_CONFIG))
    frozen = task04_joint._stage_metadata(origin, "冻结", **kwargs)
    joint = task04_joint._stage_metadata(origin, "有限解冻", **kwargs)
    origin_names = {name for name in source["parameter_requires_grad"] if name.startswith("low_fidelity_model.")}
    assert set(frozen["LF冻结参数"]) == origin_names
    assert set(joint["LF冻结参数"]) == origin_names - task04_joint.PROJECTION_NAMES
    assert "新" in frozen["两臂随机源说明"]
    assert "不声称恢复任03原数据顺序" in frozen["两臂随机源说明"]


def test_diagnostic_fork_cannot_be_promoted_to_formal_resume() -> None:
    source, _ = task04_joint.validate_task04_source(SOURCE, DECISION)
    origin = json.loads(DECISION.read_text(encoding="utf-8"))["任04共同阶段起点"]
    kwargs = dict(initial_score=1.0, best=1.0, best_epoch=0, physical=2.0,
                  physical_epoch=0, cumulative={}, source_hf_step=4800.0,
                  source_parameter_names=source["parameter_requires_grad"],
                  config_sha=task04_joint.sha256_file(TASK04_CONFIG))
    diagnostic = task04_joint._stage_metadata(origin, "有限解冻", **kwargs, diagnostic_only=True)
    assert diagnostic["运行资格"] == "短诊断；不参与任-04正式200轮采用"
    with pytest.raises(ValueError, match="诊断|资格"):
        task04_joint.validate_task04_resume_qualification(diagnostic, diagnostic_only=False)
    assert task04_joint.validate_task04_resume_qualification(diagnostic, diagnostic_only=True) is None
    formal = task04_joint._stage_metadata(origin, "冻结", **kwargs, diagnostic_only=False)
    assert task04_joint.validate_task04_resume_qualification(formal, diagnostic_only=False) is None


def test_model_view_records_new_lineage_validation_and_distinct_terminal_arm(tmp_path) -> None:
    source, architecture = task04_joint.validate_task04_source(SOURCE, DECISION)
    model, _ = task04_joint.fork_task04_model(source, architecture, "冻结", torch.device("cpu"))
    path = tmp_path / "best.pt"
    task04_joint._real_model_view(
        path, architecture, model, epoch=0, score=2.3, arm="冻结",
        source_sha=json.loads(DECISION.read_text(encoding="utf-8"))["任04共同阶段起点"]["SHA256"],
        config_sha=task04_joint.sha256_file(TASK04_CONFIG),
        top_rmse=3.1, sensor_validation={"absolute_rmse_c": 1.1, "delta_rmse_c": 0.5},
    )
    view = torch.load(path, map_location="cpu", weights_only=False)
    assert view["epoch"] == 300 and view["validation_rmse_c"] == 3.1
    assert view["validation_selection_score_c"] == 2.3
    assert view["validation_sensor"]["absolute_rmse_c"] == 1.1
    lineage = view["任04新模型视图来源"]
    assert lineage["源校正末SHA256"] == json.loads(DECISION.read_text(
        encoding="utf-8"
    ))["任04共同阶段起点"]["SHA256"]
    assert lineage["任04预登记配置SHA256"] == task04_joint.sha256_file(TASK04_CONFIG)
    assert lineage["原校正阶段旧best轮次"] == architecture["epoch"]
    assert lineage["任04本阶段实际轮次"] == 0
    assert lineage["HF温度标签训练功率"] == sorted(build_power_splits().hf_train)
    assert lineage["HF温度标签只作验证功率"] == sorted(build_power_splits().hf_validation)
    assert lineage["旧test_Data温度标签读取"] is False
    assert lineage["LF权重冻结名单"] and lineage["LF权重更新名单"] == []
    assert lineage["任04新源码SHA256"]
    assert task04_joint.task04_terminal_name("冻结") == "阶段_HF续训冻结LF末.pt"
    assert task04_joint.task04_terminal_name("有限解冻") == "阶段_联合末.pt"


def test_one_real_cpu_epoch_is_recoverable_and_never_reads_hf_test_temperatures(tmp_path, monkeypatch) -> None:
    observed_splits = []
    observed_sensor_splits = []
    original = multifidelity.load_processed_ir_observations
    original_sensors = multifidelity.load_canonical_sensor_observations

    def controlled_ir_loader(split=None):
        observed_splits.append(split)
        if split not in ("train", "validation"):
            raise AssertionError("任-04不得读取旧test_Data温度")
        return original(split)

    monkeypatch.setattr(multifidelity, "load_processed_ir_observations", controlled_ir_loader)

    def controlled_sensor_loader(*, split):
        observed_sensor_splits.append(split)
        if split not in ("train", "validation"):
            raise AssertionError("任-04不得读取旧test_Data环温标签")
        return original_sensors(split=split)

    monkeypatch.setattr(multifidelity, "load_canonical_sensor_observations", controlled_sensor_loader)
    result = task04_joint.run_task04_joint(
        arm="冻结", output_directory=tmp_path / "任务04_CPU冻结门禁",
        budget_epochs=200, session_epoch_limit=1, diagnostic_only=True,
        device_name="cpu", source_checkpoint=SOURCE, decision_path=DECISION,
    )
    output = tmp_path / "任务04_CPU冻结门禁"
    assert result["状态"] == "短门禁暂停，不得用于200轮正式采用"
    historical = json.loads(DECISION.read_text(encoding="utf-8"))["历史部署B0参考"]
    assert result["历史部署B0参考"]["SHA256"] == historical["SHA256"]
    assert result["历史部署B0参考"]["原始合法验证macro_v1_摄氏度"] == historical[
        "原始合法验证macro_v1_摄氏度"
    ]
    assert result["历史部署B0参考"]["不得被本次更差的300轮续训模型替换"] is True
    assert observed_splits and set(observed_splits) == {"train", "validation"}
    assert observed_sensor_splits and set(observed_sensor_splits) == {"train", "validation"}
    assert not (output / "阶段_联合末.pt").exists()
    initial = torch.load(output / "阶段_初始.pt", map_location="cpu", weights_only=False)
    latest = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert initial["epoch"] == 0 and latest["epoch"] == 1
    assert initial["metadata"]["源校正末SHA256"] == latest["metadata"]["源校正末SHA256"]
    assert initial["metadata"]["任04预登记配置SHA256"] == task04_joint.sha256_file(TASK04_CONFIG)
    model_view = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert model_view["任04新模型视图来源"]["任04预登记配置SHA256"] == task04_joint.sha256_file(TASK04_CONFIG)
    assert latest["budget"] == {"任04联合续训轮次": 200}
    assert len(latest["optimizer_state"]["state"]) == 10
    assert all(not flag for name, flag in latest["parameter_requires_grad"].items()
               if name.startswith("low_fidelity_model."))
    assert (output / "阶段_观测最佳.pt").is_file()
    assert (output / "阶段_物理最佳.pt").is_file()
    record = [__import__("json").loads(row) for row in (output / "training.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()]
    assert len(record) == 1
    assert record[0]["epoch"] == 1
    assert record[0]["HF观测优化步"] == 15 and record[0]["物理优化步"] == 1
    assert record[0]["LF回放监督模式"] == "low"
    assert record[0]["LF仿真回放训练点"] == 60 * 2048
    assert record[0]["LF仿真回放Cu点"] > 0 and record[0]["LF仿真回放SiC点"] > 0
    assert record[0]["LF冻结权重符合源状态"] is True
    assert record[0]["LF最后投影实际变化张量数"] == 0
    assert record[0]["LF回放计算墙钟秒"] > 0
    assert set(record[0]["HF合法验证分模态RMSE_摄氏度"]) == {"top", "hot", "cold"}
    assert set(record[0]["LF合法验证材料RMSE_摄氏度"]) == {"Cu", "SiC"}
    assert "独立局部物理损失" in record[0] and "能量四点绝对平衡均值_瓦" in record[0]
    continued = task04_joint.run_task04_joint(
        arm="冻结", output_directory=output, budget_epochs=200,
        session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
        source_checkpoint=SOURCE, decision_path=DECISION,
        resume_training_checkpoint=output / "阶段_最近.pt",
    )
    assert continued["本臂实际完成轮次"] == 2
    after = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert after["epoch"] == 2
    assert next(iter(after["optimizer_state"]["state"].values()))["step"].item() == 4832
    assert [json.loads(row)["epoch"] for row in (output / "training.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()] == [1, 2]
    with pytest.raises(ValueError, match="短诊断不得转正式"):
        task04_joint.run_task04_joint(
            arm="冻结", output_directory=output, budget_epochs=200,
            session_epoch_limit=200, diagnostic_only=False, device_name="cpu",
            source_checkpoint=SOURCE, decision_path=DECISION,
            resume_training_checkpoint=output / "阶段_最近.pt",
        )
    assert [json.loads(row)["epoch"] for row in (output / "training.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()] == [1, 2]
    original_hash = task04_joint.sha256_file

    def changed_configuration_hash(path):
        if str(path).endswith("configs/geometry.yaml"):
            return "变更后禁止伪装相同配置"
        return original_hash(path)

    monkeypatch.setattr(task04_joint, "sha256_file", changed_configuration_hash)
    with pytest.raises(ValueError, match="配置改变"):
        task04_joint.run_task04_joint(
            arm="冻结", output_directory=output, budget_epochs=200,
            session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
            source_checkpoint=SOURCE, decision_path=DECISION,
            resume_training_checkpoint=output / "阶段_最近.pt",
        )


def test_cli_rejects_short_budget_as_formal_candidate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "19_run_task04_joint.py", "--arm", "冻结", "--output", str(tmp_path / "不得启动"),
        "--session-epoch-limit", "1", "--device", "cpu",
    ])
    with pytest.raises(ValueError, match="diagnostic_only|短CPU门禁"):
        runpy.run_path(str(PROJECT_ROOT / "scripts/19_run_task04_joint.py"), run_name="__main__")
    assert not (tmp_path / "不得启动").exists()


def test_lf_volume_validation_rejects_changed_material_mesh_before_reusing_first_weights(monkeypatch) -> None:
    source, architecture = task04_joint.validate_task04_source(SOURCE, DECISION)
    model, _ = task04_joint.fork_task04_model(source, architecture, "冻结", torch.device("cpu"))
    powers = sorted(build_power_splits().simulation_validation)[:2]
    original_field = task04_joint.load_processed_field

    def mismatched_field(power):
        field = original_field(power)
        if power == powers[1]:
            material = field.material_ids.copy()
            material[0] = 1 - material[0]
            return replace(field, material_ids=material)
        return field

    monkeypatch.setattr(task04_joint, "load_processed_field", mismatched_field)
    with pytest.raises(ValueError, match="网格|材料|compatible"):
        task04_joint._lf_material_validation(model, powers, torch.device("cpu"))


def test_joint_one_real_cpu_epoch_records_only_four_lf_projection_changes(tmp_path) -> None:
    result = task04_joint.run_task04_joint(
        arm="有限解冻", output_directory=tmp_path / "任务04_CPU有限解冻门禁",
        budget_epochs=200, session_epoch_limit=1, diagnostic_only=True,
        device_name="cpu", source_checkpoint=SOURCE, decision_path=DECISION,
    )
    output = tmp_path / "任务04_CPU有限解冻门禁"
    initial = torch.load(output / "阶段_初始.pt", map_location="cpu", weights_only=False)
    latest = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert result["状态"] == "短门禁暂停，不得用于200轮正式采用"
    assert all(torch.equal(a, latest["model_state"][name]) for name, a in initial["model_state"].items()
               if name.startswith("low_fidelity_model.") and name not in task04_joint.PROJECTION_NAMES)
    changed = {name for name, a in initial["model_state"].items()
               if name.startswith("low_fidelity_model.") and not torch.equal(a, latest["model_state"][name])}
    assert changed == task04_joint.PROJECTION_NAMES
    row = json.loads((output / "training.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["LF最后投影实际变化张量数"] == 4
    assert row["LF其余权重实际变化张量数"] == 0
    assert row["LF回放计算墙钟秒"] > 0
    assert len(latest["optimizer_state"]["state"]) == 14


def test_uncommitted_latest_does_not_advance_best_model_view_and_archives_log_tail(
    tmp_path, monkeypatch,
) -> None:
    output = tmp_path / "最近状态提交前异常"
    original_save = task04_joint.save_training_state

    def fail_before_commit(path, model, optimizer, **kwargs):
        if Path(path).name == "阶段_最近.pt" and kwargs["epoch"] == 1:
            raise RuntimeError("注入最近状态提交前异常")
        return original_save(path, model, optimizer, **kwargs)

    monkeypatch.setattr(task04_joint, "save_training_state", fail_before_commit)
    with pytest.raises(RuntimeError, match="提交前异常"):
        task04_joint.run_task04_joint(
            arm="有限解冻", output_directory=output, budget_epochs=200,
            session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
            source_checkpoint=SOURCE, decision_path=DECISION,
        )
    initial = torch.load(output / "阶段_初始.pt", map_location="cpu", weights_only=False)
    view = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert not (output / "阶段_最近.pt").exists()
    assert initial["epoch"] == 0 and view["epoch"] == 300
    assert view["任04新模型视图来源"]["任04本阶段实际轮次"] == 0
    assert all(torch.equal(value, view["model_state"][name]) for name, value in initial["model_state"].items())
    assert [json.loads(row)["epoch"] for row in (output / "training.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()] == [1]
    monkeypatch.setattr(task04_joint, "save_training_state", original_save)
    recovered = task04_joint.run_task04_joint(
        arm="有限解冻", output_directory=output, budget_epochs=200,
        session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
        source_checkpoint=SOURCE, decision_path=DECISION,
        resume_training_checkpoint=output / "阶段_初始.pt",
    )
    assert recovered["本臂实际完成轮次"] == 1
    latest = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert latest["metadata"]["累计实际消耗"]["HF观测优化步"] == 15
    assert [json.loads(row)["epoch"] for row in (output / "training.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()] == [1]
    assert list(output.glob("training_中断尾行_*.jsonl"))


def test_committed_latest_reconstructs_missing_observation_physical_and_legal_view_before_next_epoch(
    tmp_path, monkeypatch,
) -> None:
    output = tmp_path / "最近已提交_最佳未提交"
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source_last = source["model_state"]["correction.8.weight"]
    real_guardrails = task04_joint.evaluate_schedule_guardrails

    def deterministic_physics(model, *args, **kwargs):
        # Fault injection makes the independent physical best occur at the same real committed epoch.
        measured = real_guardrails(model, *args, **kwargs)
        measured["独立局部物理损失"]["physics_total"] = (
            0.5 if torch.equal(model.state_dict()["correction.8.weight"].detach().cpu(), source_last)
            else 0.01
        )
        return measured

    monkeypatch.setattr(task04_joint, "evaluate_schedule_guardrails", deterministic_physics)
    real_save = task04_joint.save_training_state

    def break_sidecar(path, model, optimizer, **kwargs):
        if Path(path).name == "阶段_观测最佳.pt" and kwargs["epoch"] == 1:
            raise RuntimeError("注入最新提交后最佳副本中断")
        return real_save(path, model, optimizer, **kwargs)

    monkeypatch.setattr(task04_joint, "save_training_state", break_sidecar)
    with pytest.raises(RuntimeError, match="最佳副本中断"):
        task04_joint.run_task04_joint(
            arm="有限解冻", output_directory=output, budget_epochs=200,
            session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
            source_checkpoint=SOURCE, decision_path=DECISION,
        )
    latest = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert latest["epoch"] == latest["metadata"]["观测最佳任04轮次"] == 1
    assert latest["metadata"]["物理最佳任04轮次"] == 1
    assert torch.load(output / "阶段_观测最佳.pt", map_location="cpu", weights_only=False)["epoch"] == 0
    assert torch.load(output / "阶段_物理最佳.pt", map_location="cpu", weights_only=False)["epoch"] == 0
    assert torch.load(output / "best.pt", map_location="cpu", weights_only=False)["epoch"] == 300
    monkeypatch.setattr(task04_joint, "save_training_state", real_save)
    real_loader = task04_joint.DataLoader

    def stop_before_next_epoch(*args, **kwargs):
        if kwargs.get("shuffle") is True:
            raise RuntimeError("注入恢复完成后第2轮开始中断")
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(task04_joint, "DataLoader", stop_before_next_epoch)
    with pytest.raises(RuntimeError, match="第2轮开始中断"):
        task04_joint.run_task04_joint(
            arm="有限解冻", output_directory=output, budget_epochs=200,
            session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
            source_checkpoint=SOURCE, decision_path=DECISION,
            resume_training_checkpoint=output / "阶段_最近.pt",
        )
    assert torch.equal(torch.get_rng_state(), latest["random_state"]["torch_cpu"])
    assert random.getstate() == latest["random_state"]["python"]
    assert np.array_equal(np.random.get_state()[1], latest["random_state"]["numpy"][1])
    observed = torch.load(output / "阶段_观测最佳.pt", map_location="cpu", weights_only=False)
    physical = torch.load(output / "阶段_物理最佳.pt", map_location="cpu", weights_only=False)
    view = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert observed["epoch"] == physical["epoch"] == 1
    assert view["epoch"] == 301 and view["任04新模型视图来源"]["任04本阶段实际轮次"] == 1
    assert view["任04新模型视图来源"]["运行资格"] == latest["metadata"]["运行资格"]
    assert all(torch.equal(value, observed["model_state"][name]) and
               torch.equal(value, physical["model_state"][name]) and
               torch.equal(value, view["model_state"][name]) for name, value in latest["model_state"].items())
    assert all(torch.equal(left["exp_avg"], right["exp_avg"])
               for left, right in zip(latest["optimizer_state"]["state"].values(),
                                      observed["optimizer_state"]["state"].values()))
    assert torch.equal(latest["random_state"]["torch_cpu"], observed["random_state"]["torch_cpu"])
    assert observed["metadata"]["观测最佳选分_摄氏度"] == latest["metadata"]["观测最佳选分_摄氏度"]
    assert physical["metadata"]["物理最佳独立损失"] == latest["metadata"]["物理最佳独立损失"]
    monkeypatch.setattr(task04_joint, "DataLoader", real_loader)
    _, _, model = multifidelity._load_multifidelity_model(output / "best.pt", torch.device("cpu"))
    ir = multifidelity._evaluate_ir_model(model, "validation", torch.device("cpu"), build_power_splits().hf_validation)
    sensors = multifidelity._evaluate_sensor_model(model, "validation", torch.device("cpu"), build_power_splits().hf_validation)
    assert abs(view["validation_rmse_c"] - ir["rmse_c"]) < 1e-4
    assert abs(view["validation_sensor"]["absolute_rmse_c"] - sensors["absolute_rmse_c"]) < 1e-4
    assert [json.loads(line)["epoch"] for line in (output / "training.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()] == [1]
    assert latest["metadata"]["累计实际消耗"]["HF观测优化步"] == 15
    tampered = copy.deepcopy(view)
    tampered["validation_sensor"]["absolute_rmse_c"] += 2.0
    torch.save(tampered, output / "best.pt")
    monkeypatch.setattr(task04_joint, "DataLoader", stop_before_next_epoch)
    with pytest.raises(ValueError, match="模型视图|best.pt|传感器"):
        task04_joint.run_task04_joint(
            arm="有限解冻", output_directory=output, budget_epochs=200,
            session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
            source_checkpoint=SOURCE, decision_path=DECISION,
            resume_training_checkpoint=output / "阶段_最近.pt",
        )


def test_missing_earlier_best_full_state_refuses_resume_and_preserves_original_view(tmp_path, monkeypatch) -> None:
    output = tmp_path / "旧最佳原件不能伪造"
    original_guardrails = task04_joint.evaluate_schedule_guardrails
    original_select = task04_joint._validation_selection
    source_weight = torch.load(SOURCE, map_location="cpu", weights_only=False)["model_state"]["correction.8.weight"]
    selected_calls = 0
    first_epoch_weight = None

    def stable_physical_best(model, *args, **kwargs):
        measured = original_guardrails(model, *args, **kwargs)
        current = model.state_dict()["correction.8.weight"].detach().cpu()
        measured["独立局部物理损失"]["physics_total"] = (
            0.5 if torch.equal(current, source_weight) else
            0.01 if first_epoch_weight is None or torch.equal(current, first_epoch_weight) else 0.5
        )
        return measured

    def keep_observed_best_at_epoch_one(*args, **kwargs):
        nonlocal selected_calls
        selected_calls += 1
        score, values = original_select(*args, **kwargs)
        return (score + 100.0 if selected_calls == 5 else score), values

    monkeypatch.setattr(task04_joint, "evaluate_schedule_guardrails", stable_physical_best)
    monkeypatch.setattr(task04_joint, "_validation_selection", keep_observed_best_at_epoch_one)
    task04_joint.run_task04_joint(
        arm="有限解冻", output_directory=output, budget_epochs=200,
        session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
        source_checkpoint=SOURCE, decision_path=DECISION,
    )
    first_epoch_weight = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)[
        "model_state"
    ]["correction.8.weight"]
    task04_joint.run_task04_joint(
        arm="有限解冻", output_directory=output, budget_epochs=200,
        session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
        source_checkpoint=SOURCE, decision_path=DECISION,
        resume_training_checkpoint=output / "阶段_最近.pt",
    )
    latest = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert latest["epoch"] == 2
    assert latest["metadata"]["观测最佳任04轮次"] == 1
    assert latest["metadata"]["物理最佳任04轮次"] == 1
    assert latest["metadata"]["累计实际消耗"]["HF观测优化步"] == 30
    original_view = (output / "best.pt").read_bytes()
    original_report = (output / "阶段报告.json").read_bytes()
    assert torch.load(output / "best.pt", map_location="cpu", weights_only=False)["epoch"] == 301
    retained = output / "阶段_观测最佳_原件保留.pt"
    (output / "阶段_观测最佳.pt").rename(retained)
    with pytest.raises(ValueError, match="历史|无损|完整.*最佳"):
        task04_joint.run_task04_joint(
            arm="有限解冻", output_directory=output, budget_epochs=200,
            session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
            source_checkpoint=SOURCE, decision_path=DECISION,
            resume_training_checkpoint=output / "阶段_最近.pt",
        )
    assert retained.is_file() and not (output / "阶段_观测最佳.pt").exists()
    assert (output / "best.pt").read_bytes() == original_view
    assert (output / "阶段报告.json").read_bytes() == original_report
    assert [json.loads(line)["epoch"] for line in (output / "training.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()] == [1, 2]
