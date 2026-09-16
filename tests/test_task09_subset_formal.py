"""Formal Task-09 source schedule and honest stage accounting."""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from sic_cu.config import PROJECT_ROOT


def _formal():
    try:
        return importlib.import_module("sic_cu.train.task09_subset_formal")
    except ModuleNotFoundError:
        pytest.fail("Task-09 exclusive formal subset trainer is missing")


@pytest.mark.parametrize("hf_batches", range(3, 13))
def test_every_joint_group_replays_all_sixty_real_2048_lf_batches(hf_batches):
    formal = _formal()
    x = torch.zeros((60 * 2048, 5), dtype=torch.float32)
    x[:, 0] = torch.arange(60 * 2048)
    x[: 30 * 2048, 4] = 0.0
    x[30 * 2048 :, 4] = 1.0
    y = torch.ones((60 * 2048, 1), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x, y), batch_size=2048, shuffle=False)
    groups = list(formal.iter_task09_lf_groups(loader, hf_batches))
    assert len(groups) == hf_batches
    sizes = formal.balanced_task09_lf_groups(hf_batches)
    assert [len(group) for group in groups] == sizes
    assert sum(len(group) for group in groups) == 60
    exposed = torch.cat([block[0][:, 0] for group in groups for block in group])
    assert torch.equal(exposed, torch.arange(60 * 2048, dtype=torch.float32))
    assert sum(len(block[0]) for group in groups for block in group) == 60 * 2048
    assert {int(value) for value in torch.unique(x[:, 4])} == {0, 1}


def test_joint_schedule_rejects_short_replay_or_illegal_hf_batch_count():
    formal = _formal()
    for count in (0, 2, 13, 16):
        with pytest.raises(ValueError, match="HF|批"):
            formal.balanced_task09_lf_groups(count)
    short = DataLoader(TensorDataset(torch.zeros((60 * 2048 - 1, 5)),
                                     torch.ones((60 * 2048 - 1, 1))), batch_size=2048)
    with pytest.raises(ValueError, match="2048|60|LF"):
        list(formal.iter_task09_lf_groups(short, 4))


@pytest.mark.parametrize("hf_batches,group_sizes", [
    (15, [4] * 15), (3, [20] * 3), (4, [15] * 4),
    (7, [9, 9, 9, 9, 8, 8, 8]),
])
def test_uniform_microbatch_lf_loss_matches_old_f3_full_sixty_mean(hf_batches, group_sizes):
    formal = _formal()
    individual_scales = [formal.task09_lf_microbatch_scale(size)
                         for size in group_sizes for _ in range(size)]
    assert len(individual_scales) == 60
    assert individual_scales == [0.25] * 60
    assert sum(individual_scales) == 15.0
    assert [sum(individual_scales[sum(group_sizes[:i]):sum(group_sizes[:i + 1])])
            for i in range(len(group_sizes))] == [size / 4 for size in group_sizes]
    if hf_batches == 15:
        assert [size / 4 for size in group_sizes] == [1.0] * 15


def test_first_subset_formal_entry_never_accepts_cpu_or_unregistered_run():
    formal = _formal()
    with pytest.raises(ValueError, match="CUDA|GPU"):
        formal.require_task09_cuda("cpu")
    with pytest.raises(ValueError, match="序列|子集|登记"):
        formal.validate_task09_run_identity("wrong_arm", 3, 0)
    with pytest.raises(ValueError, match="seed|种子"):
        formal.validate_task09_run_identity("primary", 3, 7)
    assert formal.validate_task09_run_identity("right_center", 9, 4) == (
        "right_center", 9, 4
    )


class _TinyMultifidelity(nn.Module):
    def __init__(self):
        super().__init__()
        self.low_fidelity_model = nn.Linear(1, 1, bias=False)
        self.correction = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.low_fidelity_model.weight)
        nn.init.zeros_(self.correction.weight)
        self.scales = SimpleNamespace(temperature_scale_k=1.0)
        self.high_calls = self.low_calls = 0

    def forward(self, x, *, fidelity):
        if fidelity == "high":
            self.high_calls += 1
            return self.correction(x[:, :1])
        self.low_calls += 1
        return self.low_fidelity_model(x[:, :1])


class _CountSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.0)
        self.gradients = []

    def step(self, *args, **kwargs):
        self.gradients.append(float(self.param_groups[0]["params"][0].grad.item()))
        return super().step(*args, **kwargs)


def test_data_epoch_once_hf_per_step_lf60_uniform_quarter_loss_without_big_concat():
    formal = _formal()
    model = _TinyMultifidelity()
    optimizer = _CountSGD(model.low_fidelity_model.parameters())
    hf_x = torch.zeros((3 * 2048, 5))
    hf_x[:, 0] = 1.0
    hf_x[:, 3] = 55.0
    hf_x[:, 4] = 1.0
    hf_data = TensorDataset(hf_x, torch.zeros((3 * 2048, 1)),
                            torch.ones((3 * 2048, 1)))
    ring_x = torch.zeros((4, 5))
    ring_x[:, 0] = 1.0
    ring_x[:, 3] = 55.0
    ring_x[:2, 0] = 0.028
    ring_x[2:, 0] = 0.0415
    sensors = (ring_x, torch.zeros((4, 1)), torch.zeros((4, 1)),
               torch.tensor([0, 0, 2, 2]))
    lf_x = torch.zeros((60 * 2048, 5))
    lf_x[:, 0] = 1.0
    lf_x[:, 3] = torch.arange(1, 61).repeat_interleave(2048)
    lf_x[: 30 * 2048, 4] = 0.0
    lf_x[30 * 2048 :, 4] = 1.0
    lf_data = TensorDataset(lf_x, torch.ones((60 * 2048, 1)))
    record = formal.task09_data_epoch(
        model, optimizer, hf_data, sensors, torch.device("cpu"),
        seed=0, global_epoch=1, simulation_data=lf_data,
        expected_lf_powers=tuple(float(power) for power in range(1, 61)),
    )
    assert model.high_calls == 6  # one IR and one sensor forward per actual HF batch
    assert model.low_calls == 60   # no giant LF concat and no repeated LF minibatch
    assert optimizer.gradients == [-10.0] * 3
    assert record["HF观测优化步"] == 3
    assert record["HF训练观测点"] == 3 * 2048
    assert record["HF训练传感器点"] == 3 * 4
    assert record["LF联合回放batch"] == 60
    assert record["LF真实回放训练点"] == 60 * 2048
    assert record["LF60每功率真2048已核验"] is True
    assert record["LF每HF步真实分组"] == [20, 20, 20]


def test_canonical_output_uses_exclusive_task09_directory_and_never_existing_artifacts():
    formal = _formal()
    fresh = formal.TASK09_FORMAL_ROOT / "正式F3_primary_HF3_seed0_20260916T080000+0800"
    assert formal.validate_task09_output(fresh, "primary", 3, 0) == fresh
    for forged in (
        PROJECT_ROOT / "研究记录/任务07_正式五种子重训/修复后正式_E0_种子0",
        formal.TASK09_FORMAL_ROOT / "正式F3_primary_HF12_seed0_20260916T080000+0800",
        formal.TASK09_FORMAL_ROOT / "正式F3_right_center_HF3_seed0_20260916T080000+0800",
    ):
        with pytest.raises(ValueError, match="任09|目录|序列|来源"):
            formal.validate_task09_output(forged, "primary", 3, 0)


def test_per_epoch_budget_rejects_short_lf_groups_or_missing_physics_update():
    formal = _formal()
    sample = {"ir_rows": 6868, "ir_batches_2048": 4,
              "sensor_rows": {"Hot": 340, "Cold": 340}}
    complete = {
        "HF训练观测点": 6868, "HF训练传感器点": 4 * 680,
        "HF观测优化步": 4, "物理优化步": 1, "物理配点": 256,
        "LF真实回放训练点": 60 * 2048, "LF联合回放batch": 60,
        "LF_Cu真实回放点": 60 * 2048 // 2,
        "LF_SiC真实回放点": 60 * 2048 // 2,
        "LF60每功率真2048已核验": True,
        "LF每HF步真实分组": [15, 15, 15, 15],
    }
    formal.assert_task09_epoch_budget(complete, sample, joint=True)
    with pytest.raises(ValueError, match="LF|预算"):
        formal.assert_task09_epoch_budget(dict(complete, LF联合回放batch=56),
                                          sample, joint=True)
    with pytest.raises(ValueError, match="物理|配点|步"):
        formal.assert_task09_epoch_budget(dict(complete, 物理优化步=0),
                                          sample, joint=True)
    correction = dict(complete, LF真实回放训练点=0, LF联合回放batch=0,
                      LF_Cu真实回放点=0, LF_SiC真实回放点=0,
                      LF60每功率真2048已核验=False,
                      LF每HF步真实分组=[])
    formal.assert_task09_epoch_budget(correction, sample, joint=False)
    with pytest.raises(ValueError, match="LF|预算"):
        formal.assert_task09_epoch_budget(complete, sample, joint=False)


def test_source_model_immutability_enforces_only_four_lf_joint_projection_tensors():
    formal = _formal()
    sources = formal.load_task09_sources()
    model = formal.fork_task07_initialization(sources, 0, "E0", torch.device("cpu")).model
    source = sources[0]
    formal.check_task09_model(model, source, formal.CORRECTION_STAGE)
    named = dict(model.named_parameters())
    for name in formal.PROJECTION_NAMES:
        named[name].requires_grad_(True)
    formal.check_task09_model(model, source, formal.JOINT_STAGE)
    with torch.no_grad():
        named["low_fidelity_model.branch_projection.bias"].add_(0.01)
    formal.check_task09_model(model, source, formal.JOINT_STAGE)
    with torch.no_grad():
        forbidden = next(name for name in named if name.startswith("low_fidelity_model.")
                         and name not in formal.PROJECTION_NAMES)
        named[forbidden].add_(0.01)
    with pytest.raises(ValueError, match="LF|冻结|其余"):
        formal.check_task09_model(model, source, formal.JOINT_STAGE)
    named[forbidden].requires_grad_(True)
    with pytest.raises(ValueError, match="LF|冻结|投影"):
        formal.check_task09_model(model, source, formal.JOINT_STAGE)


def test_formal_entry_refuses_cpu_even_with_canonical_path_before_writing():
    formal = _formal()
    destination = (formal.TASK09_FORMAL_ROOT /
                   "正式F3_primary_HF3_seed0_20260916T080001+0800")
    with pytest.raises(ValueError, match="CUDA|GPU"):
        formal.run_task09_formal(name="primary", size=3, seed=0,
                                 output_directory=destination,
                                 device_name="cpu")
    assert not destination.exists()


def test_prior_registration_binds_source_sha_exact_subsets_and_all_five_seeds():
    formal = _formal()
    expected = formal.expected_task09_registration()
    assert expected["schema_version"] == 1
    assert expected["任务"] == "任09独立F3子集真实CUDA训练"
    assert expected["原覆盖序列前登记SHA256"] == formal.REGISTRATION_SHA256
    assert expected["V4_B0来源清单SHA256"] == formal.MANIFEST_SHA256
    assert set(expected["五seed原LF配对源"]) == set(range(5))
    assert set(expected["三序列HF功率"]) == {"primary", "left_center", "right_center"}
    assert expected["五seed正式主曲线"] == [0, 1, 2, 3, 4]
    assert expected["HF合法验证不缩减功率"] == [115.2, 403.0, 630.5]
    assert expected["LF每个2048小批同旧07有效权重"] == 0.25
    assert expected["全三序列三新增HF档五seed候选数"] == 45
    assert expected["全覆盖上界总轮次非承诺"] == 90000
    assert expected["主序列三档五seed候选数"] == 15
    assert expected["首批逐个真实先导目标个数"] == 5
    real_source = expected["每臂HF真实标签原数SHA256"]
    assert set(real_source) == {"primary", "left_center", "right_center"}
    for arm in real_source.values():
        assert set(arm) == {3, 6, 9}
        for modality in arm.values():
            assert set(modality) == {"IR", "Hot", "Cold"}
            assert all(len(sha) == 64 for sha in modality.values())
            assert modality["Hot"] != modality["Cold"]
    assert set(expected["HF三功率完整合法验证真实标签原数SHA256"]) == {
        "IR", "Hot", "Cold"
    }
    assert expected["正式源代码原件SHA256"]["trainer"]
    with pytest.raises(ValueError, match="事前|SHA|登记"):
        formal.require_task09_registration(None, None)


def test_formal_cli_requires_registry_and_never_chooses_cpu_or_existing_task07_output():
    cli = PROJECT_ROOT / "scripts/35_run_task09_subset_formal.py"
    assert cli.is_file(), "Task-09 formal CLI missing; no official entry may run"
    prefix = [sys.executable, str(cli), "--arm", "primary", "--size", "3",
              "--seed", "0", "--output", str(PROJECT_ROOT / "研究记录/任务07_正式五种子重训")]
    refused = subprocess.run(prefix, capture_output=True, text=True, check=False)
    assert refused.returncode != 0
    assert "登记" in refused.stderr or "required" in refused.stderr


def test_adamw_per_tensor_true_step_count_matches_actual_dynamic_hf_plus_physics():
    formal = _formal()
    correction = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(correction.parameters(), lr=0.001)
    named = {"correction.weight": correction.weight}
    for expected in range(1, 5):
        optimizer.zero_grad(set_to_none=True)
        correction(torch.ones((1, 1))).sum().backward()
        optimizer.step()
        if expected < 4:
            with pytest.raises(ValueError, match="AdamW|实际|step"):
                formal.assert_task09_adamw_steps(optimizer, named,
                                                 hf_batches=3, correction_epochs=1,
                                                 joint_epochs=0)
    formal.assert_task09_adamw_steps(optimizer, named,
                                     hf_batches=3, correction_epochs=1,
                                     joint_epochs=0)
    projection = nn.Linear(1, 1, bias=False)
    name = "low_fidelity_model.branch_projection.weight"
    optimizer.add_param_group({"params": [projection.weight], "lr": 0.00001})
    named[name] = projection.weight
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        (correction(torch.ones((1, 1))) + projection(torch.ones((1, 1)))).sum().backward()
        optimizer.step()
    formal.assert_task09_adamw_steps(optimizer, named,
                                     hf_batches=3, correction_epochs=1,
                                     joint_epochs=1)
    with pytest.raises(ValueError, match="AdamW|实际|step"):
        formal.assert_task09_adamw_steps(optimizer, named,
                                         hf_batches=4, correction_epochs=1,
                                         joint_epochs=1)


def test_full_phase_state_rejects_legacy_best_missing_real_optimizer_rng():
    formal = _formal()
    sources = formal.load_task09_sources()
    source = sources[0]
    sample = {"powers_w": [55.0, 364.3, 729.0],
              "ir_rows": 6868, "ir_batches_2048": 4,
              "sensor_rows": {"Hot": 340, "Cold": 340}}
    meta = formal.task09_phase_metadata(
        source, sample, stage=formal.CORRECTION_STAGE,
        correction_epochs=0, joint_epochs=0, registry_sha256="f" * 64,
        consumption=formal.task09_empty_consumption(),
    )
    assert meta["当前真实HF子集功率"] == [55.0, 364.3, 729.0]
    assert meta["实际本seed原LF整张量SHA256"] == source.lf_tensor_sha256
    assert meta["真实HF校正器AdamW期望step"] == 0
    assert meta["旧固定测试温度读取"] is False
    assert meta["本阶段不是CPU正式成绩"] is True
    with pytest.raises(ValueError, match="AdamW|RNG|完整状态"):
        formal.check_task09_resume_payload(
            {"model_state": {}, "metadata": meta}, source, sample,
            registry_sha256="f" * 64,
        )


def test_restricted_joint_switch_reuses_true_hf_adamw_and_opens_only_four_new_lf_parameters():
    formal = _formal()
    sources = formal.load_task09_sources()
    source = sources[0]
    start = formal.fork_task07_initialization(sources, 0, "E0", torch.device("cpu"))
    model, optimizer = start.model, start.optimizer
    weight = model.correction[0].weight
    optimizer.zero_grad(set_to_none=True)
    weight.sum().backward()
    optimizer.step()
    preserved = optimizer.state[weight]["step"].clone()
    formal.switch_task09_restricted_joint(model, optimizer, source)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 0.0001
    assert optimizer.param_groups[1]["lr"] == 0.00001
    assert torch.equal(optimizer.state[weight]["step"], preserved)
    projected = {name for name, parameter in model.named_parameters()
                 if name.startswith("low_fidelity_model.") and parameter.requires_grad}
    assert projected == formal.PROJECTION_NAMES
    assert all(optimizer.state.get(parameter, {}) == {}
               for name, parameter in model.named_parameters()
               if name in formal.PROJECTION_NAMES)
    formal.check_task09_model(model, source, formal.JOINT_STAGE)
    with pytest.raises(ValueError, match="组|切换|动量"):
        formal.switch_task09_restricted_joint(model, optimizer, source)


def test_best_view_contains_only_live_new_hf_tensor_and_never_optimizer_rng():
    formal = _formal()
    sources = formal.load_task09_sources()
    start = formal.fork_task07_initialization(sources, 0, "E0", torch.device("cpu"))
    model = start.model
    sample = {"powers_w": [55.0, 364.3, 729.0],
              "ir_rows": 6868, "ir_batches_2048": 4,
              "sensor_rows": {"Hot": 340, "Cold": 340}}
    view = formal.task09_model_view(
        model, sources[0], sample, registry_sha256="e" * 64,
        score=2.1, correction_epoch=1, joint_epoch=0,
    )
    assert view["method"] == "multifidelity_correction"
    assert view["hf_train_powers_w"] == [55.0, 364.3, 729.0]
    assert "optimizer_state" not in view and "random_state" not in view
    assert torch.equal(view["model_state"]["correction.0.weight"],
                       model.correction[0].weight.detach().cpu())
    assert view["任09可作训练续跑输入"] is False
    assert view["原12HF最佳校正器张量加载"] is False


def test_resume_preflight_rejects_relabelled_original_lf_tensor_despite_valid_metadata():
    formal = _formal()
    sources = formal.load_task09_sources()
    source = sources[0]
    start = formal.fork_task07_initialization(sources, 0, "E0", torch.device("cpu"))
    sample = {"powers_w": [55.0, 364.3, 729.0],
              "ir_rows": 6868, "ir_batches_2048": 4,
              "sensor_rows": {"Hot": 340, "Cold": 340}}
    meta = formal.task09_phase_metadata(
        source, sample, stage=formal.CORRECTION_STAGE,
        correction_epochs=0, joint_epochs=0, registry_sha256="c" * 64,
        consumption=formal.task09_empty_consumption(),
    )
    payload = {
        "training_state_schema_version": 1,
        "stage": formal.CORRECTION_STAGE, "epoch": 0,
        "model_state": {name: value.detach().cpu().clone() for name, value in
                        start.model.state_dict().items()},
        "optimizer_state": start.optimizer.state_dict(),
        "parameter_requires_grad": {
            name: param.requires_grad for name, param in start.model.named_parameters()
        },
        "random_state": {"python": start.random_state["python"],
                         "numpy": start.random_state["numpy"],
                         "torch_cpu": start.random_state["torch_cpu"],
                         "torch_cuda": [torch.zeros((16,), dtype=torch.uint8)]},
        "budget": dict(formal.STAGE_BUDGET), "metadata": meta,
    }
    formal.check_task09_resume_payload(payload, source, sample,
                                       registry_sha256="c" * 64)
    stolen = dict(payload, model_state=dict(payload["model_state"]))
    key = "low_fidelity_model.branch_projection.bias"
    stolen["model_state"][key] = payload["model_state"][key] + 0.01
    with pytest.raises(ValueError, match="LF|来源|张量"):
        formal.check_task09_resume_payload(stolen, source, sample,
                                           registry_sha256="c" * 64)


def test_resume_adamw_preflight_rejects_single_stolen_tensor_step():
    formal = _formal()
    sources = formal.load_task09_sources()
    source = sources[0]
    start = formal.fork_task07_initialization(sources, 0, "E0", torch.device("cpu"))
    for _ in range(4):
        start.optimizer.zero_grad(set_to_none=True)
        start.model.correction[0](torch.ones((1, 6))).sum().backward()
        start.optimizer.step()
    sample = {"powers_w": [55.0, 364.3, 729.0],
              "ir_rows": 6868, "ir_batches_2048": 3,
              "sensor_rows": {"Hot": 340, "Cold": 340}}
    meta = formal.task09_phase_metadata(
        source, sample, stage=formal.CORRECTION_STAGE,
        correction_epochs=1, joint_epochs=0, registry_sha256="d" * 64,
        consumption=formal.task09_expected_consumption(sample, 1, 0),
    )
    payload = {
        "training_state_schema_version": 1, "stage": formal.CORRECTION_STAGE,
        "epoch": 1,
        "model_state": {name: value.detach().cpu().clone()
                        for name, value in start.model.state_dict().items()},
        "optimizer_state": start.optimizer.state_dict(),
        "parameter_requires_grad": {name: param.requires_grad
                                    for name, param in start.model.named_parameters()},
        "random_state": {"python": start.random_state["python"],
                         "numpy": start.random_state["numpy"],
                         "torch_cpu": start.random_state["torch_cpu"],
                         "torch_cuda": [torch.zeros((16,), dtype=torch.uint8)]},
        "budget": dict(formal.STAGE_BUDGET), "metadata": meta,
    }
    # The nominal correction step 4 is present; all remaining correction
    # tensors are empty, so a set-membership check must not approve the state.
    assert len(payload["optimizer_state"]["state"]) < len(
        [name for name in payload["parameter_requires_grad"]
         if name.startswith("correction.")]
    )
    with pytest.raises(ValueError, match="AdamW|step|校正"):
        formal.check_task09_resume_payload(payload, source, sample,
                                           registry_sha256="d" * 64)


def test_joint_rows_cannot_precede_correction_even_if_final_lf_total_matches():
    formal = _formal()
    source = formal.load_task09_sources()[0]
    sample = {"powers_w": [55.0, 364.3, 729.0],
              "ir_rows": 6868, "ir_batches_2048": 4,
              "sensor_rows": {"Hot": 340, "Cold": 340}}
    steps = 4 * 680
    correction = {"训练阶段": formal.CORRECTION_STAGE,
                  "HF训练观测点": 6868, "HF训练传感器点": steps,
                  "HF观测优化步": 4, "物理优化步": 1, "物理配点": 256,
                  "LF真实回放训练点": 0, "LF联合回放batch": 0,
                  "LF_Cu真实回放点": 0, "LF_SiC真实回放点": 0,
                  "LF60每功率真2048已核验": False,
                  "LF每HF步真实分组": []}
    joint = dict(correction, 训练阶段=formal.JOINT_STAGE,
                 LF真实回放训练点=60 * 2048, LF联合回放batch=60,
                 LF_Cu真实回放点=30 * 2048,
                 LF_SiC真实回放点=30 * 2048,
                 LF60每功率真2048已核验=True,
                 LF每HF步真实分组=[15] * 4)
    cumulative = formal.task09_empty_consumption()
    rows = []
    for number, row in enumerate((correction, joint), 1):
        cumulative = {key: cumulative[key] + row[key] for key in cumulative}
        rows.append(dict(row, epoch=number, 运行种子=0,
                         真实HF子集功率=list(sample["powers_w"]),
                         阶段实际轮次=1, 累计实际消耗=dict(cumulative)))
    meta = formal.task09_phase_metadata(
        source, sample, stage=formal.JOINT_STAGE,
        correction_epochs=1, joint_epochs=1, registry_sha256="b" * 64,
        consumption=formal.task09_expected_consumption(sample, 1, 1),
    )
    formal.validate_task09_logged_history(rows, meta, source, sample)
    forged = [dict(rows[0], **joint), dict(rows[1], **correction)]
    forged[0]["累计实际消耗"] = formal.task09_expected_consumption(sample, 0, 1)
    forged[1]["累计实际消耗"] = formal.task09_expected_consumption(sample, 1, 1)
    with pytest.raises(ValueError, match="校正|联合|阶段|轮次"):
        formal.validate_task09_logged_history(forged, meta, source, sample)


def test_own_best_view_must_equal_complete_own_best_stage_live_tensors_and_score():
    formal = _formal()
    sources = formal.load_task09_sources()
    source = sources[0]
    start = formal.fork_task07_initialization(sources, 0, "E0", torch.device("cpu"))
    sample = {"powers_w": [55.0, 364.3, 729.0],
              "ir_rows": 6868, "ir_batches_2048": 4,
              "sensor_rows": {"Hot": 340, "Cold": 340}}
    meta = formal.task09_phase_metadata(
        source, sample, stage=formal.CORRECTION_STAGE,
        correction_epochs=0, joint_epochs=0, registry_sha256="a" * 64,
        consumption=formal.task09_empty_consumption(),
        best_score=2.1, initial_score=2.1,
    )
    full = {"model_state": start.model.state_dict(), "metadata": meta}
    view = formal.task09_model_view(start.model, source, sample,
                                    registry_sha256="a" * 64,
                                    score=2.1, correction_epoch=0, joint_epoch=0)
    formal.check_task09_best_view(view, full, source, sample,
                                   registry_sha256="a" * 64)
    tampered = dict(view, model_state=dict(view["model_state"]))
    tampered["model_state"]["correction.0.weight"] = (
        view["model_state"]["correction.0.weight"] + 1.0
    )
    with pytest.raises(ValueError, match="最佳|张量|HF|视图"):
        formal.check_task09_best_view(tampered, full, source, sample,
                                       registry_sha256="a" * 64)
    with pytest.raises(ValueError, match="最佳|选分|视图"):
        formal.check_task09_best_view(dict(view, validation_selection_score_c=1.1),
                                       full, source, sample,
                                       registry_sha256="a" * 64)


def test_formal_preregistration_generator_structured_dry_run_without_writing():
    formal = _formal()
    generator = PROJECT_ROOT / "scripts/39_register_task09_subset_formal.py"
    assert generator.is_file(), "structured Task-09 original registration writer absent"
    generated = subprocess.run([sys.executable, str(generator), "--dry-run"],
                               capture_output=True, text=True, check=False)
    assert generated.returncode == 0, generated.stderr
    assert yaml.safe_load(generated.stdout) == formal.expected_task09_registration()
    invalid = subprocess.run(
        [sys.executable, str(generator), "--output",
         str(PROJECT_ROOT / "研究记录/任务07_正式五种子重训/有效运行配置_实现修复后.yaml")],
        capture_output=True, text=True, check=False,
    )
    assert invalid.returncode != 0
    assert "任09" in invalid.stderr or "独立" in invalid.stderr


def test_resume_rejects_forged_cuda_rng_tensor_even_before_training_labels():
    formal = _formal()
    sources = formal.load_task09_sources()
    source = sources[0]
    start = formal.fork_task07_initialization(sources, 0, "E0", torch.device("cpu"))
    sample = {"powers_w": [55.0, 364.3, 729.0],
              "ir_rows": 6868, "ir_batches_2048": 4,
              "sensor_rows": {"Hot": 340, "Cold": 340}}
    meta = formal.task09_phase_metadata(
        source, sample, stage=formal.CORRECTION_STAGE,
        correction_epochs=0, joint_epochs=0, registry_sha256="2" * 64,
        consumption=formal.task09_empty_consumption(),
    )
    payload = {
        "training_state_schema_version": 1, "stage": formal.CORRECTION_STAGE,
        "epoch": 0, "model_state": start.model.state_dict(),
        "optimizer_state": start.optimizer.state_dict(),
        "parameter_requires_grad": {
            name: param.requires_grad for name, param in start.model.named_parameters()
        },
        "random_state": {"python": start.random_state["python"],
                         "numpy": start.random_state["numpy"],
                         "torch_cpu": start.random_state["torch_cpu"],
                         "torch_cuda": [torch.zeros((16,), dtype=torch.uint8)]},
        "budget": dict(formal.STAGE_BUDGET), "metadata": meta,
    }
    formal.check_task09_resume_payload(payload, source, sample,
                                       registry_sha256="2" * 64)
    forged = dict(payload, random_state=dict(payload["random_state"]))
    forged["random_state"]["torch_cuda"] = [torch.get_rng_state()]
    with pytest.raises(ValueError, match="RNG|CUDA|随机"):
        formal.check_task09_resume_payload(forged, source, sample,
                                           registry_sha256="2" * 64)


def test_resume_preflight_happens_before_any_actual_hf_label_read(monkeypatch, tmp_path):
    formal = _formal()
    destination = tmp_path / "任09只读续跑伪阶段现场"
    assert not destination.exists()
    destination.mkdir()
    sources = formal.load_task09_sources()
    start = formal.fork_task07_initialization(sources, 0, "E0", torch.device("cpu"))
    start.random_state["torch_cuda"] = [torch.zeros((16,), dtype=torch.uint8)]
    monkeypatch.setattr(formal, "require_task09_cuda", lambda device: torch.device("cpu"))
    monkeypatch.setattr(formal, "require_task09_registration", lambda path, sha: "4" * 64)
    monkeypatch.setattr(formal, "_task09_precheck_registry_bytes",
                        lambda path, sha: "4" * 64)
    monkeypatch.setattr(formal, "validate_task09_output",
                        lambda output, name, size, seed: destination)
    monkeypatch.setattr(formal, "fork_task07_initialization", lambda *args: start)
    def forbidden_labels(*args, **kwargs):
        raise AssertionError("HF train/validation temperature opened before historical preflight")
    monkeypatch.setattr(formal, "_ir_dataset", forbidden_labels)
    with pytest.raises(FileNotFoundError, match="阶段_最近|No such file"):
        formal.run_task09_formal(
            name="primary", size=3, seed=0, output_directory=destination,
            device_name="cuda", registry_path=formal.TASK09_REGISTRY,
            registry_sha256="4" * 64, session_epoch_limit=1,
            resume_checkpoint=destination / "阶段_最近.pt",
        )


def test_first_cuda_run_archives_original_sources_before_true_ring_label_hash(monkeypatch, tmp_path):
    formal = _formal()
    destination = tmp_path / "任09全新入场待验证目录"
    assert not destination.exists()
    actions = []
    monkeypatch.setattr(formal, "require_task09_cuda", lambda device: torch.device("cpu"))
    monkeypatch.setattr(formal, "validate_task09_output",
                        lambda output, name, size, seed: destination)
    monkeypatch.setattr(formal, "_task09_precheck_registry_bytes",
                        lambda path, sha: "3" * 64, raising=False)
    def spy_archive(output):
        assert output == destination
        actions.append("source_archive")
        (output / "source_snapshot").mkdir()
        return {}
    def spy_registration(*args):
        actions.append("true_ring_hash")
        assert actions == ["source_archive", "true_ring_hash"]
        raise RuntimeError("任09真标签hash已在原源码SHA快照后执行")
    monkeypatch.setattr(formal, "_task09_archive_source", spy_archive)
    monkeypatch.setattr(formal, "require_task09_registration", spy_registration)
    with pytest.raises(RuntimeError, match="源码SHA快照后"):
        formal.run_task09_formal(
            name="primary", size=3, seed=0, output_directory=destination,
            device_name="cuda", registry_path=formal.TASK09_REGISTRY,
            registry_sha256="3" * 64, session_epoch_limit=1,
        )
    assert actions == ["source_archive", "true_ring_hash"]
