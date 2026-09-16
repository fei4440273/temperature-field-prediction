from __future__ import annotations

import random
import copy
import json
import hashlib
import runpy
import sys

import numpy as np
import polars as pl
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler

from sic_cu.losses import PhysicsLossComputer
from sic_cu.eval import energy_v5
from sic_cu.eval import metrics as eval_metrics
from sic_cu.models import DeepONetPINN
from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.physics.collocation import sample_collocation
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.train import common
from sic_cu.train import multifidelity


def test_training_state_restores_optimizer_freeze_sampler_and_random_sources(tmp_path) -> None:
    torch.manual_seed(19)
    np.random.seed(19)
    random.seed(19)
    model = nn.Linear(2, 1)
    model.bias.requires_grad_(False)
    optimizer = torch.optim.AdamW([model.weight], lr=0.01)
    loss = model(torch.ones(2, 2)).square().mean()
    loss.backward()
    optimizer.step()
    sampler = DistributedSampler(TensorDataset(torch.arange(7)), num_replicas=1, rank=0)
    sampler.set_epoch(4)
    checkpoint = tmp_path / "阶段末.pt"
    common.save_training_state(
        checkpoint, model, optimizer, stage="校正", epoch=4,
        samplers={"观测": sampler}, budget={"轮次上限": 10},
    )
    expected = (torch.rand(3), np.random.rand(3), random.random())
    original_weight = model.weight.detach().clone()
    original_step = optimizer.state[model.weight]["step"].clone()

    with torch.no_grad():
        model.weight.add_(10)
    model.bias.requires_grad_(True)
    sampler.set_epoch(9)
    torch.rand(10)
    np.random.rand(10)
    random.random()
    restored = common.load_training_state(
        checkpoint, model, optimizer, samplers={"观测": sampler}
    )

    assert restored["stage"] == "校正"
    assert restored["epoch"] == 4
    assert restored["budget"] == {"轮次上限": 10}
    assert torch.equal(model.weight, original_weight)
    assert model.bias.requires_grad is False
    assert sampler.epoch == 4
    assert torch.equal(optimizer.state[model.weight]["step"], original_step)
    assert torch.equal(torch.rand(3), expected[0])
    assert np.array_equal(np.random.rand(3), expected[1])
    assert random.random() == expected[2]
    assert not (tmp_path / "阶段末.pt.tmp").exists()


def test_training_state_rejects_historical_model_only_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "历史.pt"
    torch.save({"model_state": nn.Linear(2, 1).state_dict()}, checkpoint)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="training state"):
        common.load_training_state(checkpoint, model, optimizer)


def test_initial_correction_resume_metadata_is_complete_and_only_continuous_states_resume(tmp_path) -> None:
    metadata = multifidelity.correction_resume_metadata(
        "locked-hf-hash", "separate", {"training": "source-hash"},
    )
    assert metadata["消费累计"] == {
        "hf_ir_points": 0, "hf_sensor_points": 0,
        "surface_teacher_points": 0, "lf_simulation_points": 0,
        "lf_simulation_material_points": {"copper": 0, "silicon_carbide": 0},
    }
    assert metadata["观测最佳分数"] is None
    assert metadata["观测最佳轮次"] == 0
    assert metadata["物理最佳损失"] is None
    assert metadata["停止计数"] == 0
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    initial_path = tmp_path / "阶段_初始.pt"
    common.save_training_state(
        initial_path, model, optimizer, stage="correction", epoch=0,
        budget={"校正轮次": 2, "联合轮次": 0}, metadata=metadata,
    )
    restored = common.load_training_state(initial_path, model, optimizer)
    assert restored["metadata"]["消费累计"]["hf_ir_points"] == 0
    assert multifidelity.ensure_continuous_resume_state(initial_path) is None
    assert multifidelity.ensure_continuous_resume_state(tmp_path / "阶段_最近.pt") is None
    for name in ("阶段_观测最佳.pt", "阶段_物理最佳.pt", "阶段_训练末.pt"):
        with pytest.raises(ValueError, match="continuous resume"):
            multifidelity.ensure_continuous_resume_state(tmp_path / name)


def test_correction_resume_recovers_only_one_uncommitted_log_row_with_archive(tmp_path) -> None:
    log = tmp_path / "training.jsonl"
    initial_bytes = b'{"epoch": 1}\n{"epoch": 2}\n'
    log.write_bytes(initial_bytes)
    assert multifidelity.reconcile_correction_resume_log(tmp_path, epoch=1) == 1
    assert log.read_bytes() == b'{"epoch": 1}\n'
    archives = list(tmp_path.glob("training_中断尾行_*.jsonl"))
    reports = list(tmp_path.glob("日志恢复记录_*.json"))
    assert len(archives) == len(reports) == 1
    assert archives[0].read_bytes() == initial_bytes
    assert json.loads(reports[0].read_text(encoding="utf-8"))["恢复依据轮次"] == 1
    assert multifidelity.reconcile_correction_resume_log(tmp_path, epoch=1) == 1
    assert len(list(tmp_path.glob("training_中断尾行_*.jsonl"))) == 1
    log.write_bytes(b'{"epoch": 1}\n{"epoch": 2}\n{"epoch": 3}\n')
    with pytest.raises(ValueError, match="resume log"):
        multifidelity.reconcile_correction_resume_log(tmp_path, epoch=1)
    assert log.read_bytes().endswith(b'{"epoch": 3}\n')


def test_correction_resume_does_not_roll_back_committed_rows_from_initial_state(tmp_path) -> None:
    log = tmp_path / "training.jsonl"
    log.write_bytes(b'{"epoch": 1}\n')
    (tmp_path / "阶段_最近.pt").write_bytes(b"committed-state")
    with pytest.raises(ValueError, match="latest state"):
        multifidelity.reconcile_correction_resume_log(
            tmp_path, epoch=0, state_file=tmp_path / "阶段_初始.pt",
        )
    assert log.read_bytes() == b'{"epoch": 1}\n'
    assert not list(tmp_path.glob("training_中断尾行_*.jsonl"))


def test_independent_physical_stage_ignores_training_subpackage_loss() -> None:
    guardrails = {"独立局部物理损失": {"physics_total": 0.08}}
    assert multifidelity.independent_physical_stage_score(guardrails) == 0.08
    assert not multifidelity.physical_stage_improves(0.05, guardrails)
    assert multifidelity.physical_stage_improves(0.09, guardrails)
    assert multifidelity.independent_physical_stage_score(None) is None


def test_early_stop_is_completed_with_actual_epoch_and_session_cap_is_paused() -> None:
    assert multifidelity.is_session_paused(epoch=4, planned_epochs=600, early_stopped=True) is False
    assert multifidelity.is_session_paused(epoch=4, planned_epochs=600, early_stopped=False) is True
    assert multifidelity.is_session_paused(epoch=600, planned_epochs=600, early_stopped=False) is False


def test_collocation_subpackages_preserve_paired_interface_and_point_budget() -> None:
    batch = sample_collocation(8, torch.device("cpu"), seed=11)
    packages = common.split_collocation(batch, 3)
    assert len(packages) == 3
    for name in batch.__dataclass_fields__:
        original = getattr(batch, name)
        combined = torch.cat([getattr(part, name) for part, _ in packages])
        assert torch.equal(torch.sort(combined, dim=0).values, torch.sort(original, dim=0).values)
    for part, ratio in packages:
        assert len(part.interface_sic) == len(part.interface_copper) == len(part.interface_normals)
        assert set(part.interior_material_ids.tolist()) == {0, 1}
        assert torch.equal(part.interface_sic[:, 2:4], part.interface_copper[:, 2:4])
        assert ratio == pytest.approx(len(part.interior) / len(batch.interior))
    assert sum(weight for _, weight in packages) == pytest.approx(1.0)


def test_collocation_subpackages_reject_more_parts_than_interface_pairs() -> None:
    batch = sample_collocation(4, torch.device("cpu"), seed=3)
    with pytest.raises(ValueError, match="subpackages"):
        common.split_collocation(batch, 5)


class SmallTemperatureField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.coefficient = nn.Parameter(torch.tensor(0.3))

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        r, z, t, power = coordinates[:, :1], coordinates[:, 1:2], coordinates[:, 2:3], coordinates[:, 3:4]
        return 295.15 + self.coefficient * (r.square() + z.square() + t * power * 1e-4)


def test_weighted_subpackages_match_full_physics_loss_and_gradient() -> None:
    batch = sample_collocation(8, torch.device("cpu"), seed=17)
    model = SmallTemperatureField()
    physics = PhysicsLossComputer(load_materials(), load_resolved_boundary_conditions())
    full = physics(model, batch)["physics_total"]
    full_gradient = torch.autograd.grad(full, model.coefficient)[0]
    weighted = sum(
        weight * physics(model, part)["physics_total"]
        for part, weight in common.split_collocation(batch, 3)
    )
    weighted_gradient = torch.autograd.grad(weighted, model.coefficient)[0]
    assert torch.allclose(weighted, full, rtol=1e-5, atol=1e-6)
    assert torch.allclose(weighted_gradient, full_gradient, rtol=1e-5, atol=1e-6)


def test_physics_backward_adds_to_existing_observation_gradient() -> None:
    batch = sample_collocation(8, torch.device("cpu"), seed=23)
    model = SmallTemperatureField()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    physics = PhysicsLossComputer(load_materials(), load_resolved_boundary_conditions())
    data_loss = model(batch.initial).square().mean()
    optimizer.zero_grad(set_to_none=True)
    data_loss.backward()
    observation_gradient = model.coefficient.grad.detach().clone()
    expected = observation_gradient + torch.autograd.grad(
        physics(model, batch)["physics_total"], model.coefficient
    )[0]

    common.physics_backward(model, optimizer, physics, batch)

    assert torch.allclose(model.coefficient.grad, expected, rtol=1e-5, atol=1e-5)


def test_locked_b0_start_uses_manifest_and_preserves_current_physics_semantics() -> None:
    root = PROJECT_ROOT / "reports/runs/sic-cu-v2-gpu1-20260908T072729Z-r2"
    lf = root / "deeponet_pinn_lf_seed0/best.pt"
    hf = root / "deeponet_pinn_mf_seed0/best.pt"
    checkpoint = multifidelity.validate_locked_b0_start(hf, lf, seed=0)
    assert checkpoint["epoch"] == 579
    assert checkpoint["seed"] == 0
    assert checkpoint["provenance"]["test_labels_consumed"] is False


def test_locked_b0_start_rejects_incorrect_seed() -> None:
    root = PROJECT_ROOT / "reports/runs/sic-cu-v2-gpu1-20260908T072729Z-r2"
    with pytest.raises(ValueError, match="locked B0"):
        multifidelity.validate_locked_b0_start(
            root / "deeponet_pinn_mf_seed0/best.pt",
            root / "deeponet_pinn_lf_seed0/best.pt", seed=1,
        )


@pytest.mark.parametrize("config_file", ["geometry.yaml", "materials.yaml"])
def test_locked_b0_start_rejects_changed_geometry_or_materials(monkeypatch, config_file) -> None:
    root = PROJECT_ROOT / "reports/runs/sic-cu-v2-gpu1-20260908T072729Z-r2"
    current_path = (PROJECT_ROOT / "configs" / config_file).resolve()
    original_read_bytes = type(current_path).read_bytes

    def changed_config_bytes(path):
        original = original_read_bytes(path)
        return original + b"\nchanged-physics\n" if path.resolve() == current_path else original

    monkeypatch.setattr(type(current_path), "read_bytes", changed_config_bytes)
    with pytest.raises(ValueError, match="Locked B0.*(geometry|materials)"):
        multifidelity.validate_locked_b0_start(
            root / "deeponet_pinn_mf_seed0/best.pt",
            root / "deeponet_pinn_lf_seed0/best.pt", seed=0,
        )


def test_hf_transfer_preserves_locked_correction_but_keeps_candidate_lf_weights() -> None:
    class Paired(nn.Module):
        def __init__(self):
            super().__init__()
            self.low_fidelity_model = nn.Linear(1, 1, bias=False)
            self.correction_model = nn.Linear(1, 1, bias=False)

    model = Paired()
    with torch.no_grad():
        model.low_fidelity_model.weight.fill_(5)
        model.correction_model.weight.fill_(0)
    old_state = {
        "low_fidelity_model.weight": torch.tensor([[2.0]]),
        "correction_model.weight": torch.tensor([[3.0]]),
    }
    multifidelity.initialize_locked_hf_with_lf(model, old_state, replace_lf=True)
    assert model.low_fidelity_model.weight.item() == 5.0
    assert model.correction_model.weight.item() == 3.0
    multifidelity.initialize_locked_hf_with_lf(model, old_state, replace_lf=False)
    assert model.low_fidelity_model.weight.item() == 2.0
    assert model.correction_model.weight.item() == 3.0


def test_replacement_lf_cannot_bypass_locked_hf_observation_contract() -> None:
    historical = {
        "correction_model_kwargs": {
            "width": 128, "depth": 4,
            "correction_power_scaling": "none",
            "correction_direct_power_input": True,
            "correction_power_reference_w": 400.0,
        },
        "sensors_used": True,
    }
    options = {
        "width": 128, "depth": 4, "correction_power_scaling": "none",
        "correction_direct_power_input": True, "correction_power_reference_w": 400.0,
        "skip_sensors": False, "hard_surface_residual_guide": False,
        "identify_physics_parameters": False, "hard_deployment_constraints": True,
        "hf_training_subset_w": None, "surface_teacher_weight": 0.0,
        "sensor_absolute_loss_weight": 5.0, "sensor_delta_loss_weight": 1.0,
    }
    multifidelity.validate_locked_hf_contract(historical, **options)
    with pytest.raises(ValueError, match="observation contract"):
        multifidelity.validate_locked_hf_contract(historical, **(options | {"skip_sensors": True}))
    with pytest.raises(ValueError, match="observation contract"):
        multifidelity.validate_locked_hf_contract(
            historical, **(options | {"identify_physics_parameters": True}),
        )
    with pytest.raises(ValueError, match="observation contract"):
        multifidelity.validate_locked_hf_contract(
            historical, **(options | {"sensor_absolute_loss_weight": 0.0}),
        )


def test_task03_lf_lineage_requires_300_epoch_locked_source() -> None:
    old_root = PROJECT_ROOT / "reports/runs/sic-cu-v2-gpu1-20260908T072729Z-r2"
    original = old_root / "deeponet_pinn_lf_seed0/best.pt"
    candidate = PROJECT_ROOT / (
        "研究记录/任务03_低保真精度修复/"
        "任务03_LF控制_种子0_20260915T184117+0800/best.pt"
    )
    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    multifidelity.validate_task03_lf_lineage(payload, candidate, original, seed=0)
    with pytest.raises(ValueError, match="LF.*source|LF.*origin|LF.*lineage"):
        multifidelity.validate_task03_lf_lineage(payload, candidate, original, seed=1)
    with pytest.raises(ValueError, match="LF.*source|LF.*origin|LF.*lineage"):
        multifidelity.validate_task03_lf_lineage(
            payload, original, original, seed=0,
        )
    altered = copy.deepcopy(payload)
    first = next(iter(altered["model_state"].values()))
    first.reshape(-1)[0] += 0.125
    with pytest.raises(ValueError, match="LF.*source|LF.*origin|LF.*lineage"):
        multifidelity.validate_task03_lf_lineage(altered, candidate, original, seed=0)


def test_task03_lf_lineage_rejects_different_per_epoch_point_budget(monkeypatch) -> None:
    old_root = PROJECT_ROOT / "reports/runs/sic-cu-v2-gpu1-20260908T072729Z-r2"
    candidate = PROJECT_ROOT / (
        "研究记录/任务03_低保真精度修复/"
        "任务03_LF控制_种子0_20260915T184117+0800/best.pt"
    )
    checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
    state_path = candidate.parent / "run_state.json"
    original_read = type(state_path).read_text

    def changed_run_state(path, *args, **kwargs):
        content = original_read(path, *args, **kwargs)
        if path.resolve() == state_path.resolve():
            run = json.loads(content)
            run["configuration"]["samples_per_power"] = 128
            return json.dumps(run)
        return content

    monkeypatch.setattr(type(state_path), "read_text", changed_run_state)
    with pytest.raises(ValueError, match="LF.*source|LF.*origin|LF.*lineage"):
        multifidelity.validate_task03_lf_lineage(
            checkpoint, candidate, old_root / "deeponet_pinn_lf_seed0/best.pt", seed=0,
        )


def test_real_new_lf_hf_training_rejects_changed_b0_sensor_contract(tmp_path) -> None:
    old_root = PROJECT_ROOT / "reports/runs/sic-cu-v2-gpu1-20260908T072729Z-r2"
    candidate = PROJECT_ROOT / (
        "研究记录/任务03_低保真精度修复/"
        "任务03_LF控制_种子0_20260915T184117+0800/best.pt"
    )
    with pytest.raises(ValueError, match="observation contract"):
        multifidelity.train_multifidelity(
            str(candidate), str(tmp_path / "hf"), seed=0,
            locked_historical_lf_checkpoint=str(old_root / "deeponet_pinn_lf_seed0/best.pt"),
            start_checkpoint=str(old_root / "deeponet_pinn_mf_seed0/best.pt"),
            correction_epochs=1, joint_epochs=0, skip_sensors=True,
        )
    assert not (tmp_path / "hf").exists()


def test_task03_hf_launcher_excludes_unmarked_short_budget_from_formal_candidate(
    monkeypatch, tmp_path,
) -> None:
    command = PROJECT_ROOT / "scripts/14_run_task03_hf_transfer.py"
    launcher = runpy.run_path(str(command), run_name="task03_launcher")
    monkeypatch.setattr(sys, "argv", [
        str(command), "--arm", "old", "--output", str(tmp_path / "unused"),
        "--epochs", "100",
    ])
    with pytest.raises(ValueError, match="正式|diagnostic"):
        launcher["main"]()
    assert not (tmp_path / "unused").exists()


def test_schedule_comparison_requires_locked_b0_and_frozen_lf() -> None:
    with pytest.raises(ValueError, match="locked B0"):
        multifidelity.train_multifidelity(
            "unused.pt", "unused", correction_epochs=1, joint_epochs=0,
            physics_schedule="mixed_every_five",
        )


def test_new_config_snapshot_includes_only_persistent_main_plan_hash(tmp_path) -> None:
    snapshot = common.write_config_snapshot(tmp_path)
    main_plan = PROJECT_ROOT / "多保真DeepONet预测精度优化总计划与执行台账.md"
    hashes = json.loads((snapshot / "sha256.json").read_text(encoding="utf-8"))
    assert hashes[main_plan.name] == hashlib.sha256(main_plan.read_bytes()).hexdigest()


def test_energy_terms_chinese_mapping_preserves_original_balance_and_denominator() -> None:
    source = pl.read_csv(PROJECT_ROOT / "reports/development_v5/m2_energy_terms.csv")
    mapped, aggregate = energy_v5.chinese_energy_terms(source, order=64)
    assert mapped.height == 30
    assert "原定义相对平衡" in mapped.columns
    assert "原定义相对平衡分母_瓦" in mapped.columns
    assert "绝对平衡_瓦" in mapped.columns
    assert aggregate["绝对平衡宏均值_瓦"] == pytest.approx(908.4290536194341)
    assert aggregate["绝对平衡95分位_瓦"] == pytest.approx(1851.6758955264)
    assert mapped["平衡_瓦"].to_list() == source.filter(
        pl.col("quadrature_order") == 64
    )["balance_w"].to_list()


def test_resume_option_rejects_unlocked_training_run() -> None:
    with pytest.raises(ValueError, match="locked B0"):
        multifidelity.train_multifidelity(
            "unused.pt", "unused", correction_epochs=2, joint_epochs=0,
            resume_training_checkpoint="阶段_最近.pt",
        )


def test_session_epoch_limit_rejects_nonpositive_budget() -> None:
    with pytest.raises(ValueError, match="session epoch limit"):
        multifidelity.train_multifidelity(
            "unused.pt", "unused", correction_epochs=2, joint_epochs=0,
            session_epoch_limit=0,
        )


def test_resume_status_records_last_checkpoint_and_explicit_next_action(tmp_path) -> None:
    status = tmp_path / "当前接续状态.md"
    multifidelity.write_resume_status(
        status, task="任-01", run="单轮检查", last_checkpoint="阶段_最近.pt",
        reason="训练暂停", next_action="从阶段_最近.pt继续剩余轮次",
    )
    content = status.read_text(encoding="utf-8")
    assert "当前任务：任-01" in content
    assert "最后有效检查点：阶段_最近.pt" in content
    assert "从阶段_最近.pt继续剩余轮次" in content
    assert not (tmp_path / "当前接续状态.md.tmp").exists()


def test_schedule_guardrails_use_four_fixed_energy_points_and_local_residuals() -> None:
    device = torch.device("cpu")
    materials = load_materials()
    boundaries = load_resolved_boundary_conditions()
    physics = PhysicsLossComputer(materials, boundaries)
    result = multifidelity.evaluate_schedule_guardrails(
        SmallTemperatureField(), physics, materials, boundaries, device, order=4,
    )
    assert len(result["能量四点原始值"]) == 4
    assert {(row["power_w"], row["time_s"]) for row in result["能量四点原始值"]} == {
        (55.0, 10.0), (55.0, 100.0), (630.5, 10.0), (630.5, 100.0),
    }
    assert result["独立局部物理损失"]["pde"] >= 0.0
    assert np.isfinite(result["能量四点绝对平衡均值_瓦"])


def test_schedule_energy_audit_keeps_live_float32_model_and_optimizer_contract() -> None:
    model = DeepONetPINN(width=8, latent_dim=8, blocks=1, include_material=True)
    materials = load_materials()
    boundaries = load_resolved_boundary_conditions()
    result = multifidelity.evaluate_schedule_guardrails(
        model, PhysicsLossComputer(materials, boundaries), materials, boundaries,
        torch.device("cpu"), order=4,
    )
    assert len(result["能量四点原始值"]) == 4
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_complete_energy_audit_preserves_b_equals_v_minus_j_minus_d() -> None:
    geometry = energy_v5.AxisymmetricGeometry.from_config(
        load_yaml("configs/geometry.yaml")
    )
    audit = energy_v5.audit_schedule_energy(
        SmallTemperatureField(), load_materials(), load_resolved_boundary_conditions(),
        geometry, powers_w=(55.0,), times_s=(10.0,), orders=(4, 6),
        device=torch.device("cpu"),
    )
    decomposition = audit["物理分解"]
    assert decomposition.height == 1
    row = decomposition.row(0, named=True)
    assert row["解释工程平衡_瓦"] == pytest.approx(
        row["体积分残差V_瓦"] - row["界面双侧通量J_瓦"] - row["边界失配D_瓦"]
    )
    assert audit["汇总"]["原定义相对平衡分母已保持"] is True
    assert audit["汇总"]["最大相邻阶变化对吸收功率比"] >= 0.0


def test_observation_time_windows_do_not_double_count_30_or_100_seconds() -> None:
    times = np.array([0.0, 30.0, 30.1, 100.0, 100.1, 200.0, 201.0])
    masks = [eval_metrics.observation_time_window_mask(times, name) for name in (
        "time_0_30_s", "time_30_100_s", "time_100_200_s",
    )]
    assert [np.flatnonzero(mask).tolist() for mask in masks] == [
        [0, 1], [2, 3], [4, 5],
    ]
    assert np.stack(masks).sum(axis=0).tolist() == [1, 1, 1, 1, 1, 1, 0]


def test_empty_time_window_is_explicitly_missing_in_complete_field_metrics() -> None:
    target = np.array([[300.0, 300.0], [301.0, 301.0]])
    metrics = eval_metrics.field_metrics(
        target, target, np.array([0.0, 30.0]), np.array([0, 1]),
    )
    assert metrics["time_0_30_s"]["rmse_c"] == 0.0
    assert metrics["time_30_100_s"] is None
    assert metrics["time_100_200_s"] is None


def test_validation_sensor_windows_keep_30_and_100_seconds_once() -> None:
    class ConstantTemperature(nn.Module):
        def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
            return coordinates[:, :1] * 0.0 + 295.15

    result = multifidelity._evaluate_sensor_model(
        ConstantTemperature(), split="validation", device=torch.device("cpu")
    )
    first = result["per_curve"][0]
    assert first["time_windows_absolute"]["time_0_30_s"] is not None
    assert first["window_observation_counts"]["time_0_30_s"] > 0
    assert sum(first["window_observation_counts"].values()) == first["observation_count"]


def test_chinese_schedule_report_marks_missing_windows_without_filling() -> None:
    root = PROJECT_ROOT / "研究记录/任务02_训练排程/任务02_S0正式_种子0_20260915T174917+0800"
    original = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    class ConstantTemperature(nn.Module):
        def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
            return coordinates[:, :1] * 0.0 + 295.15

    sensors = multifidelity._evaluate_sensor_model(
        ConstantTemperature(), split="validation", device=torch.device("cpu")
    )
    table = eval_metrics.chinese_schedule_observation_rows(
        "S0", original["validation_three_power_comparison"],
        original["validation_ir"], sensors,
    )
    assert "模态" in table.columns
    assert "时间窗" in table.columns
    missing = table.filter(
        (pl.col("功率_瓦") == 115.2)
        & (pl.col("模态") == "顶部")
        & (pl.col("时间窗") == "(100,200]秒")
    )
    assert missing["数据状态"].to_list() == ["无数据"]
    assert missing["RMSE_摄氏度"].null_count() == 1
