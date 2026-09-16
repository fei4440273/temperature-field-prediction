"""任08 F2审计入口严格拒绝诊断冒充五种子与物理分项混用。"""

from __future__ import annotations

import importlib.util
import json
import copy

import numpy as np

import pytest
import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.train.task07_source import validate_task07_sources


def _entry():
    path = PROJECT_ROOT / "scripts/33_audit_task08_f2_states.py"
    spec = importlib.util.spec_from_file_location("independent_task08_f2_audit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_five_observation_audit_rejects_short_seed_map_before_output(tmp_path):
    entry = _entry()
    output = tmp_path / "不能伪五份"
    with pytest.raises(ValueError, match="五|种子|0.*4"):
        entry.audit_five_observations(
            {0: tmp_path / "仅一个未完整目录"}, output_directory=output,
            registry_sha256="f" * 64, device_name="cuda")
    assert not output.exists()


def test_both_audit_outputs_refuse_external_project_directory_early(tmp_path):
    entry = _entry()
    outside = PROJECT_ROOT.parent / "任08F2禁止项目外审核工件_20260916"
    assert not outside.exists()
    with pytest.raises(ValueError, match="项目内|写入范围"):
        entry.audit_state_energy(tmp_path / "未跑", state="best", output_directory=outside,
                                 registry_sha256="f" * 64, device_name="cuda")
    with pytest.raises(ValueError, match="项目内|写入范围"):
        entry.audit_five_observations(
            {seed: tmp_path / f"未跑{seed}" for seed in range(5)},
            output_directory=outside, registry_sha256="f" * 64,
            device_name="cuda")
    assert not outside.exists()


def test_formal_state_audit_rejects_diagnostic_stage_without_energy(
    tmp_path, monkeypatch,
):
    entry = _entry()
    from sic_cu.train.task08_f2_formal import run_task08_f2_formal

    diagnostic = tmp_path / "F2短CPU一轮不能作能源正式归档"
    run_task08_f2_formal(seed=0, output_directory=diagnostic,
                         session_epoch_limit=1, diagnostic_only=True,
                         device_name="cpu")
    monkeypatch.setattr(entry, "_require_formal_registration", lambda *args: None)
    output = tmp_path / "原件未满不能启动能源"
    with pytest.raises((ValueError, FileNotFoundError), match="正式|末|截止|诊断"):
        entry.audit_state_energy(
            diagnostic, state="best", output_directory=output,
            registry_sha256="f" * 64, device_name="cpu")
    assert not output.exists()


def test_pde_training_none_and_full_audit_finite_must_be_separate():
    entry = _entry()
    valid = {"名义物理训练分项": {"pde": None, "initial": 0.0,
                                "boundary": 0.1, "interface": 0.1,
                                "physics_total": 0.2},
             "独立局部物理损失": {"pde": 0.5, "initial": 0.0,
                                   "boundary": 0.1, "interface": 0.1,
                                   "physics_total": 0.7}}
    entry.verify_physics_distinction(valid)
    for fake in (0.0, float("nan")):
        poisoned = json.loads(json.dumps(valid))
        poisoned["名义物理训练分项"]["pde"] = fake
        with pytest.raises(ValueError, match="PDE|pde|未计算"):
            entry.verify_physics_distinction(poisoned)
    poisoned = json.loads(json.dumps(valid))
    poisoned["独立局部物理损失"]["pde"] = None
    with pytest.raises(ValueError, match="PDE|pde|全项"):
        entry.verify_physics_distinction(poisoned)


def test_energy_schedule_is_same_pre_registered_30_points_and_two_orders():
    entry = _entry()
    powers, times, orders = entry.fixed_schedule()
    assert powers == [55.0, 115.2, 364.3, 403.0, 630.5, 729.0]
    assert times == [1.0, 10.0, 50.0, 100.0, 200.0]
    assert orders == [16, 64]
    assert len(powers) * len(times) == 30


def test_native_float32_hf_validation_top_endpoint_only_reinstates_outer_pixel():
    entry = _entry()
    from sic_cu.train.task08_f2_formal import fork_f2_e0

    sources = validate_task07_sources()
    source = sources[0]
    model = fork_f2_e0(sources, 0, torch.device("cpu")).model
    diagnostics = load_yaml("configs/optimization_v4.yaml")["diagnostics"]
    old, corrected = entry._native_f32_radius_records(
        model, 0, torch.device("cpu"), diagnostics, source.hf_validation_powers_w)
    assert len(old) == len(corrected) == 9
    for power, expected in ((115.2, (1919, 19)), (403.0, (2424, 24)),
                            (630.5, (2929, 29))):
        before = [row for row in old if row["power_w"] == power]
        after = [row for row in corrected if row["power_w"] == power]
        assert len(before) == len(after) == 3
        assert sum(row["sample_count"] for row in after) == expected[0]
        assert after[0]["sample_count"] == before[0]["sample_count"]
        assert after[1]["sample_count"] == before[1]["sample_count"]
        assert after[2]["sample_count"] - before[2]["sample_count"] == expected[1]


def test_true_full_five_source_stage_model_reload_before_energy(tmp_path):
    entry = _entry()
    from sic_cu.train.task08_f2_formal import run_task08_f2_formal

    diagnostic = tmp_path / "F2独立重载仅CPU短通路"
    run_task08_f2_formal(seed=0, output_directory=diagnostic,
                         session_epoch_limit=1, diagnostic_only=True,
                         device_name="cpu")
    sources = validate_task07_sources()
    best = torch.load(diagnostic / "阶段_观测最佳.pt", map_location="cpu",
                      weights_only=False)
    info = {"配对来源": sources[0], "阶段文件": {"best": diagnostic / "阶段_观测最佳.pt"}}
    loaded = entry._stage_model(info, "best", torch.device("cpu"))
    assert all(torch.equal(best["model_state"][name].cpu(), value.cpu())
               for name, value in loaded.state_dict().items())
    assert loaded.low_fidelity_model is not None
    assert best["metadata"]["当前真实LF张量SHA256"] == sources[0].lf_tensor_sha256


def test_independent_initial_recompute_rejects_hf_tensor_and_four_rng_poisoning(tmp_path):
    entry = _entry()
    from sic_cu.train.task08_f2_formal import run_task08_f2_formal

    diagnostic = tmp_path / "F2仅CPU合规初态复算"
    run_task08_f2_formal(seed=0, output_directory=diagnostic,
                         session_epoch_limit=1, diagnostic_only=True,
                         device_name="cpu")
    source_set = validate_task07_sources()
    initial = torch.load(diagnostic / "阶段_初始.pt", map_location="cpu", weights_only=False)
    entry.verify_initial_origin(initial, source_set, 0, torch.device("cpu"))
    bad = copy.deepcopy(initial)
    bad["model_state"]["correction.0.weight"][0, 0] += 1.0
    with pytest.raises(ValueError, match="初态|E0|张量"):
        entry.verify_initial_origin(bad, source_set, 0, torch.device("cpu"))
    bad = copy.deepcopy(initial)
    cpu_rng = bad["random_state"]["torch_cpu"]
    cpu_rng[0] ^= 1
    with pytest.raises(ValueError, match="初态|四|随机"):
        entry.verify_initial_origin(bad, source_set, 0, torch.device("cpu"))


def test_five_seed_observation_standard_deviation_is_sample_ddof_one():
    entry = _entry()
    scores = {seed: float(seed + 1) for seed in range(5)}
    measured = entry.five_seed_score_statistics(scores)
    assert measured["五种子宏平均选择分_摄氏度"] == pytest.approx(3.0)
    assert measured["五种子宏选择分样本标准差_摄氏度"] == pytest.approx(
        float(np.std(list(scores.values()), ddof=1)))
    assert measured["五种子样本标准差分母"] == 4
