"""Task-08 F1 HF-only admission: independent origins before any formal comparison."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from sic_cu.config import PROJECT_ROOT


def _entry():
    try:
        return importlib.import_module("sic_cu.train.task08_f1_hf_only")
    except ModuleNotFoundError:
        pytest.fail("任08真正HF-only DeepONet-PINN专属入场模块尚未实现")


def test_nominal_contact_h_and_emissivities_are_not_independent_hf_only_physics() -> None:
    entry = _entry()
    report = entry.audit_f1_physics_inputs()
    assert report["可作为纯HF-only物理输入"] is False
    assert "物理输入仍源于LF，无法称纯HF-only" in report["限制"]
    assert set(report["阻断参数"]) == {"LF辨识接触热阻", "LF表面温差所选顶部对流",
                                 "LF表面温差所选底部对流", "名义未测SiC辐射率",
                                 "名义未测Cu辐射率"}
    with pytest.raises(ValueError, match="LF|HF-only|物理|独立"):
        entry.require_independent_f1_physics_inputs()


def test_seed0_to_4_start_from_fresh_independent_deeponet_with_empty_adamw(
    monkeypatch,
) -> None:
    entry = _entry()
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail(
        "F1任意模型初始化不得读LF预训练/历史HF检查点/字典",
    ))
    originals = []
    for seed in range(5):
        model, optimizer = entry.initialize_f1_hf_only(seed, torch.device("cpu"))
        assert model.include_material is True
        assert model.branch_projection.in_features == 128
        assert model.trunk_projection.out_features == 128
        assert model.branch.input.weight.shape[1] == 1
        assert model.trunk.input.weight.shape[1] == 4
        assert not any("low_fidelity" in key or "response_features" in key or "teacher" in key
                       for key in model.state_dict())
        assert optimizer.state_dict()["state"] == {}
        assert len(optimizer.param_groups) == 1
        originals.append(model.branch.input.weight.detach().clone())
    assert all(not torch.equal(a, b) for i, a in enumerate(originals) for b in originals[i + 1:])


def test_spoofed_lf_checkpoint_kwarg_is_rejected_before_any_model_load(monkeypatch) -> None:
    entry = _entry()
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail(
        "错误F1 LF加载不许发生",
    ))
    with pytest.raises(ValueError, match="LF|字典|预训练|来源|HF-only"):
        entry.initialize_f1_hf_only(0, torch.device("cpu"),
                                    model_kwargs={"lf_checkpoint": "historical.pt"})


def test_real_hf_12_train_3_validation_ir_and_synchronized_hot_cold_only() -> None:
    entry = _entry()
    observations = entry.load_f1_hf_observations(torch.device("cpu"))
    from sic_cu.data.splits import build_power_splits

    splits = build_power_splits()
    assert len(observations["train_ir"]) == 29593
    assert {round(float(power), 4) for power in torch.unique(observations["train_ir"].tensors[0][:, 3])} == splits.hf_train
    assert {round(float(power), 4) for power in torch.unique(observations["validation_ir"].tensors[0][:, 3])} == splits.hf_validation
    for split, powers in (("train_sensor", splits.hf_train),
                          ("validation_sensor", splits.hf_validation)):
        coordinates, target, delta, baseline = observations[split]
        assert len(coordinates) == len(target) == len(delta) == len(baseline) > 0
        assert {round(float(power), 4) for power in torch.unique(coordinates[:, 3])} == powers
        for power in powers:
            condition = torch.isclose(coordinates[:, 3], torch.tensor(power), atol=1e-4, rtol=0.0)
            radii = {round(float(r), 4) for r in torch.unique(coordinates[condition][:, 0])}
            assert radii == {0.028, 0.0415}
        assert set(torch.unique(coordinates[:, 4]).tolist()) == {0.0}


def test_formal_f1_inference_forbidden_before_task07_freeze_and_independent_physics(
    tmp_path, monkeypatch,
) -> None:
    entry = _entry()
    monkeypatch.setattr(entry, "load_f1_hf_observations", lambda *_args, **_kwargs: pytest.fail(
        "任07未冻结及F1物理来源未解决前不得载训练标签或产生结果",
    ))
    output = tmp_path / "不得伪F1正式入口"
    with pytest.raises(ValueError, match="任07|冻结|LF|HF-only|物理"):
        entry.run_task08_f1_formal(seed=0, output_directory=output)
    assert not output.exists()


def test_single_step_real_hf_label_cpu_diagnostic_is_not_comparable_f1(tmp_path) -> None:
    entry = _entry()
    output = tmp_path / "HF-only单步来源诊断"
    report = entry.run_task08_f1_cpu_diagnostic(seed=0, output_directory=output)
    assert "不可作正式F1" in report["资格"]
    assert report["物理损失优化步"] == 0
    assert report["HF观测优化步"] == 1
    assert report["旧test_Data温度标签读取"] is False
    assert "物理输入仍源于LF，无法称纯HF-only" in report["物理来源限制"]
    assert report["从LF检查点或字典初始化"] is False
    assert not (output / "best.pt").exists()
    assert (output / "诊断训练状态.pt").is_file()
    state = torch.load(output / "诊断训练状态.pt", map_location="cpu", weights_only=False)
    assert state["metadata"]["正式资格"] is False
    assert state["optimizer_state"]["state"]
    for relative, expected in report["逐工件SHA256"].items():
        from sic_cu.data.common import sha256_file
        assert sha256_file(output / relative) == expected


def test_cpu_diagnostic_must_not_write_inside_historical_b0_source_tree(
    tmp_path, monkeypatch,
) -> None:
    entry = _entry()
    monkeypatch.setattr(entry, "PROJECT_ROOT", tmp_path)
    old_source = tmp_path / "reports/runs/V4旧B0/seed0"
    old_source.mkdir(parents=True)
    sentinel = old_source / "原历史来源.txt"
    sentinel.write_bytes(b"keep the source untouched")
    monkeypatch.setattr(entry, "load_f1_hf_observations", lambda *_args, **_kwargs: pytest.fail(
        "不许在旧来源目录内提早读观测/生成诊断结果",
    ))
    output = old_source / "新误入的F1结果"
    with pytest.raises(ValueError, match="历史|B0|来源|目录|写"):
        entry.run_task08_f1_cpu_diagnostic(seed=0, output_directory=output)
    assert not output.exists()
    assert sentinel.read_bytes() == b"keep the source untouched"


def test_cli_has_only_cpu_diagnostic_and_rejects_non_diag_without_side_effects(
    tmp_path, monkeypatch,
) -> None:
    script = PROJECT_ROOT / "scripts/28_check_task08_f1_hf_only.py"
    spec = importlib.util.spec_from_file_location("f1_source_cli_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "F1禁无资格命令"
    monkeypatch.setattr(sys, "argv", [str(script), "--seed", "0", "--output", str(output)])
    with pytest.raises(ValueError, match="诊断|正式|任07|LF|HF-only"):
        module.main()
    assert not output.exists()
