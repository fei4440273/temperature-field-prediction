"""任08 F2工程能源Float64审计副本，不变更冻结的训练原件。"""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.eval.energy_v5 import AxisymmetricGeometry, _volume_integrals, audit_schedule_energy
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.train.task07_source import validate_task07_sources


RUN0 = PROJECT_ROOT / (
    "研究记录/任务08_贡献消融/"
    "F2_事务修订正式_种子0_20260916T040215+0800"
)


def _entry(name: str):
    path = PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reloaded_original():
    old = _entry("33_audit_task08_f2_states.py")
    stage = RUN0 / "阶段_观测最佳.pt"
    source = validate_task07_sources()[0]
    model = old._stage_model(
        {"配对来源": source, "阶段文件": {"best": stage}},
        "best", torch.device("cpu"),
    )
    return model, stage


def _single_volume(model):
    coordinates = torch.tensor([[0.008, -0.001, 1.0, 55.0, 0.0]], dtype=torch.float64)
    weights = torch.ones(1, dtype=torch.float64)
    return _volume_integrals(model, coordinates, weights, load_materials()[0], 1)


def test_locked_old_float32_stage_reproduces_real_double_float_failure():
    model, _ = _reloaded_original()
    assert {tensor.dtype for tensor in model.parameters()} == {torch.float32}
    with pytest.raises(RuntimeError, match="same dtype|Double and Float"):
        _single_volume(model)


def test_new_energy_only_adapter_preserves_original_stage_and_every_tensor():
    new = _entry("34_audit_task08_f2_energy_double.py")
    model, stage = _reloaded_original()
    before_hash = hashlib.sha256(stage.read_bytes()).hexdigest()
    snapshot = torch.load(stage, map_location="cpu", weights_only=False)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    audited = new.prepare_double_audit_model(model, snapshot)
    assert audited is model
    assert not audited.training
    assert all(not parameter.requires_grad for parameter in audited.parameters())
    assert all(parameter.dtype == torch.float64 for parameter in audited.parameters())
    assert all(torch.equal(value, before[name].to(value.dtype))
               for name, value in audited.state_dict().items())
    assert hashlib.sha256(stage.read_bytes()).hexdigest() == before_hash
    assert audited.low_fidelity_model is not None
    storage, residual = _single_volume(audited)
    assert math.isfinite(storage) and math.isfinite(residual)


def test_audited_copy_rejects_poisoned_snapshot_without_mutating_model():
    new = _entry("34_audit_task08_f2_energy_double.py")
    model, stage = _reloaded_original()
    snapshot = torch.load(stage, map_location="cpu", weights_only=False)
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    snapshot["model_state"]["correction.0.weight"][0, 0] += 1.0
    with pytest.raises(ValueError, match="阶段|原张量|不一致"):
        new.prepare_double_audit_model(model, snapshot)
    assert all(torch.equal(value, original[name])
               for name, value in model.state_dict().items())


def test_float64_audit_copy_two_actual_orders_preserve_watts_and_divergence():
    new = _entry("34_audit_task08_f2_energy_double.py")
    model, stage = _reloaded_original()
    snapshot = torch.load(stage, map_location="cpu", weights_only=False)
    audited = new.prepare_double_audit_model(model, snapshot)
    evidence = audit_schedule_energy(
        audited, load_materials(), load_resolved_boundary_conditions(),
        AxisymmetricGeometry.from_config(load_yaml("configs/geometry.yaml")),
        powers_w=(55.0,), times_s=(1.0,), orders=(2, 3), device=torch.device("cpu"),
    )
    assert len(evidence["原始能量"]) == len(evidence["原始散度"]) == 2
    assert evidence["汇总"]["原定义相对平衡分母已保持"]
    assert evidence["汇总"]["工程平衡与原瓦数逐行一致"]
    for energy, div in zip(evidence["原始能量"], evidence["原始散度"]):
        assert energy["quadrature_order"] == div["quadrature_order"]
        assert math.isfinite(energy["balance_w"])
        assert math.isfinite(div["integrated_pde_residual_w"])
        assert math.isclose(
            div["explained_engineering_balance_w"],
            div["integrated_pde_residual_w"]
            - div["interface_two_sided_flux_w"]
            - div["boundary_flux_residual_w"],
            rel_tol=1e-8, abs_tol=1e-5,
        )


def test_new_formal_auditor_rejects_project_external_output_before_any_write():
    new = _entry("34_audit_task08_f2_energy_double.py")
    outside = PROJECT_ROOT.parent / "任08F2项目外能源输出拒绝_20260916"
    assert not outside.exists()
    with pytest.raises(ValueError, match="项目内|写入范围"):
        new.audit_state_energy_double(
            RUN0, state="best", output_directory=outside,
            registry_sha256="f" * 64, device_name="cuda",
        )
    assert not outside.exists()


def test_new_formal_auditor_cannot_publish_cpu_as_real_energy(tmp_path, monkeypatch):
    new = _entry("34_audit_task08_f2_energy_double.py")
    output = tmp_path / "CPU不冒真CUDA能源"
    monkeypatch.setattr(new, "_require_energy_registration", lambda *_: {})
    with pytest.raises(ValueError, match="CUDA|CPU"):
        new.audit_state_energy_double(
            RUN0, state="best", output_directory=output,
            registry_sha256="f" * 64, device_name="cpu",
        )
    assert not output.exists()
