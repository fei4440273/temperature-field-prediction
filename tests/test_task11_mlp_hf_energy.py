"""Task-11 MLP energy admission and unmodified Watt evidence."""

from __future__ import annotations

import copy
import importlib
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from sic_cu.eval.energy_v5 import chinese_energy_terms
from sic_cu.models import AdditiveCorrectionModel, ModelScales
from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.train.simulation import build_model


def _energy():
    try:
        return importlib.import_module("sic_cu.eval.task11_mlp_hf_energy")
    except ModuleNotFoundError:
        pytest.fail("Task-11 exclusive MLP HF energy adapter is absent")


def _registration():
    return {"独立能源审核": {
        "功率_瓦": [55.0, 115.2, 364.3, 403.0, 630.5, 729.0],
        "时刻_秒": [1.0, 10.0, 50.0, 100.0, 200.0],
        "求积阶数": [16, 64], "功率时刻点数": 30,
        "模型状态": ["best", "final"], "计算dtype": "float64",
        "名义吸收归一均值筛查": 0.05,
        "名义吸收归一95分位筛查": 0.10,
        "不用于训练选模": True, "工程安全阈值": False,
    }}


def _analytic_audit():
    # Analytic unit-test fixture, not a numerical experiment or measured data.
    energy, divergence = [], []
    for power in _registration()["独立能源审核"]["功率_瓦"]:
        for time_s in _registration()["独立能源审核"]["时刻_秒"]:
            for order in (16, 64):
                absorbed = 0.7 * power
                storage = 0.3 * power + 0.001 * order
                cooling, convection, radiation = 0.1 * power, 2.0, 1.0
                total_cooling = cooling + convection + radiation
                balance = storage + total_cooling - absorbed
                denominator = max(abs(absorbed), abs(storage),
                                  abs(total_cooling), 1e-12)
                gap = 0.064 / order
                energy.append({
                    "power_w": power, "time_s": time_s,
                    "quadrature_order": order,
                    "absorbed_power_w": absorbed, "storage_rate_w": storage,
                    "cooling_heat_w": cooling,
                    "convection_heat_w": convection,
                    "radiation_heat_w": radiation, "balance_w": balance,
                    "relative_balance_denominator_w": denominator,
                    "relative_balance": balance / denominator,
                    "boundary_gradient_mode": "autograd",
                    "outer_epsilon_m": 1e-6,
                })
                divergence.append({
                    "power_w": power, "time_s": time_s,
                    "quadrature_order": order,
                    "integrated_pde_residual_w": balance + 10.0 - gap,
                    "interface_two_sided_flux_w": 4.0,
                    "boundary_flux_residual_w": 6.0,
                    "explained_engineering_balance_w": balance - gap,
                    "engineering_balance_w": balance,
                    "engineering_explanation_gap_w": gap,
                    "silicon_carbide_pde_residual_w": balance + 10.0 - gap,
                    "copper_outer_ring_pde_residual_w": 0.0,
                    "copper_below_sic_pde_residual_w": 0.0,
                    "outer_temperature_max_abs_deviation_c": 0.0,
                })
    frame, summary = chinese_energy_terms(pl.DataFrame(energy), 64)
    decomposition = pl.DataFrame(divergence).filter(
        pl.col("quadrature_order") == 64,
    ).select(
        pl.col("power_w").alias("功率_瓦"),
        pl.col("time_s").alias("时刻_秒"),
        pl.col("integrated_pde_residual_w").alias("体积分残差V_瓦"),
        pl.col("interface_two_sided_flux_w").alias("界面双侧通量J_瓦"),
        pl.col("boundary_flux_residual_w").alias("边界失配D_瓦"),
        pl.col("explained_engineering_balance_w").alias("解释工程平衡_瓦"),
        pl.col("engineering_balance_w").alias("原工程平衡_瓦"),
        pl.col("engineering_explanation_gap_w").alias("工程解释剩余差_瓦"),
        pl.col("silicon_carbide_pde_residual_w").alias("SiC体内积分残差_瓦"),
        pl.col("copper_outer_ring_pde_residual_w").alias("Cu外环积分残差_瓦"),
        pl.col("copper_below_sic_pde_residual_w").alias("Cu下层积分残差_瓦"),
        pl.col("outer_temperature_max_abs_deviation_c").alias(
            "水冷边界最大温差_摄氏度"),
    )
    summary.update({
        "功率时刻审核行数": 30,
        "原定义相对平衡分母已保持": True,
        "工程平衡与原瓦数逐行一致": True,
        "最大相邻阶变化对吸收功率比": max(
            abs(energy[index]["balance_w"] - energy[index - 1]["balance_w"])
            / energy[index]["absorbed_power_w"]
            for index in range(1, 60, 2)),
        "最大分解剩余差_瓦": 0.001,
    })
    return {"原始能量": energy, "原始散度": divergence,
            "指标明细": frame, "物理分解": decomposition, "汇总": summary}


def test_schedule_requires_frozen_original_conditions_and_nominal_only_rules():
    module = _energy()
    powers, times, orders = module.task11_fixed_energy_schedule(_registration())
    assert powers == [55.0, 115.2, 364.3, 403.0, 630.5, 729.0]
    assert times == [1.0, 10.0, 50.0, 100.0, 200.0]
    assert orders == [16, 64]
    changed = _registration()
    changed["独立能源审核"]["工程安全阈值"] = True
    with pytest.raises(ValueError, match="名义|工程|冻结"):
        module.task11_fixed_energy_schedule(changed)
    changed = _registration()
    changed["独立能源审核"]["时刻_秒"][-1] = 199.0
    with pytest.raises(ValueError, match="固定|冻结|能源"):
        module.task11_fixed_energy_schedule(changed)


def test_energy_output_requires_canonical_seed_and_new_exact_state_child(tmp_path):
    module = _energy()
    run = tmp_path / "研究记录/任务11_外部对照/正式新MLP公平训练/正式MLP_HF_seed0"
    target = run / "独立原能源_观测最佳_20260916T160000+0800"
    assert module.task11_energy_output(run, target, "best", 0,
                                       project_root=tmp_path) == target
    assert not target.exists()
    with pytest.raises(ValueError, match="seed|目录|身份"):
        module.task11_energy_output(run, target, "best", 1,
                                    project_root=tmp_path)
    with pytest.raises(ValueError, match="状态|best|final"):
        module.task11_energy_output(run, target, "physical", 0,
                                    project_root=tmp_path)
    with pytest.raises(ValueError, match="目录|状态"):
        module.task11_energy_output(run, run / "能源", "best", 0,
                                    project_root=tmp_path)
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        module.task11_energy_output(run, target, "best", 0,
                                    project_root=tmp_path)


def test_energy_output_rejects_symlink_escape_without_creating_output(tmp_path):
    module = _energy()
    root = tmp_path / "root"
    run = root / "研究记录/任务11_外部对照/正式新MLP公平训练/正式MLP_HF_seed0"
    outside = tmp_path / "outside"
    outside.mkdir()
    run.parent.mkdir(parents=True)
    run.symlink_to(outside, target_is_directory=True)
    target = run / "独立原能源_观测最佳_20260916T160000+0800"
    with pytest.raises(ValueError, match="项目|目录|软链"):
        module.task11_energy_output(run, target, "best", 0, project_root=root)
    assert not (outside / target.name).exists()


def test_raw_energy_verifier_preserves_both_orders_watts_denominator_and_vjd():
    module = _energy()
    audit = _analytic_audit()
    verified = module.verify_task11_energy_raw(audit)
    assert math.isclose(verified["16阶V-J-D散度积分剩余差最大_瓦"], 0.004)
    assert math.isclose(verified["64阶V-J-D散度积分剩余差最大_瓦"], 0.001)
    assert math.isclose(verified["绝对平衡宏均值_瓦"],
                        np.mean([abs(row["balance_w"])
                                 for row in audit["原始能量"][1::2]]))


@pytest.mark.parametrize("field", ["relative_balance_denominator_w", "balance_w"])
def test_raw_energy_verifier_rejects_changed_original_watt_definition(field):
    module = _energy()
    audit = _analytic_audit()
    audit["原始能量"][0][field] += 1.0
    with pytest.raises(ValueError, match="瓦数|分母|原定义|一致"):
        module.verify_task11_energy_raw(audit)


def test_raw_energy_verifier_rejects_missing_row_and_wrong_summary():
    module = _energy()
    audit = _analytic_audit()
    audit["原始散度"].pop()
    with pytest.raises(ValueError, match="60|30|双阶"):
        module.verify_task11_energy_raw(audit)
    audit = _analytic_audit()
    audit["汇总"]["绝对平衡95分位_瓦"] += 0.1
    with pytest.raises(ValueError, match="汇总|原件|一致"):
        module.verify_task11_energy_raw(audit)


def test_model_view_only_accepts_actual_additive_mlp_not_deeponet():
    module = _energy()
    scales = ModelScales()
    low_kwargs = {"width": 128, "depth": 5, "activation": "tanh",
                  "include_material": True}
    correction_kwargs = {"width": 128, "depth": 4, "activation": "tanh",
                         "include_material": True,
                         "hard_initial_temperature_k": 295.15,
                         "initial_ramp_time_s": 0.05,
                         "correction_calibration_range_w": [55.0, 800.0],
                         "correction_support_range_w": [0.0, 800.0],
                         "correction_extrapolation_exponent": 2.0,
                         "correction_power_scaling": "none",
                         "correction_power_reference_w": 400.0,
                         "correction_direct_power_input": True,
                         "silicon_carbide_height_m": 0.012,
                         "surface_guide_output": "residual"}
    low = build_model("mlp_pinn", scales, **low_kwargs)
    model = AdditiveCorrectionModel(low, scales, **correction_kwargs)
    view = {"method": "multifidelity_correction", "low_fidelity_method": "mlp_pinn",
            "scales": scales.__dict__,
            "low_fidelity_model_kwargs": low_kwargs,
            "correction_model_kwargs": correction_kwargs,
            "model_state": model.state_dict()}
    reconstructed = module.task11_energy_model_from_view(view)
    assert type(reconstructed.low_fidelity_model) is type(low)
    assert all(torch.equal(value, reconstructed.state_dict()[name])
               for name, value in model.state_dict().items())
    wrong = {**view, "low_fidelity_method": "deeponet_pinn"}
    with pytest.raises(ValueError, match="MLP|mlp|方法|身份"):
        module.task11_energy_model_from_view(wrong)
    wrong = {**view, "method": "observation_ridge"}
    with pytest.raises(ValueError, match="MLP|mlp|方法|身份"):
        module.task11_energy_model_from_view(wrong)


def test_formal_energy_rejects_cpu_before_cuda_probe_or_writing():
    module = _energy()
    with pytest.raises(ValueError, match="CUDA|GPU"):
        module.require_task11_energy_cuda("cpu")


def test_energy_writer_preserves_all_raw_rows_and_refuses_overwrite(tmp_path):
    module = _energy()
    audit = _analytic_audit()
    payload = {"原定义相对平衡分母已保持": True, **audit["汇总"]}
    target = tmp_path / "原能源"
    original = copy.deepcopy(audit["原始能量"])
    digests = module.write_task11_energy_evidence(target, audit, payload,
                                                 project_root=tmp_path)
    assert set(digests) == {"原始能量.jsonl", "原始散度.jsonl", "指标明细.csv",
                            "物理分解.csv", "汇总指标.json"}
    written = [json.loads(line) for line in
               (target / "原始能量.jsonl").read_text(encoding="utf-8").splitlines()]
    assert written == original
    assert len(written) == 60
    assert json.loads((target / "审计工件SHA256.json").read_text(encoding="utf-8")) == digests
    with pytest.raises(FileExistsError):
        module.write_task11_energy_evidence(target, audit, payload,
                                            project_root=tmp_path)


def test_selected_model_loader_refuses_changed_sha_before_deserializing(tmp_path):
    module = _energy()
    assert hasattr(module, "load_task11_energy_model"), "Selected MLP audit loader is absent"
    original = tmp_path / "model.pt"
    original.write_bytes(b"unit-test-not-a-checkpoint")
    with pytest.raises(ValueError, match="SHA|原件|改变"):
        module.load_task11_energy_model({"模型原件": str(original),
                                        "模型SHA256": "0" * 64},
                                       torch.device("cpu"))


def test_formal_audit_requires_new_hf_gate_before_cuda_probe_or_output(monkeypatch):
    module = _energy()
    assert hasattr(module, "audit_task11_mlp_hf_energy"), "Formal MLP Watt audit is absent"
    monkeypatch.setattr(torch.cuda, "is_available",
                        lambda: pytest.fail("CUDA must not be probed before HF gate"))
    registry = PROJECT_ROOT / (
        "研究记录/任务11_外部对照/任11_新MLP五种子真LF训练事前登记_20260916.yaml")
    archive = PROJECT_ROOT / (
        "研究记录/任务11_外部对照/任11_新MLP五种子真LF源码事前冻结_20260916.tar.gz")
    catalog = PROJECT_ROOT / (
        "研究记录/任务11_外部对照/任11_新MLP_LF真实模拟70源目录_事前20260916.json")
    run = PROJECT_ROOT / (
        "研究记录/任务11_外部对照/正式新MLP公平训练/正式MLP_HF_seed0")
    target = run / "独立原能源_观测最佳_20260916T160000+0800"
    assert not target.exists()
    with pytest.raises(ValueError, match="任11|HF|登记|门禁|schema"):
        module.audit_task11_mlp_hf_energy(
            run=run, output=target, state="best", seed=0,
            registry=registry, registry_sha=sha256_file(registry),
            source_tar=archive, source_tar_sha=sha256_file(archive),
            lf_catalog=catalog, lf_catalog_sha=sha256_file(catalog),
            hf_data_catalog=catalog, hf_data_catalog_sha=sha256_file(catalog),
            device_name="cuda",
        )
    assert not target.exists()


def test_energy_cli_has_only_explicit_best_final_and_cuda_formal_mode():
    module = _energy()
    cli = PROJECT_ROOT / "scripts/52_audit_task11_mlp_hf_energy.py"
    assert cli.is_file(), "Exclusive Task-11 MLP energy CLI is absent"
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(cli), "--state", "physical",
                             "--device", "cpu"], capture_output=True, text=True,
                            check=False)
    assert result.returncode != 0
    assert "invalid choice" in result.stderr or "required" in result.stderr


def test_raw_evidence_rejects_nan_in_any_original_field():
    module = _energy()
    audit = _analytic_audit()
    audit["原始能量"][0]["diagnostic_extra"] = float("nan")
    with pytest.raises(ValueError, match="有限|NaN|原件|数值"):
        module.verify_task11_energy_raw(audit)


def test_writer_rejects_nonfinite_payload_before_directory_creation(tmp_path):
    module = _energy()
    audit = _analytic_audit()
    payload = {**audit["汇总"], "extra": float("nan")}
    target = tmp_path / "invalid"
    with pytest.raises(ValueError, match="有限|NaN|原件|数值"):
        module.write_task11_energy_evidence(target, audit, payload,
                                            project_root=tmp_path)
    assert not target.exists()


@pytest.mark.parametrize("changed_originals", [False, True])
def test_formal_orchestration_rechecks_originals_before_writer(monkeypatch, tmp_path,
                                                              changed_originals):
    module = _energy()
    import sic_cu.train.task11_mlp_hf_formal as formal

    calls = []
    target = tmp_path / "unit-test-energy-not-a-formal-result"
    model = torch.nn.Linear(1, 1)
    audit = _analytic_audit()

    def qualify(**kwargs):
        number = sum(item.startswith("gate") for item in calls) + 1
        calls.append(f"gate{number}")
        return {"登记原件": "unit-test-registry", "模型SHA256": "a" * 64,
                "完整原件SHA256": {"training.jsonl": (
                    "b" * 64 if changed_originals and number == 2 else "a" * 64)},
                "本seed新LF最佳检查点SHA256": "c" * 64,
                "本seedLF初始张量SHA256": "d" * 64}

    def cuda(device):
        assert calls == ["gate1"]
        calls.append("cuda-gate")
        return torch.device("cpu")

    def selected(*args):
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
        calls.append("legal-observations")
        return {"单元测试非实测": True}

    def watts(*args, **kwargs):
        assert all(parameter.dtype == torch.float64 and not parameter.requires_grad
                   for parameter in model.parameters())
        assert kwargs["powers_w"] == module.FIXED_POWERS
        assert kwargs["times_s"] == module.FIXED_TIMES
        assert kwargs["orders"] == tuple(module.FIXED_ORDERS)
        calls.append("float64-watts")
        return audit

    def writer(destination, raw, payload):
        assert destination == target and raw is audit
        assert payload["只读原件审核前后SHA一致"] is True
        assert payload["旧固定TEST温度读取"] is False
        calls.append("writer")
        return {}

    monkeypatch.setattr(module, "task11_energy_output", lambda *args: target)
    monkeypatch.setattr(formal, "require_task11_mlp_hf_downstream_qualification", qualify)
    monkeypatch.setattr(module, "load_yaml", lambda *args: {
        "独立能源审核": module.ENERGY_CONTRACT})
    monkeypatch.setattr(module, "require_task11_energy_cuda", cuda)
    monkeypatch.setattr(module, "load_task11_energy_model", lambda *args: (
        {"seed": 0}, model))
    monkeypatch.setattr(module, "_selected_observations", selected)
    monkeypatch.setattr(module, "load_materials", lambda *args: object())
    monkeypatch.setattr(module, "load_resolved_boundary_conditions", lambda *args: object())
    monkeypatch.setattr(module.AxisymmetricGeometry, "from_config", lambda *args: object())
    monkeypatch.setattr(module, "audit_schedule_energy", watts)
    monkeypatch.setattr(module, "write_task11_energy_evidence", writer)
    arguments = dict(run=tmp_path / "unit-test-run", output=target, state="best", seed=0,
                     registry="unit-test-registry", registry_sha="a" * 64,
                     source_tar="unit-test-tar", source_tar_sha="a" * 64,
                     lf_catalog="unit-test-lf", lf_catalog_sha="a" * 64,
                     hf_data_catalog="unit-test-hf", hf_data_catalog_sha="a" * 64)
    if changed_originals:
        with pytest.raises(ValueError, match="原件|改变|写盘"):
            module.audit_task11_mlp_hf_energy(**arguments)
        assert "writer" not in calls
    else:
        result = module.audit_task11_mlp_hf_energy(**arguments)
        assert result["只读原件审核前后SHA一致"] is True
        assert calls[-1] == "writer"
    assert calls[:5] == ["gate1", "cuda-gate", "legal-observations", "float64-watts", "gate2"]
    assert not target.exists()
