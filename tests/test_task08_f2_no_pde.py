"""任08 F2: remove only the HF interior PDE training loss, not other physics."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import sys
from dataclasses import replace

import pytest
import torch
from torch import nn

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.losses.physics import PhysicsLossComputer, PhysicsLossWeights
from sic_cu.physics.collocation import sample_collocation
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions


class TrainableAnalyticTemperature(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.amplitude = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 295.15 + self.amplitude * (
            100.0 * x[:, 0:1].square() + 10.0 * x[:, 1:2].square()
            + x[:, 2:3] + 0.001 * x[:, 3:4]
        )


def _entry():
    return importlib.import_module("sic_cu.train.task08_f2_no_pde")


def _physics(**kwargs) -> PhysicsLossComputer:
    return PhysicsLossComputer(
        load_materials(), load_resolved_boundary_conditions(),
        PhysicsLossWeights(pde=1.0, boundary=1.0, initial=1.0, interface=1.0),
        **kwargs,
    )


def test_default_true_preserves_original_physics_values_and_gradients() -> None:
    model = TrainableAnalyticTemperature()
    batch = sample_collocation(4, torch.device("cpu"), seed=8_080_101)
    original = _physics()(model, batch)
    gradient = torch.autograd.grad(original["physics_total"], model.amplitude)[0]
    explicit = _physics(compute_pde=True)(model, batch)
    explicit_gradient = torch.autograd.grad(explicit["physics_total"], model.amplitude)[0]
    assert set(original) == set(explicit) == {
        "pde", "initial", "boundary", "interface", "physics_total",
    }
    assert all(torch.equal(original[key], explicit[key]) for key in original)
    assert torch.equal(gradient, explicit_gradient)
    assert all(torch.isfinite(value) for value in original.values())
    baseline = {
        "pde": 7.561703387182206e-05,
        "initial": 4.6004588512005284e-06,
        "boundary": 0.1187693253159523,
        "interface": 4.5704211970587494e-08,
        "physics_total": 0.11884959042072296,
    }
    for name, measured in baseline.items():
        assert float(original[name]) == pytest.approx(measured, rel=1e-11, abs=1e-13)
    assert float(gradient) == pytest.approx(0.23386868199228772, rel=1e-11)


def test_f2_no_pde_does_not_evaluate_nan_residual_and_retains_three_gradients(
    monkeypatch,
) -> None:
    import sic_cu.losses.physics as shared

    model = TrainableAnalyticTemperature()
    batch = sample_collocation(4, torch.device("cpu"), seed=8_080_102)

    def nan_residual(_model, interior, _ids, _materials):
        return torch.full((len(interior), 1), float("nan"), device=interior.device)

    monkeypatch.setattr(shared, "axisymmetric_heat_residual", nan_residual)
    legacy_zero_weight = PhysicsLossComputer(
        load_materials(), load_resolved_boundary_conditions(),
        PhysicsLossWeights(pde=0.0, boundary=1.0, initial=1.0, interface=1.0),
    )(model, batch)
    assert torch.isnan(legacy_zero_weight["physics_total"])
    f2 = PhysicsLossComputer(
        load_materials(), load_resolved_boundary_conditions(),
        PhysicsLossWeights(pde=0.0, boundary=1.0, initial=1.0, interface=1.0),
        compute_pde=False,
    )(model, batch)
    assert f2["pde"] is None  # Never encode 'not evaluated' as a zero error.
    assert torch.isfinite(f2["physics_total"])
    assert torch.equal(
        f2["physics_total"], f2["initial"] + f2["boundary"] + f2["interface"],
    )
    assert torch.isfinite(torch.autograd.grad(f2["physics_total"], model.amplitude,
                                             retain_graph=True)[0])
    for term in ("initial", "boundary", "interface"):
        assert torch.isfinite(torch.autograd.grad(f2[term], model.amplitude,
                                                 retain_graph=True)[0])


def test_f2_no_pde_must_reject_nonzero_pde_training_weight() -> None:
    with pytest.raises(ValueError, match="PDE|pde"):
        _physics(compute_pde=False)


def test_real_pde_off_retains_exact_three_full_physics_values_and_gradients() -> None:
    model = TrainableAnalyticTemperature()
    batch = sample_collocation(4, torch.device("cpu"), seed=8_080_104)
    full = _physics()(model, batch)
    off = PhysicsLossComputer(
        load_materials(), load_resolved_boundary_conditions(),
        PhysicsLossWeights(pde=0.0, boundary=1.0, initial=1.0, interface=1.0),
        compute_pde=False,
    )(model, batch)
    assert float(full["pde"]) > 0.0
    for name in ("initial", "boundary", "interface"):
        assert torch.equal(full[name], off[name])
    assert float(off["physics_total"]) == pytest.approx(
        float(full["physics_total"] - full["pde"]), rel=1e-6, abs=1e-7,
    )
    assert torch.equal(
        torch.autograd.grad(off["physics_total"], model.amplitude)[0],
        torch.autograd.grad(full["initial"] + full["boundary"] + full["interface"],
                            model.amplitude)[0],
    )


@pytest.fixture(scope="module")
def sources():
    return _entry().load_f2_sources()


def test_five_v4_same_seed_lf_and_fresh_hf_e0_exact_initialization(sources) -> None:
    from sic_cu.train.task07_source import fork_task07_initialization

    entry = _entry()
    assert set(sources) == set(range(5))
    assert len({row.lf_tensor_sha256 for row in sources.values()}) == 5
    starts = []
    for seed in range(5):
        source = sources[seed]
        f2 = entry.fork_f2_start(sources, seed, torch.device("cpu"))
        reference = fork_task07_initialization(sources, seed, "E0", torch.device("cpu"))
        assert len(source.lf_train_powers_w) == 60
        assert len(source.lf_validation_powers_w) == 10
        assert len(source.hf_train_powers_w) == 12
        assert len(source.hf_validation_powers_w) == 3
        assert source.lf_tensor_sha256 == f2.lf_tensor_sha256
        assert source.lf_checkpoint_sha256 == f2.lf_checkpoint_sha256
        assert f2.optimizer.state_dict()["state"] == {}
        assert f2.model.correction[0].in_features == 6
        assert all(not parameter.requires_grad for parameter in
                   f2.model.low_fidelity_model.parameters())
        assert all(parameter.requires_grad for parameter in
                   f2.model.correction.parameters())
        assert set(f2.model.state_dict()) == set(reference.model.state_dict())
        assert all(torch.equal(value, reference.model.state_dict()[name])
                   for name, value in f2.model.state_dict().items())
        assert torch.equal(f2.random_state["torch_cpu"], reference.random_state["torch_cpu"])
        assert f2.random_state["torch_cuda"] is None
        assert f2.selected_for_training is False
        starts.append(f2.model.correction[0].weight.detach().cpu().clone())
    assert all(not torch.equal(a, b) for index, a in enumerate(starts)
               for b in starts[index + 1:])


def test_f2_refuses_spoofed_lf_source_before_model_initialization(sources) -> None:
    mixed = dict(sources)
    mixed[1] = replace(sources[0], seed=1)
    with pytest.raises(ValueError, match="来源|source|seed|SHA"):
        _entry().fork_f2_start(mixed, 1, torch.device("cpu"))


def test_f2_frozen_low_parameters_still_provide_coordinate_gradients(sources) -> None:
    model = _entry().fork_f2_start(sources, 0, torch.device("cpu")).model.double()
    x = torch.tensor([[0.010, -0.010, 60.0, 115.0, 1.0],
                      [0.036, -0.014, 80.0, 550.0, 0.0]],
                     dtype=torch.float64, requires_grad=True)
    low_coordinate_gradient = torch.autograd.grad(
        model(x, fidelity="low").sum(), x, retain_graph=True,
    )[0]
    high_coordinate_gradient = torch.autograd.grad(model(x).sum(), x)[0]
    assert torch.isfinite(low_coordinate_gradient).all()
    assert torch.isfinite(high_coordinate_gradient).all()
    assert low_coordinate_gradient[:, :2].abs().sum() > 0
    assert high_coordinate_gradient[:, :2].abs().sum() > 0
    assert all(parameter.grad is None and not parameter.requires_grad
               for parameter in model.low_fidelity_model.parameters())


def test_f2_real_hf_12_train_3_validation_synchronized_ir_hot_cold(sources) -> None:
    entry = _entry()
    observed = entry.load_f2_observations(sources[0], torch.device("cpu"))
    assert len(observed["train_ir"]) == 29593
    for split, allowed in (("train", set(sources[0].hf_train_powers_w)),
                           ("validation", set(sources[0].hf_validation_powers_w))):
        ir_x = observed[f"{split}_ir"].tensors[0]
        sensor_x, _target, _delta, baseline = observed[f"{split}_sensor"]
        assert {round(float(power), 4) for power in torch.unique(ir_x[:, 3])} == allowed
        assert {round(float(power), 4) for power in torch.unique(sensor_x[:, 3])} == allowed
        assert torch.all(ir_x[:, 4] == 1) and torch.all(sensor_x[:, 4] == 0)
        assert torch.all(sensor_x[baseline, 3] == sensor_x[:, 3])
        assert torch.all(sensor_x[baseline, 0] == sensor_x[:, 0])
        for power in allowed:
            mask = torch.isclose(sensor_x[:, 3], torch.tensor(power), atol=1e-4, rtol=0.0)
            assert {round(float(r), 4) for r in torch.unique(sensor_x[mask][:, 0])} == {
                0.028, 0.0415,
            }


def test_cross_power_hot_cold_baseline_is_rejected_before_one_training_round(
    sources, monkeypatch,
) -> None:
    entry = _entry()
    original = entry._sensor_tensors

    def cross_power_baseline(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[1] != "train":
            return result
        coordinates, targets, deltas, baseline = result
        altered = baseline.clone()
        altered[0] = torch.nonzero(coordinates[:, 3] != coordinates[0, 3])[0, 0]
        return coordinates, targets, deltas, altered

    monkeypatch.setattr(entry, "_sensor_tensors", cross_power_baseline)
    with pytest.raises(ValueError, match="Hot|Cold|基线|功率|同步|同一"):
        entry.load_f2_observations(sources[0])


def test_training_physics_excludes_only_pde_independent_audit_keeps_all_terms(
    monkeypatch,
) -> None:
    import sic_cu.losses.physics as shared

    f2 = _entry().make_f2_training_physics(torch.device("cpu"))
    audit = _entry().make_f2_independent_audit_physics(torch.device("cpu"))
    assert f2.weights == PhysicsLossWeights(pde=0.0, boundary=1.0, initial=1.0,
                                            interface=1.0)
    assert audit.weights == PhysicsLossWeights(pde=1.0, boundary=1.0, initial=1.0,
                                               interface=1.0)
    assert f2.compute_pde is False and audit.compute_pde is True
    model = TrainableAnalyticTemperature()
    batch = sample_collocation(4, torch.device("cpu"), seed=8_080_103)

    def called(*_args, **_kwargs):
        raise RuntimeError("PDE_CALLED")

    monkeypatch.setattr(shared, "axisymmetric_heat_residual", called)
    f2_losses = f2(model, batch)
    assert f2_losses["pde"] is None
    with pytest.raises(RuntimeError, match="PDE_CALLED"):
        audit(model, batch)


def test_no_formal_f2_training_before_task07_freeze_and_registration(
    tmp_path, monkeypatch,
) -> None:
    entry = _entry()
    monkeypatch.setattr(entry, "load_f2_observations", lambda *_args, **_kwargs:
                        pytest.fail("冻结前F2正式入口不得读观测标签"))
    output = tmp_path / "不得伪作F2正式候选"
    with pytest.raises(ValueError, match="任07|冻结|正式|事前"):
        entry.run_task08_f2_formal(seed=0, output_directory=output)
    assert not output.exists()


def test_one_true_hf_round_is_only_cpu_diagnostic_with_16_updates_and_frozen_lf(
    tmp_path,
) -> None:
    entry = _entry()
    output = tmp_path / "F2仅CPU同来源一轮诊断"
    report = entry.run_task08_f2_cpu_diagnostic(seed=0, output_directory=output)
    assert report["正式资格"] is False
    assert report["HF合法训练功率数"] == 12
    assert report["HF合法验证功率数"] == 3
    assert report["LF训练功率数"] == 60 and report["LF验证功率数"] == 10
    assert report["HF训练观测点"] == 29593
    assert report["HF观测优化步"] == 15
    assert report["物理优化步"] == 1
    assert report["物理配点"] == 256
    assert report["LF真实回放训练点"] == 0
    assert report["LF联合实际轮次"] == 0
    assert report["HF训练传感器点"] > 0
    assert report["冻结LF张量未变"] is True
    assert report["训练PDE残差"] == "未计算；不表示误差为零"
    assert report["独立能源仅用于审计不反向传播"] is True
    assert report["旧test_Data温度标签读取"] is False
    assert not (output / "best.pt").exists()
    state = torch.load(output / "诊断训练状态.pt", map_location="cpu", weights_only=False)
    assert state["metadata"]["正式资格"] is False
    assert state["metadata"]["训练PDE残差"] == "未计算；不表示误差为零"
    assert state["optimizer_state"]["state"]
    assert all(float(momentum["step"]) == 16 for momentum in
               state["optimizer_state"]["state"].values())
    for relative, expected in report["逐工件SHA256"].items():
        assert sha256_file(output / relative) == expected


def test_missing_explicit_pde_skip_blocks_cpu_even_before_real_training_labels(
    tmp_path, monkeypatch,
) -> None:
    entry = _entry()
    monkeypatch.setattr(entry, "_require_explicit_skip", lambda: (_ for _ in ()).throw(
        ValueError("PDE显式跳过未开放"),
    ))
    monkeypatch.setattr(entry, "load_f2_sources", lambda: pytest.fail(
        "任07共享物理未开放前不能打开来源检查点或合法HF观测",
    ))
    output = tmp_path / "PDE没有跳过功能的禁止入口"
    with pytest.raises(ValueError, match="PDE|跳过"):
        entry.run_task08_f2_cpu_diagnostic(seed=0, output_directory=output)
    assert not output.exists()


def test_shared_training_yaml_remains_pde_one_and_f2_has_no_energy_soft_loss() -> None:
    entry = _entry()
    before = sha256_file(PROJECT_ROOT / "configs/training.yaml")
    assert load_yaml("configs/training.yaml")["loss_weights"]["pde"] == 1.0
    assert entry.F2_TRAINING_WEIGHTS == {
        "ir": 1.0, "sensor_absolute": 5.0, "sensor_delta": 1.0,
        "low_fidelity": 1.0, "pde": 0.0, "initial": 1.0,
        "boundary": 1.0, "interface": 1.0,
    }
    assert "energy" not in entry.F2_TRAINING_WEIGHTS
    assert before == sha256_file(PROJECT_ROOT / "configs/training.yaml")


@pytest.mark.parametrize("field,wrong", [
    ("selection_metric_version", "weighted_v0"),
    ("early_stopping", {"patience": 20, "min_delta": 0.0001}),
    ("multifidelity_selection_weights", {
        "ir_rmse": 1.0, "sensor_absolute_rmse": 1.0, "sensor_delta_rmse": 1.0,
    }),
])
def test_f2_refuses_drifted_same_budget_selection_before_old_b0_checkpoint_load(
    monkeypatch, field, wrong,
) -> None:
    entry = _entry()
    original = entry.load_yaml

    def changed_training(path):
        result = original(path)
        if path != "configs/training.yaml":
            return result
        forged = copy.deepcopy(result)
        forged[field] = wrong
        return forged

    monkeypatch.setattr(entry, "load_yaml", changed_training)
    monkeypatch.setattr(entry, "validate_task07_sources", lambda: pytest.fail(
        "任08 F2固定同预算选分漂移时不得打开旧B0模型来源",
    ))
    with pytest.raises(ValueError, match="选分|早停|同预算|配置"):
        entry.load_f2_sources()


def test_f2_cli_rejects_unmarked_diagnostic_without_touching_output(
    tmp_path, monkeypatch,
) -> None:
    script = PROJECT_ROOT / "scripts/29_check_task08_f2_no_pde.py"
    spec = importlib.util.spec_from_file_location("f2_source_cli_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "无资格F2入口"
    monkeypatch.setattr(sys, "argv", [str(script), "--seed", "0", "--output", str(output)])
    with pytest.raises(ValueError, match="diagnostic|诊断|正式|冻结"):
        module.main()
    assert not output.exists()
