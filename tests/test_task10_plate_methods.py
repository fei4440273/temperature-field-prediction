"""TDD for board-only trainable networks and genuinely skipped F2 PDE."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
import torch
from torch import nn

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.eval.task10_plate_training_source import PlateTrainingBatch


REGISTRATION = (PROJECT_ROOT / "研究记录/任务10_独立双层场基准/"
                "正式人为多热流入场前登记.yaml")


class _AnalyticSteadyPlate(nn.Module):
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        z, _, q, material = coordinates.unbind(dim=-1)
        sic = 295.15 + q * ((.012 - z) / 120 + .0001 + .0055 / 401)
        cu = 295.15 + q * ((.0175 - z) / 401)
        return torch.where(material > .5, sic, cu).unsqueeze(1)


def test_independent_deeponet_uses_board_four_columns_and_rejects_old_rz() -> None:
    from sic_cu.models.task10_plate_deeponet import BoardDeepONet

    model = BoardDeepONet(width=16, latent_dim=16, blocks=1)
    rows = torch.tensor([[0.0, 5.0, 50000.0, 1.0],
                         [.013, 5.0, 50000.0, 0.0]], dtype=torch.float64)
    output = model.double()(rows)
    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()
    with pytest.raises(ValueError, match="四列|4|RZ"):
        model(torch.zeros(2, 5, dtype=torch.float64))
    with pytest.raises(ValueError, match="通量|q|flux"):
        model(torch.tensor([[.013, 5.0, 500.0, 0.0]], dtype=torch.float64))


def test_analytic_steady_plate_has_zero_pde_top_bottom_and_contact_residuals() -> None:
    from sic_cu.losses.task10_plate_physics import (
        RegisteredBoardPhysics, registered_board_collocation,
    )

    collocation = registered_board_collocation(REGISTRATION, 50000, points=4, time_s=50.0)
    physics = RegisteredBoardPhysics(REGISTRATION, method="F3")
    components = physics.components(_AnalyticSteadyPlate(), collocation)
    for key in ("pde", "top", "bottom", "interface_flux", "interface_jump"):
        assert torch.isfinite(components[key])
        assert float(components[key]) < 1e-14, (key, float(components[key]))
    assert components["initial"] > 0  # A steady heated profile is not a transient initial condition.
    assert physics.interior_pde_calls == 1
    assert collocation.interior.shape[1] == 4
    assert {float(q) for q in collocation.interior[:, 2]} == {50000.0}


def test_f2_never_invokes_second_order_interior_pde_but_keeps_other_physics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sic_cu.losses.task10_plate_physics import (
        RegisteredBoardPhysics, registered_board_collocation,
    )

    collocation = registered_board_collocation(REGISTRATION, 50000, points=3, time_s=75.0)
    physics = RegisteredBoardPhysics(REGISTRATION, method="F2")

    def forbidden_pde(*args, **kwargs):
        raise AssertionError("F2 evaluated a forbidden interior second derivative")

    monkeypatch.setattr(physics, "_interior_pde", forbidden_pde)
    components = physics.components(_AnalyticSteadyPlate(), collocation)
    assert components["pde"] is None
    assert physics.interior_pde_calls == 0
    assert {"initial", "top", "bottom", "interface_flux", "interface_jump"} <= components.keys()
    assert all(torch.isfinite(components[key]) for key in
               ("initial", "top", "bottom", "interface_flux", "interface_jump"))


def test_f1_requires_public_pde_while_f2_f3_share_exact_same_lf_start() -> None:
    from sic_cu.models.task10_plate_deeponet import BoardDeepONet
    from sic_cu.losses.task10_plate_physics import RegisteredBoardPhysics
    from sic_cu.train.task10_plate_methods import make_same_lf_pair

    torch.manual_seed(915)
    learned_low = BoardDeepONet(width=12, latent_dim=12, blocks=1)
    with torch.no_grad():
        learned_low.bias.fill_(.17)
    f2, f3 = make_same_lf_pair(learned_low, seed=11, width=12, latent_dim=12, blocks=1)
    assert f2.method == "F2" and f3.method == "F3"
    assert f2.low_model is not f3.low_model
    assert all(torch.equal(f2.state_dict()[name], f3.state_dict()[name])
               for name in f2.state_dict())
    assert all(not parameter.requires_grad for parameter in f2.low_model.parameters())
    assert all(not parameter.requires_grad for parameter in f3.low_model.parameters())
    assert all(parameter.requires_grad for parameter in f2.correction.parameters())
    assert RegisteredBoardPhysics(REGISTRATION, method="F1").compute_pde
    assert not RegisteredBoardPhysics(REGISTRATION, method="F2").compute_pde
    assert RegisteredBoardPhysics(REGISTRATION, method="F3").compute_pde
    assert f2(torch.tensor([[.013, 5.0, 50000.0, 0.0]])).shape == (1, 1)


def test_frozen_lf_weights_still_contribute_depth_derivatives_to_f3_pde() -> None:
    from sic_cu.models.task10_plate_deeponet import (
        BoardDeepONet, BoardMultifidelityDeepONet,
    )

    torch.manual_seed(916)
    learned_low = BoardDeepONet(width=12, latent_dim=12, blocks=1)
    with torch.no_grad():
        learned_low.branch_projection.weight.fill_(.03)
        learned_low.branch_projection.bias.fill_(.01)
    f3 = BoardMultifidelityDeepONet(
        learned_low, BoardDeepONet(width=12, latent_dim=12, blocks=1,
                                   with_lf_input=True), "F3",
    )
    point = torch.tensor([[.003, 5.0, 50000.0, 1.0]], requires_grad=True)
    low_gradient = torch.autograd.grad(f3.low_model(point).sum(), point,
                                      create_graph=True)[0][:, 0]
    high_gradient = torch.autograd.grad(f3(point).sum(), point,
                                       create_graph=True)[0][:, 0]
    assert abs(float(low_gradient)) > .01
    torch.testing.assert_close(high_gradient, low_gradient, atol=1e-5, rtol=1e-5)
    assert not any(parameter.requires_grad for parameter in f3.low_model.parameters())


@pytest.mark.parametrize("method", ["F1", "F2", "F3"])
def test_one_cpu_optimizer_step_is_finite_without_hidden_reference(method: str) -> None:
    from sic_cu.models.task10_plate_deeponet import BoardDeepONet
    from sic_cu.losses.task10_plate_physics import (
        RegisteredBoardPhysics, registered_board_collocation,
    )
    from sic_cu.train.task10_plate_methods import make_same_lf_pair, train_hf_method_step

    torch.manual_seed(34)
    if method == "F1":
        model = BoardDeepONet(width=12, latent_dim=12, blocks=1)
    else:
        low = BoardDeepONet(width=12, latent_dim=12, blocks=1)
        model = make_same_lf_pair(low, seed=33, width=12, latent_dim=12, blocks=1)[
            0 if method == "F2" else 1]
    times = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200]
    inputs = np.column_stack((np.tile([0.0, .013, .016], len(times)),
                              np.repeat(times, 3), np.full(42, 50000.),
                              np.tile([1., 0., 0.], len(times))))
    observed = PlateTrainingBatch(inputs, np.tile([[299.0], [296.0], [295.9]], (14, 1)),
                                  fold="train", kind="probe")
    physics = RegisteredBoardPhysics(REGISTRATION, method=method)
    collocation = registered_board_collocation(REGISTRATION, 50000, points=2, time_s=5.0)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    original = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]
    result = train_hf_method_step(model, optimizer, observed, physics, collocation)
    assert result["实际优化步数"] == 1
    assert np.isfinite(result["训练目标"])
    assert physics.interior_pde_calls == (0 if method == "F2" else 1)
    assert any(not torch.equal(previous, current) for previous, current in zip(
        original, (p for p in model.parameters() if p.requires_grad)))


def test_train_hf_step_refuses_full_field_forged_as_visible_probe() -> None:
    from sic_cu.models.task10_plate_deeponet import BoardDeepONet
    from sic_cu.losses.task10_plate_physics import (
        RegisteredBoardPhysics, registered_board_collocation,
    )
    from sic_cu.train.task10_plate_methods import train_hf_method_step

    model = BoardDeepONet(width=8, latent_dim=8, blocks=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    physics = RegisteredBoardPhysics(REGISTRATION, method="F1")
    collocation = registered_board_collocation(REGISTRATION, 50000, points=2, time_s=5.)
    times = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200]
    legal = np.column_stack((np.tile([0., .013, .016], len(times)), np.repeat(times, 3),
                             np.full(42, 50000.), np.tile([1., 0., 0.], len(times))))
    labels = np.full((42, 1), 300.)
    internal = legal.copy()
    internal[:, 0] = .004
    for forged in (PlateTrainingBatch(internal, labels, "train", "probe"),
                   PlateTrainingBatch(np.column_stack((legal[:, (0, 1)],
                                       np.full(42, 42000.), legal[:, 3])),
                                      labels, "train", "probe")):
        with pytest.raises((ValueError, PermissionError), match="探针|训练折|隐藏"):
            train_hf_method_step(model, optimizer, forged, physics, collocation)
    assert optimizer.state == {}


def test_train_lf_step_rejects_dense_hf_full_field_forged_as_coarse_lf() -> None:
    from sic_cu.models.task10_plate_deeponet import BoardDeepONet
    from sic_cu.train.task10_plate_methods import train_lf_method_step

    model = BoardDeepONet(width=8, latent_dim=8, blocks=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    first_center = .012 / 24 / 2
    legal = np.array([[first_center, 0., 50000., 1.],
                      [first_center, 2., 50000., 1.],
                      [first_center, 200., 50000., 1.]])
    labels = np.full((3, 1), 300.)
    kind = "lf_same_physics_coarse"
    forged = (np.column_stack((np.full(3, .004), legal[:, 1:])),
              np.column_stack((legal[:, :2], np.full(3, 42000.), legal[:, 3])),
              np.column_stack((legal[:, :3], np.zeros(3))),
              np.tile(legal[:1], (513, 1)))
    for coordinates in forged:
        bad_labels = np.full((len(coordinates), 1), 300.)
        with pytest.raises((ValueError, PermissionError), match="LF|粗网格|训练折|512"):
            train_lf_method_step(model, optimizer,
                PlateTrainingBatch(coordinates, bad_labels, "train", kind))
    assert optimizer.state == {}
    assert train_lf_method_step(model, optimizer,
        PlateTrainingBatch(legal, labels, "train", kind))["实际优化步数"] == 1


def test_static_training_entry_rejects_raw_hidden_hf_or_old_rz_weight(
    tmp_path: Path,
) -> None:
    from sic_cu.train.task10_plate_methods import audit_training_source_code

    safe = tmp_path / "one_dimensional_only.py"
    safe.write_text("from sic_cu.models.task10_plate_deeponet import BoardDeepONet\n",
                    encoding="utf-8")
    audit_training_source_code(safe)
    forbidden = tmp_path / "bad_method.py"
    forbidden.write_text("import numpy as np\n"
                         "np.load('HF_封存完整场/42000.npz')\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="完整|hidden|HF|非法|原RZ"):
        audit_training_source_code(forbidden)
    forbidden.write_text("from sic_cu.models.multifidelity import AdditiveCorrectionModel\n",
                         encoding="utf-8")
    with pytest.raises(PermissionError, match="旧|RZ|权重|非法"):
        audit_training_source_code(forbidden)
    forbidden.write_text("from sic_cu.physics.heat_equation import axisymmetric_heat_residual\n",
                         encoding="utf-8")
    with pytest.raises(PermissionError, match="旧|RZ|轴|非法"):
        audit_training_source_code(forbidden)


@pytest.fixture
def synthetic_only_registered_observations(tmp_path: Path) -> Path:
    """Unit-only allowed fields; never a real numerical source or a method score."""
    paths = []
    times = np.array([0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200])
    depths = np.array([0.0, .013, .016])
    z_nodes = np.r_[(np.arange(24) + .5) * .012 / 24,
                    .012 + (np.arange(11) + .5) * .0055 / 11]
    materials = np.r_[np.ones(24, dtype=np.int8), np.zeros(11, dtype=np.int8)]
    for role, powers in (("训练", (20000, 35000, 50000, 65000, 80000)),
                         ("验证", (30000, 70000))):
        for power in powers:
            path = tmp_path / f"HF_允许探针/{role}_{power}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            top = 295.15 + power * (.012 / 120 + .0001 + .0055 / 401)
            readings = np.stack([295.15 + (top - 295.15) * (1 - np.exp(-times / 10)),
                                 295.15 + power * (.0175 - .013) / 401 * (1 - np.exp(-times / 10)),
                                 295.15 + power * (.0175 - .016) / 401 * (1 - np.exp(-times / 10))], 1)
            np.savez_compressed(path, time_s=times, depths_from_top_m=depths,
                                flux_w_m2=np.array(power), temperature_k=readings)
            paths.append(path)
    for power in (20000, 35000, 50000, 65000, 80000):
        for role, correction in (("同物理", 0.0), ("接触失配", -1.0)):
            path = tmp_path / f"LF_训练场/{role}_{power}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            field = 295.15 + np.outer(1 - np.exp(-np.arange(101) * 2 / 10),
                                      power * (.0175 - z_nodes) / 401) + correction
            extras = {name: np.zeros(101) for name in
                      ("top_surface_temperature_k", "bottom_outward_flux_w_m2",
                       "storage_rate_per_area_w_m2", "balance_per_area_w_m2",
                       "interface_flux_w_m2", "interface_temperature_jump_k")}
            np.savez_compressed(path, time_s=np.arange(101) * 2.0,
                                node_depth_m=z_nodes, material_id=materials,
                                temperature_k=field, **extras)
            paths.append(path)
    (tmp_path / "探针与源场SHA清单.json").write_text(json.dumps({
        "登记SHA256": sha256_file(REGISTRATION),
        "归档文件_SHA256": {str(p.relative_to(tmp_path)): sha256_file(p) for p in paths},
    }, ensure_ascii=False), encoding="utf-8")
    assert not (tmp_path / "HF_封存完整场").exists()
    return tmp_path


def _unit_method_source(root: Path, method: str, lf: str | None = None):
    from sic_cu.eval.task10_plate_training_source import Task10PlateTrainingSource

    return Task10PlateTrainingSource(REGISTRATION, root, method, lf,
        manifest_sha256=sha256_file(root / "探针与源场SHA清单.json"))


def test_f1_unit_only_cpu_fit_selects_checkpoint_on_both_legal_probe_folds(
    synthetic_only_registered_observations: Path,
) -> None:
    from sic_cu.train.task10_plate_methods import fit_f1_method

    source = _unit_method_source(synthetic_only_registered_observations, "F1")
    result = fit_f1_method(source, seed=0, hf_steps=2, validation_every=1,
                           width=8, latent_dim=8, blocks=0, collocation_points=2,
                           device=torch.device("cpu"))
    assert result.training["实际优化步数"] == 2
    assert result.training["训练折"] == [20000, 35000, 50000, 65000, 80000]
    assert result.training["体内PDE实际调用"] == 2
    assert result.validation["合法验证仅用三探针"] is True
    assert result.validation["验证折"] == [30000, 70000]
    assert result.validation["最佳步数"] in (1, 2)
    assert result.best_state and not result.training.get("内部HF模型误差")
    assert len(result.training["逐步HF真实优化明细"]) == 2
    assert [row["HF优化步"] for row in result.training["逐步HF真实优化明细"]] == [1, 2]
    assert result.training["HF空AdamW起步"] is True


def test_actual_cli_fit_receipt_exposes_group_gate_source_tar_key(
    synthetic_only_registered_observations: Path, tmp_path: Path,
) -> None:
    from sic_cu.train.task10_plate_methods import fit_f1_method

    source_root = synthetic_only_registered_observations
    fit = fit_f1_method(_unit_method_source(source_root, "F1"), seed=0,
                        hf_steps=1, validation_every=1,
                        width=8, latent_dim=8, blocks=0, collocation_points=2,
                        device=torch.device("cpu"))
    script = runpy.run_path(str(PROJECT_ROOT / "scripts/任务10_同板F1_F2_F3重训.py"),
                            run_name="task10_fit_receipt_cpu_unit")
    expected_source_sha = "d" * 64
    output = tmp_path / "仅单元CPU方法receipt_非正式模型"
    output.mkdir()
    script["_save_fit"](output, fit, "F1", 0, source_root,
                         "b" * 64, expected_source_sha, None)
    receipt = json.loads((output / "F1/真实训练日志.json").read_text(encoding="utf-8"))
    assert receipt["方法源码tar_SHA256"] == expected_source_sha
    assert receipt["HF逐步CSV_SHA256"] == sha256_file(
        output / "F1/逐步HF实际优化原始明细.csv")
    for field, filename in (
        ("最佳HF模型状态_SHA256", "最佳HF模型_state_dict.pt"),
        ("最佳HF_AdamW机器状态_SHA256", "最佳HF_AdamW机器状态.pt"),
        ("终态HF_AdamW机器状态_SHA256", "终态HF_AdamW机器状态.pt"),
        ("终态HF模型状态_SHA256", "终态HF模型_state_dict.pt"),
        ("最佳HF四类随机态_SHA256", "最佳HF四类随机态.pt"),
        ("终态HF四类随机态_SHA256", "终态HF四类随机态.pt"),
    ):
        assert receipt[field] == sha256_file(output / "F1" / filename)
    best = torch.load(output / "F1/最佳HF_AdamW机器状态.pt",weights_only=True)
    final = torch.load(output / "F1/终态HF_AdamW机器状态.pt",weights_only=True)
    assert {int(state["step"]) for state in best["state"].values()} == {1}
    assert {int(state["step"]) for state in final["state"].values()} == {1}
    rng = torch.load(output / "F1/终态HF四类随机态.pt",weights_only=True)
    assert {"python_random", "numpy_global", "torch_cpu", "torch_cuda_all"} <= set(rng)
    assert len(rng["numpy_global"]["keys"]) == 624
    assert isinstance(rng["torch_cpu"], torch.Tensor)
    assert rng["torch_cuda_all"] == []  # CPU fixture is not a CUDA result.
    from sic_cu.eval.task10_plate_group_gate import _rng_snapshot
    _rng_snapshot(rng, device="cpu", lf=False)
    assert receipt["设备"] == "cpu"


def test_cpu_pair_remembers_actual_lf_optimizer_ends_and_four_rng_sources(
    synthetic_only_registered_observations: Path, tmp_path: Path,
) -> None:
    from sic_cu.train.task10_plate_methods import fit_same_lf_pair_methods

    root = synthetic_only_registered_observations
    fit2, fit3 = fit_same_lf_pair_methods(
        _unit_method_source(root, "F2", "lf_same_physics_coarse"),
        _unit_method_source(root, "F3", "lf_same_physics_coarse"),
        seed=0, lf_steps=2, hf_steps=2, validation_every=1,
        width=8, latent_dim=8, blocks=0, collocation_points=2,
        lf_batch_size=16, device=torch.device("cpu"))
    for fit in (fit2, fit3):
        assert {int(state["step"]) for state in fit.best_optimizer_state["state"].values()} == {
            fit.validation["最佳步数"]}
        assert {int(state["step"]) for state in fit.final_optimizer_state["state"].values()} == {2}
        assert {int(state["step"]) for state in fit.lf_optimizer_state["state"].values()} == {2}
        assert isinstance(fit.best_rng_state["torch_cpu"], torch.Tensor)
        assert fit.best_rng_state["torch_cuda_all"] == []
        assert fit.lf_rng_state["numpy_sampler_start"] != fit.lf_rng_state["numpy_sampler_final"]
        assert "python_random" in fit.lf_rng_state
        assert "numpy_global" in fit.lf_rng_state
    script = runpy.run_path(str(PROJECT_ROOT / "scripts/任务10_同板F1_F2_F3重训.py"),
                            run_name="task10_pair_receipt_cpu_unit")
    output = tmp_path / "仅CPU同LF一组两方法机读证据_非正式模型"
    output.mkdir()
    for method, fitted in (("F2",fit2),("F3",fit3)):
        script["_save_fit"](output,fitted,method,0,root,"b"*64,"d"*64,
                             "lf_same_physics_coarse")
    logs = [json.loads((output / method / "真实训练日志.json").read_text(encoding="utf-8"))
            for method in ("F2","F3")]
    assert logs[0]["共享LF终态_AdamW机器状态_SHA256"] == logs[1][
        "共享LF终态_AdamW机器状态_SHA256"]
    assert logs[0]["共享LF四类采样随机态_SHA256"] == logs[1][
        "共享LF四类采样随机态_SHA256"]
    assert logs[0]["LF终态可更新参数实测AdamW步数"] == logs[1][
        "LF终态可更新参数实测AdamW步数"] == 2
    from sic_cu.eval.task10_plate_group_gate import _rng_snapshot
    lf_rng = torch.load(output / "F2/共享LF四类采样随机态.pt",
                        map_location="cpu",weights_only=True)
    _rng_snapshot(lf_rng,device="cpu",lf=True)
    assert len(lf_rng["numpy_global"]["keys"]) == 624
    assert lf_rng["torch_cuda_all"] == []


@pytest.mark.parametrize("source_kind", ["lf_same_physics_coarse",
                                         "lf_contact_mismatch_coarse"])
def test_f2_f3_same_lf_cpu_unit_pair_has_true_skip_and_same_budget(
    synthetic_only_registered_observations: Path, source_kind: str,
) -> None:
    from sic_cu.train.task10_plate_methods import fit_same_lf_pair_methods

    root = synthetic_only_registered_observations
    f2_source = _unit_method_source(root, "F2", source_kind)
    f3_source = _unit_method_source(root, "F3", source_kind)
    fit2, fit3 = fit_same_lf_pair_methods(f2_source, f3_source, seed=1,
        lf_steps=2, hf_steps=2, validation_every=1,
        width=8, latent_dim=8, blocks=0, collocation_points=2,
        lf_batch_size=16, device=torch.device("cpu"))
    assert fit2.training["实际优化步数"] == fit3.training["实际优化步数"] == 2
    assert fit2.training["同seed共享LF真实优化步数"] == fit3.training["同seed共享LF真实优化步数"] == 2
    assert fit2.training["体内PDE实际调用"] == 0
    assert fit3.training["体内PDE实际调用"] == 2
    assert fit2.training["LF来源"] == fit3.training["LF来源"] == source_kind
    assert fit2.validation["验证折"] == fit3.validation["验证折"] == [30000, 70000]
    assert fit2.validation["最佳步数"] in (1, 2)
    assert fit3.validation["最佳步数"] in (1, 2)
    assert len(fit2.training["共享LF逐步真实优化明细"]) == 2
    assert fit2.training["共享LF逐步真实优化明细"] == fit3.training["共享LF逐步真实优化明细"]
    assert len(fit2.training["逐步HF真实优化明细"]) == 2
    assert len(fit3.training["逐步HF真实优化明细"]) == 2
    assert (fit2.training["同seed共享LF初始全张量SHA256"] ==
            fit3.training["同seed共享LF初始全张量SHA256"])
    assert (fit2.training["同seed共享HF校正初始全张量SHA256"] ==
            fit3.training["同seed共享HF校正初始全张量SHA256"])


def test_validation_metric_rejects_forged_hidden_or_training_batches() -> None:
    from sic_cu.models.task10_plate_deeponet import BoardDeepONet
    from sic_cu.train.task10_plate_methods import legal_probe_validation_rmse_k

    model = BoardDeepONet(width=8, latent_dim=8, blocks=0)
    times = np.array([0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200])
    standard = np.column_stack((
        np.tile([0.0, .013, .016], len(times)), np.repeat(times, 3),
        np.full(42, 30000.0), np.tile([1.0, 0.0, 0.0], len(times)),
    ))
    other = standard.copy()
    other[:, 2] = 70000
    labels = np.full((42, 1), 295.15)
    legal = (PlateTrainingBatch(standard, labels, "validation", "probe"),
             PlateTrainingBatch(other, labels, "validation", "probe"))
    assert legal_probe_validation_rmse_k(model, legal)["验证折"] == [30000, 70000]
    repeated = standard.copy()
    repeated[:, 1] = 5.0
    with pytest.raises(ValueError, match="时刻|探针|登记"):
        legal_probe_validation_rmse_k(model, (
            PlateTrainingBatch(repeated, labels, "validation", "probe"), legal[1]))
    with pytest.raises(ValueError, match="合法|验证|探针"):
        legal_probe_validation_rmse_k(model, (legal[0],
            PlateTrainingBatch(other, labels, "train", "probe")))
    hidden = other.copy()
    hidden[:, 2] = 42000
    with pytest.raises(ValueError, match="两个|折|隐藏|验证"):
        legal_probe_validation_rmse_k(model, (legal[0],
            PlateTrainingBatch(hidden, labels, "validation", "probe")))


def test_f2_f3_different_lf_sources_can_never_pose_as_single_pde_comparison(
    synthetic_only_registered_observations: Path,
) -> None:
    from sic_cu.train.task10_plate_methods import fit_same_lf_pair_methods

    root = synthetic_only_registered_observations
    f2_source = _unit_method_source(root, "F2", "lf_same_physics_coarse")
    f3_source = _unit_method_source(root, "F3", "lf_contact_mismatch_coarse")
    with pytest.raises(ValueError, match="同一LF|配对|同seed"):
        fit_same_lf_pair_methods(f2_source, f3_source, seed=1,
            lf_steps=1, hf_steps=1, validation_every=1,
            width=8, latent_dim=8, blocks=0, collocation_points=2,
            device=torch.device("cpu"))


def _invoke_registered_method_cli(monkeypatch: pytest.MonkeyPatch, output: Path,
                                  budget: Path, source: Path, source_sha: str) -> None:
    script = PROJECT_ROOT / "scripts/任务10_同板F1_F2_F3重训.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--arm", "F1", "--seed", "0",
                                       "--device", "cpu", "--output", str(output),
                                       "--budget", str(budget),
                                       "--budget-sha256", "0" * 64,
                                       "--source-archive", str(source),
                                       "--source-sha256", source_sha,
                                       "--methods-archive", str(source),
                                       "--methods-sha256", source_sha])
    runpy.run_path(str(script), run_name="__main__")


def test_formal_method_script_exists_separately_from_old_cylinder_trainer() -> None:
    assert (PROJECT_ROOT / "scripts/任务10_同板F1_F2_F3重训.py").is_file()


def test_preregistered_25_model_budget_requires_all_checkpoints_before_hidden_fields() -> None:
    from sic_cu.config import load_yaml

    budget = load_yaml(PROJECT_ROOT / "研究记录/任务10_独立双层场基准/"
                       "正式同板F1_F2_F3重训方法预算前登记_v2.yaml")
    assert budget["共同训练预算"]["正式优化步合计上限"] == 23000
    assert budget["共同训练预算"]["预登记模型身份总数"] == 25
    assert budget["短先导不可修订正式预算"] is True
    assert budget["完整HF后验仅全部25模型冻结后"] is True
    assert budget["单模型SHA锁不足以提前开启完整HF后验"] is True
    assert budget["能源及内部场审核延后至全部25模型及合法验证锁定"] is True


def test_budget_entry_rejects_early_hidden_or_short_pilot_rebudget(
    tmp_path: Path,
) -> None:
    from sic_cu.config import load_yaml

    entry = runpy.run_path(str(PROJECT_ROOT / "scripts/任务10_同板F1_F2_F3重训.py"),
                           run_name="task10_budget_cpu_unit")
    budget = load_yaml(PROJECT_ROOT / "研究记录/任务10_独立双层场基准/"
                       "正式同板F1_F2_F3重训方法预算前登记_v2.yaml")
    old_source = PROJECT_ROOT / budget["已封数值六源tar"]
    sha = budget["已封数值六源tar_SHA256"]
    fake_ledger = tmp_path / "非正式单元测试模拟入口凭据.txt"
    fake_ledger.write_text(sha, encoding="utf-8")
    entry["_validate_budget"].__globals__["LEDGER"] = fake_ledger
    entry["_validate_budget"](budget, old_source, sha, old_source, sha)
    for field, invalid in (("完整HF后验仅全部25模型冻结后", False),
                           ("单模型SHA锁不足以提前开启完整HF后验", False),
                           ("受限源逐热流控制CSV_SHA256", "0" * 64)):
        changed = dict(budget)
        changed[field] = invalid
        with pytest.raises(ValueError, match="预算|来源|原件"):
            entry["_validate_budget"](changed, old_source, sha, old_source, sha)
    shorter = dict(budget)
    shorter["共同训练预算"] = dict(budget["共同训练预算"])
    shorter["共同训练预算"]["正式优化步合计上限"] -= 1
    with pytest.raises(ValueError, match="预算"):
        entry["_validate_budget"](shorter, old_source, sha, old_source, sha)


def test_formal_cli_rejects_project_external_target_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = Path("/tmp/任10正式同板训练不得写项目外_此处不创建")
    source = tmp_path / "单元夹具源码.tar.gz"
    source.write_bytes(b"not a real formal source")
    with pytest.raises(ValueError, match="项目|project"):
        _invoke_registered_method_cli(monkeypatch, outside, tmp_path / "并非真实预算.yaml",
                                      source, sha256_file(source))
    assert not outside.exists()


def test_formal_cli_refuses_existing_user_directory_before_checking_any_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "原有模型目录禁止覆盖"
    existing.mkdir()
    source = tmp_path / "单元夹具源码.tar.gz"
    source.write_bytes(b"not a real formal source")
    with pytest.raises(FileExistsError):
        _invoke_registered_method_cli(monkeypatch, existing, tmp_path / "并非真实预算.yaml",
                                      source, sha256_file(source))


def test_formal_cli_refuses_bad_budget_sha_before_source_or_numerical_temperature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "单元夹具源码.tar.gz"
    source.write_bytes(b"not a real formal source")
    fake_budget = tmp_path / "未事前封存预算.yaml"
    fake_budget.write_text("schema_version: 1\n", encoding="utf-8")
    output = tmp_path / "不可出现的训练源"
    with pytest.raises(ValueError, match="预算|SHA"):
        _invoke_registered_method_cli(monkeypatch, output, fake_budget,
                                      source, sha256_file(source))
    assert not output.exists()
