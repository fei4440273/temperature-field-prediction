"""CPU-only isolated contracts; no real Howard artifacts or temperatures are read."""

from __future__ import annotations

import copy
import ast
import hashlib
import importlib
import importlib.util
import io
import json
import math
import tarfile
from pathlib import Path

import pytest
import torch
import yaml

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.models.task11_howard_composite import Task11HowardComposite
from sic_cu.physics.configuration import BoundaryConditions, CoolingBoundary, LaserBoundary
from sic_cu.physics.materials import Material, MaterialProperty
from sic_cu.train import task11_howard_hf_formal as hf


def _energy():
    try:
        return importlib.import_module("sic_cu.eval.task11_howard_hf_energy")
    except ModuleNotFoundError:
        pytest.fail("Howard HF independent energy component is absent")


def _identity():
    return {"registry": hf.REGISTRY, "registry_sha": "1" * 64,
            "source_tar": hf.SOURCE_TAR, "source_tar_sha": "2" * 64,
            "lf_catalog": hf.LF_CATALOG, "lf_catalog_sha": "3" * 64,
            "hf_data_catalog": hf.HF_DATA_CATALOG, "hf_data_catalog_sha": "4" * 64}


def _view(seed=0):
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(1)
        model = Task11HowardComposite(linear_query_points=hf._fixed_points(),
            nonlinear_query_points=hf._fixed_points(), **hf.HF_BUDGET["model_kwargs"])
    return {"schema_version": 1, "method": hf.METHOD, "seed": seed, "epoch": 10,
            "model_kwargs": model.model_kwargs, "scales": copy.deepcopy(hf.HF_BUDGET["scales"]),
            "model_state": model.state_dict(), "validation_selection_score_c": 1.0,
            "任11HowardHF事前来源": {"seed": seed, "method": hf.METHOD,
                "固定PHQH查询": hf._query_metadata("5" * 64)},
            "HF训练许可": False, **dict.fromkeys(hf.ISOLATION_FLAGS, False)}


def _run(root, seed=0):
    return root / hf.RUN_DIRECTORY / f"正式Howard_HF_seed{seed}"


def _destination(root, state="best", seed=0):
    label = "观测最佳" if state == "best" else "训练末"
    return _run(root, seed) / f"独立原能源_Howard_{label}_20260916T190000+0800"


def _pack(root, *, tar_kind=None):
    module = _energy()
    source_hashes = {}
    for name in module.SOURCE_MEMBERS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("utf-8"))
        source_hashes[name] = sha256_file(path)
    identity = _identity()
    for field in ("registry", "source_tar", "lf_catalog", "hf_data_catalog"):
        path = root / identity[field]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(field.encode())
        identity[field + "_sha"] = sha256_file(path)
    archive = root / module.ENERGY_SOURCE_TAR
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for index, name in enumerate(module.SOURCE_MEMBERS):
            info = tarfile.TarInfo(name)
            data = (root / name).read_bytes()
            info.size = len(data)
            if index == 0 and tar_kind == "link":
                info.type, info.linkname, info.size = tarfile.SYMTYPE, module.SOURCE_MEMBERS[1], 0
            tar.addfile(info, io.BytesIO(data))
        if tar_kind == "duplicate":
            name = module.SOURCE_MEMBERS[0]
            data = (root / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if tar_kind == "extra":
            info = tarfile.TarInfo("extra")
            tar.addfile(info, io.BytesIO(b""))
    contract = module.registration_contract(source_hashes=source_hashes,
        source_tar_sha=sha256_file(archive), hf_source_identity=identity,
        registered_at="2026-09-16T19:00:00+08:00", root_before_sha="6" * 64)
    registry = root / module.ENERGY_REGISTRY
    registry.write_text(yaml.safe_dump(contract, allow_unicode=True), encoding="utf-8")
    args = {"energy_registry": registry, "energy_registry_sha": sha256_file(registry),
            "energy_source_tar": archive, "energy_source_tar_sha": sha256_file(archive),
            **identity, "project_root": root}
    hashes = module.root_gate_hashes(args)
    ledger = root / hf.ROOT_LEDGER
    ledger.write_text(_gate_line(hashes), encoding="utf-8")
    return args, contract


def _gate_line(hashes, number=120):
    module = _energy()
    cell = f"{module.ROOT_TOKEN}; status=active; " + "; ".join(
        f"{key}={hashes[key]}" for key in module.ROOT_IDENTITY_FIELDS)
    return f"| 录-{number:04d} | {cell} | isolated unit fixture |\n"


def test_source_contract_owns_howard_union_and_not_a_mlp_permission():
    module = _energy()
    assert set(module.SOURCE_MEMBERS) == set(hf.SOURCE_MEMBERS) | {
        "src/sic_cu/eval/task11_howard_hf_energy.py",
        "scripts/56_audit_task11_howard_hf_energy.py",
        "tests/test_task11_howard_hf_energy.py"}
    assert module.ROOT_TOKEN == "TASK11_HOWARD_HF_ENERGY_GATE:v1"
    assert len(module.SOURCE_MEMBERS) == len(hf.SOURCE_MEMBERS) + 3


@pytest.mark.parametrize("state", ["best", "final"])
@pytest.mark.parametrize("seed", range(5))
def test_output_uses_actual_howard_run_and_exclusive_state_child(tmp_path, state, seed):
    module = _energy()
    target = _destination(tmp_path, state, seed)
    assert module.task11_howard_energy_output(_run(tmp_path, seed), target, state, seed,
                                             project_root=tmp_path) == target
    assert not target.exists()


@pytest.mark.parametrize("state", ["phase1_best", "phase2_best", "latest", "physical"])
def test_four_non_best_final_states_do_not_receive_energy_permission(tmp_path, state):
    with pytest.raises(ValueError, match="best|final|状态"):
        _energy().task11_howard_energy_output(_run(tmp_path), _destination(tmp_path),
                                             state, 0, project_root=tmp_path)


def test_output_rejects_existing_cross_seed_and_symlink(tmp_path):
    module = _energy()
    target = _destination(tmp_path)
    with pytest.raises(ValueError):
        module.task11_howard_energy_output(_run(tmp_path), target, "best", True, project_root=tmp_path)
    with pytest.raises(ValueError):
        module.task11_howard_energy_output(_run(tmp_path), target, "best", 1, project_root=tmp_path)
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        module.task11_howard_energy_output(_run(tmp_path), target, "best", 0, project_root=tmp_path)
    linked = tmp_path / "link"
    linked.symlink_to(_run(tmp_path), target_is_directory=True)
    with pytest.raises(ValueError):
        module.task11_howard_energy_output(linked, target, "best", 0, project_root=tmp_path)


def test_own_cpu_preflight_checks_six_sources_without_cuda_or_hf_qualification(tmp_path, monkeypatch):
    module = _energy()
    args, _ = _pack(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU preflight probed CUDA"))
    monkeypatch.setattr(hf, "qualify_task11_howard_hf_source", lambda **_: pytest.fail("real LF/PT read"))
    checked = module.preflight_task11_howard_energy(**args)
    assert checked["能源审核许可"] is False
    assert checked["源码普通成员数"] == len(module.SOURCE_MEMBERS)


@pytest.mark.parametrize("tar_kind", ["link", "duplicate", "extra"])
def test_bad_ordinary_source_tar_is_rejected_before_any_real_source_read(tmp_path, tar_kind):
    args, _ = _pack(tmp_path, tar_kind=tar_kind)
    with pytest.raises(ValueError, match="tar|归档|普通|成员"):
        _energy().preflight_task11_howard_energy(**args)


@pytest.mark.parametrize("change", ["source", "registry", "hf_registry", "root"])
def test_every_current_source_and_root_drift_is_rejected(tmp_path, change):
    module = _energy()
    args, _ = _pack(tmp_path)
    checked = module.preflight_task11_howard_energy(**args)
    name = {"source": module.SOURCE_MEMBERS[0], "registry": module.ENERGY_REGISTRY,
            "hf_registry": hf.REGISTRY, "root": hf.ROOT_LEDGER}[change]
    path = tmp_path / name
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="SHA|漂移|ROOT|来源"):
        module.source_unchanged(checked)


@pytest.mark.parametrize("wrapper", ["```\n{}\n```", "~~~~\n{}\n~~~~", "<!--\n{}\n-->",
    "<pre>\n{}\n</pre>", "<script>\n{}\n</script>", "<textarea>\n{}\n</textarea>",
    "<div>\n{}\n</div>\n\n", "    {}", "> {}"])
def test_root_hidden_fenced_commented_or_nonplain_tags_are_not_authority(tmp_path, wrapper):
    module = _energy()
    args, _ = _pack(tmp_path)
    line = _gate_line(module.root_gate_hashes(args)).rstrip()
    path = tmp_path / hf.ROOT_LEDGER
    path.write_text(wrapper.format(line), encoding="utf-8")
    assert not module.root_active(path, module.root_gate_hashes(args))


def test_root_record_collision_and_old_or_wrong_gate_are_rejected(tmp_path):
    module = _energy()
    args, _ = _pack(tmp_path)
    hashes = module.root_gate_hashes(args)
    ledger = tmp_path / hf.ROOT_LEDGER
    for text in (_gate_line(hashes) + "| 录-0120 | unrelated | collision |\n",
                 _gate_line(hashes) * 2, _gate_line(hashes, 107),
                 _gate_line(hashes).replace(module.ROOT_TOKEN, hf.ROOT_TOKEN)):
        ledger.write_text(text, encoding="utf-8")
        assert not module.root_active(ledger, hashes)


@pytest.mark.parametrize("kind", ["bool_counter", "int_float", "fake_permission", "new_order", "extra", "old_source"])
def test_registration_schema_is_type_strict_and_not_old_source_permission(tmp_path, kind):
    module = _energy()
    args, contract = _pack(tmp_path)
    if kind == "bool_counter": contract["独立能源审核"]["功率时刻点数"] = True
    if kind == "int_float": contract["独立能源审核"]["名义吸收归一均值筛查"] = 0
    if kind == "fake_permission": contract["HF训练许可"] = True
    if kind == "new_order": contract["独立能源审核"]["求积阶数"] = [16, 32]
    if kind == "extra": contract["extra"] = 1
    if kind == "old_source": contract["HowardHF四来源"]["registry"] = "old_mlp.yaml"
    registry = args["energy_registry"]
    registry.write_text(yaml.safe_dump(contract, allow_unicode=True), encoding="utf-8")
    args["energy_registry_sha"] = sha256_file(registry)
    (tmp_path / hf.ROOT_LEDGER).write_text(_gate_line(module.root_gate_hashes(args)), encoding="utf-8")
    with pytest.raises(ValueError): module.preflight_task11_howard_energy(**args)


def test_factory_is_real_howard_high_formula_all56_and_preserves_queries64():
    module = _energy()
    view = _view()
    original = {name: tensor.clone() for name, tensor in view["model_state"].items()}
    model = module.task11_howard_energy_model_from_view(view)
    assert type(model) is Task11HowardComposite
    assert len(list(model.parameters())) == 56
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert all(b.dtype == torch.float64 for b in model.buffers())
    coordinates = torch.tensor([[.0125, -.006, 1., 55., 1.]], dtype=torch.float32)
    with torch.no_grad():
        outputs = model.subnet_outputs(coordinates)
        expected = 295.15 + 250.0 * (outputs["linear"] + outputs["nonlinear"])
        assert torch.equal(model(coordinates), expected)
    model.to(torch.device("cpu"))
    assert all(b.dtype == torch.float64 for b in model.buffers())
    audited = module.prepare_task11_howard_energy_copy(model, view)
    assert {p.dtype for p in audited.parameters()} == {torch.float64}
    assert all(b.dtype == torch.float64 for b in audited.buffers())
    assert all(not p.requires_grad for p in audited.parameters())
    assert all(torch.equal(tensor, original[name]) for name, tensor in view["model_state"].items())


@pytest.mark.parametrize("kind", ["mlp", "lf", "method", "seed", "query32", "query_point", "meta", "kwargs", "state"])
def test_factory_rejects_wrong_source_selfmethod_state_or_query64(kind):
    module = _energy()
    view = _view()
    if kind == "mlp": view["method"] = "multifidelity_correction"
    if kind == "lf": view["method"] = hf.LF_METHOD
    if kind == "method": view["任11HowardHF事前来源"]["method"] = "mlp_pinn"
    if kind == "seed": view["任11HowardHF事前来源"]["seed"] = 1
    if kind == "query32": view["model_state"]["linear_query_points"] = hf._fixed_points().float()
    if kind == "query_point": view["model_state"]["nonlinear_query_points"][0, 0] += .001
    if kind == "meta": view["任11HowardHF事前来源"]["固定PHQH查询"]["no_temperature_selection"] = False
    if kind == "kwargs": view["model_kwargs"]["width"] = 64
    if kind == "state": view["model_state"].pop("linear_subnet.branch_layers.0.weight")
    with pytest.raises(ValueError): module.task11_howard_energy_model_from_view(view)


def test_checkpoint_current_sha_drift_is_rejected_before_deserialization(tmp_path, monkeypatch):
    module = _energy()
    run = _run(tmp_path)
    run.mkdir(parents=True)
    original = run / "best.pt"
    original.write_bytes(b"isolated-not-a-checkpoint")
    qualified = {"seed": 0, "状态": module.HF_QUALIFIED_STATUS, "已完成": True,
        "事前来源": {"seed": 0, "method": hf.METHOD}, "HF训练许可": False,
        "最佳HF检查点SHA256": sha256_file(original),
        "真实末HF检查点SHA256": "7" * 64, "真实工件SHA256": {"best.pt": sha256_file(original), "final.pt": "7" * 64}}
    original.write_bytes(b"drift")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("drifted PT was deserialized"))
    with pytest.raises(ValueError, match="SHA|漂移"):
        module.load_task11_howard_energy_model(qualified, run=run, state="best", seed=0, project_root=tmp_path)


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "mps"])
def test_formal_entry_rejects_cpu_fake_before_any_gpu_or_real_pt_read(tmp_path, monkeypatch, device):
    module = _energy()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("fake formal probed GPU"))
    monkeypatch.setattr(hf, "qualify_task11_howard_hf_source", lambda **_: pytest.fail("real PT read"))
    with pytest.raises(ValueError, match="正式|CPU|项目|cuda|CUDA"):
        module.audit_task11_howard_hf_energy(run=_run(tmp_path), output=_destination(tmp_path),
            state="best", seed=0, energy_registry="unused", energy_registry_sha="1" * 64,
            energy_source_tar="unused", energy_source_tar_sha="2" * 64,
            **_identity(), device_name=device, project_root=tmp_path)


class _Analytic(torch.nn.Module):
    def forward(self, coordinates):
        r, z, t, power, material = coordinates.split(1, dim=1)
        return 295.15 + (1.0 + material) * (2.0*r*r + 3.0*z*z + .001*t*t) + .0001*power


def _physics():
    material = Material("synthetic", 2.0, MaterialProperty("constant", 3.0), MaterialProperty("constant", 4.0))
    boundary = BoundaryConditions(295.15, 295.15, LaserBoundary("top_hat", .7, .025, True),
        CoolingBoundary("fixed_temperature", 295.15, None, "outer_radius"),
        2.0, 3.0, None, False, None, None, None)
    geometry = _energy().AxisymmetricGeometry(.05834, .025, -.0175, -.012)
    return {0: material, 1: material}, boundary, geometry


def test_real_full30_double_order_cpu_numerical_fixture_has_all_surfaces_and_not_scientific_pass(tmp_path):
    module = _energy()
    materials, boundaries, geometry = _physics()
    audit = module.audit_task11_howard_energy_fixture(_Analytic(), materials, boundaries, geometry,
                                                     project_root=tmp_path, device_name="cpu")
    verified = module.verify_task11_howard_energy_raw(audit)
    assert len(audit["原始能量"]) == len(audit["原始散度"]) == 60
    assert audit["仅隔离CPU合成测试"] is True
    assert audit["真实科学能源合格"] is False
    assert verified["64阶V-J-D散度积分剩余差最大_瓦"] < 1e-8
    for raw, divergence in zip(audit["原始能量"], audit["原始散度"]):
        assert raw["quadrature_order"] in (16, 64)
        assert math.isclose(raw["absorbed_power_w"], .7*raw["power_w"], abs_tol=1e-10)
        order = raw["quadrature_order"]
        # The original worker reports the unweighted Gauss-node mean, not an area mean.
        mean_z2 = geometry.copper_bottom_z_m**2*(3*order-2)/(4*(2*order-1))
        assert math.isclose(divergence["outer_temperature_mean_c"],
            22.0 + 2.0*geometry.copper_radius_m**2 + 3.0*mean_z2
            + .001*raw["time_s"]**2 + .0001*raw["power_w"], abs_tol=1e-9)
    decision = module.task11_howard_energy_diagnostics(audit)
    assert decision["科学或工程安全资格"] is False
    assert sum(row["功率时刻行数"] for row in decision["三不重叠时间窗"]) == 30
    assert [row["功率时刻行数"] for row in decision["三不重叠时间窗"]] == [12, 12, 6]
    assert len(decision["双阶逐点收敛"]) == 30
    for kind in ("missing_surface", "bool_float", "vjd", "temperature", "summary_count"):
        broken = copy.deepcopy(audit)
        if kind == "missing_surface": broken["原始散度"][0].pop("bottom_model_flux_w")
        if kind == "bool_float": broken["原始能量"][0]["storage_rate_w"] = True
        if kind == "vjd": broken["原始散度"][0]["boundary_flux_residual_w"] += 1.0
        if kind == "temperature": broken["原始散度"][0]["outer_temperature_mean_c"] = math.nan
        if kind == "summary_count": broken["汇总"]["功率时刻审核行数"] = True
        with pytest.raises(ValueError): module.verify_task11_howard_energy_raw(broken)


@pytest.mark.parametrize("device,real_root", [("cuda", False), ("cpu", True)])
def test_public_fixture_never_grants_project_root_or_cuda_formal_permission(tmp_path, device, real_root):
    module = _energy()
    materials, boundaries, geometry = _physics()
    with pytest.raises(ValueError, match="合成|隔离|CPU|正式"):
        module.audit_task11_howard_energy_fixture(_Analytic(), materials, boundaries, geometry,
            project_root=PROJECT_ROOT if real_root else tmp_path, device_name=device)


def test_cli_has_own_gate_arguments_and_no_cpu_or_training_mode(capsys):
    path = PROJECT_ROOT / "scripts/56_audit_task11_howard_hf_energy.py"
    assert path.is_file(), "Howard energy CLI56 is absent"
    spec = importlib.util.spec_from_file_location("howard_energy_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as exit_info: module.main(["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--energy-registry-sha" in help_text and "--energy-source-tar-sha" in help_text
    assert "--state {best,final}" in help_text and "--device {cuda}" in help_text
    assert "--train" not in help_text


def _fixture_audit(root):
    materials, boundaries, geometry = _physics()
    return _energy().audit_task11_howard_energy_fixture(_Analytic(), materials, boundaries, geometry,
                                                       project_root=root, device_name="cpu")


def _payload(root, audit):
    module = _energy()
    return {"运行种子": 0, "审核状态": "best", "运行目录": str(_run(root)),
        "结果不用于训练选模": True, "只读原件审核前后SHA一致": True,
        "真实科学能源合格": False, "审计独立成本": {},
        **audit["汇总"], **module.task11_howard_energy_diagnostics(audit)}


def test_full_surface_air_laser_interface_and_material_storage_are_numerically_traceable(tmp_path):
    module = _energy()
    audit = _fixture_audit(tmp_path)
    assert "完整物理分项" in audit, "full surface law/material storage evidence is absent"
    assert len(audit["完整物理分项"]) == 60
    for raw, divergence, parts in zip(audit["原始能量"], audit["原始散度"], audit["完整物理分项"]):
        assert parts["power_w"] == raw["power_w"] and parts["time_s"] == raw["time_s"]
        assert parts["quadrature_order"] == raw["quadrature_order"]
        assert math.isclose(parts["sic_storage_w"]+parts["copper_ring_storage_w"]+parts["copper_below_storage_w"],
                            raw["storage_rate_w"], abs_tol=1e-10)
        assert math.isclose(parts["sic_top_convection_w"]+parts["copper_shoulder_convection_w"]+parts["copper_bottom_convection_w"],
                            raw["convection_heat_w"], abs_tol=1e-10)
        assert math.isclose(parts["sic_top_radiation_w"]+parts["copper_shoulder_radiation_w"]+parts["copper_bottom_radiation_w"],
                            raw["radiation_heat_w"], abs_tol=1e-10)
        assert parts["sic_top_absorbed_laser_w"] == raw["absorbed_power_w"]
        assert parts["copper_outer_inner_cooling_w"] == raw["cooling_heat_w"]
        assert math.isclose(parts["sic_top_boundary_mismatch_w"]+parts["copper_shoulder_boundary_mismatch_w"]+
            parts["copper_bottom_boundary_mismatch_w"]+parts["copper_outer_boundary_mismatch_w"],
            divergence["boundary_flux_residual_w"], abs_tol=1e-10)
    bad = copy.deepcopy(audit)
    bad["完整物理分项"][0]["sic_top_convection_w"] += 1.0
    with pytest.raises(ValueError, match="分项|逐面|一致"):
        module.verify_task11_howard_energy_raw(bad)


def test_writer_exports_full_originals_and_separates_cost_without_overwrite(tmp_path):
    module = _energy()
    audit = _fixture_audit(tmp_path)
    payload = _payload(tmp_path, audit)
    target = _destination(tmp_path)
    hashes = module.write_task11_howard_energy_evidence(target, audit, payload, project_root=tmp_path)
    assert "完整物理分项.jsonl" in hashes
    assert all(sha256_file(target / name) == digest for name, digest in hashes.items())
    saved = json.loads((target / "汇总指标.json").read_text(encoding="utf-8"))
    assert type(saved["审计独立成本"]["原瓦数及CSV导出墙钟秒"]) is float
    assert saved["审计独立成本"]["原瓦数及CSV导出墙钟秒"] > 0.0
    assert saved["科学或工程安全资格"] is False
    with pytest.raises(FileExistsError):
        module.write_task11_howard_energy_evidence(target, audit, payload, project_root=tmp_path)


def test_writer_rechecks_export_guard_before_sealing_summary(tmp_path):
    module = _energy()
    audit = _fixture_audit(tmp_path)
    target = _destination(tmp_path)
    def guard():
        assert (target / "原始能量.jsonl").is_file()
        raise ValueError("current source/PT SHA drift during export")
    with pytest.raises(ValueError, match="SHA drift"):
        module.write_task11_howard_energy_evidence(target, audit, _payload(tmp_path, audit),
                                                   project_root=tmp_path, source_guard=guard)
    assert not (target / "汇总指标.json").exists()
    assert not (target / "审计工件SHA256.json").exists()


def test_all_local_python_imports_are_in_ordinary_source_union():
    module = _energy()
    todo = ["src/sic_cu/eval/task11_howard_hf_energy.py", "scripts/56_audit_task11_howard_hf_energy.py",
            "tests/test_task11_howard_hf_energy.py"]
    seen = set()
    while todo:
        name = todo.pop()
        if name in seen: continue
        seen.add(name)
        assert name in module.SOURCE_MEMBERS, f"unfrozen local import: {name}"
        path = PROJECT_ROOT / name
        source = ast.parse(path.read_text(encoding="utf-8"))
        package = list(path.relative_to(PROJECT_ROOT / "src").parts[:-1]) if name.startswith("src/") else []
        for node in ast.walk(source):
            if isinstance(node, ast.Import): imports = [(alias.name, []) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    prefix = package[:len(package)-node.level+1]
                    imports = [(".".join(prefix+((node.module or "").split(".") if node.module else [])),
                                [alias.name for alias in node.names])]
                else: imports = [(node.module or "", [alias.name for alias in node.names])]
            else: continue
            for imported, attributes in imports:
                if not imported.startswith("sic_cu"): continue
                candidates = [imported]+[imported+"."+attribute for attribute in attributes]
                for candidate in candidates:
                    stem = PROJECT_ROOT / "src" / Path(*candidate.split("."))
                    options = [stem.with_suffix(".py"), stem / "__init__.py"]
                    for option in options:
                        if option.is_file(): todo.append(option.relative_to(PROJECT_ROOT).as_posix())
                    for parent in stem.parents:
                        if parent == PROJECT_ROOT / "src": break
                        init = parent / "__init__.py"
                        if init.is_file(): todo.append(init.relative_to(PROJECT_ROOT).as_posix())


@pytest.mark.parametrize("poison", ["swap_air", "swap_interface", "gradient_mode", "outer_exact"])
def test_full_parts_individual_identities_cannot_hide_behind_unchanged_sums(tmp_path, poison):
    module = _energy()
    audit = _fixture_audit(tmp_path)
    parts, terms = audit["完整物理分项"][0], audit["原始散度"][0]
    if poison == "swap_air":
        parts["sic_top_convection_w"] += 1.0
        parts["copper_shoulder_convection_w"] -= 1.0
    if poison == "swap_interface":
        parts["sic_bottom_interface_w"] += 1.0
        parts["copper_under_interface_w"] -= 1.0
    if poison == "gradient_mode": terms["boundary_gradient_mode"] = "modified_gradient"
    if poison == "outer_exact": terms["outer_exact_model_flux_w"] += 1.0
    with pytest.raises(ValueError, match="分项|边界|定义|原件|一致"):
        module.verify_task11_howard_energy_raw(audit)


@pytest.mark.parametrize("section,field", [("原始能量", "sign_convention"),
                                         ("原始散度", "conductive_flux_sign_convention")])
@pytest.mark.parametrize("boundary", ["raw", "writer"])
def test_original_sign_conventions_are_frozen_at_raw_and_writer(tmp_path, section, field, boundary):
    module = _energy()
    audit = _fixture_audit(tmp_path)
    payload = _payload(tmp_path, audit)
    audit[section][0][field] = "positive_inward_wrong_original_direction"
    target = _destination(tmp_path)
    with pytest.raises(ValueError, match="符号|约定|边界|定义"):
        if boundary == "raw":
            module.verify_task11_howard_energy_raw(audit)
        else:
            module.write_task11_howard_energy_evidence(target, audit, payload, project_root=tmp_path)
    assert not target.exists()


@pytest.mark.parametrize("poison", ["nominal", "scientific", "window_counter", "nonfinite"])
def test_writer_recomputes_nominal_decision_and_all_typed_diagnostics_before_mkdir(tmp_path, poison):
    module = _energy()
    audit = _fixture_audit(tmp_path)
    payload = _payload(tmp_path, audit)
    if poison == "nominal": payload["名义吸收归一筛查"]["均值通过"] = not payload["名义吸收归一筛查"]["均值通过"]
    if poison == "scientific": payload["科学或工程安全资格"] = True
    if poison == "window_counter": payload["三不重叠时间窗"][0]["功率时刻行数"] = True
    if poison == "nonfinite": payload["bad"] = math.nan
    target = _destination(tmp_path)
    with pytest.raises(ValueError):
        module.write_task11_howard_energy_evidence(target, audit, payload, project_root=tmp_path)
    assert not target.exists()


def _synthetic_qualified(root, view):
    run = _run(root, view["seed"])
    run.mkdir(parents=True)
    for name in ("best.pt", "final.pt"): torch.save(view, run / name)
    hashes = {name: sha256_file(run / name) for name in ("best.pt", "final.pt")}
    return {"seed": view["seed"], "状态": _energy().HF_QUALIFIED_STATUS, "已完成": True,
        "事前来源": copy.deepcopy(view["任11HowardHF事前来源"]), "HF训练许可": False,
        "最佳HF检查点SHA256": hashes["best.pt"], "真实末HF检查点SHA256": hashes["final.pt"],
        "真实工件SHA256": hashes}


@pytest.mark.parametrize("state", ["best", "final"])
def test_isolated_current_pt_loader_recovers_only_own_savedmeta_and_query_buffers(tmp_path, state, monkeypatch):
    module = _energy()
    view = _view()
    qualified = _synthetic_qualified(tmp_path, view)
    mlp = importlib.import_module("sic_cu.eval.task11_mlp_hf_energy")
    monkeypatch.setattr(mlp, "task11_energy_model_from_view", lambda _: pytest.fail("MLP surrogate was used"))
    loaded, model = module.load_task11_howard_energy_model(qualified, run=_run(tmp_path), state=state,
                                                          seed=0, project_root=tmp_path)
    assert hf._same(loaded, view)
    assert hf._same(model.state_dict(), view["model_state"])
    assert all(b.dtype == torch.float64 for b in model.buffers())


def test_pt_drift_during_load_is_caught_before_factory(tmp_path, monkeypatch):
    module = _energy()
    view = _view()
    qualified = _synthetic_qualified(tmp_path, view)
    def load_and_drift(path, **kwargs):
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        path.write_bytes(b"load-period drift")
        return view
    monkeypatch.setattr(torch, "load", load_and_drift)
    monkeypatch.setattr(module, "task11_howard_energy_model_from_view", lambda _: pytest.fail("drift reached factory"))
    with pytest.raises(ValueError, match="SHA漂移"):
        module.load_task11_howard_energy_model(qualified, run=_run(tmp_path), state="best", seed=0, project_root=tmp_path)


@pytest.mark.parametrize("poison", ["status", "zero_completed", "bool_seed", "fake_train", "mlp_method", "sha"])
def test_bad_or_zero_of_ten_qualification_does_not_reach_deserialization(tmp_path, monkeypatch, poison):
    module = _energy()
    qualified = _synthetic_qualified(tmp_path, _view())
    if poison == "status": qualified["状态"] = "exit0"
    if poison == "zero_completed": qualified["已完成"] = False
    if poison == "bool_seed": qualified["seed"] = False
    if poison == "fake_train": qualified["HF训练许可"] = True
    if poison == "mlp_method": qualified["事前来源"]["method"] = "multifidelity_correction"
    if poison == "sha": qualified["真实工件SHA256"]["best.pt"] = "0" * 64
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("bad qualification read PT"))
    with pytest.raises(ValueError):
        module.load_task11_howard_energy_model(qualified, run=_run(tmp_path), state="best", seed=0, project_root=tmp_path)


def test_double_copy_rejects_poison_before_mutation_and_real56_support_second_derivative():
    module = _energy()
    view = _view()
    model = module.task11_howard_energy_model_from_view(view)
    before = copy.deepcopy(model.state_dict())
    broken = copy.deepcopy(view)
    broken["model_state"]["linear_subnet.branch_layers.0.weight"][0, 0] += 1.0
    with pytest.raises(ValueError): module.prepare_task11_howard_energy_copy(model, broken)
    assert hf._same(model.state_dict(), before)
    model = module.prepare_task11_howard_energy_copy(model, view)
    coordinates = torch.tensor([[.0125, -.006, 1., 55., 1.]], dtype=torch.float64, requires_grad=True)
    first = torch.autograd.grad(model(coordinates), coordinates, torch.ones(1, 1, dtype=torch.float64), create_graph=True)[0]
    second = torch.autograd.grad(first[:, :1], coordinates, torch.ones(1, 1, dtype=torch.float64))[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert all(b.dtype == torch.float64 for b in model.buffers())


@pytest.mark.parametrize("poison", ["duplicate_yaml", "not_tar", "source_symlink"])
def test_broken_yaml_stream_tar_and_even_internal_source_symlink_are_rejected(tmp_path, poison):
    module = _energy()
    args, contract = _pack(tmp_path)
    registry = args["energy_registry"]
    if poison == "duplicate_yaml":
        registry.write_text(registry.read_text(encoding="utf-8")+"HF训练许可: false\n", encoding="utf-8")
    if poison == "not_tar":
        archive = args["energy_source_tar"]
        archive.write_bytes(b"not a gzip archive")
        args["energy_source_tar_sha"] = sha256_file(archive)
        contract["源码冻结tarSHA256"] = sha256_file(archive)
        registry.write_text(yaml.safe_dump(contract, allow_unicode=True), encoding="utf-8")
    if poison == "source_symlink":
        source = tmp_path / module.SOURCE_MEMBERS[0]
        relocated = source.with_suffix(".relocated")
        source.rename(relocated)
        source.symlink_to(relocated)
    args["energy_registry_sha"] = sha256_file(registry)
    (tmp_path / hf.ROOT_LEDGER).write_text(_gate_line(module.root_gate_hashes(args)), encoding="utf-8")
    with pytest.raises(ValueError): module.preflight_task11_howard_energy(**args)


@pytest.mark.parametrize("available,count,world", [(False, 1, "1"), (True, 2, "1"), (True, 1, "2")])
def test_cuda_guard_has_no_cpu_fallback_or_world_size_escape(monkeypatch, available, count, world):
    module = _energy()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setenv("WORLD_SIZE", world)
    with pytest.raises(ValueError): module.require_task11_howard_energy_cuda("cuda")
