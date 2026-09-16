"""Howard HF CPU synthetic fixtures only; never formal HF results or real labels."""

from __future__ import annotations

import copy
import ast
import importlib
import io
import json
import random
import tarfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from sic_cu.models.common import ModelScales
from sic_cu.train.task11_howard_lf_formal import (
    LF_MODEL_KWARGS, LF_PARAMETER_COUNT, METHOD, Task11HowardLFTeacher,
)


def _formal():
    name = "sic_cu.train.task11_howard_hf_formal"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name == name:
            pytest.fail("Howard三网HF正式入口尚未实现；CPU RED不是正式HF训练")
        raise


def _coordinates():
    return torch.tensor([[0.010, -0.015, 5.0, 115.2, 0.0],
                         [0.015, -0.004, 100.0, 403.0, 1.0],
                         [0.035, -0.008, 200.0, 115.2, 0.0]], dtype=torch.float32)


@pytest.fixture(scope="module")
def lf_view():
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        teacher = Task11HowardLFTeacher()
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=.001, weight_decay=1e-6)
        ((teacher(_coordinates()) - 295.15) / 250.).square().mean().backward()
        optimizer.step()
    return {"schema_version": 1, "method": METHOD, "seed": 0, "epoch": 1,
            "model_kwargs": dict(LF_MODEL_KWARGS), "scales": asdict(ModelScales()),
            "LF参数量": LF_PARAMETER_COUNT, "validation_rmse_c": 0.8,
            "model_state": copy.deepcopy(teacher.state_dict()),
            "lf_subnet_state": copy.deepcopy(teacher.network.state_dict()),
            "任11Howard事前来源": {"YAML_SHA256": "1" * 64, "TAR_SHA256": "2" * 64,
                                "CATALOG_SHA256": "3" * 64},
            "原文精确三网联合训练复现": False, "旧固定TEST温度读取": False,
            "模拟测试功率温度读取": False, "HF训练许可": False}


def _geometry(root):
    path = root / "configs/geometry.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("coordinate_system: axisymmetric_rz\nunits: m\ncopper:\n  radius_m: 0.05834\n  height_m: 0.0175\nsilicon_carbide:\n  radius_m: 0.025\n  height_m: 0.012\nembedding:\n  sic_top_z_m: 0.0\n  sic_bottom_z_m: -0.012\n  copper_bottom_z_m: -0.0175\n", encoding="utf-8")
    return path


def _initialized(root, lf_view):
    m = _formal()
    if not (root / "configs/geometry.yaml").exists():
        _geometry(root)
    points, metadata = m.build_task11_howard_queries(project_root=root)
    model, optimizer, identity = m.build_task11_howard_hf_initialization(
        lf_view, seed=0, query_points=points, query_metadata=metadata)
    return model, optimizer, identity, metadata


def test_fixed_budget_explains_replay_update_difference():
    m = _formal()
    b = m.HF_BUDGET
    assert b["seeds"] == list(range(5))
    assert b["stage1_epochs"] == 1500 and b["stage2_epochs"] == 500
    assert b["lf_replay_samples_per_power"] == 512
    assert b["lf_replay_points_per_epoch"] == 30720
    assert b["lf_supervision_max_per_seed"] == 61440000
    assert b["adamw_max_steps_per_seed"] == 32000
    assert b["lf_microbatches_max_per_seed"] == 120000
    assert b["lf_mixed_updates_max_per_seed"] == 30000
    assert b["strict_update_conditions_equal"] is False
    assert b["all_three_trainable_from_epoch1"] is True
    assert b["loss_weights"]["sensor_absolute"] == 5.0
    assert b["selection_weights"]["sensor_absolute_rmse"] == .2
    assert b["branch_square_sum_weights"] == {"low_fidelity": 1e-6, "nonlinear": 1e-6}
    assert len(m.source_files()) == len(set(m.source_files()))
    assert {"src/sic_cu/train/task11_howard_hf_initialization.py",
            "tests/test_task11_howard_hf_initialization.py",
            "tests/test_task11_howard_composite.py", "pyproject.toml", "requirements.txt"} <= set(m.source_files())


def test_queries_are_unlabeled_fixed_24_cpu_float64(tmp_path):
    m = _formal()
    _geometry(tmp_path)
    points, metadata = m.build_task11_howard_queries(project_root=tmp_path)
    assert points.shape == (24, 4) and points.dtype == torch.float64
    assert points.device.type == "cpu" and not points.requires_grad
    torch.testing.assert_close(points[:3], torch.tensor([[.25*.05834, -.0175, 5., 0.],
        [.25*.05834, -.0175, 100., 0.], [.25*.05834, -.0175, 200., 0.]], dtype=torch.float64), rtol=0, atol=0)
    assert metadata["PH与QH相同"] is True and metadata["no_temperature_selection"] is True
    assert metadata["geometry_SHA256"] == m.sha256_file(tmp_path / "configs/geometry.yaml")
    assert metadata["顺序"] == "space_outer_time_inner"
    assert torch.equal(points[21:], torch.tensor([[.025, 0., 5., 1.], [.025, 0., 100., 1.], [.025, 0., 200., 1.]], dtype=torch.float64))


@pytest.mark.parametrize("change", ["geometry", "dtype", "seed", "old_method", "query_method"])
def test_initialization_rejects_geometry_seed_dtype_or_foreign_method(tmp_path, lf_view, change):
    m = _formal()
    _geometry(tmp_path)
    points, metadata = m.build_task11_howard_queries(project_root=tmp_path)
    view = copy.deepcopy(lf_view)
    if change == "geometry":
        path = tmp_path / "configs/geometry.yaml"
        path.write_text(path.read_text().replace("0.05834", "0.05835"), encoding="utf-8")
        with pytest.raises(ValueError):
            m.build_task11_howard_queries(project_root=tmp_path)
        return
    if change == "dtype":
        points = points.float()
    elif change == "seed":
        view["seed"] = 1
    elif change == "old_method":
        view["method"] = "mlp_pinn"
    else:
        metadata["方法"] = "temperature_selected_queries"
    with pytest.raises(ValueError):
        m.build_task11_howard_hf_initialization(view, seed=0, query_points=points, query_metadata=metadata)


def test_real_mixed_and_physics_steps_update_all56_without_detaching_queries(tmp_path, lf_view):
    m = _formal()
    model, optimizer, identity, metadata = _initialized(tmp_path, lf_view)
    x = _coordinates()
    y = torch.tensor([[296.], [380.], [340.]])
    sensor = (x, y, y - y[[0, 0, 0]], torch.tensor([0, 0, 0]))
    before = copy.deepcopy(model.state_dict())
    values = m.howard_hf_mixed_step(model, optimizer, (x, y, torch.ones_like(y)), sensor, (x, y))
    assert values["LF回放点"] == 3 and values["正则显式次数"] == 1
    assert len(optimizer.state_dict()["state"]) == 56
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(not torch.equal(before[name], p) for name, p in model.named_parameters())
    assert all(float(s["step"]) == 1 for s in optimizer.state_dict()["state"].values())
    def differentiated_physics(actual, batch):
        q = batch.detach().clone().requires_grad_(True)
        normalized = (actual(q) - 295.15) / 250.
        first = torch.autograd.grad(normalized.sum(), q, create_graph=True)[0]
        second = torch.autograd.grad(first[:, 0].sum(), q, create_graph=True)[0]
        objective = normalized.square().mean() + (first[:, 0] * .05834).square().mean() + (second[:, 0] * .05834**2).square().mean()
        return {"physics_total": objective, "pde": objective, "boundary": objective*0,
                "initial": objective*0, "interface": objective*0}
    m.howard_hf_physics_step(model, optimizer, differentiated_physics, x)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(float(s["step"]) == 2 for s in optimizer.state_dict()["state"].values())
    saved = copy.deepcopy(optimizer.state_dict())
    m.transition_task11_howard_hf_stage2(model, optimizer)
    assert optimizer.param_groups[0]["lr"] == .0001
    assert m._same(saved["state"], optimizer.state_dict()["state"])
    assert all(p.requires_grad is True for p in model.parameters())
    assert model.linear_query_points.dtype == torch.float64
    assert identity["HF训练许可"] is False


@pytest.mark.parametrize("change", ["missing_parameter", "duplicate", "betas", "eps", "amsgrad", "foreach", "weight_decay", "frozen", "state_step", "momentum_dtype"])
def test_adamw_contract_rejects_omission_alias_and_momentum_corruption(tmp_path, lf_view, change):
    m = _formal()
    model, optimizer, _, _ = _initialized(tmp_path, lf_view)
    x = _coordinates(); y = torch.full((3, 1), 300.)
    m.howard_hf_mixed_step(model, optimizer, (x, y, torch.ones_like(y)), (x, y, y*0, torch.tensor([0, 0, 0])), (x, y))
    if change == "missing_parameter": optimizer.param_groups[0]["params"].pop()
    elif change == "duplicate": optimizer.param_groups[0]["params"][-1] = optimizer.param_groups[0]["params"][0]
    elif change == "frozen": next(model.parameters()).requires_grad_(False)
    elif change == "state_step": next(iter(optimizer.state.values()))["step"] = torch.tensor(2.)
    elif change == "momentum_dtype": next(iter(optimizer.state.values()))["exp_avg"] = next(iter(optimizer.state.values()))["exp_avg"].double()
    else: optimizer.param_groups[0][change] = {"betas": (False, .999), "eps": True, "amsgrad": 0, "foreach": False, "weight_decay": 0.}[change]
    with pytest.raises(ValueError): m.audit_task11_howard_hf_adamw(model, optimizer, stage=m.STAGE1, expected_steps=1)


def test_selection_excludes_epoch0_and_resets_stage_patience():
    m = _formal()
    rows = []
    for epoch in range(1, 211):
        rows.append(m.make_task11_howard_hf_log_row(seed=0, global_epoch=epoch, stage=m.STAGE1,
            local_epoch=epoch, actual=m.EPOCH_CONSUMPTION, score=8. if epoch % 10 == 0 else None,
            validation={"顶部": 8., "absolute_rmse_c": 8., "delta_rmse_c": 8.} if epoch % 10 == 0 else None))
    selected = m.replay_task11_howard_hf_selection(rows)
    assert selected["全局最佳轮次"] == 10 and selected["阶段1截止轮次"] == 210
    assert selected["全局最佳分数"] == 8. and selected["已完成"] is False
    for local in range(1, 211):
        rows.append(m.make_task11_howard_hf_log_row(seed=0, global_epoch=210+local, stage=m.STAGE2,
            local_epoch=local, actual=m.EPOCH_CONSUMPTION, score=9. if local % 10 == 0 else None,
            validation={"顶部": 9., "absolute_rmse_c": 9., "delta_rmse_c": 9.} if local % 10 == 0 else None))
    selected = m.replay_task11_howard_hf_selection(rows)
    assert selected["全局最佳轮次"] == 10 and selected["阶段最佳轮次"] == 10
    assert selected["阶段2实际轮次"] == 210 and selected["已完成"] is True
    totals = m.audit_task11_howard_hf_log(rows, seed=0)
    assert totals["AdamW优化步"] == 420*16 and totals["LF回放点"] == 420*30720
    rows[-1]["LF回放点"] += 1
    with pytest.raises(ValueError): m.audit_task11_howard_hf_log(rows, seed=0)


@pytest.fixture
def fake_project(tmp_path, monkeypatch, lf_view):
    m = _formal()
    _geometry(tmp_path)
    for relative in m.SOURCE_MEMBERS:
        path = tmp_path / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic CPU ordinary source fixture " + relative, encoding="utf-8")
    source_hashes = {name: m.sha256_file(tmp_path / name) for name in m.SOURCE_MEMBERS}
    old_identity = copy.deepcopy(m.LF_SOURCE_IDENTITY)
    for name in ("registry", "source_tar", "catalog"):
        path = tmp_path / old_identity[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic CPU LF metadata " + name, encoding="utf-8")
        old_identity[name + "_sha"] = m.sha256_file(path)
    monkeypatch.setattr(m, "LF_SOURCE_IDENTITY", old_identity)
    qualifications, views = [], []
    for seed in range(5):
        view = copy.deepcopy(lf_view)
        view["seed"] = seed
        view["任11Howard事前来源"] = dict(zip(("YAML_SHA256", "TAR_SHA256", "CATALOG_SHA256"),
            (old_identity["registry_sha"], old_identity["source_tar_sha"], old_identity["catalog_sha"])))
        for name, value in view["lf_subnet_state"].items():
            value.add_(seed*.01)
            view["model_state"]["network."+name] = value.clone()
        directory = tmp_path / m.LF_RUN_DIRECTORY / f"正式Howard_LF_seed{seed}"
        directory.mkdir(parents=True)
        torch.save(view, directory / "best.pt")
        views.append(view)
        qualifications.append({"seed": seed, "累计轮次": 1, "最佳轮次": 1,
            "最佳LF验证RMSE_摄氏度": .8, "真实累计成本秒": 1.,
            "最佳LF检查点SHA256": m.sha256_file(directory / "best.pt"),
            "真实工件SHA256": {"best.pt": m.sha256_file(directory / "best.pt")},
            "事前来源": copy.deepcopy(view["任11Howard事前来源"]),
            "状态": "synthetic CPU metadata qualification; not formal LF"})
    calls = []
    def qualify(**kwargs):
        assert Path(kwargs["project_root"]) == tmp_path
        assert {key: kwargs[key] for key in old_identity} == old_identity
        calls.append(kwargs["seed"])
        return copy.deepcopy(qualifications[kwargs["seed"]])
    monkeypatch.setattr(m, "qualify_task11_howard_lf_source", qualify)
    catalog = {"schema_version": 1, "用途": "CPU synthetic fixture not a formal catalog",
        "HowardLF三身份": old_identity,
        "HowardLF固定源码35SHA256": {name: source_hashes[name] for name in m.HOWARD_LF_SOURCE_MEMBERS},
        "五LF真实完整资格": qualifications, "审核命令": "CPU fixture", "审核退出码": 0,
        "ROOT审核时SHA256": "a"*64, "限制": {"HF训练许可": False}, "五seed描述统计": {}}
    cat = tmp_path / m.LF_CATALOG
    cat.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    hf = {"schema_version": 1, "阶段": "task11_hf_development_12_3", "旧固定TEST温度读取": False,
          "模拟测试功率温度读取": False, "HF计数": {"Top训练": 29593, "Top验证": 7272,
          "环温训练": 2985, "环温验证": 752}, "协议指纹": {"synthetic": "b"*64}}
    hf_path = tmp_path / m.HF_DATA_CATALOG
    hf_path.write_text(json.dumps(hf), encoding="utf-8")
    monkeypatch.setattr(m, "collect_task11_mlp_hf_data_catalog", lambda: copy.deepcopy(hf))
    monkeypatch.setattr(m, "current_protocol_fingerprints", lambda: copy.deepcopy(hf["协议指纹"]))
    monkeypatch.setattr(m, "LF_CATALOG_SHA", m.sha256_file(cat))
    monkeypatch.setattr(m, "HF_DATA_CATALOG_SHA", m.sha256_file(hf_path))
    tar_path = tmp_path / m.SOURCE_TAR
    with tarfile.open(tar_path, "w:gz") as archive:
        for relative in m.SOURCE_MEMBERS: archive.add(tmp_path / relative, arcname=relative, recursive=False)
    _, query = m.build_task11_howard_queries(project_root=tmp_path)
    registration = m.registration_contract(source_hashes=source_hashes,
        source_tar_sha=m.sha256_file(tar_path), lf_catalog_sha=m.sha256_file(cat),
        hf_data_catalog_sha=m.sha256_file(hf_path), query_metadata=query, registered_at="synthetic CPU fixture")
    registry = tmp_path / m.REGISTRY
    registry.write_text(yaml.safe_dump(registration, allow_unicode=True, sort_keys=False), encoding="utf-8")
    args = {"registry": m.REGISTRY, "registry_sha": m.sha256_file(registry),
        "source_tar": m.SOURCE_TAR, "source_tar_sha": m.sha256_file(tar_path), "lf_catalog": m.LF_CATALOG,
        "lf_catalog_sha": m.sha256_file(cat), "hf_data_catalog": m.HF_DATA_CATALOG,
        "hf_data_catalog_sha": m.sha256_file(hf_path), "seed": 0,
        "output": str(tmp_path / m.RUN_DIRECTORY / "正式Howard_HF_seed0"), "project_root": tmp_path}
    def gate(number=108, status="active", duplicate=False):
        cell = f"{m.ROOT_TOKEN}; status={status}; YAML_SHA256={args['registry_sha']}; TAR_SHA256={args['source_tar_sha']}; LF_CATALOG_SHA256={args['lf_catalog_sha']}; HF_DATA_CATALOG_SHA256={args['hf_data_catalog_sha']}"
        line = f"| 录-{number:04d} | {cell} | synthetic CPU fixture |\n"
        (tmp_path / m.ROOT_LEDGER).write_text(line* (2 if duplicate else 1), encoding="utf-8")
    gate()
    return m, args, registration, qualifications, calls, gate, views


def test_formal_gate_registration_api_exists_before_source_or_label_reads():
    m = _formal()
    assert hasattr(m, "registration_contract"), "缺少冻结四SHA/几何/预算登记合同构造API"
    assert hasattr(m, "preflight_task11_howard_hf"), "缺少Howard自身HF事前门禁"


def test_preflight_recomputes_own_five_lf_and_initializes_cpu_before_permission(fake_project, monkeypatch):
    m, args, registration, qualifications, calls, gate, views = fake_project
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU preflight不能探测CUDA"))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: pytest.fail("CPU preflight不能探测CUDA"))
    result = m.preflight_task11_howard_hf(**args)
    assert calls == list(range(5))
    assert result["LF来源资格"] == qualifications[0]
    assert result["HF训练许可"] is False
    assert not Path(args["output"]).exists()
    assert result["查询元数据"] == registration["固定PHQH查询"]


@pytest.mark.parametrize("change", ["inactive", "duplicate_gate", "old_gate_number", "codefence_gate", "sha", "source_drift", "crossseed_output", "budget_bool", "loss_weight", "dtype", "query_method", "protocol_source", "duplicate_yaml", "lf_seed", "lf_catalog_qualification", "old_mlp_view", "tar_duplicate", "tar_link"])
def test_preflight_rejects_identity_and_contract_corruption_before_cuda(fake_project, monkeypatch, change):
    m, args, registration, qualifications, calls, gate, views = fake_project
    root = Path(args["project_root"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("拒绝不能探测CUDA"))
    if change == "inactive": gate(status="inactive")
    elif change == "duplicate_gate": gate(duplicate=True)
    elif change == "old_gate_number": gate(number=107)
    elif change == "codefence_gate":
        ledger = root / m.ROOT_LEDGER
        ledger.write_text("```\n"+ledger.read_text()+"```\n", encoding="utf-8")
    elif change == "sha": args["registry_sha"] = "0"*64
    elif change == "source_drift": (root / m.SOURCE_MEMBERS[0]).write_text("drift", encoding="utf-8")
    elif change == "crossseed_output": args["output"] = args["output"].replace("seed0", "seed1")
    elif change in ("budget_bool", "loss_weight", "dtype", "query_method", "protocol_source"):
        if change == "budget_bool": registration["正式预算"]["stage1_epochs"] = True
        elif change == "loss_weight": registration["正式预算"]["loss_weights"]["sensor_absolute"] = .2
        elif change == "dtype": registration["正式预算"]["query_buffer_dtype"] = "torch.float32"
        elif change == "query_method": registration["固定PHQH查询"]["方法"] = "label_selected"
        else: registration["固定协议指纹"] = {"other": "0"*64}
        path = root / m.REGISTRY
        path.write_text(yaml.safe_dump(registration, allow_unicode=True), encoding="utf-8")
        args["registry_sha"] = m.sha256_file(path); gate()
    elif change == "duplicate_yaml":
        path = root / m.REGISTRY
        path.write_text("schema_version: 0\n"+path.read_text(encoding="utf-8"), encoding="utf-8")
        args["registry_sha"] = m.sha256_file(path); gate()
    elif change == "lf_seed": qualifications[0]["seed"] = 1
    elif change == "lf_catalog_qualification": qualifications[0]["累计轮次"] = 2
    elif change == "old_mlp_view":
        path = root / m.LF_RUN_DIRECTORY / "正式Howard_LF_seed0/best.pt"
        views[0]["method"] = "mlp_pinn"; torch.save(views[0], path)
        qualifications[0]["最佳LF检查点SHA256"] = m.sha256_file(path)
    else:
        path = root / m.SOURCE_TAR
        with tarfile.open(path, "w:gz") as archive:
            for relative in m.SOURCE_MEMBERS: archive.add(root / relative, arcname=relative, recursive=False)
            if change == "tar_duplicate": archive.add(root / m.SOURCE_MEMBERS[0], arcname=m.SOURCE_MEMBERS[0], recursive=False)
            else:
                info = tarfile.TarInfo("symlink"); info.type = tarfile.SYMTYPE; info.linkname = m.SOURCE_MEMBERS[0]; archive.addfile(info)
        args["source_tar_sha"] = m.sha256_file(path)
        registration["源码冻结tarSHA256"] = args["source_tar_sha"]
        path = root / m.REGISTRY
        path.write_text(yaml.safe_dump(registration, allow_unicode=True), encoding="utf-8")
        args["registry_sha"] = m.sha256_file(path); gate()
    with pytest.raises(ValueError): m.preflight_task11_howard_hf(**args)
    assert not Path(args["output"]).exists()


def test_run_rejects_cpu_or_fake_cuda_before_output(fake_project, monkeypatch):
    m, args, *_ = fake_project
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError): m.run_task11_howard_hf_formal(**args, device_name="cpu")
    with pytest.raises(ValueError): m.run_task11_howard_hf_formal(**args, device_name="cuda")
    assert not Path(args["output"]).exists()


def test_complete_state_and_receipt_audit_api_exist():
    m = _formal()
    assert hasattr(m, "audit_task11_howard_hf_state"), "缺少真实全56 AdamW/RNG/query/预算完整状态审计"
    assert hasattr(m, "audit_task11_howard_hf_artifacts"), "缺少提交历史、best/final、receipt/log成本审计"
    assert hasattr(m, "verify_task11_howard_hf_resume"), "缺少仅本人最近完整提交恢复审核"
    assert hasattr(m, "qualify_task11_howard_hf_source"), "缺少只读终态资格入口"


def test_cpu_real_split_full_preserves_optimizer_rng_queries_and_stage_lr(tmp_path, lf_view):
    m = _formal()
    from sic_cu.train.common import save_training_state, load_training_state
    first, optimizer, identity, metadata = _initialized(tmp_path, lf_view)
    second, resumed_optimizer, _, _ = _initialized(tmp_path, lf_view)
    x = _coordinates(); y = torch.full((3, 1), 300.)
    sensor = (x, y, y*0, torch.tensor([0, 0, 0]))
    def update(model, opt):
        random.random(); np.random.random(); torch.rand(1)
        m.howard_hf_mixed_step(model, opt, (x, y, torch.ones_like(y)), sensor, (x, y))
    update(first, optimizer)
    path = tmp_path / "synthetic_cpu_split.pt"
    save_training_state(path, first, optimizer, stage=m.STAGE1, epoch=0,
        budget=m.STAGE_BUDGET, metadata={"CPU合成组件": True, "非正式HF消费日志": True})
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["random_state"]["torch_cuda"] is None
    update(first, optimizer)
    m.transition_task11_howard_hf_stage2(first, optimizer)
    update(first, optimizer)
    full_random = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    load_training_state(path, second, resumed_optimizer)
    update(second, resumed_optimizer)
    m.transition_task11_howard_hf_stage2(second, resumed_optimizer)
    update(second, resumed_optimizer)
    assert m._same(first.state_dict(), second.state_dict())
    assert m._same(optimizer.state_dict(), resumed_optimizer.state_dict())
    assert m._same(full_random, (random.getstate(), np.random.get_state(), torch.get_rng_state()))
    assert all(float(value["step"]) == 3 for value in resumed_optimizer.state_dict()["state"].values())
    assert second.linear_query_points.dtype == torch.float64 and resumed_optimizer.param_groups[0]["lr"] == .0001
    with pytest.raises(ValueError):
        m.audit_task11_howard_hf_state(saved, [], seed=0, identity={"seed": 0}, formal=True)


@pytest.mark.parametrize("bad", ["paused_metrics", "duplicate_receipts", "cost", "resume_sha", "log_arithmetic", "bestfinal"])
def test_artifact_audit_rejects_incomplete_or_counterfeit_submissions(bad):
    m = _formal()
    objects = {"training.jsonl": [], "HF会话收据_0001.json": {"状态": "已暂停且完整Howard HF阶段提交",
        "身份": {"seed": 0}, "起始已提交轮次": 0, "本会话实际轮次": 1, "累计实际轮次": 1,
        "训练墙钟秒": 1., "加载构建恢复墙钟秒": 1., "导出核源墙钟秒": 1., "本会话总墙钟秒": 3.,
        "前驱收据SHA256": None, "原件SHA256": {}}}
    if bad == "paused_metrics": objects["metrics.json"] = {"status": "completed"}
    elif bad == "duplicate_receipts": objects["HF会话收据_0001x.json"] = objects["HF会话收据_0001.json"]
    elif bad == "cost": objects["HF会话收据_0001.json"]["本会话总墙钟秒"] = -1.
    elif bad == "resume_sha": objects["HF会话收据_0001.json"]["恢复源before_SHA256"] = "bad"
    elif bad == "log_arithmetic": objects["training.jsonl"] = [{"LF回放点": 1}]
    else: objects["final.pt"] = {"method": "mlp_pinn"}
    with pytest.raises(ValueError):
        m.audit_task11_howard_hf_artifacts(objects, {}, seed=0, source_identity={"seed": 0}, finished=False)


def test_cli_three_modes_do_not_inherit_mlp_or_old_lf_permission():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/55_run_task11_howard_hf_formal.py"
    assert path.is_file(), "缺少Howard HF独立三模式正式CLI"
    source = path.read_text(encoding="utf-8")
    assert "--preflight-only" in source and "--audit-finished" in source
    assert "preflight_task11_howard_hf" in source and "qualify_task11_howard_hf_source" in source


@pytest.fixture(scope="module")
def cpu_epoch_artifacts(tmp_path_factory, lf_view):
    m = _formal()
    from sic_cu.losses import PhysicsLossComputer, PhysicsLossWeights
    from sic_cu.physics.materials import load_materials
    from sic_cu.physics.resolution import load_resolved_boundary_conditions
    from torch.utils.data import TensorDataset
    from sic_cu.train.common import save_training_state
    tmp_path = tmp_path_factory.mktemp("real_cpu_epoch_only")
    model, optimizer, initialization, query = _initialized(tmp_path, lf_view)
    initial_sha = m._state_content_sha(model.state_dict())
    identity = {"seed": 0, "method": m.METHOD,
        "LF初始子网内容SHA256": m._state_content_sha(model.low_fidelity_subnet.state_dict()),
        "CPU初态完整内容SHA256": initial_sha, "LF事前三SHA": copy.deepcopy(lf_view["任11Howard事前来源"]),
        "LF最佳轮次": 1, "固定PHQH查询": query, "固定协议指纹": {"CPU合成": "0"*64}}
    qualification = {"seed": 0, "最佳轮次": 1, "最佳LF检查点SHA256": "1"*64, "真实累计成本秒": 1.}
    identity["LF完整资格内容SHA256"] = m.canonical_json_sha256(qualification)
    identity["LF最佳检查点SHA256"] = qualification["最佳LF检查点SHA256"]
    directory = tmp_path / "synthetic_artifact_directory"
    directory.mkdir()
    log = directory / "training.jsonl"
    log.write_text("", encoding="utf-8")
    snapshot = directory / "事前来源快照"
    snapshot.mkdir()
    for key, name in zip(m.IDENTITY_FIELDS, ("登记.yaml", "源码.tar.gz", "LF目录.json", "HF开发目录.json")):
        path = snapshot / name
        raw = json.dumps({"CPU合成元数据": name}) if name.endswith(".json") else "CPU synthetic metadata " + name
        path.write_text(raw, encoding="utf-8")
        identity[key] = m.sha256_file(path)
    meta = m._state_metadata(identity, model, [], m.STAGE1, log_sha=m.sha256_file(log), initialization=initialization)
    save_training_state(directory / "阶段_初始.pt", model, optimizer, stage=m.STAGE1, epoch=0,
                        budget=m.STAGE_BUDGET, metadata=meta)
    def rows(count):
        x = _coordinates().repeat((count+2)//3, 1)[:count].clone()
        return x, torch.full((count, 1), 300.)
    tx, ty = rows(29593); sx, sy = rows(2985); lx, ly = rows(30720)
    train = TensorDataset(tx, ty, torch.ones_like(ty))
    replay = TensorDataset(lx, ly)
    physics = PhysicsLossComputer(load_materials(), load_resolved_boundary_conditions(), PhysicsLossWeights())
    actual = m.howard_hf_epoch(model, optimizer, train, (sx, sy, sy*0, torch.zeros(2985, dtype=torch.long)), physics,
                               replay, seed=0, global_epoch=1)
    row = m.make_task11_howard_hf_log_row(seed=0, global_epoch=1, stage=m.STAGE1, local_epoch=1,
        actual=actual, score=None, validation=None)
    log.write_text(json.dumps(row, ensure_ascii=False)+"\n", encoding="utf-8")
    meta = m._state_metadata(identity, model, [row], m.STAGE1, log_sha=m.sha256_file(log), initialization=initialization)
    save_training_state(directory / "阶段_最近.pt", model, optimizer, stage=m.STAGE1, epoch=1,
                        budget=m.STAGE_BUDGET, metadata=meta)
    source_record = {"身份": identity, "LF来源完整资格": qualification,
        "LF本人best严格迁移": True, "历史HF权重/AdamW/RNG读取": False,
        "限制": m.LIMITATIONS, **dict.fromkeys(m.ISOLATION_FLAGS, False)}
    m._write_new_json(directory / "事前真实来源登记.json", source_record, tmp_path)
    history = {}
    for original, target in (("阶段_最近.pt", "最近提交历史_0001.pt"), ("training.jsonl", "已提交日志_0001.jsonl")):
        m._exclusive_copy(directory / original, directory / target, tmp_path)
        history[target] = m.sha256_file(directory / target)
    receipt = {"身份": identity, "起始已提交轮次": 0, "本会话实际轮次": 1, "累计实际轮次": 1,
        "选模重推": m.replay_task11_howard_hf_selection([row]), "消费累计": m.audit_task11_howard_hf_log([row], seed=0),
        "训练墙钟秒": 1., "加载构建恢复墙钟秒": 1., "导出核源墙钟秒": 1., "本会话总墙钟秒": 3.,
        "峰值真实CUDA显存字节": 0, "前驱收据SHA256": None, "恢复原件": None,
        "恢复源before_SHA256": None, "恢复源after_SHA256": None,
        "提交历史SHA256": history, "原件SHA256": m._artifact_hashes(directory, tmp_path),
        "状态": "已暂停且完整Howard HF阶段提交", **dict.fromkeys(m.ISOLATION_FLAGS, False),
        "CPU合成收据结构": "成本为格式夹具值，不宣称HF真CUDA成本"}
    m._write_new_json(directory / "HF会话收据_0001.json", receipt, tmp_path)
    from sic_cu.train.common import load_training_state
    split_model, split_optimizer, _, _ = _initialized(tmp_path, lf_view)
    resume_source = directory / "最近提交历史_0001.pt"
    resume_sha = m.sha256_file(resume_source)
    load_training_state(resume_source, split_model, split_optimizer)
    rows = [row]
    full_random = None
    for epoch in range(2, 11):
        actual = m.howard_hf_epoch(model, optimizer, train, (sx, sy, sy*0, torch.zeros(2985, dtype=torch.long)), physics,
            replay, seed=0, global_epoch=epoch)
        score = modalities = None
        if epoch == 10:
            score, modalities = m._validation_selection(model, DataLoader(train, batch_size=2048),
                (sx, sy, sy*0, torch.zeros(2985, dtype=torch.long)), torch.device("cpu"), m.HF_BUDGET["selection_weights"])
        rows.append(m.make_task11_howard_hf_log_row(seed=0, global_epoch=epoch, stage=m.STAGE1,
            local_epoch=epoch, actual=actual, score=score, validation=modalities))
    log.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    meta = m._state_metadata(identity, model, rows, m.STAGE1, log_sha=m.sha256_file(log), initialization=initialization)
    m._save_stage(directory / "阶段_最近.pt", model, optimizer, stage=m.STAGE1, metadata=meta, root=tmp_path)
    full_random = copy.deepcopy(torch.load(directory / "阶段_最近.pt", map_location="cpu", weights_only=False)["random_state"])
    for canonical, target in (("阶段1_最佳.pt", "阶段1_最佳历史_0010.pt"),
                               ("阶段_观测最佳.pt", "全局最佳历史_0010.pt")):
        m._save_stage(directory / canonical, model, optimizer, stage=m.STAGE1, metadata=meta, root=tmp_path, new=True)
        m._exclusive_copy(directory / canonical, directory / target, tmp_path)
    architecture = m._architecture(0, model)
    m._save_hf_view(directory / "best.pt", architecture, model, meta, best=True, root=tmp_path)
    m._exclusive_copy(directory / "best.pt", directory / "全局最佳模型历史_0010.pt", tmp_path)
    history = {}
    for original, target in (("阶段_最近.pt", "最近提交历史_0002.pt"), ("training.jsonl", "已提交日志_0002.jsonl"),
                             ("阶段_观测最佳.pt", "观测最佳提交历史_0002.pt"), ("best.pt", "最佳视图提交历史_0002.pt")):
        m._exclusive_copy(directory / original, directory / target, tmp_path)
        history[target] = m.sha256_file(directory / target)
    second_receipt = copy.deepcopy(receipt)
    second_receipt.update({"起始已提交轮次": 1, "本会话实际轮次": 9, "累计实际轮次": 10,
        "选模重推": meta["选模重推"], "消费累计": meta["消费累计"], "提交历史SHA256": history,
        "前驱收据SHA256": m.sha256_file(directory / "HF会话收据_0001.json"),
        "恢复原件": "阶段_最近.pt", "恢复源before_SHA256": resume_sha, "恢复源after_SHA256": resume_sha,
        "原件SHA256": m._artifact_hashes(directory, tmp_path)})
    m._write_new_json(directory / "HF会话收据_0002.json", second_receipt, tmp_path)
    # Rerun the same actual array/physics updates from the saved CPU state, not
    # fabricated counters or a mocked forward/optimizer/RNG continuation.
    load_training_state(resume_source, split_model, split_optimizer)
    for epoch in range(2, 11):
        m.howard_hf_epoch(split_model, split_optimizer, train, (sx, sy, sy*0, torch.zeros(2985, dtype=torch.long)), physics,
            replay, seed=0, global_epoch=epoch)
        if epoch == 10:
            m._validation_selection(split_model, DataLoader(train, batch_size=2048),
                (sx, sy, sy*0, torch.zeros(2985, dtype=torch.long)), torch.device("cpu"), m.HF_BUDGET["selection_weights"])
    split_random = {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(), "torch_cuda": None}
    assert m._same(model.state_dict(), split_model.state_dict())
    assert m._same(optimizer.state_dict(), split_optimizer.state_dict())
    assert m._same(full_random, split_random)
    assert m.sha256_file(resume_source) == resume_sha
    objects, hashes = m._load_artifacts(directory, tmp_path)
    return m, model, optimizer, actual, objects, hashes, identity


def test_real_cpu_full_epoch_consumes_fixed_arrays_and16_real_adamw_steps(cpu_epoch_artifacts):
    m, model, optimizer, actual, *_ = cpu_epoch_artifacts
    assert {key: actual[key] for key in m.EPOCH_CONSUMPTION} == m.EPOCH_CONSUMPTION
    assert all(float(state["step"]) == 160 for state in optimizer.state_dict()["state"].values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_real_cpu_artifacts_audit_pass_is_explicitly_nonformal(cpu_epoch_artifacts):
    m, _, _, _, objects, hashes, identity = cpu_epoch_artifacts
    result = m.audit_task11_howard_hf_artifacts(objects, hashes, seed=0, source_identity=identity, finished=False, formal=False)
    assert result["全局实际轮次"] == 10 and result["消费累计"]["AdamW优化步"] == 160
    assert result["全局最佳轮次"] == 10 and result["HF训练许可"] is False
    with pytest.raises(ValueError):
        m.audit_task11_howard_hf_artifacts(objects, hashes, seed=0, source_identity=identity, finished=False, formal=True)


@pytest.mark.parametrize("change", ["missing56", "momentum_step", "id_bool_alias", "id_float_alias", "momentum_storage_alias", "query_dtype", "rng", "best0", "paused_metrics", "duplicate_receipt", "cost", "consumption", "resume_before", "initial_nl", "source_snapshot", "source_qualification"])
def test_real_cpu_artifact_chain_rejects_precise_tamper(cpu_epoch_artifacts, change):
    m, _, _, _, original, original_hashes, identity = cpu_epoch_artifacts
    objects, hashes = copy.deepcopy(original), copy.deepcopy(original_hashes)
    recent = objects["阶段_最近.pt"]
    if change == "missing56": recent["optimizer_state"]["param_groups"][0]["params"].pop()
    elif change == "momentum_step": next(iter(recent["optimizer_state"]["state"].values()))["step"].add_(1)
    elif change in ("id_bool_alias", "id_float_alias"):
        value = False if change == "id_bool_alias" else 0.
        recent["optimizer_state"]["param_groups"][0]["params"][0] = value
        record = recent["optimizer_state"]["state"].pop(0)
        recent["optimizer_state"]["state"][value] = record
    elif change == "momentum_storage_alias":
        records = recent["optimizer_state"]["state"]
        # Same-size encoder biases are independent actual momentum slots.
        records[3]["exp_avg"] = records[1]["exp_avg"]
    elif change == "query_dtype": recent["model_state"]["linear_query_points"] = recent["model_state"]["linear_query_points"].float()
    elif change == "rng": recent["random_state"]["torch_cpu"] = torch.zeros(5, dtype=torch.uint8)
    elif change == "best0": objects["best.pt"]["epoch"] = 0
    elif change == "paused_metrics": objects["metrics.json"] = {"status": "completed"}
    elif change == "duplicate_receipt": objects["HF会话收据_0001x.json"] = objects["HF会话收据_0001.json"]
    elif change == "cost": objects["HF会话收据_0001.json"]["本会话总墙钟秒"] = 4.
    elif change == "consumption": objects["HF会话收据_0001.json"]["消费累计"]["AdamW优化步"] += 1
    elif change == "resume_before": objects["HF会话收据_0001.json"]["恢复源before_SHA256"] = "1"*64
    elif change == "initial_nl":
        initial = objects["阶段_初始.pt"]
        initial["model_state"]["nonlinear_subnet.branch_encoder.weight"].add_(.1)
        initial["metadata"]["CPU初始化身份"]["三网初始化状态内容SHA256"] = m._state_content_sha(initial["model_state"])
    elif change == "source_qualification": objects["事前真实来源登记.json"]["LF来源完整资格"]["真实累计成本秒"] = 200.
    else:
        hashes["事前来源快照/登记.yaml"] = "0"*64
        objects["HF会话收据_0001.json"]["原件SHA256"]["事前来源快照/登记.yaml"] = "0"*64
    with pytest.raises(ValueError):
        m.audit_task11_howard_hf_artifacts(objects, hashes, seed=0, source_identity=identity, finished=False, formal=False)


def test_frozen_source_union_covers_local_import_closure_without_data_reads():
    m = _formal()
    root = Path(__file__).resolve().parents[1]
    frozen = set(m.source_files())
    pending = [name for name in frozen if name.startswith("src/") and name.endswith(".py")]
    discovered = set()
    while pending:
        name = pending.pop()
        if name in discovered: continue
        discovered.add(name)
        path = root / name
        parts = Path(name).with_suffix("").parts[1:]
        package = parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.Import): modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = list(package[:len(package)-node.level+1]) if node.level else []
                modules = [".".join(prefix+((node.module or "").split(".") if node.module else []))]
                modules += [base+"."+alias.name for base in modules[:] for alias in node.names]
            for module in modules:
                if not module.startswith("sic_cu"): continue
                base = Path("src").joinpath(*module.split("."))
                for candidate in (base.with_suffix(".py"), base / "__init__.py"):
                    if (root / candidate).is_file(): pending.append(candidate.as_posix())
                for length in range(1, len(module.split("."))):
                    parent = Path("src").joinpath(*module.split(".")[:length], "__init__.py")
                    if (root / parent).is_file(): pending.append(parent.as_posix())
    assert discovered <= frozen, "冻结成员缺真实本地import依赖: " + str(sorted(discovered-frozen))


@pytest.mark.parametrize("corrupt", [None, "momentum", "rng", "group", "weight", "frozen"])
def test_real_cpu_stage_boundary_preserves_actual_all56_optimizer_and_rng(cpu_epoch_artifacts, corrupt):
    m, _, _, _, objects, _, _ = cpu_epoch_artifacts
    first = copy.deepcopy(objects["阶段_最近.pt"])
    second = copy.deepcopy(first)
    second["stage"], second["epoch"] = m.STAGE2, 0
    second["optimizer_state"]["param_groups"][0]["lr"] = .0001
    # This is only the boundary tensor contract of a real ten-epoch CPU fixture,
    # not a claim that the formal 1500/500 phases were executed or shortened.
    if corrupt == "momentum": next(iter(second["optimizer_state"]["state"].values()))["exp_avg"].add_(1)
    elif corrupt == "rng": second["random_state"]["torch_cpu"][0] ^= 1
    elif corrupt == "group": second["optimizer_state"]["param_groups"].append(copy.deepcopy(second["optimizer_state"]["param_groups"][0]))
    elif corrupt == "weight": second["model_state"]["low_fidelity_subnet.branch_encoder.weight"].add_(1)
    elif corrupt == "frozen": second["parameter_requires_grad"][next(iter(second["parameter_requires_grad"]))] = False
    if corrupt:
        with pytest.raises(ValueError): m._audit_stage_boundary(first, second)
    else:
        m._audit_stage_boundary(first, second)
        assert all(float(state["step"]) == 160 for state in second["optimizer_state"]["state"].values())


def test_cpu_best_path_and_two_actual_commits_reach_first_fixed_validation(cpu_epoch_artifacts):
    m, _, optimizer, _, objects, hashes, identity = cpu_epoch_artifacts
    result = m.audit_task11_howard_hf_artifacts(objects, hashes, seed=0, source_identity=identity, finished=False, formal=False)
    assert result["全局实际轮次"] == 10, "真实CPU夹具尚未经过第10轮best/validation保存分支"
    assert result["全局最佳轮次"] == 10 and result["会话数"] == 2
    assert all(float(state["step"]) == 160 for state in optimizer.state_dict()["state"].values())


def test_formal_hash_bridge_is_local_and_preserves_real_cpu_tensor_dtypes(lf_view):
    m = _formal()
    from sic_cu.train.task11_howard_hf_initialization import _state_content_sha as cpu_hash
    assert m._state_content_sha.__module__ == m.__name__, "正式CUDA状态仍直接调用冻结仅CPU的.numpy()哈希helper"
    tensors = {**copy.deepcopy(lf_view["lf_subnet_state"]), "PH": torch.tensor([[.01, -.01, 5., 0.]], dtype=torch.float64)}
    assert m._state_content_sha(tensors) == cpu_hash(tensors)
    assert tensors["PH"].dtype == torch.float64


def test_formal_default_dtype_rejects_float64_before_cuda_or_temperature_read(fake_project, monkeypatch):
    m, arguments, *_ = fake_project
    monkeypatch.setattr(m, "PROJECT_ROOT", Path(arguments["project_root"]))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("float64默认dtype未在CUDA前拒绝，原physics sampler会产生Double坐标"))
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError): m.run_task11_howard_hf_formal(**arguments)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("variant", ["short_closer", "info_closer", "tilde_short", "duplicate_record_other_gate"])
def test_plain_gate_rejects_unclosed_commonmark_fence_or_record_collision(fake_project, variant):
    m, arguments, *_ = fake_project
    ledger = Path(arguments["project_root"]) / m.ROOT_LEDGER
    row = ledger.read_text(encoding="utf-8")
    if variant == "short_closer": raw = "````\n```\n"+row+"````\n"
    elif variant == "info_closer": raw = "```\n```python\n"+row+"```\n"
    elif variant == "tilde_short": raw = "~~~~\n~~~\n"+row+"~~~~\n"
    else: raw = "| 录-0108 | OTHER_GATE:v1; status=active | CPU fixture |\n"+row
    ledger.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError): m.preflight_task11_howard_hf(**arguments)


@pytest.mark.parametrize("opening,closing", [("<!--", "-->"), ("<script>", "</script>"), ("<pre>", "</pre>"),
    ("<style>", "</style>"), ("<textarea>", "</textarea>"), ("<div>", "</div>"),
    ("<![CDATA[", "]]>"), ("<?php", "?>"), ("<!DOCTYPE", ">")])
def test_plain_gate_rejects_hidden_html_comment_or_raw_block(fake_project, opening, closing):
    m, arguments, *_ = fake_project
    ledger = Path(arguments["project_root"]) / m.ROOT_LEDGER
    row = ledger.read_text(encoding="utf-8")
    ledger.write_text(opening+"\n"+row+closing+"\n", encoding="utf-8")
    with pytest.raises(ValueError): m.preflight_task11_howard_hf(**arguments)


@pytest.mark.parametrize("prefix", ["<!-- ordinary comment\n-->\n", "<script>ordinary text\n</script>\n",
    "<div>ordinary text\n</div>\n\n", "```text\n<!--\n```\n"])
def test_plain_gate_after_closed_html_or_fenced_html_is_still_accepted(fake_project, prefix):
    m, arguments, *_ = fake_project
    ledger = Path(arguments["project_root"]) / m.ROOT_LEDGER
    row = ledger.read_text(encoding="utf-8")
    ledger.write_text(prefix+row, encoding="utf-8")
    assert m.preflight_task11_howard_hf(**arguments)["HF训练许可"] is False


@pytest.mark.parametrize("prefix", ["<!-- closed --> <!--\n", "<!--\n--> <!--\n",
    "<script>ordinary text\n</script> <!--\n", "<script><!--\n-->\n", "<pre><!--\n-->\n",
    "<div><!-- ordinary comment -->\n", "<div><!--\n\n"])
def test_plain_gate_rejects_comment_reopened_after_html_close_on_same_line(fake_project, prefix):
    m, arguments, *_ = fake_project
    ledger = Path(arguments["project_root"]) / m.ROOT_LEDGER
    row = ledger.read_text(encoding="utf-8")
    ledger.write_text(prefix+row+"-->\n", encoding="utf-8")
    with pytest.raises(ValueError): m.preflight_task11_howard_hf(**arguments)


@pytest.mark.parametrize("opening,closing", [("<div><script>", "</script></div>"),
    ("<div><pre>", "</pre></div>"), ("<div>\n<script>", "</script>\n</div>")])
def test_plain_gate_rejects_nested_raw_html_context_across_blank_line(fake_project, opening, closing):
    m, arguments, *_ = fake_project
    ledger = Path(arguments["project_root"]) / m.ROOT_LEDGER
    row = ledger.read_text(encoding="utf-8")
    ledger.write_text(opening+"\n\n"+row+closing+"\n", encoding="utf-8")
    with pytest.raises(ValueError): m.preflight_task11_howard_hf(**arguments)
