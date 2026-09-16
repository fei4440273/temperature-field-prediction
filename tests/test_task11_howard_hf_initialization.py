"""只用真实一步合成CPU LF权重验证HF初始化；不代表任何正式训练。"""

from __future__ import annotations

import copy
import importlib
import json
import random
from dataclasses import asdict, fields, replace

import numpy as np
import pytest
import torch

from sic_cu.models.common import ModelScales
from sic_cu.models.task11_howard_composite import Task11HowardComposite
from sic_cu.train.task11_howard_lf_formal import (
    LF_MODEL_KWARGS, LF_PARAMETER_COUNT, METHOD, Task11HowardLFTeacher,
)


def _initialization():
    name = "sic_cu.train.task11_howard_hf_initialization"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name == name:
            pytest.fail("纯CPU Howard HF初始化入口尚未实现")
        raise


def _coordinates():
    return torch.tensor([[0.012, -0.009, 1.0, 115.2, 0.0],
                         [0.044, 0.006, 50.0, 403.0, 1.0],
                         [0.030, -0.004, 100.0, 115.2, 0.0]], dtype=torch.float32)


def _queries():
    return {"linear_query_points": [[0.011, -0.008, 10.0, 0.0],
                                    [0.042, 0.007, 50.0, 1.0]],
            "nonlinear_query_points": [[0.033, 0.003, 1.0, 1.0],
                                       [0.015, -0.013, 10.0, 0.0]]}


def _query_metadata():
    return {"PH": {"source": "synthetic_cpu_queries", "row_ids": [2, 0]},
            "QH": {"source": "synthetic_cpu_queries", "row_ids": [1, 3]},
            "真实查询许可已核验": False}


@pytest.fixture(scope="module")
def one_step_lf_view():
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        teacher = Task11HowardLFTeacher()
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=0.001, weight_decay=1e-6)
        loss = ((teacher(_coordinates()) - 295.15) / 250.0).square().mean()
        loss.backward()
        optimizer.step()
    assert all(float(state["step"]) == 1 for state in optimizer.state_dict()["state"].values())
    return {"schema_version": 1, "method": METHOD, "seed": 0, "epoch": 1,
            "model_kwargs": dict(LF_MODEL_KWARGS), "scales": asdict(ModelScales()),
            "LF参数量": LF_PARAMETER_COUNT, "validation_rmse_c": 0.8,
            "model_state": copy.deepcopy(teacher.state_dict()),
            "lf_subnet_state": copy.deepcopy(teacher.network.state_dict()),
            "任11Howard事前来源": {"YAML_SHA256": "1" * 64, "TAR_SHA256": "2" * 64,
                                "CATALOG_SHA256": "3" * 64},
            "原文精确三网联合训练复现": False,
            "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
            "HF训练许可": False}


def _initialize(view, *, seed=17, **kwargs):
    return _initialization().initialize_task11_howard_hf(
        view, seed=seed, **_queries(), query_metadata=_query_metadata(), **kwargs)


def _rzt_derivatives(output, coordinates):
    first = torch.autograd.grad(output.sum(), coordinates, create_graph=True)[0]
    second = torch.stack([torch.autograd.grad(first[:, axis].sum(), coordinates,
                                            create_graph=True, retain_graph=True)[0][:, axis]
                          for axis in range(3)], dim=1)
    return first[:, :3], second


def _dimensionless_graph_loss(output, coordinates):
    first, second = _rzt_derivatives(output, coordinates)
    length_time = output.new_tensor([0.05834, 0.0175, 200.0])
    return output.square().mean() + (first * length_time).square().mean() + (
        second * length_time.square()).square().mean()


def test_complete_lf_state_migration_preserves_pointwise_output(one_step_lf_view):
    model, identity = _initialize(one_step_lf_view)
    with torch.random.fork_rng(devices=[]):
        teacher = Task11HowardLFTeacher()
    teacher.network.load_state_dict(one_step_lf_view["lf_subnet_state"], strict=True)
    assert model.low_fidelity_subnet.state_dict().keys() == teacher.network.state_dict().keys()
    assert all(torch.equal(value, teacher.network.state_dict()[name])
               for name, value in model.low_fidelity_subnet.state_dict().items())
    torch.testing.assert_close(model(_coordinates(), fidelity="low"), teacher(_coordinates()), rtol=0, atol=0)
    assert identity["LF来源seed"] == 0 and identity["HF初始化seed"] == 17
    assert identity["LF参数量"] == LF_PARAMETER_COUNT
    assert identity["HF训练许可"] is False
    assert identity["LF外部SHA与完整终态资格核验"] is False
    assert identity["原文精确三网联合训练复现"] is False


def test_hf_subnets_are_fresh_seeded_and_all_three_networks_trainable(one_step_lf_view):
    model, identity = _initialize(one_step_lf_view)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.default_generator.manual_seed(17)
        fresh = Task11HowardComposite(**_queries())
    for subnet in ("linear_subnet", "nonlinear_subnet"):
        assert all(torch.equal(value, getattr(fresh, subnet).state_dict()[name])
                   for name, value in getattr(model, subnet).state_dict().items())
    assert all(parameter.requires_grad is True and parameter.device.type == "cpu"
               and parameter.dtype == torch.float32 for parameter in model.parameters())
    assert identity["新鲜HF子网"] == ["linear_subnet", "nonlinear_subnet"]


def test_init_is_seed_repeatable_without_changing_caller_rng_or_dtype(one_step_lf_view, monkeypatch):
    initializer = _initialization()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state().clone()
    initial_dtype = torch.get_default_dtype()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("初始化不能探测CUDA"))
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("不能读取CUDA RNG"))
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda *args: pytest.fail("不能修改CUDA RNG"))
    monkeypatch.setattr(torch, "manual_seed", lambda *args: pytest.fail("不能调用可能涉及CUDA的全局seed"))
    first, first_identity = _initialize(one_step_lf_view)
    try:
        torch.set_default_dtype(torch.float64)
        second, second_identity = _initialize(one_step_lf_view)
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(initial_dtype)
    assert torch.equal(cpu_state, torch.get_rng_state())
    assert python_state == random.getstate()
    now = np.random.get_state()
    assert now[0] == numpy_state[0] and np.array_equal(now[1], numpy_state[1]) and now[2:] == numpy_state[2:]
    assert first_identity == second_identity
    assert all(torch.equal(value, second.state_dict()[name]) for name, value in first.state_dict().items())
    different, different_identity = initializer.initialize_task11_howard_hf(
        one_step_lf_view, seed=18, **_queries(), query_metadata=_query_metadata())
    assert first_identity["三网初始化状态内容SHA256"] != different_identity["三网初始化状态内容SHA256"]
    assert any(not torch.equal(value, different.linear_subnet.state_dict()[name])
               for name, value in first.linear_subnet.state_dict().items())
    assert all(torch.equal(value, different.low_fidelity_subnet.state_dict()[name])
               for name, value in first.low_fidelity_subnet.state_dict().items())


def test_model_kwargs_and_caller_query_metadata_roundtrip_without_alias(one_step_lf_view):
    initializer = _initialization()
    points, metadata = _queries(), _query_metadata()
    model, identity = initializer.initialize_task11_howard_hf(
        one_step_lf_view, seed=17, **points, query_metadata=metadata, query_chunk_size=3)
    assert identity["模型构造参数"] == model.model_kwargs
    assert identity["调用者查询元数据"] == json.loads(json.dumps(metadata))
    reconstructed = Task11HowardComposite(**identity["模型构造参数"])
    reconstructed.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(reconstructed(_coordinates()), model(_coordinates()), rtol=0, atol=0)
    points["linear_query_points"][0][0] = 99.0
    metadata["PH"]["row_ids"][0] = 99
    assert model.linear_query_points[0, 0] != 99
    assert identity["调用者查询元数据"]["PH"]["row_ids"][0] == 2
    with torch.no_grad():
        model.low_fidelity_subnet.branch_encoder.weight.add_(1)
    assert not torch.equal(model.low_fidelity_subnet.branch_encoder.weight,
                           one_step_lf_view["lf_subnet_state"]["branch_encoder.weight"])


@pytest.mark.parametrize("mutation", ["method", "lf_seed_bool", "lf_seed_float", "lf_seed_range",
    "epoch_bool", "epoch_zero", "width_float", "depth_float", "latent_float", "activation_int",
    "scale", "scale_type", "parameter_count", "test_permission", "hf_permission", "paper_claim",
    "missing_state", "extra_name", "bad_shape", "bad_dtype", "nonfinite", "model_view_mismatch",
    "non_cpu", "bad_source_sha", "schema_bool"])
def test_strict_lf_view_validation_rejects_internal_alias_or_invalid_weights(one_step_lf_view, mutation):
    initializer = _initialization()
    view = copy.deepcopy(one_step_lf_view)
    raw = view["lf_subnet_state"]
    first_name = next(iter(raw))
    if mutation == "method":
        view["method"] = "mlp_pinn"
    elif mutation.startswith("lf_seed_"):
        view["seed"] = {"lf_seed_bool": False, "lf_seed_float": 0.0, "lf_seed_range": 5}[mutation]
    elif mutation.startswith("epoch_"):
        view["epoch"] = {"epoch_bool": True, "epoch_zero": 0}[mutation]
    elif mutation in {"width_float", "depth_float", "latent_float", "activation_int"}:
        key, value = {"width_float": ("width", 128.0), "depth_float": ("depth", 4.0),
                      "latent_float": ("latent_dim", 128.0), "activation_int": ("final_activation", 0)}[mutation]
        view["model_kwargs"][key] = value
    elif mutation == "scale":
        view["scales"]["temperature_scale_k"] = 251.0
    elif mutation == "scale_type":
        view["scales"]["time_max_s"] = 200
    elif mutation == "parameter_count":
        view["LF参数量"] = float(LF_PARAMETER_COUNT)
    elif mutation == "test_permission":
        view["旧固定TEST温度读取"] = 0
    elif mutation == "hf_permission":
        view["HF训练许可"] = True
    elif mutation == "paper_claim":
        view["原文精确三网联合训练复现"] = True
    elif mutation == "missing_state":
        raw.pop(first_name)
    elif mutation == "extra_name":
        raw["dummy.weight"] = torch.zeros(1)
    elif mutation == "bad_shape":
        raw[first_name] = raw[first_name].reshape(-1)
    elif mutation == "bad_dtype":
        raw[first_name] = raw[first_name].double()
    elif mutation == "nonfinite":
        raw[first_name].flatten()[0] = float("nan")
    elif mutation == "model_view_mismatch":
        view["model_state"]["network." + first_name].add_(1)
    elif mutation == "non_cpu":
        raw[first_name] = torch.empty(raw[first_name].shape, device="meta")
    elif mutation == "bad_source_sha":
        view["任11Howard事前来源"]["YAML_SHA256"] = "not_external_authority"
    elif mutation == "schema_bool":
        view["schema_version"] = True
    with pytest.raises(ValueError, match="LF|CPU|来源|构造|尺度|权重|隔离|参数"):
        initializer.initialize_task11_howard_hf(view, seed=17, **_queries(), query_metadata=_query_metadata())


@pytest.mark.parametrize("argument,value", [("seed", True), ("seed", 17.0), ("seed", -1),
                                            ("query_chunk_size", 0), ("query_chunk_size", 1.0)])
def test_hf_seed_and_chunk_are_strict_without_defining_final_training_budget(one_step_lf_view, argument, value):
    initializer = _initialization()
    arguments = {"seed": 17, "query_chunk_size": 3, **_queries(), "query_metadata": _query_metadata()}
    arguments[argument] = value
    with pytest.raises(ValueError):
        initializer.initialize_task11_howard_hf(one_step_lf_view, **arguments)


@pytest.mark.parametrize("bad", [[], [[0.0, 0.0, 1.0]], [[0.0, 0.0, float("nan"), 0]],
                                 [[0.0, 0.0, 1.0, 0.5]], torch.empty((1, 4), device="meta")])
def test_query_points_must_be_explicit_finite_cpu_material_coordinates(one_step_lf_view, bad):
    initializer = _initialization()
    points = _queries()
    points["linear_query_points"] = bad
    with pytest.raises(ValueError):
        initializer.initialize_task11_howard_hf(one_step_lf_view, seed=17, **points,
                                               query_metadata=_query_metadata())


@pytest.mark.parametrize("metadata", [{"PH": float("nan")}, {1: "invalid_key"}, {"QH": object()}])
def test_query_metadata_is_json_roundtrippable_not_a_permission_gate(one_step_lf_view, metadata):
    with pytest.raises(ValueError):
        _initialization().initialize_task11_howard_hf(one_step_lf_view, seed=17, **_queries(),
                                                     query_metadata=metadata)


def test_hf_sum_is_linear_plus_nonlinear_never_lf_plus_residual(one_step_lf_view):
    model, _ = _initialize(one_step_lf_view)
    outputs = model.subnet_outputs(_coordinates())
    correct = 295.15 + 250.0 * (outputs["linear"] + outputs["nonlinear"])
    wrong = correct + 250.0 * outputs["low_fidelity"]
    torch.testing.assert_close(model(_coordinates()), correct, rtol=0, atol=0)
    assert not torch.allclose(model(_coordinates()), wrong)


@pytest.mark.parametrize("field", ["low_fidelity", "linear", "nonlinear", "correction", "high"])
def test_migrated_fields_keep_second_rzt_derivatives_and_relevant_parameter_graphs(one_step_lf_view, field):
    model, _ = _initialize(one_step_lf_view)
    model.double()
    coordinates = _coordinates().double().requires_grad_(True)
    parts = model.subnet_outputs(coordinates)
    normalized_high = parts["linear"] + parts["nonlinear"]
    output = {**parts, "high": normalized_high,
              "correction": normalized_high - parts["low_fidelity"]}[field]
    loss = _dimensionless_graph_loss(output, coordinates)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    expected = {"low_fidelity": ("low_fidelity_subnet.",),
                "linear": ("low_fidelity_subnet.", "linear_subnet."),
                "nonlinear": ("low_fidelity_subnet.", "nonlinear_subnet."),
                "correction": ("low_fidelity_subnet.", "linear_subnet.", "nonlinear_subnet."),
                "high": ("low_fidelity_subnet.", "linear_subnet.", "nonlinear_subnet.")}[field]
    for (name, parameter), gradient in zip(model.named_parameters(), gradients):
        if name.startswith(expected):
            assert gradient is not None and gradient.shape == parameter.shape and torch.isfinite(gradient).all()
        else:
            assert gradient is None


def test_migrated_three_subnet_nominal_physics_gradients_are_finite_without_hf_step(one_step_lf_view):
    from sic_cu.physics.collocation import sample_collocation
    from sic_cu.train.simulation import _physics_ready

    model, _ = _initialize(one_step_lf_view)
    model.double()
    before = copy.deepcopy(model.state_dict())
    batch = sample_collocation(4, torch.device("cpu"), seed=123)
    batch = replace(batch, **{field.name: value.double() if value.is_floating_point() else value
                              for field in fields(batch) for value in [getattr(batch, field.name)]})
    components = _physics_ready()(model, batch)
    assert set(components) == {"pde", "initial", "boundary", "interface", "physics_total"}
    assert all(torch.isfinite(value) for value in components.values())
    gradients = torch.autograd.grad(components["physics_total"], tuple(model.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_chunk_one_and_large_chunk_keep_outputs_rzt_and_all_parameter_gradients(one_step_lf_view):
    one, _ = _initialize(one_step_lf_view, query_chunk_size=1)
    large, _ = _initialize(one_step_lf_view, query_chunk_size=16384)
    one.double()
    large.double()
    first_x = _coordinates().double().requires_grad_(True)
    second_x = first_x.detach().clone().requires_grad_(True)
    first_output, second_output = one(first_x), large(second_x)
    torch.testing.assert_close(first_output, second_output, rtol=2e-11, atol=2e-10)
    for actual, expected in zip(_rzt_derivatives(first_output, first_x),
                                _rzt_derivatives(second_output, second_x)):
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-7)
    first_loss = _dimensionless_graph_loss((first_output - 295.15) / 250.0, first_x)
    second_loss = _dimensionless_graph_loss((second_output - 295.15) / 250.0, second_x)
    for actual, expected in zip(torch.autograd.grad(first_loss, tuple(one.parameters())),
                                torch.autograd.grad(second_loss, tuple(large.parameters()))):
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-8)


def test_real_adamw_is_optional_empty_and_uses_only_caller_explicit_lr(one_step_lf_view):
    initializer = _initialization()
    model, _ = _initialize(one_step_lf_view)
    with pytest.raises(TypeError):
        initializer.build_task11_howard_hf_adamw(model, weight_decay=1e-6)
    optimizer = initializer.build_task11_howard_hf_adamw(model, lr=0.002, weight_decay=1e-6)
    assert isinstance(optimizer, torch.optim.AdamW)
    state = optimizer.state_dict()
    assert state["state"] == {}
    assert len(state["param_groups"]) == 1 and state["param_groups"][0]["lr"] == 0.002
    assert state["param_groups"][0]["params"] == list(range(len(list(model.parameters()))))
    assert {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} == {
        id(parameter) for parameter in model.parameters()}


@pytest.mark.parametrize("mutation", ["frozen_lf", "frozen_linear", "frozen_nonlinear", "bad_lr", "bool_lr"])
def test_optimizer_cannot_silently_omit_or_freeze_any_of_three_networks(one_step_lf_view, mutation):
    initializer = _initialization()
    model, _ = _initialize(one_step_lf_view)
    lr = 0.002
    if mutation.startswith("frozen_"):
        subnet = {"frozen_lf": "low_fidelity_subnet", "frozen_linear": "linear_subnet",
                  "frozen_nonlinear": "nonlinear_subnet"}[mutation]
        next(getattr(model, subnet).parameters()).requires_grad_(False)
    elif mutation == "bad_lr":
        lr = float("nan")
    else:
        lr = True
    with pytest.raises(ValueError):
        initializer.build_task11_howard_hf_adamw(model, lr=lr, weight_decay=1e-6)


@pytest.mark.parametrize("kwargs", [
    {"eps": float("inf")}, {"eps": float("nan")}, {"eps": -1.0}, {"eps": True}, {"eps": "1e-8"},
    {"betas": (float("nan"), 0.999)}, {"betas": (0.9, float("inf"))},
    {"betas": (False, 0.999)}, {"betas": (0.9, True)}, {"betas": (1.0, 0.999)},
    {"betas": (-0.1, 0.999)}, {"betas": (0.9,)}, {"betas": (0.9, 0.999, 0.1)},
    {"betas": torch.tensor([0.9, 0.999])}, {"betas": ("0.9", 0.999)}, {"betas": None},
    {"maximize": "False"}, {"amsgrad": 0}, {"amsgrad": None}, {"capturable": None},
    {"capturable": 0}, {"capturable": True}, {"differentiable": None}, {"differentiable": 0},
    {"differentiable": True}, {"foreach": "False"}, {"foreach": 0}, {"fused": "False"},
    {"fused": 0}, {"fused": True}, {"fused": True, "foreach": True},
    {"differentiable": True, "foreach": True}, {"unknown_hf_budget": 0.002},
    {"params": []}, {"decoupled_weight_decay": True},
])
def test_adamw_kwargs_are_strict_finite_and_supported_cpu_modes(one_step_lf_view, kwargs):
    initializer = _initialization()
    model, _ = _initialize(one_step_lf_view)
    with pytest.raises(ValueError):
        initializer.build_task11_howard_hf_adamw(model, lr=0.002, weight_decay=1e-6, **kwargs)


@pytest.mark.parametrize("kwargs", [{}, {"betas": (0.8, 0.95), "eps": 1e-7, "amsgrad": True},
    {"betas": [0, 0.9], "eps": 0.0, "maximize": False, "foreach": False},
    {"foreach": True, "fused": False, "maximize": True},
    {"foreach": None, "fused": None, "capturable": False, "differentiable": False}])
def test_cpu_adamw_supported_options_keep_empty_complete_unique_parameter_group(one_step_lf_view, kwargs):
    initializer = _initialization()
    model, _ = _initialize(one_step_lf_view)
    before = copy.deepcopy(model.state_dict())
    optimizer = initializer.build_task11_howard_hf_adamw(model, lr=0.002, weight_decay=1e-6, **kwargs)
    state = optimizer.state_dict()
    assert isinstance(optimizer, torch.optim.AdamW) and state["state"] == {}
    assert len(state["param_groups"]) == 1
    assert state["param_groups"][0]["params"] == list(range(56))
    parameters = optimizer.param_groups[0]["params"]
    assert len(parameters) == len({id(parameter) for parameter in parameters}) == 56
    assert {id(parameter) for parameter in parameters} == {id(parameter) for parameter in model.parameters()}
    storage_ranges = sorted((parameter.untyped_storage().data_ptr(),
                             parameter.untyped_storage().data_ptr() + parameter.untyped_storage().nbytes())
                            for parameter in parameters)
    assert len({start for start, _ in storage_ranges}) == 56
    assert all(end <= next_start for (_, end), (next_start, _) in zip(storage_ranges, storage_ranges[1:]))
    for name, value in kwargs.items():
        assert optimizer.param_groups[0][name] == value
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    if not kwargs:
        assert optimizer.defaults == {"lr": 0.002, "betas": (0.9, 0.999), "eps": 1e-8,
            "weight_decay": 1e-6, "amsgrad": False, "foreach": None, "maximize": False,
            "capturable": False, "differentiable": False, "fused": None}


@pytest.mark.parametrize("source_subnet,target_subnet", [
    ("low_fidelity_subnet", "linear_subnet"), ("low_fidelity_subnet", "nonlinear_subnet"),
    ("linear_subnet", "nonlinear_subnet"), ("low_fidelity_subnet", "low_fidelity_subnet"),
    ("linear_subnet", "linear_subnet"), ("nonlinear_subnet", "nonlinear_subnet"),
])
@pytest.mark.parametrize("alias", ["same_data_ptr", "overlapping_torch_slices",
                                 "disjoint_torch_slices", "overlapping_numpy_storages"])
def test_cpu_adamw_rejects_all_parameter_backing_storage_aliases(
        one_step_lf_view, source_subnet, target_subnet, alias):
    initializer = _initialization()
    model, _ = _initialize(one_step_lf_view)
    source_layer = getattr(model, source_subnet).trunk_layers[1]
    target_layers = "branch_layers" if source_subnet == target_subnet else "trunk_layers"
    target_layer = getattr(getattr(model, target_subnet), target_layers)[1]
    original = source_layer.weight
    count = original.numel()
    if alias == "same_data_ptr":
        target_layer.weight = torch.nn.Parameter(original.detach())
    else:
        offset = count if alias == "disjoint_torch_slices" else count // 2
        if alias == "overlapping_numpy_storages":
            owner = np.linspace(-0.2, 0.2, count + offset, dtype=np.float32)
            source = torch.from_numpy(owner[:count]).reshape_as(original)
            target = torch.from_numpy(owner[offset:offset + count]).reshape_as(original)
        else:
            owner = torch.linspace(-0.2, 0.2, count + offset, dtype=torch.float32)
            source = owner[:count].reshape_as(original)
            target = owner[offset:offset + count].reshape_as(original)
        source_layer.weight = torch.nn.Parameter(source)
        target_layer.weight = torch.nn.Parameter(target)
    source, target = source_layer.weight, target_layer.weight
    assert source is not target and source.shape == target.shape == original.shape
    assert source.device.type == target.device.type == "cpu"
    assert source.dtype == target.dtype == torch.float32
    assert source.layout == target.layout == torch.strided
    assert torch.isfinite(source).all() and torch.isfinite(target).all()
    source_storage, target_storage = source.untyped_storage(), target.untyped_storage()
    if alias == "same_data_ptr":
        assert source.data_ptr() == target.data_ptr()
    else:
        assert source.data_ptr() != target.data_ptr()
    if alias == "overlapping_numpy_storages":
        assert source_storage.data_ptr() != target_storage.data_ptr()
    else:
        assert source_storage.data_ptr() == target_storage.data_ptr()
    assert max(source_storage.data_ptr(), target_storage.data_ptr()) < min(
        source_storage.data_ptr() + source_storage.nbytes(),
        target_storage.data_ptr() + target_storage.nbytes())
    if alias == "disjoint_torch_slices":
        assert source.data_ptr() + count * source.element_size() <= target.data_ptr()
    parameters = list(model.parameters())
    assert len(parameters) == len({id(parameter) for parameter in parameters}) == 56
    values_before = [parameter.detach().clone() for parameter in parameters]
    bindings_before = [(id(parameter), parameter.data_ptr(), parameter.untyped_storage().data_ptr(),
                        parameter.untyped_storage().nbytes()) for parameter in parameters]
    source_before = copy.deepcopy(one_step_lf_view["lf_subnet_state"])
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="存储|独立|重叠"):
        initializer.build_task11_howard_hf_adamw(model, lr=0.002, weight_decay=1e-6)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert bindings_before == [(id(parameter), parameter.data_ptr(), parameter.untyped_storage().data_ptr(),
                                parameter.untyped_storage().nbytes()) for parameter in model.parameters()]
    assert all(torch.equal(parameter, before) for parameter, before in zip(model.parameters(), values_before))
    assert all(torch.equal(value, source_before[name])
               for name, value in one_step_lf_view["lf_subnet_state"].items())


@pytest.mark.parametrize("mutation", ["missing_linear", "missing_lf", "missing_nonlinear",
    "replace_linear_child", "empty_layers", "parameter_shape", "parameter_name", "shared_parameter",
    "extra_parameter", "linear_layer_replaced", "linear_layer_class", "lf_activation_int",
    "nonlinear_activation_false", "scaler_scales", "buffer_missing", "buffer_extra", "buffer_shape",
    "buffer_nonfinite", "buffer_dtype", "buffer_material", "buffer_requires_grad", "buffer_count",
    "buffer_non_cpu", "query_chunk_type", "parameter_dtype", "extra_module"])
def test_cpu_model_guard_requires_real_three_network_structure_and_fixed_query_buffers(one_step_lf_view, mutation):
    initializer = _initialization()
    model, _ = _initialize(one_step_lf_view)
    if mutation.startswith("missing_"):
        name = {"missing_linear": "linear_subnet", "missing_lf": "low_fidelity_subnet",
                "missing_nonlinear": "nonlinear_subnet"}[mutation]
        setattr(model, name, torch.nn.Identity())
    elif mutation == "replace_linear_child":
        model.linear_subnet = copy.deepcopy(model.nonlinear_subnet)
    elif mutation == "empty_layers":
        model.linear_subnet.branch_layers = torch.nn.ModuleList()
    elif mutation == "parameter_shape":
        model.linear_subnet.trunk_layers[0].weight = torch.nn.Parameter(torch.ones(128, 5))
    elif mutation == "parameter_name":
        layer = model.linear_subnet.trunk_layers[0]
        old_bias = layer.bias
        del layer.bias
        layer.register_parameter("renamed_bias", old_bias)
    elif mutation == "shared_parameter":
        model.linear_subnet.trunk_layers[0].weight = model.low_fidelity_subnet.trunk_layers[0].weight
    elif mutation == "extra_parameter":
        model.linear_subnet.register_parameter("unexpected", torch.nn.Parameter(torch.ones(1)))
    elif mutation == "linear_layer_replaced":
        model.linear_subnet.trunk_layers[0] = torch.nn.Identity()
    elif mutation == "linear_layer_class":
        class AlteredLinear(torch.nn.Linear):
            def forward(self, value):
                return super().forward(value) * 0.0

        old_layer = model.linear_subnet.trunk_layers[0]
        altered = AlteredLinear(old_layer.in_features, old_layer.out_features)
        altered.load_state_dict(old_layer.state_dict(), strict=True)
        model.linear_subnet.trunk_layers[0] = altered
    elif mutation == "lf_activation_int":
        model.low_fidelity_subnet.final_activation = 0
    elif mutation == "nonlinear_activation_false":
        model.nonlinear_subnet.final_activation = False
    elif mutation == "scaler_scales":
        model.scaler.scales = ModelScales(temperature_scale_k=123.0)
    elif mutation == "buffer_missing":
        del model.nonlinear_query_points
    elif mutation == "buffer_extra":
        model.register_buffer("extra_query_buffer", torch.zeros(1))
    elif mutation == "buffer_shape":
        model.linear_query_points = model.linear_query_points[:, :3].clone()
    elif mutation == "buffer_nonfinite":
        model.linear_query_points[0, 0] = float("nan")
    elif mutation == "buffer_dtype":
        model.linear_query_points = model.linear_query_points.float()
    elif mutation == "buffer_material":
        model.linear_query_points[0, 3] = 0.5
    elif mutation == "buffer_requires_grad":
        model.linear_query_points.requires_grad_(True)
    elif mutation == "buffer_count":
        model.linear_query_points = model.linear_query_points[:1].clone()
    elif mutation == "buffer_non_cpu":
        model.linear_query_points = torch.empty((2, 4), device="meta")
    elif mutation == "query_chunk_type":
        model.query_chunk_size = 1.0
    elif mutation == "parameter_dtype":
        model.linear_subnet.trunk_layers[0].weight = torch.nn.Parameter(
            model.linear_subnet.trunk_layers[0].weight.double())
    elif mutation == "extra_module":
        model.linear_subnet.register_module("unused_identity", torch.nn.Identity())
    with pytest.raises(ValueError):
        initializer.build_task11_howard_hf_adamw(model, lr=0.002, weight_decay=1e-6)


def test_cpu_structure_reference_preserves_rng_default_device_dtype_and_lf_weights(one_step_lf_view):
    initializer = _initialization()
    model, _ = _initialize(one_step_lf_view)
    before = copy.deepcopy(model.state_dict())
    source_before = copy.deepcopy(one_step_lf_view["lf_subnet_state"])
    old_dtype, old_device = torch.get_default_dtype(), torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float64)
        torch.set_default_device("meta")
        cpu_rng = torch.get_rng_state().clone()
        optimizer = initializer.build_task11_howard_hf_adamw(model, lr=0.002, weight_decay=1e-6)
        assert optimizer.state_dict()["state"] == {}
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.get_default_dtype() == torch.float64
        assert torch.get_default_device().type == "meta"
    finally:
        torch.set_default_dtype(old_dtype)
        torch.set_default_device(old_device)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert all(torch.equal(value, source_before[name])
               for name, value in one_step_lf_view["lf_subnet_state"].items())


def test_cpu_structure_reference_uses_actual_explicit_query_counts(one_step_lf_view):
    initializer = _initialization()
    queries = _queries()
    queries["linear_query_points"].append([0.022, -0.005, 100.0, 0.0])
    queries["nonlinear_query_points"] = queries["nonlinear_query_points"][:1]
    model, identity = initializer.initialize_task11_howard_hf(
        one_step_lf_view, seed=17, **queries, query_metadata=_query_metadata())
    optimizer = initializer.build_task11_howard_hf_adamw(model, lr=0.002, weight_decay=1e-6)
    assert optimizer.state_dict()["state"] == {}
    assert len(list(model.named_parameters())) == 56
    assert model.linear_subnet.branch_layers[0].weight.shape == (128, 3)
    assert model.nonlinear_subnet.branch_layers[0].weight.shape == (128, 2)
    assert identity["模型构造参数"] == model.model_kwargs
