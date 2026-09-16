"""Synthetic CPU-double contracts for the independent Howard-style model."""

from __future__ import annotations

import copy
import importlib

import pytest
import torch
from torch import nn

from sic_cu.models.common import ModelScales, parameter_count


def _model_class():
    return importlib.import_module("sic_cu.models.task11_howard_composite").Task11HowardComposite


def _points():
    linear = torch.tensor([[0.011, -0.008, 10.0, 0.0],
                           [0.042, 0.007, 50.0, 1.0],
                           [0.025, -0.012, 100.0, 0.0]], dtype=torch.float64)
    nonlinear = torch.tensor([[0.033, 0.003, 1.0, 1.0],
                              [0.015, -0.013, 10.0, 0.0],
                              [0.055, 0.010, 200.0, 1.0],
                              [0.027, -0.005, 50.0, 0.0]], dtype=torch.float64)
    return linear, nonlinear


def _model(**kwargs):
    linear, nonlinear = _points()
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(1107)
        return _model_class()(linear_query_points=linear, nonlinear_query_points=nonlinear,
                              width=6, depth=3, latent_dim=5, **kwargs).double()


def _coordinates():
    return torch.tensor([[0.012, -0.009, 1.0, 115.2, 0.0],
                         [0.044, 0.006, 50.0, 403.0, 1.0],
                         [0.030, -0.004, 200.0, 115.2, 0.0],
                         [0.057, 0.010, 10.0, 55.0, 1.0],
                         [0.022, 0.004, 100.0, 403.0, 1.0]], dtype=torch.float64)


def _scale(coordinates, scales):
    r, z, time, power, material = coordinates.unbind(dim=1)
    return torch.stack((2 * r / scales.r_max_m - 1,
                        2 * (z - scales.z_min_m) / (-scales.z_min_m) - 1,
                        2 * time / scales.time_max_s - 1,
                        2 * power / scales.power_max_w - 1,
                        2 * material - 1), dim=1)


def _modified_reference(subnet, branch, trunk, final_activation):
    encoder_u = torch.tanh(subnet.branch_encoder(branch))
    encoder_x = torch.tanh(subnet.trunk_encoder(trunk))
    h_u, h_x = branch, trunk
    for branch_layer, trunk_layer in zip(subnet.branch_layers[:-1], subnet.trunk_layers[:-1]):
        z_u, z_x = torch.tanh(branch_layer(h_u)), torch.tanh(trunk_layer(h_x))
        h_u = (1 - z_u) * encoder_u + z_u * encoder_x
        h_x = (1 - z_x) * encoder_u + z_x * encoder_x
    b, t = subnet.branch_layers[-1](h_u), subnet.trunk_layers[-1](h_x)
    if final_activation:
        b, t = torch.tanh(b), torch.tanh(t)
    return (b * t).sum(dim=1, keepdim=True)


def _linear_reference(subnet, branch, trunk):
    for layer in subnet.branch_layers:
        branch = layer(branch)
    for layer in subnet.trunk_layers:
        trunk = layer(trunk)
    return (branch * trunk).sum(dim=1, keepdim=True)


def _naive_subnets(model, coordinates):
    scaled = _scale(coordinates, model.scales)
    power, trunk = scaled[:, 3:4], scaled[:, [0, 1, 2, 4]]
    low = _modified_reference(model.low_fidelity_subnet, power, trunk, False)
    linear_rows, nonlinear_rows = [], []
    for row in coordinates:
        query_vectors = []
        for points in (model.linear_query_points, model.nonlinear_query_points):
            query = torch.cat((points[:, :3], row[3].expand(len(points), 1), points[:, 3:4]), dim=1)
            q_scaled = _scale(query, model.scales)
            query_vectors.append(_modified_reference(model.low_fidelity_subnet, q_scaled[:, 3:4],
                                                       q_scaled[:, [0, 1, 2, 4]], False).reshape(1, -1))
        current = _scale(row.reshape(1, 5), model.scales)
        current_trunk = current[:, [0, 1, 2, 4]]
        linear_rows.append(_linear_reference(model.linear_subnet, query_vectors[0], current_trunk))
        nonlinear_rows.append(_modified_reference(model.nonlinear_subnet,
                              torch.cat((current[:, 3:4], query_vectors[1]), dim=1), current_trunk, True))
    return {"low_fidelity": low, "linear": torch.cat(linear_rows), "nonlinear": torch.cat(nonlinear_rows)}


def _naive_high(model, coordinates):
    result = _naive_subnets(model, coordinates)
    return model.scales.temperature_offset_k + model.scales.temperature_scale_k * (
        result["linear"] + result["nonlinear"])


def _rzt_derivatives(output, coordinates):
    first = torch.autograd.grad(output.sum(), coordinates, create_graph=True)[0]
    second = torch.stack([torch.autograd.grad(first[:, axis].sum(), coordinates,
                                            create_graph=True, retain_graph=True)[0][:, axis]
                          for axis in range(3)], dim=1)
    return first[:, :3], second


def test_explicit_query_buffers_keep_shapes_values_and_point_order() -> None:
    model = _model()
    linear, nonlinear = _points()
    assert torch.equal(model.linear_query_points, linear)
    assert torch.equal(model.nonlinear_query_points, nonlinear)
    buffers = dict(model.named_buffers())
    assert {"linear_query_points", "nonlinear_query_points"} <= buffers.keys()
    assert not model.linear_query_points.requires_grad
    assert model.low_fidelity_subnet.branch_layers[0].in_features == 1
    assert model.low_fidelity_subnet.trunk_layers[0].in_features == 4
    assert model.linear_subnet.branch_layers[0].in_features == len(linear)
    assert model.nonlinear_subnet.branch_layers[0].in_features == 1 + len(nonlinear)


@pytest.mark.parametrize("name,final_activation", [("low_fidelity_subnet", False), ("nonlinear_subnet", True)])
def test_modified_subnets_follow_equations_two_to_six(name, final_activation) -> None:
    model = _model()
    subnet = getattr(model, name)
    branch = torch.linspace(-0.8, 0.7, 2 * subnet.branch_layers[0].in_features,
                            dtype=torch.float64).reshape(2, -1)
    trunk = torch.tensor([[-0.6, 0.4, 0.2, -1.0], [0.9, -0.3, -0.5, 1.0]], dtype=torch.float64)
    torch.testing.assert_close(subnet(branch, trunk), _modified_reference(subnet, branch, trunk, final_activation),
                               rtol=1e-13, atol=1e-13)


def test_linear_subnet_has_only_affine_layers_and_raw_inner_product() -> None:
    model = _model()
    subnet = model.linear_subnet
    assert all(isinstance(layer, nn.Linear) for layer in (*subnet.branch_layers, *subnet.trunk_layers))
    branch = torch.tensor([[-0.8, 0.2, 0.9]], dtype=torch.float64)
    trunk = torch.tensor([[0.4, -0.9, 0.2, -1.0]], dtype=torch.float64)
    torch.testing.assert_close(subnet(branch, trunk), _linear_reference(subnet, branch, trunk), rtol=0, atol=0)


def test_high_is_linear_plus_nonlinear_with_one_exact_temperature_offset() -> None:
    model = _model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.low_fidelity_subnet.branch_layers[-1].bias.fill_(1.0)
        model.low_fidelity_subnet.trunk_layers[-1].bias.fill_(1.0)
        model.linear_subnet.branch_layers[-1].bias.fill_(1.0)
        model.linear_subnet.trunk_layers[-1].bias.fill_(2.0)
        bias = torch.atanh(torch.tensor(0.5, dtype=torch.float64))
        model.nonlinear_subnet.branch_layers[-1].bias.fill_(bias)
        model.nonlinear_subnet.trunk_layers[-1].bias.fill_(bias)
    coordinates = _coordinates()
    coordinates[:, 2:4] = 0.0
    parts = model.subnet_outputs(coordinates)
    assert torch.equal(parts["low_fidelity"], torch.full((5, 1), 5.0, dtype=torch.float64))
    expected = torch.full((5, 1), 295.15 + 250 * (10.0 + 1.25), dtype=torch.float64)
    torch.testing.assert_close(model(coordinates, fidelity="high"), expected, rtol=0, atol=1e-12)
    assert torch.equal(model(coordinates, fidelity="low"), torch.full((5, 1), 295.15 + 250 * 5,
                                                                    dtype=torch.float64))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    assert torch.equal(model(coordinates), torch.full((5, 1), 295.15, dtype=torch.float64))


def test_grouped_live_lf_queries_equal_every_row_naive_outputs() -> None:
    model = _model()
    coordinates = _coordinates()
    expected = _naive_subnets(model, coordinates)
    actual = model.subnet_outputs(coordinates)
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(model(coordinates), _naive_high(model, coordinates), rtol=2e-12, atol=2e-12)


def test_grouped_queries_keep_first_and_second_rzt_derivatives() -> None:
    model = _model()
    coordinates = _coordinates().requires_grad_(True)
    reference_coordinates = coordinates.detach().clone().requires_grad_(True)
    actual = _rzt_derivatives(model(coordinates), coordinates)
    expected = _rzt_derivatives(_naive_high(model, reference_coordinates), reference_coordinates)
    for first, second in zip(actual, expected):
        torch.testing.assert_close(first, second, rtol=2e-10, atol=2e-10)


def test_grouped_queries_keep_hf_and_derivative_loss_gradients_for_all_subnets() -> None:
    model = _model()
    reference = copy.deepcopy(model)
    coordinates = _coordinates().requires_grad_(True)
    reference_coordinates = coordinates.detach().clone().requires_grad_(True)
    actual_output, expected_output = model(coordinates), _naive_high(reference, reference_coordinates)
    actual_derivatives = _rzt_derivatives(actual_output, coordinates)
    expected_derivatives = _rzt_derivatives(expected_output, reference_coordinates)
    actual_loss = actual_output.square().mean() + sum(item.square().mean() for item in actual_derivatives)
    expected_loss = expected_output.square().mean() + sum(item.square().mean() for item in expected_derivatives)
    actual_gradients = torch.autograd.grad(actual_loss, tuple(model.parameters()))
    expected_gradients = torch.autograd.grad(expected_loss, tuple(reference.parameters()))
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-8)
    assert any(gradient.abs().sum() > 0 for (name, _), gradient in zip(model.named_parameters(), actual_gradients)
               if name.startswith("low_fidelity_subnet."))


def test_repeated_nonadjacent_power_row_permutation_keeps_values_and_rzt_graph() -> None:
    model = _model()
    coordinates = _coordinates().requires_grad_(True)
    permutation = torch.tensor([4, 2, 3, 0, 1])
    permuted = coordinates.detach()[permutation].requires_grad_(True)
    original = model(coordinates)
    reordered = model(permuted)
    torch.testing.assert_close(reordered, original[permutation], rtol=2e-12, atol=2e-12)
    for actual, expected in zip(_rzt_derivatives(reordered, permuted), _rzt_derivatives(original, coordinates)):
        torch.testing.assert_close(actual, expected[permutation], rtol=2e-10, atol=2e-10)


def test_query_point_relabeling_preserves_operator_when_branch_columns_are_reindexed() -> None:
    model = _model()
    relabeled = copy.deepcopy(model)
    linear_order, nonlinear_order = torch.tensor([2, 0, 1]), torch.tensor([3, 1, 0, 2])
    with torch.no_grad():
        relabeled.linear_query_points.copy_(model.linear_query_points[linear_order])
        relabeled.linear_subnet.branch_layers[0].weight.copy_(model.linear_subnet.branch_layers[0].weight[:, linear_order])
        relabeled.nonlinear_query_points.copy_(model.nonlinear_query_points[nonlinear_order])
        columns = torch.cat((torch.zeros(1, dtype=torch.long), nonlinear_order + 1))
        relabeled.nonlinear_subnet.branch_encoder.weight.copy_(model.nonlinear_subnet.branch_encoder.weight[:, columns])
        relabeled.nonlinear_subnet.branch_layers[0].weight.copy_(model.nonlinear_subnet.branch_layers[0].weight[:, columns])
    torch.testing.assert_close(relabeled(_coordinates()), model(_coordinates()), rtol=2e-12, atol=2e-12)


def test_branch_regularization_is_two_raw_sums_with_only_branch_and_branch_encoders() -> None:
    model = _model()
    regularization = model.branch_regularization()
    assert regularization.keys() == {"low_fidelity", "nonlinear"}
    for key, subnet in (("low_fidelity", model.low_fidelity_subnet), ("nonlinear", model.nonlinear_subnet)):
        parameters = [*subnet.branch_encoder.parameters(), *subnet.branch_layers.parameters()]
        expected = sum(parameter.square().sum() for parameter in parameters)
        torch.testing.assert_close(regularization[key], expected, rtol=0, atol=0)
    gradients = torch.autograd.grad(sum(regularization.values()), tuple(model.parameters()), allow_unused=True)
    for (name, parameter), gradient in zip(model.named_parameters(), gradients):
        regularized = name.startswith(("low_fidelity_subnet.branch_", "nonlinear_subnet.branch_"))
        if regularized:
            torch.testing.assert_close(gradient, 2 * parameter, rtol=0, atol=0)
        else:
            assert gradient is None


def test_model_kwargs_and_state_roundtrip_include_fixed_queries_and_scales() -> None:
    scales = ModelScales(temperature_offset_k=301.25, temperature_scale_k=200.0)
    model = _model(scales=scales)
    reconstructed = _model_class()(**model.model_kwargs).double()
    reconstructed.load_state_dict(model.state_dict(), strict=True)
    assert reconstructed.model_kwargs == model.model_kwargs
    assert reconstructed.scales == scales
    assert model.parameter_count() == parameter_count(model) > 0
    torch.testing.assert_close(reconstructed(_coordinates()), model(_coordinates()), rtol=0, atol=0)


def test_query_lists_and_constructor_kwargs_keep_double_precision_without_state_loading() -> None:
    model = _model()
    reconstructed = _model_class()(**model.model_kwargs).double()
    assert reconstructed.model_kwargs == model.model_kwargs
    assert torch.equal(reconstructed.linear_query_points, model.linear_query_points)
    assert torch.equal(reconstructed.nonlinear_query_points, model.nonlinear_query_points)


def test_default_width_depth_and_latent_have_explicit_paper_layer_meaning() -> None:
    linear, nonlinear = _points()
    model = _model_class()(linear_query_points=linear, nonlinear_query_points=nonlinear)
    assert model.width == model.depth * 32 == model.latent_dim == 128
    for subnet in (model.low_fidelity_subnet, model.linear_subnet, model.nonlinear_subnet):
        assert len(subnet.branch_layers) == len(subnet.trunk_layers) == 4
        assert subnet.branch_layers[-1].out_features == subnet.trunk_layers[-1].out_features == 128


@pytest.mark.parametrize("bad", [[], [0, 0, 1, 0], [[0, 0, 1, 0, 0]],
                                     [[0, 0, float("nan"), 0]], [[0, float("inf"), 1, 0]],
                                     [[0, 0, 1, -1]], [[0, 0, 1, 0.5]], [[0, 0, 1, 2]]])
@pytest.mark.parametrize("query_name", ["linear_query_points", "nonlinear_query_points"])
def test_query_buffers_reject_bad_shape_nonfinite_and_material(query_name, bad) -> None:
    linear, nonlinear = _points()
    arguments = {"linear_query_points": linear, "nonlinear_query_points": nonlinear}
    arguments[query_name] = bad
    with pytest.raises(ValueError, match="query|查询|形状|材料|有限"):
        _model_class()(**arguments)


@pytest.mark.parametrize("argument,value", [("width", 0), ("depth", 1), ("latent_dim", 0)])
def test_invalid_network_dimensions_fail_before_model_use(argument, value) -> None:
    linear, nonlinear = _points()
    with pytest.raises(ValueError, match="width|depth|latent|层|维"):
        _model_class()(linear_query_points=linear, nonlinear_query_points=nonlinear, **{argument: value})


@pytest.mark.parametrize("bad", [torch.zeros((2, 4), dtype=torch.float64),
                                     torch.tensor([[0, 0, 1, float("nan"), 0]], dtype=torch.float64),
                                     torch.tensor([[0, 0, 1, 100, 0.5]], dtype=torch.float64)])
def test_forward_rejects_bad_five_column_interface(bad) -> None:
    with pytest.raises(ValueError, match="五|5|材料|有限|coordinates"):
        _model()(bad)


def test_unknown_fidelity_is_rejected() -> None:
    with pytest.raises(ValueError, match="fidelity|保真"):
        _model()(_coordinates(), fidelity="other")


def test_query_chunk_size_default_and_constructor_kwargs_roundtrip() -> None:
    assert _model().query_chunk_size == 16384
    model = _model(query_chunk_size=2)
    assert model.query_chunk_size == model.model_kwargs["query_chunk_size"] == 2
    reconstructed = _model_class()(**model.model_kwargs).double()
    reconstructed.load_state_dict(model.state_dict(), strict=True)
    assert reconstructed.model_kwargs == model.model_kwargs
    torch.testing.assert_close(reconstructed(_coordinates()), model(_coordinates()), rtol=0, atol=0)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2", None])
def test_query_chunk_size_rejects_non_positive_integer(bad) -> None:
    with pytest.raises(ValueError, match="query_chunk_size|分块|正整数"):
        _model(query_chunk_size=bad)


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_chunked_queries_equal_naive_outputs_and_all_parameter_high_order_gradients(chunk_size) -> None:
    model = _model(query_chunk_size=chunk_size)
    reference = copy.deepcopy(model)
    coordinates = _coordinates().requires_grad_(True)
    reference_coordinates = coordinates.detach().clone().requires_grad_(True)
    actual_parts, expected_parts = model.subnet_outputs(coordinates), _naive_subnets(reference, reference_coordinates)
    for name in actual_parts:
        torch.testing.assert_close(actual_parts[name], expected_parts[name], rtol=2e-12, atol=2e-12)
    actual, expected = model(coordinates), _naive_high(reference, reference_coordinates)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    actual_derivatives = _rzt_derivatives(actual, coordinates)
    expected_derivatives = _rzt_derivatives(expected, reference_coordinates)
    for actual_derivative, expected_derivative in zip(actual_derivatives, expected_derivatives):
        torch.testing.assert_close(actual_derivative, expected_derivative, rtol=2e-10, atol=2e-10)
    actual_loss = actual.square().mean() + sum(item.square().mean() for item in actual_derivatives)
    expected_loss = expected.square().mean() + sum(item.square().mean() for item in expected_derivatives)
    actual_gradients = torch.autograd.grad(actual_loss, tuple(model.parameters()))
    expected_gradients = torch.autograd.grad(expected_loss, tuple(reference.parameters()))
    assert len(actual_gradients) == len(expected_gradients) == len(tuple(model.named_parameters()))
    for (name, _), actual_gradient, expected_gradient in zip(model.named_parameters(),
                                                            actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-10, atol=2e-8,
                                   msg=lambda message: f"{name}: {message}")
        assert actual_gradient.isfinite().all()
    for prefix in ("low_fidelity_subnet.", "linear_subnet.", "nonlinear_subnet."):
        assert any(gradient.abs().sum() > 0 for (name, _), gradient in zip(model.named_parameters(), actual_gradients)
                   if name.startswith(prefix))


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_chunked_queries_keep_complete_dense_rzt_jacobian_and_hessian(chunk_size) -> None:
    model = _model(query_chunk_size=chunk_size)
    coordinates = _coordinates()
    rzt = coordinates[:, :3].detach().clone().requires_grad_(True)

    def fields(positions, *, naive=False):
        current = torch.cat((positions, coordinates[:, 3:]), dim=1)
        return (_naive_high(model, current) if naive else model(current)).reshape(-1)

    def jacobian(positions, *, naive=False):
        return torch.autograd.functional.jacobian(lambda values: fields(values, naive=naive), positions,
                                                  create_graph=True)

    actual_jacobian, expected_jacobian = jacobian(rzt), jacobian(rzt, naive=True)
    actual_hessian = torch.autograd.functional.jacobian(jacobian, rzt)
    expected_hessian = torch.autograd.functional.jacobian(lambda values: jacobian(values, naive=True), rzt)
    assert actual_jacobian.shape == (5, 5, 3)
    assert actual_hessian.shape == (5, 5, 3, 5, 3)
    torch.testing.assert_close(actual_jacobian, expected_jacobian, rtol=2e-10, atol=2e-10)
    torch.testing.assert_close(actual_hessian, expected_hessian, rtol=2e-10, atol=2e-10)
    for output_row in range(5):
        for input_row in range(5):
            if output_row != input_row:
                assert torch.count_nonzero(actual_jacobian[output_row, input_row]) == 0
            for other_input_row in range(5):
                if output_row != input_row or output_row != other_input_row:
                    assert torch.count_nonzero(actual_hessian[output_row, input_row, :, other_input_row, :]) == 0


@pytest.mark.parametrize("chunk_size", [1, 2])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_query_blocks_obey_total_row_limit_and_real_nonreentrant_checkpoint(monkeypatch, chunk_size,
                                                                         grad_enabled) -> None:
    module = importlib.import_module("sic_cu.models.task11_howard_composite")
    model = _model(query_chunk_size=chunk_size)
    power = torch.tensor([[55.0], [115.2], [364.3], [403.0], [630.5], [729.0], [800.0]], dtype=torch.float64)
    checkpoint_calls, low_rows = [], []
    real_checkpoint, real_low = module.checkpoint, model._low_normalized

    def checkpoint_spy(function, *arguments, **kwargs):
        checkpoint_calls.append((len(arguments[0]), kwargs.get("use_reentrant")))
        return real_checkpoint(function, *arguments, **kwargs)

    def low_spy(current):
        low_rows.append(len(current))
        return real_low(current)

    monkeypatch.setattr(module, "checkpoint", checkpoint_spy)
    monkeypatch.setattr(model, "_low_normalized", low_spy)
    with torch.set_grad_enabled(grad_enabled):
        vectors = [model._query_low_fidelity(power, points)
                   for points in (model.linear_query_points, model.nonlinear_query_points)]
        if grad_enabled:
            gradients = torch.autograd.grad(sum(vector.square().sum() for vector in vectors),
                                            tuple(model.low_fidelity_subnet.parameters()))
            assert all(gradient.isfinite().all() for gradient in gradients)
            assert any(gradient.abs().sum() > 0 for gradient in gradients)
        else:
            assert all(not vector.requires_grad for vector in vectors)
    assert low_rows and max(low_rows) <= chunk_size
    expected_calls = sum((len(power) * len(points) + chunk_size - 1) // chunk_size
                         for points in (model.linear_query_points, model.nonlinear_query_points))
    if grad_enabled:
        assert len(checkpoint_calls) == expected_calls
        assert all(rows <= chunk_size and reentrant is False for rows, reentrant in checkpoint_calls)
    else:
        assert len(low_rows) == expected_calls
        assert checkpoint_calls == []
    for actual, points in zip(vectors, (model.linear_query_points, model.nonlinear_query_points)):
        expected_rows = []
        for current_power in power:
            query = torch.cat((points[:, :3], current_power.expand(len(points), 1), points[:, 3:4]), dim=1)
            expected_rows.append(real_low(query).reshape(-1))
        torch.testing.assert_close(actual, torch.stack(expected_rows), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_chunked_queries_preserve_anchor_response_and_lf_parameter_graph(chunk_size) -> None:
    model = _model(query_chunk_size=chunk_size)
    points = model.linear_query_points.detach().clone().requires_grad_(True)
    power = torch.tensor([[115.2], [403.0]], dtype=torch.float64, requires_grad=True)
    actual = model._query_low_fidelity(power, points)
    expected_rows = []
    for current_power in power:
        query = torch.cat((points[:, :3], current_power.expand(len(points), 1), points[:, 3:4]), dim=1)
        scaled = _scale(query, model.scales)
        expected_rows.append(_modified_reference(model.low_fidelity_subnet, scaled[:, 3:4],
                                                  scaled[:, [0, 1, 2, 4]], False).reshape(-1))
    expected = torch.stack(expected_rows)
    inputs = (points, power, *model.low_fidelity_subnet.parameters())
    actual_gradients = torch.autograd.grad(actual.square().sum(), inputs)
    expected_gradients = torch.autograd.grad(expected.square().sum(), inputs)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_chunked_queries_preserve_empty_batch_shape(chunk_size) -> None:
    model = _model(query_chunk_size=chunk_size)
    coordinates = torch.empty((0, 5), dtype=torch.float64)
    assert model(coordinates).shape == model(coordinates, fidelity="low").shape == (0, 1)
