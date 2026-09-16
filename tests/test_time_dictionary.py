from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

from sic_cu.data.fields import SimulationField
from sic_cu.data.splits import build_power_splits


def _dictionary():
    try:
        return importlib.import_module("sic_cu.data.time_dictionary")
    except ModuleNotFoundError as exc:
        pytest.fail(f"New time dictionary feature is missing: {exc}")


def _field(power: float, *, drift: bool = False) -> SimulationField:
    times = np.arange(0, 201, 2, dtype=np.float32)
    nodes = np.asarray([[0.1, 0.0], [0.4, 0.3], [0.2, 0.8], [0.5, 0.5],
                        [0.0, 0.0], [0.2, 0.3], [0.3, 0.6], [0.4, 0.9]], dtype=np.float32)
    if drift:
        nodes[0, 0] += 0.01
    material = np.asarray([0] * 4 + [1] * 4, dtype=np.int64)
    labels = np.asarray([1, 2, 3, 4] * 2, dtype=np.int64)
    ramp = -np.expm1(-times / 20.0)
    delta = ramp[:, None] * np.asarray([0.0, 0.05, -0.2, 1.0,
                                        0.4, -0.5, 0.8, 1.2])[None, :] * (power / 10.0)
    return SimulationField(
        power_w=power, times_s=times, coordinates_rz_m=nodes,
        material_ids=material, node_labels=labels,
        temperature_k=(295.15 + delta).astype(np.float32),
    )


def test_spatial_selection_excludes_copper_outer_and_uses_fixed_tie_order() -> None:
    dictionary = _dictionary()
    reference = _field(10.0)
    selected = dictionary.select_spatial_nodes(reference, per_material=2)
    assert selected.excluded_copper_count == 1
    assert len(selected.nodes) == 4
    assert {(node.material_id, node.node_label) for node in selected.nodes} == {
        (0, 1), (0, 3), (1, 1), (1, 3),
    }
    assert all(not (node.material_id == 0 and node.node_label == 4)
               for node in selected.nodes)
    assert dictionary.select_spatial_nodes(reference, per_material=2) == selected


def test_spatial_tie_prefers_smaller_node_label() -> None:
    dictionary = _dictionary()
    field = _field(10.0)
    coords = field.coordinates_rz_m.copy()
    coords[1] = (0.1, 0.0)
    coords[2] = (0.1, 0.0)
    tied = SimulationField(field.power_w, field.times_s, coords, field.material_ids,
                           field.node_labels, field.temperature_k)
    selected = dictionary.select_spatial_nodes(tied, per_material=2)
    cu_labels = [node.node_label for node in selected.nodes if node.material_id == 0]
    assert 2 in cu_labels
    assert 3 not in cu_labels


def test_curve_matrix_subtracts_each_curve_at_t0_and_protects_small_amplitudes() -> None:
    dictionary = _dictionary()
    fields = [_field(10.0), _field(20.0)]
    selection = dictionary.select_spatial_nodes(fields[0], per_material=2)
    matrix = dictionary.build_curve_matrix(fields, selection)
    assert matrix.temperature_deltas_k.shape == (8, 100)
    assert np.array_equal(matrix.times_s, np.arange(2, 201, 2))
    assert np.isfinite(matrix.normalized_deltas).all()
    assert np.min(matrix.scales_k) >= 0.1
    assert matrix.power_w.tolist() == [10.0] * 4 + [20.0] * 4
    assert matrix.material_ids.tolist() == [0, 0, 1, 1] * 2
    assert np.max(matrix.initial_abs_delta_k) == 0.0


def test_curve_matrix_rejects_other_mesh_without_refitting_selection() -> None:
    dictionary = _dictionary()
    selected = dictionary.select_spatial_nodes(_field(10.0), per_material=2)
    with pytest.raises(ValueError, match="mesh|coordinate"):
        dictionary.build_curve_matrix([_field(10.0), _field(20.0, drift=True)], selected)


def test_only_exact_sixty_simulation_training_powers_may_fit() -> None:
    dictionary = _dictionary()
    train = sorted(build_power_splits().simulation_train)
    assert dictionary.validate_training_power_contract(train) == train
    with pytest.raises(ValueError, match="60|training"):
        dictionary.validate_training_power_contract(train[:-1])
    with pytest.raises(ValueError, match="60|training"):
        dictionary.validate_training_power_contract(train[:-1] + [90.0])


def test_two_registered_starts_fit_ordered_positive_tau_with_signed_amplitudes() -> None:
    dictionary = _dictionary()
    time = np.arange(2, 201, 2, dtype=np.float64)
    true_tau = np.asarray([2, 10, 50, 200], dtype=np.float64)
    response = -np.expm1(-time[:, None] / true_tau[None, :])
    amplitude = np.asarray([[1.1, -0.4, 0.3, 0.2], [0.1, 0.7, -0.6, 0.9],
                            [-0.5, 0.4, 0.8, -0.3], [0.6, -0.8, 0.4, 0.6]])
    curves = amplitude @ response.T
    result = dictionary.fit_time_dictionary(curves, time, initial_groups=((2, 10, 50, 200),
                                                                           (1, 5, 25, 100)))
    assert len(result.candidates) == 2
    for candidate in result.candidates:
        assert candidate.tau_seconds.shape == (4,)
        assert np.all(np.diff(candidate.tau_seconds) > 0)
        assert np.all(candidate.tau_seconds > 0)
        assert np.isfinite(candidate.training_normalized_rmse)
        assert candidate.training_normalized_rmse < 0.07
    assert np.any(result.training_amplitudes < 0)


def test_fixed_validation_projection_cannot_change_common_tau() -> None:
    dictionary = _dictionary()
    times = np.arange(2, 201, 2, dtype=np.float64)
    tau = np.asarray([2, 10, 50, 200], dtype=np.float64)
    reference = tau.copy()
    curves = -np.expm1(-times[None, :] / 10.0).repeat(3, axis=0)
    report = dictionary.project_fixed_dictionary(curves, times, tau)
    assert np.array_equal(tau, reference)
    assert report["normalized_rmse"] >= 0.0
    assert report["curve_count"] == 3
    assert np.isfinite(report["feature_condition_number"])


def test_torch_features_are_buffers_and_keep_first_and_second_time_derivatives() -> None:
    dictionary = _dictionary()
    features = dictionary.TimeResponseFeatures([2, 10, 50, 200]).double()
    assert "tau_seconds" in dict(features.named_buffers())
    assert not list(features.parameters())
    t = torch.tensor([[0.0], [20.0]], dtype=torch.float64, requires_grad=True)
    value = features(t)
    assert value.shape == (2, 4)
    assert torch.all(value[0] == 0)
    first = torch.autograd.grad(value[:, 0].sum(), t, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), t)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert float(first[0, 0].detach()) == pytest.approx(0.5)
    assert float(second[0, 0].detach()) == pytest.approx(-0.25)


def _entry_script():
    path = Path(__file__).resolve().parents[1] / "scripts/20_build_e2_dictionary.py"
    assert path.exists(), "Registered LF TRAIN dictionary preparation entry is missing"
    spec = importlib.util.spec_from_file_location("task06_dictionary_entry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_rejects_changed_registered_second_start() -> None:
    entry = _entry_script()
    registration = Path(__file__).resolve().parents[1] / (
        "研究记录/任务06_时间响应特征/字典拟合前登记.yaml"
    )
    policy = yaml.safe_load(registration.read_text(encoding="utf-8"))
    assert entry.validate_registration(policy) == ((2.0, 10.0, 50.0, 200.0),
                                                    (1.0, 5.0, 25.0, 100.0))
    policy["dictionary"]["tau_seconds_initial_group_2"] = [3, 5, 25, 100]
    with pytest.raises(ValueError, match="registered|初值|initial"):
        entry.validate_registration(policy)


def test_script_refuses_existing_output_before_reading_any_trajectory(tmp_path) -> None:
    entry = _entry_script()
    output = tmp_path / "older_result"
    output.mkdir()
    sentinel = output / "existing_data"
    sentinel.write_text("user", encoding="utf-8")
    with pytest.raises(FileExistsError, match="exist|已有|覆盖"):
        entry.create_fresh_output(output)
    assert sentinel.read_text(encoding="utf-8") == "user"


def _assert_registration_refused_before_output_and_lf_read(
    entry, registration: Path, output: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["task06_dictionary_entry", "--registration", str(registration),
                                     "--output", str(output)])

    def forbidden_lf_read(*_args, **_kwargs):
        pytest.fail("LF source was accessed before registration provenance passed")

    monkeypatch.setattr(entry, "_source_fields", forbidden_lf_read)
    with pytest.raises(ValueError, match="预登记|registered|registration|SHA|path"):
        entry.main()
    assert not output.exists(), "An invalid registration must not create any result directory"


def test_identical_registration_bytes_at_a_different_path_are_refused_first(tmp_path, monkeypatch) -> None:
    entry = _entry_script()
    alternate = tmp_path / "identical_registration.yaml"
    alternate.write_bytes(entry.REGISTRATION.read_bytes())
    _assert_registration_refused_before_output_and_lf_read(
        entry, alternate, tmp_path / "never_created_alias", monkeypatch,
    )


@pytest.mark.parametrize("section,field,replacement", [
    ("node_selection", "coordinate_rule", "允许依据LF验证节点另行选点"),
    ("curve_weighting", "amplitude_scale_k", "各曲线直接按峰值除，无0.1K下限"),
    ("curve_weighting", "time_points", "只拟合LF验证时间帧"),
    ("dictionary", "validation", "允许对LF验证曲线共同tau回传梯度"),
])
def test_any_unchecked_semantic_change_is_refused_before_output_and_lf_read(
    section, field, replacement, tmp_path, monkeypatch,
) -> None:
    entry = _entry_script()
    policy = yaml.safe_load(entry.REGISTRATION.read_text(encoding="utf-8"))
    policy[section][field] = replacement
    temporary_canonical = tmp_path / "registered_pre_fit.yaml"
    temporary_canonical.write_text(yaml.safe_dump(policy, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(entry, "REGISTRATION", temporary_canonical)
    _assert_registration_refused_before_output_and_lf_read(
        entry, temporary_canonical, tmp_path / "never_created_drift", monkeypatch,
    )
