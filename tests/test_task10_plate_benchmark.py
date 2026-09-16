"""Task 10 one-dimensional synthetic benchmark entry and isolation gates."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
import json
from pathlib import Path
import runpy
import sys
from dataclasses import replace

import numpy as np
import pytest

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file


REGISTRATION = (PROJECT_ROOT / "研究记录/任务10_独立双层场基准/"
                "正式人为多热流入场前登记.yaml")


def _module_with(name: str):
    module = import_module("sic_cu.eval.task10_plate_benchmark")
    assert hasattr(module, name), f"任10独立模块尚无{name}入口"
    return module


def test_task10_plate_benchmark_is_an_independent_module() -> None:
    assert find_spec("sic_cu.eval.task10_plate_benchmark") is not None


def test_load_registration_requires_the_original_immutable_source_hashes() -> None:
    module = _module_with("load_benchmark_registration")
    config = module.load_benchmark_registration(REGISTRATION)
    assert config["flux_splits_w_m2"]["train"] == [20000, 35000, 50000, 65000, 80000]
    assert config["flux_splits_w_m2"]["validation"] == [30000, 70000]
    assert config["flux_splits_w_m2"]["hidden_test"] == [42000, 58000]
    assert config["thermal_conditions"]["coarse_mismatch_contact_multiplier"] == 0.7


def test_old_numerical_control_hash_change_rejects_the_new_case(tmp_path: Path) -> None:
    module = _module_with("load_benchmark_registration")
    tampered = REGISTRATION.read_text(encoding="utf-8").replace(
        "3011eba0cb6cd1c4", "0000000000000000", 1,
    )
    copy = tmp_path / "伪造旧控制哈希.yaml"
    copy.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        module.load_benchmark_registration(copy)


def test_registered_50k_plate_has_independent_steady_and_dimensional_budget() -> None:
    module = _module_with("solve_registered_plate")
    case = module.solve_registered_plate(
        REGISTRATION, 50000, sic_cells=24, cu_cells=11,
        dt_s=2.0, end_s=200.0,
    )
    analytic_top = 295.15 + 50000 * (0.012 / 120 + 0.0001 + 0.0055 / 401)
    assert case.temperature_k.shape == (101, 35)
    assert case.material_id.shape == (35,)
    assert case.material_id.sum() == 24
    assert abs(case.top_surface_temperature_k[-1] - analytic_top) < 1e-5
    assert abs(case.interface_flux_w_m2[-1] - 50000) < 1e-5
    assert abs(case.interface_temperature_jump_k[-1] - 5.0) < 1e-5
    assert np.max(np.abs(case.balance_per_area_w_m2[1:])) < 1e-6


def test_registered_solver_rejects_unspecified_flux_and_contact_multiplier() -> None:
    module = _module_with("solve_registered_plate")
    with pytest.raises(ValueError, match="registered|登记"):
        module.solve_registered_plate(
            REGISTRATION, 50123, sic_cells=24, cu_cells=11,
            dt_s=2.0, end_s=2.0,
        )
    with pytest.raises(ValueError, match="contact|接触"):
        module.solve_registered_plate(
            REGISTRATION, 50000, sic_cells=24, cu_cells=11,
            dt_s=2.0, end_s=2.0, contact_multiplier=0.9,
        )


def test_50k_reference_refinement_keeps_original_limit_and_times() -> None:
    module = _module_with("check_registered_reference")
    result = module.check_registered_reference(REGISTRATION, 50000)
    assert result["fine_grid"] == (192, 88, 0.25)
    assert abs(result["max_adjacent_difference_c"] - 0.09753328788178806) < 1e-6
    assert result["limit_c"] == 0.1
    assert result["comparison_times_s"] == [1, 2, 5, 10, 50, 100, 200]


def test_reference_failing_max_difference_uses_only_registered_next_level() -> None:
    module = _module_with("choose_reference_grid")
    assert module.choose_reference_grid(REGISTRATION, [0.13, 0.08]) == (384, 176, 0.125)
    with pytest.raises(ValueError, match="网格|reference|参考"):
        module.choose_reference_grid(REGISTRATION, [0.15, 0.11, 0.101])


def test_50k_observation_export_contains_three_sites_not_hidden_full_field() -> None:
    module = _module_with("registered_probe_snapshot")
    fine = module.solve_registered_plate(
        REGISTRATION, 50000, sic_cells=192, cu_cells=88, dt_s=0.25, end_s=200.0,
    )
    snapshot = module.registered_probe_snapshot(REGISTRATION, 50000, "train", fine)
    assert set(snapshot) == {"time_s", "depths_from_top_m", "flux_w_m2", "temperature_k"}
    assert snapshot["temperature_k"].shape == (14, 3)
    assert snapshot["depths_from_top_m"].tolist() == [0.0, 0.013, 0.016]
    assert abs(snapshot["temperature_k"][-1, 0] - 305.83578553613355) < 1e-5
    assert not np.shares_memory(snapshot["temperature_k"], fine.temperature_k)


def _index(root: Path, *paths: Path) -> None:
    index = {
        "登记SHA256": sha256_file(REGISTRATION),
        "归档文件_SHA256": {str(path.relative_to(root)): sha256_file(path) for path in paths},
    }
    (root / "探针与源场SHA清单.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8",
    )


def test_model_method_attempt_to_read_full_hf_during_training_is_rejected(tmp_path: Path) -> None:
    module = _module_with("Task10ObservationView")
    training = module.Task10ObservationView(REGISTRATION, tmp_path, "train")
    with pytest.raises(PermissionError, match="完整|full|hidden"):
        training.open_input("full_hf", 50000)
    with pytest.raises(PermissionError, match="验证|validation"):
        training.open_input("probe", 30000)


def test_validation_view_only_opens_validation_probe_fields(tmp_path: Path) -> None:
    module = _module_with("Task10ObservationView")
    allowed = tmp_path / "HF_允许探针/验证_30000.npz"
    allowed.parent.mkdir(parents=True)
    times = np.array([0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200])
    np.savez_compressed(allowed, time_s=times,
                        depths_from_top_m=np.array([0.0, 0.013, 0.016]),
                        flux_w_m2=np.array(30000), temperature_k=np.full((14, 3), 295.15))
    _index(tmp_path, allowed)
    validation = module.Task10ObservationView(REGISTRATION, tmp_path, "validation")
    snapshot = validation.open_input("probe", 30000)
    assert set(snapshot) == {"time_s", "depths_from_top_m", "flux_w_m2", "temperature_k"}
    assert snapshot["temperature_k"].shape == (14, 3)
    with pytest.raises(PermissionError, match="训练|train"):
        validation.open_input("probe", 50000)
    with pytest.raises(PermissionError, match="LF|低保真"):
        validation.open_input("lf_same_physics_coarse", 30000)


def test_probe_archive_with_extra_internal_hf_field_is_rejected(tmp_path: Path) -> None:
    module = _module_with("Task10ObservationView")
    poisoned = tmp_path / "HF_允许探针/训练_50000.npz"
    poisoned.parent.mkdir(parents=True)
    np.savez_compressed(
        poisoned, time_s=np.array([0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200]),
        depths_from_top_m=np.array([0.0, 0.013, 0.016]), flux_w_m2=np.array(50000),
        temperature_k=np.full((14, 3), 295.15),
        full_internal_hf_temperature_k=np.full((14, 280), 1000.0),
    )
    _index(tmp_path, poisoned)
    training = module.Task10ObservationView(REGISTRATION, tmp_path, "train")
    with pytest.raises(ValueError, match="完整|field|字段"):
        training.open_input("probe", 50000)


def test_hidden_reference_rejects_access_before_model_and_validation_lock(tmp_path: Path) -> None:
    module = _module_with("Task10HiddenEvaluator")
    hidden = tmp_path / "HF_封存完整场/50000.npz"
    hidden.parent.mkdir(parents=True)
    hidden.write_bytes(b"deliberately not an npz archive")
    evaluator = module.Task10HiddenEvaluator(REGISTRATION, tmp_path,
                                              tmp_path / "not_yet_locked.json")
    with pytest.raises(PermissionError, match="锁|SHA|lock"):
        evaluator.open_full_reference(50000)


def test_model_lock_rejects_missing_real_optimizer_updates(tmp_path: Path) -> None:
    module = _module_with("seal_evaluation_model")
    model = tmp_path / "new_plate_model.pt"
    model.write_bytes(b"test-only checkpoint placeholder")
    train = tmp_path / "train.json"
    train.write_text(json.dumps({"实际优化步数": 0,
                                 "训练折": [20000, 35000, 50000, 65000, 80000]}), encoding="utf-8")
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps({"合法验证仅用三探针": True,
                                      "验证折": [30000, 70000]}), encoding="utf-8")
    with pytest.raises(ValueError, match="优化|训练|step"):
        module.seal_evaluation_model(REGISTRATION, tmp_path, model, train,
                                     validation, tmp_path / "model_lock.json")


def test_fine_hf_grid_disguised_as_low_fidelity_is_rejected(tmp_path: Path) -> None:
    module = _module_with("Task10ObservationView")
    fake_lf = tmp_path / "LF_训练场/同物理_50000.npz"
    fake_lf.parent.mkdir(parents=True)
    scalar_times = np.linspace(0, 200, 801)
    np.savez_compressed(
        fake_lf, time_s=scalar_times, node_depth_m=np.linspace(0, 0.0175, 280),
        material_id=np.r_[np.ones(192), np.zeros(88)], temperature_k=np.full((801, 280), 295.15),
        top_surface_temperature_k=np.full(801, 295.15), bottom_outward_flux_w_m2=np.zeros(801),
        storage_rate_per_area_w_m2=np.zeros(801), balance_per_area_w_m2=np.zeros(801),
        interface_flux_w_m2=np.zeros(801), interface_temperature_jump_k=np.zeros(801),
    )
    _index(tmp_path, fake_lf)
    training = module.Task10ObservationView(REGISTRATION, tmp_path, "train")
    with pytest.raises(ValueError, match="粗网格|coarse|LF|场形状"):
        training.open_input("lf_same_physics_coarse", 50000)


def test_forged_incomplete_lock_never_releases_full_hf(tmp_path: Path) -> None:
    module = _module_with("Task10HiddenEvaluator")
    lock = tmp_path / "forged_lock.json"
    lock.write_text("{}", encoding="utf-8")
    evaluator = module.Task10HiddenEvaluator(REGISTRATION, tmp_path, lock)
    with pytest.raises(PermissionError, match="锁|SHA|lock"):
        evaluator.open_full_reference(50000)


def test_task10_export_cli_is_independent_from_frozen_script22() -> None:
    assert (PROJECT_ROOT / "scripts/任务10_检查人为多热流入场.py").is_file()


def _invoke_cli(monkeypatch: pytest.MonkeyPatch, output: Path, source: Path,
                source_sha: str) -> None:
    script = PROJECT_ROOT / "scripts/任务10_检查人为多热流入场.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(output),
                                       "--source-archive", str(source),
                                       "--source-sha256", source_sha])
    runpy.run_path(str(script), run_name="__main__")


def test_export_rejects_unfrozen_source_sha_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "not_frozen.tar.gz"
    archive.write_bytes(b"not an approved source archive")
    output = tmp_path / "不得出现的新数值档案"
    with pytest.raises(ValueError, match="SHA"):
        _invoke_cli(monkeypatch, output, archive, "0" * 64)
    assert not output.exists()


def test_export_rejects_output_target_outside_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "local_source.tar.gz"
    archive.write_bytes(b"synthetic fixture")
    outside = Path("/tmp/任10禁止写项目外_单元测试不创建")
    with pytest.raises(ValueError, match="项目|project"):
        _invoke_cli(monkeypatch, outside, archive, sha256_file(archive))
    assert not outside.exists()


def test_export_never_overwrites_an_existing_accepted_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "local_source.tar.gz"
    archive.write_bytes(b"synthetic fixture")
    existing = tmp_path / "原数值原件不得覆盖"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        _invoke_cli(monkeypatch, existing, archive, sha256_file(archive))


def test_registered_probe_rejects_mislabeled_reference_heat_flux() -> None:
    module = _module_with("registered_probe_snapshot")
    fine = module.solve_registered_plate(
        REGISTRATION, 50000, sic_cells=192, cu_cells=88, dt_s=0.25, end_s=200.0,
    )
    wrong_source = replace(fine,
                           top_surface_temperature_k=fine.top_surface_temperature_k + 2.0)
    with pytest.raises(ValueError, match="热流|source|source|源"):
        module.registered_probe_snapshot(REGISTRATION, 50000, "train", wrong_source)


def test_training_data_view_rejects_external_archive_root_without_reading(tmp_path: Path) -> None:
    module = _module_with("Task10ObservationView")
    outside = Path("/tmp/任10不读项目外观察档案_测试不创建")
    with pytest.raises(ValueError, match="项目|project"):
        module.Task10ObservationView(REGISTRATION, outside, "train")


def _unit_only_sealed_model(root: Path, reference: Path, control: dict | None) -> Path:
    _index(root, reference)
    index_path = root / "探针与源场SHA清单.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if control is not None:
        index["逐热流数值控制"] = control
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    model = root / "非正式单元测试模型.pt"
    model.write_bytes(b"only a unit fixture, never a trained model")
    training = root / "非正式单元测试训练.json"
    validation = root / "非正式单元测试验证.json"
    common = {"登记SHA256": sha256_file(REGISTRATION), "模型_SHA256": sha256_file(model)}
    training.write_text(json.dumps({**common, "实际优化步数": 1,
                                    "训练折": [20000, 35000, 50000, 65000, 80000]}), encoding="utf-8")
    validation.write_text(json.dumps({**common, "合法验证仅用三探针": True,
                                      "验证折": [30000, 70000]}), encoding="utf-8")
    module = _module_with("seal_evaluation_model")
    lock = root / "非正式单元测试模型锁.json"
    module.seal_evaluation_model(REGISTRATION, root, model, training, validation, lock)
    return lock


def _placeholder_full_archive(path: Path, steps: int, sic_cells: int, cu_cells: int) -> None:
    path.parent.mkdir(parents=True)
    cells = sic_cells + cu_cells
    np.savez_compressed(
        path, time_s=np.linspace(0.0, 200.0, steps),
        node_depth_m=np.linspace(0.0001, 0.0174, cells),
        material_id=np.r_[np.ones(sic_cells), np.zeros(cu_cells)],
        temperature_k=np.full((steps, cells), 295.15),
        top_surface_temperature_k=np.full(steps, 295.15),
        bottom_outward_flux_w_m2=np.zeros(steps), storage_rate_per_area_w_m2=np.zeros(steps),
        balance_per_area_w_m2=np.zeros(steps), interface_flux_w_m2=np.zeros(steps),
        interface_temperature_jump_k=np.zeros(steps),
    )


def test_model_lock_does_not_authorize_nonregistered_hf_grid_shape(tmp_path: Path) -> None:
    module = _module_with("Task10HiddenEvaluator")
    reference = tmp_path / "HF_封存完整场/50000.npz"
    _placeholder_full_archive(reference, 2, 2, 1)
    lock = _unit_only_sealed_model(
        tmp_path, reference,
        {"50000": {"细网格": [192, 88, 0.25], "相邻差_摄氏度": 0.09753328788178806}},
    )
    evaluator = module.Task10HiddenEvaluator(REGISTRATION, tmp_path, lock)
    with pytest.raises(ValueError, match="网格|grid|reference|参考"):
        evaluator.open_full_reference(50000)


def test_hidden_test_flux_needs_its_own_refinement_control_after_model_lock(tmp_path: Path) -> None:
    module = _module_with("Task10HiddenEvaluator")
    reference = tmp_path / "HF_封存完整场/42000.npz"
    _placeholder_full_archive(reference, 801, 192, 88)
    lock = _unit_only_sealed_model(tmp_path, reference, None)
    evaluator = module.Task10HiddenEvaluator(REGISTRATION, tmp_path, lock)
    with pytest.raises(PermissionError, match="数值|网格|reference|门禁"):
        evaluator.open_full_reference(42000)


def test_all_registered_fluxes_need_an_independent_numerical_factory() -> None:
    assert (PROJECT_ROOT / "scripts/任务10_生成多热流受限数值源.py").is_file()


def _invoke_factory(monkeypatch: pytest.MonkeyPatch, output: Path, source: Path,
                    source_sha: str) -> None:
    script = PROJECT_ROOT / "scripts/任务10_生成多热流受限数值源.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(output),
                                       "--source-archive", str(source),
                                       "--source-sha256", source_sha])
    runpy.run_path(str(script), run_name="__main__")


def test_multi_factory_unauthorized_tar_sha_never_opens_registered_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "并非前登记源.tar.gz"
    archive.write_bytes(b"not signed source")
    output = tmp_path / "不得出现的任何多热流HF原件"
    with pytest.raises(ValueError, match="SHA"):
        _invoke_factory(monkeypatch, output, archive, "0" * 64)
    assert not output.exists()


def test_multi_factory_refuses_project_external_target_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "本地测试占位归档.tar.gz"
    archive.write_bytes(b"only unit fixture")
    outside = Path("/tmp/任10多热流厂禁止外部写_测试不创建")
    with pytest.raises(ValueError, match="项目|project"):
        _invoke_factory(monkeypatch, outside, archive, sha256_file(archive))
    assert not outside.exists()


def test_multi_factory_rejects_previous_user_output_before_source_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "本地测试占位归档.tar.gz"
    archive.write_bytes(b"only unit fixture")
    output = tmp_path / "已存在的多热流数值原件"
    output.mkdir()
    with pytest.raises(FileExistsError):
        _invoke_factory(monkeypatch, output, archive, sha256_file(archive))
