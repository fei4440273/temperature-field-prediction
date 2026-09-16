"""Independent post-training metrics use synthetic arrays until the group is sealed."""

import json
import importlib.util

import numpy as np
import pytest
import torch

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from test_task10_plate_group_gate import synthetic_25_locked_boards


def _synthetic_board():
    widths = np.r_[np.full(2, .012 / 2), np.full(8, .0055 / 8)]
    material = np.r_[np.ones(2, dtype=np.int8), np.zeros(8, dtype=np.int8)]
    reference = {
        "time_s": np.arange(201, dtype=np.float64),
        "node_depth_m": np.cumsum(widths) - widths / 2,
        "material_id": material,
        "temperature_k": np.full((201, len(widths)), 295.15),
    }
    setup = {
        "geometry": {"silicon_carbide_thickness_m": .012,
                     "copper_thickness_m": .0055},
        "observations": {
            "allowed_depths_from_top_m": [0.0, .013, .016],
            "train_times_s": [0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200],
            "validation_times_s": [0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200],
            "nonoverlap_time_windows_s": ["[0,30]", "(30,100]", "(100,200]"],
        },
    }
    return reference, setup


def test_true_material_widths_do_not_score_cell_count_as_equal_volume():
    from sic_cu.eval.task10_plate_posthoc import compute_plate_field_metrics

    reference, setup = _synthetic_board()
    expected = np.where(reference["material_id"] == 1, 2.0, 4.0)
    prediction = reference["temperature_k"] + expected[None, :]
    metrics = compute_plate_field_metrics(reference, prediction, setup)

    assert metrics["SiC真实厚度加权RMSE_K"] == pytest.approx(2.0)
    assert metrics["Cu真实厚度加权RMSE_K"] == pytest.approx(4.0)
    assert metrics["双材料真实厚度加权RMSE_K"] == pytest.approx(
        np.sqrt((.012 * 2.0**2 + .0055 * 4.0**2) / .0175))
    assert metrics["仅非探针近邻体内RMSE_K"] == pytest.approx(
        np.sqrt((.012 * 2.0**2 + 5 * (.0055 / 8) * 4.0**2) /
                (.012 + 5 * (.0055 / 8))))
    assert metrics["非探针近邻单元数"] == 7


def test_probes_and_observed_times_do_not_leak_into_unseen_window_scores():
    from sic_cu.eval.task10_plate_posthoc import compute_plate_field_metrics

    reference, setup = _synthetic_board()
    depths = reference["node_depth_m"]
    near_probe = np.minimum(np.abs(depths - .013), np.abs(depths - .016)) <= .0005
    prediction = reference["temperature_k"] + np.where(
        reference["material_id"] == 1, 2.0, 4.0)[None, :]
    prediction[:, near_probe] += 100.0
    observed = np.array(setup["observations"]["train_times_s"])
    prediction[np.isin(reference["time_s"], observed), :] += 200.0
    metrics = compute_plate_field_metrics(reference, prediction, setup)

    expected_unseen = np.sqrt((.012 * 2.0**2 + 5 * (.0055 / 8) * 4.0**2) /
                              (.012 + 5 * (.0055 / 8)))
    assert metrics["双材料真实厚度加权RMSE_K"] > expected_unseen
    assert metrics["仅非探针近邻体内RMSE_K"] > expected_unseen
    assert set(metrics["非探针且非观测时刻逐窗RMSE_K"]) == {
        "[0,30]", "(30,100]", "(100,200]"}
    assert all(value == pytest.approx(expected_unseen)
               for value in metrics["非探针且非观测时刻逐窗RMSE_K"].values())


def test_wrong_reference_centers_or_fold_times_cannot_look_like_a_fine_grid():
    from sic_cu.eval.task10_plate_posthoc import compute_plate_field_metrics

    reference, setup = _synthetic_board()
    wrong = {name: value.copy() for name, value in reference.items()}
    wrong["node_depth_m"][3] += .001
    with pytest.raises(ValueError, match="center|单元中心|中心"):
        compute_plate_field_metrics(wrong, reference["temperature_k"], setup)
    wrong = {name: value.copy() for name, value in reference.items()}
    wrong["time_s"][4] += .1
    with pytest.raises(ValueError, match="time|时间|时刻"):
        compute_plate_field_metrics(wrong, reference["temperature_k"], setup)


class _AnalyticSteadyBoard(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.zero = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, coordinates):
        depth, _when, flux, sic = coordinates.unbind(1)
        si = 295.15 + flux * ((.012 - depth) / 120.0 + .0001 + .0055 / 401.0)
        cu = 295.15 + flux * ((.0175 - depth) / 401.0)
        return torch.where(sic > .5, si, cu)[:, None] + self.zero * 0


def test_steady_registered_board_has_flux_and_interface_jump_but_zero_storage():
    from sic_cu.eval.task10_plate_posthoc import audit_plate_physics

    reference, setup = _synthetic_board()
    setup["geometry"]["cross_section_area_m2"] = .0028274333882308137
    setup["thermal_conditions"] = {
        "benchmark_contact_resistance_m2_k_w": .0001,
        "initial_temperature_k": 295.15,
        "bottom_fixed_temperature_k": 295.15,
    }
    setup["materials_source"] = "configs/materials.yaml"
    setup["reference_controls"] = {"comparison_times_s": [1, 2, 5, 10, 50, 100, 200]}
    result = audit_plate_physics(_AnalyticSteadyBoard(), reference, setup,
                                 flux_w_m2=42000, device=torch.device("cpu"))

    assert len(result["固定七时刻逐行网络工程热预算"]) == 7
    for row in result["固定七时刻逐行网络工程热预算"]:
        assert row["顶部网络入流_W_m2"] == pytest.approx(42000, abs=1e-8)
        assert row["底部网络外流_W_m2"] == pytest.approx(42000, abs=1e-8)
        assert row["界面SiC热流_W_m2"] == pytest.approx(42000, abs=1e-8)
        assert row["界面Cu热流_W_m2"] == pytest.approx(42000, abs=1e-8)
        assert row["界面温跳_K"] == pytest.approx(4.2, abs=1e-8)
        assert row["网络储热率_W_m2"] == pytest.approx(0, abs=1e-8)
        assert row["已知入流热预算差_W"] == pytest.approx(0, abs=1e-8)
        assert row["界面热流差_W_m2"] == pytest.approx(0, abs=1e-8)
        assert row["界面Rc温跳约束差_K"] == pytest.approx(0, abs=1e-8)
    assert result["已吸收合成等效输入功率_W"] == pytest.approx(
        42000 * .0028274333882308137)


def test_a_locked_single_model_cannot_validate_a_posthoc_first_hf_read(tmp_path):
    from sic_cu.eval.task10_plate_posthoc import verify_late_posthoc_contract

    registration = (PROJECT_ROOT / "研究记录/任务10_独立双层场基准/"
                    "正式人为多热流入场前登记.yaml")
    archive_root = (PROJECT_ROOT / "研究记录/任务10_独立双层场基准/"
                    "九热流受限数值源_台账确认后_20260916T032402+0800")
    methods = (PROJECT_ROOT / "研究记录/任务10_独立双层场基准/"
               "正式同板方法十一源源码冻结_v2_20260916T050000+0800.tar.gz")
    method_budget = (PROJECT_ROOT / "研究记录/任务10_独立双层场基准/"
                     "正式同板F1_F2_F3重训方法预算前登记_v2.yaml")
    group = tmp_path / "仅单份模型假整组.json"
    group.write_text(json.dumps({
        "登记SHA256": sha256_file(registration),
        "受限源清单_SHA256": sha256_file(archive_root / "探针与源场SHA清单.json"),
        "方法预算_SHA256": sha256_file(method_budget),
        "方法源码tar_SHA256": sha256_file(methods),
        "方法源码tar文件": str(methods),
        "身份": [{"方法": "F1", "随机种子": 0, "LF来源": None,
                 "模型目录": str(tmp_path / "一个声称已锁模型")}],
    }, ensure_ascii=False), encoding="utf-8")
    ledger = tmp_path / "假台账仅测门禁.txt"
    ledger.write_text(sha256_file(group) + "\n" + sha256_file(methods) + "\n",
                      encoding="utf-8")

    with pytest.raises(PermissionError, match="25|身份"):
        verify_late_posthoc_contract(
            group_path=group, group_sha256=sha256_file(group),
            registration_path=registration, archive_root=archive_root,
            method_budget_path=method_budget, method_budget_sha256=sha256_file(method_budget),
            posthoc_budget_path=tmp_path / "无二级冻结预算.yaml",
            posthoc_budget_sha256="1" * 64,
            posthoc_tar_path=tmp_path / "无二级冻结源码.tar.gz",
            posthoc_tar_sha256="2" * 64, ledger_path=ledger,
        )


def test_all_25_synthetic_locks_still_deny_any_field_before_separate_source_freeze(
    synthetic_25_locked_boards,
):
    from sic_cu.eval.task10_plate_posthoc import verify_late_posthoc_contract
    from test_task10_plate_group_gate import ARCHIVE, BUDGET, REGISTRATION

    group, ledger = synthetic_25_locked_boards
    with pytest.raises(PermissionError, match="后验|另封|源码|预算|SHA"):
        verify_late_posthoc_contract(
            group_path=group, group_sha256=sha256_file(group),
            registration_path=REGISTRATION, archive_root=ARCHIVE,
            method_budget_path=BUDGET, method_budget_sha256=sha256_file(BUDGET),
            posthoc_budget_path=group.parent / "未存在的真实二级指标预算.yaml",
            posthoc_budget_sha256="1" * 64,
            posthoc_tar_path=group.parent / "未存在的后验九源tar.gz",
            posthoc_tar_sha256="2" * 64, ledger_path=ledger,
        )


def test_formal_cli_rejects_25_synthetic_locks_without_second_preregistration(
    synthetic_25_locked_boards,
):
    group, _ledger = synthetic_25_locked_boards
    script = PROJECT_ROOT / "scripts/任务10_全组锁后独立一维HF后验审计.py"
    spec = importlib.util.spec_from_file_location("task10_posthoc_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = group.parent / "没有合法来源的完整HF预测"
    with pytest.raises(PermissionError, match="后验|二级|25|台账|锁"):
        module.main([
            "--全组身份文件", str(group),
            "--全组SHA256", sha256_file(group),
            "--二级预算SHA256", "1" * 64,
            "--二级源码tarSHA256", "2" * 64,
            "--输出目录", str(destination),
        ])
    assert not destination.exists()


def test_full_grid_prediction_uses_only_frozen_coordinates_not_reference_labels():
    from sic_cu.eval.task10_plate_posthoc import predict_plate_reference_grid

    reference, setup = _synthetic_board()
    del reference["temperature_k"]
    setup["flux_splits_w_m2"] = {
        "train": [20000, 35000, 50000, 65000, 80000],
        "validation": [30000, 70000],
        "hidden_test": [42000, 58000],
    }
    model = _AnalyticSteadyBoard()
    result = predict_plate_reference_grid(
        model, reference, setup, flux_w_m2=42000, device=torch.device("cpu"),
        batch_size=41)
    depth = reference["node_depth_m"]
    expected = np.where(
        reference["material_id"] == 1,
        295.15 + 42000 * ((.012 - depth) / 120.0 + .0001 + .0055 / 401.0),
        295.15 + 42000 * ((.0175 - depth) / 401.0))
    assert result.shape == (201, 10)
    assert np.allclose(result, expected[None, :], rtol=0, atol=1e-9)
    assert result.dtype == np.float64
    with pytest.raises(ValueError, match="热流|折"):
        predict_plate_reference_grid(model, reference, setup, flux_w_m2=40300,
                                     device=torch.device("cpu"), batch_size=41)


def test_formal_posthoc_does_not_create_output_for_a_partial_board_group(tmp_path):
    from sic_cu.eval.task10_plate_posthoc import run_complete_plate_posthoc

    registration = (PROJECT_ROOT / "研究记录/任务10_独立双层场基准/"
                    "正式人为多热流入场前登记.yaml")
    archive_root = (PROJECT_ROOT / "研究记录/任务10_独立双层场基准/"
                    "九热流受限数值源_台账确认后_20260916T032402+0800")
    method_budget = (PROJECT_ROOT / "研究记录/任务10_独立双层场基准/"
                     "正式同板F1_F2_F3重训方法预算前登记_v2.yaml")
    group = tmp_path / "不齐的整组清单.json"
    group.write_text(json.dumps({"身份": []}, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "绝对不得创建真实HF后验输出"

    with pytest.raises(PermissionError):
        run_complete_plate_posthoc(
            output_directory=output, group_path=group, group_sha256=sha256_file(group),
            registration_path=registration, archive_root=archive_root,
            method_budget_path=method_budget, method_budget_sha256=sha256_file(method_budget),
            posthoc_budget_path=tmp_path / "未进账预算.yaml",
            posthoc_budget_sha256="1" * 64,
            posthoc_tar_path=tmp_path / "未进账源码.tar.gz",
            posthoc_tar_sha256="2" * 64,
            ledger_path=tmp_path / "未进账项目临时台账.txt",
            device=torch.device("cpu"),
        )
    assert not output.exists()
