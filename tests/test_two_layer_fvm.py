from __future__ import annotations

import json
import numpy as np
import runpy
import sys
from pathlib import Path

import pytest

from sic_cu.eval.two_layer_fvm import check_registered_fvm_controls, solve_two_layer_fvm


REGISTRATION = "研究记录/任务10_独立双层场基准/纯数值控制前登记.yaml"


def test_fvm_steady_top_temperature_matches_independent_series_resistance() -> None:
    result = solve_two_layer_fvm(REGISTRATION, sic_cells=24, cu_cells=11, dt_s=2.0, end_s=200.0)
    independently_calculated = 295.15 + 50_000 * (0.012 / 120 + 1e-4 + 0.0055 / 401)
    assert abs(result.top_surface_temperature_k[-1] - independently_calculated) < 1e-5
    assert result.temperature_k.shape == (101, 35)
    assert np.isclose(result.time_s[0], 0)
    assert np.isclose(result.temperature_k[0], 295.15).all()


def test_fvm_each_implicit_step_satisfies_dimensional_heat_budget() -> None:
    result = solve_two_layer_fvm(REGISTRATION, sic_cells=24, cu_cells=11, dt_s=2.0, end_s=20.0)
    assert result.balance_per_area_w_m2.shape == (11,)
    assert abs(result.balance_per_area_w_m2[0]) == 0
    assert np.max(np.abs(result.balance_per_area_w_m2[1:])) < 1e-6
    assert np.max(np.abs(result.balance_per_area_w_m2[1:])) / 50_000 < 1e-10
    assert np.all(result.bottom_outward_flux_w_m2[1:] > 0)
    assert result.bottom_outward_flux_w_m2[-1] < 50_000


def test_fvm_interface_jump_uses_registered_series_resistance_not_cell_width() -> None:
    result = solve_two_layer_fvm(REGISTRATION, sic_cells=48, cu_cells=22, dt_s=1.0, end_s=200.0)
    assert abs(result.interface_temperature_jump_k[-1] - 5.0) < 1e-5
    assert abs(result.interface_flux_w_m2[-1] - 50_000) < 1e-5


def test_fvm_coarse_contact_mismatch_is_a_separate_synthetic_case() -> None:
    reference = solve_two_layer_fvm(REGISTRATION, sic_cells=24, cu_cells=11, dt_s=2.0, end_s=200.0)
    mismatch = solve_two_layer_fvm(
        REGISTRATION, sic_cells=24, cu_cells=11, dt_s=2.0, end_s=200.0,
        contact_multiplier=0.7,
    )
    assert abs(reference.top_surface_temperature_k[-1] - mismatch.top_surface_temperature_k[-1] - 1.5) < 1e-5
    assert np.array_equal(reference.node_depth_m, mismatch.node_depth_m)


def test_registered_four_level_reference_controls_are_below_fixed_refinement_limit() -> None:
    checked = check_registered_fvm_controls(REGISTRATION)
    assert len(checked["grid_steps"]) == 4
    assert checked["grid_steps"][-1] == (192, 88, 0.25)
    assert checked["checked_comparison_times_s"] == [1, 2, 5, 10, 50, 100, 200]
    assert checked["fine_grid_max_difference_c"] < 0.1
    assert checked["fine_grid_max_step_balance_w_m2"] / 50_000 < 1e-10
    assert checked["fine_grid_steady_top_error_c"] < 1e-5


def test_fvm_control_export_seals_independent_reference_and_never_overwrites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    output = tmp_path / "独立人为基准封存"
    script = Path(__file__).resolve().parents[1] / "scripts/22_check_two_layer_fvm_controls.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(output)])
    runpy.run_path(str(script), run_name="__main__")
    report = json.loads((output / "网格与时间步控制.json").read_text(encoding="utf-8"))
    provenance = json.loads((output / "来源校验.json").read_text(encoding="utf-8"))
    assert report["细两级最大温差_摄氏度"] < 0.1
    assert set(provenance["数值场SHA256"]) == {
        "封存独立参考_完整场.npz", "人为LF_同物理粗网格.npz", "人为LF_接触失配粗网格.npz",
    }
    assert provenance["旧test_Data标签读取"] is False
    with np.load(output / "封存独立参考_完整场.npz") as reference:
        assert reference["temperature_k"].shape == (801, 280)
        assert reference["material_id"].sum() == 192
    with pytest.raises(FileExistsError):
        runpy.run_path(str(script), run_name="__main__")
