#!/usr/bin/env python
"""任-10纯数值双层板控制与人为参考场封存。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.eval.two_layer_fvm import check_registered_fvm_controls, solve_two_layer_fvm


REGISTRATION = PROJECT_ROOT / "研究记录/任务10_独立双层场基准/纯数值控制前登记.yaml"


def _save_case(path: Path, result) -> None:
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, time_s=result.time_s, node_depth_m=result.node_depth_m,
            material_id=result.material_id, temperature_k=result.temperature_k,
            top_surface_temperature_k=result.top_surface_temperature_k,
            bottom_outward_flux_w_m2=result.bottom_outward_flux_w_m2,
            storage_rate_per_area_w_m2=result.storage_rate_per_area_w_m2,
            balance_per_area_w_m2=result.balance_per_area_w_m2,
            interface_flux_w_m2=result.interface_flux_w_m2,
            interface_temperature_jump_k=result.interface_temperature_jump_k,
        )
    os.replace(temporary, path)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="任-10独立双层板人为基准的纯数值控制")
    parser.add_argument("--output", required=True, help="新目录，已有产物永不覆盖")
    options = parser.parse_args()
    destination = Path(options.output)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    if destination.exists():
        raise FileExistsError(f"任-10纯数值控制目录已存在：{destination}")
    setup = load_yaml(REGISTRATION)
    controls = check_registered_fvm_controls(REGISTRATION)
    if not controls["refinement_target_met"]:
        raise RuntimeError("参考未通过预登记网格/时间步加密目标，禁止封存为数值参考")
    destination.mkdir(parents=True, exist_ok=False)
    fine_sic, fine_cu, fine_dt = controls["grid_steps"][-1]
    coarse_sic, coarse_cu, coarse_dt = controls["grid_steps"][0]
    end = float(setup["reference_controls"]["end_time_s"])
    reference = solve_two_layer_fvm(
        REGISTRATION, sic_cells=fine_sic, cu_cells=fine_cu, dt_s=fine_dt, end_s=end,
    )
    coarse = solve_two_layer_fvm(
        REGISTRATION, sic_cells=coarse_sic, cu_cells=coarse_cu, dt_s=coarse_dt, end_s=end,
    )
    mismatch = solve_two_layer_fvm(
        REGISTRATION, sic_cells=coarse_sic, cu_cells=coarse_cu, dt_s=coarse_dt, end_s=end,
        contact_multiplier=float(setup["thermal_conditions"]["coarse_mismatch_contact_multiplier"]),
    )
    files = {
        "封存独立参考_完整场.npz": reference,
        "人为LF_同物理粗网格.npz": coarse,
        "人为LF_接触失配粗网格.npz": mismatch,
    }
    for name, case in files.items():
        _save_case(destination / name, case)
    report = {
        "结论": "人为一维板求解器控制通过；主模型尚未冻结，未计算内部模型误差，非原装置或内部实验温度真值。",
        "四级SiC_Cu单元与时间步": controls["grid_steps"],
        "核查时刻_秒": controls["checked_comparison_times_s"],
        "层内核查深度_米": controls["sampling_depths_m"],
        "细两级最大温差_摄氏度": controls["fine_grid_max_difference_c"],
        "预登记加密目标_摄氏度": controls["refinement_target_c"],
        "细网格最大单步平衡_瓦每平方米": controls["fine_grid_max_step_balance_w_m2"],
        "细网格200秒顶部与独立解析稳态差_摄氏度": controls["fine_grid_steady_top_error_c"],
        "参考200秒顶部温度_摄氏度": reference.top_surface_temperature_k[-1] - 273.15,
        "同物理粗网格200秒顶部温度_摄氏度": coarse.top_surface_temperature_k[-1] - 273.15,
        "接触失配粗网格200秒顶部温度_摄氏度": mismatch.top_surface_temperature_k[-1] - 273.15,
        "人为接触失配倍率": setup["thermal_conditions"]["coarse_mismatch_contact_multiplier"],
        "全部完整数值场": "分别封存，后续训练不得使用完整参考场作标签或早停",
        "旧test_Data标签读取": False,
    }
    _write_json(destination / "网格与时间步控制.json", report)
    sources = {
        "结论": "纯数值求解器、预登记与人为源场溯源；尚无任-07后完整场模型对比资格。",
        "预登记配置_SHA256": sha256_file(REGISTRATION),
        "材料配置_SHA256": sha256_file(PROJECT_ROOT / setup["materials_source"]),
        "求解器源码_SHA256": sha256_file(PROJECT_ROOT / "src/sic_cu/eval/two_layer_fvm.py"),
        "封存入口源码_SHA256": sha256_file(PROJECT_ROOT / "scripts/22_check_two_layer_fvm_controls.py"),
        "中文控制报告_SHA256": sha256_file(destination / "网格与时间步控制.json"),
        "数值场SHA256": {name: sha256_file(destination / name) for name in files},
        "旧test_Data标签读取": False,
    }
    _write_json(destination / "来源校验.json", sources)
    print(json.dumps({"控制状态": "通过", "细两级最大温差_摄氏度":
                      controls["fine_grid_max_difference_c"], "输出目录": str(destination)},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
