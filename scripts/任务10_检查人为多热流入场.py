#!/usr/bin/env python
"""任10已登记一维板的CPU单热流入场数值控制。"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import tarfile

import numpy as np

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.eval.task10_plate_benchmark import (
    REGISTRATION, check_registered_reference, load_benchmark_registration,
    registered_probe_snapshot, solve_registered_plate,
)


SOURCES = (
    "研究记录/任务10_独立双层场基准/正式人为多热流入场前登记.yaml",
    "src/sic_cu/eval/task10_plate_benchmark.py",
    "scripts/任务10_检查人为多热流入场.py",
    "tests/test_task10_plate_benchmark.py",
    "configs/materials.yaml",
)


def _project_path(value: str) -> Path:
    raw = Path(value)
    path = (raw if raw.is_absolute() else PROJECT_ROOT / raw).resolve()
    if path != PROJECT_ROOT and PROJECT_ROOT not in path.parents:
        raise ValueError("任10源和输出目标必须位于项目目录内")
    return path


def _verify_frozen_sources(path: Path, expected_sha: str) -> None:
    if sha256_file(path) != expected_sha:
        raise ValueError("任10事前源码归档SHA不符，不得求解新人为热流")
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = set(archive.getnames())
            if not set(SOURCES) <= members:
                raise ValueError("事前tar源码归档缺少任10登记、求解器、CLI、测试或材料")
            for name in SOURCES:
                archived = archive.extractfile(name)
                if archived is None or sha256(archived.read()).hexdigest() != sha256_file(PROJECT_ROOT / name):
                    raise ValueError(f"任10源码与事前归档SHA字节不同：{name}")
    except tarfile.TarError as exc:
        raise ValueError("源码tar归档无效，禁止数值源导出") from exc


def _save_result(path: Path, result) -> None:
    np.savez_compressed(path, **{
        name: getattr(result, name) for name in result.__dataclass_fields__
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="任10纯CPU五万瓦每平方米数值入场")
    parser.add_argument("--output", required=True, help="项目内全新原件目录")
    parser.add_argument("--source-archive", required=True, help="事前已封存的项目内源码tar.gz")
    parser.add_argument("--source-sha256", required=True, help="该源码tar.gz确切SHA256")
    opts = parser.parse_args()
    output, archive = _project_path(opts.output), _project_path(opts.source_archive)
    if output.exists():
        raise FileExistsError(f"任10数值入场原件已存在，不允许覆盖：{output}")
    if len(opts.source_sha256) != 64:
        raise ValueError("归档SHA必须是事前封存的SHA256值")
    _verify_frozen_sources(archive, opts.source_sha256)
    setup = load_benchmark_registration(REGISTRATION)
    if 50000 not in setup["flux_splits_w_m2"]["train"]:
        raise ValueError("旧50,000 W/m²只可作预登记训练折纯数值控制")
    controls = check_registered_reference(REGISTRATION, 50000)
    if controls["fine_grid"] != (192, 88, 0.25):
        raise ValueError("旧50,000 W/m²数值源与事前四级加密记录不匹配")
    fine = solve_registered_plate(REGISTRATION, 50000, sic_cells=192, cu_cells=88,
                                  dt_s=0.25, end_s=200.0)
    same = solve_registered_plate(REGISTRATION, 50000, sic_cells=24, cu_cells=11,
                                  dt_s=2.0, end_s=200.0)
    mismatch = solve_registered_plate(REGISTRATION, 50000, sic_cells=24, cu_cells=11,
                                      dt_s=2.0, end_s=200.0, contact_multiplier=0.7)
    old_report = json.loads((PROJECT_ROOT / "研究记录/任务10_独立双层场基准/"
                             "纯数值控制_20260915T203723+0800/网格与时间步控制.json").read_text(encoding="utf-8"))
    if abs(fine.top_surface_temperature_k[-1] - 273.15 - old_report["参考200秒顶部温度_摄氏度"]) >= 1e-5:
        raise ValueError("新多热流数值求解器在原单热流独立顶部来源回归失败")
    output.mkdir(parents=True, exist_ok=False)
    probe_path = output / "HF_允许探针/训练_50000.npz"
    lf_same = output / "LF_训练场/同物理_50000.npz"
    lf_mismatch = output / "LF_训练场/接触失配_50000.npz"
    probe_path.parent.mkdir()
    lf_same.parent.mkdir()
    np.savez_compressed(probe_path, **registered_probe_snapshot(REGISTRATION, 50000, "train", fine))
    _save_result(lf_same, same)
    _save_result(lf_mismatch, mismatch)
    index = {
        "登记SHA256": sha256_file(REGISTRATION),
        "入场已归档源码tar_SHA256": sha256_file(archive),
        "旧单热流控制JSON_SHA256": sha256_file(PROJECT_ROOT / "研究记录/任务10_独立双层场基准/"
                                                 "纯数值控制_20260915T203723+0800/网格与时间步控制.json"),
        "归档文件_SHA256": {
            str(path.relative_to(output)): sha256_file(path)
            for path in (probe_path, lf_same, lf_mismatch)
        },
        "逐热流数值控制": {
            "50000": {"细网格": list(controls["fine_grid"]),
                      "相邻差_摄氏度": controls["max_adjacent_difference_c"]},
        },
    }
    (output / "探针与源场SHA清单.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    csv_path = output / "单热流数值入场明细.csv"
    columns = ("人为热流_瓦每平方米", "时刻_秒", "参考顶部温度_摄氏度",
               "同物理粗LF顶部温度_摄氏度", "人为失配粗LF顶部温度_摄氏度",
               "参考界面热流_瓦每平方米", "参考界面跳温_摄氏度",
               "参考单步热余额_瓦每平方米")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for time_s in controls["comparison_times_s"]:
            fine_idx, coarse_idx = round(time_s / 0.25), round(time_s / 2.0)
            lf_registered_time = coarse_idx * 2 == time_s
            writer.writerow(dict(zip(columns, (
                50000, time_s, float(fine.top_surface_temperature_k[fine_idx] - 273.15),
                float(same.top_surface_temperature_k[coarse_idx] - 273.15) if lf_registered_time else "",
                float(mismatch.top_surface_temperature_k[coarse_idx] - 273.15) if lf_registered_time else "",
                float(fine.interface_flux_w_m2[fine_idx]),
                float(fine.interface_temperature_jump_k[fine_idx]),
                float(fine.balance_per_area_w_m2[fine_idx]),
            ))))
    area = float(setup["geometry"]["cross_section_area_m2"])
    report = (
        "# 任10人为双层板：仅50,000 W/m² CPU数值入场\n\n"
        f"事前源码tar SHA256：`{sha256_file(archive)}`；专属配置SHA256：`{sha256_file(REGISTRATION)}`。"
        "测试折42,000/58,000 W/m²的隐藏温度未生成或打开；未运行任10任何主模型/基线训练。\n\n"
        f"细两级固定时刻/深度最大差 `{controls['max_adjacent_difference_c']:.12f} ℃`，门槛严格小于"
        f" `{controls['limit_c']} ℃`；单步每平方米热余额最大"
        f" `{controls['max_step_balance_w_m2']:.12g} W/m²`。人为输入热功率为"
        f" `{50000*area:.9f} W`，不是实际装置激光功率。200秒参考顶部"
        f" `{fine.top_surface_temperature_k[-1]-273.15:.9f} ℃`、界面通量"
        f" `{fine.interface_flux_w_m2[-1]:.6f} W/m²`、跳温"
        f" `{fine.interface_temperature_jump_k[-1]:.6f} ℃`；失配LF200秒顶部"
        f" `{mismatch.top_surface_temperature_k[-1]-273.15:.9f} ℃`。\n\n"
        "仅 `HF_允许探针/训练_50000.npz` 提供三个允许温度位置和14个前登记时刻；"
        "两个LF归档为完整粗网格场。1秒与5秒不在2秒LF时间网格，CSV对应LF列留空，不伪造插值。"
        "完整HF内部参考不入训练/验证目录，正式隐藏误差须模型另行同板重训、合法验证及检查点SHA锁后才可计算。\n"
    )
    (output / "单热流数值入场记录.md").write_text(report, encoding="utf-8")
    print(json.dumps({"控制": "旧来源纯数值CPU复核", "旧单热流": 50000,
                      "训练探针": str(probe_path), "最大差_摄氏度": controls["max_adjacent_difference_c"],
                      "未读隐藏测试温度": True}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
