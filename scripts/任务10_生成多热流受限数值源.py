#!/usr/bin/env python
"""任10独立数值工厂：预登记九热流参考及观察折隔离。"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import tarfile

import numpy as np

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.eval.task10_plate_benchmark import (
    REGISTRATION, check_registered_reference, load_benchmark_registration,
    registered_probe_snapshot, solve_registered_plate,
)


SOURCE_NAMES = (
    "研究记录/任务10_独立双层场基准/正式人为多热流入场前登记.yaml",
    "src/sic_cu/eval/task10_plate_benchmark.py",
    "scripts/任务10_检查人为多热流入场.py",
    "scripts/任务10_生成多热流受限数值源.py",
    "tests/test_task10_plate_benchmark.py",
    "configs/materials.yaml",
)


def _project_path(value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise ValueError("任10数值工厂的输入归档及输出均必须位于项目目录内")
    return resolved


def _verify_source_snapshot(path: Path, expected_sha: str) -> None:
    if sha256_file(path) != expected_sha:
        raise ValueError("任10数值工厂源码tar SHA与事前封存值不符")
    try:
        with tarfile.open(path, "r:gz") as snapshot:
            if not set(SOURCE_NAMES) <= set(snapshot.getnames()):
                raise ValueError("任10事前tar缺配置、两CLI、独立求解器、测试或材料")
            for name in SOURCE_NAMES:
                archived = snapshot.extractfile(name)
                if archived is None or sha256(archived.read()).hexdigest() != sha256_file(PROJECT_ROOT / name):
                    raise ValueError(f"任10事前tar内源码与生效文件SHA不一致：{name}")
    except tarfile.TarError as exc:
        raise ValueError("任10事前tar不可解析，不得运行数值工厂") from exc


def _save_case(path: Path, result) -> None:
    np.savez_compressed(path, **{
        name: getattr(result, name) for name in result.__dataclass_fields__
    })


def _validate_numeric_physics(setup: dict, q: int, reference, controls: dict) -> None:
    area = float(setup["geometry"]["cross_section_area_m2"])
    if (not np.isfinite(controls["max_step_balance_w_m2"])
            or controls["max_step_balance_w_m2"] * area >= q * area * 1e-10):
        raise ValueError("独立HF守恒数值源每步有量纲热余额未守预登记输入门禁")
    material = load_yaml(setup["materials_source"])
    geom = setup["geometry"]
    boundary = setup["thermal_conditions"]
    analytic = float(boundary["bottom_fixed_temperature_k"]) + q * (
        float(geom["silicon_carbide_thickness_m"])
        / float(material["silicon_carbide"]["conductivity_w_m_k"]["value"])
        + float(boundary["benchmark_contact_resistance_m2_k_w"])
        + float(geom["copper_thickness_m"])
        / float(material["copper"]["conductivity_w_m_k"]["value"])
    )
    if (abs(reference.top_surface_temperature_k[-1] - analytic) >= 1e-4
            or abs(reference.interface_flux_w_m2[-1] - q) >= 1e-4
            or abs(reference.interface_temperature_jump_k[-1]
                   - q * float(boundary["benchmark_contact_resistance_m2_k_w"])) >= 1e-4):
        raise ValueError("独立HF顶部解析热阻稳态/界面热流或温跳控制失败")


def main() -> None:
    parser = argparse.ArgumentParser(description="任10九档热流数值参考/受限观测来源CPU工厂")
    parser.add_argument("--output", required=True, help="项目内新编号原件目录")
    parser.add_argument("--source-archive", required=True, help="事前六源码tar.gz")
    parser.add_argument("--source-sha256", required=True, help="事前六源码tar.gz确切SHA256")
    options = parser.parse_args()
    output, source = _project_path(options.output), _project_path(options.source_archive)
    if output.exists():
        raise FileExistsError(f"已有任10多热流数值原件绝不覆盖：{output}")
    if len(options.source_sha256) != 64:
        raise ValueError("任10事前封存源码tar SHA256必须为64位")
    _verify_source_snapshot(source, options.source_sha256)
    setup = load_benchmark_registration(REGISTRATION)
    folds = setup["flux_splits_w_m2"]
    jobs = [(role, int(q)) for role in ("train", "validation", "hidden_test")
            for q in folds[role]]
    controls = {}
    for role, q in jobs:
        measured = check_registered_reference(REGISTRATION, q)
        if (not measured["max_adjacent_difference_c"] < measured["limit_c"]
                or not np.isfinite(measured["max_step_balance_w_m2"])):
            raise ValueError(f"人为热流{q}未通过独立参考相邻网格/能量数值控制")
        controls[str(q)] = measured

    output.mkdir(parents=True, exist_ok=False)
    (output / "HF_封存完整场").mkdir()
    (output / "HF_允许探针").mkdir()
    (output / "LF_训练场").mkdir()
    archived_files = []
    numeric_rows = []
    for role, q in jobs:
        measured = controls[str(q)]
        sic, cu, dt = measured["fine_grid"]
        reference = solve_registered_plate(REGISTRATION, q, sic_cells=sic, cu_cells=cu,
                                           dt_s=dt, end_s=200.0)
        _validate_numeric_physics(setup, q, reference, measured)
        full_path = output / f"HF_封存完整场/{q}.npz"
        _save_case(full_path, reference)
        archived_files.append(full_path)
        if role in ("train", "validation"):
            name = "训练" if role == "train" else "验证"
            allowed_path = output / f"HF_允许探针/{name}_{q}.npz"
            np.savez_compressed(allowed_path, **registered_probe_snapshot(REGISTRATION, q, role, reference))
            archived_files.append(allowed_path)
        if role == "train":
            for name, multiplier in (("同物理", 1.0), ("接触失配", 0.7)):
                low = solve_registered_plate(REGISTRATION, q, sic_cells=24, cu_cells=11,
                                             dt_s=2.0, end_s=200.0,
                                             contact_multiplier=multiplier)
                path = output / f"LF_训练场/{name}_{q}.npz"
                _save_case(path, low)
                archived_files.append(path)
        numeric_rows.append({
            "人为热流_瓦每平方米": q,
            "数值数据折": {"train": "训练", "validation": "验证", "hidden_test": "封存测试"}[role],
            "细SiC单元数": sic, "细Cu单元数": cu, "细时间步_秒": dt,
            "最细相邻差_摄氏度": measured["max_adjacent_difference_c"],
            "加密目标_摄氏度": measured["limit_c"],
            "细网格最大单步热余额_瓦每平方米": measured["max_step_balance_w_m2"],
            "完整参考场仅封存SHA256": sha256_file(full_path),
            "HF观察开放": "三探针训练" if role == "train" else
                          "三探针合法验证" if role == "validation" else "禁用直至锁模型SHA",
        })
    csv_path = output / "九热流数值门禁原始明细.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(numeric_rows[0]))
        writer.writeheader()
        writer.writerows(numeric_rows)
    index = {
        "登记SHA256": sha256_file(REGISTRATION),
        "事前六源tar_SHA256": sha256_file(source),
        "本数值工厂源码_SHA256": sha256_file(PROJECT_ROOT / "scripts/任务10_生成多热流受限数值源.py"),
        "逐热流数值控制": {
            str(q): {"细网格": list(controls[str(q)]["fine_grid"]),
                     "相邻差_摄氏度": controls[str(q)]["max_adjacent_difference_c"],
                     "最大单步平衡_瓦每平方米": controls[str(q)]["max_step_balance_w_m2"]}
            for _, q in jobs
        },
        "归档文件_SHA256": {
            str(path.relative_to(output)): sha256_file(path) for path in archived_files
        },
        "明细CSV_SHA256": sha256_file(csv_path),
        "旧test_Data标签读取": False,
        "方法训练/合法验证已经完成": False,
    }
    (output / "探针与源场SHA清单.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    report = (
        "# 任10一维人为九热流：仅数值源/受限观察入场\n\n"
        f"事前源码快照SHA256 `{sha256_file(source)}`；入场登记SHA256 `{sha256_file(REGISTRATION)}`。"
        "九档均由独立FVM同一常物性、顶面均匀热通量、底部固定温与已知接触热阻CPU解算，"
        "先逐档检查相邻细两级固定时刻/深度温差严格小于事前0.1℃，否则按已登记网格加密；"
        "仅相同物理的粗网格与人为0.7倍Rc失配为五训练档LF完整场。\n\n"
        "五档训练和两档验证各仅另导预登记顶部、13/16mm三个虚拟探针与14个合法采样时刻。"
        "42,000/58,000 W/m²隐藏测试档仅封存完整参考字节与SHA，不输出实际温度，"
        "不开放训练/验证读取完整参考温度、内部材料节点、测试温度或网格内部真值。"
        "逐档数值CSV只列网格、加密差、能量余额与不可解码的参考SHA；"
        "**没有任何F1/F2/F3或E0重训、早停、内部场模型误差或原装置实测真值**。\n\n"
        "下一动作必须单独事前冻结同一一维物理/已知参数、F1/F2/F3共享观察折、两LF条件下"
        "的模型预算与来源；完整参考只在训练日志、合法三探针验证及所采用检查点SHA锁定后"
        "交由后验数值评价器逐档验证网格头与数值门禁后打开。"
    )
    (output / "九热流数值入场记录.md").write_text(report + "\n", encoding="utf-8")
    print(json.dumps({"阶段": "九人为热流纯CPU参考与允许观察来源封存",
                      "登记热流": [q for _, q in jobs], "各档0.1摄氏度网格门禁": True,
                      "隐藏测试温度输出": False,
                      "主模型正式训练或内部误差": False,
                      "SHA清单": str(output / "探针与源场SHA清单.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
