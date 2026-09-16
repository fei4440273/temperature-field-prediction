"""Task-11 independent MLP HF Watt audit, without changing model selection."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.models import AdditiveCorrectionModel, ModelScales
from sic_cu.eval.energy_v5 import AxisymmetricGeometry, audit_schedule_energy
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.train.simulation import build_model


FIXED_POWERS = [55.0, 115.2, 364.3, 403.0, 630.5, 729.0]
FIXED_TIMES = [1.0, 10.0, 50.0, 100.0, 200.0]
FIXED_ORDERS = [16, 64]
ENERGY_CONTRACT = {
    "功率_瓦": FIXED_POWERS, "时刻_秒": FIXED_TIMES,
    "求积阶数": FIXED_ORDERS, "功率时刻点数": 30,
    "模型状态": ["best", "final"], "计算dtype": "float64",
    "名义吸收归一均值筛查": 0.05, "名义吸收归一95分位筛查": 0.10,
    "不用于训练选模": True, "工程安全阈值": False,
}
LF_MODEL_KWARGS = {
    "width": 128, "depth": 5, "activation": "tanh", "include_material": True,
}
CORRECTION_MODEL_KWARGS = {
    "width": 128, "depth": 4, "activation": "tanh", "include_material": True,
    "hard_initial_temperature_k": 295.15, "initial_ramp_time_s": 0.05,
    "correction_calibration_range_w": [55.0, 800.0],
    "correction_support_range_w": [0.0, 800.0],
    "correction_extrapolation_exponent": 2.0,
    "correction_power_scaling": "none", "correction_power_reference_w": 400.0,
    "correction_direct_power_input": True, "silicon_carbide_height_m": 0.012,
    "surface_guide_output": "residual",
}
MODEL_SCALES = {
    "r_max_m": 0.05834, "z_min_m": -0.0175, "time_max_s": 200.0,
    "power_max_w": 800.0, "temperature_offset_k": 295.15,
    "temperature_scale_k": 250.0,
}


def task11_fixed_energy_schedule(registration: Mapping[str, Any]):
    if registration.get("独立能源审核") != ENERGY_CONTRACT:
        raise ValueError("任11能源必须使用冻结30点双阶及仅名义筛查，不能改成工程安全阈值")
    return list(FIXED_POWERS), list(FIXED_TIMES), list(FIXED_ORDERS)


def task11_energy_output(
    run: str | Path, output: str | Path, state: str, seed: int,
    *, project_root: str | Path = PROJECT_ROOT,
) -> Path:
    if state not in ("best", "final"):
        raise ValueError("任11独立能源状态仅准best/final，不用物理诊断回调选模")
    if type(seed) is not int or seed not in range(5):
        raise ValueError("任11能源身份只准五个登记seed")
    root = Path(project_root).resolve()
    run_path, destination = Path(run), Path(output)
    if not run_path.is_absolute():
        run_path = root / run_path
    if not destination.is_absolute():
        destination = root / destination
    canonical = root / (
        "研究记录/任务11_外部对照/正式新MLP公平训练/"
        f"正式MLP_HF_seed{seed}"
    )
    run_path, destination = run_path.resolve(), destination.resolve()
    label = "观测最佳" if state == "best" else "训练末"
    if (run_path != canonical or destination.parent != canonical
            or not re.fullmatch(rf"独立原能源_{label}_\d{{8}}T\d{{6}}\+0800",
                                destination.name)):
        raise ValueError("任11能源目录须项目内同seed规范新状态子目录，不能软链越界")
    if destination.exists():
        raise FileExistsError("任11已有独立能源原件不能覆盖")
    return destination


def require_task11_energy_cuda(device_name: str) -> torch.device:
    if device_name != "cuda":
        raise ValueError("任11正式能源只准GPU CUDA，CPU诊断不可冒充")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("任11正式能源须恰好一张可见GPU CUDA")
    return torch.device("cuda")


def task11_energy_model_from_view(view: Mapping[str, Any]) -> nn.Module:
    if (view.get("method") != "multifidelity_correction"
            or view.get("low_fidelity_method") != "mlp_pinn"
            or view.get("low_fidelity_model_kwargs") != LF_MODEL_KWARGS
            or view.get("correction_model_kwargs") != CORRECTION_MODEL_KWARGS
            or view.get("scales") != MODEL_SCALES
            or not isinstance(view.get("model_state"), dict)):
        raise ValueError("任11能源只准登记MLP LF及原六输入加性校正器身份，不能冒用DeepONet")
    scales = ModelScales(**view["scales"])
    low = build_model("mlp_pinn", scales, **view["low_fidelity_model_kwargs"])
    model = AdditiveCorrectionModel(low, scales, **view["correction_model_kwargs"])
    if model.correction[0].in_features != 6:
        raise ValueError("任11能源MLP校正器必须保留原六输入结构")
    model.load_state_dict(view["model_state"], strict=True)
    return model.eval()


def verify_task11_energy_raw(audit: Mapping[str, Any]) -> dict[str, float]:
    energy, divergence = audit.get("原始能量"), audit.get("原始散度")
    mapped, decomposition, summary = (
        audit.get("指标明细"), audit.get("物理分解"), audit.get("汇总"),
    )
    if (not isinstance(energy, list) or not isinstance(divergence, list)
            or len(energy) != 60 or len(divergence) != 60
            or mapped is None or decomposition is None
            or mapped.height != 30 or decomposition.height != 30
            or not isinstance(summary, dict)):
        raise ValueError("任11双阶必须保留60原能量、60原散度及各30行指标/物理分解")
    try:
        json.dumps([energy, divergence], allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("任11全部原件字段必须是有限JSON数值，不能保留NaN或非数值对象") from error
    watt_keys = (
        "absorbed_power_w", "storage_rate_w", "cooling_heat_w",
        "convection_heat_w", "radiation_heat_w", "balance_w",
        "relative_balance_denominator_w", "relative_balance",
    )
    vjd_keys = (
        "integrated_pde_residual_w", "interface_two_sided_flux_w",
        "boundary_flux_residual_w", "explained_engineering_balance_w",
        "engineering_balance_w", "engineering_explanation_gap_w",
        "silicon_carbide_pde_residual_w", "copper_outer_ring_pde_residual_w",
        "copper_below_sic_pde_residual_w", "outer_temperature_max_abs_deviation_c",
    )

    def same(a: float, b: float, *, denominator: bool = False) -> bool:
        return math.isclose(a, b, rel_tol=1e-12 if denominator else 1e-10,
                            abs_tol=1e-9 if denominator else 1e-7)

    lower_gap, higher_gap, neighbors, high_energy, high_divergence = [], [], [], [], []
    conditions = [(p, t, q) for p in FIXED_POWERS for t in FIXED_TIMES
                  for q in FIXED_ORDERS]
    for index, (power, time_s, order) in enumerate(conditions):
        raw, terms = energy[index], divergence[index]
        if (not isinstance(raw, dict) or not isinstance(terms, dict)
                or any(row.get("power_w") != power or row.get("time_s") != time_s
                       or row.get("quadrature_order") != order for row in (raw, terms))
                or any(type(raw.get(key)) not in (int, float)
                       or not math.isfinite(raw[key]) for key in watt_keys)
                or any(type(terms.get(key)) not in (int, float)
                       or not math.isfinite(terms[key]) for key in vjd_keys)):
            raise ValueError("任11原瓦数/散度固定30点双阶身份或有限数值不一致")
        absorbed, storage = raw["absorbed_power_w"], raw["storage_rate_w"]
        cooling = sum(raw[key] for key in
                      ("cooling_heat_w", "convection_heat_w", "radiation_heat_w"))
        denominator = max(abs(absorbed), abs(storage), abs(cooling), 1e-12)
        explained = (terms["integrated_pde_residual_w"]
                     - terms["interface_two_sided_flux_w"]
                     - terms["boundary_flux_residual_w"])
        if (absorbed <= 0 or raw["relative_balance_denominator_w"] <= 0
                or not same(raw["balance_w"], storage + cooling - absorbed)
                or not same(raw["relative_balance_denominator_w"], denominator,
                            denominator=True)
                or not same(raw["relative_balance"], raw["balance_w"] / denominator)
                or not same(terms["engineering_balance_w"], raw["balance_w"])
                or not same(terms["explained_engineering_balance_w"], explained)
                or not same(terms["engineering_explanation_gap_w"],
                            raw["balance_w"] - explained)
                or not same(terms["integrated_pde_residual_w"], sum(terms[key] for key in
                            ("silicon_carbide_pde_residual_w",
                             "copper_outer_ring_pde_residual_w",
                             "copper_below_sic_pde_residual_w")))):
            raise ValueError("任11原瓦数B、原定义相对分母、V-J-D与材料分解必须逐点一致")
        gap = abs(terms["engineering_explanation_gap_w"])
        if order == 16:
            lower_gap.append(gap)
        else:
            higher_gap.append(gap)
            high_energy.append(raw)
            high_divergence.append(terms)
            neighbors.append(max(abs(raw[key] - energy[index - 1][key])
                                 for key in watt_keys[:6]) / abs(absorbed))
    energy_columns = (
        ("power_w", "功率_瓦"), ("time_s", "时刻_秒"),
        ("quadrature_order", "求积阶数"), ("absorbed_power_w", "吸收功率_瓦"),
        ("storage_rate_w", "储能率_瓦"), ("cooling_heat_w", "水冷散热_瓦"),
        ("convection_heat_w", "对流散热_瓦"), ("radiation_heat_w", "辐射散热_瓦"),
        ("balance_w", "平衡_瓦"),
        ("relative_balance_denominator_w", "原定义相对平衡分母_瓦"),
        ("relative_balance", "原定义相对平衡"), ("outer_epsilon_m", "外边界内移_米"),
    )
    vjd_columns = (
        ("power_w", "功率_瓦"), ("time_s", "时刻_秒"),
        ("integrated_pde_residual_w", "体积分残差V_瓦"),
        ("interface_two_sided_flux_w", "界面双侧通量J_瓦"),
        ("boundary_flux_residual_w", "边界失配D_瓦"),
        ("engineering_balance_w", "原工程平衡_瓦"),
        ("explained_engineering_balance_w", "解释工程平衡_瓦"),
        ("engineering_explanation_gap_w", "工程解释剩余差_瓦"),
        ("silicon_carbide_pde_residual_w", "SiC体内积分残差_瓦"),
        ("copper_outer_ring_pde_residual_w", "Cu外环积分残差_瓦"),
        ("copper_below_sic_pde_residual_w", "Cu下层积分残差_瓦"),
        ("outer_temperature_max_abs_deviation_c", "水冷边界最大温差_摄氏度"),
    )
    for raw, terms, line, vjd in zip(high_energy, high_divergence,
                                    mapped.iter_rows(named=True),
                                    decomposition.iter_rows(named=True)):
        if (any(name not in line or not same(float(line[name]), raw[key],
                    denominator=key == "relative_balance_denominator_w")
                for key, name in energy_columns)
                or not same(float(line.get("绝对平衡_瓦", math.inf)), abs(raw["balance_w"]))
                or line.get("边界梯度模式原标识") != raw.get("boundary_gradient_mode")
                or any(name not in vjd or not same(float(vjd[name]), terms[key])
                       for key, name in vjd_columns)):
            raise ValueError("任11指标/物理分解CSV与64阶原瓦数/散度逐行不一致")
    absolute = np.asarray([abs(row["balance_w"]) for row in high_energy])
    normalized = absolute / np.asarray([row["absorbed_power_w"] for row in high_energy])
    computed = {
        "绝对平衡宏均值_瓦": float(absolute.mean()),
        "绝对平衡95分位_瓦": float(np.percentile(absolute, 95)),
        "绝对平衡最大值_瓦": float(absolute.max()),
        "吸收功率归一宏均值": float(normalized.mean()),
        "吸收功率归一95分位": float(np.percentile(normalized, 95)),
        "最大相邻阶变化对吸收功率比": float(max(neighbors)),
        "最大分解剩余差_瓦": float(max(higher_gap)),
    }
    if (summary.get("原定义相对平衡分母已保持") is not True
            or summary.get("工程平衡与原瓦数逐行一致") is not True
            or summary.get("功率时刻审核行数") != 30
            or any(type(summary.get(name)) not in (int, float)
                   or not math.isfinite(summary[name]) or not same(summary[name], value)
                   for name, value in computed.items())):
        raise ValueError("任11能源汇总必须由全部原件重新计算且与原瓦数一致")
    return {
        **computed,
        "16阶V-J-D散度积分剩余差最大_瓦": max(lower_gap),
        "16阶V-J-D散度积分剩余差宏均值_瓦": float(np.mean(lower_gap)),
        "64阶V-J-D散度积分剩余差最大_瓦": max(higher_gap),
        "64阶V-J-D散度积分剩余差宏均值_瓦": float(np.mean(higher_gap)),
    }


def write_task11_energy_evidence(
    output: str | Path, audit: Mapping[str, Any], payload: Mapping[str, Any],
    *, project_root: str | Path = PROJECT_ROOT,
) -> dict[str, str]:
    root = Path(project_root).resolve()
    destination = Path(output)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    if destination == root or not destination.is_relative_to(root):
        raise ValueError("任11能源只允许写项目内全新目录")
    if destination.exists():
        raise FileExistsError("任11原能源目录已存在，不覆盖")
    verified = verify_task11_energy_raw(audit)
    if (payload.get("原定义相对平衡分母已保持") is not True
            or any(payload.get(name) != value for name, value in audit["汇总"].items())):
        raise ValueError("任11写盘摘要必须保留全部原始能源统计，不能改原相对分母")
    try:
        serialized = json.dumps({**payload, **verified}, ensure_ascii=False,
                                allow_nan=False, indent=2) + "\n"
    except (TypeError, ValueError) as error:
        raise ValueError("任11写盘摘要必须全部为有限JSON数值，不能含NaN或非数值对象") from error
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in (("指标明细.csv", audit["指标明细"]),
                       ("物理分解.csv", audit["物理分解"])):
        temporary = destination / (name + ".tmp")
        data.write_csv(temporary)
        os.replace(temporary, destination / name)
    for name, rows in (("原始能量.jsonl", audit["原始能量"]),
                       ("原始散度.jsonl", audit["原始散度"])):
        temporary = destination / (name + ".tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                      for row in rows), encoding="utf-8")
        os.replace(temporary, destination / name)
    temporary = destination / "汇总指标.json.tmp"
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, destination / "汇总指标.json")
    digests = {path.name: sha256_file(path) for path in sorted(destination.iterdir())
               if path.is_file()}
    (destination / "审计工件SHA256.json").write_text(
        json.dumps(digests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return digests


def load_task11_energy_model(
    qualified: Mapping[str, Any], device: torch.device,
) -> tuple[dict[str, Any], nn.Module]:
    original = Path(qualified["模型原件"])
    if not original.is_file() or sha256_file(original) != qualified["模型SHA256"]:
        raise ValueError("任11选中模型原件SHA在静态审计后已改变，禁止反序列化或写能源")
    view = torch.load(original, map_location="cpu", weights_only=True)
    model = task11_energy_model_from_view(view)
    return view, model.to(device).eval()


def _selected_observations(
    model: nn.Module, view: Mapping[str, Any], run: Path, device: torch.device,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader
    from sic_cu.train.multifidelity import _ir_dataset, _sensor_tensors
    from sic_cu.train.task04_joint import _lf_material_validation, _validation_selection
    from sic_cu.data.splits import build_power_splits

    powers = [115.2, 403.0, 630.5]
    data = _ir_dataset("validation", powers)
    sensors = _sensor_tensors(device, split="validation", powers_w=powers)
    if len(data) != 7272 or len(sensors[0]) != 752:
        raise ValueError("任11独审必须重算合法HF三功率全部7272 Top与752传感点")
    selected, modalities = _validation_selection(
        model, DataLoader(data, batch_size=2048, shuffle=False), sensors, device,
        load_yaml(run / "config_snapshot/training.yaml")["multifidelity_selection_weights"],
    )
    expected = view.get("validation_selection_score_c")
    if (type(expected) not in (int, float) or not math.isfinite(expected)
            or not math.isclose(selected, expected, rel_tol=0, abs_tol=1e-4)):
        raise ValueError("任11选中原模型的合法macro_v1独立重算与冻结视图得分不一致")
    splits = build_power_splits()
    if len(splits.simulation_validation) != 10:
        raise ValueError("任11LF独审仅准冻结的十个合法仿真验证功率")
    lf_metrics = _lf_material_validation(model, list(splits.simulation_validation), device)
    return {
        "合法HF验证功率_瓦": powers, "合法HF全部Top点": len(data),
        "合法HF全部传感点": len(sensors[0]),
        "合法HF原macro_v1选分独立重算_摄氏度": selected,
        "合法HF分模态独立重算": modalities,
        "合法LF验证功率_瓦": list(splits.simulation_validation),
        "合法LF逐材料节点与真实体积RMSE_摄氏度": lf_metrics,
        "本独立重算不回调训练或选模": True,
    }


def audit_task11_mlp_hf_energy(
    *, run: str | Path, output: str | Path, state: str, seed: int,
    registry: str | Path, registry_sha: str,
    source_tar: str | Path, source_tar_sha: str,
    lf_catalog: str | Path, lf_catalog_sha: str,
    hf_data_catalog: str | Path, hf_data_catalog_sha: str,
    device_name: str = "cuda",
) -> dict[str, Any]:
    from sic_cu.train.task11_mlp_hf_formal import (
        require_task11_mlp_hf_downstream_qualification,
    )

    destination = task11_energy_output(run, output, state, seed)
    identity = {
        "registry": registry, "registry_sha": registry_sha,
        "source_tar": source_tar, "source_tar_sha": source_tar_sha,
        "lf_catalog": lf_catalog, "lf_catalog_sha": lf_catalog_sha,
        "hf_data_catalog": hf_data_catalog, "hf_data_catalog_sha": hf_data_catalog_sha,
        "output": run, "seed": seed,
    }
    qualified = require_task11_mlp_hf_downstream_qualification(**identity, purpose=state)
    powers, times, orders = task11_fixed_energy_schedule(load_yaml(qualified["登记原件"]))
    before = {"模型SHA256": qualified["模型SHA256"],
              "完整原件SHA256": qualified["完整原件SHA256"]}
    device = require_task11_energy_cuda(device_name)
    view, model = load_task11_energy_model(qualified, device)
    if view.get("seed") != seed:
        raise ValueError("任11能源模型视图seed与完训身份不一致")
    run_path = Path(run)
    if not run_path.is_absolute():
        run_path = PROJECT_ROOT / run_path
    run_path = run_path.resolve()
    selected = _selected_observations(model, view, run_path, device)
    # Only the newly constructed read-only copy changes dtype and gradient flags.
    model.requires_grad_(False).to(dtype=torch.float64).eval()
    if any(value.dtype != torch.float64 for value in model.parameters()):
        raise ValueError("任11双阶能源只能用Float64模型只读副本")
    snapshot = run_path / "config_snapshot"
    audit = audit_schedule_energy(
        model, load_materials(str(snapshot / "materials.yaml")),
        load_resolved_boundary_conditions(str(snapshot / "boundary_conditions.yaml")),
        AxisymmetricGeometry.from_config(load_yaml(snapshot / "geometry.yaml")),
        powers_w=powers, times_s=times, orders=tuple(orders), device=device,
    )
    verified = verify_task11_energy_raw(audit)
    after = require_task11_mlp_hf_downstream_qualification(**identity, purpose=state)
    if before != {"模型SHA256": after["模型SHA256"],
                  "完整原件SHA256": after["完整原件SHA256"]}:
        raise ValueError("任11模型/日志/分段收据原件在能源审核中改变，不得写盘")
    summary = audit["汇总"]
    payload = {
        "中文说明": (
            "任11新MLP同seed真实LF和HF校正/全LF联合完训后，"
            "按事前固定六功率五时刻16/64阶独立审核best或final。"
            "原瓦数、原相对分母及两阶V-J-D全部保留，Float64仅为新只读副本。"
            "结果是当前模型名义热预算诊断，不证明原FEM有限盘热预算、"
            "原装置内部真实温度或工程安全；不用于训练回调、模型选择或替换B0。"
        ),
        "运行种子": seed, "运行臂": "新MLP_PiNN", "审核状态": state,
        "运行目录": str(run_path), "选择模型SHA256": qualified["模型SHA256"],
        "正式登记原件": str(registry), "正式登记SHA256": registry_sha,
        "源码归档原件": str(source_tar), "源码归档SHA256": source_tar_sha,
        "五LF身份清单SHA256": lf_catalog_sha,
        "HF开发数据清单SHA256": hf_data_catalog_sha,
        "完整原件SHA256": qualified["完整原件SHA256"],
        "本seed新LF最佳检查点SHA256": qualified["本seed新LF最佳检查点SHA256"],
        "本seedLF初始张量SHA256": qualified["本seedLF初始张量SHA256"],
        "本次合法HF_LF独立重算": selected,
        "功率_瓦": powers, "时刻_秒": times, "求积阶数": orders,
        "模型计算dtype": "float64", "只读原件审核前后SHA一致": True,
        "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
        "名义能量不证明原FEM热预算或内部温度真值": True,
        "工程安全阈值已建立": False, "结果不用于训练选模": True,
        "名义吸收归一筛查": {
            "均值阈值": 0.05, "95分位阈值": 0.10,
            "均值通过": summary["吸收功率归一宏均值"] <= 0.05,
            "95分位通过": summary["吸收功率归一95分位"] <= 0.10,
            "不代表工程安全": True,
        },
        **summary, **verified,
    }
    write_task11_energy_evidence(destination, audit, payload)
    return payload
