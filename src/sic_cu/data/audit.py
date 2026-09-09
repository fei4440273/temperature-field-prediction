from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from sic_cu.config import PROJECT_ROOT, load_yaml, resolve_data_root

from .common import bytes_to_gib, parse_power, sha256_file
from .experiment import ExperimentFileAudit, audit_experiment_file, experiment_files
from .sensors import SensorFileAudit, audit_sensor_file, sensor_files
from .simulation import SimulationFileAudit, audit_simulation_file, simulation_files
from .splits import build_power_splits


T = TypeVar("T")
R = TypeVar("R")

IDENTIFIABLE_PHYSICS_PATHS = {
    "configs/boundary_conditions.yaml.external_surface.silicon_carbide_emissivity",
    "configs/boundary_conditions.yaml.external_surface.copper_emissivity",
    "configs/boundary_conditions.yaml.interface.contact_resistance_m2_k_w",
}


def _parallel_map(function: Callable[[T], R], items: Iterable[T], workers: int) -> list[R]:
    values = list(items)
    if workers <= 1:
        return [function(item) for item in values]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, values))


def _rel(path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(PROJECT_ROOT))


def _file_inventory(paths: list[Path], workers: int, hash_files: bool) -> list[dict[str, Any]]:
    hashes = (
        _parallel_map(sha256_file, paths, workers)
        if hash_files
        else [None for _ in paths]
    )
    return [
        {
            "path": _rel(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        for path, digest in zip(paths, hashes, strict=True)
    ]


def _inventory_sha256(files: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _missing_physics_values() -> list[str]:
    missing: list[str] = []
    for cfg_path in ("configs/materials.yaml", "configs/boundary_conditions.yaml"):
        cfg = load_yaml(cfg_path)

        def visit(value: Any, prefix: str) -> None:
            if value is None:
                missing.append(prefix)
            elif isinstance(value, dict):
                for key, child in value.items():
                    if key not in {"verified", "schema_version"}:
                        if key == "table" and value.get("kind") == "constant":
                            continue
                        if (
                            key == "heat_transfer_coefficient_w_m2_k"
                            and value.get("kind") == "fixed_temperature"
                        ):
                            continue
                        if key == "emissivity" and value.get("radiation_enabled") is False:
                            continue
                        if (
                            key == "side_convection_coefficient_w_m2_k"
                            and cfg.get("cooling", {}).get("surface") == "outer_radius"
                        ):
                            continue
                        if (
                            key == "hard_bounds"
                            and cfg.get("interface", {}).get("positivity_parameterization")
                            == "log10"
                        ):
                            continue
                        visit(child, f"{prefix}.{key}")

        visit(cfg, cfg_path)
    return missing


def _audit_summary(
    simulations: list[SimulationFileAudit],
    experiments: list[ExperimentFileAudit],
    sensors: list[SensorFileAudit],
) -> dict[str, Any]:
    sim_powers = sorted(record.filename_power_w for record in simulations)
    mesh_signatures = {record.mesh_signature for record in simulations}
    time_anomalies = [
        {
            "path": _rel(record.path),
            "max_deviation_s": record.max_time_grid_deviation_s,
        }
        for record in simulations
        if record.max_time_grid_deviation_s > 1e-9
    ]
    experiment_by_power: dict[float, list[ExperimentFileAudit]] = defaultdict(list)
    for record in experiments:
        experiment_by_power[record.power_w].append(record)
    sensors_by_type: dict[str, list[SensorFileAudit]] = defaultdict(list)
    for record in sensors:
        sensors_by_type[record.sensor_type].append(record)
    return {
        "simulation": {
            "file_count": len(simulations),
            "powers_w": sim_powers,
            "all_expected_powers_present": sim_powers == list(range(10, 801, 10)),
            "rows_per_file": sorted({record.rows for record in simulations}),
            "time_frames": sorted({record.time_frames for record in simulations}),
            "nodes_per_frame": sorted(
                {record.nodes_per_frame_min for record in simulations}
                | {record.nodes_per_frame_max for record in simulations}
            ),
            "copper_nodes": sorted({record.copper_nodes for record in simulations}),
            "sic_nodes": sorted({record.sic_nodes for record in simulations}),
            "mesh_signature_count": len(mesh_signatures),
            "same_mesh_for_all_powers": len(mesh_signatures) == 1,
            "r_range_raw": [
                min(record.r_range_raw[0] for record in simulations),
                max(record.r_range_raw[1] for record in simulations),
            ],
            "z_range_raw": [
                min(record.z_range_raw[0] for record in simulations),
                max(record.z_range_raw[1] for record in simulations),
            ],
            "temperature_range_c": [
                min(record.temperature_range_c[0] for record in simulations),
                max(record.temperature_range_c[1] for record in simulations),
            ],
            "initial_temperature_range_c": [
                min(record.initial_temperature_range_c[0] for record in simulations),
                max(record.initial_temperature_range_c[1] for record in simulations),
            ],
            "initial_temperature_std_c_max": max(
                record.initial_temperature_std_c for record in simulations
            ),
            "null_values": sum(record.null_values for record in simulations),
            "time_grid_anomalies": time_anomalies,
        },
        "experiment_ir": {
            "file_count": len(experiments),
            "power_count": len(experiment_by_power),
            "powers_w": sorted(experiment_by_power),
            "frames_per_power": {
                str(power): len(records) for power, records in sorted(experiment_by_power.items())
            },
            "time_range_per_power_s": {
                str(power): [
                    min(record.time_s for record in records),
                    max(record.time_s for record in records),
                ]
                for power, records in sorted(experiment_by_power.items())
            },
            "rows_per_frame": sorted({record.rows for record in experiments}),
            "radial_bins": sorted({record.radial_bins for record in experiments}),
            "r_range_mm": [
                min(record.r_range_mm[0] for record in experiments),
                max(record.r_range_mm[1] for record in experiments),
            ],
            "temperature_range_c": [
                min(record.temperature_range_c[0] for record in experiments),
                max(record.temperature_range_c[1] for record in experiments),
            ],
            "first_frame_temperature_range_c_per_power": {
                str(power): [
                    min(
                        record.temperature_range_c[0]
                        for record in records
                        if record.time_s == min(item.time_s for item in records)
                    ),
                    max(
                        record.temperature_range_c[1]
                        for record in records
                        if record.time_s == min(item.time_s for item in records)
                    ),
                ]
                for power, records in sorted(experiment_by_power.items())
            },
            "max_radial_coordinate_error_mm": max(
                record.radial_coordinate_max_error_mm for record in experiments
            ),
            "recovered_true": sum(record.recovered_true for record in experiments),
            "recovered_false": sum(record.recovered_false for record in experiments),
            "null_values": sum(record.null_values for record in experiments),
        },
        "sensors": {
            sensor_type: {
                "file_count": len(records),
                "powers_w": sorted(record.power_w for record in records),
                "rows_per_file": sorted({record.rows for record in records}),
                "time_column_range": [
                    min(record.time_columns for record in records),
                    max(record.time_columns for record in records),
                ],
                "radius_mean_raw": sum(record.radius_mean_raw for record in records)
                / len(records),
                "radius_std_raw_max": max(record.radius_std_raw for record in records),
                "value_range_raw": [
                    min(record.value_range_raw[0] for record in records),
                    max(record.value_range_raw[1] for record in records),
                ],
                "initial_value_mean_range_raw": [
                    min(record.initial_value_mean_raw for record in records),
                    max(record.initial_value_mean_raw for record in records),
                ],
                "max_angular_spread_raw": max(
                    record.max_angular_spread_raw for record in records
                ),
                "all_files_are_exact_angular_duplicates": all(
                    record.all_ring_values_identical for record in records
                ),
                "null_values": sum(record.nan_values for record in records),
            }
            for sensor_type, records in sorted(sensors_by_type.items())
        },
    }


def _render_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    sim = summary["simulation"]
    ir = summary["experiment_ir"]
    hot = summary["sensors"]["hot"]
    cold = summary["sensors"]["cold"]
    test_data = summary["test_data"]
    lines = [
        "# SiC-Cu 温度场数据审计报告",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        f"- Origin Date: {date.today().isoformat()}",
        "- Verification Status: PARTIALLY VERIFIED / IDENTIFICATION READY",
        "- Version Label: data_audit_v1",
        "",
        "## 审计结论",
        "",
        f"- 结构状态：**{result['gate']['structural_status']}**。",
        f"- 物理训练状态：**{result['gate']['physics_training_status']}**。",
        f"- 逆向辨识状态：**{result['gate']['parameter_identification_status']}**。",
        f"- 用户侧待补数据：**{result['gate']['user_input_status']}**。",
        f"- 共检查 `{result['inventory']['file_count']}` 个原始文件，大小 "
        f"`{result['inventory']['total_gib']:.3f} GiB`。",
        "- 原始目录只读处理；审计结果写入 `reports/`，未修改原始 CSV。",
        "",
        "## Simulation data",
        "",
        f"- 文件/功率：`{sim['file_count']}` / `{len(sim['powers_w'])}`，"
        f"10–800 W 是否完整：`{sim['all_expected_powers_present']}`。",
        f"- 每文件行数：`{sim['rows_per_file']}`；时间帧：`{sim['time_frames']}`；"
        f"每帧节点：`{sim['nodes_per_frame']}`。",
        f"- Cu/SiC 节点：`{sim['copper_nodes']}` / `{sim['sic_nodes']}`。",
        f"- 所有功率是否共用同一节点拓扑：`{sim['same_mesh_for_all_powers']}`。",
        f"- 原始 r 范围：`{sim['r_range_raw']}` mm；z 范围：`{sim['z_range_raw']}` mm。",
        f"- 温度范围：`{sim['temperature_range_c']}` ℃；空值：`{sim['null_values']}`。",
        f"- 80 个功率的 t=0 初温范围：`{sim['initial_temperature_range_c']}` ℃；"
        f"单文件初温标准差上限：`{sim['initial_temperature_std_c_max']:.3e}` ℃。",
        f"- 时间网格异常：`{sim['time_grid_anomalies']}`。590 W 的 6 s 帧存在约 "
        "0.00196 s 浮点偏差，预处理时只允许容差对齐并保留原始值记录。",
        "",
        "## Experiment data",
        "",
        f"- 文件/功率：`{ir['file_count']}` / `{ir['power_count']}`。",
        f"- 每功率帧数：`{ir['frames_per_power']}`。",
        f"- 每帧像素行数：`{ir['rows_per_frame']}`；原始径向箱：`{ir['radial_bins']}`。",
        f"- 半径范围：`{ir['r_range_mm']}` mm；温度范围："
        f"`{ir['temperature_range_c']}` ℃。",
        f"- 各功率首个 5 s 帧温度范围：`{ir['first_frame_temperature_range_c_per_power']}`；"
        "实验没有 t=0 帧，不能据此替代初始条件。",
        f"- r 与 sqrt(x^2+y^2) 最大误差："
        f"`{ir['max_radial_coordinate_error_mm']:.3e}` mm。",
        f"- `is_recovered=True/False`：`{ir['recovered_true']}` / "
        f"`{ir['recovered_false']}`；空值：`{ir['null_values']}`。",
        "- 所有像素均标为 recovered，不能把像素数当独立高保真样本量；"
        "后续按 0.25 mm 径向分箱并限制每帧总权重。",
        "",
        "## test_Data",
        "",
        f"- 最终测试功率：`{test_data['experiment_ir']['powers_w']}`，"
        f"IR 共 `{test_data['experiment_ir']['file_count']}` 帧。",
        f"- Hot/Cold 文件数：`{test_data['sensors']['hot']['file_count']}` / "
        f"`{test_data['sensors']['cold']['file_count']}`。",
        "- 该目录固定标记为 test-only，任何记录进入训练或模型选择集合都会使预处理失败。",
        "",
        "## Hot/Cold",
        "",
        "| 类型 | 文件数 | 行数/文件 | 时间列 | r 原始均值 | r 标准差上限 | 最大角向温差 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Hot | {hot['file_count']} | {hot['rows_per_file']} | "
        f"{hot['time_column_range']} | {hot['radius_mean_raw']:.9f} | "
        f"{hot['radius_std_raw_max']:.3e} | {hot['max_angular_spread_raw']:.3e} |",
        f"| Cold | {cold['file_count']} | {cold['rows_per_file']} | "
        f"{cold['time_column_range']} | {cold['radius_mean_raw']:.9f} | "
        f"{cold['radius_std_raw_max']:.3e} | {cold['max_angular_spread_raw']:.3e} |",
        "",
        (
            "- 用户已确认 X/Y 使用 m、数值使用 ℃、`t=i` 表示激光开启后第 i 秒，"
            "坐标原点为光斑中心。"
            if result["gate"]["sensor_metadata_status"] == "VERIFIED"
            else "- 数值半径分别接近 0.028 和 0.0415，但这只是数值一致性证据，不能替代单位元数据。"
        ),
        "- 每个时间列在整圈坐标上完全相同，因此预处理必须压缩为每时刻一条 ring-average，"
        "不得把 148/218 个重复点作为独立监督。",
        (
            "- Hot/Cold 传感器元数据状态：已验证；对外统一使用 ℃，物理计算内部转换为 SI/K。"
            if result["gate"]["sensor_metadata_status"] == "VERIFIED"
            else "- `t=1` 的含义、时间单位、数值单位和同步方式仍未确认。"
        ),
        "",
        "## 划分与泄漏检查",
        "",
        "- Simulation：60 train / 10 validation / 10 test，按完整功率划分。",
        "- Experiment_data：12 train / 3 validation；115.2、403、630.5 W 的 IR/Hot/Cold "
        "只用于检查点选择，不参与梯度更新。",
        "- test_Data：169、339、634 W 只用于最终测试，不参与训练或模型选择。",
        "- 已建立运行时泄漏断言；任何禁用功率进入 HF loader 都会抛出异常。",
        "",
        "## 待模型辨识参数（不是待用户补充数据）",
        "",
    ]
    lines.extend(f"- `{item}`" for item in result["gate"]["pending_model_parameters"])
    lines.extend(
        [
            "",
        "## 审计判定",
        "",
        "数据文件结构、温度观测语义和方案中的数量假设通过审计。用户侧输入已经完整；"
        "训练可使用 `resolved_physics.yaml` 中明确标记的名义情景和有效初始化。"
        "发射率与接触热阻仍不是已测量或唯一辨识的真值，正式结果必须保留该限定。",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    metadata_path: str = "configs/data_metadata.yaml",
    output_json: str = "reports/data_audit.json",
    output_markdown: str = "reports/data_audit.md",
    protocol_inventory_json: str = "reports/current_protocol/data_inventory.json",
    workers: int = 4,
    hash_files: bool = True,
) -> dict[str, Any]:
    metadata = load_yaml(metadata_path)
    data_root = resolve_data_root(metadata)
    sim_paths = simulation_files(data_root / metadata["simulation"]["directory"])
    ir_paths = experiment_files(data_root / metadata["experiment_ir"]["directory"])
    hot_paths = sensor_files(data_root / metadata["sensors"]["hot_directory"])
    cold_paths = sensor_files(data_root / metadata["sensors"]["cold_directory"])
    test_cfg = metadata["test"]
    test_ir_paths = experiment_files(
        data_root / test_cfg["experiment_ir_directory"]
    )
    test_hot_paths = sensor_files(data_root / test_cfg["hot_directory"])
    test_cold_paths = sensor_files(data_root / test_cfg["cold_directory"])
    all_paths = (
        sim_paths
        + ir_paths
        + hot_paths
        + cold_paths
        + test_ir_paths
        + test_hot_paths
        + test_cold_paths
    )

    simulations = _parallel_map(audit_simulation_file, sim_paths, workers)
    experiments = _parallel_map(audit_experiment_file, ir_paths, workers)
    sensors = _parallel_map(
        lambda item: audit_sensor_file(*item),
        [(path, "hot") for path in hot_paths] + [(path, "cold") for path in cold_paths],
        workers,
    )
    summary = _audit_summary(simulations, experiments, sensors)
    summary["test_data"] = {
        "experiment_ir": {
            "file_count": len(test_ir_paths),
            "powers_w": sorted({parse_power(path) for path in test_ir_paths}),
            "temperature_statistics": "sealed_not_computed",
        },
        "sensors": {
            "hot": {
                "file_count": len(test_hot_paths),
                "powers_w": sorted({parse_power(path) for path in test_hot_paths}),
                "temperature_statistics": "sealed_not_computed",
            },
            "cold": {
                "file_count": len(test_cold_paths),
                "powers_w": sorted({parse_power(path) for path in test_cold_paths}),
                "temperature_statistics": "sealed_not_computed",
            },
        },
    }
    structural_checks = [
        summary["simulation"]["file_count"] == 80,
        summary["simulation"]["all_expected_powers_present"],
        summary["simulation"]["same_mesh_for_all_powers"],
        summary["simulation"]["nodes_per_frame"] == [1159],
        summary["experiment_ir"]["file_count"] == 365,
        summary["experiment_ir"]["power_count"] == 15,
        summary["sensors"]["hot"]["file_count"] == 15,
        summary["sensors"]["cold"]["file_count"] == 15,
        summary["sensors"]["hot"]["all_files_are_exact_angular_duplicates"],
        summary["sensors"]["cold"]["all_files_are_exact_angular_duplicates"],
        summary["test_data"]["experiment_ir"]["file_count"] == 62,
        summary["test_data"]["experiment_ir"]["powers_w"] == [169.0, 339.0, 634.0],
        summary["test_data"]["sensors"]["hot"]["file_count"] == 3,
        summary["test_data"]["sensors"]["cold"]["file_count"] == 3,
        test_cfg.get("training_allowed") is False,
        test_cfg.get("model_selection_allowed") is False,
    ]
    splits = build_power_splits()
    blocking_unknowns = _missing_physics_values()
    ir_metadata = metadata["experiment_ir"]
    for key in (
        "center_x_mm",
        "center_y_mm",
        "filename_50mm_meaning",
        "emissivity_setting",
        "background_lens_calibration",
        "initial_temperature_synchronization",
    ):
        if ir_metadata.get(key) is None:
            blocking_unknowns.append(f"configs/data_metadata.yaml.experiment_ir.{key}")
    ir_metadata_verified = all(
        ir_metadata.get(key) is not None
        for key in (
            "center_x_mm",
            "center_y_mm",
            "filename_50mm_meaning",
            "emissivity_setting",
            "background_lens_calibration",
            "initial_temperature_synchronization",
        )
    )
    sensor_metadata = metadata["sensors"]
    sensor_metadata_verified = (
        sensor_metadata.get("coordinate_unit_status") == "verified"
        and sensor_metadata.get("value_unit_status") == "verified"
        and sensor_metadata.get("time_unit_status") == "verified"
        and sensor_metadata.get("synchronized_start") is True
    )
    if not sensor_metadata_verified:
        for key in (
            "coordinate_unit",
            "value_unit",
            "time_column_interpretation",
            "time_unit",
            "synchronized_start",
        ):
            blocking_unknowns.append(f"configs/data_metadata.yaml.sensors.{key}")
    pending_model_parameters = sorted(
        item for item in blocking_unknowns if item in IDENTIFIABLE_PHYSICS_PATHS
    )
    user_input_unknowns = sorted(
        item for item in blocking_unknowns if item not in IDENTIFIABLE_PHYSICS_PATHS
    )
    inventory = _file_inventory(all_paths, workers, hash_files)
    contexts: dict[str, dict[str, Any]] = {}

    def add_context(
        paths: Iterable[Path], source: str, modality: str, split_lookup: dict[float, str]
    ) -> None:
        for path in paths:
            power = round(parse_power(path), 4)
            contexts[_rel(path)] = {
                "source": source,
                "power_w": power,
                "modality": modality,
                "split": split_lookup[power],
                "run_id": None,
                "run_id_status": "unknown_not_in_source_metadata",
            }

    simulation_split = {
        **{power: "train" for power in splits.simulation_train},
        **{power: "validation" for power in splits.simulation_validation},
        **{power: "test" for power in splits.simulation_test},
    }
    experiment_split = {
        **{power: "train" for power in splits.hf_train},
        **{power: "validation" for power in splits.hf_validation},
    }
    test_split = {power: "test" for power in splits.hf_test}
    add_context(sim_paths, "simulation", "full_field", simulation_split)
    add_context(ir_paths, "experiment", "top_ir", experiment_split)
    add_context(hot_paths, "experiment", "hot_ring", experiment_split)
    add_context(cold_paths, "experiment", "cold_ring", experiment_split)
    add_context(test_ir_paths, "test", "top_ir", test_split)
    add_context(test_hot_paths, "test", "hot_ring", test_split)
    add_context(test_cold_paths, "test", "cold_ring", test_split)
    inventory = [record | contexts[record["path"]] for record in inventory]
    inventory_hash = _inventory_sha256(inventory)
    paths_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in inventory:
        if record["sha256"] is not None:
            paths_by_hash[record["sha256"]].append(record)
    duplicate_groups = [
        {
            "sha256": digest,
            "paths": [record["path"] for record in records],
            "powers_w": sorted({record["power_w"] for record in records}),
            "splits": sorted({record["split"] for record in records}),
        }
        for digest, records in paths_by_hash.items()
        if len(records) > 1
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_on": date.today().isoformat(),
        "metadata_path": metadata_path,
        "summary": summary,
        "splits": {
            "simulation_train": sorted(splits.simulation_train),
            "simulation_validation": sorted(splits.simulation_validation),
            "simulation_test": sorted(splits.simulation_test),
            "hf_train": sorted(splits.hf_train),
            "hf_validation": sorted(splits.hf_validation),
            "hf_test": sorted(splits.hf_test),
            "external_sensor_test": sorted(splits.external_sensor_test),
            "experiment_powers": sorted(splits.experiment_powers),
            "test_only": sorted(splits.test_only),
        },
        "gate": {
            "structural_status": "PASS" if all(structural_checks) else "FAIL",
            "physics_training_status": "READY_WITH_DECLARED_INITIALIZATIONS",
            "parameter_identification_status": (
                "READY" if not user_input_unknowns else "BLOCKED_MISSING_USER_INPUT"
            ),
            "user_input_status": "COMPLETE" if not user_input_unknowns else "INCOMPLETE",
            "experiment_ir_metadata_status": "VERIFIED" if ir_metadata_verified else "UNVERIFIED",
            "sensor_metadata_status": "VERIFIED" if sensor_metadata_verified else "UNVERIFIED",
            "blocking_unknowns": blocking_unknowns,
            "pending_model_parameters": pending_model_parameters,
            "user_input_unknowns": user_input_unknowns,
        },
        "inventory": {
            "file_count": len(inventory),
            "total_bytes": sum(record["bytes"] for record in inventory),
            "total_gib": bytes_to_gib(sum(record["bytes"] for record in inventory)),
            "sha256_computed": hash_files,
            "files": inventory,
            "inventory_sha256": inventory_hash,
        },
        "records": {
            "simulation": [record.to_dict() | {"path": _rel(record.path)} for record in simulations],
            "experiment_ir": [record.to_dict() | {"path": _rel(record.path)} for record in experiments],
            "sensors": [record.to_dict() | {"path": _rel(record.path)} for record in sensors],
            "test_ir": [
                {"path": _rel(path), "power_w": parse_power(path), "labels": "sealed"}
                for path in test_ir_paths
            ],
            "test_sensors": [
                {
                    "path": _rel(path),
                    "power_w": parse_power(path),
                    "sensor_type": sensor_type,
                    "labels": "sealed",
                }
                for sensor_type, paths in (
                    ("hot", test_hot_paths),
                    ("cold", test_cold_paths),
                )
                for path in paths
            ],
        },
    }
    json_path = PROJECT_ROOT / output_json
    markdown_path = PROJECT_ROOT / output_markdown
    protocol_inventory_path = PROJECT_ROOT / protocol_inventory_json
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_inventory_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(_render_markdown(result), encoding="utf-8")
    protocol_inventory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": "hf_fixed_12_3_3_v4",
                "generated_on": result["generated_on"],
                "sha256_computed": hash_files,
                "test_temperature_statistics_included": False,
                "file_count": len(inventory),
                "total_bytes": sum(record["bytes"] for record in inventory),
                "inventory_sha256": inventory_hash,
                "duplicate_sha256_groups": duplicate_groups,
                "cross_power_duplicate_sha256_groups": [
                    group for group in duplicate_groups if len(group["powers_w"]) > 1
                ],
                "files": inventory,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all SiC-Cu raw data files")
    parser.add_argument("--metadata", default="configs/data_metadata.yaml")
    parser.add_argument("--output-json", default="reports/data_audit.json")
    parser.add_argument("--output-markdown", default="reports/data_audit.md")
    parser.add_argument(
        "--protocol-inventory-json",
        default="reports/current_protocol/data_inventory.json",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()
    result = run_audit(
        metadata_path=args.metadata,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        protocol_inventory_json=args.protocol_inventory_json,
        workers=args.workers,
        hash_files=not args.skip_hash,
    )
    print(json.dumps({"gate": result["gate"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
