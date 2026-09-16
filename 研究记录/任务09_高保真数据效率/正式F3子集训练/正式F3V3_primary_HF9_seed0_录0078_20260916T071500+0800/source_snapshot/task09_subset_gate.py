"""Task-09 read-only nested-subset source gate; it cannot produce formal scores."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping

import polars as pl
import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.processed import processed_ir_path
from sic_cu.data.splits import (
    assert_no_hf_leakage, build_power_splits, resolve_hf_training_subset,
)
from sic_cu.train.task04_joint import PROJECTION_NAMES
from sic_cu.train.task07_source import (
    MANIFEST_SHA256, SEEDS, Task07Initialization, Task07Source,
    _lf_tensor_sha256, fork_task07_initialization, validate_task07_sources,
)


REGISTRATION_PATH = Path("研究记录/任务09_高保真数据效率/额外功率覆盖序列前登记.yaml")
REGISTRATION_SHA256 = "206b91e3ea981ec81c0ed1074ea4688a16567e32cab9098b8c134ae683becd6a"
SPLITS_SHA256 = "89a1ebd7df4512ea1ffc1704799cc120182139e12d08e35d1dcb800ae9cc296e"
IR_METADATA_PATH = processed_ir_path("train")
SENSOR_METADATA_PATH = PROJECT_ROOT / "data/processed/sensor_ring_raw.parquet"
PRIMARY_LOCKED = {
    3: (55.0, 364.3, 729.0),
    6: (55.0, 216.8, 364.3, 494.2, 593.5, 729.0),
    9: (55.0, 216.8, 254.5, 364.3, 430.0, 494.2, 558.5, 593.5, 729.0),
}
TASK07_BUDGET_REGISTRATION = PROJECT_ROOT / (
    "研究记录/任务07_正式五种子重训/有效运行配置_实现修复后.yaml"
)


def validate_task09_budget_contract() -> None:
    """Check immutable source budget assumptions before projecting any subset costs."""
    training = load_yaml("configs/training.yaml")
    locked = load_yaml(TASK07_BUDGET_REGISTRATION)
    budget = locked["固定训练合同"]
    if (training["multifidelity"]["joint_simulation_samples_per_power"] != 2048
            or budget["LF每训练功率回放点"] != 2048
            or budget["HF观测batch大小"] != 2048):
        raise ValueError("任09LF2048真实训练点及HF观测batch2048预算须保持任07合同")
    optimizer = training["optimizer"]
    if (training["seeds"] != list(range(5))
            or optimizer["name"] != "adamw"
            or float(optimizer["learning_rate"]) != 0.001
            or float(optimizer["joint_learning_rate"]) != 0.0001
            or float(optimizer["weight_decay"]) != 0.000001
            or training["epochs"]["high_fidelity"] != 1500
            or training["epochs"]["joint"] != 500
            or training["early_stopping"] != {"patience": 200, "min_delta": 0.0001}
            or training["loss_weights"] != {
                "low_fidelity": 1.0, "ir": 1.0, "sensor_absolute": 5.0,
                "sensor_delta": 1.0, "pde": 1.0, "boundary": 1.0,
                "initial": 1.0, "interface": 1.0,
            }
            or budget["独立物理每轮配点"] != 256
            or budget["LF低保真回放功率数"] != 60
            or budget["HF合法验证功率数"] != 3
            or sorted(budget["受限联合仅LF四末投影"]) != sorted(PROJECTION_NAMES)):
        raise ValueError("任09原E0共同训练预算/名义物理256/四末投影合同发生变化")
    if (training["selection_metric_version"] != "macro_v1"
            or training["multifidelity_selection_weights"] != {
                "ir_rmse": 1.0, "sensor_absolute_rmse": 0.2,
                "sensor_delta_rmse": 1.0,
            }
            or budget["HF合法选分"] != "macro_v1"
            or budget["每10轮合法验证"] is not True):
        raise ValueError("任09合法选分只许共同macro_v1及原每10轮固定HF3验证")


def _power(value: Any) -> float:
    power = round(float(value), 4)
    if not math.isfinite(power):
        raise ValueError("任09功率登记含非有限数值")
    return power


def _registered_powers(raw: Any, count: int) -> tuple[float, ...]:
    if not isinstance(raw, list):
        raise ValueError("任09嵌套功率须来自登记YAML的显式数组")
    values = tuple(sorted(_power(value) for value in raw))
    if len(values) != count or len(set(values)) != count:
        raise ValueError("任09功率子集实际数量或唯一性与预登记不符")
    return values


def load_task09_registry() -> dict[str, dict[int, tuple[float, ...]]]:
    """Load original immutable YAML; reject changed bytes before inspecting its contents."""
    if (sha256_file(PROJECT_ROOT / REGISTRATION_PATH) != REGISTRATION_SHA256
            or sha256_file(PROJECT_ROOT / "configs/splits.yaml") != SPLITS_SHA256):
        raise ValueError("任09功率前登记或原HF/LF split原件SHA256发生变化")
    cfg = load_yaml(REGISTRATION_PATH)
    splits = build_power_splits()
    if (cfg.get("schema_version") != 1
            or cfg.get("source_split") != "configs/splits.yaml"
            or cfg.get("source_subset") != "high_fidelity.training_powers_w"
            or set(cfg.get("extra_coverages", {})) != {"left_center", "right_center"}
            or len(splits.simulation_train) != 60
            or len(splits.simulation_validation) != 10
            or len(splits.simulation_test) != 10
            or len(splits.hf_train) != 12
            or len(splits.hf_validation) != 3
            or len(splits.hf_test) != 3):
        raise ValueError("任09前登记结构或原HF12/3/3与LF60/10/10协议失配")
    full = _registered_powers(cfg.get("canonical_train_powers_w"), 12)
    if set(full) != splits.hf_train:
        raise ValueError("任09全量HF训练功率与原split不一致")
    registered = {"primary": cfg["primary_from_total_plan"]}
    registered.update(cfg["extra_coverages"])
    arms = {}
    for name, raw in registered.items():
        if not isinstance(raw, dict):
            raise ValueError("任09三条序列需要YAML映射")
        sequence = {size: _registered_powers(raw[f"{key}_w"], size)
                    for size, key in ((3, "three"), (6, "six"), (9, "nine"))}
        sequence[12] = full
        if (not set(sequence[3]) < set(sequence[6]) < set(sequence[9]) < set(full)
                or any(powers[0] != 55.0 or powers[-1] != 729.0
                       for powers in sequence.values())):
            raise ValueError("任09每条功率序列必须真实满足3⊂6⊂9⊂12和双端点")
        for powers in sequence.values():
            resolve_hf_training_subset(powers, splits)
        if name == "primary" and any(sequence[size] != PRIMARY_LOCKED[size]
                                     for size in (3, 6, 9)):
            raise ValueError("任09主功率序列与总台账预锁数字不一致")
        arms[name] = sequence
    return arms


def read_task09_metadata() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Project only split/power/modality columns, never temperature or test files."""
    ir = pl.scan_parquet(IR_METADATA_PATH).select("split", "power_w").collect()
    sensor = pl.scan_parquet(SENSOR_METADATA_PATH).select(
        "split", "power_w", "sensor_type"
    ).collect()
    return ir, sensor


def _metadata_counts(
    frame: pl.DataFrame, split: str, allowed: frozenset[float],
    *, modality: str | None = None,
) -> dict[float, int]:
    filtered = frame.filter(pl.col("split") == split)
    if modality is not None:
        filtered = filtered.filter(pl.col("sensor_type") == modality)
    counts = Counter(_power(value) for value in filtered["power_w"].to_list())
    if set(counts) != allowed or any(count <= 0 for count in counts.values()):
        raise ValueError(f"任09 {split} {modality or 'IR'}功率来源缺失/额外泄漏，须完整合法集合")
    return dict(counts)


def audit_task09_observations(
    arms: Mapping[str, Mapping[int, tuple[float, ...]]],
    ir_metadata: pl.DataFrame | None = None,
    sensor_metadata: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Count actual IR and both rings per subset from metadata, without label reads."""
    if ir_metadata is None or sensor_metadata is None:
        if ir_metadata is not None or sensor_metadata is not None:
            raise ValueError("任09只可同时提供IR和传感类型完整元数据")
        ir_metadata, sensor_metadata = read_task09_metadata()
    if (set(ir_metadata.columns) != {"split", "power_w"}
            or set(sensor_metadata.columns) != {"split", "power_w", "sensor_type"}
            or set(ir_metadata["split"].to_list()) != {"train", "validation"}
            or set(sensor_metadata["split"].to_list()) != {"train", "validation"}
            or set(sensor_metadata["sensor_type"].to_list()) != {"hot", "cold"}):
        raise ValueError("任09只接受train/validation元数据，拒绝旧test温度、其它列/环模态")
    splits = build_power_splits()
    ir_train = _metadata_counts(ir_metadata, "train", splits.hf_train)
    sensor_train = {modality: _metadata_counts(sensor_metadata, "train", splits.hf_train,
                                               modality=modality)
                    for modality in ("hot", "cold")}
    ir_validation = _metadata_counts(ir_metadata, "validation", splits.hf_validation)
    sensor_validation = {
        modality: _metadata_counts(sensor_metadata, "validation", splits.hf_validation,
                                   modality=modality)
        for modality in ("hot", "cold")
    }
    result = {"validation": {
        "powers_w": sorted(splits.hf_validation),
        "ir_rows": sum(ir_validation.values()),
        "sensor_rows": {"Hot": sum(sensor_validation["hot"].values()),
                        "Cold": sum(sensor_validation["cold"].values())},
    }, "arms": {}}
    for name, levels in arms.items():
        result["arms"][name] = {}
        for size, powers in levels.items():
            selected = frozenset(powers)
            if len(selected) != size or not selected <= splits.hf_train:
                raise ValueError("任09训练功率不得借合法验证或旧TEST进入")
            assert_no_hf_leakage({"IR": selected, "Hot": selected, "Cold": selected},
                                 splits.hf_validation | splits.hf_test | splits.external_sensor_test)
            ir_rows = sum(ir_train[power] for power in powers)
            result["arms"][name][size] = {
                "powers_w": list(powers), "ir_rows": ir_rows,
                "ir_batches_2048": math.ceil(ir_rows / 2048),
                "sensor_rows": {"Hot": sum(sensor_train["hot"][power] for power in powers),
                                "Cold": sum(sensor_train["cold"][power] for power in powers)},
            }
    return result


def task09_budget(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Forecast obligations; this reports no actual optimization steps or metrics."""
    batches = int(sample["ir_batches_2048"])
    rows = int(sample["ir_rows"])
    sensors = {name: int(sample["sensor_rows"][name]) for name in ("Hot", "Cold")}
    if (not 1 <= batches <= 15 or batches != math.ceil(rows / 2048)
            or rows <= 0 or any(count <= 0 for count in sensors.values())):
        raise ValueError("任09HF行数/batch/双环元数据不合格")
    quotient, remainder = divmod(60, batches)
    groups = [quotient + (index < remainder) for index in range(batches)]
    return {
        "性质": "训练前来源预测，非实际训练消耗",
        "HF顶部IR真实训练行": rows, "HF观测batch2048": batches,
        "HF训练传感器点_每批全量同步": sum(sensors.values()),
        "HF训练传感器点_一轮预测": batches * sum(sensors.values()),
        "HF验证完整功率数": 3, "物理每轮配点": 256,
        "物理独立更新步_一轮预测": 1,
        "HF观测优化步_一轮预测": batches,
        "LF固定训练功率数": 60,
        "LF每训练功率真实配点": 2048,
        "LF联合需真实回放点": 60 * 2048,
        "LF联合需真实回放batch": 60,
        "LF四批按旧HF更新可消费批次": batches * 4,
        "LF旧四批方式缺口": 60 - batches * 4,
        "LF旧四批方式可直接执行": batches == 15,
        "LF60按实际HF批数公平分组_仅预算预测": groups,
        "LF仅四末投影名称": sorted(PROJECTION_NAMES),
        "子集GPU训练已运行": False,
    }


def load_task09_sources() -> dict[int, Task07Source]:
    """Read historical HF architecture/provenance only; never reuse HF model tensors."""
    load_task09_registry()
    validate_task09_budget_contract()
    sources = validate_task07_sources()
    if set(sources) != SEEDS or len({src.lf_tensor_sha256 for src in sources.values()}) != 5:
        raise ValueError("任09必须保留五种子各自原LF张量与来源身份")
    return sources


def fork_task09_f3_start(
    sources: Mapping[int, Task07Source], seed: int, name: str, size: int,
    device: torch.device = torch.device("cpu"),
) -> Task07Initialization:
    """New seed-paired LF+HF E0 object; formal GPU is deliberately not an entry point."""
    if device.type != "cpu":
        raise ValueError("任09CPU来源门禁不启动GPU正式训练")
    registered = load_task09_registry()
    if name not in registered or size not in registered[name] or seed not in SEEDS:
        raise ValueError("任09训练序列、HF功率数量或五种子身份不在原前登记")
    start = fork_task07_initialization(sources, seed, "E0", device)
    source = sources[seed]
    if (start.selected_for_training or start.arm != "E0"
            or start.optimizer.state_dict()["state"]
            or start.model.correction[0].in_features != 6
            or _lf_tensor_sha256(start.model.low_fidelity_model.state_dict())
               != source.lf_tensor_sha256
            or any(parameter.requires_grad for parameter in
                   start.model.low_fidelity_model.parameters())
            or any(not parameter.requires_grad for parameter in
                   start.model.correction.parameters())):
        raise ValueError("任09不可从12HF权重/AdamW启动：只用同seed原LF和新HF空AdamW")
    return start


def _archive_before_entry(output: Path) -> dict[str, str]:
    """Commit an immutable-by-convention source copy before any observation scan."""
    output = output.resolve()
    try:
        output.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("任09入场工件只能新建于本项目目录") from exc
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    files = {
        "task09_subset_gate.py": PROJECT_ROOT / "src/sic_cu/train/task09_subset_gate.py",
        "34_check_task09_subset_gate.py": PROJECT_ROOT / "scripts/34_check_task09_subset_gate.py",
        "test_task09_subset_gate.py": PROJECT_ROOT / "tests/test_task09_subset_gate.py",
        "额外功率覆盖序列前登记.yaml": PROJECT_ROOT / REGISTRATION_PATH,
        "splits.yaml": PROJECT_ROOT / "configs/splits.yaml",
        "training.yaml": PROJECT_ROOT / "configs/training.yaml",
        "任07有效运行配置.yaml": PROJECT_ROOT / "研究记录/任务07_正式五种子重训/有效运行配置_实现修复后.yaml",
        "V4_B0来源清单.yaml": PROJECT_ROOT / "reports/development_v4/baseline_manifest.yaml",
    }
    hashes = {}
    for name, original in files.items():
        archived = snapshot / name
        original_sha = sha256_file(original)
        shutil.copyfile(original, archived)
        if sha256_file(archived) != original_sha:
            raise ValueError("任09源码事前原件归档SHA不一致，不继续CPU来源核查")
        hashes[name] = original_sha
    (output / "源码事前SHA256.json").write_text(
        json.dumps({"性质": "只读CPU元数据扫描和LF初态前的源码归档",
                    "源码与配置原件SHA256": hashes}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return hashes


def run_task09_cpu_entry(
    output_directory: str | Path, *, name: str = "primary", size: int = 3,
) -> dict[str, Any]:
    """One source-only report: no training, model selection, test labels or GPU."""
    arms = load_task09_registry()
    if name not in arms or size not in arms[name]:
        raise ValueError("任09仅三条原登记序列的3/6/9/12功率可入场")
    destination = Path(output_directory)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    snapshot_sha = _archive_before_entry(destination)
    validate_task09_budget_contract()
    observed = audit_task09_observations(arms)
    sources = load_task09_sources()
    from sic_cu.train.task08_f1_hf_only import audit_f1_physics_inputs

    f1 = audit_f1_physics_inputs()
    if f1["可作为纯HF-only物理输入"] is not False:
        raise ValueError("任09F1纯HF-only独立接触/对流物理来源未明确关闭，不训练F1")
    paired_lf = {}
    for seed in sorted(sources):
        initial = fork_task09_f3_start(sources, seed, name, size)
        paired_lf[seed] = {
            "原LF检查点SHA256": sources[seed].lf_checkpoint_sha256,
            "原LF全张量SHA256": sources[seed].lf_tensor_sha256,
            "当前LF全张量SHA256": _lf_tensor_sha256(
                initial.model.low_fidelity_model.state_dict()),
            "新HF校正器六列": initial.model.correction[0].in_features == 6,
            "新AdamW空状态": not bool(initial.optimizer.state_dict()["state"]),
            "LF全量参数冻结": all(not parameter.requires_grad for parameter in
                               initial.model.low_fidelity_model.parameters()),
            "CUDA随机源可冒正式GPU": False,
        }
        del initial
    forecasts = {
        arm: {count: task09_budget(sample)
              for count, sample in levels.items()}
        for arm, levels in observed["arms"].items()
    }
    report = {
        "资格": "CPU来源入场；无正式训练、误差或精度曲线",
        "选定臂": name, "选定HF功率数": size,
        "预登记原件SHA256": REGISTRATION_SHA256,
        "split原件SHA256": SPLITS_SHA256,
        "V4_B0来源清单SHA256": MANIFEST_SHA256,
        "源码事前快照SHA256": snapshot_sha,
        "LF固定训练功率数": len(build_power_splits().simulation_train),
        "HF合法验证固定功率数": len(build_power_splits().hf_validation),
        "五种子配对LF初始张量SHA256": {
            seed: row["原LF全张量SHA256"] for seed, row in paired_lf.items()},
        "五种子CPU初态": paired_lf,
        "三序列逐臂实际观测元数据": observed,
        "三序列逐臂每轮预测预算": forecasts,
        "F1纯HF-only正式物理入场": False,
        "F1限制": f1["限制"],
        "子集训练器是否已对LF60公平分组及真实消费核验": False,
        "已执行HF观测优化步": 0,
        "已执行名义物理更新步": 0,
        "已执行LF联合回放点": 0,
        "已读固定测试温度": False,
        "12功率复用限制": "任07正式冻结后仅复用合法观测F3结果；本门禁不加载12HF校正权重/优化器",
    }
    (destination / "只读入场与预算核查.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    markdown = [
        "# 任09三条子集CPU训练来源入场与每轮前瞻预算",
        "",
        "本报告只核查原前登记、IR/Hot/Cold的split/功率/模态元数据及五seed配对LF新HF第0轮；"
        "没有HF优化、正式GPU训练、验证误差、正式误差曲线或固定测试温度读取。",
        "",
        f"原功率前登记SHA256：`{REGISTRATION_SHA256}`；原split SHA256：`{SPLITS_SHA256}`。",
        f"本次源码和关键配置在扫描前原字节快照：`source_snapshot/`，完整SHA见`源码事前SHA256.json`。",
        "",
        "| 序列 | HF工况 | 真IR训练行 | HF批2048 | Hot行 | Cold行 | 每HF批同步传感行 | "
        "旧4LF批法缺额外LF批 | LF60公平分组仅预测 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm, levels in observed["arms"].items():
        for count, sample in sorted(levels.items()):
            budget = forecasts[arm][count]
            markdown.append(
                f"| {arm} | {count} | {sample['ir_rows']} | {sample['ir_batches_2048']} | "
                f"{sample['sensor_rows']['Hot']} | {sample['sensor_rows']['Cold']} | "
                f"{budget['HF训练传感器点_每批全量同步']} | "
                f"{budget['LF旧四批方式缺口']} | "
                f"{budget['LF60按实际HF批数公平分组_仅预算预测']} |"
            )
    markdown.extend([
        "",
        "HF合法验证仍为115.2/403/630.5W全3功率，顶部IR及热/冷双环始终完整；"
        f"合法验证IR原始行{observed['validation']['ir_rows']}，"
        f"Hot/Cold各{observed['validation']['sensor_rows']['Hot']}/"
        f"{observed['validation']['sensor_rows']['Cold']}行。LF原训练60/合法验证10/测试10功率不缩减。",
        "",
        "一轮预算来源预测：每HF批完整同步环温，两材料LF每训练功率2048真实点；"
        "HF独立名义物理256配点和一次更新；联合LF须60×2048真实点，"
        "且只许四末投影。小子集直接沿用任07每HF批回放4个LF批会遗漏表中批数；"
        "公平分组只是来源预测，须独立子集trainer在真实GPU训练逐批消费并核对累计日志后才有正式资格。",
        "",
        "五seed使用V4历史HF最佳检查点只读架构/协议及LF配对交叉核验；"
        "从不装入历史或任07十二功率HF校正张量、HF教师、旧AdamW或旧随机源。"
        "每seed原LF全张量保持一致并冻结，HF六输入校正器独立随机新建、AdamW状态空。",
        "",
        f"F1纯HF-only物理仍受阻：{f1['限制']}。不能把LF辨识的接触或"
        "LF温差选出的对流参数暗作独立已知HF-only输入。",
        "",
        "结论：当前仅CPU来源入场；F3小子集正式误差尚无数据。12功率F3复用须以"
        "任07乙状态冻结的合法观测结果为前提；F1及任08F2公平正式审定和专属子集GPU训练器另行核验。",
        "",
    ])
    (destination / "子集训练CPU来源入场.md").write_text(
        "\n".join(markdown), encoding="utf-8",
    )
    return report
