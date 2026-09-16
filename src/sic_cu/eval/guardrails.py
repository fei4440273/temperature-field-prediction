from __future__ import annotations

import math
from typing import Mapping, Sequence, Any

from sic_cu.data.splits import build_power_splits


def _improvement_percentage(old: float, new: float) -> float:
    if not math.isfinite(old) or not math.isfinite(new) or old <= 0 or new < 0:
        raise ValueError("Comparable error values must be finite and the baseline must be positive")
    return 100.0 * (old - new) / old


def task03_volume_guardrails(
    control: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
) -> dict[str, float | bool]:
    if set(control) != {"Cu", "SiC"} or set(candidate) != set(control):
        raise ValueError("LF volume guardrail requires paired Cu and SiC materials")
    old_mean = sum(float(control[material]["volume"]) for material in ("Cu", "SiC")) / 2.0
    new_mean = sum(float(candidate[material]["volume"]) for material in ("Cu", "SiC")) / 2.0
    improvement = _improvement_percentage(old_mean, new_mean)
    single_material_accepted = all(
        _improvement_percentage(
            float(control[material][metric]), float(candidate[material][metric]),
        ) >= -5.0
        for material in ("Cu", "SiC") for metric in ("node", "volume")
    )
    volume_accepted = improvement >= 5.0
    return {
        "原采样LF体积等权平均_摄氏度": old_mean,
        "空间LF体积等权平均_摄氏度": new_mean,
        "空间LF体积平均改善率_百分比": improvement,
        "两材料体积改善至少5百分比": volume_accepted,
        "各材料节点及体积恶化不超5百分比": single_material_accepted,
        "LF空间机制满足采用门槛": volume_accepted and single_material_accepted,
    }


def task03_hf_guardrails(
    *,
    old_score: float,
    new_score: float,
    old_modalities: Mapping[str, float],
    new_modalities: Mapping[str, float],
    old_energy: Mapping[str, float],
    new_energy: Mapping[str, float],
) -> dict[str, float | bool | dict[str, float]]:
    if set(old_modalities) != {"top", "hot", "cold"} or set(new_modalities) != set(old_modalities):
        raise ValueError("HF guardrail requires the same top/hot/cold modalities")
    if set(old_energy) != {"mean_w", "p95_w"} or set(new_energy) != set(old_energy):
        raise ValueError("HF guardrail requires the same absolute energy mean and p95")
    score_improvement = _improvement_percentage(old_score, new_score)
    allowed = {
        name: max(0.1, float(old_modalities[name]) * 0.05)
        for name in ("top", "hot", "cold")
    }
    modality_accepted = all(
        float(new_modalities[name]) - float(old_modalities[name]) <= allowed[name]
        for name in allowed
    )
    mean_improvement = _improvement_percentage(old_energy["mean_w"], new_energy["mean_w"])
    p95_unchanged_or_lower = new_energy["p95_w"] <= old_energy["p95_w"]
    return {
        "HF选分改善率_百分比": score_improvement,
        "选分方向改善": score_improvement > 0.0,
        "顶部允许增加_摄氏度": allowed["top"],
        "各单模态允许增加_摄氏度": allowed,
        "单模态护栏全部满足": modality_accepted,
        "工程能量均值改善率_百分比": mean_improvement,
        "工程能量改善至少50百分比": mean_improvement >= 50.0,
        "工程能量95分位不恶化": p95_unchanged_or_lower,
        "HF组合值得采用": score_improvement > 0.0 and modality_accepted,
        "物理修复达到保留门槛": (
            score_improvement >= -2.0 and mean_improvement >= 50.0
            and p95_unchanged_or_lower
        ),
    }


def validate_task03_hf_budget(
    metrics: Mapping[str, Any], log_rows: Sequence[Mapping[str, Any]],
    locked: Mapping[str, Any],
) -> dict[str, int]:
    config = metrics.get("configuration", {})
    splits = build_power_splits()
    epochs = int(locked["HF每臂逻辑轮次上限"])
    expected_train = sorted(splits.hf_train)
    expected_validation = sorted(splits.hf_validation)
    if (
        epochs != 300 or metrics.get("seed") != locked["种子"]
        or metrics.get("status") != "completed"
        or metrics.get("epochs_completed") != epochs
        or metrics.get("stopping_reason") != "planned_budget_completed"
        or config.get("correction_epochs") != epochs
        or config.get("joint_epochs") != 0
        or config.get("batch_size_per_rank") != locked["HF每卡batch"]
        or config.get("physics_collocation_per_rank") != locked["HF每项物理整包配点"]
        or config.get("patience") != epochs + 1
        or config.get("续训学习率") != locked["HF学习率"]
        or config.get("物理排程") != "separate"
        or config.get("验证间隔轮次") != 10
        or config.get("历史起点哈希") != locked["HF配对B0_SHA256"]
        or config.get("sensor_absolute_weight") != locked["HF传感器绝对与温升权重"][0]
        or config.get("sensor_delta_weight") != locked["HF传感器绝对与温升权重"][1]
        or config.get("checkpoint_selection") != "validation"
        or config.get("test_evaluation_enabled") is not False
        or sorted(config.get("hf_train_powers_w", [])) != expected_train
        or sorted(config.get("hf_validation_powers_w", [])) != expected_validation
        or config.get("hf_training_subset_w") is not None
        or metrics.get("test_ir") is not None or metrics.get("test_sensor") is not None
        or len(log_rows) != epochs
        or [row.get("epoch") for row in log_rows] != list(range(1, epochs + 1))
    ):
        raise ValueError("Task03 HF budget/protocol differs from pre-registered 300-epoch arms")
    data_steps = sum(int(row["data_optimizer_steps"]) for row in log_rows)
    physics_steps = sum(int(row["physics_optimizer_steps"]) for row in log_rows)
    points = sum(int(row["physics_collocation_points"]) for row in log_rows)
    ir_points = sum(int(row["epoch_hf_ir_points"]) for row in log_rows)
    if (
        data_steps != 4500 or physics_steps != 300 or points != 76800
        or ir_points != metrics.get("data_consumption_rank0", {}).get("hf_ir_points")
        or any(
            row.get("stage") != "correction" or row.get("physics_schedule") != "separate"
            or row.get("lf_parameters_frozen") is not True
            or row.get("lf_frozen_state_matches_initial") is not True
            or row.get("epoch_lf_simulation_points") != 0
            or row.get("data_optimizer_steps") != 15
            or row.get("physics_optimizer_steps") != 1
            or row.get("physics_collocation_points") != 256
            or row.get("epoch_hf_ir_points") != ir_points // epochs
            for row in log_rows
        )
    ):
        raise ValueError("Task03 HF budget/steps or frozen LF differs across arms")
    return {"观测优化步": data_steps, "物理优化步": physics_steps,
            "物理配点数": points, "训练顶部数据点暴露": ir_points}
