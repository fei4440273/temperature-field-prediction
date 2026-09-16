#!/usr/bin/env python
"""任-07五种子正式最佳视图的独立推理回归；不读取旧TEST温度。"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.data.fields import load_processed_field
from sic_cu.data.splits import build_power_splits
from sic_cu.prediction import Predictor
from sic_cu.train.multifidelity import _ir_dataset, _sensor_tensors
from sic_cu.train.task04_joint import _validation_selection
from sic_cu.train.task07_formal import (
    CORRECTION_STAGE, JOINT_STAGE, TASK07_REGISTRATION, _check_snapshot_identity,
    _equal, _log_budget, _qualification, _require_formal_registration,
)
from sic_cu.train.task07_source import validate_task07_sources
from sic_cu.visualization.export import export_prediction
from sic_cu.config import load_yaml


def _path(value: str | Path) -> Path:
    result = Path(value)
    return result if result.is_absolute() else PROJECT_ROOT / result


def _state_equal(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> bool:
    return set(first) == set(second) and all(torch.equal(first[key], second[key]) for key in first)


def check_committed_terminal(latest: Mapping[str, Any], terminal: Mapping[str, Any]) -> None:
    fields = ("stage", "epoch", "model_state", "optimizer_state", "random_state",
              "parameter_requires_grad", "metadata", "budget", "scheduler_state")
    if any(not _equal(latest.get(field), terminal.get(field)) for field in fields):
        raise ValueError("任07最近完整状态与正式训练末/联合末的模型、AdamW、随机源或冻结名单不一致")


def check_committed_best_stage(
    latest_meta: Mapping[str, Any], observed: Mapping[str, Any], physical: Mapping[str, Any],
) -> None:
    for snapshot, prefix, score_key in (
        (observed, "观测最佳", "观测最佳选分_摄氏度"),
        (physical, "物理最佳", "物理最佳独立损失"),
    ):
        global_key, stage_key, epoch_key = (f"{prefix}全局轮次", f"{prefix}阶段", f"{prefix}阶段轮次")
        best_meta = snapshot["metadata"]
        if (
            int(latest_meta[global_key]) > int(latest_meta["全局累计实际轮次"])
            or int(best_meta["全局累计实际轮次"]) != int(latest_meta[global_key])
            or int(best_meta[global_key]) != int(latest_meta[global_key])
            or snapshot["stage"] != latest_meta[stage_key]
            or int(snapshot["epoch"]) != int(latest_meta[epoch_key])
            or not math.isclose(float(best_meta[score_key]), float(latest_meta[score_key]), abs_tol=1e-4)
        ):
            raise ValueError(f"任07{prefix}完整阶段真实轮次、来源阶段或原分数与最近已提交状态不一致")


def assess_selected_best_lf_guard(
    report: Mapping[str, Any], best: Mapping[str, Any], source: Any,
) -> dict[str, Any]:
    guard = report.get("LF逐材料节点和真实体积5%护栏", {}).get(
        "LF两材料节点与真实体积均守住5%护栏",
    )
    if not isinstance(guard, bool):
        raise ValueError("任07正式联合末LF两材料节点/真实体积护栏须有真实布尔结论")
    if best["stage"] == CORRECTION_STAGE:
        lf_state = {name.removeprefix("low_fidelity_model."): tensor
                    for name, tensor in best["model_state"].items()
                    if name.startswith("low_fidelity_model.")}
        if (best["metadata"].get("当前真实LF张量SHA256") != source.lf_tensor_sha256
                or not _state_equal(lf_state, source.lf_state)):
            raise ValueError("任07校正阶段最佳LF须逐张量对应原seed配对来源，不能回退伪原LF")
        return {
            "联合末可采用": guard,
            "观察最佳是合法冻结LF校正阶段": True,
            "口径": (
                "联合末LF护栏通过；观察最佳仍为原LF冻结校正候选，非能源最终采用"
                if guard else
                "联合末不可采用：LF护栏失败；观察最佳仅为原LF冻结校正阶段，待独立精度与能源审核"
            ),
        }
    if best["stage"] != JOINT_STAGE or not guard or best["metadata"].get(
        "LF逐材料节点和真实体积5%护栏", {}
    ).get("LF两材料节点与真实体积均守住5%护栏") is not True:
        raise ValueError("任07联合最佳须自身和联合末真实LF两材料护栏均通过；失败联合不可冒充观察最佳")
    return {
        "联合末可采用": True,
        "观察最佳是合法冻结LF校正阶段": False,
        "口径": "观察最佳联合阶段及联合末均守住LF护栏；不代表独立能源及内部实验真值合格",
    }


def check_best_view(
    view: Mapping[str, Any], best: Mapping[str, Any], latest_meta: Mapping[str, Any],
    seed: int, qualification: str, validation: Mapping[str, float], *,
    correction_kwargs: Mapping[str, Any] | None = None,
    scales: Mapping[str, Any] | None = None,
    lf_kwargs: Mapping[str, Any] | None = None,
    source: Any | None = None,
) -> None:
    """Reject deployment views not identical to the committed observational best."""
    best_meta = best["metadata"]
    lineage = view.get("任07新HF模型视图来源", {})
    best_epoch = int(latest_meta["观测最佳全局轮次"])
    if (
        view.get("method") not in (None, "multifidelity_correction")
        or int(view.get("epoch", -1)) != best_epoch
        or best_epoch > int(latest_meta["全局累计实际轮次"])
        or int(best_meta["全局累计实际轮次"]) != best_epoch
        or not _state_equal(view.get("model_state", {}), best.get("model_state", {}))
        or lineage.get("训练种子") != seed
        or lineage.get("运行臂") != "E0"
        or lineage.get("本轮全局实际轮次") != best_epoch
        or lineage.get("运行资格") != qualification
        or (correction_kwargs is not None and view.get("correction_model_kwargs") != correction_kwargs)
        or (scales is not None and view.get("scales") != scales)
        or (lf_kwargs is not None and view.get("low_fidelity_model_kwargs") != lf_kwargs)
        or not math.isclose(float(view.get("validation_selection_score_c", math.inf)),
                            float(latest_meta["观测最佳选分_摄氏度"]), abs_tol=1e-4)
        or not math.isclose(float(view.get("validation_rmse_c", math.inf)),
                            float(validation["顶部"]), abs_tol=1e-4)
        or any(not math.isclose(float(view.get("validation_sensor", {}).get(key, math.inf)),
                                float(value), abs_tol=1e-4)
               for key, value in validation.items() if key != "顶部")
    ):
        raise ValueError(f"任07 seed{seed} 最佳模型视图的轮次、真实张量或合法HF评分与完整阶段不一致")
    if source is not None and (
        lineage.get("V4_B0来源清单SHA256") != source_manifest_sha()
        or lineage.get("本seed真实LF检查点SHA256") != source.lf_checkpoint_sha256
        or lineage.get("本seed历史HF架构视图SHA256") != source.hf_checkpoint_sha256
        or lineage.get("本seed来源协议SHA256") != source.source_metadata_sha256
        or lineage.get("本seed真实LF初始张量SHA256") != source.lf_tensor_sha256
        or lineage.get("当前真实LF张量SHA256") != best_meta["当前真实LF张量SHA256"]
        or lineage.get("正式预登记配置SHA256") != latest_meta["正式预登记配置SHA256"]
        or lineage.get("本轮校正轮次") != best_meta["校正实际轮次"]
        or lineage.get("本轮联合轮次") != best_meta["联合实际轮次"]
        or tuple(sorted(map(float, lineage.get("仅合法HF温度训练功率", ())))) != source.hf_train_powers_w
        or tuple(sorted(map(float, lineage.get("HF合法验证功率", ())))) != source.hf_validation_powers_w
        or tuple(sorted(map(float, lineage.get("LF真实温度只作低保真联合回放功率", ())))) != source.lf_train_powers_w
        or lineage.get("旧test_Data温度标签读取") is not False
    ):
        raise ValueError(f"任07 seed{seed} 最佳视图与历史配对来源、合法训练/验证协议不一致")


def source_manifest_sha() -> str:
    from sic_cu.train.task07_source import MANIFEST_SHA256
    return MANIFEST_SHA256


def _preflight_five_seeds(
    seed_directories: Mapping[int, str | Path], *, registry_path: Path,
    registry_sha256: str, device: torch.device,
) -> dict[int, dict[str, Any]]:
    if set(seed_directories) != set(range(5)):
        raise ValueError("任07正式五种子须完整提供seed0到4，缺失或多报seed都不可验收")
    if not registry_path.is_file() or sha256_file(registry_path) != registry_sha256:
        raise ValueError("任07正式预登记配置SHA256与真实原件不一致")
    registered = load_yaml(registry_path)
    if registered.get("种子") != list(range(5)):
        raise ValueError("任07正式配置未登记唯一五seed")
    mandatory = (
        "best.pt", "阶段报告.json", "阶段_最近.pt", "阶段_观测最佳.pt",
        "阶段_物理最佳.pt", "阶段_校正末.pt", "阶段_联合末.pt",
        "阶段_训练末.pt", "training.jsonl",
    )
    roots = {seed: _path(seed_directories[seed]) for seed in range(5)}
    for seed, root in roots.items():
        missing = [name for name in mandatory if not (root / name).is_file()]
        if missing:
            raise ValueError(f"任07正式五种子seed{seed}完整阶段/最佳视图缺失：{', '.join(missing)}")
        report = json.loads((root / "阶段报告.json").read_text(encoding="utf-8"))
        if (
            report.get("运行资格") != _qualification(False)
            or report.get("运行种子") != seed
            or report.get("运行臂") != "E0"
            or report.get("正式预登记配置SHA256") != registry_sha256
            or "短诊断" in report.get("状态", "")
            or "正式受限联合已达同组截止" not in report.get("状态", "")
            or report.get("真实最后状态") != "阶段_训练末.pt"
            or report.get("旧test_Data温度标签读取") is not False
            or not isinstance(report.get("LF逐材料节点和真实体积5%护栏", {}).get(
                "LF两材料节点与真实体积均守住5%护栏"), bool)
        ):
            raise ValueError(f"任07seed{seed}诊断、暂停、LF护栏结论缺失或未完成联合不能作正式验收")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("任07正式完整过程推理计时必须真实CUDA同步；CPU只许明确短诊断")
    sources = validate_task07_sources()
    _require_formal_registration(registry_path, registry_sha256, sources)
    if registered["V4_B0来源清单SHA256"] != source_manifest_sha():
        raise ValueError("任07五seed配对来源清单SHA漂移")
    splits = build_power_splits()
    if any(splits.hf_validation != frozenset(sources[seed].hf_validation_powers_w)
           for seed in range(5)):
        raise ValueError("任07真实HF验证功率只许预登记的三工况")
    training = load_yaml("configs/training.yaml")
    if training.get("selection_metric_version") != "macro_v1":
        raise ValueError("任07合法HF宏平均选分版本发生漂移")
    selection_weights = training["multifidelity_selection_weights"]
    checked: dict[int, dict[str, Any]] = {}
    for seed, root in roots.items():
        source = sources[seed]
        row = registered["五种子配对来源"][seed]
        if (
            row["LF检查点SHA256"] != source.lf_checkpoint_sha256
            or row["历史HF架构视图SHA256"] != source.hf_checkpoint_sha256
            or row["LF初始张量SHA256"] != source.lf_tensor_sha256
            or row["来源协议SHA256"] != source.source_metadata_sha256
        ):
            raise ValueError(f"任07seed{seed}来源SHA与前登记不一致")
        report = json.loads((root / "阶段报告.json").read_text(encoding="utf-8"))
        latest = torch.load(root / "阶段_最近.pt", map_location="cpu", weights_only=False)
        terminal = torch.load(root / "阶段_训练末.pt", map_location="cpu", weights_only=False)
        joint_terminal = torch.load(root / "阶段_联合末.pt", map_location="cpu", weights_only=False)
        best = torch.load(root / "阶段_观测最佳.pt", map_location="cpu", weights_only=False)
        correction = torch.load(root / "阶段_校正末.pt", map_location="cpu", weights_only=False)
        physical_best = torch.load(root / "阶段_物理最佳.pt", map_location="cpu", weights_only=False)
        for state in (latest, terminal, joint_terminal, best, correction, physical_best):
            _check_snapshot_identity(state, source, registry_sha256, False)
        check_committed_terminal(latest, terminal)
        check_committed_terminal(latest, joint_terminal)
        meta = latest["metadata"]
        check_committed_best_stage(meta, best, physical_best)
        best_lf_guard = assess_selected_best_lf_guard(report, best, source)
        correction_meta = correction["metadata"]
        if (
            latest["stage"] != JOINT_STAGE or terminal["stage"] != JOINT_STAGE
            or correction["stage"] != CORRECTION_STAGE
            or correction["epoch"] != meta["校正实际截止轮次"]
            or correction_meta["全局累计实际轮次"] != meta["校正实际截止轮次"]
            or correction_meta["校正实际截止轮次"] != meta["校正实际截止轮次"]
            or sha256_file(root / "阶段_校正末.pt") != meta["已提交正式校正末原件SHA256"]
            or meta["全局累计实际轮次"] != report["校正实际轮次"] + report["联合实际轮次"]
            or meta["观测最佳全局轮次"] != best["metadata"]["全局累计实际轮次"]
            or meta["观测最佳阶段"] != best["stage"]
            or meta["观测最佳阶段轮次"] != best["epoch"]
            or not math.isclose(float(meta["观测最佳选分_摄氏度"]),
                                float(best["metadata"]["观测最佳选分_摄氏度"]), abs_tol=1e-4)
            or not math.isclose(float(report["观测最佳HF合法选分_摄氏度"]),
                                float(meta["观测最佳选分_摄氏度"]), abs_tol=1e-4)
        ):
            raise ValueError(f"任07seed{seed}完整校正/联合末态、观测最佳与阶段报告不一致")
        for name, key in (("阶段_观测最佳.pt", "观测最佳全局轮次"),
                          ("阶段_物理最佳.pt", "物理最佳全局轮次")):
            stage_sha = meta.get(f"已提交旧{name}SHA256")
            if meta[key] < meta["全局累计实际轮次"] and (
                stage_sha is None or sha256_file(root / name) != stage_sha
            ):
                raise ValueError(f"任07seed{seed}历史最佳真实原件SHA承诺不一致")
        log = [json.loads(line) for line in (root / "training.jsonl").read_text(
            encoding="utf-8").splitlines()]
        train_sensor = _sensor_tensors(device, split="train", powers_w=source.hf_train_powers_w)
        _log_budget(log, meta["全局累计实际轮次"], len(train_sensor[0]), meta)
        if len(log) != meta["全局累计实际轮次"]:
            raise ValueError(f"任07seed{seed}阶段日志存在未提交尾行，不可直接验收")
        view = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
        if (
            view.get("method") != "multifidelity_correction"
            or view.get("correction_model_kwargs") != source.correction_model_kwargs
            or view.get("scales") != source.scales
            or view.get("low_fidelity_model_kwargs") != source.lf_model_kwargs
        ):
            raise ValueError(f"任07seed{seed}视图HF/LF架构和原来源不一致")
        predictor = Predictor(root / "best.pt", device=device)
        validation_data = _ir_dataset("validation", source.hf_validation_powers_w)
        loader = DataLoader(validation_data, batch_size=2048, shuffle=False)
        val_sensor = _sensor_tensors(device, split="validation",
                                     powers_w=source.hf_validation_powers_w)
        score, validation = _validation_selection(predictor.model, loader,
                                                   val_sensor, device, selection_weights)
        if not math.isclose(float(score), float(meta["观测最佳选分_摄氏度"]), abs_tol=1e-4):
            raise ValueError(f"任07seed{seed}独立重载最佳模型合法HF12/3宏选分与最佳完整状态不一致")
        check_best_view(view, best, meta, seed, _qualification(False), validation,
                        correction_kwargs=source.correction_model_kwargs,
                        scales=source.scales, lf_kwargs=source.lf_model_kwargs, source=source)
        checked[seed] = {"root": root, "source": source, "score": score,
                         "validation": validation, "predictor": predictor,
                         "阶段报告": report, "部署观测最佳与末态LF资格": best_lf_guard}
    return checked


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_cross_batch_probes(probes: np.ndarray, batch_size: int) -> np.ndarray:
    if batch_size <= 0 or probes.ndim != 2 or len(probes) == 0:
        raise ValueError("任07合法多batch点查询须为非空二维探针和正整数batch大小")
    copies = batch_size // len(probes) + 3
    return np.tile(probes, (copies, 1))


def inspect_cross_batch_consistency(
    query: np.ndarray, direct: np.ndarray, *, probe_count: int, batch_size: int,
) -> dict[str, Any]:
    if (probe_count <= 0 or batch_size <= 0 or len(query) % probe_count
            or query.shape[1:] != direct.shape[1:] or len(direct) != probe_count):
        raise ValueError("任07跨批模型探针形状或批次不合法")
    groups = query.reshape(-1, probe_count, *query.shape[1:])
    finite = bool(np.isfinite(query).all() and np.isfinite(direct).all())
    repeated = float(np.max(np.abs(groups - groups[0])))
    same_batch = float(np.max(np.abs(groups[0] - direct)))
    # One float32 ULP near 300 K exceeds the old fixed 1e-5 K audit threshold.
    one_ulp = (float(max(np.max(np.spacing(np.abs(query))),
                         np.max(np.spacing(np.abs(direct)))))
               if query.dtype == direct.dtype == np.float32 and finite else 0.0)
    tolerance = max(1e-5, one_ulp)
    return {
        "同点跨真实batch最大温差K": repeated,
        "跨批大小最大温差K": same_batch,
        "允许最大float32单ULP_K": one_ulp,
        "跨批浮点容差通过": (len(query) > batch_size and finite
                         and repeated <= tolerance and same_batch <= tolerance),
    }


def _hash_outputs(output: Path, summary: Mapping[str, Any]) -> dict[str, str]:
    (output / "运行摘要.json").write_text(json.dumps(summary, ensure_ascii=False,
                                              indent=2) + "\n", encoding="utf-8")
    hashes = {str(file.relative_to(output)): sha256_file(file)
              for file in sorted(output.rglob("*")) if file.is_file()}
    (output / "审计工件SHA256.json").write_text(json.dumps(hashes, ensure_ascii=False,
                                                        indent=2) + "\n", encoding="utf-8")
    return hashes


def _evaluate_loaded_view(
    predictor: Predictor, output: Path, *, source: Any | None,
    qualification: str, theta_resolution: int, batch_size: int,
) -> dict[str, Any]:
    if batch_size <= 0 or theta_resolution < 3:
        raise ValueError("任07查询batch_size必须正数、旋转分辨率不得低于3")
    if not predictor.supports_point_queries:
        raise ValueError("任07E0最佳视图必须支持合法HF逐点查询")
    read_clock = time.perf_counter()
    reference = load_processed_field(10.0)
    read_seconds = time.perf_counter() - read_clock
    geometry = load_yaml("configs/geometry.yaml")
    copper_radius = float(geometry["copper"]["radius_m"])
    z_cu = (float(geometry["embedding"]["copper_bottom_z_m"])
            + float(geometry["embedding"]["sic_bottom_z_m"])) / 2.0
    time0, time_last = map(float, (reference.times_s[0], reference.times_s[-1]))
    power = float(source.hf_validation_powers_w[0]) if source is not None else 100.0
    probes = np.array([
        [0.0, z_cu, time0, power, 0.0],
        [copper_radius, z_cu, time0, power, 0.0],
        [copper_radius, z_cu, time_last, power, 0.0],
        [0.0, 0.0, time0, power, 1.0],
        [0.0, 0.0, time_last, power, 1.0],
    ], dtype=np.float32)
    repeated = build_cross_batch_probes(probes, batch_size)
    _sync(predictor.device)
    clock = time.perf_counter()
    query = predictor.predict_points(repeated, batch_size=batch_size)
    _sync(predictor.device)
    query_seconds = time.perf_counter() - clock
    direct = predictor.predict_points(probes, batch_size=max(batch_size, len(probes)))
    groups = query.reshape(-1, len(probes), *query.shape[1:])
    consistency = inspect_cross_batch_consistency(
        query, direct, probe_count=len(probes), batch_size=batch_size,
    )
    repeat_deviation = consistency["同点跨真实batch最大温差K"]
    direct_deviation = consistency["跨批大小最大温差K"]
    if not consistency["跨批浮点容差通过"]:
        raise ValueError(f"任07同一坐标多批/重复查询真实最佳模型不稳定：{consistency}")
    _sync(predictor.device)
    forward_clock = time.perf_counter()
    forward_field = predictor._model_field(power, reference, reference.times_s)
    _sync(predictor.device)
    forward_seconds = time.perf_counter() - forward_clock
    _sync(predictor.device)
    field_clock = time.perf_counter()
    prediction = predictor.predict(power, times_s=reference.times_s,
                                   theta_resolution=theta_resolution)
    _sync(predictor.device)
    field_seconds = time.perf_counter() - field_clock
    if prediction.mean_temperature_k.shape != (len(reference.times_s), len(reference.material_ids)):
        raise ValueError("任07逐时完整RZ过程场与仿真来源时间/节点网格不一致")
    if not np.allclose(prediction.mean_temperature_k, forward_field, atol=1e-5, rtol=0.0):
        raise ValueError("任07独立全时HF纯前向与既有Prediction部署入口返回的场不一致")
    zero = predictor.predict(0.0, times_s=(time0, time_last),
                             theta_resolution=theta_resolution)
    low_extrap = predictor.predict(1.0, times_s=(time_last,),
                                   theta_resolution=theta_resolution)
    high_extrap = predictor.predict(800.0, times_s=(time_last,),
                                    theta_resolution=theta_resolution)
    _sync(predictor.device)
    export_clock = time.perf_counter()
    exported = export_prediction(prediction, output / "正式功率场",
                                 theta_resolution=theta_resolution,
                                 vtk_times_s=tuple(float(t) for t in (0, 50, 100, 150, 200)
                                                   if t in set(reference.times_s)))
    _sync(predictor.device)
    export_seconds = time.perf_counter() - export_clock
    kwargs = predictor.model.hard_initial_temperature_k, predictor.model.hard_cooling_radius_m
    initial_deviation = float(np.max(np.abs(query[[0, 1, 3]] - 295.15)))
    whole_initial_deviation = float(np.max(np.abs(prediction.mean_temperature_k[0] - 295.15)))
    if kwargs[0] is not None and max(initial_deviation, whole_initial_deviation) > 1e-4:
        raise ValueError("任07宣称硬初温的最佳模型在Cu/SiC t=0实测偏差超过浮点容差")
    cooling_deviation = float(np.max(np.abs(query[[1, 2]] - 295.15)))
    outer_coordinates = np.column_stack((
        np.full(len(reference.times_s), copper_radius),
        np.full(len(reference.times_s), z_cu), reference.times_s,
        np.full(len(reference.times_s), power), np.zeros(len(reference.times_s)),
    )).astype(np.float32)
    outer_field = predictor.predict_points(outer_coordinates, batch_size=batch_size)
    _sync(predictor.device)
    full_cooling_deviation = float(np.max(np.abs(outer_field - 295.15)))
    if kwargs[1] is not None and max(cooling_deviation, full_cooling_deviation) > 1e-4:
        raise ValueError("任07宣称硬水冷半径的最佳模型在Cu外径面实测偏差超过浮点容差")
    summary: dict[str, Any] = {
        "资格": qualification, "源最佳模型SHA256": sha256_file(predictor.checkpoint_path),
        "HF合法验证单工况查询功率W": power,
        "batch重复查询": {
            "批次大小": batch_size,
            "查询总行数": len(repeated),
            "实际批次数": math.ceil(len(repeated) / batch_size),
            "逐元素完全相同": bool(np.array_equal(groups, np.broadcast_to(groups[0], groups.shape))),
            "跨批浮点容差通过": consistency["跨批浮点容差通过"],
            "同点跨真实batch最大温差K": repeat_deviation,
            "跨批大小最大温差K": direct_deviation,
            "允许最大float32单ULP_K": consistency["允许最大float32单ULP_K"],
            "同步真实计时秒": query_seconds,
        },
        "完整时序场": {
            "逐时场数": len(reference.times_s), "每时RZ节点数": len(reference.material_ids),
            "起止秒": [time0, time_last], "同步真实计时秒": field_seconds,
            "读取LF网格计时秒": read_seconds,
            "模型纯前向同步计时秒": forward_seconds,
            "过程场文件导出计时秒": export_seconds,
            "逐时完整场归档": exported["field"],
        },
        "零功率锚点": {
            "初始场全部为295.15K": bool(np.allclose(zero.mean_temperature_k, 295.15, atol=1e-5)),
            "说明": "0W取既有预测入口独立初温锚点，不等于HF模型在0W坐标上的外推",
        },
        "初温t0模型诊断": {
            "Cu和SiC逐点初温最大偏差K": initial_deviation,
            "整个RZ场t0初温最大偏差K": whole_initial_deviation,
            "hard_initial_temperature_k": kwargs[0],
            "说明": "无硬初温参数时这是实测偏差，不可声称精确满足初温",
        },
        "水冷Cu外径面模型诊断": {
            "r米": copper_radius, "z米": z_cu,
            "t0和末时最大偏差K": cooling_deviation,
            "全时水冷外径面最大偏差K": full_cooling_deviation,
            "hard_cooling_radius_m": kwargs[1],
            "说明": "无硬水冷半径参数时这是实测偏差，不可声称精确满足外表面水冷",
        },
        "越界警示": {
            "低于HF训练范围有警示": any("range" in w.lower() or "extrapolation" in w.lower()
                                          for w in low_extrap.metadata.warnings),
            "超出HF训练范围有警示": any("HF experiment training range" in w
                                           for w in high_extrap.metadata.warnings),
            "补充": "1W和800W仅为支持域与文字警示回归，不作为任07合法HF验证分数",
        },
        "GPU同步真实计时": {
            "状态": "真实CUDA同步逐点和全时场" if predictor.device.type == "cuda"
                    else "CPU诊断，未作正式GPU计时",
            "逐点秒": query_seconds, "逐时全场秒": field_seconds,
        },
        "旧test_Data温度标签读取": False,
    }
    if not summary["越界警示"]["低于HF训练范围有警示"] or not summary["越界警示"]["超出HF训练范围有警示"]:
        raise ValueError("任07外推功率的现有预测入口缺必要支持域警示")
    summary["工件SHA256"] = _hash_outputs(output, summary)
    return summary


def evaluate_diagnostic_view(
    view_path: str | Path, output_directory: str | Path, *,
    device_name: str = "cpu", theta_resolution: int = 4, batch_size: int = 1024,
) -> dict[str, Any]:
    if device_name != "cpu":
        raise ValueError("任07单视图只许CPU不可作正式五种子GPU验收")
    output = _path(output_directory)
    if output.exists():
        raise FileExistsError(f"任07诊断目录已有工件，不得覆盖：{output}")
    predictor = Predictor(_path(view_path), device="cpu")
    if predictor.method != "multifidelity_correction":
        raise ValueError("任07单视图诊断只允许E0多保真HF校正部署模型")
    output.mkdir(parents=True)
    return _evaluate_loaded_view(predictor, output, source=None,
                                 qualification="CPU单视图短诊断；不可作正式五种子已验收",
                                 theta_resolution=theta_resolution, batch_size=batch_size)


def verify_task07_inference(
    seed_directories: Mapping[int, str | Path], output_directory: str | Path, *,
    formal_registry_path: str | Path = TASK07_REGISTRATION,
    formal_registry_sha256: str | None = None,
    device_name: str = "cuda", theta_resolution: int = 12,
    batch_size: int = 1024,
) -> dict[str, Any]:
    output = _path(output_directory)
    if output.exists():
        raise FileExistsError(f"任07推理审计目录已有文件，不得覆盖：{output}")
    resolved_output = output.resolve()
    if any(root.resolve() == resolved_output or root.resolve() in resolved_output.parents
           for root in (_path(path) for path in seed_directories.values())):
        raise ValueError("任07正式推理输出不可嵌套在任一冻结seed训练原件目录内")
    checked = _preflight_five_seeds(
        seed_directories, registry_path=_path(formal_registry_path),
        registry_sha256=formal_registry_sha256 or "", device=torch.device(device_name),
    )
    output.mkdir(parents=True)
    per_seed: dict[str, Any] = {}
    for seed, info in checked.items():
        subdir = output / f"seed{seed}"
        subdir.mkdir()
        result = _evaluate_loaded_view(info["predictor"], subdir, source=info["source"],
                                       qualification="五seed齐全的正式推理回归候选；非精度与能源最终采用",
                                       theta_resolution=theta_resolution, batch_size=batch_size)
        result["合法HF验证macro_v1选分_摄氏度"] = float(info["score"])
        result["合法HF验证三模态_摄氏度"] = info["validation"]
        result["部署观测最佳与末态LF资格"] = info["部署观测最佳与末态LF资格"]
        result["输入阶段工件SHA256"] = {
            name: sha256_file(info["root"] / name) for name in (
                "best.pt", "阶段报告.json", "阶段_最近.pt", "阶段_观测最佳.pt",
                "阶段_物理最佳.pt", "阶段_校正末.pt", "阶段_联合末.pt",
                "阶段_训练末.pt", "training.jsonl",
            )
        }
        per_seed[str(seed)] = result
    summary = {
        "状态": (
            "任07正式五种子校正-only最佳推理回归已验收；末联合LF护栏失败的状态不可采用，精度能源最终资格待独立审核"
            if any(not info["部署观测最佳与末态LF资格"]["联合末可采用"] for info in checked.values())
            else "任07正式五种子推理回归已验收；精度能源最终采用须另行独立审计"
        ),
        "正式预登记配置SHA256": formal_registry_sha256,
        "种子": list(range(5)), "设备": device_name,
        "五种子独立重载合法HF验证与完整过程": per_seed,
        "旧test_Data温度标签读取": False,
    }
    summary["工件SHA256"] = _hash_outputs(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="任07五seed正式最佳模型合法HF验证及完整推理回归")
    parser.add_argument("--seed0", required=True)
    parser.add_argument("--seed1", required=True)
    parser.add_argument("--seed2", required=True)
    parser.add_argument("--seed3", required=True)
    parser.add_argument("--seed4", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--registry", default=str(TASK07_REGISTRATION))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--theta-resolution", type=int, default=12)
    args = parser.parse_args()
    result = verify_task07_inference(
        {seed: getattr(args, f"seed{seed}") for seed in range(5)}, args.output,
        formal_registry_path=args.registry, formal_registry_sha256=args.registry_sha256,
        device_name=args.device, theta_resolution=args.theta_resolution,
        batch_size=args.batch_size,
    )
    print(json.dumps({key: result[key] for key in ("状态", "正式预登记配置SHA256", "种子",
                                                 "设备", "工件SHA256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
