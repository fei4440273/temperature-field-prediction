from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import statistics
import tarfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
import yaml

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.eval.task11_mlp_hf_energy import (
    CORRECTION_MODEL_KWARGS, LF_MODEL_KWARGS, MODEL_SCALES,
)
from sic_cu.models import AdditiveCorrectionModel, ModelScales
from sic_cu.train.simulation import build_model


MODULE = "sic_cu.eval.task11_mlp_observations"


def observation_module():
    assert importlib.util.find_spec(MODULE) is not None, "task11 observation implementation is missing"
    return importlib.import_module(MODULE)


@pytest.fixture(scope="module")
def observations():
    radii = np.append(np.linspace(0, 0.0249, 100, dtype=np.float32), np.float32(0.025))
    top, rings = [], []
    for power, frames, samples in ((115.2, 19, 100), (403.0, 24, 126), (630.5, 29, 150)):
        for time in np.arange(1, frames + 1, dtype=np.float32) * 5:
            for radius in radii:
                top.append((power, time, radius, 300 + float(time) * 0.1,
                            np.float32(1 / 101), "validation"))
        for kind, radius, baseline in (("hot", 0.02, 302), ("cold", 0.04, 298)):
            for time in range(1, samples + 1):
                rings.append((power, np.float32(time), kind, np.float32(radius),
                              np.float32(baseline + time * 0.1), np.float32((time - 1) * 0.1),
                              "validation", "cpu_fixture"))
    return (
        pl.DataFrame(top, schema=[("power_w", pl.Float32), ("time_s", pl.Float32),
                                  ("r_m", pl.Float32), ("temperature_mean_k", pl.Float32),
                                  ("frame_weight", pl.Float32), ("split", pl.String)], orient="row"),
        pl.DataFrame(rings, schema=[("power_w", pl.Float32), ("time_s", pl.Float32),
                                    ("sensor_type", pl.String), ("r_m", pl.Float32),
                                    ("temperature_k", pl.Float32), ("delta_temperature_k", pl.Float32),
                                    ("split", pl.String), ("source_dataset", pl.String)], orient="row"),
    )


def synthetic_predictions(top, sensors):
    top_prediction = top["temperature_mean_k"].to_numpy().astype(np.float64) + top["time_s"].to_numpy() / 5
    sensor_prediction = sensors["temperature_k"].to_numpy().astype(np.float64) + np.where(
        sensors["sensor_type"].to_numpy() == "hot", -2.0, 3.0)
    return top_prediction, sensor_prediction


def test_implementation_and_cli_exist():
    module = observation_module()
    assert (PROJECT_ROOT / "scripts/57_export_task11_mlp_observations.py").is_file()
    assert module.ROOT_TOKEN == "TASK11_NEW_MLP_OBSERVATIONS_GATE:v1"
    assert set(module.SOURCE_MEMBERS) >= {
        "src/sic_cu/eval/development_v4.py", "scripts/30_audit_task07_radial_endpoint.py",
        "src/sic_cu/eval/task11_mlp_observations.py", "scripts/57_export_task11_mlp_observations.py",
        "tests/test_task11_mlp_observations.py",
    }


def test_all_modalities_signed_errors_and_real_first_times(observations):
    module = observation_module()
    top, sensors = observations
    predictions = synthetic_predictions(top, sensors)
    result = module.evaluate_task11_observation_arrays(
        top, sensors, *predictions, seed=0, state="best", bottom_z_m=-0.0175)
    assert [len(result[key]) for key in ("points", "power", "time", "radial")] == [8024, 9, 27, 9]
    hot = next(row for row in result["power"] if row["modality"] == "Hot")
    cold = next(row for row in result["power"] if row["modality"] == "Cold")
    assert hot["signed_bias_c"] == pytest.approx(-2) and hot["peak_error_c"] == -2
    assert cold["signed_bias_c"] == pytest.approx(3) and cold["peak_error_c"] == 3
    assert hot["delta_p95_abs_error_c"] == 0
    assert all(point["reference_time_s"] == (5 if point["modality"] == "Top" else 1)
               for point in result["points"])
    for point in result["points"]:
        assert point["target_delta_k"] == pytest.approx(point["target_k"] - point["reference_target_k"])
        assert point["prediction_delta_k"] == pytest.approx(point["prediction_k"] - point["reference_prediction_k"])
    for power in module.FIXED_POWERS:
        for modality in module.MODALITIES:
            expected = next(row["sample_count"] for row in result["power"]
                            if row["power_w"] == power and row["modality"] == modality)
            assert sum(row["sample_count"] for row in result["time"]
                       if row["power_w"] == power and row["modality"] == modality) == expected
        assert sum(row["sample_count"] for row in result["radial"] if row["power_w"] == power) == next(
            row["sample_count"] for row in result["power"] if row["power_w"] == power and row["modality"] == "Top")
    empty = next(row for row in result["time"] if row["power_w"] == 115.2
                 and row["modality"] == "Top" and row["window"] == "time_100_200_s")
    assert empty["sample_count"] == 0 and empty["peak_error_c"] is None
    assert empty["delta_p95_abs_error_c"] is None and "无观测" in empty["empty_reason"]


def test_native_25mm_endpoint_and_illegal_other_endpoint():
    module = observation_module()
    masks = module.native_radial_masks(np.asarray([0.0, 0.0075, 0.0175, 0.025], dtype=np.float32))
    assert [mask.tolist() for mask in masks] == [[True, True, False, False], [False] * 4,
                                               [False, False, True, True]]
    with pytest.raises(ValueError, match="唯一合法差异"):
        module.native_radial_masks(np.asarray([0, 0.017, 0.025], dtype=np.float32))
    with pytest.raises(ValueError):
        module.native_radial_masks(np.asarray([0, 0.025], dtype=np.float64))


@pytest.mark.parametrize("change", ["prediction_nan", "target_nan", "weight_nan", "negative_weight",
                                    "zero_weight", "wrong_shape", "wrong_sensor", "missing_column",
                                    "extra_power", "train_257", "duplicate_point"])
def test_reject_nonfinite_shape_schema_or_wrong_fold(observations, change):
    module = observation_module()
    top, sensors = (frame.clone() for frame in observations)
    pred_top, pred_sensors = synthetic_predictions(top, sensors)
    if change == "prediction_nan":
        pred_top[0] = np.nan
    elif change == "target_nan":
        top = top.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(float("nan")).otherwise(
            pl.col("temperature_mean_k")).alias("temperature_mean_k"))
    elif change in {"weight_nan", "negative_weight", "zero_weight"}:
        value = {"weight_nan": float("nan"), "negative_weight": -1, "zero_weight": 0}[change]
        top = top.with_columns(pl.lit(value).alias("frame_weight"))
    elif change == "wrong_shape":
        pred_sensors = pred_sensors[:, None]
    elif change == "wrong_sensor":
        sensors = sensors.with_columns(pl.lit("Hot").alias("sensor_type"))
    elif change == "missing_column":
        top = top.drop("time_s")
    elif change == "extra_power":
        top = top.with_columns(pl.lit(257.0).alias("power_w"))
    elif change == "train_257":
        top = top.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(pl.lit("train")).otherwise(
            pl.col("split")).alias("split"))
    else:
        top = pl.concat([top.slice(0, 1), top.slice(0, top.height - 1)])
    with pytest.raises(ValueError):
        module.evaluate_task11_observation_arrays(top, sensors, pred_top, pred_sensors,
                                                  seed=0, state="best", bottom_z_m=-0.0175)


def test_signed_five_seed_statistics_and_p95_not_pooled():
    module = observation_module()
    values = [-5.0, -3.0, -1.0, 1.0, 2.0]
    result = module.five_seed_signed_statistics(values)
    assert result["均值"] == statistics.fmean(values)
    assert result["样本标准差"] == statistics.stdev(values)
    assert result["逐种子"] == values and result["ddof"] == 1
    for bad in (values[:4], values + [3], [0, 0, 0, 0, float("inf")], [True, 0, 0, 0, 0]):
        with pytest.raises(ValueError):
            module.five_seed_signed_statistics(bad)


def strict_view(seed=0):
    scales = ModelScales(**MODEL_SCALES)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        low = build_model("mlp_pinn", scales, **LF_MODEL_KWARGS)
        model = AdditiveCorrectionModel(low, scales, **CORRECTION_MODEL_KWARGS).float()
    return {"method": "multifidelity_correction", "low_fidelity_method": "mlp_pinn",
            "low_fidelity_model_kwargs": copy.deepcopy(LF_MODEL_KWARGS),
            "correction_model_kwargs": copy.deepcopy(CORRECTION_MODEL_KWARGS),
            "scales": copy.deepcopy(MODEL_SCALES), "model_state": model.state_dict(), "seed": seed}


def test_real_strict_float32_mlp_construction_and_predict():
    module = observation_module()
    view = strict_view()
    model = module.task11_observation_model_from_view(view, seed=0)
    assert model.correction[0].in_features == 6
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    prediction = module.predict_task11_observations(
        model, np.asarray([[0.0, 0.0, 1, 115.2, 1]], dtype=np.float32), torch.device("cpu"), 16)
    assert prediction.shape == (1,) and np.isfinite(prediction).all()
    for key, value in (("seed", 1), ("low_fidelity_method", "deeponet_pinn")):
        bad = {**view, key: value}
        with pytest.raises(ValueError):
            module.task11_observation_model_from_view(bad, seed=0)
    bad = copy.deepcopy(view)
    bad["correction_model_kwargs"]["correction_direct_power_input"] = False
    with pytest.raises(ValueError):
        module.task11_observation_model_from_view(bad, seed=0)
    bad = copy.deepcopy(view)
    first = next(iter(bad["model_state"]))
    bad["model_state"][first].fill_(float("nan"))
    with pytest.raises(ValueError):
        module.task11_observation_model_from_view(bad, seed=0)


@pytest.fixture
def frozen_sources(tmp_path, monkeypatch):
    module = observation_module()
    root = tmp_path / "project"
    root.mkdir()
    for relative in module.SOURCE_MEMBERS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((PROJECT_ROOT / relative).read_bytes())
    identities = {}
    for key in ("registry", "source_tar", "lf_catalog", "hf_data_catalog"):
        path = root / "synthetic_metadata" / (key + ".json")
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        identities[key] = {"原件": str(path.relative_to(root)), "SHA256": sha256_file(path)}
    monkeypatch.setattr(module, "HF_SOURCE_IDENTITY", identities)
    models = []
    for seed in range(5):
        run = root / module.RUN_DIRECTORY / f"正式MLP_HF_seed{seed}"
        run.mkdir(parents=True)
        for state in ("best", "final"):
            path = run / (state + ".pt")
            path.write_bytes(f"CPU identity only {seed} {state}".encode("ascii"))
            models.append({"种子": seed, "模型状态": state,
                           "模型原件": str(path.relative_to(root)), "模型SHA256": sha256_file(path)})
    source_tar = root / module.OBS_TAR_PATH
    source_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source_tar, "w:gz") as archive:
        for relative in module.SOURCE_MEMBERS:
            archive.add(root / relative, arcname=relative, recursive=False)
    budget = {"schema_version": 1, "阶段": "task11_new_mlp_validation_observations",
              "ROOT门禁标签": module.ROOT_TOKEN, "观察合同": copy.deepcopy(module.OBSERVATION_CONTRACT),
              "原HF四身份": identities, "模型十状态": models,
              "源码普通成员SHA256": {relative: sha256_file(root / relative) for relative in module.SOURCE_MEMBERS},
              "源码冻结tar原件": module.OBS_TAR_PATH, "源码冻结tarSHA256": sha256_file(source_tar)}
    registry = root / module.OBS_REGISTRY_PATH
    registry.write_text(yaml.safe_dump(budget, allow_unicode=True, sort_keys=False), encoding="utf-8")
    arguments = dict(observation_registry=registry, observation_registry_sha=sha256_file(registry),
                     observation_source_tar=source_tar, observation_source_tar_sha=sha256_file(source_tar),
                     output=root / module.OUTPUT_BASE / "合成观察_20260916T170000+0800", project_root=root)
    ledger = root / module.ROOT_LEDGER
    expected = (f"{module.ROOT_TOKEN}; status=active; OBS_YAML_SHA256={arguments['observation_registry_sha']}; "
                f"OBS_TAR_SHA256={arguments['observation_source_tar_sha']}; "
                f"HF_YAML_SHA256={identities['registry']['SHA256']}")
    ledger.write_text(f"| 录-0106 | {expected} | CPU合成门禁 |\n", encoding="utf-8")
    return module, root, budget, arguments, ledger


def test_static_preflight_only_reads_metadata_and_sources(frozen_sources, monkeypatch):
    module, root, _, arguments, _ = frozen_sources
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("preflight queried CUDA"))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("preflight loaded weights"))
    monkeypatch.setattr(pl, "read_parquet", lambda *a, **k: pytest.fail("preflight decoded temperatures"))
    result = module.preflight_task11_mlp_observations(**arguments)
    assert result["ROOT记录号"] == "录-0106"
    assert not arguments["output"].exists()
    assert "合法" not in result.get("状态", "")


@pytest.mark.parametrize("change", ["missing", "duplicate", "suffix", "token_substring", "same_number",
                                    "old_number", "wrong_sha", "whole_root", "external_path", "existing",
                                    "unnumbered_duplicate"])
def test_gate_rejects_without_cuda_labels_or_output(frozen_sources, monkeypatch, change):
    module, root, _, arguments, ledger = frozen_sources
    text = ledger.read_text(encoding="utf-8")
    if change == "missing":
        ledger.write_text("| 录-0106 | 尚未激活 | 合成 |\n", encoding="utf-8")
    elif change == "duplicate":
        ledger.write_text(text + text.replace("0106", "0107"), encoding="utf-8")
    elif change == "suffix":
        ledger.write_text(text.replace(" | CPU", "; 附加字段=越权 | CPU"), encoding="utf-8")
    elif change == "token_substring":
        ledger.write_text(text.replace(module.ROOT_TOKEN, "X" + module.ROOT_TOKEN), encoding="utf-8")
    elif change == "same_number":
        ledger.write_text(text + "| 录-0106 | 其他记录 | 合成 |\n", encoding="utf-8")
    elif change == "old_number":
        ledger.write_text(text.replace("0106", "0105"), encoding="utf-8")
    elif change == "wrong_sha":
        ledger.write_text(text.replace(arguments["observation_registry_sha"], "0" * 64), encoding="utf-8")
    elif change == "external_path":
        arguments["observation_registry"] = "/etc/hosts"
    elif change == "existing":
        arguments["output"].mkdir()
    elif change == "unnumbered_duplicate":
        ledger.write_text(text + text.replace("录-0106", "缺号记录"), encoding="utf-8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("failure queried CUDA"))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("failure read model"))
    if change == "whole_root":
        prereg = module.preflight_task11_mlp_observations(**arguments)
        ledger.write_text(text + "其他单元漂移\n", encoding="utf-8")
        with pytest.raises(ValueError):
            module.verify_task11_observation_runtime(prereg, {})
    else:
        with pytest.raises((ValueError, FileExistsError)):
            module.preflight_task11_mlp_observations(**arguments)
        with pytest.raises((ValueError, FileExistsError)):
            module.export_task11_mlp_observations(**arguments)
    assert not arguments["output"].exists() or change == "existing"


@pytest.mark.parametrize("change", ["source", "hf_metadata", "wrong_state", "wrong_seed", "wrong_model_sha"])
def test_source_or_ten_model_identity_drift_rejected(frozen_sources, change):
    module, root, budget, arguments, _ = frozen_sources
    prereg = module.preflight_task11_mlp_observations(**arguments)
    if change == "source":
        (root / module.SOURCE_MEMBERS[0]).write_bytes(b"drift")
    elif change == "hf_metadata":
        (root / module.HF_SOURCE_IDENTITY["hf_data_catalog"]["原件"]).write_bytes(b"drift")
    elif change == "wrong_state":
        budget["模型十状态"][0]["模型状态"] = "final"
    elif change == "wrong_seed":
        budget["模型十状态"][0]["种子"] = 1
    else:
        budget["模型十状态"][0]["模型SHA256"] = "0" * 64
    with pytest.raises(ValueError):
        if change in {"source", "hf_metadata"}:
            module.verify_task11_observation_runtime(prereg, {})
        else:
            module.validate_task11_observation_model_catalog(budget["模型十状态"], root)
    assert not arguments["output"].exists()


def test_cpu_evidence_serialization_and_all_output_hashes(frozen_sources, observations):
    module, _, _, arguments, _ = frozen_sources
    prereg = module.preflight_task11_mlp_observations(**arguments)
    top, sensors = observations
    results = []
    for seed in range(5):
        for state in ("best", "final"):
            result = module.evaluate_task11_observation_arrays(top, sensors, *synthetic_predictions(top, sensors),
                                                              seed=seed, state=state, bottom_z_m=-0.0175)
            result["macro_reproduction"] = module.reconstruct_task11_observation_macro(result["power"])
            results.append(result)
    summary = module.write_task11_observation_evidence(prereg, results, {}, {}, synthetic_cpu=True)
    output = arguments["output"]
    manifest = json.loads((output / "工件SHA256.json").read_text(encoding="utf-8"))
    assert len(manifest) >= 6
    assert set(manifest) == {path.name for path in output.iterdir()} - {"工件SHA256.json"}
    for name, digest in manifest.items():
        assert sha256_file(output / name) == digest
    assert summary["总点数"] == 80240 and summary["验收计数"] == {"逐功率": 90, "时间窗": 270, "径向窗": 90}
    assert summary["限制"]["工程安全合格主张"] is False
    assert summary["限制"]["训练成本重新统计"] is False
    assert "CPU合成" in summary["状态"]
    assert len((output / "真实逐点预测.jsonl").read_text(encoding="utf-8").splitlines()) == 80240
    assert pl.read_csv(output / "逐功率三模态.csv").height == 90
    assert pl.read_csv(output / "逐功率三模态时间窗.csv").height == 270
    assert pl.read_csv(output / "顶部逐功率原生径向窗.csv").height == 90
    negative = summary["模型双状态五种子统计"]["best"]["逐功率"]["115.2|Hot"]["有符号偏差_摄氏度"]
    assert negative["逐种子"] == pytest.approx([-2] * 5)
    assert "NaN" not in (output / "机器汇总.json").read_text(encoding="utf-8")


def test_nonfinite_serialization_and_runtime_drift_never_create_output(frozen_sources, observations):
    module, root, _, arguments, _ = frozen_sources
    prereg = module.preflight_task11_mlp_observations(**arguments)
    top, sensors = observations
    result = module.evaluate_task11_observation_arrays(top, sensors, *synthetic_predictions(top, sensors),
                                                      seed=0, state="best", bottom_z_m=-0.0175)
    result["points"][0]["prediction_k"] = float("inf")
    with pytest.raises(ValueError):
        module.write_task11_observation_evidence(prereg, [result], {}, {}, synthetic_cpu=True)
    assert not arguments["output"].exists()
    (root / module.HF_SOURCE_IDENTITY["lf_catalog"]["原件"]).write_bytes(b"drift")
    with pytest.raises(ValueError):
        module.write_task11_observation_evidence(prereg, [], {}, {}, synthetic_cpu=True)
    assert not arguments["output"].exists()


def full_results(module, observations):
    top, sensors = observations
    results = []
    for seed in range(5):
        for state in ("best", "final"):
            result = module.evaluate_task11_observation_arrays(top, sensors, *synthetic_predictions(top, sensors),
                                                              seed=seed, state=state, bottom_z_m=-0.0175)
            result["macro_reproduction"] = module.reconstruct_task11_observation_macro(result["power"])
            results.append(result)
    return results


@pytest.mark.parametrize("change", ["finite_target", "finite_reference", "finite_weight", "power_metric",
                                    "window_duplicate", "radial_duplicate", "point_duplicate", "finite_macro"])
def test_writer_recomputes_real_points_before_any_output(frozen_sources, observations, change):
    module, _, _, arguments, _ = frozen_sources
    prereg = module.preflight_task11_mlp_observations(**arguments)
    results = full_results(module, observations)
    first = results[0]
    if change == "finite_target":
        first["points"][0]["target_k"] += 1
    elif change == "finite_reference":
        first["points"][0]["reference_time_s"] = 0
    elif change == "finite_weight":
        first["points"][0]["weight"] = -1
    elif change == "power_metric":
        first["power"][0]["signed_bias_c"] += 1
    elif change == "window_duplicate":
        first["time"][1] = dict(first["time"][0])
    elif change == "radial_duplicate":
        first["radial"][1] = dict(first["radial"][0])
    elif change == "point_duplicate":
        first["points"][1] = dict(first["points"][0])
    else:
        first["macro_reproduction"]["合法HF原macro_v1_摄氏度"] += 1
    with pytest.raises(ValueError):
        module.write_task11_observation_evidence(prereg, results, {}, {}, synthetic_cpu=True)
    assert not arguments["output"].exists()


def test_weighted_errors_unweighted_p95_and_window_uses_whole_curve_reference(observations):
    module = observation_module()
    top, sensors = observations
    top = top.with_columns(pl.when(pl.col("r_m") < 0.01).then(4.0).otherwise(1.0).alias("frame_weight"))
    pred_top, pred_sensors = synthetic_predictions(top, sensors)
    pred_top += top["r_m"].to_numpy().astype(np.float64) * 100
    result = module.evaluate_task11_observation_arrays(top, sensors, pred_top, pred_sensors,
                                                      seed=0, state="best", bottom_z_m=-0.0175)
    power = next(row for row in result["power"] if row["power_w"] == 115.2 and row["modality"] == "Top")
    selected = top["power_w"].to_numpy() == 115.2
    errors = pred_top[selected] - top["temperature_mean_k"].to_numpy()[selected]
    weights = top["frame_weight"].to_numpy()[selected]
    assert power["rmse_c"] == pytest.approx(np.sqrt(np.average(errors ** 2, weights=weights)))
    assert power["signed_bias_c"] == pytest.approx(np.average(errors, weights=weights))
    assert power["p95_abs_error_c"] == pytest.approx(np.quantile(np.abs(errors), 0.95))
    window = next(row for row in result["time"] if row["power_w"] == 115.2 and row["modality"] == "Top"
                  and row["window"] == "time_30_100_s")
    selected_times = top.filter(pl.col("power_w") == 115.2)["time_s"].to_numpy()
    assert window["delta_p95_abs_error_c"] == pytest.approx(np.quantile(
        selected_times[selected_times > 30].astype(np.float64) / 5 - 1, 0.95))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_all_ten_result_serialization_finite_guard(frozen_sources, observations, value):
    module, _, _, arguments, _ = frozen_sources
    prereg = module.preflight_task11_mlp_observations(**arguments)
    results = full_results(module, observations)
    results[0]["points"][0]["prediction_k"] = value
    with pytest.raises(ValueError, match="NaN|无穷|非有限"):
        module.write_task11_observation_evidence(prereg, results, {}, {}, synthetic_cpu=True)
    assert not arguments["output"].exists()


def test_writer_never_labels_synthetic_as_real_or_accepts_qualification_drift(frozen_sources, observations):
    module, _, _, arguments, _ = frozen_sources
    prereg = module.preflight_task11_mlp_observations(**arguments)
    results = full_results(module, observations)
    with pytest.raises(ValueError):
        module.write_task11_observation_evidence(prereg, results, {}, {}, synthetic_cpu=False)
    with pytest.raises(ValueError):
        module.write_task11_observation_evidence(prereg, results, {0: {"seed": 0}}, {}, synthetic_cpu=True)
    assert not arguments["output"].exists()


def test_cuda_is_strict_with_no_fallback(monkeypatch):
    module = observation_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU name queried CUDA"))
    for name in ("cpu", "cuda:0", "cuda:1", "auto"):
        with pytest.raises(ValueError):
            module._require_cuda(name)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError):
        module._require_cuda("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError):
        module._require_cuda("cuda")


def test_runtime_pins_cannot_replace_original_begin_hash(frozen_sources):
    module, root, _, arguments, _ = frozen_sources
    prereg = module.preflight_task11_mlp_observations(**arguments)
    path = root / module.HF_SOURCE_IDENTITY["hf_data_catalog"]["原件"]
    path.write_bytes(b"drift")
    with pytest.raises(ValueError):
        module.verify_task11_observation_runtime(prereg, {str(path): sha256_file(path)})
    assert not arguments["output"].exists()


def resign_registry(frozen_sources):
    _, _, budget, arguments, ledger = frozen_sources
    registry = Path(arguments["observation_registry"])
    original = arguments["observation_registry_sha"]
    registry.write_text(yaml.safe_dump(budget, allow_unicode=True, sort_keys=False), encoding="utf-8")
    arguments["observation_registry_sha"] = sha256_file(registry)
    ledger.write_text(ledger.read_text(encoding="utf-8").replace(original, arguments["observation_registry_sha"]),
                      encoding="utf-8")


@pytest.mark.parametrize("change", ["duplicate_yaml", "boolean_as_zero", "schema_boolean"])
def test_preflight_rejects_ambiguous_yaml_contract(frozen_sources, change):
    module, _, budget, arguments, ledger = frozen_sources
    if change == "boolean_as_zero":
        budget["观察合同"]["旧固定TEST温度读取"] = 0
        resign_registry(frozen_sources)
    elif change == "schema_boolean":
        budget["schema_version"] = True
        resign_registry(frozen_sources)
    else:
        registry = Path(arguments["observation_registry"])
        original = arguments["observation_registry_sha"]
        registry.write_text(registry.read_text(encoding="utf-8") + "schema_version: 1\n", encoding="utf-8")
        arguments["observation_registry_sha"] = sha256_file(registry)
        ledger.write_text(ledger.read_text(encoding="utf-8").replace(original, arguments["observation_registry_sha"]),
                          encoding="utf-8")
    with pytest.raises(ValueError):
        module.preflight_task11_mlp_observations(**arguments)
    assert not arguments["output"].exists()


def completed_source_catalog(frozen_sources, monkeypatch, fail_seed=None):
    module, root, _, _, _ = frozen_sources
    low_rows, qualifications = [], {}
    for seed in range(5):
        run = root / module.RUN_DIRECTORY / f"正式MLP_HF_seed{seed}"
        metrics = run / "metrics.json"
        metrics.write_text("{}\n", encoding="utf-8")
        low = root / module.RUN_DIRECTORY / f"正式MLP_LF_seed{seed}" / "best.pt"
        low.parent.mkdir()
        low.write_bytes(f"CPU low source identity {seed}".encode("ascii"))
        low_rows.append({"seed": seed, "真实原件SHA256": {str(low.relative_to(root)): sha256_file(low)}})
        q = {"状态": "CPU完整HF来源资格PASS；独立能源仍须另行审核", "seed": seed, "目录": str(run),
             "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
             "完整原件SHA256": {name: sha256_file(run / name) for name in ("best.pt", "final.pt", "metrics.json")},
             "本seed新LF最佳检查点原件": str(low), "本seed新LF最佳检查点SHA256": sha256_file(low)}
        for key, path_field, sha_field in (("registry", "登记原件", "登记SHA256"),
                                         ("source_tar", "源码归档原件", "源码归档SHA256"),
                                         ("lf_catalog", "LF目录原件", "LF目录SHA256"),
                                         ("hf_data_catalog", "HF数据目录原件", "HF数据目录SHA256")):
            q[path_field] = str(root / module.HF_SOURCE_IDENTITY[key]["原件"])
            q[sha_field] = module.HF_SOURCE_IDENTITY[key]["SHA256"]
        if seed == fail_seed:
            q["状态"] = "未完成"
        qualifications[seed] = q
    lf_path = root / module.HF_SOURCE_IDENTITY["lf_catalog"]["原件"]
    lf_path.write_text(json.dumps({"五seed新LF": low_rows}, ensure_ascii=False), encoding="utf-8")
    originals = []
    for relative in ("data/processed/experiment_ir_radial.parquet", "data/processed/sensor_ring_raw.parquet"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic structural source identity, not PARQ temperature")
        originals.append({"原件": relative, "文件SHA256": sha256_file(path)})
    hf_path = root / module.HF_SOURCE_IDENTITY["hf_data_catalog"]["原件"]
    hf_path.write_text(json.dumps({"HF合法验证功率_瓦": list(module.FIXED_POWERS),
                                  "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
                                  "HF源原件": originals}, ensure_ascii=False), encoding="utf-8")
    for key in ("lf_catalog", "hf_data_catalog"):
        digest = sha256_file(root / module.HF_SOURCE_IDENTITY[key]["原件"])
        module.HF_SOURCE_IDENTITY[key]["SHA256"] = digest
        for q in qualifications.values():
            q["LF目录SHA256" if key == "lf_catalog" else "HF数据目录SHA256"] = digest
    resign_registry(frozen_sources)
    calls = []

    def qualify(**kwargs):
        calls.append(kwargs["seed"])
        assert kwargs["output"] == root / module.RUN_DIRECTORY / f"正式MLP_HF_seed{kwargs['seed']}"
        return copy.deepcopy(qualifications[kwargs["seed"]])

    monkeypatch.setattr(module, "qualify_task11_mlp_hf_source", qualify)
    return qualifications, calls


def test_all_five_cpu_qualifications_and_catalog_pins_before_labels(frozen_sources, monkeypatch):
    module, _, _, arguments, _ = frozen_sources
    expected, calls = completed_source_catalog(frozen_sources, monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU qualification queried CUDA"))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("synthetic boundary loaded real PT"))
    monkeypatch.setattr(pl, "read_parquet", lambda *a, **k: pytest.fail("CPU boundary decoded temperatures"))
    prereg = module.preflight_task11_mlp_observations(**arguments)
    before, pins = module._qualify_five(prereg)
    after, after_pins = module._qualify_five(prereg)
    assert before == after == expected and pins == after_pins and calls == list(range(5)) * 2
    assert not arguments["output"].exists()


def test_unfinished_fifth_cpu_qualification_refuses_labels(frozen_sources, monkeypatch):
    module, _, _, arguments, _ = frozen_sources
    _, calls = completed_source_catalog(frozen_sources, monkeypatch, fail_seed=4)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("unfinished source queried CUDA"))
    prereg = module.preflight_task11_mlp_observations(**arguments)
    with pytest.raises(ValueError):
        module._qualify_five(prereg)
    assert calls == list(range(5)) and not arguments["output"].exists()


def test_synthetic_helper_cannot_export_into_the_formal_project(frozen_sources, observations, monkeypatch):
    module, root, _, arguments, _ = frozen_sources
    prereg = module.preflight_task11_mlp_observations(**arguments)
    results = full_results(module, observations)
    monkeypatch.setattr(module, "PROJECT_ROOT", root)
    with pytest.raises(ValueError):
        module.write_task11_observation_evidence(prereg, results, {}, {}, synthetic_cpu=True)
    assert not arguments["output"].exists()


@pytest.mark.parametrize("change", ["state_six_input", "old_guide"])
def test_strict_mlp_rejects_corrupted_six_input_or_old_guide(change):
    module = observation_module()
    view = strict_view()
    if change == "state_six_input":
        view["model_state"]["correction.0.weight"] = view["model_state"]["correction.0.weight"][:, :5]
    else:
        view["surface_residual_guide_spec"] = {"旧臂": "任07E0"}
    with pytest.raises(ValueError):
        module.task11_observation_model_from_view(view, seed=0)


@pytest.mark.parametrize("values", [np.asarray([complex(1, float("nan"))]),
                                   np.asarray([1 + 2j]), np.asarray([True]),
                                   np.asarray([1.0], dtype=object)])
def test_vector_requires_real_numeric_dtype(values):
    module = observation_module()
    with pytest.raises(ValueError):
        module._vector(values, "原实数回归夹具", 1)


@pytest.mark.parametrize("change", ["out_of_bounds", "duplicate", "swapped", "reindexed_permutation"])
def test_source_rows_are_true_modality_source_order(observations, change):
    module = observation_module()
    results = full_results(module, observations)
    points = results[0]["points"]
    if change == "out_of_bounds":
        points[0]["source_row"] = 1000000000
    elif change == "duplicate":
        points[1]["source_row"] = points[0]["source_row"]
    elif change == "swapped":
        points[0]["source_row"], points[1]["source_row"] = points[1]["source_row"], points[0]["source_row"]
    else:
        points[0], points[1] = points[1], points[0]
        points[0]["source_row"], points[1]["source_row"] = 0, 1
    with pytest.raises(ValueError):
        module._validate_results(results)


@pytest.mark.parametrize("context", ["backtick_fence", "tilde_fence", "indented_code", "blockquote"])
def test_gate_ignores_non_table_contexts(frozen_sources, context):
    module, _, _, arguments, ledger = frozen_sources
    text = ledger.read_text(encoding="utf-8")
    if context == "backtick_fence":
        text = "```markdown\n" + text + "```\n"
    elif context == "tilde_fence":
        text = "~~~~markdown\n" + text + "~~~~\n"
    elif context == "indented_code":
        text = "    " + text
    else:
        text = "> " + text
    ledger.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        module.preflight_task11_mlp_observations(**arguments)
    assert not arguments["output"].exists()


def test_gate_keeps_plain_table_after_unrelated_code_fences(frozen_sources):
    module, _, _, arguments, ledger = frozen_sources
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text("````markdown\n| 录-0106 | 普通代码示例 | 合成 |\n```\n````\n"
                      "~~~text\n非门禁代码\n~~~\n" + text, encoding="utf-8")
    assert module.preflight_task11_mlp_observations(**arguments)["ROOT记录号"] == "录-0106"
    assert not arguments["output"].exists()


def test_public_writer_refuses_real_output_from_external_qualifications(frozen_sources, observations, monkeypatch):
    module, root, _, arguments, _ = frozen_sources
    prereg = module.preflight_task11_mlp_observations(**arguments)
    results = full_results(module, observations)
    monkeypatch.setattr(module, "PROJECT_ROOT", root)
    qualifications = {seed: {"完整原件SHA256": {}, "本seed新LF最佳检查点SHA256": "0" * 64}
                      for seed in range(5)}
    with pytest.raises(ValueError):
        module.write_task11_observation_evidence(prereg, results, qualifications, qualifications, synthetic_cpu=False)
    assert not arguments["output"].exists()


@pytest.mark.parametrize("qualification_drift", [False, True])
def test_private_real_writer_fresh_qualifications_before_output(
        frozen_sources, observations, monkeypatch, qualification_drift):
    module, root, _, arguments, _ = frozen_sources
    expected, calls = completed_source_catalog(frozen_sources, monkeypatch)
    prereg = module.preflight_task11_mlp_observations(**arguments)
    before, pins = module._qualify_five(prereg)
    results = full_results(module, observations)
    monkeypatch.setattr(module, "PROJECT_ROOT", root)
    cuda_calls = []

    def fixture_cuda(name):
        assert calls == list(range(5)) * 2
        cuda_calls.append(name)
        return torch.device("cpu")

    monkeypatch.setattr(module, "_require_cuda", fixture_cuda)
    if qualification_drift:
        expected[4]["本人HF累计真实成本秒"] = 123.0
        with pytest.raises(ValueError):
            module._write_task11_observation_evidence(prereg, results, before, before, additional_pins=pins)
        assert not arguments["output"].exists() and not cuda_calls
    else:
        module._write_task11_observation_evidence(prereg, results, before, before, additional_pins=pins)
        assert cuda_calls == ["cuda"]
    assert calls == list(range(5)) * 2


@pytest.mark.parametrize("context", ["multiline", "one_line", "unclosed", "closed_then_unclosed", "duplicate_token"])
def test_gate_rejects_disabled_html_comment_authorization(frozen_sources, context):
    module, _, _, arguments, ledger = frozen_sources
    text = ledger.read_text(encoding="utf-8")
    if context == "multiline":
        text = "<!-- disabled authorization\n" + text + "-->\n"
    elif context == "one_line":
        text = "<!-- " + text.strip() + " -->\n"
    elif context == "unclosed":
        text = "<!-- disabled authorization\n" + text
    elif context == "closed_then_unclosed":
        text = "<!-- unrelated closed comment --> <!-- disabled authorization\n" + text + "-->\n"
    else:
        text = "<!-- disabled duplicate\n" + text + "-->\n" + text
    ledger.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        module.preflight_task11_mlp_observations(**arguments)
    assert not arguments["output"].exists()


@pytest.mark.parametrize("context", ["multiline", "one_line", "multiple_one_line", "fence_literal", "comment_around_fence"])
def test_gate_keeps_plain_row_after_closed_html_comments(frozen_sources, context):
    module, _, _, arguments, ledger = frozen_sources
    text = ledger.read_text(encoding="utf-8")
    if context == "multiline":
        prefix = "<!-- unrelated disabled example\n| 录-0106 | 非门禁示例 | 合成 |\n-->\n"
    elif context == "one_line":
        prefix = "<!-- unrelated closed comment -->\n"
    elif context == "multiple_one_line":
        prefix = "<!-- first --><!-- second -->\n"
    elif context == "fence_literal":
        prefix = "```html <!-- literal in fence info\n<!-- unclosed literal inside code\n```\n"
    else:
        prefix = "<!-- unrelated comment\n```markdown\n| 录-0106 | 非门禁示例 | 合成 |\n```\n-->\n"
    ledger.write_text(prefix + text, encoding="utf-8")
    assert module.preflight_task11_mlp_observations(**arguments)["ROOT记录号"] == "录-0106"
    assert not arguments["output"].exists()
