"""Howard 三模态组件的项目内 CPU 合成回归；不读取真实模型或温度。"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import importlib.util
import io
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
from sic_cu.train import task11_howard_hf_formal as hf


MODULE = "sic_cu.eval.task11_howard_observations"


def _obs():
    assert importlib.util.find_spec(MODULE) is not None, "Howard 三模态观察实现尚不存在"
    return importlib.import_module(MODULE)


def _identity():
    return {"registry": hf.REGISTRY, "registry_sha": "1" * 64,
            "source_tar": hf.SOURCE_TAR, "source_tar_sha": "2" * 64,
            "lf_catalog": hf.LF_CATALOG, "lf_catalog_sha": "3" * 64,
            "hf_data_catalog": hf.HF_DATA_CATALOG, "hf_data_catalog_sha": "4" * 64}


def _view(seed=0):
    model = hf._fresh_cpu_model()
    return {"schema_version": 1, "method": hf.METHOD, "seed": seed, "epoch": 10,
            "model_kwargs": copy.deepcopy(model.model_kwargs), "scales": copy.deepcopy(hf.HF_BUDGET["scales"]),
            "model_state": model.state_dict(), "validation_selection_score_c": 1.0,
            "validation_modalities": {"顶部": 1., "absolute_rmse_c": 1., "absolute_mae_c": 1.,
                                      "delta_rmse_c": 1., "delta_mae_c": 1.},
            "任11HowardHF事前来源": {"seed": seed, "method": hf.METHOD,
                "固定PHQH查询": hf._query_metadata("5" * 64)},
            "HF训练许可": False, **dict.fromkeys(hf.ISOLATION_FLAGS, False)}


@pytest.fixture(scope="module")
def arrays():
    radii = np.linspace(0, .025, 101, dtype=np.float32)
    radii[32], radii[68], radii[100] = np.float32(.008), np.float32(.017), np.float32(.025)
    top, rings = [], []
    for power, frames, samples in ((115.2, 19, 100), (403., 24, 126), (630.5, 29, 150)):
        edges = np.concatenate(([0.], (radii[:-1].astype(float) + radii[1:].astype(float)) / 2, [.025]))
        weights = np.diff(edges ** 2) / .025 ** 2
        for time in np.linspace(35, 200, frames, dtype=np.float32):
            for radius, weight in zip(radii, weights):
                top.append((power, time, radius, np.float32(300 + time * .125), weight, "validation"))
        for kind, radius, baseline in (("hot", .02, 302.), ("cold", .04, 298.)):
            for time in np.linspace(35, 200, samples, dtype=np.float32):
                target = np.float32(baseline + time * .125)
                first = np.float32(baseline + 35 * .125)
                rings.append((power, time, kind, np.float32(radius), target, float(target - first), "validation"))
    return (
        pl.DataFrame(top, schema=[("power_w", pl.Float32), ("time_s", pl.Float32), ("r_m", pl.Float32),
            ("temperature_mean_k", pl.Float32), ("frame_weight", pl.Float64), ("split", pl.String)], orient="row"),
        pl.DataFrame(rings, schema=[("power_w", pl.Float32), ("time_s", pl.Float32), ("sensor_type", pl.String),
            ("r_m", pl.Float32), ("temperature_k", pl.Float32), ("delta_temperature_k", pl.Float64),
            ("split", pl.String)], orient="row"),
    )


def _evaluate(arrays, seed=0, state="best"):
    top, sensors = arrays
    pt = top["temperature_mean_k"].to_numpy().astype(float) + np.asarray(
        [{np.float32(115.2): 1., np.float32(403.): 2., np.float32(630.5): 4.}[p] for p in top["power_w"]])
    ps = sensors["temperature_k"].to_numpy().astype(float) + np.where(
        sensors["sensor_type"].to_numpy() == "hot", -2., 3.) + seed / 10
    result = _obs().evaluate_task11_howard_observation_arrays(
        top, sensors, pt, ps, seed=seed, state=state, bottom_z_m=-.0175)
    result["macro_reproduction"] = _obs().reconstruct_task11_observation_macro(result["power"])
    return result


def _gate(hashes, number=130):
    module = _obs()
    cell = f"{module.ROOT_TOKEN}; status=active; " + "; ".join(
        f"{key}={hashes[key]}" for key in module.ROOT_IDENTITY_FIELDS)
    return f"| 录-{number:04d} | {cell} | CPU合成门禁，不是活动授权 |\n"


def _pack(tmp_path, tar_kind=None):
    module = _obs()
    root = tmp_path / "隔离项目"
    root.mkdir()
    for name in module.SOURCE_MEMBERS:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / name).read_bytes())
    source_hashes = {name: sha256_file(root / name) for name in module.SOURCE_MEMBERS}
    identity = _identity()
    for name in ("registry", "source_tar", "lf_catalog", "hf_data_catalog"):
        path = root / identity[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{\"CPU合成来源\": true}\n", encoding="utf-8")
        identity[name + "_sha"] = sha256_file(path)
    models = []
    view = _view()
    for seed in range(5):
        run = root / hf.RUN_DIRECTORY / f"正式Howard_HF_seed{seed}"
        run.mkdir(parents=True)
        for state in module.STATES:
            path = run / (state + ".pt")
            torch.save({**view, "seed": seed, "任11HowardHF事前来源": {
                **view["任11HowardHF事前来源"], "seed": seed}}, path)
            models.append({"种子": seed, "模型状态": state, "模型原件": path.relative_to(root).as_posix(),
                           "模型SHA256": sha256_file(path)})
    archive = root / module.OBS_TAR_PATH
    with tarfile.open(archive, "w:gz") as tar:
        for index, name in enumerate(module.SOURCE_MEMBERS):
            data = (root / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            if index == 0 and tar_kind == "link":
                info.type, info.linkname, info.size = tarfile.SYMTYPE, module.SOURCE_MEMBERS[1], 0
            tar.addfile(info, io.BytesIO(data))
        if tar_kind == "duplicate":
            data = (root / module.SOURCE_MEMBERS[0]).read_bytes()
            info = tarfile.TarInfo(module.SOURCE_MEMBERS[0])
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        elif tar_kind == "extra":
            tar.addfile(tarfile.TarInfo("不登记的额外项"))
    contract = module.registration_contract(source_hashes=source_hashes, source_tar_sha=sha256_file(archive),
        hf_source_identity=identity, model_catalog=models, registered_at="2026-09-16T20:08:02+08:00",
        root_before_sha="6" * 64)
    registry = root / module.OBS_REGISTRY_PATH
    registry.write_text(yaml.safe_dump(contract, allow_unicode=True), encoding="utf-8")
    args = {"observation_registry": registry, "observation_registry_sha": sha256_file(registry),
        "observation_source_tar": archive, "observation_source_tar_sha": sha256_file(archive),
        "output": root / module.OUTPUT_BASE / "Howard三模态_CPU合成观察", **identity, "project_root": root}
    (root / hf.ROOT_LEDGER).write_text(_gate(module.root_gate_hashes(args)), encoding="utf-8")
    return args, contract


def test_own_source_union_and_six_identities_are_not_mlp_or_energy_authority():
    module = _obs()
    from sic_cu.eval import task11_mlp_observations as numerical
    assert set(module.SOURCE_MEMBERS) == set(hf.SOURCE_MEMBERS) | set(numerical.SOURCE_MEMBERS) | {
        "src/sic_cu/eval/task11_howard_observations.py", "scripts/58_export_task11_howard_observations.py",
        "tests/test_task11_howard_observations.py"}
    assert len(module.SOURCE_MEMBERS) == 90
    assert module.ROOT_IDENTITY_FIELDS == ("OBS_YAML_SHA256", "OBS_TAR_SHA256", *hf.IDENTITY_FIELDS)
    assert module.ROOT_TOKEN == "TASK11_HOWARD_OBSERVATIONS_GATE:v1"
    assert not any("task11_howard_hf_energy" in n for n in module.SOURCE_MEMBERS)


def test_actual_ast_import_closure_includes_dynamic_native_helper():
    module = _obs()
    frozen = set(module.SOURCE_MEMBERS)
    assert "scripts/30_audit_task07_radial_endpoint.py" in frozen
    pending = [n for n in frozen if n.endswith(".py")]
    found = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        package = list(Path(name).with_suffix("").parts[1:-1]) if name.startswith("src/") else []
        for node in ast.walk(ast.parse((PROJECT_ROOT / name).read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                modules = [".".join(prefix + ((node.module or "").split(".") if node.module else []))]
                modules += [base + "." + x.name for base in modules[:] for x in node.names]
            for imported in modules:
                if not imported.startswith("sic_cu"):
                    continue
                parts = imported.split(".")
                base = Path("src").joinpath(*parts)
                candidates = [base.with_suffix(".py"), base / "__init__.py"] + [
                    Path("src").joinpath(*parts[:n], "__init__.py") for n in range(1, len(parts))]
                pending.extend(x.as_posix() for x in candidates if (PROJECT_ROOT / x).is_file())
    assert found <= frozen, "SOURCE缺实际本地依赖：" + str(sorted(found - frozen))


def test_native_exact_8_17_25mm_endpoints_no_drop_or_double_count():
    module = _obs()
    radii = np.asarray([0., .008, .017, .025], dtype=np.float32)
    masks = module.native_radial_masks(radii)
    assert [x.tolist() for x in masks] == [[True, True, False, False], [False, False, True, False],
                                          [False, False, False, True]]
    assert np.all(sum(x.astype(int) for x in masks) == 1)
    for bad in (radii.astype(float), np.asarray([-.001, .025], dtype=np.float32),
                np.asarray([0., np.nextafter(np.float32(.025), np.float32(1))], dtype=np.float32)):
        with pytest.raises(ValueError):
            module.native_radial_masks(bad)


def test_real_array_counts_native720_endpoints_empty30_and_first_actual_time(arrays):
    result = _evaluate(arrays)
    assert [len(result[k]) for k in ("points", "power", "time", "radial")] == [8024, 9, 27, 9]
    assert sum(p["r_m"] == float(np.float32(.025)) for p in result["points"] if p["modality"] == "Top") == 72
    empty = [r for r in result["time"] if r["window"] == "time_0_30_s"]
    assert len(empty) == 9 and all(r["sample_count"] == 0 for r in empty)
    assert all(r["rmse_c"] is None and r["peak_error_c"] is None and "无观测" in r["empty_reason"] for r in empty)
    assert all(p["reference_time_s"] == 35. for p in result["points"])
    assert all(p["prediction_delta_k"] == p["target_delta_k"] for p in result["points"])
    for power in _obs().FIXED_POWERS:
        for modality in _obs().MODALITIES:
            assert sum(r["sample_count"] for r in result["time"] if r["power_w"] == power and r["modality"] == modality) == next(
                r["sample_count"] for r in result["power"] if r["power_w"] == power and r["modality"] == modality)


def test_weighted_metrics_power_balance_raw_p95_and_negative_peak(arrays):
    module = _obs()
    result = _evaluate(arrays)
    assert result["macro_reproduction"]["顶部"] == pytest.approx(7 / 3)
    assert result["macro_reproduction"]["顶部"] != pytest.approx((19 + 48 + 116) / 72)
    hot = next(r for r in result["power"] if r["modality"] == "Hot")
    assert hot["peak_error_c"] == -2. and hot["signed_bias_c"] == pytest.approx(-2.)
    top, sensors = arrays
    error = top["r_m"].to_numpy().astype(float) * 100
    changed = module.evaluate_task11_howard_observation_arrays(top, sensors,
        top["temperature_mean_k"].to_numpy().astype(float) + error, sensors["temperature_k"].to_numpy(),
        seed=0, state="final", bottom_z_m=-.0175)
    row = next(r for r in changed["power"] if r["modality"] == "Top" and r["power_w"] == 115.2)
    selected = top["power_w"].to_numpy() == np.float32(115.2)
    weights = top["frame_weight"].to_numpy()[selected]
    assert row["rmse_c"] == pytest.approx(np.sqrt(np.sum(weights * error[selected] ** 2) / weights.sum()))
    assert row["p95_abs_error_c"] == pytest.approx(np.quantile(error[selected], .95))
    assert row["mae_c"] != pytest.approx(np.mean(error[selected]))


def test_native_original_frame_weights_retained_and_unnormalized_source_rejected(arrays):
    module = _obs()
    top, sensors = arrays
    times = top["time_s"].to_numpy().astype(float)
    original = top["frame_weight"].to_numpy()
    rescaled = top.with_columns(pl.Series("frame_weight", original * (1 + times)))
    prediction = top["temperature_mean_k"].to_numpy().astype(float) + times / 10
    left = module.evaluate_task11_howard_observation_arrays(top, sensors, prediction,
        sensors["temperature_k"].to_numpy(), seed=0, state="best", bottom_z_m=-.0175)
    assert left["points"][0]["weight"] == original[0]
    with pytest.raises(ValueError, match="帧.*归一|归一.*帧"):
        module.evaluate_task11_howard_observation_arrays(rescaled, sensors, prediction,
            sensors["temperature_k"].to_numpy(), seed=0, state="best", bottom_z_m=-.0175)


def test_radial_statistics_use_raw_frame_weights_without_window_reweighting(arrays):
    module = _obs()
    top, sensors = arrays
    top = top.with_columns((pl.col("frame_weight") * (1 + pl.col("time_s") * pl.col("r_m") / .025)).alias("frame_weight"))
    top = top.with_columns((pl.col("frame_weight") / pl.col("frame_weight").sum().over("power_w", "time_s")).alias("frame_weight"))
    error = top["r_m"].to_numpy().astype(float) * 100 + top["time_s"].to_numpy().astype(float) / 50
    result = module.evaluate_task11_howard_observation_arrays(top, sensors,
        top["temperature_mean_k"].to_numpy().astype(float) + error,
        sensors["temperature_k"].to_numpy(), seed=0, state="best", bottom_z_m=-.0175)
    mask = (top["power_w"].to_numpy() == np.float32(115.2)) & module.native_radial_masks(top["r_m"].to_numpy())[2]
    weights = top["frame_weight"].to_numpy()[mask]
    row = next(r for r in result["radial"] if r["power_w"] == 115.2 and r["radial_window"] == "outer_17_25_mm")
    assert row["rmse_c"] == pytest.approx(np.sqrt(np.sum(weights * error[mask] ** 2) / weights.sum()), abs=1e-10)
    assert row["mae_c"] == pytest.approx(np.sum(weights * np.abs(error[mask])) / weights.sum(), abs=1e-10)


@pytest.mark.parametrize("change", ["prediction_nan", "prediction_shape", "target_nan", "weight_nan", "negative_weight",
    "zero_frame", "top_double_radius", "bad_sensor", "missing_column", "illegal_power", "train", "duplicate", "wrong_delta"])
def test_raw_observation_schema_finiteness_native_dtype_and_fold_fail_closed(arrays, change):
    module = _obs()
    top, sensors = [f.clone() for f in arrays]
    pt, ps = top["temperature_mean_k"].to_numpy().astype(float), sensors["temperature_k"].to_numpy().astype(float)
    if change == "prediction_nan": pt[0] = float("nan")
    elif change == "prediction_shape": ps = ps[:, None]
    elif change == "target_nan": top = top.with_columns(pl.lit(float("nan")).alias("temperature_mean_k"))
    elif change == "weight_nan": top = top.with_columns(pl.lit(float("nan")).alias("frame_weight"))
    elif change == "negative_weight": top = top.with_columns(pl.lit(-1.).alias("frame_weight"))
    elif change == "zero_frame": top = top.with_columns(pl.when(pl.col("time_s") == 35).then(0.).otherwise(pl.col("frame_weight")).alias("frame_weight"))
    elif change == "top_double_radius": top = top.with_columns(pl.col("r_m").cast(pl.Float64))
    elif change == "bad_sensor": sensors = sensors.with_columns(pl.lit("Hot").alias("sensor_type"))
    elif change == "missing_column": top = top.drop("time_s")
    elif change == "illegal_power": top = top.with_columns(pl.lit(257.).alias("power_w"))
    elif change == "train": sensors = sensors.with_columns(pl.lit("train").alias("split"))
    elif change == "duplicate": top = pl.concat([top.slice(0, 1), top.slice(0, top.height - 1)])
    else: sensors = sensors.with_columns((pl.col("delta_temperature_k") + 1).alias("delta_temperature_k"))
    with pytest.raises(ValueError):
        module.evaluate_task11_howard_observation_arrays(top, sensors, pt, ps, seed=0, state="best", bottom_z_m=-.0175)


def test_five_seed_signed_ddof1_and_empty_statistics(arrays):
    module = _obs()
    values = [-5., -3., -1., 1., 2.]
    stat = module.five_seed_signed_statistics(values)
    assert stat["均值"] == statistics.fmean(values)
    assert stat["样本标准差"] == statistics.stdev(values)
    assert stat["逐种子"] == values and stat["ddof"] == 1
    results = [_evaluate(arrays, seed=n) for n in range(5)]
    groups = module._group_statistics(results, "time")
    empty = next(v for v in groups.values() if "[0,30]" in v["分组中文名称"])
    assert empty["绝对RMSE_摄氏度"]["均值"] is None
    assert empty["绝对RMSE_摄氏度"]["逐种子"] == [None] * 5
    for bad in (values[:4], values + [3.], [True, 1., 2., 3., 4.], [0., 1., 2., 3., float("nan")]):
        with pytest.raises(ValueError):
            module.five_seed_signed_statistics(bad)


def test_factory_real_all56_queries64_formula_and_cpu_prediction():
    module = _obs()
    view = _view()
    original = {n: t.clone() for n, t in view["model_state"].items()}
    model = module.task11_howard_observation_model_from_view(view, seed=0)
    assert len(list(model.parameters())) == 56
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert {b.dtype for b in model.buffers()} == {torch.float64}
    x = torch.tensor([[.008, 0., 35., 115.2, 1.]], dtype=torch.float32)
    with torch.no_grad():
        outputs = model.subnet_outputs(x)
        assert torch.equal(model(x), 295.15 + 250. * (outputs["linear"] + outputs["nonlinear"]))
    prediction = module.predict_task11_howard_observations(model, x.numpy(), torch.device("cpu"), 1)
    assert prediction.shape == (1,) and np.isfinite(prediction).all()
    assert all(torch.equal(t, original[n]) for n, t in view["model_state"].items())


@pytest.mark.parametrize("change", ["seed", "bool_seed", "method", "lf", "query32", "querypoint", "metadata", "missing",
    "nan", "float64", "shape", "layout", "kwargs", "permission", "epoch0", "scales"])
def test_factory_rejects_wrong_saved_identity_dtype_shape_or_query(change):
    module = _obs()
    view = _view()
    name = next(n for n in view["model_state"] if "weight" in n)
    if change == "seed": view["seed"] = 1
    elif change == "bool_seed": view["seed"] = False
    elif change == "method": view["method"] = "multifidelity_correction"
    elif change == "lf": view["method"] = hf.LF_METHOD
    elif change == "query32": view["model_state"]["linear_query_points"] = hf._fixed_points().float()
    elif change == "querypoint": view["model_state"]["linear_query_points"][0, 0] += .001
    elif change == "metadata": view["任11HowardHF事前来源"]["固定PHQH查询"]["no_temperature_selection"] = False
    elif change == "missing": del view["model_state"][name]
    elif change == "nan": view["model_state"][name].fill_(float("nan"))
    elif change == "float64": view["model_state"][name] = view["model_state"][name].double()
    elif change == "shape": view["model_state"][name] = view["model_state"][name][:1]
    elif change == "layout": view["model_state"][name] = view["model_state"][name].to_sparse()
    elif change == "kwargs": view["model_kwargs"]["width"] = 64
    elif change == "permission": view["HF训练许可"] = True
    elif change == "epoch0": view["epoch"] = 0
    else: view["scales"]["temperature_offset_k"] = 0.
    with pytest.raises(ValueError):
        module.task11_howard_observation_model_from_view(view, seed=0)


def test_macro_checks_actual_howard_validation_modalities_not_mlp_sensor(arrays):
    module = _obs()
    result = _evaluate(arrays)
    expected = result["macro_reproduction"]
    view = {"validation_selection_score_c": expected["合法HF原macro_v1_摄氏度"],
            "validation_modalities": {k: v for k, v in expected.items() if k != "合法HF原macro_v1_摄氏度"}}
    assert module.verify_task11_howard_observation_macro(result, view) == expected
    bad = copy.deepcopy(view)
    bad["validation_modalities"]["顶部"] += .0002
    with pytest.raises(ValueError): module.verify_task11_howard_observation_macro(result, bad)
    with pytest.raises(ValueError): module.verify_task11_howard_observation_macro(result, {
        "validation_selection_score_c": view["validation_selection_score_c"], "validation_sensor": view["validation_modalities"]})


def test_pure_builder_and_static_preflight_do_not_read_pt_labels_or_cuda(tmp_path, monkeypatch):
    module = _obs()
    args, contract = _pack(tmp_path)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("前置核验读取PT"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("前置核验探测CUDA"))
    checked = module.preflight_task11_howard_observations(**args)
    assert checked["观察许可"] is False and checked["源码普通成员数"] == 90
    assert checked["HowardHF四来源"] == {k: args[k] for k in _identity()}
    assert len(checked["模型十身份"]) == 10
    assert not Path(args["output"]).exists()
    assert contract["科学或工程安全资格"] is False


@pytest.mark.parametrize("kind", ["link", "duplicate", "extra"])
def test_bad_tar_rejected_before_any_pt_or_cuda(tmp_path, monkeypatch, kind):
    args, _ = _pack(tmp_path, kind)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("非法来源读取PT"))
    with pytest.raises(ValueError, match="tar|归档|成员|普通"):
        _obs().preflight_task11_howard_observations(**args)


@pytest.mark.parametrize("wrapper", ["```\n{}\n```", "~~~~\n{}\n~~~~", "<!--\n{}\n-->",
    "<pre>\n{}\n</pre>", "<script>\n{}\n</script>", "<textarea>\n{}\n</textarea>",
    "<div>\n{}\n</div>\n\n", "    {}", "> {}", "<pre/>\n{}\n", "<pre\n\n{}\n"])
def test_root_fences_html_comments_raw_nested_or_nonplain_have_no_authority(tmp_path, wrapper):
    module = _obs()
    hashes = module.root_gate_hashes({"observation_registry_sha": "a" * 64,
        "observation_source_tar_sha": "b" * 64, **_identity()})
    ledger = tmp_path / "合成台账.md"
    ledger.write_text(wrapper.format(_gate(hashes).rstrip()), encoding="utf-8")
    assert not module.root_active(ledger, hashes)


def test_root_same_cell_unique_record_six_sha_and_no_other_token_occurrence(tmp_path):
    module = _obs()
    hashes = module.root_gate_hashes({"observation_registry_sha": "a" * 64,
        "observation_source_tar_sha": "b" * 64, **_identity()})
    ledger = tmp_path / "合成台账.md"
    line = _gate(hashes)
    ledger.write_text(line, encoding="utf-8")
    assert module.root_active(ledger, hashes)
    for bad in (line * 2, line + "| 录-0130 | 不相关碰撞 | 合成 |\n", _gate(hashes, 107),
                line.replace(module.ROOT_TOKEN, "TASK11_NEW_MLP_OBSERVATIONS_GATE:v1"),
                line.replace("; TAR_SHA256=", " | TAR_SHA256="), line + "<!-- " + module.ROOT_TOKEN + " -->\n"):
        ledger.write_text(bad, encoding="utf-8")
        assert not module.root_active(ledger, hashes)


@pytest.mark.parametrize("change", ["source", "registry", "hf_registry", "root", "model", "replace_pin"])
def test_current_sources_models_and_root_drift_rejected(tmp_path, change):
    module = _obs()
    args, _ = _pack(tmp_path)
    checked = module.preflight_task11_howard_observations(**args)
    root = Path(args["project_root"])
    relative = {"source": module.SOURCE_MEMBERS[0], "registry": module.OBS_REGISTRY_PATH,
                "hf_registry": hf.REGISTRY, "root": hf.ROOT_LEDGER,
                "model": f"{hf.RUN_DIRECTORY}/正式Howard_HF_seed0/best.pt"}.get(change)
    if relative:
        path = root / relative
        path.write_bytes(path.read_bytes() + b"CPU drift")
    pins = {next(iter(checked["读取原件SHA256"])): "0" * 64} if change == "replace_pin" else {}
    with pytest.raises(ValueError, match="SHA|漂移|原件|ROOT"):
        module.verify_task11_howard_observation_runtime(checked, pins)


@pytest.mark.parametrize("change", ["ten_models", "bool_count", "wrong_hf_path", "permit", "float_counter", "duplicate_yaml"])
def test_strict_registry_rejects_alias_type_or_permission_drift(tmp_path, change):
    module = _obs()
    args, contract = _pack(tmp_path)
    if change == "ten_models": contract["模型十状态"][0]["模型状态"] = "final"
    elif change == "bool_count": contract["观察合同"]["每状态总点数"] = True
    elif change == "wrong_hf_path": contract["HowardHF四来源"]["registry"] = "旧MLP.yaml"
    elif change == "permit": contract["HF训练许可"] = True
    elif change == "float_counter": contract["观察合同"]["每状态总点数"] = 8024.
    registry = Path(args["observation_registry"])
    registry.write_text(yaml.safe_dump(contract, allow_unicode=True) +
        ("HF训练许可: false\n" if change == "duplicate_yaml" else ""), encoding="utf-8")
    args["observation_registry_sha"] = sha256_file(registry)
    (Path(args["project_root"]) / hf.ROOT_LEDGER).write_text(_gate(module.root_gate_hashes(args)), encoding="utf-8")
    with pytest.raises(ValueError): module.preflight_task11_howard_observations(**args)


def test_actual_qualifier_missing_sources_rejects_before_temperature_cuda_or_output(tmp_path, monkeypatch):
    module = _obs()
    args, _ = _pack(tmp_path)
    checked = module.preflight_task11_howard_observations(**args)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("不合格父来源读取PT"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("不合格父来源探测CUDA"))
    with pytest.raises(ValueError): module.qualify_task11_howard_observation_five(checked)
    assert not Path(args["output"]).exists()


def test_cpu_cannot_be_formal_cuda_or_forge_forward_receipt():
    module = _obs()
    with pytest.raises(ValueError): module.require_task11_howard_observation_cuda("cpu")
    with pytest.raises(ValueError): module.require_task11_howard_observation_cuda("cuda")
    with pytest.raises(ValueError): module.verify_task11_howard_cuda_receipts([])
    fake = [{"种子": n, "模型状态": s, "设备": "cuda:0", "输入设备": "cuda:0", "输出设备": "cuda:0",
        "真实CUDA前向": False, "Top点数": 7272, "热环点数": 376, "冷环点数": 376} for n in range(5) for s in module.STATES]
    with pytest.raises(ValueError): module.verify_task11_howard_cuda_receipts(fake)


def test_eight_chinese_cpu_outputs_and_limits_with_actual_new_writer(tmp_path, arrays):
    module = _obs()
    args, _ = _pack(tmp_path)
    checked = module.preflight_task11_howard_observations(**args)
    results = [_evaluate(arrays, seed, state) for seed in range(5) for state in module.STATES]
    summary = module.write_task11_howard_observation_evidence(checked, results, {}, {}, synthetic_cpu=True)
    destination = Path(args["output"])
    assert len(list(destination.iterdir())) == 8
    assert summary["总点数"] == 80240 and summary["验收计数"] == {"逐功率": 90, "时间窗": 270, "径向窗": 90}
    assert "Howard" in summary["方法"] and "非原文精确" in summary["方法"]
    for name in ("科学合格主张", "工程安全合格主张", "采用B0部署", "新增测量", "任务目标完成"):
        assert summary["限制"][name] is False
    manifest = json.loads((destination / "工件SHA256.json").read_text(encoding="utf-8"))
    assert len(manifest) == 7 and all(sha256_file(destination / n) == h for n, h in manifest.items())
    assert sum(1 for _ in (destination / "真实逐点预测.jsonl").open(encoding="utf-8")) == 80240
    assert sum(p["半径_米"] == float(np.float32(.025)) and p["模态原标识"] == "Top" for p in (
        json.loads(line) for line in (destination / "真实逐点预测.jsonl").open(encoding="utf-8"))) == 720
    with pytest.raises(FileExistsError):
        module.write_task11_howard_observation_evidence(checked, results, {}, {}, synthetic_cpu=True)


@pytest.mark.parametrize("change", ["raw_error", "metric", "reference", "source_row", "material", "missing_seed", "fake_q", "real_writer"])
def test_writer_rejects_tampered_points_group_metrics_seed_or_fake_qualification(tmp_path, arrays, change):
    module = _obs()
    args, _ = _pack(tmp_path)
    checked = module.preflight_task11_howard_observations(**args)
    results = [_evaluate(arrays, seed, state) for seed in range(5) for state in module.STATES]
    if change == "raw_error": results[0]["points"][0]["error_c"] += 1
    elif change == "metric": results[0]["radial"][0]["rmse_c"] += 1
    elif change == "reference": results[0]["points"][0]["reference_time_s"] = 0.
    elif change == "source_row": results[0]["points"][0]["source_row"] = 1
    elif change == "material": results[0]["points"][0]["material_id"] = 0
    elif change == "missing_seed": results = results[:-1]
    qualifications = {0: {"状态": "伪CPU完整资格"}} if change == "fake_q" else {}
    with pytest.raises(ValueError):
        module.write_task11_howard_observation_evidence(checked, results, qualifications, qualifications,
                                                        synthetic_cpu=change != "real_writer")
    assert not Path(args["output"]).exists()


def test_cli_is_howard_own_export_and_has_no_cpu_bypass():
    path = PROJECT_ROOT / "scripts/58_export_task11_howard_observations.py"
    assert path.is_file(), "Howard 三模态导出入口不存在"
    source = path.read_text(encoding="utf-8")
    assert "export_task11_howard_observations" in source
    assert "choices=[\"cuda\"]" in source
    assert "synthetic_cpu" not in source


@pytest.mark.parametrize("kind", ["source", "model"])
@pytest.mark.parametrize("phase", ["artifact", "manifest"])
def test_writer_detects_drift_during_actual_export_and_retains_failed_output(tmp_path, arrays, monkeypatch, kind, phase):
    module = _obs()
    args, _ = _pack(tmp_path)
    checked = module.preflight_task11_howard_observations(**args)
    results = [_evaluate(arrays, seed, state) for seed in range(5) for state in module.STATES]
    root, output = Path(args["project_root"]), Path(args["output"])
    source = root / "src/sic_cu/eval/development_v4.py" if kind == "source" else Path(checked["模型十身份"][(4, "final")]["模型原件"])
    selected_name = "机器汇总.json" if phase == "artifact" else "工件SHA256.json"
    write_text = Path.write_text
    injected = []

    def write_and_drift(path, text, *positional, **keywords):
        written = write_text(path, text, *positional, **keywords)
        if path.parent == output and path.name == selected_name and not injected:
            source.write_bytes(source.read_bytes() + b"CPU fixture actual export drift")
            injected.append(True)
        return written

    monkeypatch.setattr(Path, "write_text", write_and_drift)
    with pytest.raises(ValueError, match="SHA|漂移"):
        module.write_task11_howard_observation_evidence(checked, results, {}, {}, synthetic_cpu=True)
    assert injected == [True]
    assert output.is_dir() and (output / selected_name).is_file()
