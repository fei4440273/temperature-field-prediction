"""Independent real IO controls; all models/data/ROOT/qualifications are synthetic."""

import copy
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
import yaml

from sic_cu.config import PROJECT_ROOT
from sic_cu.data import processed
from sic_cu.data.common import sha256_file
from sic_cu.data.experiment import radial_observations
from sic_cu.eval import task11_howard_observations as obs
from sic_cu.models.task11_howard_composite import Task11HowardComposite
from sic_cu.train import task11_howard_hf_formal as hf


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _gate(hashes):
    return "| 录-0140 | " + obs.ROOT_TOKEN + "; status=active; " + "; ".join(
        f"{k}={hashes[k]}" for k in obs.ROOT_IDENTITY_FIELDS) + " | 仅合成CPU，不是真实权限 |\n"


@pytest.fixture(scope="module")
def io_frames(tmp_path_factory):
    root = tmp_path_factory.mktemp("纯合成原始CSV与Parquet")
    raw = root / "原始合成CSV"
    raw.mkdir()
    top_frames, raw_rings = [], []
    checked_variance = []
    for power, count, ring_count in ((115.2, 19, 100), (403., 24, 126), (630.5, 29, 150)):
        times = np.linspace(35, 200, count, dtype=np.float32)
        times[np.argmin(abs(times - 100))] = np.float32(100)
        for index, time in enumerate(times):
            radii_mm = np.arange(101, dtype=np.float64) * .25
            spread = .11 + (np.arange(101) % 11) * (.02 + .006 * index) + .002 * index * (np.arange(101) % 5)
            center = 40 + float(time) * .05 + radii_mm * .1
            path = raw / f"{power}W-{float(time):.17g}s.csv"
            pl.DataFrame({"r_mm": np.repeat(radii_mm, 2),
                "temperature_c": np.column_stack((center - spread, center + spread)).reshape(-1)}).write_csv(path)
            native = radial_observations(path)
            expected_raw = np.clip(1 / (spread ** 2 + .01), .1, 10).astype(np.float32)
            np.testing.assert_allclose(native["reliability_weight_raw"].to_numpy(), expected_raw, rtol=1e-6, atol=1e-7)
            expected = (native["reliability_weight_raw"].to_numpy() / np.sum(
                native["reliability_weight_raw"].to_numpy(), dtype=np.float32)).astype(np.float32)
            np.testing.assert_allclose(native["frame_weight"].to_numpy(), expected, rtol=2e-7, atol=1e-9)
            checked_variance.append(native["frame_weight"].dtype == pl.Float32)
            top_frames.append(native.with_columns(pl.lit("validation").alias("split")))
        for kind, radius, base, first in (("hot", 20., 32., 5.), ("cold", 40., 29., 7.)):
            times = np.linspace(first, 200, ring_count, dtype=np.float32)
            for edge in (30., 100.):
                times[np.argmin(abs(times - edge))] = np.float32(edge)
            temperatures = base + times.astype(np.float64) * .11 + power * .002
            for time, value in zip(times, temperatures):
                raw_rings.append((np.float32(power), float(time), kind, radius, value,
                                  float(value - temperatures[0]), "validation"))
    top = pl.concat(top_frames).sort("power_w", "time_s", "r_m")
    assert all(checked_variance) and top.height == 7272
    data = root / "data/processed"
    data.mkdir(parents=True)
    train_top = top.slice(0, 1).with_columns(pl.lit("train").alias("split"))
    pl.concat([top, train_top]).write_parquet(data / "experiment_ir_radial.parquet")
    rings = pl.DataFrame(raw_rings, schema=[("power_w", pl.Float32), ("time_raw", pl.Float64),
        ("sensor_type", pl.String), ("radius_raw", pl.Float64), ("value_mean_raw", pl.Float64),
        ("delta_value_raw", pl.Float64), ("split", pl.String)], orient="row")
    train_ring = rings.slice(0, 1).with_columns(pl.lit("train").alias("split"))
    pl.concat([rings, train_ring]).write_parquet(data / "sensor_ring_raw.parquet")
    metadata = root / "configs/data_metadata.yaml"
    metadata.parent.mkdir()
    metadata.write_text(yaml.safe_dump({"sensors": {"coordinate_unit_status": "verified",
        "value_unit_status": "verified", "time_unit_status": "verified", "synchronized_start": True,
        "coordinate_unit": "mm", "time_unit": "s", "time_column_interpretation": "elapsed_time_t_equals_index",
        "value_unit": "degC"}}), encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(processed, "PROJECT_ROOT", root)
        loaded_top = obs.load_processed_ir_observations("validation").sort("power_w", "time_s", "r_m")
    loaded_sensors = obs.load_canonical_sensor_observations(metadata_path=str(metadata),
        processed_path=str(data / "sensor_ring_raw.parquet"), split="validation").sort("power_w", "sensor_type", "time_s")
    assert loaded_top.equals(top) and loaded_sensors.height == 752
    assert loaded_top["frame_weight"].dtype == loaded_top["r_m"].dtype == pl.Float32
    assert loaded_sensors["temperature_k"].dtype == loaded_sensors["delta_temperature_k"].dtype == pl.Float32
    return root, loaded_top, loaded_sensors


@pytest.fixture(scope="module")
def ten_results(io_frames):
    _, top, sensors = io_frames
    error = 3 * (top["r_m"].to_numpy().astype(float) / .025) ** 2 + (top["time_s"].to_numpy().astype(float) - 35) / 80
    results = []
    for seed in range(5):
        for state in obs.STATES:
            state_shift = .1 if state == "final" else 0.
            pt = top["temperature_mean_k"].to_numpy().astype(float) + error + seed * .03 + state_shift
            times = sensors["time_s"].to_numpy().astype(float)
            se = np.where(sensors["sensor_type"].to_numpy() == "hot", -.5 - times / 400, 2 + times / 200)
            ps = sensors["temperature_k"].to_numpy().astype(float) + se + seed * .04 + state_shift
            result = obs.evaluate_task11_howard_observation_arrays(top, sensors, pt, ps,
                seed=seed, state=state, bottom_z_m=-.0175)
            result["macro_reproduction"] = obs.reconstruct_task11_observation_macro(result["power"])
            results.append(result)
    return results


def _view(seed, source_identity):
    model = hf._fresh_cpu_model()
    return {"schema_version": 1, "method": hf.METHOD, "seed": seed, "epoch": 11,
        "model_kwargs": copy.deepcopy(model.model_kwargs), "scales": copy.deepcopy(hf.HF_BUDGET["scales"]),
        "model_state": model.state_dict(), "HF训练许可": False, **dict.fromkeys(hf.ISOLATION_FLAGS, False),
        "任11HowardHF事前来源": {"method": hf.METHOD, "seed": seed,
            "固定PHQH查询": hf._query_metadata("a" * 64),
            **dict(zip(hf.IDENTITY_FIELDS, [source_identity[n + "_sha"] for n in
                      ("registry", "source_tar", "lf_catalog", "hf_data_catalog")]))}}


@pytest.fixture
def pack(tmp_path):
    root = tmp_path / "合成来源模型与ROOT"
    root.mkdir()
    for name in obs.SOURCE_MEMBERS:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / name).read_bytes())
    hashes = {name: sha256_file(root / name) for name in obs.SOURCE_MEMBERS}
    lf_rows = []
    for seed in range(5):
        run = root / hf.LF_RUN_DIRECTORY / f"正式Howard_LF_seed{seed}"
        originals = {}
        for name in ("best.pt", "final.pt", "metrics.json", "training.jsonl", "阶段_最近.pt",
                     "事前真实来源登记.json", "config_snapshot/geometry.yaml", "额外资格原件.json"):
            path = run / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic LF seed{seed} {name}\n".encode())
            originals[name] = sha256_file(path)
        lf_rows.append({"seed": seed, "真实工件SHA256": originals})
    lf_sources = {}
    for name, relative in hf.LF_SOURCE_IDENTITY.items():
        if name.endswith("_sha"):
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic LF source {name}\n".encode())
        lf_sources[name], lf_sources[name + "_sha"] = relative, sha256_file(path)
    data_rows = []
    for relative in ("data/processed/experiment_ir_radial.parquet", "data/processed/sensor_ring_raw.parquet"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic pinned data not opened as parquet in qualification test\n")
        data_rows.append({"原件": relative, "文件SHA256": sha256_file(path)})
    identity = {}
    for name, relative in zip(("registry", "source_tar", "lf_catalog", "hf_data_catalog"),
                             (hf.REGISTRY, hf.SOURCE_TAR, hf.LF_CATALOG, hf.HF_DATA_CATALOG)):
        payload = {"synthetic_parent": name}
        if name == "lf_catalog":
            payload = {"五LF真实完整资格": lf_rows}
        elif name == "hf_data_catalog":
            payload = {"HF源原件": data_rows, "HF合法验证功率_瓦": list(obs.FIXED_POWERS),
                       "旧固定TEST温度读取": False, "模拟测试功率温度读取": False}
        _write_json(root / relative, payload)
        identity[name], identity[name + "_sha"] = relative, sha256_file(root / relative)
    models, qualifications = [], {}
    for seed in range(5):
        run = root / hf.RUN_DIRECTORY / f"正式Howard_HF_seed{seed}"
        run.mkdir(parents=True, exist_ok=True)
        view = _view(seed, identity)
        for state in obs.STATES:
            path = run / (state + ".pt")
            torch.save(view, path)
            models.append({"种子": seed, "模型状态": state, "模型原件": path.relative_to(root).as_posix(),
                           "模型SHA256": sha256_file(path)})
        originals = {state + ".pt": sha256_file(run / (state + ".pt")) for state in obs.STATES}
        for name in ("metrics.json", "training.jsonl", "阶段_最近.pt", "事前真实来源登记.json",
                     "config_snapshot/geometry.yaml", "额外资格原件.json"):
            path = run / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic HF seed{seed} {name}\n".encode())
            originals[name] = sha256_file(path)
        qualifications[seed] = {"状态": obs.HF_QUALIFIED_STATUS, "seed": seed, "已完成": True, "HF训练许可": False,
            "事前来源": view["任11HowardHF事前来源"], "真实工件SHA256": originals,
            "最佳HF检查点SHA256": originals["best.pt"], "真实末HF检查点SHA256": originals["final.pt"],
            "限制": copy.deepcopy(hf.LIMITATIONS)}
    archive = root / obs.OBS_TAR_PATH
    with tarfile.open(archive, "w:gz", format=tarfile.USTAR_FORMAT) as tar:
        for name in obs.SOURCE_MEMBERS:
            content = (root / name).read_bytes()
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    contract = obs.registration_contract(source_hashes=hashes, source_tar_sha=sha256_file(archive),
        hf_source_identity=identity, model_catalog=models, registered_at="synthetic independent review only",
        root_before_sha=hashlib.sha256(b"synthetic pre-registration ROOT").hexdigest())
    registry = root / obs.OBS_REGISTRY_PATH
    registry.write_text(yaml.safe_dump(contract, allow_unicode=True), encoding="utf-8")
    args = {"observation_registry": registry, "observation_registry_sha": sha256_file(registry),
        "observation_source_tar": archive, "observation_source_tar_sha": sha256_file(archive),
        "output": root / obs.OUTPUT_BASE / "独立真写出_CPU合成", "project_root": root, **identity}
    (root / hf.ROOT_LEDGER).write_text(_gate(obs.root_gate_hashes(args)), encoding="utf-8")
    checked = obs.preflight_task11_howard_observations(**args)
    return {"root": root, "checked": checked, "args": args, "contract": contract,
            "q": qualifications, "lf_rows": lf_rows, "lf_sources": lf_sources, "data_rows": data_rows}


def _full_pins(pack, monkeypatch):
    monkeypatch.setattr(hf, "LF_SOURCE_IDENTITY", pack["lf_sources"])
    monkeypatch.setattr(hf, "qualify_task11_howard_hf_source", lambda **a: copy.deepcopy(pack["q"][a["seed"]]))
    qualified, pins = obs.qualify_task11_howard_observation_five(pack["checked"])
    expected = {str(pack["root"] / hf.RUN_DIRECTORY / f"正式Howard_HF_seed{s}" / n): d
                for s, q in pack["q"].items() for n, d in q["真实工件SHA256"].items()}
    expected.update({str(pack["root"] / hf.LF_RUN_DIRECTORY / f"正式Howard_LF_seed{q['seed']}" / n): d
                     for q in pack["lf_rows"] for n, d in q["真实工件SHA256"].items()})
    expected.update({str(pack["root"] / relative): pack["lf_sources"][n + "_sha"]
                     for n, relative in pack["lf_sources"].items() if not n.endswith("_sha")})
    expected.update({str(pack["root"] / row["原件"]): row["文件SHA256"] for row in pack["data_rows"]})
    assert qualified == pack["q"] and pins == expected and len(pins) == 85
    assert len(pack["checked"]["读取原件SHA256"]) == 107
    assert len({**pack["checked"]["读取原件SHA256"], **pins}) == 182
    return pins


def _target(pack, kind):
    root = pack["root"]
    return {"source": root / "src/sic_cu/eval/development_v4.py",
        "model": Path(pack["checked"]["模型十身份"][(4, "final")]["模型原件"]),
        "parent": root / hf.REGISTRY, "root": root / hf.ROOT_LEDGER,
        "extra_hf": root / hf.RUN_DIRECTORY / "正式Howard_HF_seed2/额外资格原件.json",
        "extra_lf": root / hf.LF_RUN_DIRECTORY / "正式Howard_LF_seed3/额外资格原件.json",
        "data": root / "data/processed/sensor_ring_raw.parquet"}[kind]


def test_independent_full_float32_radial_ulp_and_time_boundaries(io_frames, ten_results):
    endpoints = np.asarray([0, .008, .017, .025], dtype=np.float32)
    radii = np.asarray([endpoints[0], np.nextafter(endpoints[1], np.float32(0)), endpoints[1],
        np.nextafter(endpoints[1], np.float32(1)), np.nextafter(endpoints[2], np.float32(0)), endpoints[2],
        np.nextafter(endpoints[2], np.float32(1)), np.nextafter(endpoints[3], np.float32(0)), endpoints[3]], dtype=np.float32)
    masks = obs.native_radial_masks(radii)
    assert np.argmax(np.stack(masks), axis=0).tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert np.all(sum(x.astype(int) for x in masks) == 1)
    _, _, sensors = io_frames
    for edge, expected in ((30., 0), (100., 1), (200., 2)):
        assert int((sensors["time_s"] == edge).sum()) == 6
        selected = [(edge >= w["lower"] if w["lower_closed"] else edge > w["lower"]) and edge <= w["upper"] for w in obs.TIME_WINDOWS]
        assert selected == [i == expected for i in range(3)]
    assert sum(p["r_m"] == float(np.float32(.025)) and p["modality"] == "Top" for r in ten_results for p in r["points"]) == 720
    assert sum(row["sample_count"] == 0 for r in ten_results for row in r["time"]) == 30


def test_true_csv_inverse_variance_parquet_raw_weights_and_window_statistics(io_frames, ten_results):
    root, top, sensors = io_frames
    frame_before = top.clone()
    result = ten_results[0]
    points = [p for p in result["points"] if p["modality"] == "Top"]
    assert np.array_equal(np.asarray([p["weight"] for p in points]), top["frame_weight"].to_numpy().astype(float))
    errors = np.asarray([p["error_c"] for p in points])
    power = top["power_w"].to_numpy() == np.float32(115.2)
    outer = power & obs.native_radial_masks(top["r_m"].to_numpy())[2]
    weights = top["frame_weight"].to_numpy().astype(float)[outer]
    row = next(r for r in result["radial"] if r["power_w"] == 115.2 and r["radial_window"] == "outer_17_25_mm")
    expected_rmse = np.sqrt(np.sum(weights * errors[outer] ** 2) / weights.sum())
    assert row["rmse_c"] == pytest.approx(expected_rmse, abs=1e-11)
    assert row["p95_abs_error_c"] == pytest.approx(np.quantile(abs(errors[outer]), .95), abs=1e-11)
    times = top["time_s"].to_numpy()[outer]
    altered = weights.copy()
    for time in np.unique(times):
        selected = times == time
        altered[selected] /= altered[selected].sum()
    wrong = np.sqrt(np.sum(altered * errors[outer] ** 2) / altered.sum())
    assert abs(wrong - expected_rmse) > 1e-5
    assert top.equals(frame_before)
    for modality, first in (("Hot", 5.), ("Cold", 7.)):
        assert {p["reference_time_s"] for p in result["points"] if p["modality"] == modality} == {first}
    print("REAL_SYNTHETIC_IO_COUNTS", top.height, sensors.height, "WEIGHT_DTYPES", top["frame_weight"].dtype,
          "RAW_VS_WINDOW_RENORMALIZED_RMSE", expected_rmse, wrong)


def test_true_ten_pt_catalog_and_strict_factory_precedes_cpu_device(pack, monkeypatch):
    validated, moved = [], []
    validate, move = hf._validate_model, Task11HowardComposite.to
    def tracked_validate(model):
        assert all(p.device.type == "cpu" for p in model.parameters())
        value = validate(model)
        validated.append(id(model))
        return value
    def tracked_move(model, *args, **kwargs):
        assert id(model) in validated
        assert {p.dtype for p in model.parameters()} == {torch.float32}
        assert {b.dtype for b in model.buffers()} == {torch.float64}
        moved.append(id(model))
        return move(model, *args, **kwargs)
    monkeypatch.setattr(hf, "_validate_model", tracked_validate)
    monkeypatch.setattr(Task11HowardComposite, "to", tracked_move)
    for row in pack["contract"]["模型十状态"]:
        path = pack["root"] / row["模型原件"]
        assert sha256_file(path) == row["模型SHA256"]
        view = torch.load(path, map_location="cpu", weights_only=True)
        saved = {n: t.clone() for n, t in view["model_state"].items()}
        model = obs.task11_howard_observation_model_from_view(view, seed=row["种子"]).to(torch.device("cpu"))
        assert len(list(model.parameters())) == 56
        coords = np.asarray([[.008, 0., 35., 115.2, 1.]], dtype=np.float32)
        prediction = obs.predict_task11_howard_observations(model, coords, torch.device("cpu"), 1)
        assert prediction.shape == (1,) and np.isfinite(prediction).all()
        assert all(torch.equal(saved[n], t) for n, t in view["model_state"].items())
    assert len(moved) == 10
    print("TEN_SYNTHETIC_PT_REAL_LOADS_STRICT_THEN_CPU_MOVE", len(moved))


@pytest.mark.parametrize("dtype_change", ["query32", "parameter64"])
def test_invalid_saved_dtype_rejected_without_device_repair(pack, monkeypatch, dtype_change):
    view = torch.load(pack["root"] / pack["contract"]["模型十状态"][0]["模型原件"], map_location="cpu", weights_only=True)
    name = "linear_query_points" if dtype_change == "query32" else next(n for n in view["model_state"] if "weight" in n)
    view["model_state"][name] = view["model_state"][name].float() if dtype_change == "query32" else view["model_state"][name].double()
    monkeypatch.setattr(Task11HowardComposite, "to", lambda *a, **k: pytest.fail("invalid dtype reached device move"))
    with pytest.raises(ValueError):
        obs.task11_howard_observation_model_from_view(view, seed=0).to(torch.device("cpu"))


def test_true_writer_all182_pins_full_io_and_manifest(pack, ten_results, monkeypatch):
    pins = _full_pins(pack, monkeypatch)
    summary = obs.write_task11_howard_observation_evidence(pack["checked"], ten_results, {}, {}, additional_pins=pins, synthetic_cpu=True)
    output = Path(pack["args"]["output"])
    manifest = json.loads((output / "工件SHA256.json").read_text(encoding="utf-8"))
    assert len(manifest) == 7 and len(list(output.iterdir())) == 8
    assert all(sha256_file(output / n) == digest for n, digest in manifest.items())
    assert summary["读取原件执行前后SHA256"] == {**pack["checked"]["读取原件SHA256"], **pins}
    assert len(summary["读取原件执行前后SHA256"]) == 182
    assert summary["限制"]["CPU合成回归"] is True and summary["十状态真实CUDA前向收据"] == []
    assert all(summary["限制"][n] is False for n in ("科学合格主张", "工程安全合格主张", "任务目标完成"))
    points = [json.loads(line) for line in (output / "真实逐点预测.jsonl").open(encoding="utf-8")]
    assert len(points) == 80240
    assert sum(p[obs.POINT_ZH["r_m"]] == float(np.float32(.025)) and p[obs.POINT_ZH["modality"]] == "Top" for p in points) == 720
    for file, count in (("逐功率三模态.csv", 90), ("逐功率三模态时间窗.csv", 270), ("顶部逐功率原生径向窗.csv", 90)):
        with (output / file).open(encoding="utf-8", newline="") as stream:
            assert len(list(csv.DictReader(stream))) == count
    obs.verify_task11_howard_observation_runtime(pack["checked"], pins)
    print("TRUE_CPU_WRITER_PINS", 107, len(pins), len(summary["读取原件执行前后SHA256"]), "OUTPUTS", len(manifest) + 1)


@pytest.mark.parametrize("kind", ["source", "model", "parent", "root", "extra_hf", "extra_lf", "data"])
def test_complete_original_and_additional_pins_reject_before_output(pack, ten_results, monkeypatch, kind):
    pins = _full_pins(pack, monkeypatch)
    target = _target(pack, kind)
    target.write_bytes(target.read_bytes() + b"synthetic actual pre-write byte drift\n")
    with pytest.raises(ValueError, match="SHA|漂移"):
        obs.write_task11_howard_observation_evidence(pack["checked"], ten_results, {}, {}, additional_pins=pins, synthetic_cpu=True)
    assert not Path(pack["args"]["output"]).exists()


@pytest.mark.parametrize("kind", ["source", "model", "extra_lf", "data"])
@pytest.mark.parametrize("phase", ["csv_after_actual_rows", "manifest_during_actual_hash"])
def test_true_writer_csv_and_manifest_hash_byte_drift_retained(pack, ten_results, monkeypatch, kind, phase):
    pins = _full_pins(pack, monkeypatch)
    target, output = _target(pack, kind), Path(pack["args"]["output"])
    fired = []
    def drift():
        target.write_bytes(target.read_bytes() + b"independent synthetic actual writer seal drift\n")
        fired.append(True)
    if phase == "csv_after_actual_rows":
        original = csv.DictWriter.writerows
        calls = []
        def real_rows(writer, rows):
            value = original(writer, rows)
            calls.append(True)
            if len(calls) == 3:
                drift()
            return value
        monkeypatch.setattr(csv.DictWriter, "writerows", real_rows)
    else:
        original = obs.sha256_file
        def real_hash(path):
            value = original(path)
            if Path(path).parent == output and not fired:
                drift()
            return value
        monkeypatch.setattr(obs, "sha256_file", real_hash)
    with pytest.raises(ValueError, match="SHA|漂移"):
        obs.write_task11_howard_observation_evidence(pack["checked"], ten_results, {}, {}, additional_pins=pins, synthetic_cpu=True)
    assert fired == [True] and output.is_dir()
    assert (output / "顶部逐功率原生径向窗.csv").is_file()
    assert (output / "工件SHA256.json").exists() is (phase == "manifest_during_actual_hash")
    print("REAL_WRITER_DRIFT_REJECTED_RETAINED", phase, kind, str(output))


@pytest.mark.parametrize("kind", ["hardlink", "directory", "missing"])
def test_independent_ordinary_source90_tar_boundary(pack, monkeypatch, kind):
    archive = Path(pack["args"]["observation_source_tar"])
    with tarfile.open(archive, "w:gz") as tar:
        for index, name in enumerate(obs.SOURCE_MEMBERS):
            if index == 0 and kind == "missing":
                continue
            content = (pack["root"] / name).read_bytes()
            member = tarfile.TarInfo(name)
            member.size = len(content)
            if index == 0 and kind == "hardlink":
                member.type, member.linkname, member.size = tarfile.LNKTYPE, obs.SOURCE_MEMBERS[1], 0
            elif index == 0 and kind == "directory":
                member.type, member.size = tarfile.DIRTYPE, 0
            tar.addfile(member, io.BytesIO(content))
    contract = copy.deepcopy(pack["contract"])
    contract["源码冻结tarSHA256"] = sha256_file(archive)
    registry = Path(pack["args"]["observation_registry"])
    registry.write_text(yaml.safe_dump(contract, allow_unicode=True), encoding="utf-8")
    args = {**pack["args"], "observation_source_tar_sha": sha256_file(archive), "observation_registry_sha": sha256_file(registry)}
    (pack["root"] / hf.ROOT_LEDGER).write_text(_gate(obs.root_gate_hashes(args)), encoding="utf-8")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("malformed tar reached deserialization"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("malformed tar reached CUDA"))
    with pytest.raises(ValueError, match="tar|成员|普通"):
        obs.preflight_task11_howard_observations(**args)


@pytest.mark.parametrize("change", ["cross_seed", "lf_path", "bool_seed", "duplicate_view"])
def test_independent_model_catalog_ten_views_cannot_alias(pack, change):
    rows = copy.deepcopy(pack["contract"]["模型十状态"])
    if change == "cross_seed":
        rows[0]["模型原件"] = rows[2]["模型原件"]
        rows[0]["模型SHA256"] = rows[2]["模型SHA256"]
    elif change == "lf_path":
        rows[0]["模型原件"] = f"{hf.LF_RUN_DIRECTORY}/正式Howard_LF_seed0/best.pt"
    elif change == "bool_seed":
        rows[0]["种子"] = False
    else:
        rows[1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match="十模型|本人|十个状态"):
        obs._catalog_schema(rows)


def test_independent_root_active_pin_append_inside_own_short_window_rejected(pack):
    checked = pack["checked"]
    ledger = pack["root"] / hf.ROOT_LEDGER
    assert obs.root_active(ledger, checked["ROOT六SHA"])
    ledger.write_text(ledger.read_text(encoding="utf-8") + "| 录-0141 | 合成普通无关追加 | CPU |\n", encoding="utf-8")
    assert obs.root_active(ledger, checked["ROOT六SHA"])
    with pytest.raises(ValueError, match="SHA|ROOT|漂移"):
        obs.verify_task11_howard_observation_runtime(checked, {})
    assert not Path(pack["args"]["output"]).exists()
