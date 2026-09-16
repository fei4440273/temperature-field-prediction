"""版本化顶部径向端点审计：仅合法验证，不打开旧TEST温度。"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import polars as pl
import pytest

from sic_cu.data.common import sha256_file
from sic_cu.config import PROJECT_ROOT


def _entry():
    script = Path(__file__).resolve().parents[1] / "scripts/30_audit_task07_radial_endpoint.py"
    assert script.is_file(), "版本化径向审计入口尚不存在"
    spec = importlib.util.spec_from_file_location("task07_radial_endpoint_audit", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WINDOWS = (
    {"name": "center_0_8_mm", "lower": 0.0, "upper": 8.0, "lower_closed": True},
    {"name": "middle_8_17_mm", "lower": 8.0, "upper": 17.0, "lower_closed": False},
    {"name": "outer_17_25_mm", "lower": 17.0, "upper": 25.0, "lower_closed": False},
)


def test_native_meter_masks_restore_only_represented_25_mm_endpoint() -> None:
    entry = _entry()
    radii = np.asarray([0.0, 0.004, 0.00812029, 0.01712109, 0.0247, 0.025], dtype=np.float32)
    old, fixed = entry.radial_masks(radii, WINDOWS)
    assert [int(mask.sum()) for mask in old] == [2, 1, 2]
    assert [int(mask.sum()) for mask in fixed] == [2, 1, 3]
    assert np.array_equal(fixed[-1] & ~old[-1], radii == np.float32(0.025))
    assert sum(mask.astype(np.int8) for mask in fixed).tolist() == [1] * len(radii)


def test_native_meter_mask_refuses_outside_radius_or_unsupported_dtype() -> None:
    entry = _entry()
    with pytest.raises(ValueError, match="25|定义域|半径"):
        entry.radial_masks(np.asarray([0.025, 0.02501], dtype=np.float32), WINDOWS)
    with pytest.raises(ValueError, match="float32|半径"):
        entry.radial_masks(np.asarray([0.025], dtype=np.float64), WINDOWS)


def test_source_registration_refuses_existing_directory(tmp_path) -> None:
    entry = _entry()
    existing = tmp_path / "历史已有文件"
    existing.mkdir()
    original = existing / "原件.txt"
    original.write_bytes(b"unchanged")
    with pytest.raises(FileExistsError, match="不可覆盖|已存在"):
        entry.register_source_snapshot(existing)
    assert original.read_bytes() == b"unchanged"


def test_missing_or_mislabeled_seed_refuses_audit_before_writing(tmp_path) -> None:
    entry = _entry()
    output = tmp_path / "正式审核不可产生"
    with pytest.raises(ValueError, match="五种子|种子"):
        entry.audit_radial_endpoint(
            {0: tmp_path / "seed0"}, output,
            source_manifest=tmp_path / "还不存在.json",
            source_manifest_sha256="0" * 64,
        )
    assert not output.exists()


def test_each_model_reuses_exact_predicted_top_vector_for_both_masks() -> None:
    entry = _entry()
    radii = np.asarray([0.0, 0.004, 0.00812029, 0.01712109, 0.0247, 0.025], dtype=np.float32)
    frame = pl.DataFrame({
        "r_m": radii,
        "time_s": np.full(6, 5.0, dtype=np.float32),
        "temperature_mean_k": np.full(6, 300.0, dtype=np.float32),
        "frame_weight": np.full(6, 1.0 / 6, dtype=np.float32),
    })
    predictions = np.asarray([301., 299., 300., 298., 300., 302.])
    original, corrected = entry.paired_radial_records(
        frame, predictions, WINDOWS, seed=0, power_w=115.2, arm="B0"
    )
    assert len(original) == len(corrected) == 3
    assert [item["sample_count"] for item in original] == [2, 1, 2]
    assert [item["sample_count"] for item in corrected] == [2, 1, 3]
    for idx in (0, 1):
        assert original[idx]["rmse_c"] == corrected[idx]["rmse_c"]
        assert original[idx]["p95_abs_error_c"] == corrected[idx]["p95_abs_error_c"]
    assert original[2]["rmse_c"] != corrected[2]["rmse_c"]
    assert {row["split"] for row in original + corrected} == {"validation"}


def test_registration_archives_exact_source_files_and_byte_sha_before_cuda(tmp_path) -> None:
    entry = _entry()
    registered = entry.register_source_snapshot(tmp_path / "新登记")
    manifest = Path(registered["source_manifest"])
    archive = Path(registered["source_archive"])
    assert manifest.parent == archive.parent == tmp_path / "新登记"
    assert sha256_file(manifest) == registered["source_manifest_sha256"]
    assert sha256_file(archive) == registered["source_archive_sha256"]
    loaded = entry.verify_source_snapshot(manifest, registered["source_manifest_sha256"])
    assert loaded["V4历史来源清单SHA256"] == (
        "423d287de76058d7124c9072e8d094bc5c960e48f4e7b99d7246feda2988c4de"
    )
    assert loaded["V4原配置SHA256"] == (
        "59c1e7f569ba846effc5b3fd7f1776d55f4c2341b25f012809e60b4c6b913334"
    )
    for required in ("scripts/30_audit_task07_radial_endpoint.py",
                     "tests/test_task07_radial_endpoint_audit.py",
                     "src/sic_cu/eval/development_v4.py"):
        assert required in loaded["源码逐文件SHA256"]


def test_forged_self_signed_registration_does_not_authorize_new_source(tmp_path) -> None:
    entry = _entry()
    registered = entry.register_source_snapshot(tmp_path / "真登记")
    manifest = Path(registered["source_manifest"])
    forged = tmp_path / "假登记.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["源码逐文件SHA256"]["scripts/30_audit_task07_radial_endpoint.py"] = "0" * 64
    forged.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="源码|漂移|SHA"):
        entry.verify_source_snapshot(forged, sha256_file(forged))


def test_existing_v4_and_task07_observation_metric_files_have_independent_sha() -> None:
    entry = _entry()
    sources = entry._locked_reference_metrics()
    assert len(sources["B0"]) == len(sources["任07E0"]) == 45
    for arm in ("B0", "任07E0"):
        expected = {115.2: 1919, 403.0: 2424, 630.5: 2929}
        for seed in range(5):
            for power, full in expected.items():
                rows = [row for row in sources[arm] if row["seed"] == seed
                        and row["power_w"] == power]
                assert len(rows) == 3
                assert sum(row["sample_count"] for row in rows) == full - {115.2: 19,
                                                                            403.0: 24,
                                                                            630.5: 29}[power]


def test_actual_top_validation_counts_are_untouched_by_radial_display() -> None:
    entry = _entry()
    frame = entry._locked_validation_top_frame()
    assert frame.height == 1919 + 2424 + 2929
    assert frame["r_m"].dtype == pl.Float32
    assert {round(float(value), 4) for value in frame["power_w"].unique().to_list()} == {
        115.2, 403.0, 630.5,
    }


def test_bad_manifest_shas_refused_before_cuda_or_new_output(tmp_path) -> None:
    entry = _entry()
    destination = tmp_path / "拒绝假来源"
    fake_roots = {seed: tmp_path / f"seed{seed}" for seed in range(5)}
    with pytest.raises(ValueError, match="SHA|登记|来源"):
        entry.audit_radial_endpoint(
            fake_roots, destination, source_manifest=tmp_path / "假登记.json",
            source_manifest_sha256="0" * 64,
        )
    assert not destination.exists()


def test_bare_pinn_cli_help_does_not_depend_on_ros_scripts_package() -> None:
    script = PROJECT_ROOT / "scripts/30_audit_task07_radial_endpoint.py"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        ["/home/phl/anaconda3/envs/PINN/bin/python", str(script), "--help"],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--register-source" in result.stdout and "--source-manifest-sha256" in result.stdout


def test_legacy_reference_metric_reproduction_rejects_forged_original() -> None:
    entry = _entry()
    source = entry._locked_reference_metrics()["B0"][0]
    check = source.copy()
    entry._compare_legacy_original(check, source)
    check["rmse_c"] += 0.05
    with pytest.raises(ValueError, match="RMSE|原指标|历史"):
        entry._compare_legacy_original(check, source)
    check = source.copy()
    check["sample_count"] = 1
    with pytest.raises(ValueError, match="点|原指标|历史"):
        entry._compare_legacy_original(check, source)


def test_three_windows_must_repair_exactly_one_endpoint_per_real_frame() -> None:
    entry = _entry()
    old = [{"sample_count": count} for count in (608, 684, 608)]
    fixed = [{"sample_count": count} for count in (608, 684, 627)]
    entry._check_power_partition(115.2, old, fixed, 19)
    fixed[-1] = {"sample_count": 626}
    with pytest.raises(ValueError, match="25|完整|端点"):
        entry._check_power_partition(115.2, old, fixed, 19)


def test_registered_but_missing_formal_run_rejects_before_cuda_or_output(tmp_path) -> None:
    entry = _entry()
    registered = entry.register_source_snapshot(tmp_path / "新来源")
    destination = tmp_path / "种子目录实际不存在"
    fake_roots = {seed: tmp_path / f"missing_seed{seed}" for seed in range(5)}
    with pytest.raises((FileNotFoundError, ValueError), match="正式|目录|种子|不存在"):
        entry.audit_radial_endpoint(
            fake_roots, destination,
            source_manifest=registered["source_manifest"],
            source_manifest_sha256=registered["source_manifest_sha256"],
        )
    assert not destination.exists()


def test_cli_rejects_missing_registration_without_writing_formal_output(tmp_path) -> None:
    script = PROJECT_ROOT / "scripts/30_audit_task07_radial_endpoint.py"
    output = tmp_path / "不准空登记却生成报告"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        ["/home/phl/anaconda3/envs/PINN/bin/python", str(script),
         "--audit-output", str(output), "--source-manifest", str(tmp_path / "notfound.json"),
         "--source-manifest-sha256", "0" * 64,
         *(item for seed in range(5) for item in (f"--seed{seed}", str(tmp_path / f"seed{seed}")))],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0 and not output.exists()


def test_report_outputs_four_independent_45_row_tables_and_90_row_cross_sha(tmp_path) -> None:
    entry = _entry()
    source = entry.register_source_snapshot(tmp_path / "源码真登记")
    references = entry._locked_reference_metrics()
    original = {arm: [row | {"arm": arm} for row in rows] for arm, rows in references.items()}
    corrected = {arm: [row.copy() for row in rows] for arm, rows in original.items()}
    for arm in corrected:
        for row in corrected[arm]:
            if row["radial_window"] == "outer_17_25_mm":
                row["sample_count"] += entry.POWER_COUNTS[row["power_w"]][1]
    identities = [{"种子": seed, "V4_B0配对LF检查点SHA256": "1" * 64,
                   "V4_B0观测最佳模型SHA256": "2" * 64,
                   "任07修复版观测最佳模型SHA256": "3" * 64,
                   "任07独立合法macro_v1选分_摄氏度": 2.0,
                   "任07最佳模型阶段原件SHA256": "4" * 64,
                   "任07真实末阶段原件SHA256": "5" * 64}
                  for seed in range(5)]
    destination = tmp_path / "诊断夹具只验格式非正式GPU"
    produced = entry._write_report(
        destination, source_manifest=source["source_manifest"],
        source_manifest_sha256=source["source_manifest_sha256"],
        snapshot=entry.verify_source_snapshot(source["source_manifest"],
                                              source["source_manifest_sha256"]),
        originals=original, corrected=corrected, lineage=identities,
        worst_legacy_delta=0.0, formal=False,
    )
    assert produced["报告资格"] == "CPU格式诊断；不可充当正式十模型GPU径向审计"
    manifest = json.loads((destination / "工件SHA256.json").read_text(encoding="utf-8"))
    assert set(manifest) == {
        "V4_B0旧表示原指标45.csv", "V4_B0闭端点修订指标45.csv",
        "任07_E0旧表示原指标45.csv", "任07_E0闭端点修订指标45.csv",
        "十模型原修径向逐窗90行对照.csv", "十模型端点修订摘要.json",
        "版本化顶部径向端点中文审计报告.md",
    }
    assert all(sha256_file(destination / name) == digest for name, digest in manifest.items())
    assert pl.read_csv(destination / "十模型原修径向逐窗90行对照.csv").height == 90
    assert all(pl.read_csv(destination / name).height == 45 for name in manifest if name.endswith("45.csv"))
    report = (destination / "版本化顶部径向端点中文审计报告.md").read_text(encoding="utf-8")
    assert "旧test_Data温度标签读取" in report and "25毫米" in report


def test_report_refuses_shrunk_44_row_seed_without_creating_files(tmp_path) -> None:
    entry = _entry()
    old = entry._locked_reference_metrics()
    original = {arm: [row | {"arm": arm} for row in rows] for arm, rows in old.items()}
    original["B0"].pop()
    destination = tmp_path / "缩水绝不可出"
    with pytest.raises(ValueError, match="45|十模型|缩水"):
        entry._write_report(
            destination, source_manifest=tmp_path / "任意登记.json",
            source_manifest_sha256="0" * 64, snapshot={}, originals=original,
            corrected=old, lineage=[], worst_legacy_delta=0.0, formal=False,
        )
    assert not destination.exists()


def test_official_inference_manifest_and_five_best_view_sha_are_crossed_without_gpu() -> None:
    entry = _entry()
    locked = entry._locked_formal_best_identity()
    assert set(locked) == set(range(5))
    assert locked[0]["best.pt"] == "42aab603826e627f69181fb6c76253a98b68241ec204695a9c10584626cc6ea7"
    assert locked[4]["best.pt"] == "50185092de09b58a7398f1547a9ee4ed44d82ec448f71c17ba80e863478f2a14"
    assert all(value["best.pt"] != value["阶段_训练末.pt"] for value in locked.values())


def test_non_project_or_nested_seed_output_rejected_before_cuda(tmp_path) -> None:
    entry = _entry()
    source = entry.register_source_snapshot(tmp_path / "真登记")
    roots = {
        seed: next((PROJECT_ROOT / "研究记录/任务07_正式五种子重训").glob(
            f"修复后正式_E0_种子{seed}_*")) for seed in range(5)
    }
    outside = PROJECT_ROOT.parent / f"不得将正式CUDA报告写出项目_{tmp_path.name}"
    with pytest.raises(ValueError, match="项目|外部|新目录"):
        entry.audit_radial_endpoint(
            roots, outside, source_manifest=source["source_manifest"],
            source_manifest_sha256=source["source_manifest_sha256"],
        )
    assert not outside.exists()
