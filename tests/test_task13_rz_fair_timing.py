"""Synthetic-only safety contract for the Task-13 fixed-RZ timing entry."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path
import yaml

import numpy as np
import polars as pl
import pytest
import torch

from sic_cu.eval.task13_rz_fair_timing import (
    build_fixed_queries,
    check_checkpoint_payload,
    check_checkpoint_receipt_consistency,
    check_lf_receipt,
    check_registered_origin,
    check_registered_candidate,
    check_source_tar_members,
    HighOnlyRZModel,
    check_training_receipt,
    export_identical_rz_format,
    load_registered_grid,
    measure_forward_parts,
    preflight_timing,
    benchmark_registered_models,
    registered_candidate_rows,
    registered_lf_rows,
    validate_registered_timing_protocol,
    run_model_forward,
    validate_benchmark_power,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_grid(path: Path) -> None:
    times = np.arange(101, dtype=np.float32) * 2.0
    counts = (821, 338)
    materials = np.repeat([0, 1], counts)
    labels = np.concatenate([np.arange(counts[0]), np.arange(counts[1])])
    radii = np.concatenate([
        np.linspace(0, 0.05834, counts[0], dtype=np.float32),
        np.linspace(0, 0.025, counts[1], dtype=np.float32),
    ])
    z = np.concatenate([
        np.linspace(-0.0175, 0, counts[0], dtype=np.float32),
        np.linspace(-0.012, 0, counts[1], dtype=np.float32),
    ])
    frame = pl.DataFrame({
        "time_s": np.repeat(times, 1159), "material_id": np.tile(materials, 101),
        "node_label": np.tile(labels, 101), "r_m": np.tile(radii, 101),
        "z_m": np.tile(z, 101), "temperature_k": np.full(117059, -9999.0),
    })
    frame.write_parquet(path)


def test_rejects_test_and_unregistered_powers_without_reading_labels() -> None:
    assert validate_benchmark_power(309.0) == 309.0
    for forbidden in (169.0, 339.0, 634.0, 0.0, 36.0, 403.0, 800.0, 30000.0):
        with pytest.raises(ValueError, match="309|测试|训练|功率"):
            validate_benchmark_power(forbidden)


def test_grid_reads_five_columns_only_and_keeps_registered_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "仅LF10W几何五列.parquet"
    _synthetic_grid(source)
    original = pl.read_parquet

    def guarded_read(path, *args, **kwargs):
        assert Path(path) == source
        assert set(kwargs.get("columns", [])) == {
            "time_s", "r_m", "z_m", "material_id", "node_label",
        }
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pl, "read_parquet", guarded_read)
    grid = load_registered_grid(source, _sha(source))
    query = build_fixed_queries(grid, 309.0)
    assert query.shape == (101 * 1159, 5)
    assert set(np.unique(query[:, 4])) == {0.0, 1.0}
    assert set(np.unique(query[:, 3])) == {309.0}
    assert np.array_equal(np.unique(query[:, 2]), np.arange(101) * 2)
    assert not np.any(query == -9999.0)


def test_grid_rejects_out_of_domain_and_changed_time_before_any_model(
    tmp_path: Path,
) -> None:
    source = tmp_path / "限定RZ.parquet"
    _synthetic_grid(source)
    table = pl.read_parquet(source)
    table.with_columns(pl.when(pl.col("material_id") == 1)
                       .then(0.1).otherwise(pl.col("r_m")).alias("r_m")).write_parquet(source)
    with pytest.raises(ValueError, match="域|半径|SiC|网格"):
        load_registered_grid(source, _sha(source))
    _synthetic_grid(source)
    table = pl.read_parquet(source)
    table.with_columns(pl.when(pl.col("time_s") == 200.0)
                       .then(202.0).otherwise(pl.col("time_s")).alias("time_s")
                       ).write_parquet(source)
    with pytest.raises(ValueError, match="时|200|网格"):
        load_registered_grid(source, _sha(source))


def test_model_forward_is_checkpoint_model_only_and_shape_finite() -> None:
    query = np.array([[0.0, -0.0175, 0.0, 309.0, 0.0],
                      [0.025, 0.0, 200.0, 309.0, 1.0]], dtype=np.float32)
    model = torch.nn.Linear(5, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(295.15)
    result = run_model_forward(model, query, torch.device("cpu"), batch_size=1)
    assert result.shape == (2,)
    assert np.allclose(result, 295.15, atol=1e-5)
    with pytest.raises(ValueError, match="模型|FEM|回退"):
        run_model_forward(None, query, torch.device("cpu"), batch_size=1)
    with pytest.raises(ValueError, match="域|半径|功率|时刻"):
        run_model_forward(model, np.array([[0.2, -0.01, 0, 309, 1]], dtype=np.float32),
                          torch.device("cpu"), batch_size=1)


def test_formal_requires_sha_group_and_ledger_before_opening_grid_or_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "合成事前.yaml"
    registry.write_text("schema_version: 1\n", encoding="utf-8")
    tar = tmp_path / "合成源码.tar.gz"
    tar.write_bytes(b"synthetic package")
    ledger = tmp_path / "台账.md"
    ledger.write_text("无前置登记\n", encoding="utf-8")
    output = tmp_path / "结果"
    with pytest.raises(ValueError, match="事前|SHA|台账|登记"):
        preflight_timing(registry, _sha(registry), tar, _sha(tar), ledger, output,
                         require_cuda=True)
    assert not output.exists()


def test_candidate_rejects_wrong_pde_plate_fem_and_missing_source_receipt(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "模型.pt"
    checkpoint.write_bytes(b"fake model bytes")
    receipt = tmp_path / "训练成本.json"
    receipt.write_text(json.dumps({"training_seconds": 10.0, "epochs_completed": 1700}),
                       encoding="utf-8")
    registered = {"arm": "B0", "seed": 0, "checkpoint": str(checkpoint),
                  "checkpoint_sha256": _sha(checkpoint), "receipt": str(receipt),
                  "receipt_sha256": _sha(receipt)}
    check_registered_candidate(registered, checkpoint, receipt)
    with pytest.raises(ValueError, match="来源|SHA|训练成本"):
        check_registered_candidate({**registered, "receipt_sha256": "0" * 64},
                                   checkpoint, receipt)
    with pytest.raises(ValueError, match="来源|RZ|板|FEM|方法"):
        check_registered_candidate({**registered, "arm": "任务10一维板"},
                                   checkpoint, receipt)


@pytest.mark.parametrize("arm,lf", [("B0", "deeponet_pinn"), ("MLP", "mlp_pinn"),
                                    ("E0", "deeponet_pinn"), ("F2", "deeponet_pinn")])
def test_checkpoint_identity_must_match_arm_seed_and_only_legal_powers(
    arm: str, lf: str,
) -> None:
    from sic_cu.data.splits import build_power_splits

    split = build_power_splits()
    train, val = sorted(split.hf_train), sorted(split.hf_validation)
    payload = {
        "method": "multifidelity_correction", "low_fidelity_method": lf,
        "seed": 0, "hf_train_powers_w": train, "hf_validation_powers_w": val,
        "hf_test_powers_w": sorted(split.hf_test),
        "lf_checkpoint_sha256": "b" * 64,
        "provenance": {"test_labels_consumed": False},
    }
    if arm in ("E0", "F2"):
        key = "任07新HF模型视图来源" if arm == "E0" else "任08F2模型视图来源"
        payload[key] = {"运行臂": arm, "训练种子": 0,
                        "本seed真实LF检查点SHA256": "b" * 64,
                        "旧test_Data温度标签读取": False,
                        "正式预登记配置SHA256": "a" * 64}
    row = {"arm": arm, "seed": 0, "正式预登记配置SHA256": "a" * 64,
           "lf_checkpoint_sha256": "b" * 64}
    check_checkpoint_payload(row, payload)
    with pytest.raises(ValueError, match="来源|模型|身份|seed|方法"):
        check_checkpoint_payload(row, {**payload, "low_fidelity_method": "task10_plate"})
    with pytest.raises(ValueError, match="来源|seed|身份"):
        check_checkpoint_payload(row, {**payload, "seed": 1})
    with pytest.raises(ValueError, match="训练|验证|测试|功率"):
        check_checkpoint_payload(row, {**payload, "hf_train_powers_w": train + [339.0]})
    if arm in ("E0", "F2"):
        with pytest.raises(ValueError, match="来源|身份|F2|E0|视图"):
            check_checkpoint_payload(row, {**payload, key: {**payload[key], "运行臂": "E0" if arm == "F2" else "F2"}})
    with pytest.raises(ValueError, match="LF|来源|身份"):
        check_checkpoint_payload(row, {**payload, "lf_checkpoint_sha256": "c" * 64})


@pytest.mark.parametrize("lf_method", ["deeponet_pinn", "mlp_pinn"])
def test_real_lf_pretraining_cost_is_separate_from_hf_session_and_paired(
    lf_method: str,
) -> None:
    lf_source = {"seed": 0, "method": lf_method,
                 "epochs_completed": 378, "training_seconds": 358.346,
                 "material_passport": {"experiment_data_used": False}}
    result = check_lf_receipt({"seed": 0, "lf_method": lf_method}, lf_source)
    assert result["可复用LF预训耗时秒"] == pytest.approx(358.346)
    assert result["LF预训实际轮次"] == 378
    with pytest.raises(ValueError, match="LF|来源|训练"):
        check_lf_receipt({"seed": 0, "lf_method": lf_method},
                         {**lf_source, "method": "unrelated plate model"})
    with pytest.raises(ValueError, match="LF|来源|实验"):
        check_lf_receipt({"seed": 0, "lf_method": lf_method},
                         {**lf_source, "material_passport": {"experiment_data_used": True}})


@pytest.mark.parametrize("arm", ["B0", "MLP", "E0", "F2"])
def test_checkpoint_best_identity_must_equal_locked_cost_receipt(arm: str) -> None:
    payload = {"seed": 0, "epoch": 100,
               "validation_selection_score_c": 2.0,
               "validation_selection_score_k": 2.0}
    receipt = ({"seed": 0, "best_epoch": 100,
                "best_validation_selection_score_k": 2.0}
               if arm in ("B0", "MLP") else
               {"运行种子": 0, "运行臂": arm, "观测最佳全局轮次": 100,
                "观测最佳HF合法选分_摄氏度": 2.0})
    check_checkpoint_receipt_consistency({"arm": arm, "seed": 0}, payload, receipt)
    with pytest.raises(ValueError, match="最佳|轮次|来源"):
        check_checkpoint_receipt_consistency({"arm": arm, "seed": 0},
                                             {**payload, "epoch": 101}, receipt)
    with pytest.raises(ValueError, match="最佳|验证|来源"):
        check_checkpoint_receipt_consistency({"arm": arm, "seed": 0},
                                             {**payload, "validation_selection_score_c": 3.0,
                                              "validation_selection_score_k": 3.0}, receipt)


@pytest.mark.parametrize("arm", ["B0", "MLP", "E0", "F2"])
def test_read_cost_receipt_distinguishes_full_historical_and_new_last_session(arm: str) -> None:
    old = arm in ("B0", "MLP")
    receipt = ({"seed": 0, "training_seconds": 2445.0, "epochs_completed": 1700}
               if old else {"运行种子": 0, "运行臂": arm,
                            "本会话耗时秒": 71.0, "校正实际轮次": 500,
                            "联合实际轮次": 200, "真实最后状态": "阶段_训练末.pt",
                            "旧test_Data温度标签读取": False})
    cost = check_training_receipt({"arm": arm, "seed": 0}, receipt)
    assert cost["耗时秒"] > 0
    assert cost["口径"] == ("历史单次完整训练会话含评价" if old
                           else "仅当前真实会话，非全部断点累计")
    with pytest.raises(ValueError, match="成本|训练|来源|会话"):
        bad = {**receipt, ("training_seconds" if old else "本会话耗时秒"): -1.0}
        check_training_receipt({"arm": arm, "seed": 0}, bad)


def test_cpu_synthetic_forward_parts_record_separate_model_transfer_and_readback() -> None:
    model = torch.nn.Linear(5, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(295.15)
    query = np.array([[0.0, -0.0175, 0.0, 309.0, 0.0],
                      [0.025, 0.0, 200.0, 309.0, 1.0]], dtype=np.float32)
    measured = measure_forward_parts(model, query, torch.device("cpu"),
                                     batch_size=1, warmups=1, repeats=2)
    assert len(measured["纯模型同步前向秒"]) == 2
    assert all(x >= 0 and np.isfinite(x) for x in measured["纯模型同步前向秒"])
    assert measured["查询设备传输秒"] >= 0
    assert measured["输出回读秒"] >= 0
    assert np.allclose(measured["temperature_k"], 295.15, atol=1e-5)


def test_export_same_file_schema_for_synthetic_cpu_field(tmp_path: Path) -> None:
    grid_path = tmp_path / "合成网格.parquet"
    _synthetic_grid(grid_path)
    grid = load_registered_grid(grid_path, _sha(grid_path))
    field = 295.15 + np.linspace(0.0, 5.0, 101 * 1159,
                                 dtype=np.float32).reshape(101, 1159)
    output = tmp_path / "仅合成CPU格式"
    receipt = export_identical_rz_format(grid, field, output, checkpoint_sha="a" * 64)
    assert (output / "field_rzt.npz").is_file()
    assert (output / "metadata.json").is_file()
    assert (output / "hot_cold.csv").is_file()
    assert len(receipt["vtk"]) == 5
    assert len(receipt["figures"]) == 4
    with np.load(output / "field_rzt.npz", allow_pickle=False) as field_archive:
        assert field_archive["mean_temperature_c"].shape == (101, 1159)
        assert field_archive["times_s"].shape == (101,)
    with pytest.raises(FileExistsError, match="覆盖|已有|原件"):
        export_identical_rz_format(grid, field, output, checkpoint_sha="a" * 64)


def test_origin_of_registered_models_is_fixed_RZ_not_arbitrary_same_name(
    tmp_path: Path,
) -> None:
    fake_model = tmp_path / "deeponet_pinn_mf_seed0/best.pt"
    fake_model.parent.mkdir()
    fake_model.write_bytes(b"forged RZ source")
    fake_receipt = fake_model.parent / "metrics.json"
    fake_receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="旧发布|目录|原件|来源"):
        check_registered_origin({"arm": "B0", "seed": 0,
                                 "checkpoint": str(fake_model),
                                 "receipt": str(fake_receipt)})
    with pytest.raises(ValueError, match="旧发布|目录|原件|来源"):
        check_registered_origin({"arm": "MLP", "seed": 0,
                                 "checkpoint": str(fake_model),
                                 "receipt": str(fake_receipt)})
    with pytest.raises(ValueError, match="目录|原件|来源"):
        check_registered_origin({"arm": "E0", "seed": 0,
                                 "checkpoint": str(fake_model),
                                 "receipt": str(fake_receipt)})
    with pytest.raises(ValueError, match="目录|原件|来源"):
        check_registered_origin({"arm": "F2", "seed": 0,
                                 "checkpoint": str(fake_model),
                                 "receipt": str(fake_receipt)})


def test_tar_rejects_missing_member_or_registry_after_budget_rehash(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "合成预算.yaml"
    registry.write_text("schema_version: 1\n", encoding="utf-8")
    archive = tmp_path / "缺源码档案.tar.gz"
    with tarfile.open(archive, "w:gz") as contents:
        contents.add(registry, arcname=str(registry.relative_to(Path(__file__).resolve().parents[1])))
    with pytest.raises(ValueError, match="源码|成员|快照"):
        check_source_tar_members(archive, registry)


def test_sha_index_reconstructs_exactly_twenty_fixed_sources_and_ten_shared_LF() -> None:
    sha = lambda seed, offset: format(seed + offset, "064x")
    methods = ("B0", "MLP", "E0", "F2")
    cfg = {
        "模型": {arm: [{"checkpoint_sha256": sha(seed, i * 5),
                       "receipt_sha256": sha(seed, i * 5 + 100)}
                      for seed in range(5)] for i, arm in enumerate(methods)},
        "LF来源": {method: [{"checkpoint_sha256": sha(seed, 200 + i * 5),
                           "receipt_sha256": sha(seed, 300 + i * 5)}
                          for seed in range(5)]
                 for i, method in enumerate(("deeponet_pinn", "mlp_pinn"))},
        "正式E0预登记配置SHA256": "a" * 64,
        "正式F2预登记配置SHA256": "b" * 64,
    }
    lf_rows = registered_lf_rows(cfg)
    rows = registered_candidate_rows(cfg, lf_rows)
    assert len(rows) == 20 and len(lf_rows) == 10
    assert {(row["arm"], row["seed"]) for row in rows} == {
        (arm, seed) for arm in methods for seed in range(5)}
    assert rows[0]["lf_checkpoint_sha256"] == rows[10]["lf_checkpoint_sha256"]
    assert rows[0]["lf_checkpoint_sha256"] != rows[5]["lf_checkpoint_sha256"]
    assert rows[10]["正式预登记配置SHA256"] == "a" * 64
    assert rows[15]["正式预登记配置SHA256"] == "b" * 64
    with pytest.raises(ValueError, match="五seed|20|LF|来源"):
        registered_lf_rows({**cfg, "LF来源": {"deeponet_pinn": cfg["LF来源"]["deeponet_pinn"]}})
    with pytest.raises(ValueError, match="五seed|20|模型|来源"):
        registered_candidate_rows({**cfg, "模型": {"B0": cfg["模型"]["B0"]}}, lf_rows)


def test_high_fidelity_wrapper_never_times_frozen_lf_instead_of_hf() -> None:
    class TwoFidelity(torch.nn.Module):
        def forward(self, values, fidelity="low"):
            return torch.full((len(values), 1), 295.15 if fidelity == "high" else 999.0)

    query = np.array([[0.0, -0.0175, 0, 309, 0],
                      [0.025, -0.012, 200, 309, 1]], dtype=np.float32)
    result = measure_forward_parts(HighOnlyRZModel(TwoFidelity()), query,
                                   torch.device("cpu"), batch_size=1,
                                   warmups=1, repeats=2)
    assert np.allclose(result["temperature_k"], 295.15, atol=1e-5)


def test_formal_benchmark_rejects_cpu_before_any_source_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sic_cu.eval import task13_rz_fair_timing as module

    monkeypatch.setattr(module, "load_registered_grid", lambda *_: pytest.fail(
        "CPU正式计时不得打开网格或对应HF实验",
    ))
    out = tmp_path / "不得CPU计时"
    with pytest.raises(ValueError, match="CUDA|CPU|正式"):
        benchmark_registered_models({"登记模型来源": [], "登记LF来源": []}, out,
                                     device=torch.device("cpu"))
    assert not out.exists()


def test_formal_cli_source_never_calls_prediction_fem_or_HF_temperature_loader() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/40_run_task13_rz_fair_timing.py"
    spec = importlib.util.spec_from_file_location("task13_rz_only", script)
    assert spec is not None and spec.loader is not None
    source = script.read_text(encoding="utf-8")
    import ast
    tree = ast.parse(source)
    banned = {"load_processed_field", "_ir_dataset", "load_hf_temperature",
              "interpolate_simulation_power", "predict", "predict_points"}
    calls = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
             for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, (ast.Name, ast.Attribute))}
    assert calls.isdisjoint(banned)
    from_modules = {node.module for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)}
    assert "benchmark_registered_models" in source and "preflight_timing" in source
    assert "--registry-sha" in source and "--source-tar-sha" in source
    assert "sic_cu.data.fields" not in from_modules
    assert "sic_cu.models.interpolation" not in from_modules


def test_fixed_preentry_protocol_rejects_budget_and_export_format_changes() -> None:
    cfg = {"注册功率_W": 309.0, "起止时刻_s": [0.0, 200.0],
           "时间步_s": 2.0, "batch_size": 2048, "预热重复": 2,
           "有效重复": 5, "旋转theta分辨率": 12,
           "vtk时刻_s": [0.0, 50.0, 100.0, 150.0, 200.0],
           "时间帧": 101, "每帧节点": 1159,
           "读网格五列": ["time_s", "r_m", "z_m", "material_id", "node_label"]}
    validate_registered_timing_protocol(cfg)
    for key, value in (("注册功率_W", 339.0), ("起止时刻_s", [0, 202]),
                       ("时间步_s", 1.0), ("batch_size", 999),
                       ("预热重复", 0), ("有效重复", 3),
                       ("旋转theta分辨率", 72), ("vtk时刻_s", [0.0, 200.0])):
        with pytest.raises(ValueError, match="预算|时|功率|固定|导出"):
            validate_registered_timing_protocol({**cfg, key: value})


def test_synthetic_twenty_model_ten_lf_preflight_after_same_ledger_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sic_cu.eval import task13_rz_fair_timing as entry

    root = Path(__file__).resolve().parents[1]
    grid_path = tmp_path / "合成LF10W五列网格.parquet"
    _synthetic_grid(grid_path)
    monkeypatch.setattr(entry, "GRID_PATH", grid_path)
    monkeypatch.setattr(entry, "RUN_ROOT", tmp_path / "历史runs")
    monkeypatch.setattr(entry, "E0_ROOT", tmp_path / "E0合成")
    monkeypatch.setattr(entry, "F2_ROOT", tmp_path / "F2合成")
    legacy = {}
    for arm in ("B0", "MLP"):
        path = tmp_path / f"{arm}_合成发布.yaml"
        path.write_text("status: frozen\n", encoding="utf-8")
        legacy["旧B0发布清单SHA256" if arm == "B0" else "旧MLP发布清单SHA256"] = path
    monkeypatch.setattr(entry, "OLD_RELEASES", legacy)
    cfg = {
        "schema_version": 1, "实验对象": "RZ原装置历史与新同网格推理成本",
        "注册功率_W": 309.0, "时间帧": 101, "每帧节点": 1159,
        "起止时刻_s": [0.0, 200.0], "时间步_s": 2.0,
        "batch_size": 2048, "预热重复": 2, "有效重复": 5,
        "旋转theta分辨率": 12, "vtk时刻_s": [0.0, 50.0, 100.0, 150.0, 200.0],
        "读网格五列": list(entry.GRID_COLUMNS),
        "RZ网格原件": str(grid_path), "RZ网格SHA256": _sha(grid_path),
        "正式E0预登记配置SHA256": "a" * 64,
        "正式F2预登记配置SHA256": "b" * 64,
        "模型": {arm: [{} for _ in range(5)] for arm in entry.ARMS},
        "LF来源": {method: [{} for _ in range(5)]
                 for method in ("deeponet_pinn", "mlp_pinn")},
    }
    for field, path in legacy.items():
        cfg[field] = _sha(path)
    for method in cfg["LF来源"]:
        for seed, hashes in enumerate(cfg["LF来源"][method]):
            origin = entry.RUN_ROOT / f"{method}_lf_seed{seed}"
            origin.mkdir(parents=True)
            checkpoint = origin / "best.pt"
            checkpoint.write_bytes(f"synthetic-{method}-{seed}".encode("ascii"))
            receipt = origin / "metrics.json"
            receipt.write_text(json.dumps({"method": method, "seed": seed,
                                           "epochs_completed": 100,
                                           "training_seconds": 10.0,
                                           "material_passport": {"experiment_data_used": False}}),
                               encoding="utf-8")
            hashes.update(checkpoint_sha256=_sha(checkpoint), receipt_sha256=_sha(receipt))
    for arm in cfg["模型"]:
        for seed, hashes in enumerate(cfg["模型"][arm]):
            if arm == "B0":
                origin = entry.RUN_ROOT / f"deeponet_pinn_mf_seed{seed}"
            elif arm == "MLP":
                origin = entry.RUN_ROOT / f"mlp_pinn_mf_seed{seed}"
            elif arm == "E0":
                origin = entry.E0_ROOT / entry.E0_DIRS[seed]
            else:
                origin = entry.F2_ROOT / entry.F2_DIRS[seed]
            origin.mkdir(parents=True)
            checkpoint = origin / "best.pt"
            checkpoint.write_bytes(f"synthetic-{arm}-{seed}".encode("ascii"))
            receipt = origin / ("metrics.json" if arm in ("B0", "MLP")
                                else "阶段报告.json")
            data = ({"seed": seed, "epochs_completed": 1700, "training_seconds": 20.0}
                    if arm in ("B0", "MLP") else
                    {"运行种子": seed, "运行臂": arm, "本会话耗时秒": 20.0,
                     "校正实际轮次": 500, "联合实际轮次": 200,
                     "真实最后状态": "阶段_训练末.pt",
                     "旧test_Data温度标签读取": False})
            receipt.write_text(json.dumps(data), encoding="utf-8")
            hashes.update(checkpoint_sha256=_sha(checkpoint), receipt_sha256=_sha(receipt))
    budget = tmp_path / "全合成事前.yaml"
    budget.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    tar = tmp_path / "全合成12成员.tar.gz"
    with tarfile.open(tar, mode="w:gz") as archive:
        for name in entry.SOURCE_MEMBERS:
            archive.add(root / name, arcname=name)
        archive.add(budget, arcname=str(budget.relative_to(root)))
    ledger = tmp_path / "仅合成台账.md"
    ledger.write_text(f"| 录-9999 | 合成预算 {_sha(budget)} 及源码 {_sha(tar)} 早于诊断 |\n",
                      encoding="utf-8")
    result_dir = tmp_path / "不可提前输出正式文件"
    found = preflight_timing(budget, _sha(budget), tar, _sha(tar), ledger,
                             result_dir, require_cuda=False)
    assert len(found["登记模型来源"]) == 20
    assert len(found["登记LF来源"]) == 10
    assert not result_dir.exists()
    missing_lf = entry.RUN_ROOT / "deeponet_pinn_lf_seed3/metrics.json"
    missing_lf.write_bytes(b"tampered receipt after preentry")
    with pytest.raises(ValueError, match="LF|成本|SHA"):
        preflight_timing(budget, _sha(budget), tar, _sha(tar), ledger,
                         result_dir, require_cuda=False)
    assert not result_dir.exists()
