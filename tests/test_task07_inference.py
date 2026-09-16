"""Task-07 independent inference qualification and deployment regressions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _entry():
    script = Path(__file__).resolve().parents[1] / "scripts/27_verify_task07_inference.py"
    spec = importlib.util.spec_from_file_location("task07_inference_regression", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_incomplete_five_seed_preflight_does_not_open_sources_or_create_output(
    tmp_path, monkeypatch,
) -> None:
    entry = _entry()
    monkeypatch.setattr(entry, "validate_task07_sources", lambda: pytest.fail(
        "缺seed不得读取历史源模型或任何温度观测",
    ))
    output = tmp_path / "五seed未齐拒绝"
    with pytest.raises(ValueError, match="五种子|0.*4|缺失"):
        entry.verify_task07_inference({0: tmp_path / "只报seed0"}, output)
    assert not output.exists()


def test_formal_output_must_not_be_nested_inside_any_frozen_seed_directory(
    tmp_path, monkeypatch,
) -> None:
    entry = _entry()
    roots = {seed: tmp_path / f"seed{seed}" for seed in range(5)}
    roots[0].mkdir()
    sentinel = roots[0] / "训练原件.txt"
    sentinel.write_bytes(b"frozen and untouched")
    monkeypatch.setattr(entry, "validate_task07_sources", lambda: pytest.fail(
        "输出嵌在训练源内部时不可读取配对checkpoint",
    ))
    output = roots[0] / "新独立推理结果"
    with pytest.raises(ValueError, match="输出|训练|目录|嵌套"):
        entry.verify_task07_inference(roots, output)
    assert not output.exists()
    assert sentinel.read_bytes() == b"frozen and untouched"


def test_diagnostic_view_cannot_be_certified_formal_before_any_output(
    tmp_path, monkeypatch,
) -> None:
    entry = _entry()
    roots = {seed: tmp_path / f"seed{seed}" for seed in range(5)}
    for root in roots.values():
        root.mkdir()
        (root / "best.pt").write_bytes(b"not a formal checkpoint")
        (root / "阶段报告.json").write_text(json.dumps({"运行资格": "短诊断；不可作正式验收"}),
                                         encoding="utf-8")
        for name in ("阶段_校正末.pt", "阶段_联合末.pt", "阶段_训练末.pt",
                     "阶段_最近.pt", "阶段_观测最佳.pt", "training.jsonl"):
            (root / name).write_bytes(b"placeholder")
    monkeypatch.setattr(entry, "validate_task07_sources", lambda: pytest.fail(
        "诊断须在温度数据或模型读入前拒绝",
    ))
    output = tmp_path / "诊断非法正式汇总"
    with pytest.raises(ValueError, match="诊断|正式|资格"):
        entry.verify_task07_inference(roots, output)
    assert not output.exists()


def test_view_stage_rejects_future_epoch_score_and_model_drift() -> None:
    entry = _entry()
    original = {
        "epoch": 3, "validation_selection_score_c": 2.0,
        "validation_rmse_c": 4.0, "validation_sensor": {"absolute_rmse_c": 1.0},
        "correction_model_kwargs": {"correction_width": 128},
        "model_state": {"correction.0.weight": torch.ones(128, 6)},
        "任07新HF模型视图来源": {
            "训练种子": 1, "运行臂": "E0", "本轮全局实际轮次": 3,
            "运行资格": "五种子正式E0候选",
        },
    }
    meta = {"全局累计实际轮次": 5, "观测最佳全局轮次": 3,
            "观测最佳选分_摄氏度": 2.0}
    best = {"metadata": {"全局累计实际轮次": 3},
            "model_state": {"correction.0.weight": torch.ones(128, 6)}}
    entry.check_best_view(original, best, meta, 1, "五种子正式E0候选",
                          {"顶部": 4.0, "absolute_rmse_c": 1.0})
    for field, bad in (("epoch", 6), ("validation_selection_score_c", 3.0),
                       ("correction_model_kwargs", {"correction_width": 64})):
        changed = dict(original, **{field: bad})
        with pytest.raises(ValueError, match="最佳|视图|模型|评分|轮次"):
            entry.check_best_view(changed, best, meta, 1, "五种子正式E0候选",
                                  {"顶部": 4.0, "absolute_rmse_c": 1.0},
                                  correction_kwargs={"correction_width": 128})
    changed = dict(original, model_state={"correction.0.weight": torch.zeros(128, 6)})
    with pytest.raises(ValueError, match="最佳|模型|张量"):
        entry.check_best_view(changed, best, meta, 1, "五种子正式E0候选",
                              {"顶部": 4.0, "absolute_rmse_c": 1.0})


def test_committed_terminal_rejects_adamw_rng_and_freeze_drift() -> None:
    entry = _entry()
    latest = {
        "epoch": 12, "stage": "task07_restricted_joint",
        "model_state": {"correction.0.weight": torch.ones(128, 6)},
        "optimizer_state": {"state": {0: {"step": torch.tensor(192)}}},
        "random_state": {"python": (1,), "numpy": np.array([2]),
                         "torch_cpu": torch.ones(4, dtype=torch.uint8),
                         "torch_cuda": [torch.ones(4, dtype=torch.uint8)]},
        "parameter_requires_grad": {"correction.0.weight": True},
        "metadata": {"全局累计实际轮次": 161},
    }
    terminal = dict(latest)
    entry.check_committed_terminal(latest, terminal)
    for field, bad in (
        ("optimizer_state", {"state": {0: {"step": torch.tensor(160)}}}),
        ("random_state", {**latest["random_state"], "torch_cpu": torch.zeros(4,
                                                        dtype=torch.uint8)}),
        ("parameter_requires_grad", {"correction.0.weight": False}),
    ):
        with pytest.raises(ValueError, match="完整|末|AdamW|随机|冻结"):
            entry.check_committed_terminal(latest, dict(terminal, **{field: bad}))


def test_observation_and_physical_best_must_be_original_committed_stages() -> None:
    entry = _entry()
    metadata = {
        "全局累计实际轮次": 100, "观测最佳全局轮次": 20,
        "观测最佳阶段": "task07_correction", "观测最佳阶段轮次": 20,
        "观测最佳选分_摄氏度": 2.0,
        "物理最佳全局轮次": 99, "物理最佳阶段": "task07_restricted_joint",
        "物理最佳阶段轮次": 9, "物理最佳独立损失": 0.75,
    }
    observed = {"stage": metadata["观测最佳阶段"], "epoch": 20,
                "metadata": {"全局累计实际轮次": 20,
                             "观测最佳选分_摄氏度": 2.0,
                             "观测最佳全局轮次": 20}}
    physical = {"stage": metadata["物理最佳阶段"], "epoch": 9,
                "metadata": {"全局累计实际轮次": 99,
                             "物理最佳独立损失": 0.75,
                             "物理最佳全局轮次": 99}}
    entry.check_committed_best_stage(metadata, observed, physical)
    for changed in (
        dict(observed, epoch=21),
        dict(physical, metadata={**physical["metadata"], "物理最佳独立损失": 3.0}),
        dict(physical, metadata={**physical["metadata"], "全局累计实际轮次": 101}),
    ):
        with pytest.raises(ValueError, match="最佳|物理|轮次|阶段"):
            entry.check_committed_best_stage(metadata,
                                             changed if changed.get("stage") == observed["stage"]
                                             else observed,
                                             changed if changed.get("stage") == physical["stage"]
                                             else physical)


def test_failed_final_joint_lf_can_only_replay_proven_frozen_correction_best() -> None:
    entry = _entry()
    source = SimpleNamespace(lf_tensor_sha256="paired-lf-sha",
                             lf_state={"branch_projection.weight": torch.ones(3, 3)})
    report = {"LF逐材料节点和真实体积5%护栏": {
        "LF两材料节点与真实体积均守住5%护栏": False,
    }}
    best = {
        "stage": "task07_correction",
        "metadata": {"当前真实LF张量SHA256": "paired-lf-sha"},
        "model_state": {"low_fidelity_model.branch_projection.weight": torch.ones(3, 3)},
    }
    status = entry.assess_selected_best_lf_guard(report, best, source)
    assert status["联合末可采用"] is False
    assert status["观察最佳是合法冻结LF校正阶段"] is True
    assert "联合末不可采用" in status["口径"]
    for poisoned in (
        dict(best, stage="task07_restricted_joint"),
        dict(best, metadata={"当前真实LF张量SHA256": "forged-source"}),
        dict(best, model_state={"low_fidelity_model.branch_projection.weight": torch.zeros(3, 3)}),
    ):
        with pytest.raises(ValueError, match="LF|护栏|联合|来源|校正"):
            entry.assess_selected_best_lf_guard(report, poisoned, source)


def test_failed_joint_lf_never_claims_deployed_joint_model_adoption() -> None:
    entry = _entry()
    report = {"LF逐材料节点和真实体积5%护栏": {
        "LF两材料节点与真实体积均守住5%护栏": False,
    }}
    source = SimpleNamespace(lf_tensor_sha256="original",
                             lf_state={"some.weight": torch.ones(1)})
    joint_best = {"stage": "task07_restricted_joint",
                  "metadata": {"当前真实LF张量SHA256": "original"},
                  "model_state": {"low_fidelity_model.some.weight": torch.ones(1)}}
    with pytest.raises(ValueError, match="联合|LF|护栏"):
        entry.assess_selected_best_lf_guard(report, joint_best, source)


def test_batch_regression_exceeds_batch_capacity_even_for_default_size() -> None:
    entry = _entry()
    probes = np.ones((5, 5), dtype=np.float32)
    repeated = entry.build_cross_batch_probes(probes, batch_size=1024)
    assert len(repeated) > 1024
    assert np.array_equal(repeated[:len(probes)], repeated[len(probes):2 * len(probes)])


def test_float32_cross_batch_accepts_only_one_ulp_at_operating_temperature() -> None:
    entry = _entry()
    value = np.float32(342.02496)
    one_ulp = np.nextafter(value, np.float32(np.inf))
    two_ulps = np.nextafter(one_ulp, np.float32(np.inf))
    query = np.full((1035, 1), value, dtype=np.float32)
    query[1029, 0] = one_ulp
    direct = np.full((5, 1), value, dtype=np.float32)
    one = entry.inspect_cross_batch_consistency(query, direct, probe_count=5,
                                                batch_size=1024)
    assert one["跨批浮点容差通过"]
    assert one["同点跨真实batch最大温差K"] == float(one_ulp - value)
    assert one["允许最大float32单ULP_K"] == float(one_ulp - value)
    query[1029, 0] = two_ulps
    two = entry.inspect_cross_batch_consistency(query, direct, probe_count=5,
                                                batch_size=1024)
    assert not two["跨批浮点容差通过"]


def test_query_repeat_full_time_export_zero_anchor_and_warnings_cpu_diagnostic(
    tmp_path, monkeypatch,
) -> None:
    entry = _entry()
    from sic_cu.data.fields import SimulationField
    from sic_cu.train.task07_source import validate_task07_sources

    source = validate_task07_sources()[0]
    original = torch.load(source.hf_checkpoint_path, map_location="cpu", weights_only=False)
    view = tmp_path / "只作诊断的历史模型视图.pt"
    torch.save(original, view)
    nodes = np.array([[0.0, -0.0175], [0.03, -0.0175], [0.05, -0.0175],
                      [0.0, -0.012], [0.025, -0.012], [0.018, 0.0]], dtype=np.float32)
    reference = SimulationField(
        power_w=10.0, times_s=np.array([0., 50., 100., 150., 200.], dtype=np.float32),
        coordinates_rz_m=nodes, material_ids=np.array([0, 0, 0, 1, 1, 1]),
        node_labels=np.arange(len(nodes)),
        temperature_k=np.full((5, len(nodes)), 295.15, dtype=np.float32),
    )
    monkeypatch.setattr("sic_cu.prediction.load_processed_field", lambda _: reference)
    monkeypatch.setattr(entry, "load_processed_field", lambda _: reference)
    output = tmp_path / "CPU诊断文件"
    result = entry.evaluate_diagnostic_view(view, output, device_name="cpu",
                                            theta_resolution=3, batch_size=2)
    assert "不可作正式" in result["资格"]
    assert result["batch重复查询"]["逐元素完全相同"]
    assert result["batch重复查询"]["批次大小"] == 2
    assert result["batch重复查询"]["实际批次数"] >= 2
    assert result["batch重复查询"]["跨批浮点容差通过"]
    assert result["零功率锚点"]["初始场全部为295.15K"]
    assert result["越界警示"]["低于HF训练范围有警示"]
    assert result["越界警示"]["超出HF训练范围有警示"]
    assert result["完整时序场"]["逐时场数"] == len(reference.times_s)
    assert result["初温t0模型诊断"]["Cu和SiC逐点初温最大偏差K"] <= 1e-4
    assert result["初温t0模型诊断"]["整个RZ场t0初温最大偏差K"] <= 1e-4
    assert "全时水冷外径面最大偏差K" in result["水冷Cu外径面模型诊断"]
    assert result["完整时序场"]["模型纯前向同步计时秒"] >= 0.0
    assert result["完整时序场"]["读取LF网格计时秒"] >= 0.0
    assert result["完整时序场"]["过程场文件导出计时秒"] >= 0.0
    for relative, expected_sha in result["工件SHA256"].items():
        from sic_cu.data.common import sha256_file
        assert sha256_file(output / relative) == expected_sha
    with np.load(output / "正式功率场" / "field_rzt.npz") as field:
        assert field["mean_temperature_c"].shape == (len(reference.times_s), len(nodes))
    assert result["GPU同步真实计时"]["状态"] == "CPU诊断，未作正式GPU计时"
