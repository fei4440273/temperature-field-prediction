from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tarfile

import pytest
import torch
import yaml

from sic_cu.config import PROJECT_ROOT


TASK13_REGISTRY = (
    "研究记录/任务13_发布与验收/任13_原装置四臂同RZ网格推理分项事前登记.yaml"
)
TASK13_REGISTRY_SHA = "b82b2c540ce12eeb4d8b8f8677c466211e1860fd301299c224770b8236602d42"
TASK13_SOURCE_TAR = (
    "研究记录/任务13_发布与验收/任13_四臂同RZ网格推理十二源冻结_20260916T054001+0800.tar.gz"
)
TASK13_SOURCE_SHA = "75725882f8f865a388e0da6cd0d423c5b6c8c1b1db68bb46bf7759190a0752b4"


def _synthetic_sources():
    from sic_cu.data.splits import build_power_splits

    split = build_power_splits()
    hf_powers = sorted(split.hf_train)
    hf_validation = sorted(split.hf_validation)
    lf_powers = sorted(split.simulation_train)
    lf_validation = sorted(split.simulation_validation)
    lf_rows = []
    lf_checkpoints = {}
    lf_receipts = {}
    for method in ("deeponet_pinn", "mlp_pinn"):
        for seed in range(5):
            sha = f"{seed + (1 if method == 'deeponet_pinn' else 7):064x}"
            lf_rows.append({"lf_method": method, "seed": seed,
                            "checkpoint_sha256": sha, "receipt_sha256": f"{seed + 21:064x}"})
            lf_checkpoints[method, seed] = {
                "method": method, "seed": seed,
                "train_powers_w": lf_powers, "validation_powers_w": lf_validation,
                "model_state": {"weight": torch.zeros(1)},
                "provenance": {"test_labels_consumed": False},
            }
            lf_receipts[method, seed] = {"method": method, "seed": seed,
                                         "epochs_completed": 300,
                                         "training_seconds": 10.0}
    hf_rows = []
    hf_checkpoints = {}
    hf_receipts = {}
    for arm in ("B0", "MLP", "E0", "F2"):
        for seed in range(5):
            method = "mlp_pinn" if arm == "MLP" else "deeponet_pinn"
            lf_sha = next(row["checkpoint_sha256"] for row in lf_rows
                          if row["lf_method"] == method and row["seed"] == seed)
            hf_rows.append({"arm": arm, "seed": seed, "lf_method": method,
                            "lf_checkpoint_sha256": lf_sha,
                            "checkpoint_sha256": f"{seed + 35:064x}",
                            "receipt_sha256": f"{seed + 42:064x}"})
            hf_checkpoints[arm, seed] = {
                "method": "multifidelity_correction", "seed": seed,
                "low_fidelity_method": method,
                "lf_checkpoint_sha256": lf_sha,
                "hf_train_powers_w": hf_powers,
                "hf_validation_powers_w": hf_validation,
                "model_state": {"correction.0.weight": torch.zeros(128, 6)},
                "correction_model_kwargs": {"width": 128, "depth": 4},
                "provenance": {"test_labels_consumed": False},
            }
            hf_receipts[arm, seed] = (
                {"seed": seed, "epochs_completed": 1700,
                 "training_seconds": 20.0, "best_epoch": 600}
                if arm in ("B0", "MLP") else
                {"运行种子": seed, "运行臂": arm, "校正实际轮次": 500,
                 "联合实际轮次": 200, "本会话耗时秒": 3.0,
                 "旧test_Data温度标签读取": False}
            )
    return hf_rows, lf_rows, hf_checkpoints, lf_checkpoints, hf_receipts, lf_receipts


def _summarize(sources=None, **overrides):
    from sic_cu.eval.task11_method_fairness_gate import summarize_existing_contract
    from sic_cu.config import load_yaml
    from sic_cu.data.splits import build_power_splits

    values = sources or _synthetic_sources()
    return summarize_existing_contract(
        *values, splits=build_power_splits(),
        training=load_yaml("configs/training.yaml"), **overrides,
    )


def test_pure_contract_keeps_twenty_models_and_cost_cohorts_separate():
    summary = _summarize()
    assert len(summary["已有HF来源身份"]) == 20
    assert len(summary["已有LF来源身份"]) == 10
    assert {row["HF训练成本口径"] for row in summary["已有HF来源身份"]} == {
        "旧V4完整1700轮单次会话", "新E0/F2最近断点会话；非累计总耗时",
    }
    assert summary["旧新训练墙钟直接排名许可"] is False
    assert summary["Howard原文三子网精确复现资格"] is False
    assert summary["新正式公平MLP与Howard五种子模型数"] == 0
    assert "训练墙钟排名" not in summary
    by_arm = {(row["方法"], row["seed"]): row for row in summary["已有HF来源身份"]}
    assert by_arm["F2", 0]["HF体内PDE训练"] is False
    assert by_arm["E0", 0]["HF体内PDE训练"] is True
    assert by_arm["MLP", 0]["HF校正架构"] == "LF温度加单个MLP校正器"
    assert by_arm["B0", 0]["边界初值界面训练约束保留"] is True


def test_rejects_passing_mlp_lf_as_registered_deeponet_e0():
    values = list(_synthetic_sources())
    values[2] = copy.deepcopy(values[2])
    values[2]["E0", 0]["low_fidelity_method"] = "mlp_pinn"
    with pytest.raises(ValueError, match="LF|低保真"):
        _summarize(tuple(values))


def test_rejects_existing_single_correction_as_howard_exact_reproduction():
    with pytest.raises(ValueError, match="Howard|三子网"):
        _summarize(architecture_claim="Howard论文原文精确复现")


def test_rejects_ranking_historical_full_seconds_against_new_partial_seconds():
    with pytest.raises(ValueError, match="墙钟|耗时|口径"):
        _summarize(rank_training_seconds=True)


def test_rejects_non_train_lf_powers_even_when_origin_label_is_registered():
    values = list(_synthetic_sources())
    values[3] = copy.deepcopy(values[3])
    values[3]["mlp_pinn", 0]["train_powers_w"] = [90.0]
    with pytest.raises(ValueError, match="LF|功率"):
        _summarize(tuple(values))


def test_real_registered_cpu_gate_only_reads_existing_hf_lf_origins(tmp_path):
    from sic_cu.eval.task11_method_fairness_gate import audit_registered_existing_sources

    destination = tmp_path / "reserved_new_training_output"
    summary = audit_registered_existing_sources(
        registry=TASK13_REGISTRY, registry_sha=TASK13_REGISTRY_SHA,
        source_tar=TASK13_SOURCE_TAR, source_tar_sha=TASK13_SOURCE_SHA,
        ledger="多保真DeepONet预测精度优化总计划与执行台账.md",
        output=destination,
    )
    assert len(summary["已有HF来源身份"]) == 20
    assert len(summary["已有LF来源身份"]) == 10
    assert summary["新正式公平MLP与Howard五种子模型数"] == 0
    assert not destination.exists()


def test_real_cpu_gate_rejects_forged_registry_sha_before_any_output(tmp_path):
    from sic_cu.eval.task11_method_fairness_gate import audit_registered_existing_sources

    destination = tmp_path / "reserved_new_training_output"
    with pytest.raises(ValueError, match="SHA|来源|登记"):
        audit_registered_existing_sources(
            registry=TASK13_REGISTRY, registry_sha="0" * 64,
            source_tar=TASK13_SOURCE_TAR, source_tar_sha=TASK13_SOURCE_SHA,
            ledger="多保真DeepONet预测精度优化总计划与执行台账.md",
            output=destination,
        )
    assert not destination.exists()


def test_task11_own_budget_and_seven_source_tar_require_same_0073_ledger_row(tmp_path):
    from sic_cu.eval.task11_method_fairness_gate import require_task11_registration

    sources = {}
    for index in range(7):
        path = tmp_path / f"source_{index}.txt"
        path.write_text(f"source-{index}\n", encoding="utf-8")
        relative = str(path.relative_to(PROJECT_ROOT))
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    registry = tmp_path / "budget.yaml"
    registry.write_text(yaml.safe_dump({
        "schema_version": 1, "事前台账记录编号": "录-0073",
        "性质": "任11方法级公平预算与来源身份CPU预检；不训练新模型",
        "新MLP或Howard训练许可": False,
        "旧TEST温度读取": False,
        "七源普通原件SHA256": sources,
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")
    source_tar = tmp_path / "sources.tar.gz"
    with tarfile.open(source_tar, "w:gz") as archive:
        for relative in sources:
            archive.add(PROJECT_ROOT / relative, arcname=relative)
    registry_sha = hashlib.sha256(registry.read_bytes()).hexdigest()
    source_tar_sha = hashlib.sha256(source_tar.read_bytes()).hexdigest()
    ledger = tmp_path / "ledger.md"
    ledger.write_text("| 编号 | 说明 |\n", encoding="utf-8")
    with pytest.raises(ValueError, match="0073|总账|同一"):
        require_task11_registration(
            registry, registry_sha, source_tar, source_tar_sha, ledger,
        )
    ledger.write_text(
        f"| 录-0073 | 任11预算 `{registry_sha}` 七源 `{source_tar_sha}` |\n",
        encoding="utf-8",
    )
    loaded = require_task11_registration(
        registry, registry_sha, source_tar, source_tar_sha, ledger,
    )
    assert loaded["新MLP或Howard训练许可"] is False


def test_chinese_report_writer_only_exports_contract_and_missing_preregistration(tmp_path):
    from sic_cu.eval.task11_method_fairness_gate import write_chinese_contract_report

    summary = _summarize()
    written = write_chinese_contract_report(summary, tmp_path / "fresh_report")
    report = (tmp_path / "fresh_report" / "任11方法公平预算与来源限界报告.md").read_text(
        encoding="utf-8"
    )
    machine = json.loads((tmp_path / "fresh_report" / "任11方法公平预算与来源机器摘要.json").read_text(
        encoding="utf-8"
    ))
    assert "旧新训练墙钟不能直接排名" in report
    assert "Howard" in report and "精确复现资格：否" in report
    assert "新方法成绩" not in report
    assert machine["新正式公平MLP与Howard五种子模型数"] == 0
    assert set(written) == {"中文报告", "机器摘要"}


def test_cpu_cli_exposes_only_registered_audit_arguments():
    script = PROJECT_ROOT / "scripts/41_audit_task11_method_fairness.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "任11既有方法来源与异口径预算CPU预检" in result.stdout
    for forbidden in ("--device", "--epochs", "--resume", "--train"):
        assert forbidden not in result.stdout
