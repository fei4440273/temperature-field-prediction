"""Pre-registration checks for a genuinely fresh Task-11 MLP LF source."""

from __future__ import annotations

import importlib
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.splits import build_power_splits
from sic_cu.train.task11_mlp_lf_formal import SOURCE_MEMBERS


def test_registered_real_simulation_source_excludes_test_temperature_files() -> None:
    formal = importlib.import_module("sic_cu.train.task11_mlp_lf_formal")
    catalog = formal.collect_task11_mlp_lf_sources()
    split = build_power_splits()

    training = catalog["LF训练模拟原件"]
    validation = catalog["LF合法验证模拟原件"]
    assert len(training) == 60
    assert len(validation) == 10
    assert {row["功率_瓦"] for row in training} == split.simulation_train
    assert {row["功率_瓦"] for row in validation} == split.simulation_validation
    assert not ({row["功率_瓦"] for row in training + validation} & split.simulation_test)
    assert catalog["模拟测试功率温度读取"] is False
    assert all(Path(PROJECT_ROOT / row["原件"]).is_file()
               and len(row["文件SHA256"]) == 64 for row in training + validation)


def _isolated_frozen_source(tmp_path: Path) -> tuple[object, dict, Path, str, Path, str, Path, str, Path]:
    formal = importlib.import_module("sic_cu.train.task11_mlp_lf_formal")
    catalog = formal.collect_task11_mlp_lf_sources()
    catalog_path = tmp_path / "真实模拟70源.json"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, sort_keys=True),
                            encoding="utf-8")
    catalog_sha = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    sources: dict[str, str] = {}
    tar_path = tmp_path / "新MLP_LF源码冻结.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        for name in SOURCE_MEMBERS:
            destination = tmp_path / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (PROJECT_ROOT / name).is_file():
                shutil.copyfile(PROJECT_ROOT / name, destination)
            else:
                destination.write_text("测试目录专用入口占位", encoding="utf-8")
            sources[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
            archive.add(destination, arcname=name, recursive=False)
    tar_sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    budget = {
        "schema_version": 1, "阶段": "fresh_mlp_low_fidelity",
        "ROOT门禁标签": "TASK11_NEW_MLP_LF_GATE:v1",
        "旧固定TEST温度读取": False,
        "模拟测试功率温度读取": False,
        "新HF训练许可": False,
        "真实模拟70源目录SHA256": catalog_sha,
        "源码普通成员SHA256": sources,
        "正式预算": {"seeds": [0, 1, 2, 3, 4], "method": "mlp_pinn",
                 "epochs": 2000, "samples_per_power": 8192,
                 "validation_samples_per_power": 8192, "batch_size": 8192,
                 "learning_rate": 0.001, "patience": 200,
                 "physics_collocation": 256, "physics_weight": 1.0,
                 "lf_physics_mode": "original", "sampling_mode": "material_time",
                 "device": "cuda", "world_size": 1},
    }
    budget_path = tmp_path / "任11新MLP_LF事前登记.yaml"
    budget_path.write_text(yaml.safe_dump(budget, allow_unicode=True, sort_keys=False),
                           encoding="utf-8")
    budget_sha = hashlib.sha256(budget_path.read_bytes()).hexdigest()
    ledger_path = tmp_path / "多保真DeepONet预测精度优化总计划与执行台账.md"
    return (formal, budget, budget_path, budget_sha, tar_path, tar_sha,
            catalog_path, catalog_sha, ledger_path)


def test_missing_root_gate_rejects_even_with_two_hashes_in_other_columns(tmp_path: Path) -> None:
    (formal, _, budget_path, budget_sha, tar_path, tar_sha,
     catalog_path, catalog_sha, ledger_path) = _isolated_frozen_source(tmp_path)
    ledger_path.write_text(
        f"| 录-0098 | 仅供测试的假活动 | YAML_SHA256={budget_sha} | TAR_SHA256={tar_sha} |\n",
        encoding="utf-8",
    )
    output = (tmp_path / "研究记录/任务11_外部对照/正式新MLP公平训练"
              / "正式MLP_LF_seed0")
    with pytest.raises(ValueError, match="ROOT"):
        formal.preflight_task11_mlp_lf(
            registry=budget_path, registry_sha=budget_sha,
            source_tar=tar_path, source_tar_sha=tar_sha,
            catalog=catalog_path, catalog_sha=catalog_sha,
            output=output, seed=0, ledger=ledger_path, project_root=tmp_path,
        )
    assert not output.exists()


def test_frozen_source_drift_rejected_after_real_root_gate(tmp_path: Path) -> None:
    (formal, budget, budget_path, budget_sha, tar_path, tar_sha,
     catalog_path, catalog_sha, ledger_path) = _isolated_frozen_source(tmp_path)
    ledger_path.write_text(
        ("| 录-0099 | TASK11_NEW_MLP_LF_GATE:v1; status=active; "
         f"YAML_SHA256={budget_sha}; TAR_SHA256={tar_sha}; "
         f"CATALOG_SHA256={catalog_sha} | 测试 |\n"), encoding="utf-8",
    )
    changed = tmp_path / next(iter(budget["源码普通成员SHA256"]))
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="源码|SHA"):
        formal.preflight_task11_mlp_lf(
            registry=budget_path, registry_sha=budget_sha,
            source_tar=tar_path, source_tar_sha=tar_sha,
            catalog=catalog_path, catalog_sha=catalog_sha,
            output=(tmp_path / "研究记录/任务11_外部对照/正式新MLP公平训练"
                    / "正式MLP_LF_seed1"),
            seed=1, ledger=ledger_path, project_root=tmp_path,
        )


def test_training_rejects_unactivated_root_before_cuda_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    (formal, _, budget_path, budget_sha, tar_path, tar_sha,
     catalog_path, catalog_sha, ledger_path) = _isolated_frozen_source(tmp_path)
    ledger_path.write_text(
        f"| 录-0098 | 假入口 | YAML_SHA256={budget_sha}; TAR_SHA256={tar_sha} |\n",
        encoding="utf-8",
    )
    destination = (tmp_path / "研究记录/任务11_外部对照/正式新MLP公平训练"
                   / "正式MLP_LF_seed0")

    def forbid_cuda_probe() -> bool:
        raise AssertionError("未先锁不得探测或占用CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbid_cuda_probe)
    with pytest.raises(ValueError, match="ROOT"):
        formal.run_task11_mlp_lf_formal(
            registry=budget_path, registry_sha=budget_sha,
            source_tar=tar_path, source_tar_sha=tar_sha,
            catalog=catalog_path, catalog_sha=catalog_sha,
            output=destination, seed=0, ledger=ledger_path,
            project_root=tmp_path, device_name="cuda", session_epoch_limit=200,
        )
    assert not destination.exists()


def test_resume_rejects_other_seed_checkpoint_before_cuda(tmp_path: Path) -> None:
    (formal, _, budget_path, budget_sha, tar_path, tar_sha,
     catalog_path, catalog_sha, ledger_path) = _isolated_frozen_source(tmp_path)
    ledger_path.write_text(
        ("| 录-0099 | TASK11_NEW_MLP_LF_GATE:v1; status=active; "
         f"YAML_SHA256={budget_sha}; TAR_SHA256={tar_sha}; "
         f"CATALOG_SHA256={catalog_sha} | 测试 |\n"), encoding="utf-8",
    )
    destination = (tmp_path / "研究记录/任务11_外部对照/正式新MLP公平训练"
                   / "正式MLP_LF_seed2")
    destination.mkdir(parents=True)
    wrong = tmp_path / "研究记录/任务11_外部对照/正式新MLP公平训练" / "正式MLP_LF_seed3"
    wrong.mkdir()
    (wrong / "阶段_最近.pt").write_text("其他seed无效断点", encoding="utf-8")
    with pytest.raises(ValueError, match="同seed|最近|续跑"):
        formal.preflight_task11_mlp_lf(
            registry=budget_path, registry_sha=budget_sha,
            source_tar=tar_path, source_tar_sha=tar_sha,
            catalog=catalog_path, catalog_sha=catalog_sha,
            output=destination, seed=2, ledger=ledger_path,
            project_root=tmp_path, resume_checkpoint=wrong / "阶段_最近.pt",
        )


def test_resume_rejects_hash_consistent_wrong_internal_seed_before_cuda(tmp_path: Path) -> None:
    import torch

    (formal, _, budget_path, budget_sha, tar_path, tar_sha,
     catalog_path, catalog_sha, ledger_path) = _isolated_frozen_source(tmp_path)
    ledger_path.write_text(
        ("| 录-0099 | TASK11_NEW_MLP_LF_GATE:v1; status=active; "
         f"YAML_SHA256={budget_sha}; TAR_SHA256={tar_sha}; "
         f"CATALOG_SHA256={catalog_sha} | 测试 |\n"), encoding="utf-8",
    )
    destination = (tmp_path / "研究记录/任务11_外部对照/正式新MLP公平训练"
                   / "正式MLP_LF_seed0")
    destination.mkdir(parents=True)
    recent = destination / "阶段_最近.pt"
    torch.save({"training_state_schema_version": 1,
                "stage": "low_fidelity", "epoch": 1,
                "budget": {"LF轮次": 2000},
                "metadata": {"seed": 3, "YAML_SHA256": budget_sha,
                             "TAR_SHA256": tar_sha,
                             "CATALOG_SHA256": catalog_sha}}, recent)
    log_path = destination / "training.jsonl"
    log_path.write_text(json.dumps({"epoch": 1}) + "\n", encoding="utf-8")
    receipt = {"seed": 0, "状态": "已暂停且完整阶段提交",
               "YAML_SHA256": budget_sha, "TAR_SHA256": tar_sha,
               "CATALOG_SHA256": catalog_sha,
               "累计实际轮次": 1,
               "最近阶段SHA256": hashlib.sha256(recent.read_bytes()).hexdigest(),
               "日志SHA256": hashlib.sha256(log_path.read_bytes()).hexdigest()}
    (destination / "LF会话收据_0001.json").write_text(
        json.dumps(receipt), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="seed|来源|断点|状态"):
        formal.preflight_task11_mlp_lf(
            registry=budget_path, registry_sha=budget_sha,
            source_tar=tar_path, source_tar_sha=tar_sha,
            catalog=catalog_path, catalog_sha=catalog_sha,
            output=destination, seed=0, ledger=ledger_path,
            project_root=tmp_path, resume_checkpoint=recent,
        )


def test_cpu_preflight_accepts_own_complete_pause_and_no_cuda_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    (formal, _, budget_path, budget_sha, tar_path, tar_sha,
     catalog_path, catalog_sha, ledger_path) = _isolated_frozen_source(tmp_path)
    ledger_path.write_text(
        ("| 录-0099 | TASK11_NEW_MLP_LF_GATE:v1; status=active; "
         f"YAML_SHA256={budget_sha}; TAR_SHA256={tar_sha}; "
         f"CATALOG_SHA256={catalog_sha} | 测试 |\n"), encoding="utf-8",
    )
    output = (tmp_path / "研究记录/任务11_外部对照/正式新MLP公平训练"
              / "正式MLP_LF_seed1")
    output.mkdir(parents=True)
    log_path = output / "training.jsonl"
    log_path.write_text(json.dumps({"epoch": 1, "seed": 1,
                                    "累计训练点": 60 * 8192,
                                    "累计优化步": 61,
                                    "旧固定TEST温度读取": False,
                                    "模拟测试功率温度读取": False}) + "\n",
                        encoding="utf-8")
    log_sha = hashlib.sha256(log_path.read_bytes()).hexdigest()
    checkpoint = output / "阶段_最近.pt"
    torch.save({"training_state_schema_version": 1, "stage": "low_fidelity",
                "epoch": 1, "budget": {"LF轮次": 2000},
                "model_state": {"weight": torch.ones(1)},
                "optimizer_state": {"state": {}},
                "random_state": {name: None for name in
                                 ("python", "numpy", "torch_cpu", "torch_cuda")},
                "metadata": {"seed": 1, "LF方法": "mlp_pinn", "当前轮次": 1,
                             "旧固定TEST温度读取": False,
                             "模拟测试功率温度读取": False,
                             "日志SHA256": log_sha,
                             "YAML_SHA256": budget_sha,
                             "TAR_SHA256": tar_sha,
                             "CATALOG_SHA256": catalog_sha}}, checkpoint)
    receipt = {"seed": 1, "状态": "已暂停且完整阶段提交",
               "YAML_SHA256": budget_sha, "TAR_SHA256": tar_sha,
               "CATALOG_SHA256": catalog_sha, "累计实际轮次": 1,
               "最近阶段SHA256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "日志SHA256": log_sha}
    (output / "LF会话收据_0001.json").write_text(
        json.dumps(receipt), encoding="utf-8",
    )

    def forbid_cuda_probe() -> bool:
        raise AssertionError("CPU续跑前检不得探测CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbid_cuda_probe)
    result = formal.preflight_task11_mlp_lf(
        registry=budget_path, registry_sha=budget_sha,
        source_tar=tar_path, source_tar_sha=tar_sha,
        catalog=catalog_path, catalog_sha=catalog_sha,
        output=output, seed=1, ledger=ledger_path,
        project_root=tmp_path, resume_checkpoint=checkpoint,
    )
    assert result["seed"] == 1
    assert result["状态"].startswith("CPU前置门禁PASS")


def test_nonfinite_lf_validation_cannot_be_accepted_as_best_model() -> None:
    formal = importlib.import_module("sic_cu.train.task11_mlp_lf_formal")
    with pytest.raises(ValueError, match="有限|非有限"):
        formal.require_finite_lf_metrics(0.1, float("nan"), 0.2)
    with pytest.raises(ValueError, match="有限|非有限"):
        formal.require_finite_lf_metrics(0.1, float("inf"), 0.2)
    formal.require_finite_lf_metrics(0.1, 0.2, 0.15)


def test_source_drift_after_cpu_gate_rejects_before_model_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal = importlib.import_module("sic_cu.train.task11_mlp_lf_formal")
    recorded = formal.collect_task11_mlp_lf_sources()
    path = tmp_path / "真实模拟70源.json"
    path.write_text(json.dumps(recorded), encoding="utf-8")
    prereg = {"真实源目录原件": str(path),
              "CATALOG_SHA256": hashlib.sha256(path.read_bytes()).hexdigest()}
    formal.assert_task11_lf_source_unchanged(prereg)
    changed = dict(recorded)
    changed["模拟清单原文件SHA256"] = "0" * 64
    monkeypatch.setattr(formal, "collect_task11_mlp_lf_sources", lambda: changed)
    with pytest.raises(ValueError, match="漂移|来源"):
        formal.assert_task11_lf_source_unchanged(prereg)


def test_formal_cli_requires_frozen_sources_without_old_lf_or_test_override() -> None:
    runner = PROJECT_ROOT / "scripts/49_run_task11_mlp_lf_formal.py"
    help_text = subprocess.run(
        [sys.executable, str(runner), "--help"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    assert help_text.returncode == 0
    for required in ("--registry-sha", "--source-tar-sha", "--catalog-sha",
                     "--seed", "--session-epoch-limit", "--resume-checkpoint",
                     "--freeze-catalog", "--preflight-only", "--audit-finished"):
        assert required in help_text.stdout
    for forbidden in ("--lf-checkpoint", "--start-checkpoint", "--hf-test-powers",
                      "--evaluate-test", "--hf-arm"):
        assert forbidden not in help_text.stdout


def test_missing_real_lf_completion_cannot_qualify_for_later_hf(tmp_path: Path) -> None:
    (formal, _, budget_path, budget_sha, tar_path, tar_sha,
     catalog_path, catalog_sha, ledger_path) = _isolated_frozen_source(tmp_path)
    ledger_path.write_text(
        ("| 录-0099 | TASK11_NEW_MLP_LF_GATE:v1; status=active; "
         f"YAML_SHA256={budget_sha}; TAR_SHA256={tar_sha}; "
         f"CATALOG_SHA256={catalog_sha} | 测试 |\n"), encoding="utf-8",
    )
    destination = (tmp_path / "研究记录/任务11_外部对照/正式新MLP公平训练"
                   / "正式MLP_LF_seed4")
    destination.mkdir(parents=True)
    (destination / "best.pt").write_text("旧模型不能冒作新LF", encoding="utf-8")
    with pytest.raises(ValueError, match="完成|收据|阶段|日志"):
        formal.qualify_task11_mlp_lf_source(
            registry=budget_path, registry_sha=budget_sha,
            source_tar=tar_path, source_tar_sha=tar_sha,
            catalog=catalog_path, catalog_sha=catalog_sha,
            output=destination, seed=4, ledger=ledger_path,
            project_root=tmp_path,
        )


def test_old_five_mlp_lf_checkpoints_fail_current_provenance() -> None:
    from sic_cu.eval.protocol_checks import validate_lf_checkpoint_provenance
    import torch

    root = (PROJECT_ROOT /
            "reports/runs/sic-cu-v2-gpu1-20260908T072729Z-r2")
    for seed in range(5):
        payload = torch.load(root / f"mlp_pinn_lf_seed{seed}" / "best.pt",
                             map_location="cpu", weights_only=True)
        with pytest.raises(RuntimeError, match="physics_config_sha256"):
            validate_lf_checkpoint_provenance(payload)
