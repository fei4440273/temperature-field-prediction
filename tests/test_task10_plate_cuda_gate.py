"""Synthetic-only checks for the Task 10 real-CUDA qualification gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.models.task10_plate_deeponet import BoardDeepONet
from test_task10_plate_group_gate import synthetic_25_locked_boards


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _synthetic_cuda_identity(tmp_path: Path, *, mutation: str | None = None,
                             method: str = "F1",
                             lf_source: str | None = None) -> dict:
    arm = "F1" if method == "F1" else "PAIR"
    run = tmp_path / f"正式{method}_seed0_合成CUDA收据"
    model_dir = run / method
    model_dir.mkdir(parents=True)
    model = BoardDeepONet(width=32, latent_dim=32, blocks=1)
    best = model_dir / "合法验证选定板模型_state_dict.pt"
    final = model_dir / "终态HF模型_state_dict.pt"
    torch.save(model.state_dict(), best)
    torch.save(model.state_dict(), final)
    rng = {
        "python_random": (3, (), None),
        "numpy_global": {"keys": list(range(624))},
        "torch_cpu": torch.arange(8, dtype=torch.uint8),
        "torch_cuda_all": [torch.arange(16, dtype=torch.uint8)],
    }
    if mutation == "empty_cuda_rng":
        rng["torch_cuda_all"] = []
    best_rng = model_dir / "最佳HF四类随机态.pt"
    final_rng = model_dir / "终态HF四类随机态.pt"
    torch.save(rng, best_rng)
    torch.save(rng, final_rng)
    validation = model_dir / "合法探针验证.json"
    _write_json(validation, {"模型_SHA256": sha256_file(best)})
    train = model_dir / "真实训练日志.json"
    training = {
        "方法": method, "随机种子": 0, "LF来源": lf_source,
        "设备": "cpu" if mutation == "cpu" else "cuda",
        "模型_SHA256": sha256_file(best),
        "终态HF模型状态_SHA256": sha256_file(final),
        "最佳HF四类随机态_SHA256": sha256_file(best_rng),
        "终态HF四类随机态_SHA256": sha256_file(final_rng),
    }
    if method in ("F2", "F3"):
        lf_rng = model_dir / "共享LF四类采样随机态.pt"
        torch.save(rng, lf_rng)
        training["共享LF四类采样随机态_SHA256"] = sha256_file(lf_rng)
    _write_json(train, training)
    lock = model_dir / "合法训练验证后模型SHA锁.json"
    _write_json(lock, {
        "模型文件": str(best), "模型_SHA256": sha256_file(best),
        "训练日志文件": str(train), "训练日志_SHA256": sha256_file(train),
        "合法探针验证文件": str(validation),
        "合法探针验证_SHA256": sha256_file(validation),
    })
    start = run / "开始训练源码与受限源凭据.json"
    _write_json(start, {"设备": "cuda", "运行方法": arm, "随机种子": 0,
                        "单LF来源": None if mutation == "wrong_pair_start_lf" else lf_source})
    summary = run / "合法模型训练与资源运行摘要.json"
    peak = None if mutation == "missing_peak" else 18.25
    _write_json(summary, {"CUDA峰值分配MiB": peak, "真正训练耗时_秒": 2.5,
                          "方法成绩仅合法三探针": [{"方法": method}]})
    receipt = run / "真CUDA运行设备与封存原件收据.json"
    device_count = 2 if mutation == "device_count" else 1
    device = {
        "类型": "cuda", "可见CUDA设备数": device_count, "实际设备序号": 0,
        "名称": "synthetic-cuda", "计算能力": [8, 6], "总显存字节": 8 * 2**30,
        "PyTorch版本": "" if mutation == "missing_torch" else "2.5.1+cu124",
        "CUDA运行时版本": "12.4",
    }
    artifact = {
        "模型目录": str(model_dir),
        "真实训练日志_SHA256": sha256_file(train),
        "合法验证_SHA256": sha256_file(validation),
        "单模型锁_SHA256": sha256_file(lock),
        "最佳模型_SHA256": sha256_file(best),
        "终态模型_SHA256": sha256_file(final),
        "最佳HF四类随机态_SHA256": sha256_file(best_rng),
        "终态HF四类随机态_SHA256": sha256_file(final_rng),
    }
    if method in ("F2", "F3"):
        artifact["共享LF四类采样随机态_SHA256"] = (
            training["共享LF四类采样随机态_SHA256"])
    _write_json(receipt, {
        "schema_version": 1, "正式GPU专属包装器": True, "运行状态": "成功",
        "设备": device,
        "训练": {"arm": arm, "随机种子": 0, "LF来源": lf_source,
                 "真正训练耗时_秒": 2.5, "CUDA峰值分配MiB": peak,
                 "旧冻结训练CLI_SHA256":
                 "b7ed35d95e53d2211cb15c07ef1f1eefa271deefe7f68fde188b2f27e0293cd3",
                 "开始训练凭据_SHA256": sha256_file(start),
                 "运行摘要_SHA256": sha256_file(summary)},
        "模型原件": {method: artifact},
        "失败目录不恢复且新批次重跑": True,
    })
    return {"方法": method, "随机种子": 0, "LF来源": lf_source,
            "模型目录": str(model_dir), "CUDA运行收据文件": str(receipt),
            "CUDA运行收据_SHA256": sha256_file(receipt)}


def test_existing_25_cpu_machine_artifacts_are_not_real_cuda(
    synthetic_25_locked_boards,
):
    from sic_cu.eval.task10_plate_cuda_gate import verify_25_cuda_evidence

    group, _ledger = synthetic_25_locked_boards
    identities = json.loads(group.read_text(encoding="utf-8"))["身份"]
    with pytest.raises(PermissionError, match="CUDA|cuda|设备"):
        verify_25_cuda_evidence(identities)


@pytest.mark.parametrize("mutation", [
    "cpu", "empty_cuda_rng", "device_count", "missing_torch", "missing_peak",
])
def test_cuda_receipt_rejects_missing_device_rng_version_or_peak(tmp_path, mutation):
    from sic_cu.eval.task10_plate_cuda_gate import verify_cuda_identity_receipt

    identity = _synthetic_cuda_identity(tmp_path, mutation=mutation)
    with pytest.raises(PermissionError, match="CUDA|cuda|设备|版本|显存"):
        verify_cuda_identity_receipt(identity)


def test_cuda_receipt_binds_best_final_log_and_lock_original_sha(tmp_path):
    from sic_cu.eval.task10_plate_cuda_gate import verify_cuda_identity_receipt

    identity = _synthetic_cuda_identity(tmp_path)
    result = verify_cuda_identity_receipt(identity)
    assert result["方法"] == "F1"
    final = Path(identity["模型目录"]) / "终态HF模型_state_dict.pt"
    weights = torch.load(final, map_location="cpu", weights_only=True)
    weights["bias"] += 1
    torch.save(weights, final)
    with pytest.raises(PermissionError, match="SHA|原件|绑定"):
        verify_cuda_identity_receipt(identity)


def test_pair_start_receipt_must_bind_the_exact_lf_source(tmp_path):
    from sic_cu.eval.task10_plate_cuda_gate import verify_cuda_identity_receipt

    identity = _synthetic_cuda_identity(
        tmp_path, mutation="wrong_pair_start_lf", method="F2",
        lf_source="lf_same_physics_coarse",
    )
    with pytest.raises(PermissionError, match="LF|身份|凭据"):
        verify_cuda_identity_receipt(identity)


def test_gpu_only_wrapper_builds_fixed_legacy_cuda_command(tmp_path):
    script = PROJECT_ROOT / "scripts/任务10_同板真CUDA专属训练.py"
    spec = importlib.util.spec_from_file_location("task10_cuda_only", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.build_legacy_cuda_command(
        arm="F1", lf_source=None, seed=0, output=tmp_path / "new-run")
    assert command.count("--device") == 1
    assert command[command.index("--device") + 1] == "cuda"
    assert "cpu" not in command


def _expected_cuda_identities() -> list[dict]:
    values = []
    for seed in range(5):
        values.append({"方法": "F1", "随机种子": seed, "LF来源": None})
    for kind in ("lf_same_physics_coarse", "lf_contact_mismatch_coarse"):
        for seed in range(5):
            for method in ("F2", "F3"):
                values.append({"方法": method, "随机种子": seed, "LF来源": kind})
    return values


def test_25_cuda_gate_requires_fifteen_runs_and_one_device_profile(monkeypatch):
    import sic_cu.eval.task10_plate_cuda_gate as gate

    identities = _expected_cuda_identities()

    def verify(identity):
        method, seed, kind = identity["方法"], identity["随机种子"], identity["LF来源"]
        run = f"F1-{seed}" if method == "F1" else f"PAIR-{kind}-{seed}"
        return {**identity, "CUDA运行收据_SHA256": run.ljust(64, "0"),
                "可见CUDA设备数": 1, "运行设备名称": "synthetic-cuda",
                "PyTorch版本": "2.5.1+cu124", "CUDA运行时版本": "12.4"}

    monkeypatch.setattr(gate, "verify_cuda_identity_receipt", verify)
    report = gate.verify_25_cuda_evidence(identities)
    assert report["身份总数"] == 25
    assert report["真CUDA运行批次总数"] == 15
    assert report["完整HF温度已读取"] is False

    def inconsistent(identity):
        result = verify(identity)
        if identity["方法"] == "F1" and identity["随机种子"] == 4:
            result["可见CUDA设备数"] = 2
        return result

    monkeypatch.setattr(gate, "verify_cuda_identity_receipt", inconsistent)
    with pytest.raises(PermissionError, match="设备数|版本|设备名"):
        gate.verify_25_cuda_evidence(identities)


def test_failed_cuda_directory_is_permanent_and_cannot_resume(tmp_path, monkeypatch):
    script = PROJECT_ROOT / "scripts/任务10_同板真CUDA专属训练.py"
    spec = importlib.util.spec_from_file_location("task10_cuda_failure", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_assert_frozen_inputs", lambda: None)
    monkeypatch.setattr(module, "verify_cuda_source_freeze", lambda **_kwargs: {})
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(module.torch.cuda, "get_device_properties", lambda _index:
                        SimpleNamespace(name="synthetic-cuda", major=8, minor=6,
                                        total_memory=8 * 2**30))
    monkeypatch.setattr(module.torch.version, "cuda", "12.4")
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=3, stdout="synthetic-out",
                                        stderr="synthetic-error"))
    output = tmp_path / "失败批次必须永久留档"
    with pytest.raises(RuntimeError, match="非零退出"):
        module.run_cuda_only(arm="F1", lf_source=None, seed=0, output=output,
                             cuda_budget_sha256="a" * 64,
                             cuda_tar_sha256="b" * 64)
    failure = output / "真CUDA失败目录永久保留_禁止恢复.json"
    record = json.loads(failure.read_text(encoding="utf-8"))
    assert record["禁止断点恢复"] is True
    assert record["后续只能换全新批次目录完整重跑"] is True
    with pytest.raises(FileExistsError, match="不得覆盖|恢复"):
        module.run_cuda_only(arm="F1", lf_source=None, seed=0, output=output,
                             cuda_budget_sha256="a" * 64,
                             cuda_tar_sha256="b" * 64)


def test_training_wrapper_checks_0074_before_cuda_or_subprocess(tmp_path, monkeypatch):
    script = PROJECT_ROOT / "scripts/任务10_同板真CUDA专属训练.py"
    spec = importlib.util.spec_from_file_location("task10_cuda_preregister", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_assert_frozen_inputs", lambda: None)
    monkeypatch.setattr(module, "verify_cuda_source_freeze", lambda **_kwargs:
                        (_ for _ in ()).throw(PermissionError("录0074尚未事前登记")),
                        raising=False)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda:
                        pytest.fail("录0074前不得探测CUDA"))
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs:
                        pytest.fail("录0074前不得启动旧训练CLI"))
    output = tmp_path / "录0074前不得创建目录"
    with pytest.raises(PermissionError, match="0074"):
        module.run_cuda_only(arm="F1", lf_source=None, seed=0, output=output,
                             cuda_budget_sha256="a" * 64,
                             cuda_tar_sha256="b" * 64)
    assert not output.exists()


def test_cuda_freeze_requires_budget_and_tar_in_reserved_0074_row(tmp_path):
    from sic_cu.eval.task10_plate_cuda_gate import verify_cuda_source_freeze

    root = PROJECT_ROOT / "研究记录/任务10_独立双层场基准"
    budget = root / "正式同板真CUDA训练与后验资格v4前登记.yaml"
    archive = root / "正式同板真CUDA资格十二源事前冻结.tar.gz"
    budget_sha, archive_sha = sha256_file(budget), sha256_file(archive)
    ledger = tmp_path / "纯合成录0074主账.md"
    ledger.write_text(f"| 录-0073 | {budget_sha} / {archive_sha} |\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="0074|总账|事前"):
        verify_cuda_source_freeze(
            cuda_budget_path=budget, cuda_budget_sha256=budget_sha,
            cuda_tar_path=archive, cuda_tar_sha256=archive_sha,
            ledger_path=ledger,
        )
    ledger.write_text(f"| 录-0074 | {budget_sha} / {archive_sha} |\n", encoding="utf-8")
    proof = verify_cuda_source_freeze(
        cuda_budget_path=budget, cuda_budget_sha256=budget_sha,
        cuda_tar_path=archive, cuda_tar_sha256=archive_sha,
        ledger_path=ledger,
    )
    assert proof["CUDA资格预算_SHA256"] == budget_sha
    assert proof["CUDA资格十二源tar_SHA256"] == archive_sha
    assert proof["完整HF温度已读取"] is False


def test_new_posthoc_wrapper_rejects_cpu_group_before_v3_or_hf(
    synthetic_25_locked_boards, tmp_path, monkeypatch,
):
    script = PROJECT_ROOT / "scripts/任务10_真CUDA资格后全组独立一维HF后验审计_v4.py"
    spec = importlib.util.spec_from_file_location("task10_cuda_posthoc", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    group, _ledger = synthetic_25_locked_boards
    calls = []
    monkeypatch.setattr(module, "_assert_old_frozen_bytes", lambda: calls.append("旧SHA"))
    monkeypatch.setattr(module, "verify_cuda_source_freeze",
                        lambda **_kwargs: calls.append("0074") or {})
    monkeypatch.setattr(module, "verify_late_posthoc_contract",
                        lambda **_kwargs: calls.append("0066") or {})
    monkeypatch.setattr(module, "verify_v3_source_freeze",
                        lambda **_kwargs: calls.append("0070") or {})
    monkeypatch.setattr(module, "run_complete_plate_posthoc_v3",
                        lambda **_kwargs: pytest.fail("CPU组不得进入v3或完整HF"))
    output = tmp_path / "CPU组不得创建后验输出"
    with pytest.raises(PermissionError, match="CUDA|cuda|设备"):
        module.main([
            "--全组身份文件", str(group),
            "--全组SHA256", sha256_file(group),
            "--CUDA资格预算SHA256", "a" * 64,
            "--CUDA资格源码tarSHA256", "b" * 64,
            "--输出目录", str(output),
        ])
    assert calls == ["旧SHA", "0074", "0066", "0070"]
    assert not output.exists()
