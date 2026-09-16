"""Task-09 v3 general formal entry for the remaining 14 primary models."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import copy
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest
import yaml

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file


def _v1():
    return importlib.import_module("sic_cu.train.task09_subset_formal")


def _v2():
    return importlib.import_module("sic_cu.train.task09_subset_formal_v2")


def _v3():
    try:
        return importlib.import_module("sic_cu.train.task09_subset_formal_v3")
    except ModuleNotFoundError:
        pytest.fail("Task-09 remaining-primary v3 formal entry is missing")


def _payload(*, stage: str, correction: int, joint: int) -> dict:
    return {
        "stage": stage,
        "epoch": correction if stage.endswith("correction") else joint,
        "metadata": {
            "当前阶段": stage,
            "校正实际轮次": correction,
            "联合实际轮次": joint,
            "完整全局实际轮次": correction + joint,
        },
    }


def test_v3_identity_is_exactly_primary_3_6_9_by_five_seeds_minus_completed_seed0_hf3():
    v3 = _v3()
    expected = {(size, seed) for size in (3, 6, 9) for seed in range(5)} - {(3, 0)}
    assert set(v3.remaining_task09_v3_identities()) == expected
    assert len(expected) == 14
    for size, seed in expected:
        assert v3.validate_task09_v3_identity("primary", size, seed) == (
            "primary", size, seed
        )
    for identity in (
        ("primary", 3, 0), ("left_center", 3, 1),
        ("right_center", 9, 4), ("primary", 12, 1), ("primary", 6, 5),
    ):
        with pytest.raises(ValueError, match="14|剩余|primary|HF3|seed0|身份"):
            v3.validate_task09_v3_identity(*identity)
    outputs = v3.task09_v3_canonical_outputs()
    assert set(outputs) == expected
    assert len(set(outputs.values())) == 14
    assert all(path.name.startswith("正式F3V3_primary_")
               and "_录0078_" in path.name
               and path.name.endswith("_20260916T071500+0800")
               for path in outputs.values())
    frozen_energy = importlib.import_module("sic_cu.eval.task09_energy")
    for (size, seed), path in outputs.items():
        with pytest.raises(ValueError, match="canonical|目录|任09"):
            _v1().validate_task09_output(path, "primary", size, seed)
        with pytest.raises(ValueError, match="canonical|目录|任09"):
            frozen_energy.validate_task09_output(path, "primary", size, seed)


def test_v3_fresh_and_resume_paths_are_canonical_and_never_overwrite(tmp_path, monkeypatch):
    v3 = _v3()
    root = tmp_path / "正式F3子集训练"
    root.mkdir()
    monkeypatch.setattr(v3, "TASK09_V3_FORMAL_ROOT", root)
    fresh = root / "正式F3V3_primary_HF6_seed2_录0078_20260916T071500+0800"
    assert v3.validate_task09_v3_output(
        fresh, "primary", 6, 2, resume_checkpoint=None,
    ) == fresh.resolve()
    fresh.mkdir()
    with pytest.raises(FileExistsError, match="已有|覆盖|全新"):
        v3.validate_task09_v3_output(
            fresh, "primary", 6, 2, resume_checkpoint=None,
        )
    recent = fresh / "阶段_最近.pt"
    recent.write_bytes(b"committed state")
    assert v3.validate_task09_v3_output(
        fresh, "primary", 6, 2, resume_checkpoint=recent,
    ) == fresh.resolve()
    with pytest.raises(ValueError, match="最近|checkpoint|续跑"):
        v3.validate_task09_v3_output(
            fresh, "primary", 6, 2, resume_checkpoint=fresh / "best.pt",
        )
    with pytest.raises(ValueError, match="canonical|目录|任09"):
        v3.validate_task09_v3_output(
            root / "wrong", "primary", 6, 2, resume_checkpoint=None,
        )
    with pytest.raises(ValueError, match="固定|canonical|唯一|目录"):
        v3.validate_task09_v3_output(
            root / "正式F3_primary_HF6_seed2_20260916T055618+0800",
            "primary", 6, 2, resume_checkpoint=None,
        )
    (fresh / "阶段_训练末.pt").write_bytes(b"terminal")
    with pytest.raises(ValueError, match="训练末|联合末|终态|完成"):
        v3.validate_task09_v3_output(
            fresh, "primary", 6, 2, resume_checkpoint=recent,
        )


def test_v3_identity_os_claim_rejects_concurrent_process_before_any_runner(tmp_path, monkeypatch):
    v3 = _v3()
    root = tmp_path / "正式F3子集训练"
    root.mkdir()
    monkeypatch.setattr(v3, "TASK09_V3_FORMAL_ROOT", root)
    identity = ("primary", 6, 2)
    with v3.task09_v3_run_claim(identity) as first:
        assert first["身份"] == {"序列": "primary", "HF功率数": 6, "seed": 2}
        with pytest.raises(RuntimeError, match="并发|占用|锁"):
            with v3.task09_v3_run_claim(identity):
                pytest.fail("second claim must never enter")
        with pytest.raises(RuntimeError, match="并发|占用|锁"):
            with v3.task09_v3_run_claim(identity):
                pytest.fail("failed second claim must not unlock the first")
    lock = root / ".任09V3_primary_HF6_seed2_录0078.lock"
    assert lock.is_file() and not lock.is_symlink()
    with v3.task09_v3_run_claim(identity) as second:
        assert second["锁路径"] == str(lock.resolve())


def test_v3_session_limit_requires_real_200_epoch_minimum_or_exact_short_remainder():
    v3, v1 = _v3(), _v1()
    with pytest.raises(ValueError, match="200|最低"):
        v3.validate_task09_v3_session_limit(None, 199)
    assert v3.validate_task09_v3_session_limit(None, 200)["本次最低请求轮次"] == 200
    for too_large in (201, 2000, 2001):
        with pytest.raises(ValueError, match="200|会话|上限"):
            v3.validate_task09_v3_session_limit(None, too_large)

    correction = _payload(stage=v1.CORRECTION_STAGE, correction=400, joint=0)
    with pytest.raises(ValueError, match="200|最低"):
        v3.validate_task09_v3_session_limit(correction, 100)
    assert v3.validate_task09_v3_session_limit(correction, 200)[
        "阶段剩余最多轮次"
    ] == 1600
    near_terminal = _payload(stage=v1.JOINT_STAGE, correction=500, joint=490)
    with pytest.raises(ValueError, match="剩余|10"):
        v3.validate_task09_v3_session_limit(near_terminal, 9)
    assert v3.validate_task09_v3_session_limit(near_terminal, 10)[
        "本次最低请求轮次"
    ] == 10
    with pytest.raises(ValueError, match="达到|完成|剩余"):
        v3.validate_task09_v3_session_limit(
            _payload(stage=v1.JOINT_STAGE, correction=500, joint=500), 1,
        )


def test_v3_transaction_accepts_only_log_exactly_equal_to_latest_committed_state(tmp_path):
    v3, v1 = _v3(), _v1()
    log = tmp_path / "training.jsonl"
    rows = [
        {"epoch": 1, "训练阶段": v1.CORRECTION_STAGE, "阶段实际轮次": 1},
        {"epoch": 2, "训练阶段": v1.CORRECTION_STAGE, "阶段实际轮次": 2},
    ]
    log.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                   encoding="utf-8")
    payload = _payload(stage=v1.CORRECTION_STAGE, correction=2, joint=0)
    receipt = v3.require_task09_v3_clean_transaction(log, payload)
    assert receipt["已提交日志行数"] == 2
    assert receipt["未提交尾行数"] == 0
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"epoch": 3}) + "\n")
    with pytest.raises(ValueError, match="未提交|尾行|事务"):
        v3.require_task09_v3_clean_transaction(log, payload)
    log.write_text('{"epoch": 1}\n{bad json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON|日志|事务"):
        v3.require_task09_v3_clean_transaction(log, payload)
    log.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                   encoding="utf-8")
    (tmp_path / "阶段_最近.pt.tmp").write_bytes(b"partial checkpoint")
    with pytest.raises(ValueError, match="临时|tmp|事务"):
        v3.require_task09_v3_clean_transaction(log, payload)


def test_v3_repairs_exact_correction_terminal_session_boundary_atomically(tmp_path):
    v3, v1 = _v3(), _v1()
    output = tmp_path / "run"
    output.mkdir()
    log = output / "training.jsonl"
    log.write_text(
        "".join(json.dumps({
            "epoch": epoch, "训练阶段": v1.CORRECTION_STAGE,
            "阶段实际轮次": epoch,
        }, ensure_ascii=False) + "\n" for epoch in range(1, 201)),
        encoding="utf-8",
    )
    payload = {
        **_payload(stage=v1.CORRECTION_STAGE, correction=200, joint=0),
        "model_state": {"correction.weight": __import__("torch").arange(4)},
        "optimizer_state": {"state": {0: {"step": __import__("torch").tensor(1000)}}},
        "parameter_requires_grad": {"correction.weight": True},
        "random_state": {"python": (3, (1, 2), None),
                         "numpy": ("MT19937", __import__("numpy").arange(4), 1, 0, 0.0),
                         "torch_cpu": __import__("torch").arange(8, dtype=__import__("torch").uint8),
                         "torch_cuda": [__import__("torch").arange(16, dtype=__import__("torch").uint8)]},
        "budget": {"correction": 1500, "joint": 500},
    }
    payload["metadata"].update({
        "校正实际截止轮次": 200,
        "已提交真实校正末态SHA256": None,
    })
    recent = output / "阶段_最近.pt"
    terminal = output / "阶段_校正末.pt"
    __import__("torch").save(copy.deepcopy(payload), recent)
    __import__("torch").save(copy.deepcopy(payload), terminal)
    log_sha = sha256_file(log)
    terminal_sha = sha256_file(terminal)

    receipt = v3.repair_task09_v3_terminal_boundary(output)
    repaired = __import__("torch").load(recent, map_location="cpu", weights_only=False)
    assert receipt == {
        "校正会话边界补交": True,
        "校正截止轮次": 200,
        "校正末态SHA256": terminal_sha,
    }
    assert repaired["metadata"]["已提交真实校正末态SHA256"] == terminal_sha
    assert sha256_file(terminal) == terminal_sha
    assert sha256_file(log) == log_sha
    assert not (output / "阶段_最近.pt.tmp").exists()

    mismatched = copy.deepcopy(payload)
    mismatched["model_state"]["correction.weight"][0] = 999
    __import__("torch").save(mismatched, recent)
    with pytest.raises(ValueError, match="逐张量|完全一致|边界"):
        v3.repair_task09_v3_terminal_boundary(output)


def test_v3_session_receipts_are_exclusive_contiguous_and_bind_current_chain_head(tmp_path):
    v3, v1 = _v3(), _v1()
    output = tmp_path / "run"
    output.mkdir()
    identity = ("primary", 6, 1)
    registry_sha, archive_sha, ledger_sha = "b" * 64, "c" * 64, "d" * 64

    def commit_files(end):
        (output / "阶段_最近.pt").write_bytes(f"recent-{end}".encode())
        (output / "training.jsonl").write_bytes(f"log-{end}".encode())
        (output / "任09F3阶段报告.json").write_bytes(f"v1-report-{end}".encode())
        (output / "任09V3阶段门禁报告.json").write_bytes(f"v3-report-{end}".encode())

    first = _payload(stage=v1.CORRECTION_STAGE, correction=200, joint=0)
    commit_files(200)
    receipt1 = v3.write_task09_v3_session_receipt(
        output, identity=identity, registry_sha256=registry_sha,
        archive_sha256=archive_sha, ledger_row_sha256=ledger_sha,
        start_global_epoch=0, requested_epochs=200, payload=first,
    )
    assert receipt1["会话序号"] == 1
    assert receipt1["实际新增轮次"] == 200
    with pytest.raises(FileExistsError, match="覆盖|已存在|收据"):
        v3.write_task09_v3_session_receipt(
            output, identity=identity, registry_sha256=registry_sha,
            archive_sha256=archive_sha, ledger_row_sha256=ledger_sha,
            start_global_epoch=0, requested_epochs=200, payload=first,
        )

    second = _payload(stage=v1.CORRECTION_STAGE, correction=400, joint=0)
    commit_files(400)
    receipt2 = v3.write_task09_v3_session_receipt(
        output, identity=identity, registry_sha256=registry_sha,
        archive_sha256=archive_sha, ledger_row_sha256=ledger_sha,
        start_global_epoch=200, requested_epochs=200, payload=second,
    )
    assert receipt2["会话序号"] == 2

    third = _payload(stage=v1.JOINT_STAGE, correction=400, joint=200)
    commit_files(600)
    __import__("torch").save(copy.deepcopy(second), output / "阶段_校正末.pt")
    third["metadata"].update({
        "校正实际截止轮次": 400,
        "已提交真实校正末态SHA256": sha256_file(output / "阶段_校正末.pt"),
    })
    for name in ("阶段_联合末.pt", "阶段_训练末.pt"):
        __import__("torch").save(copy.deepcopy(third), output / name)
    receipt3 = v3.write_task09_v3_session_receipt(
        output, identity=identity, registry_sha256=registry_sha,
        archive_sha256=archive_sha, ledger_row_sha256=ledger_sha,
        start_global_epoch=400, requested_epochs=200, payload=third,
    )
    assert set(receipt3["会话末终态SHA256"]) == {
        "阶段_校正末.pt", "阶段_联合末.pt", "阶段_训练末.pt",
    }
    chain = v3.validate_task09_v3_receipt_chain(
        output, identity=identity, registry_sha256=registry_sha,
        archive_sha256=archive_sha, ledger_row_sha256=ledger_sha,
        payload=third,
    )
    assert chain["会话收据数"] == 3
    assert chain["链头完整全局轮次"] == 600

    first_path = output / "任09V3会话收据_0001.json"
    original = first_path.read_bytes()
    first_path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="链|SHA|收据"):
        v3.validate_task09_v3_receipt_chain(
            output, identity=identity, registry_sha256=registry_sha,
            archive_sha256=archive_sha, ledger_row_sha256=ledger_sha,
            payload=third,
        )


def test_v3_receipt_rejects_arbitrary_short_session_without_real_terminal(tmp_path):
    v3, v1 = _v3(), _v1()
    output = tmp_path / "short"
    output.mkdir()
    for name in ("阶段_最近.pt", "training.jsonl", "任09F3阶段报告.json",
                 "任09V3阶段门禁报告.json"):
        (output / name).write_bytes(name.encode("utf-8"))
    payload = _payload(stage=v1.CORRECTION_STAGE, correction=50, joint=0)
    arguments = dict(
        output_directory=output, identity=("primary", 6, 1),
        registry_sha256="b" * 64, archive_sha256="c" * 64,
        ledger_row_sha256="d" * 64, start_global_epoch=0, payload=payload,
    )
    with pytest.raises(ValueError, match="短|200|截止|末态"):
        v3.write_task09_v3_session_receipt(requested_epochs=200, **arguments)
    payload["metadata"]["校正实际轮次"] = 10
    payload["metadata"]["完整全局实际轮次"] = 10
    payload["epoch"] = 10
    with pytest.raises(ValueError, match="短|200|截止|末态"):
        v3.write_task09_v3_session_receipt(requested_epochs=10, **arguments)


def test_v3_copy_exact_binds_prechecked_sha_and_rejects_toctou_change(tmp_path):
    v3 = _v3()
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"prechecked")
    expected = sha256_file(source)
    source.write_bytes(b"changed after precheck")
    with pytest.raises(ValueError, match="事前|SHA|改变"):
        v3._copy_exact(source, destination, expected_sha256=expected)


def test_v3_qualification_never_calls_less_than_200_real_epochs_a_formal_model():
    v3, v1 = _v3(), _v1()
    assert v3.task09_v3_minimum_qualification(
        _payload(stage=v1.CORRECTION_STAGE, correction=199, joint=0)
    )["达到正式最低200轮"] is False
    qualified = v3.task09_v3_minimum_qualification(
        _payload(stage=v1.CORRECTION_STAGE, correction=200, joint=0)
    )
    assert qualified["达到正式最低200轮"] is True
    assert qualified["可称五seed曲线"] is False
    assert qualified["可称能源完成"] is False


def test_v3_uses_frozen_v2_step_semantics_and_does_not_modify_frozen_files():
    v3, v2 = _v3(), _v2()
    assert v3.TASK09_V3_ADAMW_GATE is v2.assert_task09_v2_adamw_steps
    assert v3.FROZEN_DEPENDENCY_SHA256 == {
        "task09_subset_formal.py": "3e304eb4b2fd6c351375b7e4176ae344057b809c9c8394d62bc5c110cd0cf375",
        "task09_subset_formal_v2.py": "5173ab1df25340a5d583d455f321a81409097b4c9620541763683be3e663566d",
        "35_run_task09_subset_formal.py": "e7768f7414e5908483a37249406bbc8a98f31aee69b0b1e783e8464eeb9b8132",
        "42_run_task09_subset_formal_v2.py": "d8a41eb65a2d4ecb3f92b430606f128edf13d829dae845fae59bf5504df0a3f2",
        "task09_energy.py": "a9d3f2759794986f1135c1b2e251499e1d187bddb05f9b2944fd850bd24faaad",
        "38_audit_task09_energy.py": "9c77cdee4e1ed894180b3f2b881e326da802305a91ad609b940668044b61d213",
    }
    assert v3.verify_task09_v3_frozen_dependencies() == v3.FROZEN_DEPENDENCY_SHA256


def test_v3_runtime_scope_restores_v1_gate_and_archive_even_when_runner_fails(
    tmp_path, monkeypatch,
):
    v3, v1 = _v3(), _v1()
    original_gate = v1.assert_task09_adamw_steps
    calls = []

    def baseline_archive(output):
        calls.append(("v1", Path(output)))
        return {"old": "sha"}

    monkeypatch.setattr(v1, "_task09_archive_source", baseline_archive)
    monkeypatch.setattr(v3, "TASK09_V1_ARCHIVE_SOURCE", baseline_archive)
    original_validator = v1.validate_task09_output

    def v3_archive(output):
        calls.append(("v3", Path(output)))

    def fake_runner():
        assert v1.assert_task09_adamw_steps is v3.TASK09_V3_ADAMW_GATE
        assert v1._task09_archive_source is not baseline_archive
        assert v1.validate_task09_output is not original_validator
        assert v1._task09_archive_source(tmp_path) == {"old": "sha"}
        raise RuntimeError("synthetic runner stop")

    with pytest.raises(RuntimeError, match="synthetic"):
        v3.call_with_task09_v3_runtime(
            fake_runner, before_label_archive=v3_archive,
        )
    assert v1.assert_task09_adamw_steps is original_gate
    assert v1._task09_archive_source is baseline_archive
    assert v1.validate_task09_output is original_validator
    assert calls == [("v1", tmp_path), ("v3", tmp_path)]


def test_v3_root_gate_requires_unique_0078_line_with_yaml_and_tar_on_same_line(
    tmp_path, monkeypatch,
):
    v3 = _v3()
    ledger = tmp_path / "总台账.md"
    archive = tmp_path / "任09V3源码.tar.gz"
    archive.write_bytes(b"archive")
    archive_sha = sha256_file(archive)
    registry_sha = "b" * 64
    monkeypatch.setattr(v3, "TASK09_V3_LEDGER", ledger)
    monkeypatch.setattr(v3, "TASK09_V3_ARCHIVE_ROOT", tmp_path)
    ledger.write_text("| 录-0077 | other |\n", encoding="utf-8")
    with pytest.raises(ValueError, match="0078|台账|登记"):
        v3.require_task09_v3_root_ledger(
            registry_sha256=registry_sha, archive_path=archive,
            archive_sha256=archive_sha,
        )
    ledger.write_text(
        f"| 录-0078 | YAML SHA`{registry_sha}` |\n"
        f"| 附件 | tar {archive.name} SHA`{archive_sha}` |\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="同一行|双SHA|tar"):
        v3.require_task09_v3_root_ledger(
            registry_sha256=registry_sha, archive_path=archive,
            archive_sha256=archive_sha,
        )
    marker = (
        f"TASK09_V3_GATE:v1；status=active；"
        f"registry_sha256=`{registry_sha}`；archive_basename=`{archive.name}`；"
        f"archive_sha256=`{archive_sha}`"
    )
    for forged in (
        marker.replace("status=active", "status=disabled"),
        marker.replace("registry_sha256", "old_registry_sha256"),
        marker.replace(archive.name, archive.name + ".old"),
        marker.replace("registry_sha256", "archive_sha256", 1),
    ):
        ledger.write_text(f"| 录-0078 | {forged} |\n", encoding="utf-8")
        with pytest.raises(ValueError, match="机器标记|字段|双SHA|tar"):
            v3.require_task09_v3_root_ledger(
                registry_sha256=registry_sha, archive_path=archive,
                archive_sha256=archive_sha,
            )
    ledger.write_text(f"| 录-0078 | {marker} |\n", encoding="utf-8")
    receipt = v3.require_task09_v3_root_ledger(
        registry_sha256=registry_sha, archive_path=archive,
        archive_sha256=archive_sha,
    )
    assert receipt["ROOT主台账条目"] == "录-0078"
    assert receipt["录0078同行双SHA"] is True


def test_v3_fresh_200_epoch_session_is_forwarded_and_failure_artifacts_are_preserved(
    tmp_path, monkeypatch,
):
    v3, v1 = _v3(), _v1()
    root = tmp_path / "正式F3子集训练"
    root.mkdir()
    output = root / "正式F3V3_primary_HF6_seed1_录0078_20260916T071500+0800"
    monkeypatch.setattr(v3, "TASK09_V3_FORMAL_ROOT", root)
    monkeypatch.setattr(v3, "require_task09_v3_registration", lambda *args: "b" * 64)
    monkeypatch.setattr(v3, "validate_task09_v3_source_archive", lambda *args: {})
    monkeypatch.setattr(v3, "verify_task09_v3_frozen_dependencies", lambda: {})
    monkeypatch.setattr(v3, "require_task09_v3_root_ledger", lambda **kwargs: {
        "录0078行SHA256": "d" * 64,
    })
    monkeypatch.setattr(v1, "require_task09_registration", lambda *args: v3.ORIGINAL_REGISTRY_SHA256)

    calls = []

    def fail_after_claiming_directory(**kwargs):
        calls.append(kwargs["session_epoch_limit"])
        Path(kwargs["output_directory"]).mkdir()
        (Path(kwargs["output_directory"]) / "失败现场.bin").write_bytes(b"keep exactly")
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(v1, "run_task09_formal", fail_after_claiming_directory)
    arguments = dict(
        name="primary", size=6, seed=1, output_directory=output,
        v3_registry_path=tmp_path / "registry.yaml", v3_registry_sha256="b" * 64,
        v3_source_archive=tmp_path / "sources.tar.gz",
        v3_source_archive_sha256="c" * 64,
        original_registry_path=tmp_path / "old.yaml",
        original_registry_sha256=v3.ORIGINAL_REGISTRY_SHA256,
        root_ledger_entry="录0078", session_epoch_limit=200,
        resume_checkpoint=None, device_name="cuda",
    )
    with pytest.raises(RuntimeError, match="synthetic"):
        v3.run_task09_formal_v3(**arguments)
    assert calls == [200]
    assert (output / "失败现场.bin").read_bytes() == b"keep exactly"
    with pytest.raises(FileExistsError, match="覆盖|已有|全新"):
        v3.run_task09_formal_v3(**arguments)
    assert calls == [200]

    other = root / "正式F3V3_primary_HF6_seed2_录0078_20260916T071500+0800"
    rejected = {**arguments, "seed": 2, "output_directory": other,
                "session_epoch_limit": 2000}
    with pytest.raises(ValueError, match="200|会话|上限"):
        v3.run_task09_formal_v3(**rejected)
    assert not other.exists()
    assert calls == [200]


def test_v3_source_archive_rejects_extra_link_or_changed_member(tmp_path, monkeypatch):
    v3 = _v3()
    content = b"v3 source"
    expected = {"src/v3.py": hashlib.sha256(content).hexdigest()}
    monkeypatch.setattr(v3, "TASK09_V3_ARCHIVE_ROOT", tmp_path)
    monkeypatch.setattr(v3, "expected_task09_v3_archive_members", lambda: expected)
    archive = tmp_path / "任09F3主序列剩余14模型v3源码冻结_test.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("src/v3.py")
        info.size = len(content)
        bundle.addfile(info, io.BytesIO(content))
    assert v3.validate_task09_v3_source_archive(archive, sha256_file(archive))[
        "普通文件成员数"
    ] == 1
    top_link = tmp_path / "任09F3主序列剩余14模型v3源码冻结_link.tar.gz"
    top_link.symlink_to(archive.name)
    with pytest.raises(ValueError, match="普通|链接|tar"):
        v3.validate_task09_v3_source_archive(top_link, sha256_file(archive))
    forged = tmp_path / "任09F3主序列剩余14模型v3源码冻结_forged.tar.gz"
    with tarfile.open(forged, "w:gz") as bundle:
        info = tarfile.TarInfo("src/v3.py")
        info.size = len(content)
        bundle.addfile(info, io.BytesIO(content))
        link = tarfile.TarInfo("src/link.py")
        link.type = tarfile.SYMTYPE
        link.linkname = "v3.py"
        bundle.addfile(link)
    with pytest.raises(ValueError, match="普通|成员|链接|额外"):
        v3.validate_task09_v3_source_archive(forged, sha256_file(forged))


def test_v3_registration_rejects_top_level_symlink_before_resolve(tmp_path, monkeypatch):
    v3 = _v3()
    real = tmp_path / "real.yaml"
    real.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "registration.yaml"
    link.symlink_to(real.name)
    monkeypatch.setattr(v3, "TASK09_V3_REGISTRY", real)
    monkeypatch.setattr(v3, "expected_task09_v3_registration", lambda: {})
    with pytest.raises(ValueError, match="原件|链接|登记"):
        v3.require_task09_v3_registration(link, sha256_file(real))


def test_v3_real_entry_checks_0078_before_any_cuda_probe(monkeypatch, tmp_path):
    v3, v1 = _v3(), _v1()
    formal_root = tmp_path / "正式F3子集训练"
    formal_root.mkdir()
    monkeypatch.setattr(v3, "TASK09_V3_FORMAL_ROOT", formal_root)
    monkeypatch.setattr(v3, "require_task09_v3_registration", lambda *args: "b" * 64)
    monkeypatch.setattr(v3, "validate_task09_v3_source_archive", lambda *args: {})
    monkeypatch.setattr(v3, "verify_task09_v3_frozen_dependencies", lambda: {})
    monkeypatch.setattr(v1, "require_task09_registration", lambda *args: v3.ORIGINAL_REGISTRY_SHA256)
    monkeypatch.setattr(
        v3, "require_task09_v3_root_ledger",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("录0078未登记")),
    )
    monkeypatch.setattr(
        v1, "require_task09_cuda",
        lambda *args: (_ for _ in ()).throw(AssertionError("CUDA must not be probed")),
    )
    with pytest.raises(ValueError, match="0078"):
        v3.run_task09_formal_v3(
            name="primary", size=6, seed=1,
            output_directory=tmp_path / "never-created",
            v3_registry_path=tmp_path / "registry.yaml", v3_registry_sha256="b" * 64,
            v3_source_archive=tmp_path / "sources.tar.gz",
            v3_source_archive_sha256="c" * 64,
            original_registry_path=tmp_path / "old.yaml",
            original_registry_sha256=v3.ORIGINAL_REGISTRY_SHA256,
            root_ledger_entry="录0078", session_epoch_limit=200,
            resume_checkpoint=None, device_name="cuda",
        )
    assert not (tmp_path / "never-created").exists()


def test_v3_terminal_semantics_require_real_joint_stop_and_all_terminal_hashes():
    v3, v1 = _v3(), _v1()
    payload = _payload(stage=v1.JOINT_STAGE, correction=500, joint=300)
    payload["metadata"].update({
        "校正实际截止轮次": 500,
        "已提交真实校正末态SHA256": "a" * 64,
    })
    report = {
        "状态": "受限联合真实截止；待独立CUDA合法HF/LF及16/64阶能源审核",
        "运行种子": 4, "序列": "primary", "HF子集功率数": 9,
        "校正实际轮次": 500, "受限联合实际轮次": 300,
        "旧固定测试温度读取": False,
    }
    terminal_hashes = {
        "阶段_校正末.pt": "a" * 64,
        "阶段_联合末.pt": "b" * 64,
        "阶段_训练末.pt": "c" * 64,
    }
    assert v3.validate_task09_v3_terminal_semantics(
        payload, report, terminal_hashes=terminal_hashes,
        identity=("primary", 9, 4),
    )["真实联合终态"] is True
    bad_stage = copy.deepcopy(payload)
    bad_stage["stage"] = v1.CORRECTION_STAGE
    with pytest.raises(ValueError, match="联合|终态|stage"):
        v3.validate_task09_v3_terminal_semantics(
            bad_stage, report, terminal_hashes=terminal_hashes,
            identity=("primary", 9, 4),
        )
    for joint in (199, 201):
        bad = copy.deepcopy(payload)
        bad["metadata"]["联合实际轮次"] = joint
        bad["metadata"]["完整全局实际轮次"] = 500 + joint
        bad["epoch"] = joint
        with pytest.raises(ValueError, match="200|十轮|联合"):
            v3.validate_task09_v3_terminal_semantics(
                bad, {**report, "受限联合实际轮次": joint},
                terminal_hashes=terminal_hashes, identity=("primary", 9, 4),
            )
    with pytest.raises(ValueError, match="截止|报告|状态"):
        v3.validate_task09_v3_terminal_semantics(
            payload, {**report, "状态": "正式阶段完整会话暂停"},
            terminal_hashes=terminal_hashes, identity=("primary", 9, 4),
        )
    with pytest.raises(ValueError, match="末态|SHA|校正"):
        v3.validate_task09_v3_terminal_semantics(
            payload, report,
            terminal_hashes={"阶段_联合末.pt": "b" * 64,
                             "阶段_训练末.pt": "c" * 64},
            identity=("primary", 9, 4),
        )


def test_v3_downstream_energy_or_final_gate_requires_manifest_chain_and_terminal_states(
    tmp_path, monkeypatch,
):
    v3, v1 = _v3(), _v1()
    root = tmp_path / "正式F3子集训练"
    root.mkdir()
    output = root / "正式F3V3_primary_HF9_seed4_录0078_20260916T071500+0800"
    output.mkdir()
    payload = _payload(stage=v1.JOINT_STAGE, correction=500, joint=300)
    correction = _payload(stage=v1.CORRECTION_STAGE, correction=500, joint=0)
    __import__("torch").save(copy.deepcopy(correction), output / "阶段_校正末.pt")
    correction_sha = sha256_file(output / "阶段_校正末.pt")
    payload["metadata"].update({
        "校正实际截止轮次": 500,
        "已提交真实校正末态SHA256": correction_sha,
    })
    for name in ("阶段_最近.pt", "阶段_联合末.pt", "阶段_训练末.pt"):
        __import__("torch").save(copy.deepcopy(payload), output / name)
    (output / "任09F3阶段报告.json").write_text(json.dumps({
        "状态": "受限联合真实截止；待独立CUDA合法HF/LF及16/64阶能源审核",
        "运行种子": 4, "序列": "primary", "HF子集功率数": 9,
        "校正实际轮次": 500, "受限联合实际轮次": 300,
        "旧固定测试温度读取": False,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "training.jsonl").write_text(
        "".join(json.dumps({
            "epoch": epoch,
            "训练阶段": v1.CORRECTION_STAGE if epoch <= 500 else v1.JOINT_STAGE,
            "阶段实际轮次": epoch if epoch <= 500 else epoch - 500,
        }, ensure_ascii=False) + "\n" for epoch in range(1, 801)),
        encoding="utf-8",
    )
    monkeypatch.setattr(v3, "TASK09_V3_FORMAL_ROOT", root)
    monkeypatch.setattr(v3, "require_task09_v3_registration", lambda *args: "b" * 64)
    monkeypatch.setattr(v3, "verify_task09_v3_frozen_dependencies", lambda: {})
    monkeypatch.setattr(v3, "validate_task09_v3_source_archive", lambda *args: {})
    monkeypatch.setattr(v3, "require_task09_v3_root_ledger", lambda **kwargs: {
        "录0078行SHA256": "d" * 64,
    })
    arguments = dict(
        purpose="energy", name="primary", size=9, seed=4,
        output_directory=output, v3_registry_path=tmp_path / "registry.yaml",
        v3_registry_sha256="b" * 64, v3_source_archive=tmp_path / "sources.tar.gz",
        v3_source_archive_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="V3来源清单|谱系|manifest"):
        v3.require_task09_v3_downstream_qualification(**arguments)

    calls = []
    monkeypatch.setattr(v3, "_validate_task09_v3_lineage",
                        lambda *args, **kwargs: calls.append("manifest") or {})
    monkeypatch.setattr(v3, "validate_task09_v3_receipt_chain",
                        lambda *args, **kwargs: calls.append("chain") or {
                            "会话收据数": 4,
                            "链头终态SHA256": {
                                name: sha256_file(output / name) for name in (
                                    "阶段_校正末.pt", "阶段_联合末.pt", "阶段_训练末.pt"
                                )
                            },
                        })
    monkeypatch.setattr(v3, "_preflight_task09_v3_payload",
                        lambda *args, **kwargs: calls.append("checkpoint"))
    monkeypatch.setattr(v3, "_audit_task09_v3_committed_state",
                        lambda *args, **kwargs: calls.append("audit") or {"完整已提交轮次": 800})
    result = v3.require_task09_v3_downstream_qualification(**arguments)
    assert result["V3来源谱系与会话链合格"] is True
    assert result["旧v1或pilot产物可进入下游"] is False
    assert result["真实联合终态"]["真实联合终态"] is True
    assert json.loads(json.dumps(result, ensure_ascii=False))["真实联合终态"]["真实联合终态"] is True
    assert calls == ["manifest", "chain", "checkpoint", "audit"]


def test_v3_energy_wrapper_rejects_bad_lineage_before_energy_cuda_or_output(
    tmp_path, monkeypatch,
):
    v3 = _v3()
    energy = importlib.import_module("sic_cu.eval.task09_energy")
    destination = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        v3, "require_task09_v3_downstream_qualification",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("坏V3会话收据链")),
    )
    monkeypatch.setattr(
        energy, "task09_energy_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("output probe reached")),
    )
    monkeypatch.setattr(
        energy, "audit_task09_formal_energy",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("CUDA audit reached")),
    )
    with pytest.raises(ValueError, match="V3|收据链"):
        v3.audit_task09_v3_formal_energy(
            run=tmp_path / "fake-v1-run", name="primary", size=6, seed=1,
            state="best", output=destination, device_name="cuda",
            v3_registry_path=tmp_path / "v3.yaml", v3_registry_sha256="b" * 64,
            v3_source_archive=tmp_path / "v3.tar.gz",
            v3_source_archive_sha256="c" * 64,
            original_registry_path=tmp_path / "v1.yaml",
            original_registry_sha256=v3.ORIGINAL_REGISTRY_SHA256,
        )
    assert not destination.exists()


def test_v3_energy_wrapper_checks_original_0059_before_frozen_cuda(monkeypatch, tmp_path):
    v3, v1 = _v3(), _v1()
    energy = importlib.import_module("sic_cu.eval.task09_energy")
    destination = tmp_path / "must-not-exist"
    monkeypatch.setattr(v3, "require_task09_v3_downstream_qualification",
                        lambda **kwargs: {"V3来源谱系与会话链合格": True})
    monkeypatch.setattr(
        v1, "require_task09_registration",
        lambda *args: (_ for _ in ()).throw(ValueError("原0059登记SHA错误")),
    )
    monkeypatch.setattr(
        energy, "audit_task09_formal_energy",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("CUDA audit reached")),
    )
    with pytest.raises(ValueError, match="0059|登记|SHA"):
        v3.audit_task09_v3_formal_energy(
            run=tmp_path / "run", name="primary", size=6, seed=1,
            state="best", output=destination, device_name="cuda",
            v3_registry_path=tmp_path / "v3.yaml", v3_registry_sha256="b" * 64,
            v3_source_archive=tmp_path / "v3.tar.gz",
            v3_source_archive_sha256="c" * 64,
            original_registry_path=tmp_path / "wrong.yaml",
            original_registry_sha256="e" * 64,
        )
    assert not destination.exists()


def test_v3_energy_wrapper_writes_exclusive_provenance_receipt_after_frozen_audit(
    tmp_path, monkeypatch,
):
    v3 = _v3()
    energy = importlib.import_module("sic_cu.eval.task09_energy")
    run = tmp_path / "run"
    run.mkdir()
    destination = run / "energy"
    qualification = {"V3来源谱系与会话链合格": True, "会话链": {"会话收据数": 4}}
    calls = []
    original_energy_validator = energy.validate_task09_output
    monkeypatch.setattr(
        v3, "require_task09_v3_downstream_qualification",
        lambda **kwargs: calls.append("gate") or copy.deepcopy(qualification),
    )
    monkeypatch.setattr(energy, "task09_energy_output", lambda *args, **kwargs: destination)
    monkeypatch.setattr(_v1(), "require_task09_registration",
                        lambda *args: v3.ORIGINAL_REGISTRY_SHA256)

    def fake_audit(*args, **kwargs):
        calls.append("energy")
        assert energy.validate_task09_output is not original_energy_validator
        destination.mkdir()
        (destination / "汇总指标.json").write_text("{}\n", encoding="utf-8")
        return {"审核状态": kwargs["state"]}

    monkeypatch.setattr(energy, "audit_task09_formal_energy", fake_audit)
    result = v3.audit_task09_v3_formal_energy(
        run=run, name="primary", size=6, seed=1, state="best",
        output=destination, device_name="cuda",
        v3_registry_path=tmp_path / "v3.yaml", v3_registry_sha256="b" * 64,
        v3_source_archive=tmp_path / "v3.tar.gz",
        v3_source_archive_sha256="c" * 64,
        original_registry_path=tmp_path / "v1.yaml",
        original_registry_sha256=v3.ORIGINAL_REGISTRY_SHA256,
    )
    receipt = destination / "任09V3能源入口收据.json"
    assert calls == ["gate", "energy", "gate"]
    assert energy.validate_task09_output is original_energy_validator
    assert receipt.is_file()
    assert result["V3能源入口收据SHA256"] == sha256_file(receipt)
    with pytest.raises(FileExistsError, match="收据|覆盖|已存在"):
        v3.audit_task09_v3_formal_energy(
            run=run, name="primary", size=6, seed=1, state="best",
            output=destination, device_name="cuda",
            v3_registry_path=tmp_path / "v3.yaml", v3_registry_sha256="b" * 64,
            v3_source_archive=tmp_path / "v3.tar.gz",
            v3_source_archive_sha256="c" * 64,
            original_registry_path=tmp_path / "v1.yaml",
            original_registry_sha256=v3.ORIGINAL_REGISTRY_SHA256,
        )


def test_v3_registration_binds_14_models_budget_sources_and_no_test_temperatures():
    v3 = _v3()
    expected = v3.expected_task09_v3_registration()
    assert expected["schema_version"] == 1
    assert expected["任务"] == "任09主序列剩余14模型通用正式v3入口"
    assert expected["ROOT主台账预留条目"] == "录0078"
    assert expected["剩余正式身份数"] == 14
    assert expected["排除已完成身份"] == "primary/HF3/seed0"
    assert expected["首次正式会话最低请求轮次"] == 200
    assert expected["单次正式会话硬上限轮次"] == 200
    assert "LOCK_EX_NONBLOCK" in expected["同canonical跨进程排他"]
    assert expected["校正_联合上限轮次"] == [1500, 500]
    assert expected["联合每轮LF真实回放batch_每批点"] == [60, 2048]
    assert expected["旧固定TEST温度读取"] is False
    assert expected["旧v1或五个pilot产物资格"] is False
    assert "另行事前登记恢复" in expected["失败孤儿目录处理"]
    assert expected["能源与最终资格入口"].startswith("必须先核V3 manifest")
    assert len({item["项目内唯一固定canonical目录"]
                for item in expected["剩余正式身份"]}) == 14
    assert expected["本登记已激活GPU"] is False
    assert expected["本登记已训练模型数"] == 0
    assert set(expected["新v3源码SHA256"]) == {
        "trainer_v3", "CLI_v3", "energy_CLI_v3", "registration_generator_v3",
        "tests_v3",
    }
    generator = PROJECT_ROOT / "scripts/45_register_task09_subset_formal_v3.py"
    assert generator.is_file()
    completed = subprocess.run(
        [sys.executable, str(generator), "--dry-run"],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert yaml.safe_load(completed.stdout) == expected


def test_v3_cli_is_cuda_only_and_exposes_fresh_or_clean_resume_contract():
    script = PROJECT_ROOT / "scripts/44_run_task09_subset_formal_v3.py"
    assert script.is_file()
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--device {cuda}" in completed.stdout
    assert "--resume-checkpoint" in completed.stdout
    assert "--root-ledger-entry {录0078}" in completed.stdout
    assert "--session-epoch-limit" in completed.stdout

    energy_cli = PROJECT_ROOT / "scripts/46_audit_task09_energy_v3.py"
    assert energy_cli.is_file()
    energy_help = subprocess.run(
        [sys.executable, str(energy_cli), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert energy_help.returncode == 0, energy_help.stderr
    for option in (
        "--v3-registry", "--v3-registry-sha256", "--v3-source-archive",
        "--v3-source-archive-sha256", "--original-registry", "--device {cuda}",
    ):
        assert option in energy_help.stdout
