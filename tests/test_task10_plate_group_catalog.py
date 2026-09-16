"""任10真CUDA全组身份清单生成器的纯CPU测试。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from sic_cu.data.common import sha256_file
from sic_cu.eval import task10_plate_group_catalog as catalog


LF_KINDS = ("lf_same_physics_coarse", "lf_contact_mismatch_coarse")


@pytest.fixture
def project_tmp_path() -> Path:
    with tempfile.TemporaryDirectory(
        prefix="任10清单生成器单元测试_", dir=catalog.TASK10_ROOT,
    ) as name:
        yield Path(name)


def _write_receipt(batch: Path, *, arm: str, seed: int,
                   lf_source: str | None) -> Path:
    methods = ("F1",) if arm == "F1" else ("F2", "F3")
    artifacts = {}
    for method in methods:
        model_directory = batch / method
        model_directory.mkdir(parents=True)
        artifacts[method] = {"模型目录": str(model_directory)}
    receipt = batch / "真CUDA运行设备与封存原件收据.json"
    receipt.write_text(json.dumps({
        "schema_version": 1,
        "训练": {"arm": arm, "随机种子": seed, "LF来源": lf_source},
        "模型原件": artifacts,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt


def _complete_batches(tmp_path: Path) -> list[Path]:
    batches = []
    for seed in range(5):
        batch = tmp_path / f"F1_seed{seed}"
        _write_receipt(batch, arm="F1", seed=seed, lf_source=None)
        batches.append(batch)
    for lf_source in LF_KINDS:
        for seed in range(5):
            batch = tmp_path / f"PAIR_{lf_source}_seed{seed}"
            _write_receipt(batch, arm="PAIR", seed=seed, lf_source=lf_source)
            batches.append(batch)
    return batches


def _fake_cuda_proof(identities: list[dict]) -> dict:
    assert len(identities) == 25
    receipt_shas = {item["CUDA运行收据_SHA256"] for item in identities}
    assert len(receipt_shas) == 15
    reports = []
    for index, identity in enumerate(identities):
        reports.append({
            "方法": identity["方法"],
            "随机种子": identity["随机种子"],
            "LF来源": identity["LF来源"],
            "最佳模型_SHA256": f"{index + 1:064x}",
            "终态模型_SHA256": f"{index + 101:064x}",
        })
    return {"身份总数": 25, "真CUDA运行批次总数": 15, "逐身份": reports,
            "完整HF温度已读取": False}


def test_incomplete_batch_set_is_rejected_without_output(project_tmp_path: Path) -> None:
    batch = project_tmp_path / "only_one_batch"
    _write_receipt(batch, arm="F1", seed=0, lf_source=None)
    output = project_tmp_path / "不得生成.json"

    with pytest.raises(PermissionError, match="15|批次"):
        catalog.build_cuda_group_catalog(
            batch_directories=[batch], output_path=output,
            cuda_budget_sha256="a" * 64, cuda_tar_sha256="b" * 64,
        )

    assert not output.exists()


def test_complete_batches_generate_deterministic_25_identity_catalog(
    project_tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = _complete_batches(project_tmp_path)
    output = project_tmp_path / "任10真CUDA全25身份机读清单.json"
    seen = {}
    monkeypatch.setattr(catalog, "verify_cuda_source_freeze", lambda **_kwargs: {
        "CUDA资格预算_SHA256": "a" * 64,
        "CUDA资格十二源tar_SHA256": "b" * 64,
        "完整HF温度已读取": False,
    })
    monkeypatch.setattr(catalog, "verify_25_cuda_evidence", _fake_cuda_proof)

    def preflight(payload: bytes, digest: str) -> dict:
        seen["payload"] = payload
        seen["digest"] = digest
        return {"身份总数": 25, "只读完整HF": False}

    monkeypatch.setattr(catalog, "_preflight_complete_group_without_root", preflight)
    result = catalog.build_cuda_group_catalog(
        batch_directories=list(reversed(batches)), output_path=output,
        cuda_budget_sha256="a" * 64, cuda_tar_sha256="b" * 64,
    )

    group = json.loads(output.read_text(encoding="utf-8"))
    assert len(group["身份"]) == 25
    assert len(group["批次"]) == 15
    assert group["CUDA资格预算_SHA256"] == "a" * 64
    assert group["CUDA资格源码tar_SHA256"] == "b" * 64
    assert group["完整HF温度已读取"] is False
    assert group["身份"][0]["方法"] == "F1"
    assert group["身份"][-1]["方法"] == "F3"
    assert all(Path(item["模型目录"]).is_absolute() for item in group["身份"])
    assert all(Path(item["CUDA运行收据文件"]).is_absolute()
               for item in group["身份"])
    assert all(Path(item["CUDA运行收据文件"]).name ==
               "真CUDA运行设备与封存原件收据.json" for item in group["身份"])
    assert all(len(item["CUDA运行收据_SHA256"]) == 64 for item in group["身份"])
    assert all(len(item["最佳模型_SHA256"]) == 64 for item in group["身份"])
    assert all(len(item["终态模型_SHA256"]) == 64 for item in group["身份"])
    assert set(group["清单生成器源码_SHA256"]) == {
        "src/sic_cu/eval/task10_plate_group_catalog.py",
        "scripts/50_prepare_task10_cuda_group_catalog.py",
    }
    assert seen["payload"] == output.read_bytes()
    assert seen["digest"] == sha256_file(output)
    assert result["全组身份文件_SHA256"] == sha256_file(output)
    assert result["身份总数"] == 25
    assert result["真CUDA运行批次总数"] == 15


def test_failed_full_group_preflight_leaves_no_catalog(
    project_tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = _complete_batches(project_tmp_path)
    output = project_tmp_path / "预检失败不得落盘.json"
    monkeypatch.setattr(catalog, "verify_cuda_source_freeze", lambda **_kwargs: {})
    monkeypatch.setattr(catalog, "verify_25_cuda_evidence", _fake_cuda_proof)
    monkeypatch.setattr(
        catalog, "_preflight_complete_group_without_root",
        lambda *_args: (_ for _ in ()).throw(PermissionError("800步组门禁失败")),
    )

    with pytest.raises(PermissionError, match="组门禁失败"):
        catalog.build_cuda_group_catalog(
            batch_directories=batches, output_path=output,
            cuda_budget_sha256="a" * 64, cuda_tar_sha256="b" * 64,
        )

    assert not output.exists()


def test_catalog_sha_must_enter_root_before_formal_group_gate(
    project_tmp_path: Path,
) -> None:
    from sic_cu.eval.task10_plate_group_gate import verify_complete_group

    group = project_tmp_path / "尚未登记ROOT的全组清单.json"
    group.write_text("{}\n", encoding="utf-8")
    ledger = project_tmp_path / "不含清单SHA的非正式账本.md"
    ledger.write_text("本文件不属于正式ROOT，且不含清单SHA。\n", encoding="utf-8")

    with pytest.raises(PermissionError, match="总台账|另行进入"):
        verify_complete_group(
            group,
            group_sha256=sha256_file(group),
            registration_path=catalog.REGISTRATION,
            archive_root=catalog.SOURCE_ROOT,
            budget_path=catalog.METHOD_BUDGET,
            budget_sha256=catalog.METHOD_BUDGET_SHA256,
            ledger_path=ledger,
        )
