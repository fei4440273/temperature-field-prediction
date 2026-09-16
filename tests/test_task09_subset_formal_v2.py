"""Task-09 P1 repair: frozen-LF AdamW accounting and immutable-tail recovery."""

from __future__ import annotations

import copy
import importlib
import json
import io
from pathlib import Path
import tarfile

import pytest
import torch
from torch import nn
import yaml

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file


FAILED_RUN = PROJECT_ROOT / (
    "研究记录/任务09_高保真数据效率/正式F3子集训练/"
    "正式F3_primary_HF3_seed0_20260916T060920+0800"
)
EXPECTED_FAILED_SHAS = {
    "阶段_最近.pt": "4b7d110794046b242d4275cb795d7befcf7244bad42faf1104c883c982c33ac6",
    "阶段_联合初始.pt": "4b7d110794046b242d4275cb795d7befcf7244bad42faf1104c883c982c33ac6",
    "阶段_校正末.pt": "391f21c7b174bd5d358425b88a32b3b444b41774a78fff304d4c1049233419f8",
    "阶段_观测最佳.pt": "1089889f4e96d2e428e22bbc20e8eb27002f1570c632d2aaee9e9919c747431b",
    "阶段_物理最佳.pt": "fd1e59f18c204e82f8e97dfdb20f97d1cfe2ba58b38de602884a18a9494435ae",
    "best.pt": "4c361df61ac3c622b8f79ea1e72ff8c8e725573e7d03137c1b2af4e5536f90d4",
    "training.jsonl": "e87f207ce3df3ca4fa5870fa75c80e0940337d13f5d225c59df62f3fea17222d",
    "任09F3阶段报告.json": "17217eb79917c40a8040914c2fb581d212ee414ca3b17c8d26e5da71864961f3",
    "源码与登记事前快照SHA256.json": (
        "bc907bce480db1a910021d29c61cbc54fbde2b93b480b077c7cf49b4d203e000"
    ),
}


def _v1():
    return importlib.import_module("sic_cu.train.task09_subset_formal")


def _v2():
    try:
        return importlib.import_module("sic_cu.train.task09_subset_formal_v2")
    except ModuleNotFoundError:
        pytest.fail("Task-09 P1 revised trainer is missing")


class _TinyLf(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.branch_projection = nn.Linear(1, 1)
        self.trunk_projection = nn.Linear(1, 1)
        self.hidden = nn.Linear(1, 1)
        self.register_buffer("normalizer", torch.ones(1))
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class _TinyF3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.low_fidelity_model = _TinyLf()
        self.correction = nn.Sequential(*[nn.Linear(1, 1) for _ in range(5)])


def _stepped_joint_optimizer():
    formal = _v1()
    model = _TinyF3()
    named = dict(model.named_parameters())
    correction = [parameter for name, parameter in named.items()
                  if name.startswith("correction.")]
    projections = [parameter for name, parameter in named.items()
                   if name in formal.PROJECTION_NAMES]
    for parameter in correction + projections:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {"params": correction, "lr": 1e-4},
        {"params": projections, "lr": 1e-5},
    ])
    # hf_batches=4, correction=1, joint=1: correction step=10, projection step=5.
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        sum(parameter.sum() for parameter in correction).backward()
        optimizer.step()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        sum(parameter.sum() for parameter in correction + projections).backward()
        optimizer.step()
    return model, optimizer, named


def _joint_tail_fixture():
    formal = _v1()
    sample = {
        "powers_w": [55.0, 364.3, 729.0],
        "ir_rows": 6868,
        "ir_batches_2048": 4,
        "sensor_rows": {"Hot": 340, "Cold": 340},
    }
    committed = {
        "当前阶段": formal.JOINT_STAGE,
        "校正实际轮次": 500,
        "联合实际轮次": 0,
        "完整全局实际轮次": 500,
        "校正实际截止轮次": 500,
        "累计实际消耗": formal.task09_expected_consumption(sample, 500, 0),
    }
    actual = {
        "HF训练观测点": 6868,
        "HF训练传感器点": 2720,
        "HF观测优化步": 4,
        "物理优化步": 1,
        "物理配点": 256,
        "LF真实回放训练点": 60 * 2048,
        "LF联合回放batch": 60,
        "LF_Cu真实回放点": 61500,
        "LF_SiC真实回放点": 61380,
        "LF60每功率真2048已核验": True,
        "LF每HF步真实分组": [15, 15, 15, 15],
        "每LF2048小批损失名义系数": 0.25,
    }
    tail = {
        "epoch": 501,
        "运行种子": 0,
        "运行臂": "F3",
        "真实HF子集功率": list(sample["powers_w"]),
        "训练阶段": formal.JOINT_STAGE,
        "阶段实际轮次": 1,
        "HF合法验证选分_摄氏度": None,
        "HF合法验证分模态_摄氏度": None,
        "LF合法验证逐材料节点与体积": None,
        "LF两材料真实节点与体积5%资格": None,
        "当前本seed LF整张量SHA256": "1" * 64,
        "独立名义物理损失": None,
        "累计实际消耗": formal.task09_expected_consumption(sample, 500, 1),
        **actual,
    }
    return sample, committed, tail


def test_red_old_validator_mislabels_frozen_lf_bias_but_v2_accepts_only_four_projections():
    formal, repaired = _v1(), _v2()
    _, optimizer, named = _stepped_joint_optimizer()
    with pytest.raises(
        ValueError,
        match=r"low_fidelity_model\.bias真实AdamW step=0，期望5",
    ):
        formal.assert_task09_adamw_steps(
            optimizer, named, hf_batches=4, correction_epochs=1, joint_epochs=1,
        )
    checked = repaired.assert_task09_v2_adamw_steps(
        optimizer, named, hf_batches=4, correction_epochs=1, joint_epochs=1,
    )
    assert checked["low_fidelity_model.bias"] == 0
    assert {checked[name] for name in formal.PROJECTION_NAMES} == {5}
    assert all(checked[name] == 0 for name in checked
               if name.startswith("low_fidelity_model.")
               and name not in formal.PROJECTION_NAMES)


def test_v2_adamw_gate_rejects_wrong_projection_step_and_any_frozen_lf_state():
    formal, repaired = _v1(), _v2()
    model, optimizer, named = _stepped_joint_optimizer()
    projection = named[sorted(formal.PROJECTION_NAMES)[0]]
    optimizer.state[projection]["step"] -= 1
    with pytest.raises(ValueError, match="投影|AdamW|step"):
        repaired.assert_task09_v2_adamw_steps(
            optimizer, named, hf_batches=4, correction_epochs=1, joint_epochs=1,
        )
    optimizer.state[projection]["step"] += 1
    frozen = model.low_fidelity_model.bias
    optimizer.state[frozen] = {
        "step": torch.tensor(5.0),
        "exp_avg": torch.zeros_like(frozen),
        "exp_avg_sq": torch.zeros_like(frozen),
    }
    with pytest.raises(ValueError, match="冻结|LF|step|AdamW"):
        repaired.assert_task09_v2_adamw_steps(
            optimizer, named, hf_batches=4, correction_epochs=1, joint_epochs=1,
        )
    _, optimizer_without_state, named_without_state = _stepped_joint_optimizer()
    frozen_without_state = named_without_state["low_fidelity_model.bias"]
    optimizer_without_state.add_param_group({"params": [frozen_without_state], "lr": 1e-5})
    assert optimizer_without_state.state.get(frozen_without_state, {}) == {}
    with pytest.raises(ValueError, match="冻结|LF|参数组|AdamW"):
        repaired.assert_task09_v2_adamw_steps(
            optimizer_without_state, named_without_state,
            hf_batches=4, correction_epochs=1, joint_epochs=1,
        )


def test_single_uncommitted_joint_tail_contract_and_all_malicious_variants():
    formal, repaired = _v1(), _v2()
    sample, committed, tail = _joint_tail_fixture()
    accepted = repaired.validate_task09_uncommitted_tail(
        [tail], committed_metadata=committed, sample=sample,
        seed=0, powers=sample["powers_w"],
    )
    assert accepted == tail
    with pytest.raises(ValueError, match="唯一|一条|尾"):
        repaired.validate_task09_uncommitted_tail(
            [tail, dict(tail)], committed_metadata=committed, sample=sample,
            seed=0, powers=sample["powers_w"],
        )
    with pytest.raises(ValueError, match="联合|阶段|尾"):
        repaired.validate_task09_uncommitted_tail(
            [dict(tail, 训练阶段=formal.CORRECTION_STAGE)],
            committed_metadata=committed, sample=sample,
            seed=0, powers=sample["powers_w"],
        )
    for field, value in (
        ("epoch", 502), ("阶段实际轮次", 2), ("运行种子", 1),
        ("真实HF子集功率", [55.0, 364.3, 700.0]),
        ("HF合法验证选分_摄氏度", 1.0),
    ):
        with pytest.raises(ValueError, match="同seed|子集|joint1|阶段|验证|伪造"):
            repaired.validate_task09_uncommitted_tail(
                [dict(tail, **{field: value})],
                committed_metadata=committed, sample=sample,
                seed=0, powers=sample["powers_w"],
            )
    for field, value in (
        ("LF联合回放batch", 59),
        ("LF真实回放训练点", 59 * 2048),
        ("LF_Cu真实回放点", 61499),
        ("LF每HF步真实分组", [15, 15, 15, 14]),
        ("每LF2048小批损失名义系数", 1.0),
        ("累计实际消耗", formal.task09_expected_consumption(sample, 500, 0)),
    ):
        with pytest.raises(ValueError, match="LF|消费|预算|尾|累计"):
            repaired.validate_task09_uncommitted_tail(
                [dict(tail, **{field: value})],
                committed_metadata=committed, sample=sample,
                seed=0, powers=sample["powers_w"],
            )


def test_recovery_log_parser_rejects_bad_json_blank_or_multiple_tails():
    repaired = _v2()
    committed = b'{"epoch": 1}\n'
    tail = b'{"epoch": 2}\n'
    assert repaired.parse_task09_recovery_log(
        committed + tail, committed_rows=1,
    )[1] == [{"epoch": 2}]
    for forged in (
        committed + b'{bad json}\n',
        committed + b'\n',
        committed + tail + b'{"epoch": 3}\n',
        committed + b'{"epoch": 2}',
    ):
        with pytest.raises(ValueError, match="JSON|空|尾|完整|唯一"):
            repaired.parse_task09_recovery_log(forged, committed_rows=1)


def test_actual_failed_source_is_exact_corr500_plus_one_valid_uncommitted_joint1_tail():
    repaired = _v2()
    before = {name: sha256_file(FAILED_RUN / name) for name in EXPECTED_FAILED_SHAS}
    assert before == EXPECTED_FAILED_SHAS
    audit = repaired.audit_task09_failed_recovery_source(FAILED_RUN)
    assert audit["已提交校正轮次"] == 500
    assert audit["已提交联合轮次"] == 0
    assert audit["未提交尾行数"] == 1
    assert audit["未提交尾行全局轮次"] == 501
    assert audit["未提交尾行联合轮次"] == 1
    assert audit["未提交尾行LF回放batch"] == 60
    assert audit["未提交尾行LF回放点"] == 60 * 2048
    assert audit["未提交尾行不作已提交状态"] is True
    assert audit["失败现场完整日志SHA256"] == EXPECTED_FAILED_SHAS["training.jsonl"]
    assert audit["已提交前500行SHA256"] == (
        "b9477403ddc73d599153ddfca563e4cca575cf4fc3e6b7d07fcd1f8e37f29145"
    )
    assert audit["未提交尾行SHA256"] == (
        "502f82c00b3750fc9041224977587095f91169936dbd8243e6f665f7c0b78cbf"
    )
    assert {name: sha256_file(FAILED_RUN / name) for name in EXPECTED_FAILED_SHAS} == before


def test_actual_committed_checkpoint_rejects_tampered_frozen_lf_parameter_or_buffer():
    repaired, formal = _v2(), _v1()
    payload = torch.load(FAILED_RUN / "阶段_最近.pt", map_location="cpu", weights_only=False)
    source = formal.load_task09_sources()[0]
    sample = {
        **formal.audit_task09_observations(formal.load_task09_registry())["arms"]["primary"][3],
        "arm": "primary",
    }
    registry_sha = payload["metadata"]["正式F3事前登记SHA256"]
    repaired.check_task09_v2_resume_payload(
        payload, source, sample, registry_sha256=registry_sha,
    )
    parameter_names = {name.removeprefix("low_fidelity_model.")
                       for name, _ in formal.fork_task07_initialization(
                           formal.load_task09_sources(), 0, "E0", torch.device("cpu")
                       ).model.named_parameters()
                       if name.startswith("low_fidelity_model.")}
    candidates = [name for name in source.lf_state if name not in parameter_names]
    assert candidates, "the source model must expose at least one LF buffer"
    for lf_name in ("bias", candidates[0]):
        forged = dict(payload, model_state=dict(payload["model_state"]))
        key = f"low_fidelity_model.{lf_name}"
        forged["model_state"][key] = payload["model_state"][key].clone()
        forged["model_state"][key].view(-1)[0] += 1
        with pytest.raises(ValueError, match="LF|冻结|张量|来源"):
            repaired.check_task09_v2_resume_payload(
                forged, source, sample, registry_sha256=registry_sha,
            )


def test_recovery_stages_only_committed_prefix_in_new_directory_and_preserves_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    repaired = _v2()
    destination = tmp_path / "正式F3_primary_HF3_seed0_20260916T090000+0800"
    before = {name: sha256_file(FAILED_RUN / name) for name in EXPECTED_FAILED_SHAS}
    monkeypatch.setattr(
        repaired._v1, "validate_task09_output",
        lambda output, name, size, seed: Path(output).resolve(),
    )
    manifest = repaired.stage_task09_v2_recovery(
        FAILED_RUN, destination, name="primary", size=3, seed=0,
        v2_registry_path=repaired.TASK09_V2_REGISTRY,
        v2_registry_sha256=sha256_file(repaired.TASK09_V2_REGISTRY),
    )
    assert destination.is_dir()
    assert sha256_file(destination / "training.jsonl") == (
        "b9477403ddc73d599153ddfca563e4cca575cf4fc3e6b7d07fcd1f8e37f29145"
    )
    assert len((destination / "training.jsonl").read_text(encoding="utf-8").splitlines()) == 500
    tail = destination / "恢复源_未提交joint1尾行_仅证据.jsonl"
    assert sha256_file(tail) == (
        "502f82c00b3750fc9041224977587095f91169936dbd8243e6f665f7c0b78cbf"
    )
    assert len(tail.read_text(encoding="utf-8").splitlines()) == 1
    assert sha256_file(destination / "恢复源_完整501行失败日志_只读证据.jsonl") == (
        EXPECTED_FAILED_SHAS["training.jsonl"]
    )
    assert sha256_file(destination / "恢复源_陈旧epoch400阶段报告_只读证据.json") == (
        EXPECTED_FAILED_SHAS["任09F3阶段报告.json"]
    )
    assert sha256_file(destination / "阶段_最近.pt") == before["阶段_最近.pt"]
    assert manifest["恢复后已提交全局轮次"] == 500
    assert manifest["恢复后受限联合已提交轮次"] == 0
    assert manifest["未提交joint1将在CUDA重新计算"] is True
    assert manifest["未复用失败joint1模型_优化器_RNG"] is True
    repaired.validate_task09_v2_recovery_directory(
        destination, source_directory=FAILED_RUN,
    )
    assert {name: sha256_file(FAILED_RUN / name) for name in EXPECTED_FAILED_SHAS} == before
    with pytest.raises(FileExistsError, match="已有|覆盖|新目录"):
        repaired.stage_task09_v2_recovery(
            FAILED_RUN, destination, name="primary", size=3, seed=0,
            v2_registry_path=repaired.TASK09_V2_REGISTRY,
            v2_registry_sha256=sha256_file(repaired.TASK09_V2_REGISTRY),
        )


def test_recovery_refuses_any_source_other_than_exact_failed_original(tmp_path: Path):
    repaired = _v2()
    impostor = tmp_path / FAILED_RUN.name
    impostor.mkdir()
    with pytest.raises(ValueError, match="原失败|唯一|现场|路径"):
        repaired.audit_task09_failed_recovery_source(impostor)


def test_v2_budget_registration_and_structured_generator_are_exact():
    repaired = _v2()
    expected = repaired.expected_task09_v2_registration()
    assert expected["schema_version"] == 1
    assert expected["任务"] == "任09 P1冻结LF AdamW提交门禁修订恢复"
    assert expected["旧0059正式登记SHA256"] == repaired.ORIGINAL_FORMAL_REGISTRY_SHA256
    assert expected["旧0059源码tar_SHA256"] == (
        "0091e55a2f57077f70c1008344aff9d15c73d2e78b0070327b93f8bb172698ef"
    )
    assert expected["失败现场完整日志SHA256"] == EXPECTED_FAILED_SHAS["training.jsonl"]
    assert expected["失败现场已提交校正_联合_全局轮次"] == [500, 0, 500]
    assert expected["失败尾行只作证据不提交不复用"] is True
    assert expected["恢复后首个joint1预期AdamW_step"] == {
        "10个新HF校正参数": 2505,
        "4个LF末投影参数": 5,
        "其余LF参数": 0,
    }
    assert expected["联合每轮LF真实回放batch"] == 60
    assert expected["联合每个LF小批点"] == 2048
    assert expected["联合每LF2048小批损失系数"] == 0.25
    assert expected["校正_联合预算上限不变"] == [1500, 500]
    assert expected["ROOT主台账录0075前GPU禁止"] is True
    assert expected["旧固定TEST温度读取"] is False
    sources = expected["新修订源码SHA256"]
    assert set(sources) == {"trainer_v2", "CLI_v2", "registration_generator_v2", "tests_v2"}
    assert all(len(value) == 64 for value in sources.values())

    generator = PROJECT_ROOT / "scripts/43_register_task09_subset_formal_v2.py"
    assert generator.is_file()
    generated = __import__("subprocess").run(
        [__import__("sys").executable, str(generator), "--dry-run"],
        capture_output=True, text=True, check=False,
    )
    assert generated.returncode == 0, generated.stderr
    assert yaml.safe_load(generated.stdout) == expected


def test_v2_step_gate_is_process_scoped_and_restored_even_on_failure():
    repaired, formal = _v2(), _v1()
    original = formal.assert_task09_adamw_steps

    def observe(value):
        assert formal.assert_task09_adamw_steps is repaired.assert_task09_v2_adamw_steps
        return value

    assert repaired.call_with_task09_v2_step_gate(observe, 17) == 17
    assert formal.assert_task09_adamw_steps is original

    def explode():
        assert formal.assert_task09_adamw_steps is repaired.assert_task09_v2_adamw_steps
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic"):
        repaired.call_with_task09_v2_step_gate(explode)
    assert formal.assert_task09_adamw_steps is original


def test_v2_cli_exposes_separate_prepare_and_cuda_run_without_cpu_formal_mode():
    script = PROJECT_ROOT / "scripts/42_run_task09_subset_formal_v2.py"
    assert script.is_file()
    completed = __import__("subprocess").run(
        [__import__("sys").executable, str(script), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "prepare" in completed.stdout and "run" in completed.stdout
    run_help = __import__("subprocess").run(
        [__import__("sys").executable, str(script), "run", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert run_help.returncode == 0
    assert "--device {cuda}" in run_help.stdout
    assert "--root-ledger-entry" in run_help.stdout
    assert "--source-archive" in run_help.stdout
    assert "--source-archive-sha256" in run_help.stdout


def test_v2_cuda_gate_reads_real_0075_line_and_requires_yaml_tar_dual_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    repaired = _v2()
    ledger = tmp_path / "唯一总台账.md"
    archive = tmp_path / "任09V2源码冻结.tar.gz"
    archive.write_bytes(b"real v2 archive bytes")
    archive_sha = sha256_file(archive)
    registry_sha = "a" * 64
    monkeypatch.setattr(repaired, "TASK09_LEDGER", ledger)
    monkeypatch.setattr(repaired, "TASK09_V2_ARCHIVE_ROOT", tmp_path)

    ledger.write_text("| 录-0074 | other task |\n", encoding="utf-8")
    with pytest.raises(ValueError, match="0075|台账|登记"):
        repaired.require_task09_v2_root_ledger(
            registry_sha256=registry_sha,
            archive_path=archive, archive_sha256=archive_sha,
        )
    ledger.write_text(
        f"| 录-0075 | 任09V2 YAML SHA`{registry_sha}`，tar尚缺 |\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="同一行|双SHA|tar|归档"):
        repaired.require_task09_v2_root_ledger(
            registry_sha256=registry_sha,
            archive_path=archive, archive_sha256=archive_sha,
        )
    ledger.write_text(
        f"| 录-0075 | 任09V2 YAML SHA`{registry_sha}` |\n"
        f"| 附件 | tar SHA`{archive_sha}` {archive.name} |\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="同一行|双SHA|tar|归档"):
        repaired.require_task09_v2_root_ledger(
            registry_sha256=registry_sha,
            archive_path=archive, archive_sha256=archive_sha,
        )
    ledger.write_text(
        f"| 录-0075 | 任09V2登记SHA`{registry_sha}`；"
        f"源码tar `{archive.name}` SHA`{archive_sha}` |\n",
        encoding="utf-8",
    )
    receipt = repaired.require_task09_v2_root_ledger(
        registry_sha256=registry_sha,
        archive_path=archive, archive_sha256=archive_sha,
    )
    assert receipt["ROOT主台账条目"] == "录-0075"
    assert receipt["V2源码归档SHA256"] == archive_sha
    assert receipt["V2事前登记SHA256"] == registry_sha
    assert receipt["ROOT主台账读取前后SHA256一致"] is True


def test_v2_source_archive_requires_exact_regular_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    repaired = _v2()
    payload = b"bound source bytes"
    expected = {"src/bound.py": __import__("hashlib").sha256(payload).hexdigest()}
    monkeypatch.setattr(repaired, "TASK09_V2_ARCHIVE_ROOT", tmp_path)
    monkeypatch.setattr(repaired, "expected_task09_v2_archive_members", lambda: expected)

    archive = tmp_path / "任09F3联合首轮P1修订恢复源码冻结_test.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("src/bound.py")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    receipt = repaired.validate_task09_v2_source_archive(archive, sha256_file(archive))
    assert receipt["普通文件成员数"] == 1
    assert receipt["逐成员SHA256"] == expected

    forged = tmp_path / "任09F3联合首轮P1修订恢复源码冻结_forged.tar.gz"
    with tarfile.open(forged, "w:gz") as bundle:
        info = tarfile.TarInfo("src/bound.py")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
        link = tarfile.TarInfo("src/link.py")
        link.type = tarfile.SYMTYPE
        link.linkname = "bound.py"
        bundle.addfile(link)
    with pytest.raises(ValueError, match="普通|成员|重复|相对"):
        repaired.validate_task09_v2_source_archive(forged, sha256_file(forged))
