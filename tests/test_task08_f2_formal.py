"""任08 F2独立正式训练协议与事前来源门禁。"""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import shutil
import sys
import copy
import tarfile

import pytest
import torch

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.train.task07_source import fork_task07_initialization, validate_task07_sources


def _entry():
    return importlib.import_module("sic_cu.train.task08_f2_formal")


@pytest.fixture(scope="module")
def sources():
    return validate_task07_sources()


def test_formal_registration_fails_before_outputs_and_hf_data(tmp_path, monkeypatch):
    entry = _entry()
    output = tmp_path / "F2不许伪造登记"
    monkeypatch.setattr(entry, "validate_task07_sources", lambda: pytest.fail("不许读来源"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="登记|SHA"):
        entry.run_task08_f2_formal(seed=0, output_directory=output, device_name="cuda")
    assert not output.exists()


def test_old_locked_registration_0047_can_not_qualify_transaction_revision():
    entry = _entry()
    old = PROJECT_ROOT / "研究记录/任务08_贡献消融/F2_正式五种子有效运行配置.yaml"
    assert sha256_file(old) == (
        "4cc28923d22af6e2118f7dce34d042f4675c275fbf76a1f972cdc89cede106af")
    assert entry.TASK08_F2_REGISTRATION != old
    with pytest.raises(ValueError, match="登记|SHA|固定"):
        entry._require_formal_registration(old, sha256_file(old))


def test_project_boundary_refuses_external_output_without_reading_any_source(monkeypatch):
    entry = _entry()
    outside = PROJECT_ROOT.parent / "任08F2禁止项目外写入诊断_20260916"
    assert not outside.exists()
    monkeypatch.setattr(entry, "validate_task07_sources", lambda: pytest.fail("越界不许读来源"))
    with pytest.raises(ValueError, match="项目内|写入范围"):
        entry.run_task08_f2_formal(seed=0, output_directory=outside,
                                   diagnostic_only=True, session_epoch_limit=1,
                                   device_name="cpu")
    assert not outside.exists()


def test_project_boundary_rejects_symlink_escaping_project(tmp_path, monkeypatch):
    entry = _entry()
    outside = PROJECT_ROOT.parent / "任08F2禁止符号链接绕出_20260916"
    assert not outside.exists()
    bridge = tmp_path / "目录符号链接向外"
    bridge.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(entry, "validate_task07_sources", lambda: pytest.fail("越界不许读来源"))
    with pytest.raises(ValueError, match="项目内|写入范围"):
        entry.run_task08_f2_formal(seed=0, output_directory=bridge / "逃逸工件",
                                   diagnostic_only=True, session_epoch_limit=1,
                                   device_name="cpu")
    assert not outside.exists()


def test_hf_and_lf_70_real_input_files_are_pre_registered_by_bytes():
    entry = _entry()
    data = entry._input_hashes()
    assert data["合法HF顶部IR原件字节SHA256"] == (
        "dd8ef7e112bce2bd73dd3fe3ad5889131964db4af1b57c882b43f5b816d4b82d")
    assert data["处理manifest原件字节SHA256"] == (
        "1ea58fded68b8950d89c9012d879a7b6197dbb2b12c3c3ccc44af474843d8a82")
    assert len(data["LF60训练真场parquet逐文件SHA256"]) == 60
    assert len(data["LF10验证真场parquet逐文件SHA256"]) == 10
    assert len(set(data["LF60训练真场parquet逐文件SHA256"].values())) == 60
    entry._verify_registered_inputs({"训练与合法验证真实输入原件SHA256": data})
    altered = copy.deepcopy(data)
    altered["合法HF热冷环原件字节SHA256"] = "0" * 64
    with pytest.raises(ValueError, match="输入|环温|SHA|字节"):
        entry._verify_registered_inputs({"训练与合法验证真实输入原件SHA256": altered})
    altered = copy.deepcopy(data)
    first = next(iter(altered["LF60训练真场parquet逐文件SHA256"]))
    altered["LF60训练真场parquet逐文件SHA256"][first] = "0" * 64
    with pytest.raises(ValueError, match="LF|输入|SHA|字节"):
        entry._verify_registered_inputs({"训练与合法验证真实输入原件SHA256": altered})


def test_formal_source_tar_missing_member_or_byte_sha_must_not_be_usable(tmp_path):
    entry = _entry()
    missing = tmp_path / "未冻结F2源码快照.tar.gz"
    with pytest.raises((ValueError, FileNotFoundError), match="源码|归档|原件"):
        entry._source_archive_identity(missing)
    changed = tmp_path / "同名源码原件字节被换.tar.gz"
    poisoned = "src/sic_cu/train/task08_f2_formal.py"
    with tarfile.open(changed, "w:gz") as package:
        for name in entry.SOURCE_TAR_MEMBERS:
            if name == poisoned:
                contents = (PROJECT_ROOT / name).read_bytes() + b"\n# altered archived copy\n"
                member = tarfile.TarInfo(name)
                member.size = len(contents)
                package.addfile(member, io.BytesIO(contents))
            else:
                package.add(PROJECT_ROOT / name, arcname=name, recursive=False)
    with pytest.raises(ValueError, match="源码|tar|字节|漂移"):
        entry._source_archive_identity(changed)


def test_diagnostic_is_cpu_only_and_not_formal(tmp_path, monkeypatch):
    entry = _entry()
    output = tmp_path / "F2不许CUDA短诊断"
    monkeypatch.setattr(entry, "validate_task07_sources", lambda: pytest.fail("不许读来源"))
    with pytest.raises(ValueError, match="诊断.*CPU|CPU.*诊断"):
        entry.run_task08_f2_formal(seed=0, output_directory=output,
                                   diagnostic_only=True, session_epoch_limit=1,
                                   device_name="cuda")
    assert not output.exists()


def test_protocol_has_exactly_one_training_difference_and_full_independent_audit():
    entry = _entry()
    train, audit = entry._physics_pair(torch.device("cpu"))
    assert train.compute_pde is False and train.weights.pde == 0.0
    assert audit.compute_pde is True and audit.weights.pde == 1.0
    for name in ("initial", "boundary", "interface"):
        assert getattr(train.weights, name) == getattr(audit.weights, name) == 1.0
    assert entry.STAGE_BUDGET == {"HF校正上限轮次": 1500, "受限联合上限轮次": 500}
    assert entry.CORRECTION_STAGE != "task07_correction"
    assert entry.JOINT_STAGE != "task07_restricted_joint"


def test_five_initial_states_match_frozen_e0_per_seed(sources):
    entry = _entry()
    for seed, source in sources.items():
        start = entry.fork_f2_e0(sources, seed, torch.device("cpu"))
        reference = fork_task07_initialization(sources, seed, "E0", torch.device("cpu"))
        assert start.optimizer.state_dict()["state"] == {}
        assert set(start.random_state) == {"python", "numpy", "torch_cpu", "torch_cuda"}
        assert source.lf_tensor_sha256 == entry._lf_sha(start.model)
        assert all(torch.equal(start.model.state_dict()[name], value)
                   for name, value in reference.model.state_dict().items())
        assert all(not p.requires_grad for p in start.model.low_fidelity_model.parameters())


def test_epoch_retains_real_15_plus_1_budget_and_never_encodes_pde_as_zero(sources):
    entry = _entry()
    initial = entry.fork_f2_e0(sources, 0, torch.device("cpu"))
    train, _ = entry._physics_pair(torch.device("cpu"))
    observed = importlib.import_module("sic_cu.train.task08_f2_no_pde").load_f2_observations(
        sources[0], torch.device("cpu"),
    )
    lf_before = entry._lf_sha(initial.model)
    row = entry._hf_epoch(initial.model, initial.optimizer, observed["train_ir"],
                          observed["train_sensor"], train, torch.device("cpu"),
                          seed=0, global_epoch=1, simulation_data=None)
    assert row["HF训练观测点"] == 29593
    assert row["HF观测优化步"] == 15 and row["物理优化步"] == 1
    assert row["物理配点"] == 256 and row["LF真实回放训练点"] == 0
    assert row["名义物理训练分项"]["pde"] is None
    assert all(float(momentum["step"]) == 16 for momentum in
               initial.optimizer.state_dict()["state"].values())
    assert entry._lf_sha(initial.model) == lf_before


def test_joint_optimizer_only_unlocks_four_true_lf_projections(sources):
    entry = _entry()
    initial = entry.fork_f2_e0(sources, 0, torch.device("cpu"))
    initial.optimizer.state[initial.optimizer.param_groups[0]["params"][0]]["step"] = torch.tensor(16.)
    entry._joint_optimizer(initial.model, initial.optimizer, sources[0])
    from sic_cu.train.task04_joint import PROJECTION_NAMES

    movable = {name for name, p in initial.model.named_parameters() if p.requires_grad}
    assert movable == {name for name in movable if name.startswith("correction.")} | PROJECTION_NAMES
    assert len(initial.optimizer.param_groups) == 2
    assert initial.optimizer.param_groups[0]["lr"] == 0.0001
    assert initial.optimizer.param_groups[1]["lr"] == 0.00001


def test_real_lf_60_joint_cpu_preview_and_resume_preserve_budget_and_state(tmp_path, sources):
    entry = _entry()
    output = tmp_path / "F2_校正两轮联合真实回放和接续_绝非正式"
    entry.run_task08_f2_formal(
        seed=0, output_directory=output, session_epoch_limit=2,
        diagnostic_only=True, device_name="cpu")
    corrected = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert corrected["stage"] == entry.CORRECTION_STAGE and corrected["epoch"] == 2
    entry.run_task08_f2_formal(
        seed=0, output_directory=output, session_epoch_limit=1,
        diagnostic_only=True, diagnostic_joint_preview=True, device_name="cpu",
        resume_training_checkpoint=output / "阶段_最近.pt")
    initial = torch.load(output / "阶段_联合初始.pt", map_location="cpu", weights_only=False)
    first = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert initial["stage"] == first["stage"] == entry.JOINT_STAGE
    assert initial["epoch"] == 0 and first["epoch"] == 1
    assert len(first["optimizer_state"]["param_groups"]) == 2
    assert first["metadata"]["全局累计实际轮次"] == 3
    from sic_cu.train.task04_joint import PROJECTION_NAMES

    movable = {name for name, yes in first["parameter_requires_grad"].items() if yes}
    assert {name for name in movable if name.startswith("low_fidelity_model.")} == PROJECTION_NAMES
    for name, value in sources[0].lf_state.items():
        full = f"low_fidelity_model.{name}"
        if full not in PROJECTION_NAMES:
            assert torch.equal(first["model_state"][full], value)
    assert all(float(first["optimizer_state"]["state"][name]["step"]) >= 16
               for name in first["optimizer_state"]["state"])
    rows = [json.loads(line) for line in (output / "training.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2, 3]
    assert rows[-1]["HF观测优化步"] == 15
    assert rows[-1]["物理优化步"] == 1 and rows[-1]["物理配点"] == 256
    assert rows[-1]["LF真实回放训练点"] == 60 * 2048
    assert rows[-1]["LF联合回放batch"] == 60
    assert rows[-1]["LF_Cu真实回放点"] > 0 and rows[-1]["LF_SiC真实回放点"] > 0
    assert rows[-1]["名义物理训练分项"]["pde"] is None
    assert not (output / "阶段_训练末.pt").exists()
    entry.run_task08_f2_formal(
        seed=0, output_directory=output, session_epoch_limit=1,
        diagnostic_only=True, device_name="cpu",
        resume_training_checkpoint=output / "阶段_最近.pt")
    second = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert second["stage"] == entry.JOINT_STAGE and second["epoch"] == 2
    assert second["metadata"]["全局累计实际轮次"] == 4
    assert second["metadata"]["累计实际消耗"]["HF观测优化步"] == 60
    assert second["metadata"]["累计实际消耗"]["LF真实回放训练点"] == 2 * 60 * 2048
    assert all(float(second["optimizer_state"]["state"][name]["step"]) >=
               float(first["optimizer_state"]["state"][name]["step"])
               for name in first["optimizer_state"]["state"])
    assert not (output / "阶段_训练末.pt").exists()


def test_committed_joint_epoch_zero_resumes_after_interrupted_first_lf_epoch(tmp_path):
    entry = _entry()
    output = tmp_path / "F2联合初始已提交而首轮联合中断_历史best不得遗失"
    entry.run_task08_f2_formal(
        seed=0, output_directory=output, session_epoch_limit=2,
        diagnostic_only=True, device_name="cpu")
    entry.run_task08_f2_formal(
        seed=0, output_directory=output, session_epoch_limit=1,
        diagnostic_only=True, diagnostic_joint_preview=True, device_name="cpu",
        resume_training_checkpoint=output / "阶段_最近.pt")
    committed = torch.load(output / "阶段_联合初始.pt", map_location="cpu", weights_only=False)
    meta = committed["metadata"]
    assert committed["stage"] == entry.JOINT_STAGE and committed["epoch"] == 0
    assert meta["全局累计实际轮次"] == 2
    for name, best_key in (("阶段_观测最佳.pt", "观测最佳全局轮次"),
                           ("阶段_物理最佳.pt", "物理最佳全局轮次")):
        if meta[best_key] < meta["全局累计实际轮次"]:
            assert meta[f"已提交旧{name}SHA256"] == sha256_file(output / name)
    spoof = tmp_path / "F2联合初始旧物理best副本遭篡改须拒绝"
    shutil.copytree(output, spoof)
    shutil.copyfile(spoof / "阶段_联合初始.pt", spoof / "阶段_最近.pt")
    old_physical = torch.load(spoof / "阶段_物理最佳.pt", map_location="cpu",
                              weights_only=False)
    old_physical["model_state"]["correction.0.weight"][0, 0] += 1.0
    torch.save(old_physical, spoof / "阶段_物理最佳.pt")
    untouched = (spoof / "training.jsonl").read_bytes()
    with pytest.raises(ValueError, match="历史最佳|SHA|原件"):
        entry.run_task08_f2_formal(
            seed=0, output_directory=spoof, session_epoch_limit=1,
            diagnostic_only=True, device_name="cpu",
            resume_training_checkpoint=spoof / "阶段_最近.pt")
    assert (spoof / "training.jsonl").read_bytes() == untouched
    # 模拟联合初始已事务提交、后续首轮联合过程中断：只恢复该已提交最近态。
    shutil.copyfile(output / "阶段_联合初始.pt", output / "阶段_最近.pt")
    committed_best = {(output / name).name: sha256_file(output / name)
                      for name in ("阶段_观测最佳.pt", "阶段_物理最佳.pt")}
    entry.run_task08_f2_formal(
        seed=0, output_directory=output, session_epoch_limit=1,
        diagnostic_only=True, device_name="cpu",
        resume_training_checkpoint=output / "阶段_最近.pt")
    final = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert final["stage"] == entry.JOINT_STAGE and final["epoch"] == 1
    assert final["metadata"]["全局累计实际轮次"] == 3
    assert final["metadata"]["累计实际消耗"]["LF真实回放训练点"] == 60 * 2048
    assert all(sha256_file(output / name) == checksum
               for name, checksum in committed_best.items())
    assert [json.loads(line)["epoch"] for line in (output / "training.jsonl").read_text(
        encoding="utf-8").splitlines()] == [1, 2, 3]


def test_best_must_not_overtake_committed_latest(sources):
    entry = _entry()
    initial = entry.fork_f2_e0(sources, 0, torch.device("cpu"))
    meta = entry._metadata(sources[0], initial.model, entry.CORRECTION_STAGE,
                           "f" * 64, False, correction_epoch=0, joint_epoch=0,
                           correction_completed=None, initial_score=2.0,
                           best_score=1.9, best_global_epoch=1,
                           best_stage=entry.CORRECTION_STAGE, best_stage_epoch=1,
                           physical_score=1.0, physical_global_epoch=0,
                           physical_stage=entry.CORRECTION_STAGE, physical_stage_epoch=0,
                           phase_best_epoch=0, lf_reference={},
                           consumption=entry._empty_consumption())
    with pytest.raises(ValueError, match="最佳|最近|超前"):
        entry._validate_latest_and_best_positions(meta, 0)


@pytest.fixture(scope="module")
def diagnostic_state(tmp_path_factory):
    entry = _entry()
    output = tmp_path_factory.mktemp("任08F2_项目内两阶段源门禁") / "诊断不算正式"
    entry.run_task08_f2_formal(
        seed=0, output_directory=output,
        diagnostic_only=True, session_epoch_limit=1,
        device_name="cpu",
    )
    return output


def test_cpu_diagnostic_stage0_and_epoch1_are_full_adamw_and_four_rng(diagnostic_state):
    entry = _entry()
    stage0 = torch.load(diagnostic_state / "阶段_初始.pt", map_location="cpu", weights_only=False)
    latest = torch.load(diagnostic_state / "阶段_最近.pt", map_location="cpu", weights_only=False)
    rows = [json.loads(line) for line in (diagnostic_state / "training.jsonl").read_text(
        encoding="utf-8").splitlines()]
    source = validate_task07_sources()[0]
    assert len(rows) == 1 and rows[0]["名义物理训练分项"]["pde"] is None
    assert rows[0]["独立局部物理损失"]["pde"] >= 0
    assert rows[0]["HF训练观测点"] == 29593
    assert stage0["optimizer_state"]["state"] == {}
    assert set(stage0["random_state"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert stage0["metadata"]["训练体内PDE残差"] is None
    assert stage0["metadata"]["运行资格"] == entry._qualification(True)
    assert stage0["metadata"]["当前真实LF张量SHA256"] == source.lf_tensor_sha256
    assert latest["metadata"]["全局累计实际轮次"] == 1
    entry._check_snapshot_identity(stage0, source, None, True)
    entry._check_snapshot_identity(latest, source, None, True)
    assert not (diagnostic_state / "阶段_训练末.pt").exists()


def test_resume_rejects_best_view_ahead_of_latest_before_append(diagnostic_state, tmp_path):
    entry = _entry()
    output = tmp_path / "F2_伪超前best"
    shutil.copytree(diagnostic_state, output)
    view = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    view["epoch"] = 99
    torch.save(view, output / "best.pt")
    old_rows = (output / "training.jsonl").read_bytes()
    with pytest.raises(ValueError, match="超前|最近"):
        entry.run_task08_f2_formal(
            seed=0, output_directory=output, device_name="cpu",
            diagnostic_only=True, session_epoch_limit=1,
            resume_training_checkpoint=output / "阶段_最近.pt")
    assert (output / "training.jsonl").read_bytes() == old_rows


def test_resume_repairs_same_epoch_committed_sidecar_not_old_history(
    diagnostic_state, tmp_path,
):
    entry = _entry()
    output = tmp_path / "F2_最近提交后侧件丢失"
    shutil.copytree(diagnostic_state, output)
    latest = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert latest["metadata"]["观测最佳全局轮次"] in (0, 1)
    # The most recently selected sidecar may be lost after latest has committed;
    # historical stage-0 full snapshots can never be recreated from an epoch-1 state.
    name = ("阶段_观测最佳.pt" if latest["metadata"]["观测最佳全局轮次"] == 1
            else "阶段_物理最佳.pt" if latest["metadata"]["物理最佳全局轮次"] == 1
            else None)
    if name is None:
        pytest.skip("首轮没有新最佳：另有独立历史原件丢失拒绝注入")
    (output / name).unlink()
    entry.run_task08_f2_formal(
        seed=0, output_directory=output, device_name="cpu",
        diagnostic_only=True, session_epoch_limit=1,
        resume_training_checkpoint=output / "阶段_最近.pt",
    )
    repaired = torch.load(output / name, map_location="cpu", weights_only=False)
    assert repaired["metadata"]["全局累计实际轮次"] >= 1
    assert repaired["training_state_schema_version"] == 1


def test_resume_rejects_missing_historical_best_full_state(diagnostic_state, tmp_path):
    entry = _entry()
    output = tmp_path / "F2_原历史最佳原件丢失"
    shutil.copytree(diagnostic_state, output)
    latest = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    historic = ("阶段_观测最佳.pt" if latest["metadata"]["观测最佳全局轮次"] == 0
                else "阶段_物理最佳.pt" if latest["metadata"]["物理最佳全局轮次"] == 0
                else None)
    if historic is None:
        pytest.skip("首轮两种最佳都改进；另有SHA承诺测试必须补覆盖")
    (output / historic).unlink()
    before = (output / "training.jsonl").read_bytes()
    with pytest.raises(ValueError, match="历史最佳|SHA|原件"):
        entry.run_task08_f2_formal(
            seed=0, output_directory=output,
            diagnostic_only=True, session_epoch_limit=1, device_name="cpu",
            resume_training_checkpoint=output / "阶段_最近.pt")
    assert (output / "training.jsonl").read_bytes() == before


def test_f2_can_not_read_historical_test_temperature():
    text = (PROJECT_ROOT / "src/sic_cu/train/task08_f2_formal.py").read_text(encoding="utf-8")
    assert "test_Data/" not in text
    assert "test_Data\\" not in text


def test_cli_must_pass_registration_sha_and_never_overwrite_old_output(tmp_path, monkeypatch):
    path = PROJECT_ROOT / "scripts/32_run_task08_f2_formal.py"
    spec = importlib.util.spec_from_file_location("task08_f2_cli_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "用户原目录"
    output.mkdir()
    original = output / "原有资料.txt"
    original.write_bytes(b"user-owned")
    monkeypatch.setattr(sys, "argv", [str(path), "--seed", "0", "--output", str(output),
                                          "--device", "cuda", "--registry-sha256", "f" * 64])
    monkeypatch.setattr(module, "run_task08_f2_formal", lambda **kwargs: (_ for _ in ()).throw(
        ValueError("任08 F2预登记SHA不符")))
    with pytest.raises(ValueError, match="登记"):
        module.main()
    assert original.read_bytes() == b"user-owned"
    assert set(output.iterdir()) == {original}
