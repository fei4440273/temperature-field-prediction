"""Task-07 fresh E0 five-seed correction and bounded-joint entry gates."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from sic_cu.config import PROJECT_ROOT


def _entry():
    try:
        return importlib.import_module("sic_cu.train.task07_formal")
    except ModuleNotFoundError:
        pytest.fail("任-07独立正式训练入口尚未实现")


@pytest.fixture(scope="module")
def sources():
    from sic_cu.train.task07_source import validate_task07_sources

    return validate_task07_sources()


def test_e0_five_seed_initializations_keep_distinct_paired_lf_and_empty_hf_adamw(sources) -> None:
    entry = _entry()
    seen = set()
    for seed in range(5):
        initial = entry.fork_task07_e0(sources, seed, torch.device("cpu"))
        source = sources[seed]
        assert initial.model.correction[0].in_features == 6
        assert "response_features.tau_seconds" not in initial.model.state_dict()
        assert initial.optimizer.state_dict()["state"] == {}
        assert len(initial.optimizer.param_groups) == 1
        assert initial.lf_checkpoint_sha256 == source.lf_checkpoint_sha256
        assert initial.lf_tensor_sha256 == source.lf_tensor_sha256
        assert set(initial.random_state) == {"python", "numpy", "torch_cpu", "torch_cuda"}
        assert all(not param.requires_grad for param in initial.model.low_fidelity_model.parameters())
        for name, tensor in source.lf_state.items():
            assert torch.equal(initial.model.low_fidelity_model.state_dict()[name], tensor)
        seen.add(source.lf_tensor_sha256)
    assert len(seen) == 5


def test_formal_training_requires_audited_registration_before_creating_output(tmp_path) -> None:
    entry = _entry()
    output = tmp_path / "任07未经预登记不得训练"
    with pytest.raises(ValueError, match="预登记|正式|SHA"):
        entry.run_task07_formal(seed=0, output_directory=output, device_name="cpu")
    assert not output.exists()


def test_training_yaml_and_its_self_signed_sha_cannot_spoof_task07_registration(
    tmp_path, monkeypatch,
) -> None:
    from sic_cu.data.common import sha256_file

    entry = _entry()
    monkeypatch.setattr(entry, "validate_task07_sources", lambda: pytest.fail(
        "伪正式登记不得读五seed来源，更不得创建训练工件",
    ))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    output = tmp_path / "任07伪登记拒绝"
    training = PROJECT_ROOT / "configs/training.yaml"
    with pytest.raises(ValueError, match="预登记|专属|审计|SHA"):
        entry.run_task07_formal(
            seed=0, output_directory=output, device_name="cuda",
            formal_registry_path=training,
            formal_registry_sha256=sha256_file(training),
        )
    assert not output.exists()


def test_dedicated_registration_with_matching_ledger_sha_still_rejects_fake_budget_before_output(
    tmp_path, sources, monkeypatch,
) -> None:
    from sic_cu.data.common import sha256_file

    entry = _entry()
    registry = tmp_path / "有效运行配置_实现修复后.yaml"
    ledger = tmp_path / "事前台账.md"
    fake = entry._expected_registration(sources)
    fake["固定训练合同"]["HF校正预算上限"] = 1
    registry.write_text(yaml.safe_dump(fake, allow_unicode=True), encoding="utf-8")
    digest = sha256_file(registry)
    ledger.write_text(
        "## 九、已实施事项与改动记录表\n"
        f"| 录-0036 | 任-07登记 | 研究记录/任务07_正式五种子重训/有效运行配置_实现修复后.yaml | "
        f"事前登记SHA256 `{digest}` |\n\n## 十、指标改善明细表\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(entry, "TASK07_REGISTRATION", registry)
    monkeypatch.setattr(entry, "TASK07_LEDGER", ledger)
    monkeypatch.setattr(entry, "validate_task07_sources", lambda: sources)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    output = tmp_path / "伪budget不得生成正式模型"
    with pytest.raises(ValueError, match="登记|预算|合同|来源"):
        entry.run_task07_formal(
            seed=0, output_directory=output, device_name="cuda",
            formal_registry_path=registry, formal_registry_sha256=digest,
        )
    assert not output.exists()


def test_cli_rejected_registration_must_not_overwrite_existing_user_output(
    tmp_path, monkeypatch,
) -> None:
    entry = _entry()
    cli = PROJECT_ROOT / "scripts/25_run_task07_formal.py"
    spec = importlib.util.spec_from_file_location("task07_cli_gate_test", cli)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    existing = tmp_path / "用户旧资料目录"
    existing.mkdir()
    sentinel = existing / "原已有资料.txt"
    old_exception = existing / "异常与接续.json"
    sentinel.write_bytes(b"untouched user content")
    old_exception.write_bytes(b"untouched previous exception")
    monkeypatch.setattr(sys, "argv", [
        str(cli), "--seed", "0", "--output", str(existing),
        "--registry-sha256", "0" * 64, "--device", "cuda",
    ])
    monkeypatch.setattr(module, "run_task07_formal", lambda **kwargs: (_ for _ in ()).throw(
        ValueError("任-07专属配置未预登记"),
    ))
    with pytest.raises(ValueError, match="预登记"):
        module.main()
    assert sentinel.read_bytes() == b"untouched user content"
    assert old_exception.read_bytes() == b"untouched previous exception"
    assert set(existing.iterdir()) == {sentinel, old_exception}


def test_formal_registration_is_recorded_inside_ledger_section_nine_not_elsewhere(
    tmp_path, sources, monkeypatch,
) -> None:
    from sic_cu.data.common import sha256_file

    entry = _entry()
    registry = tmp_path / "有效运行配置_实现修复后.yaml"
    registry.write_text(yaml.safe_dump(entry._expected_registration(sources), allow_unicode=True),
                        encoding="utf-8")
    digest = sha256_file(registry)
    ledger = tmp_path / "假录登记位置.md"
    ledger.write_text(
        "## 九、已实施事项与改动记录表\n\n## 十、指标改善明细表\n"
        f"| 录-0036 | 任-07 | 研究记录/任务07_正式五种子重训/有效运行配置_实现修复后.yaml | "
        f"事前SHA256 `{digest}` |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(entry, "TASK07_REGISTRATION", registry)
    monkeypatch.setattr(entry, "TASK07_LEDGER", ledger)
    with pytest.raises(ValueError, match="台账|第九节|事前"):
        entry._require_formal_registration(registry, digest, sources)


def test_diagnostic_rejects_cuda_before_opening_seed_source_or_creating_output(
    tmp_path, monkeypatch,
) -> None:
    entry = _entry()
    monkeypatch.setattr(entry, "validate_task07_sources", lambda: pytest.fail(
        "CUDA不得用于短诊断，读取来源本身过早",
    ))
    output = tmp_path / "任07禁止CUDA诊断"
    with pytest.raises(ValueError, match="诊断.*CPU|CPU.*诊断"):
        entry.run_task07_formal(
            seed=0, output_directory=output, session_epoch_limit=1,
            diagnostic_only=True, device_name="cuda",
        )
    assert not output.exists()


def test_formal_stage_rejects_missing_cuda_rng_even_when_all_four_keys_exist(
    tmp_path, sources, monkeypatch,
) -> None:
    entry = _entry()
    source = sources[0]
    initial = entry.fork_task07_e0(sources, 0, torch.device("cpu"))
    empty_consumption = {
        "HF训练观测点": 0, "HF训练传感器点": 0,
        "LF真实回放训练点": 0, "物理配点": 0,
        "HF观测优化步": 0, "物理优化步": 0,
        "LF联合回放batch": 0,
    }
    registry_sha = "f" * 64
    meta = entry._metadata(
        source, initial.model, entry.CORRECTION_STAGE, registry_sha, False,
        correction_epoch=0, joint_epoch=0, correction_completed=None,
        initial_score=10.0, best_score=10.0, best_global_epoch=0,
        best_stage=entry.CORRECTION_STAGE, best_stage_epoch=0,
        physical_score=1.0, physical_global_epoch=0,
        physical_stage=entry.CORRECTION_STAGE, physical_stage_epoch=0,
        phase_best_epoch=0, lf_reference={"Cu": {"node": 1.0, "volume": 1.0}},
        consumption=empty_consumption,
    )
    stage = tmp_path / "原点完整状态.pt"
    from sic_cu.train.common import save_training_state

    save_training_state(
        stage, initial.model, initial.optimizer, stage=entry.CORRECTION_STAGE,
        epoch=0, budget=entry.STAGE_BUDGET, metadata=meta,
    )
    state = torch.load(stage, map_location="cpu", weights_only=False)
    state["random_state"]["torch_cuda"] = None
    with pytest.raises(ValueError, match="CUDA|随机"):
        entry._check_snapshot_identity(state, source, registry_sha, False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [torch.ones(4096, dtype=torch.uint8)])
    for fake_rng in (torch.ones(1, dtype=torch.uint8), torch.ones(4096, dtype=torch.float32)):
        state["random_state"]["torch_cuda"] = [fake_rng]
        with pytest.raises(ValueError, match="CUDA|随机"):
            entry._check_snapshot_identity(state, source, registry_sha, False)
    # CUDA seed 0 can have a valid, replayable all-zero RNG state.
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [torch.zeros(16, dtype=torch.uint8)])
    state["random_state"]["torch_cuda"] = [torch.zeros(16, dtype=torch.uint8)]
    entry._check_snapshot_identity(state, source, registry_sha, False)


def test_joint_phase_patience_uses_eligible_phase_score_not_correction_global_best() -> None:
    entry = _entry()
    # The eligible joint score improves on its own 24C start, but not on
    # the earlier 20C correction best; joint patience must nevertheless reset.
    assert entry._joint_best_update(
        score=22.0, global_best_score=20.0, phase_best_score=24.0,
        phase_best_epoch=0, joint_epoch=11, eligible=True,
    ) == (22.0, 11, 20.0, False)
    assert entry._joint_best_update(
        score=19.0, global_best_score=20.0, phase_best_score=24.0,
        phase_best_epoch=0, joint_epoch=11, eligible=False,
    ) == (24.0, 0, 20.0, False)


def test_joint_lf_unchecked_is_never_reported_as_eligible() -> None:
    entry = _entry()
    assert "未检查" in entry._lf_keep_qualification(entry.JOINT_STAGE, None)
    assert "不能判通过" in entry._lf_keep_qualification(entry.JOINT_STAGE, None)
    assert "未通过" in entry._lf_keep_qualification(
        entry.JOINT_STAGE, {"LF两材料节点与真实体积均守住5%护栏": False},
    )
    assert "守住" in entry._lf_keep_qualification(
        entry.JOINT_STAGE, {"LF两材料节点与真实体积均守住5%护栏": True},
    )


def test_hf_observation_budget_cannot_self_sign_one_point_instead_of_real_29593(
    tmp_path,
) -> None:
    entry = _entry()
    output = tmp_path / "任07伪HF观测点"
    entry.run_task07_formal(
        seed=0, output_directory=output, session_epoch_limit=1,
        diagnostic_only=True, device_name="cpu",
    )
    latest = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    row = json.loads((output / "training.jsonl").read_text(encoding="utf-8"))
    assert row["HF训练观测点"] == 29593
    fake = dict(row)
    fake["HF训练观测点"] = 1
    fake["累计实际消耗"] = row["累计实际消耗"].copy()
    fake["累计实际消耗"]["HF训练观测点"] = 1
    metadata = latest["metadata"].copy()
    metadata["累计实际消耗"] = fake["累计实际消耗"]
    sensor_count = row["HF训练传感器点"] // 15
    with pytest.raises(ValueError, match="HF|观测|预算"):
        entry._log_budget([fake], 1, sensor_count, metadata)


def test_positive_epoch_rejects_wiped_hf_adamw_momenta(
    tmp_path, sources,
) -> None:
    entry = _entry()
    output = tmp_path / "任07强制真实AdamW步数"
    entry.run_task07_formal(
        seed=0, output_directory=output, session_epoch_limit=1,
        diagnostic_only=True, device_name="cpu",
    )
    committed = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    forged = committed.copy()
    forged["optimizer_state"] = committed["optimizer_state"].copy()
    forged["optimizer_state"]["state"] = {}
    with pytest.raises(ValueError, match="AdamW|动量|真实"):
        entry._check_snapshot_identity(forged, sources[0], None, True)
    assert committed["optimizer_state"]["state"]
    assert all(float(moment["step"]) >= 16 for moment in committed[
        "optimizer_state"
    ]["state"].values())
    one_id = committed["optimizer_state"]["param_groups"][0]["params"][0]
    extra = committed.copy()
    extra["optimizer_state"] = committed["optimizer_state"].copy()
    extra["optimizer_state"]["state"] = committed["optimizer_state"]["state"].copy()
    extra_record = extra["optimizer_state"]["state"][one_id].copy()
    extra_record["step"] = torch.tensor(33.0)
    extra["optimizer_state"]["state"][one_id] = extra_record
    with pytest.raises(ValueError, match="AdamW|动量|步数"):
        entry._check_snapshot_identity(extra, sources[0], None, True)


def test_joint_committed_epoch_rejects_only_one_missing_or_short_lf_projection_momentum(
    tmp_path, sources,
) -> None:
    entry = _entry()
    output = tmp_path / "任07联合四投影部分伪动量"
    entry.run_task07_formal(
        seed=0, output_directory=output, session_epoch_limit=2,
        diagnostic_only=True, device_name="cpu",
    )
    entry.run_task07_formal(
        seed=0, output_directory=output, session_epoch_limit=1,
        diagnostic_only=True, diagnostic_joint_preview=True,
        device_name="cpu", resume_training_checkpoint=output / "阶段_最近.pt",
    )
    committed = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    groups = committed["optimizer_state"]["param_groups"]
    assert committed["epoch"] == 1 and len(groups[1]["params"]) == 4
    for identifier in groups[1]["params"]:
        assert float(committed["optimizer_state"]["state"][identifier]["step"]) >= 16
    identifier = groups[1]["params"][0]
    for tamper in ("missing", "short_step", "extra_step", "wrong_shape"):
        forged = committed.copy()
        forged["optimizer_state"] = committed["optimizer_state"].copy()
        forged["optimizer_state"]["state"] = committed["optimizer_state"]["state"].copy()
        if tamper == "missing":
            del forged["optimizer_state"]["state"][identifier]
        else:
            record = forged["optimizer_state"]["state"][identifier].copy()
            if tamper == "short_step":
                record["step"] = torch.tensor(1.0)
            elif tamper == "extra_step":
                record["step"] = torch.tensor(17.0)
            else:
                record["exp_avg"] = record["exp_avg"].reshape(-1)
            forged["optimizer_state"]["state"][identifier] = record
        with pytest.raises(ValueError, match="AdamW|动量|真实"):
            entry._check_snapshot_identity(forged, sources[0], None, True)


@pytest.mark.parametrize("tamper", ["geometry", "physics"])
def test_resume_snapshot_rejects_fake_copied_config_or_physics_before_old_writes(
    tmp_path, tamper,
) -> None:
    entry = _entry()
    output = tmp_path / f"任07原件物理快照_{tamper}"
    entry.run_task07_formal(
        seed=0, output_directory=output, session_epoch_limit=1,
        diagnostic_only=True, device_name="cpu",
    )
    if tamper == "geometry":
        target = output / "config_snapshot/geometry.yaml"
        target.write_bytes(target.read_bytes() + b"\n# fake copied geometry\n")
    else:
        target = output / "config_snapshot/resolved_physics.yaml"
        state = yaml.safe_load(target.read_text(encoding="utf-8"))
        state["values"]["silicon_carbide_emissivity"]["value"] = 0.99
        target.write_text(yaml.safe_dump(state, allow_unicode=True), encoding="utf-8")
    immutable = {name: (output / name).read_bytes() for name in (
        "阶段_最近.pt", "阶段_观测最佳.pt", "阶段报告.json", "training.jsonl",
    )}
    with pytest.raises(ValueError, match="快照|物理|SHA"):
        entry.run_task07_formal(
            seed=0, output_directory=output, session_epoch_limit=1,
            diagnostic_only=True, device_name="cpu",
            resume_training_checkpoint=output / "阶段_最近.pt",
        )
    assert all((output / name).read_bytes() == original for name, original in immutable.items())


def test_diagnostic_short_run_never_creates_formal_terminal(tmp_path) -> None:
    entry = _entry()
    output = tmp_path / "任07_seed0_诊断校正"
    report = entry.run_task07_formal(
        seed=0, output_directory=output, session_epoch_limit=1,
        diagnostic_only=True, device_name="cpu",
    )
    source, initial, latest = (
        torch.load(output / name, map_location="cpu", weights_only=False)
        for name in ("阶段_初始.pt", "阶段_观测最佳.pt", "阶段_最近.pt")
    )
    assert source["epoch"] == 0 and latest["epoch"] == 1
    assert source["optimizer_state"]["state"] == {}
    assert latest["stage"] == "task07_correction"
    assert latest["budget"] == {"HF校正上限轮次": 1500, "受限联合上限轮次": 500}
    assert set(latest["random_state"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert all(not value for name, value in latest["parameter_requires_grad"].items()
               if name.startswith("low_fidelity_model."))
    assert all(value for name, value in latest["parameter_requires_grad"].items()
               if name.startswith("correction."))
    assert report["运行资格"].startswith("短诊断")
    assert not (output / "阶段_校正末.pt").exists()
    assert not (output / "阶段_联合末.pt").exists()
    assert not (output / "阶段_训练末.pt").exists()
    rows = [json.loads(row) for row in (output / "training.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()]
    assert len(rows) == 1 and rows[0]["epoch"] == 1
    assert rows[0]["HF观测优化步"] == 15 and rows[0]["物理优化步"] == 1
    assert rows[0]["物理配点"] == 256 and rows[0]["LF真实回放训练点"] == 0
    assert rows[0]["HF训练观测点"] > 14 * 2048


def test_diagnostic_correction_one_to_two_real_epochs_restores_adamw_and_budget(tmp_path) -> None:
    entry = _entry()
    output = tmp_path / "任07_seed1_校正续跑"
    for run in (0, 1):
        report = entry.run_task07_formal(
            seed=1, output_directory=output, session_epoch_limit=1,
            diagnostic_only=True, device_name="cpu",
            **({"resume_training_checkpoint": output / "阶段_最近.pt"} if run else {}),
        )
    latest = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert latest["epoch"] == 2 and latest["stage"] == "task07_correction"
    assert len(latest["optimizer_state"]["param_groups"]) == 1
    assert latest["optimizer_state"]["state"]
    assert report["累计实际消耗"]["HF观测优化步"] == 30
    assert report["累计实际消耗"]["物理配点"] == 512
    assert [json.loads(row)["epoch"] for row in (output / "training.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()] == [1, 2]
    assert not (output / "阶段_校正末.pt").exists()


def test_rejects_wrong_seed_resume_without_rewriting_source_state(tmp_path) -> None:
    entry = _entry()
    output = tmp_path / "任07_seed0_禁止冒名"
    entry.run_task07_formal(
        seed=0, output_directory=output, session_epoch_limit=1,
        diagnostic_only=True, device_name="cpu",
    )
    latest = output / "阶段_最近.pt"
    first = latest.read_bytes()
    report = (output / "阶段报告.json").read_bytes()
    rows = (output / "training.jsonl").read_bytes()
    with pytest.raises(ValueError, match="种子|来源|SHA"):
        entry.run_task07_formal(
            seed=1, output_directory=output, session_epoch_limit=1,
            diagnostic_only=True, device_name="cpu",
            resume_training_checkpoint=latest,
        )
    assert latest.read_bytes() == first
    assert (output / "阶段报告.json").read_bytes() == report
    assert (output / "training.jsonl").read_bytes() == rows


def test_diagnostic_joint_preview_consumes_all_60_true_lf_powers_and_keeps_four_projectors(
    tmp_path, sources,
) -> None:
    entry = _entry()
    output = tmp_path / "任07_seed0_校正两轮再受限联合一轮诊断"
    entry.run_task07_formal(
        seed=0, output_directory=output, session_epoch_limit=2,
        diagnostic_only=True, device_name="cpu",
    )
    correction = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert correction["stage"] == "task07_correction" and correction["epoch"] == 2
    report = entry.run_task07_formal(
        seed=0, output_directory=output, session_epoch_limit=1,
        diagnostic_only=True, diagnostic_joint_preview=True, device_name="cpu",
        resume_training_checkpoint=output / "阶段_最近.pt",
    )
    diagnostic_switch = torch.load(output / "阶段_诊断校正切换.pt",
                                   map_location="cpu", weights_only=False)
    joint_initial = torch.load(output / "阶段_联合初始.pt", map_location="cpu", weights_only=False)
    joint = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert diagnostic_switch["stage"] == "task07_correction"
    assert diagnostic_switch["epoch"] == 2 and diagnostic_switch["metadata"]["校正实际轮次"] == 2
    assert joint_initial["stage"] == joint["stage"] == "task07_restricted_joint"
    assert joint_initial["epoch"] == 0 and joint["epoch"] == 1
    assert joint_initial["metadata"]["本阶段最低合格HF选分_摄氏度"] >= joint_initial[
        "metadata"
    ]["观测最佳选分_摄氏度"]
    assert joint["metadata"]["本阶段最低合格HF选分_摄氏度"] == joint_initial[
        "metadata"
    ]["本阶段最低合格HF选分_摄氏度"]
    assert joint["metadata"]["全局累计实际轮次"] == 3
    assert joint["budget"] == {"HF校正上限轮次": 1500, "受限联合上限轮次": 500}
    assert len(joint["optimizer_state"]["param_groups"]) == 2
    assert joint["optimizer_state"]["param_groups"][0]["lr"] == 0.0001
    assert joint["optimizer_state"]["param_groups"][1]["lr"] == 0.00001
    movable = {name for name, yes in joint["parameter_requires_grad"].items() if yes}
    assert {name for name in movable if name.startswith("low_fidelity_model.")} == {
        "low_fidelity_model.branch_projection.weight", "low_fidelity_model.branch_projection.bias",
        "low_fidelity_model.trunk_projection.weight", "low_fidelity_model.trunk_projection.bias",
    }
    for name, value in sources[0].lf_state.items():
        full = "low_fidelity_model." + name
        if full not in movable:
            assert torch.equal(joint["model_state"][full], value)
    assert any(not torch.equal(joint["model_state"][name], joint_initial["model_state"][name])
               for name in movable if name.startswith("low_fidelity_model."))
    rows = [json.loads(row) for row in (output / "training.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2, 3]
    assert rows[-1]["HF观测优化步"] == 15 and rows[-1]["物理配点"] == 256
    assert rows[-1]["LF真实回放训练点"] == 60 * 2048
    assert rows[-1]["LF联合回放batch"] == 60
    assert rows[-1]["LF_Cu真实回放点"] and rows[-1]["LF_SiC真实回放点"]
    assert report["累计实际消耗"]["HF观测优化步"] == 45
    assert report["累计实际消耗"]["LF真实回放训练点"] == 60 * 2048
    rates = rows[-1]["LF逐材料节点和真实体积5%护栏"]
    assert not rates["LF两材料节点与真实体积均守住5%护栏"]
    assert set(rates["逐材料逐口径恶化率_百分比"]) == {"Cu", "SiC"}
    assert rates["逐材料逐口径恶化率_百分比"]["SiC"]["node"] > 5.0
    assert rates["逐材料逐口径恶化率_百分比"]["SiC"]["volume"] > 5.0
    assert joint["metadata"]["观测最佳阶段"] == "task07_correction"
    assert report["LF保持资格"] == "未通过，不得采用这份联合模型"
    assert not (output / "阶段_校正末.pt").exists()
    assert not (output / "阶段_联合末.pt").exists()
    assert not (output / "阶段_训练末.pt").exists()


def test_joint_one_to_two_resume_reuses_real_hf_and_projector_adamw_momentum(
    tmp_path,
) -> None:
    entry = _entry()
    output = tmp_path / "任07_seed1_校正2联合1再联合1_真实续跑"
    entry.run_task07_formal(
        seed=1, output_directory=output, session_epoch_limit=2,
        diagnostic_only=True, device_name="cpu",
    )
    entry.run_task07_formal(
        seed=1, output_directory=output, session_epoch_limit=1,
        diagnostic_only=True, diagnostic_joint_preview=True, device_name="cpu",
        resume_training_checkpoint=output / "阶段_最近.pt",
    )
    joint_one = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    report = entry.run_task07_formal(
        seed=1, output_directory=output, session_epoch_limit=1,
        diagnostic_only=True, device_name="cpu",
        resume_training_checkpoint=output / "阶段_最近.pt",
    )
    joint_two = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert joint_one["stage"] == joint_two["stage"] == "task07_restricted_joint"
    assert joint_one["epoch"] == 1 and joint_two["epoch"] == 2
    assert joint_two["metadata"]["全局累计实际轮次"] == 4
    assert len(joint_two["optimizer_state"]["param_groups"]) == 2
    assert len(joint_two["optimizer_state"]["state"]) >= len(joint_one["optimizer_state"]["state"])
    assert all(float(state["step"]) >= 16 for state in joint_two["optimizer_state"]["state"].values())
    assert joint_one["metadata"]["已提交诊断校正切换原件SHA256"] == joint_two[
        "metadata"
    ]["已提交诊断校正切换原件SHA256"]
    assert report["累计实际消耗"]["HF观测优化步"] == 60
    assert report["累计实际消耗"]["物理配点"] == 4 * 256
    assert report["累计实际消耗"]["LF真实回放训练点"] == 2 * 60 * 2048
    rows = [json.loads(row) for row in (output / "training.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2, 3, 4]
    assert rows[-1]["训练阶段"] == "task07_restricted_joint"
    assert rows[-1]["LF真实回放训练点"] == 60 * 2048
    assert not (output / "阶段_联合末.pt").exists()
