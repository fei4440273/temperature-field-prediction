from __future__ import annotations

import importlib.util
import copy
import json
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import build_power_splits
from sic_cu.train import multifidelity


REGISTERED = PROJECT_ROOT / "研究记录/任务06_时间响应特征/HF三臂先导有效配置.yaml"
SOURCE = PROJECT_ROOT / (
    "研究记录/任务04_联合微调/"
    "任04_有限解冻_种子0_20260915T210714+0800/阶段_观测最佳.pt"
)


def _training_module():
    assert importlib.util.find_spec("sic_cu.train.task06_time_features") is not None, (
        "任06专属同源HF三臂训练入口尚未实现"
    )
    from sic_cu.train import task06_time_features

    return task06_time_features


def test_fork_clones_actual_joint_lf_and_old_six_channels_with_zero_new_columns() -> None:
    training = _training_module()
    source, architecture, config = training.validate_task06_source()
    assert sha256_file(REGISTERED) == "7a418d94493ddafd9c78e5070bca0739ab2de30aecdc22ef322ef69bc7e03b19"
    assert sha256_file(SOURCE) == "1fd334dcd3abd1e1ba0ad202afac51280dc32d9df3a9603211f42fe15dbbad6e"
    models = {}
    optimizers = {}
    for arm in ("E0", "E1", "E2"):
        model, optimizer = training.fork_task06_model(source, architecture, config, arm, torch.device("cpu"))
        models[arm], optimizers[arm] = model, optimizer
        assert len(optimizer.param_groups) == 1
        assert optimizer.state_dict()["state"] == {}
        assert all(not parameter.requires_grad for parameter in model.low_fidelity_model.parameters())
        assert all(parameter.requires_grad for parameter in model.correction.parameters())
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in source["model_state"].items()
                   if name.startswith("low_fidelity_model."))
    first = source["model_state"]["correction.0.weight"]
    assert first.shape == (128, 6)
    assert models["E0"].state_dict()["correction.0.weight"].shape == first.shape
    assert "response_features.tau_seconds" not in models["E0"].state_dict()
    for arm in ("E1", "E2"):
        expanded = models[arm].state_dict()["correction.0.weight"]
        assert expanded.shape == (128, 10)
        assert torch.equal(expanded[:, :6], first)
        assert torch.count_nonzero(expanded[:, 6:]) == 0
        assert all(torch.equal(models[arm].state_dict()[name], source["model_state"][name])
                   for name in source["model_state"] if name != "correction.0.weight")
        assert models[arm].response_features._buffers["tau_seconds"] is not None
    coordinates = torch.tensor([
        [0.002, -0.005, 0.0, 115.2, 0.0],
        [0.005, -0.002, 2.0, 403.0, 1.0],
        [0.008, -0.010, 100.0, 630.5, 0.0],
    ])
    with torch.no_grad():
        predictions = [models[arm](coordinates) for arm in ("E0", "E1", "E2")]
    assert torch.equal(predictions[0], predictions[1])
    assert torch.equal(predictions[0], predictions[2])


@pytest.mark.parametrize("arm", ("E0", "E1", "E2"))
def test_response_contract_rejects_wrong_buffer_and_old_e0_buffer(arm) -> None:
    training = _training_module()
    source, architecture, config = training.validate_task06_source()
    model, _ = training.fork_task06_model(source, architecture, config, arm, torch.device("cpu"))
    stage = model.state_dict()
    kwargs = training.task06_model_kwargs(architecture, config, arm)
    training.validate_response_contract(stage, kwargs, config, arm)
    forged = dict(stage)
    forged["response_features.tau_seconds"] = (
        torch.tensor(config["E2仅LF原始训练曲线字典"]["tau秒"], dtype=torch.float64)
        if arm == "E1" else torch.tensor([2.0, 10.0, 50.0, 200.0], dtype=torch.float64)
    )
    if arm == "E2":
        forged["response_features.tau_seconds"] = torch.tensor([2.0, 10.0, 50.0, 200.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="响应|tau|buffer|来源"):
        training.validate_response_contract(forged, kwargs, config, arm)
    if arm != "E0":
        changed = dict(kwargs)
        changed["response_tau_seconds"] = [1.0, 5.0, 25.0, 100.0]
        with pytest.raises(ValueError, match="响应|tau|buffer|来源"):
            training.validate_response_contract(stage, changed, config, arm)


def test_real_cpu_three_arm_one_epoch_and_e1_two_epoch_resume_without_test_temperatures(
    tmp_path, monkeypatch,
) -> None:
    training = _training_module()
    ir_splits, sensor_splits = [], []
    original_ir = multifidelity.load_processed_ir_observations
    original_sensor = multifidelity.load_canonical_sensor_observations

    def legal_ir(split=None):
        ir_splits.append(split)
        assert split in ("train", "validation"), "任06禁止旧test_Data温度标签"
        return original_ir(split)

    def legal_sensor(*, split):
        sensor_splits.append(split)
        assert split in ("train", "validation"), "任06禁止旧test_Data环温标签"
        return original_sensor(split=split)

    monkeypatch.setattr(multifidelity, "load_processed_ir_observations", legal_ir)
    monkeypatch.setattr(multifidelity, "load_canonical_sensor_observations", legal_sensor)
    outputs = {}
    initials = {}
    source, architecture, config = training.validate_task06_source()
    lf_sha = training._actual_lf_tensor_sha256(source["model_state"])
    for arm in ("E0", "E1", "E2"):
        output = tmp_path / f"任06_{arm}_CPU诊断"
        summary = training.run_task06_time_features(
            arm=arm, output_directory=output, budget_epochs=600, session_epoch_limit=1,
            diagnostic_only=True, device_name="cpu",
        )
        outputs[arm] = output
        assert summary["运行资格"].startswith("短诊断")
        assert summary["本臂实际完成轮次"] == 1
        assert summary["LF共同真实张量SHA256"] == lf_sha
        assert summary["LF仿真训练温度进入HF监督"] is False
        assert not (output / "阶段_HF先导末.pt").exists()
        initials[arm] = torch.load(output / "阶段_初始.pt", map_location="cpu", weights_only=False)
        recent = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
        assert initials[arm]["optimizer_state"]["state"] == {}
        assert recent["optimizer_state"]["state"]
        assert initials[arm]["epoch"] == 0 and recent["epoch"] == 1
        assert recent["metadata"]["任06预登记配置SHA256"] == sha256_file(REGISTERED)
        assert recent["metadata"]["LF共同真实张量SHA256"] == lf_sha
        assert recent["metadata"]["累计实际消耗"]["HF观测优化步"] == 15
        assert recent["metadata"]["累计实际消耗"]["物理优化步"] == 1
        assert recent["metadata"]["累计实际消耗"]["LF仿真温度训练点"] == 0
        for name in ("阶段_观测最佳.pt", "阶段_物理最佳.pt"):
            best = torch.load(output / name, map_location="cpu", weights_only=False)
            assert best["training_state_schema_version"] == 1
            assert best["stage"] == "correction_time_features"
            assert set(best["random_state"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
        record = [json.loads(line) for line in (output / "training.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()]
        assert len(record) == 1 and record[0]["epoch"] == 1
        assert record[0]["HF观测优化步"] == 15 and record[0]["物理优化步"] == 1
        assert record[0]["物理配点"] == 256
        assert record[0]["HF训练观测点"] > 0
        assert record[0]["LF仿真温度训练点"] == 0
        assert record[0]["LF实际冻结参数数"] == 33
        assert set(record[0]["LF合法验证材料RMSE_摄氏度"]) == {"Cu", "SiC"}
        view = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        _, _, loaded = multifidelity._load_multifidelity_model(output / "best.pt", torch.device("cpu"))
        assert loaded.response_features is (None if arm == "E0" else loaded.response_features)
        if arm != "E0":
            assert torch.equal(loaded.response_features.tau_seconds,
                               torch.tensor(training._registered_tau(config, arm), dtype=torch.float64))
            assert view["correction_model_kwargs"]["response_tau_seconds"] == list(
                training._registered_tau(config, arm),
            )
    assert ir_splits and set(ir_splits) == {"train", "validation"}
    assert sensor_splits and set(sensor_splits) == {"train", "validation"}
    assert all(torch.equal(initials["E0"]["random_state"]["torch_cpu"],
                           initials[arm]["random_state"]["torch_cpu"]) for arm in ("E1", "E2"))
    assert all(random.getstate() is not None for _ in ("E0", "E1", "E2"))
    e1_output = outputs["E1"]
    resumed = training.run_task06_time_features(
        arm="E1", output_directory=e1_output, budget_epochs=600, session_epoch_limit=1,
        diagnostic_only=True, device_name="cpu",
        resume_training_checkpoint=e1_output / "阶段_最近.pt",
    )
    assert resumed["本臂实际完成轮次"] == 2
    complete = torch.load(e1_output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert complete["epoch"] == 2 and complete["metadata"]["累计实际消耗"]["HF观测优化步"] == 30
    assert [json.loads(line)["epoch"] for line in (e1_output / "training.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()] == [1, 2]
    with pytest.raises(ValueError, match="短诊断|正式"):
        training.run_task06_time_features(
            arm="E1", output_directory=e1_output, budget_epochs=600,
            session_epoch_limit=600, diagnostic_only=False, device_name="cpu",
            resume_training_checkpoint=e1_output / "阶段_最近.pt",
        )


def test_short_budget_cannot_create_formal_task06_candidate(tmp_path) -> None:
    training = _training_module()
    output = tmp_path / "未授权的半程正式输出"
    with pytest.raises(ValueError, match="600|正式|诊断"):
        training.run_task06_time_features(
            arm="E1", output_directory=output, budget_epochs=600,
            session_epoch_limit=1, diagnostic_only=False, device_name="cpu",
        )
    assert not output.exists()


def test_cli_short_formal_budget_fails_before_creating_output(tmp_path, monkeypatch) -> None:
    script = PROJECT_ROOT / "scripts/23_run_task06_time_features.py"
    output = tmp_path / "任06_CLI不合法半程"
    monkeypatch.setattr(sys, "argv", [str(script), "--arm", "E2", "--output", str(output),
                                         "--device", "cpu", "--session-epoch-limit", "1"])
    with pytest.raises(ValueError, match="正式|600|诊断"):
        runpy.run_path(str(script), run_name="__main__")
    assert not output.exists()


def test_best_view_reload_uses_registered_real_tau_and_rejects_e1_disguised_as_e2(tmp_path) -> None:
    training = _training_module()
    output = tmp_path / "任06_E1_真tau重载"
    training.run_task06_time_features(
        arm="E1", output_directory=output, budget_epochs=600,
        session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
    )
    original, loaded = training.load_task06_model_view(output / "best.pt", arm="E1")
    assert loaded.response_features is not None
    assert torch.equal(loaded.response_features.tau_seconds,
                       torch.tensor([2.0, 10.0, 50.0, 200.0], dtype=torch.float64))
    assert all(torch.equal(value, loaded.state_dict()[name])
               for name, value in original["model_state"].items())
    original["model_state"]["response_features.tau_seconds"] = torch.tensor(
        [0.8341738692346146, 2.456793356991631, 11.306550331610655, 14.206694104729392],
        dtype=torch.float64,
    )
    forged = tmp_path / "best_E1冒用E2.pt"
    torch.save(original, forged)
    with pytest.raises(ValueError, match="响应|tau|buffer|来源"):
        training.load_task06_model_view(forged, arm="E1")


def test_source_registration_drift_rejected_before_any_task06_output(tmp_path, monkeypatch) -> None:
    training = _training_module()
    actual_sha = training.sha256_file

    def taint_registered_bytes(path):
        if Path(path).resolve() == REGISTERED.resolve():
            return "预登记任06配置SHA不匹配"
        return actual_sha(path)

    monkeypatch.setattr(training, "sha256_file", taint_registered_bytes)
    output = tmp_path / "任06登记漂移禁止入场"
    with pytest.raises(ValueError, match="配置.*SHA|登记"):
        training.run_task06_time_features(
            arm="E2", output_directory=output, budget_epochs=600,
            session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
        )
    assert not output.exists()


def test_recent_e1_state_cannot_disguise_e2_tau_before_resume(tmp_path) -> None:
    training = _training_module()
    output = tmp_path / "任06_E1被换E2tau"
    training.run_task06_time_features(
        arm="E1", output_directory=output, budget_epochs=600,
        session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
    )
    latest = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    original_report = (output / "阶段报告.json").read_bytes()
    latest["model_state"]["response_features.tau_seconds"] = torch.tensor(
        [0.8341738692346146, 2.456793356991631, 11.306550331610655, 14.206694104729392],
        dtype=torch.float64,
    )
    torch.save(latest, output / "阶段_最近.pt")
    with pytest.raises(ValueError, match="响应|tau|buffer|来源"):
        training.run_task06_time_features(
            arm="E1", output_directory=output, budget_epochs=600,
            session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
            resume_training_checkpoint=output / "阶段_最近.pt",
        )
    assert (output / "阶段报告.json").read_bytes() == original_report
    assert [json.loads(line)["epoch"] for line in (output / "training.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()] == [1]


@pytest.mark.parametrize("forgery", ["non_tau_width", "registered_tau_lineage"])
def test_existing_best_view_cannot_disguise_kwargs_or_tau_lineage_before_resume(
    tmp_path, forgery,
) -> None:
    training = _training_module()
    output = tmp_path / "任06模型视图非tau结构伪造拒绝续跑"
    training.run_task06_time_features(
        arm="E1", output_directory=output, budget_epochs=600,
        session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
    )
    latest_file = output / "阶段_最近.pt"
    view_file = output / "best.pt"
    latest = torch.load(latest_file, map_location="cpu", weights_only=False)
    view = torch.load(view_file, map_location="cpu", weights_only=False)
    assert latest["metadata"]["观测最佳任06轮次"] == 1
    assert view["epoch"] == 421
    assert training._deep_equal(view["model_state"], latest["model_state"])
    assert view["validation_selection_score_c"] == latest["metadata"]["观测最佳选分_摄氏度"]
    assert view["correction_model_kwargs"]["response_tau_seconds"] == [2.0, 10.0, 50.0, 200.0]
    assert view["任06新HF模型视图来源"]["响应tau秒"] == [2.0, 10.0, 50.0, 200.0]
    if forgery == "non_tau_width":
        view["correction_model_kwargs"]["width"] = 129
    else:
        view["任06新HF模型视图来源"]["响应tau秒"] = [
            0.8341738692346146, 2.456793356991631,
            11.306550331610655, 14.206694104729392,
        ]
    torch.save(view, view_file)
    prior_best = view_file.read_bytes()
    prior_latest = latest_file.read_bytes()
    prior_log = (output / "training.jsonl").read_bytes()
    prior_report = (output / "阶段报告.json").read_bytes()
    with pytest.raises(ValueError, match="模型视图|kwargs|参数|来源"):
        training.run_task06_time_features(
            arm="E1", output_directory=output, budget_epochs=600,
            session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
            resume_training_checkpoint=latest_file,
        )
    assert view_file.read_bytes() == prior_best
    assert latest_file.read_bytes() == prior_latest
    assert (output / "training.jsonl").read_bytes() == prior_log
    assert (output / "阶段报告.json").read_bytes() == prior_report


def test_latest_not_committed_keeps_initial_best_and_archives_uncommitted_tail(
    tmp_path, monkeypatch,
) -> None:
    training = _training_module()
    output = tmp_path / "任06最近未提交"
    real_save = training.save_training_state

    def break_latest(path, model, optimizer, **kwargs):
        if Path(path).name == "阶段_最近.pt" and kwargs["epoch"] == 1:
            raise RuntimeError("注入任06最近状态未提交")
        return real_save(path, model, optimizer, **kwargs)

    monkeypatch.setattr(training, "save_training_state", break_latest)
    with pytest.raises(RuntimeError, match="最近状态未提交"):
        training.run_task06_time_features(
            arm="E1", output_directory=output, budget_epochs=600,
            session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
        )
    assert not (output / "阶段_最近.pt").exists()
    assert torch.load(output / "best.pt", map_location="cpu", weights_only=False)["epoch"] == 420
    assert [json.loads(line)["epoch"] for line in (output / "training.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()] == [1]
    monkeypatch.setattr(training, "save_training_state", real_save)
    training.run_task06_time_features(
        arm="E1", output_directory=output, budget_epochs=600,
        session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
        resume_training_checkpoint=output / "阶段_初始.pt",
    )
    assert torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)["epoch"] == 1
    assert [json.loads(line)["epoch"] for line in (output / "training.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()] == [1]
    assert list(output.glob("training_中断尾行_*.jsonl"))


def test_committed_latest_repairs_both_complete_best_states_and_view_from_real_epoch(
    tmp_path, monkeypatch,
) -> None:
    training = _training_module()
    output = tmp_path / "任06最近提交但最佳未提交"
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    original_correction = source["model_state"]["correction.8.weight"]
    real_guardrails = training.evaluate_schedule_guardrails

    def guarantee_physical_best(model, *args, **kwargs):
        audit = real_guardrails(model, *args, **kwargs)
        actual = model.state_dict()["correction.8.weight"].detach().cpu()
        audit["独立局部物理损失"]["physics_total"] = (
            0.5 if torch.equal(actual, original_correction) else 0.01
        )
        return audit

    monkeypatch.setattr(training, "evaluate_schedule_guardrails", guarantee_physical_best)
    real_save = training.save_training_state

    def break_observed(path, model, optimizer, **kwargs):
        if Path(path).name == "阶段_观测最佳.pt" and kwargs["epoch"] == 1:
            raise RuntimeError("注入任06已提交最近但最佳未保存")
        return real_save(path, model, optimizer, **kwargs)

    monkeypatch.setattr(training, "save_training_state", break_observed)
    with pytest.raises(RuntimeError, match="最佳未保存"):
        training.run_task06_time_features(
            arm="E1", output_directory=output, budget_epochs=600,
            session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
        )
    latest = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert latest["epoch"] == latest["metadata"]["观测最佳任06轮次"] == 1
    assert latest["metadata"]["物理最佳任06轮次"] == 1
    assert torch.load(output / "阶段_观测最佳.pt", map_location="cpu", weights_only=False)["epoch"] == 0
    assert torch.load(output / "阶段_物理最佳.pt", map_location="cpu", weights_only=False)["epoch"] == 0
    assert torch.load(output / "best.pt", map_location="cpu", weights_only=False)["epoch"] == 420
    monkeypatch.setattr(training, "save_training_state", real_save)
    real_loader = training.DataLoader

    def stop_before_next_epoch(*args, **kwargs):
        if kwargs.get("shuffle") is True:
            raise RuntimeError("注入任06最佳恢复完成后下一轮未开始")
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(training, "DataLoader", stop_before_next_epoch)
    with pytest.raises(RuntimeError, match="下一轮未开始"):
        training.run_task06_time_features(
            arm="E1", output_directory=output, budget_epochs=600,
            session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
            resume_training_checkpoint=output / "阶段_最近.pt",
        )
    observed = torch.load(output / "阶段_观测最佳.pt", map_location="cpu", weights_only=False)
    physical = torch.load(output / "阶段_物理最佳.pt", map_location="cpu", weights_only=False)
    view = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert observed["epoch"] == physical["epoch"] == 1 and view["epoch"] == 421
    assert training._deep_equal(latest["model_state"], observed["model_state"])
    assert training._deep_equal(latest["model_state"], physical["model_state"])
    assert training._deep_equal(latest["model_state"], view["model_state"])
    assert training._deep_equal(latest["optimizer_state"], observed["optimizer_state"])
    assert training._deep_equal(latest["random_state"], physical["random_state"])
    assert view["任06新HF模型视图来源"]["运行资格"] == latest["metadata"]["运行资格"]
    assert torch.equal(torch.get_rng_state(), latest["random_state"]["torch_cpu"])
    assert latest["metadata"]["累计实际消耗"]["HF观测优化步"] == 15
    assert [json.loads(line)["epoch"] for line in (output / "training.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()] == [1]


def test_best_repair_preflights_older_physical_score_before_any_sidecar_write(
    tmp_path, monkeypatch,
) -> None:
    training = _training_module()
    output = tmp_path / "任06历史物理评分伪装禁止部分恢复"
    original_correction = torch.load(SOURCE, map_location="cpu", weights_only=False)[
        "model_state"
    ]["correction.8.weight"]
    real_guardrails = training.evaluate_schedule_guardrails

    def keep_initial_physical_best(model, *args, **kwargs):
        audit = real_guardrails(model, *args, **kwargs)
        actual = model.state_dict()["correction.8.weight"].detach().cpu()
        audit["独立局部物理损失"]["physics_total"] = (
            0.5 if torch.equal(actual, original_correction) else 0.8
        )
        return audit

    monkeypatch.setattr(training, "evaluate_schedule_guardrails", keep_initial_physical_best)
    training.run_task06_time_features(
        arm="E1", output_directory=output, budget_epochs=600,
        session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
    )
    latest_file = output / "阶段_最近.pt"
    observed_file = output / "阶段_观测最佳.pt"
    physical_file = output / "阶段_物理最佳.pt"
    latest = torch.load(latest_file, map_location="cpu", weights_only=False)
    assert latest["metadata"]["观测最佳任06轮次"] == 1
    assert latest["metadata"]["物理最佳任06轮次"] == 0
    torch.save(torch.load(output / "阶段_初始.pt", map_location="cpu", weights_only=False),
               observed_file)
    earlier_physical = torch.load(physical_file, map_location="cpu", weights_only=False)
    earlier_physical["metadata"]["物理最佳独立损失"] = 0.55
    torch.save(earlier_physical, physical_file)
    latest["metadata"]["物理最佳独立损失"] = 0.55
    latest["metadata"]["已提交旧阶段_物理最佳.ptSHA256"] = sha256_file(physical_file)
    torch.save(latest, latest_file)
    before_observed = observed_file.read_bytes()
    before_view = (output / "best.pt").read_bytes()
    before_report = (output / "阶段报告.json").read_bytes()
    with pytest.raises(ValueError, match="评分|实际|物理|不一致"):
        training.run_task06_time_features(
            arm="E1", output_directory=output, budget_epochs=600,
            session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
            resume_training_checkpoint=latest_file,
        )
    assert observed_file.read_bytes() == before_observed
    assert (output / "best.pt").read_bytes() == before_view
    assert (output / "阶段报告.json").read_bytes() == before_report


def test_lost_earlier_actual_best_refuses_resume_without_rewriting_report(tmp_path) -> None:
    training = _training_module()
    output = tmp_path / "任06历史真实最佳丢失"
    for run in range(2):
        training.run_task06_time_features(
            arm="E1", output_directory=output, budget_epochs=600,
            session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
            **({"resume_training_checkpoint": output / "阶段_最近.pt"} if run else {}),
        )
    latest = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    assert latest["epoch"] == 2 and latest["metadata"]["观测最佳任06轮次"] == 1
    assert latest["metadata"]["已提交旧阶段_观测最佳.ptSHA256"] == sha256_file(
        output / "阶段_观测最佳.pt",
    )
    old_view = (output / "best.pt").read_bytes()
    old_report = (output / "阶段报告.json").read_bytes()
    original = output / "阶段_观测最佳_原件保留.pt"
    (output / "阶段_观测最佳.pt").rename(original)
    with pytest.raises(ValueError, match="历史|原件|AdamW"):
        training.run_task06_time_features(
            arm="E1", output_directory=output, budget_epochs=600,
            session_epoch_limit=1, diagnostic_only=True, device_name="cpu",
            resume_training_checkpoint=output / "阶段_最近.pt",
        )
    assert not (output / "阶段_观测最佳.pt").exists() and original.is_file()
    assert (output / "best.pt").read_bytes() == old_view
    assert (output / "阶段报告.json").read_bytes() == old_report
