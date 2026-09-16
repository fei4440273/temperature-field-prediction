"""任-06独立能量审计只接受同源600轮完整候选。"""

from __future__ import annotations

import importlib
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.train import task06_time_features as trainer
from sic_cu.train.common import write_config_snapshot


REGISTERED = PROJECT_ROOT / "研究记录/任务06_时间响应特征/HF三臂先导有效配置.yaml"
FORMAL = "单种子正式同源600轮先导候选；须三臂审计后再决定采用"
LF_METRICS = {
    "Cu": {"node": 2.0313740794199715, "volume": 1.6831936718896254},
    "SiC": {"node": 2.3717187397752584, "volume": 2.048439869762359},
}


def _audit():
    assert (PROJECT_ROOT / "scripts/24_audit_task06_energy.py").is_file(), (
        "任-06独立30点能量审核入口尚未实现"
    )
    return importlib.import_module("scripts.24_audit_task06_energy")


@pytest.mark.parametrize("ros_pythonpath", (False, True))
def test_bare_pinn_script_help_uses_project_auditor_without_pythonpath_injection(
    ros_pythonpath: bool,
) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    if ros_pythonpath:
        environment["PYTHONPATH"] = "/opt/ros/humble/local/lib/python3.10/dist-packages"
    result = subprocess.run(
        ["/home/phl/anaconda3/envs/PINN/bin/python",
         str(PROJECT_ROOT / "scripts/24_audit_task06_energy.py"), "--help"],
        cwd=PROJECT_ROOT, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--run" in result.stdout and "--state" in result.stdout


def test_short_cpu_diagnostic_cannot_enter_formal_energy_audit(tmp_path: Path) -> None:
    run = tmp_path / "任06_E1_CPU诊断"
    run.mkdir()
    with pytest.raises((ValueError, FileNotFoundError), match="600轮|正式|阶段报告"):
        _audit().validate_completed_run(run, arm="E1")


def _synthetic_full_run(directory: Path, arm: str) -> None:
    # Synthetic states exercise provenance guards; they are not a training result.
    source, architecture, config = trainer.validate_task06_source()
    model, optimizer = trainer.fork_task06_model(
        source, architecture, config, arm, torch.device("cpu"),
    )
    lf_sha = trainer._actual_lf_tensor_sha256(source["model_state"])
    kwargs = trainer.task06_model_kwargs(architecture, config, arm)
    directory.mkdir()
    write_config_snapshot(directory)
    hf_parameters = list(model.correction.parameters())
    initial_validation = float(architecture["validation_selection_score_c"])

    def consumption(epoch: int) -> dict[str, int]:
        return {
            "HF训练观测点": 29_593 * epoch,
            "HF训练传感器点": 44_775 * epoch,
            "LF仿真温度训练点": 0,
            "物理配点": 256 * epoch,
            "HF观测优化步": 15 * epoch,
            "物理优化步": epoch,
        }

    def state(epoch: int, *, selected: int, physical: int) -> dict:
        values = {name: value.cpu().clone() for name, value in model.state_dict().items()}
        if epoch:
            values["correction.0.weight"] += epoch / 1_000_000
            if arm != "E0":
                values["correction.0.weight"][:, 6:] += 0.001
        metadata = trainer._stage_metadata(
            config, arm, lf_sha, initial_validation, 2.0 if selected else initial_validation,
            selected, 0.03 if physical else 0.04, physical,
            consumption(epoch), diagnostic_only=False,
        )
        group = optimizer.state_dict()["param_groups"][0]
        moments = {
            index: {
                "step": torch.tensor(float(epoch * 16)),
                "exp_avg": torch.zeros_like(parameter.detach().cpu()),
                "exp_avg_sq": torch.zeros_like(parameter.detach().cpu()),
            } for index, parameter in zip(group["params"], hf_parameters)
        } if epoch else {}
        return {
            "training_state_schema_version": 1,
            "stage": "correction_time_features", "epoch": epoch,
            "model_state": values,
            "parameter_requires_grad": {
                name: parameter.requires_grad for name, parameter in model.named_parameters()
            },
            "optimizer_state": {"param_groups": [group], "state": moments},
            "random_state": {
                "python": random.getstate(), "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": source["random_state"]["torch_cuda"],
            },
            "budget": {"任06HF校正先导轮次": 600}, "metadata": metadata,
        }

    initial = state(0, selected=0, physical=0)
    torch.save(initial, directory / "阶段_初始.pt")
    best = state(120, selected=120, physical=0)
    torch.save(best, directory / "阶段_观测最佳.pt")
    physical = state(180, selected=120, physical=180)
    torch.save(physical, directory / "阶段_物理最佳.pt")
    terminal = state(600, selected=120, physical=180)
    terminal["metadata"]["已提交旧阶段_观测最佳.ptSHA256"] = sha256_file(
        directory / "阶段_观测最佳.pt"
    )
    terminal["metadata"]["已提交旧阶段_物理最佳.ptSHA256"] = sha256_file(
        directory / "阶段_物理最佳.pt"
    )
    torch.save(terminal, directory / "阶段_训练末.pt")
    shutil.copyfile(directory / "阶段_训练末.pt", directory / "阶段_HF先导末.pt")
    model.load_state_dict(best["model_state"], strict=True)
    trainer._model_view(
        directory / "best.pt", architecture, model, kwargs, config, arm, lf_sha,
        epoch=120, score=2.0,
        validation={"顶部": 3.0, "absolute_rmse_c": 1.0, "delta_rmse_c": 1.0},
        diagnostic_only=False,
    )
    records = []
    for epoch in range(1, 601):
        due = epoch % 10 == 0
        records.append({
            "epoch": epoch, "任06轮次": epoch, "运行臂": arm,
            "源任04完整实际阶段轮次": 120,
            "HF观测优化步": 15, "物理优化步": 1, "物理配点": 256,
            "HF训练观测点": 29_593, "HF训练传感器点": 44_775,
            "LF仿真训练来源功率数": 60, "LF仿真温度训练点": 0,
            "LF实际冻结参数数": 33,
            "LF共同真实张量SHA256": lf_sha,
            "HF学习率": 1e-4, "LF学习率": 0.0,
            "HF合法验证选分_摄氏度": (
                2.0 if epoch == 120 else 2.3 if due else None
            ),
            "HF合法验证分模态RMSE_摄氏度": {
                "top": 3.0, "hot": 1.0, "cold": 0.5,
            } if due else None,
            "LF合法验证材料RMSE_摄氏度": LF_METRICS if due else None,
            "独立局部物理损失": {
                "pde": 0.01, "initial": 0.0, "boundary": 0.02,
                "interface": 0.02, "physics_total": 0.03 if epoch == 180 else 0.05,
            } if due else None,
            "累计实际消耗": consumption(epoch),
        })
    (directory / "training.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    (directory / "阶段报告.json").write_text(json.dumps({
        "状态": "真实完成单种子三臂原预算600轮；仍须独立能量审计和任07多种子确认",
        "运行臂": arm, "运行资格": FORMAL,
        "本臂实际完成轮次": 600, "原预登记预算": 600,
        "任06预登记配置SHA256": sha256_file(REGISTERED),
        "源任04第120轮完整状态SHA256": config["共同完整训练状态_SHA256"],
        "LF共同真实张量SHA256": lf_sha,
        "LF仿真训练温度进入HF监督": False,
        "LF合法验证逐材料节点及真实体积RMSE_摄氏度": LF_METRICS,
        "初始HF合法选分_摄氏度": initial_validation,
        "观测最佳HF合法选分_摄氏度": 2.0,
        "观测最佳任06轮次": 120,
        "物理最佳独立损失": 0.03, "物理最佳任06轮次": 180,
        "真实最后状态": "阶段_HF先导末.pt",
        "旧test_Data温度标签读取": False,
        "累计实际消耗": consumption(600),
        "最后一轮门禁证据": records[-1],
    }, ensure_ascii=False), encoding="utf-8")


@pytest.mark.parametrize("arm", ("E0", "E1", "E2"))
def test_each_arm_must_have_complete_600_epoch_best_and_final_states(tmp_path: Path, arm: str) -> None:
    run = tmp_path / f"任06_{arm}_合成完整"
    _synthetic_full_run(run, arm)
    qualified = _audit().validate_completed_run(run, arm=arm)
    assert qualified["审核点数"] == 30
    assert qualified["阶数"] == [16, 64]
    assert qualified["状态文件"]["best"].name == "best.pt"
    assert qualified["状态文件"]["final"].name == "阶段_训练末.pt"
    assert qualified["检查点哈希"]["观测最佳模型视图"] == sha256_file(run / "best.pt")
    assert qualified["检查点哈希"]["真实阶段末"] == sha256_file(run / "阶段_训练末.pt")


@pytest.mark.parametrize("arm", ("E0", "E1", "E2"))
def test_incomplete_600_line_budget_cannot_claim_formal_energy(tmp_path: Path, arm: str) -> None:
    run = tmp_path / f"任06_{arm}_少一轮"
    _synthetic_full_run(run, arm)
    rows = (run / "training.jsonl").read_text(encoding="utf-8").splitlines()
    (run / "training.jsonl").write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="600轮|日志"):
        _audit().validate_completed_run(run, arm=arm)


def test_sample_shrink_fails_even_when_epoch_count_remains_600(tmp_path: Path) -> None:
    run = tmp_path / "任06_E0_缩点"
    _synthetic_full_run(run, "E0")
    rows = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines()]
    rows[301]["HF训练观测点"] -= 1
    (run / "training.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="HF|观测|缩减|样本"):
        _audit().validate_completed_run(run, arm="E0")


@pytest.mark.parametrize("filename", ("阶段_观测最佳.pt", "阶段_物理最佳.pt"))
def test_missing_original_best_full_state_does_not_become_a_600_epoch_success(
    tmp_path: Path, filename: str,
) -> None:
    run = tmp_path / "任06_E1_丢原件"
    _synthetic_full_run(run, "E1")
    (run / filename).unlink()
    with pytest.raises(FileNotFoundError, match="最佳|状态|原件"):
        _audit().validate_completed_run(run, arm="E1")


@pytest.mark.parametrize("arm", ("E0", "E1", "E2"))
def test_tau_buffer_and_reload_kwargs_must_cross_reference_registered_source(
    tmp_path: Path, arm: str,
) -> None:
    run = tmp_path / f"任06_{arm}_伪buffer"
    _synthetic_full_run(run, arm)
    full = torch.load(run / "阶段_观测最佳.pt", map_location="cpu", weights_only=False)
    full["model_state"]["response_features.tau_seconds"] = torch.tensor(
        [1.0, 5.0, 25.0, 100.0], dtype=torch.float64,
    )
    torch.save(full, run / "阶段_观测最佳.pt")
    with pytest.raises(ValueError, match="tau|buffer|响应"):
        _audit().validate_completed_run(run, arm=arm)


def test_cpu_only_state_cannot_be_renamed_as_formal_candidate(tmp_path: Path) -> None:
    run = tmp_path / "任06_E2_CPU伪正式"
    _synthetic_full_run(run, "E2")
    for filename in ("阶段_初始.pt", "阶段_观测最佳.pt", "阶段_物理最佳.pt",
                     "阶段_训练末.pt", "阶段_HF先导末.pt"):
        full = torch.load(run / filename, map_location="cpu", weights_only=False)
        full["random_state"]["torch_cuda"] = None
        torch.save(full, run / filename)
    with pytest.raises(ValueError, match="RNG|CUDA|随机"):
        _audit().validate_completed_run(run, arm="E2")


def test_true_joint_lf_tensor_source_cannot_be_replaced_with_old_lf(tmp_path: Path) -> None:
    run = tmp_path / "任06_E2_实际LF被改"
    _synthetic_full_run(run, "E2")
    full = torch.load(run / "阶段_观测最佳.pt", map_location="cpu", weights_only=False)
    full["model_state"]["low_fidelity_model.trunk_projection.bias"] += 0.01
    torch.save(full, run / "阶段_观测最佳.pt")
    with pytest.raises(ValueError, match="LF|张量"):
        _audit().validate_completed_run(run, arm="E2")


def test_best_model_view_must_be_exactly_the_complete_best_state(tmp_path: Path) -> None:
    run = tmp_path / "任06_E1_伪模型视图"
    _synthetic_full_run(run, "E1")
    view = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    view["model_state"]["correction.0.weight"] += 0.01
    torch.save(view, run / "best.pt")
    with pytest.raises(ValueError, match="最佳|模型视图|张量"):
        _audit().validate_completed_run(run, arm="E1")


def test_named_real_terminal_includes_same_model_adamw_and_four_rng(tmp_path: Path) -> None:
    run = tmp_path / "任06_E0_专名末态伪优化器"
    _synthetic_full_run(run, "E0")
    named = torch.load(run / "阶段_HF先导末.pt", map_location="cpu", weights_only=False)
    parameter_id = named["optimizer_state"]["param_groups"][0]["params"][0]
    named["optimizer_state"]["state"][parameter_id]["exp_avg"].add_(0.1)
    torch.save(named, run / "阶段_HF先导末.pt")
    with pytest.raises(ValueError, match="真实末|AdamW|动量|同一"):
        _audit().validate_completed_run(run, arm="E0")


def test_operational_snapshot_and_self_manifest_cannot_jointly_forge_geometry(tmp_path: Path) -> None:
    run = tmp_path / "任06_E0_伪物理半径"
    _synthetic_full_run(run, "E0")
    snapshot = run / "config_snapshot"
    geometry_file = snapshot / "geometry.yaml"
    geometry = yaml.safe_load(geometry_file.read_text(encoding="utf-8"))
    geometry["copper"]["radius_m"] += 0.001
    geometry_file.write_text(yaml.safe_dump(geometry), encoding="utf-8")
    manifest_file = snapshot / "sha256.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["configs/geometry.yaml"] = sha256_file(geometry_file)
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="geometry|几何|快照"):
        _audit().validate_completed_run(run, arm="E0")


def test_reported_physical_best_loss_must_match_the_original_full_state(tmp_path: Path) -> None:
    run = tmp_path / "任06_E1_物理最佳分数伪造"
    _synthetic_full_run(run, "E1")
    physical_path = run / "阶段_物理最佳.pt"
    physical = torch.load(physical_path, map_location="cpu", weights_only=False)
    physical["metadata"]["物理最佳独立损失"] += 0.01
    torch.save(physical, physical_path)
    for filename in ("阶段_训练末.pt", "阶段_HF先导末.pt"):
        terminal = torch.load(run / filename, map_location="cpu", weights_only=False)
        terminal["metadata"]["已提交旧阶段_物理最佳.ptSHA256"] = sha256_file(physical_path)
        torch.save(terminal, run / filename)
    with pytest.raises(ValueError, match="物理最佳|分数|损失"):
        _audit().validate_completed_run(run, arm="E1")


def test_lf_material_report_must_match_all_due_validation_logs(tmp_path: Path) -> None:
    run = tmp_path / "任06_E2_冻结LF假报体积误差"
    _synthetic_full_run(run, "E2")
    report_path = run / "阶段报告.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["LF合法验证逐材料节点及真实体积RMSE_摄氏度"]["Cu"]["volume"] += 0.5
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="LF|材料|体积|冻结"):
        _audit().validate_completed_run(run, arm="E2")


def test_forged_resolved_physics_rejected_even_if_raw_boundary_matches(tmp_path: Path) -> None:
    run = tmp_path / "任06_E0_伪解析物理"
    _synthetic_full_run(run, "E0")
    resolved = run / "config_snapshot/resolved_physics.yaml"
    values = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    values["values"]["copper_emissivity"]["value"] += 0.1
    resolved.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match="解析名义物理|resolved_physics"):
        _audit().validate_completed_run(run, arm="E0")


def test_selected_hf_macro_score_is_recomputed_before_any_energy_output(tmp_path: Path) -> None:
    run = tmp_path / "任06_E1_伪造HF选分合成臂"
    _synthetic_full_run(run, "E1")
    output = tmp_path / "禁止HF互抄分数的能量输出"
    with pytest.raises(ValueError, match="合法HF|重算|选分"):
        _audit().audit_completed_state(
            run, arm="E1", state="best", output=output, device_name="cpu",
        )
    assert not output.exists()


def test_due_validation_cannot_drop_any_of_top_hot_cold_or_local_terms(tmp_path: Path) -> None:
    run = tmp_path / "任06_E0_隐匿局部失败"
    _synthetic_full_run(run, "E0")
    rows = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines()]
    rows[49]["HF合法验证分模态RMSE_摄氏度"].pop("cold")
    rows[49]["独立局部物理损失"].pop("interface")
    (run / "training.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="HF|模态|物理|局部"):
        _audit().validate_completed_run(run, arm="E0")


def test_early_legal_validation_win_prevents_false_later_observation_best(tmp_path: Path) -> None:
    run = tmp_path / "任06_E1_伪晚期最佳"
    _synthetic_full_run(run, "E1")
    log = run / "training.jsonl"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    rows[9]["HF合法验证选分_摄氏度"] = 1.0
    log.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="观测|全程|最佳"):
        _audit().validate_completed_run(run, arm="E1")


def test_early_legal_physical_win_prevents_false_later_physical_best(tmp_path: Path) -> None:
    run = tmp_path / "任06_E2_伪晚期物理最佳"
    _synthetic_full_run(run, "E2")
    log = run / "training.jsonl"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    rows[9]["独立局部物理损失"]["physics_total"] = 0.005
    log.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="物理|全程|最佳"):
        _audit().validate_completed_run(run, arm="E2")


def test_hf_adamw_betas_cannot_be_forged_with_real_parameters_and_lr(tmp_path: Path) -> None:
    run = tmp_path / "任06_E0_伪造AdamW动量算法"
    _synthetic_full_run(run, "E0")
    initial = run / "阶段_初始.pt"
    full = torch.load(initial, map_location="cpu", weights_only=False)
    full["optimizer_state"]["param_groups"][0]["betas"] = (0.5, 0.8)
    torch.save(full, initial)
    with pytest.raises(ValueError, match="AdamW|优化器|超参"):
        _audit().validate_completed_run(run, arm="E0")


def test_cpu_same_seed_cannot_claim_original_gpu_physical_best_points(tmp_path: Path) -> None:
    run = tmp_path / "任06_E0_CPU伪同点物理分数"
    _synthetic_full_run(run, "E0")
    qualified = _audit().validate_completed_run(run, arm="E0")
    _, model = trainer.load_task06_model_view(
        run / "best.pt", arm="E0", device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="CUDA|GPU|同设备|配点"):
        _audit()._recompute_physical_best(qualified, model, torch.device("cpu"))


def test_recomputed_physical_loss_must_not_accept_forged_reported_score() -> None:
    with pytest.raises(ValueError, match="物理|重算|损失"):
        _audit()._compare_physical_score(0.18, 0.03)
