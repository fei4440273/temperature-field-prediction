"""任-04独立工程能量门禁：真实训练工件须先于数值积分通过核对。"""

from __future__ import annotations

import copy
import importlib
import json
import random
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
import yaml

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.train.common import CONFIG_FILES


REGISTERED = PROJECT_ROOT / "研究记录/任务04_联合微调/有效运行配置.yaml"
OFFICIAL = "正式同源200轮候选；仍须两臂验收后决定采用"
SOURCE_SHA = load_yaml(str(REGISTERED))["起点状态_SHA256"]
CONFIG_SHA = sha256_file(REGISTERED)
SOURCE_FULL = torch.load(
    PROJECT_ROOT / load_yaml(str(REGISTERED))["起点状态"], map_location="cpu", weights_only=False,
)
SOURCE_LF = {
    name: tensor for name, tensor in SOURCE_FULL["model_state"].items()
    if name.startswith("low_fidelity_model.")
}
SOURCE_LF_PARAMETERS = {
    name for name in SOURCE_FULL["parameter_requires_grad"] if name.startswith("low_fidelity_model.")
}
SOURCE_SNAPSHOT = (PROJECT_ROOT / load_yaml(str(REGISTERED))["起点状态"]).parent / "config_snapshot"


def _subject():
    assert (PROJECT_ROOT / "scripts/21_audit_task04_energy.py").is_file(), "任-04能量审计入口尚未实现"
    return importlib.import_module("scripts.21_audit_task04_energy")


def _report(run: Path, arm: str = "冻结", *, epochs: int = 200, qualification: str = OFFICIAL) -> None:
    consumption = {
        "HF训练观测点": epochs * 29_593,
        "HF训练传感器点": epochs * 15 * 45,
        "LF仿真训练点": epochs * 122_880,
        "物理配点": epochs * 256,
        "HF观测优化步": epochs * 15,
        "物理优化步": epochs,
    }
    (run / "阶段报告.json").write_text(json.dumps({
        "状态": "完成同预算200轮，策略采用须与另一臂比较且主代理核对",
        "运行臂": arm,
        "运行资格": qualification,
        "本臂实际完成轮次": epochs,
        "原预登记预算": 200,
        "源校正末SHA256": SOURCE_SHA,
        "任04预登记配置SHA256": CONFIG_SHA,
        "观测最佳任04轮次": 7,
        "物理最佳任04轮次": 12,
        "真实最后状态": "阶段_HF续训冻结LF末.pt" if arm == "冻结" else "阶段_联合末.pt",
        "旧test_Data温度读取": False,
        "累计实际消耗": consumption,
    }, ensure_ascii=False), encoding="utf-8")


def _logs(run: Path, *, arm: str = "冻结", duplicate_epoch: bool = False, missing_physics: bool = False) -> None:
    rows = []
    for epoch in range(1, 201):
        rows.append({
            "epoch": 199 if duplicate_epoch and epoch == 200 else epoch,
            "任04轮次": epoch,
            "源校正实际轮次": 300,
            "运行臂": arm,
            "HF观测优化步": 15,
            "物理优化步": 0 if missing_physics and epoch == 105 else 1,
            "物理配点": 256,
            "HF训练观测点": 29_593,
            "LF仿真回放训练点": 122_880,
            "LF仿真回放Cu点": 82_100,
            "LF仿真回放SiC点": 40_780,
            "LF回放监督模式": "low",
            "LF回放反向更新": arm == "有限解冻",
            "LF冻结权重符合源状态": True,
            "LF其余权重实际变化张量数": 0,
            "累计实际消耗": {
                "HF训练观测点": epoch * 29_593,
                "HF训练传感器点": epoch * 15 * 45,
                "LF仿真训练点": epoch * 122_880,
                "物理配点": epoch * 256,
                "HF观测优化步": epoch * 15,
                "物理优化步": epoch,
            },
        })
    (run / "training.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
    )


def _stage(run: Path, name: str, *, epoch: int, arm: str = "冻结", model_value: float = 1.0) -> None:
    step = 4800 + epoch * 16
    model_state = copy.deepcopy(SOURCE_LF)
    model_state["correction.weight"] = torch.tensor([model_value])
    if arm == "有限解冻" and epoch > 0:
        for parameter_name in SOURCE_LF_PARAMETERS:
            if "projection." in parameter_name:
                model_state[parameter_name] = model_state[parameter_name] + 1e-5 * epoch
    low_names = SOURCE_LF_PARAMETERS
    allowed = {name for name in low_names if "projection." in name} if arm == "有限解冻" else set()
    optimizer_groups = [{"params": [0], "lr": 1e-4}]
    hf_momentum = {
        "step": torch.tensor(float(step)), "exp_avg": torch.zeros(1), "exp_avg_sq": torch.zeros(1),
    }
    optimizer_state = {0: hf_momentum}
    if arm == "有限解冻":
        optimizer_groups.append({"params": [1, 2, 3, 4], "lr": 1e-5})
        optimizer_state.update({item: copy.deepcopy(hf_momentum) for item in (1, 2, 3, 4)})
    metadata = {
        "源校正末SHA256": SOURCE_SHA,
        "任04预登记配置SHA256": CONFIG_SHA,
        "源校正末实际轮次": 300,
        "任04运行臂": arm,
        "运行资格": OFFICIAL,
        "观测最佳任04轮次": 7,
        "物理最佳任04轮次": 12 if epoch >= 12 else 0,
        "LF允许训练投影参数": sorted(allowed),
        "LF冻结参数": sorted(low_names - allowed),
        "累计实际消耗": {
            "HF训练观测点": epoch * 29_593,
            "HF训练传感器点": epoch * 15 * 45,
            "LF仿真训练点": epoch * 122_880,
            "物理配点": epoch * 256,
            "HF观测优化步": epoch * 15,
            "物理优化步": epoch,
        },
    }
    torch.save({
        "training_state_schema_version": 1,
        "stage": "joint", "epoch": epoch,
        "budget": {"任04联合续训轮次": 200},
        "metadata": metadata, "model_state": model_state,
        "parameter_requires_grad": {
            name: name.startswith("correction.") or name in allowed
            for name in model_state if name in SOURCE_LF_PARAMETERS or name.startswith("correction.")
        },
        "optimizer_state": {
            "param_groups": optimizer_groups, "state": optimizer_state,
        },
        "random_state": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": None,
        },
    }, run / name)


def _completed_run(run: Path, *, arm: str = "冻结", best_model_drift: bool = False) -> None:
    run.mkdir()
    _report(run, arm=arm)
    _logs(run, arm=arm)
    _stage(run, "阶段_观测最佳.pt", epoch=7, arm=arm)
    _stage(run, "阶段_物理最佳.pt", epoch=12, arm=arm)
    _stage(run, "阶段_训练末.pt", epoch=200, arm=arm, model_value=2.0)
    _stage(
        run, "阶段_HF续训冻结LF末.pt" if arm == "冻结" else "阶段_联合末.pt",
        epoch=200, arm=arm, model_value=2.0,
    )
    best_full = torch.load(run / "阶段_观测最佳.pt", map_location="cpu", weights_only=False)
    best_state = best_full["model_state"]
    best_parameters = best_full["parameter_requires_grad"]
    best = {
        "method": "multifidelity_correction", "epoch": 307,
        "provenance": {"test_labels_consumed": False},
        "model_state": copy.deepcopy(best_state),
        "任04新模型视图来源": {
            "源校正末SHA256": SOURCE_SHA,
            "任04预登记配置SHA256": CONFIG_SHA,
            "任04本阶段实际轮次": 7,
            "运行资格": OFFICIAL,
            "旧test_Data温度标签读取": False,
            "LF权重冻结名单": sorted(
                name for name in best_parameters if name.startswith("low_fidelity_model.")
                and (arm == "冻结" or "projection." not in name)
            ),
            "LF权重更新名单": sorted(
                name for name in best_parameters if name.startswith("low_fidelity_model.")
                and arm == "有限解冻" and "projection." in name
            ),
        },
    }
    if best_model_drift:
        best["model_state"]["correction.weight"] += 0.1
    torch.save(best, run / "best.pt")
    snapshot = run / "config_snapshot"
    snapshot.mkdir()
    hashes = {}
    for relative in CONFIG_FILES:
        filename = Path(relative).name
        original = PROJECT_ROOT / relative
        copied = snapshot / filename
        copied.write_bytes(original.read_bytes())
        hashes[relative] = sha256_file(copied)
    (snapshot / "sha256.json").write_text(json.dumps(hashes), encoding="utf-8")
    (snapshot / "resolved_physics.yaml").write_bytes(
        (SOURCE_SNAPSHOT / "resolved_physics.yaml").read_bytes()
    )


def test_short_diagnostic_never_enters_official_energy_audit(tmp_path: Path) -> None:
    run = tmp_path / "任04诊断"
    run.mkdir()
    _report(run, epochs=1, qualification="短诊断；不参与任-04正式200轮采用")
    with pytest.raises(ValueError, match="200轮"):
        _subject().validate_completed_run(run, arm="冻结", config_path=REGISTERED)


def test_empty_training_log_fails_as_incomplete_budget(tmp_path: Path) -> None:
    run = tmp_path / "任04空日志"
    run.mkdir()
    _report(run)
    (run / "training.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="200轮"):
        _subject().validate_completed_run(run, arm="冻结", config_path=REGISTERED)


@pytest.mark.parametrize("mutate, message", [
    ("duplicate_epoch", "轮次"),
    ("missing_physics", "物理优化步"),
])
def test_200_line_log_does_not_hide_broken_epoch_or_budget(tmp_path: Path, mutate: str, message: str) -> None:
    run = tmp_path / "任04异常日志"
    run.mkdir()
    _report(run)
    _logs(run, **{mutate: True})
    with pytest.raises(ValueError, match=message):
        _subject().validate_completed_run(run, arm="冻结", config_path=REGISTERED)


def test_best_model_view_must_match_complete_best_training_state(tmp_path: Path) -> None:
    run = tmp_path / "任04最佳失配"
    _completed_run(run, best_model_drift=True)
    with pytest.raises(ValueError, match="模型张量"):
        _subject().validate_completed_run(run, arm="冻结", config_path=REGISTERED)


def test_real_terminal_and_complete_best_states_remain_separate(tmp_path: Path) -> None:
    run = tmp_path / "任04正常"
    _completed_run(run)
    qualified = _subject().validate_completed_run(run, arm="冻结", config_path=REGISTERED)
    assert qualified["状态文件"]["best"].name == "best.pt"
    assert qualified["状态文件"]["final"].name == "阶段_训练末.pt"
    assert qualified["检查点哈希"]["观测最佳完整状态"] == sha256_file(run / "阶段_观测最佳.pt")
    assert qualified["检查点哈希"]["真实阶段末"] == sha256_file(run / "阶段_训练末.pt")
    assert qualified["审核点数"] == 30
    assert qualified["阶数"] == [16, 64]


def test_joint_arm_requires_projection_optimizer_and_named_true_terminal(tmp_path: Path) -> None:
    run = tmp_path / "任04联合真实完整状态"
    _completed_run(run, arm="有限解冻")
    qualified = _subject().validate_completed_run(run, arm="有限解冻", config_path=REGISTERED)
    assert qualified["检查点哈希"]["本臂专名真实阶段末"] == sha256_file(run / "阶段_联合末.pt")
    stage = torch.load(run / "阶段_联合末.pt", map_location="cpu", weights_only=False)
    stage["optimizer_state"]["param_groups"].pop()
    torch.save(stage, run / "阶段_联合末.pt")
    with pytest.raises(ValueError, match="AdamW参数组"):
        _subject().validate_completed_run(run, arm="有限解冻", config_path=REGISTERED)


def test_physical_best_requires_a_distinct_full_state_and_its_reported_epoch(tmp_path: Path) -> None:
    run = tmp_path / "任04缺少物理最佳完整状态"
    _completed_run(run)
    (run / "阶段_物理最佳.pt").unlink()
    with pytest.raises(FileNotFoundError, match="物理最佳"):
        _subject().validate_completed_run(run, arm="冻结", config_path=REGISTERED)


def test_energy_evidence_writer_preserves_archived_raw_denominator_and_sha(tmp_path: Path) -> None:
    source = PROJECT_ROOT / (
        "研究记录/任务03_低保真精度修复/"
        "任务03_HF旧LF最佳_能量审核_20260915T190937+0800"
    )
    audit = {
        "指标明细": pl.read_csv(source / "指标明细.csv"),
        "物理分解": pl.read_csv(source / "物理分解.csv"),
        "原始能量": [json.loads(line) for line in (source / "原始能量.jsonl").read_text(encoding="utf-8").splitlines()],
        "原始散度": [json.loads(line) for line in (source / "原始散度.jsonl").read_text(encoding="utf-8").splitlines()],
    }
    destination = tmp_path / "任04只读输出"
    source_denominator = audit["指标明细"]["原定义相对平衡分母_瓦"].to_numpy()
    payload = {"中文说明": "源于合法审计工件的合成输出门禁", "原定义相对平衡分母已保持": True}
    files = _subject().write_energy_evidence(destination, audit, payload)
    assert pl.read_csv(destination / "指标明细.csv").height == 30
    assert np.array_equal(
        pl.read_csv(destination / "指标明细.csv")["原定义相对平衡分母_瓦"].to_numpy(),
        source_denominator,
    )
    assert len((destination / "原始能量.jsonl").read_text(encoding="utf-8").splitlines()) == 60
    assert json.loads((destination / "汇总指标.json").read_text(encoding="utf-8"))["原定义相对平衡分母已保持"] is True
    manifest = json.loads((destination / "审计工件SHA256.json").read_text(encoding="utf-8"))
    assert manifest == files
    assert set(files) == {
        "指标明细.csv", "物理分解.csv", "原始能量.jsonl", "原始散度.jsonl", "汇总指标.json",
    }
    assert all(sha256_file(destination / name) == expected for name, expected in files.items())
    with pytest.raises(FileExistsError):
        _subject().write_energy_evidence(destination, audit, payload)


@pytest.mark.parametrize("arm, name, changing, message", [
    ("冻结", "low_fidelity_model.branch_projection.weight", False, "冻结LF"),
    ("有限解冻", "low_fidelity_model.other.weight", False, "其余LF"),
    ("有限解冻", "", True, "末投影"),
])
def test_lf_source_tensor_contract_independently_detects_illegal_drift(
    arm: str, name: str, changing: bool, message: str,
) -> None:
    original = {
        "low_fidelity_model.branch_projection.weight": torch.tensor([1.0]),
        "low_fidelity_model.other.weight": torch.tensor([2.0]),
    }
    proposed = copy.deepcopy(original)
    if name:
        proposed[name] += 1.0
    with pytest.raises(ValueError, match=message):
        _subject().validate_lf_state_against_source(
            original, proposed, arm=arm, require_joint_effect=changing,
        )


@pytest.mark.parametrize("arm, message", [("冻结", "冻结LF"), ("有限解冻", "末投影")])
def test_completed_arm_compares_the_real_lf_source_not_only_terminal_duplicates(
    tmp_path: Path, arm: str, message: str,
) -> None:
    run = tmp_path / f"任04源比较_{arm}"
    _completed_run(run, arm=arm)
    for filename in (
        "阶段_训练末.pt", "阶段_HF续训冻结LF末.pt" if arm == "冻结" else "阶段_联合末.pt",
    ):
        stage = torch.load(run / filename, map_location="cpu", weights_only=False)
        if arm == "冻结":
            stage["model_state"]["low_fidelity_model.branch_projection.weight"] += 0.02
        else:
            for key, original in SOURCE_LF.items():
                if "projection." in key:
                    stage["model_state"][key] = original.clone()
        torch.save(stage, run / filename)
    with pytest.raises(ValueError, match=message):
        _subject().validate_completed_run(run, arm=arm, config_path=REGISTERED)


def test_forged_geometry_and_matching_self_manifest_cannot_define_a_new_physical_audit(tmp_path: Path) -> None:
    run = tmp_path / "任04伪造几何及自持SHA"
    _completed_run(run)
    snapshot = run / "config_snapshot"
    geometry_path = snapshot / "geometry.yaml"
    geometry = yaml.safe_load(geometry_path.read_text(encoding="utf-8"))
    geometry["copper"]["radius_m"] += 0.001
    geometry_path.write_text(yaml.safe_dump(geometry, sort_keys=False), encoding="utf-8")
    manifest_path = snapshot / "sha256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configs/geometry.yaml"] = sha256_file(geometry_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="几何|geometry"):
        _subject().validate_completed_run(run, arm="冻结", config_path=REGISTERED)


def test_forged_resolved_physics_is_rejected_even_with_untouched_raw_boundaries(tmp_path: Path) -> None:
    run = tmp_path / "任04伪造解析名义边界"
    _completed_run(run)
    resolved = run / "config_snapshot/resolved_physics.yaml"
    physics = yaml.safe_load((SOURCE_SNAPSHOT / "resolved_physics.yaml").read_text(encoding="utf-8"))
    physics["values"]["silicon_carbide_emissivity"]["value"] += 0.1
    resolved.write_text(yaml.safe_dump(physics, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="解析名义物理|resolved_physics"):
        _subject().validate_completed_run(run, arm="冻结", config_path=REGISTERED)


def test_training_weights_and_self_manifest_cannot_drift_from_trusted_source(tmp_path: Path) -> None:
    run = tmp_path / "任04伪造HF物理损失与自持SHA"
    _completed_run(run)
    snapshot = run / "config_snapshot"
    training_path = snapshot / "training.yaml"
    training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    training["loss_weights"]["pde"] *= 1.5
    training_path.write_text(yaml.safe_dump(training, sort_keys=False), encoding="utf-8")
    manifest_path = snapshot / "sha256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configs/training.yaml"] = sha256_file(training_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="training.yaml"):
        _subject().validate_completed_run(run, arm="冻结", config_path=REGISTERED)


def test_posthoc_main_ledger_hash_is_not_part_of_locked_operational_settings(tmp_path: Path) -> None:
    run = tmp_path / "任04只更新后记总台账"
    _completed_run(run)
    manifest_path = run / "config_snapshot/sha256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["多保真DeepONet预测精度优化总计划与执行台账.md"] = "事后持续追加、不用于运营来源判定"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    approved = _subject().validate_completed_run(run, arm="冻结", config_path=REGISTERED)
    assert approved["物理快照SHA256"]["resolved_physics.yaml"] == sha256_file(
        SOURCE_SNAPSHOT / "resolved_physics.yaml"
    )
