"""Synthetic, in-memory CPU contracts for the independent Howard LF budget."""

from __future__ import annotations

import copy
import fnmatch
import hashlib
import importlib
import io
import json
import random
import subprocess
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sic_cu.config import PROJECT_ROOT
from sic_cu.models.common import ModelScales


def _formal():
    try:
        return importlib.import_module("sic_cu.train.task11_howard_lf_formal")
    except ModuleNotFoundError as error:
        if error.name != "sic_cu.train.task11_howard_lf_formal":
            raise
        pytest.fail("缺少独立Howard LF正式入口，不能借用MLP方法门禁")


def _coordinates():
    return torch.tensor([[0.01, -0.008, 10.0, 100.0, 0.0],
                         [0.04, -0.004, 100.0, 400.0, 1.0]], dtype=torch.float64)


def _sources():
    return {name: str(number) * 64 for number, name in enumerate(
        ("YAML_SHA256", "TAR_SHA256", "CATALOG_SHA256"), 1)}


def _sha(value):
    if isinstance(value, list):
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in value).encode()
    elif isinstance(value, dict) and "model_state" in value:
        buffer = io.BytesIO()
        torch.save(value, buffer)
        content = buffer.getvalue()
    else:
        content = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()


def _file_bytes(name, value):
    if name.endswith(".pt"):
        buffer = io.BytesIO()
        torch.save(value, buffer)
        return buffer.getvalue()
    if name.endswith(".jsonl"):
        return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in value).encode()
    if name.endswith(".yaml"):
        return yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode()
    if name.endswith(".md"):
        return value.encode()
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode()


def _artifact_hashes(objects):
    return {name: hashlib.sha256(_file_bytes(name, value)).hexdigest() for name, value in objects.items()}


def _synthetic_artifacts(*, finished=True, source_identity=None):
    formal = _formal()
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        teacher = formal.Task11HowardLFTeacher()
        initial_model_state = copy.deepcopy(teacher.state_dict())
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=0.001, weight_decay=1e-6)
        initial_optimizer = copy.deepcopy(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True)
        teacher(_coordinates().float()).square().mean().backward()
        optimizer.step()
        updated_optimizer = copy.deepcopy(optimizer.state_dict())
    scores = [1.0] + [1.00001] * (200 if finished else 1)
    rows = [{"epoch": epoch, "seed": 0,
             "训练阶段": "howard_adapted_low_fidelity",
             "LF训练功率数": 60, "LF合法验证功率数": 10,
             "训练样本暴露": 60 * 8192, "观测优化步": 60,
             "物理优化步": 1, "物理配点": 256,
             "累计训练点": 60 * 8192 * epoch, "累计优化步": 61 * epoch,
             "train_rmse_c": 2.0, "validation_rmse_c": score,
             "validation_mae_c": 0.8, "learning_rate": 0.001,
             "LF原物理损失": {"physics_total": 0.1},
             "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
             "HF训练许可": False}
            for epoch, score in enumerate(scores, 1)]
    identity = dict(source_identity or _sources())
    model_state = copy.deepcopy(teacher.state_dict())

    def stage(epoch, prefix_sha, final=False):
        opt = copy.deepcopy(updated_optimizer if epoch else initial_optimizer)
        for item in opt["state"].values():
            item["step"].fill_(61 * epoch)
        metadata = {**identity, "seed": 0, "LF方法": formal.METHOD,
                    "LF模型构造参数": formal.LF_MODEL_KWARGS,
                    "LF参数量": formal.LF_PARAMETER_COUNT,
                    "模型尺度": asdict(ModelScales()),
                    "当前轮次": epoch, "累计训练点": 60 * 8192 * epoch,
                    "累计优化步": 61 * epoch, "最佳LF验证轮次": 1 if epoch else 0,
                    "最佳LF验证RMSE_摄氏度": 1.0 if epoch else None,
                    "无改善轮次": max(0, epoch - 1), "日志SHA256": prefix_sha,
                    "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
                    "旧LF/HF权重读取": False, "HF训练许可": False}
        if final:
            metadata["结束原因"] = "验证耐心提前停止"
        return {"training_state_schema_version": 1, "stage": "howard_low_fidelity",
                "epoch": epoch, "model_state": copy.deepcopy(model_state if epoch else initial_model_state),
                "optimizer_state": opt, "scheduler_state": None, "sampler_epochs": {},
                "parameter_requires_grad": {name: True for name, _ in teacher.named_parameters()},
                "random_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                                 "torch_cpu": torch.get_rng_state(),
                                 "torch_cuda": [torch.zeros(16, dtype=torch.uint8)]},
                "budget": {"LF轮次": 2000}, "metadata": metadata}

    best = {"schema_version": 1, "method": formal.METHOD, "seed": 0,
            "epoch": 1, "model_state": copy.deepcopy(model_state),
            "lf_subnet_state": copy.deepcopy(teacher.network.state_dict()),
            "model_kwargs": formal.LF_MODEL_KWARGS, "scales": asdict(ModelScales()),
            "LF参数量": formal.LF_PARAMETER_COUNT, "validation_rmse_c": 1.0,
            "任11Howard事前来源": identity,
            "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
            "HF训练许可": False}
    objects = {"training.jsonl": rows, "best.pt": best,
               "阶段_初始.pt": stage(0, _sha([])),
               "阶段_LF观测最佳.pt": stage(1, _sha(rows[:1])),
               "阶段_最近.pt": stage(len(rows), _sha(rows)),
               "事前真实来源登记.json": {**identity, "LF方法": formal.METHOD,
                                          "旧LF/HF权重读取": False, "HF训练许可": False,
                                          "旧固定TEST温度读取": False,
                                          "模拟测试功率温度读取": False}}
    snapshot_names = ["config_snapshot/sha256.json", "config_snapshot/resolved_physics.yaml",
                      "config_snapshot/" + formal.ROOT_LEDGER,
                      *["config_snapshot/" + Path(name).name for name in formal.CONFIG_FILES]]
    for name in formal.CONFIG_FILES:
        objects["config_snapshot/" + Path(name).name] = {"仅内存合成配置": name}
    root_text = (f"| 录-0102 | {formal.ROOT_TOKEN}; status=active; "
                 f"YAML_SHA256={identity['YAML_SHA256']}; TAR_SHA256={identity['TAR_SHA256']}; "
                 f"CATALOG_SHA256={identity['CATALOG_SHA256']} | 合成 |\n")
    objects["config_snapshot/" + formal.ROOT_LEDGER] = root_text
    objects["config_snapshot/resolved_physics.yaml"] = {"仅内存合成名义物理": True}
    objects["config_snapshot/sha256.json"] = {
        name: hashlib.sha256(_file_bytes("config_snapshot/" + Path(name).name,
                             objects["config_snapshot/" + Path(name).name])).hexdigest()
        for name in formal.CONFIG_FILES}
    objects["config_snapshot/sha256.json"][formal.ROOT_LEDGER] = hashlib.sha256(root_text.encode()).hexdigest()
    snapshot_hashes = {name: hashlib.sha256(_file_bytes(name, objects[name])).hexdigest() for name in snapshot_names}
    if finished:
        objects["阶段_LF训练末.pt"] = stage(len(rows), _sha(rows), final=True)
    boundaries = [200, 201] if finished else [2]
    last = 0
    for number, end in enumerate(boundaries, 1):
        log_name = f"已提交日志_{number:04d}.jsonl"
        recent_name = f"最近提交历史_{number:04d}.pt"
        selected_name = f"观测最佳提交历史_{number:04d}.pt"
        view_name = f"观测最佳模型历史_{number:04d}.pt"
        objects[log_name] = rows[:end]
        objects[recent_name] = stage(end, _sha(rows[:end]))
        objects[selected_name] = copy.deepcopy(objects["阶段_LF观测最佳.pt"])
        objects[view_name] = copy.deepcopy(best)
        objects[f"LF会话收据_{number:04d}.json"] = {
            **identity, "seed": 0, "LF方法": formal.METHOD,
            "起始已提交轮次": last, "本会话实际轮次": end - last,
            "累计实际轮次": end, "正式预算上限轮次": 2000,
            "LF累计真实训练点": 60 * 8192 * end, "LF累计真实优化步": 61 * end,
            "观测最佳LF轮次": 1, "观测最佳LF验证RMSE_摄氏度": 1.0,
            "无改善轮次": end - 1, "本会话真实墙钟秒": 2.5,
            "本会话加载构建与恢复墙钟秒": 0.25,
            "本会话导出与源核验墙钟秒": 0.1,
            "本会话加载训练与导出总墙钟秒": 2.85,
            "峰值真实CUDA显存字节": 12345,
            "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
            "HF训练许可": False,
            "状态": ("已完成Howard适配LF正式训练" if finished and end == len(rows)
                     else "已暂停且完整阶段提交"),
            "日志SHA256": _sha(objects[log_name]),
            "最近阶段SHA256": _sha(objects[recent_name]),
            "观测最佳阶段SHA256": _sha(objects[selected_name]),
            "观测最佳模型SHA256": _sha(objects[view_name]),
            "初始阶段SHA256": _sha(objects["阶段_初始.pt"]),
            "配置快照SHA256": snapshot_hashes,
            "真实训练末阶段SHA256": _sha(objects["阶段_LF训练末.pt"]) if finished and end == len(rows) else None}
        last = end
    if finished:
        objects["metrics.json"] = {"method": formal.METHOD, "seed": 0,
                                   "epochs_completed": len(rows), "best_epoch": 1,
                                   "best_validation_rmse_c": 1.0,
                                   "stopping_reason": "validation_patience",
                                   "training_seconds": 5.0, "peak_gpu_memory_bytes": 12345,
                                   "setup_seconds": 0.5, "export_seconds": 0.2,
                                   "total_session_seconds": 5.7,
                                   "parameter_count": formal.LF_PARAMETER_COUNT,
                                   "LF参数量": formal.LF_PARAMETER_COUNT,
                                   "status": "completed_current_protocol_lf_only",
                                   "configuration": formal.LF_BUDGET,
                                   "source_identity": identity,
                                   "test": None, "旧固定TEST温度读取": False,
                                   "模拟测试功率温度读取": False, "HF训练许可": False}
    return objects, _artifact_hashes(objects)


def test_missing_independent_entry_is_not_silently_replaced_by_mlp():
    assert _formal().ROOT_TOKEN == "TASK11_HOWARD_LF_GATE:v1"


def test_lf_teacher_has_only_exact_modified_subnet_and_kelvin_reconstruction():
    formal = _formal()
    teacher = formal.Task11HowardLFTeacher().double()
    assert teacher.model_kwargs == {"width": 128, "depth": 4, "latent_dim": 128,
                                    "final_activation": False}
    assert teacher.parameter_count() == formal.LF_PARAMETER_COUNT == 100864
    assert all(name.startswith("network.") for name, _ in teacher.named_parameters())
    assert not list(teacher.named_buffers())
    coordinates = _coordinates()
    scaled = teacher.scaler(coordinates)
    expected = 295.15 + 250 * teacher.network(scaled[:, 3:4], scaled[:, [0, 1, 2, 4]])
    torch.testing.assert_close(teacher(coordinates), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="LF|low|HF"):
        teacher(coordinates, fidelity="high")


def test_lf_teacher_network_strictly_migrates_without_hf_initialization():
    formal = _formal()
    modified = importlib.import_module("sic_cu.models.task11_howard_composite")._ModifiedDeepONet
    teacher = formal.Task11HowardLFTeacher().double()
    receiver = modified(1, 4, 128, 4, 128, final_activation=False).double()
    receiver.load_state_dict(teacher.network.state_dict(), strict=True)
    assert all(torch.equal(value, receiver.state_dict()[name])
               for name, value in teacher.network.state_dict().items())


@pytest.mark.parametrize("kwargs", [{"width": 127}, {"depth": 5}, {"latent_dim": 64},
                                    {"final_activation": True},
                                    {"scales": {**asdict(ModelScales()), "power_max_w": 801}}])
def test_lf_teacher_rejects_unregistered_architecture_or_scale(kwargs):
    with pytest.raises(ValueError, match="固定|构造|尺度"):
        _formal().Task11HowardLFTeacher(**kwargs)


def test_budget_only_replaces_method_and_model_kwargs():
    formal = _formal()
    original = importlib.import_module("sic_cu.train.task11_mlp_lf_formal").LF_BUDGET
    assert {key: value for key, value in formal.LF_BUDGET.items()
            if key not in {"method", "model_kwargs"}} == {
                key: value for key, value in original.items() if key != "method"}
    assert formal.REGISTRY == "研究记录/任务11_外部对照/Howard适配独立LF预算前登记.yaml"
    assert formal.LF_BUDGET["model_kwargs"] == formal.LF_MODEL_KWARGS
    assert "src/sic_cu/models/task11_howard_composite.py" in formal.SOURCE_MEMBERS
    assert "src/sic_cu/train/task11_mlp_lf_formal.py" in formal.SOURCE_MEMBERS


@pytest.mark.parametrize("number,token,status,duplicate", [(101, "TASK11_HOWARD_LF_GATE:v1", "active", False),
    (102, "TASK11_NEW_MLP_LF_GATE:v1", "active", False),
    (102, "TASK11_HOWARD_LF_GATE:v1", "inactive", False),
    (102, "TASK11_HOWARD_LF_GATE:v1", "active", True)])
def test_root_requires_unique_new_numbered_howard_identity(monkeypatch, number, token, status, duplicate):
    formal = _formal()
    identity = _sources()
    text = (f"| 录-{number:04d} | {token}; status={status}; "
            f"YAML_SHA256={identity['YAML_SHA256']}; TAR_SHA256={identity['TAR_SHA256']}; "
            f"CATALOG_SHA256={identity['CATALOG_SHA256']} | 合成门禁 |\n")
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: text * (2 if duplicate else 1))
    assert not formal._root_active(Path("合成ROOT"), identity["YAML_SHA256"],
                                   identity["TAR_SHA256"], identity["CATALOG_SHA256"])


def test_valid_root_needs_all_three_exact_hashes(monkeypatch):
    formal = _formal()
    identity = _sources()
    text = ("| 录-0102 | TASK11_HOWARD_LF_GATE:v1; status=active; "
            f"YAML_SHA256={identity['YAML_SHA256']}; TAR_SHA256={identity['TAR_SHA256']}; "
            f"CATALOG_SHA256={identity['CATALOG_SHA256']} | 合成 |\n")
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: text)
    assert formal._root_active(Path("合成ROOT"), **{
        "registry_sha": identity["YAML_SHA256"], "tar_sha": identity["TAR_SHA256"],
        "catalog_sha": identity["CATALOG_SHA256"]})
    assert not formal._root_active(Path("合成ROOT"), identity["YAML_SHA256"], "f" * 64,
                                   identity["CATALOG_SHA256"])


def test_unlocked_training_refuses_before_any_cuda_probe_or_catalog(monkeypatch):
    formal = _formal()

    def forbidden(*args, **kwargs):
        raise AssertionError("未许可不能探测CUDA或读取真实LF catalog")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(formal, "collect_task11_mlp_lf_sources", forbidden)
    with pytest.raises(ValueError, match="普通|登记|项目内|文件"):
        formal.run_task11_howard_lf_formal(
            registry=formal.REGISTRY, registry_sha="1" * 64,
            source_tar="研究记录/任务11_外部对照/不存在合成tar.tar.gz", source_tar_sha="2" * 64,
            catalog="研究记录/任务11_外部对照/不存在合成catalog.json", catalog_sha="3" * 64,
            output=formal.RUN_DIRECTORY + "/正式Howard_LF_seed0", seed=0)


def test_finished_artifacts_reconstruct_best_patience_and_cost_without_cuda(monkeypatch):
    formal = _formal()
    objects, hashes = _synthetic_artifacts()

    def forbidden(*args, **kwargs):
        raise AssertionError("纯CPU完整状态审核不能调用CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    result = formal.audit_task11_howard_lf_artifacts(objects, hashes, seed=0,
                                                   source_identity=_sources(), finished=True)
    assert result["累计轮次"] == 201
    assert result["最佳轮次"] == 1
    assert result["无改善轮次"] == 200
    assert result["真实累计成本秒"] == 5.0
    assert result["HF训练许可"] is False


def test_pause_artifacts_audit_only_own_complete_state():
    formal = _formal()
    objects, hashes = _synthetic_artifacts(finished=False)
    result = formal.audit_task11_howard_lf_artifacts(objects, hashes, seed=0,
                                                   source_identity=_sources(), finished=False)
    assert result["累计轮次"] == 2
    assert result["真实累计成本秒"] == 2.5


@pytest.mark.parametrize("mutation", ["false_best", "false_patience", "false_cost", "false_steps",
    "missing_cuda_rng", "invalid_python_rng", "frozen_parameter", "adamw_betas", "adamw_momentum",
    "missing_optimizer_parameter", "initial_momentum", "final_model", "network_transfer",
    "source", "test_consumed", "nonfinite_score", "short_training", "earlier_stop_hidden",
    "receipt_number_gap", "receipt_hash", "history_log", "history_best"])
def test_hash_consistent_internal_forgery_cannot_pass_finished_chain(mutation):
    formal = _formal()
    objects, hashes = _synthetic_artifacts()
    recent = objects["阶段_最近.pt"]
    if mutation == "false_best":
        objects["best.pt"]["epoch"] = 2
    elif mutation == "false_patience":
        recent["metadata"]["无改善轮次"] = 199
    elif mutation == "false_cost":
        objects["metrics.json"]["training_seconds"] = 2.5
    elif mutation == "false_steps":
        next(iter(recent["optimizer_state"]["state"].values()))["step"].sub_(1)
    elif mutation == "missing_cuda_rng":
        recent["random_state"]["torch_cuda"] = None
    elif mutation == "invalid_python_rng":
        recent["random_state"]["python"] = ("invalid",)
    elif mutation == "frozen_parameter":
        recent["parameter_requires_grad"][next(iter(recent["parameter_requires_grad"]))] = False
    elif mutation == "adamw_betas":
        recent["optimizer_state"]["param_groups"][0]["betas"] = (0.1, 0.2)
    elif mutation == "adamw_momentum":
        next(iter(recent["optimizer_state"]["state"].values()))["exp_avg"].fill_(float("nan"))
    elif mutation == "missing_optimizer_parameter":
        recent["optimizer_state"]["state"].pop(next(iter(recent["optimizer_state"]["state"])))
    elif mutation == "initial_momentum":
        objects["阶段_初始.pt"]["optimizer_state"]["state"] = recent["optimizer_state"]["state"]
    elif mutation == "final_model":
        next(iter(objects["阶段_LF训练末.pt"]["model_state"].values())).add_(1)
    elif mutation == "network_transfer":
        next(iter(objects["best.pt"]["lf_subnet_state"].values())).add_(1)
    elif mutation == "source":
        recent["metadata"]["CATALOG_SHA256"] = "f" * 64
    elif mutation == "test_consumed":
        objects["metrics.json"]["test"] = {"rmse": 0.1}
    elif mutation == "nonfinite_score":
        objects["training.jsonl"][0]["validation_rmse_c"] = float("nan")
    elif mutation == "short_training":
        objects["training.jsonl"].pop()
    elif mutation == "earlier_stop_hidden":
        objects["training.jsonl"][0]["validation_rmse_c"] = 0.1
        objects["training.jsonl"][1]["validation_rmse_c"] = 0.05
        objects["training.jsonl"][2]["validation_rmse_c"] = 0.01
    elif mutation == "receipt_number_gap":
        objects["LF会话收据_0003.json"] = objects.pop("LF会话收据_0002.json")
    elif mutation == "receipt_hash":
        objects["LF会话收据_0001.json"]["最近阶段SHA256"] = "f" * 64
    elif mutation == "history_log":
        objects["已提交日志_0001.jsonl"][0] = {**objects["已提交日志_0001.jsonl"][0], "seed": 4}
    elif mutation == "history_best":
        objects["观测最佳提交历史_0001.pt"]["epoch"] = 2
    hashes = _artifact_hashes(objects)
    with pytest.raises(ValueError):
        formal.audit_task11_howard_lf_artifacts(objects, hashes, seed=0,
                                               source_identity=_sources(), finished=True)


def test_common_full_state_roundtrip_restores_python_numpy_cpu_and_cuda_rng(monkeypatch):
    formal = _formal()
    common = importlib.import_module("sic_cu.train.common")
    teacher = formal.Task11HowardLFTeacher().double()
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=0.001, weight_decay=1e-6)
    optimizer.zero_grad(set_to_none=True)
    teacher(_coordinates()).square().mean().backward()
    optimizer.step()
    random.seed(11)
    np.random.seed(11)
    torch.default_generator.manual_seed(11)
    cuda_state = [torch.zeros(16, dtype=torch.uint8)]
    captured = {}
    actual_save = torch.save
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: cuda_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda value: captured.update(cuda=value))
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(common.os, "replace", lambda *args: None)

    def memory_save(payload, *args, **kwargs):
        buffer = io.BytesIO()
        actual_save(payload, buffer)
        captured["bytes"] = buffer.getvalue()

    monkeypatch.setattr(torch, "save", memory_save)
    common.save_training_state(PROJECT_ROOT / "研究记录/任务11_外部对照/纯内存不落盘.pt",
                               teacher, optimizer, stage="howard_low_fidelity", epoch=1,
                               budget={"LF轮次": 2000})
    expected = (random.random(), np.random.rand(), torch.rand(3))
    with torch.no_grad():
        for parameter in teacher.parameters():
            parameter.zero_()
    state = common.load_training_state(io.BytesIO(captured["bytes"]), teacher, optimizer)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    assert torch.equal(captured["cuda"][0], cuda_state[0])
    assert all(torch.equal(value, teacher.state_dict()[name]) for name, value in state["model_state"].items())
    assert optimizer.state_dict()["state"]


def test_cli_exposes_only_independent_lf_preflight_resume_and_finished_audit():
    runner = PROJECT_ROOT / "scripts/54_run_task11_howard_lf_formal.py"
    result = subprocess.run([sys.executable, "-B", str(runner), "--help"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, "缺少独立Howard LF CLI"
    for argument in ("--registry-sha", "--source-tar-sha", "--catalog-sha", "--preflight-only",
                     "--resume-checkpoint", "--audit-finished", "--session-epoch-limit"):
        assert argument in result.stdout
    for argument in ("--lf-checkpoint", "--hf-arm", "--evaluate-test", "--freeze-catalog",
                     "--start-checkpoint", "--width", "--learning-rate"):
        assert argument not in result.stdout


def _valid_registry():
    formal = _formal()
    identity = _sources()
    return {"schema_version": 1, "阶段": "fresh_howard_adapted_low_fidelity",
            "ROOT门禁标签": formal.ROOT_TOKEN,
            "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
            "HF训练许可": False, "新HF训练许可": False,
            "原文精确三网联合训练复现": False,
            "源码冻结tarSHA256": identity["TAR_SHA256"],
            "真实模拟70源目录SHA256": identity["CATALOG_SHA256"],
            "正式预算": copy.deepcopy(formal.LF_BUDGET),
            "LF模型构造参数": copy.deepcopy(formal.LF_MODEL_KWARGS),
            "LF参数量": formal.LF_PARAMETER_COUNT, "模型尺度": asdict(ModelScales()),
            "最小改善_摄氏度": 0.0001, "AdamW固定参数": copy.deepcopy(formal.ADAMW_CONTRACT),
            "源码普通成员SHA256": {name: "a" * 64 for name in formal.SOURCE_MEMBERS}}


@pytest.mark.parametrize("mutation", ["mlp_token", "hf_permission", "new_hf_permission",
    "budget_bool", "scale", "parameter_count", "tar_sha", "catalog_sha", "source_member", "min_delta"])
def test_registry_requires_exact_independent_lf_contract(mutation):
    formal = _formal()
    registry = _valid_registry()
    if mutation == "mlp_token":
        registry["ROOT门禁标签"] = "TASK11_NEW_MLP_LF_GATE:v1"
    elif mutation == "hf_permission":
        registry["HF训练许可"] = True
    elif mutation == "new_hf_permission":
        registry["新HF训练许可"] = True
    elif mutation == "budget_bool":
        registry["正式预算"]["physics_weight"] = True
    elif mutation == "scale":
        registry["模型尺度"]["power_max_w"] = 799
    elif mutation == "parameter_count":
        registry["LF参数量"] -= 1
    elif mutation == "tar_sha":
        registry["源码冻结tarSHA256"] = "f" * 64
    elif mutation == "catalog_sha":
        registry["真实模拟70源目录SHA256"] = "f" * 64
    elif mutation == "source_member":
        registry["源码普通成员SHA256"].pop(next(iter(registry["源码普通成员SHA256"])))
    elif mutation == "min_delta":
        registry["最小改善_摄氏度"] = 0.0
    with pytest.raises(ValueError):
        formal._validate_registry_contract(registry, source_tar_sha="2" * 64, catalog_sha="3" * 64)


def test_registry_accepts_only_fixed_valid_contract():
    formal = _formal()
    registry = _valid_registry()
    assert formal._validate_registry_contract(registry, source_tar_sha="2" * 64,
                                              catalog_sha="3" * 64) == registry


def test_root_cannot_append_second_conflicting_status(monkeypatch):
    formal = _formal()
    identity = _sources()
    text = ("| 录-0102 | TASK11_HOWARD_LF_GATE:v1; status=active; "
            f"YAML_SHA256={identity['YAML_SHA256']}; TAR_SHA256={identity['TAR_SHA256']}; "
            f"CATALOG_SHA256={identity['CATALOG_SHA256']}; status=inactive | 合成 |\n")
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: text)
    assert not formal._root_active(Path("合成ROOT"), "1" * 64, "2" * 64, "3" * 64)


def test_paths_reject_parent_escape_and_symlinked_ancestors_in_memory(monkeypatch):
    formal = _formal()
    with pytest.raises(ValueError, match="越界|跳转|项目内"):
        formal._project_path(PROJECT_ROOT.parent / "不创建项目外文件.pt", PROJECT_ROOT)
    with pytest.raises(ValueError, match="越界|跳转"):
        formal._project_path("研究记录/../任11不允许跳转.pt", PROJECT_ROOT)
    original = Path.is_symlink
    fake_parent = PROJECT_ROOT / "研究记录/任11仅内存软链父目录"
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == fake_parent or original(path))
    with pytest.raises(ValueError, match="符号链接"):
        formal._project_path(fake_parent / "永不创建.pt", PROJECT_ROOT)


def test_sixty_observation_and_one_nominal_physics_steps_update_every_lf_parameter():
    formal = _formal()
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(111)
        teacher = formal.Task11HowardLFTeacher()
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=0.001, weight_decay=1e-6)
        coordinates = _coordinates().float()
        target = torch.full((2, 1), 295.15)
        for _ in range(60):
            optimizer.zero_grad(set_to_none=True)
            ((teacher(coordinates) - target) / 250).square().mean().backward()
            optimizer.step()
        before = copy.deepcopy(teacher.state_dict())
        components = formal.physics_optimizer_step(
            teacher, optimizer, formal._physics_ready(),
            formal.sample_collocation(8, torch.device("cpu"), seed=111), 1.0)
    assert set(components) == {"pde", "initial", "boundary", "interface", "physics_total"}
    assert all(torch.isfinite(value) for value in components.values())
    assert all(float(state["step"]) == 61 for state in optimizer.state_dict()["state"].values())
    assert len(optimizer.state_dict()["state"]) == len(list(teacher.parameters()))
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in teacher.parameters())
    assert any(not torch.equal(value, before[name]) for name, value in teacher.state_dict().items())


def _memory_preflight(monkeypatch, *, root_number=102, duplicate_archive=False):
    formal = _formal()
    virtual_root = PROJECT_ROOT / "研究记录/任务11_外部对照/Howard仅内存预检项目根"
    files = {virtual_root / name: name.encode() for name in formal.SOURCE_MEMBERS}
    sources = {name: hashlib.sha256(files[virtual_root / name]).hexdigest() for name in formal.SOURCE_MEMBERS}
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        for name in formal.SOURCE_MEMBERS:
            member = tarfile.TarInfo(name)
            member.size = len(files[virtual_root / name])
            archive.addfile(member, io.BytesIO(files[virtual_root / name]))
        if duplicate_archive:
            name = formal.SOURCE_MEMBERS[0]
            member = tarfile.TarInfo(name)
            member.size = len(files[virtual_root / name])
            archive.addfile(member, io.BytesIO(files[virtual_root / name]))
    archive_path = virtual_root / "研究记录/任务11_外部对照/仅合成普通源.tar.gz"
    files[archive_path] = archive_buffer.getvalue()
    tar_sha = hashlib.sha256(files[archive_path]).hexdigest()
    catalog = {"schema_version": 1, "LF训练模拟原件": [{"功率_瓦": index} for index in range(60)],
               "LF合法验证模拟原件": [{"功率_瓦": index} for index in range(60, 70)],
               "模拟测试功率温度读取": False}
    catalog_path = virtual_root / "研究记录/任务11_外部对照/仅合成catalog.json"
    files[catalog_path] = json.dumps(catalog).encode()
    catalog_sha = hashlib.sha256(files[catalog_path]).hexdigest()
    registry = _valid_registry()
    registry["源码普通成员SHA256"] = sources
    registry["源码冻结tarSHA256"] = tar_sha
    registry["真实模拟70源目录SHA256"] = catalog_sha
    registry_path = virtual_root / formal.REGISTRY
    files[registry_path] = yaml.safe_dump(registry, allow_unicode=True).encode()
    registry_sha = hashlib.sha256(files[registry_path]).hexdigest()
    ledger = virtual_root / formal.ROOT_LEDGER
    files[ledger] = (f"| 录-{root_number:04d} | {formal.ROOT_TOKEN}; status=active; "
                     f"YAML_SHA256={registry_sha}; TAR_SHA256={tar_sha}; CATALOG_SHA256={catalog_sha} | 合成 |\n").encode()
    original_read_bytes, original_open = Path.read_bytes, Path.open
    original_is_file, original_exists = Path.is_file, Path.exists
    original_read_text = Path.read_text
    actual_archive_open = tarfile.open
    monkeypatch.setattr(Path, "is_file", lambda path: path in files or original_is_file(path))
    monkeypatch.setattr(Path, "exists", lambda path: path in files or original_exists(path))
    monkeypatch.setattr(Path, "read_bytes", lambda path: files[path] if path in files else original_read_bytes(path))
    monkeypatch.setattr(Path, "read_text", lambda path, *args, **kwargs:
                        files[path].decode() if path in files else original_read_text(path, *args, **kwargs))
    monkeypatch.setattr(Path, "open", lambda path, mode="r", *args, **kwargs:
                        io.BytesIO(files[path]) if path in files and mode == "rb"
                        else original_open(path, mode, *args, **kwargs))
    monkeypatch.setattr(tarfile, "open", lambda path=None, mode="r", **kwargs:
                        actual_archive_open(fileobj=io.BytesIO(files[Path(path)]), mode=mode)
                        if path is not None and Path(path) in files
                        else actual_archive_open(path, mode, **kwargs))
    monkeypatch.setattr(formal, "collect_task11_mlp_lf_sources", lambda: catalog)
    output = virtual_root / formal.RUN_DIRECTORY / "正式Howard_LF_seed0"
    return formal, files, {"registry": registry_path, "registry_sha": registry_sha,
                           "source_tar": archive_path, "source_tar_sha": tar_sha,
                           "catalog": catalog_path, "catalog_sha": catalog_sha,
                           "output": output, "seed": 0, "ledger": ledger, "project_root": virtual_root}


def test_full_memory_preflight_checks_real_bytes_without_cuda_or_output(monkeypatch):
    formal, _, identity = _memory_preflight(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU预检禁止CUDA"))
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: pytest.fail("CPU预检禁止创建目录"))
    result = formal.preflight_task11_howard_lf(**identity)
    assert result["预算"] == formal.LF_BUDGET
    assert result["HF训练许可"] is False


@pytest.mark.parametrize("mutation", ["unactivated_root", "duplicate_archive", "current_source",
                                      "output_seed", "wrong_registry_path", "catalog_drift"])
def test_full_memory_gate_refuses_before_cuda_or_writes(monkeypatch, mutation):
    formal, files, identity = _memory_preflight(monkeypatch, root_number=101 if mutation == "unactivated_root" else 102,
                                               duplicate_archive=mutation == "duplicate_archive")
    if mutation == "current_source":
        files[Path(identity["project_root"]) / formal.SOURCE_MEMBERS[0]] += b"drift"
    elif mutation == "output_seed":
        identity["output"] = Path(identity["output"]).parent / "正式Howard_LF_seed1"
    elif mutation == "wrong_registry_path":
        wrong = Path(identity["registry"]).with_name("替代登记.yaml")
        files[wrong] = files[identity["registry"]]
        identity["registry"] = wrong
    elif mutation == "catalog_drift":
        monkeypatch.setattr(formal, "collect_task11_mlp_lf_sources", lambda: {"漂移": True})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("失败门禁不得探测CUDA"))
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: pytest.fail("失败门禁不得写目录"))
    with pytest.raises(ValueError):
        formal.run_task11_howard_lf_formal(**identity)


def _attach_memory_artifacts(monkeypatch, formal, files, identity, *, finished):
    from sic_cu.data.splits import build_power_splits
    from sic_cu.eval.protocol_checks import REQUIRED_FINGERPRINTS, validate_lf_checkpoint_provenance
    from sic_cu.physics.resolution import resolve_physics_state

    sources = {name: identity[key] for name, key in (("YAML_SHA256", "registry_sha"),
              ("TAR_SHA256", "source_tar_sha"), ("CATALOG_SHA256", "catalog_sha"))}
    objects, _ = _synthetic_artifacts(finished=finished, source_identity=sources)
    fingerprints = {name: "f" * 64 for name in REQUIRED_FINGERPRINTS}
    splits = build_power_splits()
    for name, value in objects.items():
        if name == "best.pt" or fnmatch.fnmatch(name, "观测最佳模型历史_*.pt"):
            value["provenance"] = formal.checkpoint_provenance(
                role="low_fidelity_simulation", train_powers_w=splits.simulation_train,
                validation_powers_w=splits.simulation_validation, fingerprints=fingerprints)
    for number in range(1, 3 if finished else 2):
        receipt = objects[f"LF会话收据_{number:04d}.json"]
        receipt["观测最佳模型SHA256"] = _sha(objects[f"观测最佳模型历史_{number:04d}.pt"])
    output, root = Path(identity["output"]), Path(identity["project_root"])
    objects["config_snapshot/sha256.json"] = {
        name: hashlib.sha256(files[root / name]).hexdigest() for name in formal.CONFIG_FILES}
    objects["config_snapshot/sha256.json"][formal.ROOT_LEDGER] = hashlib.sha256(files[root / formal.ROOT_LEDGER]).hexdigest()
    objects["config_snapshot/" + formal.ROOT_LEDGER] = files[root / formal.ROOT_LEDGER].decode()
    objects["config_snapshot/resolved_physics.yaml"] = resolve_physics_state()
    for name in formal.CONFIG_FILES:
        objects["config_snapshot/" + Path(name).name] = yaml.safe_load(files[root / name])

    def sync():
        for name, value in objects.items():
            files[output / name] = _file_bytes(name, value)
        for name in formal.CONFIG_FILES:
            files[output / "config_snapshot" / Path(name).name] = files[root / name]
        snapshot_hashes = {name: hashlib.sha256(files[output / name]).hexdigest()
                           for name in objects if name.startswith("config_snapshot/")}
        for name, value in objects.items():
            if fnmatch.fnmatch(name, "LF会话收据_*.json"):
                value["配置快照SHA256"] = snapshot_hashes
                files[output / name] = _file_bytes(name, value)

    sync()
    original_is_dir, original_exists, original_glob = Path.is_dir, Path.exists, Path.glob
    actual_load = torch.load
    monkeypatch.setattr(Path, "is_dir", lambda path: path == output or original_is_dir(path))
    monkeypatch.setattr(Path, "exists", lambda path: path == output or original_exists(path))
    monkeypatch.setattr(Path, "glob", lambda path, pattern: iter(sorted(
        candidate for candidate in files if candidate.parent == output and fnmatch.fnmatch(candidate.name, pattern)))
        if path == output else original_glob(path, pattern))
    monkeypatch.setattr(torch, "load", lambda path, *args, **kwargs:
                        actual_load(io.BytesIO(files[path]), *args, **kwargs)
                        if isinstance(path, Path) and path in files else actual_load(path, *args, **kwargs))
    monkeypatch.setattr(formal, "validate_lf_checkpoint_provenance", lambda payload:
                        validate_lf_checkpoint_provenance(payload, splits=splits, fingerprints=fingerprints))
    return objects, sources, sync


def test_full_memory_resume_accepts_only_own_complete_pause_before_cuda(monkeypatch):
    formal, files, identity = _memory_preflight(monkeypatch)
    _attach_memory_artifacts(monkeypatch, formal, files, identity, finished=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("本人CPU续跑预检不得CUDA"))
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: pytest.fail("预检不得写目录"))
    result = formal.preflight_task11_howard_lf(**identity, resume_checkpoint=Path(identity["output"]) / "阶段_最近.pt")
    assert result["seed"] == 0


@pytest.mark.parametrize("mutation", ["other_seed_path", "wrong_internal_seed", "cuda_rng_missing",
    "best_stage_missing", "uncommitted_log", "reused_session_copy", "source_provenance"])
def test_full_memory_resume_rejects_fake_identity_or_incomplete_transaction(monkeypatch, mutation):
    formal, files, identity = _memory_preflight(monkeypatch)
    objects, _, sync = _attach_memory_artifacts(monkeypatch, formal, files, identity, finished=False)
    resume = Path(identity["output"]) / "阶段_最近.pt"
    if mutation == "other_seed_path":
        resume = Path(identity["output"]).parent / "正式Howard_LF_seed1/阶段_最近.pt"
        files[resume] = files[Path(identity["output"]) / "阶段_最近.pt"]
    elif mutation in {"wrong_internal_seed", "cuda_rng_missing"}:
        for name in ("阶段_最近.pt", "最近提交历史_0001.pt"):
            if mutation == "wrong_internal_seed":
                objects[name]["metadata"]["seed"] = 3
            else:
                objects[name]["random_state"]["torch_cuda"] = None
        objects["LF会话收据_0001.json"]["最近阶段SHA256"] = _sha(objects["最近提交历史_0001.pt"])
        sync()
    elif mutation == "best_stage_missing":
        files.pop(Path(identity["output"]) / "阶段_LF观测最佳.pt")
    elif mutation == "uncommitted_log":
        files[Path(identity["output"]) / "training.jsonl"] += b'{"epoch":3}\n'
    elif mutation == "reused_session_copy":
        files[Path(identity["output"]) / "最近提交历史_0002.pt"] = "不可覆盖旧侧件".encode()
    elif mutation == "source_provenance":
        objects["best.pt"]["provenance"]["physics_config_sha256"] = "0" * 64
        sync()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("错误续跑不得探测CUDA"))
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: pytest.fail("错误续跑不得写目录"))
    with pytest.raises((ValueError, RuntimeError)):
        formal.preflight_task11_howard_lf(**identity, resume_checkpoint=resume)


def test_full_memory_finished_audit_never_reads_temperatures_or_cuda(monkeypatch):
    formal, files, identity = _memory_preflight(monkeypatch)
    _attach_memory_artifacts(monkeypatch, formal, files, identity, finished=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("完成CPU审核不得CUDA"))
    monkeypatch.setattr(formal, "load_sampled_points", lambda *args, **kwargs: pytest.fail("完成审核不得读取温度"))
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: pytest.fail("完成审核不得创建工件"))
    result = formal.qualify_task11_howard_lf_source(**identity)
    assert result["累计轮次"] == 201
    assert result["真实累计成本秒"] == 5.0
    assert result["HF训练许可"] is False


def test_formal_seed_directory_cannot_be_overwritten_by_fresh_session(monkeypatch):
    formal, files, identity = _memory_preflight(monkeypatch)
    _attach_memory_artifacts(monkeypatch, formal, files, identity, finished=False)
    with pytest.raises(ValueError, match="覆盖"):
        formal.preflight_task11_howard_lf(**identity)


def test_root_own_record_number_cannot_be_reused_by_other_row(monkeypatch):
    formal = _formal()
    text = ("| 录-0102 | TASK11_HOWARD_LF_GATE:v1; status=active; "
            f"YAML_SHA256={'1' * 64}; TAR_SHA256={'2' * 64}; CATALOG_SHA256={'3' * 64} | 合成 |\n"
            "| 录-0102 | 其他内容重复同编号 | 合成 |\n")
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: text)
    assert not formal._root_active(Path("合成ROOT"), "1" * 64, "2" * 64, "3" * 64)


@pytest.mark.parametrize("mutation", ["initial_seed_weights", "cuda_rng_length", "cost_components"])
def test_forged_seed_initialization_rng_size_or_cost_breakdown_rejected(mutation):
    formal = _formal()
    objects, _ = _synthetic_artifacts()
    if mutation == "initial_seed_weights":
        next(iter(objects["阶段_初始.pt"]["model_state"].values())).add_(0.01)
        for number in (1, 2):
            objects[f"LF会话收据_{number:04d}.json"]["初始阶段SHA256"] = _sha(objects["阶段_初始.pt"])
    elif mutation == "cuda_rng_length":
        for name in ("阶段_最近.pt", "最近提交历史_0002.pt", "阶段_LF训练末.pt"):
            objects[name]["random_state"]["torch_cuda"] = [torch.zeros(64, dtype=torch.uint8)]
        objects["LF会话收据_0002.json"]["最近阶段SHA256"] = _sha(objects["最近提交历史_0002.pt"])
        objects["LF会话收据_0002.json"]["真实训练末阶段SHA256"] = _sha(objects["阶段_LF训练末.pt"])
    else:
        objects["LF会话收据_0001.json"]["本会话加载训练与导出总墙钟秒"] = 2.5
    hashes = _artifact_hashes(objects)
    with pytest.raises(ValueError):
        formal.audit_task11_howard_lf_artifacts(objects, hashes, seed=0,
                                               source_identity=_sources(), finished=True)


def test_training_loop_cost_and_setup_export_total_are_separate():
    formal = _formal()
    objects, hashes = _synthetic_artifacts()
    result = formal.audit_task11_howard_lf_artifacts(objects, hashes, seed=0,
                                                   source_identity=_sources(), finished=True)
    assert result["真实累计成本秒"] == 5.0
    assert result["真实累计加载训练与导出墙钟秒"] == 5.7
    assert result["真实累计加载构建与恢复墙钟秒"] == 0.5
    assert result["真实累计导出与源核验墙钟秒"] == 0.2


def test_qualification_returns_real_config_copy_sha_not_only_hash_list(monkeypatch):
    formal, files, identity = _memory_preflight(monkeypatch)
    _attach_memory_artifacts(monkeypatch, formal, files, identity, finished=True)
    result = formal.qualify_task11_howard_lf_source(**identity)
    for name in formal.CONFIG_FILES:
        copy_name = "config_snapshot/" + Path(name).name
        assert copy_name in result["真实工件SHA256"]
        assert result["真实工件SHA256"][copy_name] == hashlib.sha256(files[Path(identity["project_root"]) / name]).hexdigest()


def test_config_copy_drift_rejected_even_when_global_source_and_hash_list_match(monkeypatch):
    formal, files, identity = _memory_preflight(monkeypatch)
    _attach_memory_artifacts(monkeypatch, formal, files, identity, finished=True)
    copy_path = Path(identity["output"]) / "config_snapshot/boundary_conditions.yaml"
    files[copy_path] = b"drifted_local_copy"
    with pytest.raises(ValueError, match="配置|快照|普通|SHA"):
        formal.qualify_task11_howard_lf_source(**identity)


def _rebind_synthetic_receipts(objects):
    for name, receipt in objects.items():
        if not fnmatch.fnmatch(name, "LF会话收据_*.json"):
            continue
        number = name.removeprefix("LF会话收据_").removesuffix(".json")
        for field, filename in (("最近阶段SHA256", f"最近提交历史_{number}.pt"),
                                ("观测最佳阶段SHA256", f"观测最佳提交历史_{number}.pt"),
                                ("观测最佳模型SHA256", f"观测最佳模型历史_{number}.pt"),
                                ("初始阶段SHA256", "阶段_初始.pt")):
            receipt[field] = _sha(objects[filename])
        if receipt["真实训练末阶段SHA256"] is not None:
            receipt["真实训练末阶段SHA256"] = _sha(objects["阶段_LF训练末.pt"])


@pytest.mark.parametrize("mutation", ["integer_trainability", "float_width", "float_depth",
                                     "float_latent", "integer_final_activation"])
def test_resign_all_history_cannot_hide_non_restorable_flags_or_constructor_types(mutation):
    formal = _formal()
    objects, _ = _synthetic_artifacts()
    for name, value in objects.items():
        if not name.endswith(".pt"):
            continue
        if mutation == "integer_trainability":
            if "parameter_requires_grad" in value:
                value["parameter_requires_grad"] = {key: 1 for key in value["parameter_requires_grad"]}
        else:
            key, replacement = {"float_width": ("width", 128.0), "float_depth": ("depth", 4.0),
                                "float_latent": ("latent_dim", 128.0),
                                "integer_final_activation": ("final_activation", 0)}[mutation]
            if "metadata" in value:
                value["metadata"]["LF模型构造参数"] = {**value["metadata"]["LF模型构造参数"], key: replacement}
            else:
                value["model_kwargs"] = {**value["model_kwargs"], key: replacement}
    _rebind_synthetic_receipts(objects)
    with pytest.raises(ValueError):
        formal.audit_task11_howard_lf_artifacts(objects, _artifact_hashes(objects), seed=0,
                                               source_identity=_sources(), finished=True)


@pytest.mark.parametrize("key,replacement", [("width", 128.0), ("depth", 4.0),
                                            ("latent_dim", 128.0), ("final_activation", 0)])
def test_registry_and_budget_constructor_types_must_match_real_teacher(key, replacement):
    formal = _formal()
    registry = _valid_registry()
    registry["LF模型构造参数"] = {**registry["LF模型构造参数"], key: replacement}
    registry["正式预算"]["model_kwargs"] = {**registry["正式预算"]["model_kwargs"], key: replacement}
    with pytest.raises(ValueError):
        formal.Task11HowardLFTeacher(**registry["LF模型构造参数"])
    with pytest.raises(ValueError):
        formal._validate_registry_contract(registry, source_tar_sha="2" * 64, catalog_sha="3" * 64)


@pytest.mark.parametrize("key,replacement", [("width", 128.0), ("depth", 4.0),
                                            ("latent_dim", 128.0), ("final_activation", 0)])
def test_metrics_budget_copy_alone_cannot_hide_constructor_type_alias(key, replacement):
    formal = _formal()
    objects, hashes = _synthetic_artifacts()
    configuration = copy.deepcopy(objects["metrics.json"]["configuration"])
    configuration["model_kwargs"][key] = replacement
    objects["metrics.json"]["configuration"] = configuration
    with pytest.raises(ValueError):
        formal.audit_task11_howard_lf_artifacts(objects, hashes, seed=0,
                                               source_identity=_sources(), finished=True)


@pytest.mark.parametrize("available,count", [(False, 0), (True, 2)])
def test_private_backend_rechecks_single_cuda_before_setting_device(monkeypatch, available, count):
    formal, _, identity = _memory_preflight(monkeypatch)
    prereg = formal.preflight_task11_howard_lf(**identity)
    monkeypatch.setattr(formal, "PROJECT_ROOT", Path(identity["project_root"]))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args, **kwargs:
                        pytest.fail("无实际单卡许可时不得设置CUDA设备"))
    with pytest.raises(ValueError, match="CUDA|GPU|单卡|单CUDA"):
        formal._run_task11_howard_lf_session(prereg, seed=0, session_epoch_limit=1,
                                             resume_checkpoint=None)
