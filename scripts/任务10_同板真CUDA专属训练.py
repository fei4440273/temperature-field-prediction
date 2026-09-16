#!/usr/bin/env python
"""Run the frozen Task 10 trainer through a CUDA-only, receipt-producing CLI."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.eval.task10_plate_cuda_gate import verify_cuda_source_freeze


ROOT = PROJECT_ROOT / "研究记录/任务10_独立双层场基准"
LEGACY_TRAIN_CLI = PROJECT_ROOT / "scripts/任务10_同板F1_F2_F3重训.py"
METHOD_BUDGET = ROOT / "正式同板F1_F2_F3重训方法预算前登记_v2.yaml"
SOURCE_ARCHIVE = ROOT / "正式人为九热流源码冻结_20260916T032032+0800.tar.gz"
METHOD_ARCHIVE = ROOT / "正式同板方法十一源源码冻结_v2_20260916T050000+0800.tar.gz"
CUDA_BUDGET = ROOT / "正式同板真CUDA训练与后验资格v4前登记.yaml"
CUDA_ARCHIVE = ROOT / "正式同板真CUDA资格十二源事前冻结.tar.gz"
LEDGER = PROJECT_ROOT / "多保真DeepONet预测精度优化总计划与执行台账.md"
LEGACY_TRAIN_CLI_SHA256 = (
    "b7ed35d95e53d2211cb15c07ef1f1eefa271deefe7f68fde188b2f27e0293cd3"
)
METHOD_BUDGET_SHA256 = (
    "cf05d16a3cd754e7f75321418ec6186e42673e33f8774492f3eea985105fba94"
)
SOURCE_ARCHIVE_SHA256 = (
    "368f0f8458c994101bd1d20c241f61f8e64d8bedb0e74b3f733d7369e7a01c2f"
)
METHOD_ARCHIVE_SHA256 = (
    "22e6df62cf1aaddf55a8a33e8d8a71e558d89110db96466dc904d1b10c03069d"
)
_LF_KINDS = ("lf_same_physics_coarse", "lf_contact_mismatch_coarse")


def _project_path(value: str | Path) -> Path:
    supplied = Path(value)
    path = (supplied if supplied.is_absolute() else PROJECT_ROOT / supplied).resolve()
    if path != PROJECT_ROOT and PROJECT_ROOT not in path.parents:
        raise ValueError("任10真CUDA专属训练输出必须位于项目内")
    return path


def _validate_identity(arm: str, lf_source: str | None, seed: int) -> None:
    if arm not in ("F1", "PAIR") or type(seed) is not int or seed not in range(5):
        raise ValueError("任10真CUDA专属训练只允许预登记方法和五个种子")
    if ((arm == "F1" and lf_source is not None)
            or (arm == "PAIR" and lf_source not in _LF_KINDS)):
        raise ValueError("F1不得带LF，PAIR必须固定一种已登记LF来源")


def build_legacy_cuda_command(*, arm: str, lf_source: str | None,
                              seed: int, output: str | Path) -> list[str]:
    """Build the sole legacy invocation; callers cannot select CPU."""
    _validate_identity(arm, lf_source, seed)
    destination = _project_path(output)
    command = [
        sys.executable, str(LEGACY_TRAIN_CLI),
        "--arm", arm,
        "--seed", str(seed),
        "--device", "cuda",
        "--output", str(destination),
        "--budget", str(METHOD_BUDGET),
        "--budget-sha256", METHOD_BUDGET_SHA256,
        "--source-archive", str(SOURCE_ARCHIVE),
        "--source-sha256", SOURCE_ARCHIVE_SHA256,
        "--methods-archive", str(METHOD_ARCHIVE),
        "--methods-sha256", METHOD_ARCHIVE_SHA256,
    ]
    if lf_source is not None:
        command.extend(["--lf-source", lf_source])
    return command


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _assert_frozen_inputs() -> None:
    expected = {
        LEGACY_TRAIN_CLI: LEGACY_TRAIN_CLI_SHA256,
        METHOD_BUDGET: METHOD_BUDGET_SHA256,
        SOURCE_ARCHIVE: SOURCE_ARCHIVE_SHA256,
        METHOD_ARCHIVE: METHOD_ARCHIVE_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise PermissionError(f"任10真CUDA包装器固定输入漂移：{path}")


def _artifact_receipt(model_directory: Path) -> dict:
    paths = {
        "真实训练日志_SHA256": model_directory / "真实训练日志.json",
        "合法验证_SHA256": model_directory / "合法探针验证.json",
        "单模型锁_SHA256": model_directory / "合法训练验证后模型SHA锁.json",
        "最佳模型_SHA256": model_directory / "合法验证选定板模型_state_dict.pt",
        "终态模型_SHA256": model_directory / "终态HF模型_state_dict.pt",
        "最佳HF四类随机态_SHA256": model_directory / "最佳HF四类随机态.pt",
        "终态HF四类随机态_SHA256": model_directory / "终态HF四类随机态.pt",
    }
    if model_directory.name in ("F2", "F3"):
        paths["共享LF四类采样随机态_SHA256"] = (
            model_directory / "共享LF四类采样随机态.pt"
        )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise PermissionError("冻结旧训练CLI成功返回后仍缺模型机件：" + ", ".join(missing))
    return {"模型目录": str(model_directory),
            **{name: sha256_file(path) for name, path in paths.items()}}


def _failure_record(output: Path, *, command: list[str], reason: str,
                    returncode: int | None = None,
                    stdout: str = "", stderr: str = "") -> None:
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "真CUDA失败目录永久保留_禁止恢复.json", {
        "schema_version": 1,
        "运行状态": "失败",
        "失败时刻": datetime.now().astimezone().isoformat(),
        "命令": command,
        "返回码": returncode,
        "原因": reason,
        "标准输出末尾": stdout[-20000:],
        "标准错误末尾": stderr[-20000:],
        "本目录永久留档": True,
        "禁止断点恢复": True,
        "后续只能换全新批次目录完整重跑": True,
        "完整HF温度已读取": False,
    })


def run_cuda_only(*, arm: str, lf_source: str | None, seed: int,
                  output: str | Path, cuda_budget_sha256: str,
                  cuda_tar_sha256: str) -> dict:
    _validate_identity(arm, lf_source, seed)
    destination = _project_path(output)
    if destination.exists():
        raise FileExistsError("真CUDA专属训练不得覆盖或恢复任何已有成功/失败目录")
    _assert_frozen_inputs()
    verify_cuda_source_freeze(
        cuda_budget_path=CUDA_BUDGET,
        cuda_budget_sha256=cuda_budget_sha256,
        cuda_tar_path=CUDA_ARCHIVE,
        cuda_tar_sha256=cuda_tar_sha256,
        ledger_path=LEDGER,
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise PermissionError("CUDA当前不可用，禁止退回CPU生成正式任10模型")
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    device = {
        "类型": "cuda",
        "可见CUDA设备数": torch.cuda.device_count(),
        "实际设备序号": device_index,
        "名称": properties.name,
        "计算能力": [properties.major, properties.minor],
        "总显存字节": properties.total_memory,
        "PyTorch版本": torch.__version__,
        "CUDA运行时版本": torch.version.cuda,
    }
    if not isinstance(device["CUDA运行时版本"], str) or not device["CUDA运行时版本"]:
        raise PermissionError("PyTorch没有真实CUDA运行时版本，不得启动正式训练")
    command = build_legacy_cuda_command(
        arm=arm, lf_source=lf_source, seed=seed, output=destination)
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError("冻结旧训练CLI非零退出")
        start = destination / "开始训练源码与受限源凭据.json"
        summary = destination / "合法模型训练与资源运行摘要.json"
        if not start.is_file() or not summary.is_file():
            raise PermissionError("冻结旧训练CLI未生成开始凭据或资源摘要")
        start_record = json.loads(start.read_text(encoding="utf-8"))
        summary_record = json.loads(summary.read_text(encoding="utf-8"))
        duration = summary_record.get("真正训练耗时_秒")
        peak = summary_record.get("CUDA峰值分配MiB")
        if (start_record.get("设备") != "cuda"
                or type(duration) not in (int, float) or not math.isfinite(duration)
                or duration <= 0 or type(peak) not in (int, float)
                or not math.isfinite(peak) or peak <= 0):
            raise PermissionError("旧训练输出未留下有限正数耗时与真实CUDA峰值显存")
        model_names = ("F1",) if arm == "F1" else ("F2", "F3")
        artifacts = {name: _artifact_receipt(destination / name)
                     for name in model_names}
        stdout_path = destination / "真CUDA包装器_旧CLI标准输出.txt"
        stderr_path = destination / "真CUDA包装器_旧CLI标准错误.txt"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        receipt = {
            "schema_version": 1,
            "正式GPU专属包装器": True,
            "运行状态": "成功",
            "设备": device,
            "训练": {
                "arm": arm,
                "随机种子": seed,
                "LF来源": lf_source,
                "真正训练耗时_秒": duration,
                "CUDA峰值分配MiB": peak,
                "旧冻结训练CLI_SHA256": LEGACY_TRAIN_CLI_SHA256,
                "开始训练凭据_SHA256": sha256_file(start),
                "运行摘要_SHA256": sha256_file(summary),
                "旧CLI标准输出_SHA256": sha256_file(stdout_path),
                "旧CLI标准错误_SHA256": sha256_file(stderr_path),
            },
            "模型原件": artifacts,
            "失败目录不恢复且新批次重跑": True,
            "完整HF温度已读取": False,
            "收据时刻": datetime.now().astimezone().isoformat(),
        }
        receipt_path = destination / "真CUDA运行设备与封存原件收据.json"
        _write_json(receipt_path, receipt)
        return {"输出目录": str(destination),
                "CUDA运行收据文件": str(receipt_path),
                "CUDA运行收据_SHA256": sha256_file(receipt_path),
                "模型方法": list(model_names),
                "完整HF温度已读取": False}
    except Exception as error:
        _failure_record(
            destination, command=command, reason=f"{type(error).__name__}: {error}",
            returncode=None if completed is None else completed.returncode,
            stdout="" if completed is None else completed.stdout,
            stderr="" if completed is None else completed.stderr,
        )
        raise


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="任10冻结旧方法的真CUDA专属训练包装器")
    parser.add_argument("--arm", required=True, choices=["F1", "PAIR"])
    parser.add_argument("--lf-source", choices=list(_LF_KINDS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--CUDA资格预算SHA256", required=True)
    parser.add_argument("--CUDA资格源码tarSHA256", required=True)
    arguments = parser.parse_args(argv)
    report = run_cuda_only(arm=arguments.arm, lf_source=arguments.lf_source,
                           seed=arguments.seed, output=arguments.output,
                           cuda_budget_sha256=arguments.CUDA资格预算SHA256,
                           cuda_tar_sha256=arguments.CUDA资格源码tarSHA256)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    main()
