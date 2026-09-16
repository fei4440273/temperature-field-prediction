#!/usr/bin/env python
"""任08 F2另版能源审核：仅审核模型副本升Float64，不改五种训练原件。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import tarfile
from pathlib import Path
from typing import Any, Mapping

import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.eval.energy_v5 import AxisymmetricGeometry, audit_schedule_energy
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.train.task07_source import validate_task07_sources
from sic_cu.train.task08_f2_formal import (
    TASK08_F2_LEDGER, TASK08_F2_REGISTRATION, _require_formal_registration,
    _project_output,
)


ENERGY_REGISTRATION = PROJECT_ROOT / (
    "研究记录/任务08_贡献消融/F2_三态能源独立审核有效配置_双精度修订后.yaml"
)
ENERGY_ARCHIVE = PROJECT_ROOT / (
    "研究记录/任务08_贡献消融/F2_三态能源独立审核事前源码快照_双精度修订后_20260916T052432+0800.tar.gz"
)
OLD_AUDITOR = PROJECT_ROOT / "scripts/33_audit_task08_f2_states.py"
OLD_ENERGY_VERIFIER = PROJECT_ROOT / "scripts/26_audit_task07_states.py"
OBSERVATIONS = PROJECT_ROOT / (
    "研究记录/任务08_贡献消融/F2_五种子合法HF逐窗与径向_20260916T051312+0800"
)


def _locked_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"项目内已冻结能源审核依赖不可加载：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_double_audit_model(model: torch.nn.Module,
                               snapshot: Mapping[str, Any]) -> torch.nn.Module:
    """Convert only the independently reloaded model, never the checkpoint or optimizer."""
    source = snapshot.get("model_state")
    actual = model.state_dict()
    if (not isinstance(source, dict) or source.keys() != actual.keys() or
        any(not isinstance(value, torch.Tensor) or
            not torch.equal(value.to(device=actual[name].device), actual[name])
            for name, value in source.items())):
        raise ValueError("F2能源审核模型与已承诺完整阶段原张量不一致")
    if (not any(parameter.dtype == torch.float32 for parameter in model.parameters()) or
        any(parameter.dtype != torch.float32 for parameter in model.parameters())):
        raise ValueError("F2原训练模型应为Float32；不得输入伪造能源精度模型")
    model.requires_grad_(False).to(dtype=torch.float64).eval()
    promoted = model.state_dict()
    if (any(not torch.equal(value.to(device=promoted[name].device,
                                     dtype=promoted[name].dtype), promoted[name])
            for name, value in source.items()) or
        any(parameter.dtype != torch.float64 or parameter.requires_grad
            for parameter in model.parameters())):
        raise ValueError("F2审核Float64模型副本升精度后原张量不同或仍有训练梯度")
    return model


def _archive_identity(registration: Mapping[str, Any]) -> None:
    declared = registration.get("审核源码逐件SHA256")
    archive_sha = registration.get("审核源码tar实际字节SHA256")
    if (not isinstance(declared, dict) or len(declared) < 6 or
        not ENERGY_ARCHIVE.is_file() or sha256_file(ENERGY_ARCHIVE) != archive_sha):
        raise ValueError("F2新Float64审计源码tar/成员事前SHA不存在或不符")
    with tarfile.open(ENERGY_ARCHIVE, mode="r:gz") as package:
        files = [member for member in package.getmembers() if member.isfile()]
        if len(files) != len(declared) or set(declared) != {item.name for item in files}:
            raise ValueError("F2双精度能源新tar成员集合与事前承诺不一致")
        for member in files:
            if (member.name.startswith("/") or ".." in Path(member.name).parts or
                not (PROJECT_ROOT / member.name).is_file()):
                raise ValueError("F2能源tar拒绝项目外路径/失踪源码")
            packed = package.extractfile(member)
            if (packed is None or hashlib.sha256(packed.read()).hexdigest() != declared[member.name]
                or sha256_file(PROJECT_ROOT / member.name) != declared[member.name]):
                raise ValueError(f"F2新审计tar归档/现场源码SHA已漂移：{member.name}")


def _require_energy_registration(expected_sha: str) -> Mapping[str, Any]:
    if (not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or
        not ENERGY_REGISTRATION.is_file() or
        sha256_file(ENERGY_REGISTRATION) != expected_sha):
        raise ValueError("任08 F2能源新版本必须使用项目内事前SHA真实登记")
    registration = load_yaml(ENERGY_REGISTRATION)
    if (registration.get("事前台账记录编号") != "录-0065" or
        registration.get("源正式F2有效配置SHA256") != sha256_file(TASK08_F2_REGISTRATION) or
        registration.get("源正式F2事前源码tar实际SHA256") !=
        "ca08e9e9c1e4305c5549adbc6a6f9affd88cb58d23e9d70b906144cb3431726d" or
        registration.get("审核模型副本dtype") != "torch.float64" or
        registration.get("求积阶数") != [16, 64] or
        registration.get("功率时刻行数") != 30 or
        registration.get("审核原始能量和散度逐件行数") != 60 or
        registration.get("冻结旧审核脚本SHA256") != sha256_file(OLD_AUDITOR) or
        registration.get("冻结原逐行能源验算脚本SHA256") !=
        sha256_file(OLD_ENERGY_VERIFIER) or
        registration.get("五种子合法观测摘要SHA256") !=
        sha256_file(OBSERVATIONS / "五种子合法观测摘要.json")):
        raise ValueError("任08 F2能源新登记源正式训练/五观测/原Double求积不等")
    rows = [line for line in TASK08_F2_LEDGER.read_text(encoding="utf-8").splitlines()
            if re.search(r"\|\s*录-0065\s*\|", line)]
    if (len(rows) != 1 or expected_sha not in rows[0] or
        sha256_file(ENERGY_ARCHIVE) not in rows[0] or
        "F2_三态能源独立审核有效配置_双精度修订后.yaml" not in rows[0]):
        raise ValueError("任08 F2新能源正式主账录-0065双SHA尚未锁定")
    _archive_identity(registration)
    _require_formal_registration(TASK08_F2_REGISTRATION,
                                 registration["源正式F2有效配置SHA256"],
                                 validate_task07_sources())
    return registration


def audit_state_energy_double(run_directory: str | Path, *, state: str,
                              output_directory: str | Path,
                              registry_sha256: str,
                              device_name: str = "cuda") -> dict[str, Any]:
    if state not in ("best", "physical", "final"):
        raise ValueError("F2新审核只能分别验完整观测最佳/物理最佳/末态")
    run = Path(run_directory)
    run = run if run.is_absolute() else PROJECT_ROOT / run
    output = _project_output(output_directory)
    if output.exists() or run.resolve() == output.resolve() or run.resolve() in output.resolve().parents:
        raise FileExistsError("F2新能源结果不能覆盖已有原件或写在训练目录")
    record = _require_energy_registration(registry_sha256)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("F2正式Float64能源只准实际CUDA双阶；CPU只能定向测试")
    old = _locked_script(OLD_AUDITOR, "task08_f2_old_locked_energy_double")
    info = old.verify_run(run, registry_sha256=record["源正式F2有效配置SHA256"],
                          device=device)
    seed = info["运行种子"]
    expected = record.get("逐种子真训练和阶段原件SHA256", {}).get(seed)
    if (not isinstance(expected, dict) or run.resolve() != (PROJECT_ROOT / expected.get("运行目录", "")).resolve()
        or expected.get("阶段best完整SHA256") != info["阶段SHA256"]["best"]
        or expected.get("阶段physical完整SHA256") != info["阶段SHA256"]["physical"]
        or expected.get("阶段final完整SHA256") != info["阶段SHA256"]["final"]
        or expected.get("训练日志SHA256") != info["日志SHA256"]
        or expected.get("阶段报告SHA256") != info["报告SHA256"]):
        raise ValueError("F2新审计输入真实seed/三阶段原pt及日志与事前登记SHA不一致")
    source_hash = info["阶段SHA256"][state]
    source = info["阶段文件"][state]
    snapshot = torch.load(source, map_location="cpu", weights_only=False)
    model = prepare_double_audit_model(old._stage_model(info, state, device), snapshot)
    powers, times, orders = old.fixed_schedule()
    evidence = audit_schedule_energy(
        model, load_materials(), load_resolved_boundary_conditions(),
        AxisymmetricGeometry.from_config(load_yaml("configs/geometry.yaml")),
        powers_w=powers, times_s=times, orders=tuple(orders), device=device,
    )
    original_checker = _locked_script(
        OLD_ENERGY_VERIFIER, "task08_f2_original_16_and_64_energy_double")
    gaps = original_checker._verify_energy_raw(evidence)
    if (len(evidence["原始能量"]) != 60 or len(evidence["原始散度"]) != 60
        or evidence["指标明细"].height != 30 or evidence["物理分解"].height != 30
        or sha256_file(source) != source_hash):
        raise ValueError("F2双阶原60能量/散度与源原pt未保持原SHA")
    summary = evidence["汇总"]
    if any(not math.isfinite(float(summary[key])) for key in
           ("绝对平衡宏均值_瓦", "绝对平衡95分位_瓦", "最大分解剩余差_瓦")):
        raise ValueError("F2新能源绝对瓦数与V-J-D有非有限值")
    payload = {
        "状态": "F2真CUDA三状态独立能源双精度审核；未定义工程通过阈值",
        "审核模型副本dtype": "torch.float64",
        "源训练完整阶段pt原dtype": "torch.float32",
        "审核原Double求积适配": "仅重载并关闭模型参数梯度；阶段pt/训练器未更改",
        "种子": seed, "状态类型": state,
        "新能源事前登记SHA256": registry_sha256,
        "源正式五种训练登记SHA256": record["源正式F2有效配置SHA256"],
        "原源码tar真实SHA256": record["源正式F2事前源码tar实际SHA256"],
        "本状态完整源阶段ptSHA256": source_hash,
        "训练日志SHA256": info["日志SHA256"],
        "训练阶段报告SHA256": info["报告SHA256"],
        "初始独立物理最佳属于第0轮": state == "physical" and
        snapshot["metadata"]["物理最佳全局轮次"] == 0,
        "训练体内PDE标记": None,
        "训练体内PDE意义": "未计算；能源双阶体内残差只在本审核计算",
        "源观测合法选分_摄氏度": info["观测合法选分_摄氏度"],
        "真训练校正和联合轮数": [info["校正真实轮次"], info["联合真实轮次"]],
        "功率_瓦": powers, "时刻_秒": times, "求积阶数": orders,
        "审核原始能量行数": 60, "审核原始散度行数": 60,
        "原汇总分母和绝对瓦数": summary,
        "16和64阶逐行V-J-D核对": gaps,
        "旧test_Data温度标签读取": False,
        "内部HF真实温度已取得": False,
    }
    hashes = old._old_auditor().write_energy_evidence(output, evidence, payload)
    if sha256_file(source) != source_hash:
        raise ValueError("F2双精度审后原阶段pt真实SHA已漂移；拒用审计结果")
    return payload | {"输出": str(output), "六工件真实SHA256": hashes}


def main() -> None:
    parser = argparse.ArgumentParser(description="任08 F2另版完整阶段Float64工程能源独立审计")
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--run", required=True)
    parser.add_argument("--state", choices=("best", "physical", "final"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evidence = audit_state_energy_double(
        args.run, state=args.state, output_directory=args.output,
        registry_sha256=args.registry_sha256, device_name=args.device,
    )
    print(json.dumps(evidence, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
