#!/usr/bin/env python
"""任08 F2能源第二独立审计版：原双阶分母字段严格来自原积分证据。"""

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
from sic_cu.train.task08_f2_formal import TASK08_F2_LEDGER, _project_output


NEW_REGISTRATION = PROJECT_ROOT / (
    "研究记录/任务08_贡献消融/F2_三态能源独立审核有效配置_写出合同修订后.yaml"
)
NEW_ARCHIVE = PROJECT_ROOT / (
    "研究记录/任务08_贡献消融/F2_三态能源独立审核事前源码快照_写出合同修订后_20260916T053341+0800.tar.gz"
)
OLD_VERSION = PROJECT_ROOT / "scripts/34_audit_task08_f2_energy_double.py"


def _script(relative: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, relative)
    if spec is None or spec.loader is None:
        raise ImportError(f"任08 F2项目冻结原能源脚本失踪：{relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_energy_payload(evidence: Mapping[str, Any],
                           payload: Mapping[str, Any]) -> dict[str, Any]:
    """The locked 16/64-point verifier must prove the denominator before reporting it."""
    old34 = _script(OLD_VERSION, "f2_0065_locked_legacy_payload_proof")
    old26 = _script(old34.OLD_ENERGY_VERIFIER, "f2_locked_16_64_energy_payload_proof")
    gaps = old26._verify_energy_raw(evidence)
    summary = evidence.get("汇总", {})
    if (not isinstance(summary, dict) or
        summary.get("原定义相对平衡分母已保持") is not True or
        summary.get("工程平衡与原瓦数逐行一致") is not True or
        payload.get("原定义相对平衡分母已保持") not in (None, True)):
        raise ValueError("任08 F2新写出必须从原双阶逐行原瓦数核真实分母；拒绝伪字段")
    return dict(payload) | {
        "原定义相对平衡分母已保持": summary["原定义相对平衡分母已保持"],
        "16和64阶逐行V-J-D核对": gaps,
    }


def _source_tar_identity(config: Mapping[str, Any]) -> None:
    members = config.get("写出修订后源码逐件SHA256")
    digest = config.get("写出修订后源码tar实际字节SHA256")
    if (not isinstance(members, dict) or len(members) < 8 or
        not NEW_ARCHIVE.is_file() or sha256_file(NEW_ARCHIVE) != digest):
        raise ValueError("任08 F2写出修订后tar原件或实际SHA不可核")
    with tarfile.open(NEW_ARCHIVE, mode="r:gz") as archived:
        originals = [item for item in archived.getmembers() if item.isfile()]
        if len(originals) != len(members) or set(members) != {item.name for item in originals}:
            raise ValueError("任08 F2写出修订后归档源码成员与事前集合不等")
        for item in originals:
            if (item.name.startswith("/") or ".." in Path(item.name).parts or
                not (PROJECT_ROOT / item.name).is_file()):
                raise ValueError("任08 F2写出修订归档含项目外/失踪源")
            content = archived.extractfile(item)
            if (content is None or hashlib.sha256(content.read()).hexdigest() != members[item.name]
                or sha256_file(PROJECT_ROOT / item.name) != members[item.name]):
                raise ValueError(f"任08 F2修订归档和现场字节不等：{item.name}")


def _require_new_registration(expected_sha: str):
    if (not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or
        not NEW_REGISTRATION.is_file() or
        sha256_file(NEW_REGISTRATION) != expected_sha):
        raise ValueError("F2第二能源版本只能以新项目内事前YAML字节SHA进入")
    config = load_yaml(NEW_REGISTRATION)
    old34 = _script(OLD_VERSION, "f2_0065_frozen_registration_reuse")
    if (config.get("事前台账记录编号") != "录-0067" or
        config.get("原0065能源登记SHA256") != sha256_file(old34.ENERGY_REGISTRATION) or
        config.get("原0065能源源码tarSHA256") != sha256_file(old34.ENERGY_ARCHIVE) or
        config.get("源录0065审核脚本SHA256") != sha256_file(OLD_VERSION) or
        config.get("源正式F2事前登记SHA256") !=
        sha256_file(PROJECT_ROOT / "研究记录/任务08_贡献消融/F2_正式五种子有效运行配置_事务修订后.yaml") or
        config.get("双阶原定义相对平衡分母须从原汇总核真") is not True or
        config.get("审计模型副本dtype") != "torch.float64" or
        config.get("功率时间点数") != 30 or
        config.get("求积阶数") != [16, 64] or
        config.get("原始能源和散度各行数") != 60):
        raise ValueError("F2写出修订不能改变老版真训练/审计输入/Double求积")
    rows = [row for row in TASK08_F2_LEDGER.read_text(encoding="utf-8").splitlines()
            if re.search(r"\|\s*录-0067\s*\|", row)]
    if (len(rows) != 1 or expected_sha not in rows[0] or
        sha256_file(NEW_ARCHIVE) not in rows[0] or
        "F2_三态能源独立审核有效配置_写出合同修订后.yaml" not in rows[0]):
        raise ValueError("F2独立能源修订录-0067双SHA未在正式主账事前锁定")
    _source_tar_identity(config)
    oldrecord = old34._require_energy_registration(config["原0065能源登记SHA256"])
    return config, old34, oldrecord


def audit_state_energy_writer_fixed(run_directory: str | Path, *, state: str,
                                    output_directory: str | Path,
                                    registry_sha256: str,
                                    device_name: str = "cuda") -> dict[str, Any]:
    if state not in ("best", "physical", "final"):
        raise ValueError("F2第二版只可审计完整best/physical/final三态")
    run = Path(run_directory)
    run = run if run.is_absolute() else PROJECT_ROOT / run
    output = _project_output(output_directory)
    if output.exists() or run.resolve() == output.resolve() or run.resolve() in output.resolve().parents:
        raise FileExistsError("F2能源独立结果必须项目内新目录且不能覆盖训练状态")
    config, old34, oldrecord = _require_new_registration(registry_sha256)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("F2第二版正式能源仍只准真CUDA，CPU仅合同单测")
    old33 = _script(old34.OLD_AUDITOR, "task08_f2_0058_original_stage_gate_0067")
    info = old33.verify_run(run, registry_sha256=oldrecord["源正式F2有效配置SHA256"],
                            device=device)
    seed = info["运行种子"]
    expected = oldrecord.get("逐种子真训练和阶段原件SHA256", {}).get(seed)
    if (not isinstance(expected, dict) or run.resolve() !=
        (PROJECT_ROOT / expected.get("运行目录", "")).resolve() or
        any(expected[f"阶段{key}完整SHA256"] != info["阶段SHA256"][key]
            for key in ("best", "physical", "final")) or
        expected["训练日志SHA256"] != info["日志SHA256"] or
        expected["阶段报告SHA256"] != info["报告SHA256"]):
        raise ValueError("任08 F2新版本真实seed和五真原件原SHA与旧0065锁定不一致")
    stage = info["阶段文件"][state]
    stage_sha = info["阶段SHA256"][state]
    snapshot = torch.load(stage, map_location="cpu", weights_only=False)
    model = old34.prepare_double_audit_model(old33._stage_model(info, state, device), snapshot)
    powers, times, orders = old33.fixed_schedule()
    evidence = audit_schedule_energy(
        model, load_materials(), load_resolved_boundary_conditions(),
        AxisymmetricGeometry.from_config(load_yaml("configs/geometry.yaml")),
        powers_w=powers, times_s=times, orders=tuple(orders), device=device,
    )
    if (len(evidence["原始能量"]) != 60 or len(evidence["原始散度"]) != 60 or
        evidence["指标明细"].height != 30 or evidence["物理分解"].height != 30 or
        sha256_file(stage) != stage_sha):
        raise ValueError("任08 F2写出修订原双阶逐行原瓦数或源阶段SHA缺失")
    summary = evidence["汇总"]
    if any(not math.isfinite(float(summary[key])) for key in
           ("绝对平衡宏均值_瓦", "绝对平衡95分位_瓦", "最大分解剩余差_瓦")):
        raise ValueError("任08 F2新写出能源绝对瓦数/分解残差非有限")
    payload = checked_energy_payload(evidence, {
        "状态": "F2真CUDA三状态能源新独立写出；无工程通过阈值",
        "事前录号": "录-0067", "本版实际YAML字节SHA256": registry_sha256,
        "本版源码tar实际SHA256": config["写出修订后源码tar实际字节SHA256"],
        "前版录0065有效YAML实际SHA256": config["原0065能源登记SHA256"],
        "源正式F2训练登记SHA256": oldrecord["源正式F2有效配置SHA256"],
        "源真训练日志SHA256": info["日志SHA256"],
        "源真阶段报告SHA256": info["报告SHA256"],
        "原状态完整阶段pt实际SHA256": stage_sha,
        "审计模型副本dtype": "torch.float64",
        "真训练阶段pt原dtype": "torch.float32",
        "训练PDE": None, "训练PDE含义": "未计算；独立审核中体内残差逐点真实积分",
        "状态种子": seed, "状态类型": state,
        "物理最佳为真实第0轮": state == "physical" and
        snapshot["metadata"].get("物理最佳全局轮次") == 0,
        "观测最佳合法HF分_摄氏度": info["观测合法选分_摄氏度"],
        "校正实际轮次": info["校正真实轮次"],
        "联合实际轮次": info["联合真实轮次"],
        "功率_瓦": powers, "时刻_秒": times, "求积阶数": orders,
        "原始能量行数": 60, "原始散度行数": 60,
        "原汇总绝对瓦数和相对定义": summary,
        "旧test_Data温度标签读取": False,
        "内部HF真实实验温度已取得": False,
    })
    old_writer = _script(old34.OLD_AUDITOR,
                         "task08_f2_0058_source_original_writer_0067")._old_auditor()
    hashes = old_writer.write_energy_evidence(output, evidence, payload)
    if sha256_file(stage) != stage_sha:
        raise ValueError("任08 F2新能源写出后原阶段SHA发生漂移；拒用结果")
    return payload | {"输出": str(output), "六工件真实SHA256": hashes}


def main() -> None:
    parser = argparse.ArgumentParser(description="任08 F2仅修订真实原分母写出合同的独立能源审核")
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--run", required=True)
    parser.add_argument("--state", choices=("best", "physical", "final"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_state_energy_writer_fixed(
        args.run, state=args.state, output_directory=args.output,
        registry_sha256=args.registry_sha256, device_name=args.device,
    )
    print(json.dumps(report, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
