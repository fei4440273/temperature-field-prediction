"""任08 F2双精度能源版本化报告字段，冻结任07与任04原验算器照旧。"""

from __future__ import annotations

import copy
import importlib.util
import json

import pytest

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file


def _module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative)
    assert spec is not None and spec.loader is not None
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


def _synthetic_writer_fixture():
    # 仅CPU单测合成合法30点双阶原瓦数，不作正式能源工件。
    prior = _module("tests/test_task07_states_audit.py", "locked_synthetic_energy_unit_fixture")
    old26 = _module("scripts/26_audit_task07_states.py", "task07_original_divergence_unit")
    return prior._synthetic_30_point_watts(old26), old26


def test_frozen_old_writer_reproduces_true_summary_but_missing_payload_red(tmp_path):
    fixture, old26 = _synthetic_writer_fixture()
    gaps = old26._verify_energy_raw(fixture)
    assert fixture["汇总"]["原定义相对平衡分母已保持"] is True
    assert gaps["16阶V-J-D散度积分剩余差最大_瓦"] == 0.0
    old21 = _module("scripts/21_audit_task04_energy.py", "task04_locked_writer_unit_red")
    output = tmp_path / "旧writer缺报告分母字段拒绝"
    with pytest.raises(ValueError, match="分母|原工程瓦数"):
        old21.write_energy_evidence(output, fixture, {"状态": "CPU构造的单测不作正式"})
    assert not output.exists()


def test_new_checked_payload_from_real_summary_passes_frozen_writer(tmp_path):
    new = _module("scripts/35_audit_task08_f2_energy_writer_fixed.py", "f2_writer_fix_unit")
    fixture, old26 = _synthetic_writer_fixture()
    expected_gaps = old26._verify_energy_raw(fixture)
    payload = new.checked_energy_payload(
        fixture, {"状态": "仅CPU合成写出合同测试；不能当GPU能源证据"},
    )
    assert payload["原定义相对平衡分母已保持"] is True
    assert payload["16和64阶逐行V-J-D核对"] == expected_gaps
    old21 = _module("scripts/21_audit_task04_energy.py", "task04_locked_writer_unit_green")
    output = tmp_path / "新合同仅单测"
    hashes = old21.write_energy_evidence(output, fixture, payload)
    assert len(hashes) == 5
    assert all(sha256_file(output / name) == digest for name, digest in hashes.items())
    assert json.loads((output / "汇总指标.json").read_text(encoding="utf-8"))[
        "原定义相对平衡分母已保持"] is True
    assert (output / "审计工件SHA256.json").is_file()


@pytest.mark.parametrize("tamper", ["false_summary", "poisoned_relative_denominator",
                                      "missing_16_order_divergence"])
def test_new_checked_payload_never_forges_false_or_poisoned_raw_denominator(tamper):
    new = _module("scripts/35_audit_task08_f2_energy_writer_fixed.py", "f2_writer_guard_unit")
    fixture, _ = _synthetic_writer_fixture()
    poisoned = copy.deepcopy(fixture)
    if tamper == "false_summary":
        poisoned["汇总"]["原定义相对平衡分母已保持"] = False
    elif tamper == "poisoned_relative_denominator":
        poisoned["原始能量"][1]["relative_balance_denominator_w"] += 1.0
    else:
        poisoned["原始散度"].pop(0)
    with pytest.raises(ValueError, match="分母|原瓦|双阶|散度|来源|逐行"):
        new.checked_energy_payload(poisoned, {"原定义相对平衡分母已保持": True})


def test_new_version_refuses_external_output_before_registration():
    new = _module("scripts/35_audit_task08_f2_energy_writer_fixed.py", "f2_writer_scope_unit")
    outside = PROJECT_ROOT.parent / "任08F2禁止项目外能源写出_20260916"
    assert not outside.exists()
    with pytest.raises(ValueError, match="项目内|写入范围"):
        new.audit_state_energy_writer_fixed(
            "研究记录/任务08_贡献消融/F2_事务修订正式_种子0_20260916T040215+0800",
            state="best", output_directory=outside,
            registry_sha256="f" * 64, device_name="cuda",
        )
    assert not outside.exists()


def test_new_version_cpu_gate_cannot_publish_formal_energy(tmp_path, monkeypatch):
    new = _module("scripts/35_audit_task08_f2_energy_writer_fixed.py", "f2_writer_cuda_unit")
    monkeypatch.setattr(new, "_require_new_registration", lambda *_: ({}, None, {}))
    output = tmp_path / "CPU单测不能冒录0067真能源"
    with pytest.raises(ValueError, match="CUDA|CPU"):
        new.audit_state_energy_writer_fixed(
            "研究记录/任务08_贡献消融/F2_事务修订正式_种子0_20260916T040215+0800",
            state="best", output_directory=output,
            registry_sha256="f" * 64, device_name="cpu",
        )
    assert not output.exists()
