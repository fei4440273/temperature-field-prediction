"""Task-09 independent formal energy admission, without opening fixed TEST labels."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import subprocess
import sys

import pytest

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file


def _audit():
    try:
        return importlib.import_module("sic_cu.eval.task09_energy")
    except ModuleNotFoundError:
        pytest.fail("Task-09 exclusive independent 30-point energy audit is absent")


def test_fixed_schedule_uses_identical_original_task04_task06_pre_registered_watts():
    audit = _audit()
    powers, times, orders = audit.task09_fixed_energy_schedule()
    assert powers == [55.0, 115.2, 364.3, 403.0, 630.5, 729.0]
    assert times == [1.0, 10.0, 50.0, 100.0, 200.0]
    assert orders == [16, 64]
    assert len(powers) * len(times) == 30
    assert sha256_file(audit.TASK04_CONFIG) == audit.TASK04_CONFIG_SHA256
    assert sha256_file(audit.TASK06_CONFIG) == audit.TASK06_CONFIG_SHA256


def test_locked_frozen_task07_comparison_only_uses_legal_validation_S_no_test():
    audit = _audit()
    scores = audit.task09_locked_task07_observed_S()
    assert len(scores) == 5 and set(scores) == set(range(5))
    assert math.isclose(scores[0], 2.124929166187284)
    assert math.isclose(scores[1], 2.0829235061817655)
    assert all(math.isfinite(value) for value in scores.values())


def test_locked_five_seed_task07_energy_comparison_is_only_verified_original_watts():
    audit = _audit()
    reference = audit.task09_locked_task07_energy_best()
    assert set(reference) == set(range(5))
    assert math.isclose(reference[0]["绝对平衡宏均值_瓦"],
                        765.5009148090098)
    assert all(data["功率时刻审核行数"] == 30
               and data["原定义相对平衡分母已保持"] is True
               and data["绝对平衡宏均值_瓦"] > 0
               for data in reference.values())


def test_independent_audit_rejects_cpu_or_old_task07_output_without_writing():
    audit = _audit()
    run = (audit.TASK09_FORMAL_ROOT /
           "正式F3_primary_HF3_seed0_20260916T080000+0800")
    destination = run / "独立原能源_观测最佳_20260916T080100+0800"
    with pytest.raises(ValueError, match="CUDA|GPU"):
        audit.audit_task09_formal_energy(
            run, name="primary", size=3, seed=0, state="best",
            output=destination, device_name="cpu",
        )
    assert not destination.exists()
    old = PROJECT_ROOT / "研究记录/任务07_正式五种子重训"
    with pytest.raises(ValueError, match="任09|目录|正式|子集"):
        audit.task09_energy_output(old, destination, "best")


def test_audit_path_only_accepts_new_run_child_and_exact_state():
    audit = _audit()
    run = (audit.TASK09_FORMAL_ROOT /
           "正式F3_left_center_HF9_seed4_20260916T080000+0800")
    destination = run / "独立原能源_训练末_20260916T080100+0800"
    assert audit.task09_energy_output(run, destination, "final") == destination
    with pytest.raises(ValueError, match="状态|best|final"):
        audit.task09_energy_output(run, destination, "teacher")
    with pytest.raises(ValueError, match="目录|审核"):
        audit.task09_energy_output(run, run, "final")


def test_independent_cli_enforces_required_canonical_and_cuda_params():
    cli = PROJECT_ROOT / "scripts/38_audit_task09_energy.py"
    assert cli.is_file()
    result = subprocess.run([sys.executable, str(cli), "--arm", "primary",
                             "--size", "3", "--seed", "0", "--device", "cpu"],
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "invalid choice" in result.stderr or "required" in result.stderr
