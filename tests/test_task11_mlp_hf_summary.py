"""Synthetic, project-local contracts for Task-11 completed HF aggregation."""

from __future__ import annotations

import copy
import importlib.util
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.train.task11_mlp_hf_formal import ROOT_LEDGER, RUN_DIRECTORY


SCRIPT = PROJECT_ROOT / "scripts/53_summarize_task11_mlp_hf_formal.py"
SUMMARY_TOKEN = "TASK11_NEW_MLP_SUMMARY_GATE:v1"
ENERGY_NAMES = ("指标明细.csv", "物理分解.csv", "原始能量.jsonl", "原始散度.jsonl",
                "汇总指标.json", "审计工件SHA256.json")


def _module():
    assert SCRIPT.is_file(), "Task-11 CPU five-seed HF summary implementation is absent"
    spec = importlib.util.spec_from_file_location("task11_hf_summary_tested", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _reseal(directory):
    names = ["指标明细.csv", "物理分解.csv", "原始能量.jsonl", "原始散度.jsonl", "汇总指标.json"]
    _write_json(directory / "审计工件SHA256.json",
                {name: sha256_file(directory / name) for name in names})


def _refresh_gate(evidence):
    arguments = evidence["arguments"]
    cell = (f"{SUMMARY_TOKEN}; status=active; ENERGY_CATALOG_SHA256={arguments['energy_catalog_sha']}; "
            f"SOURCE_SHA256={sha256_file(SCRIPT)}; HF_YAML_SHA256={arguments['registry_sha']}")
    evidence["ledger"].write_text(f"| 录-0110 | {cell} | 仅合成单元测试，不是真实执行 |\n",
                                  encoding="utf-8")


def _refresh_catalog(evidence, *, update_gate=True):
    _write_json(evidence["catalog"], evidence["directories"])
    evidence["arguments"]["energy_catalog_sha"] = sha256_file(evidence["catalog"])
    if update_gate:
        _refresh_gate(evidence)


def _pin_directory(evidence, index=0):
    row = evidence["directories"][index]
    directory = Path(row["directory"])
    row["六工件SHA256"] = {name: sha256_file(directory / name) for name in ENERGY_NAMES}
    _refresh_catalog(evidence)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    # These are generated unit-test identities, never real execution evidence.
    root = tmp_path
    run_root = root / RUN_DIRECTORY
    arguments, qualifications, lf_rows, directories = {}, {}, [], []
    for key in ("registry", "source_tar", "hf_data_catalog"):
        path = root / (key + ".json")
        _write_json(path, {"synthetic_unit_test_only": True})
        arguments[key] = str(path)
        arguments[key + "_sha"] = sha256_file(path)
    fixture = runpy.run_path(str(PROJECT_ROOT / "tests/test_task11_mlp_hf_energy.py"))
    for seed in range(5):
        run = run_root / f"正式MLP_HF_seed{seed}"
        run.mkdir(parents=True)
        split_path = run / "config_snapshot/splits.yaml"
        split_path.parent.mkdir()
        split_path.write_text(yaml.safe_dump(load_yaml("configs/splits.yaml")), encoding="utf-8")
        lf_run = run_root / f"正式MLP_LF_seed{seed}"
        lf_run.mkdir()
        lf_epochs, lf_seconds = 201 + seed, 10.0 + seed
        lf_metrics = {"epochs_completed": lf_epochs, "training_seconds": lf_seconds,
                      "status": "completed_current_protocol_lf_only", "seed": seed}
        _write_json(lf_run / "metrics.json", lf_metrics)
        lf_best = lf_run / "best.pt"
        torch.save({"synthetic_unit_test_only": True}, lf_best)
        lf_rows.append({"seed": seed, "目录": str(lf_run.relative_to(root)),
                        "最佳检查点": str(lf_best.relative_to(root)),
                        "最佳检查点SHA256": sha256_file(lf_best),
                        "LF初始张量SHA256": f"{seed + 1:064x}",
                        "实际轮次": lf_epochs, "LF真实多会话累计成本秒": lf_seconds,
                        "真实原件SHA256": {str((lf_run / "metrics.json").relative_to(root)):
                                          sha256_file(lf_run / "metrics.json")}})
        hf_epochs, hf_seconds = 400 + seed, 20.0 + seed
        consumption = {"HF观测点": 29593 * hf_epochs, "HF传感器点": 44775 * hf_epochs,
                       "HF观测优化步": 15 * hf_epochs, "物理优化步": hf_epochs,
                       "物理配点": 256 * hf_epochs, "LF回放点": 60 * 2048 * 200,
                       "LF回放小批": 60 * 200}
        _write_json(run / "metrics.json", {"consumption": consumption,
                    "training_seconds": hf_seconds, "lf_training_seconds": lf_seconds})
        for state in ("best", "final"):
            modalities = {"顶部": 3.0 + seed * 0.1,
                          "absolute_rmse_c": 1.0 + seed * 0.1,
                          "absolute_mae_c": 0.8 + seed * 0.1,
                          "delta_rmse_c": 0.5 + seed * 0.1,
                          "delta_mae_c": 0.4 + seed * 0.1}
            if state == "final":
                modalities = {key: value + 0.2 for key, value in modalities.items()}
            score = (modalities["顶部"] + 0.2 * modalities["absolute_rmse_c"]
                     + modalities["delta_rmse_c"]) / 2.2
            torch.save({"seed": seed, "validation_selection_score_c": score,
                        "validation_sensor": modalities}, run / (state + ".pt"))
        original = {str(path.relative_to(run)): sha256_file(path)
                    for path in run.rglob("*") if path.is_file()}
        qualification = {"状态": "CPU完整HF来源资格PASS；独立能源仍须另行审核", "seed": seed,
                         "目录": str(run), "登记原件": arguments["registry"],
                         "登记SHA256": arguments["registry_sha"],
                         "源码归档原件": arguments["source_tar"],
                         "源码归档SHA256": arguments["source_tar_sha"],
                         "HF数据目录原件": arguments["hf_data_catalog"],
                         "HF数据目录SHA256": arguments["hf_data_catalog_sha"],
                         "本seed新LF最佳检查点原件": str(lf_best),
                         "本seed新LF最佳检查点SHA256": sha256_file(lf_best),
                         "本seedLF初始张量SHA256": f"{seed + 1:064x}",
                         "HF校正实际轮次": 200 + seed, "HF联合实际轮次": 200,
                         "本人HF累计真实成本秒": hf_seconds, "本人LF累计真实成本秒": lf_seconds,
                         "全部会话CUDA峰值显存字节": 1000 + seed,
                         "完整原件SHA256": original,
                         "旧固定TEST温度读取": False, "模拟测试功率温度读取": False}
        qualifications[seed] = qualification
    lf_catalog = root / "lf_catalog.json"
    _write_json(lf_catalog, {"五seed新LF": lf_rows})
    arguments.update(lf_catalog=str(lf_catalog), lf_catalog_sha=sha256_file(lf_catalog))
    for seed in range(5):
        q = qualifications[seed]
        q.update(LF目录原件=str(lf_catalog), LF目录SHA256=arguments["lf_catalog_sha"])
        run = Path(q["目录"])
        for state in ("best", "final"):
            label = "观测最佳" if state == "best" else "训练末"
            directory = run / f"独立原能源_{label}_20260916T160000+0800"
            directory.mkdir()
            audit = fixture["_analytic_audit"]()
            for name, key in (("指标明细.csv", "指标明细"), ("物理分解.csv", "物理分解")):
                audit[key].write_csv(directory / name)
            for name, key in (("原始能量.jsonl", "原始能量"), ("原始散度.jsonl", "原始散度")):
                (directory / name).write_text("".join(json.dumps(row) + "\n"
                                                     for row in audit[key]), encoding="utf-8")
            view = torch.load(run / (state + ".pt"), map_location="cpu", weights_only=True)
            split = load_yaml(run / "config_snapshot/splits.yaml")
            selected = {"合法HF验证功率_瓦": [115.2, 403.0, 630.5], "合法HF全部Top点": 7272,
                        "合法HF全部传感点": 752,
                        "合法HF原macro_v1选分独立重算_摄氏度": view["validation_selection_score_c"],
                        "合法HF分模态独立重算": view["validation_sensor"],
                        "合法LF验证功率_瓦": split["simulation"]["validation_powers_w"],
                        "合法LF逐材料节点与真实体积RMSE_摄氏度": {
                            material: {"node": 1.0 + seed * 0.1, "volume": 0.9 + seed * 0.1}
                            for material in ("Cu", "SiC")},
                        "本独立重算不回调训练或选模": True}
            from sic_cu.eval.task11_mlp_hf_energy import verify_task11_energy_raw
            payload = {**audit["汇总"], **verify_task11_energy_raw(audit),
                       "运行种子": seed, "运行臂": "新MLP_PiNN", "审核状态": state,
                       "运行目录": str(run), "选择模型SHA256": q["完整原件SHA256"][state + ".pt"],
                       "正式登记原件": arguments["registry"], "正式登记SHA256": arguments["registry_sha"],
                       "源码归档原件": arguments["source_tar"], "源码归档SHA256": arguments["source_tar_sha"],
                       "五LF身份清单SHA256": arguments["lf_catalog_sha"],
                       "HF开发数据清单SHA256": arguments["hf_data_catalog_sha"],
                       "完整原件SHA256": q["完整原件SHA256"],
                       "本seed新LF最佳检查点SHA256": q["本seed新LF最佳检查点SHA256"],
                       "本seedLF初始张量SHA256": q["本seedLF初始张量SHA256"],
                       "本次合法HF_LF独立重算": selected,
                       "功率_瓦": [55.0, 115.2, 364.3, 403.0, 630.5, 729.0],
                       "时刻_秒": [1.0, 10.0, 50.0, 100.0, 200.0], "求积阶数": [16, 64],
                       "模型计算dtype": "float64", "只读原件审核前后SHA一致": True,
                       "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
                       "名义能量不证明原FEM热预算或内部温度真值": True,
                       "工程安全阈值已建立": False, "结果不用于训练选模": True,
                       "名义吸收归一筛查": {"均值阈值": 0.05, "95分位阈值": 0.10,
                           "均值通过": False, "95分位通过": False, "不代表工程安全": True}}
            _write_json(directory / "汇总指标.json", payload)
            _reseal(directory)
            directories.append({"seed": seed, "state": state, "directory": str(directory),
                                "六工件SHA256": {name: sha256_file(directory / name)
                                             for name in ENERGY_NAMES}})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU summary must not probe GPU"))
    catalog = root / "仅合成能源目录登记.json"
    arguments["energy_catalog"] = str(catalog)
    evidence = {"root": root, "run_root": run_root, "arguments": arguments,
                "qualifications": qualifications, "directories": directories,
                "catalog": catalog, "ledger": root / ROOT_LEDGER,
                "output": root / "研究记录/任务11_外部对照/新聚合验收"}
    _refresh_catalog(evidence)
    return evidence


def _execute(evidence, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "qualify_task11_mlp_hf_source",
                        lambda **kwargs: copy.deepcopy(evidence["qualifications"][kwargs["seed"]]))
    return module.summarize_task11_mlp_hf_formal(
        run_root=evidence["run_root"],
        output=evidence["output"], project_root=evidence["root"], **evidence["arguments"])


def test_five_seed_summary_recomputes_statistics_and_chinese_reports(evidence, monkeypatch):
    result = _execute(evidence, monkeypatch)
    stats = result["训练与成本统计"]["HF训练成本_秒"]
    assert stats["均值"] == 22.0
    assert stats["样本标准差"] == pytest.approx(np.std([20, 21, 22, 23, 24], ddof=1))
    assert stats["最小值"] == 20.0 and stats["最大值"] == 24.0
    assert result["训练与成本统计"]["LF训练成本_秒"]["均值"] == 12.0
    assert result["统计约定"]["能源95分位先各seed30点计算再统计"] is True
    assert set(path.name for path in evidence["output"].iterdir()) == {"中文验收.md", "机器汇总.json"}
    assert json.loads((evidence["output"] / "机器汇总.json").read_text()) == result
    report = (evidence["output"] / "中文验收.md").read_text()
    assert "样本标准差" in report and "不是工程安全" in report
    assert "两环合并" in report and "没有独立热端/冷端" in report


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_seed", "wrong_state", "wrong_run_root"])
def test_summary_requires_fixed_five_runs_and_ten_explicit_state_directories(evidence, monkeypatch, change):
    if change == "missing":
        evidence["directories"].pop()
    elif change == "duplicate":
        evidence["directories"][-1] = evidence["directories"][0]
    elif change == "wrong_seed":
        evidence["directories"][0]["seed"] = 5
    elif change == "wrong_state":
        evidence["directories"][0]["state"] = "physical"
    else:
        evidence["run_root"] = evidence["root"] / "其他已有模型"
    _refresh_catalog(evidence)
    with pytest.raises(ValueError):
        _execute(evidence, monkeypatch)
    assert not evidence["output"].exists()


@pytest.mark.parametrize("field", ["运行种子", "审核状态", "选择模型SHA256", "正式登记SHA256",
                                   "完整原件SHA256", "功率_瓦", "结果不用于训练选模"])
def test_summary_rejects_resealed_energy_identity_or_contract_drift(evidence, monkeypatch, field):
    directory = Path(evidence["directories"][0]["directory"])
    path = directory / "汇总指标.json"
    payload = json.loads(path.read_text())
    payload[field] = {"运行种子": 1, "审核状态": "final", "选择模型SHA256": "0" * 64,
                      "正式登记SHA256": "0" * 64, "完整原件SHA256": {},
                      "功率_瓦": [55], "结果不用于训练选模": False}[field]
    _write_json(path, payload)
    _reseal(directory)
    _pin_directory(evidence)
    inner_error = {"正式登记SHA256": "四份事前登记SHA", "结果不用于训练选模": "不选模合同"}.get(
        field, "冻结双阶时间功率身份")
    with pytest.raises(ValueError, match=inner_error):
        _execute(evidence, monkeypatch)
    assert not evidence["output"].exists()


def test_summary_checks_actual_five_energy_files_not_only_exit_or_manifest(evidence, monkeypatch):
    path = Path(evidence["directories"][0]["directory"]) / "原始能量.jsonl"
    path.write_text(path.read_text() + "{}\n")
    with pytest.raises(ValueError, match="SHA|原件|能源"):
        _execute(evidence, monkeypatch)
    assert not evidence["output"].exists()


def test_summary_recomputes_original_watts_even_after_all_files_resealed(evidence, monkeypatch):
    directory = Path(evidence["directories"][0]["directory"])
    path = directory / "原始能量.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["relative_balance_denominator_w"] += 1.0
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _reseal(directory)
    _pin_directory(evidence)
    with pytest.raises(ValueError, match="瓦数|原定义|分母"):
        _execute(evidence, monkeypatch)
    assert not evidence["output"].exists()


def test_summary_rejects_resealed_validation_score_different_from_original_view(evidence, monkeypatch):
    directory = Path(evidence["directories"][0]["directory"])
    path = directory / "汇总指标.json"
    payload = json.loads(path.read_text())
    payload["本次合法HF_LF独立重算"]["合法HF原macro_v1选分独立重算_摄氏度"] += 1.0
    _write_json(path, payload)
    _reseal(directory)
    _pin_directory(evidence)
    with pytest.raises(ValueError, match="选分|视图|HF|指标"):
        _execute(evidence, monkeypatch)
    assert not evidence["output"].exists()


def test_summary_requires_every_full_source_qualification_before_output(evidence, monkeypatch):
    module = _module()
    def qualify(**kwargs):
        if kwargs["seed"] == 4:
            raise ValueError("synthetic fifth completed source rejected")
        return copy.deepcopy(evidence["qualifications"][kwargs["seed"]])
    monkeypatch.setattr(module, "qualify_task11_mlp_hf_source", qualify)
    with pytest.raises(ValueError, match="fifth"):
        module.summarize_task11_mlp_hf_formal(
            run_root=evidence["run_root"],
            output=evidence["output"], project_root=evidence["root"], **evidence["arguments"])
    assert not evidence["output"].exists()


def test_summary_rechecks_full_qualification_and_refuses_changed_originals(evidence, monkeypatch):
    module = _module()
    calls = []
    def qualify(**kwargs):
        calls.append(kwargs["seed"])
        result = copy.deepcopy(evidence["qualifications"][kwargs["seed"]])
        if len(calls) > 5:
            result["完整原件SHA256"]["training.jsonl"] = "0" * 64
        return result
    monkeypatch.setattr(module, "qualify_task11_mlp_hf_source", qualify)
    with pytest.raises(ValueError, match="变化|改变|漂移|原件"):
        module.summarize_task11_mlp_hf_formal(
            run_root=evidence["run_root"],
            output=evidence["output"], project_root=evidence["root"], **evidence["arguments"])
    assert not evidence["output"].exists()


@pytest.mark.parametrize("kind", ["existing", "outside", "symlink"])
def test_summary_refuses_existing_outside_or_symlink_output(evidence, monkeypatch, kind):
    if kind == "existing":
        evidence["output"].mkdir()
        (evidence["output"] / "用户原件.txt").write_text("unchanged")
    elif kind == "outside":
        evidence["output"] = evidence["root"].parent / "不创建项目外目录"
    else:
        actual = evidence["root"] / "符号链接目标"
        actual.mkdir()
        alias = evidence["root"] / "链接"
        alias.symlink_to(actual, target_is_directory=True)
        evidence["output"] = alias / "不创建"
    with pytest.raises((ValueError, FileExistsError)):
        _execute(evidence, monkeypatch)
    if kind == "existing":
        assert (evidence["output"] / "用户原件.txt").read_text() == "unchanged"
    else:
        assert not evidence["output"].exists()


def test_summary_never_writes_report_into_formal_training_tree(evidence, monkeypatch):
    evidence["output"] = evidence["run_root"] / "正式MLP_HF_seed0/不应写入的报告"
    with pytest.raises(ValueError, match="正式|训练|输出"):
        _execute(evidence, monkeypatch)
    assert not evidence["output"].exists()


def test_summary_rejects_resealed_csv_different_from_original_watts(evidence, monkeypatch):
    import polars as pl
    directory = Path(evidence["directories"][0]["directory"])
    path = directory / "指标明细.csv"
    frame = pl.read_csv(path).with_columns(
        (pl.col("原定义相对平衡分母_瓦") + 1.0).alias("原定义相对平衡分母_瓦"))
    frame.write_csv(path)
    _reseal(directory)
    _pin_directory(evidence)
    with pytest.raises(ValueError, match="CSV|瓦数|逐行|分母"):
        _execute(evidence, monkeypatch)
    assert not evidence["output"].exists()


def test_summary_rejects_resealed_nonfinite_validation_value(evidence, monkeypatch):
    directory = Path(evidence["directories"][0]["directory"])
    path = directory / "汇总指标.json"
    payload = json.loads(path.read_text())
    payload["本次合法HF_LF独立重算"]["合法LF逐材料节点与真实体积RMSE_摄氏度"]["Cu"]["node"] = float("nan")
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    _reseal(directory)
    _pin_directory(evidence)
    with pytest.raises(ValueError, match="有限|JSON|NaN"):
        _execute(evidence, monkeypatch)
    assert not evidence["output"].exists()


@pytest.mark.parametrize("kind", ["HF", "能源"])
def test_summary_rechecks_actual_bytes_after_source_qualification(evidence, monkeypatch, kind):
    module = _module()
    calls = []
    def qualify(**kwargs):
        calls.append(kwargs["seed"])
        if len(calls) == 10:
            path = (Path(evidence["qualifications"][0]["目录"]) / "metrics.json" if kind == "HF"
                    else Path(evidence["directories"][0]["directory"]) / "原始能量.jsonl")
            path.write_text(path.read_text() + "\n")
        return copy.deepcopy(evidence["qualifications"][kwargs["seed"]])
    monkeypatch.setattr(module, "qualify_task11_mlp_hf_source", qualify)
    with pytest.raises(ValueError, match="SHA|变化|原件"):
        module.summarize_task11_mlp_hf_formal(
            run_root=evidence["run_root"],
            output=evidence["output"], project_root=evidence["root"], **evidence["arguments"])
    assert not evidence["output"].exists()


def test_cli_is_explicit_cpu_without_training_or_gpu_option():
    import os
    import subprocess
    import sys
    assert SCRIPT.is_file()
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--help"],
                            capture_output=True, text=True,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False)
    assert result.returncode == 0
    assert "纯CPU" in result.stdout and "--energy-directories" in result.stdout
    assert "--energy-directories-sha" in result.stdout
    assert "--device" not in result.stdout and "--registry-sha" in result.stdout


def test_summary_explicitly_sums_lf_hf_cost_epochs_and_consumption_once(evidence, monkeypatch):
    result = _execute(evidence, monkeypatch)
    totals = result["五种子累计总量"]
    assert totals["HF训练成本_秒"] == 110.0
    assert totals["LF训练成本_秒"] == 60.0
    assert totals["LF与HF训练成本合计_秒"] == 170.0
    assert totals["HF累计轮次"] == 2010
    assert totals["LF累计轮次"] == 1015
    assert totals["LF累计实际消费"]["物理配点"] == 256 * 1015
    assert totals["HF累计实际消费"]["HF观测点"] == 29593 * 2010
    assert result["限制"]["分段CUDA与全预算连续训练等价已验证"] is False
    report = (evidence["output"] / "中文验收.md").read_text()
    assert "五种子累计总量" in report and "连续全预算" in report


def test_summary_refuses_balanced_original_watts_and_csv_resealed_together(evidence, monkeypatch):
    import polars as pl
    directory = Path(evidence["directories"][0]["directory"])
    path = directory / "原始能量.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows[:2]:
        row["cooling_heat_w"] += 1.0
        row["convection_heat_w"] -= 1.0
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    csv = directory / "指标明细.csv"
    target = (pl.col("功率_瓦") == rows[0]["power_w"]) & (pl.col("时刻_秒") == rows[0]["time_s"])
    frame = pl.read_csv(csv).with_columns(
        pl.when(target).then(pl.col("水冷散热_瓦") + 1.0).otherwise(
            pl.col("水冷散热_瓦")).alias("水冷散热_瓦"),
        pl.when(target).then(pl.col("对流散热_瓦") - 1.0).otherwise(
            pl.col("对流散热_瓦")).alias("对流散热_瓦"))
    frame.write_csv(csv)
    _reseal(directory)
    with pytest.raises(ValueError, match="SHA|外部|原件|能源"):
        _execute(evidence, monkeypatch)
    assert not evidence["output"].exists()


@pytest.mark.parametrize("mode", ["missing", "inactive", "duplicate", "old_record",
                                  "catalog_sha", "source_sha", "registry_sha", "token_substring",
                                  "duplicate_record", "inactive_suffix"])
def test_root_summary_gate_refuses_before_catalog_source_model_or_raw_reads(evidence, monkeypatch, mode):
    ledger = evidence["ledger"]
    text = ledger.read_text()
    if mode == "missing":
        ledger.unlink()
    elif mode == "inactive":
        ledger.write_text(text.replace("status=active", "status=inactive"))
    elif mode == "duplicate":
        ledger.write_text(text + text.replace("录-0110", "录-0111"))
    elif mode == "old_record":
        ledger.write_text(text.replace("录-0110", "录-0101"))
    elif mode == "catalog_sha":
        evidence["arguments"]["energy_catalog_sha"] = "0" * 64
    elif mode == "source_sha":
        ledger.write_text(text.replace(sha256_file(SCRIPT), "0" * 64))
    elif mode == "registry_sha":
        ledger.write_text(text.replace(evidence["arguments"]["registry_sha"], "0" * 64))
    elif mode == "token_substring":
        ledger.write_text(text.replace(SUMMARY_TOKEN, "不合法前缀" + SUMMARY_TOKEN))
    elif mode == "duplicate_record":
        ledger.write_text(text + "| 录-0110 | 无关但重复编号的合成行 | 不应授权 |\n")
    else:
        ledger.write_text(text.replace(" | 仅合成", "; status=inactive | 仅合成"))
    module = _module()
    for name in ("qualify_task11_mlp_hf_source", "_read_json", "verify_task11_energy_raw"):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: pytest.fail("ROOT gate must be first"))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("ROOT gate before model load"))
    def gate_only_sha(path):
        if Path(path).resolve() not in {SCRIPT.resolve(), ledger.resolve()}:
            pytest.fail("ROOT gate must precede catalog and registered data hashes")
        return sha256_file(path)
    monkeypatch.setattr(module, "sha256_file", gate_only_sha)
    with pytest.raises(ValueError, match="ROOT|SHA|原件|活动|唯一|录号"):
        module.summarize_task11_mlp_hf_formal(
            run_root=evidence["run_root"], output=evidence["output"],
            project_root=evidence["root"], **evidence["arguments"])
    assert not evidence["output"].exists()


@pytest.mark.parametrize("mode", ["missing_hash", "missing_file", "extra_file", "bad_format",
                                  "not_mapping", "wrong_order", "boolean_seed"])
def test_energy_catalog_requires_fixed_order_and_strict_external_six_hashes(evidence, monkeypatch, mode):
    row = evidence["directories"][0]
    if mode == "missing_hash":
        row.pop("六工件SHA256")
    elif mode == "missing_file":
        row["六工件SHA256"].pop("审计工件SHA256.json")
    elif mode == "extra_file":
        row["六工件SHA256"]["自选原件.json"] = "0" * 64
    elif mode == "bad_format":
        row["六工件SHA256"]["原始能量.jsonl"] = "BAD_SHA"
    elif mode == "not_mapping":
        row["六工件SHA256"] = []
    elif mode == "wrong_order":
        evidence["directories"].reverse()
    else:
        row["seed"] = False
    _refresh_catalog(evidence)
    module = _module()
    monkeypatch.setattr(module, "qualify_task11_mlp_hf_source",
                        lambda **kwargs: pytest.fail("strict catalog schema before qualification"))
    with pytest.raises(ValueError, match="目录|SHA|字段|固定|身份|seed"):
        module.summarize_task11_mlp_hf_formal(
            run_root=evidence["run_root"], output=evidence["output"],
            project_root=evidence["root"], **evidence["arguments"])
    assert not evidence["output"].exists()


def test_energy_catalog_rejects_duplicate_json_fields(evidence, monkeypatch):
    catalog = evidence["catalog"]
    catalog.write_text(catalog.read_text().replace('"seed": 0', '"seed": 0, "seed": 0', 1))
    evidence["arguments"]["energy_catalog_sha"] = sha256_file(catalog)
    _refresh_gate(evidence)
    with pytest.raises(ValueError, match="重复"):
        _execute(evidence, monkeypatch)
    assert not evidence["output"].exists()


def test_energy_catalog_actual_byte_change_rejected_before_qualification(evidence, monkeypatch):
    evidence["catalog"].write_text(evidence["catalog"].read_text() + "\n")
    module = _module()
    monkeypatch.setattr(module, "qualify_task11_mlp_hf_source",
                        lambda **kwargs: pytest.fail("catalog actual SHA before qualification"))
    with pytest.raises(ValueError, match="SHA|目录|登记"):
        module.summarize_task11_mlp_hf_formal(
            run_root=evidence["run_root"], output=evidence["output"],
            project_root=evidence["root"], **evidence["arguments"])
    assert not evidence["output"].exists()


def test_resealed_six_hashes_and_catalog_sha_cannot_replace_root_anchor(evidence, monkeypatch):
    directory = Path(evidence["directories"][0]["directory"])
    path = directory / "原始能量.jsonl"
    path.write_text(path.read_text() + "\n")
    _reseal(directory)
    evidence["directories"][0]["六工件SHA256"] = {
        name: sha256_file(directory / name) for name in ENERGY_NAMES}
    _refresh_catalog(evidence, update_gate=False)
    module = _module()
    monkeypatch.setattr(module, "qualify_task11_mlp_hf_source",
                        lambda **kwargs: pytest.fail("root catalog SHA cannot be caller-resealed"))
    with pytest.raises(ValueError, match="ROOT|SHA|活动|唯一"):
        module.summarize_task11_mlp_hf_formal(
            run_root=evidence["run_root"], output=evidence["output"],
            project_root=evidence["root"], **evidence["arguments"])
    assert not evidence["output"].exists()


@pytest.mark.parametrize("relative", ["研究记录/任务07_旧部署/新增报告",
                                      "研究记录/任务11_外部对照/正式Howard训练/新增报告",
                                      "研究记录/任务11_外部对照/中间新层/新增报告"])
def test_output_is_only_new_direct_child_of_task11_external_base(evidence, monkeypatch, relative):
    evidence["output"] = evidence["root"] / relative
    with pytest.raises(ValueError, match="输出|直属|任务11|目录"):
        _execute(evidence, monkeypatch)
    assert not evidence["output"].exists()


@pytest.mark.parametrize("kind", ["catalog", "ROOT", "SOURCE"])
def test_catalog_root_and_source_are_locked_until_last_qualification(evidence, monkeypatch, kind):
    module = _module()
    calls, changed = [], {"source": False}
    def qualify(**kwargs):
        calls.append(kwargs["seed"])
        if len(calls) == 10:
            if kind == "SOURCE":
                changed["source"] = True
            else:
                path = evidence["catalog"] if kind == "catalog" else evidence["ledger"]
                path.write_text(path.read_text() + "\n")
        return copy.deepcopy(evidence["qualifications"][kwargs["seed"]])
    def actual_sha(path):
        if Path(path).resolve() == SCRIPT.resolve() and changed["source"]:
            return "0" * 64
        return sha256_file(path)
    monkeypatch.setattr(module, "qualify_task11_mlp_hf_source", qualify)
    monkeypatch.setattr(module, "sha256_file", actual_sha)
    with pytest.raises(ValueError, match="SHA|变化|ROOT|SOURCE|原件"):
        module.summarize_task11_mlp_hf_formal(
            run_root=evidence["run_root"], output=evidence["output"],
            project_root=evidence["root"], **evidence["arguments"])
    assert not evidence["output"].exists()
