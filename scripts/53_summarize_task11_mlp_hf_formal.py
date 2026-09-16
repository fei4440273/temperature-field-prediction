#!/usr/bin/env python
"""任11五种子真实完训与十份独立原能源的纯CPU受限中文聚合。"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Mapping

import polars as pl
import torch

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import build_power_splits
from sic_cu.eval.task11_mlp_hf_energy import (
    FIXED_ORDERS, FIXED_POWERS, FIXED_TIMES, verify_task11_energy_raw,
)
from sic_cu.train.task11_mlp_hf_formal import ROOT_LEDGER, RUN_DIRECTORY, qualify_task11_mlp_hf_source


ENERGY_FILES = (
    "指标明细.csv", "物理分解.csv", "原始能量.jsonl", "原始散度.jsonl", "汇总指标.json",
)
SUMMARY_TOKEN = "TASK11_NEW_MLP_SUMMARY_GATE:v1"
OUTPUT_BASE = Path("研究记录/任务11_外部对照")
MODALITIES = {
    "顶部": "顶部RMSE_摄氏度",
    "absolute_rmse_c": "两环合并绝对RMSE_摄氏度",
    "absolute_mae_c": "两环合并绝对MAE_摄氏度",
    "delta_rmse_c": "两环合并首时刻差分RMSE_摄氏度",
    "delta_mae_c": "两环合并首时刻差分MAE_摄氏度",
}
REGISTRATION_FIELDS = {
    "registry": ("登记原件", "登记SHA256", "正式登记原件", "正式登记SHA256"),
    "source_tar": ("源码归档原件", "源码归档SHA256", "源码归档原件", "源码归档SHA256"),
    "lf_catalog": ("LF目录原件", "LF目录SHA256", None, "五LF身份清单SHA256"),
    "hf_data_catalog": ("HF数据目录原件", "HF数据目录SHA256", None, "HF开发数据清单SHA256"),
}


def _project_path(value: str | Path, root: Path, *, file: bool = False) -> Path:
    supplied = Path(value)
    candidate = supplied if supplied.is_absolute() else root / supplied
    resolved = candidate.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("任11聚合只允许项目内来源与全新输出，不得越界")
    for path in (candidate, *candidate.parents):
        if path == root:
            break
        if path.is_symlink():
            raise ValueError("任11聚合来源与输出不得经过符号链接")
    if file and not resolved.is_file():
        raise ValueError("任11聚合原件必须是项目内真实普通文件")
    return resolved


def _new_output(value: str | Path, root: Path) -> Path:
    destination = _project_path(value, root)
    if destination.is_relative_to(root / RUN_DIRECTORY):
        raise ValueError("任11聚合输出不得写入正式训练树或污染训练/能源原件目录")
    if destination.parent != root / OUTPUT_BASE:
        raise ValueError("任11聚合输出只能是研究记录/任务11_外部对照的全新直属独立目录")
    if destination.exists():
        raise FileExistsError("任11已有聚合报告目录不可覆盖")
    return destination


def _json(text: str) -> Any:
    def reject_constant(value):
        raise ValueError(f"任11聚合JSON不得含非有限数值：{value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("任11聚合JSON不得有重复字段")
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def _read_json(path: Path) -> Any:
    return _json(path.read_text(encoding="utf-8"))


def _number(value: Any, name: str, *, integer: bool = False) -> int | float:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (integer and type(value) is not int)):
        raise ValueError(f"任11聚合指标必须是有限非负{'整数' if integer else '数值'}：{name}")
    return value


def five_seed_statistics(values: list[int | float]) -> dict[str, Any]:
    if len(values) != 5:
        raise ValueError("任11正式统计只允许五个不同seed，不得用缺席或重复样本")
    numeric = [float(_number(value, "五种子统计")) for value in values]
    return {"样本数": 5, "ddof": 1, "均值": statistics.fmean(numeric),
            "样本标准差": statistics.stdev(numeric), "最小值": min(numeric),
            "最大值": max(numeric), "逐种子": numeric}


def _statistics_tree(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    keys = set(records[0])
    if len(records) != 5 or any(set(record) != keys for record in records):
        raise ValueError("任11五种子统计字段不完整或口径不一致")
    result = {}
    for key in sorted(keys):
        values = [record[key] for record in records]
        result[key] = (_statistics_tree(values) if all(isinstance(value, dict) for value in values)
                       else five_seed_statistics(values))
    return result


def _sum_tree(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    if len(records) != 5 or any(set(record) != set(records[0]) for record in records):
        raise ValueError("任11累计总量必须来自五个固定运行身份的同口径原统计")
    result = {}
    for key in sorted(records[0]):
        values = [record[key] for record in records]
        if all(isinstance(value, dict) for value in values):
            result[key] = _sum_tree(values)
        else:
            numeric = [_number(value, "累计总量") for value in values]
            total = sum(numeric) if all(type(value) is int for value in numeric) else math.fsum(numeric)
            result[key] = _number(total, "累计总量")
    return result


def _digest(digest: Any) -> str:
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("任11原件SHA256格式不正确")
    return digest


def _check_hash(path: Path, digest: str) -> str:
    _digest(digest)
    actual = sha256_file(path)
    if actual != digest:
        raise ValueError(f"任11实际原件SHA256与登记不一致：{path.name}")
    return actual


def _summary_gate(root: Path, catalog_sha: str, registry_sha: str) -> dict[str, str]:
    _digest(catalog_sha)
    _digest(registry_sha)
    ledger = _project_path(ROOT_LEDGER, root, file=True)
    ledger_sha = sha256_file(ledger)
    text = ledger.read_text(encoding="utf-8")
    _check_hash(ledger, ledger_sha)
    source_sha = _digest(sha256_file(Path(__file__)))
    expected = (f"{SUMMARY_TOKEN}; status=active; ENERGY_CATALOG_SHA256={catalog_sha}; "
                f"SOURCE_SHA256={source_sha}; HF_YAML_SHA256={registry_sha}")
    numbered, matches = [], []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) >= 4 and re.fullmatch(r"录-\d{4}", fields[1]):
            numbered.append(fields[1])
            if fields[2].startswith(SUMMARY_TOKEN):
                matches.append((fields[1], fields[2]))
    if (len(matches) != 1 or matches[0][1] != expected
            or int(matches[0][0][2:]) <= 101 or numbered.count(matches[0][0]) != 1):
        raise ValueError("任11聚合ROOT缺录号>101的唯一活动三SHA登记，禁止读取目录/训练/能源原件")
    return {"ROOT记录号": matches[0][0], "ROOT登记原件": str(ledger),
            "ROOT登记SHA256": ledger_sha, "唯一活动行": expected,
            "独立能源目录登记SHA256": catalog_sha, "聚合器源码SHA256": source_sha,
            "HF正式登记SHA256": registry_sha}


def _original_hashes(run: Path, mapping: Mapping[str, str], root: Path) -> dict[str, str]:
    if not isinstance(mapping, dict) or not {"best.pt", "final.pt", "metrics.json"} <= set(mapping):
        raise ValueError("任11完整HF来源原件清单缺观测best/真实final/训练成本")
    result = {}
    for relative, digest in mapping.items():
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("任11HF原件清单必须是运行目录内相对路径")
        path = _project_path(run / relative, root, file=True)
        if not path.is_relative_to(run):
            raise ValueError("任11HF原件清单不得借用其他目录")
        result[str(path)] = _check_hash(path, digest)
    return result


def _fixed_energy_directories(
    rows: list[Mapping[str, Any]], run_root: Path, root: Path,
) -> tuple[dict[tuple[int, str], Path], dict[tuple[int, str], dict[str, str]]]:
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("任11必须显式登记五seed各best/final共十份能源目录")
    result, pins = {}, {}
    identities = [(seed, state) for seed in range(5) for state in ("best", "final")]
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"seed", "state", "directory", "六工件SHA256"}:
            raise ValueError("任11能源目录表每行必须精确包含seed/state/directory/六工件SHA256四字段")
        seed, state = row["seed"], row["state"]
        if type(seed) is not int or seed not in range(5) or state not in {"best", "final"}:
            raise ValueError("任11能源目录只允许五个固定seed的best/final身份")
        identity = (seed, state)
        if identity != identities[index]:
            raise ValueError("任11能源目录必须按固定seed0..4各best/final的十身份顺序登记")
        external = row["六工件SHA256"]
        if not isinstance(external, dict) or set(external) != set(ENERGY_FILES) | {"审计工件SHA256.json"}:
            raise ValueError("任11能源目录必须独立事前锁定完整六工件SHA256字段")
        pins[identity] = {name: _digest(digest) for name, digest in external.items()}
        directory = _project_path(row["directory"], root)
        label = "观测最佳" if state == "best" else "训练末"
        if (identity in result or directory.parent != run_root / f"正式MLP_HF_seed{seed}"
                or not directory.is_dir() or not re.fullmatch(
                    rf"独立原能源_{label}_\d{{8}}T\d{{6}}\+0800", directory.name)):
            raise ValueError("任11能源必须是同seed固定运行目录内不同状态的规范真实原件")
        result[identity] = directory
    if set(result) != {(seed, state) for seed in range(5) for state in ("best", "final")}:
        raise ValueError("任11十份能源身份存在缺席或重复")
    return result, pins


def _energy_hashes(directory: Path, root: Path, external: Mapping[str, str]) -> dict[str, str]:
    if {path.name for path in directory.iterdir()} != set(ENERGY_FILES) | {"审计工件SHA256.json"}:
        raise ValueError("任11能源原件必须完整包含五文件与独立SHA清单，不能有残缺临时工件")
    if not isinstance(external, dict) or set(external) != set(ENERGY_FILES) | {"审计工件SHA256.json"}:
        raise ValueError("任11能源缺目录外独立锁定的六工件SHA256，不能自封身份")
    actual = {name: _check_hash(_project_path(directory / name, root, file=True), digest)
              for name, digest in external.items()}
    manifest_path = _project_path(directory / "审计工件SHA256.json", root, file=True)
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or set(manifest) != set(ENERGY_FILES):
        raise ValueError("任11能源清单必须逐字登记五份原始工件")
    hashes = {name: _check_hash(_project_path(directory / name, root, file=True), manifest[name])
              for name in ENERGY_FILES}
    if hashes != {name: actual[name] for name in ENERGY_FILES}:
        raise ValueError("任11能源内清单与事前外部原件身份不一致")
    return actual


def _validation_metrics(payload: Mapping[str, Any], view: Mapping[str, Any], run: Path) -> dict[str, Any]:
    selected = payload.get("本次合法HF_LF独立重算")
    if (not isinstance(selected, dict) or selected.get("合法HF验证功率_瓦") != [115.2, 403.0, 630.5]
            or selected.get("合法HF全部Top点") != 7272 or selected.get("合法HF全部传感点") != 752
            or selected.get("本独立重算不回调训练或选模") is not True):
        raise ValueError("任11能源验证必须覆盖合法三功率全部HF点且不回调选模")
    score = _number(selected.get("合法HF原macro_v1选分独立重算_摄氏度"), "HF合法macro_v1选分")
    expected = _number(view.get("validation_selection_score_c"), "原模型视图选分")
    if not math.isclose(score, expected, rel_tol=0, abs_tol=1e-4):
        raise ValueError("任11能源HF选分与冻结原模型视图不一致")
    raw_modalities, original = selected.get("合法HF分模态独立重算"), view.get("validation_sensor")
    if (not isinstance(raw_modalities, dict) or set(raw_modalities) != set(MODALITIES)
            or not isinstance(original, dict) or set(original) != set(MODALITIES)):
        raise ValueError("任11HF分模态统计必须保留冻结顶部及两环合并原指标")
    modalities = {}
    for raw, chinese in MODALITIES.items():
        value = _number(raw_modalities[raw], "HF分模态")
        if not math.isclose(value, _number(original[raw], "原模型分模态"), rel_tol=0, abs_tol=1e-4):
            raise ValueError("任11能源HF分模态指标与原模型视图不一致")
        modalities[chinese] = value
    reconstructed = (raw_modalities["顶部"] + 0.2 * raw_modalities["absolute_rmse_c"]
                     + raw_modalities["delta_rmse_c"]) / 2.2
    if not math.isclose(score, reconstructed, rel_tol=0, abs_tol=1e-4):
        raise ValueError("任11合法macro_v1必须与冻结分模态权重相符")
    lf_powers = selected.get("合法LF验证功率_瓦")
    expected_powers = build_power_splits(str(run / "config_snapshot/splits.yaml")).simulation_validation
    if (not isinstance(lf_powers, list) or len(lf_powers) != 10
            or any(type(power) not in (int, float) or not math.isfinite(power) for power in lf_powers)
            or set(lf_powers) != set(expected_powers)):
        raise ValueError("任11LF统计仅允许快照冻结的十个合法验证功率")
    lf = selected.get("合法LF逐材料节点与真实体积RMSE_摄氏度")
    if (not isinstance(lf, dict) or set(lf) != {"Cu", "SiC"}
            or any(not isinstance(per_material, dict) or set(per_material) != {"node", "volume"}
                   for per_material in lf.values())):
        raise ValueError("任11合法LF必须保留Cu/SiC节点及真实体积四口径")
    return {"HF合法macro_v1_摄氏度": score, "HF合法分模态": modalities,
            "LF合法逐材料与口径": {
                material: {"节点RMSE_摄氏度": _number(lf[material]["node"], "LF节点RMSE"),
                           "体积RMSE_摄氏度": _number(lf[material]["volume"], "LF体积RMSE")}
                for material in ("Cu", "SiC")}}


def _energy_record(
    directory: Path, run: Path, seed: int, state: str, qualified: Mapping[str, Any],
    registration: Mapping[str, Any], root: Path, external: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    hashes = _energy_hashes(directory, root, external)
    payload = _read_json(directory / "汇总指标.json")
    if not isinstance(payload, dict):
        raise ValueError("任11能源汇总必须是结构化原字典")
    model_path = run / (state + ".pt")
    selected_sha = qualified["完整原件SHA256"][state + ".pt"]
    expected = {"运行种子": seed, "运行臂": "新MLP_PiNN", "审核状态": state,
                "选择模型SHA256": selected_sha, "完整原件SHA256": qualified["完整原件SHA256"],
                "本seed新LF最佳检查点SHA256": qualified["本seed新LF最佳检查点SHA256"],
                "本seedLF初始张量SHA256": qualified["本seedLF初始张量SHA256"],
                "功率_瓦": FIXED_POWERS, "时刻_秒": FIXED_TIMES, "求积阶数": FIXED_ORDERS,
                "模型计算dtype": "float64"}
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("任11能源状态、源原件、模型SHA或冻结双阶时间功率身份不一致")
    flags = {"只读原件审核前后SHA一致": True, "旧固定TEST温度读取": False,
             "模拟测试功率温度读取": False, "名义能量不证明原FEM热预算或内部温度真值": True,
             "工程安全阈值已建立": False, "结果不用于训练选模": True}
    if any(payload.get(key) is not value for key, value in flags.items()):
        raise ValueError("任11能源不能改写原件不变、TEST封存、名义安全及不选模合同")
    if _project_path(payload.get("运行目录", ""), root) != run:
        raise ValueError("任11能源运行原目录与本seed规范身份不一致")
    for key, (_, _, path_field, sha_field) in REGISTRATION_FIELDS.items():
        if payload.get(sha_field) != registration[key + "_sha"]:
            raise ValueError("任11能源四份事前登记SHA与完训资格不一致")
        if path_field and _project_path(payload.get(path_field, ""), root, file=True) != registration[key]:
            raise ValueError("任11能源事前来源路径与完训资格不一致")
    audit = {"原始能量": [_json(line) for line in (directory / "原始能量.jsonl").read_text(
                encoding="utf-8").splitlines()],
             "原始散度": [_json(line) for line in (directory / "原始散度.jsonl").read_text(
                encoding="utf-8").splitlines()],
             "指标明细": pl.read_csv(directory / "指标明细.csv"),
             "物理分解": pl.read_csv(directory / "物理分解.csv"), "汇总": payload}
    verified = verify_task11_energy_raw(audit)
    if any(not math.isclose(_number(payload.get(key), key), value, rel_tol=1e-10, abs_tol=1e-7)
           for key, value in verified.items()):
        raise ValueError("任11能源汇总中的双阶统计与原瓦数重新计算不一致")
    screening = payload.get("名义吸收归一筛查")
    screening_expected = {"均值阈值": 0.05, "95分位阈值": 0.10,
                          "均值通过": verified["吸收功率归一宏均值"] <= 0.05,
                          "95分位通过": verified["吸收功率归一95分位"] <= 0.10,
                          "不代表工程安全": True}
    if (not isinstance(screening, dict) or screening != screening_expected
            or any(type(screening[key]) is not bool for key in ("均值通过", "95分位通过", "不代表工程安全"))):
        raise ValueError("任11名义筛查布尔必须来自原瓦数且不能代表工程安全")
    _check_hash(_project_path(model_path, root, file=True), selected_sha)
    view = torch.load(model_path, map_location="cpu", weights_only=True)
    if view.get("seed") != seed:
        raise ValueError("任11能源原模型视图seed与完整来源身份不一致")
    record = _validation_metrics(payload, view, run)
    record["独立原能源"] = verified
    return record, hashes


def _training_record(qualified: Mapping[str, Any], lf_row: Mapping[str, Any], root: Path) -> dict[str, Any]:
    run = Path(qualified["目录"])
    hf = _read_json(run / "metrics.json")
    lf_run = _project_path(lf_row["目录"], root)
    lf_metrics = _project_path(lf_run / "metrics.json", root, file=True)
    if (lf_row.get("最佳检查点SHA256") != qualified["本seed新LF最佳检查点SHA256"]
            or lf_row.get("LF初始张量SHA256") != qualified["本seedLF初始张量SHA256"]):
        raise ValueError("任11聚合LF初始身份与HF完训资格不一致")
    _check_hash(lf_metrics, lf_row["真实原件SHA256"][str(lf_metrics.relative_to(root))])
    lf = _read_json(lf_metrics)
    lf_epochs = _number(lf_row["实际轮次"], "LF累计轮次", integer=True)
    hf_seconds = _number(qualified["本人HF累计真实成本秒"], "HF完整训练成本")
    lf_seconds = _number(qualified["本人LF累计真实成本秒"], "LF完整训练成本")
    if (lf.get("epochs_completed") != lf_epochs or lf.get("seed") != qualified["seed"]
            or lf.get("status") != "completed_current_protocol_lf_only"
            or lf.get("training_seconds") != lf_seconds
            or lf_row.get("LF真实多会话累计成本秒") != lf_seconds
            or hf.get("training_seconds") != hf_seconds or hf.get("lf_training_seconds") != lf_seconds):
        raise ValueError("任11HF/LF本人累计训练成本或轮次与完整来源不一致")
    correction = _number(qualified["HF校正实际轮次"], "HF校正轮次", integer=True)
    joint = _number(qualified["HF联合实际轮次"], "HF联合轮次", integer=True)
    epochs = correction + joint
    consumption = hf.get("consumption")
    expected = {"HF观测点": 29593 * epochs, "HF传感器点": 44775 * epochs,
                "HF观测优化步": 15 * epochs, "物理优化步": epochs, "物理配点": 256 * epochs,
                "LF回放点": 60 * 2048 * joint, "LF回放小批": 60 * joint}
    if not isinstance(consumption, dict) or consumption != expected:
        raise ValueError("任11HF累计实际消费与完整日志资格的逐轮预算不闭合")
    for name, value in consumption.items():
        _number(value, name, integer=True)
    return {"HF训练成本_秒": hf_seconds, "LF训练成本_秒": lf_seconds,
            "HF校正轮次": correction, "HF联合轮次": joint, "HF累计轮次": epochs,
            "LF累计轮次": lf_epochs, "HF累计实际消费": consumption,
            "LF累计实际消费": {"训练点": 60 * 8192 * lf_epochs, "优化步": 61 * lf_epochs,
                              "物理配点": 256 * lf_epochs},
            "HF会话CUDA峰值显存_字节": _number(qualified["全部会话CUDA峰值显存字节"], "峰值显存", integer=True)}


def _report(summary: Mapping[str, Any]) -> str:
    lines = ["# 任11新MLP五种子HF与双状态原能源受限验收", "",
             "ROOT唯一活动三SHA先交叉锁定独立能源目录登记、聚合器源码与HF登记；六能源工件逐份对目录外SHA复核。",
             "五份真实HF完整来源与十份独立能源原件均经CPU复核；不运行训练、GPU查询或模型选择。",
             "全部均值±样本标准差均以五个seed为样本，ddof=1；最小值与最大值是种子间范围。",
             "能源95分位先分别从每seed各30点计算，再汇总五个95分位，不混合150点。",
             "名义吸收归一筛查不是工程安全，不证明原FEM有限盘热预算或原装置内部真实温度。",
             "MLP全LF参数联合与DeepONet四投影更新不同，不能据此作跨方法严格单因素因果。",
             "冻结分模态指标只有顶部与两环合并绝对/差分指标，没有独立热端/冷端统计；不得伪造拆分。",
             "本聚合器不直接读取PARQ温度；完整来源资格复核沿用冻结API的合法开发源核验。",
             "LF消费依据完整LF日志资格已确认的实际轮次及每轮60×8192训练点、61步、256物理配点列出。",
             "五种子累计成本只计LF/HF训练身份一次，不因观测最佳与训练末双状态能源重复累计。",
             "未另训连续全预算CUDA对照，分段CUDA与连续训练的等价性仍是P2限界；不据此宣称已经验证。",
             "旧固定TEST与模拟测试温度不用于本次统计，旧B0部署不变。", ""]
    lines.extend(["## 五种子累计总量", "", "| 累计指标 | 五个运行身份合计 |", "|---|---:|"])
    def append_totals(tree, prefix=""):
        for name, value in tree.items():
            label = prefix + name
            if isinstance(value, dict):
                append_totals(value, label + " / ")
            else:
                rendered = str(value) if type(value) is int else f"{value:.9g}"
                lines.append(f"| {label} | {rendered} |")
    append_totals(summary["五种子累计总量"])
    lines.append("")
    for group in ("训练与成本统计", "观测最佳", "训练末"):
        lines.extend(["## " + group, "", "| 指标 | 五seed均值±样本标准差 | 最小值 | 最大值 |",
                      "|---|---:|---:|---:|"])
        def append_tree(tree, prefix=""):
            for name, value in tree.items():
                label = prefix + name
                if "样本标准差" in value:
                    lines.append(f"| {label} | {value['均值']:.9g} ± {value['样本标准差']:.9g} | "
                                 f"{value['最小值']:.9g} | {value['最大值']:.9g} |")
                else:
                    append_tree(value, label + " / ")
        append_tree(summary[group])
        lines.append("")
    lines.extend(["## 原件追踪", "", "精确运行身份、四份登记SHA、五份完整训练原件清单与十份能源六工件SHA见机器汇总.json。",
                  "审核前后完整来源资格及所有已读取原件SHA保持相等；只在全新项目内目录写本报告。", ""])
    return "\n".join(lines)


def summarize_task11_mlp_hf_formal(
    *, run_root: str | Path, energy_catalog: str | Path, energy_catalog_sha: str, output: str | Path,
    registry: str | Path, registry_sha: str, source_tar: str | Path, source_tar_sha: str,
    lf_catalog: str | Path, lf_catalog_sha: str, hf_data_catalog: str | Path, hf_data_catalog_sha: str,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    runs = _project_path(run_root, root)
    if runs != root / RUN_DIRECTORY:
        raise ValueError("任11聚合只允许固定新MLP公平训练根目录与五个HF身份")
    destination = _new_output(output, root)
    gate = _summary_gate(root, energy_catalog_sha, registry_sha)
    catalog_path = _project_path(energy_catalog, root, file=True)
    _check_hash(catalog_path, energy_catalog_sha)
    catalog_rows = _read_json(catalog_path)
    _check_hash(catalog_path, energy_catalog_sha)
    directories, energy_pins = _fixed_energy_directories(catalog_rows, runs, root)
    registration = {"registry": registry, "registry_sha": registry_sha,
                    "source_tar": source_tar, "source_tar_sha": source_tar_sha,
                    "lf_catalog": lf_catalog, "lf_catalog_sha": lf_catalog_sha,
                    "hf_data_catalog": hf_data_catalog, "hf_data_catalog_sha": hf_data_catalog_sha}
    read_hashes = {gate["ROOT登记原件"]: gate["ROOT登记SHA256"], str(catalog_path): energy_catalog_sha}
    for key in REGISTRATION_FIELDS:
        path = _project_path(registration[key], root, file=True)
        read_hashes[str(path)] = _check_hash(path, registration[key + "_sha"])
        registration[key] = path
    qualified = {}
    for seed in range(5):
        run = _project_path(runs / f"正式MLP_HF_seed{seed}", root)
        q = qualify_task11_mlp_hf_source(**registration, output=run, seed=seed, project_root=root)
        if (q.get("状态") != "CPU完整HF来源资格PASS；独立能源仍须另行审核"
                or q.get("seed") != seed or _project_path(q.get("目录", ""), root) != run
                or q.get("旧固定TEST温度读取") is not False or q.get("模拟测试功率温度读取") is not False):
            raise ValueError("任11五个固定HF身份必须均有完整来源资格PASS")
        for key, (path_field, sha_field, _, _) in REGISTRATION_FIELDS.items():
            if (q.get(sha_field) != registration[key + "_sha"]
                    or _project_path(q.get(path_field, ""), root, file=True) != registration[key]):
                raise ValueError("任11完整HF资格与四份登记原件不一致")
        read_hashes.update(_original_hashes(run, q["完整原件SHA256"], root))
        qualified[seed] = q
    lf_data = _read_json(registration["lf_catalog"])
    lf_rows = lf_data.get("五seed新LF") if isinstance(lf_data, dict) else None
    if (not isinstance(lf_rows, list) or len(lf_rows) != 5
            or any(not isinstance(row, dict) or type(row.get("seed")) is not int
                   or row["seed"] != seed for seed, row in enumerate(lf_rows))):
        raise ValueError("任11LF目录必须按固定seed0..4登记五份独立完整来源")
    training, state_records, energy_proof = [], {"best": [], "final": []}, []
    for seed in range(5):
        q, run = qualified[seed], Path(qualified[seed]["目录"])
        training.append(_training_record(q, lf_rows[seed], root))
        for relative, digest in lf_rows[seed]["真实原件SHA256"].items():
            path = _project_path(relative, root, file=True)
            read_hashes[str(path)] = _check_hash(path, digest)
        for state in ("best", "final"):
            directory = directories[(seed, state)]
            record, hashes = _energy_record(directory, run, seed, state, q, registration, root,
                                            energy_pins[(seed, state)])
            state_records[state].append(record)
            read_hashes.update({str(directory / name): digest for name, digest in hashes.items()})
            energy_proof.append({"运行种子": seed, "状态原标识": state, "能源原目录": str(directory),
                                 "六工件SHA256": hashes,
                                 "六工件事前外部SHA256": energy_pins[(seed, state)],
                                 "选择模型SHA256": q["完整原件SHA256"][state + ".pt"]})
    for seed in range(5):
        after = qualify_task11_mlp_hf_source(**registration, output=Path(qualified[seed]["目录"]),
                                           seed=seed, project_root=root)
        if after != qualified[seed]:
            raise ValueError("任11聚合前后完整HF资格或原件SHA发生变化，禁止写报告")
    for identity, directory in directories.items():
        expected = next(row["六工件SHA256"] for row in energy_proof
                        if (row["运行种子"], row["状态原标识"]) == identity)
        if _energy_hashes(directory, root, energy_pins[identity]) != expected:
            raise ValueError("任11聚合前后能源原件发生变化，禁止写报告")
    for path, digest in read_hashes.items():
        _check_hash(_project_path(path, root, file=True), digest)
    if _summary_gate(root, energy_catalog_sha, registry_sha) != gate:
        raise ValueError("任11聚合前后ROOT、目录登记或SOURCE身份变化，禁止写报告")
    totals = _sum_tree([{key: value for key, value in record.items()
                         if key != "HF会话CUDA峰值显存_字节"} for record in training])
    totals["LF与HF训练成本合计_秒"] = math.fsum(
        (totals["LF训练成本_秒"], totals["HF训练成本_秒"]))
    result = {"状态": "任11新MLP五seed完整HF来源及十份原能源CPU受限验收",
              "统计约定": {"种子": [0, 1, 2, 3, 4], "样本数": 5, "ddof": 1,
                          "标准差含义": "仅种子间波动，不是工程容差或独立实测置信区间",
                          "能源95分位先各seed30点计算再统计": True,
                          "训练成本只计五身份不因双能源重复": True,
                          "原件及完整资格审核前后相等": True},
              "限制": {"工程安全合格主张": False, "跨方法严格单因素因果主张": False,
                       "使用能源重新选模": False, "旧固定TEST温度读取": False,
                       "模拟测试功率温度读取": False, "GPU查询": False,
                       "独立热端冷端分拆统计": "冻结能源报告未提供，保持缺席不伪造",
                       "分段CUDA与全预算连续训练等价已验证": False,
                       "直接读取PARQ温度": False,
                       "来源核验": "只复用冻结完整资格API，不新增数据折或标签读取接口"},
              "训练与成本统计": _statistics_tree(training),
              "五种子累计总量": totals,
              "观测最佳": _statistics_tree(state_records["best"]),
              "训练末": _statistics_tree(state_records["final"]),
              "逐种子原统计": [{"运行种子": seed, "训练与成本": training[seed],
                              "观测最佳": state_records["best"][seed], "训练末": state_records["final"][seed]}
                             for seed in range(5)],
              "完整来源资格原件": [qualified[seed] for seed in range(5)],
              "十份独立能源原件": energy_proof, "所有已读取原件SHA256": read_hashes,
              "事前聚合门禁": gate,
              "独立能源目录登记": {"原件": str(catalog_path), "SHA256": energy_catalog_sha,
                               "十份精确身份与事前SHA256": catalog_rows},
              "聚合器源码SHA256": gate["聚合器源码SHA256"]}
    serialized = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    report = _report(result)
    destination = _new_output(destination, root)
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in (("机器汇总.json", serialized), ("中文验收.md", report)):
        with (destination / name).open("x", encoding="utf-8") as stream:
            stream.write(content)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="任11五seed完整HF与十原能源纯CPU中文受限聚合")
    parser.add_argument("--run-root", default=RUN_DIRECTORY)
    parser.add_argument("--energy-directories", "--energy-catalog", dest="energy_catalog", required=True,
                        help="ROOT事前锁定项目内JSON十身份列表，各含seed/state/directory/六工件SHA256")
    parser.add_argument("--energy-directories-sha", "--energy-catalog-sha", dest="energy_catalog_sha",
                        required=True, help="独立目录登记原字节SHA256，须与ROOT唯一活动行一致")
    parser.add_argument("--output", required=True)
    for key in REGISTRATION_FIELDS:
        flag = key.replace("_", "-")
        parser.add_argument("--" + flag, required=True)
        parser.add_argument("--" + flag + "-sha", required=True)
    args = vars(parser.parse_args())
    result = summarize_task11_mlp_hf_formal(**args)
    print(json.dumps({"状态": result["状态"], "完整HF身份数": 5, "独立能源状态数": 10}, ensure_ascii=False))


if __name__ == "__main__":
    main()
