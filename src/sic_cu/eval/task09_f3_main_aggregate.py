"""Read-only, all-or-nothing Task-09 F3 main-series evidence aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tarfile
from typing import Any, Callable, Mapping

import yaml

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file


TASK09_ROOT = PROJECT_ROOT / "研究记录/任务09_高保真数据效率"
ROOT_LEDGER = PROJECT_ROOT / "多保真DeepONet预测精度优化总计划与执行台账.md"
OLD_AGG_REGISTRY = TASK09_ROOT / "正式F3主序列五种子正式聚合前登记.yaml"
OLD_AGG_REGISTRY_SHA = "ae19ab85d6c7e8833986d84e65343b1b54af8fecbd9cc4988c1462d1e6d54131"
OLD_AGG_ARCHIVE = TASK09_ROOT / "任09F3主序列正式聚合五源事前冻结_20260916T094330+0800.tar.gz"
OLD_AGG_ARCHIVE_SHA = "5334745d8eb052db114e0f3e56981cd7917ba44db693e40947fff324edf9a827"
OLD_AGG_REPORT = TASK09_ROOT / "任09F3主序列十五身份正式聚合事前源码冻结与受限验收_20260916T0945.md"
OLD_AGG_REPORT_SHA = "7089327b6186c1008a76392216744dbde42ae1b6cd85329642878701b0e096e1"
NEW_AGG_REGISTRY = TASK09_ROOT / "正式F3主序列五种子正式聚合前登记_v2_严格ROOT先后门禁.yaml"
CATALOG_PATH = TASK09_ROOT / "正式F3主序列十五身份全原件证据清单.yaml"
ORIGINAL_REGISTRY = TASK09_ROOT / "正式HF数据效率F3先导前登记.yaml"
ORIGINAL_SHA = "987ba4bfd9a5b964bff88fa3314413738fa5417b5445e71cf650acbaee7d85e9"
V2_REGISTRY = TASK09_ROOT / "正式F3联合首轮P1修订恢复前登记_录0075强门禁版.yaml"
V2_SHA = "a88ff62c1b24ac288214f20d6d50c642985f9375cd75562b07cfa394e9cf23dc"
V2_ARCHIVE = TASK09_ROOT / "任09F3联合首轮P1修订恢复源码冻结_20260916T064603+0800.tar.gz"
V2_ARCHIVE_SHA = "cef17d1111542c4dbeec3f2d7573dd40cd7d719c54ba31939b87a713b4d08d7a"
V3_REGISTRY = TASK09_ROOT / "正式F3主序列剩余14模型通用入口前登记_录0078.yaml"
V3_SHA = "87816d50efe82fc7a5c218707b1323a1b0d4a8cfd9eceb7c7771bb63c9edfb28"
V3_ARCHIVE = TASK09_ROOT / "任09F3主序列剩余14模型v3源码冻结_20260916T074940+0800.tar.gz"
V3_ARCHIVE_SHA = "695e8b6c9ae9847ed2576609777dee5750c2f84c69f10667a69cd2a27a43765d"
OLD_12HF_REFERENCE = PROJECT_ROOT / (
    "研究记录/任务07_正式五种子重训/"
    "正式五种子HF合法验证逐窗明细_20260916T022125+0800/五种子观测验证摘要.json"
)
OLD_12HF_SHA = "f73f8a42bbdb7393770358809f28f59aedcb031658dd76641cc9d6e89ebac9c9"
V2_SEED0_CANONICAL = (
    "研究记录/任务09_高保真数据效率/正式F3子集训练/"
    "正式F3_primary_HF3_seed0_20260916T064515+0800"
)
V3_CANONICAL_BASE = (
    "研究记录/任务09_高保真数据效率/正式F3子集训练/"
    "正式F3V3_primary_HF{size}_seed{seed}_录0078_20260916T071500+0800"
)
POWERS = {
    3: [55.0, 364.3, 729.0],
    6: [55.0, 216.8, 364.3, 494.2, 593.5, 729.0],
    9: [55.0, 216.8, 254.5, 364.3, 430.0, 494.2, 558.5, 593.5, 729.0],
}
ENERGY_POWERS = [55.0, 115.2, 364.3, 403.0, 630.5, 729.0]
ENERGY_TIMES = [1.0, 10.0, 50.0, 100.0, 200.0]
RUN_HASH_FILES = (
    "training.jsonl", "任09F3阶段报告.json", "阶段_观测最佳.pt", "阶段_训练末.pt",
)
AUDIT_HASH_FILES = (
    "原始能量.jsonl", "原始散度.jsonl", "指标明细.csv", "物理分解.csv", "汇总指标.json",
)
SHA_PATTERN = re.compile(r"[a-f0-9]{64}\Z")
ENERGY_FOLDER_PATTERN = re.compile(r"独立原能源_(观测最佳|训练末)_\d{8}T\d{6}\+0800\Z")


def expected_task09_main_identities() -> list[dict[str, Any]]:
    """Explicit v2 seed0 plus the 14 canonical v3 identities; never scan runs."""
    return [
        {"HF功率数": size, "seed": seed, "序列": "primary",
         "来源": "v2" if (size, seed) == (3, 0) else "v3",
         "canonical目录": (V2_SEED0_CANONICAL if (size, seed) == (3, 0)
                           else V3_CANONICAL_BASE.format(size=size, seed=seed))}
        for size in (3, 6, 9) for seed in range(5)
    ]


def _regular_file(path: Path, name: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"任09{name}须为既有普通原件，不接受缺项或软链：{path}")


def _sha(path: Path, expected: str, name: str) -> None:
    if not isinstance(expected, str) or SHA_PATTERN.fullmatch(expected) is None:
        raise ValueError(f"任09{name}缺显式64位SHA")
    _regular_file(path, name)
    if sha256_file(path) != expected:
        raise ValueError(f"任09{name}现场原件SHA与事后封存清单不符：{path}")


def _project_path(root: Path, relative: Any, name: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"任09{name}须提供固定项目内相对路径")
    path = Path(relative)
    if ".." in path.parts or path.as_posix() != relative:
        raise ValueError(f"任09{name}不能以软链/父级/非规范路径绕过明确身份")
    cursor = root
    for part in path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"任09{name}路径各级不得含软链：{cursor}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"任09{name}不得指向项目外原件")
    return resolved


def validate_task09_main_registry(registry: Mapping[str, Any]) -> None:
    expected = expected_task09_main_identities()
    source_files = {
        "聚合模块": PROJECT_ROOT / "src/sic_cu/eval/task09_f3_main_aggregate.py",
        "唯一CLI": PROJECT_ROOT / "scripts/47_aggregate_task09_f3_main.py",
        "全身份证据清单生成器": PROJECT_ROOT / "scripts/48_prepare_task09_f3_main_evidence.py",
        "合成与真实原件只读测试": PROJECT_ROOT / "tests/test_task09_f3_main_aggregate.py",
    }
    expected_source_sha = {name: sha256_file(path) for name, path in source_files.items()}
    if (registry.get("schema_version") != 2
            or registry.get("旧v1事前登记SHA256") != OLD_AGG_REGISTRY_SHA
            or registry.get("旧v1五源冻结tar_SHA256") != OLD_AGG_ARCHIVE_SHA
            or registry.get("旧v1中文报告_SHA256") != OLD_AGG_REPORT_SHA
            or registry.get("ROOT第二列激活字段须精确为status_active") is not True
            or registry.get("ROOT前登记同时先于证据的账行编号与文件行序") is not True
            or registry.get("ROOT先激活事前YAML和tar唯一同行标志") !=
                "TASK09_F3_MAIN_AGG_GATE:v2"
            or registry.get("ROOT再激活完整证据清单SHA唯一同行标志") !=
                "TASK09_F3_MAIN_EVIDENCE:v2"
            or registry.get("主序列十五身份") != expected
            or registry.get("HF主曲线训练功率_瓦") != POWERS
            or registry.get("十二HF仅任07乙历史参照") is not True
            or registry.get("同一目标误差阈值_摄氏度") is not None
            or registry.get("同一目标误差所需HF数量") is not None
            or registry.get("种子变异不得冒充子集变异") is not True
            or registry.get("子集方差_额外嵌套序列结果缺席") is not None
            or registry.get("五seed样本标准差分母") != 4
            or registry.get("非单调原点必须保留") is not True
            or registry.get("旧固定TEST温度读取") is not False
            or registry.get("主曲线仅F3_F1受阻_F4不存在") is not True
            or registry.get("证据清单固定相对路径") != (
                "研究记录/任务09_高保真数据效率/正式F3主序列十五身份全原件证据清单.yaml")
            or registry.get("证据清单生成前不得正式聚合") is not True
            or registry.get("证据清单须逐份固定训练原件与best_final双能源目录及原件SHA") is not True
            or registry.get("正式结果源文件SHA256") != expected_source_sha
            or registry.get("原录0059登记SHA256") != ORIGINAL_SHA
            or registry.get("录0075_v2登记SHA256") != V2_SHA
            or registry.get("录0075_v2源码归档SHA256") != V2_ARCHIVE_SHA
            or registry.get("录0078_v3登记SHA256") != V3_SHA
            or registry.get("录0078_v3源码归档SHA256") != V3_ARCHIVE_SHA
            or registry.get("HF12历史合法观测SHA256") != OLD_12HF_SHA):
        raise ValueError("任09聚合v2修订事前登记与旧v1锁/ROOT先后/十五身份或原定义不符")


def _strict_rows(path: Path, filename: str) -> None:
    if filename.endswith(".jsonl"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        expected = 60
        fields = ("power_w", "time_s", "quadrature_order")
        schedule = {(power, time, order) for power in ENERGY_POWERS
                    for time in ENERGY_TIMES for order in (16, 64)}
    else:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        expected = 30
        fields = ("功率_瓦", "时刻_秒")
        schedule = {(power, time) for power in ENERGY_POWERS for time in ENERGY_TIMES}
    if len(rows) != expected or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"任09双能源原件{filename}必须有恰好{expected}条有效原记录")
    try:
        observed = [tuple(float(row[field]) for field in fields) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"任09双能源{filename}缺原六功率五时刻双阶固定列") from error
    if len(set(observed)) != expected or set(observed) != schedule:
        raise ValueError(f"任09双能源{filename}功率/时刻/双阶原记录重复或缩水")


def _positive(value: Any, label: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError(f"任09{label}必须是合法有限非负观测/有量纲统计")
    return float(value)


def _verify_energy(
    root: Path, identity: Mapping[str, Any], record: Mapping[str, Any],
    run: Path, run_hashes: Mapping[str, str], state: str,
) -> dict[str, float]:
    if not isinstance(record, dict) or set(record) != {
        "目录", "汇总指标SHA256", "审计工件SHA256", "V3能源入口收据SHA256",
    }:
        raise ValueError("任09best/final双能源须各提供明确目录与三类原件SHA")
    folder = _project_path(root, record["目录"], "独立能源目录")
    label = "观测最佳" if state == "best" else "训练末"
    if (folder.parent != run or folder.is_symlink()
            or ENERGY_FOLDER_PATTERN.fullmatch(folder.name) is None
            or not folder.name.startswith(f"独立原能源_{label}_")
            or not folder.is_dir()):
        raise ValueError("任09独立能源只能是该固定身份的best/final各一个项目内旧审计目录")
    manifest_path = folder / "审计工件SHA256.json"
    _sha(manifest_path, record["审计工件SHA256"], "五工件能源清单")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != set(AUDIT_HASH_FILES):
        raise ValueError("任09双能源必须各通过冻结五工件原SHA清单")
    for filename, expected in manifest.items():
        _sha(folder / filename, expected, filename)
    if manifest["汇总指标.json"] != record["汇总指标SHA256"]:
        raise ValueError("任09能源摘要SHA必须等于冻结五工件清单所载原SHA")
    for filename in AUDIT_HASH_FILES[:4]:
        _strict_rows(folder / filename, filename)
    if identity["来源"] == "v3":
        receipt_path = folder / "任09V3能源入口收据.json"
        _sha(receipt_path, record["V3能源入口收据SHA256"], "V3能源入口收据")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("审核状态") != state
                or receipt.get("运行身份") != {
                    "序列": "primary", "HF功率数": identity["HF功率数"],
                    "seed": identity["seed"],
                }
                or receipt.get("V3下游资格", {}).get("V3来源谱系与会话链合格") is not True
                or receipt.get("旧固定TEST温度读取") is not False
                or receipt.get("冻结能源输出逐文件SHA256") != {
                    **manifest, "审计工件SHA256.json": record["审计工件SHA256"],
                }):
            raise ValueError("任09V3能源收据下游资格或双能源原件SHA与原清单不一致")
    elif record["V3能源入口收据SHA256"] is not None:
        raise ValueError("任09HF3/seed0 v2不得借用V3能源入口收据")
    summary = json.loads((folder / "汇总指标.json").read_text(encoding="utf-8"))
    stage = "阶段_观测最佳.pt" if state == "best" else "阶段_训练末.pt"
    if (summary.get("序列") != "primary"
            or summary.get("HF子集功率数") != identity["HF功率数"]
            or summary.get("本seed") != identity["seed"]
            or summary.get("选中状态") != state
            or summary.get("真实阶段报告SHA256") != run_hashes["任09F3阶段报告.json"]
            or summary.get("训练日志SHA256") != run_hashes["training.jsonl"]
            or summary.get("本模型阶段SHA256") != run_hashes[stage]
            or summary.get("正式事前登记SHA256") != ORIGINAL_SHA
            or summary.get("旧固定测试温度读取") is not False):
        raise ValueError("任09双能源身份/选中状态/训练谱系SHA不符或旧固定TEST温度曾读取")
    if (summary.get("两阶原瓦数与原散度各60行") is not True
            or summary.get("工程平衡与原瓦数逐行一致") is not True
            or summary.get("原定义相对平衡分母已保持") is not True
            or summary.get("功率时刻审核行数") != 30
            or summary.get("功率_瓦") != ENERGY_POWERS
            or summary.get("时刻_秒") != ENERGY_TIMES
            or summary.get("求积阶数") != [16, 64]
            or summary.get("LF本模型逐材料节点与真实体积5%保持资格", {}).get(
                "LF两材料节点与真实体积均守住5%护栏") is not True):
        raise ValueError("任09双能源缺原30点双阶60行/名义原瓦数守恒核验或LF资格")
    return {
        "S": _positive(summary.get("任09子集合法HF选分独立重算_摄氏度"), "合法S"),
        "energy_mean": _positive(summary.get("绝对平衡宏均值_瓦"), "原能源宏均值"),
        "energy_p95": _positive(summary.get("绝对平衡95分位_瓦"), "原能源p95"),
        "normalized_mean": _positive(summary.get("吸收功率归一宏均值"), "原能源归一均值"),
        "normalized_p95": _positive(summary.get("吸收功率归一95分位"), "原能源归一p95"),
    }


def verify_task09_main_evidence(
    preregistration: Mapping[str, Any], evidence_catalog: Mapping[str, Any],
    *, project_root: Path,
    qualify_v2: Callable[[Path, int, int], Mapping[str, Any]],
    qualify_v3: Callable[[Path, int, int], Mapping[str, Any]],
    locked_12hf_scores: Mapping[int, float],
) -> dict[str, Any]:
    """Purely read registered identities and 15 signed pairs; no partial output."""
    identities = expected_task09_main_identities()
    if preregistration.get("主序列十五身份") != identities:
        raise ValueError("任09事前十五固定身份与源码冻结不符")
    entries = evidence_catalog.get("主序列十五份原件")
    if (evidence_catalog.get("schema_version") != 1
            or not isinstance(entries, list) or len(entries) != 15
            or any(not isinstance(entry, dict)
                   or set(entry) != set(identities[index]) | {"训练原件SHA256", "双能源"}
                   for index, entry in enumerate(entries))
            or [tuple(entry.get(key) for key in (
                "HF功率数", "seed", "序列", "来源", "canonical目录"))
                for entry in entries] != [tuple(identity[key] for key in (
                "HF功率数", "seed", "序列", "来源", "canonical目录"))
                for identity in identities]):
        raise ValueError("任09事后证据清单必须严格给十五固定身份且无缺失/重排/冒名")
    if set(locked_12hf_scores) != set(range(5)):
        raise ValueError("任09任07乙12HF合法历史参照五seed缺项")
    scores12 = [_positive(locked_12hf_scores[seed], "12HF乙历史合法S") for seed in range(5)]
    details: list[dict[str, Any]] = []
    for identity, entry in zip(identities, entries):
        size, seed = identity["HF功率数"], identity["seed"]
        run = _project_path(project_root, identity["canonical目录"], "十五模型canonical目录")
        if not run.is_dir() or run.is_symlink():
            raise ValueError("任09正式主曲线十五模型须全部真实目录完成后才能进入下游")
        hashes = entry.get("训练原件SHA256")
        if not isinstance(hashes, dict) or set(hashes) != set(RUN_HASH_FILES):
            raise ValueError("任09十五模型训练日志/阶段报告/best/final原SHA不完整")
        for name, expected in hashes.items():
            _sha(run / name, expected, f"训练原件{name}")
        qualification = (qualify_v2 if identity["来源"] == "v2" else qualify_v3)(run, size, seed)
        if not isinstance(qualification, Mapping) or qualification.get("qualified") is not True:
            raise ValueError("任09任一模型未通过已冻结真实下游最终资格，不准聚合")
        energy = entry.get("双能源")
        if not isinstance(energy, dict) or set(energy) != {"best", "final"}:
            raise ValueError("任09每个模型须best与真实训练final双能源独审")
        best = _verify_energy(project_root, identity, energy["best"], run, hashes, "best")
        final = _verify_energy(project_root, identity, energy["final"], run, hashes, "final")
        details.append({
            "HF功率数": size, "seed": seed, "来源": identity["来源"],
            "canonical目录": identity["canonical目录"],
            "best合法S_摄氏度": best["S"], "final合法S_摄氏度": final["S"],
        "best原能源宏均值_瓦": best["energy_mean"],
        "final原能源宏均值_瓦": final["energy_mean"],
        "best原能源95分位_瓦": best["energy_p95"],
        "final原能源95分位_瓦": final["energy_p95"],
        "best吸收功率归一宏均值": best["normalized_mean"],
        "final吸收功率归一宏均值": final["normalized_mean"],
        "best吸收功率归一95分位": best["normalized_p95"],
        "final吸收功率归一95分位": final["normalized_p95"],
            "best能源资格": (best["normalized_mean"] <= 0.05
                          and best["normalized_p95"] <= 0.10),
            "final能源资格": (final["normalized_mean"] <= 0.05
                           and final["normalized_p95"] <= 0.10),
            "best双能源原件": dict(energy["best"]),
            "final双能源原件": dict(energy["final"]),
            "训练原件SHA256": dict(hashes),
        })

    points = []
    for size in (3, 6, 9):
        subset = [detail for detail in details if detail["HF功率数"] == size]
        if len(subset) != 5 or [row["seed"] for row in subset] != list(range(5)):
            raise ValueError("任09任一主原点少于五个配对seed，不准发布统计")
        best = [row["best合法S_摄氏度"] for row in subset]
        final = [row["final合法S_摄氏度"] for row in subset]
        energy_best = [row["best原能源宏均值_瓦"] for row in subset]
        energy_final = [row["final原能源宏均值_瓦"] for row in subset]
        points.append({
            "HF功率数": size, "固定嵌套训练功率_瓦": POWERS[size],
            "观测最佳S_均值_摄氏度": statistics.mean(best),
            "观测最佳S_样本标准差_摄氏度": statistics.stdev(best),
            "训练末S_均值_摄氏度": statistics.mean(final),
            "训练末S_样本标准差_摄氏度": statistics.stdev(final),
            "best原能源宏均值_五seed均值_瓦": statistics.mean(energy_best),
            "best原能源宏均值_五seed样本标准差_瓦": statistics.stdev(energy_best),
            "final原能源宏均值_五seed均值_瓦": statistics.mean(energy_final),
            "final原能源宏均值_五seed样本标准差_瓦": statistics.stdev(energy_final),
            "best能源安全种子数": sum(row["best能源资格"] for row in subset),
            "final能源安全种子数": sum(row["final能源资格"] for row in subset),
            "五seed仅种子波动": True,
        })
    return {
        "状态": "任09F3主序列十五模型全原件合格的受限合法观测聚合",
        "主序列三个原点": points,
        "十五seed逐模型摘要": details,
        "HF12历史参照_不属新三档同轨迹": {
            "HF功率数": 12, "来源": "任07乙独立历史五seed合法验证",
            "合法S_均值_摄氏度": statistics.mean(scores12),
            "合法S_样本标准差_摄氏度": statistics.stdev(scores12),
            "十二HF历史合法观测原件SHA256": OLD_12HF_SHA,
            "与新十五模型同一HF训练轨迹": False,
        },
        "同一目标误差所需HF数量": None,
        "高保真数量节省比例": None,
        "目标误差或判定统计未事前固定": True,
        "不单调主原点原样保留": True,
        "五seed样本标准差ddof": 1,
        "种子方差不是子集方差": True,
        "旧固定TEST温度读取": False,
        "绝对能源资格不能由数值积分自洽代替": True,
    }


def build_task09_main_evidence_catalog(
    preregistration: Mapping[str, Any], *, project_root: Path,
) -> dict[str, Any]:
    """Discover only the unique two audits inside each fixed, registered run."""
    identities = expected_task09_main_identities()
    if preregistration.get("主序列十五身份") != identities:
        raise ValueError("任09证据生成不得推测或替换事前已锁十五身份")
    records = []
    for identity in identities:
        run = _project_path(project_root, identity["canonical目录"], "证据清单固定身份")
        if not run.is_dir() or run.is_symlink():
            raise ValueError("任09证据清单生成前必须全部十五正式身份完训")
        hashes = {}
        for filename in RUN_HASH_FILES:
            path = run / filename
            _regular_file(path, f"训练原件{filename}")
            hashes[filename] = sha256_file(path)
        pair = {}
        for state in ("best", "final"):
            label = "观测最佳" if state == "best" else "训练末"
            # A run's direct children are inspected only to prove uniqueness, never
            # to select among competing timestamps or borrow a sibling identity.
            folders = [candidate for candidate in run.iterdir()
                       if candidate.name.startswith(f"独立原能源_{label}_")]
            if (len(folders) != 1 or folders[0].is_symlink()
                    or not folders[0].is_dir()
                    or ENERGY_FOLDER_PATTERN.fullmatch(folders[0].name) is None):
                raise ValueError("任09每个固定身份的best/final原能源目录必须分别恰唯一")
            audit = folders[0]
            summary = audit / "汇总指标.json"
            manifest = audit / "审计工件SHA256.json"
            _regular_file(summary, "能源汇总")
            _regular_file(manifest, "冻结五工件能源清单")
            receipt = audit / "任09V3能源入口收据.json"
            if identity["来源"] == "v3":
                _regular_file(receipt, "V3能源入口收据")
            elif receipt.exists():
                raise ValueError("任09录0077的v2恢复身份不得产生V3能源收据")
            pair[state] = {
                "目录": str(audit.relative_to(project_root)),
                "汇总指标SHA256": sha256_file(summary),
                "审计工件SHA256": sha256_file(manifest),
                "V3能源入口收据SHA256": (
                    sha256_file(receipt) if identity["来源"] == "v3" else None
                ),
            }
        records.append({**identity, "训练原件SHA256": hashes, "双能源": pair})
    return {"schema_version": 1, "主序列十五份原件": records}


def validate_task09_aggregate_archive(
    registry: Path, registry_sha: str, archive: Path, archive_sha: str,
) -> None:
    _sha(registry, registry_sha, "聚合事前YAML")
    _sha(archive, archive_sha, "聚合源码tar")
    members = {
        "src/sic_cu/eval/task09_f3_main_aggregate.py",
        "scripts/47_aggregate_task09_f3_main.py",
        "scripts/48_prepare_task09_f3_main_evidence.py",
        "tests/test_task09_f3_main_aggregate.py",
        registry.relative_to(PROJECT_ROOT).as_posix(),
    }
    with tarfile.open(archive, "r:gz") as bundle:
        payloads = bundle.getmembers()
        if (len(payloads) != 5 or {m.name for m in payloads} != members
                or any(not m.isfile() or m.issym() or m.islnk()
                       or Path(m.name).is_absolute() or ".." in Path(m.name).parts
                       for m in payloads)):
            raise ValueError("任09聚合源码归档须恰五份项目内普通冻结原件")
        for member in payloads:
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError("任09聚合源码tar成员不可读取")
            expected = sha256_file(PROJECT_ROOT / member.name)
            if hashlib.sha256(stream.read()).hexdigest() != expected:
                raise ValueError("任09聚合事前tar源码成员与现场原字节SHA不一致")


def _task09_second_column_gate_rows(
    rows: list[str], marker: str,
) -> list[tuple[int, int, str]]:
    """Parse the actual second Markdown column, not matching free-text substrings."""
    accepted = []
    for position, line in enumerate(rows):
        cells = line.split("|", 3)
        if len(cells) < 3 or cells[0] != "":
            continue
        ledger_id = re.fullmatch(r"录-(\d{4})", cells[1].strip())
        if ledger_id is None:
            continue
        tokens = [token.strip() for token in cells[2].strip().split(";")]
        if not tokens or tokens[0] != marker:
            continue
        if len(tokens) < 2 or tokens[1] != "status=active":
            raise ValueError(f"任09ROOT第二列{marker}必须紧接精确status=active规范字段")
        accepted.append((position, int(ledger_id.group(1)), line))
    return accepted


def require_task09_aggregate_root_ledger(
    registry_sha: str, archive: Path, archive_sha: str,
    catalog_sha: str,
) -> None:
    _regular_file(ROOT_LEDGER, "ROOT主台账")
    before = sha256_file(ROOT_LEDGER)
    rows = ROOT_LEDGER.read_text(encoding="utf-8").splitlines()
    if sha256_file(ROOT_LEDGER) != before:
        raise ValueError("任09ROOT主台账审核期间被改写，不可聚合")
    prereg = _task09_second_column_gate_rows(rows, "TASK09_F3_MAIN_AGG_GATE:v2")
    evidence = _task09_second_column_gate_rows(rows, "TASK09_F3_MAIN_EVIDENCE:v2")
    if len(prereg) != 1 or len(evidence) != 1:
        raise ValueError("任09聚合须ROOT前登记门禁及全十五身份证据清单各独立唯一同行激活")
    if prereg[0][0] >= evidence[0][0] or prereg[0][1] >= evidence[0][1]:
        raise ValueError("任09ROOT事前YAML+tar登记须在文件行序和账行编号均先于证据清单登记")
    if (registry_sha not in prereg[0][2] or archive.name not in prereg[0][2]
            or archive_sha not in prereg[0][2]
            or catalog_sha not in evidence[0][2]
            or registry_sha not in evidence[0][2]
            or archive_sha not in evidence[0][2]):
        raise ValueError("任09事前YAML/源码tar及事后完整清单三SHA未在ROOT主账分行真实激活")


def require_task09_aggregate_prereg_root_ledger(
    registry_sha: str, archive: Path, archive_sha: str,
) -> None:
    """Allow future catalog preparation only after the frozen prereg row."""
    _regular_file(ROOT_LEDGER, "ROOT主台账")
    before = sha256_file(ROOT_LEDGER)
    rows = ROOT_LEDGER.read_text(encoding="utf-8").splitlines()
    if sha256_file(ROOT_LEDGER) != before:
        raise ValueError("任09ROOT主台账预登记读取过程中发生变化")
    prereg = _task09_second_column_gate_rows(rows, "TASK09_F3_MAIN_AGG_GATE:v2")
    if (len(prereg) != 1 or registry_sha not in prereg[0][2]
            or archive.name not in prereg[0][2]
            or archive_sha not in prereg[0][2]):
        raise ValueError("任09证据清单准备前必须先有ROOT唯一事前YAML+tar同行激活")


def _task09_real_qualifiers() -> tuple[
    Callable[[Path, int, int], Mapping[str, Any]],
    Callable[[Path, int, int], Mapping[str, Any]],
]:
    from sic_cu.eval import task09_energy
    from sic_cu.train import task09_subset_formal_v2, task09_subset_formal_v3

    task09_subset_formal_v2.require_task09_v2_registration(V2_REGISTRY, V2_SHA)
    task09_subset_formal_v2.validate_task09_v2_source_archive(V2_ARCHIVE, V2_ARCHIVE_SHA)
    task09_subset_formal_v2.require_task09_v2_root_ledger(
        registry_sha256=V2_SHA, archive_path=V2_ARCHIVE,
        archive_sha256=V2_ARCHIVE_SHA,
    )

    def qualify_v2(run: Path, size: int, seed: int) -> dict[str, bool]:
        if (size, seed) != (3, 0):
            raise ValueError("任09v2只能是录0077主HF3/seed0恢复真完整状态")
        task09_energy.qualified_task09_completed_run(
            run, name="primary", size=size, seed=seed,
            registry_path=ORIGINAL_REGISTRY, registry_sha256=ORIGINAL_SHA,
        )
        ledger_rows = ROOT_LEDGER.read_text(encoding="utf-8").splitlines()
        rows = [line for line in ledger_rows if re.match(r"^\|\s*录-0077\s*\|", line)]
        if (len(rows) != 1 or sha256_file(run / "任09F3阶段报告.json") not in rows[0]
                or sha256_file(run / "阶段_观测最佳.pt") not in rows[0]
                or sha256_file(run / "阶段_训练末.pt") not in rows[0]):
            raise ValueError("任09主HF3/seed0录0077训练报告与best/final末态未进ROOT主账")
        return {"qualified": True}

    def qualify_v3(run: Path, size: int, seed: int) -> dict[str, bool]:
        result = task09_subset_formal_v3.require_task09_v3_downstream_qualification(
            purpose="final", name="primary", size=size, seed=seed,
            output_directory=run, v3_registry_path=V3_REGISTRY,
            v3_registry_sha256=V3_SHA, v3_source_archive=V3_ARCHIVE,
            v3_source_archive_sha256=V3_ARCHIVE_SHA,
        )
        return {"qualified": result["V3来源谱系与会话链合格"] is True}

    return qualify_v2, qualify_v3


def _task09_locked_history_files() -> None:
    for path, expected in ((ORIGINAL_REGISTRY, ORIGINAL_SHA),
                           (OLD_AGG_REGISTRY, OLD_AGG_REGISTRY_SHA),
                           (OLD_AGG_ARCHIVE, OLD_AGG_ARCHIVE_SHA),
                           (OLD_AGG_REPORT, OLD_AGG_REPORT_SHA),
                           (V2_REGISTRY, V2_SHA), (V2_ARCHIVE, V2_ARCHIVE_SHA),
                           (V3_REGISTRY, V3_SHA), (V3_ARCHIVE, V3_ARCHIVE_SHA),
                           (OLD_12HF_REFERENCE, OLD_12HF_SHA)):
        _sha(path, expected, "历史冻结来源/合法观测参照")


def prepare_task09_main_evidence_from_prereg(
    *, registry_path: str | Path, registry_sha: str,
    archive_path: str | Path, archive_sha: str,
) -> dict[str, Any]:
    registry = Path(registry_path).resolve()
    archive = Path(archive_path).resolve()
    if (registry != NEW_AGG_REGISTRY or archive.parent != TASK09_ROOT
            or archive.is_symlink() or not archive.name.startswith(
                "任09F3主序列正式聚合五源v2事前冻结_")
            or not archive.name.endswith(".tar.gz")):
        raise ValueError("任09证据清单准备须固定事前YAML/五源tar")
    validate_task09_aggregate_archive(registry, registry_sha, archive, archive_sha)
    require_task09_aggregate_prereg_root_ledger(registry_sha, archive, archive_sha)
    loaded = yaml.safe_load(registry.read_text(encoding="utf-8"))
    validate_task09_main_registry(loaded)
    _task09_locked_history_files()
    catalog = build_task09_main_evidence_catalog(loaded, project_root=PROJECT_ROOT)
    qualify_v2, qualify_v3 = _task09_real_qualifiers()
    from sic_cu.eval import task09_energy
    verify_task09_main_evidence(
        loaded, catalog, project_root=PROJECT_ROOT,
        qualify_v2=qualify_v2, qualify_v3=qualify_v3,
        locked_12hf_scores=task09_energy.task09_locked_task07_observed_S(),
    )
    return catalog


def aggregate_task09_main_from_signed_catalog(
    *, registry_path: str | Path, registry_sha: str,
    archive_path: str | Path, archive_sha: str,
    evidence_path: str | Path, evidence_sha: str,
) -> dict[str, Any]:
    registry = Path(registry_path).resolve()
    archive = Path(archive_path).resolve()
    catalog = Path(evidence_path).resolve()
    if (registry != NEW_AGG_REGISTRY or catalog != CATALOG_PATH
            or archive.parent != TASK09_ROOT or archive.is_symlink()
            or not archive.name.startswith("任09F3主序列正式聚合五源v2事前冻结_")
            or not archive.name.endswith(".tar.gz")):
        raise ValueError("任09聚合只能按事前唯一YAML/项目内tar与既定全身份证据清单路径")
    validate_task09_aggregate_archive(registry, registry_sha, archive, archive_sha)
    _sha(catalog, evidence_sha, "十五身份事后原件清单")
    require_task09_aggregate_root_ledger(registry_sha, archive, archive_sha, evidence_sha)
    prereg = yaml.safe_load(registry.read_text(encoding="utf-8"))
    evidence = yaml.safe_load(catalog.read_text(encoding="utf-8"))
    validate_task09_main_registry(prereg)
    _task09_locked_history_files()
    from sic_cu.eval import task09_energy
    qualify_v2, qualify_v3 = _task09_real_qualifiers()

    locked_12hf = task09_energy.task09_locked_task07_observed_S()
    if prereg.get("HF12历史合法观测原件相对路径") != OLD_12HF_REFERENCE.relative_to(PROJECT_ROOT).as_posix():
        raise ValueError("任09只接受任07乙旧十二HF唯一合法观测原件")
    return verify_task09_main_evidence(
        prereg, evidence, project_root=PROJECT_ROOT,
        qualify_v2=qualify_v2, qualify_v3=qualify_v3,
        locked_12hf_scores=locked_12hf,
    )
