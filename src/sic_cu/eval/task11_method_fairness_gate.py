"""Read-only existing-method origin and budget contract, not a model comparison."""

from __future__ import annotations

import math
import re
import json
import hashlib
import tarfile
from pathlib import Path
from typing import Any, Mapping

import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.data.splits import build_power_splits
from sic_cu.eval.task13_rz_fair_timing import preflight_timing


ARMS = ("B0", "MLP", "E0", "F2")
LF_METHODS = ("deeponet_pinn", "mlp_pinn")
SINGLE_CORRECTION = "现有LF加单个MLP校正器；不等于Howard三子网"
SPLITS_SHA256 = "89a1ebd7df4512ea1ffc1704799cc120182139e12d08e35d1dcb800ae9cc296e"
TRAINING_SHA256 = "4c109435ec253189eaf84dd966c92c88db4235704ce5c93ce4a66682330eda41"
E0_REGISTRY = PROJECT_ROOT / "研究记录/任务07_正式五种子重训/有效运行配置_实现修复后.yaml"
F2_REGISTRY = PROJECT_ROOT / "研究记录/任务08_贡献消融/F2_正式五种子有效运行配置_事务修订后.yaml"


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _powers(value: Any) -> set[float]:
    try:
        return {round(float(power), 4) for power in value}
    except (ValueError, TypeError) as error:
        raise ValueError("HF/LF合法功率来源不是可核数字数组") from error


def summarize_existing_contract(
    hf_rows: list[Mapping[str, Any]], lf_rows: list[Mapping[str, Any]],
    hf_checkpoints: Mapping[tuple[str, int], Mapping[str, Any]],
    lf_checkpoints: Mapping[tuple[str, int], Mapping[str, Any]],
    hf_receipts: Mapping[tuple[str, int], Mapping[str, Any]],
    lf_receipts: Mapping[tuple[str, int], Mapping[str, Any]],
    *, splits: Any, training: Mapping[str, Any],
    architecture_claim: str = SINGLE_CORRECTION,
    rank_training_seconds: bool = False,
) -> dict[str, Any]:
    """Validate already frozen identities without labels, scores, ranking, or training."""
    if architecture_claim != SINGLE_CORRECTION:
        raise ValueError("现有单校正器不能冒称Howard原文线性加非线性三子网")
    if rank_training_seconds:
        raise ValueError("旧完整1700轮与新最近断点会话墙钟异口径，不得直接排名耗时")
    if (training.get("seeds") != list(range(5))
            or training.get("selection_metric_version") != "macro_v1"
            or training.get("epochs", {}).get("high_fidelity") != 1500
            or training.get("epochs", {}).get("joint") != 500
            or training.get("optimizer", {}).get("name") != "adamw"
            or training.get("loss_weights", {}).get("pde") != 1.0
            or any(training.get("loss_weights", {}).get(name) != 1.0
                   for name in ("boundary", "initial", "interface"))
            or len(splits.hf_train) != 12 or len(splits.hf_validation) != 3
            or len(splits.simulation_train) != 60
            or len(splits.simulation_validation) != 10):
        raise ValueError("任11协议/训练预算与原12/3HF、60/10LF物理合同不一致")

    expected_lf = {(method, seed) for method in LF_METHODS for seed in range(5)}
    lf_by_identity = {(row.get("lf_method"), row.get("seed")): row for row in lf_rows}
    if (len(lf_rows) != len(expected_lf) or set(lf_by_identity) != expected_lf
            or set(lf_checkpoints) != expected_lf or set(lf_receipts) != expected_lf):
        raise ValueError("任11两个真实LF方法各五seed来源身份缺失或重复")
    lf_sources = []
    for method, seed in sorted(expected_lf):
        origin = lf_by_identity[method, seed]
        checkpoint = lf_checkpoints[method, seed]
        receipt = lf_receipts[method, seed]
        if (not _sha(origin.get("checkpoint_sha256"))
                or not _sha(origin.get("receipt_sha256"))
                or checkpoint.get("method") != method or checkpoint.get("seed") != seed
                or checkpoint.get("provenance", {}).get("test_labels_consumed") is not False
                or _powers(checkpoint.get("train_powers_w", ())) != splits.simulation_train
                or _powers(checkpoint.get("validation_powers_w", ())) != splits.simulation_validation
                or receipt.get("method") != method or receipt.get("seed") != seed):
            raise ValueError("任11 LF来源方法/seed/训练验证功率或旧TEST身份不合法")
        epochs = receipt.get("epochs_completed")
        seconds = receipt.get("training_seconds")
        if (type(epochs) is not int or epochs <= 0 or epochs > training["epochs"]["simulation"]
                or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("任11 LF预训练真实正轮次与墙钟收据缺失")
        lf_sources.append({"LF方法": method, "seed": seed,
                           "检查点SHA256": origin["checkpoint_sha256"],
                           "预训练收据SHA256": origin["receipt_sha256"],
                           "LF预训实际轮次": epochs,
                           "LF预训练单独墙钟秒": float(seconds),
                           "原FEM仿真生成成本": "无可核完整墙钟原件"})

    expected_hf = {(arm, seed) for arm in ARMS for seed in range(5)}
    hf_by_identity = {(row.get("arm"), row.get("seed")): row for row in hf_rows}
    if (len(hf_rows) != len(expected_hf) or set(hf_by_identity) != expected_hf
            or set(hf_checkpoints) != expected_hf or set(hf_receipts) != expected_hf):
        raise ValueError("任11历史B0/MLP和新E0/F2四臂五seed来源身份缺失或重复")
    hf_sources = []
    for arm, seed in sorted(expected_hf):
        row = hf_by_identity[arm, seed]
        checkpoint = hf_checkpoints[arm, seed]
        receipt = hf_receipts[arm, seed]
        lf_method = "mlp_pinn" if arm == "MLP" else "deeponet_pinn"
        lf_sha = lf_by_identity[lf_method, seed]["checkpoint_sha256"]
        origin_sha = (checkpoint.get("lf_checkpoint_sha256")
                      or checkpoint.get("provenance", {}).get("lf_checkpoint_sha256"))
        model_state = checkpoint.get("model_state", {})
        first = model_state.get("correction.0.weight")
        if (not _sha(row.get("checkpoint_sha256")) or not _sha(row.get("receipt_sha256"))
                or row.get("lf_method") != lf_method
                or row.get("lf_checkpoint_sha256") != lf_sha
                or origin_sha != lf_sha
                or checkpoint.get("low_fidelity_method") != lf_method
                or checkpoint.get("method") != "multifidelity_correction"
                or checkpoint.get("seed") != seed
                or checkpoint.get("provenance", {}).get("test_labels_consumed") is not False
                or _powers(checkpoint.get("hf_train_powers_w", ())) != splits.hf_train
                or _powers(checkpoint.get("hf_validation_powers_w", ())) != splits.hf_validation
                or not isinstance(first, torch.Tensor)
                or tuple(first.shape) != (128, 6)
                or checkpoint.get("correction_model_kwargs", {}).get("width") != 128
                or checkpoint.get("correction_model_kwargs", {}).get("depth") != 4):
            raise ValueError("任11 HF来源或内嵌LF身份/旧六列单校正器错误，拒绝把MLP冒名为任07的DeepONet LF")
        if arm in ("B0", "MLP"):
            epochs = receipt.get("epochs_completed")
            seconds = receipt.get("training_seconds")
            best_epoch = receipt.get("best_epoch")
            cohort = "旧V4完整1700轮单次会话"
            if (receipt.get("seed") != seed or epochs != 1700
                    or type(best_epoch) is not int or not 0 < best_epoch <= epochs):
                raise ValueError("任11旧V4真HF完整1700轮及同seed最佳原件缺失")
        else:
            correction = receipt.get("校正实际轮次")
            joint = receipt.get("联合实际轮次")
            seconds = receipt.get("本会话耗时秒")
            epochs = (correction + joint if type(correction) is int
                      and type(joint) is int else None)
            cohort = "新E0/F2最近断点会话；非累计总耗时"
            if (receipt.get("运行种子") != seed
                    or receipt.get("运行臂") != arm
                    or receipt.get("旧test_Data温度标签读取") is not False
                    or type(correction) is not int or correction <= 0
                    or correction > training["epochs"]["high_fidelity"]
                    or type(joint) is not int or joint < 0
                    or joint > training["epochs"]["joint"]):
                raise ValueError("任11新E0/F2真HF阶段轮次、seed、旧TEST状态或上限漂移")
        if (not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("任11 HF训练墙钟必须为真实正秒且严格按原收据口径分栏")
        hf_sources.append({"方法": arm, "seed": seed,
                           "检查点SHA256": row["checkpoint_sha256"],
                           "训练收据SHA256": row["receipt_sha256"],
                           "LF方法": lf_method, "LF检查点SHA256": lf_sha,
                           "HF实际训练轮次": epochs,
                           "HF训练成本口径": cohort,
                           "原口径墙钟秒": float(seconds),
                           "HF校正架构": "LF温度加单个MLP校正器",
                           "HF体内PDE训练": arm != "F2",
                           "边界初值界面训练约束保留": True,
                           "Howard原文关系": "不是原文线性和非线性两HF子网"})

    return {
        "证据性质": "只读现有四臂二LF身份和预算；不训练任11新方法，不读取旧TEST温度",
        "协议": {
            "HF训练功率_W": sorted(splits.hf_train),
            "HF合法验证功率_W": sorted(splits.hf_validation),
            "LF训练功率_W": sorted(splits.simulation_train),
            "LF合法验证功率_W": sorted(splits.simulation_validation),
            "训练配置": "原AdamW/macro_v1；名义体内PDE、边界、初值、界面固定权重",
            "HF校正上限轮次": training["epochs"]["high_fidelity"],
            "受限联合上限轮次": training["epochs"]["joint"],
        },
        "已有HF来源身份": hf_sources,
        "已有LF来源身份": lf_sources,
        "旧新训练墙钟直接排名许可": False,
        "Howard原文三子网精确复现资格": False,
        "新正式公平MLP与Howard五种子模型数": 0,
        "新正式预登记缺口": [
            "现有任07门禁只认DeepONet LF；新MLP须独立锁LF来源/重新初始化/完整训练预算和源码",
            "Howard作者原代码版本与SHA尚无可核来源；自行三子网适配须独立标名、逐子网结构预算与源码事前封存",
            "现有旧HF完整会话与新HF最近断点会话墙钟不可直接作跨方法总成本排名",
            "原LF FEM场生成完整墙钟、原装置内HF实测和新的独立盲测没有原件",
        ],
    }


def audit_registered_existing_sources(
    *, registry: str | Path, registry_sha: str,
    source_tar: str | Path, source_tar_sha: str,
    ledger: str | Path, output: str | Path,
) -> dict[str, Any]:
    """Only open already registered CPU metadata; do not create output or train."""
    if (sha256_file(PROJECT_ROOT / "configs/splits.yaml") != SPLITS_SHA256
            or sha256_file(PROJECT_ROOT / "configs/training.yaml") != TRAINING_SHA256):
        raise ValueError("任11原划分或训练配置字节SHA漂移，禁止读取方法身份")
    cfg = preflight_timing(
        registry, registry_sha, source_tar, source_tar_sha, ledger, output,
        require_cuda=False,
    )
    expected = {"E0": cfg["正式E0预登记配置SHA256"],
                "F2": cfg["正式F2预登记配置SHA256"]}
    if (sha256_file(E0_REGISTRY) != expected["E0"]
            or sha256_file(F2_REGISTRY) != expected["F2"]):
        raise ValueError("任11 E0/F2正式来源与原事前预算SHA已漂移")
    lf_rows = cfg["登记LF来源"]
    hf_rows = [dict(row, lf_method=("mlp_pinn" if row["arm"] == "MLP"
                                    else "deeponet_pinn"))
               for row in cfg["登记模型来源"]]
    hf_checkpoints: dict[tuple[str, int], Mapping[str, Any]] = {}
    lf_checkpoints: dict[tuple[str, int], Mapping[str, Any]] = {}
    hf_receipts: dict[tuple[str, int], Mapping[str, Any]] = {}
    lf_receipts: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in lf_rows + hf_rows:
        checkpoint = Path(row["checkpoint"])
        receipt = Path(row["receipt"])
        if (sha256_file(checkpoint) != row["checkpoint_sha256"]
                or sha256_file(receipt) != row["receipt_sha256"]):
            raise ValueError("任11 LF/HF正式已登记检查点或真实收据SHA读取前漂移")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if (sha256_file(checkpoint) != row["checkpoint_sha256"]
                or sha256_file(receipt) != row["receipt_sha256"]):
            raise ValueError("任11 LF/HF正式已登记检查点或真实收据SHA读取后漂移")
        if "arm" in row:
            key = (row["arm"], row["seed"])
            hf_checkpoints[key], hf_receipts[key] = payload, record
        else:
            key = (row["lf_method"], row["seed"])
            lf_checkpoints[key], lf_receipts[key] = payload, record
    summary = summarize_existing_contract(
        hf_rows, lf_rows, hf_checkpoints, lf_checkpoints,
        hf_receipts, lf_receipts, splits=build_power_splits(),
        training=load_yaml("configs/training.yaml"),
    )
    summary["真来源与配置双锁"] = {
        "任13已有预算YAML_SHA256": registry_sha,
        "任13已有十二关键源码tar_SHA256": source_tar_sha,
        "正式E0预算SHA256": expected["E0"],
        "正式F2预算SHA256": expected["F2"],
        "划分SHA256": SPLITS_SHA256,
        "训练配置SHA256": TRAINING_SHA256,
        "档案范围": "任13十二源码仅为关键快照，不保证整个依赖环境闭包",
    }
    return summary


def _project_path(path: str | Path) -> Path:
    supplied = Path(path)
    resolved = (supplied if supplied.is_absolute() else PROJECT_ROOT / supplied).resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise ValueError("任11登记、源码和报告只能位于项目内")
    return resolved


def require_task11_registration(
    registry: str | Path, registry_sha: str,
    source_tar: str | Path, source_tar_sha: str,
    ledger: str | Path,
) -> dict[str, Any]:
    """Require the Task-11 budget and seven-source hashes on one 0073 ledger row."""
    registry_path = _project_path(registry)
    tar_path = _project_path(source_tar)
    ledger_path = _project_path(ledger)
    if (not registry_path.is_file() or not tar_path.is_file()
            or not ledger_path.is_file()
            or not _sha(registry_sha) or not _sha(source_tar_sha)
            or sha256_file(registry_path) != registry_sha
            or sha256_file(tar_path) != source_tar_sha):
        raise ValueError("任11事前预算YAML或七源tar的登记SHA缺失/漂移")
    cfg = load_yaml(registry_path)
    sources = cfg.get("七源普通原件SHA256")
    if (cfg.get("schema_version") != 1
            or cfg.get("事前台账记录编号") != "录-0073"
            or cfg.get("性质") != "任11方法级公平预算与来源身份CPU预检；不训练新模型"
            or cfg.get("新MLP或Howard训练许可") is not False
            or cfg.get("旧TEST温度读取") is not False
            or not isinstance(sources, dict) or len(sources) != 7
            or any(not isinstance(name, str) or not _sha(sha)
                   for name, sha in sources.items())):
        raise ValueError("任11录0073预算结构、禁止训练/旧TEST状态或七源清单不合法")
    try:
        with tarfile.open(tar_path, "r:gz") as archive:
            members = archive.getmembers()
            if (len(members) != 7 or any(not member.isfile() for member in members)
                    or {member.name for member in members} != set(sources)):
                raise ValueError("任11七源tar必须恰含登记的七个普通文件")
            for member in members:
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("任11七源tar普通成员无法读取")
                archived_sha = hashlib.sha256(stream.read()).hexdigest()
                current = _project_path(member.name)
                if (not current.is_file() or archived_sha != sources[member.name]
                        or sha256_file(current) != sources[member.name]):
                    raise ValueError("任11七源tar、预算声明与现场原件字节SHA不一致")
    except (tarfile.TarError, OSError) as error:
        raise ValueError("任11七源源码归档无效") from error
    rows = [line for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if re.search(r"\|\s*录-0073\s*\|", line)]
    if len(rows) != 1 or registry_sha not in rows[0] or source_tar_sha not in rows[0]:
        raise ValueError("任11录0073总账同一行尚未登记预算YAML和七源tar双SHA")
    return cfg


def write_chinese_contract_report(
    summary: Mapping[str, Any], output: str | Path,
) -> dict[str, dict[str, str]]:
    """Write only the read-only contract result, never model predictions or labels."""
    destination = _project_path(output)
    if destination.exists() or destination == PROJECT_ROOT:
        raise FileExistsError("任11合同报告只能写入全新项目内目录")
    hf = summary.get("已有HF来源身份")
    lf = summary.get("已有LF来源身份")
    gaps = summary.get("新正式预登记缺口")
    if (not isinstance(hf, list) or len(hf) != 20
            or not isinstance(lf, list) or len(lf) != 10
            or not isinstance(gaps, list) or not gaps
            or summary.get("旧新训练墙钟直接排名许可") is not False
            or summary.get("Howard原文三子网精确复现资格") is not False
            or summary.get("新正式公平MLP与Howard五种子模型数") != 0):
        raise ValueError("任11中文限界报告只接受完整20HF/10LF且无新模型结论的预检摘要")
    cohorts = sorted({row["HF训练成本口径"] for row in hf})
    arm_rounds = {
        arm: [row["HF实际训练轮次"] for row in hf if row["方法"] == arm]
        for arm in ARMS
    }
    destination.mkdir(parents=True)
    lines = [
        "# 任11方法级公平预算与来源身份预检限界报告",
        "",
        "本报告只核对已冻结方法的来源身份、合法功率折和原训练收据。它不读取旧固定TEST温度，"
        "不读取一维板隐藏内场，不启动GPU或任11新模型训练，也不形成方法精度名次。",
        "",
        "## 现有来源",
        "",
        f"- 已登记HF检查点及收据：{len(hf)}组（B0、MLP、E0、F2各五seed）。",
        f"- 已登记LF检查点及收据：{len(lf)}组（DeepONet与MLP各五seed）。",
        "- 四臂现有HF均为LF温度加单个MLP校正器；Howard原文三子网精确复现资格：否。",
        "- F2仅跳过HF体内PDE训练；边界、初值和界面训练约束仍保留。",
        "",
        "## 预算口径",
        "",
        f"- 旧B0/MLP实际轮次：{arm_rounds['B0']} / {arm_rounds['MLP']}。",
        f"- 新E0/F2实际轮次：{arm_rounds['E0']} / {arm_rounds['F2']}。",
        f"- 原收据成本口径：{'；'.join(cohorts)}。",
        "- 旧新训练墙钟不能直接排名：旧方法是完整1700轮单次会话，新方法仅有最近断点会话秒数。",
        "- LF预训练成本单独列示；原LF FEM场生成完整墙钟没有可核原件。",
        "",
        "## 后续缺口",
        "",
    ]
    lines.extend(f"- {gap}" for gap in gaps)
    lines.extend([
        "",
        "在主总账录0073登记本步骤预算YAML与七源tar双SHA之前，以及另行冻结新方法自身结构、"
        "随机初始化、五seed预算和来源之前，不许可新MLP或Howard适配训练。",
        "",
    ])
    report_path = destination / "任11方法公平预算与来源限界报告.md"
    machine_path = destination / "任11方法公平预算与来源机器摘要.json"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    machine_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n", encoding="utf-8")
    return {
        "中文报告": {"路径": str(report_path.relative_to(PROJECT_ROOT)),
                     "SHA256": sha256_file(report_path)},
        "机器摘要": {"路径": str(machine_path.relative_to(PROJECT_ROOT)),
                     "SHA256": sha256_file(machine_path)},
    }
