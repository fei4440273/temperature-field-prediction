"""Task-09 P1 repair without changing the frozen original formal trainer.

The failed CUDA process appended one genuine joint epoch to ``training.jsonl``
before the original AdamW audit rejected every frozen LF parameter as though it
were one of the four trainable projections.  This module binds that exact
failure, stages only the last committed corr500/joint0 state in a new directory,
and scopes the corrected audit to the existing training loop.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tarfile
from typing import Any, Callable, Mapping, Sequence, TypeVar

import torch
from torch import Tensor
import yaml

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.train import task09_subset_formal as _v1
from sic_cu.train.task04_joint import PROJECTION_NAMES
from sic_cu.train.task07_source import Task07Source, _lf_tensor_sha256


FAILED_RUN_DIRECTORY = PROJECT_ROOT / (
    "研究记录/任务09_高保真数据效率/正式F3子集训练/"
    "正式F3_primary_HF3_seed0_20260916T060920+0800"
)
FAILED_RUN_FILES_SHA256 = {
    "best.pt": "4c361df61ac3c622b8f79ea1e72ff8c8e725573e7d03137c1b2af4e5536f90d4",
    "training.jsonl": "e87f207ce3df3ca4fa5870fa75c80e0940337d13f5d225c59df62f3fea17222d",
    "任09F3阶段报告.json": (
        "17217eb79917c40a8040914c2fb581d212ee414ca3b17c8d26e5da71864961f3"
    ),
    "正式主HF3_seed0第200校正轮独立收据_20260916T061402+0800.md": (
        "4792335733c61f060550bc8bfa9801ea4f3f732452014c4d8450491c3d78cb7f"
    ),
    "正式主HF3_seed0第400校正轮独立收据_20260916T061637+0800.md": (
        "7fd02a4e4046d0a4295540dc79cda7f4af35e0a1f8bea23fc8f94da12f6422f0"
    ),
    "源码与登记事前快照SHA256.json": (
        "bc907bce480db1a910021d29c61cbc54fbde2b93b480b077c7cf49b4d203e000"
    ),
    "阶段_初始.pt": "fd1e59f18c204e82f8e97dfdb20f97d1cfe2ba58b38de602884a18a9494435ae",
    "阶段_最近.pt": "4b7d110794046b242d4275cb795d7befcf7244bad42faf1104c883c982c33ac6",
    "阶段_校正末.pt": "391f21c7b174bd5d358425b88a32b3b444b41774a78fff304d4c1049233419f8",
    "阶段_物理最佳.pt": "fd1e59f18c204e82f8e97dfdb20f97d1cfe2ba58b38de602884a18a9494435ae",
    "阶段_联合初始.pt": "4b7d110794046b242d4275cb795d7befcf7244bad42faf1104c883c982c33ac6",
    "阶段_观测最佳.pt": "1089889f4e96d2e428e22bbc20e8eb27002f1570c632d2aaee9e9919c747431b",
}
FAILED_LOG_PREFIX_SHA256 = (
    "b9477403ddc73d599153ddfca563e4cca575cf4fc3e6b7d07fcd1f8e37f29145"
)
FAILED_LOG_TAIL_SHA256 = (
    "502f82c00b3750fc9041224977587095f91169936dbd8243e6f665f7c0b78cbf"
)
FAILED_SOURCE_LF_TENSOR_SHA256 = (
    "c01575e2b845c25097f14d5d2cddb0a0f2763d02c9d467af0d25cbc6790ef5f8"
)
ORIGINAL_FORMAL_REGISTRY_SHA256 = (
    "987ba4bfd9a5b964bff88fa3314413738fa5417b5445e71cf650acbaee7d85e9"
)
RECOVERY_MANIFEST = "任09V2恢复来源清单.json"
RECOVERY_BASELINE = "恢复基线_已提交corr500_joint0.pt"
RECOVERY_TAIL = "恢复源_未提交joint1尾行_仅证据.jsonl"
RECOVERY_FULL_FAILED_LOG = "恢复源_完整501行失败日志_只读证据.jsonl"
RECOVERY_STALE_REPORT = "恢复源_陈旧epoch400阶段报告_只读证据.json"
TASK09_V2_TRAINER = PROJECT_ROOT / "src/sic_cu/train/task09_subset_formal_v2.py"
TASK09_V2_CLI = PROJECT_ROOT / "scripts/42_run_task09_subset_formal_v2.py"
TASK09_V2_REGISTRATION_CLI = PROJECT_ROOT / "scripts/43_register_task09_subset_formal_v2.py"
TASK09_V2_TEST = PROJECT_ROOT / "tests/test_task09_subset_formal_v2.py"
TASK09_V2_REGISTRY = PROJECT_ROOT / (
    "研究记录/任务09_高保真数据效率/"
    "正式F3联合首轮P1修订恢复前登记_录0075强门禁版.yaml"
)
FAILED_V2_REGISTRY = PROJECT_ROOT / (
    "研究记录/任务09_高保真数据效率/正式F3联合首轮P1修订恢复前登记.yaml"
)
FAILED_V2_REGISTRY_SHA256 = (
    "74352e76ef33c8ed347948a30871741adef9cf189370ac82a51a9560e12cc0d5"
)
TASK09_LEDGER = PROJECT_ROOT / "多保真DeepONet预测精度优化总计划与执行台账.md"
TASK09_V2_ARCHIVE_ROOT = PROJECT_ROOT / "研究记录/任务09_高保真数据效率"
ORIGINAL_SOURCE_TAR = PROJECT_ROOT / (
    "研究记录/任务09_高保真数据效率/正式F3来源与源码封存_20260916T041400+0800/"
    "任09F3正式来源与训练独审源码_完整校正_20260916T041500+0800.tar.gz"
)
ORIGINAL_LOCKS = {
    "正式HF数据效率F3先导前登记.yaml": ORIGINAL_FORMAL_REGISTRY_SHA256,
    "任09F3正式来源与训练独审源码_完整校正_20260916T041500+0800.tar.gz": (
        "0091e55a2f57077f70c1008344aff9d15c73d2e78b0070327b93f8bb172698ef"
    ),
    "task09_subset_formal.py": (
        "3e304eb4b2fd6c351375b7e4176ae344057b809c9c8394d62bc5c110cd0cf375"
    ),
    "35_run_task09_subset_formal.py": (
        "e7768f7414e5908483a37249406bbc8a98f31aee69b0b1e783e8464eeb9b8132"
    ),
}
_T = TypeVar("_T")


def _state_step(state: Mapping[str, Any], name: str) -> int:
    if not state:
        return 0
    step = state.get("step")
    if step is None or not math.isfinite(float(step)) or not float(step).is_integer():
        raise ValueError(f"任09V2{name}缺真实有限整数AdamW step")
    return int(step)


def assert_task09_v2_adamw_steps(
    optimizer: torch.optim.AdamW,
    named_parameters: Mapping[str, Tensor],
    *,
    hf_batches: int,
    correction_epochs: int,
    joint_epochs: int,
) -> dict[str, int]:
    """Require steps on correction/four projections and zero state on other LF.

    This is the sole behavioral correction from the frozen v1 implementation.
    Buffers are checked through the complete LF tensor SHA in the resume gate;
    buffers can never be members of an optimizer state.
    """
    if (not isinstance(hf_batches, int) or hf_batches not in range(3, 13)
            or not isinstance(correction_epochs, int) or correction_epochs < 0
            or not isinstance(joint_epochs, int) or joint_epochs < 0):
        raise ValueError("任09V2 AdamW动态HF批数与校正/联合轮次无效")
    correction_expected = (hf_batches + 1) * (correction_epochs + joint_epochs)
    projection_expected = (hf_batches + 1) * joint_epochs
    names_by_id = {id(parameter): name for name, parameter in named_parameters.items()}
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    grouped_ids = {id(parameter) for parameter in grouped}
    if len(grouped) != len(grouped_ids):
        raise ValueError("任09V2 AdamW参数组含重复张量")
    unknown = [parameter for parameter in grouped if id(parameter) not in names_by_id]
    if unknown or any(id(parameter) not in grouped_ids for parameter in optimizer.state):
        raise ValueError("任09V2 AdamW参数组或状态混入模型外张量")

    checked: dict[str, int] = {}
    corrections: set[str] = set()
    projections: set[str] = set()
    grouped_names = {names_by_id[id(parameter)] for parameter in grouped}
    for name, parameter in named_parameters.items():
        state = optimizer.state.get(parameter, {})
        actual = _state_step(state, name)
        if name.startswith("correction."):
            corrections.add(name)
            expected = correction_expected
            if name not in grouped_names:
                raise ValueError(f"任09V2新HF校正参数{name}缺AdamW参数组")
        elif name in PROJECTION_NAMES:
            projections.add(name)
            expected = projection_expected
            if projection_expected and name not in grouped_names:
                raise ValueError(f"任09V2四末投影{name}缺联合AdamW参数组")
        elif name.startswith("low_fidelity_model."):
            expected = 0
            if state or name in grouped_names:
                raise ValueError(
                    f"任09V2冻结LF参数{name}不得有AdamW状态或参数组，实际step={actual}"
                )
        else:
            raise ValueError("任09V2 AdamW不准优化来源之外的其它HF/LF参数")
        if actual != expected:
            kind = "四末投影" if name in PROJECTION_NAMES else "新HF校正"
            raise ValueError(
                f"任09V2{kind}{name}真实AdamW step={actual}，期望{expected}"
            )
        checked[name] = actual
    if len(corrections) != 10 or projections != set(PROJECTION_NAMES):
        raise ValueError("任09V2须完整10张量新HF校正器及恰四个LF末投影")
    allowed_grouped = corrections | projections
    if not grouped_names <= allowed_grouped:
        raise ValueError("任09V2 AdamW参数组含冻结LF或其它参数")
    return checked


def validate_task09_uncommitted_tail(
    trailing_rows: Sequence[Mapping[str, Any]],
    *,
    committed_metadata: Mapping[str, Any],
    sample: Mapping[str, Any],
    seed: int,
    powers: Sequence[float],
) -> Mapping[str, Any]:
    """Accept exactly one genuine joint1 receipt after committed joint0."""
    if len(trailing_rows) != 1:
        raise ValueError("任09V2恢复只接受唯一一条未提交尾行")
    row = trailing_rows[0]
    correction = committed_metadata.get("校正实际轮次")
    joint = committed_metadata.get("联合实际轮次")
    global_epoch = committed_metadata.get("完整全局实际轮次")
    if (committed_metadata.get("当前阶段") != _v1.JOINT_STAGE
            or correction != 500 or joint != 0 or global_epoch != 500
            or committed_metadata.get("校正实际截止轮次") != 500
            or committed_metadata.get("累计实际消耗") !=
            _v1.task09_expected_consumption(sample, 500, 0)):
        raise ValueError("任09V2恢复源必须是已提交corr500/joint0完整状态")
    identity = (
        row.get("epoch") == 501
        and row.get("运行种子") == seed
        and row.get("运行臂") == "F3"
        and row.get("真实HF子集功率") == list(powers)
        and row.get("训练阶段") == _v1.JOINT_STAGE
        and row.get("阶段实际轮次") == 1
    )
    if not identity:
        raise ValueError("任09V2未提交尾行须为同seed同子集的唯一joint1阶段")
    _v1.assert_task09_epoch_budget(row, sample, joint=True)
    if (row.get("累计实际消耗") != _v1.task09_expected_consumption(sample, 500, 1)
            or row.get("每LF2048小批损失名义系数") != 0.25
            or row.get("LF_Cu真实回放点") != 61500
            or row.get("LF_SiC真实回放点") != 61380):
        raise ValueError("任09V2未提交尾行LF60、双材料、1/4权重或累计消费不真")
    current_lf = row.get("当前本seed LF整张量SHA256")
    if (not isinstance(current_lf, str)
            or re.fullmatch(r"[0-9a-f]{64}", current_lf) is None
            or current_lf == FAILED_SOURCE_LF_TENSOR_SHA256):
        raise ValueError("任09V2未提交joint1尾行缺真实已动四投影LF张量SHA")
    if any(row.get(field) is not None for field in (
        "HF合法验证选分_摄氏度", "HF合法验证分模态_摄氏度",
        "LF合法验证逐材料节点与体积", "LF两材料真实节点与体积5%资格",
        "独立名义物理损失",
    )):
        raise ValueError("任09V2 joint1非十轮验证点不得伪造选分、护栏或独立物理结果")
    return row


def _task09_sample() -> dict[str, Any]:
    registered = _v1.load_task09_registry()
    sample = {
        **_v1.audit_task09_observations(registered)["arms"]["primary"][3],
        "arm": "primary",
    }
    if list(registered["primary"][3]) != [55.0, 364.3, 729.0]:
        raise ValueError("任09V2失败源所绑定primary/HF3功率原登记已改变")
    return sample


def _load_v2_live_state(
    payload: Mapping[str, Any], source: Task07Source,
) -> tuple[torch.nn.Module, torch.optim.AdamW]:
    sources = _v1.load_task09_sources()
    initial = _v1.fork_task07_initialization(sources, source.seed, "E0", torch.device("cpu"))
    model, optimizer = initial.model, initial.optimizer
    if payload["stage"] == _v1.JOINT_STAGE:
        _v1.switch_task09_restricted_joint(
            model, optimizer, source, loading_committed_state=True,
        )
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    return model, optimizer


def check_task09_v2_resume_payload(
    payload: Mapping[str, Any], source: Task07Source,
    sample: Mapping[str, Any], *, registry_sha256: str,
) -> None:
    """Retain v1 lineage checks, then enforce corrected name-to-step semantics."""
    _v1.check_task09_resume_payload(
        payload, source, sample, registry_sha256=registry_sha256,
    )
    model, optimizer = _load_v2_live_state(payload, source)
    _v1.check_task09_model(model, source, payload["stage"])
    meta = payload["metadata"]
    assert_task09_v2_adamw_steps(
        optimizer, dict(model.named_parameters()),
        hf_batches=int(sample["ir_batches_2048"]),
        correction_epochs=int(meta["校正实际轮次"]),
        joint_epochs=int(meta["联合实际轮次"]),
    )
    live_lf = model.low_fidelity_model.state_dict()
    changed = {
        name for name, original in source.lf_state.items()
        if not torch.equal(original, live_lf[name].detach().cpu())
    }
    allowed = {name.removeprefix("low_fidelity_model.") for name in PROJECTION_NAMES}
    if changed - allowed or (not meta["联合实际轮次"] and changed):
        raise ValueError("任09V2冻结LF参数/buffer改变；仅已提交joint四末投影可变")
    if _lf_tensor_sha256(live_lf) != meta["当前LF整张量SHA256"]:
        raise ValueError("任09V2当前LF整张量SHA与完整checkpoint不一致")


def parse_task09_recovery_log(
    raw: bytes, *, committed_rows: int = 500,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bytes, bytes]:
    """Parse an exact committed prefix plus one complete uncommitted JSON line."""
    if not raw or not isinstance(committed_rows, int) or committed_rows < 1:
        raise ValueError("任09V2失败恢复日志为空或已提交行数无效")
    lines = raw.splitlines(keepends=True)
    if (len(lines) != committed_rows + 1
            or any(not line.endswith(b"\n") or not line.strip() for line in lines)):
        raise ValueError("任09V2失败日志须为已提交前缀加唯一一条完整非空尾行")
    parsed: list[dict[str, Any]] = []
    try:
        for line in lines:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("JSON行不是对象")
            parsed.append(value)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError("任09V2失败日志含坏JSON、空行或非对象尾行") from error
    prefix_bytes, tail_bytes = b"".join(lines[:committed_rows]), lines[committed_rows]
    return parsed[:committed_rows], parsed[committed_rows:], prefix_bytes, tail_bytes


def _split_failed_log() -> tuple[list[dict[str, Any]], list[dict[str, Any]], bytes, bytes]:
    raw = (FAILED_RUN_DIRECTORY / "training.jsonl").read_bytes()
    prefix, tail, prefix_bytes, tail_bytes = parse_task09_recovery_log(raw)
    if (hashlib.sha256(prefix_bytes).hexdigest() != FAILED_LOG_PREFIX_SHA256
            or hashlib.sha256(tail_bytes).hexdigest() != FAILED_LOG_TAIL_SHA256):
        raise ValueError("任09V2原失败日志前500已提交行或唯一尾行字节SHA改变")
    return prefix, tail, prefix_bytes, tail_bytes


def _tree_files_sha256(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise ValueError(f"任09V2恢复源目录缺失：{root.name}")
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"任09V2恢复源目录不接受符号链接：{path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    if not files:
        raise ValueError(f"任09V2恢复源目录为空：{root.name}")
    return files


def _current_original_locks() -> dict[str, str]:
    return {
        "正式HF数据效率F3先导前登记.yaml": sha256_file(_v1.TASK09_REGISTRY),
        ORIGINAL_SOURCE_TAR.name: sha256_file(ORIGINAL_SOURCE_TAR),
        "task09_subset_formal.py": sha256_file(_v1.TASK09_FORMAL_TRAINER),
        "35_run_task09_subset_formal.py": sha256_file(_v1.TASK09_FORMAL_CLI),
    }


def _v2_source_paths() -> dict[str, Path]:
    return {
        "trainer_v2": TASK09_V2_TRAINER,
        "CLI_v2": TASK09_V2_CLI,
        "registration_generator_v2": TASK09_V2_REGISTRATION_CLI,
        "tests_v2": TASK09_V2_TEST,
    }


def expected_task09_v2_registration() -> dict[str, Any]:
    """Build the exact CPU/source/budget contract that ROOT will record as 0075."""
    current_locks = _current_original_locks()
    if current_locks != ORIGINAL_LOCKS:
        raise ValueError(f"任09V2旧0059四源锁改变，禁止制作修订登记：{current_locks}")
    v2_paths = _v2_source_paths()
    if any(not path.is_file() for path in v2_paths.values()):
        raise FileNotFoundError("任09V2 trainer/CLI/登记生成器/TDD源码必须全部存在")
    if (not FAILED_V2_REGISTRY.is_file()
            or sha256_file(FAILED_V2_REGISTRY) != FAILED_V2_REGISTRY_SHA256):
        raise ValueError("任09V2首次字符串门禁不足YAML须原字节保留为失败证据")
    return {
        "schema_version": 1,
        "任务": "任09 P1冻结LF AdamW提交门禁修订恢复",
        "ROOT主台账预留条目": "录0075",
        "ROOT主台账录0075前GPU禁止": True,
        "正式run实际读取ROOT主台账录0075同行双SHA": True,
        "正式run仅传录0075字符串可放行": False,
        "正式V2源码tar由CLI显式路径与SHA传入并逐成员核验": True,
        "首次字符串门禁不足YAML_SHA256": FAILED_V2_REGISTRY_SHA256,
        "首次字符串门禁不足YAML禁止使用": True,
        "旧0059正式登记SHA256": ORIGINAL_FORMAL_REGISTRY_SHA256,
        "旧0059源码tar_SHA256": ORIGINAL_LOCKS[ORIGINAL_SOURCE_TAR.name],
        "旧0059冻结四源SHA256": current_locks,
        "失败现场路径": str(FAILED_RUN_DIRECTORY.resolve()),
        "失败现场顶层原件SHA256": dict(FAILED_RUN_FILES_SHA256),
        "失败现场完整日志SHA256": FAILED_RUN_FILES_SHA256["training.jsonl"],
        "失败现场已提交前500行SHA256": FAILED_LOG_PREFIX_SHA256,
        "失败现场唯一未提交joint1尾行SHA256": FAILED_LOG_TAIL_SHA256,
        "失败现场已提交校正_联合_全局轮次": [500, 0, 500],
        "失败尾行只作证据不提交不复用": True,
        "失败joint1模型_优化器_RNG复用": False,
        "失败目录source_snapshot逐文件SHA256": _tree_files_sha256(
            FAILED_RUN_DIRECTORY / "source_snapshot"
        ),
        "失败目录config_snapshot逐文件SHA256": _tree_files_sha256(
            FAILED_RUN_DIRECTORY / "config_snapshot"
        ),
        "P1根因": (
            "旧校验把全部low_fidelity_model参数误要求joint step；"
            "实际仅PROJECTION_NAMES四末投影可有联合AdamW状态"
        ),
        "修订唯一行为": (
            "10个新HF校正参数step=(N+1)*(corr+joint)；四末投影step=(N+1)*joint；"
            "其余LF参数无param_group且无state，buffer由整张量SHA锁定"
        ),
        "恢复后首个joint1预期AdamW_step": {
            "10个新HF校正参数": 2505,
            "4个LF末投影参数": 5,
            "其余LF参数": 0,
        },
        "恢复后首个joint1预期日志总行": 501,
        "恢复新目录规则": (
            "正式F3_primary_HF3_seed0_YYYYMMDDTHHMMSS+0800；必须不存在且只复制已提交corr500"
        ),
        "完整501失败日志另存只读证据": True,
        "活动训练日志初始仅前500已提交行": True,
        "陈旧epoch400阶段报告仅证据不可恢复": True,
        "联合每轮LF真实回放batch": 60,
        "联合每个LF小批点": 2048,
        "联合每LF2048小批损失系数": 0.25,
        "primary_HF3每轮HF真实batch": 4,
        "每轮独立物理配点_优化步": [256, 1],
        "校正_联合预算上限不变": [1500, 500],
        "新恢复只重算未提交joint1_不增加预算": True,
        "新修订源码SHA256": {
            name: sha256_file(path) for name, path in v2_paths.items()
        },
        "旧trainer运行循环复用但旧文件不修改": True,
        "旧固定TEST温度读取": False,
        "CPU_TDD或恢复准备可作正式成绩": False,
        "当前正式GPU新增已完成轮次": 0,
    }


def require_task09_v2_registration(
    path: str | Path | None, sha256: str | None,
) -> str:
    if path is None or sha256 is None or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError("任09V2须显式给出修订事前YAML及实际SHA256")
    candidate = Path(path).resolve()
    if candidate != TASK09_V2_REGISTRY.resolve() or not candidate.is_file():
        raise ValueError("任09V2只接受项目内专属修订事前YAML原件")
    if sha256_file(candidate) != sha256:
        raise ValueError("任09V2修订事前YAML命令行SHA与原件不一致")
    loaded = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if loaded != expected_task09_v2_registration():
        raise ValueError("任09V2修订登记与失败源、预算或当前源码SHA不一致")
    return sha256


def expected_task09_v2_archive_members() -> dict[str, str]:
    paths = [
        TASK09_V2_TRAINER, TASK09_V2_CLI, TASK09_V2_REGISTRATION_CLI,
        TASK09_V2_TEST, TASK09_V2_REGISTRY, FAILED_V2_REGISTRY,
        _v1.TASK09_FORMAL_TRAINER, _v1.TASK09_FORMAL_CLI, _v1.TASK09_REGISTRY,
    ]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("任09V2源码tar九份普通原件尚未齐全")
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path) for path in paths
    }


def validate_task09_v2_source_archive(
    archive_path: str | Path, archive_sha256: str,
) -> dict[str, Any]:
    archive = Path(archive_path).resolve()
    root = TASK09_V2_ARCHIVE_ROOT.resolve()
    if (re.fullmatch(r"[0-9a-f]{64}", archive_sha256) is None
            or archive.parent != root
            or not archive.name.startswith("任09F3联合首轮P1修订恢复源码冻结_")
            or not archive.name.endswith(".tar.gz")
            or not archive.is_file() or archive.is_symlink()
            or sha256_file(archive) != archive_sha256):
        raise ValueError("任09V2源码tar须为项目任务目录内显式普通原件且SHA匹配")
    expected = expected_task09_v2_archive_members()
    actual: dict[str, str] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if (len(names) != len(set(names)) or set(names) != set(expected)
                or any(not member.isfile() or member.issym() or member.islnk()
                       or Path(member.name).is_absolute() or ".." in Path(member.name).parts
                       for member in members)):
            raise ValueError("任09V2源码tar仅准九份无重复普通相对路径原件")
        for member in members:
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError("任09V2源码tar普通成员不可读取")
            actual[member.name] = hashlib.sha256(stream.read()).hexdigest()
    if actual != expected:
        raise ValueError("任09V2源码tar逐成员与当前V2/旧0059原件SHA不一致")
    return {
        "V2源码归档路径": str(archive),
        "V2源码归档SHA256": archive_sha256,
        "普通文件成员数": len(actual),
        "逐成员SHA256": actual,
    }


def require_task09_v2_root_ledger(
    *, registry_sha256: str, archive_path: str | Path, archive_sha256: str,
) -> dict[str, Any]:
    """Require the real ROOT-owned 0075 row to bind YAML and source tar together."""
    archive = Path(archive_path).resolve()
    archive_root = TASK09_V2_ARCHIVE_ROOT.resolve()
    if (re.fullmatch(r"[0-9a-f]{64}", registry_sha256) is None
            or re.fullmatch(r"[0-9a-f]{64}", archive_sha256) is None
            or not archive.is_file() or archive.is_symlink()
            or not archive.is_relative_to(archive_root)
            or sha256_file(archive) != archive_sha256):
        raise ValueError("任09V2录0075核验前须给真实项目内源码归档路径与SHA")
    ledger = TASK09_LEDGER
    if not ledger.is_file() or ledger.is_symlink():
        raise ValueError("任09V2 ROOT唯一总台账原件缺失")
    before = sha256_file(ledger)
    text = ledger.read_text(encoding="utf-8")
    after = sha256_file(ledger)
    if before != after:
        raise ValueError("任09V2 ROOT总台账读取期间发生变化")
    rows = [line for line in text.splitlines()
            if re.match(r"^\|\s*录-0075\s*\|", line)]
    if not rows:
        raise ValueError("任09V2 ROOT主台账尚无录0075真实登记，禁止GPU")
    if len(rows) != 1:
        raise ValueError("任09V2 ROOT主台账录0075必须唯一")
    row = rows[0]
    if (registry_sha256 not in row or archive_sha256 not in row
            or archive.name not in row):
        raise ValueError("任09V2录0075同一行必须同时包含YAML与源码tar双SHA及归档名")
    return {
        "ROOT主台账条目": "录-0075",
        "ROOT主台账路径": str(ledger.resolve()),
        "ROOT主台账SHA256": before,
        "ROOT主台账读取前后SHA256一致": True,
        "V2事前登记SHA256": registry_sha256,
        "V2源码归档路径": str(archive),
        "V2源码归档SHA256": archive_sha256,
        "录0075同行双SHA": True,
    }


def _equal_tensor_mapping(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (set(left) == set(right)
            and all(isinstance(left[name], torch.Tensor)
                    and isinstance(right[name], torch.Tensor)
                    and torch.equal(left[name], right[name]) for name in left))


def audit_task09_failed_recovery_source(
    source_directory: str | Path = FAILED_RUN_DIRECTORY,
) -> dict[str, Any]:
    """Read-only audit of the one authorized corr500/joint0 failure source."""
    source_path = Path(source_directory).resolve()
    if source_path != FAILED_RUN_DIRECTORY.resolve():
        raise ValueError("任09V2只接受已登记唯一原失败现场路径")
    if not source_path.is_dir():
        raise FileNotFoundError(f"任09V2原失败现场缺失：{source_path}")
    bad = {
        name: (sha256_file(source_path / name) if (source_path / name).is_file() else None, sha)
        for name, sha in FAILED_RUN_FILES_SHA256.items()
        if not (source_path / name).is_file() or sha256_file(source_path / name) != sha
    }
    if bad:
        raise ValueError(f"任09V2原失败现场文件SHA改变：{bad}")
    prefix, tail, _, _ = _split_failed_log()
    payload = torch.load(source_path / "阶段_最近.pt", map_location="cpu", weights_only=False)
    joint_initial = torch.load(source_path / "阶段_联合初始.pt", map_location="cpu",
                               weights_only=False)
    correction_terminal = torch.load(source_path / "阶段_校正末.pt", map_location="cpu",
                                     weights_only=False)
    sources = _v1.load_task09_sources()
    source = sources[0]
    sample = _task09_sample()
    registry_sha = payload["metadata"]["正式F3事前登记SHA256"]
    check_task09_v2_resume_payload(
        payload, source, sample, registry_sha256=registry_sha,
    )
    meta = payload["metadata"]
    _v1.validate_task09_logged_history(prefix, meta, source, sample)
    validated_tail = validate_task09_uncommitted_tail(
        tail, committed_metadata=meta, sample=sample, seed=0,
        powers=sample["powers_w"],
    )
    if (source.lf_tensor_sha256 != FAILED_SOURCE_LF_TENSOR_SHA256
            or registry_sha != ORIGINAL_FORMAL_REGISTRY_SHA256
            or payload["stage"] != _v1.JOINT_STAGE or payload["epoch"] != 0):
        raise ValueError("任09V2失败源LF原张量、旧0059登记或corr500/joint0身份改变")
    if (sha256_file(source_path / "阶段_最近.pt") !=
            sha256_file(source_path / "阶段_联合初始.pt")
            or payload["metadata"] != joint_initial["metadata"]
            or not _equal_tensor_mapping(payload["model_state"], joint_initial["model_state"])
            or correction_terminal["stage"] != _v1.CORRECTION_STAGE
            or correction_terminal["epoch"] != 500
            or payload["metadata"].get("已提交真实校正末态SHA256") !=
            FAILED_RUN_FILES_SHA256["阶段_校正末.pt"]
            or not _equal_tensor_mapping(payload["model_state"],
                                         correction_terminal["model_state"])):
        raise ValueError("任09V2 recent/joint0或corr500终态引用与模型逐张量不一致")
    source_snapshot = _tree_files_sha256(source_path / "source_snapshot")
    config_snapshot = _tree_files_sha256(source_path / "config_snapshot")
    if len(source_snapshot) != 24 or len(config_snapshot) != 10:
        raise ValueError("任09V2失败源source/config快照文件数不符原现场")
    return {
        "状态": "P1提交门禁失败；旧目录只读保留",
        "失败现场路径": str(source_path),
        "失败现场完整日志SHA256": FAILED_RUN_FILES_SHA256["training.jsonl"],
        "已提交最近状态SHA256": FAILED_RUN_FILES_SHA256["阶段_最近.pt"],
        "已提交前500行SHA256": FAILED_LOG_PREFIX_SHA256,
        "未提交尾行SHA256": FAILED_LOG_TAIL_SHA256,
        "已提交校正轮次": 500,
        "已提交联合轮次": 0,
        "未提交尾行数": 1,
        "未提交尾行全局轮次": validated_tail["epoch"],
        "未提交尾行联合轮次": validated_tail["阶段实际轮次"],
        "未提交尾行LF回放batch": validated_tail["LF联合回放batch"],
        "未提交尾行LF回放点": validated_tail["LF真实回放训练点"],
        "未提交尾行不作已提交状态": True,
        "失败joint1模型_优化器_RNG可复用": False,
        "最近状态等于联合初始逐字节": True,
        "校正末与joint0模型逐张量相等": True,
        "最近状态校正末引用SHA256": FAILED_RUN_FILES_SHA256["阶段_校正末.pt"],
        "失败目录顶层文件SHA256": dict(FAILED_RUN_FILES_SHA256),
        "失败目录source_snapshot逐文件SHA256": source_snapshot,
        "失败目录config_snapshot逐文件SHA256": config_snapshot,
    }


def _copy_exact(source: Path, destination: Path) -> str:
    expected = sha256_file(source)
    shutil.copyfile(source, destination)
    actual = sha256_file(destination)
    if actual != expected:
        raise ValueError(f"任09V2复制原件SHA不一致：{source.name}")
    return actual


def stage_task09_v2_recovery(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    name: str,
    size: int,
    seed: int,
    v2_registry_path: str | Path | None = None,
    v2_registry_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a new canonical directory from committed corr500 only."""
    repair_registration = require_task09_v2_registration(
        v2_registry_path, v2_registry_sha256,
    )
    audit = audit_task09_failed_recovery_source(source_directory)
    if (name, size, seed) != ("primary", 3, 0):
        raise ValueError("任09V2本次P1恢复只接受原失败primary/HF3/seed0")
    output = _v1.validate_task09_output(output_directory, name, size, seed)
    if output.exists():
        raise FileExistsError("任09V2恢复输出已有内容；只准全新目录且不得覆盖")
    output.mkdir(parents=True, exist_ok=False)
    source = FAILED_RUN_DIRECTORY
    shutil.copytree(source / "source_snapshot", output / "source_snapshot")
    shutil.copytree(source / "config_snapshot", output / "config_snapshot")
    v2_snapshot = output / "source_snapshot_v2"
    v2_snapshot.mkdir()
    v2_copied: dict[str, str] = {}
    for label, path in _v2_source_paths().items():
        destination = v2_snapshot / path.name
        v2_copied[label] = _copy_exact(path, destination)
    v2_copied["registration_v2"] = _copy_exact(
        TASK09_V2_REGISTRY, v2_snapshot / TASK09_V2_REGISTRY.name,
    )
    immutable_files = (
        "源码与登记事前快照SHA256.json", "阶段_初始.pt", "阶段_最近.pt",
        "阶段_校正末.pt", "阶段_联合初始.pt", "阶段_观测最佳.pt",
        "阶段_物理最佳.pt", "best.pt",
    )
    copied = {name: _copy_exact(source / name, output / name) for name in immutable_files}
    copied[RECOVERY_BASELINE] = _copy_exact(source / "阶段_最近.pt",
                                             output / RECOVERY_BASELINE)
    _, _, prefix_bytes, tail_bytes = _split_failed_log()
    (output / "training.jsonl").write_bytes(prefix_bytes)
    (output / RECOVERY_TAIL).write_bytes(tail_bytes)
    copied[RECOVERY_FULL_FAILED_LOG] = _copy_exact(
        source / "training.jsonl", output / RECOVERY_FULL_FAILED_LOG,
    )
    copied[RECOVERY_STALE_REPORT] = _copy_exact(
        source / "任09F3阶段报告.json", output / RECOVERY_STALE_REPORT,
    )
    for receipt in (
        "正式主HF3_seed0第200校正轮独立收据_20260916T061402+0800.md",
        "正式主HF3_seed0第400校正轮独立收据_20260916T061637+0800.md",
    ):
        copied[f"恢复源_{receipt}"] = _copy_exact(source / receipt,
                                                   output / f"恢复源_{receipt}")
    manifest = {
        "schema_version": 1,
        "任务": "任09 P1冻结LF AdamW提交门禁修订恢复",
        "V2事前登记SHA256": repair_registration,
        "恢复源失败现场": str(source.resolve()),
        "恢复源失败现场完整日志SHA256": audit["失败现场完整日志SHA256"],
        "恢复源已提交最近状态SHA256": audit["已提交最近状态SHA256"],
        "恢复源已提交前500行SHA256": FAILED_LOG_PREFIX_SHA256,
        "恢复源未提交joint1尾行SHA256": FAILED_LOG_TAIL_SHA256,
        "恢复后已提交全局轮次": 500,
        "恢复后校正已提交轮次": 500,
        "恢复后受限联合已提交轮次": 0,
        "未提交joint1将在CUDA重新计算": True,
        "未复用失败joint1模型_优化器_RNG": True,
        "旧失败目录写入": False,
        "旧固定TEST温度读取": False,
        "陈旧epoch400阶段报告仅作失败现场证据_不可恢复": True,
        "恢复源source_snapshot逐文件SHA256": audit["失败目录source_snapshot逐文件SHA256"],
        "恢复源config_snapshot逐文件SHA256": audit["失败目录config_snapshot逐文件SHA256"],
        "新目录初始复制文件SHA256": copied,
        "新目录V2源码快照SHA256": v2_copied,
    }
    (output / RECOVERY_MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    validate_task09_v2_recovery_directory(output, source_directory=source)
    return manifest


def validate_task09_v2_recovery_directory(
    output_directory: str | Path,
    *, source_directory: str | Path = FAILED_RUN_DIRECTORY,
    v2_registry_path: str | Path | None = None,
    v2_registry_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate immutable recovery baseline while permitting later new commits."""
    audit_task09_failed_recovery_source(source_directory)
    output = Path(output_directory).resolve()
    manifest_path = output / RECOVERY_MANIFEST
    if not manifest_path.is_file():
        raise ValueError("任09V2恢复目录缺结构化恢复来源清单")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registered_sha = v2_registry_sha256 or manifest.get("V2事前登记SHA256")
    require_task09_v2_registration(
        v2_registry_path or TASK09_V2_REGISTRY, registered_sha,
    )
    if (manifest.get("schema_version") != 1
            or manifest.get("恢复源失败现场") != str(FAILED_RUN_DIRECTORY.resolve())
            or manifest.get("恢复源已提交前500行SHA256") != FAILED_LOG_PREFIX_SHA256
            or manifest.get("恢复源未提交joint1尾行SHA256") != FAILED_LOG_TAIL_SHA256
            or manifest.get("V2事前登记SHA256") != registered_sha
            or manifest.get("未复用失败joint1模型_优化器_RNG") is not True
            or manifest.get("旧失败目录写入") is not False):
        raise ValueError("任09V2恢复来源清单未绑定唯一失败源或把未提交joint1冒作状态")
    baseline = output / RECOVERY_BASELINE
    tail = output / RECOVERY_TAIL
    full_failed_log = output / RECOVERY_FULL_FAILED_LOG
    stale_report = output / RECOVERY_STALE_REPORT
    log = output / "training.jsonl"
    if (not baseline.is_file() or sha256_file(baseline) !=
            FAILED_RUN_FILES_SHA256["阶段_最近.pt"]
            or not tail.is_file() or sha256_file(tail) != FAILED_LOG_TAIL_SHA256
            or not full_failed_log.is_file() or sha256_file(full_failed_log) !=
            FAILED_RUN_FILES_SHA256["training.jsonl"]
            or not stale_report.is_file() or sha256_file(stale_report) !=
            FAILED_RUN_FILES_SHA256["任09F3阶段报告.json"]
            or not log.is_file()):
        raise ValueError("任09V2恢复基线、未提交尾行证据或训练日志缺失/改变")
    lines = log.read_bytes().splitlines(keepends=True)
    if (len(lines) < 500 or hashlib.sha256(b"".join(lines[:500])).hexdigest()
            != FAILED_LOG_PREFIX_SHA256):
        raise ValueError("任09V2新目录已提交前500行训练历史改变")
    for name, expected in manifest["新目录初始复制文件SHA256"].items():
        path = output / name
        if name in {"阶段_最近.pt", "阶段_观测最佳.pt", "阶段_物理最佳.pt", "best.pt"}:
            continue
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"任09V2恢复不可变原件改变：{name}")
    if (_tree_files_sha256(output / "source_snapshot") !=
            manifest.get("恢复源source_snapshot逐文件SHA256")
            or _tree_files_sha256(output / "config_snapshot") !=
            manifest.get("恢复源config_snapshot逐文件SHA256")):
        raise ValueError("任09V2恢复source/config快照逐文件SHA改变")
    expected_v2_snapshot = {
        path.name: manifest["新目录V2源码快照SHA256"][label]
        for label, path in _v2_source_paths().items()
    }
    expected_v2_snapshot[TASK09_V2_REGISTRY.name] = manifest[
        "新目录V2源码快照SHA256"
    ]["registration_v2"]
    if _tree_files_sha256(output / "source_snapshot_v2") != expected_v2_snapshot:
        raise ValueError("任09V2恢复目录的新trainer/CLI/登记/TDD源码快照SHA改变")
    payload = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    source = _v1.load_task09_sources()[0]
    sample = _task09_sample()
    registry_sha = payload["metadata"]["正式F3事前登记SHA256"]
    check_task09_v2_resume_payload(
        payload, source, sample, registry_sha256=registry_sha,
    )
    # v1 verifies every committed row and both complete best tracks.  The v2
    # AdamW function is scoped here as well so later joint commits are legal.
    original = _v1.assert_task09_adamw_steps
    _v1.assert_task09_adamw_steps = assert_task09_v2_adamw_steps
    try:
        _v1._task09_checkpoint_preflight(
            output, payload, source, sample, registry_sha256=registry_sha,
        )
    finally:
        _v1.assert_task09_adamw_steps = original
    return manifest


def call_with_task09_v2_step_gate(
    function: Callable[..., _T], *args: Any, **kwargs: Any,
) -> _T:
    """Apply the one-line semantic repair only for a single v1 call."""
    original = _v1.assert_task09_adamw_steps
    if original is assert_task09_v2_adamw_steps:
        raise RuntimeError("任09V2提交门禁不允许嵌套或并发替换")
    _v1.assert_task09_adamw_steps = assert_task09_v2_adamw_steps
    try:
        return function(*args, **kwargs)
    finally:
        _v1.assert_task09_adamw_steps = original


def audit_task09_v2_committed_state(
    output_directory: str | Path,
    *, v2_registry_path: str | Path,
    v2_registry_sha256: str,
) -> dict[str, Any]:
    """Audit a newly committed state after v1 returned successfully."""
    validate_task09_v2_recovery_directory(
        output_directory, v2_registry_path=v2_registry_path,
        v2_registry_sha256=v2_registry_sha256,
    )
    output = Path(output_directory).resolve()
    payload = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    meta = payload["metadata"]
    source = _v1.load_task09_sources()[0]
    sample = _task09_sample()
    model, optimizer = _load_v2_live_state(payload, source)
    steps = assert_task09_v2_adamw_steps(
        optimizer, dict(model.named_parameters()),
        hf_batches=sample["ir_batches_2048"],
        correction_epochs=meta["校正实际轮次"], joint_epochs=meta["联合实际轮次"],
    )
    rows = (output / "training.jsonl").read_text(encoding="utf-8").splitlines()
    if len(rows) != meta["完整全局实际轮次"]:
        raise ValueError("任09V2提交后活动日志行数与最近完整状态不一致")
    corrections = {steps[name] for name in steps if name.startswith("correction.")}
    projections = {steps[name] for name in PROJECTION_NAMES}
    frozen = {steps[name] for name in steps
              if name.startswith("low_fidelity_model.") and name not in PROJECTION_NAMES}
    if meta["联合实际轮次"] == 1 and (
        corrections != {2505} or projections != {5} or frozen != {0} or len(rows) != 501
    ):
        raise ValueError("任09V2重算joint1后须correction2505/projection5/frozen0/log501")
    return {
        "校正已提交轮次": meta["校正实际轮次"],
        "联合已提交轮次": meta["联合实际轮次"],
        "全局已提交轮次": meta["完整全局实际轮次"],
        "活动日志行数": len(rows),
        "新HF校正AdamW_step集合": sorted(corrections),
        "LF四末投影AdamW_step集合": sorted(projections),
        "其余冻结LF参数AdamW_step集合": sorted(frozen),
        "未提交旧joint1模型_优化器_RNG复用": False,
    }


def run_task09_formal_v2(
    *, name: str, size: int, seed: int, output_directory: str | Path,
    v2_registry_path: str | Path, v2_registry_sha256: str,
    original_registry_path: str | Path, original_registry_sha256: str,
    root_ledger_entry: str, session_epoch_limit: int,
    resume_checkpoint: str | Path, device_name: str = "cuda",
    v2_source_archive: str | Path | None = None,
    v2_source_archive_sha256: str | None = None,
) -> dict[str, Any]:
    """Resume the staged source through the frozen loop with the scoped P1 gate."""
    repair_sha = require_task09_v2_registration(v2_registry_path, v2_registry_sha256)
    if root_ledger_entry != "录0075":
        raise ValueError("任09V2真CUDA须ROOT主台账先登记录0075")
    if v2_source_archive is None or v2_source_archive_sha256 is None:
        raise ValueError("任09V2真CUDA须显式给出录0075所锁源码tar路径与SHA")
    archive_receipt = validate_task09_v2_source_archive(
        v2_source_archive, v2_source_archive_sha256,
    )
    ledger_receipt = require_task09_v2_root_ledger(
        registry_sha256=repair_sha, archive_path=v2_source_archive,
        archive_sha256=v2_source_archive_sha256,
    )
    if device_name != "cuda":
        raise ValueError("任09V2正式恢复只允许真实CUDA；CPU仅做来源/TDD")
    original_sha = _v1.require_task09_registration(
        original_registry_path, original_registry_sha256,
    )
    if original_sha != ORIGINAL_FORMAL_REGISTRY_SHA256:
        raise ValueError("任09V2必须沿用原0059预算登记SHA，不改训练预算")
    output = _v1.validate_task09_output(output_directory, name, size, seed)
    if (name, size, seed) != ("primary", 3, 0):
        raise ValueError("任09V2此次只恢复已登记失败的primary/HF3/seed0")
    checkpoint = Path(resume_checkpoint).resolve()
    if checkpoint != output / "阶段_最近.pt":
        raise ValueError("任09V2只从新恢复目录的最近完整corr500/joint0状态续跑")
    validate_task09_v2_recovery_directory(
        output, v2_registry_path=v2_registry_path,
        v2_registry_sha256=repair_sha,
    )
    report = call_with_task09_v2_step_gate(
        _v1.run_task09_formal,
        name=name, size=size, seed=seed, output_directory=output,
        device_name=device_name, registry_path=original_registry_path,
        registry_sha256=original_sha, session_epoch_limit=session_epoch_limit,
        resume_checkpoint=checkpoint,
    )
    committed = audit_task09_v2_committed_state(
        output, v2_registry_path=v2_registry_path,
        v2_registry_sha256=repair_sha,
    )
    v2_report = {
        "状态": "任09V2 P1修订门禁已提交真实CUDA阶段",
        "ROOT主台账条目": root_ledger_entry,
        "ROOT主台账真实同行双SHA收据": ledger_receipt,
        "V2源码tar逐成员收据": archive_receipt,
        "V2事前登记SHA256": repair_sha,
        "旧0059事前登记SHA256": original_sha,
        "恢复源失败日志SHA256": FAILED_RUN_FILES_SHA256["training.jsonl"],
        "恢复源未提交joint1只作证据": True,
        "提交后独审": committed,
        "旧固定TEST温度读取": False,
    }
    destination = output / "任09V2提交门禁报告.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(v2_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(destination)
    return {**report, "V2提交门禁": v2_report}
