"""General formal entry for the 14 untrained Task-09 primary identities.

This module leaves the frozen v1/v2 implementations byte-identical.  It gates
the old training loop with a new source registration, a real ROOT ledger row,
the corrected v2 AdamW name-to-step semantics, and a 200-epoch minimum formal
session.  No TEST-temperature source is introduced here.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile
from typing import Any, BinaryIO, Callable, Iterator, Mapping, TypeVar

import numpy as np
import torch
import yaml

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.eval import task09_energy as _energy
from sic_cu.train import task09_subset_formal as _v1
from sic_cu.train import task09_subset_formal_v2 as _v2


TASK09_V3_FORMAL_ROOT = _v1.TASK09_FORMAL_ROOT
TASK09_V3_TRAINER = PROJECT_ROOT / "src/sic_cu/train/task09_subset_formal_v3.py"
TASK09_V3_CLI = PROJECT_ROOT / "scripts/44_run_task09_subset_formal_v3.py"
TASK09_V3_REGISTRATION_CLI = PROJECT_ROOT / "scripts/45_register_task09_subset_formal_v3.py"
TASK09_V3_ENERGY_CLI = PROJECT_ROOT / "scripts/46_audit_task09_energy_v3.py"
TASK09_V3_TEST = PROJECT_ROOT / "tests/test_task09_subset_formal_v3.py"
TASK09_V3_REGISTRY = PROJECT_ROOT / (
    "研究记录/任务09_高保真数据效率/正式F3主序列剩余14模型通用入口前登记_录0078.yaml"
)
TASK09_V3_LEDGER = PROJECT_ROOT / "多保真DeepONet预测精度优化总计划与执行台账.md"
TASK09_V3_ARCHIVE_ROOT = PROJECT_ROOT / "研究记录/任务09_高保真数据效率"
TASK09_V3_ADAMW_GATE = _v2.assert_task09_v2_adamw_steps
ORIGINAL_REGISTRY_SHA256 = (
    "987ba4bfd9a5b964bff88fa3314413738fa5417b5445e71cf650acbaee7d85e9"
)
ORIGINAL_SOURCE_TAR_SHA256 = (
    "0091e55a2f57077f70c1008344aff9d15c73d2e78b0070327b93f8bb172698ef"
)
V2_REGISTRY_SHA256 = (
    "a88ff62c1b24ac288214f20d6d50c642985f9375cd75562b07cfa394e9cf23dc"
)
FROZEN_DEPENDENCY_SHA256 = {
    "task09_subset_formal.py": (
        "3e304eb4b2fd6c351375b7e4176ae344057b809c9c8394d62bc5c110cd0cf375"
    ),
    "task09_subset_formal_v2.py": (
        "5173ab1df25340a5d583d455f321a81409097b4c9620541763683be3e663566d"
    ),
    "35_run_task09_subset_formal.py": (
        "e7768f7414e5908483a37249406bbc8a98f31aee69b0b1e783e8464eeb9b8132"
    ),
    "42_run_task09_subset_formal_v2.py": (
        "d8a41eb65a2d4ecb3f92b430606f128edf13d829dae845fae59bf5504df0a3f2"
    ),
    "task09_energy.py": (
        "a9d3f2759794986f1135c1b2e251499e1d187bddb05f9b2944fd850bd24faaad"
    ),
    "38_audit_task09_energy.py": (
        "9c77cdee4e1ed894180b3f2b881e326da802305a91ad609b940668044b61d213"
    ),
}
V3_SOURCE_SNAPSHOT = "source_snapshot_v3"
V3_LINEAGE_MANIFEST = "任09V3事前来源清单.json"
V3_CANONICAL_TIMESTAMP = "20260916T071500+0800"
V3_RECEIPT_PREFIX = "任09V3会话收据_"
TASK09_V1_ADAMW_GATE = _v1.assert_task09_adamw_steps
TASK09_V1_ARCHIVE_SOURCE = _v1._task09_archive_source
TASK09_V1_OUTPUT_VALIDATOR = _v1.validate_task09_output
TASK09_ENERGY_OUTPUT_VALIDATOR = _energy.validate_task09_output
_T = TypeVar("_T")


def remaining_task09_v3_identities() -> tuple[tuple[int, int], ...]:
    return tuple((size, seed) for size in (3, 6, 9) for seed in range(5)
                 if (size, seed) != (3, 0))


def task09_v3_canonical_outputs() -> dict[tuple[int, int], Path]:
    return {
        (size, seed): (
            TASK09_V3_FORMAL_ROOT
            / f"正式F3V3_primary_HF{size}_seed{seed}_录0078_{V3_CANONICAL_TIMESTAMP}"
        ).resolve()
        for size, seed in remaining_task09_v3_identities()
    }


def validate_task09_v3_identity(name: str, size: int, seed: int) -> tuple[str, int, int]:
    if name != "primary" or (size, seed) not in remaining_task09_v3_identities():
        raise ValueError(
            "任09V3只准primary的剩余14身份；已完成primary/HF3/seed0明确排除"
        )
    return name, size, seed


@contextmanager
def task09_v3_run_claim(
    identity: tuple[str, int, int],
) -> Iterator[dict[str, Any]]:
    name, size, seed = validate_task09_v3_identity(*identity)
    root = TASK09_V3_FORMAL_ROOT
    if not root.is_dir() or root.is_symlink():
        raise ValueError("任09V3跨进程锁根必须是项目内正式普通目录")
    lock_path = root / f".任09V3_{name}_HF{size}_seed{seed}_录0078.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise RuntimeError("任09V3身份排他锁不可安全打开") from error
    stream: BinaryIO | None = None
    locked = False
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or lock_path.is_symlink():
            raise RuntimeError("任09V3身份排他锁必须是唯一普通文件")
        stream = os.fdopen(descriptor, "r+b", closefd=True)
        descriptor = -1
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as error:
            raise RuntimeError("任09V3同canonical身份已有并发进程占用排他锁") from error
        current = os.lstat(lock_path)
        if current.st_ino != info.st_ino or current.st_dev != info.st_dev:
            raise RuntimeError("任09V3身份排他锁路径在获取期间被替换")
        yield {
            "身份": {"序列": name, "HF功率数": size, "seed": seed},
            "锁路径": str(lock_path.resolve()),
            "跨进程非阻塞排他": True,
        }
    finally:
        if stream is not None:
            try:
                if locked:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()
        elif descriptor >= 0:
            os.close(descriptor)


def validate_task09_v3_output(
    output_directory: str | Path, name: str, size: int, seed: int,
    *, resume_checkpoint: str | Path | None,
) -> Path:
    validate_task09_v3_identity(name, size, seed)
    raw_output = Path(output_directory)
    output = raw_output.resolve()
    expected = task09_v3_canonical_outputs()[(size, seed)]
    if output != expected or raw_output.is_symlink():
        raise ValueError("任09V3输出必须是录0078绑定该身份的唯一固定canonical目录")
    if resume_checkpoint is None:
        if output.exists():
            raise FileExistsError("任09V3 fresh正式输出已有内容，不得覆盖且须换全新目录")
    else:
        raw_checkpoint = Path(resume_checkpoint)
        checkpoint = raw_checkpoint.resolve()
        if (not output.is_dir() or checkpoint != output / "阶段_最近.pt"
                or raw_checkpoint.is_symlink() or not checkpoint.is_file()):
            raise ValueError("任09V3续跑只接受同canonical目录的阶段_最近.pt checkpoint")
        if any((output / name).exists() for name in ("阶段_联合末.pt", "阶段_训练末.pt")):
            raise ValueError("任09V3已有联合末/训练末正式终态，不得追加隐形轮次")
    return output


def _validate_task09_v3_runtime_output(
    output_directory: str | Path, name: str, size: int, seed: int,
) -> Path:
    validate_task09_v3_identity(name, size, seed)
    output = Path(output_directory).resolve()
    if output != task09_v3_canonical_outputs()[(size, seed)]:
        raise ValueError("任09V3冻结运行时只接受录0078唯一固定canonical目录")
    return output


def validate_task09_v3_session_limit(
    payload: Mapping[str, Any] | None, session_epoch_limit: int,
) -> dict[str, int]:
    if not isinstance(session_epoch_limit, int) or session_epoch_limit < 1:
        raise ValueError("任09V3正式会话轮次须为正整数")
    if session_epoch_limit > 200:
        raise ValueError("任09V3每个正式会话最多200轮；2000仅是单模型数学阶段上限")
    if payload is None:
        remaining = 2000
    else:
        meta = payload.get("metadata", {})
        stage = payload.get("stage")
        correction = meta.get("校正实际轮次")
        joint = meta.get("联合实际轮次")
        if (not isinstance(correction, int) or not isinstance(joint, int)
                or correction < 0 or joint < 0
                or meta.get("完整全局实际轮次") != correction + joint):
            raise ValueError("任09V3续跑阶段轮次元数据无效")
        if stage == _v1.CORRECTION_STAGE:
            remaining = (500 if meta.get("校正实际截止轮次") is not None
                         else 1500 - correction + 500)
        elif stage == _v1.JOINT_STAGE:
            remaining = 500 - joint
        else:
            raise ValueError("任09V3续跑阶段必须是原校正或受限联合")
    if remaining <= 0:
        raise ValueError("任09V3该模型原阶段预算已达到完成上限，无剩余轮次")
    minimum = min(200, remaining)
    if session_epoch_limit < minimum:
        raise ValueError(f"任09V3正式会话最低请求{minimum}轮；不足200仅许精确短剩余")
    if session_epoch_limit > remaining:
        raise ValueError(f"任09V3本模型阶段剩余最多{remaining}轮，不得超额请求")
    return {"阶段剩余最多轮次": remaining, "本次最低请求轮次": minimum,
            "本次请求轮次": session_epoch_limit}


def require_task09_v3_clean_transaction(
    log_path: str | Path, payload: Mapping[str, Any],
) -> dict[str, int]:
    path = Path(log_path)
    if not path.is_file():
        raise ValueError("任09V3续跑事务缺已提交训练日志")
    temporary = [candidate.name for candidate in path.parent.glob("*.tmp")]
    if temporary:
        raise ValueError(f"任09V3阶段事务残留未提交临时文件：{sorted(temporary)}")
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if any(not line.endswith(b"\n") or not line.strip() for line in lines):
        raise ValueError("任09V3训练事务日志含空行或非完整JSON行")
    rows: list[dict[str, Any]] = []
    try:
        for line in lines:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("not a JSON object")
            rows.append(value)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError("任09V3训练事务日志含坏JSON或非对象") from error
    meta = payload.get("metadata", {})
    committed = meta.get("完整全局实际轮次")
    correction = meta.get("校正实际轮次")
    if not isinstance(committed, int) or not isinstance(correction, int):
        raise ValueError("任09V3最近状态缺已提交事务轮次")
    if len(rows) != committed:
        suffix = max(0, len(rows) - committed)
        raise ValueError(
            f"任09V3活动日志与最近状态事务不等；未提交尾行数{suffix}，禁止自动裁尾"
        )
    for index, row in enumerate(rows, 1):
        expected_stage = (_v1.CORRECTION_STAGE if index <= correction
                          else _v1.JOINT_STAGE)
        local = index if index <= correction else index - correction
        if (row.get("epoch") != index or row.get("训练阶段") != expected_stage
                or row.get("阶段实际轮次") != local):
            raise ValueError("任09V3训练事务日志全局/阶段轮次顺序不真")
    return {"已提交日志行数": committed, "未提交尾行数": 0}


def task09_v3_minimum_qualification(payload: Mapping[str, Any]) -> dict[str, Any]:
    total = payload.get("metadata", {}).get("完整全局实际轮次")
    if not isinstance(total, int) or total < 0:
        raise ValueError("任09V3资格须来自完整阶段真实轮次")
    return {
        "完整已提交轮次": total,
        "达到正式最低200轮": total >= 200,
        "可称五seed曲线": False,
        "可称能源完成": False,
        "可称精度改善": False,
    }


def _frozen_dependency_paths() -> dict[str, Path]:
    return {
        "task09_subset_formal.py": _v1.TASK09_FORMAL_TRAINER,
        "task09_subset_formal_v2.py": _v2.TASK09_V2_TRAINER,
        "35_run_task09_subset_formal.py": _v1.TASK09_FORMAL_CLI,
        "42_run_task09_subset_formal_v2.py": _v2.TASK09_V2_CLI,
        "task09_energy.py": PROJECT_ROOT / "src/sic_cu/eval/task09_energy.py",
        "38_audit_task09_energy.py": PROJECT_ROOT / "scripts/38_audit_task09_energy.py",
    }


def verify_task09_v3_frozen_dependencies() -> dict[str, str]:
    actual = {name: sha256_file(path) for name, path in _frozen_dependency_paths().items()}
    if actual != FROZEN_DEPENDENCY_SHA256:
        raise ValueError("任09V3冻结v1/v2 trainer或CLI原字节已改变")
    return actual


def _v3_source_paths() -> dict[str, Path]:
    return {
        "trainer_v3": TASK09_V3_TRAINER,
        "CLI_v3": TASK09_V3_CLI,
        "energy_CLI_v3": TASK09_V3_ENERGY_CLI,
        "registration_generator_v3": TASK09_V3_REGISTRATION_CLI,
        "tests_v3": TASK09_V3_TEST,
    }


def expected_task09_v3_registration() -> dict[str, Any]:
    frozen = verify_task09_v3_frozen_dependencies()
    paths = _v3_source_paths()
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("任09V3 trainer/CLI/登记生成器/TDD源码尚未齐全")
    if (sha256_file(_v1.TASK09_REGISTRY) != ORIGINAL_REGISTRY_SHA256
            or sha256_file(_v2.TASK09_V2_REGISTRY) != V2_REGISTRY_SHA256):
        raise ValueError("任09V3继承的录0059或v2修订登记原件SHA改变")
    registered = _v1.load_task09_registry()
    observations = _v1.audit_task09_observations(registered)["arms"]["primary"]
    return {
        "schema_version": 1,
        "任务": "任09主序列剩余14模型通用正式v3入口",
        "ROOT主台账预留条目": "录0078",
        "ROOT录0078前GPU禁止": True,
        "正式入口实际读取录0078同行YAML_tar双SHA": True,
        "原录0059正式登记SHA256": ORIGINAL_REGISTRY_SHA256,
        "原录0059四十七源tar_SHA256": ORIGINAL_SOURCE_TAR_SHA256,
        "v2修订登记SHA256": V2_REGISTRY_SHA256,
        "v2已确认AdamW语义复用": True,
        "冻结依赖SHA256": frozen,
        "剩余正式身份": [
            {
                "序列": "primary", "HF功率数": size, "seed": seed,
                "项目内唯一固定canonical目录": str(
                    task09_v3_canonical_outputs()[(size, seed)].relative_to(PROJECT_ROOT)
                ),
            }
            for size, seed in remaining_task09_v3_identities()
        ],
        "剩余正式身份数": 14,
        "排除已完成身份": "primary/HF3/seed0",
        "primary三档HF功率": {
            size: list(registered["primary"][size]) for size in (3, 6, 9)
        },
        "primary三档真实每轮来源预算": {
            size: {
                "IR真行": observations[size]["ir_rows"],
                "IR真实2048批": observations[size]["ir_batches_2048"],
                "Hot真行": observations[size]["sensor_rows"]["Hot"],
                "Cold真行": observations[size]["sensor_rows"]["Cold"],
                "LF60公平分组": _v1.balanced_task09_lf_groups(
                    observations[size]["ir_batches_2048"]
                ),
            } for size in (3, 6, 9)
        },
        "同seed原LF完整权重": True,
        "新六输入HF校正器": True,
        "fresh空AdamW与独立四RNG": True,
        "合法HF验证三功率完整不缩": [115.2, 403.0, 630.5],
        "首次正式会话最低请求轮次": 200,
        "不足200只许阶段精确短剩余": True,
        "单次正式会话硬上限轮次": 200,
        "校正_联合上限轮次": [1500, 500],
        "联合每轮LF真实回放batch_每批点": [60, 2048],
        "联合每个LF2048小批损失系数": 0.25,
        "每轮独立物理配点_优化步": [256, 1],
        "阶段事务要求": "活动日志必须恰等最近完整checkpoint；未提交尾行禁止自动裁剪",
        "会话收据链要求": "每段排他新收据绑定起止轮次、recent、日志、两份阶段报告及前收据SHA",
        "同canonical跨进程排他": "身份固定普通lock文件自预检前至收据提交后持有LOCK_EX_NONBLOCK",
        "校正早停恰逢会话边界": "逐张量核recent与terminal后仅原子补交terminal SHA，不增加轮次",
        "输出目录不得覆盖": True,
        "失败孤儿目录处理": "原样保留且固定路径不复用；只能另行事前登记恢复或废弃",
        "旧v1或五个pilot产物资格": False,
        "能源与最终资格入口": "必须先核V3 manifest、完整会话收据链和训练末态",
        "旧固定TEST温度读取": False,
        "新v3源码SHA256": {name: sha256_file(path) for name, path in paths.items()},
        "本登记已激活GPU": False,
        "本登记已训练模型数": 0,
        "剩余14模型数学上限轮次非承诺": 28000,
        "五seed曲线或能源结论": False,
    }


def require_task09_v3_registration(
    path: str | Path | None, sha256: str | None,
) -> str:
    if path is None or sha256 is None or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError("任09V3须显式给出事前登记YAML和SHA256")
    raw = Path(path)
    candidate = raw.resolve()
    if (candidate != TASK09_V3_REGISTRY.resolve() or not candidate.is_file()
            or raw.is_symlink() or sha256_file(candidate) != sha256):
        raise ValueError("任09V3只接受项目内专属登记原件及匹配SHA")
    if yaml.safe_load(candidate.read_text(encoding="utf-8")) != expected_task09_v3_registration():
        raise ValueError("任09V3登记与14身份、预算、数据或源码现场不一致")
    return sha256


def expected_task09_v3_archive_members() -> dict[str, str]:
    paths = [
        TASK09_V3_TRAINER, TASK09_V3_CLI, TASK09_V3_REGISTRATION_CLI,
        TASK09_V3_ENERGY_CLI, TASK09_V3_TEST, TASK09_V3_REGISTRY,
        _v1.TASK09_FORMAL_TRAINER, _v2.TASK09_V2_TRAINER,
        _v1.TASK09_FORMAL_CLI, _v2.TASK09_V2_CLI,
        PROJECT_ROOT / "src/sic_cu/eval/task09_energy.py",
        PROJECT_ROOT / "scripts/38_audit_task09_energy.py",
        _v1.TASK09_REGISTRY, _v2.TASK09_V2_REGISTRY,
    ]
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise FileNotFoundError("任09V3源码tar十四份普通原件未齐")
    return {path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path) for path in paths}


def validate_task09_v3_source_archive(
    archive_path: str | Path, archive_sha256: str,
) -> dict[str, Any]:
    raw_archive = Path(archive_path)
    archive = raw_archive.resolve()
    root = TASK09_V3_ARCHIVE_ROOT.resolve()
    if (re.fullmatch(r"[0-9a-f]{64}", archive_sha256) is None
            or archive.parent != root or raw_archive.is_symlink() or not archive.is_file()
            or not archive.name.startswith("任09F3主序列剩余14模型v3源码冻结_")
            or not archive.name.endswith(".tar.gz")
            or sha256_file(archive) != archive_sha256):
        raise ValueError("任09V3源码tar须为任务目录内匹配SHA的普通原件")
    expected = expected_task09_v3_archive_members()
    actual: dict[str, str] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if (len(names) != len(set(names)) or set(names) != set(expected)
                or any(not member.isfile() or member.issym() or member.islnk()
                       or Path(member.name).is_absolute() or ".." in Path(member.name).parts
                       for member in members)):
            raise ValueError("任09V3源码tar只许十四份无重复、无链接、无额外的普通相对成员")
        for member in members:
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError("任09V3源码tar普通成员不可读取")
            actual[member.name] = hashlib.sha256(stream.read()).hexdigest()
    if actual != expected:
        raise ValueError("任09V3源码tar成员内容与现场SHA不一致")
    return {"V3源码归档路径": str(archive), "V3源码归档SHA256": archive_sha256,
            "普通文件成员数": len(actual), "逐成员SHA256": actual}


def require_task09_v3_root_ledger(
    *, registry_sha256: str, archive_path: str | Path, archive_sha256: str,
) -> dict[str, Any]:
    raw_archive = Path(archive_path)
    archive = raw_archive.resolve()
    root = TASK09_V3_ARCHIVE_ROOT.resolve()
    if (re.fullmatch(r"[0-9a-f]{64}", registry_sha256) is None
            or re.fullmatch(r"[0-9a-f]{64}", archive_sha256) is None
            or not archive.is_file() or raw_archive.is_symlink()
            or not archive.is_relative_to(root) or sha256_file(archive) != archive_sha256):
        raise ValueError("任09V3录0078核验须给项目内真实tar与SHA")
    ledger = TASK09_V3_LEDGER
    if not ledger.is_file() or ledger.is_symlink():
        raise ValueError("任09V3唯一ROOT总台账缺失")
    before = sha256_file(ledger)
    text = ledger.read_text(encoding="utf-8")
    after = sha256_file(ledger)
    if before != after:
        raise ValueError("任09V3总台账读取期间改变")
    rows = [line for line in text.splitlines()
            if re.match(r"^\|\s*录-0078\s*\|", line)]
    if not rows:
        raise ValueError("任09V3 ROOT总台账尚无录0078登记，禁止GPU")
    if len(rows) != 1:
        raise ValueError("任09V3 ROOT总台账录0078必须唯一")
    row = rows[0]
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    marker = task09_v3_ledger_marker(
        registry_sha256=registry_sha256,
        archive_basename=archive.name,
        archive_sha256=archive_sha256,
    )
    if cells.count(marker) != 1:
        raise ValueError("任09V3录0078同一行须含唯一精确active机器标记及YAML/tar双SHA字段")
    return {
        "ROOT主台账条目": "录-0078", "ROOT主台账SHA256": before,
        "ROOT主台账读取前后SHA256一致": True,
        "V3事前登记SHA256": registry_sha256,
        "V3源码归档SHA256": archive_sha256,
        "V3源码归档路径": str(archive), "录0078同行双SHA": True,
        "录0078行SHA256": hashlib.sha256(row.encode("utf-8")).hexdigest(),
    }


def task09_v3_ledger_marker(
    *, registry_sha256: str, archive_basename: str, archive_sha256: str,
) -> str:
    """Return the exact standalone Markdown-cell marker ROOT must register."""
    if (re.fullmatch(r"[0-9a-f]{64}", registry_sha256) is None
            or re.fullmatch(r"[0-9a-f]{64}", archive_sha256) is None
            or not archive_basename
            or Path(archive_basename).name != archive_basename
            or "|" in archive_basename):
        raise ValueError("任09V3录0078机器标记字段无效")
    return (
        "TASK09_V3_GATE:v1；status=active；"
        f"registry_sha256=`{registry_sha256}`；"
        f"archive_basename=`{archive_basename}`；"
        f"archive_sha256=`{archive_sha256}`"
    )


def _task09_v3_exact_value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape)
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and left.shape == right.shape
                and np.array_equal(left, right, equal_nan=True))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping)
                and set(left) == set(right)
                and all(_task09_v3_exact_value_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_task09_v3_exact_value_equal(a, b) for a, b in zip(left, right)))
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def repair_task09_v3_terminal_boundary(output_directory: str | Path) -> dict[str, Any]:
    """Commit the terminal SHA omitted by frozen v1 at an exact session boundary.

    The correction terminal and recent payloads must be byte-independent but
    semantically identical complete states.  Only recent metadata is replaced;
    neither the terminal checkpoint nor the training log is rewritten.
    """
    output = Path(output_directory)
    recent_path = output / "阶段_最近.pt"
    terminal_path = output / "阶段_校正末.pt"
    if not recent_path.is_file():
        raise ValueError("任09V3校正会话边界缺阶段_最近完整状态")
    recent = torch.load(recent_path, map_location="cpu", weights_only=False)
    meta = recent.get("metadata", {})
    if recent.get("stage") != _v1.CORRECTION_STAGE:
        return {"校正会话边界补交": False}
    completed = meta.get("校正实际截止轮次")
    committed_sha = meta.get("已提交真实校正末态SHA256")
    if completed is None:
        if terminal_path.exists():
            raise ValueError("任09V3校正未截止却存在校正末态，阶段事务不一致")
        return {"校正会话边界补交": False}
    if (not isinstance(completed, int) or completed < 1
            or meta.get("校正实际轮次") != completed
            or meta.get("联合实际轮次") != 0
            or recent.get("epoch") != completed
            or not terminal_path.is_file() or terminal_path.is_symlink()):
        raise ValueError("任09V3校正会话边界轮次或校正末态原件不完整")
    terminal_sha = sha256_file(terminal_path)
    if committed_sha is not None:
        if committed_sha != terminal_sha:
            raise ValueError("任09V3已提交校正末态SHA与原件不一致")
        return {"校正会话边界补交": False}

    terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
    if not _task09_v3_exact_value_equal(recent, terminal):
        raise ValueError("任09V3校正边界recent与terminal模型/AdamW/RNG逐张量并非完全一致")
    log_path = output / "training.jsonl"
    require_task09_v3_clean_transaction(log_path, recent)
    log_sha = sha256_file(log_path)
    terminal_before = terminal_sha
    repaired = copy.deepcopy(recent)
    repaired["metadata"]["已提交真实校正末态SHA256"] = terminal_sha
    temporary = recent_path.with_name(recent_path.name + ".tmp")
    torch.save(repaired, temporary)
    temporary.replace(recent_path)
    verified = torch.load(recent_path, map_location="cpu", weights_only=False)
    if (not _task09_v3_exact_value_equal(repaired, verified)
            or sha256_file(log_path) != log_sha
            or sha256_file(terminal_path) != terminal_before):
        raise ValueError("任09V3校正会话边界原子补交后原件核验失败")
    return {
        "校正会话边界补交": True,
        "校正截止轮次": completed,
        "校正末态SHA256": terminal_sha,
    }


def _task09_v3_receipt_paths(output: Path) -> list[Path]:
    candidates = sorted(output.glob(f"{V3_RECEIPT_PREFIX}*.json"))
    for index, path in enumerate(candidates, 1):
        if (path.is_symlink() or not path.is_file()
                or path.name != f"{V3_RECEIPT_PREFIX}{index:04d}.json"):
            raise ValueError("任09V3会话收据文件名、序号或普通文件资格不连续")
    return candidates


def _task09_v3_bound_artifact_hashes(output: Path) -> dict[str, str]:
    names = (
        "阶段_最近.pt", "training.jsonl", "任09F3阶段报告.json",
        "任09V3阶段门禁报告.json",
    )
    paths = {name: output / name for name in names}
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise ValueError("任09V3会话收据缺recent、日志或两份阶段报告普通原件")
    return {name: sha256_file(path) for name, path in paths.items()}


def _task09_v3_terminal_hashes(output: Path) -> dict[str, str]:
    names = ("阶段_校正末.pt", "阶段_联合末.pt", "阶段_训练末.pt")
    paths = {name: output / name for name in names}
    present = {name for name, path in paths.items() if path.exists() or path.is_symlink()}
    allowed = (set(), {"阶段_校正末.pt"}, set(names))
    if present not in allowed or any(
        not paths[name].is_file() or paths[name].is_symlink() for name in present
    ):
        raise ValueError("任09V3终态须无、仅校正末，或校正/联合/训练末三份普通原件齐全")
    return {name: sha256_file(paths[name]) for name in names if name in present}


def _load_task09_v3_receipt_chain(
    output: Path, *, identity: tuple[str, int, int], registry_sha256: str,
    archive_sha256: str, ledger_row_sha256: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    paths = _task09_v3_receipt_paths(output)
    receipts: list[dict[str, Any]] = []
    hashes: list[str] = []
    previous_sha: str | None = None
    previous_end = 0
    expected_identity = {
        "序列": identity[0], "HF功率数": identity[1], "seed": identity[2],
    }
    for index, path in enumerate(paths, 1):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("任09V3会话收据不是完整JSON原件") from error
        if not isinstance(receipt, dict):
            raise ValueError("任09V3会话收据必须是JSON对象")
        start = receipt.get("起始完整全局轮次")
        end = receipt.get("结束完整全局轮次")
        added = receipt.get("实际新增轮次")
        requested = receipt.get("请求最多轮次")
        stage = receipt.get("会话末阶段")
        correction = receipt.get("会话末校正实际轮次")
        joint = receipt.get("会话末联合实际轮次")
        terminal_hashes = receipt.get("会话末终态SHA256", {})
        artifacts = receipt.get("会话末原件SHA256", {})
        if not isinstance(terminal_hashes, dict) or not isinstance(artifacts, dict):
            raise ValueError("任09V3会话收据的原件SHA字段必须是对象")
        final_terminal = set(terminal_hashes) == {
            "阶段_校正末.pt", "阶段_联合末.pt", "阶段_训练末.pt",
        }
        if (receipt.get("schema_version") != 1
                or receipt.get("会话序号") != index
                or receipt.get("运行身份") != expected_identity
                or receipt.get("项目内唯一固定canonical目录") != str(output)
                or receipt.get("V3事前登记SHA256") != registry_sha256
                or receipt.get("V3源码归档SHA256") != archive_sha256
                or receipt.get("录0078行SHA256") != ledger_row_sha256
                or receipt.get("前一会话收据SHA256") != previous_sha
                or not isinstance(start, int) or start != previous_end
                or not isinstance(end, int) or not isinstance(added, int)
                or end - start != added or not 1 <= added <= 200
                or not isinstance(requested, int) or not added <= requested <= 200
                or stage not in (_v1.CORRECTION_STAGE, _v1.JOINT_STAGE)
                or not isinstance(correction, int) or not isinstance(joint, int)
                or correction < 0 or joint < 0 or correction + joint != end
                or (stage == _v1.CORRECTION_STAGE and joint != 0)
                or (stage == _v1.JOINT_STAGE and joint < 0)
                or set(terminal_hashes) not in (
                    set(), {"阶段_校正末.pt"},
                    {"阶段_校正末.pt", "阶段_联合末.pt", "阶段_训练末.pt"},
                )
                or any(re.fullmatch(r"[0-9a-f]{64}", value) is None
                       for value in terminal_hashes.values())
                or (terminal_hashes.get("阶段_校正末.pt") !=
                    receipt.get("已提交真实校正末态SHA256"))
                or ((added < requested or requested < 200) and not final_terminal)
                or set(artifacts) != {
                    "阶段_最近.pt", "training.jsonl", "任09F3阶段报告.json",
                    "任09V3阶段门禁报告.json",
                }
                or any(re.fullmatch(r"[0-9a-f]{64}", value) is None
                       for value in artifacts.values())):
            raise ValueError("任09V3会话收据链身份、轮次、来源或原件字段不真")
        receipt_sha = sha256_file(path)
        receipts.append(receipt)
        hashes.append(receipt_sha)
        previous_sha = receipt_sha
        previous_end = end
    return receipts, hashes


def validate_task09_v3_receipt_chain(
    output_directory: str | Path, *, identity: tuple[str, int, int],
    registry_sha256: str, archive_sha256: str, ledger_row_sha256: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(output_directory).resolve()
    receipts, hashes = _load_task09_v3_receipt_chain(
        output, identity=identity, registry_sha256=registry_sha256,
        archive_sha256=archive_sha256, ledger_row_sha256=ledger_row_sha256,
    )
    if not receipts:
        raise ValueError("任09V3续跑或下游资格缺不可覆盖的会话收据链")
    total = payload.get("metadata", {}).get("完整全局实际轮次")
    latest = receipts[-1]
    terminal_hashes = _task09_v3_terminal_hashes(output)
    if (latest["结束完整全局轮次"] != total
            or latest["会话末原件SHA256"] != _task09_v3_bound_artifact_hashes(output)
            or latest["会话末终态SHA256"] != terminal_hashes):
        raise ValueError("任09V3会话收据链头与当前recent/日志/报告SHA不一致")
    return {
        "会话收据数": len(receipts),
        "链头完整全局轮次": total,
        "链头收据SHA256": hashes[-1],
        "链头收据路径": str(_task09_v3_receipt_paths(output)[-1]),
        "链头终态SHA256": terminal_hashes,
    }


def write_task09_v3_session_receipt(
    output_directory: str | Path, *, identity: tuple[str, int, int],
    registry_sha256: str, archive_sha256: str, ledger_row_sha256: str,
    start_global_epoch: int, requested_epochs: int, payload: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(output_directory).resolve()
    receipts, hashes = _load_task09_v3_receipt_chain(
        output, identity=identity, registry_sha256=registry_sha256,
        archive_sha256=archive_sha256, ledger_row_sha256=ledger_row_sha256,
    )
    previous_end = receipts[-1]["结束完整全局轮次"] if receipts else 0
    if start_global_epoch < previous_end:
        raise FileExistsError("任09V3会话收据已存在，不得覆盖或回滚重写")
    if start_global_epoch != previous_end:
        raise ValueError("任09V3新会话起点未连续承接上一收据链头")
    end = payload.get("metadata", {}).get("完整全局实际轮次")
    added = end - start_global_epoch if isinstance(end, int) else None
    if (not isinstance(start_global_epoch, int) or start_global_epoch < 0
            or not isinstance(end, int) or not isinstance(added, int)
            or not 1 <= added <= requested_epochs <= 200):
        raise ValueError("任09V3会话实际新增轮次须为1至请求上限且每段最多200")
    terminal_hashes = _task09_v3_terminal_hashes(output)
    if ((added < requested_epochs or requested_epochs < 200)
            and set(terminal_hashes) != {
                "阶段_校正末.pt", "阶段_联合末.pt", "阶段_训练末.pt",
            }):
        raise ValueError("任09V3不足200的短会话只许真实联合截止/精确短剩余并绑定三末态")
    meta = payload.get("metadata", {})
    stage = payload.get("stage")
    correction = meta.get("校正实际轮次")
    joint = meta.get("联合实际轮次")
    if (stage not in (_v1.CORRECTION_STAGE, _v1.JOINT_STAGE)
            or not isinstance(correction, int) or not isinstance(joint, int)
            or correction + joint != end):
        raise ValueError("任09V3会话收据须绑定真实校正/联合阶段轮次")
    correction_terminal_sha = meta.get("已提交真实校正末态SHA256")
    if terminal_hashes.get("阶段_校正末.pt") != correction_terminal_sha:
        raise ValueError("任09V3会话收据的校正末态SHA须与最近完整状态一致")
    sequence = len(receipts) + 1
    destination = output / f"{V3_RECEIPT_PREFIX}{sequence:04d}.json"
    if destination.exists():
        raise FileExistsError("任09V3同序号会话收据已存在，不得覆盖")
    receipt = {
        "schema_version": 1,
        "会话序号": sequence,
        "运行身份": {"序列": identity[0], "HF功率数": identity[1], "seed": identity[2]},
        "项目内唯一固定canonical目录": str(output),
        "V3事前登记SHA256": registry_sha256,
        "V3源码归档SHA256": archive_sha256,
        "录0078行SHA256": ledger_row_sha256,
        "前一会话收据SHA256": hashes[-1] if hashes else None,
        "起始完整全局轮次": start_global_epoch,
        "结束完整全局轮次": end,
        "实际新增轮次": added,
        "请求最多轮次": requested_epochs,
        "会话末阶段": stage,
        "会话末校正实际轮次": correction,
        "会话末联合实际轮次": joint,
        "已提交真实校正末态SHA256": correction_terminal_sha,
        "会话末原件SHA256": _task09_v3_bound_artifact_hashes(output),
        "会话末终态SHA256": terminal_hashes,
        "本收据不等于五seed曲线或能源结论": True,
    }
    encoded = (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    validate_task09_v3_receipt_chain(
        output, identity=identity, registry_sha256=registry_sha256,
        archive_sha256=archive_sha256, ledger_row_sha256=ledger_row_sha256,
        payload=payload,
    )
    return receipt


def call_with_task09_v3_runtime(
    function: Callable[..., _T], *args: Any,
    before_label_archive: Callable[[Path], None] | None = None,
    **kwargs: Any,
) -> _T:
    original_gate = _v1.assert_task09_adamw_steps
    original_archive = _v1._task09_archive_source
    original_validator = _v1.validate_task09_output
    if (original_gate is not TASK09_V1_ADAMW_GATE
            or original_archive is not TASK09_V1_ARCHIVE_SOURCE
            or original_validator is not TASK09_V1_OUTPUT_VALIDATOR):
        raise RuntimeError("任09V3不允许嵌套、并发或预先篡改冻结v1运行时")

    def archived(output: Path) -> dict[str, str]:
        hashes = original_archive(output)
        if before_label_archive is not None:
            before_label_archive(output)
        return hashes

    _v1.assert_task09_adamw_steps = TASK09_V3_ADAMW_GATE
    _v1.validate_task09_output = _validate_task09_v3_runtime_output
    if before_label_archive is not None:
        _v1._task09_archive_source = archived
    try:
        return function(*args, **kwargs)
    finally:
        _v1.assert_task09_adamw_steps = original_gate
        _v1._task09_archive_source = original_archive
        _v1.validate_task09_output = original_validator


def _copy_exact(
    source: Path, destination: Path, *, expected_sha256: str,
) -> str:
    if (re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
            or not source.is_file() or source.is_symlink()
            or sha256_file(source) != expected_sha256):
        raise ValueError(f"任09V3事前SHA原件已改变或不是普通文件：{source.name}")
    shutil.copyfile(source, destination)
    if (sha256_file(source) != expected_sha256
            or not destination.is_file() or destination.is_symlink()
            or sha256_file(destination) != expected_sha256):
        raise ValueError(f"任09V3事前源码复制SHA不一致：{source.name}")
    return expected_sha256


def _write_task09_v3_prelabel_snapshot(
    output: Path, *, registry_sha256: str, archive_path: Path,
    archive_sha256: str, ledger_receipt: Mapping[str, Any],
    identity: tuple[str, int, int],
) -> None:
    require_task09_v3_registration(TASK09_V3_REGISTRY, registry_sha256)
    validate_task09_v3_source_archive(archive_path, archive_sha256)
    frozen = verify_task09_v3_frozen_dependencies()
    registration = yaml.safe_load(TASK09_V3_REGISTRY.read_text(encoding="utf-8"))
    expected_sources = registration["新v3源码SHA256"]
    snapshot = output / V3_SOURCE_SNAPSHOT
    snapshot.mkdir()
    copied = {label: _copy_exact(
        path, snapshot / path.name, expected_sha256=expected_sources[label],
    )
              for label, path in _v3_source_paths().items()}
    copied["registration_v3"] = _copy_exact(
        TASK09_V3_REGISTRY, snapshot / TASK09_V3_REGISTRY.name,
        expected_sha256=registry_sha256,
    )
    copied["source_archive_v3"] = _copy_exact(
        archive_path, snapshot / archive_path.name,
        expected_sha256=archive_sha256,
    )
    require_task09_v3_registration(TASK09_V3_REGISTRY, registry_sha256)
    validate_task09_v3_source_archive(archive_path, archive_sha256)
    if verify_task09_v3_frozen_dependencies() != frozen:
        raise ValueError("任09V3事前归档期间冻结依赖发生改变")
    manifest = {
        "schema_version": 1,
        "任务": "任09主序列剩余14模型通用正式v3入口",
        "运行身份": {"序列": identity[0], "HF功率数": identity[1], "seed": identity[2]},
        "V3事前登记SHA256": registry_sha256,
        "V3源码归档SHA256": archive_sha256,
        "录0078行SHA256": ledger_receipt["录0078行SHA256"],
        "冻结v1_v2与能源依赖SHA256": frozen,
        "首读训练或验证标签之前归档": True,
        "旧固定TEST温度读取": False,
        "V3源码快照SHA256": copied,
    }
    (output / V3_LINEAGE_MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def _validate_task09_v3_lineage(
    output: Path, *, identity: tuple[str, int, int], registry_sha256: str,
    archive_path: Path, archive_sha256: str, ledger_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = output / V3_LINEAGE_MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("任09V3续跑缺首读标签前的V3来源清单")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("任09V3来源清单必须是JSON对象")
    expected_identity = {"序列": identity[0], "HF功率数": identity[1], "seed": identity[2]}
    frozen = verify_task09_v3_frozen_dependencies()
    if (manifest.get("schema_version") != 1
            or manifest.get("运行身份") != expected_identity
            or manifest.get("V3事前登记SHA256") != registry_sha256
            or manifest.get("V3源码归档SHA256") != archive_sha256
            or manifest.get("录0078行SHA256") != ledger_receipt["录0078行SHA256"]
            or manifest.get("冻结v1_v2与能源依赖SHA256") != frozen
            or manifest.get("首读训练或验证标签之前归档") is not True
            or manifest.get("旧固定TEST温度读取") is not False):
        raise ValueError("任09V3续跑身份、录0078或事前源码谱系不一致")
    expected = manifest.get("V3源码快照SHA256", {})
    registration = yaml.safe_load(TASK09_V3_REGISTRY.read_text(encoding="utf-8"))
    expected_hashes = {
        **registration["新v3源码SHA256"],
        "registration_v3": registry_sha256,
        "source_archive_v3": archive_sha256,
    }
    paths = {label: output / V3_SOURCE_SNAPSHOT / path.name
             for label, path in _v3_source_paths().items()}
    paths["registration_v3"] = output / V3_SOURCE_SNAPSHOT / TASK09_V3_REGISTRY.name
    paths["source_archive_v3"] = output / V3_SOURCE_SNAPSHOT / archive_path.name
    if expected != expected_hashes or set(paths) != set(expected) or any(
        not path.is_file() or path.is_symlink() or sha256_file(path) != expected[label]
        for label, path in paths.items()
    ):
        raise ValueError("任09V3续跑事前源码快照原件缺失或SHA改变")
    return manifest


def _task09_v3_sample(size: int) -> dict[str, Any]:
    registered = _v1.load_task09_registry()
    return {**_v1.audit_task09_observations(registered)["arms"]["primary"][size],
            "arm": "primary"}


def _audit_task09_v3_committed_state(
    output: Path, *, size: int, seed: int, registry_sha256: str,
) -> dict[str, Any]:
    checkpoint = output / "阶段_最近.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sources = _v1.load_task09_sources()
    source = sources[seed]
    sample = _task09_v3_sample(size)
    _v2.check_task09_v2_resume_payload(
        payload, source, sample, registry_sha256=ORIGINAL_REGISTRY_SHA256,
    )
    transaction = require_task09_v3_clean_transaction(output / "training.jsonl", payload)
    call_with_task09_v3_runtime(
        _v1._task09_checkpoint_preflight,
        output, payload, source, sample, registry_sha256=ORIGINAL_REGISTRY_SHA256,
    )
    return {**transaction, **task09_v3_minimum_qualification(payload),
            "校正已提交轮次": payload["metadata"]["校正实际轮次"],
            "联合已提交轮次": payload["metadata"]["联合实际轮次"]}


def _preflight_task09_v3_payload(
    output: Path, payload: Mapping[str, Any], *, size: int, seed: int,
) -> None:
    source = _v1.load_task09_sources()[seed]
    sample = _task09_v3_sample(size)
    _v2.check_task09_v2_resume_payload(
        payload, source, sample, registry_sha256=ORIGINAL_REGISTRY_SHA256,
    )
    call_with_task09_v3_runtime(
        _v1._task09_checkpoint_preflight,
        output, payload, source, sample, registry_sha256=ORIGINAL_REGISTRY_SHA256,
    )


def validate_task09_v3_terminal_semantics(
    payload: Mapping[str, Any], report: Mapping[str, Any], *,
    terminal_hashes: Mapping[str, str], identity: tuple[str, int, int],
) -> dict[str, Any]:
    meta = payload.get("metadata", {})
    correction = meta.get("校正实际轮次")
    joint = meta.get("联合实际轮次")
    expected_terminals = {"阶段_校正末.pt", "阶段_联合末.pt", "阶段_训练末.pt"}
    if (payload.get("stage") != _v1.JOINT_STAGE
            or not isinstance(correction, int) or not 200 <= correction <= 1500
            or correction % 10
            or not isinstance(joint, int) or not 200 <= joint <= 500
            or joint % 10
            or payload.get("epoch") != joint
            or meta.get("完整全局实际轮次") != correction + joint
            or meta.get("校正实际截止轮次") != correction):
        raise ValueError("任09V3真实联合终态须JOINT stage、校正/联合至少200且在十轮截止点")
    if (set(terminal_hashes) != expected_terminals
            or any(re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in terminal_hashes.values())
            or meta.get("已提交真实校正末态SHA256") != terminal_hashes["阶段_校正末.pt"]):
        raise ValueError("任09V3真实联合终态须绑定校正/联合/训练三末态SHA")
    if (report.get("运行种子") != identity[2]
            or report.get("序列") != identity[0]
            or report.get("HF子集功率数") != identity[1]
            or report.get("校正实际轮次") != correction
            or report.get("受限联合实际轮次") != joint
            or report.get("旧固定测试温度读取") is not False
            or "受限联合真实截止" not in str(report.get("状态", ""))):
        raise ValueError("任09V3终态报告必须是同身份真实受限联合截止状态")
    return {
        "真实联合终态": True,
        "校正实际轮次": correction,
        "联合实际轮次": joint,
        "三末态SHA256": dict(terminal_hashes),
    }


def require_task09_v3_downstream_qualification(
    *, purpose: str, name: str, size: int, seed: int,
    output_directory: str | Path, v3_registry_path: str | Path,
    v3_registry_sha256: str, v3_source_archive: str | Path,
    v3_source_archive_sha256: str,
) -> dict[str, Any]:
    """Read-only mandatory gate for future energy or final-result consumers."""
    if purpose not in ("energy", "final"):
        raise ValueError("任09V3下游资格用途只允许energy或final")
    registration = require_task09_v3_registration(v3_registry_path, v3_registry_sha256)
    verify_task09_v3_frozen_dependencies()
    validate_task09_v3_source_archive(v3_source_archive, v3_source_archive_sha256)
    ledger = require_task09_v3_root_ledger(
        registry_sha256=registration, archive_path=v3_source_archive,
        archive_sha256=v3_source_archive_sha256,
    )
    identity = validate_task09_v3_identity(name, size, seed)
    output = Path(output_directory).resolve()
    if output != task09_v3_canonical_outputs()[(size, seed)] or not output.is_dir():
        raise ValueError("任09V3下游资格只接受录0078固定canonical目录")
    recent_path = output / "阶段_最近.pt"
    correction_terminal = output / "阶段_校正末.pt"
    terminals = [output / "阶段_联合末.pt", output / "阶段_训练末.pt"]
    if (not recent_path.is_file() or recent_path.is_symlink()
            or not correction_terminal.is_file() or correction_terminal.is_symlink()
            or any(not path.is_file() or path.is_symlink() for path in terminals)):
        raise ValueError("任09V3能源/最终资格须完整校正末、联合末和训练末正式状态")
    payload = torch.load(recent_path, map_location="cpu", weights_only=False)
    require_task09_v3_clean_transaction(output / "training.jsonl", payload)
    report_path = output / "任09F3阶段报告.json"
    if not report_path.is_file() or report_path.is_symlink():
        raise ValueError("任09V3能源/最终资格缺真实阶段报告普通原件")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("任09V3能源/最终资格阶段报告不是完整JSON") from error
    if not isinstance(report, dict):
        raise ValueError("任09V3能源/最终资格阶段报告必须是JSON对象")
    _validate_task09_v3_lineage(
        output, identity=identity, registry_sha256=registration,
        archive_path=Path(v3_source_archive).resolve(),
        archive_sha256=v3_source_archive_sha256, ledger_receipt=ledger,
    )
    chain = validate_task09_v3_receipt_chain(
        output, identity=identity, registry_sha256=registration,
        archive_sha256=v3_source_archive_sha256,
        ledger_row_sha256=ledger["录0078行SHA256"], payload=payload,
    )
    terminal_semantics = validate_task09_v3_terminal_semantics(
        payload, report, terminal_hashes=chain["链头终态SHA256"], identity=identity,
    )
    for terminal_path in terminals:
        terminal_payload = torch.load(terminal_path, map_location="cpu", weights_only=False)
        if not _task09_v3_exact_value_equal(payload, terminal_payload):
            raise ValueError("任09V3能源/最终资格的recent与两份训练末态不完全一致")
    _preflight_task09_v3_payload(output, payload, size=size, seed=seed)
    committed = _audit_task09_v3_committed_state(
        output, size=size, seed=seed, registry_sha256=registration,
    )
    return {
        "V3下游用途": purpose,
        "V3来源谱系与会话链合格": True,
        "运行身份": {"序列": name, "HF功率数": size, "seed": seed},
        "会话链": chain,
        "真实联合终态": terminal_semantics,
        "完整状态": committed,
        "旧v1或pilot产物可进入下游": False,
    }


def call_with_task09_v3_energy_runtime(
    function: Callable[..., _T], *args: Any, **kwargs: Any,
) -> _T:
    original_validator = _energy.validate_task09_output
    if original_validator is not TASK09_ENERGY_OUTPUT_VALIDATOR:
        raise RuntimeError("任09V3不允许嵌套、并发或预先篡改冻结能源运行时")
    _energy.validate_task09_output = _validate_task09_v3_runtime_output
    try:
        return function(*args, **kwargs)
    finally:
        _energy.validate_task09_output = original_validator


def audit_task09_v3_formal_energy(
    *, run: str | Path, name: str, size: int, seed: int, state: str,
    output: str | Path, device_name: str,
    v3_registry_path: str | Path, v3_registry_sha256: str,
    v3_source_archive: str | Path, v3_source_archive_sha256: str,
    original_registry_path: str | Path, original_registry_sha256: str,
) -> dict[str, Any]:
    """Gate v3 provenance before the frozen Task-09 CUDA energy implementation."""
    gate_arguments = {
        "purpose": "energy", "name": name, "size": size, "seed": seed,
        "output_directory": run, "v3_registry_path": v3_registry_path,
        "v3_registry_sha256": v3_registry_sha256,
        "v3_source_archive": v3_source_archive,
        "v3_source_archive_sha256": v3_source_archive_sha256,
    }
    qualification = require_task09_v3_downstream_qualification(**gate_arguments)
    json.dumps(qualification, ensure_ascii=False)
    old_registration = _v1.require_task09_registration(
        original_registry_path, original_registry_sha256,
    )
    if old_registration != ORIGINAL_REGISTRY_SHA256:
        raise ValueError("任09V3能源必须在CUDA前原样核实录0059登记路径和SHA")
    if device_name != "cuda":
        raise ValueError("任09V3能源包装入口只允许真实CUDA；CPU仅作来源门禁测试")
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError("任09V3能源输出或入口收据已存在，不得覆盖")
    validated = _energy.task09_energy_output(run, output, state)
    if validated != destination:
        raise ValueError("任09V3能源输出未通过冻结38入口canonical校验")
    result = call_with_task09_v3_energy_runtime(
        _energy.audit_task09_formal_energy,
        run, name=name, size=size, seed=seed, state=state, output=destination,
        device_name=device_name, registry_path=original_registry_path,
        registry_sha256=old_registration,
    )
    after = require_task09_v3_downstream_qualification(**gate_arguments)
    if after != qualification:
        raise ValueError("任09V3能源计算期间manifest、收据链或训练末态发生变化")
    if not destination.is_dir() or destination.is_symlink():
        raise ValueError("任09V3冻结能源实现未提交唯一普通输出目录")
    files = sorted(path for path in destination.iterdir()
                   if path.name != "任09V3能源入口收据.json")
    if not files or any(not path.is_file() or path.is_symlink() for path in files):
        raise ValueError("任09V3能源输出须为冻结实现写出的普通原件")
    receipt = {
        "schema_version": 1,
        "任务": "任09V3正式能源包装入口收据",
        "运行身份": {"序列": name, "HF功率数": size, "seed": seed},
        "审核状态": state,
        "V3下游资格": qualification,
        "V3事前登记SHA256": v3_registry_sha256,
        "V3源码归档SHA256": v3_source_archive_sha256,
        "原录0059登记SHA256": original_registry_sha256,
        "冻结能源输出逐文件SHA256": {path.name: sha256_file(path) for path in files},
        "旧固定TEST温度读取": False,
    }
    receipt_path = destination / "任09V3能源入口收据.json"
    encoded = (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, receipt_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        **result,
        "V3下游资格": qualification,
        "V3能源入口收据SHA256": sha256_file(receipt_path),
    }


def _run_task09_formal_v3_claimed(
    *, name: str, size: int, seed: int, output_directory: str | Path,
    v3_registry_path: str | Path, v3_registry_sha256: str,
    v3_source_archive: str | Path, v3_source_archive_sha256: str,
    original_registry_path: str | Path, original_registry_sha256: str,
    root_ledger_entry: str, session_epoch_limit: int,
    resume_checkpoint: str | Path | None, device_name: str = "cuda",
) -> dict[str, Any]:
    """Gate then delegate one remaining-primary formal session to frozen v1."""
    registration = require_task09_v3_registration(v3_registry_path, v3_registry_sha256)
    verify_task09_v3_frozen_dependencies()
    archive_receipt = validate_task09_v3_source_archive(
        v3_source_archive, v3_source_archive_sha256,
    )
    if root_ledger_entry != "录0078":
        raise ValueError("任09V3正式CUDA须ROOT总台账录0078")
    ledger_receipt = require_task09_v3_root_ledger(
        registry_sha256=registration, archive_path=v3_source_archive,
        archive_sha256=v3_source_archive_sha256,
    )
    if device_name != "cuda":
        raise ValueError("任09V3正式入口只允许真实CUDA，CPU仅作来源/TDD")
    old_registration = _v1.require_task09_registration(
        original_registry_path, original_registry_sha256,
    )
    if old_registration != ORIGINAL_REGISTRY_SHA256:
        raise ValueError("任09V3必须原样继承录0059正式预算登记")
    identity = validate_task09_v3_identity(name, size, seed)
    output = validate_task09_v3_output(
        output_directory, name, size, seed, resume_checkpoint=resume_checkpoint,
    )
    previous = None
    start_global_epoch = 0
    if resume_checkpoint is not None:
        previous = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        require_task09_v3_clean_transaction(output / "training.jsonl", previous)
        _validate_task09_v3_lineage(
            output, identity=identity, registry_sha256=registration,
            archive_path=Path(v3_source_archive).resolve(),
            archive_sha256=v3_source_archive_sha256, ledger_receipt=ledger_receipt,
        )
        _preflight_task09_v3_payload(output, previous, size=size, seed=seed)
        start_global_epoch = previous["metadata"]["完整全局实际轮次"]
        validate_task09_v3_receipt_chain(
            output, identity=identity, registry_sha256=registration,
            archive_sha256=v3_source_archive_sha256,
            ledger_row_sha256=ledger_receipt["录0078行SHA256"], payload=previous,
        )
    validate_task09_v3_session_limit(previous, session_epoch_limit)

    archive_callback = None
    if resume_checkpoint is None:
        archive_callback = lambda directory: _write_task09_v3_prelabel_snapshot(
            directory, registry_sha256=registration,
            archive_path=Path(v3_source_archive).absolute(),
            archive_sha256=v3_source_archive_sha256,
            ledger_receipt=ledger_receipt, identity=identity,
        )
    report = call_with_task09_v3_runtime(
        _v1.run_task09_formal,
        name=name, size=size, seed=seed, output_directory=output,
        device_name=device_name, registry_path=original_registry_path,
        registry_sha256=old_registration, session_epoch_limit=session_epoch_limit,
        resume_checkpoint=resume_checkpoint,
        before_label_archive=archive_callback,
    )
    boundary_receipt = repair_task09_v3_terminal_boundary(output)
    _validate_task09_v3_lineage(
        output, identity=identity, registry_sha256=registration,
        archive_path=Path(v3_source_archive).resolve(),
        archive_sha256=v3_source_archive_sha256, ledger_receipt=ledger_receipt,
    )
    committed = _audit_task09_v3_committed_state(
        output, size=size, seed=seed, registry_sha256=registration,
    )
    v3_report = {
        "状态": "任09V3正式会话已提交；仍须按实际模型数形成五seed曲线",
        "运行身份": {"序列": name, "HF功率数": size, "seed": seed},
        "V3事前登记SHA256": registration,
        "V3源码tar收据": archive_receipt,
        "ROOT录0078收据": ledger_receipt,
        "提交后事务与最低门禁": committed,
        "校正会话边界补交收据": boundary_receipt,
        "旧固定TEST温度读取": False,
        "本单模型可称五seed曲线": False,
        "本单模型可称能源完成": False,
    }
    destination = output / "任09V3阶段门禁报告.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(v3_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(destination)
    latest_payload = torch.load(output / "阶段_最近.pt", map_location="cpu", weights_only=False)
    session_receipt = write_task09_v3_session_receipt(
        output, identity=identity, registry_sha256=registration,
        archive_sha256=v3_source_archive_sha256,
        ledger_row_sha256=ledger_receipt["录0078行SHA256"],
        start_global_epoch=start_global_epoch,
        requested_epochs=session_epoch_limit, payload=latest_payload,
    )
    return {**report, "V3通用入口门禁": v3_report, "V3会话收据": session_receipt}


def run_task09_formal_v3(
    *, name: str, size: int, seed: int, output_directory: str | Path,
    v3_registry_path: str | Path, v3_registry_sha256: str,
    v3_source_archive: str | Path, v3_source_archive_sha256: str,
    original_registry_path: str | Path, original_registry_sha256: str,
    root_ledger_entry: str, session_epoch_limit: int,
    resume_checkpoint: str | Path | None, device_name: str = "cuda",
) -> dict[str, Any]:
    """Hold one OS claim from preflight through the final immutable receipt."""
    identity = validate_task09_v3_identity(name, size, seed)
    with task09_v3_run_claim(identity) as claim:
        result = _run_task09_formal_v3_claimed(
            name=name, size=size, seed=seed, output_directory=output_directory,
            v3_registry_path=v3_registry_path,
            v3_registry_sha256=v3_registry_sha256,
            v3_source_archive=v3_source_archive,
            v3_source_archive_sha256=v3_source_archive_sha256,
            original_registry_path=original_registry_path,
            original_registry_sha256=original_registry_sha256,
            root_ledger_entry=root_ledger_entry,
            session_epoch_limit=session_epoch_limit,
            resume_checkpoint=resume_checkpoint, device_name=device_name,
        )
    return {**result, "V3跨进程排他锁": claim}
