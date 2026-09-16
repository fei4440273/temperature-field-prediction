"""Build the Task 10 25-model catalog without reading complete HF fields."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.eval.task10_plate_cuda_gate import (
    METHOD_BUDGET_SHA256,
    METHOD_TAR_SHA256,
    verify_25_cuda_evidence,
    verify_cuda_source_freeze,
)
from sic_cu.eval.task10_plate_group_gate import verify_complete_group


TASK10_ROOT = PROJECT_ROOT / "研究记录/任务10_独立双层场基准"
REGISTRATION = TASK10_ROOT / "正式人为多热流入场前登记.yaml"
SOURCE_ROOT = TASK10_ROOT / "九热流受限数值源_台账确认后_20260916T032402+0800"
SOURCE_INDEX = SOURCE_ROOT / "探针与源场SHA清单.json"
METHOD_BUDGET = TASK10_ROOT / "正式同板F1_F2_F3重训方法预算前登记_v2.yaml"
METHOD_TAR = TASK10_ROOT / "正式同板方法十一源源码冻结_v2_20260916T050000+0800.tar.gz"
CUDA_BUDGET = TASK10_ROOT / "正式同板真CUDA训练与后验资格v4前登记.yaml"
CUDA_TAR = TASK10_ROOT / "正式同板真CUDA资格十二源事前冻结.tar.gz"
MAIN_LEDGER = PROJECT_ROOT / "多保真DeepONet预测精度优化总计划与执行台账.md"
GENERATOR_SOURCES = (
    "src/sic_cu/eval/task10_plate_group_catalog.py",
    "scripts/50_prepare_task10_cuda_group_catalog.py",
)
_LF_KINDS = ("lf_same_physics_coarse", "lf_contact_mismatch_coarse")
_RECEIPT_NAME = "真CUDA运行设备与封存原件收据.json"


def _project_path(value: str | Path, *, directory: bool | None = None) -> Path:
    supplied = Path(value)
    path = (supplied if supplied.is_absolute() else PROJECT_ROOT / supplied).resolve()
    if path != PROJECT_ROOT and PROJECT_ROOT not in path.parents:
        raise PermissionError("任10全组身份生成器只接受项目内canonical路径")
    if directory is True and not path.is_dir():
        raise PermissionError("任10全组身份生成器缺批次目录")
    if directory is False and not path.is_file():
        raise PermissionError("任10全组身份生成器缺项目内原件")
    return path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PermissionError("任10真CUDA批次收据不可机读") from error
    if not isinstance(value, dict):
        raise PermissionError("任10真CUDA批次收据必须是JSON对象")
    return value


def _identity_key(item: dict) -> tuple[str, int, str | None]:
    return item["方法"], item["随机种子"], item["LF来源"]


def _identity_order(item: dict) -> tuple[int, int, int, int]:
    method, seed, lf_source = _identity_key(item)
    if method == "F1":
        return 0, 0, seed, 0
    return 1, _LF_KINDS.index(lf_source), seed, (0 if method == "F2" else 1)


def _collect_batches(batch_directories: list[str | Path]) -> tuple[list[dict], list[dict]]:
    if len(batch_directories) != 15:
        raise PermissionError("任10全组清单必须显式提供恰15个成功真CUDA批次目录")
    directories = [_project_path(value, directory=True) for value in batch_directories]
    if len(set(directories)) != 15:
        raise PermissionError("任10全组清单的15个真CUDA批次目录不得重复")

    identities: list[dict] = []
    batches: list[dict] = []
    for directory in directories:
        receipt_path = _project_path(directory / _RECEIPT_NAME, directory=False)
        receipt = _read_json(receipt_path)
        training = receipt.get("训练")
        artifacts = receipt.get("模型原件")
        if not isinstance(training, dict) or not isinstance(artifacts, dict):
            raise PermissionError("任10真CUDA批次收据缺训练身份或模型原件")
        arm = training.get("arm")
        seed = training.get("随机种子")
        lf_source = training.get("LF来源")
        if (arm == "F1" and lf_source is None):
            methods = ("F1",)
        elif arm == "PAIR" and lf_source in _LF_KINDS:
            methods = ("F2", "F3")
        else:
            raise PermissionError("任10批次必须是F1或固定LF来源的PAIR身份")
        if type(seed) is not int or seed not in range(5) or set(artifacts) != set(methods):
            raise PermissionError("任10批次种子或F1/PAIR模型原件集合非法")

        receipt_sha = sha256_file(receipt_path)
        batch_identities = []
        for method in methods:
            artifact = artifacts[method]
            if not isinstance(artifact, dict) or not isinstance(artifact.get("模型目录"), str):
                raise PermissionError("任10批次模型原件缺模型目录")
            model_directory = _project_path(artifact["模型目录"], directory=True)
            if model_directory.parent != directory:
                raise PermissionError("任10模型目录与真CUDA批次收据必须相邻")
            identity = {
                "方法": method,
                "随机种子": seed,
                "LF来源": lf_source,
                "模型目录": str(model_directory),
                "CUDA运行收据文件": str(receipt_path),
                "CUDA运行收据_SHA256": receipt_sha,
            }
            identities.append(identity)
            batch_identities.append({
                "方法": method,
                "模型目录": str(model_directory),
            })
        batches.append({
            "arm": arm,
            "随机种子": seed,
            "LF来源": lf_source,
            "CUDA运行收据文件": str(receipt_path),
            "CUDA运行收据_SHA256": receipt_sha,
            "模型身份": batch_identities,
        })

    identities.sort(key=_identity_order)
    batches.sort(key=lambda item: (0, 0, item["随机种子"])
                 if item["arm"] == "F1" else
                 (1, _LF_KINDS.index(item["LF来源"]), item["随机种子"]))
    return identities, batches


def _preflight_complete_group_without_root(payload: bytes, digest: str) -> dict:
    """Replay the old all-25 gate with an ephemeral non-authoritative ledger."""
    with tempfile.TemporaryDirectory(prefix="任10全组清单生成预检_", dir=TASK10_ROOT) as name:
        temporary = Path(name)
        group_path = temporary / "待ROOT登记全25身份清单.json"
        ledger_path = temporary / "仅生成器预检_不属于正式ROOT.md"
        group_path.write_bytes(payload)
        ledger_path.write_text(
            f"仅供生成器CPU预检；清单SHA={digest}；方法源码tarSHA={METHOD_TAR_SHA256}\n",
            encoding="utf-8",
        )
        return verify_complete_group(
            group_path,
            group_sha256=digest,
            registration_path=REGISTRATION,
            archive_root=SOURCE_ROOT,
            budget_path=METHOD_BUDGET,
            budget_sha256=METHOD_BUDGET_SHA256,
            ledger_path=ledger_path,
        )


def build_cuda_group_catalog(
    *, batch_directories: list[str | Path], output_path: str | Path,
    cuda_budget_sha256: str, cuda_tar_sha256: str,
) -> dict:
    """Validate 15 CUDA batches and emit one deterministic 25-identity catalog."""
    output = _project_path(output_path)
    if output.exists():
        raise FileExistsError("任10全组身份清单不得覆盖已有文件")
    if output.suffix.lower() != ".json":
        raise ValueError("任10全组身份清单输出必须是JSON文件")

    identities, batches = _collect_batches(batch_directories)
    freeze = verify_cuda_source_freeze(
        cuda_budget_path=CUDA_BUDGET,
        cuda_budget_sha256=cuda_budget_sha256,
        cuda_tar_path=CUDA_TAR,
        cuda_tar_sha256=cuda_tar_sha256,
        ledger_path=MAIN_LEDGER,
    )
    cuda_proof = verify_25_cuda_evidence(identities)
    reports = {_identity_key(item): item for item in cuda_proof["逐身份"]}
    for identity in identities:
        report = reports[_identity_key(identity)]
        identity["最佳模型_SHA256"] = report["最佳模型_SHA256"]
        identity["终态模型_SHA256"] = report["终态模型_SHA256"]

    source_shas = {
        name: sha256_file(_project_path(PROJECT_ROOT / name, directory=False))
        for name in GENERATOR_SOURCES
    }
    group = {
        "schema_version": 1,
        "阶段": "任10真CUDA全25身份ROOT登记前机读清单",
        "登记SHA256": sha256_file(REGISTRATION),
        "受限源清单_SHA256": sha256_file(SOURCE_INDEX),
        "方法预算文件": _relative(METHOD_BUDGET),
        "方法预算_SHA256": METHOD_BUDGET_SHA256,
        "方法源码tar文件": _relative(METHOD_TAR),
        "方法源码tar_SHA256": METHOD_TAR_SHA256,
        "CUDA资格预算文件": _relative(CUDA_BUDGET),
        "CUDA资格预算_SHA256": cuda_budget_sha256,
        "CUDA资格源码tar文件": _relative(CUDA_TAR),
        "CUDA资格源码tar_SHA256": cuda_tar_sha256,
        "清单生成器源码_SHA256": source_shas,
        "身份": identities,
        "批次": batches,
        "身份总数": 25,
        "真CUDA运行批次总数": 15,
        "完整HF温度已读取": False,
        "后验启用条件": "本JSON自身SHA另行进入正式ROOT后，v4后验仍须逐级重放录0066/0070/0074门禁",
    }
    payload = (json.dumps(group, ensure_ascii=False, indent=2, sort_keys=True,
                          allow_nan=False) + "\n").encode("utf-8")
    digest = sha256(payload).hexdigest()
    legacy_proof = _preflight_complete_group_without_root(payload, digest)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)
    if sha256_file(output) != digest:
        raise RuntimeError("任10全组身份清单落盘后SHA与预检字节不一致")
    return {
        "全组身份文件": str(output),
        "全组身份文件_SHA256": digest,
        "身份总数": cuda_proof["身份总数"],
        "真CUDA运行批次总数": cuda_proof["真CUDA运行批次总数"],
        "录0074冻结门禁": freeze,
        "旧800步全组门禁": legacy_proof,
        "完整HF温度已读取": False,
        "下一步": "将清单SHA另行登记正式ROOT；登记前v4后验严格拒绝",
    }
