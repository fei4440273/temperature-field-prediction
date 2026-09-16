"""Task 10 real-CUDA evidence gate, evaluated before any complete-HF read."""

from __future__ import annotations

import json
from hashlib import sha256
import math
from pathlib import Path
import tarfile

import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file


LEGACY_TRAIN_CLI_SHA256 = (
    "b7ed35d95e53d2211cb15c07ef1f1eefa271deefe7f68fde188b2f27e0293cd3"
)
METHOD_BUDGET_SHA256 = (
    "cf05d16a3cd754e7f75321418ec6186e42673e33f8774492f3eea985105fba94"
)
METHOD_TAR_SHA256 = (
    "22e6df62cf1aaddf55a8a33e8d8a71e558d89110db96466dc904d1b10c03069d"
)
ORIGINAL_POSTHOC_BUDGET_SHA256 = (
    "5e69e7b29aacd11e5a1923aff03974dd777d754676ee9463a2b15c4d810436da"
)
ORIGINAL_POSTHOC_TAR_SHA256 = (
    "10777b6e0b99e6b8b4563b59e2d9d575505eda6cdea49235fca02ba1f1c54fb0"
)
V3_BUDGET_SHA256 = (
    "2e4c48149cd335accc31cdb00d7b38b24611c8aee9ee2ffd7bf05981afa6e173"
)
V3_TAR_SHA256 = (
    "653c9c7bc90511fca8e0baa2d0534255d21c42b55a33e7f42a5a7279209a669e"
)
CUDA_FREEZE_NAMES = (
    "研究记录/任务10_独立双层场基准/正式同板真CUDA训练与后验资格v4前登记.yaml",
    "研究记录/任务10_独立双层场基准/正式同板F1_F2_F3重训方法预算前登记_v2.yaml",
    "研究记录/任务10_独立双层场基准/正式同板全组后验指标及源码前登记.yaml",
    "研究记录/任务10_独立双层场基准/正式同板全组后验真实LF配对v3前登记.yaml",
    "scripts/任务10_同板F1_F2_F3重训.py",
    "src/sic_cu/eval/task10_plate_group_gate.py",
    "src/sic_cu/eval/task10_plate_posthoc.py",
    "src/sic_cu/eval/task10_plate_posthoc_v3.py",
    "src/sic_cu/eval/task10_plate_cuda_gate.py",
    "scripts/任务10_同板真CUDA专属训练.py",
    "scripts/任务10_真CUDA资格后全组独立一维HF后验审计_v4.py",
    "tests/test_task10_plate_cuda_gate.py",
)
_LF_KINDS = ("lf_same_physics_coarse", "lf_contact_mismatch_coarse")


def _project_file(value: str | Path) -> Path:
    supplied = Path(value)
    path = (supplied if supplied.is_absolute() else PROJECT_ROOT / supplied).resolve()
    if path != PROJECT_ROOT and PROJECT_ROOT not in path.parents:
        raise PermissionError("真CUDA资格原件必须位于项目目录内canonical实路径")
    if not path.is_file():
        raise PermissionError("真CUDA资格缺项目内原件，禁止读取完整HF")
    return path


def _project_directory(value: str | Path) -> Path:
    supplied = Path(value)
    path = (supplied if supplied.is_absolute() else PROJECT_ROOT / supplied).resolve()
    if path != PROJECT_ROOT and PROJECT_ROOT not in path.parents:
        raise PermissionError("真CUDA模型目录必须位于项目内")
    if not path.is_dir():
        raise PermissionError("真CUDA模型目录不存在")
    return path


def _json_file(path: str | Path) -> dict:
    try:
        value = json.loads(_project_file(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PermissionError("真CUDA收据JSON不可解析") from error
    if not isinstance(value, dict):
        raise PermissionError("真CUDA收据JSON必须是对象")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PermissionError(f"{label} SHA未按原件绑定")
    return value


def _same_file_sha(path: Path, expected: object, label: str) -> None:
    if sha256_file(_project_file(path)) != _sha(expected, label):
        raise PermissionError(f"{label}原件SHA漂移或未绑定")


def _positive_finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _cuda_rng_count(path: Path, expected_sha: object, label: str) -> int:
    _same_file_sha(path, expected_sha, label)
    try:
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PermissionError(f"{label}不能以weights_only安全读取") from error
    cuda = snapshot.get("torch_cuda_all") if isinstance(snapshot, dict) else None
    if (not isinstance(cuda, list) or not cuda
            or any(not isinstance(state, torch.Tensor)
                   or state.dtype != torch.uint8 or state.numel() < 1
                   for state in cuda)):
        raise PermissionError(f"{label}缺非空全部CUDA设备随机态")
    return len(cuda)


def _identity_tuple(identity: dict) -> tuple[str, int, str | None]:
    try:
        method = identity["方法"]
        seed = identity["随机种子"]
        lf_source = identity["LF来源"]
    except (KeyError, TypeError) as error:
        raise PermissionError("真CUDA身份缺方法、种子或LF来源") from error
    if (method not in ("F1", "F2", "F3") or type(seed) is not int
            or seed not in range(5)
            or (method == "F1" and lf_source is not None)
            or (method in ("F2", "F3") and lf_source not in _LF_KINDS)):
        raise PermissionError("真CUDA身份方法、种子或LF来源非法")
    return method, seed, lf_source


def verify_cuda_identity_receipt(identity: dict) -> dict:
    """Bind one declared identity to actual CUDA, RNG and model artifacts."""
    if not isinstance(identity, dict):
        raise PermissionError("真CUDA身份必须为机读对象")
    method, seed, lf_source = _identity_tuple(identity)
    try:
        model_directory = _project_directory(identity["模型目录"])
        receipt_path = _project_file(identity["CUDA运行收据文件"])
    except (KeyError, TypeError) as error:
        raise PermissionError("真CUDA身份缺设备运行收据") from error
    if receipt_path != model_directory.parent / "真CUDA运行设备与封存原件收据.json":
        raise PermissionError("CUDA运行收据必须与其模型批次目录相邻绑定")
    _same_file_sha(receipt_path, identity.get("CUDA运行收据_SHA256"), "CUDA运行收据")
    receipt = _json_file(receipt_path)

    device = receipt.get("设备")
    training = receipt.get("训练")
    artifacts = receipt.get("模型原件")
    if (receipt.get("schema_version") != 1
            or receipt.get("正式GPU专属包装器") is not True
            or receipt.get("运行状态") != "成功"
            or receipt.get("失败目录不恢复且新批次重跑") is not True
            or not isinstance(device, dict) or not isinstance(training, dict)
            or not isinstance(artifacts, dict)):
        raise PermissionError("CUDA收据不是GPU专属成功批次或失败目录策略不符")
    count = device.get("可见CUDA设备数")
    ordinal = device.get("实际设备序号")
    capability = device.get("计算能力")
    if (device.get("类型") != "cuda" or type(count) is not int or count < 1
            or type(ordinal) is not int or not 0 <= ordinal < count
            or not isinstance(device.get("名称"), str) or not device["名称"].strip()
            or not isinstance(capability, list) or len(capability) != 2
            or any(type(part) is not int or part < 0 for part in capability)
            or type(device.get("总显存字节")) is not int
            or device["总显存字节"] <= 0
            or not isinstance(device.get("PyTorch版本"), str)
            or not device["PyTorch版本"].strip()
            or not isinstance(device.get("CUDA运行时版本"), str)
            or not device["CUDA运行时版本"].strip()):
        raise PermissionError("CUDA设备数量、运行设备或PyTorch/CUDA版本收据不完整")

    if (training.get("arm") != ("F1" if method == "F1" else "PAIR")
            or training.get("随机种子") != seed
            or training.get("LF来源") != lf_source
            or training.get("旧冻结训练CLI_SHA256") != LEGACY_TRAIN_CLI_SHA256
            or not _positive_finite(training.get("真正训练耗时_秒"))
            or not _positive_finite(training.get("CUDA峰值分配MiB"))):
        raise PermissionError("CUDA训练身份、固定CLI、耗时或真实峰值显存收据非法")

    start = model_directory.parent / "开始训练源码与受限源凭据.json"
    summary = model_directory.parent / "合法模型训练与资源运行摘要.json"
    _same_file_sha(start, training.get("开始训练凭据_SHA256"), "开始训练凭据")
    _same_file_sha(summary, training.get("运行摘要_SHA256"), "运行摘要")
    start_record, summary_record = _json_file(start), _json_file(summary)
    expected_arm = "F1" if method == "F1" else "PAIR"
    if (start_record.get("设备") != "cuda"
            or start_record.get("运行方法") != expected_arm
            or start_record.get("随机种子") != seed
            or start_record.get("单LF来源") != lf_source
            or not math.isclose(float(summary_record.get("真正训练耗时_秒", float("nan"))),
                                float(training["真正训练耗时_秒"]), rel_tol=0, abs_tol=0)
            or not math.isclose(float(summary_record.get("CUDA峰值分配MiB", float("nan"))),
                                float(training["CUDA峰值分配MiB"]), rel_tol=0, abs_tol=0)):
        raise PermissionError("CUDA开始凭据与运行摘要的设备、身份或资源字段不一致")

    artifact = artifacts.get(method)
    if not isinstance(artifact, dict):
        raise PermissionError("CUDA收据缺当前方法的模型原件绑定")
    try:
        artifact_directory = Path(artifact["模型目录"]).resolve()
    except (KeyError, TypeError) as error:
        raise PermissionError("CUDA模型原件目录缺失") from error
    if artifact_directory != model_directory:
        raise PermissionError("CUDA收据模型目录与身份目录不一致")

    files = {
        "真实训练日志_SHA256": model_directory / "真实训练日志.json",
        "合法验证_SHA256": model_directory / "合法探针验证.json",
        "单模型锁_SHA256": model_directory / "合法训练验证后模型SHA锁.json",
        "最佳模型_SHA256": model_directory / "合法验证选定板模型_state_dict.pt",
        "终态模型_SHA256": model_directory / "终态HF模型_state_dict.pt",
        "最佳HF四类随机态_SHA256": model_directory / "最佳HF四类随机态.pt",
        "终态HF四类随机态_SHA256": model_directory / "终态HF四类随机态.pt",
    }
    if method in ("F2", "F3"):
        files["共享LF四类采样随机态_SHA256"] = (
            model_directory / "共享LF四类采样随机态.pt"
        )
    for field, path in files.items():
        _same_file_sha(path, artifact.get(field), field)

    train = _json_file(files["真实训练日志_SHA256"])
    lock = _json_file(files["单模型锁_SHA256"])
    if (train.get("方法") != method or train.get("随机种子") != seed
            or train.get("LF来源") != lf_source or train.get("设备") != "cuda"
            or train.get("模型_SHA256") != artifact["最佳模型_SHA256"]
            or train.get("终态HF模型状态_SHA256") != artifact["终态模型_SHA256"]
            or train.get("最佳HF四类随机态_SHA256")
               != artifact["最佳HF四类随机态_SHA256"]
            or train.get("终态HF四类随机态_SHA256")
               != artifact["终态HF四类随机态_SHA256"]
            or lock.get("模型_SHA256") != artifact["最佳模型_SHA256"]
            or lock.get("训练日志_SHA256") != artifact["真实训练日志_SHA256"]
            or lock.get("合法探针验证_SHA256") != artifact["合法验证_SHA256"]
            or Path(lock.get("模型文件", "")).resolve()
               != files["最佳模型_SHA256"]
            or Path(lock.get("训练日志文件", "")).resolve()
               != files["真实训练日志_SHA256"]
            or Path(lock.get("合法探针验证文件", "")).resolve()
               != files["合法验证_SHA256"]):
        raise PermissionError("CUDA最佳/终态/训练日志/原锁SHA未相互绑定")

    for field in ("最佳HF四类随机态_SHA256", "终态HF四类随机态_SHA256"):
        if _cuda_rng_count(files[field], artifact[field], field) != count:
            raise PermissionError("CUDA随机态数量与可见CUDA设备数不一致")
    if method in ("F2", "F3"):
        field = "共享LF四类采样随机态_SHA256"
        if (train.get(field) != artifact[field]
                or _cuda_rng_count(files[field], artifact[field], field) != count):
            raise PermissionError("共享LF CUDA随机态原件或设备数不一致")

    return {
        "方法": method,
        "随机种子": seed,
        "LF来源": lf_source,
        "CUDA运行收据_SHA256": sha256_file(receipt_path),
        "可见CUDA设备数": count,
        "运行设备名称": device["名称"],
        "PyTorch版本": device["PyTorch版本"],
        "CUDA运行时版本": device["CUDA运行时版本"],
        "CUDA峰值分配MiB": training["CUDA峰值分配MiB"],
        "最佳模型_SHA256": artifact["最佳模型_SHA256"],
        "终态模型_SHA256": artifact["终态模型_SHA256"],
    }


def _expected_identities() -> set[tuple[str, int, str | None]]:
    return {(method, seed, kind)
            for method, kinds in (("F1", (None,)), ("F2", _LF_KINDS),
                                  ("F3", _LF_KINDS))
            for kind in kinds for seed in range(5)}


def verify_25_cuda_evidence(identities: list[dict]) -> dict:
    """Require exactly 25 real-CUDA identities and 15 immutable run receipts."""
    if not isinstance(identities, list) or len(identities) != 25:
        raise PermissionError("真CUDA资格要求完整25个模型身份")
    actual = [_identity_tuple(identity) for identity in identities]
    if len(set(actual)) != 25 or set(actual) != _expected_identities():
        raise PermissionError("真CUDA资格的25个F1/F2/F3身份缺失或重复")
    reports = {key: verify_cuda_identity_receipt(identity)
               for key, identity in zip(actual, identities)}
    device_profiles = {(report["可见CUDA设备数"], report["运行设备名称"],
                        report["PyTorch版本"], report["CUDA运行时版本"])
                       for report in reports.values()}
    if len(device_profiles) != 1:
        raise PermissionError("25个真CUDA批次的设备数、设备名或运行版本不一致")
    for kind in _LF_KINDS:
        for seed in range(5):
            left = reports[("F2", seed, kind)]
            right = reports[("F3", seed, kind)]
            if left["CUDA运行收据_SHA256"] != right["CUDA运行收据_SHA256"]:
                raise PermissionError("同seed同LF的F2/F3必须来自同一真CUDA配对运行收据")
    receipt_shas = {report["CUDA运行收据_SHA256"] for report in reports.values()}
    if len(receipt_shas) != 15:
        raise PermissionError("真CUDA全组必须由5个F1和10个F2/F3配对批次组成")
    return {
        "身份总数": 25,
        "真CUDA运行批次总数": 15,
        "可见CUDA设备数": next(iter(device_profiles))[0],
        "逐身份设备RNG资源与原件SHA绑定通过": True,
        "完整HF温度已读取": False,
        "逐身份": [reports[key] for key in actual],
    }


def is_registered_cuda_row(ledger_text: str, budget_sha256: str,
                           archive_sha256: str) -> bool:
    return any(line.startswith("| 录-0074 |") and budget_sha256 in line
               and archive_sha256 in line for line in ledger_text.splitlines())


def verify_cuda_source_freeze(
    *, cuda_budget_path: str | Path, cuda_budget_sha256: str,
    cuda_tar_path: str | Path, cuda_tar_sha256: str,
    ledger_path: str | Path,
) -> dict:
    """Verify the row-0074 source archive before CUDA training or full-HF use."""
    budget_path = _project_file(cuda_budget_path)
    archive_path = _project_file(cuda_tar_path)
    ledger = _project_file(ledger_path)
    if (sha256_file(budget_path) != _sha(cuda_budget_sha256, "CUDA资格预算")
            or sha256_file(archive_path) != _sha(cuda_tar_sha256, "CUDA资格十二源tar")):
        raise PermissionError("CUDA资格预算或十二源tar原件SHA漂移")
    if not is_registered_cuda_row(ledger.read_text(encoding="utf-8"),
                                  cuda_budget_sha256, cuda_tar_sha256):
        raise PermissionError("CUDA资格预算和十二源tar双SHA须事前进入正式总账录0074同一行")
    budget = load_yaml(budget_path)
    immutable = {
        "旧方法预算_SHA256": METHOD_BUDGET_SHA256,
        "旧方法十一源tar_SHA256": METHOD_TAR_SHA256,
        "旧二级预算_SHA256": ORIGINAL_POSTHOC_BUDGET_SHA256,
        "旧二级十源tar_SHA256": ORIGINAL_POSTHOC_TAR_SHA256,
        "旧v3预算_SHA256": V3_BUDGET_SHA256,
        "旧v3七源tar_SHA256": V3_TAR_SHA256,
        "旧冻结训练CLI_SHA256": LEGACY_TRAIN_CLI_SHA256,
    }
    required_true = (
        "真CUDA专属包装器固定旧方法且不可选择CPU",
        "逐模型训练日志设备必须为cuda",
        "逐模型全部CUDA设备RNG必须非空且数量一致",
        "逐批设备名称数量PyTorch及CUDA版本和峰值显存收据",
        "最佳终态训练日志验证及单锁原件SHA相互绑定",
        "失败目录永久留档禁止恢复且另新目录完整重跑",
        "旧0064_0066_0070原件保持不变",
        "CUDA资格先于任一完整HF温度首读",
        "正式25模型当前仍为零",
        "真实完整HF温度当前读取仍为零",
    )
    if (budget.get("schema_version") != 1
            or budget.get("阶段") != "任10真CUDA训练及全组后验资格v4事前冻结"
            or any(budget.get(field) != digest for field, digest in immutable.items())
            or any(budget.get(field) is not True for field in required_true)
            or budget.get("CUDA资格十二源tar成员") != list(CUDA_FREEZE_NAMES)):
        raise PermissionError("CUDA资格预算未完整锁定设备证据、失败策略或旧三级来源")
    sources = budget.get("CUDA资格生效源码_SHA256")
    if (not isinstance(sources, dict)
            or set(sources) != set(CUDA_FREEZE_NAMES[1:])
            or any(sha256_file(_project_file(PROJECT_ROOT / name)) != digest
                   for name, digest in sources.items())):
        raise PermissionError("CUDA资格十二源在场源码与预算逐字节SHA不符")
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if ([member.name for member in members] != list(CUDA_FREEZE_NAMES)
                    or any(not member.isfile() or member.issym() or member.islnk()
                           for member in members)):
                raise PermissionError("CUDA资格tar只能含预算列出的十二份普通文件")
            for member in members:
                extracted = archive.extractfile(member)
                expected = (cuda_budget_sha256 if member.name == CUDA_FREEZE_NAMES[0]
                            else sources[member.name])
                if extracted is None or sha256(extracted.read()).hexdigest() != expected:
                    raise PermissionError("CUDA资格tar成员与在场预算锁定SHA不一致")
    except (OSError, tarfile.TarError) as error:
        raise PermissionError("CUDA资格十二源tar不可安全读取") from error
    return {
        "CUDA资格预算_SHA256": cuda_budget_sha256,
        "CUDA资格十二源tar_SHA256": cuda_tar_sha256,
        "旧方法录0064_SHA256": METHOD_BUDGET_SHA256,
        "旧二级录0066_SHA256": ORIGINAL_POSTHOC_BUDGET_SHA256,
        "旧v3录0070_SHA256": V3_BUDGET_SHA256,
        "正式CUDA模型已在本门禁产生": False,
        "完整HF温度已读取": False,
    }
