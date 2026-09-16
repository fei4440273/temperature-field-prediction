"""Require all 25 predeclared board fits before any complete-HF postmortem."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.eval.task10_plate_benchmark import load_benchmark_registration
from sic_cu.models.task10_plate_deeponet import BoardDeepONet, BoardMultifidelityDeepONet


MAIN_LEDGER = PROJECT_ROOT / "多保真DeepONet预测精度优化总计划与执行台账.md"
_TRAIN = [20000, 35000, 50000, 65000, 80000]
_VALIDATION = [30000, 70000]
_LF_KINDS = ("lf_same_physics_coarse", "lf_contact_mismatch_coarse")


def _project_file(path: str | Path) -> Path:
    supplied = Path(path)
    resolved = (supplied if supplied.is_absolute() else PROJECT_ROOT / supplied).resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise ValueError("全组后验凭据必须位于项目内canonical实路径")
    if not resolved.is_file():
        raise PermissionError("完整HF首读前缺少项目内已封存凭据文件")
    return resolved


def _json_file(path: str | Path) -> dict:
    content = json.loads(_project_file(path).read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise PermissionError("全组后验JSON必须为机读对象，不能释放任何完整HF")
    return content


def _finite_positive(value: object) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def _csv_rows(path: Path, pinned_sha: str, expected_steps: int,
              step_key: str, point_key: str, max_points: int) -> list[dict]:
    if sha256_file(_project_file(path)) != pinned_sha:
        raise PermissionError("真实优化逐步CSV原件SHA缺失或漂移")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_steps:
        raise PermissionError("真实优化逐步CSV实际行数不足正式固定预算")
    for step, row in enumerate(rows, 1):
        try:
            q = int(row["训练热流_W_m2"])
            points = int(row[point_key])
            objective = float(row["实际训练目标" if step_key == "HF优化步"
                                  else "实际LF训练目标"])
            number = int(row[step_key])
        except (KeyError, TypeError, ValueError) as error:
            raise PermissionError("优化CSV必须逐行包含热流、点数、真实目标和步数") from error
        if (number != step or q != _TRAIN[(step - 1) % len(_TRAIN)]
                or (points != 42 if step_key == "HF优化步" else not 0 < points <= max_points)
                or not math.isfinite(objective) or objective < 0):
            raise PermissionError("真实优化逐步CSV热流折、消费点数或顺序不符合同板预算")
    return rows


def _expected_identities() -> set[tuple[str, int, str | None]]:
    return {(method, seed, kind) for method, kinds in
            (("F1", (None,)), ("F2", _LF_KINDS), ("F3", _LF_KINDS))
            for kind in kinds for seed in range(5)}


def _machine(folder: Path, train: dict, name: str, filename: str):
    path = folder / filename
    if not path.is_file():
        raise PermissionError("AdamW/模型/RNG机器状态原件缺失，完整HF首读被拒绝")
    sha = train.get(name + "_SHA256")
    if not isinstance(sha, str) or len(sha) != 64 or sha256_file(_project_file(path)) != sha:
        raise PermissionError("实际AdamW/模型/RNG机器原件SHA与逐模型训练日志不符")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, ValueError, TypeError, OSError) as error:
        raise PermissionError("模型/优化器/RNG机读原件须可安全重载，不能单信任CSV") from error


def _adamw_steps(state: object, parameters: list[torch.nn.Parameter],
                 expected: int, claimed_count: object) -> None:
    if not isinstance(state, dict) or len(parameters) != claimed_count:
        raise PermissionError("AdamW机读优化器可更新参数数与一维板网络不符")
    try:
        groups = state["param_groups"]
        entries = state["state"]
        ids = [key for group in groups for key in group["params"]]
    except (KeyError, TypeError) as error:
        raise PermissionError("AdamW机器原件缺参数组或状态对象") from error
    if (len(ids) != len(parameters) or len(set(ids)) != len(ids)
            or set(ids) != set(entries)
            or any(group.get("lr") != .001 or group.get("weight_decay") != .000001
                   for group in groups)):
        raise PermissionError("AdamW完整可更新参数、学习率/衰减与固定正式预算不符")
    for param_id, parameter in zip(ids, parameters):
        record = entries[param_id]
        if (not isinstance(record, dict) or not all(isinstance(record.get(key), torch.Tensor)
                                                    for key in ("step", "exp_avg", "exp_avg_sq"))
                or record["step"].numel() != 1
                or not bool(torch.isfinite(record["step"]).all())
                or int(record["step"]) != expected
                or record["exp_avg"].shape != parameter.shape
                or record["exp_avg_sq"].shape != parameter.shape
                or not bool(torch.isfinite(record["exp_avg"]).all())
                or not bool(torch.isfinite(record["exp_avg_sq"]).all())):
            raise PermissionError("HF最佳/终态或共享LF300的真实AdamW逐参数机器步数非法")


def _rng_snapshot(value: object, *, device: str, lf: bool) -> None:
    if not isinstance(value, dict):
        raise PermissionError("四类随机态必须为机读原件而非声称随机种子")
    python_state = value.get("python_random")
    numpy_global = value.get("numpy_global")
    torch_cpu = value.get("torch_cpu")
    cuda = value.get("torch_cuda_all")
    if (not isinstance(python_state, (tuple, list)) or len(python_state) != 3
            or not isinstance(numpy_global, dict)
            or len(numpy_global.get("keys", [])) != 624
            or not isinstance(torch_cpu, torch.Tensor) or torch_cpu.dtype != torch.uint8
            or torch_cpu.numel() < 1 or not isinstance(cuda, list)
            or any(not isinstance(item, torch.Tensor) or item.dtype != torch.uint8
                   or item.numel() < 1 for item in cuda)
            or (device == "cpu" and bool(cuda)) or (device == "cuda" and not cuda)):
        raise PermissionError("CPU/全部CUDA卡/NumPy/Python四类真实随机态不完整")
    if lf and (not isinstance(value.get("numpy_sampler_start"), dict)
               or not isinstance(value.get("numpy_sampler_final"), dict)):
        raise PermissionError("共享LF300随机采样Generator起点与终态缺机读原件")


def _check_fit(entry: dict, root: Path, registration_sha: str,
               source_sha: str, budget_sha: str, methods_sha: str,
               ) -> tuple[dict, dict, str | None]:
    method, seed, lf_kind = entry["方法"], entry["随机种子"], entry["LF来源"]
    folder = Path(entry["模型目录"]).resolve()
    if folder != PROJECT_ROOT and PROJECT_ROOT not in folder.parents:
        raise PermissionError("模型目录不能逃逸项目目录读取任何完整HF")
    checkpoint = _project_file(folder / "合法验证选定板模型_state_dict.pt")
    train_file = _project_file(folder / "真实训练日志.json")
    val_file = _project_file(folder / "合法探针验证.json")
    lock_file = _project_file(folder / "合法训练验证后模型SHA锁.json")
    train, val, lock = map(_json_file, (train_file, val_file, lock_file))
    checkpoint_sha = sha256_file(checkpoint)
    if (train.get("方法") != method or train.get("随机种子") != seed
            or train.get("LF来源") != lf_kind or train.get("训练折") != _TRAIN
            or train.get("实际优化步数") != 800
            or train.get("可见HF探针实际优化消费点数") != 800*42
            or train.get("HF空AdamW起步") is not True
            or train.get("完整HF内部真值进入训练") is not False
            or train.get("体内PDE实际调用") != (0 if method == "F2" else 800)
            or train.get("正式受限数值源清单_SHA256") != source_sha
            or train.get("源场清单SHA256") != source_sha
            or train.get("方法预算_SHA256") != budget_sha
            or train.get("方法源码tar_SHA256") != methods_sha
            or val.get("验证折") != _VALIDATION
            or val.get("合法验证仅用三探针") is not True
            or val.get("合法验证温度只来自三探针") is not True
            or train.get("登记SHA256") != registration_sha
            or val.get("登记SHA256") != registration_sha
            or train.get("模型_SHA256") != checkpoint_sha
            or val.get("模型_SHA256") != checkpoint_sha):
        raise PermissionError("模型训练折、源SHA、合法验证或PDE预算未逐份完整锁定")
    with torch.random.fork_rng(devices=[]):
        if method == "F1":
            network = BoardDeepONet(width=32, latent_dim=32, blocks=1)
        else:
            network = BoardMultifidelityDeepONet(
                BoardDeepONet(width=32, latent_dim=32, blocks=1),
                BoardDeepONet(width=32, latent_dim=32, blocks=1, with_lf_input=True),
                method)
    trainable = [param for param in network.parameters() if param.requires_grad]
    best_checkpoint = torch.load(checkpoint,map_location="cpu",weights_only=True)
    best_machine = _machine(folder,train,"最佳HF模型状态", "最佳HF模型_state_dict.pt")
    final_machine = _machine(folder,train,"终态HF模型状态", "终态HF模型_state_dict.pt")
    if (not isinstance(best_checkpoint, dict) or not isinstance(best_machine, dict)
            or set(best_checkpoint) != set(best_machine)
            or any(not isinstance(left, torch.Tensor) or not isinstance(best_machine[key],torch.Tensor)
                   or not torch.equal(left,best_machine[key]) for key,left in best_checkpoint.items())):
        raise PermissionError("最佳checkpoint与HF最佳模型机读态逐参数键值必须一致")
    for weights in (best_checkpoint, best_machine, final_machine):
        if (not isinstance(weights,dict)
                or any(not isinstance(tensor,torch.Tensor)
                       or not bool(torch.isfinite(tensor).all())
                       for tensor in weights.values())):
            raise PermissionError("HF最佳/终态新板模型权重键值不得含NaN或非法非Tensor")
        try:
            network.load_state_dict(weights, strict=True)
        except (RuntimeError, TypeError, ValueError) as error:
            raise PermissionError("一维板新HF模型最佳及终态必须严格state_dict可重载") from error
    if (train.get("设备") not in ("cpu", "cuda")
            or train.get("HF最佳可更新参数实测AdamW步数") != val.get("最佳步数")
            or train.get("HF终态可更新参数实测AdamW步数") != 800):
        raise PermissionError("HF机读优化器最佳/终态步数及设备须从正式CLI实际保留")
    for name, filename, count in (
        ("最佳HF_AdamW机器状态", "最佳HF_AdamW机器状态.pt", val["最佳步数"]),
        ("终态HF_AdamW机器状态", "终态HF_AdamW机器状态.pt", 800),
    ):
        _adamw_steps(_machine(folder,train,name,filename),trainable,
                     count,train.get("HF可更新参数状态数"))
    for name, filename in (("最佳HF四类随机态", "最佳HF四类随机态.pt"),
                           ("终态HF四类随机态", "终态HF四类随机态.pt")):
        _rng_snapshot(_machine(folder,train,name,filename),device=train["设备"],lf=False)
    scores = val.get("逐档探针RMSE_K", {})
    overall = val.get("两档探针RMSE_K")
    selected = val.get("最佳步数")
    if (set(scores) != {str(q) for q in _VALIDATION}
            or not all(_finite_positive(value) for value in scores.values())
            or not _finite_positive(overall)
            or not math.isclose(overall,
                                math.sqrt(sum(value*value for value in scores.values())/2),
                                abs_tol=1e-5, rel_tol=1e-5)
            or type(selected) is not int or not 0 < selected <= 800
            or selected % 20 or selected != train.get("最佳HF优化步数")):
        raise PermissionError("30/70千三探针验证分数、逐档聚合或最佳步缺失非法")
    hf_file = folder / "逐步HF实际优化原始明细.csv"
    hf_rows = _csv_rows(hf_file, train.get("HF逐步CSV_SHA256"), 800,
                        "HF优化步", "HF三探针消费点数", 42)
    sampled_scores = []
    for step, row in enumerate(hf_rows, 1):
        try:
            pde_calls = int(row["累计体内PDE计算次数"])
            measure = row["合法两档探针验证RMSE_K"]
        except KeyError as error:
            raise PermissionError("HF逐步CSV缺体内PDE及合法验证逐步字段") from error
        if (pde_calls != (0 if method == "F2" else step)
                or (step % 20 == 0) != (measure not in ("", "None"))):
            raise PermissionError("F2/F3 PDE调用、20步验证CSV或真实逐步预算不配对")
        if step % 20 == 0:
            try:
                score = float(measure)
            except (TypeError, ValueError) as error:
                raise PermissionError("每20步合法两档三探针验证CSV分数必须是数字") from error
            if not _finite_positive(score):
                raise PermissionError("合法两档三探针验证CSV不得有负数或非有限分数")
            sampled_scores.append((score, step))
    if not math.isclose(float(hf_rows[selected - 1]["合法两档探针验证RMSE_K"]),
                        overall, abs_tol=1e-7, rel_tol=1e-7):
        raise PermissionError("已选合法探针RMSE须吻合逐步真实优化CSV选定步")
    if min(sampled_scores)[0] != overall or next(step for score, step in sampled_scores
                                               if score == overall) != selected:
        raise PermissionError("完整HF前最佳步须与全部合法20步验证最低分一致")
    if lf_kind is None:
        if method != "F1" or any(key in train for key in ("共享LF逐步CSV_SHA256",
                                                       "同seed共享LF真实优化步数")):
            raise PermissionError("F1 HF-only不可偷载或预训任一LF")
        lf_csv_sha = None
    else:
        lf_file = folder / "同seed单源LF预训真实优化明细.csv"
        lf_csv_sha = train.get("共享LF逐步CSV_SHA256")
        lf_rows = _csv_rows(lf_file, lf_csv_sha, 300,
                            "LF优化步", "LF训练消费点数", 512)
        if (train.get("同seed共享LF真实优化步数") != 300
                or train.get("同seed共享LF真实监督消费点数") !=
                sum(int(row["LF训练消费点数"]) for row in lf_rows)
                or any(not isinstance(train.get(field), str)
                       or len(train[field]) != 64 for field in
                       ("同seed共享LF初始全张量SHA256", "同seed共享HF校正初始全张量SHA256"))):
            raise PermissionError("F2/F3共享LF真实优化或逐张量同起点缺可信凭据")
        if train.get("LF终态可更新参数实测AdamW步数") != 300:
            raise PermissionError("共享LF机读优化器终态必须是300个真实可更新参数step")
        with torch.random.fork_rng(devices=[]):
            lf_network = BoardDeepONet(width=32, latent_dim=32, blocks=1)
        lf_trainable = [parameter for parameter in lf_network.parameters()
                        if parameter.requires_grad]
        _adamw_steps(_machine(folder,train,"共享LF终态_AdamW机器状态",
                             "共享LF终态_AdamW机器状态.pt"), lf_trainable,300,
                     train.get("LF可更新参数状态数"))
        _rng_snapshot(_machine(folder,train,"共享LF四类采样随机态",
                               "共享LF四类采样随机态.pt"),device=train["设备"],lf=True)
    if (lock.get("登记SHA256") != registration_sha
            or lock.get("源场SHA清单SHA256") != source_sha
            or lock.get("模型_SHA256") != checkpoint_sha
            or lock.get("实际优化步数") != 800
            or lock.get("完整HF仅后验评价") is not True
            or Path(lock.get("模型文件", "")).resolve() != checkpoint
            or Path(lock.get("训练日志文件", "")).resolve() != train_file
            or Path(lock.get("合法探针验证文件", "")).resolve() != val_file
            or lock.get("训练日志_SHA256") != sha256_file(train_file)
            or lock.get("合法探针验证_SHA256") != sha256_file(val_file)):
        raise PermissionError("模型/训练/验证/登记来源单模型SHA锁并不完整")
    return train, val, lf_csv_sha


def verify_complete_group(
    group_path: str | Path, *, group_sha256: str,
    registration_path: str | Path, archive_root: str | Path,
    budget_path: str | Path, budget_sha256: str,
    ledger_path: str | Path = MAIN_LEDGER,
) -> dict:
    group_file = _project_file(group_path)
    registration = _project_file(registration_path)
    source_index = _project_file(Path(archive_root) / "探针与源场SHA清单.json")
    budget_file = _project_file(budget_path)
    if (sha256_file(group_file) != group_sha256 or sha256_file(budget_file) != budget_sha256
            or group_sha256 not in _project_file(ledger_path).read_text(encoding="utf-8")):
        raise PermissionError("全25身份清单与预算SHA须在首次完整HF评价前另行进入总台账")
    setup = load_benchmark_registration(registration)
    budget = load_yaml(budget_file)
    group = _json_file(group_file)
    registration_sha, source_sha = sha256_file(registration), sha256_file(source_index)
    methods_sha = group.get("方法源码tar_SHA256")
    methods_file = group.get("方法源码tar文件")
    if (group.get("登记SHA256") != registration_sha
            or group.get("受限源清单_SHA256") != source_sha
            or group.get("方法预算_SHA256") != budget_sha256
            or budget.get("一维数值登记_SHA256") != registration_sha
            or budget.get("受限源清单_SHA256") != source_sha
            or budget.get("完整HF后验仅全部25模型冻结后") is not True
            or budget.get("单模型SHA锁不足以提前开启完整HF后验") is not True
            or not isinstance(methods_sha, str) or len(methods_sha) != 64
            or not isinstance(methods_file, str)
            or sha256_file(_project_file(methods_file)) != methods_sha
            or methods_sha not in _project_file(ledger_path).read_text(encoding="utf-8")
            or setup["flux_splits_w_m2"]["train"] != _TRAIN
            or setup["flux_splits_w_m2"]["validation"] != _VALIDATION):
        raise PermissionError("全组训练预算/正式数值来源SHA/方法源码总台账未绑定")
    identities = group.get("身份")
    if (not isinstance(identities, list) or len(identities) != 25
            or any(not isinstance(item, dict) or
                   not {"方法", "随机种子", "LF来源", "模型目录"} <= set(item)
                   for item in identities)):
        raise PermissionError("25个预登记模型身份缺一不可提前打开任何折的完整HF")
    actual = [(item["方法"], item["随机种子"], item["LF来源"])
              for item in identities]
    if len(set(actual)) != 25 or set(actual) != _expected_identities():
        raise PermissionError("全组F1/F2/F3五seed与同LF配对身份不完整或重复")
    fits = {}
    for identity, item in zip(actual, identities):
        fits[identity] = _check_fit(item, Path(archive_root), registration_sha,
                                    source_sha, budget_sha256, methods_sha)
    for kind in _LF_KINDS:
        for seed in range(5):
            f2, _, lf2 = fits[("F2", seed, kind)]
            f3, _, lf3 = fits[("F3", seed, kind)]
            if (lf2 != lf3 or any(f2[field] != f3[field] for field in
                ("同seed共享LF初始全张量SHA256",
                 "同seed共享HF校正初始全张量SHA256",
                 "同seed共享LF真实监督消费点数",
                 "共享LF终态_AdamW机器状态_SHA256",
                 "共享LF四类采样随机态_SHA256"))):
                raise PermissionError("同LF/同seed的F2与F3预训CSV、RNG机器SHA、AdamW机器SHA及起点全张量必须相同")
    return {"身份总数": 25, "逐步CSV与机器状态结构及SHA校验通过": True,
            "只读完整HF": False,
            "身份清单_SHA256": group_sha256, "预算_SHA256": budget_sha256,
            "数值源清单_SHA256": source_sha, "模型身份": [list(identity) for identity in actual]}


class BoardCompleteGroupEvaluationGate:
    """No full-HF read until a separate frozen metrics evaluator is supplied."""

    def __init__(self, group_path: str | Path, *, group_sha256: str,
                 registration_path: str | Path, archive_root: str | Path,
                 budget_path: str | Path, budget_sha256: str,
                 ledger_path: str | Path = MAIN_LEDGER):
        self.arguments = dict(group_path=group_path, group_sha256=group_sha256,
                              registration_path=registration_path, archive_root=archive_root,
                              budget_path=budget_path, budget_sha256=budget_sha256,
                              ledger_path=ledger_path)

    def open_full_reference(self, flux_w_m2: int):
        verify_complete_group(**self.arguments)
        raise PermissionError(
            "全25锁只是必要条件；场RMSE/界面/能量指标与后验评估源码尚未另行登记冻结，"
            "当前不能用单模型锁打开任一折完整HF")
