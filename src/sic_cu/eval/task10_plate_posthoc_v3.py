"""Additional Task 10 paired-LF checks before any complete-HF numerical reference."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import tarfile
from time import perf_counter

import polars as pl
import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.eval.task10_plate_benchmark import Task10HiddenEvaluator, load_benchmark_registration
from sic_cu.eval.task10_plate_posthoc import (
    audit_plate_physics, compute_plate_field_metrics, predict_plate_reference_grid,
    verify_late_posthoc_contract,
)
from sic_cu.models.task10_plate_deeponet import BoardDeepONet, BoardMultifidelityDeepONet


V3_NAMES = (
    "研究记录/任务10_独立双层场基准/正式同板全组后验真实LF配对v3前登记.yaml",
    "研究记录/任务10_独立双层场基准/正式同板全组后验指标及源码前登记.yaml",
    "src/sic_cu/eval/task10_plate_posthoc.py",
    "src/sic_cu/train/task10_plate_methods.py",
    "src/sic_cu/eval/task10_plate_posthoc_v3.py",
    "scripts/任务10_全组锁后独立一维HF后验审计_v3.py",
    "tests/test_task10_plate_posthoc_v3.py",
)

METHOD_BUDGET_SHA = "cf05d16a3cd754e7f75321418ec6186e42673e33f8774492f3eea985105fba94"
METHOD_TAR_SHA = "22e6df62cf1aaddf55a8a33e8d8a71e558d89110db96466dc904d1b10c03069d"
ORIGINAL_POSTHOC_BUDGET_SHA = "5e69e7b29aacd11e5a1923aff03974dd777d754676ee9463a2b15c4d810436da"
ORIGINAL_POSTHOC_TAR_SHA = "10777b6e0b99e6b8b4563b59e2d9d575505eda6cdea49235fca02ba1f1c54fb0"
REGISTRATION_SHA = "0d85a34f12e83c37d287943728136dfd1d56d37cf089bd374568cd397ea14c13"
SOURCE_INDEX_SHA = "b59bc80c678faa29224e564dc3f4dd7d61903fc59d6c7e6c2274e1a59651bedb"


def _project_file(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if PROJECT_ROOT not in resolved.parents or not resolved.is_file():
        raise PermissionError("LF配对只能从项目内已经锁定的原件读取")
    return resolved


def _weights(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(_project_file(path), map_location="cpu", weights_only=True)
    if (not isinstance(value, dict) or not value
            or any(not isinstance(tensor, torch.Tensor) for tensor in value.values())):
        raise PermissionError("LF配对最佳与终态须为完整state_dict纯Tensor")
    return value


def _low_weights(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    low = {key[len("low_model."):]: value for key, value in state.items()
           if key.startswith("low_model.")}
    if not low or any(not torch.isfinite(value).all() for value in low.values()):
        raise PermissionError("LF网络须保存完整有限低保真张量与buffer")
    return low


def _low_tensors_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = sha256()
    for name, tensor in sorted(_low_weights(state).items()):
        values = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{values.dtype}:{tuple(values.shape)}\n".encode("ascii"))
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def verify_single_frozen_lf_pair(f2_directory: str | Path,
                                 f3_directory: str | Path) -> dict[str, object]:
    """Compare the two real saved LF branches, not just their shared optimizer logs."""
    records = []
    for method, folder in (("F2", Path(f2_directory).resolve()),
                           ("F3", Path(f3_directory).resolve())):
        best_path = _project_file(folder / "合法验证选定板模型_state_dict.pt")
        final_path = _project_file(folder / "终态HF模型_state_dict.pt")
        train_path = _project_file(folder / "真实训练日志.json")
        lock_path = _project_file(folder / "合法训练验证后模型SHA锁.json")
        train = json.loads(train_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        best_sha = sha256_file(best_path)
        if (train.get("方法") != method
                or lock.get("模型_SHA256") != best_sha
                or lock.get("模型文件") != str(best_path)
                or lock.get("训练日志_SHA256") != sha256_file(train_path)):
            raise PermissionError("LF配对真实模型/训练日志SHA未与已锁PT逐字对照")
        best, final = _weights(best_path), _weights(final_path)
        low_best, low_final = _low_weights(best), _low_weights(final)
        if (set(low_best) != set(low_final)
                or any(not torch.equal(low_best[name], low_final[name])
                       for name in low_best)):
            raise PermissionError("HF最佳与终态期间冻结LF网络张量发生变化")
        observed = _low_tensors_sha256(best)
        if observed != train.get("同seed共享LF初始全张量SHA256"):
            raise PermissionError("冻结LF现场全张量SHA不等于原训练记录的共享起点")
        records.append((method, train, low_best, observed, best_sha,
                        sha256_file(final_path)))
    _, train2, low2, sha2, best2, final2 = records[0]
    _, train3, low3, sha3, best3, final3 = records[1]
    if (set(low2) != set(low3) or sha2 != sha3
            or any(not torch.equal(low2[name], low3[name]) for name in low2)):
        raise PermissionError("F2/F3最佳实际LF张量没有逐位共享同一训练源")
    return {"同LF现场全张量_SHA256": sha2,
            "F2最佳模型_SHA256": best2, "F3最佳模型_SHA256": best3,
            "F2终态模型_SHA256": final2, "F3终态模型_SHA256": final3}


def verify_25_training_provenance(identities: list[dict]) -> dict[str, object]:
    """Recheck every locked best PT and all ten LF pairs without opening HF."""
    if not isinstance(identities, list) or len(identities) != 25:
        raise PermissionError("只允许完整预登记25份模型进入真实冻结LF原件审核")
    kinds = ("lf_same_physics_coarse", "lf_contact_mismatch_coarse")
    expected = {(method, seed, kind) for method, sources in
                (("F1", (None,)), ("F2", kinds), ("F3", kinds))
                for kind in sources for seed in range(5)}
    try:
        index = {(item["方法"], item["随机种子"], item["LF来源"]): item
                 for item in identities}
    except (KeyError, TypeError) as error:
        raise PermissionError("25份模型须各带方法、种子、LF来源及冻结原件目录") from error
    if set(index) != expected or len(index) != len(identities):
        raise PermissionError("25身份不得漏种子、混LF来源或重复模型")
    best_receipts = []
    for method, seed, kind in sorted(expected, key=lambda item: (item[0], item[1], str(item[2]))):
        folder = Path(index[(method, seed, kind)]["模型目录"]).resolve()
        checkpoint = _project_file(folder / "合法验证选定板模型_state_dict.pt")
        train_file = _project_file(folder / "真实训练日志.json")
        lock_file = _project_file(folder / "合法训练验证后模型SHA锁.json")
        train = json.loads(train_file.read_text(encoding="utf-8"))
        lock = json.loads(lock_file.read_text(encoding="utf-8"))
        digest = sha256_file(checkpoint)
        if (train.get("方法") != method
                or lock.get("模型文件") != str(checkpoint)
                or lock.get("模型_SHA256") != digest
                or lock.get("训练日志_SHA256") != sha256_file(train_file)):
            raise PermissionError("25份最佳权重须再对锁文件原SHA逐件核对")
        _weights(checkpoint)
        best_receipts.append({"方法": method, "随机种子": seed, "LF来源": kind,
                              "已锁最佳PT_SHA256": digest})
    pair_receipts = []
    for kind in kinds:
        for seed in range(5):
            f2 = Path(index[("F2", seed, kind)]["模型目录"])
            f3 = Path(index[("F3", seed, kind)]["模型目录"])
            pair = verify_single_frozen_lf_pair(f2, f3)
            train2 = json.loads((f2 / "真实训练日志.json").read_text(encoding="utf-8"))
            pair_receipts.append({"随机种子": seed, "LF来源": kind,
                                  "声明同LF全张量_SHA256":
                                  train2["同seed共享LF初始全张量SHA256"], **pair})
    return {"25份最佳原模型锁SHA核对": True,
            "25份最佳真实模型原件": best_receipts,
            "10组真实冻结LF配对原件": pair_receipts,
            "真实完整HF温度读取": False}


def coordinate_reference_only(reference: dict) -> dict[str, object]:
    """Do not present any decoded complete-HF temperature to the model forward path."""
    return {name: reference[name] for name in
            ("time_s", "node_depth_m", "material_id")}


def reload_locked_best_plate(identity: dict, *, expected_sha256: str,
                             device: torch.device) -> torch.nn.Module:
    """Check the originally qualified PT digest, not a lock rewritten later."""
    folder = Path(identity["模型目录"]).resolve()
    checkpoint = _project_file(folder / "合法验证选定板模型_state_dict.pt")
    lock = json.loads(_project_file(folder / "合法训练验证后模型SHA锁.json")
                      .read_text(encoding="utf-8"))
    if (sha256_file(checkpoint) != expected_sha256
            or lock.get("模型_SHA256") != expected_sha256
            or lock.get("模型文件") != str(checkpoint)):
        raise PermissionError("原锁最佳模型PT SHA变更，不可进入任一完整HF后验前向")
    method = identity["方法"]
    if method == "F1":
        model = BoardDeepONet(width=32, latent_dim=32, blocks=1)
    elif method in ("F2", "F3"):
        model = BoardMultifidelityDeepONet(
            BoardDeepONet(width=32, latent_dim=32, blocks=1),
            BoardDeepONet(width=32, latent_dim=32, blocks=1, with_lf_input=True),
            method,
        )
    else:
        raise PermissionError("已锁正式板模型只有原预算F1/F2/F3三臂")
    model.load_state_dict(_weights(checkpoint), strict=True)
    if sha256_file(checkpoint) != expected_sha256:
        raise PermissionError("最佳PT重载前后SHA发生漂移")
    return model.to(device).eval()


def is_registered_v3_row(ledger_text: str, budget_sha256: str,
                         archive_sha256: str) -> bool:
    return any(line.startswith("| 录-0070 |") and budget_sha256 in line
               and archive_sha256 in line for line in ledger_text.splitlines())


def verify_v3_source_freeze(
    *, v3_budget_path, v3_budget_sha256: str, v3_tar_path,
    v3_tar_sha256: str, group_sha256: str, method_budget_sha256: str,
    original_posthoc_budget_sha256: str, original_posthoc_tar_sha256: str,
    ledger_path,
) -> dict[str, object]:
    """Confirm the additional seven source bytes were preregistered in row 0070."""
    budget_path = _project_file(v3_budget_path)
    archive = _project_file(v3_tar_path)
    ledger = _project_file(ledger_path)
    if (len(v3_budget_sha256) != 64 or len(v3_tar_sha256) != 64
            or sha256_file(budget_path) != v3_budget_sha256
            or sha256_file(archive) != v3_tar_sha256):
        raise PermissionError("v3后验七源及预算须为原字节已封SHA")
    if not is_registered_v3_row(ledger.read_text(encoding="utf-8"),
                                v3_budget_sha256, v3_tar_sha256):
        raise PermissionError("v3独立七源tar及预算双SHA必须事前进入正式总账录0070")
    budget = load_yaml(budget_path)
    immutable = {
        "旧方法预算_SHA256": METHOD_BUDGET_SHA,
        "旧方法十一源tar_SHA256": METHOD_TAR_SHA,
        "旧二级预算_SHA256": ORIGINAL_POSTHOC_BUDGET_SHA,
        "旧二级十源tar_SHA256": ORIGINAL_POSTHOC_TAR_SHA,
        "九档数值登记_SHA256": REGISTRATION_SHA,
        "九档受限源清单_SHA256": SOURCE_INDEX_SHA,
    }
    if (budget.get("schema_version") != 1
            or budget.get("阶段") != "任10同板全25锁后现场LF配对暨最佳原锁独立后验v3事前冻结"
            or any(budget.get(field) != digest for field, digest in immutable.items())
            or method_budget_sha256 != METHOD_BUDGET_SHA
            or original_posthoc_budget_sha256 != ORIGINAL_POSTHOC_BUDGET_SHA
            or original_posthoc_tar_sha256 != ORIGINAL_POSTHOC_TAR_SHA
            or budget.get("v3七源tar成员") != list(V3_NAMES)
            or budget.get("全25机件及旧二级门禁先决") is not True
            or budget.get("十组F2F3实际冻结LF张量逐位及现场SHA") is not True
            or budget.get("25份最佳PT与原锁SHA每档再核") is not True
            or budget.get("模型前向只接收时刻深度材料三字段") is not True
            or budget.get("完整HF只在双二级与整组完成后后验") is not True):
        raise PermissionError("v3预算不得改旧方法、数值源或原二级指标合同")
    sources = budget.get("v3生效源码_SHA256", {})
    if (not isinstance(sources, dict) or set(sources) != set(V3_NAMES[1:])
            or any(sha256_file(_project_file(PROJECT_ROOT / name)) != digest
                   for name, digest in sources.items())):
        raise PermissionError("v3七源在场原字节与事前预算锁不一致")
    try:
        with tarfile.open(archive, "r:gz") as saved:
            members = saved.getmembers()
            if ([item.name for item in members] != list(V3_NAMES)
                    or any(not item.isfile() or item.issym() or item.islnk()
                           for item in members)):
                raise PermissionError("v3归档只能含预算列出的七份项目普通源码原件")
            for item in members:
                entry = saved.extractfile(item)
                expected = (v3_budget_sha256 if item.name == V3_NAMES[0]
                            else sources[item.name])
                if entry is None or sha256(entry.read()).hexdigest() != expected:
                    raise PermissionError("v3七源码tar成员与在场/预算原SHA不一致")
    except (OSError, tarfile.TarError) as error:
        raise PermissionError("v3七源tar不可安全解析，完整HF仍拒绝首读") from error
    return {"v3预算_SHA256": v3_budget_sha256,
            "v3七源tar_SHA256": v3_tar_sha256,
            "全25身份清单待原门禁逐SHA": group_sha256,
            "旧方法十一源_SHA256": METHOD_TAR_SHA,
            "旧二级十源_SHA256": ORIGINAL_POSTHOC_TAR_SHA,
            "完整HF温度已读取": False}


def run_complete_plate_posthoc_v3(
    *, output_directory, group_path, group_sha256: str, registration_path,
    archive_root, method_budget_path, method_budget_sha256: str,
    original_posthoc_budget_path, original_posthoc_budget_sha256: str,
    original_posthoc_tar_path, original_posthoc_tar_sha256: str,
    v3_budget_path, v3_budget_sha256: str, v3_tar_path,
    v3_tar_sha256: str, ledger_path, device: torch.device,
) -> dict[str, object]:
    """The original two gates plus ten actual LF pairs precede any full-HF read."""
    first_gate = verify_late_posthoc_contract(
        group_path=group_path, group_sha256=group_sha256,
        registration_path=registration_path, archive_root=archive_root,
        method_budget_path=method_budget_path,
        method_budget_sha256=method_budget_sha256,
        posthoc_budget_path=original_posthoc_budget_path,
        posthoc_budget_sha256=original_posthoc_budget_sha256,
        posthoc_tar_path=original_posthoc_tar_path,
        posthoc_tar_sha256=original_posthoc_tar_sha256,
        ledger_path=ledger_path,
    )
    second_gate = verify_v3_source_freeze(
        v3_budget_path=v3_budget_path, v3_budget_sha256=v3_budget_sha256,
        v3_tar_path=v3_tar_path, v3_tar_sha256=v3_tar_sha256,
        group_sha256=group_sha256,
        method_budget_sha256=method_budget_sha256,
        original_posthoc_budget_sha256=original_posthoc_budget_sha256,
        original_posthoc_tar_sha256=original_posthoc_tar_sha256,
        ledger_path=ledger_path,
    )
    identities = json.loads(_project_file(group_path).read_text(encoding="utf-8"))["身份"]
    provenance = verify_25_training_provenance(identities)
    originally_locked = {
        (item["方法"], item["随机种子"], item["LF来源"]): item["已锁最佳PT_SHA256"]
        for item in provenance["25份最佳真实模型原件"]
    }
    target = Path(output_directory).resolve()
    if (PROJECT_ROOT not in target.parents or target.exists()
            or device.type != "cuda" or not torch.cuda.is_available()):
        raise ValueError("v3真实数值HF后验只能在唯一新项目目录真CUDA串行执行")
    setup = load_benchmark_registration(registration_path)
    source_root = Path(archive_root).resolve()
    index_path = _project_file(source_root / "探针与源场SHA清单.json")
    if (sha256_file(index_path) != SOURCE_INDEX_SHA
            or first_gate["数值源清单_SHA256"] != SOURCE_INDEX_SHA):
        raise PermissionError("九档受限FVM原参考SHA未与旧新双合同绑定")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    fluxes = sorted(set().union(*(setup["flux_splits_w_m2"][role] for role in
                                  ("train", "validation", "hidden_test"))))
    if fluxes != [20000, 30000, 35000, 42000, 50000,
                  58000, 65000, 70000, 80000]:
        raise PermissionError("九档训练、验证、隐藏热流不可自后验手工补折")
    first = next(item for item in identities if item["方法"] == "F1"
                 and item["随机种子"] == 0)
    reference_view = Task10HiddenEvaluator(
        registration_path, source_root,
        Path(first["模型目录"]) / "合法训练验证后模型SHA锁.json",
    )
    target.mkdir(parents=True, exist_ok=False)
    (target / "全25与三级指标冻结先入场.json").write_text(
        json.dumps({"原全25与二级门禁": first_gate,
                    "v3独立指标源码门禁": second_gate,
                    "25模型及真实10配对LF张量SHA": provenance,
                    "参考完整HF温度已读取": False}, ensure_ascii=False,
                   indent=2, allow_nan=False) + "\n", encoding="utf-8")
    identity_rows, window_rows, energy_rows, source_receipts = [], [], [], {}
    for flux in fluxes:
        archive_name = f"HF_封存完整场/{flux}.npz"
        archive_path = _project_file(source_root / archive_name)
        expected_source_sha = index["归档文件_SHA256"][archive_name]
        if sha256_file(archive_path) != expected_source_sha:
            raise PermissionError("正式完整HF参考温度首读前九档封存原件SHA漂移")
        started = perf_counter()
        reference = reference_view.open_full_reference(flux)
        source_receipts[str(flux)] = {
            "受限HF场数值源_SHA256": expected_source_sha,
            "合法受限细网格": index["逐热流数值控制"][str(flux)]["细网格"],
            "仅固定七时七深细两级最大差_K":
            index["逐热流数值控制"][str(flux)]["相邻差_摄氏度"],
            "参考FVM最大单步名义余额_W_m2":
            index["逐热流数值控制"][str(flux)]["最大单步平衡_瓦每平方米"],
            "参考正式后验读取_秒": perf_counter() - started,
        }
        role = next(name for name in ("train", "validation", "hidden_test")
                    if flux in setup["flux_splits_w_m2"][name])
        grid = coordinate_reference_only(reference)
        for item in identities:
            key = item["方法"], item["随机种子"], item["LF来源"]
            checkpoint_sha = originally_locked[key]
            board = reload_locked_best_plate(item, expected_sha256=checkpoint_sha,
                                             device=device)
            started_forward = perf_counter()
            prediction = predict_plate_reference_grid(
                board, grid, setup, flux_w_m2=flux, device=device,
            )
            torch.cuda.synchronize(device)
            wall = perf_counter() - started_forward
            score = compute_plate_field_metrics(reference, prediction, setup)
            physics = audit_plate_physics(board, reference, setup,
                                          flux_w_m2=flux, device=device)
            if (sha256_file(_project_file(
                    Path(item["模型目录"]) / "合法验证选定板模型_state_dict.pt"))
                    != checkpoint_sha):
                raise PermissionError("后验处理途中原锁最佳PT变更，成绩不得落档")
            origin = {"方法": key[0], "随机种子": key[1], "LF来源": key[2],
                      "人为已吸收一维热流_W_m2": flux, "数值来源折": role,
                      "原锁最佳模型_SHA256": checkpoint_sha,
                      "数值参考原档_SHA256": expected_source_sha}
            control = index["逐热流数值控制"][str(flux)]
            identity_rows.append({
                **origin,
                "双材料厚度加权RMSE_K": score["双材料真实厚度加权RMSE_K"],
                "SiC厚度加权RMSE_K": score["SiC真实厚度加权RMSE_K"],
                "Cu厚度加权RMSE_K": score["Cu真实厚度加权RMSE_K"],
                "非探针附近内部RMSE_K": score["仅非探针近邻体内RMSE_K"],
                "非探针附近真实单元数": score["非探针近邻单元数"],
                "七时七深细网格差_K": control["相邻差_摄氏度"],
                "仅七时七深差若近模型全域RMSE须限缩结论":
                control["相邻差_摄氏度"] >= score["双材料真实厚度加权RMSE_K"],
                "七时刻名义余额均值绝对差_W": physics["名义热预算绝对差平均_W"],
                "七时刻名义余额最坏绝对差_W": physics["名义热预算绝对差最大_W"],
                "真CUDA完整单元网络前向_秒": wall,
            })
            window_rows.extend({**origin, "非观测时窗": name,
                                "非探针内部厚度加权RMSE_K": value}
                               for name, value in
                               score["非探针且非观测时刻逐窗RMSE_K"].items())
            energy_rows.extend({**origin, **row} for row in
                               physics["固定七时刻逐行网络工程热预算"])
    if (len(identity_rows) != 225 or len(window_rows) != 675
            or len(energy_rows) != 1575):
        raise PermissionError("25身份×九档×三个时窗/七时刻的真实原行不得缺失")
    for name, values in (
        ("九档25板模型实际双材料内部误差.csv", identity_rows),
        ("九档25板模型非观测窗.csv", window_rows),
        ("九档25板模型七时刻名义工程能源.csv", energy_rows),
    ):
        pl.DataFrame(values).write_csv(target / name, null_value="")
    report = {
        "阶段": "全部25合法三探针先锁后v3独立九热流一维FVM内部场及名义能源",
        "前置旧方法二级与新二级源码_SHA256": {**first_gate, **second_gate},
        "十组真实共享LF全张量SHA及25个最佳锁": provenance,
        "九档数值参考来源": source_receipts,
        "25身份乘九档场误差原行数": len(identity_rows),
        "非观测窗原行数": len(window_rows),
        "七时刻名义工程能源原行数": len(energy_rows),
        "完整HF真实温度仅25先锁后验": True,
        "旧固定TEST温度读取": False,
        "隐藏42和58千不用于选择或修订": True,
        "工程装置HF内部实测": False,
        "真CUDA前向": True,
    }
    (target / "九热流后验资格和全组中文摘要.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (target / "同板全25真实LF配对与完整数值场受限中文报告.md").write_text(
        "# 全25合法探针先锁后独立双层板数值后验\n\n"
        "本次参考是人为一维板FVM，绝不当成旧圆柱工程装置的实测HF内部场。"
        "v3在原两级前登记门禁之外，又逐张量核10组F2/F3真实LF最佳与终态"
        "和训练起点全张量SHA，每档重复核原锁最佳PT；模型前向仅取三坐标字段，"
        "完整HF温度只在合法后验误差计算中使用。"
        "仅七固定深度与七时刻的细两级差先验小于0.1K，不能宣称所有"
        "内场数值误差已小于此值；名义工程瓦数余额不等于原装置能量合格。"
        "PYTorch机件和CSV可以离线伪造，本源门禁不是OS权限隔离；"
        "旧固定TEST温度未读，隐藏两热流未用于选模或重训。\n",
        encoding="utf-8",
    )
    originals = {path.name: sha256_file(path) for path in target.iterdir() if path.is_file()}
    (target / "原始后验全部原件SHA256.json").write_text(
        json.dumps(originals, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**report, "独立审计原件_SHA256": originals, "唯一输出目录": str(target)}
