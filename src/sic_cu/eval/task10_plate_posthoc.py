"""Post-training plate metrics; raw full-HF access belongs to a separate group gate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
import tarfile
from time import perf_counter

import numpy as np
import polars as pl
import torch

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.eval.task10_plate_benchmark import (
    Task10HiddenEvaluator, load_benchmark_registration,
)
from sic_cu.eval.task10_plate_group_gate import verify_complete_group
from sic_cu.models.task10_plate_deeponet import (
    BoardDeepONet, BoardMultifidelityDeepONet,
)


POSTHOC_NAMES = (
    "研究记录/任务10_独立双层场基准/正式同板全组后验指标及源码前登记.yaml",
    "研究记录/任务10_独立双层场基准/正式人为多热流入场前登记.yaml",
    "研究记录/任务10_独立双层场基准/正式同板F1_F2_F3重训方法预算前登记_v2.yaml",
    "configs/materials.yaml",
    "src/sic_cu/eval/task10_plate_benchmark.py",
    "src/sic_cu/eval/task10_plate_group_gate.py",
    "src/sic_cu/eval/task10_plate_posthoc.py",
    "src/sic_cu/models/task10_plate_deeponet.py",
    "scripts/任务10_全组锁后独立一维HF后验审计.py",
    "tests/test_task10_plate_posthoc.py",
)


def compute_plate_field_metrics(
    reference: Mapping[str, np.ndarray], prediction_k: np.ndarray,
    setup: Mapping[str, object],
) -> dict[str, object]:
    """Score actual 1D cell volumes, excluding probe neighborhoods for unseen rows."""
    times = np.asarray(reference["time_s"], dtype=np.float64)
    depths = np.asarray(reference["node_depth_m"], dtype=np.float64)
    material = np.asarray(reference["material_id"])
    true_k = np.asarray(reference["temperature_k"], dtype=np.float64)
    predicted = np.asarray(prediction_k, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 3 or not np.isclose(times[0], 0.0)
            or not np.isclose(times[-1], 200.0)
            or np.any(np.diff(times) <= 0)
            or not np.allclose(np.diff(times), times[1] - times[0], rtol=0, atol=1e-9)):
        raise ValueError("正式一维板参考时间网格必须均匀覆盖0..200秒")
    if (material.ndim != 1 or depths.shape != material.shape
            or not 2 <= int(np.sum(material == 1)) < len(material)
            or set(material.tolist()) != {0, 1}
            or np.any(material[:-1] < material[1:])):
        raise ValueError("参考单元中心两材料顺序必须为SiC在上Cu在下")
    geometry = setup["geometry"]
    sic_cells = int(np.sum(material == 1))
    cu_cells = len(material) - sic_cells
    widths = np.r_[np.full(sic_cells, float(geometry["silicon_carbide_thickness_m"])
                           / sic_cells),
                   np.full(cu_cells, float(geometry["copper_thickness_m"]) / cu_cells)]
    centers = np.cumsum(widths) - widths / 2
    if not np.allclose(depths, centers, rtol=0, atol=1e-10):
        raise ValueError("正式一维HF单元中心必须按已登记两材料真实厚度重新推导")
    if (true_k.shape != (len(times), len(depths)) or predicted.shape != true_k.shape
            or not np.isfinite(true_k).all() or not np.isfinite(predicted).all()):
        raise ValueError("一维场必须是每个参考单元和时刻的有限原值与模型预测")
    observed = setup["observations"]
    probe = np.asarray(observed["allowed_depths_from_top_m"], dtype=np.float64)
    observed_times = np.asarray(sorted(set(observed["train_times_s"])
                                       | set(observed["validation_times_s"])),
                                dtype=np.float64)
    if (probe.shape != (3,) or not np.allclose(probe, [0, .013, .016], atol=1e-12)
            or np.any(np.diff(observed_times) < 0)):
        raise ValueError("仅按事前固定0/13/16mm三位置排除合法探针近邻")
    unseen_depth = np.all(np.abs(depths[:, None] - probe) > .0005, axis=1)
    unseen_time = ~np.isin(times, observed_times)
    squared = (predicted - true_k) ** 2

    def weighted_rmse(time_mask: np.ndarray, cell_mask: np.ndarray) -> float:
        if not bool(time_mask.any()) or not bool(cell_mask.any()):
            raise ValueError("当前参考场缺已登记未观测时刻或非探针近邻单元")
        region = squared[np.ix_(time_mask, cell_mask)]
        return float(np.sqrt(np.average(region, axis=1, weights=widths[cell_mask]).mean()))

    all_times = np.ones(len(times), dtype=bool)
    sic = material == 1
    cu = material == 0
    windows = {
        "[0,30]": times <= 30,
        "(30,100]": (times > 30) & (times <= 100),
        "(100,200]": times > 100,
    }
    if observed["nonoverlap_time_windows_s"] != list(windows):
        raise ValueError("后验时窗只能使用数值前登记的三个无重叠时段")
    return {
        "SiC真实厚度加权RMSE_K": weighted_rmse(all_times, sic),
        "Cu真实厚度加权RMSE_K": weighted_rmse(all_times, cu),
        "双材料真实厚度加权RMSE_K": weighted_rmse(all_times, sic | cu),
        "仅非探针近邻体内RMSE_K": weighted_rmse(all_times, unseen_depth),
        "非探针近邻单元数": int(unseen_depth.sum()),
        "非探针且非观测时刻逐窗RMSE_K": {
            name: weighted_rmse(mask & unseen_time, unseen_depth)
            for name, mask in windows.items()
        },
    }


def audit_plate_physics(
    model: torch.nn.Module, reference: Mapping[str, np.ndarray],
    setup: Mapping[str, object], *, flux_w_m2: int, device: torch.device,
) -> dict[str, object]:
    """Evaluate actual network top/interface/bottom fluxes and 1D storage per area."""
    if not isinstance(model, torch.nn.Module) or device.type not in ("cpu", "cuda"):
        raise ValueError("同一维板后验必须是真实可求导PyTorch网络与指定设备")
    times = np.asarray(reference["time_s"], dtype=np.float64)
    depths = np.asarray(reference["node_depth_m"], dtype=np.float64)
    material = np.asarray(reference["material_id"])
    if (times.ndim != 1 or len(times) < 3 or not np.isclose(times[0], 0)
            or not np.isclose(times[-1], 200) or
            not np.allclose(np.diff(times), times[1] - times[0], atol=1e-9, rtol=0)
            or material.shape != depths.shape or set(material.tolist()) != {0, 1}
            or np.any(material[:-1] < material[1:])):
        raise ValueError("工程储热必须使用同一维板的真实0..200s双材料均匀参考网格")
    geometry = setup["geometry"]
    sic_cells = int(np.sum(material == 1))
    cu_cells = len(material) - sic_cells
    if min(sic_cells, cu_cells) < 2:
        raise ValueError("储热必须至少有两材料各两个真实参考单元")
    sic_depth = float(geometry["silicon_carbide_thickness_m"])
    total_depth = sic_depth + float(geometry["copper_thickness_m"])
    widths = np.r_[np.full(sic_cells, sic_depth / sic_cells),
                   np.full(cu_cells, (total_depth - sic_depth) / cu_cells)]
    if not np.allclose(depths, np.cumsum(widths) - widths / 2, atol=1e-10, rtol=0):
        raise ValueError("工程储热必须使用受限HF封存网格真实材料单元中心")
    area = float(geometry["cross_section_area_m2"])
    boundary = setup["thermal_conditions"]
    contact = float(boundary["benchmark_contact_resistance_m2_k_w"])
    values = load_yaml(setup["materials_source"])
    si, cu = values["silicon_carbide"], values["copper"]
    conductivity = (float(si["conductivity_w_m_k"]["value"]),
                    float(cu["conductivity_w_m_k"]["value"]))
    capacity = np.where(material == 1,
                        float(si["density_kg_m3"]) * float(si["heat_capacity_j_kg_k"]["value"]),
                        float(cu["density_kg_m3"]) * float(cu["heat_capacity_j_kg_k"]["value"]))
    sampled = setup["reference_controls"]["comparison_times_s"]
    if (sampled != [1, 2, 5, 10, 50, 100, 200] or area <= 0 or contact < 0
            or min(conductivity) <= 0 or min(capacity) <= 0):
        raise ValueError("后验工程7时刻/一维材料/面积/Rc须与前登记公共参数相同")
    dtype = next(model.parameters()).dtype
    model.eval()

    def coordinates(depth: np.ndarray, when: np.ndarray, mat: np.ndarray) -> torch.Tensor:
        data = np.column_stack((depth, when, np.full(len(depth), flux_w_m2), mat))
        return torch.as_tensor(data, dtype=dtype, device=device)

    rows = []
    dt_s = float(times[1] - times[0])
    for moment in sampled:
        if moment < dt_s or not np.any(np.isclose(times, moment, atol=1e-9, rtol=0)):
            raise ValueError("全部7个前登记热预算时刻须在受限参考HF时间网格")
        boundaries = coordinates(np.array([0.0, sic_depth, sic_depth, total_depth]),
                                 np.full(4, moment), np.array([1, 1, 0, 0]))
        boundaries.requires_grad_(True)
        with torch.enable_grad():
            temperature = model(boundaries).reshape(4)
            derivative = torch.autograd.grad(temperature.sum(), boundaries)[0][:, 0]
        if not torch.isfinite(temperature).all() or not torch.isfinite(derivative).all():
            raise ValueError("网络顶底/界面后验温度或导数非有限，不得宣称能源合格")
        values_k = temperature.detach().cpu().numpy()
        dz = derivative.detach().cpu().numpy()
        top, interface_sic, interface_cu = -conductivity[0] * dz[:3]
        interface_cu = -conductivity[1] * dz[2]
        bottom = -conductivity[1] * dz[3]
        current = coordinates(depths, np.full(len(depths), moment), material)
        previous = coordinates(depths, np.full(len(depths), moment - dt_s), material)
        with torch.no_grad():
            now = model(current).reshape(-1).detach().cpu().numpy()
            before = model(previous).reshape(-1).detach().cpu().numpy()
        storage = float(np.dot(capacity * widths, (now - before) / dt_s))
        if not np.isfinite(now).all() or not np.isfinite(before).all():
            raise ValueError("网络真实网格相邻时刻储热温度非有限")
        balance = storage + bottom - float(flux_w_m2)
        rows.append({
            "时刻_s": int(moment), "顶部网络入流_W_m2": float(top),
            "顶部已知入流差_W_m2": float(top - flux_w_m2),
            "底部网络外流_W_m2": float(bottom),
            "底部固定温边界差_K": float(values_k[3] - boundary["bottom_fixed_temperature_k"]),
            "界面SiC热流_W_m2": float(interface_sic),
            "界面Cu热流_W_m2": float(interface_cu),
            "界面热流差_W_m2": float(interface_sic - interface_cu),
            "界面温跳_K": float(values_k[1] - values_k[2]),
            "界面Rc温跳约束差_K": float(values_k[1] - values_k[2] - contact * interface_sic),
            "网络储热率_W_m2": storage,
            "已知入流热预算差_W_m2": float(balance),
            "已知入流热预算差_W": float(balance * area),
        })
    return {
        "固定七时刻逐行网络工程热预算": rows,
        "已吸收合成等效输入功率_W": float(flux_w_m2 * area),
        "名义热预算绝对差平均_W": float(np.mean([abs(row["已知入流热预算差_W"])
                                             for row in rows])),
        "名义热预算绝对差最大_W": float(max(abs(row["已知入流热预算差_W"])
                                             for row in rows)),
    }


def predict_plate_reference_grid(
    model: torch.nn.Module, reference_grid: Mapping[str, np.ndarray],
    setup: Mapping[str, object], *, flux_w_m2: int, device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """Predict the registered cell-time grid without ever receiving HF temperature."""
    roles = setup["flux_splits_w_m2"]
    allowed = {q for fold in ("train", "validation", "hidden_test") for q in roles[fold]}
    if flux_w_m2 not in allowed or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("完整网格预测仅限9档已登记折的正批量人为热流")
    times = np.asarray(reference_grid["time_s"], dtype=np.float64)
    depths = np.asarray(reference_grid["node_depth_m"], dtype=np.float64)
    material = np.asarray(reference_grid["material_id"], dtype=np.float64)
    if (times.ndim != 1 or depths.ndim != 1 or len(times) < 3 or len(depths) < 4
            or material.shape != depths.shape or set(material.tolist()) != {0.0, 1.0}
            or not np.isfinite(times).all() or not np.isfinite(depths).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("只接受已封时刻/单元中心/两材料ID生成模型坐标，不接受HF温度")
    parameter = next(model.parameters(), None)
    if parameter is None or device.type not in ("cpu", "cuda"):
        raise ValueError("完整网格仅允许已有可重载冻结网络在指定CPU/GPU进行前向")
    total = len(times) * len(depths)
    predicted = np.empty(total, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for offset in range(0, total, batch_size):
            indexes = np.arange(offset, min(total, offset + batch_size))
            moments, cells = np.divmod(indexes, len(depths))
            values = np.column_stack((depths[cells], times[moments],
                                      np.full(len(indexes), flux_w_m2), material[cells]))
            coordinates = torch.as_tensor(values, dtype=parameter.dtype, device=device)
            output = model(coordinates).reshape(-1)
            if len(output) != len(indexes) or not torch.isfinite(output).all():
                raise ValueError("冻结模型完整网格前向批次缺行或温度非有限")
            predicted[offset:offset + len(indexes)] = output.detach().cpu().numpy()
    return predicted.reshape(len(times), len(depths))


def verify_late_posthoc_contract(
    *, group_path, group_sha256: str, registration_path, archive_root,
    method_budget_path, method_budget_sha256: str, posthoc_budget_path,
    posthoc_budget_sha256: str, posthoc_tar_path, posthoc_tar_sha256: str,
    ledger_path,
) -> dict[str, object]:
    """Demand all 25 training locks before considering a separate posthoc contract."""
    group_proof = verify_complete_group(
        group_path, group_sha256=group_sha256,
        registration_path=registration_path, archive_root=archive_root,
        budget_path=method_budget_path, budget_sha256=method_budget_sha256,
        ledger_path=ledger_path,
    )
    budget_path = Path(posthoc_budget_path).resolve()
    archive_path = Path(posthoc_tar_path).resolve()
    ledger = Path(ledger_path).resolve()
    for path in (budget_path, archive_path, ledger):
        if (PROJECT_ROOT not in path.parents or not path.is_file()):
            raise PermissionError("全25后独立后验预算与源码仍未另行项目内封存")
    if (sha256_file(budget_path) != posthoc_budget_sha256
            or sha256_file(archive_path) != posthoc_tar_sha256
            or posthoc_budget_sha256 not in ledger.read_text(encoding="utf-8")
            or posthoc_tar_sha256 not in ledger.read_text(encoding="utf-8")):
        raise PermissionError("完整HF首读前独立后验预算和源码tar SHA必须逐字事前进入总账")
    budget = load_yaml(budget_path)
    method_sha = sha256_file(Path(method_budget_path).resolve())
    registration_sha = sha256_file(Path(registration_path).resolve())
    source_sha = sha256_file(Path(archive_root).resolve() / "探针与源场SHA清单.json")
    group = json.loads(Path(group_path).resolve().read_text(encoding="utf-8"))
    fixed = budget.get("后验固定合同", {})
    if (budget.get("schema_version") != 1
            or budget.get("阶段") != "任10同板全25锁后独立纯数值后验事前指标冻结"
            or budget.get("后验十源tar成员") != list(POSTHOC_NAMES)
            or budget.get("方法预算_SHA256") != method_sha
            or budget.get("数值前登记_SHA256") != registration_sha
            or budget.get("九档受限源清单_SHA256") != source_sha
            or budget.get("方法十一源tar_SHA256") != group.get("方法源码tar_SHA256")
            or budget.get("全部25机读身份必需") is not True
            or budget.get("完整HF不用于训练验证调参选早停") is not True
            or fixed.get("25种子来源身份数") != 25
            or fixed.get("固定七能源时刻_s") != [1, 2, 5, 10, 50, 100, 200]
            or fixed.get("探针附近排除半径_m") != .0005
            or fixed.get("一维板真实材料单元厚度权重") is not True
            or fixed.get("储热前差分参考时步") is not True
            or fixed.get("全九档功率_W_m2") != [20000, 30000, 35000, 42000, 50000,
                                           58000, 65000, 70000, 80000]
            or fixed.get("隐藏两档始终后验") is not True):
        raise PermissionError("全25之后二级指标预算/源身份/固定后验口径必须另封且一致")
    source_shas = budget.get("后验生效源码_SHA256", {})
    if (set(source_shas) != set(POSTHOC_NAMES[1:])
            or any(sha256_file(PROJECT_ROOT / name) != digest
                   for name, digest in source_shas.items())):
        raise PermissionError("二级指标代码/模型/核25门禁/材料/测试与预算SHA漂移")
    try:
        with tarfile.open(archive_path, "r:gz") as saved:
            members = saved.getmembers()
            if ([entry.name for entry in members] != list(POSTHOC_NAMES)
                    or any(not entry.isfile() or entry.issym() or entry.islnk()
                           for entry in members)):
                raise PermissionError("二级指标tar成员只可为前登记10个项目内普通原字节")
            for entry in members:
                contained = saved.extractfile(entry)
                expected = (posthoc_budget_sha256 if entry.name == POSTHOC_NAMES[0]
                            else source_shas[entry.name])
                if contained is None or sha256(contained.read()).hexdigest() != expected:
                    raise PermissionError("二级指标tar某成员SHA与方法前登记源码不一致")
    except (tarfile.TarError, OSError) as error:
        raise PermissionError("完整HF首读前独立后验源码tar须可安全解析") from error
    return {**group_proof, "二级后验预算_SHA256": posthoc_budget_sha256,
            "二级后验源码tar_SHA256": posthoc_tar_sha256,
            "全25和两级源码事前SHA已核": True,
            "完整HF温度已读取": False}


def run_complete_plate_posthoc(
    *, output_directory, group_path, group_sha256: str, registration_path,
    archive_root, method_budget_path, method_budget_sha256: str,
    posthoc_budget_path, posthoc_budget_sha256: str, posthoc_tar_path,
    posthoc_tar_sha256: str, ledger_path, device: torch.device,
) -> dict[str, object]:
    """Only after both immutable gates, compare 25 new boards against nine FVM fields."""
    gate = verify_late_posthoc_contract(
        group_path=group_path, group_sha256=group_sha256,
        registration_path=registration_path, archive_root=archive_root,
        method_budget_path=method_budget_path,
        method_budget_sha256=method_budget_sha256,
        posthoc_budget_path=posthoc_budget_path,
        posthoc_budget_sha256=posthoc_budget_sha256,
        posthoc_tar_path=posthoc_tar_path,
        posthoc_tar_sha256=posthoc_tar_sha256,
        ledger_path=ledger_path,
    )
    destination = Path(output_directory).resolve()
    if (PROJECT_ROOT not in destination.parents or destination.exists()
            or device.type != "cuda" or not torch.cuda.is_available()):
        raise ValueError("正式完整HF后验须唯一新项目内目录及串行真实CUDA设备")
    setup = load_benchmark_registration(registration_path)
    index_file = Path(archive_root).resolve() / "探针与源场SHA清单.json"
    index = json.loads(index_file.read_text(encoding="utf-8"))
    identities = json.loads(Path(group_path).resolve().read_text(encoding="utf-8"))["身份"]
    if len(identities) != 25 or sha256_file(index_file) != gate["数值源清单_SHA256"]:
        raise PermissionError("九档FVM参考清单和事前25锁不一致，完整场仍拒绝首读")
    first = next(entry for entry in identities if entry["方法"] == "F1"
                 and entry["随机种子"] == 0)
    reference_view = Task10HiddenEvaluator(
        registration_path, archive_root,
        Path(first["模型目录"]) / "合法训练验证后模型SHA锁.json",
    )
    fluxes = sorted({flux for role in ("train", "validation", "hidden_test")
                     for flux in setup["flux_splits_w_m2"][role]})
    if fluxes != [20000, 30000, 35000, 42000, 50000, 58000, 65000, 70000, 80000]:
        raise PermissionError("真实完整场九档功率必须完全等于二级事前预算原三折")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "两级事前入场与来源.json").write_text(json.dumps({
        "全25身份清单_SHA256": group_sha256,
        "方法预算_SHA256": method_budget_sha256,
        "二级后验预算_SHA256": posthoc_budget_sha256,
        "二级后验源码tar_SHA256": posthoc_tar_sha256,
        "正式九档参考清单_SHA256": sha256_file(index_file),
        "已打开完整HF温度": False,
        "设备": "真实串行CUDA",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    by_identity, by_window, by_energy, reference_sources = [], [], [], {}
    for flux in fluxes:
        control = index["逐热流数值控制"][str(flux)]
        reference_path = Path(archive_root).resolve() / f"HF_封存完整场/{flux}.npz"
        expected = index["归档文件_SHA256"][f"HF_封存完整场/{flux}.npz"]
        if sha256_file(reference_path) != expected:
            raise PermissionError("首读前九档HF完整数值参考原件SHA漂移")
        start_read = perf_counter()
        reference = reference_view.open_full_reference(flux)
        reference_sources[str(flux)] = {
            "参考场原档_SHA256": expected,
            "fine_grid": control["细网格"],
            "固定七深度七时刻相邻两级最大差_K": control["相邻差_摄氏度"],
            "参考FVM最大单步热预算差_W_m2": control["最大单步平衡_瓦每平方米"],
            "参考读取_秒": perf_counter() - start_read,
        }
        role = next(name for name in ("train", "validation", "hidden_test")
                    if flux in setup["flux_splits_w_m2"][name])
        for entry in identities:
            method, seed, lf_kind = entry["方法"], entry["随机种子"], entry["LF来源"]
            if method == "F1":
                board = BoardDeepONet(width=32, latent_dim=32, blocks=1)
            else:
                board = BoardMultifidelityDeepONet(
                    BoardDeepONet(width=32, latent_dim=32, blocks=1),
                    BoardDeepONet(width=32, latent_dim=32, blocks=1, with_lf_input=True),
                    method,
                )
            checkpoint = Path(entry["模型目录"]) / "合法验证选定板模型_state_dict.pt"
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            board.load_state_dict(weights, strict=True)
            board.to(device).eval()
            start_forward = perf_counter()
            predicted = predict_plate_reference_grid(board, reference, setup,
                                                     flux_w_m2=flux, device=device)
            torch.cuda.synchronize(device)
            elapsed = perf_counter() - start_forward
            scores = compute_plate_field_metrics(reference, predicted, setup)
            energy = audit_plate_physics(board, reference, setup,
                                         flux_w_m2=flux, device=device)
            base = {"方法": method, "随机种子": seed, "LF来源": lf_kind,
                    "一维合成热流_W_m2": flux, "来源折": role,
                    "已锁最佳HF模型_SHA256": sha256_file(checkpoint),
                    "参考HF场_SHA256": expected}
            by_identity.append({**base,
                                "双材料厚度加权RMSE_K": scores["双材料真实厚度加权RMSE_K"],
                                "SiC厚度加权RMSE_K": scores["SiC真实厚度加权RMSE_K"],
                                "Cu厚度加权RMSE_K": scores["Cu真实厚度加权RMSE_K"],
                                "非探针近邻内部RMSE_K": scores["仅非探针近邻体内RMSE_K"],
                                "非探针近邻真实单元数": scores["非探针近邻单元数"],
                                "均值绝对名义余额_W": energy["名义热预算绝对差平均_W"],
                                "最坏绝对名义余额_W": energy["名义热预算绝对差最大_W"],
                                "仅固定七深度七时刻参考网格差_K": control["相邻差_摄氏度"],
                                "网格误差不小于模型全域RMSE须限缩结论":
                                control["相邻差_摄氏度"] >=
                                scores["双材料真实厚度加权RMSE_K"],
                                "网络完整网格前向_秒": elapsed})
            by_window.extend({**base, "无观测时窗": window, "内部厚度RMSE_K": value}
                             for window, value in
                             scores["非探针且非观测时刻逐窗RMSE_K"].items())
            by_energy.extend({**base, **item} for item in energy["固定七时刻逐行网络工程热预算"])
    if len(by_identity) != 225 or len(by_window) != 675 or len(by_energy) != 1575:
        raise PermissionError("正式25身份×九档×三分窗或七能源时刻缺原行，不输出受限成绩")
    for filename, rows in (("九档25模型两材料与内部误差.csv", by_identity),
                           ("九档25模型非观测时窗.csv", by_window),
                           ("九档25模型七时刻网络工程能源.csv", by_energy)):
        pl.DataFrame(rows).write_csv(destination / filename, null_value="")
    report = {"阶段": "全25模型合法探针先锁后独立一维数值FVM完整HF后验；非原装置实验内场",
              "全25身份先锁与二级预算源码SHA": gate,
              "正式九档源": reference_sources,
              "内部误差原行": len(by_identity), "非观测窗口原行": len(by_window),
              "网络能源七时刻原行": len(by_energy),
              "完整HF仅后验首读": True, "真CUDA模型前向": True,
              "隐藏42/58千绝未用于早停或重训": True,
              "工程实际装置HF内部温度真值": False,
              "旧固定TEST温度读取": False}
    (destination / "九档全组资格与后验摘要.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    (destination / "同板三方法完整内场与名义能源受限报告.md").write_text(
        "# 任10同板全25已锁后独立数值后验\n\n"
        "本次九档FVM参考不是原圆柱装置HF内部实测；后验不能反向修订模型或选早停。"
        "仅固定七深度、七时刻最细相邻网格差先验小于0.1℃，不代表所有场点。"
        "各身份/热流及非观测窗完整原数、数值源SHA和七时刻每平方米/瓦数能源"
        "均由同目录CSV及摘要逐字追溯；物理名义余额不能冒称原装置热预算通过。\n",
        encoding="utf-8")
    files = {file.name: sha256_file(file) for file in destination.iterdir() if file.is_file()}
    (destination / "审计原件SHA256.json").write_text(
        json.dumps(files, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**report, "审计原件SHA256": files, "输出": str(destination)}
