"""Registered synthetic one-dimensional SiC/Cu plate benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.eval.two_layer_fvm import TwoLayerFvmResult


REGISTRATION = (PROJECT_ROOT / "研究记录/任务10_独立双层场基准/"
                "正式人为多热流入场前登记.yaml")


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _require_project_file(path: Path) -> Path:
    resolved = path.resolve()
    if PROJECT_ROOT != resolved and PROJECT_ROOT not in resolved.parents:
        raise ValueError("任10读写对象必须位于项目目录内")
    return resolved


def load_benchmark_registration(path: str | Path = REGISTRATION) -> dict:
    setup = load_yaml(_path(path))
    if setup.get("schema_version") != 1 or not setup.get("stage", "").startswith("任-10乙冻结后"):
        raise ValueError("任10必须使用独立乙冻结后人为板登记")
    for field, source in (
        ("original_numerical_control_sha256", setup["original_numerical_control"]),
        ("original_solver_sha256", "src/sic_cu/eval/two_layer_fvm.py"),
        ("original_export_sha256", "scripts/22_check_two_layer_fvm_controls.py"),
        ("materials_sha256", setup["materials_source"]),
    ):
        if sha256_file(_path(source)) != setup[field]:
            raise ValueError(f"任10前登记SHA与冻结源不符：{field}")
    old = load_yaml(setup["original_numerical_control"])
    for key in ("silicon_carbide_thickness_m", "copper_thickness_m", "cross_section_area_m2"):
        if setup["geometry"][key] != old["geometry"][key]:
            raise ValueError(f"板几何与原纯数值控制不符：{key}")
    for key in ("initial_temperature_k", "bottom_fixed_temperature_k",
                "benchmark_contact_resistance_m2_k_w", "coarse_mismatch_contact_multiplier"):
        if setup["thermal_conditions"][key] != old["thermal_conditions"][key]:
            raise ValueError(f"板热条件与原纯数值控制不符：{key}")
    depths = setup["observations"]["allowed_depths_from_top_m"]
    if depths != [0.0, *old["geometry"]["virtual_probe_depths_from_top_m"]]:
        raise ValueError("HF只可开放顶部与已登记13/16mm虚拟探针")
    sets = [setup["flux_splits_w_m2"][role] for role in ("train", "validation", "hidden_test")]
    powers = [int(power) for group in sets for power in group]
    if (not all(sets) or len(powers) != len(set(powers))
            or any(power <= 0 or power != raw for group in sets for raw, power in
                   ((raw, int(raw)) for raw in group))):
        raise ValueError("各人为面热通量折须非空、无交叉且为正整数")
    return setup


def solve_registered_plate(
    registration_path: str | Path, flux_w_m2: int, *, sic_cells: int, cu_cells: int,
    dt_s: float, end_s: float, contact_multiplier: float = 1.0,
) -> TwoLayerFvmResult:
    setup = load_benchmark_registration(registration_path)
    allowed = {int(value) for group in setup["flux_splits_w_m2"].values() for value in group}
    if flux_w_m2 not in allowed:
        raise ValueError("该人为顶面热通量不在事前registered数值折")
    mismatch = float(setup["thermal_conditions"]["coarse_mismatch_contact_multiplier"])
    if contact_multiplier not in (1.0, mismatch):
        raise ValueError("仅允许已登记同物理或人为contact接触热阻0.7倍率")
    if min(sic_cells, cu_cells) < 2 or dt_s <= 0 or end_s <= 0:
        raise ValueError("板两材料至少各2单元且时间步及终时刻必须为正")
    steps = round(end_s / dt_s)
    if not np.isclose(steps * dt_s, end_s, rtol=0, atol=1e-12):
        raise ValueError("终时刻必须属于事前数值时间网格")
    geometry, boundary = setup["geometry"], setup["thermal_conditions"]
    materials = load_yaml(setup["materials_source"])
    if materials.get("verified") is not True:
        raise ValueError("任10仅开放已核实材料常物性")
    widths = np.concatenate((
        np.full(sic_cells, float(geometry["silicon_carbide_thickness_m"]) / sic_cells),
        np.full(cu_cells, float(geometry["copper_thickness_m"]) / cu_cells),
    ))
    sic, cu = materials["silicon_carbide"], materials["copper"]
    for material in (sic, cu):
        if material["conductivity_w_m_k"]["kind"] != "constant" or material["heat_capacity_j_kg_k"]["kind"] != "constant":
            raise ValueError("任10FVM一维板只实现事前已锁的常物性")
    conductivity = np.concatenate((
        np.full(sic_cells, float(sic["conductivity_w_m_k"]["value"])),
        np.full(cu_cells, float(cu["conductivity_w_m_k"]["value"])),
    ))
    thermal_capacity = np.concatenate((
        np.full(sic_cells, float(sic["density_kg_m3"]) * float(sic["heat_capacity_j_kg_k"]["value"])),
        np.full(cu_cells, float(cu["density_kg_m3"]) * float(cu["heat_capacity_j_kg_k"]["value"])),
    )) * widths
    contact = float(boundary["benchmark_contact_resistance_m2_k_w"]) * contact_multiplier
    if min(widths.min(), conductivity.min(), thermal_capacity.min(), contact) < 0:
        raise ValueError("单元物性与人为接触热阻必须为非负且有效")
    face_resistance = widths[:-1] / (2 * conductivity[:-1]) + widths[1:] / (2 * conductivity[1:])
    face_resistance[sic_cells - 1] += contact
    face_conductance = 1.0 / face_resistance
    bottom_conductance = 2 * conductivity[-1] / widths[-1]
    diagonal = thermal_capacity / dt_s
    diagonal[:-1] += face_conductance
    diagonal[1:] += face_conductance
    diagonal[-1] += bottom_conductance
    implicit = splu(diags(
        [-face_conductance, diagonal, -face_conductance],
        offsets=[-1, 0, 1], shape=(len(widths), len(widths)), format="csc",
    ))
    bottom_fixed = float(boundary["bottom_fixed_temperature_k"])
    flux = float(flux_w_m2)
    forcing = np.zeros(len(widths))
    forcing[0] = flux
    forcing[-1] += bottom_conductance * bottom_fixed
    times = np.linspace(0.0, end_s, steps + 1)
    field = np.empty((steps + 1, len(widths)), dtype=np.float64)
    initial = float(boundary["initial_temperature_k"])
    field[0] = initial
    top = np.full(steps + 1, initial)
    bottom_flux = np.zeros(steps + 1)
    storage = np.zeros(steps + 1)
    balance = np.zeros(steps + 1)
    interface_flux = np.zeros(steps + 1)
    interface_jump = np.zeros(steps + 1)
    for n in range(1, steps + 1):
        field[n] = implicit.solve(thermal_capacity / dt_s * field[n - 1] + forcing)
        storage[n] = np.dot(thermal_capacity, field[n] - field[n - 1]) / dt_s
        bottom_flux[n] = bottom_conductance * (field[n, -1] - bottom_fixed)
        balance[n] = storage[n] + bottom_flux[n] - flux
        top[n] = field[n, 0] + flux * widths[0] / (2 * conductivity[0])
        interface_flux[n] = face_conductance[sic_cells - 1] * (
            field[n, sic_cells - 1] - field[n, sic_cells]
        )
        interface_jump[n] = (
            field[n, sic_cells - 1]
            - interface_flux[n] * widths[sic_cells - 1] / (2 * conductivity[sic_cells - 1])
            - field[n, sic_cells]
            - interface_flux[n] * widths[sic_cells] / (2 * conductivity[sic_cells])
        )
    return TwoLayerFvmResult(
        time_s=times,
        node_depth_m=np.cumsum(widths) - widths / 2,
        material_id=np.concatenate((np.ones(sic_cells, dtype=np.int8), np.zeros(cu_cells, dtype=np.int8))),
        temperature_k=field,
        top_surface_temperature_k=top,
        bottom_outward_flux_w_m2=bottom_flux,
        storage_rate_per_area_w_m2=storage,
        balance_per_area_w_m2=balance,
        interface_flux_w_m2=interface_flux,
        interface_temperature_jump_k=interface_jump,
    )


def _registered_grids(setup: dict) -> list[tuple[int, int, float]]:
    return [(
        int(pair["silicon_carbide_cells"]), int(pair["copper_cells"]),
        float(pair["time_step_s"]),
    ) for pair in setup["reference_controls"]["mesh_time_pairs"]]


def choose_reference_grid(
    registration_path: str | Path, adjacent_differences_c: list[float],
) -> tuple[int, int, float]:
    setup = load_benchmark_registration(registration_path)
    grids = _registered_grids(setup)
    if len(grids) != 6 or not 0 < len(adjacent_differences_c) <= 3:
        raise ValueError("任10参考只能采用已登记96/44至768/352三次相邻加密")
    limit = float(setup["reference_controls"]["per_flux_max_adjacent_difference_c"])
    for n, difference in enumerate(adjacent_differences_c):
        if not np.isfinite(difference) or difference < 0:
            raise ValueError("数值参考相邻网格差须有限且非负")
        if difference < limit:
            return grids[n + 3]
    raise ValueError("数值reference参考差超过预登记网格0.1℃门禁，不能封存该热流")


def _temperature_at_depth(result: TwoLayerFvmResult, depth_m: float, time_index: int,
                          interface_m: float) -> float:
    chosen = result.material_id == (1 if depth_m < interface_m else 0)
    return float(np.interp(depth_m, result.node_depth_m[chosen],
                           result.temperature_k[time_index, chosen]))


def check_registered_reference(registration_path: str | Path, flux_w_m2: int) -> dict:
    setup = load_benchmark_registration(registration_path)
    grids = _registered_grids(setup)
    end = float(setup["reference_controls"]["end_time_s"])
    depths = setup["reference_controls"]["comparison_depths_from_top_m"]
    times = setup["reference_controls"]["comparison_times_s"]
    interface = float(setup["geometry"]["silicon_carbide_thickness_m"])
    prior = solve_registered_plate(registration_path, flux_w_m2, sic_cells=grids[2][0],
                                   cu_cells=grids[2][1], dt_s=grids[2][2], end_s=end)
    differences = []
    for level in grids[3:]:
        candidate = solve_registered_plate(registration_path, flux_w_m2, sic_cells=level[0],
                                           cu_cells=level[1], dt_s=level[2], end_s=end)
        deviation = 0.0
        for when in times:
            first_index, second_index = round(when / grids[2 + len(differences)][2]), round(when / level[2])
            if (not np.isclose(prior.time_s[first_index], when, rtol=0, atol=1e-12)
                    or not np.isclose(candidate.time_s[second_index], when, rtol=0, atol=1e-12)):
                raise ValueError("参考核查时刻不在相邻两级登记网格")
            for depth in depths:
                if depth == 0.0:
                    delta = abs(prior.top_surface_temperature_k[first_index]
                                - candidate.top_surface_temperature_k[second_index])
                else:
                    delta = abs(_temperature_at_depth(prior, depth, first_index, interface)
                                - _temperature_at_depth(candidate, depth, second_index, interface))
                deviation = max(deviation, float(delta))
        differences.append(deviation)
        if deviation < float(setup["reference_controls"]["per_flux_max_adjacent_difference_c"]):
            selected = choose_reference_grid(registration_path, differences)
            return {
                "flux_w_m2": flux_w_m2, "fine_grid": selected,
                "max_adjacent_difference_c": deviation, "all_adjacent_differences_c": differences,
                "limit_c": float(setup["reference_controls"]["per_flux_max_adjacent_difference_c"]),
                "comparison_times_s": list(times), "comparison_depths_m": list(depths),
                "max_step_balance_w_m2": float(np.max(np.abs(candidate.balance_per_area_w_m2[1:]))),
            }
        prior = candidate
    choose_reference_grid(registration_path, differences)
    raise AssertionError("unreachable")


def registered_probe_snapshot(
    registration_path: str | Path, flux_w_m2: int, role: str, result: TwoLayerFvmResult,
) -> dict[str, np.ndarray]:
    setup = load_benchmark_registration(registration_path)
    if role not in ("train", "validation") or flux_w_m2 not in setup["flux_splits_w_m2"][role]:
        raise PermissionError("只有对应训练/验证折可制备三探针观察")
    end = float(setup["reference_controls"]["end_time_s"])
    if not np.isclose(result.time_s[-1], end, rtol=0, atol=1e-12):
        raise ValueError("探针源必须覆盖登记全部0..200秒")
    materials = load_yaml(setup["materials_source"])
    geometry, boundary = setup["geometry"], setup["thermal_conditions"]
    expected_top = float(boundary["bottom_fixed_temperature_k"]) + flux_w_m2 * (
        float(geometry["silicon_carbide_thickness_m"])
        / float(materials["silicon_carbide"]["conductivity_w_m_k"]["value"])
        + float(boundary["benchmark_contact_resistance_m2_k_w"])
        + float(geometry["copper_thickness_m"])
        / float(materials["copper"]["conductivity_w_m_k"]["value"])
    )
    if abs(result.top_surface_temperature_k[-1] - expected_top) >= 1e-4:
        raise ValueError("HF三探针来源温度与所标registered热流的稳态不同")
    samples = np.asarray(setup["observations"][f"{role}_times_s"], dtype=np.float64)
    depths = np.asarray(setup["observations"]["allowed_depths_from_top_m"], dtype=np.float64)
    interface = float(setup["geometry"]["silicon_carbide_thickness_m"])
    readings = np.empty((len(samples), len(depths)), dtype=np.float64)
    for row, moment in enumerate(samples):
        index = round(moment / (result.time_s[1] - result.time_s[0]))
        if index >= len(result.time_s) or not np.isclose(result.time_s[index], moment, atol=1e-12, rtol=0):
            raise ValueError("探针观察采样时刻不在已登记HF参考时间网格")
        readings[row, 0] = result.top_surface_temperature_k[index]
        for column, depth in enumerate(depths[1:], 1):
            readings[row, column] = _temperature_at_depth(result, float(depth), index, interface)
    return {
        "time_s": samples.copy(), "depths_from_top_m": depths.copy(),
        "flux_w_m2": np.array(flux_w_m2, dtype=np.int64), "temperature_k": readings,
    }


def _read_source_index(registration_path: Path, archive_root: Path) -> dict:
    index = json.loads(_require_project_file(archive_root / "探针与源场SHA清单.json")
                       .read_text(encoding="utf-8"))
    if index.get("登记SHA256") != sha256_file(registration_path):
        raise ValueError("任10源场登记SHA和观察档案不符")
    return index


def _read_verified_archive(archive_root: Path, index: dict, relative_name: str,
                           allowed_fields: set[str],
                           expected_shapes: dict[str, tuple[int, ...]] | None = None,
                           ) -> dict[str, np.ndarray]:
    path = _require_project_file(archive_root / relative_name)
    expected = index.get("归档文件_SHA256", {}).get(relative_name)
    if expected is None or sha256_file(path) != expected:
        raise ValueError("任10源场档案SHA与已锁清单不符")
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != allowed_fields:
            raise ValueError("观察档案包含未允许的完整HF字段或非法源场field字段")
        if expected_shapes:
            with ZipFile(path) as archive:
                for field, shape in expected_shapes.items():
                    with archive.open(field + ".npy") as header:
                        major, minor = np.lib.format.read_magic(header)
                        if (major, minor) == (1, 0):
                            found_shape, _, _ = np.lib.format.read_array_header_1_0(header)
                        elif (major, minor) in ((2, 0), (3, 0)):
                            found_shape, _, _ = np.lib.format.read_array_header_2_0(header)
                        else:
                            raise ValueError("非标准LF场npy格式，不可读取温度")
                    if found_shape != shape:
                        raise ValueError("LF低保真粗网格场形状与已登记源不同，不可冒充HF细网格")
        return {key: np.array(source[key], copy=True) for key in allowed_fields}


class Task10ObservationView:
    """A training or validation view; neither has a full-HF reference operation."""

    def __init__(self, registration_path: str | Path, archive_root: str | Path, mode: str):
        if mode not in ("train", "validation"):
            raise ValueError("仅可指定训练或验证探针观察视图")
        self.registration_path = _require_project_file(_path(registration_path))
        self.setup = load_benchmark_registration(self.registration_path)
        self.archive_root = _require_project_file(_path(archive_root))
        self.mode = mode

    def open_input(self, kind: str, flux_w_m2: int) -> dict[str, np.ndarray]:
        if kind in ("full_hf", "hidden_hf", "hidden_test"):
            raise PermissionError("训练/验证绝对不能打开完整full HF或hidden测试温度")
        if kind == "probe":
            valid = self.setup["flux_splits_w_m2"][self.mode]
            if flux_w_m2 not in valid:
                raise PermissionError("训练view不能读取验证validation探针；验证view不能读取训练train探针")
            name = f"HF_允许探针/{'训练' if self.mode == 'train' else '验证'}_{flux_w_m2}.npz"
            index = _read_source_index(self.registration_path, self.archive_root)
            snapshot = _read_verified_archive(self.archive_root, index, name,
                                              {"time_s", "depths_from_top_m", "flux_w_m2", "temperature_k"})
            samples = self.setup["observations"][f"{self.mode}_times_s"]
            if (snapshot["temperature_k"].shape != (len(samples), 3)
                    or snapshot["time_s"].tolist() != samples
                    or snapshot["depths_from_top_m"].tolist() != self.setup["observations"]["allowed_depths_from_top_m"]
                    or snapshot["flux_w_m2"].ndim != 0
                    or int(snapshot["flux_w_m2"]) != flux_w_m2):
                raise ValueError("HF探针档案必须仅含本折已登记时刻及三个合法深度")
            return snapshot
        if kind in ("lf_same_physics_coarse", "lf_contact_mismatch_coarse"):
            if self.mode != "train":
                raise PermissionError("验证视图仅打开三探针HF；LF低保真训练完整场另由训练视图访问")
            if flux_w_m2 not in self.setup["flux_splits_w_m2"]["train"]:
                raise PermissionError("LF测试或验证热流不参与低保真训练")
            name = f"LF_训练场/{'同物理' if kind == 'lf_same_physics_coarse' else '接触失配'}_{flux_w_m2}.npz"
            index = _read_source_index(self.registration_path, self.archive_root)
            coarse = self.setup["low_fidelity_sources"][
                "same_physics_coarse" if kind == "lf_same_physics_coarse"
                else "synthetic_contact_mismatch_coarse"
            ]
            cells = coarse["silicon_carbide_cells"] + coarse["copper_cells"]
            steps = round(self.setup["reference_controls"]["end_time_s"] / coarse["time_step_s"]) + 1
            return _read_verified_archive(self.archive_root, index, name,
                                          {"time_s", "node_depth_m", "material_id", "temperature_k",
                                           "top_surface_temperature_k", "bottom_outward_flux_w_m2",
                                           "storage_rate_per_area_w_m2", "balance_per_area_w_m2",
                                           "interface_flux_w_m2", "interface_temperature_jump_k"},
                                          {"temperature_k": (steps, cells),
                                           "time_s": (steps,), "node_depth_m": (cells,),
                                           "material_id": (cells,)})
        raise PermissionError("任10模型输入仅允许本折probe探针或对应LF训练完整场，禁止其他field")


def seal_evaluation_model(
    registration_path: str | Path, archive_root: str | Path, model_checkpoint: str | Path,
    training_log: str | Path, validation_report: str | Path, lock_file: str | Path,
) -> dict:
    registration, root = _require_project_file(_path(registration_path)), _require_project_file(_path(archive_root))
    setup = load_benchmark_registration(registration)
    checkpoint, training, validation, lock = [
        _require_project_file(_path(path)) for path in
        (model_checkpoint, training_log, validation_report, lock_file)
    ]
    if lock.exists():
        raise FileExistsError("完整场模型锁定原件不可覆盖")
    model_sha = sha256_file(checkpoint)
    train = json.loads(training.read_text(encoding="utf-8"))
    val = json.loads(validation.read_text(encoding="utf-8"))
    if not isinstance(train.get("实际优化步数"), int) or train["实际优化步数"] <= 0:
        raise ValueError("任10必须记录真实训练优化step步数才准完整场模型锁")
    if (train.get("训练折") != setup["flux_splits_w_m2"]["train"]
            or val.get("验证折") != setup["flux_splits_w_m2"]["validation"]
            or val.get("合法验证仅用三探针") is not True
            or any(record.get("模型_SHA256") != model_sha
                   or record.get("登记SHA256") != sha256_file(registration)
                   for record in (train, val))):
        raise ValueError("完整场模型锁须同一权重SHA、登记折和独立合法HF探针验证原件")
    index = _read_source_index(registration, root)
    entry = {
        "登记SHA256": sha256_file(registration),
        "源场SHA清单SHA256": sha256_file(root / "探针与源场SHA清单.json"),
        "模型文件": str(checkpoint), "模型_SHA256": model_sha,
        "训练日志文件": str(training), "训练日志_SHA256": sha256_file(training),
        "合法探针验证文件": str(validation), "合法探针验证_SHA256": sha256_file(validation),
        "训练折": train["训练折"], "验证折": val["验证折"],
        "实际优化步数": train["实际优化步数"],
        "完整HF仅后验评价": True,
    }
    if not index.get("归档文件_SHA256"):
        raise ValueError("探针源场尚未封存逐档SHA")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return entry


class Task10HiddenEvaluator:
    """The full reference is released only for an already sealed model."""

    def __init__(self, registration_path: str | Path, archive_root: str | Path,
                 lock_file: str | Path):
        self.registration_path = _require_project_file(_path(registration_path))
        self.setup = load_benchmark_registration(self.registration_path)
        self.archive_root = _require_project_file(_path(archive_root))
        self.lock_file = _require_project_file(_path(lock_file))

    def open_full_reference(self, flux_w_m2: int) -> dict[str, np.ndarray]:
        if not self.lock_file.is_file():
            raise PermissionError("完整HF必须先经合法训练/验证与模型SHA锁定lock")
        locked = json.loads(self.lock_file.read_text(encoding="utf-8"))
        required = {"模型文件", "模型_SHA256", "训练日志文件", "训练日志_SHA256",
                    "合法探针验证文件", "合法探针验证_SHA256", "登记SHA256",
                    "源场SHA清单SHA256", "完整HF仅后验评价", "实际优化步数"}
        if not isinstance(locked, dict) or not required <= set(locked):
            raise PermissionError("不完整的模型SHA锁定记录不得释放完整HF")
        if (not isinstance(locked["实际优化步数"], int)
                or locked["实际优化步数"] <= 0):
            raise PermissionError("未证明合法训练优化步的模型锁不可释放完整HF")
        for path_key, sha_key in (
            ("模型文件", "模型_SHA256"),
            ("训练日志文件", "训练日志_SHA256"),
            ("合法探针验证文件", "合法探针验证_SHA256"),
        ):
            if sha256_file(_require_project_file(_path(locked[path_key]))) != locked[sha_key]:
                raise PermissionError("模型、训练或验证SHA锁定后漂移，完整HF不得打开")
        if (locked.get("登记SHA256") != sha256_file(self.registration_path)
                or locked.get("源场SHA清单SHA256") != sha256_file(self.archive_root / "探针与源场SHA清单.json")
                or locked.get("完整HF仅后验评价") is not True):
            raise PermissionError("登记者/封存源场SHA漂移，完整HF不得打开")
        if flux_w_m2 not in {int(value) for group in self.setup["flux_splits_w_m2"].values()
                             for value in group}:
            raise ValueError("完整HF只可在事前登记数值热流中后验评价")
        index = _read_source_index(self.registration_path, self.archive_root)
        control = index.get("逐热流数值控制", {}).get(str(flux_w_m2))
        if not isinstance(control, dict):
            raise PermissionError("该人为热流尚无独立数值参考网格门禁，不可打开完整HF")
        grids = _registered_grids(self.setup)
        grid_raw = control.get("细网格")
        difference = control.get("相邻差_摄氏度")
        if (not isinstance(grid_raw, list) or tuple(grid_raw) not in grids[3:]
                or not isinstance(difference, (int, float)) or not np.isfinite(difference)
                or difference < 0
                or difference >= self.setup["reference_controls"]["per_flux_max_adjacent_difference_c"]):
            raise PermissionError("该热流参考细网格或0.1℃相邻差数值门禁未过，不可释放完整HF")
        sic, cu, dt = grid_raw
        n_time = round(float(self.setup["reference_controls"]["end_time_s"]) / dt) + 1
        n_cells = sic + cu
        return _read_verified_archive(
            self.archive_root, index, f"HF_封存完整场/{flux_w_m2}.npz",
            {"time_s", "node_depth_m", "material_id", "temperature_k",
             "top_surface_temperature_k", "bottom_outward_flux_w_m2",
             "storage_rate_per_area_w_m2", "balance_per_area_w_m2",
             "interface_flux_w_m2", "interface_temperature_jump_k"},
            {"temperature_k": (n_time, n_cells), "time_s": (n_time,),
             "node_depth_m": (n_cells,), "material_id": (n_cells,)},
        )
