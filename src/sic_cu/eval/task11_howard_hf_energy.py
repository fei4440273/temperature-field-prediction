"""Own-source, read-only Howard HF Watt audit; never a training/selection callback."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import re
import tarfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from sic_cu.config import PROJECT_ROOT, load_yaml
from sic_cu.data.common import sha256_file
from sic_cu.eval.energy_v5 import (
    AxisymmetricGeometry, audit_schedule_energy, _coordinates, _constant_material_values, _radiative_flux,
    axisymmetric_rectangle_quadrature, axisymmetric_horizontal_quadrature,
)
from sic_cu.eval.metrics import observation_time_window_mask
from sic_cu.eval.task11_mlp_hf_energy import verify_task11_energy_raw
from sic_cu.models.task11_howard_composite import Task11HowardComposite
from sic_cu.physics.boundary import convection_flux
from sic_cu.physics.materials import load_materials
from sic_cu.physics.resolution import load_resolved_boundary_conditions
from sic_cu.train import task11_howard_hf_formal as hf


ROOT_TOKEN = "TASK11_HOWARD_HF_ENERGY_GATE:v1"
ENERGY_REGISTRY = "研究记录/任务11_外部对照/Howard适配三网HF双状态独立能源事前登记.yaml"
ENERGY_SOURCE_TAR = "研究记录/任务11_外部对照/Howard适配三网HF双状态独立能源源码事前冻结.tar.gz"
HF_QUALIFIED_STATUS = "CPU Howard三网HF本人真实终态与当前来源审核PASS"
ROOT_IDENTITY_FIELDS = ("ENERGY_YAML_SHA256", "ENERGY_TAR_SHA256", *hf.IDENTITY_FIELDS)
SOURCE_MEMBERS = tuple(sorted(set(hf.SOURCE_MEMBERS) | {
    "src/sic_cu/eval/task11_howard_hf_energy.py",
    "scripts/56_audit_task11_howard_hf_energy.py",
    "tests/test_task11_howard_hf_energy.py",
}))
FIXED_POWERS = (55.0, 115.2, 364.3, 403.0, 630.5, 729.0)
FIXED_TIMES = (1.0, 10.0, 50.0, 100.0, 200.0)
FIXED_ORDERS = (16, 64)
TIME_WINDOWS = ("time_0_30_s", "time_30_100_s", "time_100_200_s")
ENERGY_CONTRACT = {
    "种子": [0, 1, 2, 3, 4], "模型状态": ["best", "final"], "总状态数": 10,
    "功率_瓦": list(FIXED_POWERS), "时刻_秒": list(FIXED_TIMES),
    "求积阶数": list(FIXED_ORDERS), "功率时刻点数": 30,
    "每状态原能量行数": 60, "每状态原散度行数": 60, "每状态完整分项行数": 60,
    "原模型参数dtype": "torch.float32", "PHQH缓冲dtype": "torch.float64",
    "能源只读副本参数dtype": "torch.float64", "求积及导数dtype": "torch.float64",
    "PHQH": "本人savedmeta及model_state恢复固定无温度选择PH=QH各24乘4；不生成新查询",
    "HF公式": "Kelvin=295.15+250*(Fl+Fnl);not_LF_plus_residual",
    "摄氏转换": "temperature_K-273.15；温差K与摄氏度等值",
    "时间窗_秒": ["[0,30]", "(30,100]", "(100,200]"],
    "体积域": ["SiC", "Cu外环", "Cu下层"],
    "边界域": ["SiC顶部激光与air", "Cu顶部肩部air", "Cu底部air", "Cu外壁水冷内侧1e-6米"],
    "界面域": ["SiC底与Cu下层双侧", "SiC侧与Cu外环双侧"],
    "平衡定义": "B=storage+cooling+convection+radiation-absorbed=V-J-D",
    "名义吸收归一均值筛查": 0.05, "名义吸收归一95分位筛查": 0.10,
    "不用于训练选模": True, "工程安全阈值": False,
    "独立能源审核不证明内部真值": True,
    "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
    "设备": "cuda单卡；CPU合成样例不能取得正式权限",
    "审计成本不参与训练或标准化推理排名": True,
}
ENERGY_FLOAT_FIELDS = ("power_w", "time_s", "absorbed_power_w", "storage_rate_w",
    "cooling_heat_w", "convection_heat_w", "radiation_heat_w", "balance_w",
    "relative_balance_denominator_w", "relative_balance", "outer_epsilon_m")
ENERGY_TEXT_FIELDS = ("boundary_gradient_mode", "sign_convention", "balance_definition")
DIVERGENCE_FLOAT_FIELDS = ("power_w", "time_s", "integrated_pde_residual_w", "storage_rate_w",
    "external_model_conductive_flux_w", "interface_two_sided_flux_w", "storage_plus_external_plus_interface_w",
    "divergence_identity_gap_w", "boundary_law_outward_w", "boundary_flux_residual_w",
    "engineering_balance_w", "explained_engineering_balance_w", "engineering_explanation_gap_w",
    "sic_top_model_flux_w", "copper_top_model_flux_w", "bottom_model_flux_w", "outer_model_flux_w",
    "outer_exact_model_flux_w", "outer_inner_limit_cooling_flux_w", "sic_bottom_interface_flux_w",
    "copper_under_interface_flux_w", "sic_side_interface_flux_w", "copper_outer_ring_interface_flux_w",
    "silicon_carbide_pde_residual_w", "copper_outer_ring_pde_residual_w", "copper_below_sic_pde_residual_w",
    "outer_temperature_mean_c", "outer_temperature_max_abs_deviation_c", "outer_radial_gradient_mean_k_m",
    "outer_inner_temperature_mean_c", "outer_inner_radial_gradient_mean_k_m", "outer_epsilon_m")
DIVERGENCE_TEXT_FIELDS = ("boundary_gradient_mode", "conductive_flux_sign_convention", "interface_definition")
COMPONENT_FLOAT_FIELDS = ("power_w", "time_s", "sic_storage_w", "copper_ring_storage_w", "copper_below_storage_w",
    "sic_top_convection_w", "sic_top_radiation_w", "sic_top_area_mean_temperature_c",
    "copper_shoulder_convection_w", "copper_shoulder_radiation_w", "copper_shoulder_area_mean_temperature_c",
    "copper_bottom_convection_w", "copper_bottom_radiation_w", "copper_bottom_area_mean_temperature_c",
    "sic_top_absorbed_laser_w", "copper_outer_inner_cooling_w",
    "sic_top_boundary_mismatch_w", "copper_shoulder_boundary_mismatch_w", "copper_bottom_boundary_mismatch_w",
    "copper_outer_boundary_mismatch_w", "sic_bottom_interface_w", "copper_under_interface_w",
    "sic_side_interface_w", "copper_ring_interface_w", "copper_outer_exact_node_mean_temperature_c",
    "copper_outer_inner_node_mean_temperature_c")


def source_files() -> tuple[str, ...]:
    return SOURCE_MEMBERS


def _path(value: str | Path, root: Path, *, file: bool = True) -> Path:
    if ".." in Path(value).parts:
        raise ValueError("Howard能源来源/输出禁止父路径跳转")
    return hf._path(value, root, file=file)


def _identity(identity: Mapping[str, Any]) -> dict[str, str]:
    names = ("registry", "source_tar", "lf_catalog", "hf_data_catalog")
    if (not isinstance(identity, dict) or set(identity) != {x for n in names for x in (n, n+"_sha")}
            or any(type(identity[n]) is not str or identity[n] != expected for n, expected in
                   zip(names, (hf.REGISTRY, hf.SOURCE_TAR, hf.LF_CATALOG, hf.HF_DATA_CATALOG)))
            or any(not hf._sha(identity[n+"_sha"]) for n in names)):
        raise ValueError("Howard能源只接受本人当前HF四独立固定来源，不能借旧source pack或MLP身份")
    return copy.deepcopy(identity)


def root_gate_hashes(arguments: Mapping[str, Any]) -> dict[str, str]:
    return dict(zip(ROOT_IDENTITY_FIELDS, (arguments["energy_registry_sha"], arguments["energy_source_tar_sha"],
        arguments["registry_sha"], arguments["source_tar_sha"], arguments["lf_catalog_sha"], arguments["hf_data_catalog_sha"])))


def registration_contract(*, source_hashes: Mapping[str, str], source_tar_sha: str,
    hf_source_identity: Mapping[str, Any], registered_at: str, root_before_sha: str) -> dict[str, Any]:
    """Pure builder; does not register, archive, qualify a real LF or touch ROOT."""
    identity = _identity(hf_source_identity)
    if (not isinstance(source_hashes, dict) or set(source_hashes) != set(SOURCE_MEMBERS)
            or any(not hf._sha(value) for value in source_hashes.values())
            or not hf._sha(source_tar_sha) or not hf._sha(root_before_sha)
            or type(registered_at) is not str or not registered_at.strip()):
        raise ValueError("Howard能源独立登记必须冻结完整普通source及事前ROOT/SHA")
    return {"schema_version": 1, "登记时间": registered_at, "登记前ROOT_SHA256": root_before_sha,
        "阶段": "read_only_completed_own_howard_hf_energy", "ROOT门禁标签": ROOT_TOKEN,
        "能源审核许可": True, "HF训练许可": False, "HowardHF四来源": identity,
        "源码冻结tar": ENERGY_SOURCE_TAR, "源码冻结tarSHA256": source_tar_sha,
        "源码普通成员SHA256": copy.deepcopy(source_hashes), "独立能源审核": copy.deepcopy(ENERGY_CONTRACT),
        "真实资格API": "sic_cu.train.task11_howard_hf_formal.qualify_task11_howard_hf_source",
        "重新训练或优化": False, "重新选择best": False, "修改物理导数": False,
        "旧固定TEST温度读取": False, "模拟测试功率温度读取": False,
        "科学或工程安全资格": False}


def root_active(ledger: Path, hashes: Mapping[str, str]) -> bool:
    if set(hashes) != set(ROOT_IDENTITY_FIELDS) or any(not hf._sha(v) for v in hashes.values()):
        return False
    expected = f"{ROOT_TOKEN}; status=active; " + "; ".join(f"{key}={hashes[key]}" for key in ROOT_IDENTITY_FIELDS)
    rows, numbers = [], {}
    fence = None
    html_block, html_end = False, None
    html = hf._RootHTMLGuard()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if html_block:
            try: html.feed(line+"\n")
            except (AssertionError, ValueError): return False
            if html_end is None:
                if not line.strip() and not html.hidden: html_block = False
            elif re.search(html_end, line, re.I):
                html_block = html.hidden
                if html_block: html_end = html.continuation_end()
            continue
        match = re.match(r" {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence is not None:
            if match and match[1][0] == fence[0] and len(match[1]) >= fence[1] and not match[2].strip(): fence = None
            continue
        if match and (match[1][0] != "`" or "`" not in match[2]):
            fence = (match[1][0], len(match[1]))
            continue
        raw_tag = re.match(r" {0,3}<(script|pre|style|textarea)(?:\s|>|$)", line, re.I)
        if "<!--" in line or re.match(r" {0,3}<", line):
            try: html.feed(line+"\n")
            except (AssertionError, ValueError): return False
            if raw_tag: html_end = rf"</{raw_tag[1]}\s*>"
            elif re.match(r" {0,3}<!--", line): html_end = r"-->"
            elif re.match(r" {0,3}<\?", line): html_end = r"\?>"
            elif re.match(r" {0,3}<!\[CDATA\[", line): html_end = r"\]\]>"
            elif re.match(r" {0,3}<![A-Z]", line): html_end = r">"
            elif re.match(r" {0,3}<", line): html_end = None
            else: html_end = r"-->"
            html_block = True
            if html_end is not None and re.search(html_end, line, re.I):
                html_block = html.hidden
                if html_block: html_end = html.continuation_end()
            continue
        if not line.startswith("|"): continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) >= 4 and re.fullmatch(r"录-\d{4}", fields[1]):
            number = int(fields[1][2:])
            numbers[number] = numbers.get(number, 0)+1
            if fields[2].startswith(ROOT_TOKEN): rows.append((number, fields[2]))
    return len(rows) == 1 and rows[0][0] > 107 and numbers[rows[0][0]] == 1 and rows[0][1] == expected


def _audit_source_tar(archive_file: Path, hashes: Mapping[str, str], root: Path) -> None:
    try:
        with tarfile.open(archive_file, "r:gz") as archive:
            members = archive.getmembers()
            if (len(members) != len(SOURCE_MEMBERS) or {m.name for m in members} != set(SOURCE_MEMBERS)
                    or len({m.name for m in members}) != len(members) or any(not m.isfile() for m in members)):
                raise ValueError("Howard能源源码tar必须完整普通成员；禁止link/重复/额外/缺失")
            for member in members:
                stream = archive.extractfile(member)
                if (stream is None or hashlib.sha256(stream.read()).hexdigest() != hashes[member.name]
                        or sha256_file(_path(member.name, root)) != hashes[member.name]):
                    raise ValueError("Howard能源tar、登记及当前普通源码SHA不一致")
    except (tarfile.TarError, OSError) as error:
        raise ValueError("Howard能源普通源码归档无法核验") from error


def preflight_task11_howard_energy(*, energy_registry: str | Path, energy_registry_sha: str,
    energy_source_tar: str | Path, energy_source_tar_sha: str,
    registry: str | Path, registry_sha: str, source_tar: str | Path, source_tar_sha: str,
    lf_catalog: str | Path, lf_catalog_sha: str, hf_data_catalog: str | Path, hf_data_catalog_sha: str,
    project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """Own gate/source CPU checks only; successful fixtures grant no formal permission."""
    root = Path(project_root).resolve()
    identity = _identity({"registry": str(Path(registry).relative_to(root)) if Path(registry).is_absolute() else str(registry),
        "registry_sha": registry_sha,
        "source_tar": str(Path(source_tar).relative_to(root)) if Path(source_tar).is_absolute() else str(source_tar),
        "source_tar_sha": source_tar_sha,
        "lf_catalog": str(Path(lf_catalog).relative_to(root)) if Path(lf_catalog).is_absolute() else str(lf_catalog),
        "lf_catalog_sha": lf_catalog_sha,
        "hf_data_catalog": str(Path(hf_data_catalog).relative_to(root)) if Path(hf_data_catalog).is_absolute() else str(hf_data_catalog),
        "hf_data_catalog_sha": hf_data_catalog_sha})
    arguments = {"energy_registry_sha": energy_registry_sha, "energy_source_tar_sha": energy_source_tar_sha, **identity}
    hashes = root_gate_hashes(arguments)
    files = [_path(energy_registry, root), _path(energy_source_tar, root)] + [_path(identity[n], root)
        for n in ("registry", "source_tar", "lf_catalog", "hf_data_catalog")]
    if files[:2] != [root / ENERGY_REGISTRY, root / ENERGY_SOURCE_TAR]:
        raise ValueError("Howard能源仅独立固定正式YAML和普通tar，不准借旧source替身")
    if any(not hf._sha(v) for v in hashes.values()) or any(sha256_file(p) != d for p, d in zip(files, hashes.values())):
        raise ValueError("Howard能源六来源当前原件SHA缺失或漂移")
    ledger = _path(hf.ROOT_LEDGER, root)
    if not root_active(ledger, hashes): raise ValueError("Howard能源需要独立六SHA plain ROOT活动门禁及不碰撞录号>107")
    recorded = hf._load_yaml(files[0])
    if not isinstance(recorded, dict): raise ValueError("Howard能源登记必须为严格完整schema")
    expected = registration_contract(source_hashes=recorded.get("源码普通成员SHA256"),
        source_tar_sha=energy_source_tar_sha, hf_source_identity=identity,
        registered_at=recorded.get("登记时间"), root_before_sha=recorded.get("登记前ROOT_SHA256"))
    if not hf._same(recorded, expected): raise ValueError("Howard能源冻结schema/float/bool/counter/来源/双阶合同不一致")
    _audit_source_tar(files[1], recorded["源码普通成员SHA256"], root)
    return {"项目根": str(root), "登记": recorded, "六来源原件": [str(p) for p in files],
        "ROOT六SHA": hashes, "ROOT审核SHA256": sha256_file(ledger),
        "HowardHF四来源": identity, "源码普通成员数": len(SOURCE_MEMBERS), "能源审核许可": False}


def source_unchanged(checked: Mapping[str, Any]) -> None:
    root = Path(checked["项目根"])
    if any(sha256_file(_path(p, root)) != digest for p, digest in
           zip(checked["六来源原件"], checked["ROOT六SHA"].values())):
        raise ValueError("Howard能源六事前来源当前原件SHA漂移")
    for name, digest in checked["登记"]["源码普通成员SHA256"].items():
        if sha256_file(_path(name, root)) != digest: raise ValueError("Howard能源冻结普通源码SHA漂移")
    ledger = _path(hf.ROOT_LEDGER, root)
    if sha256_file(ledger) != checked["ROOT审核SHA256"] or not root_active(ledger, checked["ROOT六SHA"]):
        raise ValueError("Howard能源ROOT审核前后原件或独立活动门禁漂移")


def task11_howard_energy_output(run: str | Path, output: str | Path, state: str, seed: int,
    *, project_root: str | Path = PROJECT_ROOT) -> Path:
    if state not in ("best", "final"): raise ValueError("Howard能源只准本人best/final，四其他state不能借能源选模")
    if type(seed) is not int or seed not in range(5): raise ValueError("Howard能源仅五个登记seed")
    root = Path(project_root).resolve()
    run_path, destination = _path(run, root, file=False), _path(output, root, file=False)
    canonical = root / hf.RUN_DIRECTORY / f"正式Howard_HF_seed{seed}"
    label = "观测最佳" if state == "best" else "训练末"
    if (run_path != canonical or destination.parent != canonical or not re.fullmatch(
            rf"独立原能源_Howard_{label}_\d{{8}}T\d{{6}}\+0800", destination.name)):
        raise ValueError("Howard能源目录必须本人同seed实际RUN_DIRECTORY的固定新状态子目录")
    if destination.exists(): raise FileExistsError("Howard独立能源原件不能覆盖")
    return destination


def task11_howard_energy_model_from_view(view: Mapping[str, Any]) -> Task11HowardComposite:
    try:
        identity, state, kwargs = view["任11HowardHF事前来源"], view["model_state"], view["model_kwargs"]
        seed = view["seed"]
        if (type(view.get("schema_version")) is not int or view["schema_version"] != 1
                or view.get("method") != hf.METHOD or type(seed) is not int or seed not in range(5)
                or type(view.get("epoch")) is not int or view["epoch"] <= 0
                or not isinstance(identity, dict) or identity.get("method") != hf.METHOD
                or type(identity.get("seed")) is not int or identity["seed"] != seed
                or view.get("HF训练许可") is not False or any(view.get(n) is not False for n in hf.ISOLATION_FLAGS)
                or not isinstance(state, dict) or not isinstance(kwargs, dict)
                or not hf._same(view.get("scales"), hf.HF_BUDGET["scales"])):
            raise ValueError("Howard能源必须本人真实三网HF savedmeta/state方法及seed身份；不能借MLP/LF")
        ph, qh = state["linear_query_points"], state["nonlinear_query_points"]
        metadata = identity["固定PHQH查询"]
        hf._validate_queries(ph, metadata)
        hf._validate_queries(qh, metadata)
        expected_kwargs = {**hf.HF_BUDGET["model_kwargs"], "scales": hf.HF_BUDGET["scales"],
            "linear_query_points": ph.tolist(), "nonlinear_query_points": qh.tolist()}
        if not hf._same(kwargs, expected_kwargs): raise ValueError("Howard能源本人saved构造参数和PHQH state不一致")
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            with torch.random.fork_rng(devices=[]), torch.device("cpu"):
                torch.default_generator.manual_seed(0)
                model = Task11HowardComposite(**copy.deepcopy(kwargs))
        finally:
            torch.set_default_dtype(previous)
        reference = model.state_dict()
        if set(state) != set(reference) or any(not isinstance(v, Tensor) or v.device.type != "cpu"
            or v.dtype != reference[n].dtype or v.shape != reference[n].shape or not torch.isfinite(v).all()
            for n, v in state.items()): raise ValueError("Howard能源全部56实参和双Float64 query必须完整有限CPU本人视图")
        model.load_state_dict(state, strict=True)
        hf._validate_model(model)
        return model.eval()
    except (KeyError, TypeError, AttributeError, RuntimeError) as error:
        raise ValueError("Howard能源本人三网savedmeta/state或完整56参数/query64工厂核验失败") from error


def load_task11_howard_energy_model(qualified: Mapping[str, Any], *, run: str | Path, state: str,
    seed: int, project_root: str | Path = PROJECT_ROOT) -> tuple[dict[str, Any], Task11HowardComposite]:
    if (state not in ("best", "final") or type(seed) is not int or seed not in range(5)
            or not isinstance(qualified, dict) or qualified.get("状态") != HF_QUALIFIED_STATUS
            or type(qualified.get("seed")) is not int or qualified["seed"] != seed
            or qualified.get("已完成") is not True or qualified.get("HF训练许可") is not False
            or not isinstance(qualified.get("事前来源"), dict)
            or qualified["事前来源"].get("method") != hf.METHOD
            or type(qualified["事前来源"].get("seed")) is not int or qualified["事前来源"]["seed"] != seed):
        raise ValueError("Howard能源只能接本人HF资格API真实终态，不接受退出0或0/10伪资格")
    root = Path(project_root).resolve()
    run_path = _path(run, root, file=False)
    if run_path != root / hf.RUN_DIRECTORY / f"正式Howard_HF_seed{seed}": raise ValueError("Howard能源只读本人seed固定PT")
    original = _path(run_path / (state+".pt"), root)
    digest = qualified["最佳HF检查点SHA256" if state == "best" else "真实末HF检查点SHA256"]
    if (not hf._sha(digest) or qualified.get("真实工件SHA256", {}).get(state+".pt") != digest
            or sha256_file(original) != digest): raise ValueError("Howard能源当前选中PT SHA漂移，禁止反序列化")
    try:
        view = torch.load(original, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError("Howard能源本人CPU模型视图无法读取") from error
    if sha256_file(original) != digest: raise ValueError("Howard能源读取PT期间当前SHA漂移")
    model = task11_howard_energy_model_from_view(view)
    if view["seed"] != seed or not hf._same(view["任11HowardHF事前来源"], qualified["事前来源"]):
        raise ValueError("Howard能源当前PT和本人完整HF资格来源不一致")
    return view, model


def prepare_task11_howard_energy_copy(model: Task11HowardComposite, view: Mapping[str, Any]) -> Task11HowardComposite:
    hf._validate_model(model)
    if not hf._same(model.state_dict(), view["model_state"]):
        raise ValueError("Howard能源CPU工厂/移动后本人原state不一致，不能改导数或换查询")
    model.requires_grad_(False).to(dtype=torch.float64).eval()
    if (any(p.dtype != torch.float64 for p in model.parameters()) or any(b.dtype != torch.float64 for b in model.buffers())
            or not hf._same(model.model_kwargs, view["model_kwargs"])):
        raise ValueError("Howard能源求积只读副本必须float64并保持saved PHQH构造语义")
    return model


def _complete_components(model: nn.Module, materials: Mapping[int, Any], boundaries: Any,
    geometry: AxisymmetricGeometry, audit: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    parts = []
    rc, rs, zb, zs = (geometry.copper_radius_m, geometry.silicon_carbide_radius_m,
                      geometry.copper_bottom_z_m, geometry.silicon_carbide_bottom_z_m)
    domains = (("sic", 1, (0., rs), (zs, 0.)), ("copper_ring", 0, (rs, rc), (zb, 0.)),
               ("copper_below", 0, (0., rs), (zb, zs)))
    surfaces = (("sic_top", 1, (0., rs), 0., boundaries.top_convection_coefficient_w_m2_k,
                 boundaries.silicon_carbide_emissivity),
                ("copper_shoulder", 0, (rs, rc), 0., boundaries.top_convection_coefficient_w_m2_k,
                 boundaries.copper_emissivity),
                ("copper_bottom", 0, (0., rc), zb, boundaries.bottom_convection_coefficient_w_m2_k,
                 boundaries.copper_emissivity))
    for energy, divergence in zip(audit["原始能量"], audit["原始散度"]):
        power, time_s, order = (energy["power_w"], energy["time_s"], energy["quadrature_order"])
        row = {"power_w": power, "time_s": time_s, "quadrature_order": order}
        for name, material_id, rb, z_bounds in domains:
            rz, weights = axisymmetric_rectangle_quadrature(rb, z_bounds, order, device)
            coordinates = _coordinates(rz[:, 0], rz[:, 1], time_s, power, material_id)
            density, _, heat_capacity = _constant_material_values(materials[material_id])
            storage = 0.
            for offset in range(0, len(coordinates), 2048):
                x = coordinates[offset:offset+2048].detach().clone().requires_grad_(True)
                temperature = model(x)
                derivative = torch.autograd.grad(temperature, x, torch.ones_like(temperature), create_graph=False)[0][:, 2]
                storage += float(torch.sum(weights[offset:offset+2048]*density*heat_capacity*derivative.detach()).cpu())
            row[name+"_storage_w"] = float(storage)
        for name, material_id, rb, z, coefficient, emissivity in surfaces:
            if coefficient is None: raise ValueError("Howard完整逐面air分项需要同源resolved对流系数")
            r, weights = axisymmetric_horizontal_quadrature(rb, order, device)
            coordinates = _coordinates(r, torch.full_like(r, z), time_s, power, material_id)
            with torch.no_grad(): temperature = model(coordinates)[:, 0]
            conv = convection_flux(temperature[:, None], boundaries.ambient_temperature_k, coefficient)[:, 0]
            rad = _radiative_flux(temperature, boundaries.ambient_temperature_k, emissivity, boundaries.radiation_enabled)
            row[name+"_convection_w"] = float(torch.sum(weights*conv).cpu())
            row[name+"_radiation_w"] = float(torch.sum(weights*rad).cpu())
            row[name+"_area_mean_temperature_c"] = float((torch.sum(weights*temperature)/weights.sum()).cpu())-273.15
        row.update({"sic_top_absorbed_laser_w": energy["absorbed_power_w"],
            "copper_outer_inner_cooling_w": energy["cooling_heat_w"],
            "sic_top_boundary_mismatch_w": divergence["sic_top_model_flux_w"]
                -(row["sic_top_convection_w"]+row["sic_top_radiation_w"]-energy["absorbed_power_w"]),
            "copper_shoulder_boundary_mismatch_w": divergence["copper_top_model_flux_w"]
                -(row["copper_shoulder_convection_w"]+row["copper_shoulder_radiation_w"]),
            "copper_bottom_boundary_mismatch_w": divergence["bottom_model_flux_w"]
                -(row["copper_bottom_convection_w"]+row["copper_bottom_radiation_w"]),
            "copper_outer_boundary_mismatch_w": divergence["outer_model_flux_w"]-energy["cooling_heat_w"],
            "sic_bottom_interface_w": divergence["sic_bottom_interface_flux_w"],
            "copper_under_interface_w": divergence["copper_under_interface_flux_w"],
            "sic_side_interface_w": divergence["sic_side_interface_flux_w"],
            "copper_ring_interface_w": divergence["copper_outer_ring_interface_flux_w"],
            "copper_outer_exact_node_mean_temperature_c": divergence["outer_temperature_mean_c"],
            "copper_outer_inner_node_mean_temperature_c": divergence["outer_inner_temperature_mean_c"]})
        parts.append(row)
    return {**audit, "完整物理分项": parts}


def verify_task11_howard_energy_raw(audit: Mapping[str, Any]) -> dict[str, float]:
    verified = verify_task11_energy_raw(audit)
    if type(audit["汇总"].get("功率时刻审核行数")) is not int:
        raise ValueError("Howard能源计数必须真实int，不能bool伪装")
    def same(a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7)
    components = audit.get("完整物理分项")
    if not isinstance(components, list) or len(components) != 60:
        raise ValueError("Howard能源完整逐面air、laser、储能及interface分项必须保留60行")
    for energy, terms, parts in zip(audit["原始能量"], audit["原始散度"], components):
        for row, floats, texts in ((energy, ENERGY_FLOAT_FIELDS, ENERGY_TEXT_FIELDS),
                                   (terms, DIVERGENCE_FLOAT_FIELDS, DIVERGENCE_TEXT_FIELDS)):
            if (set(row) != set(floats) | set(texts) | {"quadrature_order"}
                    or type(row["quadrature_order"]) is not int or row["quadrature_order"] not in FIXED_ORDERS
                    or any(type(row[n]) is not float or not math.isfinite(row[n]) for n in floats)
                    or any(type(row[n]) is not str or not row[n] for n in texts)):
                raise ValueError("Howard能源完整边界/内侧/interface原件schema必须保留全部有限float、int及说明")
        interface = sum(terms[n] for n in ("sic_bottom_interface_flux_w", "copper_under_interface_flux_w",
            "sic_side_interface_flux_w", "copper_outer_ring_interface_flux_w"))
        external = sum(terms[n] for n in ("sic_top_model_flux_w", "copper_top_model_flux_w", "bottom_model_flux_w", "outer_model_flux_w"))
        law = sum(energy[n] for n in ("cooling_heat_w", "convection_heat_w", "radiation_heat_w"))-energy["absorbed_power_w"]
        rhs = energy["storage_rate_w"]+external+interface
        if (energy["outer_epsilon_m"] != 1e-6 or terms["outer_epsilon_m"] != 1e-6
                or energy["boundary_gradient_mode"] != "inner_autodiff_at_R_minus_epsilon"
                or terms["boundary_gradient_mode"] != "exact_R_for_divergence_and_inner_limit_for_engineering_cooling"
                or energy["sign_convention"] != ("storage_positive_for_internal_energy_increase;"
                    "cooling_convection_radiation_positive_outward;"
                    "absorbed_power_recorded_as_positive_inward_magnitude")
                or terms["conductive_flux_sign_convention"] != "positive_outward_from_each_material"
                or energy["balance_definition"] != "storage+cooling+convection+radiation-absorbed"
                or terms["interface_definition"] != "sum_of_two_material_outward_fluxes"
                or not same(terms["outer_model_flux_w"], terms["outer_exact_model_flux_w"])
                or not same(terms["storage_rate_w"], energy["storage_rate_w"])
                or not same(terms["outer_inner_limit_cooling_flux_w"], energy["cooling_heat_w"])
                or not same(terms["interface_two_sided_flux_w"], interface)
                or not same(terms["external_model_conductive_flux_w"], external)
                or not same(terms["boundary_law_outward_w"], law)
                or not same(terms["boundary_flux_residual_w"], external-law)
                or not same(terms["storage_plus_external_plus_interface_w"], rhs)
                or not same(terms["divergence_identity_gap_w"], terms["integrated_pde_residual_w"]-rhs)):
            raise ValueError("Howard能源原边界定义及完整Cu底/外壁/肩部、SiC顶部与双侧interface的B=V-J-D逐点不闭合")
        if (not isinstance(parts, dict) or set(parts) != set(COMPONENT_FLOAT_FIELDS) | {"quadrature_order"}
                or type(parts["quadrature_order"]) is not int or parts["quadrature_order"] != energy["quadrature_order"]
                or any(type(parts[n]) is not float or not math.isfinite(parts[n]) for n in COMPONENT_FLOAT_FIELDS)
                or parts["power_w"] != energy["power_w"] or parts["time_s"] != energy["time_s"]
                or not same(sum(parts[n] for n in ("sic_storage_w", "copper_ring_storage_w", "copper_below_storage_w")), energy["storage_rate_w"])
                or not same(sum(parts[s+"_convection_w"] for s in ("sic_top", "copper_shoulder", "copper_bottom")), energy["convection_heat_w"])
                or not same(sum(parts[s+"_radiation_w"] for s in ("sic_top", "copper_shoulder", "copper_bottom")), energy["radiation_heat_w"])
                or not same(sum(parts[s+"_boundary_mismatch_w"] for s in ("sic_top", "copper_shoulder", "copper_bottom", "copper_outer")), terms["boundary_flux_residual_w"])
                or not same(parts["sic_top_absorbed_laser_w"], energy["absorbed_power_w"])
                or not same(parts["copper_outer_inner_cooling_w"], energy["cooling_heat_w"])
                or not same(sum(parts[n] for n in ("sic_bottom_interface_w", "copper_under_interface_w", "sic_side_interface_w", "copper_ring_interface_w")), terms["interface_two_sided_flux_w"])
                or not same(parts["copper_outer_exact_node_mean_temperature_c"], terms["outer_temperature_mean_c"])
                or not same(parts["copper_outer_inner_node_mean_temperature_c"], terms["outer_inner_temperature_mean_c"])):
            raise ValueError("Howard能源全部逐面air/laser/界面/材料储能完整分项与原件不一致")
        surface_flux = {"sic_top": "sic_top_model_flux_w", "copper_shoulder": "copper_top_model_flux_w",
                        "copper_bottom": "bottom_model_flux_w"}
        for surface, flux_key in surface_flux.items():
            expected_mismatch = terms[flux_key]-(parts[surface+"_convection_w"]+parts[surface+"_radiation_w"]
                -(energy["absorbed_power_w"] if surface == "sic_top" else 0.))
            if not same(parts[surface+"_boundary_mismatch_w"], expected_mismatch):
                raise ValueError("Howard能源每个独立逐面air/laser分项及原边界失配定义不一致")
        pairs = (("sic_bottom_interface_w", "sic_bottom_interface_flux_w"),
                 ("copper_under_interface_w", "copper_under_interface_flux_w"),
                 ("sic_side_interface_w", "sic_side_interface_flux_w"),
                 ("copper_ring_interface_w", "copper_outer_ring_interface_flux_w"))
        if (any(not same(parts[p], terms[t]) for p, t in pairs)
                or not same(parts["copper_outer_boundary_mismatch_w"], terms["outer_model_flux_w"]-energy["cooling_heat_w"])):
            raise ValueError("Howard能源双侧界面与外壁逐个原件分项不能用总和抵消改变")
    return verified


def _statistics(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    absolute = np.asarray([abs(row["balance_w"]) for row in rows])
    normalized = absolute / np.asarray([row["absorbed_power_w"] for row in rows])
    return {"绝对平衡宏均值_瓦": float(absolute.mean()), "绝对平衡95分位_瓦": float(np.percentile(absolute, 95)),
        "绝对平衡最大值_瓦": float(absolute.max()), "吸收功率归一宏均值": float(normalized.mean()),
        "吸收功率归一95分位": float(np.percentile(normalized, 95))}


def task11_howard_energy_diagnostics(audit: Mapping[str, Any]) -> dict[str, Any]:
    verified = verify_task11_howard_energy_raw(audit)
    energy, terms = audit["原始能量"], audit["原始散度"]
    high = energy[1::2]
    windows = []
    counts = np.zeros(len(high), dtype=np.int64)
    for name in TIME_WINDOWS:
        mask = observation_time_window_mask(np.asarray([r["time_s"] for r in high]), name)
        counts += mask
        rows = [r for r, selected in zip(high, mask) if selected]
        windows.append({"时间窗": name, "功率时刻行数": len(rows), **_statistics(rows)})
    if not np.all(counts == 1): raise ValueError("Howard能源三时间窗不得漏点或重叠双计")
    convergence = []
    for index in range(0, 60, 2):
        differences = {n: float(energy[index+1][n]-energy[index][n]) for n in ENERGY_FLOAT_FIELDS[2:8]}
        vjd = {n: float(terms[index+1][n]-terms[index][n]) for n in
            ("integrated_pde_residual_w", "interface_two_sided_flux_w", "boundary_flux_residual_w", "engineering_explanation_gap_w")}
        convergence.append({"功率_瓦": energy[index]["power_w"], "时刻_秒": energy[index]["time_s"],
            "64减16阶能量分项_瓦": differences, "64减16阶VJD分项_瓦": vjd,
            "最大能量阶差对吸收功率比": float(max(abs(v) for v in differences.values())/energy[index+1]["absorbed_power_w"])})
    failures = [{"功率_瓦": r["power_w"], "时刻_秒": r["time_s"], "原平衡B_瓦": r["balance_w"],
        "原定义相对分母_瓦": r["relative_balance_denominator_w"], "原定义相对平衡": r["relative_balance"],
        "吸收功率归一绝对残差": float(abs(r["balance_w"])/r["absorbed_power_w"])} for r in high
        if abs(r["balance_w"])/r["absorbed_power_w"] > .10]
    magnitudes = {n: {"有符号均值_瓦": float(np.mean([r[n] for r in high])),
        "绝对最大值_瓦": float(max(abs(r[n]) for r in high)), "最小值_瓦": float(min(r[n] for r in high)),
        "最大值_瓦": float(max(r[n] for r in high))} for n in ENERGY_FLOAT_FIELDS[2:9]}
    return {**verified, "名义吸收归一筛查": {"均值阈值": .05, "95分位阈值": .10,
        "均值通过": verified["吸收功率归一宏均值"] <= .05,
        "95分位通过": verified["吸收功率归一95分位"] <= .10,
        "名义同时通过": verified["吸收功率归一宏均值"] <= .05 and verified["吸收功率归一95分位"] <= .10,
        "不代表工程安全": True}, "科学或工程安全资格": False,
        "所有逐点超过名义95分位阈值原残差": failures, "失败功率时刻计数": len(failures),
        "三不重叠时间窗": windows, "双阶逐点收敛": convergence, "能源分项量级": magnitudes,
        "全部30点原残差均另存原件": True, "双阶变化是诊断不回调训练或重新选模": True}


def audit_task11_howard_energy_fixture(model: nn.Module, materials: Mapping[str, Any], boundaries: Any,
    geometry: AxisymmetricGeometry, *, project_root: str | Path, device_name: str = "cpu") -> dict[str, Any]:
    root = Path(project_root).resolve()
    if (root == PROJECT_ROOT.resolve() or not root.is_relative_to(PROJECT_ROOT.resolve())
            or device_name != "cpu" or any(p.device.type != "cpu" for p in model.parameters())
            or any(b.device.type != "cpu" for b in model.buffers())):
        raise ValueError("Howard能源合成fixture仅项目内隔离CPU，不能正式PROJECT_ROOT或CUDA")
    audit = audit_schedule_energy(model, materials, boundaries, geometry, powers_w=FIXED_POWERS,
        times_s=FIXED_TIMES, orders=FIXED_ORDERS, device=torch.device("cpu"))
    audit = _complete_components(model, materials, boundaries, geometry, audit, torch.device("cpu"))
    verify_task11_howard_energy_raw(audit)
    return {**audit, "仅隔离CPU合成测试": True, "真实科学能源合格": False}


def require_task11_howard_energy_cuda(device_name: str) -> torch.device:
    if device_name != "cuda": raise ValueError("Howard正式能源只准cuda，禁止CPU自动回退")
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or any(os.environ.get(k, d) != d for k, d in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))):
        raise ValueError("Howard正式能源必须真实恰好一张CUDA单进程；不静默CPU fallback")
    return torch.device("cuda")


def _cuda_forward_probe(model: Task11HowardComposite, device: torch.device) -> dict[str, Any]:
    hf._validate_model(model)
    if next(model.parameters()).device.type != "cuda": raise ValueError("Howard正式forward必须真实CUDA本人三网")
    coordinates = torch.tensor([[.0125, -.006, t, p, 1.] for p in FIXED_POWERS for t in FIXED_TIMES],
                               dtype=torch.float32, device=device)
    torch.cuda.synchronize(device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    with torch.no_grad(): prediction = model(coordinates)
    end.record()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-wall
    if prediction.shape != (30, 1) or prediction.dtype != torch.float32 or not torch.isfinite(prediction).all():
        raise ValueError("Howard真实CUDA float32本人视图30行forward非有限或身份漂移")
    return {"设备": str(prediction.device), "输入行数": 30, "参数dtype": "torch.float32",
        "PHQHdtype": "torch.float64", "CUDA事件秒": float(start.elapsed_time(end)/1000.),
        "同步forward墙钟秒": float(elapsed), "本探针不参与标准化推理速度排名": True}


def write_task11_howard_energy_evidence(output: str | Path, audit: Mapping[str, Any], payload: Mapping[str, Any],
    *, project_root: str | Path = PROJECT_ROOT, source_guard: Callable[[], None] | None = None) -> dict[str, str]:
    root = Path(project_root).resolve()
    if root == PROJECT_ROOT.resolve() and source_guard is None:
        raise ValueError("Howard正式能源写盘必须具备本人完整当前来源和PT导出核源guard")
    destination = task11_howard_energy_output(payload["运行目录"], output, payload["审核状态"],
        payload["运行种子"], project_root=root)
    verified = verify_task11_howard_energy_raw(audit)
    diagnostics = task11_howard_energy_diagnostics(audit)
    if (payload.get("结果不用于训练选模") is not True or payload.get("只读原件审核前后SHA一致") is not True
            or any(not hf._same(payload.get(n), v) for n, v in audit["汇总"].items())
            or any(not hf._same(payload.get(n), v) for n, v in verified.items())
            or any(not hf._same(payload.get(n), v) for n, v in diagnostics.items())
            or not isinstance(payload.get("审计独立成本"), dict)):
        raise ValueError("Howard能源写盘必须保留完整原始汇总与只读身份，不得改分母/选模")
    json.dumps(dict(payload), allow_nan=False)
    started = time.perf_counter()
    destination.mkdir(parents=True, exist_ok=False)
    for name, frame in (("指标明细.csv", audit["指标明细"]), ("物理分解.csv", audit["物理分解"])):
        with (destination / name).open("x", encoding="utf-8", newline="") as stream: frame.write_csv(stream)
    for name, rows in (("原始能量.jsonl", audit["原始能量"]), ("原始散度.jsonl", audit["原始散度"]),
                       ("完整物理分项.jsonl", audit["完整物理分项"])):
        with (destination / name).open("x", encoding="utf-8") as stream:
            for row in rows: stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    exported = float(time.perf_counter()-started)
    if source_guard is not None: source_guard()
    result = copy.deepcopy(dict(payload))
    result["审计独立成本"]["原瓦数及CSV导出墙钟秒"] = exported
    result["审计独立成本"]["导出计时范围"] = "原瓦数JSONL及CSV写盘；不含摘要和SHA封口；不参与训练或推理排名"
    with (destination / "汇总指标.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+"\n")
    digests = {p.name: sha256_file(p) for p in sorted(destination.iterdir()) if p.is_file()}
    with (destination / "审计工件SHA256.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(digests, ensure_ascii=False, allow_nan=False, indent=2)+"\n")
    return digests


def audit_task11_howard_hf_energy(*, run: str | Path, output: str | Path, state: str, seed: int,
    energy_registry: str | Path, energy_registry_sha: str, energy_source_tar: str | Path, energy_source_tar_sha: str,
    registry: str | Path, registry_sha: str, source_tar: str | Path, source_tar_sha: str,
    lf_catalog: str | Path, lf_catalog_sha: str, hf_data_catalog: str | Path, hf_data_catalog_sha: str,
    device_name: str = "cuda", project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if root != PROJECT_ROOT.resolve() or device_name != "cuda":
        raise ValueError("Howard正式能源只能实际PROJECT_ROOT及cuda；CPU合成fixture不可冒充")
    destination = task11_howard_energy_output(run, output, state, seed, project_root=root)
    started = time.perf_counter()
    checked = preflight_task11_howard_energy(energy_registry=energy_registry, energy_registry_sha=energy_registry_sha,
        energy_source_tar=energy_source_tar, energy_source_tar_sha=energy_source_tar_sha,
        registry=registry, registry_sha=registry_sha, source_tar=source_tar, source_tar_sha=source_tar_sha,
        lf_catalog=lf_catalog, lf_catalog_sha=lf_catalog_sha, hf_data_catalog=hf_data_catalog,
        hf_data_catalog_sha=hf_data_catalog_sha, project_root=root)
    identity = {**checked["HowardHF四来源"], "output": run, "seed": seed, "project_root": root}
    qualified = hf.qualify_task11_howard_hf_source(**identity)
    qualification_seconds = float(time.perf_counter()-started)
    source_unchanged(checked)
    started = time.perf_counter()
    view, model = load_task11_howard_energy_model(qualified, run=run, state=state, seed=seed, project_root=root)
    load_seconds = float(time.perf_counter()-started)
    # The real CPU factory and both immutable Float64 buffers are checked before CUDA discovery.
    source_unchanged(checked)
    device = require_task11_howard_energy_cuda(device_name)
    started = time.perf_counter()
    model.to(device)
    torch.cuda.synchronize(device)
    move_seconds = float(time.perf_counter()-started)
    forward = _cuda_forward_probe(model, device)
    started = time.perf_counter()
    prepare_task11_howard_energy_copy(model, view)
    torch.cuda.synchronize(device)
    prepare_seconds = float(time.perf_counter()-started)
    run_path = _path(run, root, file=False)
    snapshot = _path(run_path / "config_snapshot", root, file=False)
    started = time.perf_counter()
    materials = load_materials(str(_path(snapshot / "materials.yaml", root)))
    boundaries = load_resolved_boundary_conditions(str(_path(snapshot / "boundary_conditions.yaml", root)))
    geometry = AxisymmetricGeometry.from_config(load_yaml(_path(snapshot / "geometry.yaml", root)))
    audit = audit_schedule_energy(model, materials, boundaries, geometry,
        powers_w=FIXED_POWERS, times_s=FIXED_TIMES, orders=FIXED_ORDERS, device=device)
    audit = _complete_components(model, materials, boundaries, geometry, audit, device)
    torch.cuda.synchronize(device)
    energy_seconds = float(time.perf_counter()-started)
    diagnostics = task11_howard_energy_diagnostics(audit)
    started = time.perf_counter()
    after = hf.qualify_task11_howard_hf_source(**identity)
    source_unchanged(checked)
    if not hf._same(qualified, after): raise ValueError("Howard能源前后本人完整HF/LF资格、PT或来源SHA漂移，不得写盘")
    requalification_seconds = float(time.perf_counter()-started)
    payload = {"schema_version": 1, "中文说明": "本人Howard数学三网适配真实终态best/final，只读30点16/64阶名义Watt审计；保留B、原相对分母及V-J-D，失败残差全部保留；不复现作者exact训练、不证明内部真值或工程安全，不改导数、不重新训练或选模。",
        "运行种子": seed, "运行臂": hf.METHOD, "审核状态": state, "运行目录": str(run_path),
        "选择模型SHA256": qualified["最佳HF检查点SHA256" if state == "best" else "真实末HF检查点SHA256"],
        "真实完整HowardHF来源资格": qualified, "完整原件SHA256": qualified["真实工件SHA256"],
        "能源事前登记原件": str(checked["六来源原件"][0]), "能源事前源码原件": str(checked["六来源原件"][1]),
        "独立ROOT标签": ROOT_TOKEN, "独立ROOT六来源SHA256": checked["ROOT六SHA"],
        "ROOT审核前后SHA256": checked["ROOT审核SHA256"],
        "冻结全部普通源码SHA256": checked["登记"]["源码普通成员SHA256"],
        "功率_瓦": list(FIXED_POWERS), "时刻_秒": list(FIXED_TIMES), "求积阶数": list(FIXED_ORDERS),
        "本人savedPHQH元数据": view["任11HowardHF事前来源"]["固定PHQH查询"],
        "原参数dtype": "torch.float32", "能源副本参数dtype": "torch.float64", "PHQHdtype": "torch.float64",
        "真实CUDAforward探针": forward, "只读原件审核前后SHA一致": True,
        "旧固定TEST温度读取": False, "模拟测试功率温度读取": False, "工程安全阈值已建立": False,
        "结果不用于训练选模": True, "导数改造或再次训练": False, "HF训练许可": False,
        "审计独立成本": {"来源资格及独立门禁CPU核验墙钟秒": qualification_seconds,
            "模型读取及CPU工厂墙钟秒": load_seconds, "CUDA移动同步墙钟秒": move_seconds,
            "只读能源副本准备同步墙钟秒": prepare_seconds, "能源求积审计同步墙钟秒": energy_seconds,
            "来源及真实工件复核墙钟秒": requalification_seconds,
            "不作为真实训练成本或标准化推理成本": True}, **audit["汇总"], **diagnostics}
    def export_source_guard() -> None:
        source_unchanged(checked)
        if any(sha256_file(_path(run_path / name, root)) != digest
               for name, digest in qualified["真实工件SHA256"].items()):
            raise ValueError("Howard能源导出期间全部本人PT/日志/历史/温度配置原件SHA漂移；不封口")
    hashes = write_task11_howard_energy_evidence(destination, audit, payload, project_root=root, source_guard=export_source_guard)
    # The writer's final summary contains its separately timed export cost.
    result = json.loads((destination / "汇总指标.json").read_text(encoding="utf-8"))
    return {**result, "输出目录": str(destination), "审计工件SHA256": hashes}
