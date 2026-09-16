"""Task 10 board-only method initialization, CPU steps, and source audit."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import random

import numpy as np
import torch
from torch import Tensor, nn

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.eval.task10_plate_benchmark import load_benchmark_registration
from sic_cu.eval.task10_plate_training_source import PlateTrainingBatch, Task10PlateTrainingSource
from sic_cu.losses.task10_plate_physics import (
    BoardCollocation, RegisteredBoardPhysics, registered_board_collocation,
)
from sic_cu.models.task10_plate_deeponet import BoardDeepONet, BoardMultifidelityDeepONet


def make_same_lf_pair(low_model: BoardDeepONet, *, seed: int, width: int = 64,
                      latent_dim: int = 64, blocks: int = 2,
                      ) -> tuple[BoardMultifidelityDeepONet, BoardMultifidelityDeepONet]:
    if not isinstance(low_model, BoardDeepONet) or low_model.with_lf_input:
        raise ValueError("F2/F3须由同一已训一维板LF DeepONet张量复制，不接受旧RZ权重")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("F2/F3必须固定相同且非负的随机种子")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        correction = BoardDeepONet(width=width, latent_dim=latent_dim, blocks=blocks,
                                    with_lf_input=True)
    parameter = next(low_model.parameters())
    correction = correction.to(device=parameter.device, dtype=parameter.dtype)
    return (
        BoardMultifidelityDeepONet(deepcopy(low_model), deepcopy(correction), "F2"),
        BoardMultifidelityDeepONet(deepcopy(low_model), deepcopy(correction), "F3"),
    )


def _method_tensors(model: nn.Module, batch: PlateTrainingBatch,
                    allowed_kinds: tuple[str, ...]) -> tuple[Tensor, Tensor]:
    if (not isinstance(batch, PlateTrainingBatch) or batch.fold != "train"
            or batch.kind not in allowed_kinds
            or batch.coordinates_z_t_q_material.ndim != 2
            or batch.coordinates_z_t_q_material.shape[1] != 4
            or batch.temperature_k.shape != (len(batch.coordinates_z_t_q_material), 1)):
        raise ValueError("重训只接受训练折合法四列板探针或该训练折LF单源")
    parameter = next(model.parameters())
    if parameter.device.type not in ("cpu", "cuda"):
        raise ValueError("任10只可在前登记且获GPU排队许可的CPU或CUDA运行")
    coordinates = torch.as_tensor(batch.coordinates_z_t_q_material, dtype=parameter.dtype,
                                  device=parameter.device)
    labels = torch.as_tensor(batch.temperature_k, dtype=parameter.dtype, device=parameter.device)
    if not torch.isfinite(coordinates).all() or not torch.isfinite(labels).all():
        raise ValueError("合法同板训练批次输入或温度非有限")
    return coordinates, labels


def _registered_probe_coordinates(times: tuple | list, depths: tuple | list,
                                  flux_w_m2: int, sic_length_m: float) -> np.ndarray:
    time_values = np.asarray(times, dtype=np.float64)
    depth_values = np.asarray(depths, dtype=np.float64)
    material = (depth_values < sic_length_m).astype(np.float64)
    return np.column_stack((np.tile(depth_values, len(time_values)),
                            np.repeat(time_values, len(depth_values)),
                            np.full(len(time_values) * len(depth_values), flux_w_m2),
                            np.tile(material, len(time_values))))


def _require_registered_probe(batch: PlateTrainingBatch, expected: np.ndarray) -> None:
    coords = batch.coordinates_z_t_q_material
    if (coords.shape != expected.shape or batch.temperature_k.shape != (len(expected), 1)
            or not np.isfinite(coords).all() or not np.isfinite(batch.temperature_k).all()
            or not np.allclose(coords, expected, atol=1e-12, rtol=0)):
        raise ValueError("HF探针批次须逐行严格匹配前登记折热流、14时刻、三位置和材料")


def _require_registered_coarse_lf(batch: PlateTrainingBatch) -> None:
    coordinates = batch.coordinates_z_t_q_material
    if (batch.fold != "train" or batch.kind not in ("lf_same_physics_coarse",
                                                 "lf_contact_mismatch_coarse")
            or coordinates.ndim != 2 or coordinates.shape[1] != 4
            or not 0 < len(coordinates) <= 512 or not np.isfinite(coordinates).all()):
        raise ValueError("LF直接优化只准登记训练折粗网格四列且单步不超过512点")
    setup = load_benchmark_registration()
    geometry = setup["geometry"]
    coarse = setup["low_fidelity_sources"]["same_physics_coarse"]
    sic, cu = coarse["silicon_carbide_cells"], coarse["copper_cells"]
    widths = np.r_[np.full(sic, geometry["silicon_carbide_thickness_m"] / sic),
                   np.full(cu, geometry["copper_thickness_m"] / cu)]
    depths = np.cumsum(widths) - widths / 2
    material = np.r_[np.ones(sic), np.zeros(cu)]
    near = np.abs(coordinates[:, 0:1] - depths[None, :])
    indexes = near.argmin(axis=1)
    times = np.arange(round(setup["reference_controls"]["end_time_s"] /
                            coarse["time_step_s"]) + 1) * coarse["time_step_s"]
    if (not np.all(near[np.arange(len(coordinates)), indexes] <= 1e-12)
            or not np.array_equal(coordinates[:, 3], material[indexes])
            or not np.isin(coordinates[:, 1], times).all()
            or not np.all(coordinates[:, 2] == coordinates[0, 2])
            or coordinates[0, 2] not in setup["flux_splits_w_m2"]["train"]):
        raise ValueError("LF训练行不得冒充HF密网格/隐藏热流，只接受24/11粗中心、2秒时刻和训练折")


def train_lf_method_step(model: BoardDeepONet, optimizer: torch.optim.Optimizer,
                         lf_training: PlateTrainingBatch) -> dict:
    if not isinstance(model, BoardDeepONet) or model.with_lf_input:
        raise ValueError("LF完整场仅训练新一维板单输出DeepONet")
    if not isinstance(lf_training, PlateTrainingBatch):
        raise ValueError("LF训练入口只接受受限来源的一维粗网格批次")
    _require_registered_coarse_lf(lf_training)
    coordinates, labels = _method_tensors(model, lf_training,
                                          ("lf_same_physics_coarse", "lf_contact_mismatch_coarse"))
    optimizer.zero_grad(set_to_none=True)
    objective = ((model(coordinates) - labels) / model.scale_k).square().mean()
    if not torch.isfinite(objective):
        raise ValueError("任10 LF训练目标非有限，不得保存权重")
    objective.backward()
    optimizer.step()
    return {"实际优化步数": 1, "LF训练目标": float(objective.detach())}


def train_hf_method_step(model: nn.Module, optimizer: torch.optim.Optimizer,
                         hf_training: PlateTrainingBatch, physics: RegisteredBoardPhysics,
                         collocation: BoardCollocation) -> dict:
    if ((physics.method == "F1" and
         (not isinstance(model, BoardDeepONet) or model.with_lf_input))
            or (physics.method in ("F2", "F3") and
                (not isinstance(model, BoardMultifidelityDeepONet)
                 or model.method != physics.method))):
        raise ValueError("同板F1/F2/F3模型与PDE分支须各自一致")
    flux = float(collocation.interior[0, 2])
    if flux not in physics.training_fluxes_w_m2 or any(not torch.all(rows[:, 2] == flux)
                                                       for rows in (collocation.initial,
                                                                    collocation.top,
                                                                    collocation.bottom,
                                                                    collocation.interface_sic,
                                                                    collocation.interface_cu,
                                                                    collocation.interior)):
        raise ValueError("公开物理配点只能与同一预登记训练折热流配对")
    _require_registered_probe(hf_training, _registered_probe_coordinates(
        physics.training_probe_times_s, physics.training_probe_depths_m,
        int(flux), physics.sic_length))
    coordinates, labels = _method_tensors(model, hf_training, ("probe",))
    optimizer.zero_grad(set_to_none=True)
    observed = ((model(coordinates) - labels) / physics.temperature_scale_k).square().mean()
    pieces = physics.components(model, collocation)
    objective = observed + sum(value for value in pieces.values() if value is not None)
    if not torch.isfinite(objective):
        raise ValueError("一维同板HF可见探针/公开物理训练目标非有限，不得优化")
    objective.backward()
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
           for parameter in model.parameters() if parameter.requires_grad):
        raise ValueError("同板物理梯度非有限，不得优化或当模型成绩")
    optimizer.step()
    return {"实际优化步数": 1, "训练目标": float(objective.detach()),
            "HF探针目标": float(observed.detach()),
            "体内PDE实际调用": physics.interior_pde_calls}


def legal_probe_validation_rmse_k(model: nn.Module,
                                  batches: tuple[PlateTrainingBatch, ...]) -> dict:
    if (not batches or any(not isinstance(batch, PlateTrainingBatch)
                           or batch.fold != "validation" or batch.kind != "probe"
                           or batch.coordinates_z_t_q_material.shape != (42, 4)
                           or batch.temperature_k.shape != (42, 1) for batch in batches)):
        raise ValueError("模型早停仅验证折30/70千合法三探针×14时刻，不得使用完整HF")
    registered = load_benchmark_registration()
    allowed = set(registered["flux_splits_w_m2"]["validation"])
    q_values = [int(batch.coordinates_z_t_q_material[0, 2]) for batch in batches]
    if set(q_values) != allowed or len(batches) != 2 or len(set(q_values)) != len(q_values):
        raise ValueError("同板模型验证必须各含两个预登记功率，无训练/数值隐藏折")
    for batch, q in zip(batches, q_values):
        _require_registered_probe(batch, _registered_probe_coordinates(
            registered["observations"]["validation_times_s"],
            registered["observations"]["allowed_depths_from_top_m"],
            q, registered["geometry"]["silicon_carbide_thickness_m"]))
    parameter = next(model.parameters())
    with torch.no_grad():
        errors = []
        per_flux = {}
        for batch, q in zip(batches, q_values):
            x = torch.as_tensor(batch.coordinates_z_t_q_material, dtype=parameter.dtype,
                                device=parameter.device)
            y = torch.as_tensor(batch.temperature_k, dtype=parameter.dtype,
                                device=parameter.device)
            delta = model(x) - y
            if not torch.isfinite(delta).all():
                raise ValueError("合法探针验证非有限，不能选检查点")
            errors.append(delta.square())
            per_flux[str(q)] = float(delta.square().mean().sqrt())
    return {"合法验证仅用三探针": True,
            "验证折": sorted(allowed),
            "两档探针RMSE_K": float(torch.cat(errors).mean().sqrt()),
            "逐档探针RMSE_K": per_flux}


def audit_training_source_code(path: str | Path, *, expected_sha256: str | None = None) -> str:
    """Catch direct full-HF or original-cylinder ingress before trainer execution."""
    source = Path(path).resolve()
    if source != PROJECT_ROOT and PROJECT_ROOT not in source.parents:
        raise ValueError("训练程序静态审核只可读取项目内实路径源码")
    current_sha = sha256_file(source)
    if expected_sha256 is not None and current_sha != expected_sha256:
        raise PermissionError("独立一维板训练脚本源码SHA漂移，不得执行")
    parsed = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    banned_imports = (
        "sic_cu.models.multifidelity", "sic_cu.models.deeponet_pinn",
        "sic_cu.physics.heat_equation", "sic_cu.train.task07_source",
        "sic_cu.train.task08_f2_no_pde", "sic_cu.eval.task10_hidden_evaluator",
    )
    banned_literals = ("HF_封存完整场", "test_Data", "full_internal_hf",
                       "旧RZ权重", "hidden_test")
    for node in ast.walk(parsed):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if any(node.module == name or node.module.startswith(name + ".")
                   for name in banned_imports):
                raise PermissionError("禁止把旧RZ圆柱网络/方程导入一维板训练")
            if (node.module == "sic_cu.eval.task10_plate_benchmark" and
                    any(alias.name in ("Task10HiddenEvaluator", "solve_registered_plate",
                                       "check_registered_reference") for alias in node.names)):
                raise PermissionError("一维训练禁止直接导入HF完整参考数值工厂/后验评价器")
        if isinstance(node, ast.Import) and any(alias.name in banned_imports
                                                 for alias in node.names):
            raise PermissionError("禁止导入旧RZ几何网络或方程")
        if isinstance(node, ast.Constant):
            if (isinstance(node.value, str)
                    and any(fragment in node.value for fragment in banned_literals)):
                raise PermissionError("训练程序包含完整HF或数值隐藏折源场路径，不得运行")
            if type(node.value) is int and node.value in (42000, 58000):
                raise PermissionError("训练程序不能直接编码42/58千隐藏评价热流")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and (node.func.attr in ("open_full_reference", "open_full_hf")
                     or (node.func.attr == "load" and isinstance(node.func.value, ast.Name)
                         and node.func.value.id in ("np", "numpy", "torch")))):
            raise PermissionError("训练源码不得直接加载完整HF或任07/08旧模型权重")
    return current_sha


@dataclass(frozen=True)
class BoardMethodFit:
    model: nn.Module
    best_state: dict[str, Tensor]
    training: dict
    validation: dict
    best_optimizer_state: dict
    final_optimizer_state: dict
    final_model_state: dict[str, Tensor]
    best_rng_state: dict
    final_rng_state: dict
    lf_optimizer_state: dict | None = None
    lf_rng_state: dict | None = None


def _snapshot_rng(device: torch.device) -> dict:
    numpy_state = np.random.get_state()
    cuda = ([state.detach().clone() for state in torch.cuda.get_rng_state_all()]
            if device.type == "cuda" else [])
    if device.type == "cuda" and not cuda:
        raise ValueError("正式CUDA四类随机态必须有至少一张卡的torch RNG原件")
    return {
        "python_random": random.getstate(),
        "numpy_global": {"generator": numpy_state[0],
                         "keys": numpy_state[1].tolist(), "position": int(numpy_state[2]),
                         "has_gauss": int(numpy_state[3]),
                         "cached_gaussian": float(numpy_state[4])},
        "torch_cpu": torch.get_rng_state().detach().clone(),
        "torch_cuda_all": cuda,
    }


def _optimizer_real_state(optimizer: torch.optim.Optimizer, *, steps: int) -> dict:
    eligible = [p for group in optimizer.param_groups for p in group["params"]
                if p.requires_grad]
    if (not eligible or len(optimizer.state) != len(eligible)
            or any("step" not in optimizer.state[p]
                   or int(optimizer.state[p]["step"]) != steps for p in eligible)):
        raise ValueError("真实AdamW每个可更新参数的步数必须与HF/LF真实优化步逐位相同")
    return deepcopy(optimizer.state_dict())


def _state_tensors_sha256(model: nn.Module) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        values = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{values.dtype}:{tuple(values.shape)}\n".encode("ascii"))
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def _check_unit_budget(seed: int, hf_steps: int, validation_every: int,
                       collocation_points: int) -> None:
    if (not isinstance(seed, int) or seed < 0 or not isinstance(hf_steps, int)
            or hf_steps <= 0 or not isinstance(validation_every, int)
            or validation_every <= 0 or not isinstance(collocation_points, int)
            or collocation_points < 2):
        raise ValueError("同板方法预算须训练前冻结非负种子、正优化/验证步及至少2个配点")


def _check_device(device: torch.device) -> None:
    if not isinstance(device, torch.device) or device.type not in ("cpu", "cuda"):
        raise ValueError("设备须由事前预算及GPU串行排队明确选CPU或CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA尚不可用，不得伪装正式GPU重训")


def _source_batches(source: Task10PlateTrainingSource,
                    ) -> tuple[list[int], list[PlateTrainingBatch], tuple[PlateTrainingBatch, ...]]:
    if not isinstance(source, Task10PlateTrainingSource):
        raise ValueError("正式方法仅可使用钉SHA的同板F1/F2/F3受限训练来源")
    powers = [int(q) for q in source._training.setup["flux_splits_w_m2"]["train"]]
    train = [source.training_probe(q) for q in powers]
    validation = tuple(source.validation_probe(q) for q in
                       source._validation.setup["flux_splits_w_m2"]["validation"])
    if len(powers) != 5 or len(validation) != 2 or any(len(batch.temperature_k) != 42
                                                       for batch in (*train, *validation)):
        raise ValueError("同板方法必须全部加载5档训练和2档合法三探针验证")
    return powers, train, validation


def _fit_hf(model: nn.Module, source: Task10PlateTrainingSource,
            physics: RegisteredBoardPhysics, powers: list[int],
            training: list[PlateTrainingBatch], validation: tuple[PlateTrainingBatch, ...],
            *, hf_steps: int, validation_every: int, collocation_points: int,
            learning_rate: float) -> BoardMethodFit:
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters()
                                   if parameter.requires_grad), lr=learning_rate,
                                  weight_decay=1e-6)
    if optimizer.state:
        raise ValueError("HF AdamW必须从同seed空状态开始，禁止延用旧RZ动量")
    selected = None
    best_state = None
    best_optimizer_state = None
    best_rng_state = None
    best_report = None
    best_step = 0
    step_rows = []
    moments = [float(t) for t in source._training.setup["observations"]["train_times_s"] if t > 0]
    total_points = 0
    for offset in range(hf_steps):
        index = offset % len(powers)
        time_s = moments[(offset // len(powers)) % len(moments)]
        collocation = registered_board_collocation(
            source._training.registration_path, powers[index],
            points=collocation_points, time_s=time_s,
        )
        objective = train_hf_method_step(model, optimizer, training[index], physics, collocation)
        total_points += len(training[index].temperature_k)
        step = offset + 1
        step_row = {"HF优化步": step, "训练热流_W_m2": powers[index],
                    "物理配点时刻_s": time_s, "HF三探针消费点数": len(training[index].temperature_k),
                    "实际训练目标": objective["训练目标"],
                    "HF可见探针目标": objective["HF探针目标"],
                    "累计体内PDE计算次数": objective["体内PDE实际调用"],
                    "合法两档探针验证RMSE_K": None}
        if step % validation_every == 0 or step == hf_steps:
            report = legal_probe_validation_rmse_k(model, validation)
            step_row["合法两档探针验证RMSE_K"] = report["两档探针RMSE_K"]
            if selected is None or report["两档探针RMSE_K"] < selected:
                selected = report["两档探针RMSE_K"]
                best_step = step
                best_state = deepcopy(model.state_dict())
                best_optimizer_state = _optimizer_real_state(optimizer, steps=step)
                best_rng_state = _snapshot_rng(next(model.parameters()).device)
                best_report = report
        step_rows.append(step_row)
    assert best_state is not None and best_report is not None
    assert best_optimizer_state is not None and best_rng_state is not None
    final_optimizer_state = _optimizer_real_state(optimizer, steps=hf_steps)
    final_rng_state = _snapshot_rng(next(model.parameters()).device)
    best_report["最佳步数"] = best_step
    train_report = {
        "方法": physics.method, "训练折": powers,
        "实际优化步数": hf_steps, "最佳HF优化步数": best_step,
        "可见HF探针实际优化消费点数": total_points,
        "体内PDE实际调用": physics.interior_pde_calls,
        "源场清单SHA256": source._manifest_sha,
        "逐步HF真实优化明细": step_rows,
        "HF空AdamW起步": True,
    }
    return BoardMethodFit(model, best_state, train_report, best_report,
                          best_optimizer_state, final_optimizer_state,
                          deepcopy(model.state_dict()), best_rng_state, final_rng_state)


def fit_f1_method(source: Task10PlateTrainingSource, *, seed: int, hf_steps: int,
                  validation_every: int, width: int = 64, latent_dim: int = 64,
                  blocks: int = 2, collocation_points: int = 8,
                  learning_rate: float = .001,
                  device: torch.device = torch.device("cpu")) -> BoardMethodFit:
    _check_unit_budget(seed, hf_steps, validation_every, collocation_points)
    _check_device(device)
    if source.method != "F1" or source.lf_source is not None:
        raise ValueError("F1 HF-only仅合法训练探针与同板公开物理，不能消费LF")
    powers, train, validation = _source_batches(source)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = BoardDeepONet(width=width, latent_dim=latent_dim, blocks=blocks)
    model = model.to(device)
    physics = RegisteredBoardPhysics(source._training.registration_path, method="F1")
    return _fit_hf(model, source, physics, powers, train, validation,
                   hf_steps=hf_steps, validation_every=validation_every,
                   collocation_points=collocation_points, learning_rate=learning_rate)


def fit_same_lf_pair_methods(source_f2: Task10PlateTrainingSource,
                             source_f3: Task10PlateTrainingSource, *, seed: int,
                             lf_steps: int, hf_steps: int, validation_every: int,
                             width: int = 64, latent_dim: int = 64, blocks: int = 2,
                             collocation_points: int = 8, lf_batch_size: int = 512,
                             learning_rate: float = .001,
                             device: torch.device = torch.device("cpu"),
                             ) -> tuple[BoardMethodFit, BoardMethodFit]:
    _check_unit_budget(seed, hf_steps, validation_every, collocation_points)
    _check_device(device)
    if (source_f2.method != "F2" or source_f3.method != "F3"
            or source_f2.lf_source != source_f3.lf_source
            or source_f2.lf_source is None
            or source_f2._manifest_sha != source_f3._manifest_sha
            or source_f2._training.archive_root != source_f3._training.archive_root
            or source_f2._training.registration_path != source_f3._training.registration_path):
        raise ValueError("解释单PDE贡献须F2/F3同一LF来源、同一源SHA、同seed观察折配对")
    if (not isinstance(lf_steps, int) or lf_steps <= 0
            or not isinstance(lf_batch_size, int) or lf_batch_size <= 0):
        raise ValueError("两方法须共享LF预训正优化步预算和正采样数")
    powers, f2_train, f2_validation = _source_batches(source_f2)
    f3_powers, f3_train, f3_validation = _source_batches(source_f3)
    if (powers != f3_powers or any(not np.array_equal(a.temperature_k, b.temperature_k)
                                   or not np.array_equal(a.coordinates_z_t_q_material,
                                                         b.coordinates_z_t_q_material)
                                   for a, b in zip((*f2_train, *f2_validation),
                                                   (*f3_train, *f3_validation)))):
        raise ValueError("F2/F3必须逐探针共享相同HF训练和合法验证原件")
    all_lf = [source_f2.training_low_fidelity(q) for q in powers]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        pretrained = BoardDeepONet(width=width, latent_dim=latent_dim, blocks=blocks)
    pretrained = pretrained.to(device)
    lf_optimizer = torch.optim.AdamW(pretrained.parameters(), lr=learning_rate,
                                     weight_decay=1e-6)
    generator = np.random.default_rng(seed)
    sampler_initial = deepcopy(generator.bit_generator.state)
    consumed = 0
    lf_rows = []
    for offset in range(lf_steps):
        batch = all_lf[offset % len(powers)]
        indexes = generator.integers(0, len(batch.temperature_k),
                                     size=min(lf_batch_size, len(batch.temperature_k)))
        chosen = PlateTrainingBatch(batch.coordinates_z_t_q_material[indexes],
                                    batch.temperature_k[indexes], "train", batch.kind)
        result = train_lf_method_step(pretrained, lf_optimizer, chosen)
        consumed += len(indexes)
        lf_rows.append({"LF优化步": offset + 1, "训练热流_W_m2": powers[offset % len(powers)],
                        "LF训练消费点数": len(indexes), "实际LF训练目标": result["LF训练目标"]})
    lf_optimizer_state = _optimizer_real_state(lf_optimizer, steps=lf_steps)
    lf_rng_state = _snapshot_rng(device)
    lf_rng_state["numpy_sampler_start"] = sampler_initial
    lf_rng_state["numpy_sampler_final"] = deepcopy(generator.bit_generator.state)
    model_f2, model_f3 = make_same_lf_pair(pretrained, seed=seed, width=width,
                                           latent_dim=latent_dim, blocks=blocks)
    if (_state_tensors_sha256(model_f2.low_model) != _state_tensors_sha256(model_f3.low_model)
            or _state_tensors_sha256(model_f2.correction)
            != _state_tensors_sha256(model_f3.correction)):
        raise ValueError("F2/F3同LF/同seed初始网络全张量并非逐位相同，不得配对")
    initial_lf_sha = _state_tensors_sha256(model_f2.low_model)
    initial_hf_sha = _state_tensors_sha256(model_f2.correction)
    fit2 = _fit_hf(model_f2, source_f2,
                   RegisteredBoardPhysics(source_f2._training.registration_path, method="F2"),
                   powers, f2_train, f2_validation, hf_steps=hf_steps,
                   validation_every=validation_every, collocation_points=collocation_points,
                   learning_rate=learning_rate)
    fit3 = _fit_hf(model_f3, source_f3,
                   RegisteredBoardPhysics(source_f3._training.registration_path, method="F3"),
                   powers, f3_train, f3_validation, hf_steps=hf_steps,
                   validation_every=validation_every, collocation_points=collocation_points,
                   learning_rate=learning_rate)
    for fit in (fit2, fit3):
        fit.training["LF来源"] = source_f2.lf_source
        fit.training["同seed共享LF真实优化步数"] = lf_steps
        fit.training["同seed共享LF真实监督消费点数"] = consumed
        fit.training["随机种子"] = seed
        fit.training["共享LF逐步真实优化明细"] = lf_rows
        fit.training["同seed共享LF初始全张量SHA256"] = initial_lf_sha
        fit.training["同seed共享HF校正初始全张量SHA256"] = initial_hf_sha
    return (replace(fit2, lf_optimizer_state=deepcopy(lf_optimizer_state),
                    lf_rng_state=deepcopy(lf_rng_state)),
            replace(fit3, lf_optimizer_state=deepcopy(lf_optimizer_state),
                    lf_rng_state=deepcopy(lf_rng_state)))
