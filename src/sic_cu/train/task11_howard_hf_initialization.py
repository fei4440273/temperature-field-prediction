"""仅CPU构造Howard三网并严格迁移LF；不读取数据或授权任何HF训练。

调用方仍须单独核验LF检查点外部SHA、完整LF终态资格及PH/QH查询许可。
这里的内容哈希只标识当前内存张量，不能替代上游事前登记或训练证据。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from sic_cu.models.common import ModelScales
from sic_cu.models.task11_howard_composite import Task11HowardComposite
from sic_cu.train.task11_howard_lf_formal import (
    IDENTITY_FIELDS, ISOLATION_FLAGS, LF_MODEL_KWARGS, LF_PARAMETER_COUNT,
    METHOD, Task11HowardLFTeacher, _same,
)


@contextmanager
def _cpu_float32_rng(seed: int):
    previous_dtype = torch.get_default_dtype()
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        try:
            torch.set_default_dtype(torch.float32)
            torch.default_generator.manual_seed(seed)
            yield
        finally:
            torch.set_default_dtype(previous_dtype)


def _state_content_sha(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        header = json.dumps([name, str(value.dtype), list(value.shape)],
                            ensure_ascii=True, separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate_lf_view(view: Mapping[str, Any]) -> tuple[dict[str, Tensor], dict[str, Any]]:
    if (not isinstance(view, Mapping) or type(view.get("schema_version")) is not int
            or view["schema_version"] != 1 or view.get("method") != METHOD
            or type(view.get("seed")) is not int or view["seed"] not in range(5)
            or type(view.get("epoch")) is not int or not 1 <= view["epoch"] <= 2000
            or not _same(view.get("model_kwargs"), LF_MODEL_KWARGS)
            or not _same(view.get("scales"), asdict(ModelScales()))
            or type(view.get("LF参数量")) is not int or view["LF参数量"] != LF_PARAMETER_COUNT
            or type(view.get("validation_rmse_c")) not in (int, float)
            or not math.isfinite(view["validation_rmse_c"]) or view["validation_rmse_c"] < 0
            or view.get("原文精确三网联合训练复现") is not False
            or any(view.get(name) is not False for name in ISOLATION_FLAGS)):
        raise ValueError("Howard LF视图方法/seed/构造/尺度/参数量/标签隔离或适配身份不符")
    source = view.get("任11Howard事前来源")
    if (not isinstance(source, Mapping) or set(source) != set(IDENTITY_FIELDS)
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in source.values())):
        raise ValueError("Howard LF来源三SHA格式不完整；仍须上游外部资格核验")
    with _cpu_float32_rng(view["seed"]):
        teacher = Task11HowardLFTeacher()
    reference = teacher.network.state_dict()
    raw, complete = view.get("lf_subnet_state"), view.get("model_state")
    if (not isinstance(raw, Mapping) or set(raw) != set(reference)
            or not isinstance(complete, Mapping)
            or set(complete) != set(teacher.state_dict())):
        raise ValueError("Howard LF完整teacher与子网权重必须含全部唯一真实参数名")
    for name, expected in reference.items():
        value, full_value = raw[name], complete["network." + name]
        if (not isinstance(value, Tensor) or not isinstance(full_value, Tensor)
                or value.device.type != "cpu" or full_value.device.type != "cpu"
                or value.layout != torch.strided or full_value.layout != torch.strided
                or value.shape != expected.shape or full_value.shape != expected.shape
                or value.dtype != expected.dtype or full_value.dtype != expected.dtype
                or not torch.isfinite(value).all() or not torch.isfinite(full_value).all()
                or not torch.equal(value, full_value)):
            raise ValueError("Howard LF权重须为有限CPU float32、严格尺寸且完整/子网视图逐项相等")
    weights = {name: value.detach().clone().contiguous() for name, value in raw.items()}
    teacher.network.load_state_dict(weights, strict=True)
    return weights, {"LF来源方法": METHOD, "LF来源seed": view["seed"],
                     "LF来源epoch": view["epoch"], "LF参数量": LF_PARAMETER_COUNT,
                     "LF来源三SHA": dict(source),
                     "LF子网权重内容SHA256": _state_content_sha(weights)}


def _json_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(type(name) is not str for name in value):
            raise ValueError("调用者PH/QH查询元数据的JSON字段名必须为字符串")
        return {name: _json_metadata(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_json_metadata(item) for item in value]
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("调用者PH/QH查询元数据必须为有限且可严格JSON往返的对象")


def _trainable_cpu_model(model: Task11HowardComposite) -> None:
    """要求完整真实CPU三网，且全部参数的底层存储字节区间互不重叠。"""
    if (type(model) is not Task11HowardComposite
            or not _same({"width": model.width, "depth": model.depth,
                          "latent_dim": model.latent_dim},
                         {name: LF_MODEL_KWARGS[name] for name in ("width", "depth", "latent_dim")})
            or type(model.scales) is not ModelScales
            or type(model.query_chunk_size) is not int or model.query_chunk_size < 1
            or not _same(asdict(model.scales), asdict(ModelScales()))):
        raise ValueError("仅允许与固定Howard LF相容的真实CPU三网模型")
    buffers = dict(model.named_buffers())
    if set(buffers) != {"linear_query_points", "nonlinear_query_points"}:
        raise ValueError("Howard三网必须保留全部且仅有真实PH/QH固定查询缓冲")
    if any(value.device.type != "cpu" or value.dtype != torch.float64
           or value.layout != torch.strided or value.ndim != 2 or value.shape[1] != 4
           or len(value) == 0 or value.requires_grad is not False or not torch.isfinite(value).all()
           or not ((value[:, 3] == 0) | (value[:, 3] == 1)).all() for value in buffers.values()):
        raise ValueError("Howard PH/QH固定查询缓冲必须为非空有限CPU float64 N乘4、材料0/1且不求梯度")
    # Only the architecture and query values are compared with the seed-0
    # skeleton; migrated LF and fresh HF parameter values are not overwritten.
    with _cpu_float32_rng(0):
        reference = Task11HowardComposite(**model.model_kwargs)
    modules, reference_modules = dict(model.named_modules()), dict(reference.named_modules())
    if (modules.keys() != reference_modules.keys()
            or any(type(value) is not type(reference_modules[name]) for name, value in modules.items())):
        raise ValueError("Howard必须为完整真实三网及其固定仿射模块结构，不能替换或增删子网")
    if (model.low_fidelity_subnet.final_activation is not False
            or model.nonlinear_subnet.final_activation is not True
            or type(model.scaler.scales) is not ModelScales
            or not _same(asdict(model.scaler.scales), asdict(ModelScales()))):
        raise ValueError("Howard LF/NL末激活与实际CoordinateScaler尺度必须严格固定")
    parameters, reference_parameters = dict(model.named_parameters()), dict(reference.named_parameters())
    if (parameters.keys() != reference_parameters.keys()
            or len({id(value) for value in parameters.values()}) != len(parameters)
            or any(value.shape != reference_parameters[name].shape
                   or value.requires_grad is not True or value.device.type != "cpu"
                   or value.layout != torch.strided or value.dtype != reference_parameters[name].dtype
                   or not torch.isfinite(value).all() for name, value in parameters.items())):
        raise ValueError("Howard三网全部唯一参数名字/尺寸/有限CPU float32与真正bool True标志必须完整")
    storage_ranges = []
    for value in parameters.values():
        storage = value.untyped_storage()
        start = storage.data_ptr()
        storage_ranges.append((start, start + storage.nbytes()))
    storage_ranges.sort()
    # Compare whole backing stores, not tensor starts: even disjoint slices
    # sharing one allocation violate the independent-parameter contract.
    if (any(start <= 0 or end <= start for start, end in storage_ranges)
            or any(next_start < end for (_, end), (next_start, _)
                   in zip(storage_ranges, storage_ranges[1:]))):
        raise ValueError("Howard三网全部参数必须具有独立且无重叠的底层存储，即使张量切片互不相交")
    state, reference_state = model.state_dict(), reference.state_dict()
    if (state.keys() != reference_state.keys()
            or any(value.shape != reference_state[name].shape or value.device.type != "cpu"
                   or value.dtype != reference_state[name].dtype or value.layout != torch.strided
                   or not torch.isfinite(value).all() for name, value in state.items())
            or any(not torch.equal(value, reference_state[name]) for name, value in buffers.items())):
        raise ValueError("Howard真实完整三网状态与PH/QH构造参数的名字/尺寸/dtype/缓冲值不一致")


def initialize_task11_howard_hf(
    lf_view: Mapping[str, Any], *, seed: int,
    linear_query_points: Tensor | Sequence[Sequence[float]],
    nonlinear_query_points: Tensor | Sequence[Sequence[float]],
    query_metadata: Mapping[str, Any], query_chunk_size: int = 16384,
) -> tuple[Task11HowardComposite, dict[str, Any]]:
    """先校验内存LF视图，再CPU fresh三网+LF strict迁移；不构造默认HF优化器。

    HF seed只限制为合法非负CPU RNG整数，不在此决定正式HF seed/预算。
    PH/QH坐标由调用者显式传入；查询元数据仅忠实往返，不被当作许可。
    """
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Howard HF初始化seed必须为非负合法整数，不能使用bool或浮点别名")
    if type(query_chunk_size) is not int or query_chunk_size < 1:
        raise ValueError("Howard LF查询chunk总行数必须为正整数")
    if not isinstance(query_metadata, Mapping):
        raise ValueError("调用者PH/QH查询元数据必须为JSON对象；不在此虚构查询来源")
    metadata = _json_metadata(query_metadata)
    for points in (linear_query_points, nonlinear_query_points):
        if isinstance(points, Tensor) and points.device.type != "cpu":
            raise ValueError("Howard HF初始化只接受调用者显式CPU PH/QH查询坐标")
    weights, source = _validate_lf_view(lf_view)
    with _cpu_float32_rng(seed):
        model = Task11HowardComposite(
            linear_query_points=linear_query_points, nonlinear_query_points=nonlinear_query_points,
            scales=ModelScales(), width=128, depth=4, latent_dim=128,
            query_chunk_size=query_chunk_size)
    model.low_fidelity_subnet.load_state_dict(weights, strict=True)
    _trainable_cpu_model(model)
    if not _same(model.low_fidelity_subnet.state_dict(), weights):
        raise ValueError("Howard LF strict迁移后权重与来源子网不一致")
    identity = {**source, "状态": "仅CPU三网初始化与LF严格迁移；没有执行HF训练",
                "HF初始化seed": seed, "新鲜HF子网": ["linear_subnet", "nonlinear_subnet"],
                "三网全部参数可训练": True, "LF严格迁移": True,
                "HF训练许可": False, "旧固定TEST温度读取": False,
                "模拟测试功率温度读取": False, "原文精确三网联合训练复现": False,
                "LF外部SHA与完整终态资格核验": False,
                "查询许可核验": False, "调用者查询元数据": metadata,
                "模型构造参数": _json_metadata(model.model_kwargs),
                "HF公式": "Kelvin=295.15+250*(Fl+Fnl)，不额外加LF基线",
                "三网初始化状态内容SHA256": _state_content_sha(model.state_dict()),
                "三网初始化参数量": model.parameter_count(),
                "优化器": "未构造；需调用者显式提供HF学习率并另行完成训练许可门禁"}
    return model, identity


def build_task11_howard_hf_adamw(
    model: Task11HowardComposite, *, lr: float, weight_decay: float, **kwargs: Any,
) -> torch.optim.AdamW:
    """真实空AdamW包含全部三网；lr/weight_decay显式提供，不授权HF。

    只支持PyTorch 2.5普通不可微CPU模式：有限Python数值betas/eps、
    bool amsgrad/maximize、foreach为bool/None；capturable/differentiable
    只能为False，fused只能为False/None。不承诺高级优化器模式支持。
    """
    _trainable_cpu_model(model)
    if (type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0
            or type(weight_decay) not in (int, float)
            or not math.isfinite(weight_decay) or weight_decay < 0):
        raise ValueError("Howard HF AdamW的显式学习率/weight_decay必须为有限合法实数")
    supported = {"betas", "eps", "amsgrad", "maximize", "foreach", "capturable",
                 "differentiable", "fused"}
    if set(kwargs) - supported:
        raise ValueError("Howard CPU AdamW仅接受PyTorch 2.5明确支持的标准参数")
    if "betas" in kwargs:
        betas = kwargs["betas"]
        if (not isinstance(betas, (list, tuple)) or len(betas) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or not 0 <= value < 1 for value in betas)):
            raise ValueError("Howard CPU AdamW betas须为两个有限非bool实数，各位于[0,1)")
    if "eps" in kwargs:
        eps = kwargs["eps"]
        if type(eps) not in (int, float) or not math.isfinite(eps) or eps < 0:
            raise ValueError("Howard CPU AdamW eps须为有限非bool非负实数")
    if any(name in kwargs and type(kwargs[name]) is not bool
           for name in ("amsgrad", "maximize", "capturable", "differentiable")):
        raise ValueError("Howard CPU AdamW标准开关必须为真正bool，不接受数值或字符串别名")
    if any(name in kwargs and kwargs[name] is not None and type(kwargs[name]) is not bool
           for name in ("foreach", "fused")):
        raise ValueError("Howard CPU AdamW foreach/fused必须为真正bool或None")
    if (kwargs.get("capturable", False) or kwargs.get("differentiable", False)
            or kwargs.get("fused", False)):
        raise ValueError("Howard CPU AdamW仅支持普通不可微模式，禁止capturable/differentiable/fused True")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    if optimizer.state_dict()["state"]:
        raise ValueError("Howard HF初始化AdamW必须为空状态，不得借LF或旧HF动量")
    return optimizer
