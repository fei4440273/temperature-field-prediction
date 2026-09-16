"""Read-only original Howard HF qualification of completed own seeds 0, 1 and 3."""

import hashlib
import inspect
import json
import math
import os
import stat
import sys
import tarfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN_BASE = ROOT / "研究记录/任务11_外部对照/正式Howard适配公平训练"
HF4 = RUN_BASE / "正式Howard_HF_seed4"
SEEDS = (0, 1, 3)
EXPECTED = {
    0: (108, 860, 360, 500, 3.682372202733206, 160, 9.77071200217873),
    1: (75, 730, 400, 330, 3.1338988354975648, 200, 5.073958195345687),
    3: (104, 870, 400, 470, 3.0032344132943343, 670, 3.0650510873185643),
}
COUNTS = Counter()
LOAD_RECORDS = []
STRUCTURE_RECORDS = []
STATE_RECORDS = []
LF_RECORDS = []
ALLOWED_PT = {}
ALLOWED_PARQUET = {}
CURRENT_SEED = None
CURRENT_PHASE = "startup"


def refuse(kind, message):
    COUNTS[kind] += 1
    raise RuntimeError(message)


def audit_path(value, dir_fd=None):
    path = Path(os.fsdecode(value))
    if not path.is_absolute() and type(dir_fd) is int and dir_fd >= 0:
        path = Path(os.readlink(f"/proc/self/fd/{dir_fd}")) / path
    lexical = Path(os.path.abspath(path))
    if lexical == HF4 or HF4 in lexical.parents:
        refuse("HF4访问拒绝尝试", "This audit never accesses HF seed4: " + str(lexical))
    return lexical.resolve()


def audit_io(event, args):
    targets = []
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = audit_path(args[0])
        if path.suffix.lower() in {".parq", ".parquet"} and path not in ALLOWED_PARQUET:
            refuse("非许可Parquet打开尝试", "Only pinned structural or LF row-count sources may open")
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        targets = [(path, None)] if writing else []
    elif event in {"os.listdir", "os.scandir"} and not isinstance(args[0], int):
        audit_path(args[0])
    elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.chown", "os.utime", "os.truncate"}:
        fd_index = {"os.mkdir": 2, "os.remove": 1, "os.rmdir": 1, "os.chmod": 2,
                    "os.chown": 3, "os.utime": 3}.get(event)
        descriptor = args[fd_index] if fd_index is not None and len(args) > fd_index else None
        targets = [(args[0], descriptor)]
    elif event in {"os.rename", "os.replace", "os.link"}:
        targets = [(args[0], args[2] if len(args) > 2 else None),
                   (args[1], args[3] if len(args) > 3 else None)]
    elif event == "os.symlink":
        targets = [(args[1], args[2] if len(args) > 2 else None)]
    for value, descriptor in targets:
        if isinstance(value, int):
            continue
        path = audit_path(value, descriptor)
        if not path.is_relative_to(HERE):
            refuse("档案外写入尝试", "No original or outside-archive writes: " + str(path))
        COUNTS["许可自身档案写入事件"] += 1


original_stat = os.stat


def guarded_stat(path, *args, **kwargs):
    if not isinstance(path, int):
        candidate = Path(os.fsdecode(path))
        if not candidate.is_absolute() and type(kwargs.get("dir_fd")) is int and kwargs["dir_fd"] >= 0:
            candidate = Path(os.readlink(f'/proc/self/fd/{kwargs["dir_fd"]}')) / candidate
        candidate = Path(os.path.abspath(candidate))
        if candidate == HF4 or HF4 in candidate.parents:
            refuse("HF4访问拒绝尝试", "HF seed4 stat is forbidden")
    return original_stat(path, *args, **kwargs)


os.stat = guarded_stat
sys.addaudithook(audit_io)

import torch
import polars as pl
import yaml


def no_cuda(*args, **kwargs):
    refuse("CUDA初始化尝试", "Pure CPU qualification never initializes CUDA")


def no_training(*args, **kwargs):
    refuse("训练或optimizer_step尝试", "Qualification must never perform a training step")


assert not torch.cuda.is_initialized()
torch.cuda._lazy_init = no_cuda
torch.cuda.init = no_cuda
if hasattr(torch._C, "_cuda_init"):
    torch._C._cuda_init = no_cuda
torch.optim.AdamW.step = no_training


def sha_stream(stream):
    digest = hashlib.sha256()
    while True:
        block = stream.read(1024 * 1024)
        if not block:
            return digest.hexdigest()
        digest.update(block)


def sha(path):
    with audit_path(path).open("rb") as stream:
        return sha_stream(stream)


def file_identity(path):
    path = audit_path(path)
    info = path.stat()
    assert stat.S_ISREG(info.st_mode) and not path.is_symlink()
    return {"路径": str(path), "字节": info.st_size, "权限": oct(stat.S_IMODE(info.st_mode)), "SHA256": sha(path)}


def ordinary_map(directory):
    result = {}
    directory = audit_path(directory)
    for path in sorted(directory.rglob("*")):
        path = audit_path(path)
        assert not path.is_symlink()
        if path.is_file():
            assert stat.S_ISREG(path.stat().st_mode)
            result[path.relative_to(directory).as_posix()] = sha(path)
        else:
            assert path.is_dir()
    return result


def write_json(name, value):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


original_load = torch.load


def pinned_load(path, *args, **kwargs):
    path = audit_path(path)
    if path not in ALLOWED_PT:
        refuse("非许可PT反序列化尝试", "Only pinned completed HF013 and prerequisite LF01234 PTs are allowed")
    assert not args and kwargs.get("map_location") == "cpu"
    assert kwargs.get("weights_only") in (True, False)
    before = sha(path)
    assert before == ALLOWED_PT[path]
    saved = original_load(path, **kwargs)
    after = sha(path)
    assert after == before and not torch.cuda.is_initialized()
    LOAD_RECORDS.append({"审核HFseed": CURRENT_SEED, "阶段": CURRENT_PHASE, "路径": str(path),
                         "before_SHA256": before, "after_SHA256": after,
                         "map_location": kwargs["map_location"], "weights_only": kwargs["weights_only"]})
    COUNTS["实际CPU_PT反序列化数"] += 1
    return saved


torch.load = pinned_load

from sic_cu.train import task11_howard_hf_formal as hf
from sic_cu.train import task11_howard_lf_formal as lf

for name in ("run_task11_howard_hf_formal", "howard_hf_epoch", "howard_hf_mixed_step",
             "howard_hf_physics_step", "_backward_step"):
    setattr(hf, name, no_training)
for name in ("run_task11_howard_lf_formal", "howard_lf_epoch", "physics_optimizer_step"):
    if hasattr(lf, name):
        setattr(lf, name, no_training)

original_schema = pl.read_parquet_schema
original_scan = pl.scan_parquet


def no_full_parquet(*args, **kwargs):
    refuse("完整Parquet或温度列读取尝试", "No full Parquet or temperature/target loading")


def structure_schema(path, *args, **kwargs):
    path = audit_path(path)
    if path not in ALLOWED_PARQUET:
        refuse("非许可Parquet_schema尝试", "Only source-pinned schemas are allowed")
    assert not args and not kwargs
    COUNTS["实际Parquet_schema读取数"] += 1
    return original_schema(path)


class ProjectionOnly:
    def __init__(self, frame, path, projected=False, columns=None):
        self.frame, self.path, self.projected, self.columns = frame, path, projected, columns

    def select(self, expression):
        assert not self.projected
        kind, columns, _ = ALLOWED_PARQUET[self.path]
        if kind == "HF开发结构":
            assert type(expression) is list and expression == columns
            recorded = expression
        else:
            assert isinstance(expression, pl.Expr) and str(expression) == str(pl.len())
            recorded = ["pl.len()；仅行数，不投影温度列"]
        return ProjectionOnly(self.frame.select(expression), self.path, True, recorded)

    def collect(self):
        assert self.projected
        kind, allowed, expected_sha = ALLOWED_PARQUET[self.path]
        assert sha(self.path) == expected_sha
        result = self.frame.collect()
        if kind == "HF开发结构":
            assert result.columns == allowed
        else:
            assert result.shape == (1, 1) and result.columns == ["len"]
        assert sha(self.path) == expected_sha
        STRUCTURE_RECORDS.append({"审核HFseed": CURRENT_SEED, "阶段": CURRENT_PHASE,
                                  "路径": str(self.path), "类别": kind, "投影": self.columns,
                                  "结果列": result.columns, "结果shape": list(result.shape), "SHA256": expected_sha})
        COUNTS["实际Parquet结构或行数collect数"] += 1
        return result


def structure_scan(path, *args, **kwargs):
    path = audit_path(path)
    if path not in ALLOWED_PARQUET:
        refuse("非许可Parquet_scan尝试", "Only pinned HF structural or LF row-count scans are allowed")
    assert not args and not kwargs
    COUNTS["实际Parquet_scan调用数"] += 1
    return ProjectionOnly(original_scan(path), path)


pl.read_parquet = no_full_parquet
pl.read_parquet_schema = structure_schema
pl.scan_parquet = structure_scan

original_state_audit = hf.audit_task11_howard_hf_state
original_lf_qualify = hf.qualify_task11_howard_lf_source
original_hf_view = hf._audit_hf_view
original_boundary = hf._audit_stage_boundary


def state_audit(saved, rows, *, seed, identity, formal=True):
    result = original_state_audit(saved, rows, seed=seed, identity=identity, formal=formal)
    states, optimizer = saved["model_state"], saved["optimizer_state"]
    records = list(optimizer["state"].values())
    parameter_names = saved["parameter_requires_grad"]
    assert len(parameter_names) == 56 and all(value is True for value in parameter_names.values())
    assert len(states) == 58 and all(states[name].dtype == torch.float32 for name in parameter_names)
    assert all(states[name].dtype == torch.float64 for name in ("linear_query_points", "nonlinear_query_points"))
    entry = {"seed": seed, "日志前缀实际轮次": len(rows), "本阶段epoch": saved["epoch"],
             "stage": saved["stage"], "formal": formal, "原参数数": len(parameter_names),
             "原参数dtype": "torch.float32", "原PHQH_buffer数": 2, "原buffer_dtype": "torch.float64",
             "原AdamW状态数": len(records), "原AdamWstep集合": sorted({float(record["step"]) for record in records}),
             "原所有动量float32且有限": all(record[key].dtype == torch.float32 and bool(torch.isfinite(record[key]).all())
                                             for record in records for key in ("exp_avg", "exp_avg_sq")),
             "四RNG键": sorted(saved["random_state"]),
             "原CUDA_RNGshape": list(saved["random_state"]["torch_cuda"][0].shape),
             "原CUDA_RNGdtype": str(saved["random_state"]["torch_cuda"][0].dtype),
             "原执行设备声明": saved["metadata"]["执行设备"]}
    assert formal is True and entry["原所有动量float32且有限"]
    STATE_RECORDS.append(entry)
    COUNTS["原完整HF状态真实审核数"] += 1
    return result


def lf_qualify(**arguments):
    result = original_lf_qualify(**arguments)
    assert result["状态"] == "CPU Howard适配LF真实终态与当前来源审核PASS"
    LF_RECORDS.append({"审核HFseed": CURRENT_SEED, "阶段": CURRENT_PHASE,
                       "LFseed": arguments["seed"], "最佳LF检查点SHA256": result["最佳LF检查点SHA256"],
                       "完整资格内容SHA256": hf.canonical_json_sha256(result),
                       "真实LF工件数": len(result["真实工件SHA256"])})
    COUNTS["五LF前提中的原完整LF资格审核数"] += 1
    return result


def hf_view(*args, **kwargs):
    result = original_hf_view(*args, **kwargs)
    COUNTS["原HF结果view真实审核数"] += 1
    return result


def boundary(*args, **kwargs):
    result = original_boundary(*args, **kwargs)
    COUNTS["原全56四RNG阶段边界真实审核数"] += 1
    return result


hf.audit_task11_howard_hf_state = state_audit
hf.qualify_task11_howard_lf_source = lf_qualify
hf._audit_hf_view = hf_view
hf._audit_stage_boundary = boundary


def main():
    global CURRENT_SEED, CURRENT_PHASE
    started = time.perf_counter()
    assert Path.cwd().resolve() == ROOT and ROOT == hf.PROJECT_ROOT.resolve()
    assert sys.executable == "/home/phl/anaconda3/envs/PINN/bin/python" and sys.flags.dont_write_bytecode
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "1"
    assert not torch.cuda.is_initialized()
    for key in ("TMPDIR", "TMP", "TEMP", "CUDA_CACHE_PATH", "XDG_CACHE_HOME", "MPLCONFIGDIR",
                "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        assert audit_path(os.environ[key]).is_relative_to(HERE)
    fixture_fd = os.open(HERE / "临时文件", os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert audit_path("dirfd_no_delete_fixture", fixture_fd) == HERE / "临时文件/dirfd_no_delete_fixture"
        audit_io("os.remove", ("dirfd_no_delete_fixture", fixture_fd))
        assert audit_path("cwd_no_delete_fixture", None) == ROOT / "cwd_no_delete_fixture"
        assert audit_path("cwd_no_delete_fixture", -1) == ROOT / "cwd_no_delete_fixture"
    finally:
        os.close(fixture_fd)
    COUNTS["自身dir_fd纯解析无删除回归PASS"] += 1
    parent_names = (hf.REGISTRY, hf.SOURCE_TAR, hf.LF_CATALOG, hf.HF_DATA_CATALOG)
    parents_before = {name: file_identity(ROOT / name) for name in parent_names}
    root_before = file_identity(ROOT / hf.ROOT_LEDGER)
    source_before = {name: sha(ROOT / name) for name in hf.SOURCE_MEMBERS}
    registration = yaml.safe_load((ROOT / hf.REGISTRY).read_text(encoding="utf-8"))
    assert len(source_before) == 82 and source_before == registration["源码普通成员SHA256"]
    hf._audit_source_tar(ROOT / hf.SOURCE_TAR, source_before, ROOT)
    tar_entries = []
    with tarfile.open(ROOT / hf.SOURCE_TAR, "r:gz") as archive:
        for member in archive.getmembers():
            assert member.isreg() and not member.issym() and not member.islnk() and not member.isdir()
            tar_entries.append({"成员": member.name, "字节": member.size, "类型": member.type.decode("ascii"),
                                "内容SHA256": sha_stream(archive.extractfile(member))})
    assert len(tar_entries) == 82 and len({row["成员"] for row in tar_entries}) == 82
    hf_before = {str(seed): ordinary_map(RUN_BASE / f"正式Howard_HF_seed{seed}") for seed in SEEDS}
    lf_before = {str(seed): ordinary_map(RUN_BASE / f"正式Howard_LF_seed{seed}") for seed in range(5)}
    for seed in SEEDS:
        assert len(hf_before[str(seed)]) == EXPECTED[seed][0]
    for kind, maps in (("HF", hf_before), ("LF", lf_before)):
        for seed, files in maps.items():
            directory = RUN_BASE / f"正式Howard_{kind}_seed{seed}"
            ALLOWED_PT.update({directory / name: digest for name, digest in files.items() if name.endswith(".pt")})
    lf_catalog = json.loads((ROOT / hf.LF_SOURCE_IDENTITY["catalog"]).read_text(encoding="utf-8"))
    for key in ("LF训练模拟原件", "LF合法验证模拟原件"):
        for item in lf_catalog[key]:
            path = audit_path(ROOT / item["原件"])
            ALLOWED_PARQUET[path] = ("LF训练验证仅行数", [], item["文件SHA256"])
    assert len(ALLOWED_PARQUET) == 70
    hf_catalog = json.loads((ROOT / hf.HF_DATA_CATALOG).read_text(encoding="utf-8"))
    hf_source_pins = {item["原件"]: item["文件SHA256"] for item in hf_catalog["HF源原件"]}
    for name, columns in (("data/processed/experiment_ir_radial.parquet", ["power_w", "split"]),
                          ("data/processed/sensor_ring_raw.parquet", ["power_w", "sensor_type", "split"])):
        path = ROOT / name
        ALLOWED_PARQUET[path] = ("HF开发结构", columns, hf_source_pins[name])
    assert len(ALLOWED_PARQUET) == 72
    for path, (_, _, digest) in ALLOWED_PARQUET.items():
        assert sha(path) == digest
    lf_parent_before = {key: file_identity(ROOT / hf.LF_SOURCE_IDENTITY[key])
                        for key in ("registry", "source_tar", "catalog")}
    arguments_base = {"registry": hf.REGISTRY, "registry_sha": parents_before[hf.REGISTRY]["SHA256"],
                      "source_tar": hf.SOURCE_TAR, "source_tar_sha": parents_before[hf.SOURCE_TAR]["SHA256"],
                      "lf_catalog": hf.LF_CATALOG, "lf_catalog_sha": parents_before[hf.LF_CATALOG]["SHA256"],
                      "hf_data_catalog": hf.HF_DATA_CATALOG,
                      "hf_data_catalog_sha": parents_before[hf.HF_DATA_CATALOG]["SHA256"], "project_root": str(ROOT)}
    assert hf._root_active(ROOT / hf.ROOT_LEDGER, {key: parents_before[name]["SHA256"]
                                                for key, name in zip(hf.IDENTITY_FIELDS, parent_names)})
    results = {}
    for seed in SEEDS:
        CURRENT_SEED, CURRENT_PHASE = seed, "独立显式正CPUpreflight"
        arguments = {**arguments_base, "seed": seed, "output": str(RUN_BASE / f"正式Howard_HF_seed{seed}")}
        print(f"SEED{seed}_PREFLIGHT_START", flush=True)
        checked = hf.preflight_task11_howard_hf(**arguments, qualified_source_only=True)
        assert checked["HF训练许可"] is False and not torch.cuda.is_initialized()
        points = checked["查询点"]
        checked_json = {key: value for key, value in checked.items() if key != "查询点"}
        checked_json["查询点原tensor只读摘要"] = {"dtype": str(points.dtype), "shape": list(points.shape),
                                            "SHA256": hashlib.sha256(points.detach().cpu().numpy().tobytes()).hexdigest()}
        write_json(f"seed{seed}_原正CPUpreflight.json", checked_json)
        CURRENT_PHASE = "原qualify_source内部preflight与全finished审核"
        print(f"SEED{seed}_ORIGINAL_QUALIFY_START", flush=True)
        before_counter = dict(COUNTS)
        q = hf.qualify_task11_howard_hf_source(**arguments)
        assert q["状态"] == "CPU Howard三网HF本人真实终态与当前来源审核PASS"
        count, epochs, first, second, best_score, best_epoch, final_score = EXPECTED[seed]
        assert q["已完成"] is True and q["阶段1实际轮次"] == first and q["阶段2实际轮次"] == second
        assert first + second == epochs and q["全局最佳分数"] == best_score and q["全局最佳轮次"] == best_epoch
        assert q["最近合法分数"] == final_score and q["真实工件SHA256"] == hf_before[str(seed)]
        assert len(q["真实工件SHA256"]) == count and q["HF训练许可"] is False
        assert q["原文精确三网联合训练复现"] is False and not torch.cuda.is_initialized()
        q_text = json.dumps(q, ensure_ascii=False, indent=2, allow_nan=False)
        with (HERE / f"seed{seed}_原Python完整资格q.json").open("x", encoding="utf-8") as stream:
            stream.write(q_text + "\n")
        results[str(seed)] = {"实际APIkwargs": arguments, "完整原PythonqJSON字符串": q_text,
                              "全资格真实审核计数增量": {key: value - before_counter.get(key, 0)
                                                          for key, value in COUNTS.items()},
                              "实际工件数": count, "实际完整轮次": epochs,
                              "实际best_S": q["全局最佳分数"], "实际best_epoch": q["全局最佳轮次"],
                              "实际final_S": q["最近合法分数"], "会话数": q["会话数"],
                              "best_SHA256": q["最佳HF检查点SHA256"], "final_SHA256": q["真实末HF检查点SHA256"]}
        print(f"SEED{seed}_FULL_QUALIFICATION_PASS " + json.dumps({key: value for key, value in results[str(seed)].items()
                                                                if key not in {"完整原PythonqJSON字符串", "实际APIkwargs"}},
                                                               ensure_ascii=False, allow_nan=False), flush=True)
        del checked, checked_json, points, q
    CURRENT_PHASE = "只读收尾重新核源"
    source_after = {name: sha(ROOT / name) for name in hf.SOURCE_MEMBERS}
    parents_after = {name: file_identity(ROOT / name) for name in parent_names}
    hf_after = {str(seed): ordinary_map(RUN_BASE / f"正式Howard_HF_seed{seed}") for seed in SEEDS}
    lf_after = {str(seed): ordinary_map(RUN_BASE / f"正式Howard_LF_seed{seed}") for seed in range(5)}
    lf_parent_after = {key: file_identity(ROOT / hf.LF_SOURCE_IDENTITY[key])
                       for key in ("registry", "source_tar", "catalog")}
    root_after = file_identity(ROOT / hf.ROOT_LEDGER)
    assert source_after == source_before and parents_after == parents_before
    assert hf_after == hf_before and lf_after == lf_before and lf_parent_after == lf_parent_before
    assert hf._root_active(ROOT / hf.ROOT_LEDGER, {key: parents_after[name]["SHA256"]
                                                for key, name in zip(hf.IDENTITY_FIELDS, parent_names)})
    for path, (_, _, digest) in ALLOWED_PARQUET.items():
        assert sha(path) == digest
    hf._audit_source_tar(ROOT / hf.SOURCE_TAR, source_after, ROOT)
    assert COUNTS["五LF前提中的原完整LF资格审核数"] == 30
    assert COUNTS["原全56四RNG阶段边界真实审核数"] == 3
    for key in ("HF4访问拒绝尝试", "CUDA初始化尝试", "训练或optimizer_step尝试", "非许可PT反序列化尝试",
                "非许可Parquet打开尝试", "完整Parquet或温度列读取尝试", "非许可Parquet_schema尝试",
                "非许可Parquet_scan尝试", "档案外写入尝试"):
        assert COUNTS[key] == 0
    assert not torch.cuda.is_initialized()
    machine = {"状态": "独立CPU原API成功HF013完整资格全部PASS；不代表五HF完训或十态评估",
               "北京时间": datetime.now().astimezone().isoformat(), "本次实际审核墙钟秒": time.perf_counter() - started,
               "真实API签名": str(inspect.signature(hf.preflight_task11_howard_hf)),
               "真实资格API": "qualify_task11_howard_hf_source(**arguments)，内部qualified_source_only=True、finished=True",
               "运行环境": {"Python": sys.version, "executable": sys.executable, "Torch": torch.__version__,
                            "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"], "CUDA_is_initialized终态": torch.cuda.is_initialized(),
                            "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"], "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
                            "全部临时缓存路径": {key: os.environ[key] for key in ("TMPDIR", "TMP", "TEMP", "CUDA_CACHE_PATH", "XDG_CACHE_HOME",
                                                                       "MPLCONFIGDIR", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR")}},
               "scope": {"完整成功HF": list(SEEDS), "前提完整LF": list(range(5)), "禁止HF4": str(HF4),
                         "温度标签读取": False, "训练step": False, "GPU初始化": False,
                         "先前工具选路偏差": "独审开始时一次rg --files全项目目录枚举实际exit0、输出3497行截断；未限制HF013，不能声称全任务零HF4目录枚举。未打开HF4内容/PT/log/receipt，无原件写入。此守卫进程只显式HF013及已冻结LF01234。"},
               "结果": results, "四HF父身份_before": parents_before, "四HF父身份_after": parents_after,
               "源码82_before": source_before, "源码82_after": source_after, "源码82零漂移": True,
               "普通tar82严格复核": tar_entries, "普通tar无目录链接重复额外缺失": True,
               "成功HF工件_before": hf_before, "成功HF工件_after": hf_after, "成功HF零漂移": True,
               "五LF工件_before": lf_before, "五LF工件_after": lf_after, "五LF零漂移": True,
               "三LF父原件_before": lf_parent_before, "三LF父原件_after": lf_parent_after,
               "ROOT历史身份_before": root_before, "ROOT历史身份_after": root_after,
               "ROOTwholeSHA必须恒等": False, "ROOT自身四SHA真实plain正门禁前后PASS": True,
               "原PT真实CPU读入记录": LOAD_RECORDS, "原所有完整HF状态实际审核记录": STATE_RECORDS,
               "原所有LF资格实际审核记录": LF_RECORDS, "真实Parquet结构或LF行数IO": STRUCTURE_RECORDS,
               "许可Parquet路径与源SHA": {str(path): {"类别": kind, "允许投影列": columns, "SHA256": digest}
                                        for path, (kind, columns, digest) in ALLOWED_PARQUET.items()},
               "守卫计数_before最终JSON写出": dict(COUNTS),
               "计数时间边界": "JSON写出前的进程观测；最终完整stdout/stderr与actualexit由独立外层捕获，不冒称退出后全时计数。",
               "科学边界": ["仅HF013三成功自身完整资格，不算五mean/std，不以success subset冒五seed。",
                           "失败HF2/HF4不在本读入scope；不得据三成功宣布五HF或十能源/十OBS门槛满足。",
                           "不称Howard普遍失败、不称作者exact、不采纳为B0；没有内部真值、安全或泛化结论。",
                           "HF开发真实IO仅power_w/split/sensor_type；LF70真实IO仅schema与pl.len()行数及文件SHA，无温度/target列或TEST数据解码。",
                           "只保存审核证据，不改ROOT/全局状态/源/模型/日志/收据/冻结登记/既有档案。"]}
    write_json("独立成功013资格原Python机器证据.json", machine)
    print("HF013_INDEPENDENT_ORIGINAL_API_AUDIT_COMPLETE " + json.dumps(
        {"完成seed": list(SEEDS), "实际总PT读入数": len(LOAD_RECORDS), "实际完整HF状态审核数": len(STATE_RECORDS),
         "实际五LF前提资格审核数": len(LF_RECORDS), "实际Parquet结构或行数collect数": len(STRUCTURE_RECORDS),
         "SOURCE82和四父件与三成功HF及五LF零漂移": True, "CUDA_is_initialized": torch.cuda.is_initialized()},
        ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
