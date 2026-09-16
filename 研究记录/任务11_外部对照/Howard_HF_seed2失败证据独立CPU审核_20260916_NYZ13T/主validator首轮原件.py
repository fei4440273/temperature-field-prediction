"""Pinned seed2 failure review: two real CPU states, hashes and saved terminal evidence only."""

import ast
import hashlib
import json
import math
import os
import stat
import sys
import tarfile
from datetime import datetime
from pathlib import Path, PurePosixPath

import torch
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RECORD_DIR = ROOT / "研究记录/任务11_外部对照"
RUN = RECORD_DIR / "正式Howard适配公平训练/正式Howard_HF_seed2"
MD = RECORD_DIR / "任11_Howard_HF_seed2第304轮数值失败中文诊断_20260916T205117+0800.md"
MACHINE = RECORD_DIR / "任11_Howard_HF_seed2数值失败现场与原303断点CPU304真实复现_20260916T205117+0800.json"
ARCHIVE = RECORD_DIR / "Howard_HF_seed2第304轮数值失败完整普通字节归档_20260916T205117+0800.tar.gz"
EXPECTED = {
    MD: "5f9f128596e11fd9817f1ae01544651501cc2d35315fb548f0d6f60ff2dfa732",
    MACHINE: "25d8ccc3a98307314dc37119f0105133e6a407d5ddc068bc933e23617fec8a13",
    ARCHIVE: "61f681baced196bbb94587a16ec4506aaf672096eb7cbfe67259c31b547f100e",
}
ALLOWED_LOADS = {RUN / "最近提交历史_0001.pt", RUN / "阶段_最近.pt"}
COUNTS = {"CUDA初始化尝试": 0, "训练或step尝试": 0, "非许可PT反序列化尝试": 0,
          "真实温度Parquet读取尝试": 0, "档案外写入尝试": 0, "实际CPU_PT反序列化数": 0}


def no_cuda(*args, **kwargs):
    COUNTS["CUDA初始化尝试"] += 1
    raise RuntimeError("Bounded review forbids all CUDA initialization")


def no_training(*args, **kwargs):
    COUNTS["训练或step尝试"] += 1
    raise RuntimeError("Bounded review must not repeat epoch304 or any optimizer step")


def audit_io(event, args):
    targets = []
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        if path.suffix.lower() in {".parq", ".parquet"}:
            COUNTS["真实温度Parquet读取尝试"] += 1
            raise RuntimeError("No real temperature or Parquet reads allowed")
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        targets = [path] if writing else []
    elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.chown", "os.utime", "os.truncate"}:
        targets = [args[0]]
    elif event in {"os.rename", "os.replace", "os.link"}:
        targets = args[:2]
    elif event == "os.symlink":
        targets = [args[1]]
    for value in targets:
        if isinstance(value, int):
            continue
        path = Path(os.fsdecode(value)).resolve()
        if not path.is_relative_to(HERE):
            COUNTS["档案外写入尝试"] += 1
            raise RuntimeError("Read-only failure evidence outside own new archive: " + str(path))


assert not torch.cuda.is_initialized()
torch.cuda._lazy_init = no_cuda
torch.cuda.init = no_cuda
if hasattr(torch._C, "_cuda_init"):
    torch._C._cuda_init = no_cuda
torch.optim.AdamW.step = no_training
sys.addaudithook(audit_io)
original_load = torch.load


def pinned_cpu_load(path, *, expected):
    path = Path(path).resolve()
    if path not in ALLOWED_LOADS:
        COUNTS["非许可PT反序列化尝试"] += 1
        raise RuntimeError("Only pinned real seed2 epoch200 and303 may be loaded")
    assert sha(path) == expected
    COUNTS["实际CPU_PT反序列化数"] += 1
    saved = original_load(path, map_location="cpu", weights_only=False)
    assert sha(path) == expected
    return saved


torch.load = pinned_cpu_load
from sic_cu.train import task11_howard_hf_formal as hf
hf.howard_hf_epoch = no_training
hf.howard_hf_mixed_step = no_training
hf.howard_hf_physics_step = no_training
hf._backward_step = no_training


def sha_stream(reader):
    digest = hashlib.sha256()
    while True:
        block = reader.read(1024 * 1024)
        if not block:
            return digest.hexdigest()
        digest.update(block)


def sha(path):
    with Path(path).open("rb") as stream:
        return sha_stream(stream)


def identity(path):
    path = hf._path(path, ROOT)
    return {"路径": str(path), "字节": path.stat().st_size, "权限": oct(stat.S_IMODE(path.stat().st_mode)), "SHA256": sha(path)}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def fresh_files():
    result = {}
    for path in sorted(RUN.rglob("*")):
        hf._path(path, ROOT, file=False)
        assert not path.is_symlink()
        if path.is_file():
            assert stat.S_ISREG(path.stat().st_mode)
            result[path.relative_to(RUN).as_posix()] = sha(path)
        else:
            assert path.is_dir()
    return result


def numerical_copy_match(raw, copy, location="$", conversions=None):
    if conversions is None:
        conversions = []
    if type(raw) is float:
        assert type(copy) in (int, float) and math.isfinite(raw) and float(copy) == raw, location
        if type(copy) is not float:
            conversions.append(location)
    elif isinstance(raw, dict):
        assert isinstance(copy, dict) and raw.keys() == copy.keys(), location
        for key in raw:
            numerical_copy_match(raw[key], copy[key], location + "." + key, conversions)
    elif isinstance(raw, list):
        assert isinstance(copy, list) and len(raw) == len(copy), location
        for index, (left, right) in enumerate(zip(raw, copy)):
            numerical_copy_match(left, right, location + f"[{index}]", conversions)
    else:
        assert type(raw) is type(copy) and raw == copy, location
    return conversions


def actual_stage(data, name, terminal_key="write_stdin真实完整终态"):
    stage = data[name]
    terminal = stage[terminal_key]
    assert terminal["exit_code"] == 0
    raw = json.loads(terminal["output"])
    conversions = numerical_copy_match(raw, stage["结构化原stdout"])
    return raw, {"完整原实际命令": stage["完整实际命令"], "实际工具终态": terminal,
                 "原始stdout解析成功": True, "结构化阅读副本float变int路径": conversions}


def moments_of(saved):
    names = list(dict(hf._fresh_cpu_model().named_parameters()))
    result = []
    for index, state in saved["optimizer_state"]["state"].items():
        row = {"参数顺序": index, "参数名称": names[index], "step": float(state["step"])}
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            row[name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                "所有值有限": bool(torch.isfinite(value).all()), "绝对最大值": float(value.abs().max()),
                "正无穷数": int(torch.isposinf(value).sum()), "负无穷数": int(torch.isneginf(value).sum()),
                "NaN数": int(torch.isnan(value).sum())}
        result.append(row)
    return result


def main():
    assert Path.cwd().resolve() == ROOT and ROOT == hf.PROJECT_ROOT.resolve()
    assert sys.executable == "/home/phl/anaconda3/envs/PINN/bin/python" and sys.flags.dont_write_bytecode
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "1"
    for path, digest in EXPECTED.items():
        assert sha(path) == digest
    data = read_json(MACHINE)
    md_text = MD.read_text(encoding="utf-8")
    scene, scene_evidence = actual_stage(data, "真实失败现场只读核对")
    cpu, cpu_evidence = actual_stage(data, "从原303副本只诊断一个CPU304轮")
    archived, archive_evidence = actual_stage(data, "失败50普通原文件完整字节归档")
    before = fresh_files()
    assert len(before) == 50 and before == scene["失败后整个本人普通文件SHA目录"] == archived["全部失败真实原始文件SHA"]
    source_before = {name: sha(hf._path(name, ROOT)) for name in hf.SOURCE_MEMBERS}
    registration = yaml.safe_load((ROOT / hf.REGISTRY).read_text(encoding="utf-8"))
    assert len(source_before) == 82 and source_before == registration["源码普通成员SHA256"]
    hf._audit_source_tar(ROOT / hf.SOURCE_TAR, source_before, ROOT)
    assert stat.S_IMODE(ARCHIVE.stat().st_mode) == 0o444 and ARCHIVE.stat().st_size == 87312127
    tar_map = {}
    tar_members = []
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        members = archive.getmembers()
        assert len(members) == 50
        for member in members:
            assert member.isreg() and not member.islnk() and not member.issym() and not member.isdir()
            parts = PurePosixPath(member.name).parts
            assert parts[0] == RUN.name and not member.name.startswith("/") and ".." not in parts
            relative = PurePosixPath(*parts[1:]).as_posix()
            assert relative in before and relative not in tar_map
            reader = archive.extractfile(member)
            assert reader is not None
            with reader:
                content_sha = sha_stream(reader)
            assert member.size == (RUN / relative).stat().st_size and content_sha == before[relative]
            tar_map[relative] = content_sha
            tar_members.append({"包内名称": member.name, "原相对路径": relative, "普通type": repr(member.type),
                "字节": member.size, "contentSHA256": content_sha})
    assert tar_map == before
    child = data["本人seed2失败真实子退出码_stdout_stderr"]
    assert child["seed"] == 2 and child["真实子CLI包装退出码"] == 1
    child_lines = child["真实完整stdout"].splitlines()
    assert len(child_lines) == 1 and child_lines[0].startswith("HOWARD_HF_CHILD_START=")
    start = json.loads(child_lines[0].split("=", 1)[1])
    assert start["PID"] == 1870464 and start["seed"] == 2 and start["接续起点实际轮次"] == 200
    assert start["本人最近断点SHA256"] == before["最近提交历史_0001.pt"]
    assert start["前驱收据SHA256"] == before["HF会话收据_0001.json"]
    assert "HOWARD_HF_CHILD_COMPLETE" not in child["真实完整stdout"]
    assert "line 1310" in child["真实完整stderr"] and "line 302" in child["真实完整stderr"]
    queue_terminal = data["原必要串行队列实际末工具返回"]
    assert queue_terminal["exit_code"] == 1 and "AssertionError: Howard本人真实会话失败" in queue_terminal["output"]
    queue_events = [json.loads(line.split("=", 1)[1]) for line in data["原必要串行队列原完整stdout及stderr工具合并输出"].splitlines()
        if line.startswith("HOWARD_HF_QUEUE_CHILD_TERMINAL=")]
    assert queue_events[-1] == child
    queue_starts = [json.loads(line.split("=", 1)[1]) for line in data["原必要串行队列原完整stdout及stderr工具合并输出"].splitlines()
        if line.startswith("HOWARD_HF_QUEUE_NEXT=")]
    assert queue_starts[-1]["seed"] == 2 and queue_starts[-1]["父队列PID"] == 1851759
    receipt = read_json(RUN / "HF会话收据_0001.json")
    rows = [json.loads(line) for line in (RUN / "training.jsonl").read_text(encoding="utf-8").splitlines()]
    committed_rows = [json.loads(line) for line in (RUN / "已提交日志_0001.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 303 and len(committed_rows) == 200 and rows[:200] == committed_rows
    assert receipt["累计实际轮次"] == receipt["本会话实际轮次"] == 200 and receipt["起始已提交轮次"] == 0
    assert receipt["状态"] == "已暂停且完整Howard HF阶段提交" and receipt["选模重推"]["已完成"] is False
    assert receipt["提交历史SHA256"]["最近提交历史_0001.pt"] == before["最近提交历史_0001.pt"]
    for name, digest in receipt["提交历史SHA256"].items():
        assert before[name] == digest
    current_drift = {name: {"收据SHA256": digest, "当前原件SHA256": before[name]} for name, digest in receipt["原件SHA256"].items()
                     if before.get(name) != digest}
    assert set(current_drift) == {"阶段_最近.pt", "training.jsonl"}
    audit200 = pinned_cpu_load(RUN / "最近提交历史_0001.pt", expected=before["最近提交历史_0001.pt"])
    audit303 = pinned_cpu_load(RUN / "阶段_最近.pt", expected=before["阶段_最近.pt"])
    state200 = hf.audit_task11_howard_hf_state(audit200, committed_rows, seed=2, identity=receipt["身份"], formal=True)
    state303 = hf.audit_task11_howard_hf_state(audit303, rows, seed=2, identity=receipt["身份"], formal=True)
    assert hf._same(state200, scene["首200提交断点完整状态CPU审核返回"])
    assert hf._same(state303, scene["最新durable断点完整状态CPU审核返回"])
    assert state303["已完成"] is False and state303["阶段1截止轮次"] is None
    assert state303["全局实际轮次"] == 303 and state200["全局实际轮次"] == 200
    assert audit303["metadata"]["日志SHA256"] == before["training.jsonl"]
    moment303 = moments_of(audit303)
    assert moment303 == scene["最新durable断点56真实动量检查"]
    assert all(row["step"] == 4848 for row in moment303)
    assert all(row["step"] == 3200 for row in moments_of(audit200))
    assert all(not (RUN / name).exists() for name in ("final.pt", "metrics.json", "HF会话收据_0002.json", "阶段_训练末.pt"))
    assert scene["当前formal接续门禁真实拒绝"] == "Howard HF当前原件或静态初态/事前快照与收据SHA漂移"
    assert scene["当前原日志末3行"] == rows[-3:]
    assert state303["全局最佳轮次"] == 150 and state303["全局最佳分数"] == 3.457685370974757
    assert state303["最近合法分数"] == 2497.152174288104
    assert [row["名义物理训练分项"]["boundary"] for row in rows[-3:]] == [30827.12890625, 738797.1875, 2801033216.0]
    assert [row["名义物理训练分项"]["physics_total"] for row in rows[-3:]] == [31266.865234375, 740236.875, 2801037056.0]
    moments = cpu["56参数与原AdamW当前实际逐项"]
    names = list(dict(hf._fresh_cpu_model().named_parameters()))
    assert len(moments) == 56 and [row["参数名称"] for row in moments] == names
    assert cpu["诊断CPU轮次数"] == 1 and cpu["诊断epoch实际ValueError"] is None
    assert cpu["诊断AdamW实际ValueError"] == "Howard HF必须保存全部56真实一致AdamW step与有限同dtype/shape动量"
    assert cpu["原303断点SHA256"] == before["阶段_最近.pt"]
    bad = [row for row in moments if row["exp_avg_sq"]["非有限元素数"]]
    assert len(bad) == cpu["实际非有限动量参数数"] == 32
    assert all(row["参数全有限"] is True and row["最终梯度全有限"] is True and row["step"] == 4864 for row in moments)
    assert all(row["exp_avg"]["非有限元素数"] == 0 and row["exp_avg"]["NaN数"] == 0 for row in moments)
    assert all(row["exp_avg_sq"]["非有限元素数"] == row["exp_avg_sq"]["正无穷数"] and row["exp_avg_sq"]["NaN数"] == 0 for row in moments)
    assert all(row[kind]["dtype"] == "torch.float32" for row in moments for kind in ("exp_avg", "exp_avg_sq"))
    for raw, original in zip(moments, moment303):
        assert all(raw[kind]["shape"] == original[kind]["shape"] for kind in ("exp_avg", "exp_avg_sq"))
    consumption = cpu["诊断全轮实际消费若成功"]
    assert all(consumption[key] == expected for key, expected in hf.EPOCH_CONSUMPTION.items())
    assert cpu["CUDA未初始化"] is True and cpu["原单CUDA Philox不调用或伪造为CPU"] is True
    gradient = moments[0]["最终梯度最大有限值"]
    assert gradient == 2.401524235096994e22
    float32_max = float(torch.finfo(torch.float32).max)
    second_moment_contribution = (1 - hf.ADAMW_CONTRACT["betas"][1]) * gradient ** 2
    assert math.isfinite(gradient) and second_moment_contribution > float32_max
    square = torch.tensor(gradient, dtype=torch.float32).square()
    assert square.dtype == torch.float32 and bool(torch.isposinf(square))
    source_text = (ROOT / "src/sic_cu/train/task11_howard_hf_formal.py").read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    backward = functions["_backward_step"]
    backward_calls = sorted((node.lineno, ast.unparse(node.func)) for node in ast.walk(backward) if isinstance(node, ast.Call))
    assert [name for _, name in backward_calls if name.endswith(".backward") or name.endswith(".step")] == ["objective.backward", "optimizer.step"]
    session = functions["_run_session"]
    session_lines = sorted((node.lineno, ast.unparse(node.func)) for node in ast.walk(session) if isinstance(node, ast.Call)
        and ast.unparse(node.func) in ("howard_hf_epoch", "audit_task11_howard_hf_adamw", "log.open", "_save_stage"))
    assert (1310, "audit_task11_howard_hf_adamw") in session_lines
    assert "不能" in md_text and "五" in md_text and data["Task11完整五HF十能源十视图验收已完成"] is False
    after = fresh_files()
    source_after = {name: sha(ROOT / name) for name in hf.SOURCE_MEMBERS}
    assert before == after and source_before == source_after
    assert all(sha(path) == expected for path, expected in EXPECTED.items())
    assert COUNTS["实际CPU_PT反序列化数"] == 2 and all(value == 0 for key, value in COUNTS.items() if key != "实际CPU_PT反序列化数")
    assert not torch.cuda.is_initialized()
    result = {"性质": "新有界独立只读失败证据审核，不重跑304、不授训练GPU或完整五HF资格",
        "记录时间": datetime.now().astimezone().isoformat(timespec="seconds"), "输入三原件fresh身份": {path.name: identity(path) for path in EXPECTED},
        "原现场命令及原工具终态": scene_evidence, "原CPU304命令及原工具终态": cpu_evidence,
        "原50普通归档命令及原工具终态": archive_evidence,
        "50当前原件beforeSHA": before, "50当前原件afterSHA": after, "50tar普通逐成员核验": tar_members,
        "50成员无link目录重复缺失额外": True, "SOURCE82_before": source_before, "SOURCE82_after": source_after,
        "HF82普通父tar实际审查": True, "原HF四身份": {name: identity(ROOT / name) for name in (hf.REGISTRY, hf.SOURCE_TAR, hf.LF_CATALOG, hf.HF_DATA_CATALOG)},
        "原队列30719父实际终态exit": queue_terminal["exit_code"], "父PID": 1851759,
        "真实seed2子START": start, "真实seed2子exit": 1, "真实seed2子stdout": child["真实完整stdout"], "真实seed2子stderr": child["真实完整stderr"],
        "最后合法200收据": receipt, "303相对收据两原件漂移": current_drift,
        "真实200完整状态独立CPU审核": state200, "真实303完整状态独立CPU审核": state303,
        "真实303全56原动量独立CPU审核": moment303, "原CPU304全56参数动量逐项": moments,
        "CPU304正无穷二阶动量参数数": len(bad), "CPU304正无穷二阶动量元素数": sum(row["exp_avg_sq"]["正无穷数"] for row in moments),
        "CPU304参数及最终梯度均有限": True, "原GPU失败非有限二阶动量实际计数": None,
        "纯Float32平方验证": {"梯度实际float": gradient, "Float32最大有限值": float32_max,
            "数学0.001乘梯度平方": second_moment_contribution, "实际Float32平方正无穷": True, "不执行优化器step或训练": True},
        "冻结源码实际调用次序": {"backward函数calls": backward_calls, "会话关键函数calls": session_lines},
        "原GPU失败段结束时间": None, "原GPU失败额外训练墙钟秒": None, "五seedmean": None, "五seedstd_ddof1": None,
        "科学口径": "seed2不完训且缺final/metrics，5mean/std保持NULL；不以成功子集冒五seed，不移除或替补失败；五全资格和十能/十OBS未满足；不泛化Howard普遍失败，不声称作者exact或B0替换",
        "CPU审核安全守卫": {**COUNTS, "CUDA_initialized": torch.cuda.is_initialized(), "torch_version": torch.__version__},
        "新任务或重训启动": False, "全局或原件写删改": False}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
