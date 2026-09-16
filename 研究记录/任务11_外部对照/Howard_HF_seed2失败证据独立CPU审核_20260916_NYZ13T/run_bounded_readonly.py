"""Exclusive raw evidence capture for one bounded read-only CPU failure audit."""

import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


here = Path(__file__).resolve().parent
root = here.parents[2]
assert Path.cwd().resolve() == root
assert sys.executable == "/home/phl/anaconda3/envs/PINN/bin/python" and sys.flags.dont_write_bytecode
assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "1"
keys = ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTHONPATH", "TMPDIR", "TMP", "TEMP", "CUDA_CACHE_PATH",
        "XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR")
environment = {key: os.environ[key] for key in keys}
argv = [sys.executable, "-B", str(here / "independent_failure_audit.py")]
assert len(sys.argv) == 1 or sys.argv[1:] == ["再核"]
prefix = "独立CPU只读实跑" if len(sys.argv) == 1 else "独立CPU只读再核"
started = datetime.now().astimezone().isoformat(timespec="seconds")
clock = time.perf_counter()
completed = subprocess.run(argv, cwd=root, env=os.environ.copy(), capture_output=True, text=True, check=False)
elapsed = time.perf_counter() - clock
stdout_path = here / (prefix + ".stdout.txt")
stderr_path = here / (prefix + ".stderr.txt")
for path, text in ((stdout_path, completed.stdout), (stderr_path, completed.stderr)):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
record = {"性质": "有界独立CPU只读失败证据审核原始实命令与完整终态", "完整实际命令":
    shlex.join(["env", *(f"{key}={value}" for key, value in environment.items()), *argv]),
    "外层实际argv": [sys.executable, "-B", *sys.argv], "实际环境": environment,
    "工作目录": str(root), "实际开始": started, "实际终止": datetime.now().astimezone().isoformat(timespec="seconds"),
    "墙钟秒": elapsed, "实际退出码": completed.returncode, "完整stdout": completed.stdout,
    "完整stderr": completed.stderr, "stdout原件": str(stdout_path), "stderr原件": str(stderr_path)}
if completed.returncode == 0:
    actual = json.loads(completed.stdout)
    record["实际安全守卫"] = actual["CPU审核安全守卫"]
    print(json.dumps({"实际退出码": completed.returncode, "CPU守卫": record["实际安全守卫"],
        "50普通成员": len(actual["50tar普通逐成员核验"]), "HF源成员": len(actual["SOURCE82_before"]),
        "200与303完整独立审核": [actual["真实200完整状态独立CPU审核"]["全局实际轮次"], actual["真实303完整状态独立CPU审核"]["全局实际轮次"]],
        "CPU304原证据+inf参数和元素": [actual["CPU304正无穷二阶动量参数数"], actual["CPU304正无穷二阶动量元素数"]],
        "纯数学Float32平方": actual["纯Float32平方验证"], "原303相对200收据漂移项": list(actual["303相对收据两原件漂移"])}, ensure_ascii=False))
else:
    print(completed.stderr, end="", file=sys.stderr)
with (here / (prefix + ".command.json")).open("x", encoding="utf-8") as stream:
    json.dump(record, stream, ensure_ascii=False, allow_nan=False, indent=2)
    stream.write("\n")
print("RAW_EVIDENCE " + str(here / (prefix + ".command.json")))
raise SystemExit(completed.returncode)
