"""Capture real RED/GREEN of the own new audit hook, with no deletion or model reads."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
assert len(sys.argv) == 2 and sys.argv[1] in {"RED", "GREEN"}
argv = [sys.executable, "-B", str(here / "test_guard_dir_fd_readonly.py")]
keys = ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTHONPATH", "TMPDIR", "TMP", "TEMP", "CUDA_CACHE_PATH",
        "XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR")
environment = {key: os.environ[key] for key in keys}
completed = subprocess.run(argv, cwd=here.parents[2], env=os.environ.copy(), capture_output=True, text=True, check=False)
value = {"性质": "仅自身档案guard相对dir_fd路径回归，不实际unlink/训练/模型读取", "阶段": sys.argv[1],
    "完整实际命令": shlex.join(["env", *(f"{key}={item}" for key, item in environment.items()), *argv]),
    "实际退出码": completed.returncode, "完整stdout": completed.stdout, "完整stderr": completed.stderr}
with (here / ("自身guard_dir_fd_" + sys.argv[1] + ".command.json")).open("x", encoding="utf-8") as stream:
    json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
    stream.write("\n")
print(completed.stdout, end="")
print(completed.stderr, end="", file=sys.stderr)
raise SystemExit(completed.returncode)
