"""Capture full real terminal output without a JavaScript JSON round trip."""

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
command = ["/home/phl/anaconda3/envs/PINN/bin/python", "-B", str(HERE / "independent_success_013_audit.py")]
assert sys.executable == command[0] and sys.flags.dont_write_bytecode
assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "1"
started = time.perf_counter()
result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
for name, data in (("原真实资格进程完整stdout.txt", result.stdout), ("原真实资格进程完整stderr.txt", result.stderr)):
    with (HERE / name).open("xb") as stream:
        stream.write(data)
terminal = {"实际子进程命令argv": command, "工作目录": str(ROOT), "实际exit_code": result.returncode,
            "实际完整stdout": result.stdout.decode("utf-8"), "实际完整stderr": result.stderr.decode("utf-8"),
            "原stdout字节": len(result.stdout), "原stderr字节": len(result.stderr),
            "原stdoutSHA256": hashlib.sha256(result.stdout).hexdigest(),
            "原stderrSHA256": hashlib.sha256(result.stderr).hexdigest(),
            "外层实际墙钟秒": time.perf_counter() - started, "北京时间": datetime.now().astimezone().isoformat(),
            "原Python数字表示保留": True,
            "实际外层环境": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                                "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                                                                "TMPDIR", "TMP", "TEMP", "CUDA_CACHE_PATH", "XDG_CACHE_HOME", "MPLCONFIGDIR",
                                                                "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "WORLD_SIZE", "RANK", "LOCAL_RANK")}}
with (HERE / "真实原资格命令与完整进程终态.json").open("x", encoding="utf-8") as stream:
    json.dump(terminal, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps(terminal, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
sys.exit(result.returncode)
