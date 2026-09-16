"""Capture only the two pure evidence-closing commands, never a training command."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
targets = {"finalize": ("finalize_013_evidence.py", "汇总准确身份"),
           "seal": ("seal_013_archive.py", "机械0444封存验证")}
target, prefix = targets[sys.argv[1]]
command = ["/home/phl/anaconda3/envs/PINN/bin/python", "-B", str(HERE / target)]
assert sys.executable == command[0] and sys.flags.dont_write_bytecode
assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "1"
started = time.perf_counter()
result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
names = []
for suffix, data in (("完整stdout.txt", result.stdout), ("完整stderr.txt", result.stderr)):
    name = prefix + suffix
    with (HERE / name).open("xb") as stream:
        stream.write(data)
    names.append(name)
terminal = {"实际命令argv": command, "外层argv": sys.argv, "实际工作目录": str(ROOT), "actual_exit_code": result.returncode,
            "实际完整stdout": result.stdout.decode("utf-8"), "实际完整stderr": result.stderr.decode("utf-8"),
            "实际stdout字节": len(result.stdout), "实际stderr字节": len(result.stderr),
            "实际stdoutSHA256": hashlib.sha256(result.stdout).hexdigest(),
            "实际stderrSHA256": hashlib.sha256(result.stderr).hexdigest(), "实际外层墙钟秒": time.perf_counter() - started,
            "完整resolved实际环境": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                                       "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                                                                       "TMPDIR", "TMP", "TEMP", "CUDA_CACHE_PATH", "XDG_CACHE_HOME", "MPLCONFIGDIR",
                                                                       "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "WORLD_SIZE", "RANK", "LOCAL_RANK")}}
name = prefix + "实际命令与完整终态.json"
with (HERE / name).open("x", encoding="utf-8") as stream:
    json.dump(terminal, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write("\n")
names.append(name)
for name in names:
    os.chmod(HERE / name, 0o444)
print(json.dumps(terminal, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
sys.exit(result.returncode)
