"""Exclusive evidence capture for independent synthetic CPU OBS review."""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from sic_cu.config import PROJECT_ROOT
from sic_cu.eval import task11_howard_observations as obs
from sic_cu.eval import task11_howard_hf_energy as energy
from sic_cu.eval import task11_mlp_observations as old
from sic_cu.train import task11_howard_hf_formal as hf


ROOT = PROJECT_ROOT.resolve()
REVIEW = Path(__file__).resolve().parent
ENERGY_THREE = sorted(set(energy.SOURCE_MEMBERS) - set(hf.SOURCE_MEMBERS))
PARENT87 = sorted(set(hf.SOURCE_MEMBERS) | set(old.SOURCE_MEMBERS))
ALL93 = sorted(set(obs.SOURCE_MEMBERS) | set(ENERGY_THREE))
FIXED_FILES = (hf.REGISTRY, hf.SOURCE_TAR, hf.LF_CATALOG, hf.HF_DATA_CATALOG,
               energy.ENERGY_REGISTRY, energy.ENERGY_SOURCE_TAR)
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTHONPATH", "TMPDIR", "TMP", "TEMP", "CUDA_CACHE_PATH",
    "XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR")
EXPECTED_CANDIDATE = {
    "src/sic_cu/eval/task11_howard_observations.py": "0db2f84797d27f5933e8af46dfc2df820ce0e18be512c7a6e5a9e9ad2b5ff807",
    "scripts/58_export_task11_howard_observations.py": "11ba79106d63fb2040b37ee6d3be18e83fd31eeeb81913718399df476f528368",
    "tests/test_task11_howard_observations.py": "f6092c84f442939f3c849bf83054f395a873e00690643b5c61b52e34f680cae3",
}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _map(names):
    return {name: _sha(hf._path(name, ROOT)) for name in names}


def _json_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def _identity(name):
    path = hf._path(name, ROOT)
    return {"路径": str(path), "字节": path.stat().st_size, "SHA256": _sha(path)}


def _snapshot():
    return {"全部93源码": _map(ALL93), "父闭包87": _map(PARENT87), "父HF82": _map(hf.SOURCE_MEMBERS),
        "旧OBS72": _map(old.SOURCE_MEMBERS), "能源自身3": _map(ENERGY_THREE),
        "候选自身3": _map(EXPECTED_CANDIDATE), "固定父原件": {name: _identity(name) for name in FIXED_FILES},
        "真实ROOT当前身份_只作上下文允许根授权追加": _identity(hf.ROOT_LEDGER)}


def main():
    assert len(sys.argv) == 2 and sys.argv[1] in {"83", "independent"}
    assert Path.cwd().resolve() == ROOT
    assert sys.executable == "/home/phl/anaconda3/envs/PINN/bin/python" and sys.flags.dont_write_bytecode
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert (len(PARENT87), len(hf.SOURCE_MEMBERS), len(old.SOURCE_MEMBERS), len(obs.SOURCE_MEMBERS), len(ENERGY_THREE), len(ALL93)) == (87, 82, 72, 90, 3, 93)
    phase = sys.argv[1]
    prefix = REVIEW / ("独审本人83" if phase == "83" else "独立追加IO病例")
    basetemp = REVIEW / ("pytest临时_本人83" if phase == "83" else "pytest临时_独立IO")
    assert not basetemp.exists()
    before = _snapshot()
    assert before["候选自身3"] == EXPECTED_CANDIDATE
    _json_new(prefix.with_suffix(".before.json"), before)
    test = str(ROOT / "tests/test_task11_howard_observations.py") if phase == "83" else str(REVIEW / "test_independent_obs_review.py")
    argv = [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:logging", "-p", "obs_review_guard",
        "--capture=sys", "-o", "addopts=", "--basetemp=" + str(basetemp), test, "-rA"]
    env = {key: os.environ[key] for key in ENV_KEYS}
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    clock = time.perf_counter()
    completed = subprocess.run(argv, cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - clock
    with prefix.with_suffix(".stdout.txt").open("x", encoding="utf-8") as stream:
        stream.write(completed.stdout)
    with prefix.with_suffix(".stderr.txt").open("x", encoding="utf-8") as stream:
        stream.write(completed.stderr)
    after = _snapshot()
    _json_new(prefix.with_suffix(".after.json"), after)
    groups = ("全部93源码", "父闭包87", "父HF82", "旧OBS72", "能源自身3", "候选自身3", "固定父原件")
    zero = {name: before[name] == after[name] for name in groups}
    record = {"结构版本": 1, "证据性质": "本人独立CPU纯合成真实pytest，不授正式OBS或能源/训练权限",
        "外层完整实际argv": [sys.executable, "-B", *sys.argv],
        "完整实际pytest命令": shlex.join(["env", *(f"{key}={value}" for key, value in env.items()), *argv]),
        "实际环境": env, "工作目录": str(ROOT), "开始时间": started,
        "结束时间": datetime.now().astimezone().isoformat(timespec="seconds"), "实际退出码": completed.returncode,
        "墙钟秒": elapsed, "stdout": completed.stdout, "stderr": completed.stderr,
        "前后逐件一致": zero, "before": before, "after": after,
        "wholeROOT授权追加不是源码漂移": True, "独立审查守卫": _identity(str(REVIEW / "obs_review_guard.py")),
        "未执行真实十态OBS或能源或训练或GPU": True, "失败目录原样保留": True}
    _json_new(prefix.with_suffix(".command.json"), record)
    print(completed.stdout, end="")
    print(completed.stderr, end="", file=sys.stderr)
    print("INDEPENDENT_SOURCE_PINS_UNCHANGED " + json.dumps(zero, ensure_ascii=False, sort_keys=True))
    print("COMMAND_EVIDENCE " + json.dumps(_identity(str(prefix.with_suffix(".command.json"))), ensure_ascii=False))
    assert all(zero.values()), "Real current source/frozen parent drift"
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
