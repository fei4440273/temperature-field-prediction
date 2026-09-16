"""Read-only final check of saved CPU review evidence and current source pins."""

import json
from pathlib import Path

import torch
import obs_review_guard as guard
import run_independent_cpu as runner


here = Path(__file__).resolve().parent
with (here / "独立复核汇总机器证据.json").open(encoding="utf-8") as stream:
    saved = json.load(stream)
with (here / "封存实际终态补录.json").open(encoding="utf-8") as stream:
    terminal = json.load(stream)
assert terminal["实际退出码"] == 0
for item in terminal["实际终态stdout要点"].values():
    if "文件" in item:
        actual = runner._identity(str(here / item["文件"]))
        assert (actual["字节"], actual["SHA256"]) == (item["字节"], item["SHA256"])
for name, initial in saved["同目录不可变材料完整身份"].items():
    assert runner._identity(str(here / name)) == initial
fresh = runner._snapshot()
groups = ("全部93源码", "父闭包87", "父HF82", "旧OBS72", "能源自身3", "候选自身3", "固定父原件")
same = {name: fresh[name] == saved["封存现场完整源码及父六件身份"][name] for name in groups}
assert all(same.values())
assert fresh["候选自身3"] == runner.EXPECTED_CANDIDATE
top_files = [path for path in here.iterdir() if path.is_file()]
assert top_files and all(path.stat().st_mode & 0o222 == 0 for path in top_files)
assert not torch.cuda.is_initialized()
assert guard.COUNTS["cuda_initialization_attempts"] == guard.COUNTS["outside_data_io_denied"] == 0
names = ("独立复核中文报告.md", "独立复核汇总机器证据.json", "独审本人83.command.json",
         "独立追加IO病例.command.json", "独立追加IO病例_修对照.command.json", "Float32诊断首轮补录.json",
         "Float32求和根因实际证据.json", "封存实际终态补录.json", "封存只读终验.py")
print("FINAL_READONLY_CPU_REVIEW " + json.dumps({"源码及固定父六件zero_drift": same,
    "保存终态": [(item["名称"], item["实际退出码"], item["实际终态输出"]) for item in saved["实际调用三轮"]],
    "顶层只读文件数": len(top_files), "核心鲜验身份": {name: runner._identity(str(here / name)) for name in names},
    "候选三源鲜验": {name: runner._identity(str(runner.ROOT / name)) for name in runner.EXPECTED_CANDIDATE},
    "GPU或真实模型数据运行许可": False, "终验CUDA_initialized": torch.cuda.is_initialized()},
    ensure_ascii=False, sort_keys=True))
