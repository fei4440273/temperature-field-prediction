"""Finalize only this independent CPU audit's new evidence; frozen material is read-only."""

import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("freeze_review_readonly", HERE / "只读冻包复核.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
energy, hf = review.imports()
assert Path.cwd() == review.ROOT
assert all(os.environ.get(k) == v for k, v in review.ENV.items())
started, clock = review.now(), time.perf_counter()
before85 = review.current_map(energy.SOURCE_MEMBERS)
before82 = review.current_map(hf.SOURCE_MEMBERS)
own_files = ("只读冻包复核.py", "交付现场再核.py", "中文冻包独审报告.md", "冻包独审机器证据.json",
             "冻包独审机器证据_恢复复核1.json", "冻包独审机器证据_完整终核.json", "外层命令真实终态与自身错误补存.json")
own_before = {name: review.metadata(HERE / name) for name in own_files}
errors = []
command = review.execute([review.PYTHON, "-B", str(HERE / "只读冻包复核.py"), "pack"])
try:
    assert command["实际退出码"] == 0
    pack = json.loads(command["stdout"])
    final = json.loads((HERE / "冻包独审机器证据_完整终核.json").read_text(encoding="utf-8"))
    assert final["本轮内部核验结论退出码"] == 0 and final["实际错误原样"] == []
    assert before85 == final["SOURCE85前"] == final["SOURCE85后"] == pack["SOURCE85"]
    assert before82 == pack["SOURCE82"]
    assert review.canonical(before85) == review.MAP_SHA
    assert (own_before["冻包独审机器证据_完整终核.json"]["字节"],
            own_before["冻包独审机器证据_完整终核.json"]["SHA256"]) == (171511, "4d3766c192c0a409fc831f6bd8b3fd14fe060b82a6a32123aade1d83e2d5aefa")
    assert own_before["只读冻包复核.py"]["SHA256"] == "96152934c1c3461c2e548f968c3c4075998bcd393800c658722e27088319df49"
    assert json.loads(final["真实子命令"]["pureCPU门禁fixture"]["stdout"])["门禁fixture数"] == 16
    assert json.loads(final["真实子命令"]["pureCPU严格合同类型fixture"]["stdout"])["严格合同负例数"] == 7
    callbacks = json.loads(final["真实子命令"]["真实CLI56六SHA动态负门禁"]["stdout"])
    assert len(callbacks["禁止回调计数"]) == 21 and set(callbacks["禁止回调计数"].values()) == {0}
    assert final["真实子命令"]["真实CLI56六SHA动态负门禁"]["实际退出码"] == 1
    assert callbacks["输出目录执行前存在"] is False and callbacks["输出目录执行后存在"] is False
except BaseException:
    errors.append(traceback.format_exc())
after85, after82 = review.current_map(energy.SOURCE_MEMBERS), review.current_map(hf.SOURCE_MEMBERS)
own_after = {name: review.metadata(HERE / name) for name in own_files}
if before85 != after85 or before82 != after82 or own_before != own_after:
    errors.append("source85/source82/own archival material drift")
six = review.actual_six(energy, hf)
evidence = {"结构版本": 1, "实际起止": [started, review.now()], "PID": os.getpid(), "父PID": os.getppid(),
    "墙钟秒": time.perf_counter()-clock, "工作目录": str(review.ROOT), "实际环境": review.ENV,
    "交付内部实际退出码": int(bool(errors)), "实际错误原样": errors, "真实再核子命令": command,
    "SOURCE85前": before85, "SOURCE85后": after85, "SOURCE82前": before82, "SOURCE82后": after82,
    "SOURCE85规范JSON摘要SHA256": review.canonical(after85), "SOURCE82规范JSON摘要SHA256": review.canonical(after82),
    "本独审新档案前": own_before, "本独审新档案后": own_after, "SOURCE85及82前后漂移数": int(before85 != after85) + int(before82 != after82),
    "ROOT当前_仅上下文": review.metadata(review.ROOT / hf.ROOT_LEDGER),
    "本人父HF四SHA活动": hf._root_active(review.ROOT / hf.ROOT_LEDGER, {k: six[k] for k in hf.IDENTITY_FIELDS}),
    "本人能源六SHA活动": energy.root_active(review.ROOT / hf.ROOT_LEDGER, six),
    "真实五HF_十能源_GPU_温度_PT_新实测_工程安全_B0_总目标完成": False}
with (HERE / "交付现场再核机器证据.json").open("x", encoding="utf-8") as stream:
    json.dump(evidence, stream, ensure_ascii=False, allow_nan=False, indent=2)
    stream.write("\n")
print(json.dumps({"交付内部实际退出码": int(bool(errors)), "实际再核子命令退出码": command["实际退出码"],
    "SOURCE85": len(after85), "SOURCE82": len(after82), "SOURCE85摘要": review.canonical(after85),
    "前后源码漂移数": evidence["SOURCE85及82前后漂移数"], "父HF活动": evidence["本人父HF四SHA活动"],
    "能源本人活动": evidence["本人能源六SHA活动"], "独审中文报告": own_after["中文冻包独审报告.md"],
    "外层真实补存": own_after["外层命令真实终态与自身错误补存.json"],
    "交付机器证据": review.metadata(HERE / "交付现场再核机器证据.json"), "实际错误原样": errors}, ensure_ascii=False, allow_nan=False))
raise SystemExit(int(bool(errors)))
