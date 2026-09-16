"""Mechanical archival binding or read-only final verification; no model loads."""

import json
import math
import os
import shlex
import sys
from pathlib import Path

import independent_failure_audit as audit


here = Path(__file__).resolve().parent
record_path = here / "独立CPU只读实跑.command.json"
report_path = here / "独立失败证据中文审核报告.md"
output = here / "独立失败证据审核机器证据.json"
record = audit.read_json(record_path)
assert record["实际退出码"] == 0 and record["完整stderr"] == ""
stdout_path = here / "独立CPU只读实跑.stdout.txt"
stderr_path = here / "独立CPU只读实跑.stderr.txt"
assert record["完整stdout"] == stdout_path.read_text(encoding="utf-8")
assert record["完整stderr"] == stderr_path.read_text(encoding="utf-8")
proof = json.loads(record["完整stdout"])
fresh_files = audit.fresh_files()
fresh_source = {name: audit.sha(audit.ROOT / name) for name in audit.hf.SOURCE_MEMBERS}
assert fresh_files == proof["50当前原件beforeSHA"] == proof["50当前原件afterSHA"]
assert fresh_source == proof["SOURCE82_before"] == proof["SOURCE82_after"]
assert len(fresh_files) == 50 and len(fresh_source) == 82
for path, expected in audit.EXPECTED.items():
    assert audit.sha(path) == expected
parent_fresh = {name: audit.identity(audit.ROOT / name) for name in proof["原HF四身份"]}
assert parent_fresh == proof["原HF四身份"]
args = proof["真实seed2子START"]["真实CLI55参数"]
receipt = proof["最后合法200收据"]
for index, (key, name) in enumerate(zip(audit.hf.IDENTITY_FIELDS, parent_fresh)):
    assert parent_fresh[name]["SHA256"] == receipt["身份"][key]
    flag = ("--registry-sha", "--source-tar-sha", "--lf-catalog-sha", "--hf-data-catalog-sha")[index]
    assert args[args.index(flag) + 1] == parent_fresh[name]["SHA256"]
cost_fields = ("训练墙钟秒", "加载构建恢复墙钟秒", "导出核源墙钟秒")
assert all(type(receipt[key]) is float and math.isfinite(receipt[key]) and receipt[key] > 0 for key in cost_fields)
assert math.isclose(receipt["本会话总墙钟秒"], sum(receipt[key] for key in cost_fields), rel_tol=1e-12, abs_tol=1e-7)
assert proof["原GPU失败额外训练墙钟秒"] is proof["五seedmean"] is proof["五seedstd_ddof1"] is None
assert audit.COUNTS["实际CPU_PT反序列化数"] == 0 and all(value == 0 for value in audit.COUNTS.values())
assert not audit.torch.cuda.is_initialized()
check = len(sys.argv) == 2 and sys.argv[1] == "--check"
assert check or len(sys.argv) == 1
if check:
    saved = audit.read_json(output)
    for name, initial in saved["本目录证据完整绑定"].items():
        current = audit.identity(here / name)
        assert current["SHA256"] == initial["SHA256"] and current["字节"] == initial["字节"]
    assert saved["SOURCE82终验SHA"] == fresh_source and saved["50原件终验SHA"] == fresh_files
    assert all(path.stat().st_mode & 0o222 == 0 for path in here.iterdir() if path.is_file())
    print("READONLY_FINAL " + json.dumps({"报告": audit.identity(report_path), "机器证据": audit.identity(output),
        "raw实命令": audit.identity(record_path), "50原件和SOURCE82及HF四件终验零漂移": True,
        "指定三原件SHA吻合": True, "CPU_PT反序列化本终验0": True, "CUDA_initialized": False,
        "未启动训练或其他任务": True}, ensure_ascii=False))
else:
    assert not output.exists()
    names = ("independent_failure_audit.py", "run_bounded_readonly.py", "seal_bounded_review.py",
             "独立失败证据中文审核报告.md", "独立CPU只读实跑.command.json", "独立CPU只读实跑.stdout.txt", "独立CPU只读实跑.stderr.txt")
    environment = record["实际环境"]
    value = {"性质": "独立只读有界seed2失败证据审核封存，非正式训练完成、模型复现、能源/OBS或全目标验收",
        "独立主要实际命令": record["完整实际命令"], "独立主要实际退出码": record["实际退出码"],
        "独立主要实际墙钟秒": record["墙钟秒"], "本目录证据完整绑定": {name: audit.identity(here / name) for name in names},
        "原三指定物料终验身份": {path.name: audit.identity(path) for path in audit.EXPECTED},
        "SOURCE82终验SHA": fresh_source, "50原件终验SHA": fresh_files, "原HF四身份终验": parent_fresh,
        "200与303完整独立审核": [proof["真实200完整状态独立CPU审核"], proof["真实303完整状态独立CPU审核"]],
        "原CPU304+inf二阶动量参数与元素": [proof["CPU304正无穷二阶动量参数数"], proof["CPU304正无穷二阶动量元素数"]],
        "原GPU失败+inf计数未知": None, "原GPU失败结束时间未知": None, "原GPU失败额外墙钟未知": None,
        "五seedmean": None, "五seedstd_ddof1": None, "五完整HF十能源十OBS条件未满足": True,
        "原200已封口HF成本": {key: receipt[key] for key in (*cost_fields, "本会话总墙钟秒", "峰值真实CUDA显存字节")},
        "原200成本分项算术实际核验": True, "独立主要安全守卫": proof["CPU审核安全守卫"],
        "没有重跑304训练或GPU": True, "HF四SHA严格等于原收据身份及原CLI55参数": True,
        "ROOT和三全局未写或用作长时锁": True, "其他任务启动": False,
        "本机械封存完整规范化实际命令": shlex.join(["env", *(f"{key}={item}" for key, item in environment.items()), sys.executable, "-B", *sys.argv])}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    assert audit.read_json(output) == value
    print("SEALED_BOUNDED_REVIEW " + json.dumps({"报告": audit.identity(report_path), "机器证据": audit.identity(output),
        "raw实命令": audit.identity(record_path), "源和原件零漂移": True, "没有重跑304或GPU": True}, ensure_ascii=False))
