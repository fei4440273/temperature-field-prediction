"""Mechanically bind actual independent CPU runs and a handwritten review report."""

import json
import os
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path

import torch
import obs_review_guard as guard
import run_independent_cpu as runner


HERE = Path(__file__).resolve().parent
ROOT = runner.ROOT
GROUPS = ("全部93源码", "父闭包87", "父HF82", "旧OBS72", "能源自身3", "候选自身3", "固定父原件")


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def identity(path):
    return runner._identity(str(path))


def main():
    assert Path.cwd().resolve() == ROOT
    assert sys.executable == "/home/phl/anaconda3/envs/PINN/bin/python" and sys.flags.dont_write_bytecode
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert not torch.cuda.is_initialized()
    output = HERE / "独立复核汇总机器证据.json"
    assert not output.exists()
    specs = [
        ("独审本人83", 0, "83 passed in 41.41s"),
        ("独立追加IO病例", 1, "11 passed, 18 errors in 3.33s"),
        ("独立追加IO病例_修对照", 0, "29 passed in 39.06s"),
    ]
    runs = []
    records = []
    for name, code, terminal in specs:
        command_path = HERE / (name + ".command.json")
        record = read_json(command_path)
        assert record["实际退出码"] == code and terminal in record["stdout"]
        assert all(record["前后逐件一致"].values())
        raw = {suffix: HERE / (name + suffix) for suffix in (".stdout.txt", ".stderr.txt", ".before.json", ".after.json")}
        assert raw[".stdout.txt"].read_text(encoding="utf-8") == record["stdout"]
        assert raw[".stderr.txt"].read_text(encoding="utf-8") == record["stderr"]
        assert read_json(raw[".before.json"]) == record["before"]
        assert read_json(raw[".after.json"]) == record["after"]
        for group in GROUPS:
            assert record["before"][group] == record["after"][group]
        match = re.search(r"INDEPENDENT_CPU_SAFETY (\{[^\n]+\})", record["stdout"])
        assert match
        safety = json.loads(match.group(1))
        assert safety["pytest_exitstatus"] == code
        assert safety["cuda_initialized"] is False
        assert safety["cuda_initialization_attempts"] == safety["outside_data_io_denied"] == 0
        runs.append({"名称": name, "实际退出码": code, "实际终态输出": terminal,
            "开始": record["开始时间"], "终止": record["结束时间"], "墙钟秒": record["墙钟秒"],
            "完整实际pytest命令": record["完整实际pytest命令"], "外层完整实际argv": record["外层完整实际argv"],
            "完整实际环境": record["实际环境"], "CPU守卫实际终态": safety,
            "前后逐件一致": record["前后逐件一致"], "完整raw命令证据": identity(command_path),
            "独立raw文件": {suffix: identity(path) for suffix, path in raw.items()},
            "真实ROOT历史上下文_before": record["before"]["真实ROOT当前身份_只作上下文允许根授权追加"],
            "真实ROOT历史上下文_after": record["after"]["真实ROOT当前身份_只作上下文允许根授权追加"]})
        records.append(record)
    fresh = runner._snapshot()
    same = {group: all(fresh[group] == record["before"][group] == record["after"][group] for record in records) for group in GROUPS}
    assert all(same.values()) and fresh["候选自身3"] == runner.EXPECTED_CANDIDATE
    assert (len(fresh["全部93源码"]), len(fresh["父闭包87"]), len(fresh["父HF82"]),
            len(fresh["旧OBS72"]), len(fresh["能源自身3"]), len(fresh["候选自身3"])) == (93, 87, 82, 72, 3, 3)
    source90 = {name: fresh["全部93源码"][name] for name in runner.obs.SOURCE_MEMBERS}
    assert len(source90) == 90 and not set(source90).intersection(fresh["能源自身3"])
    author_dir = ROOT / "研究记录/任务11_外部对照/Howard三模态组件_CPU验证_20260916T200802+0800"
    author_machine = author_dir / "CPU验证完整证据.json"
    author_report = author_dir / "中文组件核验与边界.md"
    author = read_json(author_machine)
    author_report_text = author_report.read_text(encoding="utf-8")
    assert author_report_text
    author_sources = author["SOURCE闭包"]["源码前后鲜验"]
    assert author_sources["新三源SHA256"] == runner.EXPECTED_CANDIDATE
    assert author_sources["既有闭包SHA256"] == fresh["父闭包87"]
    author_stages = author["TDD实际命令与输出"]
    assert author_stages[-1]["退出码"] == 0 and "83 passed in 42.30s" in author_stages[-1]["标准输出"]
    supplement_path = HERE / "Float32诊断首轮补录.json"
    supplement = read_json(supplement_path)
    assert supplement["实际退出码"] == 1 and "AssertionError" in supplement["stderr"]
    diagnostic_path = HERE / "Float32求和根因实际证据.json"
    diagnostic = read_json(diagnostic_path)
    assert diagnostic["实际退出码"] == 0
    diagnostic_records = diagnostic["记录"]
    assert len(diagnostic_records) == 24
    assert len({str(Path(row["合成CSV"]).resolve()) for row in diagnostic_records}) == 12
    assert all(row["Float64理想归一最大绝对差"] < 3e-8 and row["原帧Float64权重和对1偏差"] < 1e-6 for row in diagnostic_records)
    diagnostic_command = shlex.join(["env", *(f"{key}={value}" for key, value in records[1]["实际环境"].items()), *supplement["实际argv"]])
    final_stdout = records[-1]["stdout"]
    for line in ("REAL_SYNTHETIC_IO_COUNTS 7272 752", "TEN_SYNTHETIC_PT_REAL_LOADS_STRICT_THEN_CPU_MOVE 10",
                 "TRUE_CPU_WRITER_PINS 107 85 182 OUTPUTS 8"):
        assert line in final_stdout
    retained = [line for line in final_stdout.splitlines() if line.startswith("REAL_WRITER_DRIFT_REJECTED_RETAINED")]
    assert len(retained) == 8
    assert runs[-1]["CPU守卫实际终态"]["synthetic_torch_loads"] == 12
    assert runs[-1]["CPU守卫实际终态"]["synthetic_parquet_reads"] == 2
    archive_files = {path.name: identity(path) for path in sorted(HERE.iterdir()) if path.is_file()}
    own_env = {key: os.environ[key] for key in runner.ENV_KEYS}
    payload = {"结构版本": 1, "记录时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "复核结论": "本轮有界独立CPU代码复核通过，未发现新增确定P1/P2；不是正式观察/能源/训练/GPU或全目标完成",
        "技能边界": {"已完整读取": ["requesting-code-review", "code-reviewer.md", "verification-before-completion",
            "systematic-debugging", "root-cause-tracing.md"], "制定计划技能使用": False,
            "再次分派独立静态review": "实际失败：collab spawn failed: agent thread limit reached，不计通过证据"},
        "实际调用三轮": runs, "最终通过两轮项数": [83, 29], "合计112并非一次实跑": True,
        "自身失败如实保留": {"追加首轮": runs[1], "初诊补录": identity(supplement_path),
            "初诊完整规范化实际命令": diagnostic_command, "初诊actualexit": 1,
            "初诊stdout": supplement["stdout"], "初诊stderr": supplement["stderr"],
            "成功诊断完整规范化实际命令": diagnostic_command, "成功诊断actualexit": 0,
            "成功诊断结构化实际终态stdout": identity(diagnostic_path),
            "成功诊断stderr": "", "唯一原始CSV数": 12, "含currentalias路径记录数": 24,
            "夹具原件": identity(HERE / "独立首轮夹具原件.py"),
            "执行器首轮原件": identity(HERE / "CPU执行器首轮原件.py"),
            "初诊脚本原件": identity(HERE / "Float32诊断首轮脚本原件.py"),
            "只修自身对照不改候选或原frame_weight": True},
        "封存现场完整源码及父六件身份": fresh, "三轮及封存现场逐件一致": same,
        "SOURCE90逐成员SHA": source90,
        "候选自身3完整身份": {name: identity(ROOT / name) for name in runner.EXPECTED_CANDIDATE},
        "作者档案fresh实际读取绑定": {"机器证据": identity(author_machine), "中文边界": identity(author_report),
            "现场完整JSON解析成功": True, "作者最终阶段": author_stages[-1],
            "作者历史RED不作为本人新RED": True},
        "真实CPUwriter与漂移实际终态": {"静态pins": 107, "附加pins": 85, "去重全部pins": 182,
            "真实输出文件数": 8, "manifest逐件封7输出": True, "合成十态逐点行": 80240,
            "功率时间径向行": [90, 270, 90], "写前实际字节漂移病例数": 7,
            "真实CSV写后及manifest实际hash中漂移病例数": 8, "实际失败输出保留原文": retained,
            "HF资格只接合成API及原件不授正式资格": True, "公开CPUwriter空前后资格与空CUDAreceipt": True,
            "反序列化合成合法模型数": 10, "合成非法dtype拒绝模型数": 2},
        "wholeROOT语义": "真实ROOT跨轮只作历史上下文，根授权追加不判冻结源码漂移；合成ROOT同一短审计窗仍严格wholeSHA拒绝变化",
        "写入范围": str(HERE), "临时与缓存范围": own_env,
        "禁止行为均未执行": ["正式PT/真实PARQUET/真实温度/固定TEST读取", "CUDA初始化", "正式OBS或能源或训练",
            "候选及父源码改动", "ROOT/status/continuation及三全局写入", "父冻结YAMLtarcatalog写入", "安装", "删除", "计划扩展"],
        "尚未授予许可": "正式OBS待本人5HF全完训及best/final完整own资格，再由根OBS六SHA活动ROOT明确授权；非十态能源/安全/内部真值/作者exact验收",
        "同目录不可变材料完整身份": archive_files,
        "本机械封存完整规范化实际命令": shlex.join(["env", *(f"{key}={value}" for key, value in own_env.items()), sys.executable, "-B", *sys.argv]),
        "封存CPU守卫": {**guard.COUNTS, "cuda_initialized": torch.cuda.is_initialized()}}
    assert all(payload["三轮及封存现场逐件一致"].values())
    assert payload["封存CPU守卫"]["cuda_initialized"] is False
    assert guard.COUNTS["cuda_initialization_attempts"] == guard.COUNTS["outside_data_io_denied"] == 0
    runner._json_new(output, payload)
    assert read_json(output) == payload
    print("ARCHIVE_SOURCE_AND_PARENT_ZERO_DRIFT " + json.dumps(same, ensure_ascii=False, sort_keys=True))
    print("INDEPENDENT_REVIEW_ARCHIVE " + json.dumps(identity(output), ensure_ascii=False))
    print("INDEPENDENT_REVIEW_REPORT " + json.dumps(identity(HERE / "独立复核中文报告.md"), ensure_ascii=False))
    print("CANDIDATES_FRESH " + json.dumps(payload["候选自身3完整身份"], ensure_ascii=False))
    print("AUTHOR_ARCHIVE_FRESH " + json.dumps({key: payload["作者档案fresh实际读取绑定"][key] for key in ("机器证据", "中文边界")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
