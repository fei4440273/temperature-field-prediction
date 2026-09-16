"""Bind fresh immutable audit evidence; never load models or rerun qualification."""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import independent_success_013_audit as own


HERE, ROOT, hf = own.HERE, own.ROOT, own.hf


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def main():
    machine = read("独立成功013资格原Python机器证据.json")
    cross = read("纯收据前缀成本交叉核对与原q字符串保真.json")
    original = read("真实原资格命令与完整进程终态.json")
    closing = read("纯收尾实际命令与完整终态.json")
    tool = read("原真实工具终态绑定.json")
    for actual, key in ((original, "原完整资格"), (closing, "纯收尾")):
        assert actual["实际exit_code"] == tool[key]["actual_exit_code"] == 0
        assert actual["实际完整stderr"] == "" and actual["原stderr字节"] == 0
    forbidden = ("HF4访问拒绝尝试", "CUDA初始化尝试", "训练或optimizer_step尝试", "非许可PT反序列化尝试",
                 "非许可Parquet打开尝试", "完整Parquet或温度列读取尝试", "非许可Parquet_schema尝试",
                 "非许可Parquet_scan尝试", "档案外写入尝试")
    observed = {key: machine["守卫计数_before最终JSON写出"].get(key, 0) for key in forbidden}
    assert all(value == 0 for value in observed.values())
    pt_kinds = Counter("HF013" if "/正式Howard_HF_seed" in item["路径"] else "LF01234前提"
                       for item in machine["原PT真实CPU读入记录"])
    assert pt_kinds == {"HF013": 208, "LF01234前提": 624}
    projected = Counter(item["类别"] for item in machine["真实Parquet结构或LF行数IO"])
    assert projected == {"LF训练验证仅行数": 4200, "HF开发结构": 12}
    state_counts = Counter(item["seed"] for item in machine["原所有完整HF状态实际审核记录"])
    assert state_counts == {0: 55, 1: 27, 3: 45}
    assert len(machine["原所有LF资格实际审核记录"]) == 30
    assert machine["守卫计数_before最终JSON写出"]["原全56四RNG阶段边界真实审核数"] == 3
    assert machine["守卫计数_before最终JSON写出"]["原HF结果view真实审核数"] == 52
    report = (HERE / "独立成功013完整资格中文报告.md").read_text(encoding="utf-8")
    for text in ("五HF mean/std = NULL", "4212", "全项目目录枚举", "不独立诊断HF4", "任务", "停止"):
        assert text in report
    seeds = {}
    hf_maps = {}
    for seed in own.SEEDS:
        q_text = (HERE / f"seed{seed}_原Python完整资格q.json").read_text(encoding="utf-8")
        assert q_text == machine["结果"][str(seed)]["完整原PythonqJSON字符串"] + "\n"
        q = json.loads(q_text)
        assert q["已完成"] is True and q["原文精确三网联合训练复现"] is False and q["HF训练许可"] is False
        assert cross["结果"][str(seed)]["原qJSON字符串逐字相同"] is True
        assert len(cross["结果"][str(seed)]["全收据"]) == q["会话数"]
        points_times = q["事前来源"]["固定PHQH查询"]["时刻_秒"]
        assert points_times == [5.0, 100.0, 200.0] and all(type(value) is float for value in points_times)
        current = own.ordinary_map(own.RUN_BASE / f"正式Howard_HF_seed{seed}")
        assert current == q["真实工件SHA256"] == machine["成功HF工件_before"][str(seed)] == machine["成功HF工件_after"][str(seed)]
        hf_maps[str(seed)] = current
        seeds[str(seed)] = {"普通工件数": len(current), "全局实际轮次": q["全局实际轮次"],
                           "阶段1实际轮次": q["阶段1实际轮次"], "阶段2实际轮次": q["阶段2实际轮次"],
                           "best_S": q["全局最佳分数"], "best_epoch": q["全局最佳轮次"], "final_S": q["最近合法分数"],
                           "best_SHA256": q["最佳HF检查点SHA256"], "final_SHA256": q["真实末HF检查点SHA256"],
                           "会话数": q["会话数"], "原完整状态实际审核次数": state_counts[seed],
                           "训练累计秒": q["真实HF训练累计成本秒"], "加载构建恢复累计秒": q["真实加载构建恢复累计秒"],
                           "导出核源累计秒": q["真实导出核源累计秒"], "真实会话总累计秒": q["真实HF会话累计总秒"],
                           "完整原PythonqJSON字符串": q_text[:-1]}
    lf_maps = {str(seed): own.ordinary_map(own.RUN_BASE / f"正式Howard_LF_seed{seed}") for seed in range(5)}
    assert lf_maps == machine["五LF工件_before"] == machine["五LF工件_after"]
    sources = {name: own.sha(ROOT / name) for name in hf.SOURCE_MEMBERS}
    assert sources == machine["源码82_before"] == machine["源码82_after"] == cross["源码82_after"]
    parents = {name: own.file_identity(ROOT / name) for name in machine["四HF父身份_before"]}
    assert parents == machine["四HF父身份_before"] == machine["四HF父身份_after"]
    hf._audit_source_tar(ROOT / hf.SOURCE_TAR, sources, ROOT)
    assert hf._root_active(ROOT / hf.ROOT_LEDGER, {key: parents[name]["SHA256"]
                                                for key, name in zip(hf.IDENTITY_FIELDS, parents)})
    assert not own.torch.cuda.is_initialized()
    assert own.COUNTS["实际CPU_PT反序列化数"] == 0
    evidence_names = [path.name for path in sorted(HERE.iterdir()) if path.is_file()]
    evidence = {name: own.file_identity(HERE / name) for name in evidence_names}
    result = {"状态": "任11有界HF013原完整资格及纯收据成本/前缀独审完成；交付停止",
              "北京时间": datetime.now().astimezone().isoformat(), "seed结果": seeds,
              "原完整资格actualexit": original["实际exit_code"], "原完整资格完整stderr字节": original["原stderr字节"],
              "原完整资格真实墙钟秒": original["外层实际墙钟秒"],
              "纯收尾actualexit": closing["实际exit_code"], "纯收尾完整stderr字节": closing["原stderr字节"],
              "纯收尾真实墙钟秒": closing["外层实际墙钟秒"],
              "实际PT读取计数": dict(pt_kinds), "实际Parquet_collect计数": dict(projected),
              "实际原完整LF资格次数": 30, "实际完整HF状态次数": 127,
              "实际全56四RNG阶段边界次数": 3, "实际HF_view次数": 52,
              "原资格守卫显式禁止事件计数_before最后JSON写出": observed,
              "原资格计数时点": "原程序最终JSON写出前；退出后完整stderr/actualexit另由真实外层捕获，不冒全时守卫。",
              "原四HF父准确身份": parents, "原SOURCE82完整SHA映射": sources,
              "原成功HF完整当前SHA映射": hf_maps, "原五LF完整当前SHA映射": lf_maps,
              "所有源父模型日志收据零漂移": True,
              "封存时ROOT历史身份": own.file_identity(ROOT / hf.ROOT_LEDGER), "原自身四SHAplain活动门禁当前PASS": True,
              "先前工具偏差": read("原有工具选路偏差保留.json"),
              "HF4内容读取": False, "CUDA_is_initialized": own.torch.cuda.is_initialized(),
              "五HFmean": None, "五HFstd": None, "三成功subset冒五seed": False,
              "十能源OBS门槛满足": False, "SOURCE90新冻结": False,
              "B0采纳": False, "作者exact": False, "内部真值或安全结论": False,
              "本收尾模型反序列化数": 0, "已有成功及失败原件删改": False,
              "新独审档案准确身份_汇总写出前": evidence,
              "后续动作": "只对本新档案做0444机械封存与fresh验证；完成即停，不训练、不读HF4、不12/13、能源、OBS或90冻结。"}
    own.write_json("独立成功013收尾汇总与准确证据身份.json", result)
    print(json.dumps({"完成seed": list(own.SEEDS), "原完整资格actualexit": 0, "纯收尾actualexit": 0,
                      "PT真实读入": dict(pt_kinds), "Parquet实际collect": dict(projected),
                      "SOURCE82及四父三HF五LF当前零漂移": True, "原q浮点5.0保真": True,
                      "CUDA_is_initialized": own.torch.cuda.is_initialized(), "五HFmean": None, "五HFstd": None},
                     ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
