"""Pure receipt-prefix and cost cross-check after the original full qualification."""

import json
import math
from pathlib import Path

import independent_success_013_audit as own


HERE, ROOT, hf = own.HERE, own.ROOT, own.hf


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    terminal = read_json(HERE / "真实原资格命令与完整进程终态.json")
    assert terminal["实际exit_code"] == 0 and terminal["实际完整stderr"] == ""
    assert (HERE / "原真实资格进程完整stdout.txt").read_text(encoding="utf-8") == terminal["实际完整stdout"]
    assert (HERE / "原真实资格进程完整stderr.txt").read_bytes() == b""
    machine = read_json(HERE / "独立成功013资格原Python机器证据.json")
    source_before = {name: own.sha(ROOT / name) for name in hf.SOURCE_MEMBERS}
    assert source_before == machine["源码82_before"] == machine["源码82_after"]
    parent_before = {name: own.file_identity(ROOT / name) for name in machine["四HF父身份_before"]}
    assert parent_before == machine["四HF父身份_before"] == machine["四HF父身份_after"]
    own.hf._audit_source_tar(ROOT / hf.SOURCE_TAR, source_before, ROOT)
    result = {}
    for seed in own.SEEDS:
        directory = own.RUN_BASE / f"正式Howard_HF_seed{seed}"
        q_text = (HERE / f"seed{seed}_原Python完整资格q.json").read_text(encoding="utf-8")
        assert q_text == machine["结果"][str(seed)]["完整原PythonqJSON字符串"] + "\n"
        q = json.loads(q_text)
        files_before = own.ordinary_map(directory)
        assert files_before == q["真实工件SHA256"] == machine["成功HF工件_before"][str(seed)]
        rows = [json.loads(line) for line in (directory / "training.jsonl").read_text(encoding="utf-8").splitlines()]
        selected = hf.replay_task11_howard_hf_selection(rows)
        assert selected["已完成"] is True and all(hf._same(value, q[key]) for key, value in selected.items())
        assert hf._same(hf.audit_task11_howard_hf_log(rows, seed=seed), q["消费累计"])
        names = sorted(name for name in files_before if name.startswith("HF会话收据_"))
        assert names == [f"HF会话收据_{number:04d}.json" for number in range(1, q["会话数"] + 1)]
        receipts = []
        end_before, predecessor = 0, None
        sums = {key: 0.0 for key in ("训练墙钟秒", "加载构建恢复墙钟秒", "导出核源墙钟秒", "本会话总墙钟秒")}
        peak = 0
        for number, name in enumerate(names, 1):
            text = (directory / name).read_text(encoding="utf-8")
            receipt = json.loads(text)
            end = receipt["累计实际轮次"]
            assert receipt["起始已提交轮次"] == end_before and 1 <= end - end_before <= 200
            assert receipt["本会话实际轮次"] == end - end_before and receipt["前驱收据SHA256"] == predecessor
            assert hf._same(receipt["身份"], q["事前来源"])
            assert receipt["状态"] == ("已完成Howard三网HF正式训练" if number == len(names) else "已暂停且完整Howard HF阶段提交")
            prefix = [json.loads(line) for line in (directory / f"已提交日志_{number:04d}.jsonl").read_text(encoding="utf-8").splitlines()]
            assert hf._same(prefix, rows[:end])
            assert hf._same(receipt["选模重推"], hf.replay_task11_howard_hf_selection(prefix))
            assert hf._same(receipt["消费累计"], hf.audit_task11_howard_hf_log(prefix, seed=seed))
            history = receipt["提交历史SHA256"]
            assert all(files_before[relative] == digest == own.sha(directory / relative) for relative, digest in history.items())
            assert receipt["原件SHA256"]["阶段_最近.pt"] == history[f"最近提交历史_{number:04d}.pt"]
            assert receipt["原件SHA256"]["training.jsonl"] == history[f"已提交日志_{number:04d}.jsonl"]
            resume_sha = None if number == 1 else receipts[-1]["原收据Python解析"]["原件SHA256"]["阶段_最近.pt"]
            assert receipt["恢复源before_SHA256"] == resume_sha == receipt["恢复源after_SHA256"]
            assert receipt["恢复原件"] == (None if number == 1 else "阶段_最近.pt")
            for key in sums:
                assert type(receipt[key]) is float and math.isfinite(receipt[key]) and receipt[key] > 0
                sums[key] += receipt[key]
            assert math.isclose(receipt["本会话总墙钟秒"], sum(receipt[key] for key in ("训练墙钟秒", "加载构建恢复墙钟秒", "导出核源墙钟秒")), rel_tol=1e-12, abs_tol=1e-7)
            peak = max(peak, receipt["峰值真实CUDA显存字节"])
            receipts.append({"收据名称": name, "收据SHA256": files_before[name], "原收据JSON字符串": text,
                             "原收据Python解析": receipt, "提交日志前缀真实行数": len(prefix),
                             "全提交历史SHA逐文件重新核对PASS": True})
            end_before, predecessor = end, files_before[name]
        assert end_before == len(rows) == q["全局实际轮次"]
        for receipt_key, q_key in (("训练墙钟秒", "真实HF训练累计成本秒"), ("加载构建恢复墙钟秒", "真实加载构建恢复累计秒"),
                                   ("导出核源墙钟秒", "真实导出核源累计秒"), ("本会话总墙钟秒", "真实HF会话累计总秒")):
            assert sums[receipt_key] == q[q_key]
        assert peak == q["峰值CUDA显存字节"]
        metrics = read_json(directory / "metrics.json")
        assert metrics["test"] is None and metrics["status"] == "completed_current_protocol_howard_hf"
        assert own.ordinary_map(directory) == files_before
        result[str(seed)] = {"原Pythonq文件字节及SHA": own.file_identity(HERE / f"seed{seed}_原Python完整资格q.json"),
                             "原qJSON字符串逐字相同": True, "全收据": receipts, "真实累计成本重加": sums,
                             "真实日志行数": len(rows), "日志选模及消费全重推": selected,
                             "HF工件前后零漂移": True, "TEST结果": metrics["test"]}
    lf_after = {str(seed): own.ordinary_map(own.RUN_BASE / f"正式Howard_LF_seed{seed}") for seed in range(5)}
    assert lf_after == machine["五LF工件_before"] == machine["五LF工件_after"]
    source_after = {name: own.sha(ROOT / name) for name in hf.SOURCE_MEMBERS}
    parent_after = {name: own.file_identity(ROOT / name) for name in machine["四HF父身份_before"]}
    assert source_after == source_before and parent_after == parent_before
    assert not own.torch.cuda.is_initialized()
    assert own.COUNTS["实际CPU_PT反序列化数"] == 0
    assert own.COUNTS["HF4访问拒绝尝试"] == own.COUNTS["CUDA初始化尝试"] == own.COUNTS["训练或optimizer_step尝试"] == 0
    output = {"状态": "原实际资格完整终态0且stderr空；纯收据/日志前缀/成本独立交叉核对PASS",
              "结果": result, "源码82_before": source_before, "源码82_after": source_after,
              "四HF父件_before": parent_before, "四HF父件_after": parent_after,
              "原资格完整PT审核不替代": "本步骤不重新反序列化PT；原步骤已实际CPU审核全部完整状态、四RNG、56原AdamW与view。",
              "本步骤守卫计数_before最后JSON写出": dict(own.COUNTS), "CUDA_is_initialized": own.torch.cuda.is_initialized()}
    own.write_json("纯收据前缀成本交叉核对与原q字符串保真.json", output)
    print(json.dumps({"完成seed": list(own.SEEDS), "原资格actualexit": terminal["实际exit_code"],
                      "原资格完整stderr字节": terminal["原stderr字节"], "纯交叉核对PASS": True,
                      "每seed真实收据数": {seed: len(item["全收据"]) for seed, item in result.items()},
                      "SOURCE82与父四件及三成功HF五LF零漂移": True, "本步骤PT读入数": 0,
                      "CUDA_is_initialized": own.torch.cuda.is_initialized()}, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
