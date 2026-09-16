"""Mechanical permission seal and fresh byte verification of only new audit files."""

import json
import os
import stat

import independent_success_013_audit as own


HERE, ROOT, hf = own.HERE, own.ROOT, own.hf


def main():
    summary = json.loads((HERE / "独立成功013收尾汇总与准确证据身份.json").read_text(encoding="utf-8"))
    terminal = json.loads((HERE / "汇总准确身份实际命令与完整终态.json").read_text(encoding="utf-8"))
    assert terminal["actual_exit_code"] == 0 and terminal["实际完整stderr"] == ""
    names = [path.name for path in sorted(HERE.iterdir()) if path.is_file()]
    for name in names:
        path = HERE / name
        assert not path.is_symlink() and stat.S_ISREG(path.stat().st_mode)
        os.chmod(path, 0o444)
    evidence = {name: own.file_identity(HERE / name) for name in names}
    assert all(item["权限"] == "0o444" for item in evidence.values())
    sources = {name: own.sha(ROOT / name) for name in hf.SOURCE_MEMBERS}
    parents = {name: own.file_identity(ROOT / name) for name in summary["原四HF父准确身份"]}
    hf_maps = {str(seed): own.ordinary_map(own.RUN_BASE / f"正式Howard_HF_seed{seed}") for seed in own.SEEDS}
    lf_maps = {str(seed): own.ordinary_map(own.RUN_BASE / f"正式Howard_LF_seed{seed}") for seed in range(5)}
    assert sources == summary["原SOURCE82完整SHA映射"] and parents == summary["原四HF父准确身份"]
    assert hf_maps == summary["原成功HF完整当前SHA映射"] and lf_maps == summary["原五LF完整当前SHA映射"]
    hf._audit_source_tar(ROOT / hf.SOURCE_TAR, sources, ROOT)
    assert hf._root_active(ROOT / hf.ROOT_LEDGER, {key: parents[name]["SHA256"]
                                                for key, name in zip(hf.IDENTITY_FIELDS, parents)})
    assert not own.torch.cuda.is_initialized() and own.COUNTS["实际CPU_PT反序列化数"] == 0
    for kind in ("HF4访问拒绝尝试", "CUDA初始化尝试", "训练或optimizer_step尝试", "档案外写入尝试"):
        assert own.COUNTS[kind] == 0
    manifest = {"状态": "本新独审普通档案机械0444封存；准确SHA逐件重新读核PASS",
                "普通档案数量_不含此清单及其外层终态": len(evidence), "封存普通档案准确身份": evidence,
                "范围": "仅本项目新独审目录顶层普通证据，不chmod/root/source/PT/log/receipt/已有档案；TMP/cache不是模型来源或冻结证据。",
                "闭合非循环约定": "本清单本身以及之后真实捕获的机械0444封存验证完整stdout/stderr/终态三件不进入自身SHA映射；三件由外层捕获后各0444，准确SHA在最终交付fresh读核。",
                "source82与四父三HF五LF当前零漂移": True, "原自身四SHAROOTplain正门禁": True,
                "ROOT当前历史身份": own.file_identity(ROOT / hf.ROOT_LEDGER), "wholeROOT行政变化不作为源漂移": True,
                "本封存步骤PT反序列化数": 0, "CUDA_is_initialized": own.torch.cuda.is_initialized(),
                "资格3成功不是五成功": True, "五HFmean": None, "五HFstd": None,
                "新十能源OBS或90冻结": False, "后续": "交付后停止。"}
    own.write_json("封存普通档案准确字节SHA清单.json", manifest)
    os.chmod(HERE / "封存普通档案准确字节SHA清单.json", 0o444)
    for name, expected in evidence.items():
        assert own.file_identity(HERE / name) == expected
    assert own.file_identity(HERE / "封存普通档案准确字节SHA清单.json")["权限"] == "0o444"
    print(json.dumps({"source82与四父三HF五LF当前零漂移": True, "档案成员0444及逐件fresh_SHA核对PASS": True,
                      "封存清单绑定成员数": len(evidence), "PT反序列化数": 0, "CUDA_is_initialized": own.torch.cuda.is_initialized(),
                      "五HFmean": None, "五HFstd": None, "结束": "完成即停，不读HF4，不新实验。"},
                     ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
