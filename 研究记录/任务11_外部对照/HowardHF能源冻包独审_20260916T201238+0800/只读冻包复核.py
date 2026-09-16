"""CPU-only independent pack/contract/gate audit; no real qualification or model reads."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import runpy
import shlex
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/phl/lyf/Temperature Field Prediction")
OWN = Path(__file__).resolve().parent
AUTHOR = ROOT / "研究记录/任务11_外部对照/HowardHF独立能源事前冻结_20260916T195529+0800"
RUNTIME = ROOT / "研究记录/任务11_外部对照/HowardLF正式执行环境_20260916T172037+0800"
PYTHON = "/home/phl/anaconda3/envs/PINN/bin/python"
HISTORICAL_ROOT = "aa6785eadfab41ada86398abafcaf644634bac1e1f4458d2f9af4c9ad3351be5"
MAP_SHA = "1345b9389f9f98319cffa4852a17b9b12fa6fc52e2ef9f2b81422b4eb55866cc"
ENV = {
    "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "PYTHONPATH": str(ROOT / "src"),
    "TMPDIR": str(RUNTIME / "临时文件"), "TMP": str(RUNTIME / "临时文件"),
    "TEMP": str(RUNTIME / "临时文件"), "XDG_CACHE_HOME": str(RUNTIME / "应用缓存"),
    "CUDA_CACHE_PATH": str(RUNTIME / "GPU缓存"), "MPLCONFIGDIR": str(RUNTIME / "绘图缓存"),
    "TORCH_EXTENSIONS_DIR": str(RUNTIME / "Torch扩展缓存"),
    "TORCHINDUCTOR_CACHE_DIR": str(RUNTIME / "Inductor缓存"),
}
EXPECTED = {
    "研究记录/任务11_外部对照/Howard适配三网HF双状态独立能源事前登记.yaml": (11689, "e5c7038a82d8c7022b3152ab158c886387bd139f8ce05836fe34ffaa19edbc22"),
    "研究记录/任务11_外部对照/Howard适配三网HF双状态独立能源源码事前冻结.tar.gz": (369661, "0939ffe69c12bcdee49e12dbd10a294be2b41257ae5ffc8d0a6bebc616e195e0"),
    "研究记录/任务11_外部对照/Howard适配三网HF联合预算前登记.yaml": (15408, "ed5513d20fc136457bc33b025728b42b33c41f6d40c777cdfc31e72e2916bf33"),
    "研究记录/任务11_外部对照/Howard适配三网HF联合源码事前冻结.tar.gz": (351593, "61fe4167dad454a91aa4877a027b0eea290f2705c01afb4becaa1bddab9b4249"),
    "研究记录/任务11_外部对照/Howard适配五LF完训身份清单.json": (44182, "b3ad1acc8e56654a935d07d5275d98fb82648a39cf70d28a70814e2184e419de"),
    "研究记录/任务11_外部对照/任11_新MLP_HF12_3开发结构来源清单_20260916T153147.json": (2343, "7665603abb31fa225c5ef36394ddf7c3b97461340b8b3307863766cbae252e18"),
}
AUTHOR_EXPECTED = {
    "启动前中文报告.md": (12022, "7b4e3abe90cb15e16672af38ab374a8ac995e89d8aa3fa7a73c26f922f21b49f"),
    "真实事前冻结机器证据.json": (126217, "1cba48f346ab8a3bc78aeaecf4880d058ddb2e5f6b4566e2c837944eb35529e9"),
    "登记前现场身份.json": (23759, "fbd9f9a6d145b22f558888979d0d483b395752e7a5b19f38c51822d6259a5454"),
    "冻结命令实际终态补存.json": (4910, "9065e2d6061f12f224ff13a3dd8ce0ea011dd9e693893ae6aa40401f067a72fd"),
    "交付前真实复核机器证据.json": (26093, None),
}


def now():
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def metadata(path):
    path = Path(path)
    assert path.is_relative_to(ROOT), path
    info = path.lstat()
    assert stat.S_ISREG(info.st_mode) and not path.is_symlink(), path
    value = path.read_bytes()
    return {"相对路径": str(path.relative_to(ROOT)), "字节": len(value), "SHA256": digest_bytes(value),
            "普通文件": True, "权限": oct(stat.S_IMODE(info.st_mode))}


def canonical(mapping):
    return digest_bytes(json.dumps(mapping, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())


def current_map(names):
    return {name: metadata(ROOT / name)["SHA256"] for name in sorted(names)}


def imports():
    from sic_cu.eval import task11_howard_hf_energy as energy
    from sic_cu.train import task11_howard_hf_formal as hf
    return energy, hf


def actual_identity(hf):
    return {key: value for name, path in zip(("registry", "source_tar", "lf_catalog", "hf_data_catalog"),
               (hf.REGISTRY, hf.SOURCE_TAR, hf.LF_CATALOG, hf.HF_DATA_CATALOG))
            for key, value in ((name, path), (name + "_sha", metadata(ROOT / path)["SHA256"]))}


def actual_six(energy, hf):
    return {"ENERGY_YAML_SHA256": metadata(ROOT / energy.ENERGY_REGISTRY)["SHA256"],
            "ENERGY_TAR_SHA256": metadata(ROOT / energy.ENERGY_SOURCE_TAR)["SHA256"],
            **{field: actual_identity(hf)[name + "_sha"] for field, name in
               zip(hf.IDENTITY_FIELDS, ("registry", "source_tar", "lf_catalog", "hf_data_catalog"))}}


def ordinary_tar(path, expected):
    hashes, names = {}, []
    counters = {"目录数": 0, "符号链接数": 0, "硬链接数": 0, "其他非普通数": 0}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            names.append(member.name)
            counters["目录数"] += int(member.isdir())
            counters["符号链接数"] += int(member.issym())
            counters["硬链接数"] += int(member.islnk())
            counters["其他非普通数"] += int(not member.isfile() and not (member.isdir() or member.issym() or member.islnk()))
            assert member.isfile(), (path, member.name, member.type)
            assert member.name not in hashes, (path, "duplicate", member.name)
            assert member.name in expected, (path, "extra", member.name)
            assert not Path(member.name).is_absolute() and ".." not in Path(member.name).parts, member.name
            stream = archive.extractfile(member)
            assert stream is not None, member.name
            value = stream.read()
            assert len(value) == member.size, member.name
            hashes[member.name] = digest_bytes(value)
    assert set(hashes) == set(expected), (path, "missing", set(expected) - set(hashes))
    assert hashes == expected, (path, "member digest mismatch")
    return {"成员数": len(names), "唯一成员数": len(hashes), **counters, "重复数": len(names)-len(hashes),
            "额外数": len(set(names)-set(expected)), "缺失数": len(set(expected)-set(names)), "逐成员SHA256": hashes}


def basic_author_meta(expected, actual):
    return {key: actual[key] for key in expected} == expected


def pack_audit():
    energy, hf = imports()
    before85, before82 = current_map(energy.SOURCE_MEMBERS), current_map(hf.SOURCE_MEMBERS)
    assert len(before85) == 85 and len(before82) == 82 and canonical(before85) == MAP_SHA
    assert set(before85)-set(before82) == {"src/sic_cu/eval/task11_howard_hf_energy.py",
                                         "scripts/56_audit_task11_howard_hf_energy.py", "tests/test_task11_howard_hf_energy.py"}
    originals = {}
    for name, (size, digest) in EXPECTED.items():
        item = metadata(ROOT / name)
        assert (item["字节"], item["SHA256"]) == (size, digest), item
        if name in (energy.ENERGY_REGISTRY, energy.ENERGY_SOURCE_TAR):
            assert item["权限"] == "0o444", item
        originals[name] = item
    author_files = {}
    for name, (size, digest) in AUTHOR_EXPECTED.items():
        item = metadata(AUTHOR / name)
        assert item["字节"] == size and (digest is None or item["SHA256"] == digest), item
        assert item["权限"] == "0o444", item
        author_files[name] = item
    identity = actual_identity(hf)
    assert energy._identity(identity) == identity and len(identity) == 8
    actual = hf._load_yaml(ROOT / energy.ENERGY_REGISTRY)
    expected = energy.registration_contract(source_hashes=before85, source_tar_sha=originals[energy.ENERGY_SOURCE_TAR]["SHA256"],
        hf_source_identity=identity, registered_at=actual["登记时间"], root_before_sha=HISTORICAL_ROOT)
    assert hf._same(actual, expected), "actual energy schema differs from fresh pure contract"
    parent = hf._load_yaml(ROOT / hf.REGISTRY)
    parent_expected = hf.registration_contract(source_hashes=before82, source_tar_sha=identity["source_tar_sha"],
        lf_catalog_sha=identity["lf_catalog_sha"], hf_data_catalog_sha=identity["hf_data_catalog_sha"],
        query_metadata=parent["固定PHQH查询"], registered_at=parent["登记时间"])
    assert hf._same(parent, parent_expected), "actual parent schema differs from fresh pure contract"
    parent_tar = ordinary_tar(ROOT / hf.SOURCE_TAR, before82)
    energy_tar = ordinary_tar(ROOT / energy.ENERGY_SOURCE_TAR, before85)
    hf._audit_source_tar(ROOT / hf.SOURCE_TAR, before82, ROOT)
    energy._audit_source_tar(ROOT / energy.ENERGY_SOURCE_TAR, before85, ROOT)
    pre = json.loads((AUTHOR / "登记前现场身份.json").read_text(encoding="utf-8"))
    made = json.loads((AUTHOR / "真实事前冻结机器证据.json").read_text(encoding="utf-8"))
    supplement = json.loads((AUTHOR / "冻结命令实际终态补存.json").read_text(encoding="utf-8"))
    delivery = json.loads((AUTHOR / "交付前真实复核机器证据.json").read_text(encoding="utf-8"))
    assert made["登记前现场"] == pre
    for stage in (pre, made["登记后现场"]):
        assert stage["SOURCE85"] == before85 and stage["SOURCE82"] == before82
        for name, item in stage["HowardHF四原件"].items():
            assert basic_author_meta(item, originals[identity[name]]), name
    assert pre["HowardHF四来源严格8键"] == identity
    assert pre["登记前ROOT"]["SHA256"] == made["登记后现场"]["ROOT整件"]["SHA256"] == HISTORICAL_ROOT
    assert pre["SOURCE85规范JSON摘要SHA256"] == MAP_SHA
    command_records = {}
    for key, expected_rc in (("纯合同普通tar首次实际核验", 0), ("纯合同普通tar终态实际核验", 0),
                             ("真实CLI56帮助", 0), ("真实CLI56不活动ROOT负门禁", 1)):
        item = made[key]
        assert type(item["实际退出码"]) is int and item["实际退出码"] == expected_rc, key
        assert item["工作目录"] == str(ROOT)
        assert type(item["stdout"]) is str and type(item["stderr"]) is str
        command_records[key] = item
        if "纯合同" in key:
            stdout = json.loads(item["stdout"])
            assert stdout["SOURCE85普通tar审查"]["逐成员内容SHA256"] == before85
            assert stdout["父SOURCE82普通tar审查"]["逐成员内容SHA256"] == before82
            assert all(stdout[k] == 0 and type(stdout[k]) is int for k in
                       ("SOURCE85前后漂移数", "SOURCE82前后漂移数", "四HF原件前后漂移数", "ROOT整件前后漂移数"))
    assert made["首次核验摘要"] == json.loads(made["纯合同普通tar首次实际核验"]["stdout"])
    assert "line 233" in made["真实CLI56不活动ROOT负门禁"]["stderr"]
    assert "第235行" in made["负门禁说明"]["证据方法"]
    assert supplement["exec_command实际退出码"] == 0
    final = supplement["终态输出JSON对象"]
    assert final["冻结实际退出码"] == 0 and final["纯合同_tar核验退出码"] == [0, 0]
    assert final["真实CLI56_help退出码"] == 0 and final["真实CLI56_ROOT负门禁退出码"] == 1
    assert final["SOURCE85摘要SHA256"] == MAP_SHA
    assert final["ROOT前后SHA256"] == HISTORICAL_ROOT
    for field, name in (("YAML", energy.ENERGY_REGISTRY), ("普通tar", energy.ENERGY_SOURCE_TAR)):
        assert basic_author_meta(made["能源两新原件"][field], originals[name])
    recorded_outer = shlex.split(made["本轮完整实际命令"])
    actual_outer = shlex.split(supplement["外层完整实际命令"])
    assert len(recorded_outer) == len(actual_outer) and recorded_outer[-1] == actual_outer[-1] == "freeze"
    assert recorded_outer[:-2] == actual_outer[:-2]
    assert Path(recorded_outer[-2]) == ROOT / actual_outer[-2]
    assert "233" in supplement["不改旧证据的准确性补注"]["负门禁定位"]
    assert delivery["实际退出码"] == 0 and delivery["SOURCE85当前对登记前逐项漂移数"] == 0
    assert delivery["SOURCE82当前对登记前逐项漂移数"] == 0 and delivery["四父HF原件当前对登记前漂移数"] == 0
    assert delivery["普通SOURCE85审查"]["逐成员内容SHA256"] == before85
    assert delivery["普通父SOURCE82审查"]["逐成员内容SHA256"] == before82
    for item in delivery["所有交付物当前准确身份"].values():
        name = item["相对路径"]
        if name in originals:
            assert basic_author_meta(item, originals[name]), name
        elif Path(name).name in author_files:
            assert basic_author_meta(item, author_files[Path(name).name]), name
    six = actual_six(energy, hf)
    assert made["未来ROOT独立六SHA身份"] == six == delivery["六SHA身份"]
    gate_before = {"ROOT上下文": metadata(ROOT / hf.ROOT_LEDGER),
                   "父HF活动": hf._root_active(ROOT / hf.ROOT_LEDGER, {k: six[k] for k in hf.IDENTITY_FIELDS}),
                   "能源本人活动": energy.root_active(ROOT / hf.ROOT_LEDGER, six)}
    assert gate_before["父HF活动"] is True and gate_before["能源本人活动"] is False, gate_before
    after85, after82 = current_map(energy.SOURCE_MEMBERS), current_map(hf.SOURCE_MEMBERS)
    assert before85 == after85 and before82 == after82
    assert all(metadata(ROOT / name) == item for name, item in originals.items())
    assert all(metadata(AUTHOR / name) == item for name, item in author_files.items())
    return {"严格本人能源合同": True, "严格本人8键来源": True, "严格父HF合同": True,
            "SOURCE85": before85, "SOURCE82": before82, "SOURCE85规范JSON摘要SHA256": canonical(before85),
            "SOURCE82规范JSON摘要SHA256": canonical(before82), "SOURCE85普通tar": energy_tar, "SOURCE82普通tar": parent_tar,
            "四父及能源两原件": originals, "作者五物料": author_files, "实际6SHA": six, "ROOT实际门禁": gate_before,
            "SOURCE85前后漂移数": 0, "SOURCE82前后漂移数": 0, "四父及七冻包物料前后漂移数": 0,
            "作者记录命令原样核对": command_records, "作者外层实际终态补存原样": supplement,
            "历史ROOT说明": "历史aa6785只核对原证据冻结短窗口；当前ROOT合法追加不要求等于历史原件",
            "真实资格_PT_温度_PARQ_TEST_CUDA_能源数值": False}


def root_fixtures():
    energy, hf = imports()
    six = actual_six(energy, hf)
    token = energy.ROOT_TOKEN + "; status=active; " + "; ".join(f"{k}={six[k]}" for k in energy.ROOT_IDENTITY_FIELDS)
    row = f"| 录-9000 | {token} | 本项目CPU合成表格，不是真实ROOT |"
    wrong_parent = row.replace(six["YAML_SHA256"], "0" * 64)
    wrong_energy = row.replace(six["ENERGY_YAML_SHA256"], "0" * 64)
    parent_token = hf.ROOT_TOKEN + "; status=active; " + "; ".join(f"{k}={six[k]}" for k in hf.IDENTITY_FIELDS)
    cases = {
        "完整同格本人六SHA": (row, True), "错误父SHA": (wrong_parent, False), "错误能源SHA": (wrong_energy, False),
        "继承父HF活动": (f"| 录-9000 | {parent_token} | 假借父活动 |", False),
        "HTML注释隐藏": ("<!--\n" + row + "\n-->", False),
        "HTMLscript隐藏": ("<script>\n" + row + "\n</script>", False),
        "HTML嵌套raw隐藏": ("<pre><textarea>\n</pre>\n\n" + row + "\n</textarea>", False),
        "普通HTML跨空白隐藏": ("<div>\n" + row + "\n</div>", False),
        "反引号fence隐藏": ("```\n" + row + "\n```", False),
        "波浪fence隐藏": ("~~~~\n" + row + "\n~~~~", False),
        "长fence短关闭仍隐藏": ("`````\n```\n" + row + "\n`````", False),
        "重复录号": (row + "\n| 录-9000 | 普通追加 | 碰撞 |", False),
        "重复本人活动": (row + "\n" + row.replace("9000", "9001"), False),
        "录号不大于107": (row.replace("9000", "0107"), False),
        "六SHA跨不同格": (row.replace("; ENERGY_TAR_SHA256", " | ENERGY_TAR_SHA256"), False),
        "非本人的四SHA字段": (row.replace("ENERGY_YAML_SHA256=" + six["ENERGY_YAML_SHA256"] + "; ", ""), False),
    }
    temporary = Path(tempfile.mkdtemp(prefix="HowardHF冻包门禁独审_", dir=ENV["TMPDIR"]))
    results = {}
    for number, (name, (text, expected)) in enumerate(cases.items()):
        path = temporary / f"root_{number:02d}.md"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
        actual = energy.root_active(path, six)
        assert actual is expected, (name, actual, expected)
        results[name] = {"期望": expected, "实际": actual, "纯CPU合成": True, "工件": metadata(path)}
    return {"项目内合成目录": str(temporary), "门禁fixture数": len(results), "逐项": results,
            "真实ROOT或正式能源许可修改": False}


def schema_fixtures():
    energy, hf = imports()
    actual = hf._load_yaml(ROOT / energy.ENERGY_REGISTRY)
    cases = {}
    for label, key, replacement in (("等值整数冒充功率float", "功率_瓦", [55, 115.2, 364.3, 403.0, 630.5, 729.0]),
        ("等值浮点冒充点数int", "功率时刻点数", 30.0),
        ("数字1冒充布尔true", "不用于训练选模", 1),
        ("错误query_dtype标识", "PHQH缓冲dtype", "torch.float32"),
        ("微小实数功率漂移", "功率_瓦", [55.001, 115.2, 364.3, 403.0, 630.5, 729.0])):
        changed = copy.deepcopy(actual)
        changed["独立能源审核"][key] = replacement
        accepted = hf._same(changed, actual)
        assert accepted is False, label
        cases[label] = {"严格_same接受": accepted, "纯CPU合成": True}
    changed = copy.deepcopy(actual["HowardHF四来源"])
    changed["registry"] = "研究记录/任务11_外部对照/任11_MLP_HF联合预算前登记.yaml"
    try:
        energy._identity(changed)
    except ValueError as error:
        cases["借MLP固定路径身份"] = {"实际拒绝": type(error).__name__, "原因": str(error), "纯CPU合成": True}
    else:
        raise AssertionError("wrong fixed parent identity accepted")
    temporary = Path(tempfile.mkdtemp(prefix="HowardHF冻包严格YAML独审_", dir=ENV["TMPDIR"]))
    duplicate = temporary / "重复键纯合成.yaml"
    with duplicate.open("x", encoding="utf-8") as stream:
        stream.write("schema_version: 1\nschema_version: 1\n")
    try:
        hf._load_yaml(duplicate)
    except ValueError as error:
        cases["严格加载器同值重复键"] = {"实际拒绝": type(error).__name__, "原因": str(error), "工件": metadata(duplicate), "纯CPU合成": True}
    else:
        raise AssertionError("duplicate YAML key accepted")
    return {"严格合同负例数": len(cases), "逐项": cases, "真实工件或温度读取": False}


def negative_cli():
    import torch
    import polars as pl
    energy, hf = imports()
    counters = {}
    def poison(name):
        counters[name] = 0
        def reject(*args, **kwargs):
            counters[name] += 1
            raise AssertionError("forbidden real callback: " + name)
        return reject
    hf.qualify_task11_howard_hf_source = poison("HF完整真实资格")
    hf.qualify_task11_howard_lf_source = poison("LF完整真实资格")
    torch.load = poison("torch.load真实模型")
    pl.read_parquet = poison("polars.read_parquet温度")
    pl.scan_parquet = poison("polars.scan_parquet温度")
    hf.load_sampled_points = poison("HF模拟温度采样")
    hf._sensor_tensors = poison("HF传感器温度")
    hf._ir_dataset = poison("HF红外温度")
    hf.collect_task11_mlp_hf_data_catalog = poison("MLP来源API禁止借用")
    energy.load_task11_howard_energy_model = poison("本人PT视图工厂")
    energy.load_materials = poison("materials实际温度流程")
    energy.load_resolved_boundary_conditions = poison("boundary实际温度流程")
    energy.load_yaml = poison("geometry实际温度流程")
    energy.audit_schedule_energy = poison("实际能源积分")
    energy.require_task11_howard_energy_cuda = poison("正式CUDA发现入口")
    for name in ("is_available", "device_count", "synchronize", "init", "current_device"):
        setattr(torch.cuda, name, poison("torch.cuda." + name))
    Path.mkdir = poison("Path.mkdir")
    requested = sys.argv[2:]
    script = ROOT / "scripts/56_audit_task11_howard_hf_energy.py"
    sys.argv = [str(script), *requested]
    output = ROOT / requested[requested.index("--output") + 1]
    before = output.exists()
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        print(json.dumps({"实际CLI": str(script), "实际argv": requested, "PID": os.getpid(),
                          "禁止回调计数": counters, "输出目录执行前存在": before,
                          "输出目录执行后存在": output.exists()}, ensure_ascii=False, allow_nan=False), flush=True)


def execute(argv):
    started, clock = now(), time.perf_counter()
    complete_env = {**os.environ, **ENV}
    command = ["env", *(f"{k}={v}" for k, v in ENV.items()), *argv]
    process = subprocess.Popen(argv, cwd=ROOT, env=complete_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return {"完整实际命令": shlex.join(command), "实际argv": argv, "工作目录": str(ROOT),
            "开始时间": started, "结束时间": now(), "子进程PID": process.pid,
            "实际退出码": process.returncode, "墙钟秒": time.perf_counter()-clock, "stdout": stdout, "stderr": stderr}


def run_audit(evidence_name):
    energy, hf = imports()
    assert Path.cwd() == ROOT and all(os.environ.get(k) == v for k, v in ENV.items()), "strict actual env missing"
    assert OWN.is_relative_to(ROOT) and Path(evidence_name).name == evidence_name
    assert not (OWN / evidence_name).exists()
    source_before = current_map(energy.SOURCE_MEMBERS)
    originals_before = {name: metadata(ROOT / name) for name in EXPECTED}
    author_before = {name: metadata(AUTHOR / name) for name in AUTHOR_EXPECTED}
    root_before = metadata(ROOT / hf.ROOT_LEDGER)
    started, clock = now(), time.perf_counter()
    records, errors = {}, []
    try:
        for name, mode in (("fresh冻结合同普通包", "pack"), ("pureCPU门禁fixture", "fixtures"),
                           ("pureCPU严格合同类型fixture", "types")):
            records[name] = execute([PYTHON, "-B", str(Path(__file__).resolve()), mode])
            assert records[name]["实际退出码"] == 0, name
        records["真实CLI56帮助"] = execute([PYTHON, "-B", str(ROOT / "scripts/56_audit_task11_howard_hf_energy.py"), "--help"])
        assert records["真实CLI56帮助"]["实际退出码"] == 0
        six, identity = actual_six(energy, hf), actual_identity(hf)
        run = hf.RUN_DIRECTORY + "/正式Howard_HF_seed0"
        output = run + "/独立原能源_Howard_观测最佳_20260916T201238+0800"
        arguments = ["--run", run, "--output", output, "--seed", "0", "--state", "best",
                     "--energy-registry", energy.ENERGY_REGISTRY, "--energy-registry-sha", six["ENERGY_YAML_SHA256"],
                     "--energy-source-tar", energy.ENERGY_SOURCE_TAR, "--energy-source-tar-sha", six["ENERGY_TAR_SHA256"]]
        for name in ("registry", "source_tar", "lf_catalog", "hf_data_catalog"):
            arguments += ["--" + name.replace("_", "-"), identity[name],
                          "--" + name.replace("_", "-") + "-sha", identity[name + "_sha"]]
        arguments += ["--device", "cuda"]
        records["真实CLI56六SHA动态负门禁"] = execute([PYTHON, "-B", str(Path(__file__).resolve()), "negative", *arguments])
        negative = records["真实CLI56六SHA动态负门禁"]
        assert negative["实际退出码"] == 1 and "ValueError: Howard能源需要独立六SHA plain ROOT活动门禁" in negative["stderr"]
        assert "AssertionError" not in negative["stderr"]
        callbacks = json.loads(negative["stdout"])
        assert all(type(v) is int and v == 0 for v in callbacks["禁止回调计数"].values())
        assert callbacks["输出目录执行前存在"] is False and callbacks["输出目录执行后存在"] is False
        assert not (ROOT / output).exists()
    except BaseException:
        errors.append(traceback.format_exc())
    source_after = current_map(energy.SOURCE_MEMBERS)
    originals_after = {name: metadata(ROOT / name) for name in EXPECTED}
    author_after = {name: metadata(AUTHOR / name) for name in AUTHOR_EXPECTED}
    root_after = metadata(ROOT / hf.ROOT_LEDGER)
    if source_before != source_after or originals_before != originals_after or author_before != author_after:
        errors.append("current ordinary source/parents/frozen7 material drift")
    six = actual_six(energy, hf)
    evidence = {"结构版本": 1, "证据性质": "本代理独立fresh CPU纯合同普通冻包及动态拒绝；不继承作者测试/科学效果计数",
        "开始时间": started, "结束时间": now(), "审计PID": os.getpid(), "父PID": os.getppid(),
        "墙钟秒": time.perf_counter()-clock, "工作目录": str(ROOT), "实际环境": ENV,
        "本轮完整实际命令": shlex.join(["env", *(f"{k}={v}" for k, v in ENV.items()), PYTHON, "-B", str(Path(__file__).resolve()), "audit", evidence_name]),
        "本轮内部核验结论退出码": int(bool(errors)), "真实子命令": records, "实际错误原样": errors,
        "SOURCE85前": source_before, "SOURCE85后": source_after, "SOURCE85规范JSON摘要SHA256": canonical(source_after),
        "SOURCE85前后漂移数": sum(source_before[k] != source_after[k] for k in source_before),
        "SOURCE82前后漂移数": sum(source_before[k] != source_after[k] for k in hf.SOURCE_MEMBERS),
        "四父及能源两原件前": originals_before, "四父及能源两原件后": originals_after,
        "作者五物料前": author_before, "作者五物料后": author_after,
        "ROOT短窗口前_仅上下文": root_before, "ROOT短窗口后_仅上下文": root_after,
        "ROOT整件变化不当源码漂移": True,
        "当前父HF四SHA活动": hf._root_active(ROOT / hf.ROOT_LEDGER, {k: six[k] for k in hf.IDENTITY_FIELDS}),
        "当前能源本人六SHA活动": energy.root_active(ROOT / hf.ROOT_LEDGER, six), "当前本人六SHA": six,
        "真实五HF_十能源_GPU_新测量_工程安全_B0_总目标完成": False,
        "被审源码_ROOT_状态_YAML_tar_catalog_报告写入": False}
    with (OWN / evidence_name).open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"实际内部退出码": int(bool(errors)), "真实子命令退出码": {k: v["实际退出码"] for k, v in records.items()},
        "SOURCE85": len(source_after), "SOURCE82": len(hf.SOURCE_MEMBERS), "SOURCE85摘要": canonical(source_after),
        "源码前后漂移数": evidence["SOURCE85前后漂移数"], "当前父HF活动": evidence["当前父HF四SHA活动"],
        "当前能源活动": evidence["当前能源本人六SHA活动"], "机器证据": metadata(OWN / evidence_name),
        "错误原样": errors}, ensure_ascii=False, allow_nan=False), flush=True)
    return int(bool(errors))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "negative":
        negative_cli()
    elif mode in ("pack", "fixtures", "types"):
        function = {"pack": pack_audit, "fixtures": root_fixtures, "types": schema_fixtures}[mode]
        print(json.dumps(function(), ensure_ascii=False, allow_nan=False, indent=2))
    elif mode == "audit":
        raise SystemExit(run_audit(sys.argv[2] if len(sys.argv) > 2 else "冻包独审机器证据.json"))
    else:
        raise ValueError(mode)
