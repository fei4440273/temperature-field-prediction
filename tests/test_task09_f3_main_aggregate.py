"""Synthetic-only checks for the Task-09 F3 main-series publication gate."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from sic_cu.data.common import sha256_file
from sic_cu.eval.task09_f3_main_aggregate import (
    ORIGINAL_SHA,
    PROJECT_ROOT,
    RUN_HASH_FILES,
    TASK09_ROOT,
    _verify_energy,
    build_task09_main_evidence_catalog,
    expected_task09_main_identities,
    require_task09_aggregate_prereg_root_ledger,
    require_task09_aggregate_root_ledger,
    validate_task09_main_registry,
    verify_task09_main_evidence,
)


def _write(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return sha256_file(path)


def _catalog_fixture(tmp_path: Path) -> tuple[dict, dict]:
    registered = {"主序列十五身份": expected_task09_main_identities()}
    catalog = {"schema_version": 1, "主序列十五份原件": []}
    scores = {3: [1, 3, 2, 4, 5], 6: [3, 4, 5, 6, 7], 9: [2, 2, 2, 2, 2]}
    for identity in registered["主序列十五身份"]:
        size, seed = identity["HF功率数"], identity["seed"]
        run = tmp_path / identity["canonical目录"]
        artifacts = {
            name: _write(run / name, {"identity": [size, seed], "name": name})
            for name in ("training.jsonl", "任09F3阶段报告.json",
                         "阶段_观测最佳.pt", "阶段_训练末.pt")
        }
        energy = {}
        for state in ("best", "final"):
            label = "观测最佳" if state == "best" else "训练末"
            dirname = f"独立原能源_{label}_20260916T091200+0800"
            folder = run / dirname
            summary = {
                "序列": "primary", "HF子集功率数": size, "本seed": seed,
                "选中状态": state, "旧固定测试温度读取": False,
                "两阶原瓦数与原散度各60行": True,
                "功率时刻审核行数": 30,
                "功率_瓦": [55.0, 115.2, 364.3, 403.0, 630.5, 729.0],
                "时刻_秒": [1.0, 10.0, 50.0, 100.0, 200.0],
                "求积阶数": [16, 64],
                "原定义相对平衡分母已保持": True,
                "工程平衡与原瓦数逐行一致": True,
                "任09子集合法HF选分独立重算_摄氏度": scores[size][seed]
                   + (0.5 if state == "final" else 0),
                "LF本模型逐材料节点与真实体积5%保持资格": {
                    "LF两材料节点与真实体积均守住5%护栏": True,
                },
                "绝对平衡宏均值_瓦": 100 + seed,
                "绝对平衡95分位_瓦": 200 + seed,
                "吸收功率归一宏均值": 2.0,
                "吸收功率归一95分位": 3.0,
                "真实阶段报告SHA256": artifacts["任09F3阶段报告.json"],
                "训练日志SHA256": artifacts["training.jsonl"],
                "本模型阶段SHA256": artifacts[
                    "阶段_观测最佳.pt" if state == "best" else "阶段_训练末.pt"
                ],
                "正式事前登记SHA256": ORIGINAL_SHA,
            }
            hashes = {}
            for filename in ("原始能量.jsonl", "原始散度.jsonl"):
                path = folder / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(
                    json.dumps({"power_w": power, "time_s": time,
                                "quadrature_order": order}) + "\n"
                    for power in summary["功率_瓦"]
                    for time in summary["时刻_秒"]
                    for order in (16, 64)
                ), encoding="utf-8")
                hashes[filename] = sha256_file(path)
            for filename, rows in (("指标明细.csv", 30), ("物理分解.csv", 30)):
                path = folder / filename
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["功率_瓦", "时刻_秒"])
                    writer.writerows([[power, time] for power in summary["功率_瓦"]
                                     for time in summary["时刻_秒"]])
                hashes[filename] = sha256_file(path)
            hashes["汇总指标.json"] = _write(folder / "汇总指标.json", summary)
            manifest_sha = _write(folder / "审计工件SHA256.json", hashes)
            evidence = {
                "目录": str(Path(identity["canonical目录"]) / dirname),
                "汇总指标SHA256": hashes["汇总指标.json"],
                "审计工件SHA256": manifest_sha,
                "V3能源入口收据SHA256": None,
            }
            if identity["来源"] == "v3":
                receipt = {
                    "审核状态": state,
                    "运行身份": {"序列": "primary", "HF功率数": size, "seed": seed},
                    "冻结能源输出逐文件SHA256": {
                        **hashes, "审计工件SHA256.json": manifest_sha,
                    },
                    "旧固定TEST温度读取": False,
                    "V3下游资格": {"V3来源谱系与会话链合格": True},
                }
                evidence["V3能源入口收据SHA256"] = _write(
                    folder / "任09V3能源入口收据.json", receipt,
                )
            energy[state] = evidence
        catalog["主序列十五份原件"].append({
            **identity, "训练原件SHA256": artifacts, "双能源": energy,
        })
    return registered, catalog


def _verify(tmp_path: Path, registered: dict, catalog: dict) -> dict:
    return verify_task09_main_evidence(
        registered, catalog, project_root=tmp_path,
        qualify_v2=lambda _run, _size, _seed: {"qualified": True},
        qualify_v3=lambda _run, _size, _seed: {"qualified": True},
        locked_12hf_scores={seed: 2.0 + seed * 0.1 for seed in range(5)},
    )


def test_all_fifteen_ddof1_and_nonmonotonic_retained(tmp_path: Path) -> None:
    registered, catalog = _catalog_fixture(tmp_path)
    output = _verify(tmp_path, registered, catalog)
    assert [point["HF功率数"] for point in output["主序列三个原点"]] == [3, 6, 9]
    assert output["主序列三个原点"][0]["观测最佳S_均值_摄氏度"] == 3.0
    assert output["主序列三个原点"][0]["观测最佳S_样本标准差_摄氏度"] == pytest.approx((5 / 2)**0.5)
    assert output["主序列三个原点"][1]["观测最佳S_均值_摄氏度"] == 5.0
    assert output["主序列三个原点"][2]["观测最佳S_均值_摄氏度"] == 2.0
    assert output["同一目标误差所需HF数量"] is None
    assert output["种子方差不是子集方差"] is True
    assert output["HF12历史参照_不属新三档同轨迹"]["HF功率数"] == 12


def test_missing_identity_rejected_before_partial_curve(tmp_path: Path) -> None:
    registered, catalog = _catalog_fixture(tmp_path)
    catalog["主序列十五份原件"].pop()
    with pytest.raises(ValueError, match="十五"):
        _verify(tmp_path, registered, catalog)


def test_downstream_qualification_failure_rejected(tmp_path: Path) -> None:
    registered, catalog = _catalog_fixture(tmp_path)
    with pytest.raises(ValueError, match="下游"):
        verify_task09_main_evidence(
            registered, catalog, project_root=tmp_path,
            qualify_v2=lambda *_: {"qualified": False},
            qualify_v3=lambda *_: {"qualified": True},
            locked_12hf_scores={seed: 2.0 for seed in range(5)},
        )


def test_energy_manifest_poison_rejected(tmp_path: Path) -> None:
    registered, catalog = _catalog_fixture(tmp_path)
    first = catalog["主序列十五份原件"][0]
    summary = tmp_path / first["双能源"]["best"]["目录"] / "汇总指标.json"
    _write(summary, {"poison": True})
    with pytest.raises(ValueError, match="SHA"):
        _verify(tmp_path, registered, catalog)


def test_v3_energy_receipt_or_fixed_test_flag_rejected(tmp_path: Path) -> None:
    registered, catalog = _catalog_fixture(tmp_path)
    first_v3 = catalog["主序列十五份原件"][1]
    first_v3["双能源"]["final"]["V3能源入口收据SHA256"] = None
    with pytest.raises(ValueError, match="V3"):
        _verify(tmp_path, registered, catalog)
    registered, catalog = _catalog_fixture(tmp_path)
    first = catalog["主序列十五份原件"][0]
    path = tmp_path / first["双能源"]["final"]["目录"] / "汇总指标.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["旧固定测试温度读取"] = True
    first["双能源"]["final"]["汇总指标SHA256"] = _write(path, data)
    manifest_path = path.parent / "审计工件SHA256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["汇总指标.json"] = first["双能源"]["final"]["汇总指标SHA256"]
    first["双能源"]["final"]["审计工件SHA256"] = _write(manifest_path, manifest)
    with pytest.raises(ValueError, match="TEST"):
        _verify(tmp_path, registered, catalog)


@pytest.mark.parametrize("identity_index,energy_folder,state", [
    (0, "独立原能源_观测最佳_20260916T065823+0800", "best"),
    (0, "独立原能源_训练末_20260916T065941+0800", "final"),
    (4, "独立原能源_观测最佳_20260916T085811+0800", "best"),
    (4, "独立原能源_训练末_20260916T085845+0800", "final"),
])
def test_completed_v2_v3_energy_originals_readonly(
    identity_index: int, energy_folder: str, state: str,
) -> None:
    identity = expected_task09_main_identities()[identity_index]
    run = PROJECT_ROOT / identity["canonical目录"]
    audit = run / energy_folder
    if not audit.is_dir():
        pytest.skip("已有训练身份只读能源样本缺席")
    hashes = {name: sha256_file(run / name) for name in RUN_HASH_FILES}
    record = {
        "目录": str(audit.relative_to(PROJECT_ROOT)),
        "汇总指标SHA256": sha256_file(audit / "汇总指标.json"),
        "审计工件SHA256": sha256_file(audit / "审计工件SHA256.json"),
        "V3能源入口收据SHA256": (
            sha256_file(audit / "任09V3能源入口收据.json")
            if identity["来源"] == "v3" else None
        ),
    }
    verified = _verify_energy(PROJECT_ROOT, identity, record, run, hashes, state)
    assert verified["S"] >= 0


def test_root_ledger_two_sha_gates_need_two_unique_active_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sic_cu.eval.task09_f3_main_aggregate as aggregate

    path = tmp_path / "synthetic_ledger.md"
    monkeypatch.setattr(aggregate, "ROOT_LEDGER", path)
    digest = "a" * 64
    archive_digest = "b" * 64
    evidence_digest = "c" * 64
    archive = tmp_path / "source.tar.gz"
    first = f"| 录-0084 | TASK09_F3_MAIN_AGG_GATE:v2; status=active; {digest}; {archive.name}; {archive_digest} |"
    second = f"| 录-0090 | TASK09_F3_MAIN_EVIDENCE:v2; status=active; {evidence_digest}; {digest}; {archive_digest} |"
    path.write_text(first + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="各独立唯一"):
        require_task09_aggregate_root_ledger(digest, archive, archive_digest, evidence_digest)
    path.write_text(first + "\n" + second + "\n", encoding="utf-8")
    require_task09_aggregate_root_ledger(digest, archive, archive_digest, evidence_digest)
    path.write_text(first + "\n" + second + "\n" + second + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="各独立唯一"):
        require_task09_aggregate_root_ledger(digest, archive, archive_digest, evidence_digest)


def test_root_ledger_rejects_evidence_before_prereg_in_file_or_ledger_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sic_cu.eval.task09_f3_main_aggregate as aggregate

    path = tmp_path / "reverse_ledger.md"
    monkeypatch.setattr(aggregate, "ROOT_LEDGER", path)
    digest, archive_digest, evidence_digest = "a" * 64, "b" * 64, "c" * 64
    archive = tmp_path / "source.tar.gz"
    prereg = f"| 录-0084 | TASK09_F3_MAIN_AGG_GATE:v2; status=active; {digest}; {archive.name}; {archive_digest} |"
    evidence = f"| 录-0090 | TASK09_F3_MAIN_EVIDENCE:v2; status=active; {evidence_digest}; {digest}; {archive_digest} |"
    path.write_text(evidence + "\n" + prereg + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="先于"):
        require_task09_aggregate_root_ledger(digest, archive, archive_digest, evidence_digest)
    prereg_higher = prereg.replace("录-0084", "录-0091")
    path.write_text(prereg_higher + "\n" + evidence + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="先于"):
        require_task09_aggregate_root_ledger(digest, archive, archive_digest, evidence_digest)


def test_root_ledger_rejects_substring_active_and_status_in_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sic_cu.eval.task09_f3_main_aggregate as aggregate

    path = tmp_path / "fake_active_ledger.md"
    monkeypatch.setattr(aggregate, "ROOT_LEDGER", path)
    digest, archive_digest, evidence_digest = "a" * 64, "b" * 64, "c" * 64
    archive = tmp_path / "source.tar.gz"
    prereg = f"| 录-0084 | TASK09_F3_MAIN_AGG_GATE:v2; status=active-ish; {digest}; {archive.name}; {archive_digest} |"
    evidence = f"| 录-0090 | TASK09_F3_MAIN_EVIDENCE:v2; status=active; {evidence_digest}; {digest}; {archive_digest} |"
    path.write_text(prereg + "\n" + evidence + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="第二列|active"):
        require_task09_aggregate_root_ledger(digest, archive, archive_digest, evidence_digest)
    with pytest.raises(ValueError, match="第二列|active"):
        require_task09_aggregate_prereg_root_ledger(digest, archive, archive_digest)
    poisoned = prereg.replace("status=active-ish", "status=inactive; note=status=active")
    path.write_text(poisoned + "\n" + evidence + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="第二列|active"):
        require_task09_aggregate_root_ledger(digest, archive, archive_digest, evidence_digest)


def test_v2_root_gate_requires_v2_marker_and_rejects_frozen_v1_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sic_cu.eval.task09_f3_main_aggregate as aggregate

    path = tmp_path / "versioned_ledger.md"
    monkeypatch.setattr(aggregate, "ROOT_LEDGER", path)
    digest, archive_digest, evidence_digest = "a" * 64, "b" * 64, "c" * 64
    archive = tmp_path / "source_v2.tar.gz"
    prereg = f"| 录-0084 | TASK09_F3_MAIN_AGG_GATE:v2; status=active; {digest}; {archive.name}; {archive_digest} |"
    evidence = f"| 录-0090 | TASK09_F3_MAIN_EVIDENCE:v2; status=active; {evidence_digest}; {digest}; {archive_digest} |"
    path.write_text(prereg + "\n" + evidence + "\n", encoding="utf-8")
    require_task09_aggregate_prereg_root_ledger(digest, archive, archive_digest)
    require_task09_aggregate_root_ledger(digest, archive, archive_digest, evidence_digest)
    path.write_text(prereg.replace(":v2", ":v1") + "\n"
                    + evidence.replace(":v2", ":v1") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="唯一|v2"):
        require_task09_aggregate_prereg_root_ledger(digest, archive, archive_digest)
    with pytest.raises(ValueError, match="唯一|v2"):
        require_task09_aggregate_root_ledger(digest, archive, archive_digest, evidence_digest)


def test_v2_preregistered_schema_rejects_old_frozen_v1_definition() -> None:
    original = yaml.safe_load((TASK09_ROOT / "正式F3主序列五种子正式聚合前登记.yaml")
                              .read_text(encoding="utf-8"))
    original["正式结果源文件SHA256"]["聚合模块"] = sha256_file(
        PROJECT_ROOT / "src/sic_cu/eval/task09_f3_main_aggregate.py")
    original["正式结果源文件SHA256"]["合成与真实原件只读测试"] = sha256_file(
        PROJECT_ROOT / "tests/test_task09_f3_main_aggregate.py")
    new = {**original,
           "schema_version": 2,
           "旧v1事前登记SHA256": "ae19ab85d6c7e8833986d84e65343b1b54af8fecbd9cc4988c1462d1e6d54131",
           "旧v1五源冻结tar_SHA256": "5334745d8eb052db114e0f3e56981cd7917ba44db693e40947fff324edf9a827",
           "旧v1中文报告_SHA256": "7089327b6186c1008a76392216744dbde42ae1b6cd85329642878701b0e096e1",
           "ROOT第二列激活字段须精确为status_active": True,
           "ROOT前登记同时先于证据的账行编号与文件行序": True,
           "ROOT先激活事前YAML和tar唯一同行标志": "TASK09_F3_MAIN_AGG_GATE:v2",
           "ROOT再激活完整证据清单SHA唯一同行标志": "TASK09_F3_MAIN_EVIDENCE:v2"}
    with pytest.raises(ValueError, match="v2|修订"):
        validate_task09_main_registry(original)
    validate_task09_main_registry(new)
    with pytest.raises(ValueError, match="v2|修订"):
        validate_task09_main_registry({**new, "旧v1中文报告_SHA256": "0" * 64})


def test_catalog_generator_uses_fixed_identities_and_unique_energy_pairs(
    tmp_path: Path,
) -> None:
    registered, expected = _catalog_fixture(tmp_path)
    built = build_task09_main_evidence_catalog(registered, project_root=tmp_path)
    assert built == expected
    first = expected["主序列十五份原件"][0]
    competing = tmp_path / first["canonical目录"] / "独立原能源_观测最佳_20260916T091201+0800"
    competing.mkdir()
    with pytest.raises(ValueError, match="恰唯一"):
        build_task09_main_evidence_catalog(registered, project_root=tmp_path)
