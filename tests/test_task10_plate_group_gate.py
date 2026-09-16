"""Synthetic-only evidence for mandatory all-25 board-model gate."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import random

import pytest
import torch

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.eval.task10_plate_benchmark import REGISTRATION, seal_evaluation_model
from sic_cu.models.task10_plate_deeponet import BoardDeepONet, BoardMultifidelityDeepONet


ARCHIVE = PROJECT_ROOT / ("研究记录/任务10_独立双层场基准/"
                          "九热流受限数值源_台账确认后_20260916T032402+0800")
MANIFEST_SHA = "b59bc80c678faa29224e564dc3f4dd7d61903fc59d6c7e6c2274e1a59651bedb"
BUDGET = PROJECT_ROOT / ("研究记录/任务10_独立双层场基准/"
                         "正式同板F1_F2_F3重训方法预算前登记_v2.yaml")
METHODS_UNIT_SOURCE = PROJECT_ROOT / ("研究记录/任务10_独立双层场基准/"
                                      "正式同板方法九源源码冻结_20260916T041718+0800.tar.gz")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fake_optimizer_state(model: torch.nn.Module, steps: int) -> dict:
    """A synthetically forged state to check structure, never proof of 800 updates."""
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=.001,weight_decay=.000001)
    for parameter in (p for p in model.parameters() if p.requires_grad):
        optimizer.state[parameter] = {"step": torch.tensor(float(steps)),
                                      "exp_avg": torch.zeros_like(parameter),
                                      "exp_avg_sq": torch.zeros_like(parameter)}
    return optimizer.state_dict()


@pytest.fixture
def synthetic_25_locked_boards(tmp_path: Path) -> tuple[Path, Path]:
    """Fake scores/checkpoint bytes, not trained weights or reference temperatures."""
    registration_sha = sha256_file(REGISTRATION)
    budget_sha = sha256_file(BUDGET)
    methods_sha = sha256_file(METHODS_UNIT_SOURCE)
    group = {"登记SHA256": registration_sha, "受限源清单_SHA256": MANIFEST_SHA,
             "方法预算_SHA256": budget_sha, "方法源码tar_SHA256": methods_sha,
             "方法源码tar文件": str(METHODS_UNIT_SOURCE), "身份": []}
    for method, kinds in (("F1", (None,)),
                          ("F2", ("lf_same_physics_coarse", "lf_contact_mismatch_coarse")),
                          ("F3", ("lf_same_physics_coarse", "lf_contact_mismatch_coarse"))):
        for kind in kinds:
            for seed in range(5):
                folder = tmp_path / f"synthetic_{method}_{kind}_{seed}"
                folder.mkdir()
                checkpoint = folder / "合法验证选定板模型_state_dict.pt"
                torch.manual_seed(seed)
                low = BoardDeepONet(width=32,latent_dim=32,blocks=1)
                if method == "F1":
                    model = low
                else:
                    model = BoardMultifidelityDeepONet(
                        low, BoardDeepONet(width=32,latent_dim=32,blocks=1,
                                          with_lf_input=True), method)
                torch.save(model.state_dict(),checkpoint)  # Untrained synthetic state.
                model_sha = sha256_file(checkpoint)
                machines = {
                    "最佳HF模型状态": ("最佳HF模型_state_dict.pt",model.state_dict()),
                    "最佳HF_AdamW机器状态": ("最佳HF_AdamW机器状态.pt",
                                             _fake_optimizer_state(model,800)),
                    "终态HF_AdamW机器状态": ("终态HF_AdamW机器状态.pt",
                                             _fake_optimizer_state(model,800)),
                    "终态HF模型状态": ("终态HF模型_state_dict.pt",model.state_dict()),
                }
                snapshot = {"python_random": random.getstate(),
                            "numpy_global": {"generator": "MT19937", "keys": list(range(624)),
                                             "position": 0, "has_gauss": 0,
                                             "cached_gaussian": 0.},
                            "torch_cpu": torch.get_rng_state().clone(), "torch_cuda_all": []}
                machines["最佳HF四类随机态"] = ("最佳HF四类随机态.pt",snapshot)
                machines["终态HF四类随机态"] = ("终态HF四类随机态.pt",snapshot)
                if kind is not None:
                    lf_model = BoardDeepONet(width=32,latent_dim=32,blocks=1)
                    machines["共享LF终态_AdamW机器状态"] = (
                        "共享LF终态_AdamW机器状态.pt",_fake_optimizer_state(lf_model,300))
                    machines["共享LF四类采样随机态"] = (
                        "共享LF四类采样随机态.pt",{**snapshot,
                        "numpy_sampler_start": {"state": {"state": 1}},
                        "numpy_sampler_final": {"state": {"state": 2}}})
                machine_shas = {}
                for name,(filename,state) in machines.items():
                    path = folder / filename
                    torch.save(state,path)
                    machine_shas[name+"_SHA256"] = sha256_file(path)
                hf_csv = folder / "逐步HF实际优化原始明细.csv"
                rows = [{"HF优化步": step, "训练热流_W_m2": (20000,35000,50000,65000,80000)[(step-1)%5],
                         "物理配点时刻_s": 5., "HF三探针消费点数": 42,
                         "实际训练目标": 1., "HF可见探针目标": .5,
                         "累计体内PDE计算次数": 0 if method == "F2" else step,
                         "合法两档探针验证RMSE_K": (2. - step / 800)
                         if step % 20 == 0 else None}
                        for step in range(1, 801)]
                _write_csv(hf_csv, rows)
                train = {"方法": method, "训练折": [20000,35000,50000,65000,80000],
                         "实际优化步数": 800, "最佳HF优化步数": 800,
                         "可见HF探针实际优化消费点数": 33600,
                         "体内PDE实际调用": 0 if method == "F2" else 800,
                         "HF空AdamW起步": True, "登记SHA256": registration_sha,
                         "模型_SHA256": model_sha, "方法预算_SHA256": budget_sha,
                         "方法源码tar_SHA256": methods_sha,
                         "正式受限数值源清单_SHA256": MANIFEST_SHA,
                         "源场清单SHA256": MANIFEST_SHA,
                         "随机种子": seed, "LF来源": kind,
                         "设备": "cpu",
                         "HF最佳可更新参数实测AdamW步数": 800,
                         "HF终态可更新参数实测AdamW步数": 800,
                         "HF可更新参数状态数": len(machines["终态HF_AdamW机器状态"][1]["state"]),
                         "HF逐步CSV_SHA256": sha256_file(hf_csv),
                         "完整HF内部真值进入训练": False, **machine_shas}
                if kind is not None:
                    lf_csv = folder / "同seed单源LF预训真实优化明细.csv"
                    lf_rows = [{"LF优化步": step,
                                "训练热流_W_m2": (20000,35000,50000,65000,80000)[(step-1)%5],
                                "LF训练消费点数": 512, "实际LF训练目标": 1.}
                               for step in range(1, 301)]
                    _write_csv(lf_csv, lf_rows)
                    pair_sha = sha256(f"{kind}:{seed}".encode()).hexdigest()
                    train.update({"共享LF逐步CSV_SHA256": sha256_file(lf_csv),
                                  "同seed共享LF真实优化步数": 300,
                                  "同seed共享LF真实监督消费点数": 300*512,
                                  "LF终态可更新参数实测AdamW步数": 300,
                                  "LF可更新参数状态数": len(machines[
                                      "共享LF终态_AdamW机器状态"][1]["state"]),
                                  "同seed共享LF初始全张量SHA256": pair_sha,
                                  "同seed共享HF校正初始全张量SHA256": pair_sha})
                val = {"验证折": [30000,70000], "合法验证仅用三探针": True,
                       "最佳步数": 800, "两档探针RMSE_K": 1.,
                       "逐档探针RMSE_K": {"30000": 1., "70000": 1.},
                       "登记SHA256": registration_sha, "模型_SHA256": model_sha,
                       "合法验证温度只来自三探针": True}
                train_file = folder / "真实训练日志.json"
                val_file = folder / "合法探针验证.json"
                train_file.write_text(json.dumps(train,ensure_ascii=False)+"\n", encoding="utf-8")
                val_file.write_text(json.dumps(val,ensure_ascii=False)+"\n", encoding="utf-8")
                lock = folder / "合法训练验证后模型SHA锁.json"
                seal_evaluation_model(REGISTRATION, ARCHIVE, checkpoint, train_file,
                                      val_file, lock)
                group["身份"].append({"方法": method, "随机种子": seed,
                                        "LF来源": kind, "模型目录": str(folder)})
    manifest = tmp_path / "合成25身份完全冻结机读清单.json"
    manifest.write_text(json.dumps(group, ensure_ascii=False)+"\n",encoding="utf-8")
    ledger = tmp_path / "单元测试非正式总账占位.txt"
    ledger.write_text(sha256_file(manifest) + "\n" + methods_sha + "\n",encoding="utf-8")
    return manifest, ledger


def test_all_25_csv_checkpoint_locks_and_legal_scores_can_be_preflighted_without_hf(
    synthetic_25_locked_boards: tuple[Path, Path],
) -> None:
    from sic_cu.eval.task10_plate_group_gate import verify_complete_group

    manifest, ledger = synthetic_25_locked_boards
    proof = verify_complete_group(manifest, group_sha256=sha256_file(manifest),
        registration_path=REGISTRATION, archive_root=ARCHIVE,
        budget_path=BUDGET, budget_sha256=sha256_file(BUDGET),
        ledger_path=ledger)
    assert proof["身份总数"] == 25
    assert proof["逐步CSV与机器状态结构及SHA校验通过"] is True
    assert proof["只读完整HF"] is False


def test_missing_identity_provenance_or_probe_score_is_red_before_hf(
    synthetic_25_locked_boards: tuple[Path, Path],
) -> None:
    from sic_cu.eval.task10_plate_group_gate import verify_complete_group

    manifest, ledger = synthetic_25_locked_boards
    original = json.loads(manifest.read_text(encoding="utf-8"))
    for mutate in (lambda x: x["身份"].pop(),
                   lambda x: x.__setitem__("受限源清单_SHA256", "0" * 64),
                   lambda x: x.__setitem__("方法源码tar文件", str(PROJECT_ROOT /
                      "研究记录/任务10_独立双层场基准/正式人为九热流源码冻结_20260916T032032+0800.tar.gz")),
                   lambda x: x["身份"][0].__setitem__("LF来源", "lf_same_physics_coarse")):
        item = json.loads(json.dumps(original))
        mutate(item)
        bad = manifest.parent / f"恶意清单_{len(list(manifest.parent.glob('恶意清单_*')))}.json"
        bad.write_text(json.dumps(item,ensure_ascii=False)+"\n",encoding="utf-8")
        ledger.write_text(sha256_file(bad) + "\n" + original["方法源码tar_SHA256"] + "\n",
                          encoding="utf-8")
        with pytest.raises((PermissionError, ValueError), match="身份|来源|源|F1|25"):
            verify_complete_group(bad,group_sha256=sha256_file(bad),
                registration_path=REGISTRATION,archive_root=ARCHIVE,
                budget_path=BUDGET,budget_sha256=sha256_file(BUDGET),ledger_path=ledger)
    first_folder = Path(original["身份"][0]["模型目录"])
    validation = first_folder / "合法探针验证.json"
    report = json.loads(validation.read_text(encoding="utf-8"))
    report["两档探针RMSE_K"] = -1
    validation.write_text(json.dumps(report,ensure_ascii=False)+"\n",encoding="utf-8")
    ledger.write_text(sha256_file(manifest) + "\n" + original["方法源码tar_SHA256"] + "\n",
                      encoding="utf-8")
    with pytest.raises((PermissionError, ValueError), match="验证|分数|SHA"):
        verify_complete_group(manifest,group_sha256=sha256_file(manifest),
            registration_path=REGISTRATION,archive_root=ARCHIVE,
            budget_path=BUDGET,budget_sha256=sha256_file(BUDGET),ledger_path=ledger)


def test_no_hidden_or_any_validation_full_hf_before_frozen_metrics(
    synthetic_25_locked_boards: tuple[Path, Path],
) -> None:
    from sic_cu.eval.task10_plate_group_gate import BoardCompleteGroupEvaluationGate

    manifest, ledger = synthetic_25_locked_boards
    gate = BoardCompleteGroupEvaluationGate(manifest, group_sha256=sha256_file(manifest),
        registration_path=REGISTRATION, archive_root=ARCHIVE,
        budget_path=BUDGET,budget_sha256=sha256_file(BUDGET),ledger_path=ledger)
    for flux in (30000, 42000):
        with pytest.raises(PermissionError,match="指标|源码|25"):
            gate.open_full_reference(flux)


def test_fake_csv_claiming_41_hf_points_with_recomputed_shas_still_red(
    synthetic_25_locked_boards: tuple[Path, Path],
) -> None:
    from sic_cu.eval.task10_plate_group_gate import verify_complete_group

    manifest, ledger = synthetic_25_locked_boards
    group = json.loads(manifest.read_text(encoding="utf-8"))
    folder = Path(group["身份"][0]["模型目录"])
    hf_file = folder / "逐步HF实际优化原始明细.csv"
    with hf_file.open(newline="",encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    rows[0]["HF三探针消费点数"] = "41"
    _write_csv(hf_file, rows)
    train_file = folder / "真实训练日志.json"
    train = json.loads(train_file.read_text(encoding="utf-8"))
    train["HF逐步CSV_SHA256"] = sha256_file(hf_file)
    train_file.write_text(json.dumps(train,ensure_ascii=False)+"\n",encoding="utf-8")
    lock_file = folder / "合法训练验证后模型SHA锁.json"
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    lock["训练日志_SHA256"] = sha256_file(train_file)
    lock_file.write_text(json.dumps(lock,ensure_ascii=False)+"\n",encoding="utf-8")
    with pytest.raises(PermissionError,match="消费|点数|CSV|HF"):
        verify_complete_group(manifest,group_sha256=sha256_file(manifest),
            registration_path=REGISTRATION,archive_root=ARCHIVE,
            budget_path=BUDGET,budget_sha256=sha256_file(BUDGET),ledger_path=ledger)


def test_missing_best_adamw_machine_original_is_red_despite_group_receipts(
    synthetic_25_locked_boards: tuple[Path, Path],
) -> None:
    from sic_cu.eval.task10_plate_group_gate import verify_complete_group

    manifest, ledger = synthetic_25_locked_boards
    folder = Path(json.loads(manifest.read_text(encoding="utf-8"))["身份"][0]["模型目录"])
    missing = folder / "最佳HF_AdamW机器状态.pt"
    missing.rename(folder / "最佳HF_AdamW机器状态_单元测试可恢复保管.pt")
    with pytest.raises(PermissionError,match="AdamW|优化器|机器|状态"):
        verify_complete_group(manifest,group_sha256=sha256_file(manifest),
            registration_path=REGISTRATION,archive_root=ARCHIVE,
            budget_path=BUDGET,budget_sha256=sha256_file(BUDGET),ledger_path=ledger)


def test_better_earlier_val_csv_and_different_pair_lf_machine_sha_are_red(
    synthetic_25_locked_boards: tuple[Path, Path],
) -> None:
    from sic_cu.eval.task10_plate_group_gate import verify_complete_group

    manifest, ledger = synthetic_25_locked_boards
    group = json.loads(manifest.read_text(encoding="utf-8"))
    f1 = Path(group["身份"][0]["模型目录"])
    hf_file = f1 / "逐步HF实际优化原始明细.csv"
    with hf_file.open(newline="",encoding="utf-8-sig") as original:
        rows = list(csv.DictReader(original))
    rows[19]["合法两档探针验证RMSE_K"] = "0.01"
    _write_csv(hf_file,rows)
    train_path, lock_path = f1 / "真实训练日志.json", f1 / "合法训练验证后模型SHA锁.json"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    train["HF逐步CSV_SHA256"] = sha256_file(hf_file)
    train_path.write_text(json.dumps(train,ensure_ascii=False)+"\n",encoding="utf-8")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["训练日志_SHA256"] = sha256_file(train_path)
    lock_path.write_text(json.dumps(lock,ensure_ascii=False)+"\n",encoding="utf-8")
    with pytest.raises(PermissionError,match="最佳|最低|分数"):
        verify_complete_group(manifest,group_sha256=sha256_file(manifest),
            registration_path=REGISTRATION,archive_root=ARCHIVE,
            budget_path=BUDGET,budget_sha256=sha256_file(BUDGET),ledger_path=ledger)

    # Separate fixture edit: restore F1 receipt before targeting paired LF state.
    rows[19]["合法两档探针验证RMSE_K"] = str(2. - 20 / 800)
    _write_csv(hf_file,rows)
    train["HF逐步CSV_SHA256"] = sha256_file(hf_file)
    train_path.write_text(json.dumps(train,ensure_ascii=False)+"\n",encoding="utf-8")
    lock["训练日志_SHA256"] = sha256_file(train_path)
    lock_path.write_text(json.dumps(lock,ensure_ascii=False)+"\n",encoding="utf-8")
    f3 = next(Path(item["模型目录"]) for item in group["身份"]
              if item["方法"] == "F3" and item["随机种子"] == 0
              and item["LF来源"] == "lf_same_physics_coarse")
    lf_rng = f3 / "共享LF四类采样随机态.pt"
    value = torch.load(lf_rng,map_location="cpu",weights_only=True)
    value["numpy_sampler_final"] = {"state": {"state": 3}}
    torch.save(value,lf_rng)
    f3_train = f3 / "真实训练日志.json"
    changed = json.loads(f3_train.read_text(encoding="utf-8"))
    changed["共享LF四类采样随机态_SHA256"] = sha256_file(lf_rng)
    f3_train.write_text(json.dumps(changed,ensure_ascii=False)+"\n",encoding="utf-8")
    f3_lock = f3 / "合法训练验证后模型SHA锁.json"
    lock = json.loads(f3_lock.read_text(encoding="utf-8"))
    lock["训练日志_SHA256"] = sha256_file(f3_train)
    f3_lock.write_text(json.dumps(lock,ensure_ascii=False)+"\n",encoding="utf-8")
    with pytest.raises(PermissionError,match="同LF|随机|机器|配对"):
        verify_complete_group(manifest,group_sha256=sha256_file(manifest),
            registration_path=REGISTRATION,archive_root=ARCHIVE,
            budget_path=BUDGET,budget_sha256=sha256_file(BUDGET),ledger_path=ledger)
