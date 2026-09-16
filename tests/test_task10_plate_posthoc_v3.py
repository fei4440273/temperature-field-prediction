"""Synthetic-only regression for the independent Task 10 full-HF gate v3."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json

import numpy as np
import pytest
import torch

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file
from sic_cu.models.task10_plate_deeponet import BoardDeepONet, BoardMultifidelityDeepONet
from test_task10_plate_group_gate import synthetic_25_locked_boards


def _low_tensor_sha(weights: dict[str, torch.Tensor]) -> str:
    digest = sha256()
    for name, tensor in sorted(weights.items()):
        if name.startswith("low_model."):
            stripped = name[len("low_model."):]
            value = tensor.detach().cpu().contiguous()
            digest.update(f"{stripped}:{value.dtype}:{tuple(value.shape)}\n".encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _synthetic_pair(tmp_path):
    left = tmp_path / "F2_同LF合成夹具"
    right = tmp_path / "F3_同LF合成夹具"
    left.mkdir()
    right.mkdir()
    torch.manual_seed(5)
    low = BoardDeepONet(width=32, latent_dim=32, blocks=1)
    correction = BoardDeepONet(width=32, latent_dim=32, blocks=1, with_lf_input=True)
    digest = None
    for folder, method in ((left, "F2"), (right, "F3")):
        model = BoardMultifidelityDeepONet(low, correction, method)
        weights = model.state_dict()
        digest = _low_tensor_sha(weights)
        best = folder / "合法验证选定板模型_state_dict.pt"
        final = folder / "终态HF模型_state_dict.pt"
        torch.save(weights, best)
        torch.save(weights, final)
        train = folder / "真实训练日志.json"
        train.write_text(json.dumps({
            "方法": method, "同seed共享LF初始全张量SHA256": digest,
        }, ensure_ascii=False), encoding="utf-8")
        lock = folder / "合法训练验证后模型SHA锁.json"
        lock.write_text(json.dumps({
            "模型文件": str(best), "模型_SHA256": sha256_file(best),
            "训练日志文件": str(train), "训练日志_SHA256": sha256_file(train),
        }, ensure_ascii=False), encoding="utf-8")
    return left, right, digest


def test_true_frozen_lf_tensors_and_logged_sha_pass_synthetic_pair(tmp_path):
    from sic_cu.eval.task10_plate_posthoc_v3 import verify_single_frozen_lf_pair

    f2, f3, expected = _synthetic_pair(tmp_path)
    result = verify_single_frozen_lf_pair(f2, f3)
    assert result["同LF现场全张量_SHA256"] == expected
    assert result["F2最佳模型_SHA256"] == sha256_file(
        f2 / "合法验证选定板模型_state_dict.pt")
    assert result["F3终态模型_SHA256"] == sha256_file(
        f3 / "终态HF模型_state_dict.pt")


def test_f2_f3_would_be_falsely_paired_if_its_best_real_lf_weight_diverges(tmp_path):
    from sic_cu.eval.task10_plate_posthoc_v3 import verify_single_frozen_lf_pair

    f2, f3, _expected = _synthetic_pair(tmp_path)
    target = f3 / "合法验证选定板模型_state_dict.pt"
    changed = torch.load(target, map_location="cpu", weights_only=True)
    changed["low_model.bias"] += 1.0
    torch.save(changed, target)
    with pytest.raises(PermissionError, match="LF|张量|SHA"):
        verify_single_frozen_lf_pair(f2, f3)


def test_a_wrong_logged_lf_sha_cannot_be_salvaged_by_equal_checkpoint_weights(tmp_path):
    from sic_cu.eval.task10_plate_posthoc_v3 import verify_single_frozen_lf_pair

    f2, f3, _expected = _synthetic_pair(tmp_path)
    train = f3 / "真实训练日志.json"
    record = json.loads(train.read_text(encoding="utf-8"))
    record["同seed共享LF初始全张量SHA256"] = "0" * 64
    train.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(PermissionError, match="LF|张量|SHA"):
        verify_single_frozen_lf_pair(f2, f3)


def _synthetic_25_real_lf_weights(tmp_path):
    identities = []
    for seed in range(5):
        folder = tmp_path / f"F1_{seed}"
        folder.mkdir()
        model = BoardDeepONet(width=32, latent_dim=32, blocks=1)
        best = folder / "合法验证选定板模型_state_dict.pt"
        torch.save(model.state_dict(), best)
        train = folder / "真实训练日志.json"
        train.write_text(json.dumps({"方法": "F1"}), encoding="utf-8")
        (folder / "合法训练验证后模型SHA锁.json").write_text(json.dumps({
            "模型文件": str(best), "模型_SHA256": sha256_file(best),
            "训练日志文件": str(train), "训练日志_SHA256": sha256_file(train),
        }, ensure_ascii=False), encoding="utf-8")
        identities.append({"方法": "F1", "随机种子": seed,
                           "LF来源": None, "模型目录": str(folder)})
    for kind in ("lf_same_physics_coarse", "lf_contact_mismatch_coarse"):
        for seed in range(5):
            base = tmp_path / f"pair_{kind}_{seed}"
            base.mkdir()
            f2, f3, _expected = _synthetic_pair(base)
            for method, folder in (("F2", f2), ("F3", f3)):
                identities.append({"方法": method, "随机种子": seed,
                                   "LF来源": kind, "模型目录": str(folder)})
    assert len(identities) == 25
    return identities


def test_25_locked_real_lf_branches_produce_ten_pair_source_receipts(tmp_path):
    from sic_cu.eval.task10_plate_posthoc_v3 import verify_25_training_provenance

    identities = _synthetic_25_real_lf_weights(tmp_path)
    report = verify_25_training_provenance(identities)
    assert report["25份最佳原模型锁SHA核对"] is True
    assert len(report["10组真实冻结LF配对原件"]) == 10
    assert all(item["同LF现场全张量_SHA256"] == item["声明同LF全张量_SHA256"]
               for item in report["10组真实冻结LF配对原件"])


def test_all_25_fake_machine_receipts_cannot_replace_actual_lf_tensor_sha(
    synthetic_25_locked_boards,
):
    from sic_cu.eval.task10_plate_posthoc_v3 import verify_25_training_provenance

    group, _ledger = synthetic_25_locked_boards
    identities = json.loads(group.read_text(encoding="utf-8"))["身份"]
    with pytest.raises(PermissionError, match="LF|全张量|SHA"):
        verify_25_training_provenance(identities)


def test_25_snapshot_rechecks_original_best_lock_before_any_reference(tmp_path):
    from sic_cu.eval.task10_plate_posthoc_v3 import verify_25_training_provenance

    identities = _synthetic_25_real_lf_weights(tmp_path)
    target = tmp_path / "F1_3" / "合法验证选定板模型_state_dict.pt"
    weights = torch.load(target, map_location="cpu", weights_only=True)
    weights["bias"] += 1
    torch.save(weights, target)
    with pytest.raises(PermissionError, match="SHA|锁|最佳"):
        verify_25_training_provenance(identities)


def test_model_forward_sees_only_reference_coordinates_not_hf_temperature():
    from sic_cu.eval.task10_plate_posthoc_v3 import coordinate_reference_only

    reference = {
        "time_s": np.array([0, 1, 2]),
        "node_depth_m": np.array([.0001, .015]),
        "material_id": np.array([1, 0]),
        "temperature_k": np.full((3, 2), 310.0),
        "interface_temperature_jump_k": np.ones(3),
    }
    coordinates = coordinate_reference_only(reference)
    assert set(coordinates) == {"time_s", "node_depth_m", "material_id"}
    assert all(coordinates[key] is reference[key] for key in coordinates)
    assert "temperature_k" not in coordinates


def test_best_checkpoint_reloading_cannot_accept_updated_lock_after_first_gate(tmp_path):
    from sic_cu.eval.task10_plate_posthoc_v3 import reload_locked_best_plate

    identities = _synthetic_25_real_lf_weights(tmp_path)
    f1 = next(identity for identity in identities if identity["方法"] == "F1"
              and identity["随机种子"] == 2)
    folder = tmp_path / "F1_2"
    original_sha = sha256_file(folder / "合法验证选定板模型_state_dict.pt")
    target = folder / "合法验证选定板模型_state_dict.pt"
    changed = torch.load(target, map_location="cpu", weights_only=True)
    changed["bias"] += 1.0
    torch.save(changed, target)
    lock_path = folder / "合法训练验证后模型SHA锁.json"
    rewritten = json.loads(lock_path.read_text(encoding="utf-8"))
    rewritten["模型_SHA256"] = sha256_file(target)
    lock_path.write_text(json.dumps(rewritten, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(PermissionError, match="SHA|原锁|最佳"):
        reload_locked_best_plate(f1, expected_sha256=original_sha,
                                 device=torch.device("cpu"))


def test_formal_v3_never_makes_directory_for_forged_only_one_board(tmp_path):
    from sic_cu.eval.task10_plate_posthoc_v3 import run_complete_plate_posthoc_v3

    root = PROJECT_ROOT / "研究记录/任务10_独立双层场基准"
    reference_root = root / "九热流受限数值源_台账确认后_20260916T032402+0800"
    group = tmp_path / "仅一模型不是全组.json"
    group.write_text(json.dumps({"身份": []}, ensure_ascii=False), encoding="utf-8")
    destination = tmp_path / "不许存在的后验HF输出"
    with pytest.raises(PermissionError):
        run_complete_plate_posthoc_v3(
            output_directory=destination, group_path=group,
            group_sha256=sha256_file(group),
            registration_path=root / "正式人为多热流入场前登记.yaml",
            archive_root=reference_root,
            method_budget_path=root / "正式同板F1_F2_F3重训方法预算前登记_v2.yaml",
            method_budget_sha256="cf05d16a3cd754e7f75321418ec6186e42673e33f8774492f3eea985105fba94",
            original_posthoc_budget_path=root / "正式同板全组后验指标及源码前登记.yaml",
            original_posthoc_budget_sha256="5e69e7b29aacd11e5a1923aff03974dd777d754676ee9463a2b15c4d810436da",
            original_posthoc_tar_path=root / "正式同板后验十源事前冻结.tar.gz",
            original_posthoc_tar_sha256="10777b6e0b99e6b8b4563b59e2d9d575505eda6cdea49235fca02ba1f1c54fb0",
            v3_budget_path=tmp_path / "无新预算.yaml", v3_budget_sha256="0" * 64,
            v3_tar_path=tmp_path / "无新归档.tar.gz", v3_tar_sha256="1" * 64,
            ledger_path=tmp_path / "未登记主账.md",
            device=torch.device("cpu"),
        )
    assert not destination.exists()


def test_exact_0070_row_contains_both_real_budget_and_tar_sha():
    from sic_cu.eval.task10_plate_posthoc_v3 import is_registered_v3_row

    budget, archive = "a" * 64, "b" * 64
    assert not is_registered_v3_row("| 录-0069 | " + budget + " / " + archive + " |",
                                    budget, archive)
    assert not is_registered_v3_row("| 录-0070 | " + budget + " |", budget, archive)
    assert is_registered_v3_row("| 录-0070 | " + budget + " / " + archive + " |",
                                budget, archive)


def test_v3_two_source_shas_must_be_in_the_actual_reserved_0070_row(tmp_path):
    from sic_cu.eval.task10_plate_posthoc_v3 import verify_v3_source_freeze

    root = PROJECT_ROOT / "研究记录/任务10_独立双层场基准"
    budget = root / "正式同板全组后验真实LF配对v3前登记.yaml"
    archive = root / "正式同板后验真实LF配对v3七源事前冻结.tar.gz"
    budget_sha, archive_sha = sha256_file(budget), sha256_file(archive)
    ledger = tmp_path / "纯合成机读门禁主账.md"
    ledger.write_text("| 录-0069 | " + budget_sha + " / " + archive_sha + " |\n",
                      encoding="utf-8")
    arguments = {
        "v3_budget_path": budget, "v3_budget_sha256": budget_sha,
        "v3_tar_path": archive, "v3_tar_sha256": archive_sha,
        "group_sha256": "2" * 64,
        "method_budget_sha256":
        "cf05d16a3cd754e7f75321418ec6186e42673e33f8774492f3eea985105fba94",
        "original_posthoc_budget_sha256":
        "5e69e7b29aacd11e5a1923aff03974dd777d754676ee9463a2b15c4d810436da",
        "original_posthoc_tar_sha256":
        "10777b6e0b99e6b8b4563b59e2d9d575505eda6cdea49235fca02ba1f1c54fb0",
        "ledger_path": ledger,
    }
    with pytest.raises(PermissionError, match="0070|总账|事前"):
        verify_v3_source_freeze(**arguments)
    ledger.write_text("| 录-0070 | " + budget_sha + " / " + archive_sha + " |\n",
                      encoding="utf-8")
    proof = verify_v3_source_freeze(**arguments)
    assert proof["v3预算_SHA256"] == budget_sha
    assert proof["v3七源tar_SHA256"] == archive_sha
    assert proof["完整HF温度已读取"] is False


def test_unique_v3_cli_never_creates_output_before_complete_real_group(tmp_path):
    script = PROJECT_ROOT / "scripts/任务10_全组锁后独立一维HF后验审计_v3.py"
    spec = importlib.util.spec_from_file_location("task10_only_v3_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    group = tmp_path / "只锁一份假全组.json"
    group.write_text(json.dumps({"身份": []}, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "伪单模型不得出现目录"
    with pytest.raises(PermissionError):
        module.main([
            "--全组身份文件", str(group),
            "--全组SHA256", sha256_file(group),
            "--v3预算SHA256", "0" * 64,
            "--v3七源tarSHA256", "1" * 64,
            "--输出目录", str(output),
        ])
    assert not output.exists()
