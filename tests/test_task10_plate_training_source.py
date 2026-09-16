"""A separate one-dimensional Task 10 trainer must receive only permitted inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sic_cu.config import PROJECT_ROOT
from sic_cu.data.common import sha256_file


REGISTRATION = (PROJECT_ROOT / "研究记录/任务10_独立双层场基准/"
                "正式人为多热流入场前登记.yaml")
TIMES = np.array([0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200], dtype=float)
DEPTHS = np.array([0.0, 0.013, 0.016], dtype=float)


@pytest.fixture
def limited_source(tmp_path: Path) -> Path:
    train = tmp_path / "HF_允许探针/训练_50000.npz"
    val = tmp_path / "HF_允许探针/验证_30000.npz"
    same = tmp_path / "LF_训练场/同物理_50000.npz"
    mismatch = tmp_path / "LF_训练场/接触失配_50000.npz"
    train.parent.mkdir(parents=True)
    same.parent.mkdir(parents=True)
    for filename, q in ((train, 50000), (val, 30000)):
        np.savez_compressed(filename, time_s=TIMES,
                            depths_from_top_m=DEPTHS, flux_w_m2=np.array(q),
                            temperature_k=np.arange(42, dtype=float).reshape(14, 3) + 295.15)
    depth = np.r_[((np.arange(24) + .5) * .012 / 24),
                  .012 + (np.arange(11) + .5) * .0055 / 11]
    material = np.r_[np.ones(24, dtype=np.int8), np.zeros(11, dtype=np.int8)]
    for filename, correction in ((same, 0.0), (mismatch, -1.5)):
        names = {name: np.zeros(101) for name in
                 ("top_surface_temperature_k", "bottom_outward_flux_w_m2",
                  "storage_rate_per_area_w_m2", "balance_per_area_w_m2",
                  "interface_flux_w_m2", "interface_temperature_jump_k")}
        np.savez_compressed(filename, time_s=np.arange(101, dtype=float) * 2,
                            node_depth_m=depth, material_id=material,
                            temperature_k=np.arange(101 * 35, dtype=float).reshape(101, 35)
                                          + 295.15 + correction, **names)
    paths = (train, val, same, mismatch)
    (tmp_path / "探针与源场SHA清单.json").write_text(json.dumps({
        "登记SHA256": sha256_file(REGISTRATION),
        "归档文件_SHA256": {
            str(path.relative_to(tmp_path)): sha256_file(path) for path in paths
        },
    }, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def _training_source(root: Path, method: str, lf_source: str | None = None):
    from sic_cu.eval.task10_plate_training_source import Task10PlateTrainingSource

    manifest_sha = sha256_file(root / "探针与源场SHA清单.json")
    return Task10PlateTrainingSource(REGISTRATION, root, method, lf_source,
                                     manifest_sha256=manifest_sha)


def test_plate_training_source_has_distinct_schema_not_old_rz_power(limited_source: Path) -> None:
    source = _training_source(limited_source, "F1")
    batch = source.training_probe(50000)
    assert batch.coordinates_z_t_q_material.shape == (42, 4)
    assert batch.temperature_k.shape == (42, 1)
    assert batch.kind == "probe" and batch.fold == "train"
    assert batch.coordinates_z_t_q_material[0].tolist() == [0.0, 0.0, 50000.0, 1.0]
    assert batch.coordinates_z_t_q_material[1].tolist() == [0.013, 0.0, 50000.0, 0.0]
    assert batch.coordinates_z_t_q_material[-1].tolist() == [0.016, 200.0, 50000.0, 0.0]
    np.testing.assert_array_equal(batch.temperature_k.ravel(),
                                  np.arange(42, dtype=float) + 295.15)
    assert batch.coordinates_z_t_q_material.shape[-1] != 5


def test_f1_hf_only_never_accepts_either_lf_source(limited_source: Path) -> None:
    with pytest.raises(ValueError, match="F1|LF|HF-only"):
        _training_source(limited_source, "F1", "lf_same_physics_coarse")
    pure = _training_source(limited_source, "F1")
    with pytest.raises(PermissionError, match="F1|LF|HF-only"):
        pure.training_low_fidelity(50000)
    with pytest.raises(ValueError, match="F1|F2|F3"):
        _training_source(limited_source, "E0")


@pytest.mark.parametrize("method", ["F2", "F3"])
def test_f2_f3_each_train_with_one_pre_registered_lf_physics(
    limited_source: Path, method: str,
) -> None:
    with pytest.raises(ValueError, match="LF|低保真"):
        _training_source(limited_source, method)
    same = _training_source(limited_source, method, "lf_same_physics_coarse")
    mismatch = _training_source(limited_source, method, "lf_contact_mismatch_coarse")
    assert same.training_probe(50000).coordinates_z_t_q_material.shape == (42, 4)
    a, b = same.training_low_fidelity(50000), mismatch.training_low_fidelity(50000)
    assert a.coordinates_z_t_q_material.shape == (101 * 35, 4)
    assert b.coordinates_z_t_q_material.shape == (101 * 35, 4)
    assert a.temperature_k.shape == (101 * 35, 1)
    assert a.coordinates_z_t_q_material[0].tolist() == [.012 / 48, 0.0, 50000.0, 1.0]
    assert a.coordinates_z_t_q_material[-1][1:].tolist() == [200.0, 50000.0, 0.0]
    assert a.temperature_k[0, 0] - b.temperature_k[0, 0] == 1.5
    assert a.kind == "lf_same_physics_coarse"
    assert b.kind == "lf_contact_mismatch_coarse"


def test_validation_only_uses_its_own_three_probe_positions_and_times(limited_source: Path) -> None:
    source = _training_source(limited_source, "F3", "lf_contact_mismatch_coarse")
    valid = source.validation_probe(30000)
    assert valid.kind == "probe" and valid.fold == "validation"
    assert valid.coordinates_z_t_q_material.shape == (42, 4)
    assert np.unique(valid.coordinates_z_t_q_material[:, 2]).tolist() == [30000.0]
    with pytest.raises(PermissionError, match="训练|train"):
        source.validation_probe(50000)
    with pytest.raises(PermissionError, match="验证|validation"):
        source.training_probe(30000)


def test_hidden_and_complete_hf_never_enter_training_or_validation(limited_source: Path) -> None:
    source = _training_source(limited_source, "F2", "lf_same_physics_coarse")
    for power in (42000, 58000):
        with pytest.raises(PermissionError):
            source.training_probe(power)
        with pytest.raises(PermissionError):
            source.validation_probe(power)
        with pytest.raises(PermissionError):
            source.training_low_fidelity(power)
    with pytest.raises(PermissionError, match="HF|完整|hidden"):
        source.open_full_hf(50000)
    with pytest.raises(PermissionError, match="HF|完整|hidden"):
        source.open_full_hf(42000)


def test_source_manifest_is_pinned_and_rechecked_before_any_future_training_read(
    limited_source: Path,
) -> None:
    from sic_cu.eval.task10_plate_training_source import Task10PlateTrainingSource

    original_sha = sha256_file(limited_source / "探针与源场SHA清单.json")
    with pytest.raises(ValueError, match="SHA|清单"):
        Task10PlateTrainingSource(REGISTRATION, limited_source, "F1",
                                  manifest_sha256="0" * 64)
    trained = _training_source(limited_source, "F1")
    manifest = limited_source / "探针与源场SHA清单.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert sha256_file(manifest) != original_sha
    with pytest.raises(ValueError, match="SHA|清单"):
        trained.training_probe(50000)


def test_manifest_symlink_cannot_move_training_reads_outside_project(limited_source: Path) -> None:
    source = _training_source(limited_source, "F1")
    manifest = limited_source / "探针与源场SHA清单.json"
    manifest.rename(limited_source / "测试清单原件保留.json")
    manifest.symlink_to("/tmp/任10同板方法来源不得项目外读_此处无文件")
    with pytest.raises(ValueError, match="项目|project"):
        source.training_probe(50000)
