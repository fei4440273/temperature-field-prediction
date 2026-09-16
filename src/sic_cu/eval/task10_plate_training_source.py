"""One-dimensional Task 10 method batches from the frozen restricted source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sic_cu.data.common import sha256_file
from sic_cu.eval.task10_plate_benchmark import Task10ObservationView, _require_project_file


@dataclass(frozen=True)
class PlateTrainingBatch:
    """Rows are [top depth m, time s, synthetic flux W/m2, material id]."""

    coordinates_z_t_q_material: np.ndarray
    temperature_k: np.ndarray
    fold: str
    kind: str


class Task10PlateTrainingSource:
    """Only a preregistered trainer input, not a raw full-HF archive reader."""

    _LF_KINDS = ("lf_same_physics_coarse", "lf_contact_mismatch_coarse")

    def __init__(self, registration_path: str | Path, archive_root: str | Path,
                 method: str, lf_source: str | None = None, *,
                 manifest_sha256: str):
        if method not in ("F1", "F2", "F3"):
            raise ValueError("仅F1/F2/F3同一维板重新训练，旧RZ的E0模型不得直接输入")
        if (method == "F1" and lf_source is not None):
            raise ValueError("F1 HF-only只使用合法探针，不可消费任一LF源")
        if method in ("F2", "F3") and lf_source not in self._LF_KINDS:
            raise ValueError("F2/F3必须各自固定单一已登记的LF低保真训练来源")
        if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
            raise ValueError("源场清单SHA256须训练前独立字节封存")
        self.method = method
        self.lf_source = lf_source
        self._training = Task10ObservationView(registration_path, archive_root, "train")
        self._validation = Task10ObservationView(registration_path, archive_root, "validation")
        self._manifest_sha = manifest_sha256
        self._check_manifest()

    def _check_manifest(self) -> None:
        index = self._training.archive_root / "探针与源场SHA清单.json"
        if sha256_file(_require_project_file(index)) != self._manifest_sha:
            raise ValueError("训练/验证源场清单SHA漂移，禁止继续读取合法温度")

    def _probe(self, mode: str, q: int) -> PlateTrainingBatch:
        self._check_manifest()
        view = self._training if mode == "train" else self._validation
        snapshot = view.open_input("probe", q)
        times = snapshot["time_s"]
        depths = snapshot["depths_from_top_m"]
        interface = float(view.setup["geometry"]["silicon_carbide_thickness_m"])
        material = np.where(depths < interface, 1.0, 0.0)
        coordinates = np.column_stack((
            np.tile(depths, len(times)), np.repeat(times, len(depths)),
            np.full(len(times) * len(depths), float(q)), np.tile(material, len(times)),
        ))
        return PlateTrainingBatch(coordinates, snapshot["temperature_k"].reshape(-1, 1),
                                  mode, "probe")

    def training_probe(self, flux_w_m2: int) -> PlateTrainingBatch:
        return self._probe("train", flux_w_m2)

    def validation_probe(self, flux_w_m2: int) -> PlateTrainingBatch:
        return self._probe("validation", flux_w_m2)

    def training_low_fidelity(self, flux_w_m2: int) -> PlateTrainingBatch:
        if self.method == "F1":
            raise PermissionError("F1 HF-only禁止打开任何LF完整温度")
        self._check_manifest()
        snapshot = self._training.open_input(self.lf_source, flux_w_m2)
        coarse = self._training.setup["low_fidelity_sources"][
            "same_physics_coarse" if self.lf_source == "lf_same_physics_coarse"
            else "synthetic_contact_mismatch_coarse"]
        geometry = self._training.setup["geometry"]
        sic, cu = coarse["silicon_carbide_cells"], coarse["copper_cells"]
        widths = np.r_[np.full(sic, geometry["silicon_carbide_thickness_m"] / sic),
                       np.full(cu, geometry["copper_thickness_m"] / cu)]
        expected_depths = np.cumsum(widths) - widths / 2
        expected_material = np.r_[np.ones(sic, dtype=np.int8), np.zeros(cu, dtype=np.int8)]
        if (not np.allclose(snapshot["node_depth_m"], expected_depths, atol=1e-12, rtol=0)
                or not np.array_equal(snapshot["material_id"], expected_material)
                or not np.allclose(snapshot["time_s"],
                                   np.arange(len(snapshot["time_s"])) * coarse["time_step_s"],
                                   atol=1e-12, rtol=0)):
            raise ValueError("LF网格材料/时间坐标与预登记一维双层板不符")
        steps = len(snapshot["time_s"])
        cells = len(snapshot["node_depth_m"])
        coordinates = np.column_stack((
            np.tile(snapshot["node_depth_m"], steps),
            np.repeat(snapshot["time_s"], cells),
            np.full(steps * cells, float(flux_w_m2)),
            np.tile(snapshot["material_id"], steps),
        ))
        return PlateTrainingBatch(coordinates, snapshot["temperature_k"].reshape(-1, 1),
                                  "train", self.lf_source)

    def open_full_hf(self, flux_w_m2: int):
        raise PermissionError("F1/F2/F3训练或验证期间禁止访问完整HF/hidden参考温度")
