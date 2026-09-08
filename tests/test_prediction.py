from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from sic_cu.data.fields import load_processed_field
from sic_cu.prediction import Predictor, rotate_axisymmetric, stable_time_from_maximum
from sic_cu.models import MLPPINN, ModelScales
from sic_cu.visualization import export_prediction


def test_36w_is_exact_linear_power_interpolation() -> None:
    predictor = Predictor(interpolation_kind="linear", device="cpu")
    result = predictor.predict(36.0, times_s=[0.0, 20.0, 200.0])
    low = predictor.predict(30.0, times_s=[0.0, 20.0, 200.0])
    high = predictor.predict(40.0, times_s=[0.0, 20.0, 200.0])
    expected = 0.4 * low.mean_temperature_k + 0.6 * high.mean_temperature_k
    assert np.allclose(result.mean_temperature_k, expected)
    assert result.metadata.deterministic


@pytest.mark.parametrize("power", [-1.0, 800.01, 900.0])
def test_prediction_rejects_unsupported_power(power: float) -> None:
    with pytest.raises(ValueError, match="0-800"):
        Predictor(device="cpu").predict(power)


def test_zero_power_is_uniform_observed_initial_field() -> None:
    result = Predictor(device="cpu").predict(0.0, times_s=[0.0, 100.0, 200.0])
    assert np.allclose(result.mean_temperature_k, 295.15)
    assert result.stable_time_s == 0.0
    assert result.metadata.support_domain == "zero_power_physics_anchor"


def test_sub_10w_prediction_is_anchored_to_zero_and_10w() -> None:
    predictor = Predictor(device="cpu")
    zero = predictor.predict(0.0, times_s=[0.0, 100.0, 200.0])
    five = predictor.predict(5.0, times_s=[0.0, 100.0, 200.0])
    ten = predictor.predict(10.0, times_s=[0.0, 100.0, 200.0])
    assert np.allclose(five.mean_temperature_k, 0.5 * (zero.mean_temperature_k + ten.mean_temperature_k))
    assert five.metadata.support_domain == "between_zero_anchor_and_10W_simulation"


@pytest.mark.parametrize("power", [10.0, 800.0])
def test_simulation_power_boundaries_are_queryable(power: float) -> None:
    result = Predictor(device="cpu").predict(power, times_s=[0.0, 200.0])
    assert result.mean_temperature_k.shape == (2, 1159)
    assert result.metadata.support_domain == "inside_10_800W_simulation_domain"


def test_stable_time_uses_maximum_temperature_changes() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    maximum = np.array([20.0, 21.0, 21.005, 21.012, 21.013])
    assert stable_time_from_maximum(times, maximum, tolerance_k=0.01) == 1.0
    assert stable_time_from_maximum(times, maximum, tolerance_k=0.001) is None


def test_rotation_preserves_axisymmetric_values() -> None:
    coordinates = np.array([[0.01, -0.01], [0.02, 0.0]])
    xyz, temperature, materials = rotate_axisymmetric(
        coordinates,
        np.array([300.0, 400.0]),
        np.array([0, 1]),
        theta_resolution=8,
    )
    assert xyz.shape == (16, 3)
    assert np.allclose(np.hypot(xyz[:8, 0], xyz[:8, 1]), 0.01)
    assert np.all(temperature[:8] == 300.0)
    assert np.all(materials[8:] == 1)


def test_export_roundtrip(tmp_path) -> None:
    prediction = Predictor(device="cpu").predict(36.0, times_s=[0.0, 100.0, 200.0])
    paths = export_prediction(
        prediction,
        tmp_path,
        theta_resolution=4,
        vtk_times_s=(0.0,),
    )
    with np.load(paths["field"]) as saved:
        assert np.array_equal(saved["mean_temperature_k"], prediction.mean_temperature_k)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["power_w"] == 36.0
    assert len(paths["vtk"]) == 1
    assert "POINT_DATA" in (tmp_path / "vtk/field_0s.vtk").read_text()


def test_coordinate_checkpoint_roundtrip(tmp_path) -> None:
    model = MLPPINN(width=8, depth=2)
    torch.manual_seed(17)
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.1, 0.1)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "method": "mlp",
            "scales": ModelScales().__dict__,
            "model_kwargs": {"width": 8, "depth": 2, "activation": "tanh"},
            "model_state": model.state_dict(),
            "material_passport": {"experiment_data_used": False},
        },
        checkpoint,
    )
    result = Predictor(checkpoint, device="cpu").predict(36.0, times_s=[0.0, 2.0])
    assert result.mean_temperature_k.shape == (2, 1159)
    assert result.metadata.source.startswith("mlp:")
    assert any("without experiment data" in warning for warning in result.metadata.warnings)

    exact = Predictor(checkpoint, device="cpu").predict(36.0, times_s=[1.0])
    reference = load_processed_field(10.0)
    coordinate = torch.tensor(
        [[*reference.coordinates_rz_m[0], 1.0, 36.0]], dtype=torch.float32
    )
    with torch.no_grad():
        expected = model(coordinate).item()
    assert exact.mean_temperature_k[0, 0] == pytest.approx(expected, abs=1e-5)
