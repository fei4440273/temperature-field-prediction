from __future__ import annotations

from sic_cu.config import load_yaml


def test_local_training_configuration_uses_one_cuda_device() -> None:
    training = load_yaml("configs/training.yaml")

    assert training["device"] == "cuda"
    assert training["distributed"]["enabled"] is False
    assert training["distributed"]["world_size"] == 1
