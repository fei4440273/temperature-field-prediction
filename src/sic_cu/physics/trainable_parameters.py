from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _logit(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError("Emissivity initialization must be strictly between 0 and 1")
    return math.log(value / (1.0 - value))


class TrainableBoundaryParameters(nn.Module):
    """Constrained scalar physics parameters used only during inverse identification."""

    def __init__(
        self,
        silicon_carbide_emissivity: float,
        copper_emissivity: float,
        contact_resistance_m2_k_w: float,
    ) -> None:
        super().__init__()
        if contact_resistance_m2_k_w <= 0:
            raise ValueError("Contact-resistance initialization must be positive")
        self.silicon_carbide_emissivity_logit = nn.Parameter(
            torch.tensor(_logit(silicon_carbide_emissivity), dtype=torch.float32)
        )
        self.copper_emissivity_logit = nn.Parameter(
            torch.tensor(_logit(copper_emissivity), dtype=torch.float32)
        )
        self.contact_resistance_log10 = nn.Parameter(
            torch.tensor(math.log10(contact_resistance_m2_k_w), dtype=torch.float32)
        )

    @property
    def silicon_carbide_emissivity(self) -> Tensor:
        return torch.sigmoid(self.silicon_carbide_emissivity_logit)

    @property
    def copper_emissivity(self) -> Tensor:
        return torch.sigmoid(self.copper_emissivity_logit)

    @property
    def contact_resistance_m2_k_w(self) -> Tensor:
        return torch.pow(
            torch.tensor(
                10.0,
                dtype=self.contact_resistance_log10.dtype,
                device=self.contact_resistance_log10.device,
            ),
            self.contact_resistance_log10,
        )

    def snapshot(self) -> dict[str, float]:
        return {
            "silicon_carbide_emissivity": float(
                self.silicon_carbide_emissivity.detach().cpu()
            ),
            "copper_emissivity": float(self.copper_emissivity.detach().cpu()),
            "contact_resistance_m2_k_w": float(
                self.contact_resistance_m2_k_w.detach().cpu()
            ),
            "contact_resistance_log10": float(
                self.contact_resistance_log10.detach().cpu()
            ),
        }

