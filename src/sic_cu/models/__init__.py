"""Candidate temperature-field architectures."""

from .common import ModelScales
from .deeponet_pinn import DeepONetPINN
from .gno_pinn import GNOPINN
from .lstm_pinn import LSTMPINN
from .mlp_pinn import MLPPINN
from .multifidelity import AdditiveCorrectionModel
from .prc import PRCMultifidelityModel, PRC_VARIANTS
from .pod_pinn import (
    MaterialWisePODPINN,
    PODBasis,
    PODPINN,
    fit_pod,
    load_pod_basis,
    save_pod_basis,
)
from .surface_residual import SurfaceResidualMLP
from .residual_interpolation import ChebyshevSurfaceResidualGuide

__all__ = [
    "DeepONetPINN",
    "GNOPINN",
    "LSTMPINN",
    "MLPPINN",
    "AdditiveCorrectionModel",
    "PRCMultifidelityModel",
    "PRC_VARIANTS",
    "ModelScales",
    "PODBasis",
    "PODPINN",
    "MaterialWisePODPINN",
    "fit_pod",
    "load_pod_basis",
    "save_pod_basis",
    "SurfaceResidualMLP",
    "ChebyshevSurfaceResidualGuide",
]
