"""Causality, differentiation and inference contracts for the new comparison."""
from pathlib import Path
from dataclasses import replace
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from joint_temperature_core import Geometry, PhysicalSettings, physics_loss


def module(name):
    assert importlib.util.find_spec(name) is not None, f"Missing implementation: {name}"
    return importlib.import_module(name)


def physical():
    return PhysicalSettings(295.15, 298.15, 295.15, 295.15, 295.15,
                            .8, .02, 9., 4.3, .5, .5, True,
                            8900., 400., 401., 3170., 700., 120., 7.34e-5)


def curves(future=0.):
    t = np.arange(11, dtype=float)
    y = np.stack((295.15 + t, 295.15 + .5*t), axis=-1)
    y[5:] += future
    return {100.: (t, y), 200.: (t, y+1.)}


def config():
    return dict(hidden_dim=8, latent_dim=8, trunk_width=12, trunk_depth=2,
                long_seq_len=8, local_seq_len=4, local_window_s=3.,
                sensor_lag_s=1., sensor_scale_k=80., ssm_state_dim=4,
                temperature_scale_k=250., boundary_attention=False)


def test_history_excludes_present_future_and_preserves_channel_units():
    data = module("sequential_deeponet_data")
    a = data.HistoryProvider(curves(), Geometry(), config())
    b = data.HistoryProvider(curves(999.), Geometry(), config())
    long, local = a.build(np.array([100.]), np.array([5.]))
    other_long, other_local = b.build(np.array([100.]), np.array([5.]))
    np.testing.assert_array_equal(long, other_long)
    np.testing.assert_array_equal(local, other_local)
    assert long.shape == (1, 8, 5) and local.shape == (1, 4, 5)
    assert long[0, -1, 0] == pytest.approx(4./200.)
    assert long[0, -1, 2] == pytest.approx(4./80.)
    assert long[0, -1, 1] == pytest.approx(100./800.)


def test_interpolation_cannot_use_current_simulation_node_temperature():
    data = module("sequential_deeponet_data")
    times = np.array([0.,2.,4.])
    a = {100.:(times,np.array([[295.15,295.15],[300.,299.],[305.,301.]]))}
    b = {100.:(times,np.array([[295.15,295.15],[900.,900.],[999.,999.]]))}
    pa = data.HistoryProvider(a,Geometry(),config())
    pb = data.HistoryProvider(b,Geometry(),config())
    for left,right in zip(pa.build([100.],[2.]),pb.build([100.],[2.])):
        np.testing.assert_array_equal(left,right)


def test_metrics_keep_kelvin_and_temperature_rise_denominators_distinct():
    data = module("sequential_deeponet_data")
    result = data.metrics(np.array([296.15,297.15]),np.array([298.15,295.15]))
    assert result["rmse_k"] == pytest.approx(2.)
    assert result["mae_k"] == pytest.approx(2.)
    assert result["relative_l2_rise_pct"] == pytest.approx(100*np.sqrt(8/5))
    assert result["relative_l2_kelvin_pct"] < 1.


@pytest.mark.parametrize("method", ["fnn", "gru", "lstm", "asl"])
def test_operator_supports_physics_and_has_no_boundary_attention(method):
    core = module("sequential_deeponet_core")
    data = module("sequential_deeponet_data")
    provider = data.HistoryProvider(curves(), Geometry(), config())
    model = core.SequentialDeepONet(method, Geometry(), physical(), config())
    model.set_history_providers(low=provider, high=provider)
    assert not any("attention" in name for name, _ in model.named_modules())
    x = torch.tensor([[.01, -.003, 5., 100., 1.], [.035, -.015, 8., 200., 0.]])
    assert model(x).shape == (2, 1)
    parts = physics_loss(model, 4, torch.Generator().manual_seed(7))
    assert all(torch.isfinite(value) for value in parts.values())
    sum(parts.values()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.branch.parameters())


@pytest.mark.parametrize("method", ["fnn", "gru", "lstm", "asl"])
def test_checkpoint_preserves_prediction_and_1000_epoch_identity(method, tmp_path):
    core = module("sequential_deeponet_core")
    data = module("sequential_deeponet_data")
    provider = data.HistoryProvider(curves(), Geometry(), config())
    model = core.SequentialDeepONet(method, Geometry(), physical(), config()).eval()
    model.set_history_providers(low=provider, high=provider)
    x = torch.tensor([[.01, -.003, 5., 100., 1.]])
    path = tmp_path / "final.pt"
    core.save_checkpoint(path, model, epoch=1000, config={"model":config()},
                         history=[{"epoch":1000}], seed=123)
    restored, state = core.load_checkpoint(path)
    restored.set_history_providers(low=provider, high=provider)
    torch.testing.assert_close(restored(x), model(x))
    assert state["epoch"] == 1000 and state["method"] == method


def test_affine_parallel_scan_matches_recurrence_and_gradients():
    core = module("sequential_deeponet_core")
    a = torch.rand(2, 9, 3, 4, dtype=torch.float64, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    state = torch.zeros_like(b[:,0])
    expected = []
    for i in range(9):
        state = a[:,i]*state+b[:,i]
        expected.append(state)
    expected = torch.stack(expected, 1)
    actual = core.parallel_affine_scan(a,b)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.sum(), (a,b), retain_graph=True)
    gb = torch.autograd.grad(expected.sum(), (a,b))
    for left,right in zip(ga,gb):
        torch.testing.assert_close(left,right)


def test_common_trunk_and_heads_use_identical_initial_weights():
    core = module("sequential_deeponet_core")
    states = []
    for method in core.METHODS:
        torch.manual_seed(123)
        model = core.SequentialDeepONet(method,Geometry(),physical(),config())
        states.append({key:value for key,value in model.state_dict().items()
                       if not key.startswith("branch.")})
    for state in states[1:]:
        for key in states[0]:
            torch.testing.assert_close(states[0][key],state[key])


def test_asl_encoder_matches_authenticated_source_with_same_weights(monkeypatch):
    source_path = ROOT/"references/ASL-PINN-code/src/models/ssm_lstm.py"
    if not source_path.exists():
        pytest.skip("Authenticated reference snapshot is not present.")
    core = module("sequential_deeponet_core")
    reference_config = SimpleNamespace(SSM_ACTIVATION_CHECKPOINTING=False,
        SENSOR_HISTORY_FEATURE_DIM=5,SSM_STATE_DIM=4,SSM_EXPAND=1,SSM_CONV_KERNEL=3,
        SSM_PARALLEL_SCAN=True,TIME_FEATURE_DIM=8,TIME_LONG_SEQ_LEN=8,TIME_LOCAL_SEQ_LEN=4,
        TOP_RESPONSE_TAU=18.,MAX_TIME=200.)
    monkeypatch.setitem(sys.modules,"config",reference_config)
    spec = importlib.util.spec_from_file_location("asl_reference",source_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    source = reference.CascadedSelectiveSSMLSTMTimeEncoder(out_dim=8,hidden_dim=8)
    adapted = core.ASLBranch(8,8,4,18.,200.)
    adapted.load_state_dict(source.state_dict())
    long = torch.randn(3,8,5)
    long[:,:,0] = torch.linspace(0.,.5,8)
    long[:,:,4] = 1.
    local = long[:,-4:]
    t = torch.tensor([.1,.3,.8])
    torch.testing.assert_close(adapted(long,local,t),source(t,t,sensor_long=long,sensor_local=local),
                               atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize("method",["fnn","gru","lstm","asl"])
def test_time_derivative_holds_history_fixed_but_differentiates_query_gate(method):
    core = module("sequential_deeponet_core")
    torch.manual_seed(123)
    model = core.SequentialDeepONet(method,Geometry(),physical(),config()).double()
    long = torch.randn(2,8,5,dtype=torch.float64)
    long[:,:,0] = torch.linspace(0.,.2,8)
    long[:,:,4] = 1.
    local = long[:,-4:]
    x = torch.tensor([[.01,-.003,40.,100.,1.],[.035,-.015,55.,200.,0.]],
                     dtype=torch.float64,requires_grad=True)
    inverse = torch.arange(2)
    value = model.forward_explicit(x,long,local,inverse)
    analytic = torch.autograd.grad(value.sum(),x)[0][:,2]
    displacement = torch.zeros_like(x)
    displacement[:,2] = 1e-4
    upper = model.forward_explicit(x.detach()+displacement,long,local,inverse)
    lower = model.forward_explicit(x.detach()-displacement,long,local,inverse)
    numerical = ((upper-lower)/(2e-4)).reshape(-1)
    torch.testing.assert_close(analytic,numerical,rtol=1e-5,atol=1e-6)


def test_shared_history_keeps_coordinate_jacobian_pointwise():
    core = module("sequential_deeponet_core")
    model = core.SequentialDeepONet("asl",Geometry(),physical(),config()).double()
    long = torch.randn(1,8,5,dtype=torch.float64)
    long[:,:,0] = torch.linspace(0.,.2,8)
    local = long[:,-4:]
    inverse = torch.zeros(2,dtype=torch.long)
    x = torch.tensor([[.01,-.003,40.,100.,1.],[.035,-.015,40.,100.,0.]],dtype=torch.float64)
    jacobian = torch.autograd.functional.jacobian(
        lambda query:model.forward_explicit(query,long,local,inverse),x)
    torch.testing.assert_close(jacobian[0,0,1],torch.zeros(5,dtype=x.dtype),rtol=0,atol=0)
    torch.testing.assert_close(jacobian[1,0,0],torch.zeros(5,dtype=x.dtype),rtol=0,atol=0)
