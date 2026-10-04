"""ASL adaptation must remove artificial corners without changing legacy models."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from joint_temperature_core import Geometry,Table
from sequential_deeponet_core import ASLBranch
from sequential_deeponet_data import HistoryProvider


def test_smooth_gate_has_no_switch_corner_and_supports_physics_derivatives():
    branch = ASLBranch(8,4,3,18.,200.,maturity_mode="smooth")
    seconds = torch.tensor([0.,17.999,18.001,71.999,72.001,200.],dtype=torch.float64,
                           requires_grad=True)
    values = branch.maturity_scale(seconds/200.)
    slope = torch.autograd.grad(values.sum(),seconds,create_graph=True)[0]
    curvature = torch.autograd.grad(slope.sum(),seconds)[0]
    assert values[0]==0 and values[-1]<=1
    assert torch.all(values[1:]>values[:-1])
    assert torch.isfinite(curvature).all()
    assert abs(slope[1]-slope[2])<1e-5
    assert abs(slope[3]-slope[4])<1e-5
    assert abs(curvature[1]-curvature[2])<1e-5


def test_smooth_gate_keeps_legacy_weights_and_default_predictions():
    torch.manual_seed(123)
    legacy = ASLBranch(8,4,3,18.,200.)
    smooth = ASLBranch(8,4,3,18.,200.,maturity_mode="smooth")
    smooth.load_state_dict(legacy.state_dict(),strict=True)
    assert sum(p.numel() for p in legacy.parameters())==sum(p.numel() for p in smooth.parameters())
    q = torch.tensor([.02,.09,.12,.4])
    expected = -torch.expm1(-3*((q*200.-18.)/54.).clamp(0.,1.))/(-np.expm1(-3))
    torch.testing.assert_close(legacy.maturity_scale(q),expected,rtol=0,atol=0)


def test_soft_transition_removes_corner_without_moving_original_time_scale():
    branch = ASLBranch(8,4,3,18.,200.,maturity_mode="soft_transition",maturity_smoothing_s=3.)
    q = torch.tensor([0.,17.999,18.001,36.,71.999,72.001,200.],dtype=torch.float64,requires_grad=True)
    values = branch.maturity_scale(q/200.)
    slope = torch.autograd.grad(values.sum(),q,create_graph=True)[0]
    assert values[0]==0 and values[-1]<=1
    assert abs(slope[1]-slope[2])<1e-5
    assert abs(slope[4]-slope[5])<1e-5
    legacy = ASLBranch(8,4,3,18.,200.).maturity_scale(q/200.)
    assert torch.max(abs(values-legacy))<.12


def test_replacement_retains_every_baseline_array_and_rejects_wrong_length(tmp_path):
    from evaluate_asl_iteration import replace_asl
    path = tmp_path/"predictions.npz"
    original = dict(x=np.zeros((4,5),np.float32),y=np.arange(4,dtype=np.float32),
                    fnn=np.arange(4)+1.,gru=np.arange(4)+2.,lstm=np.arange(4)+3.,asl=np.arange(4)+4.)
    np.savez_compressed(path,**original)
    with np.load(path) as pack:
        replacement = replace_asl(pack,np.full(4,25.))
        for key in ("x","y","fnn","gru","lstm"):
            np.testing.assert_array_equal(replacement[key],original[key])
        with pytest.raises(ValueError,match="shape"):
            replace_asl(pack,np.zeros(3))


def history_config():
    return dict(long_seq_len=8,local_seq_len=4,local_window_s=20.,sensor_lag_s=1.,sensor_scale_k=80.)


def test_smooth_window_has_no_21_second_start_corner_and_stays_causal():
    cfg = dict(history_config(),local_window_mode="smooth")
    t = np.arange(101.)
    y = np.column_stack((298.15+.2*t,298.15+.1*t))
    provider = HistoryProvider({100.:(t,y)},Geometry(),cfg,initial_temperature_k=298.15)
    queries = np.array([20.98,20.99,21.,21.01,21.02])
    _,local = provider.build(np.full(5,100.),queries)
    slopes = np.diff(local[:,0,0].astype(float)*200.)/.01
    assert abs(slopes[2]-slopes[1])<.002
    assert np.all(local[:,:,0]*200.<=queries[:,None]-1.+1e-5)
    future = y.copy()
    future[t>20.] += 1000.
    altered = HistoryProvider({100.:(t,future)},Geometry(),cfg,initial_temperature_k=298.15)
    for original,changed in zip(provider.build([100.],[21.]),altered.build([100.],[21.])):
        np.testing.assert_array_equal(original,changed)
    legacy = HistoryProvider({100.:(t,y)},Geometry(),history_config())
    _,old = legacy.build([100.,100.],[20.,22.])
    np.testing.assert_allclose(old[:,0,0]*200.,[0.,1.])


def test_balanced_selection_rejects_sensor_regression_despite_better_top(monkeypatch):
    import train_sequential_deeponet as training
    x = np.array([[.001,0.,10.,100.,1.],[.001,0.,20.,100.,1.]])
    tables = {name:Table(x,np.full(2,300.),np.ones(2),np.zeros(2)).validate()
              for name in ("顶部","热端","冷端")}
    errors = {"顶部":6.,"热端":.6,"冷端":.6}
    def prediction(model,table,provider,fidelity):
        name = next(k for k,v in tables.items() if v is table)
        return table.y[:,0]+errors[name]
    monkeypatch.setattr(training,"table_predict",prediction)
    selection = dict(criterion="max_reference_rmse_ratio",
                     reference_rmse_k={"顶部":8.,"热端":1.,"冷端":1.})
    balanced,_ = training.validate(None,tables,None,selection)
    errors.update({"顶部":4.,"热端":1.2})
    regression,_ = training.validate(None,tables,None,selection)
    assert balanced==pytest.approx(.75)
    assert regression==pytest.approx(1.2,abs=2e-5)
    assert regression>balanced
