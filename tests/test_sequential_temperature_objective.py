"""Measured temperature-curve supervision and signed-error contracts."""
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from joint_temperature_core import Geometry,Table
from sequential_deeponet_data import HistoryProvider


def objective():
    assert importlib.util.find_spec("sequential_temperature_objective") is not None
    return importlib.import_module("sequential_temperature_objective")


def observations():
    rows,values,groups = [],[],[]
    for group,(power,end) in enumerate(((100.,30.),(200.,50.))):
        for time in np.arange(5.,end+1,5.):
            for radius in (.001,.01):
                rows.append([radius,0.,time,power,1.])
                values.append(295.15+2*time-radius*100)
                groups.append(group)
    return Table(np.array(rows),np.array(values),np.ones(len(rows)),np.array(groups)).validate()


def provider(table):
    curves = {}
    for power in np.unique(table.x[:,3]):
        end = table.x[table.x[:,3]==power,2].max()
        times = np.arange(end+1)
        curves[float(power)] = times,np.column_stack((295.15+times,295.15+.5*times))
    cfg = dict(long_seq_len=8,local_seq_len=4,local_window_s=20.,sensor_lag_s=1.,sensor_scale_k=80.)
    return HistoryProvider(curves,Geometry(),cfg)


class Response(torch.nn.Module):
    def __init__(self,offset=0.,opposing_slope=0.):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(float(offset)))
        self.slope = torch.nn.Parameter(torch.tensor(float(opposing_slope)))
        self.temperature_scale = 250.

    def forward_explicit(self,x,long,local,inverse,fidelity):
        sign = torch.where(x[:,0]<.005,1.,-1.)
        self.x = x
        return (295.15+2*x[:,2]-100*x[:,0]+self.offset+self.slope*sign*x[:,2])[:,None]


def test_center_tail_uses_each_powers_horizon_and_actual_radius():
    api = objective()
    table = observations()
    ids = api.center_tail_indices(table,.7)
    selected = table.x[ids]
    np.testing.assert_allclose(selected[:,0],.001)
    np.testing.assert_array_equal(selected[selected[:,3]==100.,2],[25.,30.])
    np.testing.assert_array_equal(selected[selected[:,3]==200.,2],[35.,40.,45.,50.])
    center = api.center_table(table,.005)
    assert np.all(center.x[:,0]<=.005)
    assert set(center.group)=={0,1}


def test_measured_tail_constraints_separate_constant_bias_and_slope():
    api = objective()
    table = observations()
    constraints = api.TopCurveConstraints(table,provider(table),"cpu",20.)
    model = Response(offset=2.)
    parts = constraints.losses(model)
    assert parts["top_tail"].item()==pytest.approx((2./250.)**2,rel=1e-5)
    assert parts["top_tail_endpoint"].item()==pytest.approx((2./250.)**2,rel=1e-5)
    assert parts["top_tail_slope"]<1e-12
    sum(parts.values()).backward()
    assert model.offset.grad.item()>0.
    assert torch.isfinite(model.slope.grad)
    assert set(model.x[:,2].tolist()).issubset(set(table.x[:,2].tolist()))


def test_opposite_radial_tail_slopes_do_not_cancel():
    api = objective()
    table = observations()
    constraints = api.TopCurveConstraints(table,provider(table),"cpu",20.)
    model = Response(opposing_slope=.5)
    loss = constraints.losses(model)["top_tail_slope"]
    assert loss.item()==pytest.approx((.5*20./250.)**2,rel=1e-5)
    loss.backward()
    assert model.slope.grad.item()>0.


def test_tail_constraints_require_three_distinct_measured_times():
    api = objective()
    table = observations()
    with pytest.raises(ValueError,match="three"):
        api.TopCurveConstraints(table,provider(table),"cpu",5.)


def test_validation_score_distinguishes_center_tail_and_preserves_legacy(monkeypatch):
    import train_sequential_deeponet as training

    top = observations()
    tables = {"顶部":top,"热端":observations(),"冷端":observations()}
    error = np.zeros(len(top.x))
    central = top.x[:,0]<.005
    error[central] = 4.
    error[(top.x[:,2]==5.)&~central] = 20.
    predictions = {id(top):top.y[:,0]+error,
                   id(tables["热端"]):tables["热端"].y[:,0]+1.,
                   id(tables["冷端"]):tables["冷端"].y[:,0]-3.}
    monkeypatch.setattr(training,"table_predict",lambda model,table,provider,fidelity:predictions[id(table)])
    score,measured = training.validate(None,tables,None,
        dict(tail_fraction=.7,weights={"顶部":.5,"中心后段":.25,"热端":.125,"冷端":.125}))
    assert measured["中心后段"]["n"]==6
    assert measured["中心后段"]["rmse_k"]==4.
    assert score==pytest.approx(np.sqrt(.5*np.mean(error**2)+.25*16+.125*1+.125*9),rel=1e-5)
    legacy,_ = training.validate(None,tables,None)
    assert legacy==pytest.approx(np.sqrt(.5*np.mean(error**2)+.25*1+.25*9),rel=1e-5)
