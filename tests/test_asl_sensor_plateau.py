"""Causal sensor assimilation and supervision for measured late heating."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from joint_temperature_core import Geometry,Table
from sequential_deeponet_data import HistoryProvider
import sequential_temperature_objective as objective


def configuration():
    return dict(long_seq_len=8,local_seq_len=4,local_window_s=20.,sensor_lag_s=1.,
                sensor_scale_k=80.,local_window_mode="smooth",history_arrival_mode="smooth")


def test_new_measurement_is_assimilated_continuously_and_stays_past_only():
    t = np.arange(31.)
    y = np.column_stack((298.15+.1*t,298.15+.05*t))
    y[t>=15.] += 2.
    p = HistoryProvider({100.:(t,y)},Geometry(),configuration(),initial_temperature_k=298.15)
    before,at,after = [p.build([100.],[q])[0] for q in (15.9999,16.,16.0001)]
    assert np.max(abs(at-before))<1e-5
    assert np.max(abs(after-at))<1e-5
    later = p.build([100.],[17.])[0]
    assert later[0,-1,2]>at[0,-1,2]+.01
    future = y.copy()
    future[t>15.] += 1000.
    other = HistoryProvider({100.:(t,future)},Geometry(),configuration(),initial_temperature_k=298.15)
    for original,changed in zip(p.build([100.],[16.5]),other.build([100.],[16.5])):
        np.testing.assert_array_equal(original,changed)


@pytest.mark.parametrize("t",[
    np.array([0.,1.,2.,2.5,3.5,4.5]),
    np.array([0.,1.,2.,6.,7.,8.]),
])
def test_overlapping_or_delayed_arrivals_remain_continuous(t):
    y = np.column_stack((298.15+.3*t,298.15+.1*t))
    y[3:] += 1.
    p = HistoryProvider({100.:(t,y)},Geometry(),configuration(),initial_temperature_k=298.15)
    for arrival in t[2:-1]:
        points = np.array([.5,1.,arrival])
        before = p._curve(100.,points,arrival-1e-6)[0]
        at = p._curve(100.,points,arrival)[0]
        after = p._curve(100.,points,arrival+1e-6)[0]
        assert np.max(abs(at-before))<1e-5
        assert np.max(abs(after-at))<1e-5
    future = y.copy()
    future[t>t[3]] += 1000.
    other = HistoryProvider({100.:(t,future)},Geometry(),configuration(),initial_temperature_k=298.15)
    query = t[3]+1.+.1*(t[4]-t[3])
    for original,changed in zip(p.build([100.],[query]),other.build([100.],[query])):
        np.testing.assert_array_equal(original,changed)


def sensor_tables():
    tables = {}
    for name,radius,rate in (("热端",.028,.01),("冷端",.0415,.005)):
        rows,values,groups = [],[],[]
        for group,(power,end) in enumerate(((100.,40.),(200.,60.))):
            for time in np.arange(1.,end+1):
                rows.append([radius,-.01,time,power,0.])
                values.append(298.15+rate*time)
                groups.append(group)
        tables[name] = Table(np.array(rows),np.array(values),np.ones(len(rows)),np.array(groups)).validate()
    return tables


class SensorResponse(torch.nn.Module):
    def __init__(self,extra_slope=0.):
        super().__init__()
        self.extra = torch.nn.Parameter(torch.tensor(float(extra_slope)))
        self.temperature_scale = 250.

    def forward_explicit(self,x,long,local,inverse,fidelity):
        rate = torch.where(x[:,0]<.03,.01,.005)
        return (298.15+(rate+self.extra)*x[:,2])[:,None]


def test_sensor_tail_preserves_real_slow_heating_and_penalizes_excess_rate():
    api = getattr(objective,"SensorCurveConstraints",None)
    assert api is not None,"Measured sensor plateau constraints are missing."
    tables = sensor_tables()
    t = np.arange(61.)
    curves = {p:(t,np.column_stack((298.15+.01*t,298.15+.005*t))) for p in (100.,200.)}
    provider = HistoryProvider(curves,Geometry(),configuration(),initial_temperature_k=298.15)
    constraints = api(tables,provider,"cpu",20.,shape_points=8,shape_delta_s=2.)
    real = constraints.losses(SensorResponse())
    assert real["sensor_tail_slope"]<1e-10
    assert real["sensor_shape"]<1e-10
    model = SensorResponse(extra_slope=.05)
    losses = constraints.losses(model)
    assert losses["sensor_tail_slope"].item()==pytest.approx((.05*20/250.)**2,rel=.001)
    assert losses["sensor_tail_endpoint"]>0
    sum(losses.values()).backward()
    assert model.extra.grad.item()>0


def test_training_budget_counts_warm_start_ancestry(tmp_path):
    import train_sequential_deeponet as training
    api = getattr(training,"training_updates_in_checkpoint",None)
    assert api is not None,"Repeated ASL adaptation must preserve its training ancestry."
    torch.save(dict(epoch=1000,config={"training":{}},extra={}),tmp_path/"original.pt")
    previous = dict(epoch=750,config={"training":{"initial_checkpoint":"original.pt"}},extra={})
    assert api(previous,root=tmp_path)==1750
    assert api(dict(epoch=100,config={},extra={"initial_training_updates":1750}),root=tmp_path)==1850


def test_execution_budget_includes_full_ancestor_runs(tmp_path,monkeypatch):
    import evaluate_asl_iteration as evaluation
    monkeypatch.setattr(evaluation,"ROOT",tmp_path)
    for name,initial in (("original",None),("previous","original/asl/best.pt")):
        directory = tmp_path/name
        (directory/"asl").mkdir(parents=True)
        (directory/"asl/final_info.json").write_text(json.dumps({"epochs_completed":1000}))
        training = {} if initial is None else {"initial_checkpoint":initial}
        (directory/"config.json").write_text(json.dumps({"training":training}))
    config = {"training":{"initial_checkpoint":"previous/asl/best.pt"}}
    assert evaluation.executed_updates_in_lineage(config,1000)==3000
