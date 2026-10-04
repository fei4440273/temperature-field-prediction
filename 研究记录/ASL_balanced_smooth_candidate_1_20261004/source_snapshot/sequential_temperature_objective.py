"""Supervision for measured spatial peaks and late temperature curves."""
from __future__ import annotations

import math
import numpy as np
import torch

from joint_temperature_core import Table,grouped_mse
from sequential_deeponet_data import BatchTable


def center_table(table,radius_m):
    if not math.isfinite(radius_m) or radius_m<=0:
        raise ValueError("Center radius must be finite and positive.")
    ids = np.flatnonzero(table.x[:,0]<=radius_m)
    if set(table.group[ids])!=set(table.group):
        raise ValueError("Every training power needs measured central observations.")
    return Table(table.x[ids],table.y[ids],table.weight[ids],table.group[ids]).validate()


def center_tail_indices(table,fraction=.7):
    if not math.isfinite(fraction) or not 0<fraction<1:
        raise ValueError("Tail fraction must be strictly between zero and one.")
    selected = []
    for power in np.unique(table.x[:,3]):
        ids = np.flatnonzero(table.x[:,3]==power)
        times = table.x[ids,2]
        for time in np.unique(times[times>=fraction*times.max()]):
            frame = ids[times==time]
            selected.append(frame[np.argmin(table.x[frame,0])])
    return np.asarray(selected,dtype=np.int64)


class TopCurveConstraints:
    """Cached real tail observations, with independent power/radius trends."""
    def __init__(self,table,provider,device,window_s):
        if not math.isfinite(window_s) or window_s<=0:
            raise ValueError("Measured tail window must be finite and positive.")
        _,groups = np.unique(table.x[:,[3,0]],axis=0,return_inverse=True)
        selected,endpoints = [],[]
        for group in np.unique(groups):
            ids = np.flatnonzero(groups==group)
            times = table.x[ids,2]
            tail = ids[times>=times.max()-window_s-1e-5]
            observed = table.x[tail,2]
            if len(np.unique(observed))<3:
                raise ValueError("Each radial tail curve needs three distinct measured times.")
            if observed.max()-observed.min()+1e-5<window_s:
                raise ValueError("Measured tail support does not cover the requested window.")
            selected.extend(tail)
        selected = np.asarray(selected,dtype=np.int64)
        tail = Table(table.x[selected],table.y[selected],np.ones(len(selected)),groups[selected]).validate()
        self.pool = BatchTable(tail,provider,device)
        self.ids = torch.arange(len(selected),device=device)
        self.groups = torch.as_tensor(tail.group,device=device)
        self.count = int(self.groups.max())+1
        times = self.pool.x[:,2]
        counts = torch.zeros(self.count,device=device).scatter_add_(0,self.groups,torch.ones_like(times))
        mean = torch.zeros_like(counts).scatter_add_(0,self.groups,times)/counts
        self.centered_time = times-mean[self.groups]
        self.variance = torch.zeros_like(counts).scatter_add_(0,self.groups,self.centered_time.square())
        for group in np.unique(tail.group):
            ids = np.flatnonzero(tail.group==group)
            endpoints.extend(ids[tail.x[ids,2]==tail.x[ids,2].max()])
        self.endpoints = torch.as_tensor(endpoints,device=device)
        self.window_s = window_s

    def losses(self,model):
        x,long,local,inverse = self.pool.inputs(self.ids)
        prediction = model.forward_explicit(x,long,local,inverse,"high")
        error = (prediction-self.pool.y)/model.temperature_scale
        weights = torch.ones_like(error)
        temperature = grouped_mse(error,weights,self.groups)
        numerator = torch.zeros(self.count,device=x.device).scatter_add_(
            0,self.groups,self.centered_time*error[:,0])
        change = numerator/self.variance*self.window_s
        endpoint = grouped_mse(error[self.endpoints],weights[self.endpoints],self.groups[self.endpoints])
        return dict(top_tail=temperature,top_tail_slope=change.square().mean(),top_tail_endpoint=endpoint)
