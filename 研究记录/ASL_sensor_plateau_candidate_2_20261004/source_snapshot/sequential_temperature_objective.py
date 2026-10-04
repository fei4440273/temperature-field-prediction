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


def sensor_tail_metrics(table,prediction,window_s=20.):
    prediction = np.asarray(prediction).reshape(-1)
    rows = []
    for power in np.unique(table.x[:,3]):
        ids = np.flatnonzero(table.x[:,3]==power)
        ids = ids[table.x[ids,2]>=table.x[ids,2].max()-window_s-1e-5]
        t = table.x[ids,2].astype(float)
        if len(np.unique(t))<3:
            raise ValueError("Sensor tails require three distinct measured times.")
        centered = t-t.mean()
        observed = table.y[ids,0].astype(float)
        predicted = prediction[ids].astype(float)
        actual_rate = float(np.dot(centered,observed)/np.dot(centered,centered))
        predicted_rate = float(np.dot(centered,predicted)/np.dot(centered,centered))
        rows.append(dict(power_w=float(power),observed_rate_k_per_s=actual_rate,
            predicted_rate_k_per_s=predicted_rate,rate_error_k_per_s=predicted_rate-actual_rate,
            tail_bias_k=float((predicted-observed).mean())))
    return dict(rate_rmse_k_per_s=float(np.sqrt(np.mean([r["rate_error_k_per_s"]**2 for r in rows]))),curves=rows)


class SensorCurveConstraints:
    """Measured tails and interior curvature, independently for each sensor/power."""
    def __init__(self,tables,provider,device,window_s,*,shape_points=12,shape_delta_s=2.):
        if set(tables)!={"热端","冷端"}:
            raise ValueError("Both measured sensor tables are required.")
        if int(shape_points)!=shape_points or shape_points<2 or not math.isfinite(shape_delta_s) or shape_delta_s<=0:
            raise ValueError("Sensor shape probes require at least two points and a positive spacing.")
        combined = Table(np.concatenate([t.x for t in tables.values()]),
            np.concatenate([t.y for t in tables.values()]),
            np.concatenate([t.weight for t in tables.values()]),
            np.concatenate([t.group for t in tables.values()])).validate()
        self.tail = TopCurveConstraints(combined,provider,device,window_s)
        rows,values = [],[]
        for table in tables.values():
            for power in np.unique(table.x[:,3]):
                ids = np.flatnonzero(table.x[:,3]==power)
                ids = ids[np.argsort(table.x[ids,2])]
                t,y = table.x[ids,2],table.y[ids,0]
                start,end = max(5.,float(t.min())+shape_delta_s),float(t.max())-shape_delta_s
                if end<=start:
                    raise ValueError("Sensor recording is too short for interior shape probes.")
                for center in np.linspace(start,end,int(shape_points)):
                    times = center+np.array([-shape_delta_s,0.,shape_delta_s])
                    query = np.repeat(table.x[ids[:1]],3,axis=0)
                    query[:,2] = times
                    rows.append(query)
                    values.append(np.interp(times,t,y))
        shape = Table(np.concatenate(rows),np.concatenate(values),np.ones(3*len(rows)),
                      np.repeat(np.arange(len(rows)),3)).validate()
        self.shape = BatchTable(shape,provider,device)
        self.ids = torch.arange(len(shape.x),device=device)
        self.shape_delta_s = shape_delta_s

    def losses(self,model):
        result = {name.replace("top_","sensor_"):value for name,value in self.tail.losses(model).items()}
        x,long,local,inverse = self.shape.inputs(self.ids)
        error = (model.forward_explicit(x,long,local,inverse,"high")-self.shape.y).reshape(-1,3)
        curvature = (error[:,0]-2*error[:,1]+error[:,2])*(10./self.shape_delta_s)**2/model.temperature_scale
        result["sensor_shape"] = curvature.square().mean()
        return result
