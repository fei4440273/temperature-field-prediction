"""Project data and strictly past-only sensor histories for operator learning."""
from __future__ import annotations

import numpy as np
import polars as pl
import torch

from joint_temperature_core import Table, load_observations, load_simulation


class HistoryProvider:
    def __init__(self, curves, geometry, config, *, initial_temperature_k=295.15):
        self.initial_temperature_k = float(initial_temperature_k)
        if not np.isfinite(self.initial_temperature_k) or self.initial_temperature_k<=0:
            raise ValueError("History initial temperature must be finite and positive in kelvin.")
        self.curves = {float(p): (np.asarray(t, dtype=float), np.asarray(y, dtype=float))
                       for p, (t, y) in curves.items()}
        if not self.curves:
            raise ValueError("At least one sensor history is required.")
        for t, y in self.curves.values():
            if (t.ndim != 1 or y.shape != (len(t), 2) or len(t) < 2
                    or (np.diff(t) <= 0).any() or not np.isfinite(y).all()):
                raise ValueError("Sensor curves require finite temperatures and increasing timestamps.")
        self.powers = np.array(sorted(self.curves))
        self.geometry = geometry
        self.config = config.copy()
        self.long_len = int(config["long_seq_len"])
        self.local_len = int(config["local_seq_len"])
        self.window = float(config["local_window_s"])
        self.lag = float(config["sensor_lag_s"])
        self.scale = float(config["sensor_scale_k"])
        if min(self.long_len, self.local_len) < 2 or min(self.window, self.lag, self.scale) <= 0:
            raise ValueError("History sizes/scales and past-only lag must be positive.")
        self.cache = {}

    def _curve(self, power, times, cutoff):
        upper = int(np.searchsorted(self.powers, power))
        lo = self.powers[max(0, min(upper - 1, len(self.powers)-1))]
        hi = self.powers[min(upper, len(self.powers)-1)]
        # An exact measured power must not inherit a neighboring curve's mask.
        if upper < len(self.powers) and self.powers[upper] == power:
            lo = hi = self.powers[upper]
        alpha = 0. if lo == hi else np.clip((power-lo)/(hi-lo), 0., 1.)
        values = []
        valid = []
        for p in (lo, hi):
            t, y = self.curves[float(p)]
            recording_end = t[-1]
            # Restrict interpolation endpoints too: a future observation cannot
            # influence an apparently past timestamp through linear interpolation.
            eligible = t <= cutoff + 1e-7
            if not eligible.any():
                y0 = np.full((len(times), 2), self.initial_temperature_k)
                values.append(y0)
                valid.append(np.zeros(len(times)))
                continue
            t, y = t[eligible], y[eligible]
            sampled = np.column_stack([np.interp(times, t, y[:,i]) for i in (0,1)])
            # The endpoint trend avoids sawtooth heating rates from a stale hold.
            # Warm up past the ambient anchor and cap extension to one cadence.
            if len(t) >= 3:
                extend = (times > t[-1]) & (times <= recording_end+1e-7)
                cadence = t[-1]-t[-2]
                elapsed = np.minimum(times[extend]-t[-1], cadence)
                sampled[extend] = y[-1]+elapsed[:,None]*(y[-1]-y[-2])/cadence
            values.append(sampled)
            # Sampling gaps use a causal estimate; the mask describes the declared
            # recording window, not the arrival time of its latest sample.
            valid.append((times <= recording_end+1e-7).astype(float))
        return (1-alpha)*values[0]+alpha*values[1], np.minimum(valid[0],valid[1])

    def build(self, powers, times):
        powers = np.asarray(powers, dtype=float).reshape(-1)
        times = np.asarray(times, dtype=float).reshape(-1)
        if powers.shape != times.shape or not np.isfinite(powers).all() or not np.isfinite(times).all():
            raise ValueError("Power/time queries must be finite and have matching shapes.")
        longs, locals_ = [], []
        for power,time in zip(powers,times):
            key = (float(power), float(time))
            if key not in self.cache:
                end = max(0., time-self.lag)
                local_start = max(0., end-self.window)
                out = []
                for query in (np.linspace(0.,end,self.long_len),
                              np.linspace(local_start,end,self.local_len)):
                    y, valid = self._curve(power, query, end)
                    out.append(np.column_stack((query/self.geometry.time_max,
                        np.full(len(query),power/self.geometry.power_max),
                        (y-self.initial_temperature_k)/self.scale,valid)).astype(np.float32))
                if len(self.cache) < 30000:
                    self.cache[key] = tuple(out)
                long, local = out
            else:
                long, local = self.cache[key]
            longs.append(long)
            locals_.append(local)
        return np.stack(longs), np.stack(locals_)


def experiment_history(root, split, splits, geometry, config):
    initial = float(config.get("experiment_copper_initial_c",22.))+273.15
    tables = load_observations(root, split, splits, geometry)
    hot,cold = tables["热端"],tables["冷端"]
    curves = {}
    for power in np.unique(hot.x[:,3]):
        blocks = []
        for table in (hot,cold):
            mask = np.isclose(table.x[:,3],power,atol=1e-3,rtol=0)
            t,y = table.x[mask,2], table.y[mask,0]
            order = np.argsort(t)
            blocks.append((t[order], y[order]))
        # Observed sensor cadences can differ; use the common timestamps only.
        times = np.intersect1d(blocks[0][0],blocks[1][0])
        y = np.column_stack([np.interp(times,t,values) for t,values in blocks])
        if times[0] > 0:
            times = np.r_[0.,times]
            y = np.vstack((np.full(2,initial),y))
        curves[float(power)] = times,y
    return tables,HistoryProvider(curves,geometry,config,initial_temperature_k=initial)


def simulation_history(root, powers, geometry, config):
    curves, nodes = {}, []
    for power in powers:
        f = pl.read_parquet(root/"data/processed/simulation"/f"{power:g}W.parquet")
        copper = f.filter(pl.col("material_id")==0)
        unique = copper.select("node_label","r_m","z_m").unique()
        r,z = unique["r_m"].to_numpy(), unique["z_m"].to_numpy()
        values = []
        for radius in (.028,.0415):
            index = int(np.argmin((r-radius)**2+(z-geometry.bottom_cu)**2))
            node = int(unique["node_label"][index])
            curve = copper.filter(pl.col("node_label")==node).sort("time_s")
            times = curve["time_s"].to_numpy()
            values.append(curve["temperature_k"].to_numpy())
            nodes.append(dict(power_w=float(power),nominal_radius_m=radius,node_label=node,
                              actual_radius_m=float(r[index]),actual_height_m=float(z[index])))
        curves[float(power)] = times,np.column_stack(values)
    return HistoryProvider(curves,geometry,config),nodes


class BatchTable:
    """A balanced point pool with one stored history per power/time pair."""
    def __init__(self, table, provider, device):
        self.table = table
        self.x = torch.as_tensor(table.x,device=device)
        self.y = torch.as_tensor(table.y,device=device)
        pairs, inverse = np.unique(table.x[:,[3,2]],axis=0,return_inverse=True)
        self.case_ids = torch.as_tensor(inverse,device=device)
        long, local = provider.build(pairs[:,0],pairs[:,1])
        self.long = torch.as_tensor(long,device=device)
        self.local = torch.as_tensor(local,device=device)
        self.groups = [np.flatnonzero(table.group==gid) for gid in np.unique(table.group)]
        self.device = device

    def sample(self, count_per_group, rng):
        ids = np.concatenate([rng.choice(group,count_per_group,replace=len(group)<count_per_group)
                              for group in self.groups])
        return torch.as_tensor(ids,device=self.device)

    def inputs(self, ids):
        cases,inverse = torch.unique(self.case_ids[ids],sorted=True,return_inverse=True)
        return self.x[ids],self.long[cases],self.local[cases],inverse


def metrics(y, prediction):
    y = np.asarray(y,dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction,dtype=np.float64).reshape(-1)
    error = prediction-y
    if not len(y) or y.shape != prediction.shape or not np.isfinite(error).all():
        raise ValueError("Cannot evaluate empty, mismatched or nonfinite predictions.")
    denominator = np.sum((y-y.mean())**2)
    rise_norm = np.linalg.norm(y-295.15)
    return dict(n=len(y),rmse_k=float(np.sqrt(np.mean(error**2))),
        mae_k=float(np.mean(np.abs(error))),max_abs_error_k=float(np.abs(error).max()),
        relative_l2_kelvin_pct=float(100*np.linalg.norm(error)/np.linalg.norm(y)),
        relative_l2_rise_pct=(float(100*np.linalg.norm(error)/rise_norm) if rise_norm>1e-8 else None),
        r2=(float(1-np.sum(error**2)/denominator) if denominator>1e-8 else None))
