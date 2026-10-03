"""Sequential branch operators adapted from S-DeepONet and ASL-PINN.

ASL learned blocks follow src/models/ssm_lstm.py of ASL-PINN commit 4477c971.
The thermal-trend input mode adapts rate estimation to this experiment.
The comparison owns its implementation and has no runtime reference-repo imports.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from joint_temperature_core import Geometry, PhysicalSettings, mlp

METHODS = ("fnn","gru","lstm","asl")
LABELS = {key: key.upper()+"-DeepONet" for key in METHODS}


def parallel_affine_scan(multipliers,increments):
    a,b = multipliers,increments
    offset = 1
    while offset < increments.shape[1]:
        b = torch.cat((b[:,:offset],b[:,offset:]+a[:,offset:]*b[:,:-offset]),1)
        a = torch.cat((a[:,:offset],a[:,offset:]*a[:,:-offset]),1)
        offset *= 2
    return b


def causal_heating_rate_history(history):
    t = history[...,0]
    mean_temperature = .5*(history[...,2]+history[...,3])
    dt = t[:,1:]-t[:,:-1]
    rate = torch.where(dt>torch.finfo(t.dtype).eps,
        (mean_temperature[:,1:]-mean_temperature[:,:-1])/dt.clamp_min(torch.finfo(t.dtype).eps),
        torch.zeros_like(dt))
    rate = rate*history[:,1:,4]*history[:,:-1,4]
    rate = torch.cat((torch.zeros_like(rate[:,:1]),rate),1).tanh()
    return torch.cat((history[...,:4],rate[...,None]),-1)


def thermal_heating_rate_history(history,time_max,sensor_scale,window_s,rate_scale):
    """Causal weighted linear slopes in K/s, using valid past tokens only."""
    t = history[...,0]*time_max
    temperature = .5*(history[...,2]+history[...,3])*sensor_scale
    age = t[:,:,None]-t[:,None,:]
    causal = torch.ones(t.shape[1],t.shape[1],device=t.device,dtype=torch.bool).tril()
    weights = (1-age/window_s).clamp(0.,1.)*causal
    weights = weights*history[:,None,:,4]*history[:,:,None,4]
    epsilon = torch.finfo(t.dtype).eps
    total = weights.sum(-1).clamp_min(epsilon)
    centered_time = -age-(weights*(-age)).sum(-1,keepdim=True)/total[...,None]
    centered_temperature = temperature[:,None,:]-(weights*temperature[:,None,:]).sum(-1,keepdim=True)/total[...,None]
    variance = (weights*centered_time.square()).sum(-1)
    slope = (weights*centered_time*centered_temperature).sum(-1)/variance.clamp_min(epsilon)
    rate = torch.where(variance>epsilon,(slope/rate_scale).tanh(),torch.zeros_like(slope))
    return torch.cat((history[...,:4],rate[...,None]),-1)


class SelectiveSSMBlock(nn.Module):
    def __init__(self,dim,d_state):
        super().__init__()
        self.d_state = d_state
        self.dt_rank = max(4,dim//16)
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim,2*dim,bias=False)
        self.conv1d = nn.Conv1d(dim,dim,3,groups=dim)
        self.x_proj = nn.Linear(dim,self.dt_rank+2*d_state,bias=False)
        self.dt_proj = nn.Linear(self.dt_rank,dim)
        self.out_proj = nn.Linear(dim,dim,bias=False)
        self.A_log = nn.Parameter(torch.arange(1,d_state+1,dtype=torch.float32).log().repeat(dim,1))
        self.D = nn.Parameter(torch.ones(dim))
        nn.init.constant_(self.dt_proj.bias,-3.)

    def forward(self,x):
        u,z = self.in_proj(self.norm(x)).chunk(2,-1)
        u = F.silu(self.conv1d(F.pad(u.transpose(1,2),(2,0))).transpose(1,2))
        dt_raw,B,C = torch.split(self.x_proj(u),(self.dt_rank,self.d_state,self.d_state),-1)
        dt = F.softplus(self.dt_proj(dt_raw))
        A = -self.A_log.exp()
        state = parallel_affine_scan(torch.exp(dt[...,None]*A[None,None]),
            dt[...,None]*B[:,:,None,:]*u[...,None])
        y = (state*C[:,:,None,:]).sum(-1)+self.D*u
        return x+self.out_proj(y*F.silu(z))


class ASLBranch(nn.Module):
    """SSM -> LSTM with dual-state initialization and maturity-gated context."""
    def __init__(self,hidden,rank,state_dim,tau,time_max,*,rate_mode="original",
                 sensor_scale=80.,rate_window_s=18.,rate_scale=.5):
        super().__init__()
        if rate_mode not in ("original","thermal_trend"):
            raise ValueError("ASL rate mode must be original or thermal_trend.")
        if any(not math.isfinite(v) or v<=0 for v in (time_max,sensor_scale,rate_window_s,rate_scale)):
            raise ValueError("ASL thermal rate scales must be finite and positive.")
        self.long_input_proj = nn.Linear(5,hidden)
        self.local_input_proj = nn.Linear(5,hidden)
        self.ssm = nn.Sequential(SelectiveSSMBlock(hidden,state_dim),SelectiveSSMBlock(hidden,state_dim))
        self.lstm = nn.LSTM(hidden,hidden,batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,rank),nn.SiLU())
        self.tau,self.time_max = tau,time_max
        self.rate_mode,self.sensor_scale = rate_mode,sensor_scale
        self.rate_window_s,self.rate_scale = rate_window_s,rate_scale

    def forward(self,long,local,query_time):
        rates = (thermal_heating_rate_history(long,self.time_max,self.sensor_scale,
                 self.rate_window_s,self.rate_scale) if self.rate_mode=="thermal_trend"
                 else causal_heating_rate_history(long))
        context = torch.tanh(self.ssm(self.long_input_proj(rates))[:,-1])
        progress = ((query_time*self.time_max-self.tau)/(3*self.tau)).clamp(0.,1.)
        maturity = -torch.expm1(-3*progress)/(-math.expm1(-3))
        features = self.local_input_proj(local)+maturity[:,None,None]*context[:,None,:]
        state = context[None]
        with torch.backends.cudnn.flags(enabled=False):
            hidden,_ = self.lstm(features,(state,state.clone()))
        return self.out(hidden[:,-1])


class RecurrentBranch(nn.Module):
    """A long-history recurrent encoder conditions a local recurrent decoder."""
    def __init__(self,kind,hidden,rank):
        super().__init__()
        recurrent = nn.GRU if kind=="gru" else nn.LSTM
        self.long_input_proj = nn.Linear(5,hidden)
        self.local_input_proj = nn.Linear(5,hidden)
        self.encoder = recurrent(hidden,hidden,batch_first=True)
        self.decoder = recurrent(hidden,hidden,batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,rank),nn.SiLU())

    def forward(self,long,local,query_time):
        with torch.backends.cudnn.flags(enabled=False):
            _,state = self.encoder(self.long_input_proj(long))
            output,_ = self.decoder(self.local_input_proj(local),state)
        return self.out(output[:,-1])


class FNNBranch(nn.Module):
    def __init__(self,length,hidden,rank):
        super().__init__()
        self.network = mlp(5*length,rank,hidden*2,3)

    def forward(self,long,local,query_time):
        return self.network(torch.cat((long.flatten(1),local.flatten(1)),1))


class SequentialDeepONet(nn.Module):
    def __init__(self,method,geometry,settings,config):
        super().__init__()
        if method not in METHODS or config.get("boundary_attention",False):
            raise ValueError("Choose one of four encoders without boundary attention.")
        self.method,self.geometry,self.settings = method,geometry,settings
        self.model_config = copy.deepcopy(config)
        hidden,rank = int(config["hidden_dim"]),int(config["latent_dim"])
        self.rank = rank
        if method=="asl":
            self.branch = ASLBranch(hidden,rank,int(config["ssm_state_dim"]),
                float(config.get("maturity_tau_s",18.)),geometry.time_max,
                rate_mode=config.get("asl_rate_mode","original"),
                sensor_scale=float(config.get("sensor_scale_k",80.)),
                rate_window_s=float(config.get("asl_rate_window_s",18.)),
                rate_scale=float(config.get("asl_rate_scale_k_per_s",.5)))
        elif method=="fnn":
            self.branch = FNNBranch(int(config["long_seq_len"])+int(config["local_seq_len"]),hidden,rank)
        else:
            self.branch = RecurrentBranch(method,hidden,rank)
        # Branch sizes consume different RNG amounts; common components use a
        # separate seed so architecture is the controlled comparison variable.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed()+3000)
            self.trunk = mlp(4,rank,int(config["trunk_width"]),int(config["trunk_depth"]))
            self.low_projection = nn.Linear(rank,rank)
            self.high_projection = nn.Linear(rank,rank)
        self.low_bias = nn.Parameter(torch.zeros(1))
        self.high_bias = nn.Parameter(torch.zeros(1))
        self.log_contact = nn.Parameter(torch.tensor(math.log(settings.contact_initial)))
        self.register_buffer("temperature_scale",torch.tensor(float(config["temperature_scale_k"])))
        self.providers = {}

    @property
    def contact_resistance(self):
        return self.log_contact.exp()

    def set_history_providers(self,*,low,high):
        self.providers = {"low":low,"high":high}

    def forward_explicit(self,x,long,local,inverse,fidelity="high"):
        if fidelity not in ("low","high"):
            raise ValueError("Unknown fidelity.")
        g = self.geometry
        # Squared radius makes axis symmetry differentiable at r=0.
        z = torch.stack(((x[:,0]/g.radius_cu).square(),
            (x[:,1]-g.bottom_cu)/(-g.bottom_cu),x[:,2]/g.time_max,x[:,4]),1)
        trunk = self.trunk(z)
        if x.requires_grad:
            # Coordinate derivatives must remain pointwise even when points
            # share history. ASL's explicit maturity gate still depends on t.
            long,local = long[inverse],local[inverse]
            inverse = torch.arange(len(x),device=x.device)
            query_t = x[:,2]/g.time_max
        else:
            query_t = torch.zeros(len(long),device=x.device,dtype=x.dtype)
            query_t = query_t.scatter(0,inverse,x[:,2]/g.time_max)
        branch = self.branch(long,local,query_t)[inverse]
        normalized = (self.low_projection(branch)*trunk).sum(-1,keepdim=True)/math.sqrt(self.rank)+self.low_bias
        initial = self.settings.initial_simulation_k
        if fidelity=="high":
            normalized = normalized+(self.high_projection(branch)*trunk).sum(-1,keepdim=True)/math.sqrt(self.rank)+self.high_bias
            initial = self.settings.initial_experiment_k
        anchor = 1.-torch.exp(-x[:,2:3]/float(self.model_config.get("initial_tau_s",5.)))
        return initial+self.temperature_scale*anchor*normalized

    def forward(self,x,fidelity="high"):
        if fidelity not in self.providers:
            raise ValueError("Attach the correct split's history provider before predicting.")
        pairs,inverse = torch.unique(x[:,[3,2]].detach(),dim=0,return_inverse=True)
        long,local = self.providers[fidelity].build(pairs[:,0].cpu().numpy(),pairs[:,1].cpu().numpy())
        return self.forward_explicit(x,torch.as_tensor(long,device=x.device,dtype=x.dtype),
                                    torch.as_tensor(local,device=x.device,dtype=x.dtype),inverse,fidelity)


def save_checkpoint(path,model,*,epoch,config,history,seed,optimizer=None,scheduler=None,
                    numpy_generator=None,extra=None):
    state = dict(schema="sequential_deeponet_v1",method=model.method,epoch=int(epoch),seed=int(seed),
        config=copy.deepcopy(config),model_config=model.model_config,
        geometry=asdict(model.geometry),physical_settings=asdict(model.settings),
        model_state=model.state_dict(),history=copy.deepcopy(history),
        optimizer=optimizer.state_dict() if optimizer else None,
        scheduler=scheduler.state_dict() if scheduler else None,
        numpy_generator=copy.deepcopy(numpy_generator.bit_generator.state) if numpy_generator else None,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        extra=copy.deepcopy(extra or {}))
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(state,temporary)
    temporary.replace(path)
    return state


def load_checkpoint(path,device="cpu"):
    state = torch.load(path,map_location=device,weights_only=False)
    if state.get("schema")!="sequential_deeponet_v1":
        raise ValueError("Not a sequential operator comparison checkpoint.")
    model = SequentialDeepONet(state["method"],Geometry(**state["geometry"]),
                              PhysicalSettings(**state["physical_settings"]),state["model_config"]).to(device)
    model.load_state_dict(state["model_state"])
    return model,state
