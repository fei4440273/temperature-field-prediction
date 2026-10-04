"""多保真 DeepONet 的连续联合训练核心。所有温度计算使用 K，输出展示使用 ℃。"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn, Tensor

KELVIN = 273.15
TOTAL_EPOCHS = 1000
COPPER_MODULE_FLAGS = ('copper_amplitude', 'copper_time_scale', 'copper_power_modulation')
HIGH_PARAMETER_PREFIXES = ('correction.',) + tuple(name + '_net.' for name in COPPER_MODULE_FLAGS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding='utf-8'))


@dataclass(frozen=True)
class Geometry:
    radius_cu: float = 0.05834
    radius_sic: float = 0.025
    bottom_cu: float = -0.0175
    bottom_sic: float = -0.012
    time_max: float = 200.0
    power_max: float = 800.0

    @classmethod
    def from_project(cls, root: Path):
        d = read_yaml(root / 'configs/geometry.yaml')
        return cls(float(d['copper']['radius_m']),
                   float(d['silicon_carbide']['radius_m']),
                   float(d['embedding']['copper_bottom_z_m']),
                   float(d['embedding']['sic_bottom_z_m']))


@dataclass(frozen=True)
class PhysicalSettings:
    cooling_simulation_k: float
    cooling_experiment_k: float
    initial_simulation_k: float
    initial_experiment_k: float
    ambient_k: float
    absorption: float
    beam_radius: float
    h_top: float
    h_bottom: float
    emissivity_sic: float
    emissivity_cu: float
    radiation: bool
    density_cu: float
    capacity_cu: float
    conductivity_cu: float
    density_sic: float
    capacity_sic: float
    conductivity_sic: float
    contact_initial: float

    @classmethod
    def from_project(cls, root: Path, config: dict):
        b = read_yaml(root / 'configs/boundary_conditions.yaml')
        m = read_yaml(root / 'configs/materials.yaml')
        if b['cooling']['kind'] != 'fixed_temperature' or b['cooling']['surface'] != 'outer_radius':
            raise ValueError('本入口要求当前装置的铜外圆柱面定温水冷配置。')
        if b['laser']['profile'] != 'gaussian':
            raise ValueError('本入口沿用已确认的高斯激光热流，不静默替换热源类型。')
        if (config.get('model', {}).get('time_response', 'free') == 'monotone_heating'
                and b['laser'].get('constant_during_heating') is not True):
            raise ValueError('持续升温模式仅适用于恒定加热，不能用于关断热源后的冷却过程。')
        ext = b['external_surface']
        nominal = ext.get('nominal_modeling_values', {})
        def eps(name):
            value = ext.get(name + '_emissivity')
            if value is None:
                value = nominal.get(name)
            if value is None or not 0 <= float(value) <= 1:
                raise ValueError('缺少原项目的名义发射率配置：' + name)
            return float(value)
        def prop(material, name):
            item = m[material][name]
            if item.get('kind') != 'constant':
                raise ValueError('本轮沿用现有常物性；温变物性不能静默按常数处理。')
            return float(item['value'])
        p = config['physical']
        initial = float(b['initial_temperature_k'])
        contact = b['interface'].get('contact_resistance_m2_k_w')
        if contact is None:
            contact = b['interface']['contact_resistance_initialization_m2_k_w']
        return cls(
            float(p['simulation_cooling_c']) + KELVIN,
            float(p['experiment_cooling_c']) + KELVIN,
            initial,
            initial if p.get('experiment_initial_c') is None else float(p['experiment_initial_c']) + KELVIN,
            float(b['ambient_temperature_k']), float(b['laser']['absorption_fraction']),
            float(b['laser']['beam_radius_m']), float(ext['top_convection_coefficient_w_m2_k']),
            float(ext['bottom_convection_coefficient_w_m2_k']), eps('silicon_carbide'), eps('copper'),
            bool(ext['radiation_enabled']),
            float(m['copper']['density_kg_m3']), prop('copper','heat_capacity_j_kg_k'),
            prop('copper','conductivity_w_m_k'), float(m['silicon_carbide']['density_kg_m3']),
            prop('silicon_carbide','heat_capacity_j_kg_k'), prop('silicon_carbide','conductivity_w_m_k'),
            float(contact))


def mlp(ni: int, no: int, width: int, depth: int, zero_last: bool = False):
    layers: list[nn.Module] = [nn.Linear(ni, width), nn.Tanh()]
    for _ in range(depth-1):
        layers.extend([nn.Linear(width, width), nn.Tanh()])
    out = nn.Linear(width, no)
    if zero_last:
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(x + self.net(x))


class Encoder(nn.Module):
    def __init__(self, ni, width, blocks):
        super().__init__()
        self.input = nn.Linear(ni,width)
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
    def forward(self,x):
        return self.blocks(self.input(x))


class JointDeepONet(nn.Module):
    """功率分支与空间分支联合训练；新模式从结构上保证持续加热时不降温。"""
    def __init__(self, geometry: Geometry, settings: PhysicalSettings, config: dict):
        super().__init__()
        self.geometry, self.settings = geometry, settings
        self.model_config = copy.deepcopy(config)
        self.time_response = config.get('time_response', 'free')
        if self.time_response not in ('free', 'monotone_heating'):
            raise ValueError('未知温度响应模式：' + str(self.time_response))
        heating = self.time_response == 'monotone_heating'
        self.correction_power_degree = config.get('correction_power_degree')
        if self.correction_power_degree is not None:
            if (not heating or type(self.correction_power_degree) is not int
                    or not 1 <= self.correction_power_degree <= 3):
                raise ValueError('低阶功率校正要求持续升温模式，功率多项式次数必须为1、2或3。')
        self.power_independent_initial = config.get('power_independent_initial', False)
        if type(self.power_independent_initial) is not bool or (self.power_independent_initial
                and self.correction_power_degree is None):
            raise ValueError('功率无关初温仅支持低阶功率校正的持续升温模式。')
        center_max = config.get('high_response_center_max_s')
        if center_max is not None and (type(center_max) not in (int, float)
                or not math.isfinite(float(center_max))
                or not 0 < float(center_max) < geometry.time_max
                or not self.power_independent_initial):
            raise ValueError('高保真观测时间基中心上限须位于0与训练时间范围内，并使用当前持续升温模型。')
        self.high_response_center_max_s = float(center_max) if center_max is not None else None
        low_floor_width = config.get('low_initial_floor_width_k')
        if low_floor_width is not None and (type(low_floor_width) not in (int, float)
                or not math.isfinite(float(low_floor_width)) or float(low_floor_width) <= 0
                or self.high_response_center_max_s is None):
            raise ValueError('低保真初温平滑约束需要正的开尔文宽度及高保真实测时间基上限。')
        self.low_initial_floor_width_k = float(low_floor_width) if low_floor_width is not None else None
        for name in COPPER_MODULE_FLAGS:
            if type(config.get(name, False)) is not bool:
                raise ValueError('铜区模块开关必须为布尔值：' + name)
        if any(config.get(name, False) for name in COPPER_MODULE_FLAGS) and (
                not heating or self.correction_power_degree != 2
                or not self.power_independent_initial or self.high_response_center_max_s is None
                or self.low_initial_floor_width_k is None):
            raise ValueError('铜区模块只支持方案3采用的完整持续升温结构。')
        if heating:
            if (settings.initial_simulation_k > min(settings.ambient_k, settings.cooling_simulation_k)
                    or settings.initial_experiment_k > min(settings.ambient_k, settings.cooling_experiment_k)):
                raise ValueError('持续升温模式要求名义初温不高于环境和水冷温度；预热后的冷却需另用模型。')
            if geometry.time_max <= 0 or float(config.get('temperature_scale_k', 250.0)) <= 0:
                raise ValueError('持续升温模式的时间范围和温度尺度必须为正数。')
        width, rank = int(config['width']), int(config['latent_dim'])
        self.branch = Encoder(1,width,int(config['blocks']))
        self.trunk = Encoder(3 if heating else 4,width,int(config['blocks']))
        self.branch_projection = nn.Linear(width,rank)
        self.trunk_projection = nn.Linear(width,rank)
        nn.init.zeros_(self.branch_projection.weight)
        nn.init.zeros_(self.branch_projection.bias)
        self.low_bias = nn.Parameter(torch.zeros(1))
        correction_inputs = 5 if heating else 6
        correction_outputs = rank + 1 if heating else 1
        if self.correction_power_degree is not None:
            correction_inputs = 3
            correction_outputs *= self.correction_power_degree + 1
        self.correction = mlp(correction_inputs,correction_outputs,
                              int(config['correction_width']),int(config['correction_depth']),True)
        self.copper_amplitude_net = mlp(5,1,32,2,True) if config.get('copper_amplitude',False) else None
        self.copper_time_scale_net = mlp(5,1,32,2,True) if config.get('copper_time_scale',False) else None
        self.copper_power_modulation_net = (mlp(3,4*rank,32,2,True)
                                            if config.get('copper_power_modulation',False) else None)
        if heating:
            self.heating_bias = nn.Parameter(torch.tensor(math.log(math.expm1(.02 / rank))))
            centers = torch.expm1(torch.linspace(0, math.log1p(geometry.time_max), rank))
            self.register_buffer('heating_centers', centers)
            self.register_buffer('heating_widths', .5 + .15 * centers)
            self.register_buffer('experiment_temperature_offset', torch.tensor(settings.initial_experiment_k))
        self.register_buffer('temperature_scale', torch.tensor(float(config.get('temperature_scale_k',250.0))))
        self.register_buffer('temperature_offset', torch.tensor(settings.initial_simulation_k))
        self.log_contact = nn.Parameter(torch.tensor(math.log(settings.contact_initial)),
                                        requires_grad=bool(config.get('learn_contact',True)))

    @property
    def contact_resistance(self):
        return self.log_contact.exp()

    def scaled(self,x):
        g = self.geometry
        return torch.stack((2*x[:,0]/g.radius_cu-1,
            2*(x[:,1]-g.bottom_cu)/(-g.bottom_cu)-1,
            2*x[:,2]/g.time_max-1, 2*x[:,3]/g.power_max-1, 2*x[:,4]-1),dim=1)

    def forward_low(self,x):
        if self.time_response == 'monotone_heating':
            _, raw, basis = self.heating_features(x)
            coefficients = torch.nn.functional.softplus(raw)
            if self.low_initial_floor_width_k is not None:
                rise = self.temperature_scale * (coefficients * basis).sum(1, keepdim=True)
                offset = self.temperature_scale * self.low_bias
                width = self.low_initial_floor_width_k
                # 零秒严格等于仿真初温，正温升区域仍尽量沿用原有预测。
                return self.temperature_offset + width * (
                    torch.nn.functional.softplus((rise + offset) / width)
                    - torch.nn.functional.softplus(offset / width))
            return self.temperature_offset + self.temperature_scale * (
                self.low_bias + (coefficients * basis).sum(1, keepdim=True))
        z = self.scaled(x)
        b = self.branch_projection(self.branch(z[:,3:4]))
        t = self.trunk_projection(self.trunk(z[:,[0,1,2,4]]))
        return self.temperature_offset + self.temperature_scale*((b*t).sum(1,keepdim=True)/math.sqrt(b.shape[1])+self.low_bias)

    def heating_features(self, x):
        z = self.scaled(x)
        branch = self.branch_projection(self.branch(z[:, 3:4]))
        spatial = self.trunk_projection(self.trunk(z[:, [0, 1, 4]]))
        raw = branch * spatial / math.sqrt(branch.shape[1]) + self.heating_bias
        # 系数不含时间；延迟升温基函数在0秒为0，导数非负，也能表达S型升温。
        initial = torch.sigmoid(-self.heating_centers / self.heating_widths)
        basis = (torch.sigmoid((x[:, 2:3] - self.heating_centers) / self.heating_widths)
                 - initial) / (1 - initial)
        return z, raw, basis

    def copper_features(self, z, raw):
        power = z[:, 3:4]
        return torch.cat((z[:, :2], power, (3*power.square()-1)/2,
                          torch.nn.functional.softplus(raw).mean(1, keepdim=True)), dim=1)

    def heating_correction(self, z, raw):
        if self.correction_power_degree is None:
            low_amplitude = torch.nn.functional.softplus(raw).sum(1, keepdim=True) + self.low_bias
            return self.correction(torch.cat((z[:, [0, 1, 3, 4]], low_amplitude), dim=1))
        # 空间网络不接收功率；功率只能通过低阶多项式进入校正，减少工况间的窄谷。
        radius = torch.where(z[:,4]>0,self.geometry.radius_sic,self.geometry.radius_cu)
        relative_radius = (z[:,0]+1)*self.geometry.radius_cu/(2*radius)
        spatial = torch.stack((2*relative_radius.square()-1, z[:, 1], z[:, 4]), dim=1)
        p = z[:, 3:4]
        polynomials = [torch.ones_like(p), p]
        if self.correction_power_degree >= 2:
            polynomials.append((3 * p.square() - 1) / 2)
        if self.correction_power_degree >= 3:
            polynomials.append((5 * p.pow(3) - 3 * p) / 2)
        coefficients = self.correction(spatial).reshape(len(z), self.correction_power_degree + 1, -1)
        correction = (coefficients * torch.cat(polynomials, dim=1).unsqueeze(-1)).sum(1)
        if self.power_independent_initial:
            # 加热前没有激光功率效应；只让功率多项式调整后续的升温系数。
            correction = torch.cat((correction[:, :-1], coefficients[:, 0, -1:]), dim=1)
        if self.copper_power_modulation_net is not None:
            centers = torch.linspace(10.,self.geometry.power_max,4,device=z.device,dtype=z.dtype)
            width = (self.geometry.power_max-10.)/3
            power_w = (p+1)*self.geometry.power_max/2
            basis = torch.exp(-.5*((power_w-centers)/width).square())
            amplitudes = self.copper_power_modulation_net(spatial).reshape(len(z),4,-1)
            residual = (amplitudes*basis.unsqueeze(-1)).sum(1)
            residual = residual*(z[:,4:5]<0)
            correction = torch.cat((correction[:, :-1]+residual,correction[:, -1:]),dim=1)
        return correction

    def forward(self,x,fidelity='high'):
        if x.ndim != 2 or x.shape[1] != 5:
            raise ValueError('坐标列必须为：半径、高度、时间、功率、材料标签。')
        if self.time_response == 'monotone_heating':
            if fidelity == 'low':
                return self.forward_low(x)
            if fidelity != 'high':
                raise ValueError('未知保真度。')
            z, raw, basis = self.heating_features(x)
            correction = self.heating_correction(z, raw)
            coefficients = torch.nn.functional.softplus(raw + correction[:, :-1])
            if self.copper_time_scale_net is not None:
                features = self.copper_features(z, raw)
                effective_time = x[:,2:3]/torch.exp(.5*torch.tanh(self.copper_time_scale_net(features)))
                effective_time = torch.where(z[:,4:5]<0,effective_time,x[:,2:3])
                initial = torch.sigmoid(-self.heating_centers/self.heating_widths)
                basis = (torch.sigmoid((effective_time-self.heating_centers)/self.heating_widths)
                         -initial)/(1-initial)
            if self.high_response_center_max_s is not None:
                basis = basis * (self.heating_centers <= self.high_response_center_max_s)
            rise = (coefficients * basis).sum(1,keepdim=True)
            if self.copper_amplitude_net is not None:
                features = self.copper_features(z, raw)
                gain = torch.exp(.5*torch.tanh(self.copper_amplitude_net(features)))
                rise = torch.where(z[:,4:5]<0,rise*gain,rise)
            # 初温偏移可学习，仍接受原初始条件软约束；不修改真实初温标签。
            return self.experiment_temperature_offset + self.temperature_scale * (
                self.low_bias + correction[:, -1:] + rise)
        low = self.forward_low(x)
        if fidelity == 'low':
            return low
        if fidelity != 'high':
            raise ValueError('未知保真度。')
        z = torch.cat((self.scaled(x),(low-self.temperature_offset)/self.temperature_scale),dim=1)
        # 不使用点值覆写水冷，不把整个高保真场机械加3度，不切断低保真坐标导数。
        return low + self.temperature_scale*self.correction(z)


def grad(y,x):
    return torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True,retain_graph=True)[0]


def axisymmetric_radial_laplacian(first_r, second_r, radius):
    # Symmetry at the axis gives lim_(r->0) (dT/dr)/r = d2T/dr2.
    return torch.where(radius > 0, second_r + first_r / radius.clamp_min(1e-9), 2 * second_r)


def points(r,z,t,p,m):
    return torch.cat((r,z,t,p,torch.full_like(r,float(m))),1)


def physics_loss(model: nn.Module, count: int, generator: torch.Generator,
                 *, fidelity: str = 'high', include_low_cooling: bool = True):
    """Constrain the selected field with its own initial and cooling conditions."""
    if fidelity not in ('low', 'high'):
        raise ValueError('未知物理约束的保真度。')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    g,s = model.geometry, model.settings
    n = max(4,int(count))
    def rand(n,lo=0.,hi=1.):
        return lo+(hi-lo)*torch.rand(n,1,device=device,dtype=dtype,generator=generator)
    def tp(n):
        return rand(n,0.001,g.time_max),rand(n,10.,g.power_max)
    def fixed(n,value):
        return torch.full((n,1),float(value),device=device,dtype=dtype)
    def eval_grad(x):
        x = x.detach().requires_grad_(True)
        value = model(x, fidelity=fidelity)
        return value,grad(value,x),x
    kmax=max(s.conductivity_sic,s.conductivity_cu)
    length=-g.bottom_cu
    ts=float(model.temperature_scale)
    pde_scale=max(kmax*ts/length**2,max(s.density_cu*s.capacity_cu,s.density_sic*s.capacity_sic)*ts/g.time_max)
    flux_scale=kmax*ts/length
    pdes=[]; initial=[]; axis=[]
    for material in (0,1):
        r=rand(n).sqrt()*(g.radius_sic if material else g.radius_cu)
        z=rand(n,g.bottom_sic if material else g.bottom_cu,0.)
        if not material:
            mask=(r<=g.radius_sic)&(z>=g.bottom_sic)
            while bool(mask.any()):
                k=int(mask.sum()); r[mask]=(rand(k).sqrt()*g.radius_cu).view(-1)
                z[mask]=rand(k,g.bottom_cu,0.).view(-1)
                mask=(r<=g.radius_sic)&(z>=g.bottom_sic)
        t,p=tp(n)
        x=points(r,z,t,p,material)
        temp,first,xg=eval_grad(x)
        drr=grad(first[:,0:1],xg)[:,0:1]
        dzz=grad(first[:,1:2],xg)[:,1:2]
        k=s.conductivity_sic if material else s.conductivity_cu
        cap=(s.density_sic*s.capacity_sic) if material else (s.density_cu*s.capacity_cu)
        residual=cap*first[:,2:3]-k*(axisymmetric_radial_laplacian(first[:,0:1],drr,xg[:,0:1])+dzz)
        pdes.append((residual/pde_scale).square().mean())
        x0=x.detach().clone(); x0[:,2]=0
        initial_target = s.initial_simulation_k if fidelity == 'low' else s.initial_experiment_k
        if hasattr(model, 'initial_temperature'):
            initial_target = model.initial_temperature(x0, fidelity)
        initial.append(((model(x0, fidelity=fidelity)-initial_target)/ts).square().mean())
        za=rand(n,g.bottom_sic,0.) if material else rand(n,g.bottom_cu,g.bottom_sic)
        _,ga,_=eval_grad(points(fixed(n,0),za,t,p,material))
        axis.append((ga[:,0:1]*length/ts).square().mean())
    def environmental(temp,h,eps):
        value=h*(temp-s.ambient_k)
        if s.radiation:
            value=value+eps*5.670374419e-8*(temp.pow(4)-s.ambient_k**4)
        return value
    boundaries=[]
    for material,rlo,rhi in ((1,0.,g.radius_sic),(0,g.radius_sic,g.radius_cu)):
        t,p=tp(n); r=rand(n,rlo,rhi)
        temp,der,_=eval_grad(points(r,fixed(n,0),t,p,material))
        k=s.conductivity_sic if material else s.conductivity_cu
        eps=s.emissivity_sic if material else s.emissivity_cu
        inward=(2*s.absorption*p/(math.pi*s.beam_radius**2)*torch.exp(-2*r.square()/s.beam_radius**2)) if material else 0.
        boundaries.append(((-k*der[:,1:2]+inward-environmental(temp,s.h_top,eps))/flux_scale).square().mean())
    t,p=tp(n)
    bottom=points(rand(n,0.,g.radius_cu),fixed(n,g.bottom_cu),t,p,0)
    temp,der,_=eval_grad(bottom)
    boundaries.append(((s.conductivity_cu*der[:,1:2]-environmental(temp,s.h_bottom,s.emissivity_cu))/flux_scale).square().mean())
    outer=points(fixed(n,g.radius_cu),rand(n,g.bottom_cu,0.),t,p,0)
    cooling_target = s.cooling_simulation_k if fidelity == 'low' else s.cooling_experiment_k
    boundaries.append(((model(outer, fidelity=fidelity)-cooling_target)/ts).square().mean())
    low_cooling = (((model(outer, fidelity='low')-s.cooling_simulation_k)/ts).square().mean()
                   if include_low_cooling else None)
    interfaces=[]
    for side in (False,True):
        t,p=tp(n); eps=1e-6
        if side:
            z=rand(n,g.bottom_sic,0.)
            xs=points(fixed(n,g.radius_sic-eps),z,t,p,1)
            xc=points(fixed(n,g.radius_sic+eps),z,t,p,0)
            index,normal=0,1.
        else:
            r=rand(n,0.,g.radius_sic)
            xs=points(r,fixed(n,g.bottom_sic+eps),t,p,1)
            xc=points(r,fixed(n,g.bottom_sic-eps),t,p,0)
            index,normal=1,-1.
        st,sg,_=eval_grad(xs); ct,cg,_=eval_grad(xc)
        qs=-s.conductivity_sic*sg[:,index:index+1]*normal
        qc=-s.conductivity_cu*cg[:,index:index+1]*normal
        interfaces.append(((qs-qc)/flux_scale).square().mean()+((st-ct-model.contact_resistance*qs)/ts).square().mean())
    losses = {'传热方程':torch.stack(pdes).mean(),
              '边界条件':torch.stack(boundaries+axis).sum(),
              '初始条件':torch.stack(initial).mean(),
              '材料界面':torch.stack(interfaces).mean()}
    if include_low_cooling:
        losses['仿真水冷'] = low_cooling
    losses['热阻正则'] = (model.log_contact-math.log(s.contact_initial)).square()
    return losses


@dataclass
class Table:
    x: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    group: np.ndarray
    refs: np.ndarray | None = None

    def validate(self):
        self.x=np.asarray(self.x,dtype=np.float32)
        self.y=np.asarray(self.y,dtype=np.float32).reshape(-1,1)
        self.weight=np.asarray(self.weight,dtype=np.float32).reshape(-1,1)
        self.group=np.asarray(self.group,dtype=np.int64).reshape(-1)
        n=len(self.x)
        if not n or self.x.shape!=(n,5) or self.y.shape!=(n,1) or len(self.group)!=n:
            raise ValueError('数据为空或规范化数据列不一致。')
        if not all(np.isfinite(a).all() for a in (self.x,self.y,self.weight)) or (self.weight<0).any():
            raise ValueError('数据含非有限数或负权重。')
        for gid in np.unique(self.group):
            if self.weight[self.group==gid].sum()<=0:
                raise ValueError('某个工况没有有效权重。')
        if self.refs is not None:
            self.refs=np.asarray(self.refs,dtype=np.int64)
            if len(self.refs)!=n or self.refs.min()<0 or self.refs.max()>=n:
                raise ValueError('温升参考索引无效。')
        return self


def read_parquet(path: Path):
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError('读取项目现有数据需要环境中的polars；请使用项目原训练环境。') from exc
    if not path.is_file():
        raise FileNotFoundError(f'现有规范化数据未找到：{path}。请在保有完整数据的项目目录运行。')
    return pl.read_parquet(path)


def power_keys(values):
    return np.rint(np.asarray(values,dtype=np.float64)*10000).astype(np.int64)


def fixed_splits(root: Path, low_fidelity_mode: str = 'legacy_split'):
    """训练入口显式选择低保真模式；旧配置和调用保留原60/10/10划分。"""
    if low_fidelity_mode not in ('legacy_split','all_training'):
        raise ValueError('未知低保真数据模式；只支持legacy_split或all_training。')
    d=read_yaml(root/'configs/splits.yaml')
    hf={k:tuple(float(x) for x in d['high_fidelity'][k+'_powers_w']) for k in ('training','validation','test')}
    if low_fidelity_mode=='all_training':
        lf={'training':tuple(float(p) for p in range(10,801,10)),'validation':(),'test':()}
        lf_counts=(80,0,0)
    else:
        lf={k:tuple(float(x) for x in d['simulation'][k+'_powers_w']) for k in ('validation','test')}
        lf['training']=tuple(float(p) for p in range(10,801,10) if p not in lf['validation']+lf['test'])
        lf_counts=(60,10,10)
    for source,expected in ((hf,(12,3,3)),(lf,lf_counts)):
        for i,k in enumerate(('training','validation','test')):
            if len(source[k])!=expected[i]: raise ValueError(f'功率划分数量不符合当前模式要求：高保真12/3/3，低保真{lf_counts}。')
        sets=[set(source[k]) for k in ('training','validation','test')]
        if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
            raise ValueError('功率划分出现交集。')
    return {'high':hf,'low':lf}


def select_split(frame,split,allowed):
    import polars as pl
    frame=frame.filter(pl.col('split')==split)
    keys=power_keys(frame['power_w'].to_numpy())
    expected=set(power_keys(allowed))
    if set(keys)!=expected:
        raise ValueError(f'{split}实际功率与锁定清单不一致。')
    if 'source_dataset' in frame.columns:
        sources=set(frame['source_dataset'].to_list())
        if split!='test' and 'test' in sources:
            raise ValueError('训练或验证记录混入测试来源。')
    return frame


def load_observations(root: Path, split: str, splits: dict, geometry: Geometry):
    if split not in ('train','validation','test'): raise ValueError('未知数据划分。')
    allowed=splits['high']['training' if split=='train' else split]
    processed=root/'data/processed'
    ir_name='test_ir_radial.parquet' if split=='test' else 'experiment_ir_radial.parquet'
    sensor_name='test_sensor_ring_raw.parquet' if split=='test' else 'sensor_ring_raw.parquet'
    ir=select_split(read_parquet(processed/ir_name),split,allowed).sort(['power_w','time_s','r_m'])
    keys=power_keys(ir['power_w'].to_numpy()); _,group=np.unique(keys,return_inverse=True)
    x=np.column_stack((ir['r_m'].to_numpy(),np.zeros(ir.height),ir['time_s'].to_numpy(),ir['power_w'].to_numpy(),np.ones(ir.height)))
    top=Table(x,ir['temperature_mean_k'].to_numpy(),ir['frame_weight'].to_numpy(),group).validate()
    metadata=read_yaml(root/'configs/data_metadata.yaml')['sensors']
    if metadata['coordinate_unit']!='m' or metadata['value_unit']!='degC' or metadata['time_unit']!='s':
        raise ValueError('本入口使用已确认的米、摄氏度、秒传感器数据。')
    sf=select_split(read_parquet(processed/sensor_name),split,allowed).sort(['power_w','sensor_type','time_raw'])
    tables={'顶部':top}
    for name,tag in (('热端','hot'),('冷端','cold')):
        import polars as pl
        f=sf.filter(pl.col('sensor_type')==tag)
        if set(power_keys(f['power_w'].to_numpy()))!=set(power_keys(allowed)):
            raise ValueError(name+'工况缺失，不允许静默跳过。')
        keys=power_keys(f['power_w'].to_numpy()); _,grp=np.unique(keys,return_inverse=True)
        ref=np.zeros(f.height,dtype=np.int64)
        for gid in np.unique(grp):
            mask=grp==gid; ids=np.flatnonzero(mask)
            ref[mask]=ids[np.argmin(f['time_raw'].to_numpy()[ids])]
        x=np.column_stack((f['radius_raw'].to_numpy(),np.full(f.height,geometry.bottom_cu),f['time_raw'].to_numpy(),f['power_w'].to_numpy(),np.zeros(f.height)))
        tables[name]=Table(x,f['value_mean_raw'].to_numpy()+KELVIN,np.ones(f.height),grp,ref).validate()
    return tables


def load_simulation(root: Path, powers):
    blocks=[]; targets=[]; groups=[]
    for i,power in enumerate(powers):
        f=read_parquet(root/'data/processed/simulation'/f'{power:g}W.parquet')
        x=f.select('r_m','z_m','time_s','power_w','material_id').to_numpy().astype(np.float32)
        if set(power_keys(x[:,3]))!={int(round(power*10000))}:
            raise ValueError('仿真文件功率内容不符。')
        if set(np.unique(x[:,4]))!={0.,1.}: raise ValueError('仿真必须包括铜和碳化硅完整场。')
        blocks.append(x); targets.append(f['temperature_k'].to_numpy())
        groups.append(2*i+x[:,4].astype(np.int64))
    x=np.concatenate(blocks); y=np.concatenate(targets)
    return Table(x,y,np.ones(len(x)),np.concatenate(groups)).validate()


class Pool:
    def __init__(self,table:Table,device):
        self.table=table
        self.device=device
        self.groups=[np.flatnonzero(table.group==g) for g in np.unique(table.group)]
    def tensors(self,ids):
        table=self.table
        return (torch.as_tensor(table.x[ids],device=self.device),
                torch.as_tensor(table.y[ids],device=self.device),
                torch.as_tensor(table.weight[ids],device=self.device))
    def sample(self,per_group,rng):
        return np.concatenate([rng.choice(g,int(per_group),replace=len(g)<per_group) for g in self.groups])


def sensor_tail_pool(table: Table, device, window_seconds=20.):
    """Keep only observed tail points from each training sensor power."""
    window=float(window_seconds)
    if not math.isfinite(window) or window<=0:
        raise ValueError('传感器末段窗口必须为有限正秒数。')
    selected=[]
    for group in np.unique(table.group):
        ids=np.flatnonzero(table.group==group)
        times=table.x[ids,2].astype(np.float64)
        if float(times.max()-times.min())+1e-5<window:
            raise ValueError(f'传感器末段不足{window:g}秒，不可外推实测斜率。')
        tail=ids[times>=times.max()-window-1e-5]
        if len(np.unique(table.x[tail,2]))<3:
            raise ValueError('传感器末段至少需要三个不同的实测时刻。')
        selected.append(tail)
    ids=np.concatenate(selected)
    return Pool(Table(table.x[ids],table.y[ids],table.weight[ids],table.group[ids]).validate(),device)


def sensor_early_pool(table: Table, device, end_seconds=5.):
    """Use only observed seconds 1..end for each training sensor power."""
    end=float(end_seconds)
    if not math.isfinite(end) or end<=1.:
        raise ValueError('传感器早期温升的末时刻必须大于1秒。')
    selected=[]; refs=[]; offset=0
    for group in np.unique(table.group):
        ids=np.flatnonzero(table.group==group)
        times=table.x[ids,2].astype(np.float64)
        early=ids[(times>=1.-1e-5)&(times<=end+1e-5)]
        early=early[np.argsort(table.x[early,2])]
        observed=table.x[early,2]
        if (len(np.unique(observed))<2 or not np.isclose(observed[0],1.,atol=1e-5)
                or not np.isclose(observed[-1],end,atol=1e-5)):
            raise ValueError(f'传感器早期温升必须有真实的1秒和{end:g}秒测点。')
        selected.append(early)
        refs.extend([offset]*len(early))
        offset+=len(early)
    ids=np.concatenate(selected)
    early_table=Table(table.x[ids],table.y[ids],table.weight[ids],table.group[ids],
                      np.asarray(refs,dtype=np.int64)).validate()
    return Pool(early_table,device)


def sensor_early_rise_loss(model, pool: Pool, worst_fraction=0.):
    """Match each actual early temperature rise relative to its observed second 1."""
    if pool.table.refs is None:
        raise ValueError('传感器早期温升缺少实测参考时刻。')
    x,y,weight=pool.tensors(np.arange(len(pool.table.x)))
    rx,ry,_=pool.tensors(pool.table.refs)
    _,groups=np.unique(pool.table.group,return_inverse=True)
    error=((model(x)-model(rx))-(y-ry))/model.temperature_scale
    return grouped_mse(error,weight,torch.as_tensor(groups,device=x.device),worst_fraction)


def sensor_tail_slope_loss(model, pool: Pool, window_seconds=20., worst_fraction=0.):
    """Match measured late-time slopes, scaled as temperature change over the window."""
    x,y,weight=pool.tensors(np.arange(len(pool.table.x)))
    _,groups=np.unique(pool.table.group,return_inverse=True)
    group=torch.as_tensor(groups,device=x.device)
    count=len(pool.groups)
    w=weight.view(-1)
    t=x[:,2]
    sum_w=torch.zeros(count,device=x.device,dtype=t.dtype).scatter_add_(0,group,w)
    mean_t=torch.zeros_like(sum_w).scatter_add_(0,group,w*t)/sum_w
    centered=t-mean_t[group]
    denominator=torch.zeros_like(sum_w).scatter_add_(0,group,w*centered.square())
    if (denominator<=0).any():
        raise ValueError('传感器末段实测时刻没有足够变化，无法计算斜率。')
    errors=(model(x)-y).view(-1)
    numerator=torch.zeros_like(sum_w).scatter_add_(0,group,w*centered*errors)
    changes=(numerator/denominator)*float(window_seconds)/model.temperature_scale
    return grouped_mse(changes[:,None],torch.ones_like(changes[:,None]),
                       torch.arange(count,device=x.device),worst_fraction)


def sensor_tail_endpoint_loss(model, pool: Pool, worst_fraction=0.):
    """Anchor each training curve at its last observed sensor temperature."""
    table=pool.table
    selected=np.concatenate([ids[np.isclose(table.x[ids,2],table.x[ids,2].max(),
                                        rtol=0.,atol=1e-5)] for ids in pool.groups])
    x,y,weight=pool.tensors(selected)
    _,groups=np.unique(table.group[selected],return_inverse=True)
    error=(model(x)-y)/model.temperature_scale
    return grouped_mse(error,weight,torch.as_tensor(groups,device=x.device),worst_fraction)


def grouped_mse(error:Tensor,weights:Tensor,group:Tensor,worst_fraction=0.,group_weights=None):
    if not math.isfinite(float(worst_fraction)) or not 0 <= float(worst_fraction) <= 1:
        raise ValueError('最差工况损失比例必须在0到1之间。')
    group=group.view(-1)
    n=int(group.max())+1
    den=torch.zeros(n,device=error.device,dtype=error.dtype).scatter_add_(0,group,weights.view(-1))
    num=torch.zeros_like(den).scatter_add_(0,group,(weights*error.square()).view(-1))
    good=den>0
    values=num[good]/den[good]
    if not len(values):
        raise ValueError('损失没有有效权重。')
    if group_weights is None:
        average=values.mean()
    else:
        gw=group_weights[good]
        if not torch.isfinite(gw).all() or (gw<0).any() or gw.sum()<=0:
            raise ValueError('工况损失权重无效。')
        average=(values*gw).sum()/gw.sum()
    return (1-float(worst_fraction))*average+float(worst_fraction)*values.max()


def observation_loss(model:JointDeepONet,pool:Pool,ids,with_delta=False,
                     worst_fraction=0.,power_weight=0.,edge_weight=0.):
    x,y,w=pool.tensors(ids); pred=model(x)
    _,group=np.unique(pool.table.group[ids],return_inverse=True)
    group=torch.as_tensor(group,device=x.device)
    group_weights=None
    if power_weight or edge_weight:
        if min(float(power_weight),float(edge_weight))<0 or not all(math.isfinite(float(v)) for v in (power_weight,edge_weight)):
            raise ValueError('功率和边缘损失权重必须是有限非负数。')
        if edge_weight:
            w=w*(1+float(edge_weight)*(x[:,0:1]/model.geometry.radius_sic).square())
        if power_weight:
            group_weights=torch.stack([1+float(power_weight)*x[group==gid,3].mean()/model.geometry.power_max
                                      for gid in range(int(group.max())+1)])
    absolute=grouped_mse((pred-y)/model.temperature_scale,w,group,worst_fraction,group_weights)
    if not with_delta: return absolute
    refs=pool.table.refs[ids]
    rx,ry,_=pool.tensors(refs)
    base=model(rx)
    delta=grouped_mse(((pred-base)-(y-ry))/model.temperature_scale,w,group,worst_fraction,group_weights)
    return absolute,delta


def low_parameters(model):
    return [p for name,p in model.named_parameters()
            if not name.startswith(HIGH_PARAMETER_PREFIXES) and name!='log_contact']


def high_parameters(model):
    return [p for name,p in model.named_parameters() if name.startswith(HIGH_PARAMETER_PREFIXES)]


def set_low_trainable(model,enabled):
    for parameter in low_parameters(model):
        parameter.requires_grad_(bool(enabled))


def initialize_low_network(model,path:Path):
    """只借用已有低保真权重；实验校正与热阻重新拟合，不冒充续训。"""
    source,state=load_model(path,'cpu')
    if model.time_response!='monotone_heating' or source.time_response!='monotone_heating':
        raise ValueError('低保真初始化只支持相同的持续升温结构。')
    if model.geometry!=source.geometry:
        raise ValueError('低保真初始化的几何不一致。')
    simulation_settings={name:value for name,value in asdict(model.settings).items()
                         if name not in ('cooling_experiment_k','initial_experiment_k')}
    if any(getattr(source.settings,name)!=value for name,value in simulation_settings.items()):
        raise ValueError('低保真初始化的仿真物性或边界条件不一致。')
    if float(model.temperature_scale)!=float(source.temperature_scale):
        raise ValueError('低保真初始化的温度尺度不一致，不能静默覆盖新配置。')
    current=model.state_dict()
    copied={name:value for name,value in source.state_dict().items()
            if not name.startswith(HIGH_PARAMETER_PREFIXES)
            and name not in ('log_contact','experiment_temperature_offset')}
    expected={name for name in current
              if not name.startswith(HIGH_PARAMETER_PREFIXES)
              and name not in ('log_contact','experiment_temperature_offset')}
    if set(copied)!=expected or any(current[name].shape!=value.shape for name,value in copied.items()):
        raise ValueError('低保真初始化的网络尺寸不一致，不能混接权重。')
    current.update(copied)
    model.load_state_dict(current)
    return {'来源检查点':str(Path(path).resolve()),'检查点SHA256':sha256(Path(path)),
            '来源轮次':int(state['epoch']),'迁移内容':'仅低保真网络；校正网络和接触热阻未迁移'}


def power_smoothness_loss(model,count,generator,step_w=40.):
    """对最终温度的功率二阶差分施加软约束，不强制温度按功率排序。"""
    g=model.geometry
    if not math.isfinite(float(step_w)) or not 0<float(step_w)<(g.power_max-10)/2:
        raise ValueError('功率平滑间隔不在模型功率范围内。')
    n=max(4,int(count))
    parameter=next(model.parameters())
    device,dtype=parameter.device,parameter.dtype
    random=torch.rand(n,5,device=device,dtype=dtype,generator=generator)
    material=(random[:,4]>=.5).to(dtype)
    radius=torch.where(material>0,g.radius_sic,g.radius_cu)
    bottom=torch.where(material>0,g.bottom_sic,g.bottom_cu)
    r=random[:,0].sqrt()*radius
    z=bottom+random[:,1]*(-bottom)
    # 铜点落入嵌入体时移到其下方；SiC与铜不混用材料标签。
    inside=(material==0)&(r<g.radius_sic)&(z>g.bottom_sic)
    z=torch.where(inside,g.bottom_cu+random[:,1]*(g.bottom_sic-g.bottom_cu),z)
    x=torch.stack((r,z,random[:,2]*g.time_max,
                   10+step_w+random[:,3]*(g.power_max-10-2*step_w),material),dim=1)
    left=x.clone(); right=x.clone()
    left[:,3]-=step_w; right[:,3]+=step_w
    return torch.stack([((model(left,fidelity)-2*model(x,fidelity)+model(right,fidelity))
                         /model.temperature_scale).square().mean() for fidelity in ('low','high')]).mean()


@torch.no_grad()
def predict(model,coordinates,batch_size=8192,fidelity='high'):
    device=next(model.parameters()).device
    dtype=next(model.parameters()).dtype
    outputs=[]; model.eval()
    for j in range(0,len(coordinates),batch_size):
        x=torch.as_tensor(coordinates[j:j+batch_size],device=device,dtype=dtype)
        outputs.append(model(x,fidelity=fidelity).detach().cpu().numpy().reshape(-1))
    return np.concatenate(outputs) if outputs else np.empty(0)


def table_metrics(table:Table,prediction):
    y=table.y.reshape(-1); prediction=np.asarray(prediction).reshape(-1)
    error=prediction-y
    rows=[]
    for gid in np.unique(table.group):
        mask=table.group==gid; w=table.weight[mask].reshape(-1).astype(np.float64); w=w/w.sum()
        e=error[mask].astype(np.float64)
        row={'功率_W':float(table.x[mask][0,3]),'均方根误差_℃':float(np.sqrt(np.sum(w*e**2))),
             '平均绝对误差_℃':float(np.sum(w*np.abs(e))),'平均偏差_℃':float(np.sum(w*e)),
             '最大绝对误差_℃':float(np.max(np.abs(e)))}
        if table.refs is not None:
            rise_error=(prediction-prediction[table.refs])-(y-y[table.refs])
            row['温升均方根误差_℃']=float(np.sqrt(np.sum(w*rise_error[mask]**2)))
        rows.append(row)
    summary={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='功率_W'}
    return {'汇总':summary,'逐功率':rows}


def evaluate(model,tables,selection_config=None):
    from joint_temperature_selection import selection_score
    detail={name:table_metrics(table,predict(model,table.x)) for name,table in tables.items()}
    return {'综合选择分数_℃':selection_score(detail,selection_config),'分项':detail}


def snapshot(model,optimizer,scheduler,epoch,history,config,splits,best_score,rng):
    schema = ('joint_deeponet_8000_v7_copper_modules'
              if any(model.model_config.get(name, False) for name in COPPER_MODULE_FLAGS) else
              'joint_deeponet_8000_v6_fixed_low_initial' if model.low_initial_floor_width_k is not None else
              'joint_deeponet_8000_v5_high_observed_window' if model.high_response_center_max_s is not None else
              'joint_deeponet_8000_v4_power_independent_initial' if model.power_independent_initial else
              'joint_deeponet_8000_v3_smooth_power' if model.correction_power_degree is not None else
              'joint_deeponet_8000_v2_heating' if model.time_response == 'monotone_heating'
              else 'joint_deeponet_8000_v1')
    planned_epochs=config.get('training',{}).get('epochs',TOTAL_EPOCHS)
    return {'schema':schema,'epoch':int(epoch),'planned_epochs':planned_epochs,
            'model_state':model.state_dict(),'model_config':model.model_config,
            'geometry':asdict(model.geometry),'physical_settings':asdict(model.settings),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'history':history,'config':config,'splits':splits,'best_score':float(best_score),
            'numpy_generator':copy.deepcopy(rng.bit_generator.state),
            'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def load_model(path:Path,device='cpu'):
    state=torch.load(path,map_location=device,weights_only=False)
    schemas = {'joint_deeponet_8000_v1': 'free', 'joint_deeponet_8000_v2_heating': 'monotone_heating',
               'joint_deeponet_8000_v3_smooth_power':'monotone_heating',
               'joint_deeponet_8000_v4_power_independent_initial':'monotone_heating',
               'joint_deeponet_8000_v5_high_observed_window':'monotone_heating',
               'joint_deeponet_8000_v6_fixed_low_initial':'monotone_heating',
               'joint_deeponet_8000_v7_copper_modules':'monotone_heating'}
    if state.get('schema') not in schemas:
        raise ValueError('该入口只读取本轮联合训练的检查点，不冒充旧模型兼容。')
    if state['model_config'].get('time_response', 'free') != schemas[state['schema']]:
        raise ValueError('检查点版本与温度响应模式不一致，禁止把旧权重解释为新的持续升温模型。')
    if ((state['model_config'].get('correction_power_degree') is not None)
            != (state['schema'] in ('joint_deeponet_8000_v3_smooth_power',
                                   'joint_deeponet_8000_v4_power_independent_initial',
                                   'joint_deeponet_8000_v5_high_observed_window',
                                   'joint_deeponet_8000_v6_fixed_low_initial',
                                   'joint_deeponet_8000_v7_copper_modules'))
            or state['model_config'].get('power_independent_initial', False)
            != (state['schema'] in ('joint_deeponet_8000_v4_power_independent_initial',
                                    'joint_deeponet_8000_v5_high_observed_window',
                                    'joint_deeponet_8000_v6_fixed_low_initial',
                                    'joint_deeponet_8000_v7_copper_modules'))
            or (state['model_config'].get('high_response_center_max_s') is not None)
            != (state['schema'] in ('joint_deeponet_8000_v5_high_observed_window',
                                    'joint_deeponet_8000_v6_fixed_low_initial',
                                    'joint_deeponet_8000_v7_copper_modules'))
            or (state['model_config'].get('low_initial_floor_width_k') is not None)
            != (state['schema'] in ('joint_deeponet_8000_v6_fixed_low_initial',
                                    'joint_deeponet_8000_v7_copper_modules'))
            or any(state['model_config'].get(name, False) for name in COPPER_MODULE_FLAGS)
            != (state['schema'] == 'joint_deeponet_8000_v7_copper_modules')):
        raise ValueError('检查点版本与低阶功率校正配置不一致，禁止混接权重。')
    model=JointDeepONet(Geometry(**state['geometry']),PhysicalSettings(**state['physical_settings']),state['model_config']).to(device)
    model.load_state_dict(state['model_state'])
    return model,state


def save_atomic(state,path:Path):
    tmp=path.with_name(path.name+'.tmp')
    torch.save(state,tmp); tmp.replace(path)
