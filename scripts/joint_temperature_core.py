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
TOTAL_EPOCHS = 8000


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
    """保留功率分支/时空分支与温度残差结构；全部参数从第1轮共同训练。"""
    def __init__(self, geometry: Geometry, settings: PhysicalSettings, config: dict):
        super().__init__()
        self.geometry, self.settings = geometry, settings
        self.model_config = copy.deepcopy(config)
        width, rank = int(config['width']), int(config['latent_dim'])
        self.branch = Encoder(1,width,int(config['blocks']))
        self.trunk = Encoder(4,width,int(config['blocks']))
        self.branch_projection = nn.Linear(width,rank)
        self.trunk_projection = nn.Linear(width,rank)
        nn.init.zeros_(self.branch_projection.weight)
        nn.init.zeros_(self.branch_projection.bias)
        self.low_bias = nn.Parameter(torch.zeros(1))
        self.correction = mlp(6,1,int(config['correction_width']),int(config['correction_depth']),True)
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
        z = self.scaled(x)
        b = self.branch_projection(self.branch(z[:,3:4]))
        t = self.trunk_projection(self.trunk(z[:,[0,1,2,4]]))
        return self.temperature_offset + self.temperature_scale*((b*t).sum(1,keepdim=True)/math.sqrt(b.shape[1])+self.low_bias)

    def forward(self,x,fidelity='high'):
        if x.ndim != 2 or x.shape[1] != 5:
            raise ValueError('坐标列必须为：半径、高度、时间、功率、材料标签。')
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


def points(r,z,t,p,m):
    return torch.cat((r,z,t,p,torch.full_like(r,float(m))),1)


def physics_loss(model: JointDeepONet, count: int, generator: torch.Generator):
    """同一步计算最终高保真场PDE/边界/初值/界面，并单独约束仿真水冷22℃。"""
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
        value = model(x)
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
        residual=cap*first[:,2:3]-k*(drr+first[:,0:1]/xg[:,0:1].clamp_min(1e-9)+dzz)
        pdes.append((residual/pde_scale).square().mean())
        x0=x.detach().clone(); x0[:,2]=0
        initial.append(((model(x0)-s.initial_experiment_k)/ts).square().mean())
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
    boundaries.append(((model(outer)-s.cooling_experiment_k)/ts).square().mean())
    low_cooling=((model(outer,fidelity='low')-s.cooling_simulation_k)/ts).square().mean()
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
    return {'传热方程':torch.stack(pdes).mean(),
            '边界条件':torch.stack(boundaries+axis).sum(),
            '初始条件':torch.stack(initial).mean(),
            '材料界面':torch.stack(interfaces).mean(),
            '仿真水冷':low_cooling,
            '热阻正则':(model.log_contact-math.log(s.contact_initial)).square()}


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


def fixed_splits(root: Path):
    d=read_yaml(root/'configs/splits.yaml')
    hf={k:tuple(float(x) for x in d['high_fidelity'][k+'_powers_w']) for k in ('training','validation','test')}
    lf={k:tuple(float(x) for x in d['simulation'][k+'_powers_w']) for k in ('validation','test')}
    lf['training']=tuple(float(p) for p in range(10,801,10) if p not in lf['validation']+lf['test'])
    for source,expected in ((hf,(12,3,3)),(lf,(60,10,10))):
        for i,k in enumerate(('training','validation','test')):
            if len(source[k])!=expected[i]: raise ValueError('现有固定功率划分不符合12/3/3或60/10/10。')
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


def grouped_mse(error:Tensor,weights:Tensor,group:Tensor):
    group=group.view(-1)
    n=int(group.max())+1
    den=torch.zeros(n,device=error.device,dtype=error.dtype).scatter_add_(0,group,weights.view(-1))
    num=torch.zeros_like(den).scatter_add_(0,group,(weights*error.square()).view(-1))
    good=den>0
    return (num[good]/den[good]).mean()


def observation_loss(model:JointDeepONet,pool:Pool,ids,with_delta=False):
    x,y,w=pool.tensors(ids); pred=model(x)
    _,group=np.unique(pool.table.group[ids],return_inverse=True)
    group=torch.as_tensor(group,device=x.device)
    absolute=grouped_mse((pred-y)/model.temperature_scale,w,group)
    if not with_delta: return absolute
    refs=pool.table.refs[ids]
    rx,ry,_=pool.tensors(refs)
    base=model(rx)
    delta=grouped_mse(((pred-base)-(y-ry))/model.temperature_scale,w,group)
    return absolute,delta


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


def evaluate(model,tables):
    detail={name:table_metrics(table,predict(model,table.x)) for name,table in tables.items()}
    top=detail['顶部']['汇总']['均方根误差_℃']
    absolute=np.mean([detail[n]['汇总']['均方根误差_℃'] for n in ('热端','冷端')])
    delta=np.mean([detail[n]['汇总']['温升均方根误差_℃'] for n in ('热端','冷端')])
    return {'综合选择分数_℃':float((top+.2*absolute+delta)/2.2),'分项':detail}


def snapshot(model,optimizer,scheduler,epoch,history,config,splits,best_score,rng):
    return {'schema':'joint_deeponet_8000_v1','epoch':int(epoch),'planned_epochs':TOTAL_EPOCHS,
            'model_state':model.state_dict(),'model_config':model.model_config,
            'geometry':asdict(model.geometry),'physical_settings':asdict(model.settings),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'history':history,'config':config,'splits':splits,'best_score':float(best_score),
            'numpy_generator':copy.deepcopy(rng.bit_generator.state),
            'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def load_model(path:Path,device='cpu'):
    state=torch.load(path,map_location=device,weights_only=False)
    if state.get('schema')!='joint_deeponet_8000_v1':
        raise ValueError('该入口只读取本轮联合训练的检查点，不冒充旧模型兼容。')
    model=JointDeepONet(Geometry(**state['geometry']),PhysicalSettings(**state['physical_settings']),state['model_config']).to(device)
    model.load_state_dict(state['model_state'])
    return model,state


def save_atomic(state,path:Path):
    tmp=path.with_name(path.name+'.tmp')
    torch.save(state,tmp); tmp.replace(path)
