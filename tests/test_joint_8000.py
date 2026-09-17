"""联合训练入口的数值与数据约束测试；合成数据只用于测试，不发布为预测结果。"""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from joint_temperature_core import *


def settings(cooling=25.):
    return PhysicalSettings(295.15,cooling+273.15,295.15,295.15,295.15,.8,.02,9.,4.3,.5,.5,True,
                            8900.,400.,401.,3170.,700.,120.,7.34072435302768e-5)


def model(cooling=25.):
    torch.manual_seed(42)
    return JointDeepONet(Geometry(),settings(cooling),{'width':8,'latent_dim':8,'blocks':1,
                'correction_width':8,'correction_depth':2,'temperature_scale_k':250.,'learn_contact':True})


def coordinates():
    return torch.tensor([[.01,-.003,5.,400.,1.],[.035,-.015,20.,400.,0.]],requires_grad=True)


def test_fixed_8000():
    assert TOTAL_EPOCHS==8000
    cfg=read_yaml(Path(__file__).resolve().parents[1]/'configs/联合训练8000轮.yaml')
    assert cfg['training']['epochs']==8000
    assert cfg['physical']['simulation_cooling_c']==22.
    assert cfg['physical']['experiment_cooling_c']==25.
    assert 'early_stopping' not in cfg['training']


def test_all_parameters_enabled_from_start():
    assert all(p.requires_grad for p in model().parameters())


def test_low_supervision_does_not_update_correction():
    m=model();x=coordinates()
    ((m(x,'low')-310)**2).mean().backward()
    assert all(p.grad is None for p in m.correction.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.branch_projection.parameters())


def test_joint_step_updates_both_branches():
    m=model();opt=torch.optim.AdamW(m.parameters(),lr=.0001)
    before={n:p.detach().clone() for n,p in m.named_parameters()}
    for _ in range(2):
        opt.zero_grad();x=coordinates()
        loss=((m(x,'low')-310)/250).square().mean()+((m(x)-320)/250).square().mean()
        physics=physics_loss(m,4,torch.Generator().manual_seed(5))
        loss=loss+sum(physics.values());loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        opt.step()
    assert any(not torch.equal(before[n],p) for n,p in m.named_parameters() if n.startswith('branch'))
    assert any(not torch.equal(before[n],p) for n,p in m.named_parameters() if n.startswith('correction'))
    assert float(m.contact_resistance)>0


def test_distinct_cooling_and_unchanged_initial():
    a=model(22.);b=model(25.)
    pa=physics_loss(a,4,torch.Generator().manual_seed(7))
    pb=physics_loss(b,4,torch.Generator().manual_seed(7))
    assert float(pb['边界条件']-pa['边界条件'])==pytest.approx((3/250)**2,rel=.01)
    assert float(pa['仿真水冷'])==float(pb['仿真水冷'])==0.
    assert b.settings.initial_experiment_k==295.15


def test_no_constant_boundary_overwrite():
    m=model();x=coordinates();x=x.detach().clone();x[1,0]=m.geometry.radius_cu;x.requires_grad_()
    out=m(x)
    assert out.requires_grad
    assert 'isclose' not in __import__('inspect').getsource(JointDeepONet.forward)
    deriv=grad(out,x);second=grad(deriv[:,0:1],x)
    assert torch.isfinite(deriv).all() and torch.isfinite(second).all()


def test_group_metrics_and_sensor_reference():
    x=np.array([[.028,-.0175,1,100,0],[.028,-.0175,2,100,0]],np.float32)
    t=Table(x,np.array([298.15,299.15]),np.ones(2),np.zeros(2),np.array([0,0])).validate()
    r=table_metrics(t,t.y[:,0]+2)['汇总']
    assert r['均方根误差_℃']==pytest.approx(2.)
    assert r['温升均方根误差_℃']==0.


def test_save_reload_and_sources(tmp_path):
    m=model();opt=torch.optim.AdamW(m.parameters(),lr=.0001)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    state=snapshot(m,opt,scheduler,123,[],{}, {},2.5,np.random.default_rng(1))
    path=tmp_path/'最近模型.pt';save_atomic(state,path)
    loaded,payload=load_model(path)
    assert payload['epoch']==123 and payload['planned_epochs']==8000
    assert torch.equal(m(coordinates()),loaded(coordinates()))
    assert loaded.settings.cooling_experiment_k==298.15


def test_test_export_guard(tmp_path):
    from joint_temperature_figures import export_results
    with pytest.raises(RuntimeError,match='锁定'):
        export_results(model(),tmp_path,tmp_path,{}, {})


def test_plot_outputs_are_real_files(tmp_path):
    from joint_temperature_figures import training_plots,save_comparison_curve
    history=[]
    for i in range(1,4):
        history.append({'轮次':i,'接触热阻_m2K_W':7e-5,'总损失':1/i,
                        '验证综合分数_℃':2.,'顶部验证均方根误差_℃':3.,
                        '热端验证均方根误差_℃':1.,'冷端验证均方根误差_℃':1.})
    training_plots(history,tmp_path,{})
    assert (tmp_path/'训练曲线/界面接触热阻变化.png').is_file()
    assert (tmp_path/'训练曲线/损失函数变化.png').is_file()
    save_comparison_curve(tmp_path/'单元测试.png',np.arange(3),np.array([25.,26.,27.]),np.array([25.,25.5,27.]),'单元测试，不是项目结果','时间 / s')
    assert (tmp_path/'单元测试.png').stat().st_size>1000
