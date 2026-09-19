"""联合训练入口的数值与数据约束测试；合成数据只用于测试，不发布为预测结果。"""
import argparse
import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from joint_temperature_core import *

ROOT=Path(__file__).resolve().parents[1]


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


def test_default_config_uses_all_low_fidelity_for_training():
    cfg=read_yaml(ROOT/'configs/联合训练8000轮.yaml')
    assert cfg.get('data',{}).get('low_fidelity_mode')=='all_training'


def test_all_low_fidelity_split_keeps_high_fidelity_unchanged():
    original_hash=sha256(ROOT/'configs/splits.yaml')
    splits=fixed_splits(ROOT,'all_training')
    assert splits['low']=={'training':tuple(float(p) for p in range(10,801,10)),
                           'validation':(),'test':()}
    assert splits['high']==fixed_splits(ROOT)['high']
    assert tuple(len(splits['high'][k]) for k in ('training','validation','test'))==(12,3,3)
    assert sha256(ROOT/'configs/splits.yaml')==original_hash


def test_legacy_low_fidelity_split_is_explicitly_preserved():
    splits=fixed_splits(ROOT,'legacy_split')
    assert splits==fixed_splits(ROOT)
    assert tuple(len(splits['low'][k]) for k in ('training','validation','test'))==(60,10,10)


def test_unknown_low_fidelity_mode_is_rejected():
    with pytest.raises(ValueError,match='低保真'):
        fixed_splits(ROOT,'unknown_mode')


@pytest.fixture
def small_joint_project(tmp_path):
    """真实读写流程用的小数据；不含任何高保真测试文件。"""
    import polars as pl
    root=tmp_path/'单元测试项目'
    configs=root/'configs'; configs.mkdir(parents=True)
    for name in ('splits.yaml','geometry.yaml','materials.yaml','boundary_conditions.yaml','data_metadata.yaml'):
        shutil.copy2(ROOT/'configs'/name,configs/name)
    splits=fixed_splits(ROOT)
    processed=root/'data/processed'; (processed/'simulation').mkdir(parents=True)
    top=[]; sensors=[]
    for name,split in (('training','train'),('validation','validation')):
        for power in splits['high'][name]:
            for t in (5.,10.):
                top.append({'split':split,'source_dataset':'experiment','power_w':power,
                            'r_m':.01,'time_s':t,'temperature_mean_k':295.15+power*t/1000.,'frame_weight':1.})
            for tag,r in (('hot',.028),('cold',.0415)):
                for t in (1.,2.):
                    sensors.append({'split':split,'source_dataset':'experiment','power_w':power,
                                    'sensor_type':tag,'time_raw':t,'radius_raw':r,'value_mean_raw':25.+power*t/10000.})
    pl.DataFrame(top).write_parquet(processed/'experiment_ir_radial.parquet')
    pl.DataFrame(sensors).write_parquet(processed/'sensor_ring_raw.parquet')
    for power in range(10,801,10):
        old_split=next('train' if k=='training' else k for k,ps in splits['low'].items() if power in ps)
        rows=[{'r_m':.035 if mat==0 else .01,'z_m':-.015 if mat==0 else -.005,
               'time_s':t,'power_w':float(power),'material_id':mat,
               'temperature_k':295.15+power*t/1000.,'split':old_split}
              for mat in (0,1) for t in (0.,2.)]
        pl.DataFrame(rows).write_parquet(processed/'simulation'/f'{power}W.parquet')
    return root


@pytest.mark.parametrize('mode,powers,samples',[('all_training',80,2560),('legacy_split',60,1920)])
def test_simulation_sampling_covers_every_power_and_material(small_joint_project,mode,powers,samples):
    splits=fixed_splits(small_joint_project,mode)
    table=load_simulation(small_joint_project,splits['low']['training'])
    ids=Pool(table,torch.device('cpu')).sample(16,np.random.default_rng(7))
    groups,counts=np.unique(table.group[ids],return_counts=True)
    assert len(ids)==samples and len(groups)==2*powers
    assert (counts==16).all()
    assert set(table.x[ids,3])==set(splits['low']['training'])


def small_training_entrypoint(monkeypatch):
    spec=importlib.util.spec_from_file_location('joint_8000_test_entrypoint',ROOT/'scripts/联合训练8000轮.py')
    trainer=importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    # 仅在测试进程把循环缩到1轮；正式配置与入口仍强制8000轮。
    monkeypatch.setattr(trainer,'TOTAL_EPOCHS',1)
    return trainer


def small_training_config(mode):
    cfg=copy.deepcopy(read_yaml(ROOT/'configs/联合训练8000轮.yaml'))
    if mode=='all_training': cfg['data']={'low_fidelity_mode':mode}
    else:
        cfg.pop('data',None)
        cfg['model'].pop('time_response',None)
    cfg['model'].update(width=8,latent_dim=8,blocks=1,correction_width=8,correction_depth=2)
    cfg['training'].update(epochs=1,device='cpu',ir_batch_size=24,physics_points_per_region=4,
                           validation_every=1,plots_every=2)
    cfg['visualization']['export_test_after_training']=False
    return cfg


@pytest.mark.parametrize('mode,counts',[('all_training',(80,0,0)),('legacy_split',(60,10,10))])
def test_joint_training_uses_selected_split_without_high_fidelity_test_labels(small_joint_project,monkeypatch,mode,counts):
    import yaml
    trainer=small_training_entrypoint(monkeypatch)
    cfg=small_training_config(mode)
    path=small_joint_project/'configs/联合训练8000轮.yaml'
    path.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
    trainer.train(argparse.Namespace(root=str(small_joint_project),config=str(path),device='cpu',output=None,resume=None))
    output,=tuple((small_joint_project/'研究记录').iterdir())
    assert output.name.startswith('联合训练8000轮_低保真全量_持续升温_' if mode=='all_training' else '联合训练8000轮_')
    source=json.loads((output/'运行来源.json').read_text(encoding='utf-8'))
    assert tuple(len(source['功率划分']['low'][k]) for k in ('training','validation','test'))==counts
    assert tuple(len(source['功率划分']['high'][k]) for k in ('training','validation','test'))==(12,3,3)
    row=json.loads((output/'训练记录.jsonl').read_text(encoding='utf-8'))
    assert row['轮次']==1 and row['优化步数']==1
    assert '顶部验证均方根误差_℃' in row
    assert ('仿真验证抽样均方根误差_℃' in row)==(mode=='legacy_split')
    _,state=load_model(output/'最近模型.pt')
    assert state['config']==cfg
    assert state['schema']==('joint_deeponet_8000_v2_heating' if mode=='all_training' else 'joint_deeponet_8000_v1')
    assert tuple(len(state['splits']['low'][k]) for k in ('training','validation','test'))==counts
    best_metrics=json.loads((output/'验证最佳指标.json').read_text(encoding='utf-8'))
    assert best_metrics['验证最佳轮次']==1
    assert best_metrics['综合选择分数_℃']==pytest.approx(state['best_score'])
    assert not (small_joint_project/'data/processed/test_ir_radial.parquet').exists()
    assert not (small_joint_project/'data/processed/test_sensor_ring_raw.parquet').exists()


def test_old_checkpoint_rejects_full_low_fidelity_config_without_overwrite(small_joint_project,monkeypatch):
    import yaml
    trainer=small_training_entrypoint(monkeypatch)
    path=small_joint_project/'configs/联合训练8000轮.yaml'
    old=small_training_config('legacy_split')
    path.write_text(yaml.safe_dump(old,allow_unicode=True),encoding='utf-8')
    output=small_joint_project/'单元测试旧划分'
    args=argparse.Namespace(root=str(small_joint_project),config=str(path),device='cpu',output=str(output),resume=None)
    trainer.train(args)
    frozen={name:sha256(output/name) for name in ('最近模型.pt','验证最佳模型.pt','训练记录.jsonl','运行来源.json')}
    changed=small_training_config('all_training')
    path.write_text(yaml.safe_dump(changed,allow_unicode=True),encoding='utf-8')
    args.resume=str(output/'最近模型.pt')
    with pytest.raises(ValueError,match='恢复时配置、划分或物理条件发生变化'):
        trainer.train(args)
    assert {name:sha256(output/name) for name in frozen}==frozen
    path.write_text(yaml.safe_dump(old,allow_unicode=True),encoding='utf-8')
    trainer.train(args)
    _,state=load_model(output/'最近模型.pt')
    assert state['epoch']==1 and len(state['splits']['low']['training'])==60


@pytest.mark.parametrize('mode',['legacy_split','all_training'])
def test_config_resumes_interrupted_training_with_a_new_update(small_joint_project,monkeypatch,mode):
    import yaml
    trainer=small_training_entrypoint(monkeypatch)
    monkeypatch.setattr(trainer,'TOTAL_EPOCHS',2)
    cfg=small_training_config(mode); cfg['training']['epochs']=2
    path=small_joint_project/'configs/联合训练8000轮.yaml'
    path.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
    output=small_joint_project/'单元测试中断恢复'
    args=argparse.Namespace(root=str(small_joint_project),config=str(path),device='cpu',output=str(output),resume=None)
    save=trainer.save_atomic

    def interrupt_after_complete_checkpoint(state,target):
        save(state,target)
        if state['epoch']==1 and target.name=='验证最佳模型.pt':
            raise KeyboardInterrupt('单元测试模拟中断，正式训练未运行')

    monkeypatch.setattr(trainer,'save_atomic',interrupt_after_complete_checkpoint)
    with pytest.raises(KeyboardInterrupt,match='单元测试模拟中断'):
        trainer.train(args)
    _,before=load_model(output/'最近模型.pt')
    assert before['epoch']==1 and len(before['history'])==1
    monkeypatch.setattr(trainer,'save_atomic',save)
    args.resume=str(output/'最近模型.pt')
    trainer.train(args)
    _,after=load_model(output/'最近模型.pt')
    rows=[json.loads(line) for line in (output/'训练记录.jsonl').read_text(encoding='utf-8').splitlines()]
    assert after['epoch']==2 and [r['轮次'] for r in rows]==[1,2]
    assert rows[-1]['优化步数']==1 and after['config']==cfg and after['splits']==before['splits']
    assert any(not torch.equal(value,after['model_state'][name]) for name,value in before['model_state'].items())
    old_step=max(float(s['step']) for s in before['optimizer']['state'].values())
    new_step=max(float(s['step']) for s in after['optimizer']['state'].values())
    assert new_step==old_step+1


def test_old_full_lf_model_cannot_resume_as_heating_without_overwrite(small_joint_project,monkeypatch):
    import yaml
    trainer=small_training_entrypoint(monkeypatch)
    cfg=small_training_config('all_training')
    cfg['model'].pop('time_response')
    path=small_joint_project/'configs/联合训练8000轮.yaml'
    path.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
    output=small_joint_project/'单元测试同划分旧响应'
    args=argparse.Namespace(root=str(small_joint_project),config=str(path),device='cpu',output=str(output),resume=None)
    trainer.train(args)
    frozen={name:sha256(output/name) for name in ('最近模型.pt','验证最佳模型.pt','训练记录.jsonl','运行来源.json')}
    cfg['model']['time_response']='monotone_heating'
    path.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
    args.resume=str(output/'最近模型.pt')
    with pytest.raises(ValueError,match='恢复时配置、划分或物理条件发生变化'):
        trainer.train(args)
    assert {name:sha256(output/name) for name in frozen}==frozen
