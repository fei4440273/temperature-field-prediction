#!/usr/bin/env python3
"""连续8000轮多保真DeepONet联合训练；完成后生成测试集中文结果图册。"""
from __future__ import annotations
import argparse
import copy
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from joint_temperature_core import (
    TOTAL_EPOCHS, Geometry, PhysicalSettings, JointDeepONet, Pool,
    read_yaml, sha256, fixed_splits, load_observations, load_simulation,
    grouped_mse, observation_loss, physics_loss, evaluate, snapshot,
    load_model, save_atomic, predict, table_metrics,
)


def stamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def append_record(out,step,change,result):
    path=out/'实施记录.md'
    if not path.exists():
        path.write_text('# 联合训练实施记录\n\n|时间|实施内容|实际改动|结果与指标|\n|---|---|---|---|\n',encoding='utf-8')
    with path.open('a',encoding='utf-8') as f:
        f.write(f'|{stamp()}|{step}|{change}|{result}|\n')


def prepare_sources(root,out,config_path,config,splits):
    archive=out/'运行源码'
    archive.mkdir()
    paths=[Path(__file__),Path(__file__).with_name('joint_temperature_core.py'),
           Path(__file__).with_name('joint_temperature_figures.py'),config_path,
           root/'configs/splits.yaml',root/'configs/materials.yaml',
           root/'configs/geometry.yaml',root/'configs/boundary_conditions.yaml',root/'configs/data_metadata.yaml']
    hashes={}
    for i,path in enumerate(paths):
        target=archive/f'{i:02d}_{path.name}'
        shutil.copy2(path,target); hashes[str(path)]=sha256(path)
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): commit='未处于可查询的Git工作区'
    (out/'运行来源.json').write_text(json.dumps({'记录时间':stamp(),'源码提交':commit,
        '命令':sys.argv,'文件SHA256':hashes,'功率划分':splits,
        'Python':platform.python_version(),'PyTorch':torch.__version__,
        '说明':'原始数据和旧检查点不改写；新训练随机初始化，全部分支共同更新。'},ensure_ascii=False,indent=2),encoding='utf-8')
    import yaml
    (out/'实际配置.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')


def reduced_lf_validation(table,rng,per_group=1024):
    from joint_temperature_core import Table
    ids=np.concatenate([rng.choice(np.flatnonzero(table.group==gid),min(per_group,int(np.sum(table.group==gid))),replace=False)
                        for gid in np.unique(table.group)])
    return Table(table.x[ids],table.y[ids],table.weight[ids],table.group[ids]).validate()


def train(args):
    root=Path(args.root).resolve()
    config_path=Path(args.config)
    if not config_path.is_absolute(): config_path=root/config_path
    config=read_yaml(config_path)
    tc=config['training']
    if int(tc['epochs'])!=TOTAL_EPOCHS:
        raise ValueError('正式入口固定连续8000轮，不接受3000+2000+3000等分段训练。')
    if int(tc['ir_batch_size'])<1 or int(tc['validation_every'])<1:
        raise ValueError('批大小和验证间隔必须为正整数。')
    chosen=args.device or tc['device']
    chosen=('cuda' if torch.cuda.is_available() else 'cpu') if chosen=='auto' else chosen
    device=torch.device(chosen)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('当前环境没有可用CUDA；请在原GPU训练环境运行。')
    if int(__import__('os').environ.get('WORLD_SIZE','1'))!=1:
        raise RuntimeError('本入口使用单GPU，避免与旧分阶段分布式入口混用。')
    seed=int(tc['seed'])
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed)
    from joint_temperature_figures import setup_font
    setup_font()  # 开始长训练前检查中文图形环境。
    splits=fixed_splits(root)
    geometry=Geometry.from_project(root)
    physical=PhysicalSettings.from_project(root,config)
    # 训练前只读取训练、验证。测试温度留到完成8000轮以后。
    obs=load_observations(root,'train',splits,geometry)
    val=load_observations(root,'validation',splits,geometry)
    sim=load_simulation(root,splits['low']['training'])
    sim_val=reduced_lf_validation(load_simulation(root,splits['low']['validation']),np.random.default_rng(20260917))
    if args.resume:
        resume=Path(args.resume).resolve()
        model,state=load_model(resume,device)
        if state['config']!=config or state['splits']!=splits or state['physical_settings']!=__import__('dataclasses').asdict(physical):
            raise ValueError('恢复时配置、划分或物理条件发生变化；不能混接为同一次8000轮训练。')
        out=resume.parent
        epoch0=int(state['epoch']); history=state['history']; best=float(state['best_score'])
        rng.bit_generator.state=state['numpy_generator']
        torch.set_rng_state(state['torch_rng'].cpu())
        if torch.cuda.is_available() and state['cuda_rng']:
            torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        append_record(out,'恢复同一次训练',f'从已完成第{epoch0}轮继续','总目标仍为8000轮，不是新阶段')
    else:
        out=Path(args.output).resolve() if args.output else root/'研究记录'/('联合训练8000轮_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('输出目录不是空目录，禁止覆盖既有结果。')
        out.mkdir(parents=True,exist_ok=True)
        model=JointDeepONet(geometry,physical,config['model']).to(device)
        epoch0=0; history=[]; best=float('inf'); state=None
        prepare_sources(root,out,config_path,config,splits)
        append_record(out,'开始连续联合训练','实验水冷25.000℃；仿真水冷22.000℃；全部网络参数从第1轮共同训练',
                      '计划8000轮；无早停、无冻结阶段；尚无新训练结果')
    neural=[p for name,p in model.named_parameters() if name!='log_contact' and p.requires_grad]
    groups=[{'params':neural,'lr':float(tc['learning_rate'])}]
    if model.log_contact.requires_grad:
        groups.append({'params':[model.log_contact],'lr':float(tc['contact_learning_rate']),'weight_decay':0.0})
    optimizer=torch.optim.AdamW(groups,weight_decay=float(tc['weight_decay']))
    # 全程平滑变化的学习率，不改变数据来源、可训练参数或训练阶段。
    initial_lr=float(tc['learning_rate']); final_lr=float(tc['final_learning_rate'])
    ratio=final_lr/initial_lr
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda ep:ratio+(1-ratio)*.5*(1+math.cos(math.pi*min(ep,TOTAL_EPOCHS)/TOTAL_EPOCHS)))
    if state is not None:
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    pools={name:Pool(table,device) for name,table in obs.items()}
    sim_pool=Pool(sim,device)
    if epoch0==0:
        save_atomic(snapshot(model,optimizer,scheduler,0,history,config,splits,best,rng),out/'最近模型.pt')
    log=out/'训练记录.jsonl'
    if args.resume:
        # 检查点保存完整已提交历史；中断后未提交的日志保留备份。
        if log.exists(): shutil.copy2(log,out/('恢复前日志_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.jsonl'))
        log.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in history),encoding='utf-8')
    batch=int(tc['ir_batch_size']); weights=config['loss_weights']
    validation_every=int(tc['validation_every']); plots_every=int(tc['plots_every'])
    from joint_temperature_figures import training_plots, export_results
    payload=state
    for epoch in range(epoch0+1,TOTAL_EPOCHS+1):
        started=time.perf_counter(); model.train()
        order=rng.permutation(len(obs['顶部'].x)); totals={}; steps=0
        for begin in range(0,len(order),batch):
            optimizer.zero_grad(set_to_none=True)
            ir_ids=order[begin:begin+batch]
            parts={'顶部温度':observation_loss(model,pools['顶部'],ir_ids)}
            sensor_abs=[]; sensor_delta=[]
            for name in ('热端','冷端'):
                ids=pools[name].sample(tc['sensor_points_per_curve'],rng)
                absolute,delta=observation_loss(model,pools[name],ids,True)
                sensor_abs.append(absolute); sensor_delta.append(delta)
            parts['环温绝对值']=torch.stack(sensor_abs).mean()
            parts['环温温升']=torch.stack(sensor_delta).mean()
            si=sim_pool.sample(tc['simulation_points_per_power_material'],rng)
            sx,sy,sw=sim_pool.tensors(si)
            _,sg=np.unique(sim.group[si],return_inverse=True)
            parts['仿真温度']=grouped_mse((model(sx,fidelity='low')-sy)/model.temperature_scale,sw,torch.as_tensor(sg,device=device))
            generator=torch.Generator(device=device)
            generator.manual_seed(seed*10**9+epoch*100000+steps)
            parts.update(physics_loss(model,int(tc['physics_points_per_region']),generator))
            total=sum(float(weights[name])*value for name,value in parts.items())
            if not torch.isfinite(total):
                raise FloatingPointError(f'第{epoch}轮出现非有限损失；已保留上一完整轮检查点，不伪造完成。')
            total.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(tc['gradient_clip_norm']),error_if_nonfinite=True)
            optimizer.step()
            totals['总损失']=totals.get('总损失',0.0)+float(total.detach())
            for name,value in parts.items(): totals[name]=totals.get(name,0.)+float(value.detach())
            steps+=1
        scheduler.step()
        row={'轮次':epoch,'时间':stamp(),'耗时_秒':time.perf_counter()-started,
             '优化步数':steps,'学习率':float(optimizer.param_groups[0]['lr']),
             '接触热阻_m2K_W':float(model.contact_resistance.detach()),
             **{key:value/steps for key,value in totals.items()}}
        improved=False
        if epoch==1 or epoch%validation_every==0 or epoch==TOTAL_EPOCHS:
            metrics=evaluate(model,val)
            row['验证综合分数_℃']=metrics['综合选择分数_℃']
            for name,info in metrics['分项'].items():
                row[name+'验证均方根误差_℃']=info['汇总']['均方根误差_℃']
            lp=predict(model,sim_val.x,fidelity='low')
            row['仿真验证抽样均方根误差_℃']=float(np.sqrt(np.mean((lp-sim_val.y.reshape(-1))**2)))
            if row['验证综合分数_℃']<best:
                best=row['验证综合分数_℃']; improved=True
            (out/'最近验证指标.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
        history.append(row)
        with log.open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        payload=snapshot(model,optimizer,scheduler,epoch,history,config,splits,best,rng)
        # 每轮保存连续状态，不将一次训练人为拆成几个阶段。
        save_atomic(payload,out/'最近模型.pt')
        if improved:
            save_atomic(payload,out/'验证最佳模型.pt')
            append_record(out,f'完成第{epoch}轮','仅按验证集更新最佳模型',
                          f'综合分数={best:.3f}℃；顶部={row["顶部验证均方根误差_℃"]:.3f}℃；热端={row["热端验证均方根误差_℃"]:.3f}℃；冷端={row["冷端验证均方根误差_℃"]:.3f}℃')
        if epoch%plots_every==0 or epoch==TOTAL_EPOCHS:
            training_plots(history,out,config)
        if epoch==1 or epoch%10==0:
            print(f'第{epoch}/8000轮 | 总损失 {row["总损失"]:.3e} | 最近最佳验证 {best:.3f}℃',flush=True)
    save_atomic(payload,out/'第8000轮模型.pt')
    append_record(out,'完成8000轮','同一联合训练循环结束',f'已执行8000轮；验证最优分数={best:.3f}℃；随后才读取测试温度出图')
    frozen={'完成轮数':TOTAL_EPOCHS,'出图检查点':'验证最佳模型.pt','检查点SHA256':sha256(out/'验证最佳模型.pt'),
            '时间':stamp(),'选择依据':'仅训练期间的验证集；完整训练结束后锁定',
            '测试标签参与训练':False,'测试标签参与选模':False,'旧测试历史已查看':True}
    (out/'测试出图锁定记录.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    selected,_=load_model(out/'验证最佳模型.pt',device)
    if bool(config['visualization']['export_test_after_training']):
        export_results(selected,root,out,splits,config)
        append_record(out,'测试图册完成','表面并排图、温度曲线、三维动画、损失、接触热阻','详见结果总览.html；本次测试只展示结果，不用于再次调参')
    print(f'训练完成。图册：{out / "结果总览.html"}',flush=True)


def main():
    parser=argparse.ArgumentParser(description='连续8000轮联合训练及中文图册导出')
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--config',default='configs/联合训练8000轮.yaml')
    parser.add_argument('--output')
    parser.add_argument('--device')
    parser.add_argument('--resume',help='恢复本次训练的最近模型.pt，不新增训练阶段')
    args=parser.parse_args()
    train(args)


if __name__=='__main__':
    main()
