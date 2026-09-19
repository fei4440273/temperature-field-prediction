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
    initialize_low_network, low_parameters, set_low_trainable, power_smoothness_loss,
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
    selection_source=Path(__file__).with_name('joint_temperature_selection.py')
    paths.append(selection_source)
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
        '说明':'原始数据和旧检查点不改写；初始化及低保真稳定设置以实际配置和独立初始化记录为准。'},ensure_ascii=False,indent=2),encoding='utf-8')
    import yaml
    (out/'实际配置.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')


def reduced_lf_validation(table,rng,per_group=1024):
    from joint_temperature_core import Table
    ids=np.concatenate([rng.choice(np.flatnonzero(table.group==gid),min(per_group,int(np.sum(table.group==gid))),replace=False)
                        for gid in np.unique(table.group)])
    return Table(table.x[ids],table.y[ids],table.weight[ids],table.group[ids]).validate()


def training_limit(planned,max_epochs,validation_only):
    if max_epochs is None:
        return int(planned)
    if not validation_only:
        raise ValueError('缩短训练只允许用于仅训练/验证检查，不能冒充完整训练并读取测试。')
    if type(max_epochs) is not int or not 1<=max_epochs<=planned:
        raise ValueError('验证训练轮数必须是计划轮数以内的正整数。')
    return max_epochs


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
    validation_only=bool(getattr(args,'validation_only',False))
    limit=training_limit(TOTAL_EPOCHS,getattr(args,'max_epochs',None),validation_only)
    freeze_epochs=int(tc.get('low_freeze_epochs',0))
    if not 0<=freeze_epochs<TOTAL_EPOCHS:
        raise ValueError('低保真稳定轮数必须小于总训练轮数。')
    sensor_sampling=tc.get('sensor_sampling','sampled')
    if sensor_sampling not in ('sampled','full'):
        raise ValueError('热端、冷端只支持抽样或完整训练曲线。')
    selection_config=config.get('validation_selection')
    observation_weights=config.get('observation_weighting',{})
    separate_sensors=all(name in config['loss_weights'] for name in ('热端绝对温度','冷端绝对温度'))
    if any(name in config['loss_weights'] for name in ('热端绝对温度','冷端绝对温度')) and not separate_sensors:
        raise ValueError('热端、冷端绝对温度权重必须同时配置。')
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
    low_fidelity_mode=config.get('data',{}).get('low_fidelity_mode','legacy_split')
    splits=fixed_splits(root,low_fidelity_mode)
    geometry=Geometry.from_project(root)
    physical=PhysicalSettings.from_project(root,config)
    # 训练前只读取训练、验证。测试温度留到完成8000轮以后。
    obs=load_observations(root,'train',splits,geometry)
    val=load_observations(root,'validation',splits,geometry)
    sim=load_simulation(root,splits['low']['training'])
    sim_val=None
    if splits['low']['validation']:
        sim_val=reduced_lf_validation(load_simulation(root,splits['low']['validation']),np.random.default_rng(20260917))
    if args.resume:
        resume=Path(args.resume).resolve()
        model,state=load_model(resume,device)
        if state['config']!=config or state['splits']!=splits or state['physical_settings']!=__import__('dataclasses').asdict(physical):
            raise ValueError('恢复时配置、划分或物理条件发生变化；不能混接为同一次8000轮训练。')
        out=resume.parent
        epoch0=int(state['epoch']); history=state['history']; best=float(state['best_score'])
        if root not in out.parents or epoch0>limit:
            raise ValueError('恢复目录必须位于本项目内，已完成轮数不能大于本次训练上限。')
        rng.bit_generator.state=state['numpy_generator']
        torch.set_rng_state(state['torch_rng'].cpu())
        if torch.cuda.is_available() and state['cuda_rng']:
            torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        append_record(out,'恢复同一次训练',f'从已完成第{epoch0}轮继续','总目标仍为8000轮，不是新阶段')
    else:
        prefix='联合训练8000轮_低保真全量_' if low_fidelity_mode=='all_training' else '联合训练8000轮_'
        if config['model'].get('time_response', 'free') == 'monotone_heating':
            prefix += '持续升温_'
        out=Path(args.output).resolve() if args.output else root/'研究记录'/(prefix+datetime.now().strftime('%Y%m%d_%H%M%S'))
        if root not in out.parents:
            raise ValueError('训练输出必须位于本项目内，禁止写入项目外目录。')
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('输出目录不是空目录，禁止覆盖既有结果。')
        out.mkdir(parents=True,exist_ok=True)
        model=JointDeepONet(geometry,physical,config['model']).to(device)
        initialization=None
        if tc.get('initialize_low_from'):
            source=root/tc['initialize_low_from']
            initialization=initialize_low_network(model,source)
        epoch0=0; history=[]; best=float('inf'); state=None
        prepare_sources(root,out,config_path,config,splits)
        append_record(out,'开始训练','实验水冷25.000℃；仿真水冷22.000℃；原初温标签不改写',
                      f'计划8000轮；低保真稳定轮数={freeze_epochs}；尚无本次训练精度结果')
        append_record(out,'记录本次数据分配',f'低保真模式={low_fidelity_mode}；高保真训练/验证/测试=12/3/3',
                      f'低保真训练/验证/测试={len(splits["low"]["training"])}/{len(splits["low"]["validation"])}/{len(splits["low"]["test"])}；训练池={len(sim.x)}条；仅按高保真验证选模')
        if model.time_response == 'monotone_heating':
            append_record(out,'启用持续升温响应','低保真与高保真输出均使用非负升温系数；初温仍可拟合，原初始与水冷软约束保留',
                          '新结构不能接续旧v1权重；不用于热源关断后的冷却')
        if initialization:
            (out/'低保真初始化记录.json').write_text(json.dumps(initialization,ensure_ascii=False,indent=2),encoding='utf-8')
            append_record(out,'借用已有低保真权重','只迁移低保真参数，不继承原实验校正、热阻和优化器',
                          f'来源第{initialization["来源轮次"]}轮；低保真先稳定{freeze_epochs}轮，再按较小学习率共同更新')
        if model.correction_power_degree is not None:
            append_record(out,'启用误差改进配置','低阶功率校正；热端、冷端绝对温度分别拟合；完整传感器曲线',
                          f'检查点v3_smooth_power；功率校正次数={model.correction_power_degree}；选模设置={selection_config}')
        if validation_only:
            append_record(out,'仅训练/验证检查',f'本次最多执行{limit}轮，不读取测试标签，不替换正式图册',
                          '这不是已完成8000轮的新结果')
    if 'low_learning_rate' in tc:
        groups=[{'params':low_parameters(model),'lr':float(tc['low_learning_rate'])},
                {'params':list(model.correction.parameters()),'lr':float(tc['learning_rate'])}]
    else:
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
    for epoch in range(epoch0+1,limit+1):
        started=time.perf_counter(); model.train()
        set_low_trainable(model,epoch>freeze_epochs)
        order=rng.permutation(len(obs['顶部'].x)); totals={}; steps=0
        for begin in range(0,len(order),batch):
            optimizer.zero_grad(set_to_none=True)
            ir_ids=order[begin:begin+batch]
            parts={'顶部温度':observation_loss(model,pools['顶部'],ir_ids,
                    worst_fraction=float(observation_weights.get('top_worst_fraction',0.)),
                    power_weight=float(observation_weights.get('top_power_weight',0.)),
                    edge_weight=float(observation_weights.get('top_edge_weight',0.)))}
            sensor_abs=[]; sensor_delta=[]
            for name in ('热端','冷端'):
                ids=(np.arange(len(pools[name].table.x)) if sensor_sampling=='full'
                     else pools[name].sample(tc['sensor_points_per_curve'],rng))
                absolute,delta=observation_loss(model,pools[name],ids,True,
                    worst_fraction=float(observation_weights.get('sensor_worst_fraction',0.)))
                sensor_abs.append(absolute); sensor_delta.append(delta)
                if separate_sensors:
                    parts[name+'绝对温度']=absolute
            if not separate_sensors:
                parts['环温绝对值']=torch.stack(sensor_abs).mean()
            parts['环温温升']=torch.stack(sensor_delta).mean()
            si=sim_pool.sample(tc['simulation_points_per_power_material'],rng)
            sx,sy,sw=sim_pool.tensors(si)
            _,sg=np.unique(sim.group[si],return_inverse=True)
            parts['仿真温度']=grouped_mse((model(sx,fidelity='low')-sy)/model.temperature_scale,sw,torch.as_tensor(sg,device=device))
            generator=torch.Generator(device=device)
            generator.manual_seed(seed*10**9+epoch*100000+steps)
            parts.update(physics_loss(model,int(tc['physics_points_per_region']),generator))
            if '功率平滑' in weights:
                parts['功率平滑']=power_smoothness_loss(model,int(tc.get('power_smoothness_points',64)),
                                                      generator,float(tc.get('power_smoothness_step_w',40.)))
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
        correction_group=1 if 'low_learning_rate' in tc else 0
        row={'轮次':epoch,'时间':stamp(),'耗时_秒':time.perf_counter()-started,
             '优化步数':steps,'学习率':float(optimizer.param_groups[correction_group]['lr']),
             '接触热阻_m2K_W':float(model.contact_resistance.detach()),
             **{key:value/steps for key,value in totals.items()}}
        improved=False
        if model.correction_power_degree is not None:
            row['低保真参数参与更新']=epoch>freeze_epochs
            row['低保真学习率']=float(optimizer.param_groups[0]['lr'])
            row['传感器训练方式']=sensor_sampling
        if epoch==1 or epoch%validation_every==0 or epoch==limit:
            metrics=evaluate(model,val,selection_config)
            row['验证综合分数_℃']=metrics['综合选择分数_℃']
            for name,info in metrics['分项'].items():
                row[name+'验证均方根误差_℃']=info['汇总']['均方根误差_℃']
            if sim_val is not None:
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
            (out/'验证最佳指标.json').write_text(json.dumps({'验证最佳轮次':epoch,**metrics},
                ensure_ascii=False,indent=2),encoding='utf-8')
            append_record(out,f'完成第{epoch}轮','仅按验证集更新最佳模型',
                          f'综合分数={best:.3f}℃；顶部={row["顶部验证均方根误差_℃"]:.3f}℃；热端={row["热端验证均方根误差_℃"]:.3f}℃；冷端={row["冷端验证均方根误差_℃"]:.3f}℃')
        if epoch%plots_every==0 or epoch==limit:
            training_plots(history,out,config)
        if epoch==1 or epoch%10==0:
            print(f'第{epoch}/8000轮 | 总损失 {row["总损失"]:.3e} | 最近最佳验证 {best:.3f}℃',flush=True)
    if validation_only:
        selected,selected_state=load_model(out/'验证最佳模型.pt',device)
        final_metrics={'训练':evaluate(selected,obs,selection_config),
                       '验证':evaluate(selected,val,selection_config)}
        (out/'最佳模型训练验证指标.json').write_text(json.dumps(final_metrics,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'验证最佳指标.json').write_text(json.dumps({'验证最佳轮次':selected_state['epoch'],
            **final_metrics['验证']},ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'验证训练完成记录.json').write_text(json.dumps({'计划轮数':TOTAL_EPOCHS,'实际完成轮数':limit,
            '验证最佳轮次':selected_state['epoch'],'验证最佳检查点SHA256':sha256(out/'验证最佳模型.pt'),
            '测试标签参与训练':False,'测试标签参与选模':False,'测试数据已读取':False,
            '固定图册已更新':False,'结束时间':stamp()},ensure_ascii=False,indent=2),encoding='utf-8')
        append_record(out,'训练/验证检查结束',f'实际执行{limit}轮；验证最佳在第{selected_state["epoch"]}轮',
                      '指标已保存；未读取测试、未伪造8000轮完成、未更新固定图册')
        print(f'训练/验证检查结束，未读取测试：{out}',flush=True)
        return
    save_atomic(payload,out/'第8000轮模型.pt')
    append_record(out,'完成8000轮','同一联合训练循环结束',f'已执行8000轮；验证最优分数={best:.3f}℃；随后才读取测试温度出图')
    frozen={'完成轮数':TOTAL_EPOCHS,'出图检查点':'验证最佳模型.pt','检查点SHA256':sha256(out/'验证最佳模型.pt'),
            '时间':stamp(),'选择依据':'仅训练期间的验证集；完整训练结束后锁定',
            '测试标签参与训练':False,'测试标签参与选模':False,'旧测试历史已查看':True}
    (out/'测试出图锁定记录.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    selected,selected_state=load_model(out/'验证最佳模型.pt',device)
    selected_validation=evaluate(selected,val,selection_config)
    (out/'验证最佳指标.json').write_text(json.dumps({'验证最佳轮次':selected_state['epoch'],
        **selected_validation},ensure_ascii=False,indent=2),encoding='utf-8')
    if bool(config['visualization']['export_test_after_training']):
        fixed_gallery=export_results(selected,root,out,splits,config)
        append_record(out,'测试图册完成','表面并排图、温度曲线、三维动画、损失、接触热阻',
                      f'本次结果总览.html已保存；固定入口已更新：{fixed_gallery}；本次测试只展示结果，不用于再次调参')
        print(f'固定图册已更新：{fixed_gallery}',flush=True)
    print(f'训练完成。图册：{out / "结果总览.html"}',flush=True)


def main():
    parser=argparse.ArgumentParser(description='连续8000轮联合训练及中文图册导出')
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--config',default='configs/联合训练误差改进.yaml')
    parser.add_argument('--output')
    parser.add_argument('--device')
    parser.add_argument('--resume',help='恢复本次训练的最近模型.pt，不新增训练阶段')
    parser.add_argument('--validation-only',action='store_true',help='只检查训练与验证，不读取测试、不更新正式图册')
    parser.add_argument('--max-epochs',type=int,help='仅训练/验证检查的轮数上限；正式训练仍需8000轮')
    args=parser.parse_args()
    train(args)


if __name__=='__main__':
    main()
