#!/usr/bin/env python3
"""仅对已完成的联合训练重建图册，不再训练或选择模型。"""
import argparse
from pathlib import Path
import torch
from joint_temperature_core import load_model
from joint_temperature_figures import training_plots,export_results,completed_gallery_lock

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='重新生成已完成联合训练的中文图册')
    parser.add_argument('--run',required=True)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args();out=Path(args.run).resolve()
    lock=completed_gallery_lock(out)
    checkpoint=out/lock['出图检查点']
    model,state=load_model(checkpoint,args.device)
    epochs=lock['完成轮数']
    if state['planned_epochs']!=epochs or state['config']['training']['epochs']!=epochs:
        raise RuntimeError('出图检查点的计划轮数与本次训练不一致。')
    final=torch.load(out/f'第{epochs}轮模型.pt',map_location='cpu',weights_only=False)
    if final['epoch']!=epochs or final['planned_epochs']!=epochs or final['config']!=state['config']:
        raise RuntimeError('最终模型不属于本次完整训练，不能重出图。')
    training_plots(final['history'],out,state['config'])
    fixed_gallery=export_results(model,Path(args.root).resolve(),out,state['splits'],state['config'])
    print(out/'结果总览.html')
    print(f'固定图册已更新：{fixed_gallery}')
