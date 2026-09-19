#!/usr/bin/env python3
"""仅对本入口已经完成的8000轮训练重建图册，不再训练或选择模型。"""
import argparse
import json
from pathlib import Path
import torch
from joint_temperature_core import load_model,sha256
from joint_temperature_figures import training_plots,export_results

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='重新生成已完成联合训练的中文图册')
    parser.add_argument('--run',required=True)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args();out=Path(args.run).resolve()
    lock=json.loads((out/'测试出图锁定记录.json').read_text(encoding='utf-8'))
    checkpoint=out/lock['出图检查点']
    if lock['完成轮数']!=8000 or sha256(checkpoint)!=lock['检查点SHA256']:
        raise RuntimeError('训练未完成或出图模型发生变化，不能静默重新选模型。')
    model,state=load_model(checkpoint,args.device)
    final=torch.load(out/'第8000轮模型.pt',map_location='cpu',weights_only=False)
    training_plots(final['history'],out,state['config'])
    fixed_gallery=export_results(model,Path(args.root).resolve(),out,state['splits'],state['config'])
    print(out/'结果总览.html')
    print(f'固定图册已更新：{fixed_gallery}')
