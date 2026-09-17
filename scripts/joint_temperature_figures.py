"""从模型和实测数值生成中文图册，不生成示意温度或虚构实验曲线。"""
from __future__ import annotations
import csv
import html
import json
import math
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from PIL import Image
from joint_temperature_core import KELVIN, predict, load_observations, table_metrics, read_parquet


def setup_font():
    preferred=['Noto Sans CJK SC','Noto Sans CJK JP','Source Han Sans SC','WenQuanYi Micro Hei','Microsoft YaHei','SimHei','Arial Unicode MS']
    available={f.name for f in font_manager.fontManager.ttflist}
    chosen=next((name for name in preferred if name in available),None)
    if not chosen:
        raise RuntimeError('未找到中文字体。Ubuntu可安装fonts-noto-cjk后运行；程序不生成乱码图。')
    plt.rcParams['font.sans-serif']=[chosen,'DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    return chosen


def save_curve(path:Path,x,series,title,xlabel,ylabel,log=False):
    fig,ax=plt.subplots(figsize=(8,4.7))
    for label,y in series:
        ax.plot(x,np.asarray(y),label=label,linewidth=1.6)
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
    if log: ax.set_yscale('log')
    else: ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.grid(True,alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(path,dpi=140); plt.close(fig)


def training_plots(history:list,out:Path,config:dict):
    setup_font(); folder=out/'训练曲线'; folder.mkdir(exist_ok=True)
    epochs=np.asarray([row['轮次'] for row in history])
    names=['总损失','仿真温度','顶部温度','环温绝对值','环温温升','传热方程','边界条件','材料界面']
    series=[(name,np.maximum([row.get(name,np.nan) for row in history],1e-16)) for name in names]
    save_curve(folder/'损失函数变化.png',epochs,series,'联合训练损失变化','训练轮次','无量纲损失（对数）',True)
    rows=[r for r in history if '验证综合分数_℃' in r]
    if rows:
        save_curve(folder/'验证温度误差.png',[r['轮次'] for r in rows],
                   [(name,[r[name] for r in rows]) for name in ('验证综合分数_℃','顶部验证均方根误差_℃','热端验证均方根误差_℃','冷端验证均方根误差_℃')],
                   '验证集预测误差','训练轮次','误差 / ℃')
    rc=np.asarray([row['接触热阻_m2K_W'] for row in history])*1e5
    save_curve(folder/'界面接触热阻变化.png',epochs,[('拟合接触热阻',rc)],
               '界面接触热阻随训练轮次的变化','训练轮次','接触热阻 / (10⁻⁵ m²·K/W)')


def save_comparison_curve(path:Path,x,measured,predicted,title,xlabel):
    fig,ax=plt.subplots(figsize=(8,4.7))
    ax.plot(x,measured,'o',label='实验值',markersize=3)
    ax.plot(x,predicted,'-',label='预测值',linewidth=1.7)
    rmse=float(np.sqrt(np.mean((np.asarray(predicted)-measured)**2)))
    ax.set(title=f'{title}\n均方根误差 {rmse:.3f} ℃',xlabel=xlabel,ylabel='温度 / ℃')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.grid(True,alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)


def save_map(path:Path,values,extent,title,low,high,dpi):
    fig,ax=plt.subplots(figsize=(5.2,4.6))
    image=ax.imshow(values,origin='lower',extent=extent,vmin=low,vmax=high)
    ax.set(title=title,xlabel='横向位置 / mm',ylabel='纵向位置 / mm',aspect='equal')
    bar=fig.colorbar(image,ax=ax,pad=.025); bar.set_label('温度 / ℃' if '误差' not in title else '预测−实验 / ℃')
    bar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig)


def join_images(paths,destination):
    images=[Image.open(path).convert('RGB') for path in paths]
    canvas=Image.new('RGB',(sum(im.width for im in images),max(im.height for im in images)),(255,255,255))
    pos=0
    for im in images:
        canvas.paste(im,(pos,0)); pos+=im.width; im.close()
    canvas.save(destination)


def raw_surface_frames(root:Path,config:dict,power:float):
    directory=root/config['visualization']['raw_test_surface']
    pattern=re.compile(r'(\d+(?:\.\d+)?)W-(\d+(?:\.\d+)?)s_50mm_temperature\.csv$',re.IGNORECASE)
    matches=[]
    if directory.exists():
        for path in directory.rglob('*.csv'):
            match=pattern.search(path.name)
            if match and abs(float(match[1])-power)<1e-3:
                matches.append((float(match[2]),path))
    return sorted(matches)


def surface_images(model,root,out,power,table,config):
    folder=out/f'{power:.3f}W'/'表面逐时刻'; folder.mkdir(parents=True,exist_ok=True)
    mask=np.abs(table.x[:,3]-power)<1e-3
    x=table.x[mask]; measured=table.y[mask,0]-KELVIN
    times=np.unique(x[:,2]); dpi=int(config['visualization']['image_dpi'])
    raw=dict(raw_surface_frames(root,config,power)); frames=[]
    for t in times:
        rows=x[:,2]==t; radius=x[rows,0]; reference=measured[rows]
        pred_curve=predict(model,x[rows])-KELVIN
        save_comparison_curve(folder/f'{float(t):08.3f}秒_径向对比.png',radius*1000,reference,pred_curve,
                              f'{power:.3f} W，{float(t):.3f} s，碳化硅表面','半径 / mm')
        entry=next((path for rt,path in raw.items() if abs(rt-float(t))<1e-4),None)
        if entry is not None:
            import polars as pl
            f=pl.read_csv(entry,columns=['x_mm','y_mm','r_mm','temperature_c'])
            px=f['x_mm'].to_numpy(); py=f['y_mm'].to_numpy(); rr=f['r_mm'].to_numpy()/1000
            query=np.column_stack((rr,np.zeros(len(rr)),np.full(len(rr),t),np.full(len(rr),power),np.ones(len(rr))))
            truth=f['temperature_c'].to_numpy(); estimate=predict(model,query)-KELVIN
            ux,ix=np.unique(px,return_inverse=True); uy,iy=np.unique(py,return_inverse=True)
            if len(ux)*len(uy)>2_000_000:
                raise ValueError('原始表面不是支持的规则坐标图，停止而不虚构像素。')
            a=np.full((len(uy),len(ux)),np.nan); b=a.copy()
            a[iy,ix]=truth; b[iy,ix]=estimate
            extent=[float(ux.min()),float(ux.max()),float(uy.min()),float(uy.max())]
            source='实验温度'
        else:
            line=np.linspace(-model.geometry.radius_sic,model.geometry.radius_sic,161)
            xx,yy=np.meshgrid(line,line); rr=np.hypot(xx,yy); inside=rr<=model.geometry.radius_sic
            order=np.argsort(radius); a=np.full_like(rr,np.nan); b=a.copy()
            a[inside]=np.interp(rr[inside],radius[order],reference[order])
            query=np.column_stack((rr[inside],np.zeros(inside.sum()),np.full(inside.sum(),t),np.full(inside.sum(),power),np.ones(inside.sum())))
            b[inside]=predict(model,query)-KELVIN
            extent=[line.min()*1000,line.max()*1000,line.min()*1000,line.max()*1000]
            source='实验环平均数据还原'
        frames.append((float(t),a,b,extent,source))
    if not frames: raise ValueError('测试表面没有实测时刻。')
    lower=min(float(np.nanmin(a)) for _,a,b,_,_ in frames)
    lower=min(lower,min(float(np.nanmin(b)) for _,a,b,_,_ in frames))
    upper=max(max(float(np.nanmax(a)),float(np.nanmax(b))) for _,a,b,_,_ in frames)
    if upper<=lower: upper=lower+1.
    error_range=max(float(np.nanmax(np.abs(b-a))) for _,a,b,_,_ in frames)
    error_range=max(error_range,.001)
    boards=[]
    for t,a,b,extent,source in frames:
        prefix=folder/f'{t:08.3f}秒'
        paths=[Path(str(prefix)+suffix) for suffix in ('_实验.png','_预测.png','_误差.png')]
        save_map(paths[0],a,extent,f'{source}\n{power:.3f} W，{t:.3f} s',lower,upper,dpi)
        save_map(paths[1],b,extent,f'预测温度\n{power:.3f} W，{t:.3f} s',lower,upper,dpi)
        save_map(paths[2],b-a,extent,f'温度误差\n{power:.3f} W，{t:.3f} s',-error_range,error_range,dpi)
        board=Path(str(prefix)+'_并排对比.png'); join_images(paths,board); boards.append(board)
    imgs=[Image.open(p).convert('P',palette=Image.Palette.ADAPTIVE) for p in boards]
    imgs[0].save(folder.parent/'表面实验与预测对比.gif',save_all=True,append_images=imgs[1:],duration=500,loop=0)
    for im in imgs: im.close()


def ring_images(model,out,power,name,table):
    mask=np.abs(table.x[:,3]-power)<1e-3
    ids=np.flatnonzero(mask); ids=ids[np.argsort(table.x[ids,2])]
    x=table.x[ids]; measured=table.y[ids,0]-KELVIN; estimate=predict(model,x)-KELVIN
    folder=out/f'{power:.3f}W'; folder.mkdir(exist_ok=True)
    save_comparison_curve(folder/f'{name}温度对比.png',x[:,2],measured,estimate,f'{power:.3f} W，{name}','时间 / s')
    save_curve(folder/f'{name}误差随时间.png',x[:,2],[('预测−实验',estimate-measured)],f'{power:.3f} W，{name}误差','时间 / s','误差 / ℃')
    with (folder/f'{name}逐时刻数值.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['时间_s','实验温度_℃','预测温度_℃','误差_℃'])
        for t,y,p in zip(x[:,2],measured,estimate):writer.writerow([f'{float(v):.3f}' for v in (t,y,p,p-y)])


def three_dimensional_animation(model,out,power,config,fidelity='high'):
    g=model.geometry; dt=float(config['visualization']['animation_time_step_s'])
    times=np.arange(0,g.time_max+dt/2,dt)
    theta=np.linspace(0,np.pi,49); zz=np.linspace(g.bottom_cu,0,35)
    ang,z=np.meshgrid(theta,zz); r=np.full_like(z,g.radius_cu)
    surfaces=[(r*np.cos(ang),r*np.sin(ang),z,r,np.zeros_like(r))]
    for rlo,rhi,material in ((0.,g.radius_sic,1.),(g.radius_sic,g.radius_cu,0.)):
        ang,r=np.meshgrid(theta,np.linspace(rlo,rhi,35));z=np.zeros_like(r)
        surfaces.append((r*np.cos(ang),r*np.sin(ang),z,r,np.full_like(r,material)))
    sx,z=np.meshgrid(np.linspace(-g.radius_cu,g.radius_cu,91),zz)
    r=np.abs(sx); m=((r<=g.radius_sic)&(z>=g.bottom_sic)).astype(float)
    surfaces.append((sx,np.zeros_like(sx),z,r,m))
    predictions=[]
    for t in times:
        vals=[]
        for x,y,z,r,m in surfaces:
            query=np.column_stack((r.ravel(),z.ravel(),np.full(r.size,t),np.full(r.size,power),m.ravel()))
            vals.append((predict(model,query,fidelity=fidelity)-KELVIN).reshape(r.shape))
        predictions.append(vals)
    low=min(float(a.min()) for v in predictions for a in v);high=max(float(a.max()) for v in predictions for a in v)
    normal=Normalize(low,max(high,low+.001)); cmap=plt.get_cmap()
    folder=out/f'{power:.3f}W'/'三维逐时刻'; folder.mkdir(parents=True,exist_ok=True)
    mode='实验校正预测' if fidelity=='high' else '仿真分支预测'
    cut=surfaces[-1]
    np.savez_compressed(folder.parent/f'轴对称体场_{mode}.npz',
        x_m=cut[0], z_m=cut[2], material_id=cut[4], time_s=times,
        temperature_c=np.stack([values[-1] for values in predictions]))
    frames=[]
    for t,values in zip(times,predictions):
        fig=plt.figure(figsize=(8,5.8));ax=fig.add_subplot(111,projection='3d')
        for (x,y,z,r,m),v in zip(surfaces,values):
            ax.plot_surface(x*1000,y*1000,z*1000,facecolors=cmap(normal(v)),rstride=1,cstride=1,shade=False,linewidth=0)
        ax.set(xlabel='横向 / mm',ylabel='纵向 / mm',zlabel='高度 / mm',
               title=f'{power:.3f} W，{float(t):.3f} s\nSiC–Cu三维温度场：{mode}（半剖显示）')
        ax.set_box_aspect((2,1,.45));ax.view_init(elev=24,azim=-60)
        bar=fig.colorbar(plt.cm.ScalarMappable(norm=normal,cmap=cmap),ax=ax,shrink=.65,pad=.10)
        bar.set_label('温度 / ℃');bar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        fig.tight_layout();path=folder/f'{mode}_{float(t):08.3f}秒.png';fig.savefig(path,dpi=120);plt.close(fig)
        frames.append(Image.open(path).convert('P',palette=Image.Palette.ADAPTIVE))
    fps=int(config['visualization']['animation_fps'])
    frames[0].save(folder.parent/f'整体三维温度_{mode}.gif',save_all=True,append_images=frames[1:],duration=int(1000/fps),loop=0)
    for frame in frames:frame.close()


def write_gallery(out:Path,metrics:dict):
    pieces=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>温度预测结果总览</title>',
            '<style>body{font-family:sans-serif;max-width:1500px;margin:auto;padding:24px}img{max-width:100%;height:auto}section{margin:30px 0}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}</style>',
            '<h1>多保真DeepONet温度预测结果</h1><p>连续8000轮训练后，依据验证集确定模型，再对测试集生成以下结果。三维图为模型预测；不是虚构的内部实测。</p>',
            '<p>表面实验与预测采用相同温度色标；三维动画全时段采用固定色标。表面原始像素不可读取时，图题明确标注“实验环平均数据还原”。</p>',
            '<h2>测试集分项误差</h2><table><tr><th>功率/W</th><th>位置</th><th>均方根误差/℃</th><th>平均绝对误差/℃</th></tr>']
    for name,info in metrics.items():
        for row in info['逐功率']:
            pieces.append(f'<tr><td>{row["功率_W"]:.3f}</td><td>{name}</td><td>{row["均方根误差_℃"]:.3f}</td><td>{row["平均绝对误差_℃"]:.3f}</td></tr>')
    pieces.append('</table>')
    for path in sorted((out/'训练曲线').glob('*.png')):
        rel=path.relative_to(out).as_posix();pieces.append(f'<section><h2>{html.escape(path.stem)}</h2><img loading="lazy" src="{html.escape(rel)}"></section>')
    for folder in sorted(out.glob('*W')):
        if not folder.is_dir():continue
        pieces.append(f'<h2>{html.escape(folder.name)}</h2>')
        for path in sorted(folder.glob('*.png'))+sorted(folder.glob('*.gif')):
            rel=path.relative_to(out).as_posix();pieces.append(f'<section><h3>{html.escape(path.stem)}</h3><img loading="lazy" src="{html.escape(rel)}"></section>')
        pieces.append('<details><summary>展开全部实测时刻的表面对比图</summary>')
        for path in sorted((folder/'表面逐时刻').glob('*_并排对比.png')):
            rel=path.relative_to(out).as_posix();pieces.append(f'<h3>{html.escape(path.stem)}</h3><img loading="lazy" src="{html.escape(rel)}">')
        pieces.append('</details>')
    pieces.append('</html>');(out/'结果总览.html').write_text('\n'.join(pieces),encoding='utf-8')


def export_results(model,root:Path,out:Path,splits:dict,config:dict):
    setup_font()
    if not (out/'测试出图锁定记录.json').exists():
        raise RuntimeError('必须先完成8000轮并锁定检查点，再读取测试温度出图。')
    lock=json.loads((out/'测试出图锁定记录.json').read_text(encoding='utf-8'))
    if lock['完成轮数']!=8000:raise RuntimeError('完整训练尚未结束。')
    tables=load_observations(root,'test',splits,model.geometry)
    metrics={name:table_metrics(table,predict(model,table.x)) for name,table in tables.items()}
    (out/'测试指标_原始精度.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    for power in splits['high']['test']:
        surface_images(model,root,out,power,tables['顶部'],config)
        for name in ('热端','冷端'):ring_images(model,out,power,name,tables[name])
        three_dimensional_animation(model,out,power,config,'high')
        three_dimensional_animation(model,out,power,config,'low')
    write_gallery(out,metrics)
