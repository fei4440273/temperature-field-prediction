"""从模型和实测数值生成中文图册，不生成示意温度或虚构实验曲线。"""
from __future__ import annotations
import base64
import csv
import html
import json
import math
import mimetypes
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from PIL import Image
from joint_temperature_core import KELVIN, predict, load_observations, table_metrics, read_parquet, sha256

FIXED_GALLERY = Path('研究记录/联合训练8000轮_20260917_114704/结果总览.html')


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
    names=['总损失','仿真温度','顶部温度','环温绝对值','热端绝对温度','冷端绝对温度',
           '环温温升','功率平滑','传热方程','边界条件','材料界面']
    names=[name for name in names if any(name in row for row in history)]
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


def surface_error_rows(frames):
    rows=[]
    for t,measured,predicted,_,source in sorted(frames,key=lambda frame:frame[0]):
        measured=np.asarray(measured,dtype=np.float64)
        predicted=np.asarray(predicted,dtype=np.float64)
        if not np.isfinite(t) or measured.shape!=predicted.shape:
            raise ValueError('顶面误差数据的时间或实验与预测网格不一致。')
        valid=np.isfinite(measured)&np.isfinite(predicted)
        if not valid.any():
            raise ValueError(f'顶面{float(t):.3f}秒没有可比较的有效点。')
        error=predicted[valid]-measured[valid]
        rows.append({'时间_s':float(t),'有效点数':int(valid.sum()),
                     '均方根误差_℃':float(np.sqrt(np.mean(error**2))),
                     '平均绝对误差_℃':float(np.mean(np.abs(error))),
                     '平均偏差_℃':float(np.mean(error)),
                     '最大绝对误差_℃':float(np.max(np.abs(error))),'数据来源':source})
    if not rows:
        raise ValueError('顶面没有实测时刻，不能生成误差曲线。')
    return rows


def save_surface_error_curve(out:Path,power:float,frames):
    rows=surface_error_rows(frames)
    setup_font()
    folder=out/f'{power:.3f}W';folder.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,5.2))
    series=(('均方根误差_℃','#0072B2','-','o'),
            ('平均绝对误差_℃','#D55E00','--','s'),('平均偏差_℃','#009E73','-.','^'))
    for name,color,style,marker in series:
        ax.plot([row['时间_s'] for row in rows],[row[name] for row in rows],
                label=name.removesuffix('_℃'),color=color,linestyle=style,
                marker=marker,markersize=3,linewidth=1.6)
    ax.axhline(0.,color='#666666',linestyle=':',linewidth=.9)
    ax.set(title=f'{power:.3f} W 顶面温度误差随时间变化',xlabel='时间 / s',ylabel='误差 / ℃')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.grid(True,alpha=.25);ax.legend(fontsize=10)
    sources={row['数据来源'] for row in rows}
    scope='顶面原始有效像素' if sources=={'实验温度'} else (
        '实验环平均数据还原的有效网格点' if sources=={'实验环平均数据还原'} else '各时刻云图的有效网格点')
    fig.text(.5,.07,f'统计范围：{scope}',ha='center',fontsize=10)
    fig.text(.5,.025,'平均偏差 = 预测 - 实验；负值表示预测偏低',ha='center',fontsize=10)
    fig.tight_layout(rect=(0.,.12,1.,1.))
    fig.savefig(folder/'顶面温度误差随时间.png',dpi=300)
    fig.savefig(folder/'顶面温度误差随时间.pdf')
    plt.close(fig)
    with (folder/'顶面温度误差逐时刻.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    return rows


def surface_images(model,root,out,power,table,config,*,error_curve_only:bool=False):
    folder=out/f'{power:.3f}W'/'表面逐时刻'; folder.mkdir(parents=True,exist_ok=True)
    mask=np.abs(table.x[:,3]-power)<1e-3
    x=table.x[mask]; measured=table.y[mask,0]-KELVIN
    times=np.unique(x[:,2]); dpi=int(config['visualization']['image_dpi'])
    raw=dict(raw_surface_frames(root,config,power)); frames=[]
    for t in times:
        rows=x[:,2]==t; radius=x[rows,0]; reference=measured[rows]
        if not error_curve_only:
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
    error_rows=save_surface_error_curve(out,power,frames)
    if error_curve_only:
        return error_rows
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


def gallery_html(out:Path,metrics:dict,destination:Path,publication:dict|None=None,*,embed_images:bool=False):
    def asset_url(path):
        relative = Path(os.path.relpath(path, destination.parent)).as_posix()
        return html.escape(quote(relative, safe='/'), quote=True)
    def image_url(path):
        if not embed_images:
            return asset_url(path)
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in ('image/png', 'image/gif'):
            raise ValueError(f'不支持内嵌此图像格式：{path.name}')
        encoded = base64.b64encode(path.read_bytes()).decode('ascii')
        return f'data:{mime};base64,{encoded}'
    pieces=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>温度预测结果总览</title>',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<style>body{font-family:sans-serif;max-width:1500px;margin:auto;padding:24px}p{overflow-wrap:anywhere}img{max-width:100%;height:auto}section{margin:30px 0}table{border-collapse:collapse;width:100%;max-width:760px;table-layout:fixed;font-size:14px}td,th{padding:6px;border:1px solid #aaa}</style>',
            '<h1>多保真DeepONet温度预测结果</h1><p>连续8000轮训练后，依据验证集确定模型，再对测试集生成以下结果。三维图为模型预测；不是虚构的内部实测。</p>',
            '<p>表面实验与预测采用相同温度色标；三维动画全时段采用固定色标。表面原始像素不可读取时，图题明确标注“实验环平均数据还原”。</p>']
    if publication is not None:
        source = out / publication['来源图册文件名']
        pieces.extend([
            f'<p>来源运行：<a href="{asset_url(source)}">{html.escape(out.name)}</a></p>',
            f'<p>完成轮数：{publication["完成轮数"]}；展示检查点：{html.escape(publication["出图检查点"])}；'
            f'训练完成时间：{html.escape(publication["训练完成记录时间"])}；更新于：{html.escape(publication["更新时刻"])}</p>',
        ])
    pieces.append('<h2>测试集分项误差</h2><table><tr><th>功率/W</th><th>位置</th><th>均方根误差/℃</th><th>平均绝对误差/℃</th></tr>')
    for name,info in metrics.items():
        for row in info['逐功率']:
            pieces.append(f'<tr><td>{row["功率_W"]:.3f}</td><td>{name}</td><td>{row["均方根误差_℃"]:.3f}</td><td>{row["平均绝对误差_℃"]:.3f}</td></tr>')
    pieces.append('</table>')
    top_curves=sorted(out.glob('*W/顶面温度误差随时间.png'))
    if top_curves:
        pieces.append('<h2>顶面温度误差随时间变化</h2>')
        pieces.append('<p>曲线按每个实测时刻的顶面云图有效点计算；上方表格按径向环平均温度及其数据权重计算，统计方式不同。</p>')
        for path in top_curves:
            pieces.append(f'<section><h3>{html.escape(path.parent.name)}</h3><img loading="lazy" src="{image_url(path)}"></section>')
    for path in sorted((out/'训练曲线').glob('*.png')):
        pieces.append(f'<section><h2>{html.escape(path.stem)}</h2><img loading="lazy" src="{image_url(path)}"></section>')
    for folder in sorted(out.glob('*W')):
        if not folder.is_dir():continue
        pieces.append(f'<h2>{html.escape(folder.name)}</h2>')
        for path in sorted(folder.glob('*.png'))+sorted(folder.glob('*.gif')):
            if path.name=='顶面温度误差随时间.png':continue
            pieces.append(f'<section><h3>{html.escape(path.stem)}</h3><img loading="lazy" src="{image_url(path)}"></section>')
        pieces.append('<details><summary>展开全部实测时刻的表面对比图</summary>')
        for path in sorted((folder/'表面逐时刻').glob('*_并排对比.png')):
            pieces.append(f'<h3>{html.escape(path.stem)}</h3><img loading="lazy" src="{image_url(path)}">')
        pieces.append('</details>')
    pieces.append('</html>')
    return '\n'.join(pieces)


def write_text_atomic(path:Path,text:str):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                     prefix='临时图册发布_',delete=False) as handle:
        temporary=Path(handle.name)
        handle.write(text)
    try:
        temporary.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o644)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_gallery(out:Path,metrics:dict,*,destination:Path|None=None,publication:dict|None=None):
    destination=out/'结果总览.html' if destination is None else destination
    write_text_atomic(destination,gallery_html(out,metrics,destination,publication))
    return destination


def completed_gallery_lock(out:Path):
    out=out.resolve()
    lock_path=out/'测试出图锁定记录.json'
    if not lock_path.is_file():
        raise RuntimeError('缺少测试出图锁定记录，不能更新固定图册。')
    lock=json.loads(lock_path.read_text(encoding='utf-8'))
    if lock.get('完成轮数')!=8000:
        raise RuntimeError('完整8000轮训练尚未结束，不能更新固定图册。')
    checkpoint=(out/lock['出图检查点']).resolve()
    if out not in checkpoint.parents or not checkpoint.is_file() or sha256(checkpoint)!=lock['检查点SHA256']:
        raise RuntimeError('出图检查点不存在、超出运行目录或SHA256不一致，不能更新固定图册。')
    return lock


def publish_latest_gallery(root:Path,out:Path):
    root,out=root.resolve(),out.resolve()
    destination=root/FIXED_GALLERY
    backup=destination.with_name('结果总览_原始运行.html')
    record_path=destination.with_name('固定图册更新记录.json')
    if root not in out.parents or any(root not in p.resolve().parents for p in (destination,backup,record_path)):
        raise ValueError('图册来源及固定入口必须位于本项目内，不能发布到项目外。')
    lock=completed_gallery_lock(out)
    source=out/'结果总览.html'
    regenerated=out/'结果总览_本次重出图.html'
    if out==destination.parent and regenerated.is_file():
        source=regenerated
    if not source.is_file():
        raise RuntimeError('来源运行的结果图册尚未生成，不能更新固定图册。')
    metrics=json.loads((out/'测试指标_原始精度.json').read_text(encoding='utf-8'))
    record={'更新时刻':datetime.now().astimezone().isoformat(timespec='seconds'),
            '来源运行目录':out.relative_to(root).as_posix(),'固定入口':FIXED_GALLERY.as_posix(),
            '完成轮数':lock['完成轮数'],'出图检查点':lock['出图检查点'],
            '检查点SHA256':lock['检查点SHA256'],'训练完成记录时间':lock.get('时间','未记录'),
            '来源图册文件名':backup.name if source==destination else source.name,
            '图片保存方式':'内嵌原始PNG/GIF，无需外部图片路径'}
    # 先完成读取、校验和渲染，再备份及原子替换，避免坏结果破坏当前入口。
    content=gallery_html(out,metrics,destination,record,embed_images=True)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and not backup.exists():
        shutil.copy2(destination,backup)
    write_text_atomic(destination,content)
    write_text_atomic(record_path,json.dumps(record,ensure_ascii=False,indent=2))
    return destination


def export_results(model,root:Path,out:Path,splits:dict,config:dict):
    setup_font()
    completed_gallery_lock(out)
    tables=load_observations(root,'test',splits,model.geometry)
    metrics={name:table_metrics(table,predict(model,table.x)) for name,table in tables.items()}
    (out/'测试指标_原始精度.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    for power in splits['high']['test']:
        surface_images(model,root,out,power,tables['顶部'],config)
        for name in ('热端','冷端'):ring_images(model,out,power,name,tables[name])
        three_dimensional_animation(model,out,power,config,'high')
        three_dimensional_animation(model,out,power,config,'low')
    # 固定入口所在的旧运行重出图时，先写独立来源页，保留当前入口待发布。
    destination=out/'结果总览_本次重出图.html' if out.resolve()==(root/FIXED_GALLERY).parent.resolve() else None
    write_gallery(out,metrics,destination=destination)
    return publish_latest_gallery(root,out)
