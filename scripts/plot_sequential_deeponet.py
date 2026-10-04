"""Export paper-style comparison figures from actual saved predictions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.tri as mtri
import numpy as np

from sequential_deeponet_core import METHODS,LABELS
from train_sequential_deeponet import ROOT,write_json,digest

COLORS = {"fnn":"#0072B2","gru":"#D55E00","lstm":"#009E73","asl":"#CC79A7"}
FIGURES = []
SURFACE_MAXIMA = []
TEMPERATURE_CMAP = "RdYlBu_r"
ERROR_CMAP = "YlOrRd"


def save(fig,directory,name):
    fig.savefig(directory/(name+".png"),dpi=300,bbox_inches="tight",facecolor="white")
    (directory/(name+".pdf")).unlink(missing_ok=True)
    plt.close(fig)
    FIGURES.append(("appendix/" if directory.name=="appendix" else "")+name+".png")


def band_norm(values, *, error=False):
    low = 0. if error else float(np.min(values))
    high = max(low+1e-6,float(np.max(values)))
    levels = MaxNLocator(nbins=12).tick_values(low,high)
    return BoundaryNorm(levels,256)


def mark_maximum(ax,coordinates,errors):
    index = int(np.argmax(errors))
    point = coordinates[index]
    ax.scatter(*point,marker="X",s=35,c="black",edgecolors="white",linewidths=.8,zorder=10)
    return index


def history_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [{key:float(value) if value else np.nan for key,value in row.items()}
                for row in csv.DictReader(handle)]


def training_figures(output,directory,*,history_sources=None,comparable_validation=False):
    fig,axes = plt.subplots(1,2,figsize=(10.,3.7),layout="constrained")
    for method in METHODS:
        source = history_sources[method] if history_sources else output/method/"history.csv"
        history = history_rows(source)
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs,[row["loss"] for row in history],color=COLORS[method],alpha=.8,lw=1.,label=LABELS[method])
        valid = [row for row in history if np.isfinite(row.get("validation_score_k",np.nan))]
        scores = ([np.sqrt(.5*row["validation_顶部_rmse_k"]**2+.25*row["validation_热端_rmse_k"]**2
                           +.25*row["validation_冷端_rmse_k"]**2) for row in valid]
                  if comparable_validation else [row["validation_score_k"] for row in valid])
        axes[1].plot([row["epoch"] for row in valid],scores,
                     color=COLORS[method],lw=1.5,label=LABELS[method])
    for ax,title in zip(axes,("训练损失","实验验证集误差")):
        ax.set(xlabel="训练轮次",title=title,xlim=(0,1000))
        ax.grid(alpha=.2)
    axes[0].set(yscale="log",ylabel="加权归一化损失")
    axes[1].set(yscale="log",ylabel="实验验证综合 RMSE / K" if comparable_validation else "验证选取分数 / K")
    axes[1].legend(fontsize=9)
    save(fig,directory,"Figure_1_training_curves")


def metric_figures(results,directory):
    fig,axes = plt.subplots(1,3,figsize=(12.4,3.7),layout="constrained")
    definitions = [
        ("顶部实验测试误差","high_test","顶部","rmse_k","RMSE / K"),
        ("热端实验测试误差","high_test","热端","rmse_k","RMSE / K"),
        ("冷端实验测试误差","high_test","冷端","rmse_k","RMSE / K"),
    ]
    labels = [key.upper() for key in METHODS]
    for ax,(title,outer,inner,key,ylabel) in zip(axes.flat,definitions):
        values = []
        for method in METHODS:
            value = results[method]
            if outer:
                value = value[outer]
            if inner:
                value = value[inner]
            values.append(value[key])
        ax.bar(labels,values,color=[COLORS[key] for key in METHODS],width=.6)
        ax.set(title=title,ylabel=ylabel,ylim=(0,max(values)*1.22))
        ax.grid(axis="y",alpha=.2)
        for i,value in enumerate(values):
            ax.text(i,value+max(values)*.025,f"{value:,.0f}" if key=="parameters" else f"{value:.2f}",
                    ha="center",va="bottom",fontsize=10)
    save(fig,directory,"Figure_2_test_accuracy")


def test_power_figures(results,directory):
    powers = sorted(results["fnn"]["high_test_by_power"],key=float)
    fig,axes = plt.subplots(1,3,figsize=(13.,3.9),layout="constrained")
    centers,width = np.arange(len(powers)),.19
    for ax,name in zip(axes,("顶部","热端","冷端")):
        for i,method in enumerate(METHODS):
            values = [results[method]["high_test_by_power"][p][name]["rmse_k"] for p in powers]
            locations = centers+(i-1.5)*width
            ax.bar(locations,values,width=width,color=COLORS[method],label=LABELS[method])
            for location,value in zip(locations,values):
                ax.text(location,value,f"{value:.2f}",ha="center",va="bottom",fontsize=8,rotation=90)
        ax.set(title=f"{name}：各测试功率",xlabel="测试功率 / W",ylabel="RMSE / K",
               xticks=centers,xticklabels=powers)
        ax.set_ylim(0,ax.get_ylim()[1]*1.3)
        ax.grid(axis="y",alpha=.15)
    axes[0].legend(fontsize=8,ncol=2)
    save(fig,directory,"Figure_3_test_errors_by_power")


def histograms(cases,directory):
    fig,axes = plt.subplots(2,4,figsize=(14.4,6.8),layout="constrained")
    for row,tag in enumerate(("simulation","top")):
        max_error = max(item["relative_l2_kelvin_pct"] for method in METHODS for item in cases[method][tag])
        bins = np.linspace(0.,max_error*1.025,26)
        for column,method in enumerate(METHODS):
            errors = np.array([item["relative_l2_kelvin_pct"] for item in cases[method][tag]])
            ax = axes[row,column]
            ax.hist(errors,bins=bins,color=COLORS[method],edgecolor="white",linewidth=.5)
            ax.axvline(errors.mean(),color="black",lw=1.,ls="--")
            ax.set(title=LABELS[method],xlabel="Relative L2 in Kelvin (%)",xlim=(0.,bins[-1]))
            ax.text(.97,.94,f"n = {len(errors)}\nmean = {errors.mean():.3f}%\n90th = {np.percentile(errors,90):.3f}%",
                    transform=ax.transAxes,ha="right",va="top",fontsize=9)
            ax.grid(axis="y",alpha=.15)
        axes[row,0].set_ylabel("Simulation full-field cases" if tag=="simulation" else "Experimental surface frames")
    save(fig,directory,"Figure_3_relative_error_histograms")


def draw_section(ax,x,values,norm,cmap):
    artists = []
    for material in (0,1):
        selected = x[:,4]==material
        r,z,v = x[selected,0]*1000,x[selected,1]*1000,values[selected]
        positive = r>1e-7
        r,z,v = np.r_[r,-r[positive]],np.r_[z,z[positive]],np.r_[v,v[positive]]
        points,inverse = np.unique(np.column_stack((r,z)),axis=0,return_inverse=True)
        sums = np.bincount(inverse,weights=v)
        vals = sums/np.bincount(inverse)
        triangulation = mtri.Triangulation(points[:,0],points[:,1])
        center = points[triangulation.triangles].mean(axis=1)
        if material==0:
            triangulation.set_mask((np.abs(center[:,0])<25.-1e-3)&(center[:,1]>-12.+1e-3))
        artists.append(ax.tricontourf(triangulation,vals,levels=norm.boundaries,cmap=cmap,norm=norm))
        ax.tricontour(triangulation,vals,levels=norm.boundaries[::3],colors="black",linewidths=.3,alpha=.45)
    ax.plot([-25,-25,25,25],[0,-12,-12,0],color="white",lw=.5)
    ax.set(xlim=(-58.34,58.34),ylim=(-17.5,0),xlabel="Diameter coordinate (mm)",ylabel="z (mm)")
    ax.set_aspect("equal")
    return artists[-1]


def percentile_fields(simulation,cases,directory,*,model_epochs=None):
    selection_rows = []
    x,y = simulation["x"],simulation["y"]
    for method in METHODS:
        ordered = sorted(cases[method]["simulation"],key=lambda row:row["relative_l2_kelvin_pct"])
        selected = [ordered[int(round(q*(len(ordered)-1)))] for q in (0.,.9,1.)]
        masks = [np.isclose(x[:,3],row["power_w"])&np.isclose(x[:,2],row["time_s"]) for row in selected]
        temp_values = np.concatenate([np.r_[y[mask],simulation[method][mask]]-273.15 for mask in masks])
        errors = np.concatenate([np.abs(simulation[method][mask]-y[mask]) for mask in masks])
        norm = band_norm(temp_values)
        error_norm = band_norm(errors,error=True)
        fig,axes = plt.subplots(3,3,figsize=(13.1,5.7),layout="constrained")
        for i,(label,row,mask) in enumerate(zip(("Best","90th percentile","Worst"),selected,masks)):
            for j,values in enumerate((y[mask]-273.15,simulation[method][mask]-273.15,
                                       np.abs(simulation[method][mask]-y[mask]))):
                artist = draw_section(axes[i,j],x[mask],values,error_norm if j==2 else norm,
                                      ERROR_CMAP if j==2 else TEMPERATURE_CMAP)
                axes[i,j].set_title(("Simulation reference","Prediction","Absolute error")[j],fontsize=10)
                if j==2:
                    mark_maximum(axes[i,j],x[mask,:2]*1000,values)
                    axes[i,j].set_title(f"Absolute error; Max = {values.max():.2f} K",fontsize=10)
                axes[i,j].set_xlabel("Diameter coordinate (mm)" if i==2 else "")
                if j:
                    axes[i,j].set_ylabel("")
                if j in (1,2):
                    fig.colorbar(artist,ax=axes[i,j],fraction=.032,pad=.02,label="Error (K)" if j==2 else "Temperature (deg C)")
            axes[i,0].text(0.,1.42,f"{label}: {row['power_w']:g} W, {row['time_s']:g} s; L2 = {row['relative_l2_kelvin_pct']:.3f}%",
                           transform=axes[i,0].transAxes,fontsize=10,ha="left")
            selection_rows.append(dict(method=method,percentile=label,**row))
        epoch = 1000 if model_epochs is None else model_epochs[method]
        fig.suptitle(f"{LABELS[method]} | held-out simulation fields | update {epoch}",fontsize=13)
        save(fig,directory,f"Figure_4_{method}_percentile_fields")
    return selection_rows


def shared_simulation(simulation,directory,*,model_caption="epoch 1000"):
    x,y = simulation["x"],simulation["y"]
    mask = np.isclose(x[:,3],610.)&np.isclose(x[:,2],100.)
    all_temps = np.concatenate([simulation[key][mask] for key in METHODS]+[y[mask]])-273.15
    max_error = max(np.abs(simulation[key][mask]-y[mask]).max() for key in METHODS)
    norm,error_norm = band_norm(all_temps),band_norm([0.,max_error],error=True)
    fig,axes = plt.subplots(4,3,figsize=(13.1,6.6),layout="constrained")
    for i,method in enumerate(METHODS):
        for j,values in enumerate((y[mask]-273.15,simulation[method][mask]-273.15,np.abs(simulation[method][mask]-y[mask]))):
            artist = draw_section(axes[i,j],x[mask],values,error_norm if j==2 else norm,ERROR_CMAP if j==2 else TEMPERATURE_CMAP)
            axes[i,j].set_title(("Simulation reference",LABELS[method],"Absolute error")[j],fontsize=10)
            if j==2:
                mark_maximum(axes[i,j],x[mask,:2]*1000,values)
                axes[i,j].set_title(f"Absolute error; Max = {values.max():.2f} K",fontsize=10)
            axes[i,j].set_xlabel("Diameter coordinate (mm)" if i==3 else "")
            if j:
                axes[i,j].set_ylabel("")
            if j in (1,2):
                fig.colorbar(artist,ax=axes[i,j],fraction=.03,pad=.02,label="Error (K)" if j==2 else "Temperature (deg C)")
    fig.suptitle(f"Common held-out case: 610 W, 100 s | {model_caption}",fontsize=13)
    save(fig,directory,"Figure_5_common_full_field_and_errors")


def surface_fields(evaluation,directory,*,model_caption="第 1000 轮模型"):
    for time_s in (5,30,60,90,120):
        pack = np.load(evaluation/f"surface_634W_{time_s}s.npz")
        xy,y = pack["xy_mm"],pack["y"]
        unique_x,unique_y = np.unique(xy[:,0]),np.unique(xy[:,1])
        ix,iy = np.searchsorted(unique_x,xy[:,0]),np.searchsorted(unique_y,xy[:,1])
        all_temps = np.concatenate([y]+[pack[method] for method in METHODS])-273.15
        norm = band_norm(all_temps)
        error_norm = band_norm([0.,max(np.abs(pack[method]-y).max() for method in METHODS)],error=True)
        fig,axes = plt.subplots(2,5,figsize=(14.2,7.),layout="constrained")
        for i,key in enumerate(("reference",)+METHODS):
            values = y if key=="reference" else pack[key]
            grid = np.full((len(unique_y),len(unique_x)),np.nan)
            grid[iy,ix] = values-273.15
            im = axes[0,i].contourf(unique_x,unique_y,grid,levels=norm.boundaries,cmap=TEMPERATURE_CMAP,norm=norm)
            contours = axes[0,i].contour(unique_x,unique_y,grid,levels=norm.boundaries[::3],colors="black",linewidths=.4,alpha=.6)
            axes[0,i].clabel(contours,inline=True,fontsize=8,fmt="%g")
            axes[0,i].set_title("实验测量温度场" if key=="reference" else LABELS[key],fontsize=10)
            if i:
                error_grid = np.full_like(grid,np.nan)
                error_grid[iy,ix] = np.abs(values-y)
                err = axes[1,i].contourf(unique_x,unique_y,error_grid,levels=error_norm.boundaries,cmap=ERROR_CMAP,norm=error_norm)
                error = np.abs(values-y)
                index = mark_maximum(axes[1,i],xy,error)
                rmse = float(np.sqrt(np.mean((values-y)**2)))
                axes[1,i].set_title(f"RMSE = {rmse:.2f} K\n最大误差 = {error[index]:.2f} K",fontsize=10)
                SURFACE_MAXIMA.append(dict(time_s=time_s,method=key,rmse_k=rmse,
                    max_abs_error_k=float(error[index]),x_mm=float(xy[index,0]),y_mm=float(xy[index,1]),
                    source_npz_sha256=digest(evaluation/f"surface_634W_{time_s}s.npz")))
            else:
                axes[1,i].axis("off")
        for ax in axes.flat:
            if ax.axison:
                ax.set(xlabel="x (mm)",ylabel="y (mm)",aspect="equal",xticks=(-25,0,25),yticks=(-25,0,25),
                       xlim=(-26,26),ylim=(-26,26))
        fig.colorbar(im,ax=axes[0,:].tolist(),fraction=.016,pad=.015,label="温度 / ℃")
        fig.colorbar(err,ax=axes[1,1:].tolist(),fraction=.02,pad=.015,label="绝对误差 / K")
        fig.suptitle(f"实验测试：634 W，{time_s} s，{model_caption}",fontsize=13)
        save(fig,directory,f"Figure_6_surface_634W_{time_s}s")


def profile_figures(evaluation,directory):
    top = np.load(evaluation/"top_predictions.npz")
    x,y = top["x"],top["y"]
    fig,axes = plt.subplots(1,3,figsize=(13.2,3.8),layout="constrained")
    for ax,power in zip(axes,(169.,339.,634.)):
        power_mask = np.isclose(x[:,3],power)
        times = np.unique(x[power_mask,2])
        time_s = times[np.argmin(np.abs(times-min(120.,times.max())))]
        mask = power_mask&np.isclose(x[:,2],time_s)
        order = np.argsort(x[mask,0])
        r = x[mask,0][order]*1000
        ax.plot(r,y[mask][order]-273.15,color="black",lw=2.,label="实验径向平均温度")
        for method in METHODS:
            ax.plot(r,top[method][mask][order]-273.15,color=COLORS[method],lw=1.4,label=LABELS[method])
        ax.set(title=f"{power:g} W，{time_s:g} s",xlabel="半径 / mm",ylabel="温度 / ℃")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig,directory,"Figure_7_top_radial_profiles")
    fig,axes = plt.subplots(3,3,figsize=(13.2,10.),layout="constrained")
    packs = [top,np.load(evaluation/"hot_predictions.npz"),np.load(evaluation/"cold_predictions.npz")]
    for i,(tag,pack) in enumerate(zip(("顶部近中心","热端","冷端"),packs)):
        x,y = pack["x"],pack["y"]
        for j,power in enumerate((169.,339.,634.)):
            ids = np.flatnonzero(np.isclose(x[:,3],power))
            if i==0:
                ids = np.array([group[np.argmin(x[group,0])] for t in np.unique(x[ids,2])
                                for group in [ids[np.isclose(x[ids,2],t)]]])
            ids = ids[np.argsort(x[ids,2])]
            ax = axes[i,j]
            ax.plot(x[ids,2],y[ids]-273.15,"k-o",lw=1.8,ms=2.5,markevery=8,label="实验测量")
            dense = np.load(evaluation/f"dense_{'hot' if i==1 else 'cold'}_{power:g}W_predictions.npz") if i else None
            for method in METHODS:
                styles = {"fnn":"-","gru":"--","lstm":"-.","asl":"-"}
                plot_times = x[ids,2] if dense is None else dense["x"][:int(dense["grid_count"]),2]
                plot_values = pack[method][ids] if dense is None else dense[method][:int(dense["grid_count"])]
                ax.plot(plot_times,plot_values-273.15,color=COLORS[method],
                        ls=styles[method],lw=1.8 if method=="asl" else 1.3,label=LABELS[method])
            ax.set(title=f"{tag}，{power:g} W",xlabel="时间 / s",ylabel="温度 / ℃")
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    save(fig,directory,"Figure_8_temporal_response")


def parity_figure(evaluation,results,directory):
    top = np.load(evaluation/"top_predictions.npz")
    y = top["y"]-273.15
    lo,hi = y.min()-5,y.max()+5
    fig,axes = plt.subplots(2,2,figsize=(9.,8.5),layout="constrained")
    ids = np.arange(0,len(y),max(1,len(y)//5000))
    for ax,method in zip(axes.flat,METHODS):
        ax.scatter(y[ids],top[method][ids]-273.15,s=5,alpha=.35,color=COLORS[method],rasterized=True)
        ax.plot([lo,hi],[lo,hi],color="black",lw=1,ls="--")
        ax.set(title=LABELS[method],xlabel="Supplied radial-mean temperature (deg C)",
               ylabel="Predicted temperature (deg C)",xlim=(lo,hi),ylim=(lo,hi))
        ax.set_aspect("equal")
        ax.text(.05,.92,f"R2 = {results[method]['high_test']['顶部']['r2']:.5f}",transform=ax.transAxes)
    save(fig,directory,"Figure_9_actual_vs_predicted")


def revised_sensor_report(output,results,figures):
    cfg = json.loads((output/"config.json").read_text(encoding="utf-8"))
    temporal = json.loads((output/"temporal_audit.json").read_text(encoding="utf-8"))
    bias = json.loads((output/"diagnostics/temperature_bias.json").read_text(encoding="utf-8"))
    revision = json.loads((output/"data_revision/data_revision.json").read_text(encoding="utf-8"))
    lines = ["# 更新冷端数据与 25℃初温后的四方法结果", "",
        "FNN、GRU、LSTM、ASL 已分别真实训练 1000 轮，随机种子 123。主表与图像均使用第 1000 轮模型和当前测量数据。", "",
        "## 本次数据与初温", "",
        "169 W 冷端使用用户修订后的 colddata169W.csv，已重新生成对应测试传感器处理数据。"
        "其他训练、验证、仿真和实验处理数据未改变，保留逐文件 SHA256 核对记录。",
        f"该文件有 {revision['samples']} 个时刻，首条实测为 {revision['first_time_s']:g} s、"
        f"{revision['first_measurement_c']:.5f}℃，末条为 {revision['last_measurement_c']:.5f}℃。"
        f"新旧观测最大差值为 {revision['maximum_measurement_change_c']:.5f}℃。", "",
        "实验铜区（包含热端、冷端）的 0 s 预测严格固定为 25℃，由模型初值锚点直接保证。"
        "实验传感器历史的 0 s 锚点和温度归一化基准也使用 25℃，物理初始条件损失使用相同初值。"
        "顶部 SiC 和低保真仿真的初温仍为 22℃。铜与 SiC 使用不同的初始温度；这是本轮明确采用的材料初值设定。"
        "原始实测温度没有平移或重写，也没有为了连接 25℃起点而平滑预测输出。", "",
        "ASL 保留两个选择性 SSM、因果卷积、LSTM、h/c 初值和成熟度门控，参数结构未改变。"
        "沿用上一轮验证选定的热趋势输入、时间正则、顶部中心和后段约束。"
        "本轮只按用户要求更新数据与铜区初温，不使用修订后的测试标签进行超参数搜索。", "",
        "旧结果含有旧版 169 W 冷端标签和不同的初温，不能直接用缓存指标作新旧效果比较。"
        "本报告只列出本轮四种方法与当前测试测量的对比。数据修订与固定配置记录分别见"
        "[data_revision.json](data_revision/data_revision.json) 和 [revision_protocol.json](revision_protocol.json)。", "",
        "## 当前实验测试结果", "",
        "| 方法 | 顶部 RMSE / K | 热端 RMSE / K | 冷端 RMSE / K | 综合 RMSE / K |",
        "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        value = results[method]
        high = value["high_test"]
        lines.append(f"| {LABELS[method]} | {high['顶部']['rmse_k']:.4f} | {high['热端']['rmse_k']:.4f} | "
                     f"{high['冷端']['rmse_k']:.4f} | {value['combined_test_rmse_k']:.4f} |")
    lines += ["", "综合 RMSE = sqrt(0.5 × 顶部 RMSE² + 0.25 × 热端 RMSE² + 0.25 × 冷端 RMSE²)。温差 1 K 等于 1℃。", "",
        "## ASL 各功率顶部后段", "",
        "| 测试功率 / W | 全顶部 RMSE / K | 中心后段 RMSE / K | 中心末时刻偏差 / K |",
        "|---|---:|---:|---:|"]
    measured = bias["runs"]["current"]["splits"]["test"]["asl"]["顶部"]
    for power,row in sorted(measured["by_power"].items(),key=lambda item:float(item[0])):
        lines.append(f"| {power} | {row['full']['rmse_k']:.4f} | {row['center_tail']['rmse_k']:.4f} | {row['center_final_bias_k']:+.4f} |")
    lines += ["", "后段定义为每个功率实测结束时间的最后 30%，中心取每帧最小半径处的实际观测。"
        "训练、验证、测试的完整有符号偏差记录见 [temperature_bias.json](diagnostics/temperature_bias.json)。", "",
        "## 原始时间曲线检查", "",
        "检查了三个测试功率、四种方法的 24 条热端/冷端曲线，以及 12 条顶部近中心曲线。"
        "使用 0.1 s 密集网格，并检查整数秒前后 0.001 s 的原始模型输出。历史输入只使用 t−1 s 及更早的测量；"
        "已知采集范围决定有效性，精确功率的历史不受邻居功率结束时间影响。", "",
        "| 传感器 | 功率 / W | FNN 后期最大 0.1 s 变化 / K | GRU | LSTM | ASL |",
        "|---|---:|---:|---:|---:|---:|"]
    for sensor in ("hot","cold"):
        for power in (169.,339.,634.):
            values = [next(r for r in temporal["dense_curves"] if
                (r["sensor"],r["power_w"],r["method"])==(sensor,power,m))["late_max_abs_step_k"] for m in METHODS]
            lines.append(f"| {'热端' if sensor=='hot' else '冷端'} | {power:g} | "+" | ".join(f"{v:.5f}" for v in values)+" |")
    for rows,title in ((temporal["dense_curves"],"热端/冷端"),(temporal["dense_top_curves"],"顶部近中心")):
        asl = [r for r in rows if r["method"]=="asl"]
        late = max(r["late_incoming_integer_max_jump_k"] for r in asl)
        early = max(asl,key=lambda r:r["incoming_integer_max_jump_k"])
        lines += ["",f"ASL {title}在 t ≥ 75 s 的最大新观测到达变化为 {late:.6f} K；"
            f"全时间最大到达变化为 {early['incoming_integer_max_jump_k']:.5f} K"
            f"（{early['power_w']:g} W、{early['incoming_peak_time_s']:g} s）。"
            "早期历史建立阶段与后期分别记录，不能把后期检查解释为全时间强制单调或严格连续。"]
    lines += ["", "完整记录见 [temporal_audit.json](temporal_audit.json)。Figure 8 的热端、冷端使用密集网格原始预测，顶部使用实际观测时刻的预测，没有输出后处理。", "",
        "## 图像", "",
        "本轮和后续导出只保存 300 DPI PNG。主图直接对比 169、339、634 W 的实验测试数据。"
        "Figure 3 为各测试功率的顶部、热端、冷端 RMSE；Figure 8 为实际时间响应。"
        "仿真内部场缺少对应实验内部测量，因此作为附录参考。LF 指仿真，HF 指实验，不是额外算法。", "",
        "温度云图使用分级蓝—浅黄—红色带和等温线，误差使用浅黄—深红色带。"
        "同一工况四种方法共用色标；误差云图标注最大绝对误差、RMSE 和最大误差位置（黑色 X）。", ""]
    lines.extend(f"- [{name}](figures/{name})" for name in figures)
    lines += ["", "## 比较口径与复现", "",
        "仿真功率划分为 60/10/10，实验为 12/3/3。每轮是覆盖全部训练功率和物理采样的一次 AdamW 更新，"
        "并非遍历所有仿真网格点。四种方法共用损失、随机种子、主干初始权重和学习率计划。"
        "验证最优检查点保存在 best.pt，主图仍统一用 epoch_1000.pt。", "",
        "模型是基于过去热端/冷端观测的条件温度场重构；不是只输入功率便完全不依赖传感器的预测。"
        "PDE 求坐标导数时固定历史输入。二维顶部图保留 IR 派生数据的 is_recovered 标记；"
        "本轮为一个随机种子，未据此证明任何结构普遍最优。", "",
        "```bash", "python scripts/refresh_sequential_sensor_data.py --destination '研究记录/Sequential_DeepONet_sensor25_new/data_revision'",
        "python scripts/train_sequential_deeponet.py --config configs/sequential_deeponet_sensor25.yaml --device cuda --output '研究记录/Sequential_DeepONet_sensor25_new'",
        "python scripts/evaluate_sequential_deeponet.py --device cuda --output '研究记录/Sequential_DeepONet_sensor25_new'",
        "python scripts/analyze_sequential_temperature_bias.py --output '研究记录/Sequential_DeepONet_sensor25_new' --methods fnn gru lstm asl --splits train validation test",
        "python scripts/audit_sequential_temporal.py --output '研究记录/Sequential_DeepONet_sensor25_new'",
        "python scripts/plot_sequential_deeponet.py --output '研究记录/Sequential_DeepONet_sensor25_new'",
        "python scripts/verify_sequential_deeponet.py --output '研究记录/Sequential_DeepONet_sensor25_new'", "```", ""]
    (output/"对比总结.md").write_text("\n".join(lines),encoding="utf-8")


def report(output,results,figures):
    cfg = json.loads((output/"config.json").read_text(encoding="utf-8"))
    if cfg.get("experiment",{}).get("data_revision"):
        return revised_sensor_report(output,results,figures)
    thermal = cfg["model"].get("asl_rate_mode","original")=="thermal_trend"
    adaptation_note = ("在此基础上进行下述 ASL 项目适配，保持种子和训练预算不变，" if thermal
                       else "保持架构、损失、种子和训练预算不变，")
    source_alignment = ("原始预处理模式在相同权重下已与原始源码核对；本项目正式 ASL 使用上述 thermal_trend 输入适配，不宣称其输出与原算法逐值相同。" if thermal
                        else "相同权重下的分支输出已与原始源码核对。")
    extra_loss = "和共同时间连续性" if cfg["loss_weights"].get("temporal",0)>0 else ""
    baseline = results["fnn"]["combined_test_rmse_k"]
    ranked = sorted(METHODS,key=lambda key:results[key]["combined_test_rmse_k"])
    lines = ["# 四种 DeepONet 的 1000 epochs 对比", "",
        "本次已真实完成 FNN、GRU、LSTM、ASL 各 1000 epochs，随机种子为 123，全部关闭边界注意力。主表统一使用第 1000 轮模型。", "",
        "## 测试结果", "",
        "| 方法 | 顶部 RMSE / K | 热端 RMSE / K | 冷端 RMSE / K | 综合 RMSE / K |",
        "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        value = results[method]
        high = value["high_test"]
        lines.append(f"| {LABELS[method]} | {high['顶部']['rmse_k']:.4f} | {high['热端']['rmse_k']:.4f} | "
            f"{high['冷端']['rmse_k']:.4f} | {value['combined_test_rmse_k']:.4f} |")
    lines += ["", "综合 RMSE = sqrt(0.5 × 顶部 RMSE² + 0.25 × 热端 RMSE² + 0.25 × 冷端 RMSE²)。温差 1 K 等于 1 ℃。", "",
        "按第 1000 轮实验测试综合 RMSE 排序："+" < ".join(LABELS[key] for key in ranked)+"。"]
    for method in METHODS[1:]:
        delta = 100*(baseline-results[method]["combined_test_rmse_k"])/baseline
        lines.append(f"{LABELS[method]} 相对 FNN 的综合 RMSE {'降低' if delta>=0 else '增加'} {abs(delta):.2f}%。")
    top_ranked = sorted(METHODS,key=lambda key:results[key]["high_test"]["顶部"]["rmse_k"])
    lines += ["", "## 比较分析", "",
        f"实验顶部 RMSE 最低的是 {LABELS[top_ranked[0]]}，综合误差最低的是 {LABELS[ranked[0]]}。",
        "主图统一对比实验测试集中的 169、339、634 W。仿真全场误差不混入实验主表。"]
    lines += ["", "## 后期突跳检查", "",
        "V5 历史选择存在缺陷：精确查询某个已测功率时，温度虽然取自该功率，"
        "有效性标记却同时受相邻功率约束。339 W 错误继承了 169 W 的 80 s 结束信息，"
        "634 W 错误继承了 339 W 的 100 s 结束信息。标记突然变化会改变循环分支输出，"
        "造成原 Figure 8 中的尖峰和反复变化。另一个缺陷是有效标记使用最新已到达样本的时刻，"
        "在整数秒到非整数秒之间错误翻转；例如仅修复邻居选择后的 ASL 在 634 W 热端的 "
        "80.000 → 80.001 s 仍会跳变 8.28 K。",
        "本次修复精确功率的历史选择，并用已知采集时间范围判断有效性。"
        "此外，末段只保持旧温度会使 ASL 升温率特征反复下降、重置，产生细密锯齿。"
        "现在用最近两条已到达观测的线性趋势估计末段温度，最多延伸一个历史采样间隔，"
        "仅在已知采集范围内使用，并等待至少三个历史点以避开初始锚点的首段斜率。"
        "该末段值是过去观测构建的输入估计，不是非整数秒实测温度，"
        "不引入未来温度作为插值端点。"+adaptation_note+
        "重新训练全部四种方法各 1000 轮。训练、验证、测试所有功率均检查了相邻曲线独立性。"
        "Figure 8 的热端/冷端使用 0.1 s 网格的原始模型预测，顶部使用实际采集时刻；"
        "没有平滑、插值修饰或强制单调后处理。", ""]
    temporal = json.loads((output/"temporal_audit.json").read_text(encoding="utf-8"))
    if "asl_adaptation" in temporal:
        adaptation = temporal["asl_adaptation"]
        old_jump = max(r["previous"]["incoming_integer_max_jump_k"] for r in adaptation["curves"])
        new_jump = max(r["adapted"]["incoming_integer_max_jump_k"] for r in adaptation["curves"])
        old_top = adaptation["previous"]["high_test"]["顶部"]["rmse_k"]
        new_top = adaptation["adapted"]["high_test"]["顶部"]["rmse_k"]
        worst = max((r for r in temporal["dense_curves"] if r["method"]=="asl"),
                    key=lambda r:r["incoming_integer_max_jump_k"])
        accuracy_tradeoff = "；".join(f"{name} RMSE {adaptation['previous']['high_test'][name]['rmse_k']:.4f} → "
            f"{adaptation['adapted']['high_test'][name]['rmse_k']:.4f} K" for name in ("热端","冷端"))
        lines += ["## ASL 针对本项目的改造", "",
            "此前 history_fix_v3 已修复历史掩码和末段估计，但原 ASL 仍对相邻历史点直接求导。"
            "新观测替换末段预测时的小幅温差会被升温率通道放大：634 W、29 s 附近的输入修正约 0.089 K，"
            "热端和冷端输出却分别跳变约 0.326 K 和 0.342 K。固定升温率通道的诊断使该处输出变化降至约 0.008 K / 0.003 K。", "",
            "本次保留两个选择性 SSM → LSTM、h/c 初始化、成熟度上下文门控，以及全部可训练模块与参数数量。"
            f"仅把升温率计算改为过去 {adaptation['rate_window_s']:g} s 有效历史的因果加权线性拟合，"
            f"使用物理单位 K/s 并按 {adaptation['rate_scale_k_per_s']:g} K/s 归一化。"
            "每个历史位置只使用该位置及更早的有效采样点。", "",
            f"四种方法统一增加训练传感器到达时刻前后 ±{adaptation['delta_s']:g} s 的预测二阶差分损失，"
            "按温度尺度及时间间隔平方归一化，使用单独的可恢复随机数状态。"
            "该项作用于原始模型输出并实际重新构造两侧的历史输入，补充固定历史的 PDE 偏导。"
            "它只使用训练功率、时间和测点坐标，不读取测试标签，也不强制单调。", "",
            f"六条 ASL 热端/冷端曲线的全时间段最大到达跳变（整数秒前 0.001 s → 整数秒）"
            f"由 {old_jump:.6f} K 变为 {new_jump:.6f} K；顶部测试 RMSE 由 {old_top:.4f} K 变为 {new_top:.4f} K。"
            "这是当前完整训练方案相对 history_fix_v3 的比较；若另有温度约束，亦包含其影响。", "",
            f"当前全时段最大到达跳变位于 {worst['power_w']:g} W {'热端' if worst['sensor']=='hot' else '冷端'}、"
            f"{worst['incoming_peak_time_s']:g} s，即初始历史建立阶段。"
            f"两端误差变化为：{accuracy_tradeoff}；综合 RMSE "
            f"{adaptation['previous']['combined_test_rmse_k']:.4f} → {adaptation['adapted']['combined_test_rmse_k']:.4f} K。", "",
            "旧检查点配置缺少 asl_rate_mode 时仍走 original 预处理；新模型明确保存 thermal_trend 配置。"
            "history_fix_v3 的原检查点、原始密集预测及其哈希保留用于复核。", ""]
    if cfg["loss_weights"].get("top_center",0)>0:
        bias = json.loads((output/"diagnostics/temperature_bias.json").read_text(encoding="utf-8"))
        before = bias["runs"]["baseline"]["splits"]["test"]["asl"]
        after = bias["runs"]["current"]["splits"]["test"]["asl"]
        lines += ["## 顶部后段误差优化", "",
            "本轮以前一版 thermal_trend ASL 为基线，保持 SSM → LSTM、所有可训练模块、参数数量、输入历史与初始化不变。"
            "旧损失把顶部各半径混合采样，中心偏低、边缘偏高可以同时出现；后期趋势和末时刻偏差没有单独约束。"
            "训练数据本身也存在这一现象，故不能只归因为测试功率插值。", "",
            f"四种方法统一增加半径≤{cfg['training']['top_center_radius_m']*1000:g} mm 的训练观测，"
            f"并用每个训练功率最后 {cfg['training']['top_tail_window_s']:g} s 的实测时刻约束温度、末时刻与升温趋势。"
            "趋势误差按功率和半径分别拟合、分别平方，中心与边缘的相反偏差不能互相抵消。"
            f"完整顶部温度损失权重为 {cfg['loss_weights']['top']:g}，避免仅改善后段而损害前段拟合。"
            "网络输出未经平滑、偏移校准或强制单调。", "",
            "配置比较使用训练与验证集；诊断中曾查看过旧测试结果，故不宣称盲测。"
            "验证选取同时考虑整个顶部、实测近中心后段以及两端误差。"
            "本表后段定义为每个功率自身实测末时刻的后30%，中心为各帧实际测得的最小半径。", "",
            "验证选取分数 = sqrt(0.5×顶部RMSE² + 0.25×中心后段RMSE² + 0.125×热端RMSE² + 0.125×冷端RMSE²)。"
            "它用于验证选型，与主表三类测点的测试综合RMSE不同。", "",
            "| 测试功率 / W | 中心后段 RMSE：原→新 / K | 末时刻有符号误差：原→新 / K |",
            "|---|---:|---:|"]
        for power in ("169","339","634"):
            old,new = before["顶部"]["by_power"][power],after["顶部"]["by_power"][power]
            lines.append(f"| {power} | {old['center_tail']['rmse_k']:.4f} → {new['center_tail']['rmse_k']:.4f} | "
                         f"{old['center_final_bias_k']:+.4f} → {new['center_final_bias_k']:+.4f} |")
        lines += ["", f"整个顶部测试 RMSE：{before['顶部']['full']['rmse_k']:.4f} → {after['顶部']['full']['rmse_k']:.4f} K；"
            f"中心后段汇总 RMSE：{before['顶部']['center_tail']['rmse_k']:.4f} → {after['顶部']['center_tail']['rmse_k']:.4f} K。",
            "两端误差变化为："+"；".join(f"{name} RMSE {before[name]['full']['rmse_k']:.4f} → {after[name]['full']['rmse_k']:.4f} K"
                                     for name in ("热端","冷端"))+"。",
            "末时刻偏差减小仍不等于每条曲线的后段斜率误差均降低；339 W 仍存在低估，"
            "634 W 后段由较大负偏差趋向较小负偏差。逐功率斜率误差另存于诊断文件，保留实际结果。"]
        lines += ["", "正误差表示模型高于实测，负误差表示低于实测。"
            "完整逐功率、逐区域、后段斜率及实际中心时刻数据保存在 [temperature_bias.json](diagnostics/temperature_bias.json)，"
            "验证选型记录保存在 [selection.json](diagnostics/selection.json)。", ""]
    lines += ["以下为 t ≥ 75 s 的最大相邻测量时刻跳变，单位 K，单元格为 V5 → 修复后。", "",
        "| 测点 | 功率 / W | FNN | GRU | LSTM | ASL |", "|---|---:|---:|---:|---:|---:|"]
    for sensor in ("hot","cold"):
        for power in (169.,339.,634.):
            cells = []
            for method in METHODS:
                row = next(r for r in temporal["curves"] if r["sensor"]==sensor and r["power_w"]==power and r["method"]==method)
                cells.append(f"{row['original']['late_max_abs_step_k']:.4f} → {row['corrected']['late_max_abs_step_k']:.4f}")
            lines.append(f"| {'热端' if sensor=='hot' else '冷端'} | {power:g} | "+" | ".join(cells)+" |")
    lines += ["", "完整跳变时刻、向下变化量、各曲线 RMSE 和输入掩码检查保存在 [temporal_audit.json](temporal_audit.json)。", "",
        "另外对三个测试功率的热端、冷端和四种方法逐一计算 0.1 s 时间网格及整数秒前后 0.001 s 的原始预测。"]
    outgoing = max(r["late_outgoing_integer_max_jump_k"] for r in temporal["dense_curves"])
    incoming = max(r["late_incoming_integer_max_jump_k"] for r in temporal["dense_curves"])
    dense_step = max(r["late_max_abs_step_k"] for r in temporal["dense_curves"])
    lines += [f"t ≥ 75 s 时，全部 24 条曲线的最大 0.1 s 变化为 {dense_step:.5f} K；"
        f"整数秒之后 0.001 s 的最大变化为 {outgoing:.6f} K。"
        f"整数秒新观测到达时的最大变化为 {incoming:.5f} K，允许输入更新，"
        "不能把这一检查解释为严格数学连续或强制单调。", "",
    ]
    if "dense_top_curves" in temporal:
        top_rows = [r for r in temporal["dense_top_curves"] if r["method"]=="asl"]
        late_jump = max(r["late_incoming_integer_max_jump_k"] for r in top_rows)
        warm_jump = max(r["post_warmup_incoming_integer_max_jump_k"] for r in top_rows)
        initial = max(top_rows,key=lambda r:r["incoming_integer_max_jump_k"])
        lines += [f"另外检查12条顶部近中心原始密集预测。ASL在 t ≥ 75 s 的最大观测到达变化为 {late_jump:.6f} K，"
            f"t ≥ 3 s 为 {warm_jump:.6f} K。初始历史建立阶段仍存在 {initial['incoming_integer_max_jump_k']:.4f} K 的变化"
            f"（{initial['power_w']:g} W、{initial['incoming_peak_time_s']:g} s）。"
            "顶部实测从5 s开始，保留早期原始诊断，不把这段视为已验证平滑。", ""]
    lines += [
        "## LF、HF 与旧 Figure 3", "",
        "LF（Low Fidelity）指低保真的有限元仿真参考，HF（High Fidelity）指高保真的实验测量。"
        "两者是训练与参考数据来源，不是另外两种模型。实验没有内部全场测量，"
        "因此仿真内部场对比只能作为补充，不能替代实验测试。现在主图直接展示实验测试结果，仿真图放入 appendix。", "",
        "旧 Figure 3 是误差直方图：横轴为每个功率/时刻工况的相对 L2 误差，纵轴为工况数，"
        "虚线为平均值，90th 为 90% 工况误差不超过的数值。它用来观察误差是否集中、是否有少量很差的工况，"
        "横轴不是时间。使用开尔文温度作分母会使百分比显得较小，不适合直观阅读温差。", "",
        "新的 Figure 3 改为按 169、339、634 W 分组的顶部/热端/冷端 RMSE，柱子越低越好，"
        "可直接判断哪个方法在哪个测试功率上误差较大。原论文形式的直方图保留在附录。", "",
        "## 验证最优模型（补充）", "",
        "| 方法 | 验证选定轮次 | 对应测试综合 RMSE / K |", "|---|---:|---:|"]
    for method in METHODS:
        value = results[method]
        lines.append(f"| {LABELS[method]} | {value['best_epoch']} | {value['best_combined_test_rmse_k']:.4f} |")
    lines += ["", "训练没有提前终止。最优轮次只由验证集选择，以上补充表不会替换主表的第 1000 轮结果。", "",
        "## 实现与比较口径", "",
        "参考 [S-DeepONet 论文](https://arxiv.org/abs/2306.08218)及[作者代码](https://github.com/Jasiuk-Research-Group/S-DeepONet)，将序列编码器作为分支网络，与 FNN 主干输出做点积。GRU/LSTM 使用长历史编码器和局部历史解码器，隐藏宽度 48；这是适配本项目输入的实现，未照搬原文 101 点负载及其大型编码器/解码器层数。", "",
        "ASL 来自[用户 ASL-PINN 仓库](https://github.com/fei4440273/Temperature-Reconstruction-ASL-PINN)，提交 4477c971007624133eec92bd7eeb5bb58d3b9c26。保留两个选择性 SSM、因果卷积、输入相关 Δ/B/C、稳定对角 A、D 跳连、门控残差、长历史升温率特征、SSM 终态初始化 LSTM 的 h/c，以及成熟度门控持续上下文。"+source_alignment+"主干与高/低保真输出头采用相同初始权重。边界注意力未导入或实例化。", "",
        "共同输入为 32 点长历史与 8 点局部历史，每点包含时间、功率、hot 温度、cold 温度和有效性。局部窗口 20 s，历史截止于 t−1 s；插值和有界末段趋势估计也只允许使用截止时刻之前的观测。有效性表示已知采集时间范围，而非该重采样时刻恰好有一条测量；范围是离线实验的元数据，不读取未来温度。在线采集若结束时刻未知，需要显式提供采集状态。初温统一为 22 ℃，时间/功率尺度为本项目的 200 s / 800 W。", "",
        "仿真按现有配置使用 60/10/10 个训练/验证/测试功率；实验使用 12/3/3 个功率，实验测试功率为 169、339、634 W。仿真传感器历史来自最接近两个底面传感器位置的铜网格节点；对应节点坐标保存在 JSON。", "",
        "每轮按全部训练功率分层随机采样，低保真每个功率/材料 8 点，顶部每功率 64 点，两端每功率各 16 点，并计算传热方程、边界、初值、界面、热阻先验"+extra_loss+"损失，再执行一次 AdamW 更新。因此 1000 epochs 指当前项目的采样训练轮次，不是对全部 702 万个仿真点循环 1000 次。所有方法使用同一配置、采样种子和学习率计划。", "",
        "## 结果适用范围", "",
        "这是 T(r,z,t,P | 过去 hot/cold 测量) 的条件重构。测试功率的过去传感器测量用于推理输入，当前/未来温度不进入输入；因此传感器误差含历史相关信息，顶部误差更适合评估从测量到空间场的泛化。PDE 对坐标求导时保持历史输入固定，沿用 ASL 源码定义，不能把该残差解释为沿整个更新历史的总时间导数。", "",
        "本项目为恒定激光加热数据，没有据此验证任意变动负载。仿真内部温度只作为低保真全场参考；实验没有内部全场标签。二维顶部图使用用户提供的 IR 派生温度 CSV，保留 is_recovered 标志，不把已恢复像素当成独立原始辐射测量。", "",
        "本轮为单一随机种子，结论限定于这次 1000 轮预算和数据配置，不能据此断言某种结构在统计上始终更优。训练时间包含定期验证和检查点之间的运行开销；推理时间仅为本机单次测试集计时，不外推有限元加速比。", "",
        "此前诊断轮和 V5 测试结果曾被查看，本轮属于针对已发现实现缺陷的修复复训，不宣称从未查看过测试数据的盲测。", "",
        "## 图像与复现", "",
        "主图只显示实验测试对比。温度场使用分级蓝—浅黄—红色带，并叠加带数值的等温线；"
        "绝对误差采用浅黄—深红色带。同一工况的四个模型共用温度和误差范围，"
        "每幅误差云图标注 RMSE、最大绝对误差及其位置（黑色 X）。"
        "顶部二维图固定为 634 W 的 5、30、60、90、120 s，只保存 300 DPI PNG。", ""]
    lines.extend(f"- [{name}](figures/{name})" for name in figures if not name.startswith("appendix/"))
    lines += ["", "## 附录：仿真参考与论文形式误差分布", "",
        "下面仅为仿真参考与原论文展示形式的补充。共同仿真工况为 610 W、100 s；"
        "最好/90 分位/最坏工况按各自工况相对 L2 排序，排除 t=0。"]
    lines.extend(f"- [{name}](figures/{name})" for name in figures if name.startswith("appendix/"))
    lines += ["", "```bash", "/home/phl/anaconda3/envs/PINN/bin/python scripts/train_sequential_deeponet.py --device cuda --output '研究记录/Sequential_DeepONet_new'",
        "/home/phl/anaconda3/envs/PINN/bin/python scripts/evaluate_sequential_deeponet.py --device cuda --output '研究记录/Sequential_DeepONet_new'",
        "/home/phl/anaconda3/envs/PINN/bin/python scripts/audit_sequential_temporal.py --output '研究记录/Sequential_DeepONet_new'",
        "/home/phl/anaconda3/envs/PINN/bin/python scripts/analyze_sequential_temperature_bias.py --output '研究记录/Sequential_DeepONet_new' --methods fnn gru lstm asl --splits train validation test --selection-from '研究记录/Sequential_DeepONet_1000epochs_temperature_tail/diagnostics/selection.json'",
        "/home/phl/anaconda3/envs/PINN/bin/python scripts/plot_sequential_deeponet.py --output '研究记录/Sequential_DeepONet_new'", "```", "",
        "同一次训练中断时添加 --resume。每种方法保存 initial.pt、best.pt、latest.pt、epoch_1000.pt、history.csv、training.log、final_info.json。evaluation 保存逐点预测 NPZ、逐工况 CSV、汇总表、模型 SHA256 与指标 JSON。"]
    (output/"对比总结.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"研究记录/ASL_sensor_plateau_candidate_3_20261004")
    args = parser.parse_args()
    output = args.output.resolve()
    cfg = json.loads((output/"config.json").read_text(encoding="utf-8"))
    if cfg["experiment"].get("asl_iteration"):
        from evaluate_asl_iteration import export
        export(output)
        print(f"Exported ASL revision PNGs to {output/'figures'}",flush=True)
        return
    evaluation,directory = output/"evaluation",output/"figures"
    directory.mkdir(exist_ok=True)
    appendix = directory/"appendix"
    appendix.mkdir(exist_ok=True)
    FIGURES.clear()
    SURFACE_MAXIMA.clear()
    plt.rcParams.update({"font.size":10,"axes.titlesize":11,"axes.labelsize":10,"legend.fontsize":9,
        "font.family":["Noto Sans CJK JP","DejaVu Sans"],"axes.spines.top":False,"axes.spines.right":False,
        "savefig.dpi":300})
    results = json.loads((evaluation/"metrics.json").read_text(encoding="utf-8"))
    cases = json.loads((evaluation/"case_metrics.json").read_text(encoding="utf-8"))
    simulation = np.load(evaluation/"simulation_predictions.npz")
    training_figures(output,directory)
    metric_figures(results,directory)
    test_power_figures(results,directory)
    histograms(cases,appendix)
    selections = percentile_fields(simulation,cases,appendix)
    write_json(evaluation/"percentile_field_selections.json",selections)
    shared_simulation(simulation,appendix)
    surface_fields(evaluation,directory)
    profile_figures(evaluation,directory)
    parity_figure(evaluation,results,directory)
    write_json(evaluation/"cloud_maxima.json",SURFACE_MAXIMA)
    report(output,results,FIGURES)
    write_json(directory/"manifest.json",dict(dpi=300,epoch=1000,formats=["png"],figures=FIGURES,
        plot_script_sha256=digest(Path(__file__)),png_sha256={name:digest(directory/name) for name in FIGURES}))
    print(f"Exported {len(FIGURES)} PNGs and {output/'对比总结.md'}",flush=True)


if __name__=="__main__":
    main()
