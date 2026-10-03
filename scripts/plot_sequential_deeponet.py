"""Export paper-style comparison figures from actual saved predictions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.tri as mtri
import numpy as np

from sequential_deeponet_core import METHODS,LABELS
from train_sequential_deeponet import ROOT,write_json,digest

COLORS = {"fnn":"#0072B2","gru":"#D55E00","lstm":"#009E73","asl":"#CC79A7"}
FIGURES = []


def save(fig,directory,name):
    fig.savefig(directory/(name+".png"),dpi=300,bbox_inches="tight",facecolor="white")
    fig.savefig(directory/(name+".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(fig)
    FIGURES.append(name+".png")


def history_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [{key:float(value) if value else np.nan for key,value in row.items()}
                for row in csv.DictReader(handle)]


def training_figures(output,directory):
    fig,axes = plt.subplots(1,3,figsize=(13.2,3.7),layout="constrained")
    for method in METHODS:
        history = history_rows(output/method/"history.csv")
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs,[row["loss"] for row in history],color=COLORS[method],alpha=.8,lw=1.,label=LABELS[method])
        valid = [row for row in history if np.isfinite(row.get("validation_score_k",np.nan))]
        axes[1].plot([row["epoch"] for row in valid],[row["validation_score_k"] for row in valid],
                     color=COLORS[method],lw=1.5,label=LABELS[method])
        axes[2].plot(epochs,[row["training_seconds"] for row in history],color=COLORS[method],label=LABELS[method])
    for ax,title in zip(axes,("Training objective","HF validation error","Elapsed training time")):
        ax.set(xlabel="Epoch",title=title,xlim=(0,1000))
        ax.grid(alpha=.2)
    axes[0].set(yscale="log",ylabel="Weighted normalized loss")
    axes[1].set(yscale="log",ylabel="Combined RMSE (K)")
    axes[2].set(ylabel="Time (s)")
    axes[2].legend(fontsize=9)
    save(fig,directory,"Figure_1_training_curves")


def metric_figures(results,directory):
    fig,axes = plt.subplots(2,3,figsize=(12.4,7.1),layout="constrained")
    definitions = [
        ("HF top-surface RMSE","high_test","顶部","rmse_k","RMSE (K)"),
        ("HF combined RMSE",None,None,"combined_test_rmse_k","RMSE (K)"),
        ("LF simulated full-field RMSE","low_test",None,"rmse_k","RMSE (K)"),
        ("HF top-surface relative L2","high_test","顶部","relative_l2_rise_pct","Relative L2 of temperature rise (%)"),
        ("Training cost",None,None,"training_seconds","Time (s)"),
        ("Trainable parameters",None,None,"parameters","Parameter count"),
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
    save(fig,directory,"Figure_2_accuracy_and_cost")


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
        axes[row,0].set_ylabel("LF full-field cases" if tag=="simulation" else "HF surface frames")
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
        artists.append(ax.tripcolor(triangulation,vals,cmap=cmap,norm=norm,shading="gouraud",rasterized=True))
    ax.plot([-25,-25,25,25],[0,-12,-12,0],color="white",lw=.5)
    ax.set(xlim=(-58.34,58.34),ylim=(-17.5,0),xlabel="Diameter coordinate (mm)",ylabel="z (mm)")
    ax.set_aspect("equal")
    return artists[-1]


def percentile_fields(simulation,cases,directory):
    selection_rows = []
    x,y = simulation["x"],simulation["y"]
    for method in METHODS:
        ordered = sorted(cases[method]["simulation"],key=lambda row:row["relative_l2_kelvin_pct"])
        selected = [ordered[int(round(q*(len(ordered)-1)))] for q in (0.,.9,1.)]
        masks = [np.isclose(x[:,3],row["power_w"])&np.isclose(x[:,2],row["time_s"]) for row in selected]
        temp_values = np.concatenate([np.r_[y[mask],simulation[method][mask]]-273.15 for mask in masks])
        errors = np.concatenate([np.abs(simulation[method][mask]-y[mask]) for mask in masks])
        norm = Normalize(temp_values.min(),temp_values.max())
        error_norm = Normalize(0.,max(1.,errors.max()))
        fig,axes = plt.subplots(3,3,figsize=(13.1,5.7),layout="constrained")
        for i,(label,row,mask) in enumerate(zip(("Best","90th percentile","Worst"),selected,masks)):
            for j,values in enumerate((y[mask]-273.15,simulation[method][mask]-273.15,
                                       np.abs(simulation[method][mask]-y[mask]))):
                artist = draw_section(axes[i,j],x[mask],values,error_norm if j==2 else norm,
                                      "magma" if j==2 else "viridis")
                axes[i,j].set_title(("Simulation reference","Prediction","Absolute error")[j],fontsize=10)
                axes[i,j].set_xlabel("Diameter coordinate (mm)" if i==2 else "")
                if j:
                    axes[i,j].set_ylabel("")
                if j in (1,2):
                    fig.colorbar(artist,ax=axes[i,j],fraction=.032,pad=.02,label="Error (K)" if j==2 else "Temperature (deg C)")
            axes[i,0].text(0.,1.42,f"{label}: {row['power_w']:g} W, {row['time_s']:g} s; L2 = {row['relative_l2_kelvin_pct']:.3f}%",
                           transform=axes[i,0].transAxes,fontsize=10,ha="left")
            selection_rows.append(dict(method=method,percentile=label,**row))
        fig.suptitle(f"{LABELS[method]} | held-out simulation fields | epoch 1000",fontsize=13)
        save(fig,directory,f"Figure_4_{method}_percentile_fields")
    return selection_rows


def shared_simulation(simulation,directory):
    x,y = simulation["x"],simulation["y"]
    mask = np.isclose(x[:,3],610.)&np.isclose(x[:,2],100.)
    all_temps = np.concatenate([simulation[key][mask] for key in METHODS]+[y[mask]])-273.15
    max_error = max(np.abs(simulation[key][mask]-y[mask]).max() for key in METHODS)
    norm,error_norm = Normalize(all_temps.min(),all_temps.max()),Normalize(0.,max_error)
    fig,axes = plt.subplots(4,3,figsize=(13.1,6.6),layout="constrained")
    for i,method in enumerate(METHODS):
        for j,values in enumerate((y[mask]-273.15,simulation[method][mask]-273.15,np.abs(simulation[method][mask]-y[mask]))):
            artist = draw_section(axes[i,j],x[mask],values,error_norm if j==2 else norm,"magma" if j==2 else "viridis")
            axes[i,j].set_title(("Simulation reference",LABELS[method],"Absolute error")[j],fontsize=10)
            axes[i,j].set_xlabel("Diameter coordinate (mm)" if i==3 else "")
            if j:
                axes[i,j].set_ylabel("")
            if j in (1,2):
                fig.colorbar(artist,ax=axes[i,j],fraction=.03,pad=.02,label="Error (K)" if j==2 else "Temperature (deg C)")
    fig.suptitle("Common held-out case: 610 W, 100 s | epoch 1000",fontsize=13)
    save(fig,directory,"Figure_5_common_full_field_and_errors")


def surface_fields(evaluation,directory):
    for time_s in (5,30,60,90,120):
        pack = np.load(evaluation/f"surface_634W_{time_s}s.npz")
        xy,y = pack["xy_mm"],pack["y"]
        unique_x,unique_y = np.unique(xy[:,0]),np.unique(xy[:,1])
        ix,iy = np.searchsorted(unique_x,xy[:,0]),np.searchsorted(unique_y,xy[:,1])
        all_temps = np.concatenate([y]+[pack[method] for method in METHODS])-273.15
        norm = Normalize(all_temps.min(),all_temps.max())
        error_norm = Normalize(0.,max(np.abs(pack[method]-y).max() for method in METHODS))
        fig,axes = plt.subplots(2,5,figsize=(14.2,6.3),layout="constrained")
        for i,key in enumerate(("reference",)+METHODS):
            values = y if key=="reference" else pack[key]
            grid = np.full((len(unique_y),len(unique_x)),np.nan)
            grid[iy,ix] = values-273.15
            im = axes[0,i].imshow(grid,origin="lower",extent=(-25,25,-25,25),cmap="viridis",norm=norm,
                                  interpolation="nearest")
            axes[0,i].set_title("Supplied IR-derived field" if key=="reference" else LABELS[key],fontsize=10)
            if i:
                error_grid = np.full_like(grid,np.nan)
                error_grid[iy,ix] = np.abs(values-y)
                err = axes[1,i].imshow(error_grid,origin="lower",extent=(-25,25,-25,25),cmap="magma",norm=error_norm)
                axes[1,i].set_title(f"RMSE = {np.sqrt(np.mean((values-y)**2)):.2f} K",fontsize=10)
            else:
                axes[1,i].axis("off")
        for ax in axes.flat:
            if ax.axison:
                ax.set(xlabel="x (mm)",ylabel="y (mm)",aspect="equal",xticks=(-25,0,25),yticks=(-25,0,25))
        fig.colorbar(im,ax=axes[0,:].tolist(),fraction=.016,pad=.015,label="Temperature (deg C)")
        fig.colorbar(err,ax=axes[1,1:].tolist(),fraction=.02,pad=.015,label="Absolute error (K)")
        fig.suptitle(f"Held-out 634 W top surface at {time_s} s | epoch 1000",fontsize=13)
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
        ax.plot(r,y[mask][order]-273.15,color="black",lw=2.,label="Supplied radial mean")
        for method in METHODS:
            ax.plot(r,top[method][mask][order]-273.15,color=COLORS[method],lw=1.4,label=LABELS[method])
        ax.set(title=f"{power:g} W, {time_s:g} s",xlabel="Radius (mm)",ylabel="Temperature (deg C)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig,directory,"Figure_7_top_radial_profiles")
    fig,axes = plt.subplots(3,3,figsize=(13.2,10.),layout="constrained")
    packs = [top,np.load(evaluation/"hot_predictions.npz"),np.load(evaluation/"cold_predictions.npz")]
    for i,(tag,pack) in enumerate(zip(("Top near center","Hot sensor","Cold sensor"),packs)):
        x,y = pack["x"],pack["y"]
        for j,power in enumerate((169.,339.,634.)):
            ids = np.flatnonzero(np.isclose(x[:,3],power))
            if i==0:
                ids = np.array([group[np.argmin(x[group,0])] for t in np.unique(x[ids,2])
                                for group in [ids[np.isclose(x[ids,2],t)]]])
            ids = ids[np.argsort(x[ids,2])]
            ax = axes[i,j]
            ax.plot(x[ids,2],y[ids]-273.15,"k-",lw=2,label="Supplied observation")
            for method in METHODS:
                ax.plot(x[ids,2],pack[method][ids]-273.15,color=COLORS[method],lw=1.3,label=LABELS[method])
            ax.set(title=f"{tag}, {power:g} W",xlabel="Time (s)",ylabel="Temperature (deg C)")
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


def report(output,results,figures):
    baseline = results["fnn"]["combined_test_rmse_k"]
    ranked = sorted(METHODS,key=lambda key:results[key]["combined_test_rmse_k"])
    lines = ["# 四种 DeepONet 的 1000 epochs 对比", "",
        "本次已真实完成 FNN、GRU、LSTM、ASL 各 1000 epochs，随机种子为 123，全部关闭边界注意力。主表统一使用第 1000 轮模型。", "",
        "## 测试结果", "",
        "| 方法 | 顶部 RMSE / K | 热端 RMSE / K | 冷端 RMSE / K | 综合 RMSE / K | 仿真全场 RMSE / K | 训练时间 / s | 参数量 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        value = results[method]
        high = value["high_test"]
        lines.append(f"| {LABELS[method]} | {high['顶部']['rmse_k']:.4f} | {high['热端']['rmse_k']:.4f} | "
            f"{high['冷端']['rmse_k']:.4f} | {value['combined_test_rmse_k']:.4f} | {value['low_test']['rmse_k']:.4f} | "
            f"{value['training_seconds']:.1f} | {value['parameters']:,} |")
    lines += ["", "综合 RMSE = sqrt(0.5 × 顶部 RMSE² + 0.25 × 热端 RMSE² + 0.25 × 冷端 RMSE²)。温差 1 K 等于 1 ℃。", "",
        "按第 1000 轮实验测试综合 RMSE 排序："+" < ".join(LABELS[key] for key in ranked)+"。"]
    for method in METHODS[1:]:
        delta = 100*(baseline-results[method]["combined_test_rmse_k"])/baseline
        lines.append(f"{LABELS[method]} 相对 FNN 的综合 RMSE {'降低' if delta>=0 else '增加'} {abs(delta):.2f}%。")
    lf_ranked = sorted(METHODS,key=lambda key:results[key]["low_test"]["rmse_k"])
    top_ranked = sorted(METHODS,key=lambda key:results[key]["high_test"]["顶部"]["rmse_k"])
    lf_delta = 100*(results["fnn"]["low_test"]["rmse_k"]-results["asl"]["low_test"]["rmse_k"])/results["fnn"]["low_test"]["rmse_k"]
    time_ratio = results["asl"]["training_seconds"]/results["fnn"]["training_seconds"]
    lines += ["", "## 比较分析", "",
        f"仿真全场 RMSE 最低的是 {LABELS[lf_ranked[0]]}，实验顶部 RMSE 最低的是 {LABELS[top_ranked[0]]}。"
        "两种评价分别衡量对仿真内部场的拟合和对实验表面场的重构，应分别解读。",
        f"ASL 相对 FNN 的仿真全场 RMSE {'降低' if lf_delta>=0 else '增加'} {abs(lf_delta):.2f}%，"
        f"本机训练时间约为 FNN 的 {time_ratio:.2f} 倍。更复杂的序列分支需同时考虑空间误差和计算成本。",
        "GRU/LSTM 的最终轮与验证最优轮结果可用于检查后期泛化是否下降；训练目标含多个监督项和物理约束，"
        "目标下降不保证所有测试功率的顶部误差同步下降。当前结果不足以确定具体成因，也不能从单种子比较推断普遍优劣。"]
    lines += ["", "## 论文对应指标", "",
        "| 方法 | 顶部相对 L2 / %（K 分母） | 顶部相对 L2 / %（温升分母） | 顶部 R² | 仿真全场相对 L2 / %（K 分母） | 仿真全场 R² |",
        "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        high,low = results[method]["high_test"]["顶部"],results[method]["low_test"]
        lines.append(f"| {LABELS[method]} | {high['relative_l2_kelvin_pct']:.4f} | {high['relative_l2_rise_pct']:.4f} | "
            f"{high['r2']:.6f} | {low['relative_l2_kelvin_pct']:.4f} | {low['r2']:.6f} |")
    lines += ["", "相对 L2 = 100 × ||预测−参考||₂ / ||参考||₂；温升分母为 ||参考−295.15 K||₂。K 分母会被初温抬高，不能与采用摄氏度分母的论文数值直接比较。直方图按每个功率/时刻工况计算；工况间标准差不是多随机种子的统计不确定性。", "",
        "## 仿真材料分区误差", "",
        "| 方法 | 铜区 RMSE / K | SiC 区 RMSE / K |", "|---|---:|---:|"]
    for method in METHODS:
        regions = results[method]["low_test_by_material"]
        lines.append(f"| {LABELS[method]} | {regions['copper']['rmse_k']:.4f} | {regions['sic']['rmse_k']:.4f} |")
    lines += ["", "全场指标按现有有限元节点等权汇总，两个材料的节点数不同，分区表补充展示各自精度。", "",
        "## 验证最优模型（补充）", "",
        "| 方法 | 验证选定轮次 | 对应测试综合 RMSE / K |", "|---|---:|---:|"]
    for method in METHODS:
        value = results[method]
        lines.append(f"| {LABELS[method]} | {value['best_epoch']} | {value['best_combined_test_rmse_k']:.4f} |")
    lines += ["", "训练没有提前终止。最优轮次只由验证集选择，以上补充表不会替换主表的第 1000 轮结果。", "",
        "## 实现与比较口径", "",
        "参考 [S-DeepONet 论文](https://arxiv.org/abs/2306.08218)及[作者代码](https://github.com/Jasiuk-Research-Group/S-DeepONet)，将序列编码器作为分支网络，与 FNN 主干输出做点积。GRU/LSTM 使用长历史编码器和局部历史解码器，隐藏宽度 48；这是适配本项目输入的实现，未照搬原文 101 点负载及其大型编码器/解码器层数。", "",
        "ASL 来自[用户 ASL-PINN 仓库](https://github.com/fei4440273/Temperature-Reconstruction-ASL-PINN)，提交 4477c971007624133eec92bd7eeb5bb58d3b9c26。保留两个选择性 SSM、因果卷积、输入相关 Δ/B/C、稳定对角 A、D 跳连、门控残差、长历史升温率特征、SSM 终态初始化 LSTM 的 h/c，以及成熟度门控持续上下文。相同权重下的分支输出已与原始源码核对。主干与高/低保真输出头采用相同初始权重。边界注意力未导入或实例化。", "",
        "共同输入为 32 点长历史与 8 点局部历史，每点包含时间、功率、hot 温度、cold 温度和有效性。局部窗口 20 s，历史截止于 t−1 s；插值也只允许使用截止时刻之前的观测。初温统一为 22 ℃，时间/功率尺度为本项目的 200 s / 800 W。", "",
        "仿真按现有配置使用 60/10/10 个训练/验证/测试功率；实验使用 12/3/3 个功率，实验测试功率为 169、339、634 W。仿真传感器历史来自最接近两个底面传感器位置的铜网格节点；对应节点坐标保存在 JSON。", "",
        "每轮按全部训练功率分层随机采样，低保真每个功率/材料 8 点，顶部每功率 64 点，两端每功率各 16 点，并计算传热方程、边界、初值、界面和热阻先验损失，再执行一次 AdamW 更新。因此 1000 epochs 指当前项目的采样训练轮次，不是对全部 702 万个仿真点循环 1000 次。所有方法使用同一配置、采样种子和学习率计划。", "",
        "## 结果适用范围", "",
        "这是 T(r,z,t,P | 过去 hot/cold 测量) 的条件重构。测试功率的过去传感器测量用于推理输入，当前/未来温度不进入输入；因此传感器误差含历史相关信息，顶部误差更适合评估从测量到空间场的泛化。PDE 对坐标求导时保持历史输入固定，沿用 ASL 源码定义，不能把该残差解释为沿整个更新历史的总时间导数。", "",
        "本项目为恒定激光加热数据，没有据此验证任意变动负载。仿真内部温度只作为低保真全场参考；实验没有内部全场标签。二维顶部图使用用户提供的 IR 派生温度 CSV，保留 is_recovered 标志，不把已恢复像素当成独立原始辐射测量。", "",
        "本轮为单一随机种子，结论限定于这次 1000 轮预算和数据配置，不能据此断言某种结构在统计上始终更优。训练时间包含定期验证和检查点之间的运行开销；推理时间仅为本机单次测试集计时，不外推有限元加速比。", "",
        "正式训练前完成了短运行；首轮完整运行在源码审查中发现查询时间门控梯度被缓存截断，已归档为 diagnostic。随后仅修复求导定义，保持架构、损失权重、数据划分和优化配置不变，并重新训练全部四种方法。诊断轮测试结果曾被查看，因此本报告不宣称从未查看过测试功率的盲测。", "",
        "## 图像与复现", "",
        "PNG 均为 300 DPI，并保存同名 PDF。误差直方图及最好/90 分位/最坏工况图对应论文的同类展示。百分位按每种模型自己的工况误差排序，排除 t=0 的平凡初始场；共同工况图固定为 610 W、100 s。顶部二维图固定为 634 W 的 5、30、60、90、120 s。", ""]
    lines.extend(f"- [{name}](figures/{name})" for name in figures)
    lines += ["", "```bash", "/home/phl/anaconda3/envs/PINN/bin/python scripts/train_sequential_deeponet.py --device cuda --output '研究记录/Sequential_DeepONet_new'",
        "/home/phl/anaconda3/envs/PINN/bin/python scripts/evaluate_sequential_deeponet.py --device cuda --output '研究记录/Sequential_DeepONet_new'",
        "/home/phl/anaconda3/envs/PINN/bin/python scripts/plot_sequential_deeponet.py --output '研究记录/Sequential_DeepONet_new'", "```", "",
        "同一次训练中断时添加 --resume。每种方法保存 initial.pt、best.pt、latest.pt、epoch_1000.pt、history.csv、training.log、final_info.json。evaluation 保存逐点预测 NPZ、逐工况 CSV、汇总表、模型 SHA256 与指标 JSON。"]
    (output/"对比总结.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"研究记录/Sequential_DeepONet_1000epochs")
    args = parser.parse_args()
    output = args.output.resolve()
    evaluation,directory = output/"evaluation",output/"figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.titlesize":11,"axes.labelsize":10,"legend.fontsize":9,
        "font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,
        "savefig.dpi":300,"pdf.fonttype":42})
    results = json.loads((evaluation/"metrics.json").read_text(encoding="utf-8"))
    cases = json.loads((evaluation/"case_metrics.json").read_text(encoding="utf-8"))
    simulation = np.load(evaluation/"simulation_predictions.npz")
    training_figures(output,directory)
    metric_figures(results,directory)
    histograms(cases,directory)
    selections = percentile_fields(simulation,cases,directory)
    write_json(evaluation/"percentile_field_selections.json",selections)
    shared_simulation(simulation,directory)
    surface_fields(evaluation,directory)
    profile_figures(evaluation,directory)
    parity_figure(evaluation,results,directory)
    report(output,results,FIGURES)
    write_json(directory/"manifest.json",dict(dpi=300,epoch=1000,figures=FIGURES,
        plot_script_sha256=digest(Path(__file__)),png_sha256={name:digest(directory/name) for name in FIGURES}))
    print(f"Exported {len(FIGURES)} PNGs, matching PDFs, and {output/'对比总结.md'}",flush=True)


if __name__=="__main__":
    main()
