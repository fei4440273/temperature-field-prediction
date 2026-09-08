# SiC–Cu 激光烧蚀多保真瞬态温度场预测项目工作方案

> **用途**：本文件用于指导 Codex 完成数据审计、数据划分、基线模型、候选模型、训练、验证、测试、可视化和消融实验。  
> **最终目标**：输入任意激光功率 \(P\)，预测整个加热过程中的 SiC–Cu 轴对称瞬态温度场 \(T(r,z,t,P)\)，并通过轴对称旋转恢复三维温度场 \(T(x,y,z,t,P)\)。  
> **工程特色**：多保真数据融合 + SiC–Cu 跨材料传热 + 少量高保真实验 + 大量低保真仿真 + 稠密表面红外观测 + 稀疏底部 Hot/Cold 观测。

---

## 1. 数据现状

GitHub：`https://github.com/fei4440273/temperature-field-prediction-data`

### 1.1 Simulation data：低保真完整场

- 激光功率：10–800 W，间隔 10 W，共 80 个工况。
- 时间：0–200 s，间隔 2 s，共 101 帧。
- 每帧 1159 个轴对称节点：Cu 821 个、SiC 338 个。
- 坐标单位：mm；温度单位：°C。
- 提供 SiC + Cu 全域温度场。
- 定义为低保真温度场 \(T_{LF}\)。

### 1.2 Experiment data：高保真 SiC 表面场

实验功率：

```text
115.2, 216.8, 254.5, 309.0, 364.3,
403.0, 494.2, 558.5, 593.5, 630.5 W
```

特点：

- 共 10 个真实功率工况、264 个 CSV。
- 每帧为 SiC 顶部二维红外温度场。
- 不同功率有效时间长度不同。
- 当前所有像素 `is_recovered=True`，不能把每个像素无差别视为完全可靠真值。
- 建议转换为径向统计观测：
  - 环平均温度 \(\mu_T(r,t,P)\)
  - 环内标准差 \(\sigma_T(r,t,P)\)
  - 有效像素数 \(N(r,t,P)\)

### 1.3 hotdata / colddata：底部高保真传感器

两个目录均包含 17 个功率：

```text
50, 115.2, 216.8, 250, 254.5, 309, 350, 364.3,
403, 494.2, 500, 558.5, 593.5, 600, 630.5, 700, 800 W
```

实验位置按现有说明：

- Hot：\(r=0.028\,m\)
- Cold：\(r=0.0415\,m\)

文件结构：

- 前两列 `X`,`Y`
- 后续列 `t=1 ... t=n`
- 不同功率约 100–160 个时间列

**阻塞项**：仓库 README 明确说明 Hot/Cold 的坐标单位、时间单位和数值单位尚未在源文件中显式记录。Codex 不得自行假设，必须在 `reports/data_audit.md` 中列为待确认项。

### 1.4 几何

Cu：

- \(R_{Cu}=0.05834\,m\)
- \(H_{Cu}=0.0175\,m\)

SiC：

- \(R_{SiC}=0.025\,m\)
- \(H_{SiC}=0.012\,m\)

第一版采用轴对称模型：

\[
T=T(r,z,t,P)
\]

最终通过

\[
x=r\cos\theta,\qquad y=r\sin\theta
\]

恢复三维场。

---

# 2. 数据划分总原则

## 2.1 禁止随机按像素、节点或时间点拆分高保真实验

同一功率内部的像素、时间帧高度相关。若把 309 W 的部分时间/像素用于训练、其余用于测试，会造成严重信息泄漏。

因此：

> **训练、验证、测试必须按“完整功率工况”分组。**

某个功率进入测试集后，该功率对应的：

- IR
- Hot
- Cold

全部禁止参与训练和调参。

---

# 3. “留一功率”是否科学

## 3.1 结论

可以采用，但规范名称应理解为：

- Leave-One-Group-Out（LOGO）
- 本项目中 group = power condition
- 工程论文中也常称 Leave-One-Condition-Out（LOCO）

它不是临时发明的方法。

Scikit-learn 官方文档明确指出，LeaveOneGroupOut 可以把“不同实验”作为不同 group，每轮完整留出一个实验：

https://scikit-learn.org/stable/modules/cross_validation.html

2026 年已有 physics-constrained 工程代理模型采用 leave-one-condition-out 来避免同一运行条件泄漏到训练和测试：

`A high-fidelity physics-constrained deep learning framework for cross-condition wake prediction in tidal current turbines`, Energy, 2026.

## 3.2 本项目不把 LOPO/LOCO 作为唯一划分

原因：

- 10 个 IR 工况少，LOGO 对最终稳健性评估很有价值；
- 但对所有模型都做 10 次完整 PINN 重训练代价很大；
- 模型研发阶段需要固定验证集来快速比较。

因此采用：

> **固定工况级 train/validation/test 作为主实验 + LOGO 作为最终稳健性复核。**

---

# 4. 第一阶段：Simulation-only 未见功率筛选

目的：先比较网络结构本身是否具有 \(P_{unseen}\rightarrow T(r,z,t)\) 泛化能力。

Simulation 共 80 个功率。

## 4.1 Test powers（10个）

```text
50, 130, 210, 290, 370, 450, 530, 610, 690, 770 W
```

## 4.2 Validation powers（10个）

```text
90, 170, 250, 330, 410, 490, 570, 650, 730, 800 W
```

## 4.3 Training powers

剩余 60 个功率。

### 硬规则

- 一个功率的 101 个时间帧全部属于同一个 split。
- 一个功率的 1159 个空间节点全部属于同一个 split。
- 所有方法必须使用同一个 split。

### 评价指标

- RMSE
- MAE
- \(R^2\)
- Relative \(L_2\)
- 最大温度误差
- 0–30 s RMSE
- 30–100 s RMSE
- 100–200 s RMSE
- SiC RMSE
- Cu RMSE
- 全域 RMSE

---

# 5. 高保真固定主划分

10 个 IR 功率采用 6/2/2 工况级划分。

## 5.1 Train

```text
115.2
254.5
364.3
403.0
558.5
630.5
```

## 5.2 Validation

```text
216.8
494.2
```

## 5.3 Test

```text
309.0
593.5
```

选择逻辑：

- train 覆盖低、中、高功率；
- validation 位于区间内部；
- test 是训练范围内的完全未见功率，主要测试插值泛化能力；
- 不用极端端点作为主 test，避免把插值和外推混为一谈。

### 数据泄漏断言

若 309 W 为 test：

- 所有 309 W IR 禁止训练；
- `HotData-309W.csv` 禁止训练；
- 对应 Cold 禁止训练。

593.5 W 同理。

---

# 6. Hot/Cold 的使用策略

## 6.1 与 IR 重合的 10 个功率

```text
115.2, 216.8, 254.5, 309, 364.3,
403, 494.2, 558.5, 593.5, 630.5 W
```

完全跟随 IR split：

- IR train → Hot/Cold 可训练
- IR val → Hot/Cold 只能验证
- IR test → Hot/Cold 只能最终测试

## 6.2 只有 Hot/Cold、没有 IR 的 7 个功率

```text
50, 250, 350, 500, 600, 700, 800 W
```

第一版全部保留为 **External Sensor Test**，不参加高保真训练。

这样可以检验：

> 在没有 SiC 红外监督的功率下，模型是否能预测 Cu 底部温升。

分为：

### IR 覆盖区间内部传感器测试

```text
250, 350, 500, 600 W
```

### 相对 IR 高保真区间外部测试

```text
50, 700, 800 W
```

注意：这里“外部/外推”是相对 IR 实验覆盖范围，不是相对 10–800 W simulation 范围。

---

# 7. 最终 LOGO/LOPO 稳健性验证

只对以下模型做：

- MLP-PINN 基线
- 最优模型
- 计算资源允许时加第二名

10 个 IR power 每次留 1 个完整 power 做 test。

每一 fold：

- test power 的 IR + Hot + Cold 全部删除；
- 其余 9 个 power 训练；
- **不重新搜索超参数**；
- 网络深度、学习率、训练轮数等采用主实验已经锁定的配置。

最终报告：

\[
RMSE=mean\pm std
\]

\[
MAE=mean\pm std
\]

并报告每个 power 的逐工况误差。

---

# 8. 统一 Physics Module

所有 PINN 类方法共用相同物理模块，以保证公平比较。

轴对称热传导：

\[
\rho c_p\frac{\partial T}{\partial t}
=
\frac{1}{r}\frac{\partial}{\partial r}
\left(rk\frac{\partial T}{\partial r}\right)
+
\frac{\partial}{\partial z}
\left(k\frac{\partial T}{\partial z}\right)
\]

SiC、Cu 分别使用自己的 \(\rho,c_p,k\)。

如果有温变物性：

\[
k=k(T),\qquad c_p=c_p(T)
\]

不要擅自改成常数。

## 8.1 SiC–Cu 跨材料条件

跨材料传热是工程特色，但不是网络方法名称。

至少满足热流连续：

\[
q_{SiC}=q_{Cu}
\]

若确认存在接触热阻：

\[
T_{SiC}-T_{Cu}=R_{tc}q_n
\]

## 8.2 初始条件

\[
T(r,z,0,P)=T_0
\]

## 8.3 激光边界

若确认 Gaussian beam：

\[
q(r,P)=\frac{2\eta P}{\pi w^2}
\exp\left(-\frac{2r^2}{w^2}\right)
\]

**吸收率 \(\eta\)、光斑 \(w\) 未确认前禁止硬编码。**

---

# 9. 统一多保真框架

所有候选方法都在相同 multi-fidelity protocol 中比较。

推荐：

\[
T_{HF}=T_{LF}+\Delta T
\]

其中：

- \(T_{LF}\)：由 Simulation data 学到的低保真场；
- \(\Delta T\)：真实实验对仿真场的校正。

禁止把海量 simulation 点和少量 experiment 点简单拼接后做同权重 MSE。

统一采用：

### Stage A：LF pretraining

只用 simulation。

### Stage B：HF correction

冻结或部分冻结 LF backbone，使用：

- IR train powers
- Hot/Cold train powers
- physics loss

学习 correction。

### Stage C：joint fine-tuning

小学习率联合微调。

---

# 10. 第一批必须实现的方法

方法名称只表示“网络/算法”，不要把 MF、跨材料写进方法简称。

---

## 10.1 MLP-PINN

### 作用

统一 PINN 基线，必须第一个完成。

### 输入

```text
r, z, t, P
```

### 输出

```text
T
```

### 初始结构

```text
4 → 128 → 128 → 128 → 128 → 128 → 1
```

优先测试：

- tanh
- SiLU

必须完成：

- simulation-only benchmark
- multi-fidelity benchmark
- PDE residual
- IR prediction
- Hot/Cold prediction

任何高级方法必须显著超过 MLP-PINN 才有保留价值。

---

## 10.2 LSTM-PINN

### 目标

检验显式时间记忆是否改善瞬态预测。

建议：

- 空间/功率：MLP encoder
- 时间历史：LSTM
- fusion 后预测 T

时间窗口先测试：

```text
8 / 16 / 32
```

重点比较：

- 0–30 s RMSE
- 全程 RMSE
- Tmax error

若只改善早期瞬态但全场不提升，则作为未来组合模块，不直接作为最终主模型。

参考：

Physics-informed LSTM for inverse modeling transient temperature field in frozen soils under data-scarce conditions, 2026.

https://doi.org/10.1016/j.geoai.2026.100107

---

## 10.3 POD-PINN

### 目标

检验温度场是否具有低秩结构，从而降低小样本学习难度。

对 simulation snapshots 做 POD/SVD：

\[
T(r,z,t,P)\approx \bar T+
\sum_{k=1}^{K}a_k(t,P)\phi_k(r,z)
\]

必须先画累计能量曲线：

```text
K = 5, 10, 20, 30, 50
```

报告：

```text
95%
99%
99.9%
```

PINN 主要预测：

```text
(P,t) → [a1,...,aK]
```

同时比较：

1. Global POD
2. Material-wise POD（SiC/Cu 分开 POD）

跨材料问题中第二种可能更合理。

---

## 10.4 DeepONet-PINN

### 目标

重点解决：

\[
P_{unseen}\rightarrow T(r,z,t)
\]

### Branch

第一版：

```text
P
```

后续建议升级为采样后的激光热流函数 `q(r;P)`。

### Trunk

```text
r, z, t
```

优先参考 **Res-DeepONet**，而非 vanilla DeepONet。

高度相关文献：

Deep operator networks for bioheat transfer problems with parameterized laser source functions, IJHMT, 2024.

https://doi.org/10.1016/j.ijheatmasstransfer.2024.125659

代码：

https://github.com/adi-roy/Res-DeepONet

该工作同样研究 parameterized laser heating，因此是本项目的重要实现参考。

重点指标：

- simulation unseen-power RMSE
- IR unseen-power RMSE
- inference speed
- full-field relative error

---

## 10.5 GNO-PINN

### 目标

利用原始 FEM 非规则节点，避免为了 FNO 强行规则网格插值。

Graph：

```text
node = FEM node
feature = [r, z, material_id, P, t, optional T_LF]
edge = FEM adjacency 或 KNN
```

如果没有 element connectivity：

```text
KNN k = 6~12
```

必须保证节点连接具有物理空间意义。

第一版以预测精度为主，不要一开始实现过度复杂的 graph differential operator。

若 simulation unseen-power benchmark 不优于 DeepONet-PINN / POD-PINN，则暂停此路线。

---

# 11. 第二阶段候选方法

不要求第一轮全部实现。

## 11.1 FNO/PINO

优点：field-to-field operator learning 强。

缺点：当前 FEM 非规则，需要插值到规则 r-z 网格，可能引入插值误差。

## 11.2 KAN-PINN

作为 MLP backbone 替换实验，不作为第一优先创新。

## 11.3 Diffusion-PINN

第一阶段不建议直接用于确定性 \(P\rightarrow T\) forward prediction。

更合理的后期用途：

```text
T_LF + sparse HF observations
→ Conditional Diffusion Corrector
→ distribution of T_HF
```

适合：

- 稀疏反演
- 不确定性量化
- 病态问题

只有最优确定性模型完成后再考虑。

---

# 12. 必须加入的工程基线

## 12.1 FEM Power Linear Interpolation

例如 36 W：

\[
T_{36}=0.4T_{30}+0.6T_{40}
\]

若 FEM 所有功率使用相同节点拓扑和时间网格，可直接实现。

## 12.2 Cubic/Spline Power Interpolation

可作为进一步传统基线。

目的：

> 高级神经网络不能只超过 PINN，还必须证明相比简单仿真功率插值有实际价值。

---

# 13. Hot/Cold 数据预处理

Codex 必须逐文件检查：

1. 行数
2. 列数
3. X/Y 范围
4. 计算 \(r=\sqrt{X^2+Y^2}\) 的均值/标准差
5. 是否接近 0.028 / 0.0415
6. 同一时间列圆周温度是否完全一致
7. 若一致，只保留一条 ring-average time series，禁止把数百个重复坐标当独立传感器监督
8. NaN
9. 时间列数量
10. 初始温度偏置

同时生成：

```text
absolute_temperature
delta_temperature = T(t) - T(t0)
```

传感器 loss 可写为：

\[
L_{sensor}=\lambda_{abs}L_{abs}
+\lambda_{\Delta T}L_{\Delta T}
\]

初始建议：

```text
lambda_abs = 0.2
lambda_delta = 1.0
```

但最终必须由 validation powers 调整。

---

# 14. IR 预处理

轴对称第一版不要直接把全部二维像素喂给 PINN。

计算：

\[
r=\sqrt{(x-x_c)^2+(y-y_c)^2}
\]

初始径向 bin：

```text
dr = 0.25 mm
```

可配置。

输出统一表：

```text
power
time
r
mean_temperature
std_temperature
n_pixels
```

可用：

\[
w=\frac{1}{\sigma_T^2+\epsilon}
\]

作为观测可靠性权重，但必须 clipping，防止 \(\sigma\to0\) 权重爆炸。

原始二维 IR 数据必须保留用于最终可视化比较。

---

# 15. 统一 Loss

基础形式：

\[
L=
\lambda_{LF}L_{LF}
+\lambda_{IR}L_{IR}
+\lambda_{sensor}L_{sensor}
+\lambda_{PDE}L_{PDE}
+\lambda_{BC}L_{BC}
+\lambda_{IC}L_{IC}
+\lambda_{interface}L_{interface}
\]

要求：

- 所有方法共用同一 physics loss；
- 初版用固定权重；
- 第二阶段才实现 gradient-based adaptive loss；
- 不要一开始同时加大量复杂训练技巧，否则无法判断结构改进的真实贡献。

---

# 16. 公平比较规则

统一随机种子：

```text
0, 1, 2, 3, 4
```

每种核心方法至少 5 次独立训练。

统一：

- train/val/test power
- preprocessing
- physics collocation set
- optimizer
- 最大 epoch
- early stopping
- evaluation grid
- metrics

最终报告 `mean ± std`，禁止只选最好一次。

---

# 17. 模型筛选流程

## Round 0：Data Audit

无高级模型训练。

必须生成：

```text
reports/data_audit.md
configs/data_metadata.yaml
```

## Round 1：Simulation-only

比较：

- FEM linear interpolation
- 数据驱动 MLP
- MLP-PINN
- LSTM-PINN
- POD-PINN
- DeepONet-PINN
- GNO-PINN

筛选前 3。

## Round 2：Multi-fidelity HF correction

前 3 + MLP-PINN baseline。

使用：

- Simulation
- train IR
- train Hot/Cold
- Physics

## Round 3：External Sensor Test

完全不训练：

```text
50, 250, 350, 500, 600, 700, 800 W Hot/Cold
```

测试隐藏 Cu 区域泛化。

## Round 4：LOGO robustness

只运行：

- MLP-PINN
- 最优模型
- 可选第二名

## Round 5：最终组合创新

根据前面结果再组合，而不是提前硬拼。

例如只有当：

- DeepONet 未见功率最好；
- LSTM 早期瞬态明显更好；

才考虑 DeepONet + temporal encoder + PINN。

---

# 18. 评价指标

## 全场

- RMSE
- MAE
- \(R^2\)
- Relative \(L_2\)

## 局部

- SiC RMSE
- Cu RMSE
- IR surface RMSE
- Hot RMSE
- Cold RMSE

## 时间分段

- 0–30 s RMSE
- 30–100 s RMSE
- >100 s RMSE

## 工程指标

- Tmax error
- Tmax location error
- 末帧/稳态误差
- Hot/Cold 最大温升误差
- Hot/Cold 曲线相关系数

## 计算成本

- training time
- inference time
- parameter count
- peak GPU memory

---

# 19. 当前还缺少/必须确认的内容

这些是正式 PINN 训练前的关键输入。

## 19.1 边界条件

需要提供或确认：

1. SiC 吸收率 \(\eta\)
2. 激光光斑半径/直径
3. 光斑分布：Gaussian / top-hat / measured
4. 激光是否全程恒定功率
5. 水冷边界：定温 or 对流
6. 等效换热系数
7. 环境温度
8. 外表面对流
9. 是否考虑辐射
10. SiC–Cu 接触热阻
11. Simulation 中以上参数的真实设置

## 19.2 热物性

需要：

### SiC

- \(\rho(T)\)
- \(k(T)\)
- \(c_p(T)\)

### Cu

- \(\rho(T)\)
- \(k(T)\)
- \(c_p(T)\)

若目前只有常数，先实现常数版，但配置必须支持温变物性。

## 19.3 Hot/Cold metadata

必须确认：

- X/Y 单位
- 温度单位
- `t=1` 是 1 s 还是采样序号
- 是否为真实测量后的环平均结果
- 不同功率时间起点是否同步

写入：

```text
configs/data_metadata.yaml
```

## 19.4 IR metadata

必须确认：

- 图像中心 \(x_c,y_c\)
- 文件名中的 `50mm` 含义
- 红外发射率设置
- 是否完成背景/镜头标定
- 每个功率时间帧对应关系
- 初始温度同步方式

## 19.5 真实内部场验证局限

目前没有 SiC/Cu 内部完整实验真值。

因此论文中：

- 内部场不能声称“直接实验真值验证”；
- IR 验证 SiC 顶面；
- Hot/Cold 独立验证 Cu 底面；
- PDE residual 验证物理一致性；
- Simulation 只作为低保真参考，不等同真实内部真值。

---

# 20. 推荐工程目录

```text
temperature-field-prediction/
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt
├── configs/
│   ├── geometry.yaml
│   ├── materials.yaml
│   ├── boundary_conditions.yaml
│   ├── data_metadata.yaml
│   ├── splits.yaml
│   └── training.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
├── src/
│   ├── data/
│   │   ├── audit.py
│   │   ├── simulation.py
│   │   ├── experiment.py
│   │   ├── sensors.py
│   │   ├── splits.py
│   │   └── normalization.py
│   ├── physics/
│   │   ├── heat_equation.py
│   │   ├── boundary.py
│   │   ├── interface.py
│   │   └── materials.py
│   ├── models/
│   │   ├── mlp_pinn.py
│   │   ├── lstm_pinn.py
│   │   ├── pod_pinn.py
│   │   ├── deeponet_pinn.py
│   │   ├── gno_pinn.py
│   │   └── common.py
│   ├── losses/
│   ├── train/
│   ├── eval/
│   └── visualization/
├── scripts/
│   ├── 00_audit_data.py
│   ├── 01_build_processed_data.py
│   ├── 02_run_simulation_benchmark.py
│   ├── 03_train_multifidelity.py
│   ├── 04_external_sensor_test.py
│   ├── 05_logo_cv.py
│   └── 06_predict_power.py
├── reports/
├── checkpoints/
└── tests/
```

---

# 21. Codex 实施顺序

## Milestone 1：数据审计

**不要先写 DeepONet/LSTM/GNO。**

完成：

- clone + Git LFS 检查
- Simulation parser
- Experiment parser
- Hot/Cold parser
- 单位/坐标/时间检查
- 文件完整性
- `reports/data_audit.md`

## Milestone 2：统一数据层

Simulation：

```text
power,time,r,z,material,temperature
```

IR：

```text
power,time,r,temperature_mean,temperature_std
```

Sensor：

```text
power,time,sensor_type,r,temperature,delta_temperature
```

## Milestone 3：FEM interpolation + MLP baseline

## Milestone 4：MLP-PINN

先确认 PDE autodiff、material mask、boundary sampling、interface loss 正确。

## Milestone 5：LSTM-PINN + POD-PINN

## Milestone 6：DeepONet-PINN

优先参考 Res-DeepONet 的架构和训练流程，但不要直接复制其生物热参数。

## Milestone 7：GNO-PINN

## Milestone 8：统一 Multi-fidelity correction

## Milestone 9：External sensor test + LOGO

## Milestone 10：任意功率预测接口

示例：

```bash
python scripts/06_predict_power.py \
  --power 36 \
  --t-start 0 \
  --t-end 200 \
  --dt 1 \
  --checkpoint checkpoints/best.pt
```

输出：

```text
predictions/36W/field_rzt.npz
predictions/36W/vtk/*.vtk
predictions/36W/hot_cold.csv
predictions/36W/figures/*.png
```

---

# 22. Codex 编码约束

1. PyTorch。
2. 所有随机种子可配置。
3. 所有物理参数从 YAML 读取；未知参数禁止硬编码。
4. 原始数据只读。
5. 各模型共享同一个 data layer、physics module、metrics。
6. 保存 train/val loss、各 loss component、learning rate、epoch。
7. 保存 best checkpoint。
8. 保存 config snapshot。
9. 实验结果输出 CSV/JSON。
10. 支持 CPU 调试、CUDA 训练。
11. 单元测试至少覆盖：
   - coordinate conversion
   - power split leakage
   - PDE residual shape
   - material mask
   - sensor ring radius
12. 若 test power 出现在任何 HF training loader 中，立即 raise error。

---

# 23. 最终核心实验表

| Method | Simulation unseen RMSE | IR test RMSE | Hot RMSE | Cold RMSE | Relative L2 | Params | Inference |
|---|---:|---:|---:|---:|---:|---:|---:|
| FEM linear interp | | - | | | | - | |
| MLP | | | | | | | |
| MLP-PINN | | | | | | | |
| LSTM-PINN | | | | | | | |
| POD-PINN | | | | | | | |
| DeepONet-PINN | | | | | | | |
| GNO-PINN | | | | | | | |

LOGO 表：

| Held-out Power | MLP-PINN RMSE | Best Model RMSE | Improvement |
|---:|---:|---:|---:|
| 115.2 | | | |
| ... | | | |
| 630.5 | | | |
| Mean ± Std | | | |

---

# 24. 研究决策原则

本项目不预设最终一定是 DeepONet、LSTM、POD 或 GNO。

最终方法由以下结果共同决定：

1. 未见功率全场精度；
2. 高保真 IR 精度；
3. Hot/Cold 外部验证；
4. SiC–Cu 物理一致性；
5. 训练稳定性；
6. 推理效率；
7. 是否显著优于 MLP-PINN 与 FEM interpolation。

最终创新结构必须从 benchmark 结果中产生，而不是把热门模块直接堆叠。

---

# 25. 参考文献与代码

### Leave-One-Group-Out

https://scikit-learn.org/stable/modules/cross_validation.html

### Res-DeepONet / Parameterized laser heating

Roy et al., International Journal of Heat and Mass Transfer, 2024, 228, 125659.

https://doi.org/10.1016/j.ijheatmasstransfer.2024.125659

https://github.com/adi-roy/Res-DeepONet

### Physics-informed LSTM / transient temperature field

https://doi.org/10.1016/j.geoai.2026.100107

---

# 26. Codex 第一个任务

**先不要实现高级模型。**

首先执行：

```text
1. 克隆并检查 GitHub 仓库；
2. 读取 README、Simulation、Experiment、hotdata、colddata；
3. 对所有文件进行数据审计；
4. 检查本方案假设和真实文件是否一致；
5. 生成 reports/data_audit.md；
6. 生成 configs/data_metadata.yaml 草稿；
7. 列出所有无法从数据本身确认的物理参数和单位；
8. 在问题解决前不得自行编造参数；
9. 数据审计通过后再进入 Milestone 2。
```

---

# 27. 最终目标

实现：

\[
P\rightarrow T(r,z,t,P)
\]

并轴对称恢复：

\[
T(x,y,z,t,P)
\]

例如输入：

```text
P = 36 W
```

输出：

- 0–200 s SiC–Cu 完整瞬态温度场；
- SiC 表面预测；
- Cu 内部预测；
- Hot/Cold 曲线；
- VTK/ParaView 三维可视化；
- 模型适用范围和误差报告。

**核心研究问题：在多保真、跨材料、高保真实验工况稀缺的条件下，哪一种 PINN 增强方法能够在完全未见功率上获得最高、最稳定、最可信的瞬态温度场预测精度。**
