# SiC-Cu 温度场预测项目实时进度

> 2026-09-08 协议更新：当前机器可读协议使用 Experiment_data 的 12 个训练功率和
> 3 个验证功率（115.2/403/630.5 W），test_Data 的 169/339/634 W 只作最终测试。
> train/validation/test 按完整功率和 Top/Hot/Cold 联动隔离；LOGO 与循环交叉验证已停用。
> 完整协议见 `DATA_USAGE_PLAN.md`。下文已完成模型和数值均为旧协议历史结果，必须按固定
> 三方划分重训后才能更新为当前结论。

> 最后更新：2026-09-03 06:03 UTC  
> 工程状态：`HARD-SURFACE-GUIDED MF-PINN / MEAN SURFACE GATES PASSED`  
> 当前阶段：50 模型 LOGO 五种子验证、全数据部署重训和 36 W 端到端导出均已完成。  
> 当前运行：没有本项目训练任务正在运行；主机已有长期容器未被停止或修改。  
> 本文件用途：作为项目根目录的主进度看板，后续实施、训练和验证结果应同步更新到这里。

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent`
- Origin Mode：`run + validate status`
- Origin Date：2026-09-02
- Verification Status：`SURFACE MEAN GATES PASSED / INTERNAL FIELD UNVERIFIABLE`
- Version Label：`sic_cu_project_progress_v4`
- 原则：区分真实数据结果、软件能力和待验证推断；未知物理参数不使用经验值代填。

## 一句话结论

项目工程主体已经建立，旧扩散模型代码已删除；原始数据审计、统一数据层、固定功率划分、
FEM/MLP/LSTM/POD/DeepONet 基线、确定性多保真表面校正、统一物理损失、5 折功率隔离协议、
0-800 W 查询和三维导出均已有可运行实现。

当前已能训练和部署确定性多保真 PINN，但仍不能宣称“经实验验证的内部三维高保真温度场”。材料常数、初始温度、
Gaussian 热源定义、吸收率、光斑半径、恒定加载、水冷温度与准确边界面，以及 Hot/Cold
语义均已确认。实验 CSV 已是最终摄氏温度场，直接作为权威观测，不要求相机原始辐射标定。
文件名中的 `50mm` 仅表示 SiC 直径。自然对流系数已有可追溯的工程估计；未知的材料表面
辐射率和 SiC-Cu 接触热阻已完成第一轮可辨识性分析：`R_c` 得到稳定的低保真有效初值，
辐射率在现有数据下不可唯一辨识，已转为 low/mid/high 敏感性场景，不再列为用户待提供信息。

历史的普通残差 MF-PINN 在 **5 折留二完整功率 x 5 seeds** 中表面
`MAE=7.734 +/- 1.152 K`，未通过门槛。新增功率结构化和表面残差插值消融后，最终锁定
**DeepONet 低保真体场 + 硬表面绝对温度引导 + 材料感知 PINN 内部修正**。

最终候选按每次留出一个完整实验功率的 **10 折 LOGO x 5 seeds** 运行 50 个模型。折外
SiC 顶面 `RMSE=4.635 K`、`MAE=3.626 K`、最高温度相对误差 `2.412%`；Hot/Cold 环
绝对/温升 RMSE 为 `3.321 +/- 0.256 K` 和 `2.395 +/- 0.080 K`。平均表面 MAE 和峰值
门槛通过，但 115.2 W 与 558.5 W 的逐功率表面 MAE 分别为 `5.144 K` 和 `7.374 K`，
故“每个功率均小于 5 K”仍未通过。表面引导确定后不依赖神经网络随机种子，所以表面指标
跨 seed 标准差为 0；内部修正和传感器指标仍有 seed 波动。现有单次实验不足以校准概率分布，
因此当前不进入潜扩散阶段。

## 当前阶段判定

项目需要从两个维度理解进度：

| 维度 | 当前状态 | 说明 |
|---|---|---|
| 工程实现 | 主体完成 | 数据、模型、训练器、评估器、预测器、导出、Docker 和测试均已建立 |
| 数据准备 | 完成 | 378 个原始文件审计通过，统一 Parquet 数据层已生成 |
| Simulation-only 筛选 | 完成 | FEM、MLP、LSTM、DeepONet、Global POD 已按冻结功率完成评估 |
| 高保真表面校正 | 完成 | 表面插值消融与硬表面引导 LOGO 验证完成；只代表 SiC 顶面 |
| 参数逆向辨识 | 第一轮完成 | `R_c=7.340724e-5 m^2*K/W` 为低保真有效初值；辐射率不可唯一辨识 |
| 正式 PINN 训练 | 最佳候选完成 | 硬表面温度引导候选通过 LOGO 平均表面门槛 |
| 内部高保真三维场 | 尚不可验证 | 没有内部实验真值；需在物理约束下推断，不能写成实验重建真值 |
| Hot/Cold 验证 | 已完成 | 最终 LOGO 绝对/温升 RMSE 为 3.321/2.395 K |
| 完整功率交叉验证 | 10 折 LOGO x 5 seeds 完成 | 50 个模型；表面 MAE 3.626 K，平均门槛通过 |
| 扩散模型 | 当前方案不实施 | 旧实现已删除；只在确定性模型证据支持后再决定是否进入第二阶段 |

因此，按项目工作方案的顺序，当前工程进度已到 **Milestone 10：任意功率部署接口完成**。
科学验证仍有一项不可由现有数据解除的限制：内部完整三维场没有实验真值，只能报告为表面
与稀疏铜环观测约束下的物理推断。

## 里程碑看板

| 里程碑 | 状态 | 已完成内容 | 尚缺内容 |
|---|---|---|---|
| 1. 数据审计 | 完成 | 结构、功率、时间、节点、单位线索、SHA-256、异常记录 | 无 |
| 2. 统一数据层 | 完成 | Simulation/IR/Hot/Cold 转为规范数据层，原始数据不修改；Hot/Cold 单位转换已启用 | 无 |
| 3. FEM + MLP 基线 | 完成 | 线性/三次插值、无材料标签消融、材料感知 MLP、5 seeds | 无 |
| 4. MLP-PINN | 参数路线已验证 | PDE/边界/初值/界面损失、可训练参数、测试集封存和多初值诊断 | simulation-only 5-seed 正式对照 |
| 5. LSTM-PINN + POD-PINN | 数据基线完成、物理训练阻塞 | LSTM 8/16/32 窗口选择、POD 变体选择、5 seeds | 正式 PINN 训练 |
| 6. DeepONet-PINN | 数据基线完成、物理训练阻塞 | 材料感知 DeepONet 5 seeds | 正式 PINN 训练 |
| 7. GNO-PINN | 仅数据路径可用 | 固定 FEM 图、DDP 和完整场推理已验证 | 有效的图微分/Jacobian PDE 残差 |
| 8. 多保真校正 | 完成 | DeepONet LF + 硬表面温度引导 + 环观测 + PINN 损失 | 内部实验真值缺失 |
| 9. External sensor + LOGO | 完成 | 10 折 x 5 seeds，共 50 模型；IR/Hot/Cold 同功率隔离 | 逐功率门槛仍有失败项 |
| 10. 任意功率预测 | 完成 | 全数据部署检查点；36 W 初值、温度下限、外圆水冷和全部导出 | 36 W 属实验域外推 |

## 数据准备状态

### 原始数据

| 数据 | 数量 | 当前用途 | 状态 |
|---|---:|---|---|
| Simulation data | 80 个功率文件 | 10-800 W 低保真完整轴对称场 | 可用 |
| Experiment data | 264 个 IR 帧 | 10 个功率的 SiC 顶面高保真观测 | 可用 |
| hotdata | 17 个文件 | Cu 底部 `r=0.028 m` 环位置观测 | 米、摄氏度、秒及同步语义已确认，可用 |
| colddata | 17 个文件 | Cu 底部 `r=0.0415 m` 环位置观测 | 米、摄氏度、秒及同步语义已确认，可用 |

总计 `378` 个原始文件、`2.159 GiB`。原始数据与此前 GitHub 数据仓库副本逐文件
SHA-256 一致，项目处理过程没有修改原始 CSV。

### Simulation 审计结果

- 功率：10-800 W，每 10 W 一组，共 80 组。
- 每组：101 帧、每帧 1,159 个节点、共 117,059 行。
- 节点组成：Cu 821 个，SiC 338 个。
- 所有功率使用同一网格。
- 温度范围：`22.0-533.579 degC`。
- 所有 80 个功率、共 92,720 个 `t=0` 节点均为 `22 degC`，单文件标准差为 0。
- 590 W 的 6 s 帧存在约 `0.00196 s` 时间浮点偏差；预处理按容差对齐并保留异常记录。

### Experiment IR 审计结果

- 功率：115.2、216.8、254.5、309、364.3、403、494.2、558.5、593.5、630.5 W。
- 每帧：125,629 个像素，共 33,166,056 条像素记录。
- 温度范围：`52.0-420.9 degC`。
- 所有像素均为 `is_recovered=True`，无法区分原始和修复像素。
- 已按 `0.25 mm` 径向分箱，每帧权重总和限制为 1，避免把像素数当作独立重复试验。
- 首帧为 5 s 且已经升温，不能用于替代 `t=0` 初始条件。

### Hot/Cold 审计结果

- Hot 与 Cold 各 17 个功率文件。
- 同一时刻整圈数据完全重复，已压缩为每时刻一个环平均，避免伪造样本量。
- X/Y 坐标单位为 m，原点为激光光斑中心；数值列为 degC。
- `t=1`、`t=2` 表示激光开启后的第 1 s、第 2 s，数据记录与激光开启同步。
- Hot 位于 `r=0.028 m` 圆环，Cold 位于 `r=0.0415 m` 圆环；每圈温度视为相等。
- 规范加载器已验证转换为 `r_m`、`time_s` 和 `temperature_k`，可进入监督或外部评估。

## 固定数据划分

所有拆分均按完整功率执行，禁止随机拆分像素、节点或时间帧。

### Simulation

- Train：60 个功率。
- Validation：90、170、250、330、410、490、570、650、730、800 W。
- Test：50、130、210、290、370、450、530、610、690、770 W。

### 高保真主划分

- Train：115.2、254.5、364.3、403、558.5、630.5 W。
- Validation：216.8、494.2 W。
- Test：309、593.5 W。

### 外部传感器测试

- IR 覆盖区间内：250、350、500、600 W。
- 相对 IR 覆盖区间外：50、700、800 W。
- 5 折留二功率：每个实验功率恰好作为测试一次；每折测试功率的 IR/Hot/Cold 同时删除。
- 首轮采用嵌套 `6 train / 2 validation / 2 test`，验证功率同样不参与参数更新。

## 已实现的软件模块

### 数据与评估

- 原始文件审计和不可修改数据清单。
- Simulation、IR 径向统计、Hot/Cold 环观测的统一 Parquet 层。
- 功率级 train/validation/test 和 5 折留二防泄漏断言；float32 功率使用 0.0001 W 整数键匹配。
- 全场、材料分区、时间分段、最高温度、热点位置、IR 径向和原始像素指标。
- 界面同坐标双材料节点专项评估。
- 5 种子均值、样本标准差和 95% 置信区间汇总。

### 模型与训练器

- FEM 线性和三次功率插值。
- 材料感知 MLP、LSTM、DeepONet。
- Global POD 和 Material-wise POD。
- 固定网格 GNO 数据模型。
- 确定性多保真表面残差模型。
- LF 预训练检查、HF correction 和 joint fine-tuning 训练器。
- MLP/LSTM/POD/DeepONet 共用 Physics Module。
- 单卡与 PyTorch DDP 双卡入口、最佳检查点、训练日志、配置快照和 Material Passport。

### 物理模块

- SiC/Cu 双材料轴对称热方程残差。
- 中心轴、初始条件、SiC 激光表面、Cu 顶面、铜外圆柱恒温水冷和底面自然对流/辐射边界。
- 对流、辐射、固定温度或对流冷却分支。
- SiC-Cu 热流连续。
- 理想接触与接触热阻两种温度跃迁分支。
- 所有物理量从 YAML 加载；空值或未验证状态直接拒绝正式训练。

### 预测和导出

- Python/CLI 输入功率范围：`0-800 W`。
- 默认时间：0-200 s；支持自定义严格递增时间网格。
- 稳定时间：从某帧开始，后续所有最高温度相邻帧变化均不超过 `0.01 K`。
- 0 W：返回所有仿真共同观测到的 `295.15 K` 均匀初始场；环境与水冷温度均已确认为 22 degC。
- 0-10 W：在 0 W 锚点和 10 W 仿真之间线性插值，并明确标记无直接数据支撑。
- 10-800 W：默认可使用低保真 FEM 功率插值；传入最佳检查点时使用多保真部署模型。
- 输出：轴对称 NPZ、Hot/Cold CSV、三维旋转 VTK、截面图和最高温度曲线。

## 已验证结果

### Simulation-only 冻结功率测试

| 方法 | 运行数 | 全场 RMSE (K) | 全场 MAE (K) | 结论 |
|---|---:|---:|---:|---|
| FEM 线性功率插值 | 1 | 0.0157 +/- 0.0098 | 0.0051 +/- 0.0032 | 当前最强低保真基线 |
| FEM 三次功率插值 | 1 | 0.0193 +/- 0.0099 | 0.0064 +/- 0.0035 | 略弱于线性插值 |
| Global POD data-only | 5 seeds | 0.1774 +/- 0.0195 | 0.0867 +/- 0.0144 | 最强神经/降阶候选 |
| 材料感知 DeepONet | 5 seeds | 0.5383 +/- 0.0529 | 0.3055 +/- 0.0352 | 优于 LSTM/MLP |
| 材料感知 LSTM，窗口 32 | 5 seeds | 0.6528 +/- 0.0258 | 0.4226 +/- 0.0148 | 未超过 DeepONet/POD |
| 材料感知 MLP | 5 seeds | 1.0523 +/- 0.0954 | 0.6460 +/- 0.0564 | 基础神经基线 |
| 无材料标签 MLP 消融 | 5 seeds | 4.4473 +/- 0.0353 | 1.7297 +/- 0.0358 | 结构上不能表示界面温差 |

标准差口径：插值行是 10 个测试功率之间的标准差；5-seed 行是随机种子之间的样本标准差。
这些数值不是重复物理试验的不确定性。

### LSTM 决策

- Seed 0 验证集窗口 8/16/32 的最佳 RMSE：`0.906/0.994/0.725 K`。
- 按验证集锁定窗口 32，测试集不参与窗口选择。
- 窗口 32 的最高温度曲线 MAE：`2.116 +/- 0.108 K`。
- DeepONet 对应值：`1.011 +/- 0.378 K`。
- Global POD 对应值：`0.355 +/- 0.060 K`。
- 决策：LSTM 保留为可复现基线，但当前不作为最终组合中的优先时序模块。

### 高保真 SiC 顶面结果

锁定测试功率为 309 W 和 593.5 W。

| 方法 | 评价口径 | RMSE (K) | MAE (K) | 结论 |
|---|---|---:|---:|---|
| LF FEM 线性插值 | 径向可靠度加权 | 26.12 +/- 6.48 | 21.63 +/- 4.35 | 未做实验校正 |
| 确定性 MF 表面残差 | 径向可靠度加权，5 seeds | 5.53 +/- 0.52 | 4.51 +/- 0.44 | 达到平均表面 MAE 5 K 门槛 |
| LF FEM 线性插值 | 原始像素分级聚合 | 29.92 +/- 10.60 | 26.59 +/- 9.60 | 未做实验校正 |
| 确定性 MF 表面残差 | 原始像素分级聚合，5 seeds | 5.75 +/- 0.41 | 5.33 +/- 0.51 | 未达到像素口径 MAE 5 K |

补充结果：

- 平均峰值绝对误差：`2.85 +/- 0.46 K`。
- 平均峰值相对误差：`1.19 +/- 0.19%`。
- 径向梯度 MAE：`0.538 +/- 0.166 K/mm`。
- Seed 4 在 593.5 W 的 MAE 为 `6.13 K`，说明随机种子稳定性仍需改进。
- 以上结果只适用于 SiC 顶面，不能当作内部三维高保真场验收结果。

### 最佳方法：硬表面温度引导多保真 DeepONet-PINN

方法把 80 个 Simulation 工况训练得到的 DeepONet 作为低保真完整场先验，并在每个训练折内
对实验 SiC 顶面绝对温升拟合二维 Chebyshev 引导面。SiC 顶面由引导面硬约束，修正沿厚度
平滑过渡到材料感知 PINN 内部修正；铜内部仍由低保真场、Hot/Cold 环观测和物理损失共同
约束。测试功率的 IR、Hot、Cold 均不进入该折拟合。

严格 10 折 LOGO x 5 seeds（共 50 个模型）的结果：

| 指标 | 均值 | seed 标准差 | 95% CI 半宽 | 判定 |
|---|---:|---:|---:|---|
| SiC 顶面 RMSE | 4.635 K | 0 | 0 | 参考指标 |
| SiC 顶面 MAE | 3.626 K | 0 | 0 | `<=5 K` 通过 |
| 最高温度绝对误差 | 3.765 K | 0 | 0 | 参考指标 |
| 最高温度相对误差 | 2.412% | 0 | 0 | `<=5%` 通过 |
| Hot/Cold 绝对 RMSE | 3.321 K | 0.256 K | 0.318 K | 内部稀疏观测 |
| Hot/Cold 温升 RMSE | 2.395 K | 0.080 K | 0.099 K | 内部稀疏观测 |

表面指标的 seed 标准差为 0 是硬引导确定性的预期结果，不代表物理试验没有不确定性。逐功率
仍有两个 MAE 超过 5 K：115.2 W 为 `5.144 K`，558.5 W 为 `7.374 K`；115.2 W 的峰值
相对误差也为 `7.583%`。因此只判定“跨功率平均门槛通过”，不判定“每个功率均通过”。
完整证据位于 `reports/mf_pinn_hard_surface_temperature_logo_5seed_summary.json`。

仅预测 SiC 表面时，`regression_residual_per_watt` 的 LOGO MAE 略低，为 `3.591 K`，但它
没有铜/SiC 内部场，也没有 PINN 体场约束，不能作为最终三维方法。综合最终目标后，硬表面
温度引导多保真 DeepONet-PINN 是当前最佳选择。

### 历史基线：普通确定性多保真 PINN

名义物理场景在测试集打开前锁定：SiC/Cu 热辐射率均取 `0.5`，只作为中性敏感性值；
接触热阻取低保真界面平衡结果 `7.34072435302768e-5 m^2*K/W`。模型使用 DeepONet
低保真算子、材料感知残差网络、IR、Hot/Cold、低保真全场监督和统一物理损失。每个 seed
训练 300 轮 correction + 100 轮 joint，模型选择仅看 216.8 W 与 494.2 W 验证功率。

旧版 `sensor_absolute=0.2` 仅按 IR 选检查点，虽然表面 MAE 为 `4.914 K`，主测试铜环绝对
RMSE 高达 `20.469 K`。该结果保留为开发基线，不能作为当前候选。

使用同一 seed 42、完全不读取测试集的权重消融后，按
`(IR_RMSE + 0.2*SensorAbs_RMSE + SensorDelta_RMSE)/2.2` 锁定
`sensor_absolute=5.0`、`sensor_delta=1.0`，并重新训练 5 seeds：

| 指标 | 5-seed 均值 | seed 标准差 | 95% CI 半宽 | 判定 |
|---|---:|---:|---:|---|
| 验证综合分数 | 3.251 K | 0.270 K | 0.335 K | 仅用于模型选择 |
| 验证 IR RMSE | 4.592 K | 0.274 K | 0.340 K | 仅用于模型选择 |
| 测试 IR RMSE | 6.674 K | 0.784 K | 0.973 K | 待改善 |
| 测试 IR MAE | 5.111 K | 0.583 K | 0.724 K | `<=5 K` 未通过 |
| 测试峰值相对误差 | 1.266% | 0.196% | 0.243% | `<=5%` 通过 |
| 主测试铜环绝对 RMSE | 4.053 K | 0.173 K | 0.214 K | 明显改善 |
| 主测试铜环温升 RMSE | 2.749 K | 0.257 K | 0.319 K | 原型水平 |

按测试功率聚合：

| 功率 | IR RMSE | IR MAE | 峰值相对误差 | 判定 |
|---:|---:|---:|---:|---|
| 309 W | 4.779 +/- 0.499 K | 3.384 +/- 0.637 K | 1.025 +/- 0.413% | 通过 |
| 593.5 W | 7.950 +/- 1.118 K | 6.615 +/- 0.758 K | 1.507 +/- 0.061% | MAE 未通过 |

7 个完全外部 Hot/Cold 功率（50/250/350/500/600/700/800 W）的 5-seed 结果为：绝对
温度 RMSE `7.015 +/- 0.758 K`，温升曲线 RMSE `4.584 +/- 1.023 K`。完整结果见
`reports/mf_pinn_sensor5_5seed_summary.json` 和
`reports/mf_pinn_sensor5_external_sensor_5seed_summary.json`。

### 历史基线的 5 折留二功率交叉验证（5 seeds）

该历史协议采用 5 折留二完整功率。每折使用 6 个功率训练、
2 个功率选择检查点、2 个功率只作测试；IR、Hot 和 Cold 按功率共同隔离。每个功率恰好
测试一次，也恰好验证一次。共训练 25 个模型；下表均值和标准差先在每个 seed 内对 10 个
折外功率等权汇总，再跨 5 seeds 统计：

| 指标 | 5-seed 均值 | seed 标准差 | 95% CI 半宽 | 判定 |
|---|---:|---:|---:|---|
| 表面 RMSE | 9.140 K | 1.302 K | 1.617 K | 未通过 |
| 表面 MAE | 7.734 K | 1.152 K | 1.430 K | `<=5 K` 未通过 |
| 峰值相对误差 | 4.088% | 0.531% | 0.660% | 平均通过，115.2 W 失败 |
| 铜环绝对 RMSE | 3.430 K | 0.870 K | 1.080 K | 原型水平 |
| 铜环温升 RMSE | 2.408 K | 0.653 K | 0.810 K | 原型水平 |

跨 seed 的表面 MAE 较大功率为 115.2 W `8.369 K`、558.5 W `8.587 K`、593.5 W
`13.627 K` 和 630.5 W `19.101 K`；309 W 也为 `5.594 K`。该配置的跨功率泛化明确
失败，并直接促成后续功率结构化与硬表面引导方案。证据见
`reports/grouped_5fold_protocol.json` 和 `reports/mf_pinn_grouped_5fold_5seed_summary.json`。

### 参数辨识与敏感性结论（本轮新增）

- 使用 60 个 Simulation 训练功率拟合、10 个验证功率核验，测试功率没有访问。
- `R_c=7.34072435302768e-5 m^2*K/W`；验证归一化残差 RMSE 为 `0.2850%`，可作为低保真
  有效初始化，但不是内部实验真值。
- SiC/Cu 辐射率的无约束平衡解分别为 `-87.66/-0.390`，违反 `[0,1]`，表明当前仿真场
  与已声明表面边界不能闭合；约束解 0 不得当作物理辐射率。
- 同 seed 多初值联合优化基本保留初值，进一步证明当前观测不能唯一辨识这三个联合参数。
- 固定 low/mid/high 辐射率场景各训练 400 轮后，最佳验证 RMSE 为
  `4.371848/4.369855/4.366256 K`，最大差仅 `0.005592 K`，低于 `0.1 K` 判定阈值。
- 因此锁定中性 mid 场景用于模型开发，并把 low/high 作为敏感性边界，不作材料真值声明。
- 证据：`reports/physics_parameter_identification.md`、
  `reports/inverse_multistart_diagnostic.md`、`reports/physics_sensitivity.md`。

## 关键界面发现

- 仿真网格存在 38 对坐标相同、材料侧不同的 SiC/Cu 界面节点。
- 全部仿真中的最大材料侧温差为 `95.90 K`，位置为 800 W、158 s、
  `(r,z)=(0.025,0) m`。
- 这否定了“可以直接默认界面温度连续”的假设，但没有界面热流，因此不能仅由温差反推出
  接触热阻。
- 无材料标签 MLP 在同坐标只能输出相同温度，冻结测试集的界面温差预测恒为 0，温差
  MAE 为 `31.67 K`。
- 加入 `material_id` 后，界面温差 MAE：MLP `1.63 +/- 0.16 K`，LSTM
  `1.571 +/- 0.070 K`，DeepONet `0.930 +/- 0.075 K`。
- 结论：所有点式模型和多保真校正必须显式输入材料标签。

## 当前可查看的预测结果

### 36 W 最佳部署结果

目录：`predictions/36W_best_deployment/`

- 检查点：`reports/deployment/mf_pinn_hard_surface_temperature_all_data_seed0_constrained.pt`。
- 训练：全部 10 个实验功率，300 轮 correction + 100 轮 joint，最终轮部署；未用测试集选轮次。
- 帧数/节点数：201 帧 / 1,159 个轴对称节点，时间步长 1 s。
- `t=0` 全场严格为 `22.0 degC`；0-200 s 全场范围 `22.0-44.240 degC`。
- 铜外圆柱面共 19 个节点，全时刻固定温度最大误差 `0 K`；全场数值有限且不低于 22 degC。
- 200 s 的最高温度为 `44.240 degC`，最后 1 s 仍变化 `0.0286 K`，故 200 s 内未满足
  `0.01 K` 稳定判据；接口返回 `None`，报告记为 `>200 s`。
- 输出包括 NPZ、Hot/Cold CSV、0/50/100/150/200 s 的 72 角向 VTK 和 PNG，均已回读。
- 36 W 位于 10-800 W 仿真支持域，但低于 115.2 W 红外实验下界，属于实验校正域外推。
- 内部温度场是物理引导推断，不是内部实验真值。

### 36 W 低保真演示

目录：`predictions/36W/`

- 来源：30 W 和 40 W FEM 场的线性功率插值。
- 帧数/节点数：101 帧 / 1,159 节点。
- 预测最高温度：约 `45.02 degC`。
- 稳定时刻：`70 s`。
- 状态：可用于接口与导出演示，不是高保真 PINN 结果。
- 警告：36 W 低于 IR 实验功率覆盖范围。

### 36 W 历史受约束多保真 PINN 原型

目录：`predictions/36W_mf_pinn_sensor5_seed4/`

- 检查点：`reports/deployment/mf_pinn_sensor5_seed4_constrained.pt`；seed 4 由验证综合分数选择。
- `t=0` 全部 1,159 个节点严格为 `22.0 degC`。
- 0-200 s 全场范围：`22.0-42.586 degC`；铜外圆柱面全时刻严格为 `22.0 degC`。
- 按后续所有相邻 Tmax 帧变化均不超过 `0.01 K` 的规则，稳定时刻为 `140 s`。
- 输出包括 101 帧轴对称 NPZ、Hot/Cold CSV、5 个时刻的 72 角向 VTK 和 PNG。
- 36 W 低于 IR 最小功率，实验残差按功率距离二次收缩到低保真先验，并保留外推警告。
- 部署投影还强制全场温度不低于 22 degC；该投影不进入训练图，避免 softplus 饱和造成塌缩。
- 状态：历史原型，已由上面的全数据最佳部署检查点取代。

未约束检查点曾在 36 W 给出 `t=0` 约 `16.36-41.24 degC` 且 Tmax 明显回落，已保留在
`predictions/36W_mf_pinn_nominal_seed1/` 作为失败诊断，不得用于展示或部署。

### 0 W 演示

目录：`predictions/0W/`

- 全场温度：`295.15 K`。
- 稳定时刻：`0 s`。
- 依据：所有 80 个仿真功率共同观测到的均匀初始场。
- 限制：这是基于已确认 `22 degC` 初始/环境/水冷条件的解析锚点，不是激光开启后的实验验证场。

### 不可用于部署的 MLP 示例

- `predictions/36W_mlp_seed0/`：最低约 `9.17 degC`，200 s 内未稳定。
- `predictions/36W_mlp_material_seed0/`：最低约 `7.58 degC`，200 s 内未稳定。
- 两者均违反已观测的 22 degC 初始场，只保留为失败基线证据。

## 验证与运行环境

### 自动化验证

- 单元/集成测试：`74 passed`；含功率键、5/10 折隔离、硬表面引导及部署硬约束回归测试。
- 固定 seed 0 复现：模型权重位级一致，验证和测试指标完全一致；计时字段按预期变化。
- 双 A40 DDP：模型训练、显式梯度平均和参数同步已验证。
- 双 GPU 物理反向传播：使用明确标记的合成常数，只验证软件计算图，不作为科学结果。
- 0 W、10 W、800 W、36 W 和非法功率查询测试通过。
- 连续坐标模型现在直接在请求时刻推理，不再把 `t=1 s` 从仿真偶数秒网格插值出来。
- 多保真部署模型支持 `t=0` 硬初值、22 degC 下限、铜外圆水冷和低功率残差收缩。
- 最终 10 折 LOGO x 5 seeds 的 50 个模型均完成；表面 MAE 3.626 K、峰值相对误差
  2.412%，平均门槛通过。
- 最佳部署检查点的 36 W NPZ、CSV、VTK 和 PNG 已生成并完成回读。
- 所有报告 JSON 可解析。
- 原始 CSV 哈希与数据仓库副本一致。

### Docker

- 基础镜像：`ra-msml-pinn:local-cu124`。
- Conda 环境：`/opt/conda/envs/PINN`。
- PyTorch/CUDA：PyTorch 2.6.0 / CUDA 12.4。
- 工程镜像：`sic-cu-temperature:test`。
- 当前镜像 ID：`sha256:cdb90bc3e82395ff40bcc1c28a1410f44895ab8cd03038beb147ad3241747870`。
- 镜像大小：约 17.51 GB。
- 原始数据由 `.dockerignore` 排除，运行时通过只读挂载提供。

### 2026-09-03 06:03 UTC 资源快照

| GPU | 型号 | 显存 | 利用率 | 本项目状态 |
|---|---|---:|---:|---|
| 0 | NVIDIA A40 | 0 / 46,068 MiB | 0% | 无本项目训练任务 |
| 1 | NVIDIA A40 | 0 / 46,068 MiB | 0% | 无本项目训练任务 |

本项目当前没有正在运行的训练容器。主机上已有的长期容器未被停止或修改。

## 当前物理配置与硬阻塞项

### 1. SiC/Cu 热物性（已解除）

已由用户确认采用不随温度变化的常数：

- SiC：`rho=3170 kg/m^3`、`k=120 W/(m*K)`。
- Cu：`rho=8900 kg/m^3`、`k=401 W/(m*K)`。
- 用户已确认比热原报值采用 `kJ/(kg*K)`，规范值为 SiC `700 J/(kg*K)`、Cu
  `400 J/(kg*K)`。
- 两种材料均按常物性处理，`configs/materials.yaml` 已设为 `verified: true`。

### 2. 激光与散热边界

已确认：SiC 顶面由 Gaussian 光束垂直照射，吸收率 `0.8`；`w=0.02 m` 是公式
`q=2*eta*P/(pi*w^2)*exp(-2*r^2/w^2)` 中的 `1/e^2` 强度半径；加热期间功率恒定。
初始温度、环境温度和水冷固定温度均为 `295.15 K`。水冷只施加到铜外圆柱面
`r=0.05834 m`，不施加到底面。

其余暴露水平表面采用自然对流和辐射：上表面有效常数 `h=9.0 W/(m^2*K)`，下表面
`h=4.3 W/(m^2*K)`；敏感性区间分别为 `5.1-10.7` 和 `2.4-5.1 W/(m^2*K)`。取值推导与
来源记录在 `reports/natural_convection_parameter_basis.md`。

SiC 和裸露铜表面的真实热辐射率仍未知。激光波段吸收率 `0.8` 不自动等于热辐射率，因此
不直接复用。直接边界平衡和同 seed 多初值均表明辐射率不可唯一辨识；固定 low/mid/high
场景的验证 RMSE 最大差仅 `0.005592 K`。后续名义训练采用 `0.5/0.5` 中性值并报告两端
敏感性，不把它们写成材料实测值。

配置文件：`configs/boundary_conditions.yaml`。

### 3. SiC-Cu 界面

已确认 SiC-Cu 界面必须采用接触热阻模型，不使用理想温度连续。由 Simulation 训练功率的
界面温差/法向热流平衡得到 `R_c=7.34072435302768e-5 m^2*K/W`，10 个验证功率的归一化
残差 RMSE 为 `0.2850%`。该值作为“低保真有效初始化”使用，不声称是内部实验真值。
联合梯度辨识未从不同初值收敛到共同解，因此正式模型当前固定该值，并在后续 profile
likelihood/体场改进阶段继续检查。

### 4. IR 元数据（已解除）

- CSV 中的 `temperature_c` 是用户提供的最终温度观测，模型不处理相机原始辐射信号，
  因此相机发射率、背景和镜头参数不再作为训练门禁。
- `50mm` 已确认表示 SiC 圆片直径，用于标识该文件是 SiC 而不是 Cu 的红外数据。
- `r_mm` 与 `sqrt(x_mm^2+y_mm^2)` 的最大误差仅 `1.846e-6 mm`，故文件坐标中心按
  `(0,0) mm` 使用。
- 正式物理初始温度为 `22 degC`；IR 首个已保存温度场位于激光开启后 `5 s`，不冒充 `t=0`。

## 门禁状态

| 路径 | 当前返回状态 | 原因 |
|---|---|---|
| 参数逆向辨识 | `COMPLETE_WITH_LIMITED_IDENTIFIABILITY` | `R_c` 有低保真有效初值；辐射率不可唯一辨识 |
| Simulation MLP-PINN | `READY_FOR_SENSITIVITY_PROTOCOL` | 可通过显式场景运行；默认生产配置仍 fail-closed |
| 完整多保真训练 | `SURFACE_MEAN_GATES_PASSED` | 最终 LOGO 表面 MAE 3.626 K；内部场无实验真值 |
| Hot/Cold 元数据加载 | `PASS` | 米、摄氏度、逐秒时间和激光开启同步已确认 |
| Hot/Cold 外部评估 | `COMPLETE_PROTOTYPE` | 7 功率绝对 RMSE 7.015 K、温升 RMSE 4.584 K |
| 10 折 LOGO 功率验证 | `5SEED_COMPLETE_MEAN_ACCEPTED` | 50 模型，表面 MAE 3.626 K；逐功率门槛未全部通过 |
| GNO-PINN | `BLOCKED_INVALID_GRAPH_PDE` | 当前消息传递输出不能使用点式 Jacobian 作为有效 PDE 导数 |

这些门禁是设计行为，不是程序崩溃。其作用是阻止未经核实的参数进入论文结果。

## 后续执行顺序

1. **已完成**：参数辨识、多初值和固定场景敏感性检查。
2. **已完成**：铜环绝对监督权重验证集消融、5-seed 开发模型及外部 7 功率评估。
3. **已完成**：历史 5 折留二 x 5 seeds，识别普通残差模型的功率泛化问题。
4. **已完成**：功率结构化、五种表面插值和硬表面引导消融。
5. **已完成**：最佳候选 10 折 LOGO x 5 seeds，共 50 个模型。
6. **已完成**：用全部高保真功率重训部署模型，保留初值、最低温度、外圆水冷和低功率投影。
7. **已完成**：最佳部署检查点的 36 W 轴对称与旋转三维场导出和回读。
8. 论文阶段需把内部场明确写为物理推断；若要验证内部精度，需新增内部温度或截面实验。
9. 潜扩散暂不加入；只有取得重复实验并证明 CRPS/覆盖率优于确定性集合后才重新评估。

## 主要文件入口

| 内容 | 路径 |
|---|---|
| 项目说明 | `README.md` |
| 原始工作方案 | `SiC_Cu_temperature_prediction_project_plan.md` |
| 本实时进度 | `PROJECT_PROGRESS.md` |
| 详细实施报告 | `reports/implementation_status.md` |
| 数据审计 | `reports/data_audit.md`、`reports/data_audit.json` |
| 模型对比 | `reports/model_comparison.csv` |
| LSTM 窗口选择 | `reports/lstm_window_selection.json` |
| 界面审计 | `reports/simulation_interface_audit.json` |
| 元数据门禁记录 | `reports/metadata_gate_checks.json` |
| 低保真参数辨识 | `reports/physics_parameter_identification.md` |
| 多初值诊断 | `reports/inverse_multistart_diagnostic.md` |
| 辐射率敏感性 | `reports/physics_sensitivity.md` |
| 历史 MF-PINN 5-seed 汇总 | `reports/mf_pinn_sensor5_5seed_summary.json` |
| 历史外部 Hot/Cold 5-seed 汇总 | `reports/mf_pinn_sensor5_external_sensor_5seed_summary.json` |
| 传感器权重选择 | `reports/sensor_weight_selection.md` |
| 5 折留二协议 | `reports/grouped_5fold_protocol.json` |
| 5 折 5-seed 总汇 | `reports/mf_pinn_grouped_5fold_5seed_summary.json` |
| 最终 10 折 LOGO x 5-seed 总汇 | `reports/mf_pinn_hard_surface_temperature_logo_5seed_summary.json` |
| 表面插值消融 | `reports/residual_interpolation_cv.json` |
| 当前最佳部署检查点 | `reports/deployment/mf_pinn_hard_surface_temperature_all_data_seed0_constrained.pt` |
| 当前最佳 36 W 结果 | `predictions/36W_best_deployment/` |
| 当前最佳 36 W 验收报告 | `reports/deployment/36W_best_deployment_validation.json` |
| 训练产物 | `reports/runs/` |
| 预测结果 | `predictions/` |
| 物性配置 | `configs/materials.yaml` |
| 边界配置 | `configs/boundary_conditions.yaml` |
| 数据元信息 | `configs/data_metadata.yaml` |

## 实时查看命令

查看本进度文件：

```bash
cd '/home/lyf/Temperature Field Prediction'
less PROJECT_PROGRESS.md
```

查看 GPU：

```bash
watch -n 2 nvidia-smi
```

查看当前容器：

```bash
docker ps
```

查看某次训练最后 20 轮：

```bash
tail -n 20 reports/runs/运行目录/training.jsonl
```

查看最终模型对比：

```bash
column -s, -t reports/model_comparison.csv | less -S
```

运行完整测试：

```bash
PROJECT_DIR='/home/lyf/Temperature Field Prediction'
docker run --rm --ipc=host \
  -e PYTHONPATH=/workspace/src \
  -v "$PROJECT_DIR:/workspace" -w /workspace ra-msml-pinn:local-cu124 \
  /opt/conda/envs/PINN/bin/python -m pytest -q tests
```

## 当前需要用户提供的资料

**无。** 当前数据输入已经完整。第一轮分析已经确认辐射率在现有观测下不可唯一辨识，项目
已按敏感性场景处理；这不是用户欠缺数据。`R_c` 已得到低保真有效初值。相机原始标定不属于
当前最终温度 CSV 的模型输入。后续工作可以直接继续，不需要用户再次确认这些项目。
