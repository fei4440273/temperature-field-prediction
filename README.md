# SiC-Cu 多保真瞬态温度场预测

> 项目的当前阶段、实时运行状态、已完成结果和阻塞项统一记录在
> [`PROJECT_PROGRESS.md`](PROJECT_PROGRESS.md) 中。

本工程当前依据 `temperature_field_prediction_multifidelity_plan_v2.md` 完善，用于研究激光功率到
SiC-铜复合结构轴对称瞬态温度场的映射：

```text
(r, z, t, P, material_id) -> T(r, z, t, P, material_id)
```

最终三维结果由轴对称截面绕中心轴旋转得到。当前方案以确定性模型筛选为主，首批包含
MLP-PINN、LSTM-PINN、POD-PINN、DeepONet-PINN 和 GNO 候选；扩散模型不属于第一阶段。
当前 GNO 只允许数据消融，图上的有效 PDE 微分算子实现前不能标为 GNO-PINN。

温度单位约定：文档、图表、预测导出及评价指标全部使用 `℃`；JSON/CSV 的温度误差字段
统一以 `_c` 结尾。辐射边界、材料物性和模型内部张量仍使用绝对温标 K，输出时统一转换，
不能把 Stefan-Boltzmann 项直接改为摄氏温度。

## 当前状态

- 已按 V2 完成 P0/P1 工程改造：543 个原始 CSV 的来源、功率、模态、split、大小和
  SHA-256 清单，以及 processed manifest、解析物理快照和协议审计均可复现。
- 训练检查点必须记录 raw/processed/split/physics 指纹；来源不明、旧全数据或旧 LOGO
  检查点不能进入当前正式对照。
- 连续坐标模型直接在真实 IR/Hot/Cold 坐标评价；主选择指标固定为 `macro_v1`，按功率
  和曲线等权，不再由记录数决定权重。
- 普通多保真联合阶段默认以 LF 输出拟合 simulation train；原 HF 向仿真回拉仅保留为
  `high_fidelity_pullback` 命名消融。
- 已新增 PRC 全域 LF 响应、V/A/AT 校正、平衡仿真采样、冻结发布清单和测试门禁。
- 已按固定协议完成 MLP-PINN、DeepONet-PINN 和硬表面引导的 5 种子训练与独立测试；
  PRC-V/A/AT 已完成 seed 0 先导对照，尚未扩展到 5 种子。
- `test_Data` 已在模型选择完成后按用户授权读取；从现在起不得依据测试结果回调模型或超参数。
- 80 个仿真功率的全部 `t=0` 节点均为 `22℃`；实验首帧从 5 s 开始，不能代替初始场。
- 已生成统一 Parquet 数据层，并保留原始文件不变。
- 已固定 simulation `60/10/10`，并将高保真数据固定为 Experiment `12/3` 训练/验证、
  test_Data `3` 功率最终测试；三组功率和全部模态互不重叠。
- 已实现 FEM 线性/三次插值、MLP/LSTM/POD/DeepONet PINN 候选、GNO 数据候选及统一物理损失。
- 已实现确定性多保真表面残差基线、任意功率查询、三维旋转和 NPZ/CSV/VTK/PNG 导出。
- MLP、POD 和 GNO 训练入口已在两张 NVIDIA A40 上通过 DDP 冒烟验证。
- 已完成材料感知 MLP/LSTM/DeepONet、无材料标签 MLP 消融与 Global POD 的 5 种子 simulation-only 冻结功率测试。
- 已实现原始二维 IR 像素回投、轴对称误差下限、末帧对比图和 5 种子汇总。
- 历史版本已完成参数可辨识性分析、辐射率敏感性筛选、确定性多保真 PINN 5-seed
  开发测试及 5 折留二 x 5 seeds 的 25 模型验证。
- 历史版本已完成“硬表面温度引导”多保真 DeepONet-PINN 的 10 折 LOGO x 5 seeds，
  并生成了旧数据部署检查点；LOGO 和循环交叉验证现已停用，旧结果仅作历史参考。

### 当前固定测试结果

独立测试功率为 169、339、634 W。五种子总体 RMSE/MAE：DeepONet-PINN 为
`1.9600 ± 0.2743℃ / 1.6947 ± 0.2790℃`，MLP-PINN 为
`2.4361 ± 0.5547℃ / 2.1881 ± 0.5774℃`，硬表面引导为
`3.1736 ± 0.0705℃ / 3.0127 ± 0.0683℃`。PRC-AT 的 seed 0 RMSE/MAE 为
`2.0404℃ / 1.6805℃`，没有超过 DeepONet-PINN 对应 seed 0 的 `1.7511℃ / 1.4707℃`。
三个功率的 Top/Hot/Cold、RMSE、MAE、偏差、最大绝对误差及 PRC-V/A/AT 明细见
[`reports/final_test_20260909_user_authorized/README.md`](reports/final_test_20260909_user_authorized/README.md)。

### 历史结果，仅作背景

历史结果中综合效果最好的部署路线是：**DeepONet 低保真体场 + 训练折内拟合的硬表面绝对温度
引导 + 材料感知 PINN 内部修正**。在严格 LOGO x 5 seeds 中，其 SiC 顶面
`RMSE=4.635 ℃`、`MAE=3.626 ℃`、最高温度相对误差 `2.412%`，铜 Hot/Cold 环绝对/温升
`RMSE=3.321 +/- 0.256 ℃` 和 `2.395 +/- 0.080 ℃`。平均表面 MAE 与峰值误差通过第一阶段
门槛；115.2 W 和 558.5 W 的逐功率 MAE 仍超过 5 ℃。表面引导是确定性的，所以表面指标
跨 seed 标准差为 0；内部修正网络的传感器指标仍随 seed 变化。内部三维场没有实验真值，
仍只能称为物理引导推断。完整结果与限制见
[`reports/implementation_status.md`](reports/implementation_status.md)。

在 simulation-only 冻结测试功率上，材料感知数据驱动 MLP、LSTM 和 DeepONet 的全场 RMSE
分别为 `1.052 ± 0.095 ℃`、`0.653 ± 0.026 ℃` 和 `0.538 ± 0.053 ℃`，无材料标签 MLP
消融为 `4.447 ± 0.035 ℃`，Global POD 数据消融为 `0.177 ± 0.019 ℃`。它们都未超过 FEM
线性插值的 `0.0157 ℃`。

LSTM 使用验证集从 8/16/32 中锁定窗口 32，其最高温度曲线 MAE 为 `2.116 ± 0.108 ℃`，弱于 DeepONet 的
`1.011 ± 0.378 ℃` 和 Global POD 的 `0.355 ± 0.060 ℃`。因此按预设精度筛选规则，LSTM
只保留为可复现基线，不作为当前优先组合模块。

材料常数、22℃ 初始/环境/水冷温度、Gaussian 半径定义、外圆柱水冷边界以及全部观测数据
语义均已确认。IR CSV 已是最终温度场，`50mm` 表示 SiC 直径，不再要求相机原始辐射标定。
接触热阻的低保真有效初值为 `7.340724e-5 m^2*K/W`。SiC/铜热辐射率在当前数据下不可
唯一辨识，固定 low/mid/high 场景的验证 RMSE 最大差仅 `0.005592 ℃`，名义模型采用中性
`0.5/0.5`，不作材料真值声明。详见 [`PROJECT_PROGRESS.md`](PROJECT_PROGRESS.md)。

## 数据目录

```text
data/
├── Simulation_data/       # 80 个功率，10-800 W，完整轴对称场
├── Experiment_data/       # 15 个功率，12 个训练、3 个验证
│   ├── Topdata/           # SiC 顶面二维红外场
│   ├── HotData/           # Cu 底部 r=0.028 m 环数据
│   └── ColdData/          # Cu 底部 r=0.0415 m 环数据
├── test_Data/             # 169/339/634 W，只允许最终测试
│   ├── topdata/
│   ├── hotdota/           # 原始目录名如此
│   └── colddata/
├── processed/             # 自动生成的统一 Parquet 数据
└── cache/                 # POD 等可重建缓存
```

原始数据只读使用。`data/processed/` 和 `data/cache/` 可由脚本重建，不应手工编辑。

## Docker 环境

本机已验证镜像为 `ra-msml-pinn:local-cu124`，其 PINN 环境位于
`/opt/conda/envs/PINN`，包含 PyTorch 2.6.0 与 CUDA 12.4。

定义工程路径：

```bash
PROJECT_DIR='/home/lyf/Temperature Field Prediction'
IMAGE='ra-msml-pinn:local-cu124'
```

运行全部测试：

```bash
docker run --rm --ipc=host \
  -e PYTHONPATH=/workspace/src \
  -v "$PROJECT_DIR:/workspace" -w /workspace "$IMAGE" \
  /opt/conda/envs/PINN/bin/python -m pytest -q tests
```

双 A40 训练统一使用：

```bash
docker run --rm --gpus all --ipc=host \
  -e CUDA_VISIBLE_DEVICES=0,1 -e PYTHONPATH=/workspace/src \
  -v "$PROJECT_DIR:/workspace" -w /workspace "$IMAGE" \
  /opt/conda/envs/PINN/bin/torchrun --standalone --nproc_per_node=2 SCRIPT.py ARGS
```

也可使用 [`docker/compose.yaml`](docker/compose.yaml) 启动开发容器。

## 数据准备与审计

```bash
python scripts/00_audit_data.py
python scripts/01_build_processed_data.py
```

关键产物：

- `reports/data_audit.md`
- `reports/data_audit.json`
- `data/processed/manifest.json`
- `data/processed/simulation/*.parquet`
- `data/processed/experiment_ir_radial.parquet`
- `data/processed/sensor_ring_raw.parquet`
- `data/processed/test_ir_radial.parquet`（仅冻结测试入口读取）
- `data/processed/test_sensor_ring_raw.parquet`（仅冻结测试入口读取）
- `reports/current_protocol/data_inventory.json`
- `reports/current_protocol/protocol_audit.json`
- `reports/current_protocol/resolved_physics.yaml`

IR 像素按 `0.25 mm` 径向分箱，每帧权重和固定为 1；Hot/Cold 中完全重复的角向值
被压缩为每时刻一条环平均，避免伪造样本量。

## 第一阶段基线与候选模型

FEM 插值基线：

```bash
python scripts/02_run_simulation_benchmark.py --kind linear
python scripts/02_run_simulation_benchmark.py \
  --kind cubic --output reports/interpolation_cubic_benchmark.json
```

POD 能量分析只使用 simulation 训练功率：

```bash
python scripts/02_analyze_pod.py
```

累计能量图输出到 [`reports/pod_energy.png`](reports/pod_energy.png)。Global/材料分离
POD 的变体选择使用重建后的验证全场 RMSE，不能比较维数不同的系数均方误差。

纯数据 MLP 双卡训练：

```bash
torchrun --standalone --nproc_per_node=2 scripts/02_train_simulation_model.py \
  --method mlp --seed 0 --output reports/runs/mlp_seed0
```

纯数据时序/算子基线可使用 `--method lstm` 或 `--method deeponet`。材料和边界配置确认后，
可将方法改为 `mlp_pinn`、`lstm_pinn` 或 `deeponet_pinn`。POD-PINN 和 GNO 使用独立入口：

```bash
torchrun --standalone --nproc_per_node=2 scripts/02_train_pod_model.py
torchrun --standalone --nproc_per_node=2 scripts/02_train_gno_model.py --data-only
```

`--data-only` 仅用于软件检查或消融，不能标为 PINN 结果。GNO 的消息传递会耦合节点，当前
逐点自动微分不能给出正确的节点 Jacobian/PDE 导数；未实现图微分残差前，程序会拒绝 GNO
物理训练。

## 多保真训练与验证

确定性 SiC 顶面残差基线无需未知物理量，可以先运行：

```bash
torchrun --standalone --nproc_per_node=2 scripts/03_train_surface_residual.py \
  --seed 0 --output reports/runs/surface_residual_seed0
```

完整多保真校正需要已训练的低保真检查点和全部已核实元数据：

```bash
torchrun --standalone --nproc_per_node=2 scripts/03_train_multifidelity.py \
  --lf-checkpoint reports/runs/mlp_pinn_seed0/best.pt \
  --output reports/runs/multifidelity_seed0
```

低保真检查点必须由当前 60 个 simulation train 功率训练并带完整来源指纹。PRC 先拟合
SiC/Cu 全域 LF，再使用同一个 LF 检查点比较 V/A/AT：

```bash
python scripts/03_train_prc.py lf --modes 8 \
  --output reports/runs/prc_lf_k8_seed0
python scripts/03_train_prc.py correct \
  --lf-checkpoint reports/runs/prc_lf_k8_seed0/best.pt \
  --variant amplitude_time --output reports/runs/prc_at_seed0
```

默认高保真划分为 Experiment 12 个训练功率和 3 个验证功率
（115.2/403/630.5 W），test_Data 的 169/339/634 W 只用于最终测试。验证数据只用于
早停和检查点选择，不进入梯度更新；测试数据既不训练也不选模型。

开发阶段只评估验证功率的 SiC 表面：

```bash
python scripts/02_evaluate_ir_surface.py --checkpoint CHECKPOINT --split validation
```

原始二维像素回投与末帧对比图：

```bash
python scripts/02_evaluate_ir_pixels.py --checkpoint CHECKPOINT --split validation
```

像素指标按“帧内像素、等权帧、等权功率”聚合，不把像素当独立重复实验。当前表面残差
五种子像素等权结果为 `RMSE = 5.748 ± 0.409 ℃`、`MAE = 5.327 ± 0.506 ℃`；它与径向
可靠度加权的 `MAE = 4.507 ± 0.443 ℃` 是不同评价口径。

五个种子的模型、配置与评价口径全部冻结后，先创建 release；随后一次性输出三个
test_Data 功率的 Top/Hot/Cold 对比误差、MAE、RMSE、最大绝对误差及二维 IR 回投：

```bash
python scripts/06_freeze_release.py --release-id RELEASE_ID \
  --checkpoint CHECKPOINT_SEED0 --checkpoint CHECKPOINT_SEED1 \
  --checkpoint CHECKPOINT_SEED2 --checkpoint CHECKPOINT_SEED3 \
  --checkpoint CHECKPOINT_SEED4 --model-config configs/training.yaml \
  --command 'RECORDED_TRAINING_COMMAND'
python scripts/06_evaluate_test_data.py \
  --checkpoint CHECKPOINT_SEED0 --checkpoint CHECKPOINT_SEED1 \
  --checkpoint CHECKPOINT_SEED2 --checkpoint CHECKPOINT_SEED3 \
  --checkpoint CHECKPOINT_SEED4 \
  --release-manifest reports/releases/RELEASE_ID/release_manifest.yaml
```

脚本会先完成五个种子的一致性预检，再一次性读取测试集；最终 JSON/CSV 同时包含逐种子
结果、跨种子均值/标准差，以及 169/339/634 W 各自的 Top/Hot/Cold 偏差、MAE 和 RMSE。

Hot/Cold 元数据已确认；也可对任意多保真检查点运行外部功率评估：

```bash
python scripts/04_external_sensor_test.py --checkpoint CHECKPOINT
```

训练产物包含最佳检查点、逐 epoch JSONL、最终 JSON 指标、配置快照和 Material Passport。

当前协议不使用 LOGO 或循环交叉验证。`scripts/05_logo_cv.py` 和
`scripts/05_grouped_cv.py` 仅保留为旧命令的显式停用提示，不能用于生成新结果。

## 任意功率预测与导出

只有当前固定协议冻结发布中的检查点可以作为新推荐模型。36 W 位于 LF 范围内，但低于
HF 训练最小功率 55 W，必须标为高保真校正外推。旧全数据检查点仅保留历史复现用途。

冻结检查点的预测命令为：

```bash
python scripts/06_predict_power.py \
  --power 36 --t-start 0 --t-end 200 --dt 1 --theta-resolution 72 \
  --checkpoint FROZEN_CHECKPOINT \
  --output predictions/36W_frozen_release

python scripts/06_validate_prediction.py \
  --input predictions/36W_frozen_release \
  --output reports/deployment/36W_frozen_release_validation.json
```

输出目录为：

```text
predictions/36W_frozen_release/
├── field_rzt.npz
├── metadata.json
├── hot_cold.csv
├── vtk/*.vtk
└── figures/*.png
```

Python 接口：

```python
from sic_cu import Predictor

predictor = Predictor(
    checkpoint="FROZEN_CHECKPOINT"
)
result = predictor.predict(
    power_w=36.0,
    times_s=range(0, 201, 2),
    theta_resolution=72,
)
print(result.mean_temperature_c.shape)
print(result.metadata.warnings)
```

不传检查点时仍可使用 LF FEM 功率插值基线。接口范围为 `0-800 W`；
0 W 返回全部仿真共同观测到的 `22℃` 均匀初始场，`0-10 W` 使用该锚点与 10 W 仿真的
线性插值并给出缺少直接支撑的警告。`55-729 W` 只表示处于当前 HF 训练功率范围，不等于
该功率没有对应的实验功率真值，只能作为功率插值查询；范围外输入直接拒绝。

以下旧部署结果仅为历史记录：`predictions/36W_best_deployment/` 有 201 帧、1,159 个轴对称节点，
全场 `22.0-44.240℃`，`t=0` 与铜外圆柱面全时刻严格为 `22℃`。按最高温度后续所有
相邻帧变化不超过 `0.01℃` 的规则，200 s 内尚未稳定，接口返回 `None`，报告中记为
`>200 s`。36 W 低于当前 HF 训练最小功率 55 W，因此必须保留外推警告。

数据驱动 MLP 检查点也可完成同样导出，但无材料标签版本
`predictions/36W_mlp_seed0/` 最低约 `9.17℃`；材料感知版本
`predictions/36W_mlp_material_seed0/` 最低约 `7.58℃`。二者均低于已观测的 `22℃` 初始场，
且 200 s 内未稳定，只保留为失败基线证据。旧受约束多保真原型位于
`predictions/36W_mf_pinn_sensor5_seed4/`：`t=0` 和铜外圆柱面均严格为 `22℃`，全场
不低于 `22℃`，最高温度约 `42.586℃`，稳定时刻 `140 s`；它已被新的全数据部署检查点
取代，仅作为历史对照保留。

## 参数辨识状态

用户侧数据和元数据已经完整：

- [`configs/materials.yaml`](configs/materials.yaml)：常物性已核实，SiC/Cu 比热分别为 `700/400 J/(kg*K)`。
- [`configs/boundary_conditions.yaml`](configs/boundary_conditions.yaml)：已知边界已登记；SiC/铜辐射率与 `R_c` 由模型辨识。
- [`configs/data_metadata.yaml`](configs/data_metadata.yaml)：IR 与 Hot/Cold 语义均已确认；`50mm` 是 SiC 直径。

第一轮结果表明：`R_c` 可从低保真界面场稳定提取为有效初值；辐射率从不同初值不能收敛到
共同解，且直接表面平衡给出非物理解，因此按敏感性参数处理。生产配置保持 fail-closed，
显式敏感性训练入口可以运行。当前不需要用户继续提供相机或材料参数。

特别注意：全部仿真功率中 38 对同坐标 SiC/Cu 界面节点存在非零温差，最大可达
`95.90 ℃`。这提示原模型可能包含接触热阻或其他界面处理，禁止默认温度连续；详见
`reports/simulation_interface_audit.json`。

无材料标签 MLP 在同坐标只能输出相同温度，冻结测试集的界面温差预测恒为 0，界面温差
`MAE = 31.67 ℃`。加入 `material_id` 后，5 种子界面温差 `MAE = 1.63 ± 0.16 ℃`；因此后续
点式模型和多保真修正必须保留显式材料输入。该结果仍不能用于辨识接触热阻或界面热流。

## 已知科学边界

实验只直接约束 SiC 顶面和 Cu 底部两个圆环，没有 SiC/Cu 内部完整实验真值。
因此内部三维场属于“表面和稀疏传感器约束下的物理推断”，不能写成内部实验真值重建。
所有新模型必须先超过 FEM 插值强基线，并按固定功率划分和预先锁定的随机种子报告；
169/339/634 W 始终只作最终测试，不参与任何模型选择。
