# SiC-Cu 多保真温度场固定数据协议

> 版本：v4.0  
> 日期：2026-09-08  
> 状态：配置、预处理、训练加载器和测试入口已实现

## 1. 核心规则

本项目只使用一个预先固定的 train/validation/test 协议，不再使用 LOGO 或循环交叉验证。
所有高保真划分均以完整功率为最小单位，同一功率的 Top、Hot、Cold 数据始终一起移动。

- train：参与损失和梯度更新；
- validation：只用于早停、选择检查点和超参数，不参与梯度更新；
- test：只在模型结构、超参数和检查点全部冻结后评估，不参与训练或模型选择。

不得按像素、节点、时间帧或传感器时刻随机拆分，否则同一实验功率会跨集合泄漏。

## 2. 高保真固定划分

Experiment_data 共 15 个功率，其中 3 个按功率范围分层选作验证集，约占 20%。选择只依据
功率覆盖范围，不依据模型误差：115.2 W 覆盖低功率区，403 W 覆盖中功率区，630.5 W
覆盖高功率区。其余 12 个功率训练。

| 集合 | 数据目录 | 功率（W） | 允许用途 |
| --- | --- | --- | --- |
| train | `Experiment_data` | `55, 216.8, 254.5, 257, 309, 364.3, 430, 494.2, 532, 558.5, 593.5, 729` | 梯度更新 |
| validation | `Experiment_data` | `115.2, 403, 630.5` | 早停、检查点与超参数选择 |
| test | `test_Data` | `169, 339, 634` | 最终冻结测试 |

三个功率集合两两交集为空。Experiment_data 只产生 `experiment/train` 和
`experiment/validation` 记录；test_Data 只能产生 `test/test` 记录。

## 3. Simulation 固定划分

Simulation_data 继续按完整功率使用 `60/10/10` 划分。低保真模型只在 simulation train
上拟合，在 simulation validation 上选择，在 simulation test 上报告完整场泛化指标。
具体功率位于 `configs/splits.yaml`。

多保真模型的梯度数据只包括 simulation train 与 Experiment train。高保真验证只读取
Experiment validation；高保真最终测试只读取 test_Data。低保真仿真与高保真实验是不同
保真度的数据源，所有报告必须分别标明证据来源。

## 4. 机器约束

`configs/data_metadata.yaml` 将 test_Data 固定声明为：

```yaml
role: final_test_only
training_allowed: false
model_selection_allowed: false
```

预处理器和训练器执行以下检查：

1. train、validation、test 功率两两无交集；
2. Experiment 必须恰好是 12 train + 3 validation；
3. test_Data 必须恰好是 3 test；
4. 显式传入功率时仍必须同时匹配对应 `split`；
5. test_Data 记录一旦出现在 train 或 validation，立即终止；
6. 检查点记录的三组功率若与当前固定协议不一致，测试入口拒绝评估。

旧 LOGO 和 grouped-CV 入口已停用，调用时只返回迁移提示。

## 5. 训练和最终测试

默认多保真训练直接读取固定划分：

```bash
python scripts/03_train_multifidelity.py \
  --lf-checkpoint reports/runs/deeponet_data_seed0/best.pt \
  --output reports/runs/multifidelity_fixed_split_seed0
```

模型结构、损失权重、轮数策略和检查点全部冻结后，单独运行一次 test_Data：

```bash
python scripts/06_evaluate_test_data.py \
  --checkpoint reports/runs/multifidelity_fixed_split_seed0/best.pt \
  --output-json reports/test_three_power_comparison.json \
  --output-csv reports/test_three_power_comparison.csv
```

最终结果对 169、339、634 W 分别输出 Top/Hot/Cold 的有符号平均误差、MAE、RMSE、最大
绝对误差及等权综合指标。

## 6. 证据边界

169/339/634 W 在旧协议中曾以“验证数据”名义运行过并查看过结果，因此从研究过程角度，
它们已经不是从未查看的严格盲测集。当前代码可以保证它们今后不参与梯度更新和模型选择，
但无法消除历史查看事实。若论文需要严格的一次性盲测结论，应再采集至少一组从未查看的
独立 test_Data，并在方法完全冻结后只评估一次。

Top 数据验证 SiC 顶面，Hot/Cold 只验证铜底面两个半径位置；这些观测不能当作完整三维
体积真值。内部温度场结论仍需结合 simulation test 完整场误差和物理残差说明。
