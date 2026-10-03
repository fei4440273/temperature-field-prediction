# SiC-Cu 多保真温度场预测

## V5 修复版：四种序列 DeepONet 对比

V5 新增 FNN、GRU、LSTM、ASL 四种 DeepONet，均不含边界注意力。
四种方法使用统一配置、数据划分、主干/输出头初始化和采样种子，
已分别完成 1000 个采样 epochs，主表统一评价第 1000 轮模型。
分支采用过去 hot/cold 测量历史，历史截止于查询时刻前 1 秒。

修复了精确功率混用相邻曲线采集终点、非整数秒有效标记错误翻转两处历史处理问题，
并对四种方法分别重新训练1000轮。原始V5结果受这些缺陷影响，保留作诊断对照，
当前正式结论以修复版报告为准。这是单种子、恒定加热数据的条件重构比较。
同时用最近已到达观测的有界趋势重构采样间隔内的末段输入，减少旧温度保持导致的
ASL升温率锯齿；这一输入处理统一用于四种方法，不读取未来温度。

主图直接对比实验测试集。Figure 3改为按功率分组的RMSE，仿真参考放入附录；
温度云图增加分级色带与等温线，误差云图标出最大值和位置。
Figure 8热端/冷端使用0.1秒网格的原始预测，没有平滑或强制单调处理。

| 方法 | 实验顶部 RMSE / K | 热端 RMSE / K | 冷端 RMSE / K | 综合 RMSE / K |
|---|---:|---:|---:|---:|
| FNN-DeepONet | 8.3112 | 1.2985 | **0.6750** | 5.9223 |
| GRU-DeepONet | 8.7332 | 0.8824 | 0.6789 | 6.2004 |
| LSTM-DeepONet | 7.9019 | 0.9150 | 0.7781 | 5.6197 |
| ASL-DeepONet | **6.8127** | **0.6419** | 0.6957 | **4.8405** |

- [完整修复版报告与 17 张 300 DPI PNG](研究记录/Sequential_DeepONet_1000epochs_history_fix_v3/对比总结.md)
- [汇总指标 CSV](研究记录/Sequential_DeepONet_1000epochs_history_fix_v3/evaluation/summary.csv)
- [密集时间步检查](研究记录/Sequential_DeepONet_1000epochs_history_fix_v3/temporal_audit.json)
- [独立审计结果](研究记录/Sequential_DeepONet_1000epochs_history_fix_v3/verification.json)
- [统一配置](configs/sequential_deeponet.yaml)
- [版本文件与复现说明](docs/V5-release.md)

每种方法均发布 `initial.pt`、`best.pt`、`latest.pt`、`epoch_1000.pt`、
完整训练历史和日志，并保留逐点预测、源码快照、模型/输入哈希及同名 PDF。
以下命令在项目根目录执行；新训练必须使用新输出目录。

```bash
pip install -e '.[test]'
python scripts/train_sequential_deeponet.py --device cuda --output '研究记录/Sequential_DeepONet_new'
python scripts/evaluate_sequential_deeponet.py --device cuda --output '研究记录/Sequential_DeepONet_new'
python scripts/audit_sequential_temporal.py --output '研究记录/Sequential_DeepONet_new'
python scripts/plot_sequential_deeponet.py --output '研究记录/Sequential_DeepONet_new'
```

训练中断后，在同一次训练命令中添加 `--resume`。原始与处理后数据仍按仓库惯例
不随代码发布，训练、重新推理和完整输入审计需要准备项目 `data/`；
重绘已发布的图像可直接使用保存的预测结果。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_sequential_deeponet.py -q -o addopts=''
python scripts/verify_sequential_deeponet.py --output '研究记录/Sequential_DeepONet_1000epochs_history_fix_v3'
```

ASL 与原始源码逐权重对齐的测试需要本地 `references/ASL-PINN-code`，
没有该参考仓库时只跳过该项测试。本机已完成全部 25 项测试和完整实验审计。

## V4 保留代码

当前代码使用联合训练 DeepONet：低保真分支拟合仿真温度场，高保真校正分支拟合实验顶面、铜板热端和冷端温度。模型只适用于持续加热过程，不用于激光关闭后的冷却。

训练入口为 `scripts/联合训练8000轮.py`，模型和预测函数位于 `scripts/joint_temperature_core.py`。入口文件名是历史名称，当前配置实际训练600轮。

## V4 保留模型

- 训练配置：[第09轮600轮配置](configs/联合训练600轮_低保真初温平滑锚定.yaml)。低保真使用80个仿真功率训练、不另划验证或测试；高保真实验为12个训练、3个验证和3个留用测试功率。当前开发训练与分析没有读取测试温度。
- **正式预测只用第09轮的`验证最佳模型.pt`（第575轮）**。同目录的`低保真初始化模型.pt`是复训起点，不是正式预测模型；该起点来自曾使用实验训练数据的旧联合训练检查点，不应称为纯仿真预训练。
- 验证集顶部、热端、冷端RMSE分别为3.269、1.072、0.650℃。80个仿真训练功率的全场RMSE为3.292℃；低保真零秒固定为仿真初温22℃，但800 W顶面近中心0至2秒温升仍低估约62.43℃。热/冷端末段趋势和115.2 W低功率曲线仍存在偏差。
- 按冷热端同时降低整条验证曲线RMSE、顶部误差保持可接受、修正仿真初温错误的目标，第09轮是当前保留的版本；它不是所有单项指标的最小值。第14轮综合选模分数2.308℃低于第09轮2.347℃，但两端RMSE均高于第09轮；第19轮热端RMSE为1.001℃，冷端则升至0.692℃。旧轮次完整结果未随 V4 发布。
- 低功率长时段实验数据不足，传感器轴向安装位置仍需核实；这些问题限制了现有误差判断。

## V4 文件范围

- GitHub 的 V4 包含当前 `configs/`、`scripts/`、`src/`、`tests/`、依赖清单和这份说明。随代码发布第09轮的最佳预测权重、低保真初始化权重及少量核验记录，均在 `研究记录/联合训练600轮_第09轮_低保真初温平滑锚定_20260920/`。
- 原始与处理后的 `data/`、独立的 `GitHub data repository/`、`reports/`、训练曲线、结果总览网页及其图片、其他任务文档不在 V4 发布包内。完整训练和分析需要自行准备项目原始数据；本机保留的研究文件不等于远端也提供这些文件。
- 发布目录不含第600轮模型和完整运行配置，不能直接把发布目录传给两项分析脚本；它们用于在数据齐全时重新完成训练所产生的新运行目录。
- 最佳预测权重 SHA256：`29c3ca716b63402704f88067a581da636dc7c637103b8a3ef515f739407723c5`；低保真初始化权重 SHA256：`ef2793367e335513189a72961337c84754d18d788fd326d4f52c4f4d14169a02`。

## 运行与查看

建议使用 Python 3.9 及以上和 `pip install -e '.[test]'` 安装依赖。入口文件名虽沿用“8000轮”，实际按当前配置连续训练600轮。新运行必须使用新输出目录，不覆盖第09轮权重；在运行前需备齐未上传的原始与处理后数据。

```bash
PYTHONPATH='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_joint_selection.py tests/test_best_release.py
python scripts/联合训练8000轮.py --config configs/联合训练600轮_低保真初温平滑锚定.yaml --device cuda --output '研究记录/联合训练600轮_新运行'
python scripts/联合训练开发结果分析.py --run '研究记录/联合训练600轮_新运行'
python scripts/联合训练低保真训练集分析.py --run '研究记录/联合训练600轮_新运行' --device cuda
```

前述两项测试只需 V4 发布包；完整 `pytest tests`、训练和分析还需要未发布的本地数据及图册。后两条分析分别只使用高保真实验的训练/验证数据、80个仿真训练功率。现有实验最晚观测时间不超过160秒，200秒的实验温度没有对应实测，不能从模型外推断言已经稳定。

在仓库根目录可以直接加载最佳模型并预测单点温度（五列依次为半径m、高度m、时间s、功率W、材料编号；温度输出为K）：

```python
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path('scripts').resolve()))
from joint_temperature_core import load_model, predict

checkpoint = Path('研究记录/联合训练600轮_第09轮_低保真初温平滑锚定_20260920/验证最佳模型.pt')
model, state = load_model(checkpoint, 'cpu')
query = np.array([[0.0, 0.0, 20.0, 430.0, 1.0]], dtype=np.float32)
print(predict(model, query, fidelity='high')[0] - 273.15)  # 摄氏度
```
