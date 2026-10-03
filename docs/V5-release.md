# V5 项目适配版发布说明

V5 在 V4 基础上新增 FNN、GRU、LSTM 和 ASL 四种 DeepONet 比较，
参考 [S-DeepONet 论文](https://arxiv.org/abs/2306.08218)、
[作者实现](https://github.com/Jasiuk-Research-Group/S-DeepONet) 和
[ASL-PINN 实现](https://github.com/fei4440273/Temperature-Reconstruction-ASL-PINN)。
ASL 分支保留选择性 SSM、长历史升温率、LSTM 状态初始化及成熟度上下文门控，
未加入边界注意力。这里是适配本项目传感器历史的比较，并非原文实验的逐项复现。

修复版解决两处导致温度曲线突跳的历史有效性缺陷：精确功率不再继承邻居的
采集终点；非整数秒使用已知采集范围判断有效性，温度插值端点仍限于过去观测。
有效标记表示采集覆盖范围这一离线元数据，温度输入仅用已到达的历史观测构建。
末段采样点使用最近两条过去观测的趋势估计，最多延伸一个已观察采样间隔；
等待三个历史点后启用，并限制在已知采集范围内，以减少旧温度保持造成的升温率锯齿。
项目适配版保留 ASL 两层选择性 SSM → LSTM、h/c 初始化及成熟度上下文门控，
将相邻历史点求导改为过去18秒有效历史的因果加权线性趋势，按0.5 K/s归一化。
不增加可训练参数，ASL仍为76,227个参数。旧模型配置缺少新模式时使用原预处理，
新模型明确保存 `asl_rate_mode: thermal_trend`。
四种方法共同加入训练传感器更新时刻前后±0.1秒的二阶差分损失，
它重新构造两侧的过去历史并约束原始预测，使用独立、可恢复的随机数状态。
四种方法均从同一初始化重新训练1000轮，使用共同损失、数据及学习率计划。
原始V5标签保留原始快照，V5分支包含修复结果；旧运行仅作为诊断对照。

本轮继续优化顶部温度误差。旧损失混合各半径的误差，中心低估和边缘高估可能同时存在，
且没有单独约束后段温度、末时刻和实测升温趋势。新增半径5 mm内的训练观测，
以及每个训练功率最后20秒的逐半径温度、末时刻与线性趋势损失；
趋势分别拟合后平方，不允许不同半径的误差互相抵消。
第一候选改善中心却损害完整顶部，依据验证集拒绝；第二候选只将完整顶部权重2提高到8，
验证顶部与中心后段均改善后锁定配置，再完成四种方法各1000次更新。
ASL输入、可训练模块和参数数量均与前一版ASL_thermal一致。

## 发布内容

- `configs/sequential_deeponet_temperature.yaml`：最新四种方法共用的训练配置。
- `configs/sequential_deeponet_tail.yaml`：第一候选配置，保留选型依据。
- `configs/sequential_deeponet.yaml`：前一版输入适配配置。
- `scripts/sequential_deeponet_core.py`：四种分支、共同主干和模型检查点。
- `scripts/sequential_deeponet_data.py`：因果历史和分层采样。
- `scripts/train_sequential_deeponet.py`：连续训练和断点恢复。
- `scripts/sequential_temperature_objective.py`：训练观测的中心、后段、末时刻与趋势约束。
- `scripts/analyze_sequential_temperature_bias.py`：逐功率、逐半径有符号误差与真实时刻诊断。
- `scripts/evaluate_sequential_deeponet.py`：最终轮与验证最优轮的测试评价。
- `scripts/plot_sequential_deeponet.py`：从真实保存预测导出 PNG/PDF 和中文报告。
- `scripts/audit_sequential_temporal.py`：原始V5对照、全功率历史掩码和密集时间步检查。
- `scripts/verify_sequential_deeponet.py`：检查实际优化步数、哈希、指标和图像。
- `tests/test_sequential_deeponet.py`：31 项因果性、数值、求导、连续性及模型/报告契约测试。
- `tests/test_sequential_temperature_objective.py`、`tests/test_sequential_temperature_bias.py`：7项真实时刻、偏差/趋势、空间抵消与验证选型回归测试。
- `研究记录/Sequential_DeepONet_1000epochs_temperature_tail/`：最新四组训练模型、历史、逐点预测、
  指标、17 张 PNG（11张实验主图、6张附录）、同名 PDF、源码快照与对比报告。
- `研究记录/Sequential_DeepONet_1000epochs_ASL_thermal/`：前一版输入适配结果，作为本轮基线。
- `研究记录/Sequential_DeepONet_1000epochs_history_fix_v3/`：保留适配前的检查点和密集预测作对照。
- `研究记录/Sequential_DeepONet_1000epochs/`：保留原始V5输出作缺陷对照。

公共模块 `scripts/joint_temperature_core.py` 同步为本次实验真实使用的源码，
包含轴对称轴线极限、高/低保真物理条件和已有铜区可选模块。
其默认轮数为 1000；V4 的历史入口仍按自身 600 轮配置运行，
对应默认轮数测试同步为 1000。V4 的配置、入口和已发布模型继续保留。

本次新增正式修复实验结果。原始/处理后 `data/`、第三方参考仓库、
诊断运行、短运行及工作区其他未提交任务不进入 V5。

## 结果口径

四种方法均完成 1000 次 AdamW 更新。每轮按训练功率分层采样并计算物理损失，
不是遍历全部约 702 万个仿真点的一次全量 epoch。主结果采用第 1000 轮模型，
验证最优检查点另列补充结果，训练没有提前结束。

这是给定过去 hot/cold 测量的温度场条件重构。仿真内部场作为低保真参考，
实验表面场作为高保真参考；二者指标分别解释。当前为单随机种子、恒定加热数据，
尚未验证任意时变负载。更多实现细节和诊断轮披露见
[完整对比报告](../研究记录/Sequential_DeepONet_1000epochs_temperature_tail/对比总结.md)。

ASL顶部RMSE由6.7109降至6.1333 K，综合RMSE由4.7706降至4.3873 K；
中心后段汇总RMSE由6.1361降至3.6970 K。169/339/634 W中心后段分别由
3.5420/7.9046/5.6227变为0.8061/5.7943/2.0941 K，末时刻绝对偏差均减小。
热端/冷端RMSE由0.6834/0.7034变为0.9489/0.9259 K；
339 W仍低估约6.26 K，后段斜率误差并非所有曲线均下降。
75秒之后ASL传感器最大到达跳变0.01428 K，顶部近中心为0.08569 K；
顶部2秒仍有约4.24 K的初始历史建立变化，顶部实测从5秒开始，保留这段原始诊断。
结果来自原始预测，不使用绘图平滑或强制单调；本轮比较包含温度约束和损失再加权，
不宣称此前查看过测试集的修复实验是盲测，也不宣称ASL全面优于其他结构。

## 检查与复现

在仓库根目录安装 `pip install -e '.[test]'` 后运行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_sequential_deeponet.py tests/test_sequential_temperature_objective.py tests/test_sequential_temperature_bias.py -q -o addopts=''
python scripts/plot_sequential_deeponet.py --output '研究记录/Sequential_DeepONet_1000epochs_temperature_tail'
```

无需原始数据即可测试核心实现并从保存预测重绘图像。对照上游 ASL 源码的测试
需要本地参考路径 `references/ASL-PINN-code/src/models/ssm_lstm.py`；缺少时跳过
这一项。本机全部 38 项已通过，对应上游提交为
`4477c971007624133eec92bd7eeb5bb58d3b9c26`。

重新训练、重新推理和审计输入数据哈希需要按照原项目格式准备 `data/`。
正式实验目录内 `verification.json` 记录已执行的完整审计。
本次隔离发布目录执行全量 `pytest tests`：275项通过，1项因没有上游参考仓库跳过；
2项因缺少未发布的V4旧图册、实际配置和第600轮模型失败。
相应测试及开发分析源码与上一提交完全一致，旧文件在上一提交中也未发布。
本项目ASL的38项针对性测试已在数据及参考源码齐全的工作区全部通过。

```bash
python scripts/train_sequential_deeponet.py --device cuda --output '研究记录/Sequential_DeepONet_new'
python scripts/evaluate_sequential_deeponet.py --device cuda --output '研究记录/Sequential_DeepONet_new'
python scripts/audit_sequential_temporal.py --output '研究记录/Sequential_DeepONet_new'
python scripts/analyze_sequential_temperature_bias.py --output '研究记录/Sequential_DeepONet_new' --methods fnn gru lstm asl --splits train validation test --selection-from '研究记录/Sequential_DeepONet_1000epochs_temperature_tail/diagnostics/selection.json'
python scripts/plot_sequential_deeponet.py --output '研究记录/Sequential_DeepONet_new'
python scripts/verify_sequential_deeponet.py --output '研究记录/Sequential_DeepONet_new'
```
