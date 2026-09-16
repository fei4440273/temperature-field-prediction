# 任08 F1：HF-only DeepONet-PINN来源入场与物理来源限制

日期：2026-09-16。状态：**仅HF观测代码与CPU一优化步诊断已入场；正式F1 PINN训练、五种子消融和跨方法排名均未开始**。任07尚在独立五seed真实GPU训练，不以其旧10轮失败seed0、任06单种子先导或任何历史LF最佳替代新F1来源；未读取旧`test_Data`温度标签。

## 已可独立核实的来源

- 模型是`src/sic_cu/models/deeponet_pinn.py`的原`DeepONetPINN`，固定宽128、潜维128、3个残差块、tanh、材料通道；`src/sic_cu/train/task08_f1_hf_only.py`为0—4分别从全新随机源构建模型和空AdamW。构造入口不读取LF检查点、历史HF权重、字典、仿真伪标签或来源LF接触初值；非法来源字段拒绝。
- 合法HF观测来自`src/sic_cu/train/multifidelity.py`的`_ir_dataset('train'/'validation')`及`_sensor_tensors(...,split='train'/'validation')`。真实12个HF训练功率顶部IR为29,593行；3个固定合法验证功率不参加训练。Hot/Cold在每个同折功率下均有各自的真实曲线、绝对与差分基线；固定测试折不读温度。CPU一优化步只取55 W训练IR真实64行与该功率真实两条环温曲线，并使用现行5/1训练传感器损失；**没有执行15批HF或独立物理优化步，不能当作同预算F1结果**。
- 源码`src/sic_cu/train/task08_f1_hf_only.py`、CPU专属`scripts/28_check_task08_f1_hf_only.py`与`tests/test_task08_f1_hf_only.py`构成唯一新增来源通路。诊断原件名为`诊断训练状态.pt`而非`best.pt`，随机源只有CPU记录，中文摘要和独立SHA侧车均标记正式资格为假。该诊断不产生HF合法验证成绩，不作为任08 F1实测分数。

## 阻断：尚无独立PINN物理输入

总计划任08明确禁止F1以LF预训练、字典、伪标签或LF辨识接触参数冒充“仅HF”。不仅`configs/boundary_conditions.yaml`的接触热阻初始化约`7.340724e-5 m²K/W`来自LF训练辨识，`src/sic_cu/physics/resolution.py`的解析入口还直接注入这个值。`reports/natural_convection_parameter_basis.md`明确顶部`9.0`和底部`4.3 W/(m²K)`取自80个LF模拟功率的表面温差中位数；`reports/physics_sensitivity.md`的SiC和Cu辐射率`0.5/0.5`只在含LF接触初值的候选中作名义筛选，非独立测量。`PhysicsLossComputer`在缺发射率时拒绝辐射边界；它仍会计算界面损失，即使界面权重置零也不能消除输入来源污染。

`interface_residuals(..., None)`施加SiC/Cu温度与热流连续，即**完美接触建模假设**，不是缺少界面物理，也不等价于F3已有接触热阻模型。未对上下对流、辐射率和接触处理取得独立、可审计的来源并事前固定之前，`require_independent_f1_physics_inputs()`与专属正式F1入口拒绝建立所谓“纯HF-only PINN”物理训练。保留现行名义物理值只能另列“共享LF标定输入”的额外对照，不能写成纯HF-only，也不能与任07主模型在“共同物理配置”口径上暗中混用。

## 后续资格

任07达到甲或乙、完成来源冻结后，另行登记F1真实五seed独立物理来源、与F3不同的完美接触或共享LF标定口径、相同12/3 HF训练/验证及实际1500/500/早停和各项优化消耗；使用完整阶段和合法验证产生正式证据。若无独立h/辐射率/接触证据，保持F1物理与跨方法比较**阻断**，不凭空添值、改共享配置或报F1优于主线。
