# 任11：Howard等复合多保真DeepONet原文与代码来源入场限界

核对日期：2026-09-16。此项仅为文献／工程来源只读入场，不是该方法在本装置的训练、实验成绩、实现复现或公平模型比较。任07已乙状态冻结（观测可复核，名义物理不合格），任08/09正式真实消融及数据效率和任10完整场模型对比仍缺；不提前写任11方法名次。

## 原文所述具体方法

[作者版v2全文](https://arxiv.org/html/2204.09157v2)与[论文登记／期刊DOI](https://arxiv.org/abs/2204.09157)核实：Howard、Perego、Karniadakis、Stinis于2023年《Journal of Computational Physics》发表论文，期刊DOI `10.1016/j.jcp.2023.112462`。原文§2.1/2.2.1使用三个子块同时训练：修改型LF DeepONet、建模LF→HF**线性相关**的标准DeepONet，以及建模**非线性相关**的修改型DeepONet，HF输出为后两块相加；线性分支输入含若干LF子网查询点，非线性分支同时含原算子输入和LF子网输出，且线性／非线性块结构与正则分工不同。其数据驱动目标含HF和LF拟合项与两项分支正则，物理版本另外加入体内PDE、初值和边界，论文特别分列“有HF数据”与“无HF成对输入输出、仅物理”的实验设定。[原文方法与公式(7)—(16)](https://arxiv.org/html/2204.09157v2)。

本项目现有`src/sic_cu/models/multifidelity.py` `AdditiveCorrectionModel`源码SHA256 `18bcc1c6c7d4c4f385ff9144b706488a60d54da7e5d10c1d319e2e3fb598f90f`，主温度形式为`T_HF=T_LF+C(r,z,t,power,material,T_LF)`；LF与单个MLP校正阶段冻结，随后有限受限联合，并未实现原文分立线性/非线性DeepONet子网、LF多点branch输入及对应的同时训练和正则。因此现有F3/E0绝不冒称Howard**同架构原文复现**；即使将来借鉴其三块思想，亦只能登记为对原热场数据和边界的**适配实现**，单列新增参数数目、训练与LF来源、预算和与当前工程校正器的区别。[原文三子网解释](https://arxiv.org/html/2204.09157v2)。

## 作者实现与工程对照限界

截至本次核对，[Amanda Howard本人代码页](https://amanda-howard.github.io/code/)列出SINDy-KANs、FBKANs、MFKANs、Stacking PINNs等代码，**未列这篇2023复合MF DeepONet的可核作者仓库**；[作者版论文HTML](https://arxiv.org/html/2204.09157v2)没有明确给出本论文自身的公开实现仓库链接，文中引用的其他physics-informed DeepONet Burgers数据源不能据此冒充本论文源码。查不到可核链接不等于证明作者从未释出代码；本轮登记为**作者实现未核得、精确复现资格未成立**，不会将其他作者Lu等2022年另篇`multifidelity-deeponet`仓库改名成Howard实现。[Howard作者代码页](https://amanda-howard.github.io/code/)。

任11正式方法对比若后续有可核原作者仓库，须先锁仓库原地址/版本/SHA并逐项对照三子网结构与本装置观测／物理适配；若仍没有，才可独立TDD写明“依据论文的三块架构自行适配”，记录和F3相同合法HF12训练／3验证、LF60训练／10验证合同、各seed新初始化、合理同预算与参考观测指标。旧方法未统一协议／预算不能进入公平五seed排名。HF内部场无真实实验温度、名义物理未闭合，即使新适配观测分数改善也不获得原装置内部可信资格；当前**没有任何任11新版GPU结果、模型SHA或改善率**，必须保持正式比较未开始。

只核原文／作者页及本项目来源与已有冻结报告；未读取旧固定TEST温度，不动旧部署、任07已封训练源码和任11方法排名。
