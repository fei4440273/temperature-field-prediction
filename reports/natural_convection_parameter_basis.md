# 自然对流换热系数取值依据

## Material Passport

- Origin Skill：`academic-research-suite / deep-research / fact-check`
- Origin Date：2026-09-02
- Verification Status：`SUPPORTED ENGINEERING ESTIMATE`
- Version Label：`natural_convection_h_v1`

## 结论

未施加固定水冷的水平外表面采用分方向常数换热系数：

| 表面 | 采用值 | 敏感性区间 |
|---|---:|---:|
| 水平上表面，热面朝上 | `9.0 W/(m^2*K)` | `5.1-10.7 W/(m^2*K)` |
| 水平下表面，热面朝下 | `4.3 W/(m^2*K)` | `2.4-5.1 W/(m^2*K)` |

铜外圆柱面 `r=0.05834 m` 使用用户确认的 `22℃` 固定温度水冷边界，不再叠加自然对流。

## 关联式和几何

COMSOL 的 External Natural Convection 文档要求水平板按上表面和下表面分别处理，水平板的
特征长度取 `L=A/P`。对本项目半径 `R=0.05834 m` 的圆面：

```text
L = A/P = pi*R^2/(2*pi*R) = R/2 = 0.02917 m
```

上向热表面采用经典实验关联：

```text
Nu = 0.54*Ra^(1/4),  2.2e4 <= Ra <= 8e6
h  = Nu*k_air/L
```

该关联及其适用范围来自 Goldstein、Sparrow 和 Jones 对不同平面形状水平表面的实验研究。
下向热表面采用工程文献中的 `Nu=0.27*Ra^(1/4)` 分支。COMSOL 同样区分 horizontal plate
upside/downside，并建议空气物性在膜温处评价。

## 项目数据计算

为满足当前“常物性”设定，计算固定使用约 `26.85℃`（内部热力学温度 300 K）的空气参数：`k=0.0263 W/(m*K)`、
`nu=15.89e-6 m^2/s`、`alpha=22.5e-6 m^2/s`、`beta=1/300 K^-1`。空气输运物性的权威
数据源类型为 NIST REFPROP；这里的常数化是工程近似，不是 REFPROP 全温区逐点求值。

使用全部 80 个 Simulation 功率的水平表面面积加权平均温度，并排除小于 `1℃` 的温差后：

| 表面 | 样本数 | 温差 P05/中位/P95/最大 (℃) | h P05/中位/P95/最大 (W/(m^2*K)) |
|---|---:|---|---|
| 上表面 | 7,987 | `5.50 / 50.88 / 103.13 / 110.00` | `5.15 / 8.98 / 10.71 / 10.88` |
| 下表面 | 7,953 | `4.10 / 41.29 / 83.74 / 89.32` | `2.39 / 4.26 / 5.08 / 5.17` |

因此采用中位数圆整值 `9.0` 和 `4.3 W/(m^2*K)`，敏感性分析使用 P05-P95 区间。

## 来源

- [COMSOL 6.3: Heat Transfer Coefficients - External Natural Convection](https://doc.comsol.com/6.3/doc/com.comsol.help.heat/heat_ug_theory.07.096.html)
- [Goldstein, Sparrow, and Jones (1974), Natural Convection Adjacent to Horizontal Surface of Various Planforms, DOI 10.1115/1.3450224](https://doi.org/10.1115/1.3450224)
- [NIST REFPROP: thermodynamic and transport properties](https://www.nist.gov/srd/refprop)

## 限制

- `h` 是由关联式和当前低保真表面温差得到的有效常数，不是本装置的直接实验测量。
- 下表面一部分低温差样本位于常用关联式适用范围边缘，因此必须执行上述敏感性分析。
- 自然对流和接触热阻可能相关；参数辨识时不能把二者的作用互相替代。
- 辐射发射率未知，将与接触热阻在训练折内联合辨识；辨识完成前完整边界配置保持
  `verified: false`，但不再视为用户待补数据。
