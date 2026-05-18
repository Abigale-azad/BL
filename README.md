# Black-Litterman 大类资产配置系统

基于 **Black-Litterman 模型**的大类资产配置 Web 应用，使用 Python + Streamlit 构建。

---

## 功能概览

| 步骤 | 功能 |
|------|------|
| ① 数据统计 | 自动计算年化收益、滚动波动率、相关系数矩阵 |
| ② BL 参数 | 设置 λ、τ、市场初始权重、隐含均衡收益 |
| ③ 观点设置 | 动态添加/删除绝对/相对观点，构建 P、q、Ω 矩阵 |
| ④ 约束条件 | 设置各资产权重上下限、组合最大波动率 |
| ⑤ BL 计算 | 一键执行后验均值-方差优化，输出最优权重 |
| ⑥ 结果对比 | 权重图、绩效指标、风险贡献三模型对比 |
| ⑦ 历史回测 | 净值曲线、最大回撤、年化收益/夏普比率 |

---

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动应用

```bash
streamlit run app.py
```

浏览器将自动打开 `http://localhost:8501`

### 3. 使用示例数据

启动后，在侧栏勾选 **"使用内置示例数据"** 即可立即体验，无需上传文件。

---

## 上传自定义数据

Excel 文件格式要求：

| Date       | 上证指数 | 沪深300 | 中证800 | 中证全债 | 股票基金指数 | 债券基金指数 |
|------------|----------|---------|---------|----------|------------|------------|
| 2011-12-30 | 2199.42  | 2345.74 | 2500.65 | 139.50   | 3770.85    | 1791.95    |
| 2012-01-04 | 2169.39  | 2298.75 | 2448.86 | 139.55   | 3692.23    | 1790.01    |

- **第一列**：日期（任意格式，pandas 可识别即可）
- **其余列**：各资产价格/净值序列（数值型）
- 资产列数不限，系统自动适配

---

## 生成测试数据脚本

将以下脚本保存为 `generate_test_data.py`，运行后生成 `test_data.xlsx`：

```python
"""
generate_test_data.py
生成 6 类资产日线价格数据（2011-2020 年），用于测试 BL 系统
运行: python generate_test_data.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)

dates = pd.date_range("2011-01-01", "2020-12-31", freq="B")
n = len(dates)

assets = ["上证指数", "沪深300", "中证800", "中证全债", "股票基金指数", "债券基金指数"]
init_prices = [2200.0, 2346.0, 2500.0, 139.5, 3770.0, 1792.0]
ann_rets    = [0.040,  0.050,  0.055,  0.040, 0.060,  0.040]
ann_vols    = [0.220,  0.220,  0.230,  0.020, 0.180,  0.015]

# 相关系数矩阵
corr = np.array([
    [1.00, 0.95, 0.93, 0.05, 0.85, 0.04],
    [0.95, 1.00, 0.97, 0.06, 0.88, 0.05],
    [0.93, 0.97, 1.00, 0.06, 0.86, 0.05],
    [0.05, 0.06, 0.06, 1.00, 0.07, 0.92],
    [0.85, 0.88, 0.86, 0.07, 1.00, 0.06],
    [0.04, 0.05, 0.05, 0.92, 0.06, 1.00],
])

dt = 1.0 / 252
vols = np.array(ann_vols)
mus  = np.array(ann_rets) - 0.5 * vols**2

L = np.linalg.cholesky(corr)
z = np.random.randn(len(assets), n)
eps = (L @ z).T  # (n, 6)

daily_log_ret = mus * dt + vols * np.sqrt(dt) * eps
prices = np.exp(np.cumsum(daily_log_ret, axis=0))
prices = prices / prices[0] * np.array(init_prices)

df = pd.DataFrame(prices, index=dates, columns=assets)
df.index.name = "Date"
df.to_excel("test_data.xlsx")
print(f"已生成 test_data.xlsx：{df.shape[0]} 行 x {df.shape[1]} 资产")
print(df.head())
```

---

## BL 模型数学原理

### 1. 隐含均衡收益
$$\Pi = \lambda \Sigma w_{mkt}$$

### 2. 不确定性矩阵（Idzorek 简化方法）
$$\Omega_{ii} = \left(\frac{1}{c_i} - 1\right) \cdot (\tau P \Sigma P^T)_{ii}$$

### 3. 后验均值
$$\mu^* = \left[(\tau\Sigma)^{-1} + P^T\Omega^{-1}P\right]^{-1} \left[(\tau\Sigma)^{-1}\Pi + P^T\Omega^{-1}q\right]$$

### 4. 后验协方差
$$\Sigma^* = \Sigma + \left[(\tau\Sigma)^{-1} + P^T\Omega^{-1}P\right]^{-1}$$

### 5. 均值-方差优化
$$\max_w \ \mu^{*T} w - \frac{1}{2}\lambda w^T \Sigma^* w$$
$$\text{s.t.} \quad \sum w_i = 1, \quad w_L \leq w \leq w_U, \quad \sqrt{w^T\Sigma^* w} \leq \sigma_{max}$$

---

## 文件结构

```
bl_app/
├── app.py              # 主应用（Streamlit UI）
├── bl_model.py         # BL 数学核心
├── utils.py            # 数据处理 & 图表工具
├── requirements.txt    # Python 依赖
└── README.md           # 本文档
```

---

## 扩展资产（无需改代码）

1. 在 Excel 中新增资产列（如：美股指数、黄金）
2. 重新上传文件，系统自动识别新资产
3. 在「观点设置」和「约束条件」页面为新资产配置参数
