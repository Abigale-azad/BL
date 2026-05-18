"""
utils.py — 数据处理与组合统计工具函数
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────

def load_price_data(file_obj) -> tuple[pd.DataFrame | None, str | None]:
    """
    从 Excel 文件加载价格数据。
    自动识别日期列（第一列或含 'date'/'日期' 的列名），其余列为资产价格。
    Returns: (df_prices, error_msg)
    """
    try:
        df = pd.read_excel(file_obj, engine="openpyxl")
    except Exception:
        try:
            df = pd.read_excel(file_obj, engine="xlrd")
        except Exception as e:
            return None, f"读取文件失败: {e}"

    # 识别日期列
    date_col = None
    for col in df.columns:
        col_str = str(col).lower()
        if "date" in col_str or "日期" in col_str or "time" in col_str or "时间" in col_str:
            date_col = col
            break
    if date_col is None:
        date_col = df.columns[0]

    # 解析日期
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        return None, f"日期列解析失败: {e}"

    df = df.set_index(date_col).sort_index()

    # 只保留数值列
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] < 2:
        return None, "有效资产列不足（至少需要 2 列）"

    # 删除全 NaN 行
    df = df.dropna(how="all")

    return df, None


def generate_demo_data() -> pd.DataFrame:
    """生成示例数据（6 类资产，2011-2020 年日线价格）"""
    np.random.seed(42)
    dates = pd.date_range("2011-01-01", "2020-12-31", freq="B")
    n = len(dates)

    assets = ["上证指数", "沪深300", "中证800", "中证全债", "股票基金指数", "债券基金指数"]
    init_prices = [2200, 2346, 2500, 139.5, 3770, 1792]
    ann_rets = [0.04, 0.05, 0.055, 0.04, 0.06, 0.04]
    ann_vols = [0.22, 0.22, 0.23, 0.02, 0.18, 0.015]
    corr = np.array([
        [1.00, 0.95, 0.93, 0.05, 0.85, 0.04],
        [0.95, 1.00, 0.97, 0.06, 0.88, 0.05],
        [0.93, 0.97, 1.00, 0.06, 0.86, 0.05],
        [0.05, 0.06, 0.06, 1.00, 0.07, 0.92],
        [0.85, 0.88, 0.86, 0.07, 1.00, 0.06],
        [0.04, 0.05, 0.05, 0.92, 0.06, 1.00],
    ])

    dt = 1 / 252
    vols = np.array(ann_vols)
    mus = np.array(ann_rets) - 0.5 * vols**2
    L = np.linalg.cholesky(corr)

    daily_rets = (mus * dt + vols * np.sqrt(dt) * (L @ np.random.randn(len(assets), n)).T)
    prices = np.exp(np.cumsum(daily_rets, axis=0))
    prices = prices / prices[0] * np.array(init_prices)

    df = pd.DataFrame(prices, index=dates, columns=assets)
    df.index.name = "Date"
    return df


# ─────────────────────────────────────────────
# 统计计算
# ─────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """从价格序列计算日收益率"""
    return prices.pct_change().dropna()


def compute_statistics(
    returns: pd.DataFrame, rolling_window: int = 120
) -> dict:
    """
    计算年化统计指标
    Returns dict with keys:
        mean_returns, cov_matrix, corr_matrix, rolling_vol, ann_vol
    """
    ann = 252
    mean_returns = returns.mean() * ann
    cov_matrix = returns.cov() * ann
    corr_matrix = returns.corr()
    rolling_vol = returns.rolling(rolling_window).std() * np.sqrt(ann)
    ann_vol = returns.std() * np.sqrt(ann)

    return {
        "mean_returns": mean_returns,
        "cov_matrix": cov_matrix,
        "corr_matrix": corr_matrix,
        "rolling_vol": rolling_vol,
        "ann_vol": ann_vol,
    }


# ─────────────────────────────────────────────
# 组合统计
# ─────────────────────────────────────────────

def portfolio_stats(
    weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = 0.03
) -> dict:
    """返回组合预期收益、年化波动率、夏普比率"""
    w = np.array(weights)
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 1e-10 else 0.0
    return {"return": ret, "volatility": vol, "sharpe": sharpe}


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """边际风险贡献（绝对值，年化）"""
    w = np.array(weights)
    vol = np.sqrt(w @ cov @ w)
    if vol < 1e-10:
        return np.zeros(len(w))
    mrc = cov @ w / vol
    rc = w * mrc
    return rc


def backtest_portfolio(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """历史回测净值曲线"""
    w = np.array(weights)
    port_ret = returns @ w
    nav = (1 + port_ret).cumprod()
    nav = nav / nav.iloc[0]
    nav.name = "组合净值"
    return nav


# ─────────────────────────────────────────────
# Plotly 图表工具
# ─────────────────────────────────────────────

COLORS = px.colors.qualitative.Plotly


def plot_rolling_vol(rolling_vol: pd.DataFrame, assets: list[str]) -> go.Figure:
    """滚动波动率折线图"""
    fig = go.Figure()
    for i, col in enumerate(assets):
        fig.add_trace(go.Scatter(
            x=rolling_vol.index, y=rolling_vol[col],
            name=col, line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            mode="lines"
        ))
    fig.update_layout(
        title="滚动波动率（120日，年化）",
        xaxis_title="日期", yaxis_title="年化波动率",
        yaxis_tickformat=".1%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380, margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_white",
    )
    return fig


def plot_corr_matrix(corr: pd.DataFrame) -> go.Figure:
    """相关系数热力图"""
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11),
        hoverongaps=False,
    ))
    fig.update_layout(
        title="资产相关系数矩阵",
        height=420, margin=dict(l=80, r=20, t=60, b=60),
        template="plotly_white",
    )
    return fig


def plot_weight_comparison(
    asset_names: list[str],
    weights_dict: dict[str, np.ndarray],
) -> go.Figure:
    """权重对比柱状图"""
    fig = go.Figure()
    for i, (label, w) in enumerate(weights_dict.items()):
        fig.add_trace(go.Bar(
            name=label, x=asset_names, y=w,
            text=[f"{v:.1%}" for v in w],
            textposition="outside",
            marker_color=COLORS[i % len(COLORS)],
        ))
    fig.update_layout(
        title="资产配置权重对比",
        barmode="group",
        yaxis_tickformat=".0%",
        xaxis_title="资产", yaxis_title="权重",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420, margin=dict(l=40, r=20, t=60, b=60),
        template="plotly_white",
    )
    return fig


def plot_risk_contribution(
    asset_names: list[str],
    rc_dict: dict[str, np.ndarray],
) -> go.Figure:
    """风险贡献饼图（子图）"""
    n = len(rc_dict)
    fig = make_subplots(rows=1, cols=n, specs=[[{"type": "pie"}] * n],
                        subplot_titles=list(rc_dict.keys()))
    for i, (label, rc) in enumerate(rc_dict.items()):
        rc_pct = np.abs(rc) / np.abs(rc).sum() if np.abs(rc).sum() > 0 else rc
        fig.add_trace(go.Pie(
            labels=asset_names, values=rc_pct,
            name=label, hole=0.35,
            textinfo="label+percent",
            showlegend=(i == 0),
        ), row=1, col=i + 1)
    fig.update_layout(
        title="风险贡献分布",
        height=380, margin=dict(l=20, r=20, t=80, b=20),
        template="plotly_white",
    )
    return fig


def plot_nav_curves(nav_dict: dict[str, pd.Series]) -> go.Figure:
    """净值曲线对比"""
    fig = go.Figure()
    for i, (label, nav) in enumerate(nav_dict.items()):
        fig.add_trace(go.Scatter(
            x=nav.index, y=nav.values,
            name=label, mode="lines",
            line=dict(color=COLORS[i % len(COLORS)], width=2),
        ))
    fig.update_layout(
        title="组合历史净值曲线（回测）",
        xaxis_title="日期", yaxis_title="净值",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400, margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_white",
    )
    return fig
