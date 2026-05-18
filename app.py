"""
app.py — Black-Litterman 大类资产配置 Web 应用
启动方式: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io

from bl_model import (
    compute_implied_returns,
    compute_omega,
    bl_posterior,
    optimize_portfolio,
)
from utils import (
    load_price_data,
    generate_demo_data,
    compute_returns,
    compute_statistics,
    portfolio_stats,
    risk_contributions,
    backtest_portfolio,
    plot_rolling_vol,
    plot_corr_matrix,
    plot_weight_comparison,
    plot_risk_contribution,
    plot_nav_curves,
)

# ─── 页面配置 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Black-Litterman 大类资产配置",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS 美化 ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700; color: #1a3a5c;
        border-bottom: 3px solid #2563eb; padding-bottom: 8px; margin-bottom: 1rem;
    }
    .step-badge {
        display: inline-block; background: #2563eb; color: white;
        border-radius: 50%; width: 26px; height: 26px; text-align: center;
        line-height: 26px; font-weight: bold; font-size: 0.85rem; margin-right: 8px;
    }
    .metric-card {
        background: #f0f4ff; border-left: 4px solid #2563eb;
        border-radius: 6px; padding: 12px 16px; margin: 4px 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px; background: #f3f4f6; border-radius: 8px 8px 0 0;
        font-weight: 600; color: #374151;
    }
    .stTabs [aria-selected="true"] {
        background: #2563eb !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State 初始化 ──────────────────────────────────────────────────
def init_session():
    defaults = {
        "prices": None,
        "returns": None,
        "stats": None,
        "assets": [],
        "views": [],           # list of view dicts
        "bl_result": None,
        "use_demo": False,
        "last_assets": None,   # 用于检测资产列表变化
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 BL 资产配置系统")
    st.divider()

    # ── 数据上传 ──────────────────────────────────────────────────────────────
    st.markdown("### 📁 数据导入")
    uploaded = st.file_uploader(
        "上传 Excel 文件 (.xlsx/.xls)",
        type=["xlsx", "xls"],
        help="第一列为日期，后续列为各资产价格/净值",
    )

    use_demo = st.checkbox("使用内置示例数据", value=(st.session_state.prices is None))

    # 加载数据
    if uploaded is not None:
        prices, err = load_price_data(uploaded)
        if err:
            st.error(f"读取失败：{err}")
        else:
            st.session_state.prices = prices
            st.session_state.use_demo = False
            st.success(f"✅ 已加载 {prices.shape[1]} 类资产，{prices.shape[0]} 行")
    elif use_demo:
        if st.session_state.use_demo is False or st.session_state.prices is None:
            st.session_state.prices = generate_demo_data()
            st.session_state.use_demo = True
        st.info("📌 使用示例数据（6 类资产，2011-2020）")

    if st.session_state.prices is not None:
        prices = st.session_state.prices
        min_d, max_d = prices.index.min().date(), prices.index.max().date()

        st.markdown("### 📅 回测区间")
        start_d = st.date_input("开始日期", value=min_d, min_value=min_d, max_value=max_d)
        end_d = st.date_input("截止日期", value=max_d, min_value=min_d, max_value=max_d)

        if start_d >= end_d:
            st.warning("开始日期须早于截止日期")
        else:
            prices_sel = prices.loc[str(start_d):str(end_d)]
            returns = compute_returns(prices_sel)
            stats = compute_statistics(returns)
            st.session_state.returns = returns
            st.session_state.stats = stats
            st.session_state.assets = list(prices_sel.columns)
            st.caption(f"有效交易日：{len(returns)} 天")

    st.divider()

    # ── BL 全局参数 ────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ BL 全局参数")
    lambda_ = st.number_input(
        "风险厌恶系数 λ", min_value=0.1, max_value=20.0, value=3.0, step=0.1,
        help="控制组合对风险的敏感度，默认 3"
    )
    tau = st.number_input(
        "τ (Tau)", min_value=0.001, max_value=1.0, value=0.05, step=0.005,
        format="%.3f",
        help="观点相对先验的不确定性标量，默认 0.05"
    )
    tau_auto = st.checkbox(
        "自动计算 τ = 1/T",
        value=False,
        help="Adzorek 方法：τ = 1 / 样本数"
    )
    rf = st.number_input(
        "无风险利率 (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.1
    ) / 100.0

    max_vol = st.number_input(
        "组合最大年化波动率 (%)", min_value=1.0, max_value=50.0, value=15.0, step=0.5,
        help="0 表示不设上限"
    ) / 100.0

    opt_method = st.selectbox(
        "优化目标",
        ["最大化效用函数 (μ - 0.5λΣ)", "最大化夏普比率"],
        index=0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — 7 个步骤标签页
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">Black-Litterman 大类资产配置系统</div>',
            unsafe_allow_html=True)

if st.session_state.prices is None:
    st.info('👈 请先在侧栏上传 Excel 文件或勾选"使用示例数据"')
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "①数据统计", "②BL参数", "③观点设置", "④约束条件", "⑤BL计算", "⑥结果对比", "⑦历史回测"
])

# 快捷引用
assets = st.session_state.assets
n_assets = len(assets)
stats = st.session_state.stats
returns = st.session_state.returns


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数：清理资产相关状态（当资产列表发生变化时）
# ═══════════════════════════════════════════════════════════════════════════════
def reset_asset_dependent_state():
    """清空依赖于资产列表的所有用户状态"""
    st.session_state.views = []
    st.session_state.bl_result = None
    # 注意：bounds_df、mkt_weights_df、pi_manual 将在各自的 tab 中按需重建


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — 数据统计
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if stats is None:
        st.warning("请先在侧栏选择有效数据区间")
        st.stop()

    st.subheader("📈 数据统计概览")

    # 关键指标汇总表
    summary = pd.DataFrame({
        "年化均值收益": stats["mean_returns"].map("{:.2%}".format),
        "年化波动率": stats["ann_vol"].map("{:.2%}".format),
    })
    st.dataframe(summary.T, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_rolling_vol(stats["rolling_vol"], assets),
                        use_container_width=True)
    with col2:
        st.plotly_chart(plot_corr_matrix(stats["corr_matrix"]),
                        use_container_width=True)

    with st.expander("📋 查看完整协方差矩阵（年化）"):
        st.dataframe(
            stats["cov_matrix"].style.format("{:.4f}").background_gradient(
                cmap="Blues", axis=None
            ),
            use_container_width=True,
        )
    with st.expander("📋 查看原始日收益率（前 20 行）"):
        st.dataframe(
            returns.head(20).style.format("{:.4%}"),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BL 参数设置
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("⚙️ BL 参数设置")

    if stats is None:
        st.warning("请先加载数据")
        st.stop()

    # ================= 资产同步检查 =================
    # 如果当前资产列表与上次存储的不同，则重置所有依赖资产的状态
    if st.session_state.last_assets != assets:
        reset_asset_dependent_state()
        # 强制重建 mkt_weights_df 和 pi_manual（后续会按需创建）
        if "mkt_weights_df" in st.session_state:
            del st.session_state["mkt_weights_df"]
        if "pi_manual" in st.session_state:
            del st.session_state["pi_manual"]
        st.session_state.last_assets = assets.copy()
        # 不需要 rerun，但后续代码会重新创建这些 DataFrame

    cov = stats["cov_matrix"].values
    tau_val = tau if not tau_auto else 1.0 / len(returns)
    if tau_auto:
        st.info(f"自动计算 τ = 1/T = 1/{len(returns)} ≈ {tau_val:.5f}")

    # ── 市场初始权重 ────────────────────────────────────────────────────────
    st.markdown("#### 市场初始权重（可编辑）")
    col_hint, col_btn = st.columns([4, 1])
    with col_hint:
        st.caption("修改各资产权重，系统自动归一化。默认等权。")
    with col_btn:
        if st.button("🔄 重置等权"):
            st.session_state["mkt_weights_df"] = pd.DataFrame(
                {"权重 (%)": [round(100.0 / n_assets, 2)] * n_assets},
                index=assets,
            )

    # 如果 mkt_weights_df 不存在或索引与当前 assets 不一致，则重新创建
    if ("mkt_weights_df" not in st.session_state or
        list(st.session_state["mkt_weights_df"].index) != assets):
        st.session_state["mkt_weights_df"] = pd.DataFrame(
            {"权重 (%)": [round(100.0 / n_assets, 2)] * n_assets},
            index=assets,
        )

    mkt_df_edit = st.data_editor(
        st.session_state["mkt_weights_df"],
        use_container_width=True,
        num_rows="fixed",
        column_config={"权重 (%)": st.column_config.NumberColumn(
            min_value=0.0, max_value=100.0, step=0.1, format="%.2f"
        )},
        key="mkt_weight_editor",
    )
    st.session_state["mkt_weights_df"] = mkt_df_edit

    raw_w = mkt_df_edit["权重 (%)"].values.astype(float)
    raw_w = np.clip(raw_w, 0, None)
    if raw_w.sum() > 0:
        w_mkt = raw_w / raw_w.sum()
    else:
        w_mkt = np.ones(n_assets) / n_assets

    st.caption(f"归一化后权重合计：{w_mkt.sum():.4f}")

    # ── 隐含均衡收益 ────────────────────────────────────────────────────────
    st.markdown("#### 隐含均衡收益 Π")
    pi_mode = st.radio(
        "计算方式",
        ["从市场权重计算 (Π = λΣw)", "手动输入"],
        horizontal=True,
    )

    if pi_mode == "从市场权重计算 (Π = λΣw)":
        # 确保 w_mkt 长度与 cov 一致（已经一致，因为 w_mkt 基于当前 assets 构建）
        Pi = compute_implied_returns(lambda_, cov, w_mkt)
        pi_df = pd.DataFrame({"隐含均衡收益（年化）": Pi}, index=assets)
        st.dataframe(pi_df.style.format("{:.2%}"), use_container_width=True)
    else:
        st.caption("输入各资产年化预期收益（%）")
        # 如果 pi_manual 不存在或索引不匹配，则基于当前市场权重计算默认值
        if ("pi_manual" not in st.session_state or
            list(st.session_state["pi_manual"].index) != assets):
            default_pi = compute_implied_returns(lambda_, cov, w_mkt) * 100
            st.session_state["pi_manual"] = pd.DataFrame(
                {"隐含均衡收益 (%)": np.round(default_pi, 3)}, index=assets
            )
        pi_edit = st.data_editor(
            st.session_state["pi_manual"],
            use_container_width=True,
            num_rows="fixed",
            column_config={"隐含均衡收益 (%)": st.column_config.NumberColumn(
                format="%.3f", step=0.1
            )},
            key="pi_manual_editor",
        )
        st.session_state["pi_manual"] = pi_edit
        Pi = pi_edit["隐含均衡收益 (%)"].values.astype(float) / 100.0

    # 保存到 session
    st.session_state["Pi"] = Pi
    st.session_state["w_mkt"] = w_mkt
    st.session_state["tau_val"] = tau_val
    st.session_state["lambda_"] = lambda_
    st.session_state["rf"] = rf
    st.session_state["max_vol"] = max_vol
    st.session_state["opt_method"] = opt_method


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BL 观点设置
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("👁 BL 观点设置（核心）")

    if stats is None:
        st.warning("请先加载数据")
        st.stop()

    st.markdown(
        "支持**绝对观点**（某资产预期收益=Y%）和**相对观点**（A资产比B资产高/低X%）。"
        "每条观点可独立设置信心水平。"
    )

    # ── 添加观点 UI ────────────────────────────────────────────────────────
    with st.expander("➕ 添加新观点", expanded=len(st.session_state.views) == 0):
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            new_type = st.selectbox("观点类型", ["相对观点 (A - B = X%)", "绝对观点 (A = Y%)"],
                                    key="new_view_type")
        with vcol2:
            new_conf = st.slider("信心水平 (%)", 1, 100, 60, key="new_view_conf")

        if "相对" in new_type:
            vc1, vc2, vc3 = st.columns(3)
            with vc1:
                asset_a = st.selectbox("资产 A（多）", assets, key="va_a")
            with vc2:
                asset_b = st.selectbox("资产 B（空）",
                                       [a for a in assets if a != asset_a],
                                       key="va_b")
            with vc3:
                view_val = st.number_input("A 比 B 高 (%)", value=5.0, step=0.1,
                                           key="va_val", format="%.2f")
            view_desc = f"{asset_a} 比 {asset_b} 高 {view_val:.2f}%"
            new_view = {
                "type": "relative",
                "asset_a": asset_a,
                "asset_b": asset_b,
                "value": view_val / 100.0,
                "confidence": new_conf / 100.0,
                "desc": view_desc,
            }
        else:
            vc1, vc2 = st.columns(2)
            with vc1:
                asset_a = st.selectbox("资产", assets, key="va_abs")
            with vc2:
                view_val = st.number_input("预期年化收益 (%)", value=8.0, step=0.1,
                                           key="va_abs_val", format="%.2f")
            view_desc = f"{asset_a} 预期收益 = {view_val:.2f}%"
            new_view = {
                "type": "absolute",
                "asset_a": asset_a,
                "asset_b": None,
                "value": view_val / 100.0,
                "confidence": new_conf / 100.0,
                "desc": view_desc,
            }

        if st.button("✅ 添加此观点", type="primary"):
            st.session_state.views.append(new_view)
            st.success(f"已添加：{view_desc}，信心 {new_conf}%")
            st.rerun()

    # ── 当前观点列表 ──────────────────────────────────────────────────────────
    st.markdown(f"#### 当前观点列表（共 {len(st.session_state.views)} 条）")
    if len(st.session_state.views) == 0:
        st.info("暂无观点，BL 后验收益将等于市场均衡收益")
    else:
        for idx, v in enumerate(st.session_state.views):
            col_desc, col_conf, col_del = st.columns([5, 2, 1])
            with col_desc:
                st.markdown(f"**{idx+1}.** {v['desc']}")
            with col_conf:
                new_c = st.slider(
                    "", 1, 100, int(v["confidence"] * 100),
                    key=f"conf_slider_{idx}",
                    label_visibility="collapsed",
                )
                st.session_state.views[idx]["confidence"] = new_c / 100.0
            with col_del:
                if st.button("🗑", key=f"del_view_{idx}", help="删除此观点"):
                    st.session_state.views.pop(idx)
                    st.rerun()

        if st.button("🗑 清空所有观点", type="secondary"):
            st.session_state.views = []
            st.rerun()

    # ── 构建 P 矩阵、q 向量 ─────────────────────────────────────────────────
    def build_Pq(views, assets):
        k = len(views)
        n = len(assets)
        P = np.zeros((k, n))
        q = np.zeros(k)
        for i, v in enumerate(views):
            idx_a = assets.index(v["asset_a"])
            q[i] = v["value"]
            if v["type"] == "relative":
                idx_b = assets.index(v["asset_b"])
                P[i, idx_a] = 1.0
                P[i, idx_b] = -1.0
            else:
                P[i, idx_a] = 1.0
        return P, q

    P, q = build_Pq(st.session_state.views, assets)
    confidences = np.array([v["confidence"] for v in st.session_state.views])

    if len(st.session_state.views) > 0:
        st.markdown("#### P 矩阵 / q 向量 预览")
        p_df = pd.DataFrame(P, columns=assets,
                             index=[f"观点{i+1}" for i in range(len(P))])
        p_df["q (年化收益)"] = [f"{qi:.2%}" for qi in q]
        p_df["信心"] = [f"{c:.0%}" for c in confidences]
        st.dataframe(p_df.style.format("{:.0f}", subset=assets), use_container_width=True)

    # 保存
    st.session_state["P"] = P
    st.session_state["q"] = q
    st.session_state["confidences"] = confidences


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — 约束条件
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔒 约束条件设置")

    st.markdown(
        "设置各资产权重上下限，以及组合整体波动率上限。"
        "权重之和 = 1（强约束，自动满足）。"
    )

    # ── 全局波动率上限提示 ───────────────────────────────────────────────────
    st.markdown(f"**组合最大年化波动率**（在侧栏设置）：`{max_vol:.1%}`")

    # ── 权重上下限表格 ──────────────────────────────────────────────────────
    st.markdown("#### 各资产权重上下限")

    if "bounds_df" not in st.session_state or \
            list(st.session_state["bounds_df"].index) != assets:
        st.session_state["bounds_df"] = pd.DataFrame({
            "下限 (%)": [0.0] * n_assets,
            "上限 (%)": [100.0] * n_assets,
        }, index=assets)

    bounds_edit = st.data_editor(
        st.session_state["bounds_df"],
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "下限 (%)": st.column_config.NumberColumn(
                min_value=0.0, max_value=100.0, step=1.0, format="%.1f"
            ),
            "上限 (%)": st.column_config.NumberColumn(
                min_value=0.0, max_value=100.0, step=1.0, format="%.1f"
            ),
        },
        key="bounds_editor",
    )
    st.session_state["bounds_df"] = bounds_edit

    # 验证约束可行性
    lower_arr = bounds_edit["下限 (%)"].values.astype(float) / 100.0
    upper_arr = bounds_edit["上限 (%)"].values.astype(float) / 100.0
    lower_arr = np.clip(lower_arr, 0, 1)
    upper_arr = np.clip(upper_arr, lower_arr, 1)

    if lower_arr.sum() > 1.0:
        st.warning(f"⚠️ 所有下限之和 = {lower_arr.sum():.1%} > 1，优化可能无可行解，请调整")
    if upper_arr.sum() < 1.0:
        st.warning(f"⚠️ 所有上限之和 = {upper_arr.sum():.1%} < 1，优化可能无可行解，请调整")
    else:
        st.success(f"✅ 约束合理：下限合计 {lower_arr.sum():.1%}，上限合计 {upper_arr.sum():.1%}")

    st.session_state["lower_arr"] = lower_arr
    st.session_state["upper_arr"] = upper_arr

    # 快捷设置按钮
    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        if st.button("全部 0%~100%（无限制）"):
            st.session_state["bounds_df"]["下限 (%)"] = 0.0
            st.session_state["bounds_df"]["上限 (%)"] = 100.0
            st.rerun()
    with bcol2:
        if st.button("均匀约束 5%~30%"):
            st.session_state["bounds_df"]["下限 (%)"] = 5.0
            st.session_state["bounds_df"]["上限 (%)"] = 30.0
            st.rerun()
    with bcol3:
        if st.button("均匀约束 0%~25%"):
            st.session_state["bounds_df"]["下限 (%)"] = 0.0
            st.session_state["bounds_df"]["上限 (%)"] = 25.0
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BL 计算
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🚀 BL 计算")

    if stats is None:
        st.warning("请先加载数据")
        st.stop()

    # 参数汇总确认
    st.markdown("#### 当前参数确认")
    param_cols = st.columns(4)
    param_cols[0].metric("风险厌恶 λ", f"{lambda_:.2f}")
    param_cols[1].metric("τ (Tau)", f"{st.session_state.get('tau_val', tau):.4f}")
    param_cols[2].metric("无风险利率", f"{rf:.1%}")
    param_cols[3].metric("最大波动率", f"{max_vol:.1%}")

    st.markdown(f"**资产数量**: {n_assets}，**观点数量**: {len(st.session_state.views)}")

    # ── 计算按钮 ────────────────────────────────────────────────────────────
    run_btn = st.button("▶️ 执行 BL 计算", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("正在计算中..."):
            try:
                cov = stats["cov_matrix"].values
                Pi = st.session_state.get("Pi", compute_implied_returns(lambda_, cov, st.session_state.get("w_mkt", np.ones(n_assets)/n_assets)))
                tau_v = st.session_state.get("tau_val", tau)
                P = st.session_state.get("P", np.zeros((0, n_assets)))
                q = st.session_state.get("q", np.zeros(0))
                confs = st.session_state.get("confidences", np.zeros(0))
                lower = st.session_state.get("lower_arr", np.zeros(n_assets))
                upper = st.session_state.get("upper_arr", np.ones(n_assets))

                # 计算 Omega
                if len(st.session_state.views) > 0:
                    Omega = compute_omega(P, cov, tau_v, confs)
                else:
                    Omega = np.eye(0)  # 空矩阵，bl_posterior 会处理

                # BL 后验
                mu_post, sigma_post = bl_posterior(Pi, cov, tau_v, P, q, Omega if P.shape[0] > 0 else np.eye(0))

                # 优化
                if "最大化夏普" in opt_method:
                    from bl_model import max_sharpe
                    w_bl, success, msg = max_sharpe(mu_post, sigma_post, lower, upper, rf, max_vol)
                else:
                    w_bl, success, msg = optimize_portfolio(
                        mu_post, sigma_post, lambda_, lower, upper, max_vol
                    )

                if not success:
                    st.warning(f"⚠️ 优化器未完全收敛（{msg}），结果仍可参考")

                # 等权基准
                w_eq = np.clip(np.ones(n_assets) / n_assets, lower, upper)
                w_eq = w_eq / w_eq.sum()

                # 保存结果
                st.session_state["bl_result"] = {
                    "Pi": Pi, "mu_post": mu_post, "sigma_post": sigma_post,
                    "w_bl": w_bl, "w_mkt": st.session_state.get("w_mkt", np.ones(n_assets)/n_assets),
                    "w_eq": w_eq, "cov": cov, "success": success,
                }
                st.success("✅ BL 计算完成！请查看「⑥结果对比」和「⑦历史回测」标签页")
                st.rerun()  # 强制刷新页面，使 Tab6/Tab7 立即显示最新结果

            except Exception as e:
                st.error(f"计算出错：{e}")
                import traceback
                st.code(traceback.format_exc())

    # ── 显示最近一次结果 ────────────────────────────────────────────────────
    res = st.session_state.get("bl_result")
    if res is not None:
        st.markdown("#### BL 后验收益 vs 隐含均衡收益")
        comp_df = pd.DataFrame({
            "隐含均衡收益 Π": res["Pi"],
            "BL 后验收益 μ*": res["mu_post"],
            "差异": res["mu_post"] - res["Pi"],
        }, index=assets).map(lambda x: f"{x:.2%}")
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("#### 最优权重（BL 模型）")
        w_df = pd.DataFrame({
            "BL 最优权重": res["w_bl"],
            "市场初始权重": res["w_mkt"],
            "等权基准": res["w_eq"],
        }, index=assets).map(lambda x: f"{x:.2%}")
        st.dataframe(w_df, use_container_width=True)

        # 导出权重
        out_buf = io.BytesIO()
        pd.DataFrame({
            "资产": assets,
            "BL最优权重": res["w_bl"],
            "市场初始权重": res["w_mkt"],
            "等权基准": res["w_eq"],
        }).to_excel(out_buf, index=False)
        st.download_button(
            "⬇️ 导出权重 Excel",
            out_buf.getvalue(),
            file_name="bl_weights.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — 结果对比
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("📊 模型结果对比")

    res = st.session_state.get("bl_result")
    if res is None:
        st.info("请先在「⑤BL计算」标签页执行计算")
        st.stop()

    cov = res["cov"]
    mu_post = res["mu_post"]
    w_bl = res["w_bl"]
    w_mkt = res["w_mkt"]
    w_eq = res["w_eq"]

    # ── 权重对比图 ──────────────────────────────────────────────────────────
    st.plotly_chart(
        plot_weight_comparison(assets, {
            "BL 最优": w_bl,
            "市场初始权重": w_mkt,
            "等权基准": w_eq,
        }),
        use_container_width=True,
    )

    # ── 收益/风险/夏普对比 ────────────────────────────────────────────────
    st.markdown("#### 组合绩效指标对比")
    mu_hist = stats["mean_returns"].values

    def fmt_stats(w, mu, label):
        s = portfolio_stats(w, mu, cov, rf)
        return {
            "模型": label,
            "预期年化收益": f"{s['return']:.2%}",
            "年化波动率": f"{s['volatility']:.2%}",
            "夏普比率": f"{s['sharpe']:.3f}",
        }

    perf_rows = [
        fmt_stats(w_bl, mu_post, "BL 最优（后验收益）"),
        fmt_stats(w_mkt, mu_post, "市场初始权重（后验收益）"),
        fmt_stats(w_eq, mu_hist, "等权基准（历史均值）"),
    ]
    perf_df = pd.DataFrame(perf_rows).set_index("模型")
    st.dataframe(perf_df, use_container_width=True)

    # ── 风险贡献 ─────────────────────────────────────────────────────────
    st.markdown("#### 风险贡献分布")
    rc_bl = risk_contributions(w_bl, cov)
    rc_mkt = risk_contributions(w_mkt, cov)
    rc_eq = risk_contributions(w_eq, cov)

    st.plotly_chart(
        plot_risk_contribution(assets, {
            "BL 最优": rc_bl,
            "市场初始权重": rc_mkt,
            "等权基准": rc_eq,
        }),
        use_container_width=True,
    )

    # 数值明细
    with st.expander("📋 风险贡献数值明细"):
        rc_df = pd.DataFrame({
            "BL 最优": rc_bl / rc_bl.sum() if rc_bl.sum() > 0 else rc_bl,
            "市场初始权重": rc_mkt / rc_mkt.sum() if rc_mkt.sum() > 0 else rc_mkt,
            "等权基准": rc_eq / rc_eq.sum() if rc_eq.sum() > 0 else rc_eq,
        }, index=assets).map(lambda x: f"{x:.2%}")
        st.dataframe(rc_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — 历史回测
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("📈 组合历史回测")

    res = st.session_state.get("bl_result")
    if res is None:
        st.info("请先在「⑤BL计算」标签页执行计算")
        st.stop()

    bt_col1, bt_col2 = st.columns([3, 1])
    with bt_col1:
        bt_window = st.selectbox(
            "回测区间",
            ["全部历史", "最近1年", "最近3年", "最近5年"],
            index=0,
        )
    with bt_col2:
        show_single = st.checkbox("叠加单资产净值", value=False)

    ret_bt = returns.copy()
    if bt_window == "最近1年":
        ret_bt = ret_bt.iloc[-252:]
    elif bt_window == "最近3年":
        ret_bt = ret_bt.iloc[-756:]
    elif bt_window == "最近5年":
        ret_bt = ret_bt.iloc[-1260:]

    nav_dict = {
        "BL 最优": backtest_portfolio(ret_bt, res["w_bl"]),
        "市场初始权重": backtest_portfolio(ret_bt, res["w_mkt"]),
        "等权基准": backtest_portfolio(ret_bt, res["w_eq"]),
    }
    if show_single:
        for col in assets:
            single_ret = ret_bt[[col]]
            single_nav = (1 + single_ret[col]).cumprod()
            single_nav = single_nav / single_nav.iloc[0]
            single_nav.name = col
            nav_dict[col] = single_nav

    st.plotly_chart(plot_nav_curves(nav_dict), use_container_width=True)

    # 回测统计
    st.markdown("#### 回测统计")
    bt_stats_rows = []
    for label, nav in nav_dict.items():
        total_ret = float(nav.iloc[-1] - 1)
        days = len(nav)
        ann_ret = float((1 + total_ret) ** (252 / days) - 1) if days > 0 else 0
        daily_ret = nav.pct_change().dropna()
        ann_vol_bt = float(daily_ret.std() * np.sqrt(252))
        sharpe_bt = (ann_ret - rf) / ann_vol_bt if ann_vol_bt > 1e-10 else 0
        # 最大回撤
        roll_max = nav.cummax()
        dd = (nav - roll_max) / roll_max
        max_dd = float(dd.min())
        bt_stats_rows.append({
            "组合": label,
            "累计收益": f"{total_ret:.2%}",
            "年化收益": f"{ann_ret:.2%}",
            "年化波动率": f"{ann_vol_bt:.2%}",
            "夏普比率": f"{sharpe_bt:.3f}",
            "最大回撤": f"{max_dd:.2%}",
        })
    bt_df = pd.DataFrame(bt_stats_rows).set_index("组合")
    st.dataframe(bt_df, use_container_width=True)

    # 导出回测净值
    nav_export = pd.DataFrame({k: v for k, v in nav_dict.items()})
    out2 = io.BytesIO()
    nav_export.to_excel(out2)
    st.download_button(
        "⬇️ 导出回测净值 Excel",
        out2.getvalue(),
        file_name="bl_backtest_nav.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )